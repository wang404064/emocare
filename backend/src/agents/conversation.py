"""
对话Agent (Conversation Agent)
职责：
- 基于感知结果生成共情对话
- 基于Qwen3-8B微调的模型,部署在阿里云百炼平台，使用百炼平台的API调用
- 自主决定使用什么策略和语言组织
- 判断是否需要触发工具Agent
"""
from typing import List, Optional
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from ..core.config import settings
from ..core.state import AgentState


class ConversationAgent:
    """对话Agent - 负责生成共情对话响应"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=30.0,  # 30秒超时
            max_retries=2  # 最多重试2次
        )
        
        # 主对话Prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}")
        ])
        
        # 工具需求判断Prompt
        self.tool_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分析助手。请分析用户消息和对话上下文，判断是否需要触发以下工具：

可用工具：
1. weather - 天气查询（当用户询问天气、想了解某地天气时）
   - 参数: {{"city": "城市名称"}}（如果用户没有指定城市，可以不传city参数，工具会询问用户所在城市）
2. web_search - 互联网搜索（当用户需要查找信息、了解某个话题时）
   - 参数: {{"query": "搜索关键词"}}
3. emotion_tracker - 情绪记录（当对话结束或用户表达了明显的情绪时，用于长期追踪）
   - 参数: {{"action": "record", "emotion": "情绪类型", "intensity": 强度值}}
4. reminder - 提醒功能（当用户说"提醒我"、"明天记得"等）
   - 参数: {{"time": "提醒时间", "content": "提醒内容"}}
5. proactive_message - 主动关怀消息（当用户情绪较低落，需要后续跟进时）
   - 参数: {{"action": "schedule"}}

请以JSON格式返回：
{{
    "needs_tools": true/false,
    "tool_requests": [
        {{
            "tool_name": "工具名称",
            "parameters": {{"参数名": "参数值"}},
            "reason": "调用原因"
        }}
    ]
}}

如果不需要任何工具，返回：
{{
    "needs_tools": false,
    "tool_requests": []
}}"""),
            ("human", "用户消息: {user_input}\n感知结果: {perception}\n助手回复: {response}")
        ])
        
        self.json_parser = JsonOutputParser()
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位温暖、专业的情感陪伴助手，名叫"小暖"。你的核心使命是提供真诚的陪伴。

【你的性格特点】
- 温暖真诚：像一个懂你的朋友，不居高临下
- 善于倾听：认真理解用户的感受，不急于给建议
- 适度共情：理解并反映用户的情绪，但不过度代入
- 语言自然：说话像真人，不机械，不用过多表情符号

【当前感知信息】
- 用户情绪: {emotion}（强度: {intensity}）
- 场景提示: {scene}
- 当前对话策略: {strategy}

【对话策略指导】
根据 current_strategy 严格执行：
- empathy_first: 高强度负面情绪，先充分共情，暂缓建议，多追问感受
- empathy_first_gentle_probe: 用户可能有一些深层痛苦。核心是让对方感到被真正倾听了——
    * 先共情、先理解，不要急于给建议或解决问题
    * 柔性地多问一句"最近发生什么了吗？"或"这种感觉是从什么时候开始的？"
    * 如果对话自然推进，可以在结尾轻柔地带一句："有时候找个专业的人聊聊，也是一种对自己的善待"
    * 不要机械地在每条回复都贴热线——只在对方明确表达无助或请求帮助时才提供
- gentle_explore: 中等情绪，共情后可以轻柔地探索、询问
- normal_chat: 低强度或积极情绪，自然地交流
- crisis_immediate: 危机状态（此分支由危机Agent处理，不应进入此处）

根据场景调整：
- work_stress: 理解职场压力，可探讨边界感
- relationship: 倾听为主，不轻易评判他人
- family: 理解复杂性，尊重用户的处境
- loneliness: 陪伴感要强，让用户感到被理解
- health_anxiety: 温和安抚，不诊断也不否定担忧

【重要原则】
1. 永远不要说"我理解你的感受"这种空话，要具体说出你理解了什么
2. 不要给出过多建议，除非用户明确请求
3. 可以适当追问，帮助用户表达
4. 如果用户想聊别的，就自然地跟着聊
5. 回复长度适中，不要太长让人有压力
6. 严禁重复：不要在回复中重复相同的句子或段落，确保每个表达都是唯一的
7. 永远不要做医疗、法律诊断或建议，超出能力范围时建议用户寻求专业帮助

记住：你是陪伴者，不是解决问题的人。有时候，被听见本身就是最好的支持。"""
    
    def _build_enhanced_prompt(self, perception: dict, current_strategy: str = "normal_chat") -> str:
        """根据感知结果增强系统提示"""
        emotion_info = perception.get("emotion", {})
        emotion = emotion_info.get("emotion", "calm")
        intensity = emotion_info.get("intensity", 0.5)
        scene = perception.get("scene_hint", "other")

        return self._get_system_prompt().format(
            emotion=emotion,
            intensity=intensity,
            scene=scene,
            strategy=current_strategy
        )
    
    def _prepare_history(self, messages: list, max_length: int = None) -> list:
        """准备对话历史"""
        if max_length is None:
            max_length = settings.MAX_HISTORY_LENGTH
        
        # 只保留最近的消息
        recent = messages[-max_length:] if len(messages) > max_length else messages
        return recent
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """
        去除重复的句子
        检测并移除文本中完全重复的句子
        """
        import re
        
        # 按句号、问号、感叹号分割句子
        sentences = re.split(r'([。！？\n])', text)
        
        # 重新组合句子（保留分隔符）
        result_sentences = []
        seen_sentences = set()
        
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                # 句子 + 分隔符
                sentence = sentences[i] + sentences[i + 1]
                # 去除首尾空白，用于比较
                sentence_clean = sentence.strip()
                
                # 如果句子长度太短（可能是标点符号），直接添加
                if len(sentence_clean) <= 2:
                    result_sentences.append(sentence)
                    i += 2
                    continue
                
                # 检查是否重复（使用清理后的句子进行比较）
                if sentence_clean not in seen_sentences:
                    seen_sentences.add(sentence_clean)
                    result_sentences.append(sentence)
                # 如果重复，跳过
                i += 2
            else:
                # 最后一个元素（可能是单独的标点）
                result_sentences.append(sentences[i])
                i += 1
        
        return ''.join(result_sentences)
    
    async def generate_response(
        self,
        user_input: str,
        perception: dict,
        history: list,
        current_strategy: str = "normal_chat"
    ) -> str:
        """生成对话响应"""
        try:
            # 构建增强的系统提示
            enhanced_system = self._build_enhanced_prompt(perception, current_strategy)
            
            # 准备历史消息
            prepared_history = self._prepare_history(history)
            
            # 创建带有增强系统提示的prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", enhanced_system),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_input}")
            ])
            
            chain = prompt | self.llm
            
            result = await chain.ainvoke({
                "history": prepared_history,
                "user_input": user_input
            })
            
            # 后处理：去除重复的句子
            response = self._remove_duplicate_sentences(result.content)
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"对话生成失败: {error_msg}")
            # 提供更详细的错误信息
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                logger.error(f"LLM API连接失败，请检查: base_url={settings.LLM_API_BASE}, api_key是否有效")
            return "我在这里听着呢。你想和我聊聊吗？"
    
    async def check_tool_needs(
        self, 
        user_input: str, 
        perception: dict, 
        response: str
    ) -> dict:
        """检查是否需要触发工具"""
        try:
            chain = self.tool_check_prompt | self.llm | self.json_parser
            
            result = await chain.ainvoke({
                "user_input": user_input,
                "perception": str(perception),
                "response": response
            })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"工具需求检查失败: {error_msg}")
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                logger.error(f"LLM API连接失败，请检查: base_url={settings.LLM_API_BASE}, api_key是否有效")
            return {"needs_tools": False, "tool_requests": []}
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        对话Agent主入口
        """
        user_input = state["user_input"]
        perception = state.get("perception", {})
        messages = state.get("messages", [])
        has_tool_results = state.get("has_tool_results", False)
        tool_results_formatted = state.get("tool_results_formatted", "")
        current_strategy = state.get("current_strategy", "normal_chat")
        
        logger.info(f"对话Agent处理: {user_input[:50]}... 策略: {current_strategy}")
        
        # 如果已经有工具结果（且格式化文本非空），根据工具结果生成最终回答
        if has_tool_results and tool_results_formatted:
            logger.info("根据工具结果生成最终回答")
            enhanced_input = f"{user_input}\n\n[工具执行结果]\n{tool_results_formatted}"
            response = await self.generate_response(
                user_input=enhanced_input,
                perception=perception,
                history=messages,
                current_strategy=current_strategy
            )
        elif has_tool_results and not tool_results_formatted:
            # 工具已执行但无格式化结果（如仅做了情绪记录），直接生成正常回复
            logger.info("工具有结果但无可展示文本，生成正常回复")
            response = await self.generate_response(
                user_input=user_input,
                perception=perception,
                history=messages,
                current_strategy=current_strategy
            )
        else:
            # 1. 先检查是否需要工具（基于用户输入，不依赖响应）
            tool_check = await self.check_tool_needs(
                user_input=user_input,
                perception=perception,
                response=""  # 先不生成响应，基于用户输入判断
            )
            
            needs_tools = tool_check.get("needs_tools", False)
            tool_requests = tool_check.get("tool_requests", [])
            
            if needs_tools:
                logger.info(f"需要工具: {[t['tool_name'] for t in tool_requests]}")
                # 如果需要工具，先不生成响应，让工具Agent执行后再生成
                response = ""  # 暂时不生成响应，等待工具执行后生成
            else:
                # 如果不需要工具，正常生成响应
                response = await self.generate_response(
                    user_input=user_input,
                    perception=perception,
                    history=messages,
                    current_strategy=current_strategy
                )
            
            logger.info(f"生成响应: {response[:100]}...")
            
            # 2. 更新状态
            return {
                **state,
                "response": response,
                "needs_tools": needs_tools,
                "tool_requests": tool_requests
            }
        
        # 有工具结果的情况，返回最终响应
        logger.info(f"生成最终响应: {response[:100]}...")
        return {
            **state,
            "response": response,
            "needs_tools": False,  # 工具已执行，不再需要工具
            "has_tool_results": False  # 重置标志
        }


# 创建全局实例
conversation_agent = ConversationAgent()


async def run_conversation(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await conversation_agent(state)
