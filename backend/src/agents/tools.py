"""
工具Agent (Tool Agent)
职责：
- 根据对话Agent的请求执行工具
- 管理工具调用结果
- 可选触发：用户请求特定功能或对话结束需要跟进
"""
from typing import Dict, Any, List
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage

from ..core.state import AgentState
from ..tools.emotion_tracker import emotion_tracker
from ..tools.scheduler import proactive_scheduler, reminder_tool
from ..tools.weather import weather_tool
from ..tools.web_search import web_search_tool


class ToolAgent:
    """工具Agent - 负责执行各种辅助工具"""
    
    def __init__(self):
        # 注册可用工具
        self.tools = {
            "emotion_tracker": emotion_tracker,
            "proactive_message": proactive_scheduler,
            "reminder": reminder_tool,
            "weather": weather_tool,
            "web_search": web_search_tool
        }
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行单个工具"""
        if tool_name not in self.tools:
            logger.warning(f"未知工具: {tool_name}")
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Unknown tool: {tool_name}"
            }
        
        tool = self.tools[tool_name]
        
        try:
            result = await tool.run(parameters)
            logger.info(f"工具执行成功: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"工具执行失败: {tool_name}, error={e}")
            return {
                "success": False,
                "tool_name": tool_name,
                "error": str(e)
            }
    
    async def execute_all(
        self, 
        tool_requests: List[Dict[str, Any]],
        user_id: str = "anonymous"
    ) -> List[Dict[str, Any]]:
        """执行所有工具请求"""
        results = []
        
        for request in tool_requests:
            tool_name = request.get("tool_name")
            parameters = request.get("parameters", {})
            
            # 注入user_id
            parameters["user_id"] = user_id
            
            result = await self.execute_tool(tool_name, parameters)
            result["request_reason"] = request.get("reason", "")
            results.append(result)
        
        return results
    
    def format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化工具结果为可读文本"""
        if not results:
            return ""
        
        formatted_parts = []
        
        for result in results:
            tool_name = result.get("tool_name")
            
            # 处理需要用户输入的情况（如天气工具缺少城市）
            if not result.get("success") and result.get("needs_user_input"):
                question = result.get("question", "")
                if question:
                    formatted_parts.append(question)
                continue
            
            if not result.get("success"):
                logger.debug(f"工具执行失败，跳过: {tool_name}, error={result.get('error')}")
                continue
            
            tool_result = result.get("result", {})
            
            if tool_name == "weather":
                # 天气信息
                formatted_message = tool_result.get("formatted_message", "")
                if formatted_message:
                    formatted_parts.append(formatted_message)
                else:
                    logger.warning(f"天气工具返回结果中没有formatted_message: {tool_result}")
            
            elif tool_name == "web_search":
                # 搜索结果
                formatted_message = tool_result.get("formatted_message", "")
                if formatted_message:
                    formatted_parts.append(formatted_message)
            
            elif tool_name == "reminder":
                # 提醒确认
                confirmation = tool_result.get("confirmation", "")
                if confirmation:
                    formatted_parts.append(confirmation)
            
            elif tool_name == "emotion_tracker":
                action = tool_result.get("action")
                if action == "summary":
                    summary = tool_result.get("summary", {})
                    if summary.get("total_records", 0) > 0:
                        trend = summary.get("trend", "")
                        trend_text = {
                            "improving": "整体在好转",
                            "worsening": "可能需要多关注自己",
                            "stable": "比较稳定"
                        }.get(trend, "")
                        if trend_text:
                            formatted_parts.append(f"📊 最近的情绪{trend_text}")
            
            elif tool_name == "proactive_message":
                action = tool_result.get("action")
                if action == "scheduled":
                    formatted_parts.append("我会记得过来看看你的 💫")
        
        return "\n\n".join(formatted_parts)
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        工具Agent主入口
        """
        tool_requests = state.get("tool_requests", [])
        user_id = state.get("user_id", "anonymous")
        perception = state.get("perception", {})
        
        if not tool_requests:
            logger.info("无工具请求，跳过工具Agent")
            return state
        
        logger.info(f"工具Agent处理 {len(tool_requests)} 个请求")
        
        # 自动记录情绪（如果有感知结果）
        emotion_info = perception.get("emotion", {})
        if emotion_info:
            # 提取最近3轮对话的上下文
            messages = state.get("messages", [])
            recent_conversations = []
            if messages:
                # 获取最近6条消息（3轮对话：每轮包含用户消息和助手回复）
                recent_messages = messages[-6:] if len(messages) > 6 else messages
                # 按消息类型配对，提取对话
                i = 0
                while i < len(recent_messages):
                    msg = recent_messages[i]
                    # 检查是否是用户消息
                    if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                        user_content = getattr(msg, 'content', str(msg))
                        # 查找下一条助手消息
                        assistant_content = ""
                        if i + 1 < len(recent_messages):
                            next_msg = recent_messages[i + 1]
                            if isinstance(next_msg, AIMessage) or (hasattr(next_msg, 'type') and next_msg.type == 'ai'):
                                assistant_content = getattr(next_msg, 'content', str(next_msg))
                                i += 2  # 跳过用户和助手消息
                            else:
                                i += 1  # 只跳过用户消息
                        else:
                            i += 1
                        
                        conversation = {
                            "user": user_content,
                            "assistant": assistant_content
                        }
                        recent_conversations.append(conversation)
                    else:
                        i += 1
                
                # 只保留最近3轮对话
                recent_conversations = recent_conversations[-3:] if len(recent_conversations) > 3 else recent_conversations
            
            # 添加情绪记录请求
            emotion_request = {
                "tool_name": "emotion_tracker",
                "parameters": {
                    "action": "record",
                    "emotion": emotion_info.get("emotion", "calm"),
                    "intensity": emotion_info.get("intensity", 0.5),
                    "scene": perception.get("scene_hint", "other"),
                    "recent_conversations": recent_conversations
                },
                "reason": "自动情绪记录"
            }
            tool_requests.append(emotion_request)
        
        # 执行所有工具
        results = await self.execute_all(tool_requests, user_id)
        
        # 检查是否有工具需要用户输入
        needs_user_input = any(
            not r.get("success") and r.get("needs_user_input") 
            for r in results
        )
        
        # 格式化结果
        formatted_results = self.format_tool_results(results)
        
        # 工具Agent只执行工具，不生成最终响应
        # 将工具结果保存到状态中，让对话Agent根据工具结果生成最终回答
        # 如果需要用户输入，直接返回询问
        if needs_user_input:
            formatted_results = self.format_tool_results(results)
            response = formatted_results if formatted_results else "请问你想查询哪个城市的天气呢？"
            return {
                **state,
                "tool_results": results,
                "response": response,
                "needs_tools": False  # 标记不需要再次调用工具
            }
        
        # 工具执行成功，标记需要对话Agent根据工具结果生成回答
        return {
            **state,
            "tool_results": results,
            "tool_results_formatted": formatted_results,  # 保存格式化的工具结果
            "needs_tools": False,  # 标记工具已执行，不需要再次调用
            "has_tool_results": True  # 标记有工具结果，对话Agent需要根据工具结果生成回答
        }


# 创建全局实例
tool_agent = ToolAgent()


async def run_tool_agent(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await tool_agent(state)
