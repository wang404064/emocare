"""
感知Agent (Perception Agent)
职责：
1. 情绪识别 (9分类 + 强度) - 使用BERT模型
2. 危机检测 (关键词硬匹配 + 风险等级)
3. 场景识别 (简单分类)
"""
import re
from typing import List, Tuple
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..core.config import settings
from ..core.state import AgentState, PerceptionResult, EmotionAnalysis
from ..models.emotion_recognizer import get_emotion_recognizer


class PerceptionAgent:
    """感知Agent - 负责情绪识别、危机检测和场景分类"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=0.3,  # 感知任务需要更确定性的输出
            max_tokens=512,
            timeout=30.0,  # 30秒超时
            max_retries=2  # 最多重试2次
        )
        self.crisis_keywords = settings.CRISIS_KEYWORDS
        self.emotion_categories = settings.EMOTION_CATEGORIES
        self.scene_categories = settings.SCENE_CATEGORIES
        
        # 初始化情绪识别器（延迟加载）
        self.emotion_recognizer = None
        
        # 场景识别Prompt（情绪识别已由模型完成，LLM主要用于场景识别）
        self.scene_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的场景分析师。请分析用户消息可能涉及的生活场景。

可选场景类别: {scene_categories}

请以JSON格式返回分析结果，格式如下：
{{
    "scene": "场景类别",
    "scene_confidence": 0.0-1.0之间的数值
}}

注意：
- 场景识别要结合上下文判断用户可能面临的情境
- 如果场景不明显，选择"other"并给予较低的confidence"""),
            ("human", "用户消息: {user_input}\n\n历史上下文: {context}\n\n识别到的情绪: {emotion_info}")
        ])
        
        self.json_parser = JsonOutputParser()
    
    def detect_crisis(self, text: str) -> Tuple[bool, List[str]]:
        """
        危机检测 - 使用关键词硬匹配
        返回: (是否危机, 匹配的关键词列表)
        """
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)
        
        is_crisis = len(matched_keywords) > 0
        
        if is_crisis:
            logger.warning(f"检测到危机信号! 匹配关键词: {matched_keywords}")
        
        return is_crisis, matched_keywords
    
    def _get_emotion_recognizer(self):
        """获取情绪识别器实例（延迟加载）"""
        if self.emotion_recognizer is None:
            try:
                self.emotion_recognizer = get_emotion_recognizer()
            except Exception as e:
                logger.warning(f"情绪识别器加载失败，将使用LLM备用方案: {e}")
        return self.emotion_recognizer
    
    def recognize_emotion(self, user_input: str) -> dict:
        """
        使用BERT模型进行情绪识别（优先）
        
        Returns:
            {
                "emotion": str,
                "intensity": float,
                "confidence": float,
                "risk_level": str,
                "risk_score": float
            }
        """
        recognizer = self._get_emotion_recognizer()
        if recognizer is None:
            return {
                "emotion": "calm",  # 默认使用calm（9种情绪之一）
                "intensity": 0.5,
                "confidence": 0.3,
                "risk_level": "low",
                "risk_score": 0.0
            }
        
        try:
            result = recognizer.recognize(user_input)
            logger.info(f"情绪识别结果: {result['emotion']} (强度: {result['intensity']:.2f}, "
                       f"风险: {result['risk_level']})")
            return result
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return {
                "emotion": "calm",  # 默认使用calm（9种情绪之一）
                "intensity": 0.5,
                "confidence": 0.3,
                "risk_level": "low",
                "risk_score": 0.0
            }
    
    async def analyze_scene(self, user_input: str, context: str = "", emotion_info: dict = None) -> dict:
        """
        使用LLM进行场景分析
        """
        try:
            emotion_str = f"{emotion_info.get('emotion', 'calm')} (强度: {emotion_info.get('intensity', 0.5):.2f})"
            
            chain = self.scene_prompt | self.llm | self.json_parser
            result = await chain.ainvoke({
                "user_input": user_input,
                "context": context,
                "emotion_info": emotion_str,
                "scene_categories": ", ".join(self.scene_categories)
            })
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"场景分析失败: {error_msg}")
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                logger.error(f"LLM API连接失败，请检查: base_url={settings.LLM_API_BASE}, api_key是否有效")
            return {
                "scene": "other",
                "scene_confidence": 0.3
            }
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        感知Agent主入口
        """
        user_input = state["user_input"]
        messages = state.get("messages", [])
        
        logger.info(f"感知Agent处理: {user_input[:50]}...")
        
        # 1. 情绪识别（优先使用BERT模型）
        emotion_result = self.recognize_emotion(user_input)
        
        # 2. 危机检测（关键词硬匹配 + 风险等级）
        is_crisis_keyword, matched_keywords = self.detect_crisis(user_input)
        risk_level = emotion_result.get("risk_level", "low")
        risk_score = emotion_result.get("risk_score", 0.0)
        
        # 如果风险等级为high，也标记为危机
        is_crisis = is_crisis_keyword or (risk_level == "high" and risk_score > 0.7)
        
        # 3. 构建上下文
        context = ""
        if messages:
            recent_messages = messages[-6:]  # 最近3轮对话
            context = "\n".join([
                f"{'用户' if msg.type == 'human' else '助手'}: {msg.content}"
                for msg in recent_messages
            ])
        
        # 4. LLM场景分析（情绪已由模型识别，LLM只负责场景）
        scene_result = await self.analyze_scene(user_input, context, emotion_result)
        
        # 5. 构建感知结果
        perception_result = {
            "emotion": {
                "emotion": emotion_result.get("emotion", "calm"),  # 默认calm
                "intensity": emotion_result.get("intensity", 0.5),
                "confidence": emotion_result.get("confidence", 0.5)
            },
            "crisis_detected": is_crisis,
            "crisis_keywords_matched": matched_keywords,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "scene_hint": scene_result.get("scene", "other"),
            "scene_confidence": scene_result.get("scene_confidence", 0.5)
        }
        
        logger.info(f"感知结果: emotion={perception_result['emotion']['emotion']} "
                   f"(强度: {perception_result['emotion']['intensity']:.2f}), "
                   f"crisis={is_crisis}, risk={risk_level}, "
                   f"scene={perception_result['scene_hint']}")
        
        # 6. 更新状态
        return {
            **state,
            "perception": perception_result,
            "is_crisis": is_crisis
        }


# 创建全局实例
perception_agent = PerceptionAgent()


async def run_perception(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await perception_agent(state)
