"""
感知Agent (Perception Agent)
职责：
1. 情绪识别 (9分类 + 强度) - 使用BERT模型
2. 危机检测 (分层：强信号/弱信号/正常，含误报过滤)
3. 场景识别 (简单分类)
4. 策略决定 (根据情绪+危机等级选择对话策略)
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
    """感知Agent - 负责情绪识别、危机检测、场景分析和策略决定"""

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=0.3,
            max_tokens=512,
            timeout=30.0,
            max_retries=2
        )
        self.crisis_keywords = settings.CRISIS_KEYWORDS
        self.weak_crisis_keywords = getattr(settings, 'WEAK_CRISIS_KEYWORDS', [])
        self.false_positive_patterns = getattr(settings, 'CRISIS_FALSE_POSITIVE_PATTERNS', [])
        self.emotion_categories = settings.EMOTION_CATEGORIES
        self.scene_categories = settings.SCENE_CATEGORIES

        self.emotion_recognizer = None

        # 危机二次判断 Prompt（轻量，只输出 JSON，不生成回复）
        self.crisis_judge_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是危机评估专家，只负责判断用户消息是否存在真实的自杀/自伤风险。

关键区分标准：
- 口语夸张（"烦死了""困死了""累死了"）= 无风险
- 日常抱怨（"周一不想上班""这工作活不下去了"）= 无风险，用户只是在吐槽
- 抽象表达（"有时候觉得活着没意思""人为什么要活着"）= 低风险，可能是哲学思考或轻度情绪
- 具体计划/行动（"我买了药""站在阳台上""不想再醒来了"）= 高风险
- 告别倾向（"对不起所有人""以后不会麻烦你们了"）= 高风险

请只输出JSON，不要加任何其他文字：
{{"level": "high|low|uncertain", "brief": "一句话判断依据"}}

- high: 存在真实的自伤意图或具体计划
- low: 口语夸张、日常吐槽、哲学思考，无实际危险
- uncertain: 信息不足，无法确定"""),
            ("human", "用户消息: {user_input}\n匹配到的敏感词: {keywords}\nBERT风险评估: {risk_level} (置信度: {risk_score:.2f})")
        ])

        # 场景识别Prompt
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
    
    def _is_false_positive(self, text: str) -> bool:
        """
        检测是否是口语化误报（正则匹配）。
        覆盖"想死你了""开心死了""笑死我了"等日常夸张表达。
        Returns: True 表示是误报，应跳过危机处理
        """
        for pattern in self.false_positive_patterns:
            if re.search(pattern, text):
                logger.debug(f"口语化误报匹配: pattern={pattern}, text={text[:30]}")
                return True
        return False
    
    def detect_crisis(self, text: str) -> Tuple[bool, str, List[str]]:
        """
        分层危机检测
        
        Returns:
            (is_strong_crisis, crisis_level, matched_keywords)
            - is_strong_crisis: 是否为强危机信号（需进入危机分支）
            - crisis_level: "strong" | "weak" | "none"
            - matched_keywords: 匹配到的关键词列表
        """
        text_lower = text.lower()
        matched_keywords = []
        
        # 首先检测误报（口语化表达）
        if self._is_false_positive(text):
            logger.debug(f"口语化表达，跳过危机检测: {text[:30]}")
            return False, "none", []
        
        # 强信号检测
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)
        
        if matched_keywords:
            logger.warning(f"检测到强危机信号! 匹配关键词: {matched_keywords}")
            return True, "strong", matched_keywords
        
        # 弱信号检测
        weak_matched = []
        for keyword in self.weak_crisis_keywords:
            if keyword in text_lower:
                weak_matched.append(keyword)
        
        if weak_matched:
            logger.info(f"检测到弱危机信号: {weak_matched}")
            return False, "weak", weak_matched
        
        return False, "none", []
    
    def _get_emotion_recognizer(self):
        """获取情绪识别器实例（延迟加载）"""
        if self.emotion_recognizer is None:
            try:
                self.emotion_recognizer = get_emotion_recognizer()
            except Exception as e:
                logger.warning(f"情绪识别器加载失败，将使用LLM备用方案: {e}")
        return self.emotion_recognizer
    
    async def recognize_emotion(self, user_input: str) -> dict:
        """
        使用BERT模型进行情绪识别（在线程池中执行，避免阻塞事件循环）

        Returns:
            {
                "emotion": str,
                "intensity": float,
                "confidence": float,
                "risk_level": str,
                "risk_score": float
            }
        """
        import asyncio

        recognizer = self._get_emotion_recognizer()
        if recognizer is None:
            return self._default_emotion_result()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, recognizer.recognize, user_input)
            logger.info(f"情绪识别结果: {result['emotion']} (强度: {result['intensity']:.2f}, "
                       f"风险: {result['risk_level']})")
            return result
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return self._default_emotion_result()

    def _default_emotion_result(self) -> dict:
        return {
            "emotion": "calm",
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
    
    async def _crisis_judge(self, user_input: str, keywords: list[str], risk_level: str, risk_score: float) -> dict:
        """
        LLM 危机二次判断。
        只在关键词命中强信号时调用，用于区分"烦死了"和"我想结束生命"。
        返回 {"level": "high|low|uncertain", "brief": "..."}
        API 失败时安全回退为 high（宁可过度反应，不可漏报）。
        """
        try:
            chain = self.crisis_judge_prompt | self.llm | self.json_parser
            result = await chain.ainvoke({
                "user_input": user_input,
                "keywords": ", ".join(keywords) if keywords else "无",
                "risk_level": risk_level,
                "risk_score": risk_score
            })
            level = result.get("level", "high")  # 默认 high 安全
            brief = result.get("brief", "")
            logger.info(f"Crisis Judge: level={level}, brief={brief}")
            return {"level": level, "brief": brief}
        except Exception as e:
            logger.warning(f"Crisis Judge 调用失败，安全回退为 high: {e}")
            return {"level": "high", "brief": "judge unavailable"}

    async def __call__(self, state: AgentState) -> AgentState:
        """
        感知Agent主入口。

        决策流程（自上而下）：
        1. 口语误报           → normal_chat（直接放行）
        2. 强关键词命中       → Crisis Judge 二次判断
              ├─ high         → crisis_immediate（硬编码安全模板）
              ├─ uncertain    → empathy_first_gentle_probe（LLM 生成 + safety 约束）
              └─ low          → 按情绪强度走正常策略
        3. BERT 高风险(无关键词) → empathy_first_gentle_probe（LLM 生成 + safety 约束）
        4. 弱信号             → empathy_first_gentle_probe（温和关注）
        5. 正常               → 按情绪强度选择策略
        """
        user_input = state["user_input"]
        messages = state.get("messages", [])

        logger.info(f"感知Agent处理: {user_input[:50]}...")

        # 1. 情绪识别
        emotion_result = await self.recognize_emotion(user_input)

        # 2. 关键词危机检测
        is_strong_crisis, crisis_level, matched_keywords = self.detect_crisis(user_input)
        risk_level = emotion_result.get("risk_level", "low")
        risk_score = emotion_result.get("risk_score", 0.0)
        emotion_intensity = emotion_result.get("intensity", 0.5)

        # ── 3. 策略决策 ──────────────────────────────────────────────────
        judge_result = None  # Crisis Judge 的判决结果

        if is_strong_crisis and matched_keywords:
            # 强关键词命中 → LLM 二次判断
            judge_result = await self._crisis_judge(
                user_input, matched_keywords, risk_level, risk_score
            )
            if judge_result["level"] == "high":
                current_strategy = "crisis_immediate"
                is_crisis = True
                is_weak_crisis = False
            elif judge_result["level"] == "uncertain":
                current_strategy = "empathy_first_gentle_probe"
                is_crisis = False
                is_weak_crisis = True
            else:  # low — LLM 认为不是危机
                is_crisis = False
                is_weak_crisis = False
                if emotion_intensity > 0.7:
                    current_strategy = "empathy_first"
                elif emotion_intensity > 0.3:
                    current_strategy = "gentle_explore"
                else:
                    current_strategy = "normal_chat"

        elif risk_level == "high" and risk_score > 0.7:
            # BERT 高风险但无关键词
            is_crisis = False
            is_weak_crisis = True
            current_strategy = "empathy_first_gentle_probe"

        elif crisis_level == "weak":
            # 弱信号
            is_crisis = False
            is_weak_crisis = True
            current_strategy = "empathy_first_gentle_probe"

        else:
            # 正常
            is_crisis = False
            is_weak_crisis = False
            if emotion_intensity > 0.7:
                current_strategy = "empathy_first"
            elif emotion_intensity > 0.3:
                current_strategy = "gentle_explore"
            else:
                current_strategy = "normal_chat"

        # ── 4. 构建上下文 ───────────────────────────────────────────────
        context = ""
        if messages:
            recent_messages = messages[-6:]
            context = "\n".join([
                f"{'用户' if msg.type == 'human' else '助手'}: {msg.content}"
                for msg in recent_messages
            ])

        # ── 5. 场景分析 ─────────────────────────────────────────────────
        scene_result = await self.analyze_scene(user_input, context, emotion_result)

        # ── 6. 构建感知结果 ─────────────────────────────────────────────
        perception_result = {
            "emotion": {
                "emotion": emotion_result.get("emotion", "calm"),
                "intensity": emotion_result.get("intensity", 0.5),
                "confidence": emotion_result.get("confidence", 0.5)
            },
            "crisis_detected": is_crisis,
            "crisis_level": crisis_level,
            "is_weak_crisis": is_weak_crisis,
            "crisis_keywords_matched": matched_keywords,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "scene_hint": scene_result.get("scene", "other"),
            "scene_confidence": scene_result.get("scene_confidence", 0.5),
            "judge_result": judge_result  # 供 conversation agent 参考
        }

        logger.info(f"感知结果: emotion={perception_result['emotion']['emotion']} "
                   f"(强度: {perception_result['emotion']['intensity']:.2f}), "
                   f"crisis={is_crisis} (judge={judge_result['level'] if judge_result else 'N/A'}), "
                   f"keywords={matched_keywords}, risk={risk_level}, "
                   f"scene={perception_result['scene_hint']}, strategy={current_strategy}")

        return {
            **state,
            "perception": perception_result,
            "is_crisis": is_crisis,
            "current_strategy": current_strategy
        }


# 创建全局实例
perception_agent = PerceptionAgent()


async def run_perception(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await perception_agent(state)
