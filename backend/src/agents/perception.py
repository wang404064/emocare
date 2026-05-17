"""
感知Agent (Perception Agent)
职责：
1. 情绪识别 (7分类 + 强度) - 使用BERT模型
2. 语音情绪识别 - 使用SenseVoice + 多模态融合
3. 危机检测 (分层：强信号/弱信号/正常，含误报过滤)
4. 场景识别 (简单分类)
5. 策略决定 (根据情绪+危机等级选择对话策略)
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
from ..models.audio_emotion import get_audio_recognizer, EMOTION_ORDER as AUDIO_EMOTION_ORDER


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
    
    def _format_history(self, messages: list, max_turns: int = 3) -> str:
        """取最近 N 轮对话，格式化为文本供 BERT 拼接（ERC 上下文感知）"""
        if not messages:
            return ""
        recent = messages[-(max_turns * 2):]  # 每轮 = 用户 + 助手
        lines = []
        for msg in recent:
            role = "用户" if msg.type == "human" else "助手"
            content = getattr(msg, "content", str(msg))
            lines.append(f"{role}: {content[:200]}")
        return "\n".join(lines)

    async def recognize_emotion(self, user_input: str, history_text: str = "") -> dict:
        """
        上下文感知情绪识别（在线程池中执行）。

        Args:
            user_input: 当前用户消息
            history_text: 最近 N 轮对话的格式化文本（已拼接好的）
        """
        import asyncio

        recognizer = self._get_emotion_recognizer()
        if recognizer is None:
            return self._default_emotion_result()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, recognizer.recognize, user_input, history_text
            )
            logger.info(
                f"情绪识别: {result['emotion']} (强度: {result['intensity']:.2f}), "
                f"风险: {result['risk_level']} (分数: {result['risk_score']:.2f}), "
                f"转变: {result.get('shift_label', 'N/A')}"
            )
            return result
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return self._default_emotion_result()

    def _default_emotion_result(self) -> dict:
        return {
            "emotion": "calm", "intensity": 0.5, "confidence": 0.3,
            "risk_level": "low", "risk_score": 0.0, "shift_label": "stable",
        }

    def _emotion_based_strategy(self, intensity: float) -> str:
        """按情绪强度返回策略（不含危机逻辑）"""
        if intensity > 0.7:
            return "empathy_first"
        elif intensity > 0.3:
            return "gentle_explore"
        return "normal_chat"

    async def recognize_emotion_from_audio(self, audio_data: bytes) -> dict:
        """
        语音情绪识别 + 多模态融合。

        流程:
          1. SenseVoice ASR → 转录文本 + 音频情绪向量
          2. BERT → 文本情绪向量
          3. 多模态加权融合
        """
        import asyncio

        audio_recognizer = get_audio_recognizer()
        loop = asyncio.get_event_loop()

        # 1. 音频情绪识别（在线程池中执行）
        audio_result = await loop.run_in_executor(
            None, audio_recognizer.recognize, audio_data
        )
        transcript = audio_result.get("text", "").strip()
        audio_vector = audio_result["audio_emotion_vector"]
        audio_events = audio_result.get("audio_events", [])
        audio_emotion = audio_result.get("audio_emotion", "calm")

        logger.info(
            f"音频识别: text='{transcript[:60]}...' "
            f"audio_emotion={audio_emotion}, "
            f"events={audio_events}"
        )

        if not transcript:
            return {
                "emotion_result": self._default_emotion_result(),
                "audio_emotion_details": {},
                "transcript": "",
            }

        # 2. 文本情绪识别（BERT）
        text_emotion_result = await self.recognize_emotion(transcript, history_text="")
        text_vector = self._emotion_dict_to_vector(text_emotion_result)

        # 3. 多模态融合
        fused_vector = self._multimodal_fusion(
            text_vector=text_vector,
            audio_vector=audio_vector,
            audio_events=audio_events,
            audio_emotion=audio_emotion,
        )

        # 4. 用融合后的向量替换 BERT 结果
        merged_result = self._vector_to_emotion_result(
            fused_vector,
            text_emotion_result.get("risk_level", "low"),
            text_emotion_result.get("risk_score", 0.0),
            text_emotion_result.get("shift_label", "stable"),
        )
        return {
            "emotion_result": merged_result,
            "audio_emotion_details": dict(zip(
                ["sadness", "anxiety", "anger", "loneliness",
                 "hopelessness", "calm", "joy"],
                audio_vector,
            )),
            "transcript": transcript,
        }

    def _emotion_dict_to_vector(self, result: dict) -> list:
        """从 BERT 结果中提取 7 维情绪向量"""
        details = result.get("emotion_details", {})
        order = [
            "sadness", "anxiety", "anger", "loneliness",
            "hopelessness", "calm", "joy"
        ]
        return [details.get(k, 0.0) for k in order]

    def _vector_to_emotion_result(
        self, vector: list, risk_level: str, risk_score: float, shift_label: str
    ) -> dict:
        """将 7 维向量转回与 BERT 一致的 emotion_result 格式"""
        order = [
            "sadness", "anxiety", "anger", "loneliness",
            "hopelessness", "calm", "joy"
        ]
        emotion_dict = dict(zip(order, vector))
        primary_idx = max(range(len(order)), key=lambda i: vector[i])
        primary_emotion = order[primary_idx]
        primary_intensity = vector[primary_idx]

        return {
            "emotion": primary_emotion,
            "intensity": primary_intensity,
            "confidence": min(primary_intensity * 1.2, 1.0),
            "emotion_details": emotion_dict,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "all_emotions": emotion_dict,
            "shift_label": shift_label,
        }

    def _multimodal_fusion(
        self,
        text_vector: list,
        audio_vector: list,
        audio_events: list,
        audio_emotion: str,
    ) -> list:
        """
        多模态情绪融合: 音频情绪 + 文本情绪 → 加权合并。

        规则:
          - 默认权重: text 0.7, audio 0.3
          - 音频事件（哭/笑/叹息）直接注入增量
          - 音频有高置信标签且与文本不一致时，提升音频权重到 0.4
        """
        text_w = 0.7
        audio_w = settings.AUDIO_EMOTION_WEIGHT  # default 0.3

        order = [
            "sadness", "anxiety", "anger", "loneliness",
            "hopelessness", "calm", "joy"
        ]

        # ── 1. 加权合并 ──────────────────────────────────────
        fused = [
            text_w * text_vector[i] + audio_w * audio_vector[i]
            for i in range(7)
        ]

        # ── 2. 音频事件增强（强力信号，直接叠加） ──────────────
        EVENT_BOOST = {
            "CRY":      {"sadness": 0.3, "hopelessness": 0.2},
            "LAUGHTER": {"joy": 0.3},
            "LAUGH":    {"joy": 0.25},
            "SOB":      {"sadness": 0.25},
            "SIGH":     {"sadness": 0.15},
            "SCREAM":   {"anxiety": 0.3, "anger": 0.2},
        }
        for event in audio_events:
            boosts = EVENT_BOOST.get(event, {})
            for emo_key, boost_val in boosts.items():
                idx = order.index(emo_key)
                fused[idx] = min(fused[idx] + boost_val, 1.0)

        # ── 3. 抑制无音频时的 calm 偏向 ───────────────────────
        if not audio_events and audio_emotion == "calm":
            # 音频确认平静 → 微调权重，但不过度压过文本
            pass

        # ── 4. 归一化 ─────────────────────────────────────────
        total = sum(fused)
        if total > 0:
            fused = [round(v / total, 4) for v in fused]
        else:
            fused = [round(v, 4) for v in fused]

        logger.info(
            f"多模态融合: text[{text_vector[0]:.2f},{text_vector[3]:.2f}...] "
            f"+ audio[{audio_vector[0]:.2f},{audio_vector[3]:.2f}...] "
            f"→ fused[{fused[0]:.2f},{fused[3]:.2f}...] "
            f"(events={audio_events})"
        )
        return fused

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
        感知Agent主入口 (v2 — ERC + 多层融合危机检测)。

        决策流程：
        1. 正则误报           → normal_chat
        2. 强关键词命中       → BERT risk gating
              ├─ BERT low    → 直接放行（不调 LLM）  ← NEW
              ├─ BERT high   → crisis_immediate（不调 LLM） ← NEW
              └─ BERT medium → LLM Crisis Judge 二次判断    ← 保留
        3. BERT 高风险(无关键词) → empathy_first_gentle_probe
        4. 弱信号             → empathy_first_gentle_probe
        5. 正常               → 按情绪强度选策略
        """
        user_input = state["user_input"]
        messages = state.get("messages", [])
        audio_data = state.get("audio_data")

        if audio_data:
            logger.info(f"感知Agent处理: 语音输入 ({len(audio_data)} bytes)")
        else:
            logger.info(f"感知Agent处理: {user_input[:50]}...")

        # ── 1. 上下文感知情绪识别（ERC / 多模态） ─────────────────────────
        audio_emotion = None
        if audio_data:
            # 语音路径：SenseVoice ASR → 转录文本 + 音频情绪
            # 2. BERT 文本情绪 → 3. 多模态融合
            fused = await self.recognize_emotion_from_audio(audio_data)
            emotion_result = fused["emotion_result"]
            audio_emotion = fused.get("audio_emotion_details", {})
            # 用转录文本作为 user_input（供危机检测 + 对话使用）
            user_input = fused.get("transcript", "") or user_input
        else:
            history_text = self._format_history(messages)
            emotion_result = await self.recognize_emotion(user_input, history_text)

        # ── 2. 关键词危机检测 ────────────────────────────────────────────
        is_strong_crisis, crisis_level, matched_keywords = self.detect_crisis(user_input)
        bert_risk = emotion_result.get("risk_level", "low")
        bert_risk_score = emotion_result.get("risk_score", 0.0)
        emotion_intensity = emotion_result.get("intensity", 0.5)
        shift_label = emotion_result.get("shift_label", "stable")

        # ── 3. 策略决策（多层融合） ───────────────────────────────────────
        judge_result = None
        judge_skipped = False  # 标记是否跳过了 LLM Judge

        if is_strong_crisis and matched_keywords:
            # ── BERT Risk Gating（优化5）─────────────────────────────────
            if bert_risk == "low" and bert_risk_score < 0.3:
                # BERT 高置信认定不是危机 → 跳过 LLM Judge
                logger.info(
                    f"BERT risk gate: low ({bert_risk_score:.2f}) → 跳过 LLM Judge, "
                    f"关键词={matched_keywords}"
                )
                is_crisis = False
                is_weak_crisis = False
                current_strategy = self._emotion_based_strategy(emotion_intensity)
                judge_result = {"level": "low", "brief": "BERT low confidence"}
                judge_skipped = True

            elif bert_risk == "high" and bert_risk_score > 0.7:
                # BERT 高置信认定是危机 → 跳过 LLM Judge, 直接进危机流程
                logger.warning(
                    f"BERT risk gate: high ({bert_risk_score:.2f}) → 直接危机流程, "
                    f"关键词={matched_keywords}"
                )
                is_crisis = True
                is_weak_crisis = False
                current_strategy = "crisis_immediate"
                judge_result = {"level": "high", "brief": "BERT high confidence"}
                judge_skipped = True

            else:
                # BERT medium → 调 LLM Judge
                judge_result = await self._crisis_judge(
                    user_input, matched_keywords, bert_risk, bert_risk_score
                )
                if judge_result["level"] == "high":
                    current_strategy = "crisis_immediate"
                    is_crisis = True
                    is_weak_crisis = False
                elif judge_result["level"] == "uncertain":
                    current_strategy = "empathy_first_gentle_probe"
                    is_crisis = False
                    is_weak_crisis = True
                else:
                    is_crisis = False
                    is_weak_crisis = False
                    current_strategy = self._emotion_based_strategy(emotion_intensity)

        elif bert_risk == "high" and bert_risk_score > 0.7:
            # BERT 高风险但无关键词 → 弱危机
            is_crisis = False
            is_weak_crisis = True
            current_strategy = "empathy_first_gentle_probe"

        elif crisis_level == "weak":
            # 弱信号 → 温和关注
            is_crisis = False
            is_weak_crisis = True
            current_strategy = "empathy_first_gentle_probe"

        else:
            is_crisis = False
            is_weak_crisis = False
            current_strategy = self._emotion_based_strategy(emotion_intensity)

        # ── 4. 情绪恶化检测 ──────────────────────────────────────────────
        if not is_crisis and shift_label == "up" and emotion_intensity > 0.6:
            logger.info(f"情绪持续恶化 (shift=up, intensity={emotion_intensity:.2f})"
                        f" → 升级为 empathy_first_gentle_probe")
            is_weak_crisis = True
            if current_strategy == "normal_chat":
                current_strategy = "empathy_first_gentle_probe"

        # ── 5. 构建场景分析上下文 ─────────────────────────────────────────
        context = ""
        if messages:
            recent_messages = messages[-6:]
            context = "\n".join([
                f"{'用户' if msg.type == 'human' else '助手'}: {msg.content}"
                for msg in recent_messages
            ])

        scene_result = await self.analyze_scene(user_input, context, emotion_result)

        # ── 6. 构建感知结果 ──────────────────────────────────────────────
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
            "risk_level": bert_risk,
            "risk_score": bert_risk_score,
            "scene_hint": scene_result.get("scene", "other"),
            "scene_confidence": scene_result.get("scene_confidence", 0.5),
            "judge_result": judge_result,
            "judge_skipped": judge_skipped,
            "shift_label": shift_label,
        }

        logger.info(
            f"感知结果: emotion={perception_result['emotion']['emotion']} "
            f"(强度: {perception_result['emotion']['intensity']:.2f}, 转变: {shift_label}), "
            f"crisis={is_crisis} (BERT risk={bert_risk}/{bert_risk_score:.2f}, "
            f"judge={'skipped' if judge_skipped else (judge_result['level'] if judge_result else 'N/A')}), "
            f"keywords={matched_keywords}, scene={perception_result['scene_hint']}, "
            f"strategy={current_strategy}"
        )

        return {
            **state,
            "user_input": user_input,       # 语音路径时为转录文本
            "perception": perception_result,
            "is_crisis": is_crisis,
            "current_strategy": current_strategy,
            "audio_emotion": audio_emotion,  # 语音情绪向量（仅语音路径）
            "audio_data": None,              # 用完即清，避免重复处理
        }


# 创建全局实例
perception_agent = PerceptionAgent()


async def run_perception(state: AgentState) -> AgentState:
    """LangGraph节点函数"""
    return await perception_agent(state)
