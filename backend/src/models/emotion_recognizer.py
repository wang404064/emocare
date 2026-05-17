"""
情绪识别器 (v2)
基于 BERT 的上下文感知情绪识别 + 风险检测 + 情绪转变检测。

v2 升级：
  - recognize() 新增 history_text 参数，拼接最近 3 轮对话历史做上下文感知推理
  - 后处理提取 shift_logits（情绪转变方向）
  - max_length 128 → 256 以适应拼接后的上下文

参考论文：
  - EmoShiftNet (Frontiers AI 2025): 上下文 + 情绪转变检测
  - BERT Context-Aware ERC (ACM 2025): 上下文拼接提升压抑型情绪识别
"""
import torch
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
from transformers import AutoTokenizer

from .model import EmotionRiskClassifier

EMOTION_ORDER = [
    "sadness", "anxiety", "anger", "loneliness",
    "hopelessness", "calm", "joy"
]

RISK_LEVELS = ["low", "medium", "high"]

# 情绪转变方向标签
SHIFT_LABELS = ["down", "stable", "up"]


class EmotionRecognizer:
    """情绪识别器 — 上下文感知 BERT 推理"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if model_path is None:
            try:
                from ..core.config import settings
                model_path = settings.EMOTION_MODEL_PATH
            except Exception:
                raise ValueError("请配置 EMOTION_MODEL_PATH 或传入 model_path 参数")

        if device is None:
            try:
                from ..core.config import settings
                device = settings.EMOTION_MODEL_DEVICE
            except Exception:
                pass

        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._prev_emotion = None  # 上一轮情绪，用于 shift 计算

    def _load_model(self):
        if self._loaded:
            return
        try:
            logger.info(f"加载情绪识别器模型: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = EmotionRiskClassifier.from_pretrained(
                str(self.model_path), ignore_mismatched_sizes=True
            )
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"情绪识别器加载成功，设备: {self.device}")
        except Exception as e:
            logger.error(f"加载情绪识别器失败: {e}")
            raise

    def _postprocess_prediction(
        self,
        emotion_logits: torch.Tensor,
        risk_logits: torch.Tensor,
        shift_logits: Optional[torch.Tensor] = None
    ) -> Dict:
        emotion_intensities = torch.clamp(emotion_logits, 0.0, 1.0).tolist()
        risk_level = torch.argmax(risk_logits).item()
        risk_probs = torch.softmax(risk_logits, dim=-1).tolist()

        result = {
            "emotion_intensities": emotion_intensities,
            "risk_level": risk_level,
            "risk_probs": risk_probs,
        }
        if shift_logits is not None:
            shift_idx = torch.argmax(shift_logits).item()
            result["shift_label"] = SHIFT_LABELS[shift_idx]
            result["shift_probs"] = torch.softmax(shift_logits, dim=-1).tolist()
        else:
            result["shift_label"] = "stable"
            result["shift_probs"] = [0.0, 1.0, 0.0]

        return result

    def recognize(self, text: str, history_text: str = "") -> Dict:
        """
        上下文感知情绪识别。

        Args:
            text: 当前用户消息
            history_text: 最近 N 轮对话的格式化文本。
                          格式: "用户: ...\n助手: ...\n用户: ..."

        Returns:
            { emotion, intensity, confidence, emotion_details,
              risk_level, risk_score, all_emotions, shift_label }
        """
        if not self._loaded:
            self._load_model()

        if not text or not text.strip():
            return self._get_default_result()

        try:
            # ── 上下文拼接（ERC） ──────────────────────────────────────
            if history_text:
                full_text = f"[历史]\n{history_text}\n[当前]\n{text}"
                max_len = 256
            else:
                full_text = text
                max_len = 128

            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # ── 推理 ───────────────────────────────────────────────────
            with torch.no_grad():
                outputs = self._model(**inputs)
                shift_logits_tensor = outputs.get("shift_logits")
                result = self._postprocess_prediction(
                    outputs["emotion_logits"][0].cpu(),
                    outputs["risk_logits"][0].cpu(),
                    shift_logits_tensor[0].cpu() if shift_logits_tensor is not None else None,
                )

            emotion_dict = dict(zip(EMOTION_ORDER, result["emotion_intensities"]))
            primary_idx = max(range(len(EMOTION_ORDER)),
                              key=lambda i: result["emotion_intensities"][i])
            primary_emotion = EMOTION_ORDER[primary_idx]
            primary_intensity = result["emotion_intensities"][primary_idx]
            confidence = min(primary_intensity * 1.2, 1.0)

            risk_level_str = RISK_LEVELS[result["risk_level"]]
            risk_score = result["risk_probs"][result["risk_level"]]

            # ── 情绪转变（fallback：与上一轮对比） ─────────────────────
            shift_label = result.get("shift_label", "stable")
            if self._prev_emotion is not None and shift_label == "stable":
                prev_idx = EMOTION_ORDER.index(self._prev_emotion) if self._prev_emotion in EMOTION_ORDER else -1
                if prev_idx >= 0:
                    intensity_delta = primary_intensity - result["emotion_intensities"][prev_idx]
                    if intensity_delta > 0.15:
                        shift_label = "up"
                    elif intensity_delta < -0.15:
                        shift_label = "down"

            self._prev_emotion = primary_emotion

            return {
                "emotion": primary_emotion,
                "intensity": primary_intensity,
                "confidence": confidence,
                "emotion_details": emotion_dict,
                "risk_level": risk_level_str,
                "risk_score": risk_score,
                "all_emotions": emotion_dict,
                "shift_label": shift_label,
            }

        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return self._get_default_result()

    def _get_default_result(self) -> Dict:
        default_emotions = {emo: 0.0 for emo in EMOTION_ORDER}
        default_emotions["calm"] = 0.5
        return {
            "emotion": "calm", "intensity": 0.5, "confidence": 0.3,
            "emotion_details": default_emotions, "risk_level": "low",
            "risk_score": 0.0, "all_emotions": default_emotions,
            "shift_label": "stable",
        }


# 创建全局实例（延迟加载）
_emotion_recognizer: Optional[EmotionRecognizer] = None


def get_emotion_recognizer() -> EmotionRecognizer:
    """获取情绪识别器单例"""
    global _emotion_recognizer
    if _emotion_recognizer is None:
        _emotion_recognizer = EmotionRecognizer()
    return _emotion_recognizer
