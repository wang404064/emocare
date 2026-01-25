"""
情绪识别器
基于BERT模型进行情绪识别和风险检测
"""
import torch
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
from transformers import AutoTokenizer

# 从本地导入模型类
from .model import EmotionRiskClassifier

# 情绪标签顺序（与训练一致）- 9种情绪，不进行映射
EMOTION_ORDER = [
    "sadness",      # 悲伤
    "anxiety",      # 焦虑
    "anger",        # 愤怒
    "loneliness",   # 孤独
    "shame_guilt",  # 羞耻/内疚
    "hopelessness", # 绝望
    "hope",         # 希望
    "calm",         # 平静
    "joy"           # 喜悦
]

# 风险等级映射
RISK_LEVELS = ["low", "medium", "high"]


class EmotionRecognizer:
    """情绪识别器 - 基于BERT模型"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化情绪识别器
        
        Args:
            model_path: 模型路径，默认为预定义路径或从配置读取
            device: 设备（cuda/cpu），默认自动选择
        """
        if model_path is None:
            # 尝试从配置读取
            try:
                from ..core.config import settings
                model_path = settings.EMOTION_MODEL_PATH
            except:
                # 默认模型路径（如果配置中没有，需要用户指定）
                raise ValueError("请配置 EMOTION_MODEL_PATH 或传入 model_path 参数")
        
        if device is None:
            # 尝试从配置读取
            try:
                from ..core.config import settings
                device = settings.EMOTION_MODEL_DEVICE
            except:
                pass
        
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 延迟加载模型
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return
        
        try:
            logger.info(f"加载情绪识别器模型: {self.model_path}")
            
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 加载模型（使用本地导入的 EmotionRiskClassifier）
            self._model = EmotionRiskClassifier.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()
            
            self._loaded = True
            logger.info(f"情绪识别器模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"加载情绪识别器模型失败: {e}")
            raise
    
    def _postprocess_prediction(self, emotion_logits: torch.Tensor, risk_logits: torch.Tensor) -> Dict:
        """
        后处理模型输出
        
        Args:
            emotion_logits: (9,) 情绪强度logits
            risk_logits: (3,) 风险等级logits
        
        Returns:
            处理后的结果字典
        """
        # 情绪强度限制在 [0, 1]
        emotion_intensities = torch.clamp(emotion_logits, 0.0, 1.0).tolist()
        
        # 风险等级：取argmax
        risk_level = torch.argmax(risk_logits).item()
        
        # 风险概率分布
        risk_probs = torch.softmax(risk_logits, dim=-1).tolist()
        
        return {
            "emotion_intensities": emotion_intensities,
            "risk_level": risk_level,
            "risk_probs": risk_probs
        }
    
    def recognize(self, text: str) -> Dict:
        """
        识别文本中的情绪
        
        Args:
            text: 输入文本
        
        Returns:
            {
                "emotion": str,           # 主要情绪类别（9种情绪之一）
                "intensity": float,       # 主要情绪强度 0-1
                "confidence": float,      # 置信度
                "emotion_details": dict,  # 所有9种情绪的强度详情
                "risk_level": str,        # 风险等级 (low/medium/high)
                "risk_score": float,      # 风险分数
                "all_emotions": dict      # 所有9种情绪的强度（与emotion_details相同）
            }
        """
        if not self._loaded:
            self._load_model()
        
        if not text or not text.strip():
            return self._get_default_result()
        
        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self._model(**inputs)
                result = self._postprocess_prediction(
                    outputs["emotion_logits"][0].cpu(),
                    outputs["risk_logits"][0].cpu()
                )
            
            # 构建情绪强度字典（9种情绪）
            emotion_dict = dict(zip(EMOTION_ORDER, result["emotion_intensities"]))
            
            # 找到主要情绪（强度最高的）
            primary_emotion_idx = max(range(len(EMOTION_ORDER)), 
                                    key=lambda i: result["emotion_intensities"][i])
            primary_emotion = EMOTION_ORDER[primary_emotion_idx]
            primary_intensity = result["emotion_intensities"][primary_emotion_idx]
            
            # 计算置信度（使用主要情绪强度）
            confidence = min(primary_intensity * 1.2, 1.0)
            
            # 风险等级
            risk_level_str = RISK_LEVELS[result["risk_level"]]
            risk_score = result["risk_probs"][result["risk_level"]]
            
            return {
                "emotion": primary_emotion,  # 直接使用9种情绪之一，不映射
                "intensity": primary_intensity,
                "confidence": confidence,
                "emotion_details": emotion_dict,  # 所有9种情绪的强度
                "risk_level": risk_level_str,
                "risk_score": risk_score,
                "all_emotions": emotion_dict  # 与emotion_details相同，保持兼容性
            }
            
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """返回默认结果"""
        default_emotions = {emo: 0.0 for emo in EMOTION_ORDER}
        default_emotions["calm"] = 0.5  # 默认平静情绪
        return {
            "emotion": "calm",  # 默认使用calm而不是neutral
            "intensity": 0.5,
            "confidence": 0.3,
            "emotion_details": default_emotions,
            "risk_level": "low",
            "risk_score": 0.0,
            "all_emotions": default_emotions
        }


# 创建全局实例（延迟加载）
_emotion_recognizer: Optional[EmotionRecognizer] = None


def get_emotion_recognizer() -> EmotionRecognizer:
    """获取情绪识别器单例"""
    global _emotion_recognizer
    if _emotion_recognizer is None:
        _emotion_recognizer = EmotionRecognizer()
    return _emotion_recognizer
