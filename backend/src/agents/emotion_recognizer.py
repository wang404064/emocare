"""
情绪识别器 (Emotion Recognizer)
基于Transformer模型进行情绪识别
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from ..core.config import settings


class EmotionRecognizer:
    """基于Transformer的情绪识别器"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化情绪识别器
        
        Args:
            model_path: 模型路径，如果为None则使用配置中的路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"情绪识别器使用设备: {self.device}")
        
        # 情绪标签映射（与训练脚本保持一致）
        self.emotion_labels = {
            0: "joy",
            1: "sadness",
            2: "anger",
            3: "fear",
            4: "surprise",
            5: "disgust",
            6: "trust",
            7: "anticipation"
        }
        
        # 反向映射：标签名 -> 索引
        self.label_to_idx = {v: k for k, v in self.emotion_labels.items()}
        
        # 如果没有neutral，添加它（映射到索引8）
        if "neutral" not in self.label_to_idx:
            self.label_to_idx["neutral"] = 8
            self.emotion_labels[8] = "neutral"
        
        # 模型路径
        if model_path is None:
            model_path = getattr(settings, 'EMOTION_RECOGNIZER_MODEL_PATH', None)
        
        if model_path is None:
            # 默认路径：使用预训练的中文模型
            model_path = "hfl/chinese-roberta-wwm-ext"
            logger.warning(f"未指定情绪识别器模型路径，使用默认模型: {model_path}")
            self.model = None
            self.tokenizer = None
            self._load_default_model(model_path)
        else:
            self.model_path = Path(model_path)
            self._load_model()
    
    def _load_default_model(self, model_name: str):
        """加载默认的预训练模型（用于测试，实际需要训练后的模型）"""
        try:
            logger.info(f"加载默认模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 注意：这里使用默认模型，实际应该使用训练后的模型
            # 如果没有训练好的模型，可以先用这个作为占位符
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.emotion_labels)
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("默认模型加载完成（注意：这是未训练的模型，仅用于测试）")
        except Exception as e:
            logger.error(f"加载默认模型失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            if not self.model_path.exists():
                logger.warning(f"模型路径不存在: {self.model_path}，使用默认模型")
                self._load_default_model("hfl/chinese-roberta-wwm-ext")
                return
            
            logger.info(f"加载情绪识别器模型: {self.model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                num_labels=len(self.emotion_labels)
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("情绪识别器模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}，使用默认模型")
            self._load_default_model("hfl/chinese-roberta-wwm-ext")
    
    def recognize(self, text: str, max_length: int = 128) -> Dict:
        """
        识别文本中的情绪
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
        
        Returns:
            {
                "emotion": str,  # 情绪类别
                "intensity": float,  # 情绪强度 0-1
                "confidence": float,  # 识别置信度 0-1
                "probabilities": dict  # 各类别的概率分布
            }
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("情绪识别器模型未加载，返回默认值")
            return {
                "emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.3,
                "probabilities": {}
            }
        
        try:
            # 编码文本
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 计算概率分布
                probs = torch.softmax(logits, dim=-1)
                probs_np = probs.cpu().numpy()[0]
                
                # 获取预测类别和置信度
                pred_idx = torch.argmax(logits, dim=-1).item()
                confidence = probs_np[pred_idx]
                
                # 获取情绪标签
                emotion = self.emotion_labels.get(pred_idx, "neutral")
                
                # 计算情绪强度（使用置信度作为强度指标）
                # 如果置信度很高，强度也高；如果置信度低，强度也低
                intensity = float(confidence)
                
                # 构建概率分布字典
                probabilities = {
                    self.emotion_labels.get(i, "unknown"): float(probs_np[i])
                    for i in range(len(probs_np))
                }
                
                return {
                    "emotion": emotion,
                    "intensity": intensity,
                    "confidence": float(confidence),
                    "probabilities": probabilities
                }
                
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return {
                "emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.3,
                "probabilities": {}
            }
    
    def recognize_batch(self, texts: list, max_length: int = 128) -> list:
        """
        批量识别情绪
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
        
        Returns:
            情绪识别结果列表
        """
        return [self.recognize(text, max_length) for text in texts]


# 创建全局实例（延迟加载）
_emotion_recognizer: Optional[EmotionRecognizer] = None


def get_emotion_recognizer() -> EmotionRecognizer:
    """获取情绪识别器实例（单例模式）"""
    global _emotion_recognizer
    if _emotion_recognizer is None:
        _emotion_recognizer = EmotionRecognizer()
    return _emotion_recognizer
