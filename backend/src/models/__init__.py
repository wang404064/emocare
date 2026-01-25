"""
情绪识别器模块
"""
from .emotion_recognizer import EmotionRecognizer, get_emotion_recognizer
from .model import EmotionRiskClassifier

__all__ = ["EmotionRecognizer", "get_emotion_recognizer", "EmotionRiskClassifier"]
