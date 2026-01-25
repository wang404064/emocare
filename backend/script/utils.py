# utils.py
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmotionRiskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["user_utterance"]
        emotion = item["emotion_intensities"]  # list of 9 floats
        risk = item["risk_level"]  # int: 0,1,2

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),  # ← 新增这一行！
            "emotion_labels": torch.tensor(emotion, dtype=torch.float),
            "risk_labels": torch.tensor(risk, dtype=torch.long)
        }


def postprocess_prediction(emotion_logits, risk_logits):
    """
    后处理模型原始输出
    Args:
        emotion_logits: (9,) tensor of raw regression outputs
        risk_logits: (3,) tensor of classification logits
    Returns:
        dict with clamped emotions and risk label
    """
    # 情绪强度限制在 [0, 1]
    emotion_clamped = torch.clamp(emotion_logits, 0.0, 1.0).tolist()

    # 风险等级：取 argmax
    risk_pred = torch.argmax(risk_logits).item()

    # 可选：返回概率分布
    risk_probs = torch.softmax(risk_logits, dim=-1).tolist()

    return {
        "emotion_intensities": emotion_clamped,
        "risk_level": risk_pred,
        "risk_probs": risk_probs
    }


# 情绪标签顺序
EMOTION_ORDER = [
    "sadness", "anxiety", "anger", "loneliness",
    "shame_guilt", "hopelessness", "hope", "calm", "joy"
]