# model.py
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn


class EmotionRiskClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)

        # 情绪强度回归头 (9 维连续值)
        self.emotion_regressor = nn.Linear(config.hidden_size, 9)
        # 风险等级分类头 (3 类: low=0, medium=1, high=2)
        self.risk_classifier = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  # ← 保留这个参数
        emotion_labels=None,
        risk_labels=None,
        return_dict=True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        emotion_logits = self.emotion_regressor(cls_output)  # (batch, 9)
        risk_logits = self.risk_classifier(cls_output)       # (batch, 3)

        loss = None
        if emotion_labels is not None and risk_labels is not None:
            # 情绪：回归 → MSE
            loss_emotion = nn.MSELoss()(emotion_logits, emotion_labels.float())
            # 风险：分类 → CrossEntropy
            loss_risk = nn.CrossEntropyLoss()(risk_logits, risk_labels)
            loss = loss_emotion + 1.5 * loss_risk

        if not return_dict:
            output = (emotion_logits, risk_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "risk_logits": risk_logits,
        }
