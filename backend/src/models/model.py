# model.py — EmotionRiskClassifier v2
# 升级点：
#   1. 共享中间表示层 (LayerNorm → 512d → GELU)，替代单层 Linear
#   2. 新增情绪转变检测头 (emotion_shift_head)：下降/持平/上升，3 分类
#   3. Huber Loss 替代 MSE（抗离群点）
#   4. 情绪强度加权（hopelessness 2×, shame_guilt 1.5×, loneliness 1.5×）
#   5. 风险序数回归意识（通过加权 CE 实现）
#   6. 向后兼容：shift_labels 为 None 时自动跳过
#
# 参考论文：
#   - EmoShiftNet (Frontiers AI 2025): shift-aware multi-task + Focal Loss
#   - Opinion-BERT (Nature 2025): multi-task emotion + risk joint training

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class EmotionRiskClassifier(BertPreTrainedModel):
    """
    多任务情绪识别 + 风险评估 + 情绪转变检测。

    Heads:
        emotion_regressor : 7 维情绪强度 [0,1]
        risk_classifier   : 3 级风险 (low/medium/high)
        emotion_shift_head: 3 类情绪转变 (down/stable/up)
    """

    # 7 种情绪对应的训练损失权重（list，运行时按 device 创建 tensor）
    # hopelessness, loneliness 临床意义更大，给予更高权重
    # sadness, anxiety, anger, loneliness, hopelessness, calm, joy
    _emotion_weight_values = [1.2, 1.3, 0.8, 1.5, 2.0, 0.6, 0.6]
    # 3 级风险对应的训练损失权重  # low, medium, high
    _risk_weight_values = [0.5, 1.0, 2.0]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        # ── 共享中间表示（替代原单层 Linear） ──────────────────────────
        self.shared_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── 三个任务头 ────────────────────────────────────────────────
        self.emotion_regressor = nn.Linear(512, 7)     # 情绪强度回归
        self.risk_classifier = nn.Linear(512, 3)        # 风险等级分类
        self.emotion_shift_head = nn.Linear(512, 3)     # 情绪转变分类 (NEW)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        emotion_labels=None,
        risk_labels=None,
        shift_labels=None,          # (NEW) 情绪转变标签
        return_dict=True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        shared_repr = self.shared_head(cls_output)  # (batch, 512)

        emotion_logits = self.emotion_regressor(shared_repr)   # (batch, 7)
        risk_logits = self.risk_classifier(shared_repr)        # (batch, 3)
        shift_logits = self.emotion_shift_head(shared_repr)    # (batch, 3)

        loss = None
        if emotion_labels is not None and risk_labels is not None:
            # ── 1. 情绪强度回归：Huber Loss + 类别加权 ──────────────────
            loss_emotion = 0.0
            device = emotion_logits.device
            weights = torch.tensor(self._emotion_weight_values, dtype=torch.float32, device=device)

            for i in range(7):
                loss_i = F.huber_loss(
                    emotion_logits[:, i],
                    emotion_labels[:, i].float(),
                    delta=0.2,
                    reduction="none"
                )
                loss_emotion = loss_emotion + (weights[i] * loss_i).mean()
            loss_emotion = loss_emotion / 7.0

            # ── 2. 风险分类：加权 CrossEntropy ──────────────────────────
            risk_weights = torch.tensor(self._risk_weight_values, dtype=torch.float32, device=device)
            loss_risk = F.cross_entropy(
                risk_logits, risk_labels, weight=risk_weights
            )

            loss = loss_emotion + 1.5 * loss_risk

            # ── 3. 情绪转变（辅助任务） ──────────────────────────────────
            if shift_labels is not None:
                loss_shift = F.cross_entropy(shift_logits, shift_labels)
                loss = loss + 0.5 * loss_shift

        if not return_dict:
            output = (emotion_logits, risk_logits, shift_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        result = {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "risk_logits": risk_logits,
            "shift_logits": shift_logits,
        }
        return result
