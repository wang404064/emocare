# 情绪识别器训练脚本
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from model import EmotionRiskClassifier
from utils import EmotionRiskDataset, postprocess_prediction


# === 配置 ===
MODEL_NAME = "google-bert/bert-base-chinese"
TRAIN_DATA = "data/train.json"
VAL_DATA = "data/val.json"
OUTPUT_DIR = "emotion_risk_model_v1"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EmotionRiskClassifier.from_pretrained(MODEL_NAME)
model.to(device)

train_dataset = EmotionRiskDataset(TRAIN_DATA, tokenizer, MAX_LENGTH)
val_dataset = EmotionRiskDataset(VAL_DATA, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# === 训练循环 ===
best_val_mae = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)  # ← 新增
        emotion_labels = batch["emotion_labels"].to(device)
        risk_labels = batch["risk_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # ← 传入
            emotion_labels=emotion_labels,
            risk_labels=risk_labels
        )

        loss = outputs["loss"]
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)

    # === 验证 ===
    model.eval()
    val_emotion_preds = []
    val_emotion_labels = []
    val_risk_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)  # ← 验证时也需要获取标签
            risk_labels = batch["risk_labels"].to(device)  # ← 验证时也需要获取标签

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids  # ← 传入
            )
            emotion_logits = outputs["emotion_logits"]
            risk_logits = outputs["risk_logits"]

            # 收集情绪预测（用于 MAE）
            val_emotion_preds.append(emotion_logits.cpu())
            val_emotion_labels.append(emotion_labels.cpu())

            # 风险准确率
            preds = torch.argmax(risk_logits, dim=1)
            val_risk_correct += (preds == risk_labels).sum().item()
            val_total += risk_labels.size(0)

    # 计算 MAE（平均绝对误差）
    all_preds = torch.cat(val_emotion_preds, dim=0)
    all_labels = torch.cat(val_emotion_labels, dim=0)
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    risk_acc = val_risk_correct / val_total

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val MAE: {mae:.4f}, Risk Acc: {risk_acc:.4f}")

    # === 早停 & 保存最佳模型 ===
    if mae < best_val_mae:
        best_val_mae = mae
        patience_counter = 0
        # 保存模型和 tokenizer
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  → New best MAE! Saved to {OUTPUT_DIR}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training finished.")