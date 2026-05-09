# Emocare x LlamaFactory 训练配置

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `sft_qwen3_8b.yaml` | SFT LoRA 训练配置 |
| `dpo_qwen3_8b.yaml` | DPO LoRA 训练配置 |
| `dataset_info.json` | LlamaFactory 数据集注册表 |
| `emocare_sft_combined.jsonl` | SFT 数据 (3800 条) |
| `emocare_dpo.jsonl` | DPO 数据 (2769 条) |
| `train.sh` | 一键训练 + 评测 pipeline |
| `setup.sh` | 云服务器环境安装 |
| `evaluate.py` | 离线评测脚本 |

---

## 训练参数

### SFT

| 参数 | 值 |
|------|-----|
| batch_size | 4 × 4 = 16 |
| epochs | 3 |
| learning_rate | 2e-4 |
| LoRA rank/alpha | 64 / 128 |
| cutoff_len | 4096 |

### DPO

| 参数 | 值 |
|------|-----|
| batch_size | 2 × 8 = 16 |
| epochs | 2 |
| learning_rate | 5e-5 |
| beta | 0.3 |
| adapter | 加载 SFT LoRA |

---

## 使用

```bash
# 完整流程
bash train.sh

# 跳过评测
bash train.sh --skip-eval

# 单独评测
python evaluate.py --base_model <path> --dpo_adapter <path> \
    --dpo_data emocare_dpo.jsonl --sft_data emocare_sft_combined.jsonl
```

详见 `DEPLOY.md`
