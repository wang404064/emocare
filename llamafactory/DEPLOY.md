# EmoCare 云服务器部署 — 完整流程

## 核心原则

> **所有配置在本地改，上传即跑。不在服务器上改一个字。**

## 本地只需改一个文件

打开 `llamafactory/train.sh`，改第 29-30 行：

```bash
MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-8B"   # 模型下载位置
DATA_ROOT="/root/emocare"                             # 项目上传位置
```

| 平台 | MODEL_PATH 示例 |
|------|----------------|
| AutoDL | `/root/autodl-tmp/models/Qwen/Qwen3-8B` |
| 恒源云 | `/hy-tmp/models/Qwen/Qwen3-8B` |
| Vast.ai | `/workspace/models/Qwen/Qwen3-8B` |
| 自己机器 | `/data/models/Qwen3-8B` |

YAML 配置文件不用动，路径由 `envsubst` 自动注入。

---

## 完整操作步骤

### 步骤 1：本地打包（跳过 git 和大文件）

```bash
cd E:/project/agent/emo

# 只传 llamafactory 目录，其他不需要
tar -czvf emocare.tar.gz llamafactory/
```

### 步骤 2：上传到服务器

```bash
# AutoDL（替换 IP 和端口）
scp -P 42151 emocare.tar.gz root@connect.westb.seetacloud.com:/root/

# 恒源云
scp -P 22 emocare.tar.gz root@123.45.67.89:/hy-tmp/

# 或用 AutoDL 的网盘上传面板，直接拖拽
```

### 步骤 3：SSH 登服务器，解压

```bash
ssh -p 42151 root@connect.westb.seetacloud.com

# 解压
mkdir -p /root/emocare
tar -xzvf /root/emocare.tar.gz -C /root/emocare/

# 看下结构对不对
ls /root/emocare/llamafactory/
# 应该有: sft_qwen3_8b.yaml  dpo_qwen3_8b.yaml  train.sh  setup.sh
#          evaluate.py  dataset_info.json  emocare_sft_combined.jsonl  emocare_dpo.jsonl
```

### 步骤 4：下载基座模型（只需第一次）

```bash
# AutoDL 用镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Qwen3-8B（约15GB，需要5-10分钟）
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir /root/autodl-tmp/models/Qwen/Qwen3-8B
```

### 步骤 5：一键安装环境（只需第一次）

```bash
cd /root/emocare/llamafactory
bash setup.sh
# 大约 10-15 分钟，会装 conda 环境 + PyTorch + LlamaFactory + peft
```

### 步骤 6：开始训练

```bash
cd /root/emocare/llamafactory

# 完整流程：SFT → DPO → 导出 → 评测
conda run -n llamafactory bash train.sh

# 或跳过评测（省 GPU 时间）
conda run -n llamafactory bash train.sh --skip-eval

# 跳过 base 模型评测（只测 SFT vs DPO）
conda run -n llamafactory bash train.sh --skip-base
```

---

## 各步骤耗时参考（A100 80G）

| 步骤 | 内容 | 时间 |
|------|------|------|
| 模型下载 | Qwen3-8B 15GB | ~10min |
| setup.sh | 装环境 | ~15min |
| SFT 训练 | 3800条 × 3epoch | ~2.5h |
| DPO 训练 | 4792条 × 2epoch | ~1.5h |
| 合并导出 | LoRA → 完整权重 | ~2min |
| 评测 | 200 样本，3个模型 | ~20min |
| **合计** | | **~4.5h** |

---

## 后续迭代（改代码/数据后更新）

```bash
# 本地改完代码后，只传 llamafactory 目录
cd E:/project/agent/emo
tar -czvf emocare_update.tar.gz llamafactory/
scp -P 42151 emocare_update.tar.gz root@connect.westb.seetocloud.com:/root/
# 服务器上
tar -xzvf /root/emocare_update.tar.gz -C /root/emocare/
```

---

## 常见问题

### Q: 怎么看训练有没有崩？
```bash
# 看训练日志
tail -50 /root/emocare/output/logs/sft_train.log
tail -50 /root/emocare/output/logs/dpo_train.log

# 看 loss 是否下降
grep "loss" /root/emocare/output/logs/sft_train.log | tail -20
```

### Q: 怎么单独跑评测（不重新训练）？
```bash
conda run -n llamafactory python evaluate.py \
    --base_model /root/autodl-tmp/models/Qwen/Qwen3-8B \
    --sft_adapter /root/emocare/output/sft \
    --dpo_adapter /root/emocare/output/dpo \
    --dpo_data /root/emocare/llamafactory/emocare_dpo.jsonl \
    --sft_data /root/emocare/llamafactory/emocare_sft_combined.jsonl \
    --n_samples 200 \
    --output /root/emocare/output/eval_results.json
```

### Q: A10 24G 显存不够怎么办？
在 `train.sh` 的 SFT 步骤后，手动把 `/tmp/sft_run.yaml` 里的：
```yaml
bf16: true    → 改为 fp16: true
per_device_train_batch_size: 4  → 改为 2
gradient_accumulation_steps: 4  → 改为 8
cutoff_len: 4096                → 改为 2048
```
