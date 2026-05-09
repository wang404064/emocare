#!/bin/bash
# =====================================================================
# LlamaFactory 环境安装脚本 — 修复版
# 在云主机（AutoDL / 恒源云 / Vast.ai）上运行一次即可
#
# 修复内容：
#   1. conda activate 在非交互 shell 中失效 → 改用 conda run
#   2. 添加 evaluate.py 所需依赖（peft / rouge-score / bert_score）
#   3. 增加 PyTorch 版本检测（不重复安装）
#   4. 添加 HuggingFace 镜像源（国内云主机加速下载）
# =====================================================================

set -e

echo "=================================================="
echo "  EmoCare — 环境安装"
echo "=================================================="

# ── 配置 ──────────────────────────────────────────────
CONDA_ENV="llamafactory"
PYTHON_VER="3.11"
CUDA_VERSION="121"       # 121 = CUDA 12.1（AutoDL A100 默认）

# HuggingFace 镜像（国内云主机需要，海外注释掉下面这行）
export HF_ENDPOINT="https://hf-mirror.com"

# ── Step 1: 创建 conda 环境 ────────────────────────────
echo ""
echo "[1/5] 创建 conda 环境 (${CONDA_ENV})..."

# 初始化 conda（非交互 shell 必须）
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    # AutoDL 特殊路径
    if [ -d "/root/miniconda3" ]; then
        source "/root/miniconda3/etc/profile.d/conda.sh"
    else
        echo "⚠️  未找到 conda.sh，尝试直接使用 conda..."
    fi
fi

# 检查环境是否已存在
if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
    echo "  环境 ${CONDA_ENV} 已存在，跳过创建"
else
    conda create -n "${CONDA_ENV}" python="${PYTHON_VER}" -y
    echo "  ✓ 环境创建成功"
fi

# ── Step 2: 安装 PyTorch ───────────────────────────────
echo ""
echo "[2/5] 检查 / 安装 PyTorch (CUDA ${CUDA_VERSION})..."

TORCH_INSTALLED=$(conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

if [ -z "${TORCH_INSTALLED}" ]; then
    echo "  安装 PyTorch 2.1.2 + CUDA 12.1..."
    conda run -n "${CONDA_ENV}" pip install \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}" \
        --no-cache-dir
else
    echo "  PyTorch 已安装: ${TORCH_INSTALLED}，跳过"
fi

# ── Step 3: 安装 LlamaFactory ─────────────────────────
echo ""
echo "[3/5] 安装 LlamaFactory..."

# 优先从源码安装（支持最新功能 + DPO）
LF_INSTALLED=$(conda run -n "${CONDA_ENV}" llamafactory-cli --help 2>/dev/null | head -1 || echo "")

if [ -z "${LF_INSTALLED}" ]; then
    # 方法 A：源码安装（推荐，确保 DPO/SFT 功能完整）
    LF_DIR="/root/LLaMA-Factory"
    if [ ! -d "${LF_DIR}" ]; then
        echo "  克隆 LlamaFactory 仓库..."
        git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LF_DIR}"
    fi
    conda run -n "${CONDA_ENV}" pip install -e "${LF_DIR}[torch,metrics]" --no-cache-dir
else
    echo "  LlamaFactory 已安装，跳过"
fi

# ── Step 4: 安装评测依赖 ───────────────────────────────
echo ""
echo "[4/5] 安装评测依赖（evaluate.py 需要）..."

conda run -n "${CONDA_ENV}" pip install \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    transformers>=4.41.0 \
    sentencepiece \
    protobuf \
    --no-cache-dir

# ROUGE / BERTScore（可选，跳过不影响核心评测）
conda run -n "${CONDA_ENV}" pip install \
    rouge-score \
    bert-score \
    jieba \
    --no-cache-dir || echo "  ⚠️  可选评测库安装失败，跳过（不影响核心指标）"

# ── Step 5: 验证安装 ───────────────────────────────────
echo ""
echo "[5/5] 验证安装..."

conda run -n "${CONDA_ENV}" python - << 'PYEOF'
import sys
print(f"Python:     {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA 可用:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        print(f"显存:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError as e:
    print(f"❌ PyTorch 未安装: {e}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("❌ transformers 未安装")

try:
    import peft
    print(f"PEFT:       {peft.__version__}")
except ImportError:
    print("⚠️  peft 未安装（evaluate.py 的 LoRA adapter 加载需要）")

try:
    import llamafactory
    print(f"LlamaFactory: 已安装 ✓")
except ImportError:
    # 有时 llamafactory 不暴露 __version__，用 CLI 验证
    print("LlamaFactory: (通过 CLI 验证)")

PYEOF

echo ""
echo "=================================================="
echo "✅ 安装完成！"
echo "=================================================="
echo ""
echo "激活环境:    conda activate ${CONDA_ENV}"
echo "开始训练:    conda run -n ${CONDA_ENV} bash train.sh"
echo "单独评测:    conda run -n ${CONDA_ENV} python evaluate.py --help"
echo ""
echo "⚠️  重要提醒："
echo "  1. 确认 train.sh 中 MODEL_PATH 和 DATA_ROOT 已修改为实际路径"
echo "  2. 国内云主机如需下载模型，已设置 HF_ENDPOINT=https://hf-mirror.com"
echo "     下载命令: huggingface-cli download Qwen/Qwen3-8B --local-dir /root/autodl-tmp/models/Qwen/Qwen3-8B"
echo ""
