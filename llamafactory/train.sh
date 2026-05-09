#!/bin/bash
# =====================================================================
# LlamaFactory 一键训练 + 自动评测脚本
# 用法: bash train.sh [--skip-eval] [--skip-base]
#
# ★ 唯一需要修改的地方：第 29-30 行的 MODEL_PATH 和 DATA_ROOT
#    YAML 文件通过 envsubst 自动注入路径，不用手动改
#
# 执行顺序：
#   步骤 1  SFT LoRA 训练
#   步骤 2  DPO LoRA 训练
#   步骤 3  合并导出 LoRA 权重
#   步骤 4  自动评测（Win Rate + Reward Margin + PPL）
# =====================================================================

set -e

# ====================== ★ 只改这两行 ★ ======================
MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-8B"
DATA_ROOT="/root/autodl-tmp/my_project"
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====================== 参数解析 ======================
SKIP_EVAL=false
SKIP_BASE=false
for arg in "$@"; do
    case $arg in
        --skip-eval) SKIP_EVAL=true ;;
        --skip-base) SKIP_BASE=true ;;
    esac
done

# ====================== 路径定义（以下不需要改）======================

SFT_CONFIG="${SCRIPT_DIR}/sft_qwen3_8b.yaml"
DPO_CONFIG="${SCRIPT_DIR}/dpo_qwen3_8b.yaml"
EVALUATE_SCRIPT="${SCRIPT_DIR}/evaluate.py"

SFT_OUTPUT="${DATA_ROOT}/output/sft"
DPO_OUTPUT="${DATA_ROOT}/output/dpo"
FINAL_OUTPUT="${DATA_ROOT}/output/emocare_final"
EVAL_OUTPUT="${DATA_ROOT}/output/eval_results.json"

SFT_LOG="${DATA_ROOT}/output/logs/sft_train.log"
DPO_LOG="${DATA_ROOT}/output/logs/dpo_train.log"
EVAL_LOG="${DATA_ROOT}/output/logs/eval.log"

# SFT adapter 路径（DPO 训练时加载，评测时加载）
SFT_ADAPTER="${SFT_OUTPUT}"

# =====================================================================
echo "=================================================="
echo "  EmoCare — 训练 + 评测 Pipeline"
echo "=================================================="
echo "模型:       ${MODEL_PATH}"
echo "数据根目录: ${DATA_ROOT}"
echo "脚本目录:   ${SCRIPT_DIR}"
echo "跳过评测:   ${SKIP_EVAL}"
echo ""

# ====================== 预检 ======================
echo "[预检] 检查配置文件..."
[ ! -f "${SFT_CONFIG}" ] && echo "❌ SFT配置不存在: ${SFT_CONFIG}" && exit 1
[ ! -f "${DPO_CONFIG}" ] && echo "❌ DPO配置不存在: ${DPO_CONFIG}" && exit 1

echo "[预检] 检查模型路径..."
[ ! -d "${MODEL_PATH}" ] && echo "❌ 模型路径不存在: ${MODEL_PATH}" && exit 1

echo "[预检] 检查数据集..."
[ ! -f "${SCRIPT_DIR}/emocare_sft_combined.jsonl" ] && echo "❌ SFT数据不存在: ${SCRIPT_DIR}/emocare_sft_combined.jsonl" && exit 1
[ ! -f "${SCRIPT_DIR}/emocare_dpo.jsonl" ]         && echo "❌ DPO数据不存在: ${SCRIPT_DIR}/emocare_dpo.jsonl" && exit 1

echo "[预检] 检查 dataset_info.json..."
[ ! -f "${SCRIPT_DIR}/dataset_info.json" ] && echo "❌ dataset_info.json 不存在: ${SCRIPT_DIR}/dataset_info.json" && exit 1

echo "[预检] 检查 llamafactory..."
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ llamafactory-cli 未找到，请先执行 bash setup.sh"
    exit 1
fi

mkdir -p "${SFT_OUTPUT}" "${DPO_OUTPUT}" "${FINAL_OUTPUT}" "${DATA_ROOT}/output/logs"

echo "[预检] ✅ 全部通过"
echo ""

# =====================================================================
# 步骤 1：SFT 训练
# =====================================================================
echo ""
echo "=================================================="
echo "步骤 1/4  SFT LoRA 训练"
echo "=================================================="
echo "日志: ${SFT_LOG}"
echo ""

# envsubst 注入 MODEL_PATH / DATA_ROOT / SCRIPT_DIR
export MODEL_PATH DATA_ROOT SCRIPT_DIR
envsubst '${MODEL_PATH} ${DATA_ROOT} ${SCRIPT_DIR}' < "${SFT_CONFIG}" > /tmp/sft_run.yaml

llamafactory-cli train /tmp/sft_run.yaml 2>&1 | tee "${SFT_LOG}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ SFT 训练失败！日志: ${SFT_LOG}"
    exit 1
fi

echo ""
echo "✅ SFT 训练完成！"
# LlamaFactory 输出目录结构：output_dir/checkpoint-xxx/ 或 output_dir/
ls -lh "${SFT_OUTPUT}/" 2>/dev/null | head -5 || true

# 智能探测 SFT adapter 实际路径
if [ -d "${SFT_OUTPUT}/final" ]; then
    SFT_ADAPTER="${SFT_OUTPUT}/final"
elif [ -d "${SFT_OUTPUT}/checkpoint-"* ]; then
    # 取最后一个 checkpoint
    SFT_ADAPTER=$(ls -d "${SFT_OUTPUT}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
fi
echo "SFT adapter: ${SFT_ADAPTER}"
echo ""

# =====================================================================
# 步骤 2：DPO 训练
# =====================================================================
echo ""
echo "=================================================="
echo "步骤 2/4  DPO LoRA 训练"
echo "=================================================="
echo "日志: ${DPO_LOG}"
echo ""

export MODEL_PATH DATA_ROOT SCRIPT_DIR SFT_ADAPTER
envsubst '${MODEL_PATH} ${DATA_ROOT} ${SCRIPT_DIR} ${SFT_ADAPTER}' < "${DPO_CONFIG}" > /tmp/dpo_run.yaml

llamafactory-cli train /tmp/dpo_run.yaml 2>&1 | tee "${DPO_LOG}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ DPO 训练失败！日志: ${DPO_LOG}"
    exit 1
fi

echo ""
echo "✅ DPO 训练完成！"
ls -lh "${DPO_OUTPUT}/" 2>/dev/null | head -5 || true
echo ""

# =====================================================================
# 步骤 3：合并导出 LoRA 权重
# =====================================================================
echo ""
echo "=================================================="
echo "步骤 3/4  合并 LoRA 权重并导出"
echo "=================================================="
echo ""

# 智能探测 DPO adapter 路径
DPO_ADAPTER="${DPO_OUTPUT}"
if [ -d "${DPO_OUTPUT}/final" ]; then
    DPO_ADAPTER="${DPO_OUTPUT}/final"
elif [ -d "${DPO_OUTPUT}/checkpoint-"* ]; then
    DPO_ADAPTER=$(ls -d "${DPO_OUTPUT}/checkpoint-"* 2>/dev/null | sort -V | tail -1)
fi
echo "DPO adapter: ${DPO_ADAPTER}"

llamafactory-cli export \
    --model_name_or_path "${MODEL_PATH}" \
    --adapter_name_or_path "${DPO_ADAPTER}" \
    --template qwen3_nothink \
    --finetuning_type lora \
    --export_dir "${FINAL_OUTPUT}" \
    --export_size 4 \
    --export_legacy_format false

if [ $? -ne 0 ]; then
    echo "❌ 合并导出失败"
    exit 1
fi

echo ""
echo "✅ 模型导出完成: ${FINAL_OUTPUT}"
ls -lh "${FINAL_OUTPUT}/" | head -10
echo ""

# =====================================================================
# 步骤 4：自动评测
# =====================================================================
if [ "${SKIP_EVAL}" = "true" ]; then
    echo "⏭  跳过评测（--skip-eval）"
else
    echo ""
    echo "=================================================="
    echo "步骤 4/4  自动评测"
    echo "=================================================="
    echo "日志: ${EVAL_LOG}"
    echo ""

    EVAL_CMD="python ${EVALUATE_SCRIPT} \
        --base_model    ${MODEL_PATH} \
        --sft_adapter   ${SFT_ADAPTER} \
        --dpo_adapter   ${DPO_ADAPTER} \
        --dpo_data      ${SCRIPT_DIR}/emocare_dpo.jsonl \
        --sft_data      ${SCRIPT_DIR}/emocare_sft_combined.jsonl \
        --n_samples     200 \
        --n_ppl         100 \
        --output        ${EVAL_OUTPUT} \
        --device        cuda"

    if [ "${SKIP_BASE}" = "true" ]; then
        EVAL_CMD="${EVAL_CMD} --skip_base"
    fi

    echo "运行: ${EVAL_CMD}"
    echo ""

    eval ${EVAL_CMD} 2>&1 | tee "${EVAL_LOG}"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo ""
        echo "⚠️  评测脚本报错，但训练已完成。"
        echo "   手动运行: ${EVAL_CMD}"
    else
        echo ""
        echo "✅ 评测完成！结果: ${EVAL_OUTPUT}"
    fi
fi

# =====================================================================
# 完成
# =====================================================================
echo ""
echo "=================================================="
echo "🎉 全部流程完成！"
echo "=================================================="
echo ""
echo "最终模型: ${FINAL_OUTPUT}"
echo "评测结果: ${EVAL_OUTPUT}"
echo "训练日志: ${DATA_ROOT}/output/logs/"
echo ""
echo "快速推理测试:"
echo "  llamafactory-cli chat \\"
echo "    --model_name_or_path ${FINAL_OUTPUT} \\"
echo "    --template qwen3"
echo ""
echo "下载模型:"
echo "  cd ${DATA_ROOT}/output && tar -czvf emocare_final.tar.gz emocare_final/"
echo ""
