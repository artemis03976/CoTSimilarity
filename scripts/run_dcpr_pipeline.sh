#!/usr/bin/env bash
# =============================================================================
# run_dcpr_pipeline.sh —— DCPR 完整流程测试脚本
#
# 功能：数据划分 → 训练 → 推理 → 指标统计
#
# 使用前提（确认以下文件存在）：
#   - data/qwen/dcpr_train.jsonl
#   - data/qwen/dcpr_val.jsonl
#   - data/math_paired.jsonl（用于推理测试）
#
# 输出：
#   checkpoints/<RUN_NAME>/         —— 训练产物（dual_prefix + router）
#   output/<RUN_NAME>/all_records.jsonl  —— DCPR 推理结果
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 配置区（按需修改）
# ---------------------------------------------------------------------------
RUN_NAME="dcpr_pipeline"

# 训练
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dcpr_train.jsonl"
VAL_DATA="data/qwen/dcpr_val.jsonl"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

# 推理测试
TEST_DATA="data/math_paired.jsonl"
INFER_OUTPUT_DIR="output/${RUN_NAME}"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# ---------------------------------------------------------------------------
# Step 1: 训练 DCPR
# ---------------------------------------------------------------------------
echo "========== [1/3] 开始训练 DCPR =========="
python scripts/train_dcpr.py \
    --model_name "${MODEL_NAME}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --output_path "${CHECKPOINT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 2: 推理测试
# ---------------------------------------------------------------------------
# 取最后一个 epoch 的 checkpoint
LATEST_CKPT=$(ls -td "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | head -1)
LATEST_CKPT="${LATEST_CKPT%/}"
if [ -z "${LATEST_CKPT}" ]; then
    echo "错误: 未找到训练 checkpoint，训练可能失败。"
    exit 1
fi

echo "========== [2/3] DCPR 推理测试 =========="
echo "使用 checkpoint: ${LATEST_CKPT}"
mkdir -p "${INFER_OUTPUT_DIR}"

python src/dcpr_inference_test.py \
    --checkpoint "${LATEST_CKPT}/dcpr_trainable.pt" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${INFER_OUTPUT_DIR}" \
    --data_path "${TEST_DATA}" \
    --device cuda
echo ""

# ---------------------------------------------------------------------------
# Step 3: 指标统计  
# ---------------------------------------------------------------------------
echo "========== [3/3] 指标统计 =========="
echo ""
echo "--- DCPR 推理结果 ---"
python scripts/calculate_accuracy.py \
    "${INFER_OUTPUT_DIR}/all_records.jsonl"

echo ""
echo "--- 对比 Baseline（如已有 baseline 结果）---"
if [ -f "output/base_all_records.jsonl" ]; then
    python scripts/calculate_accuracy.py \
        "${INFER_OUTPUT_DIR}/all_records.jsonl" \
        --compare-file "output/qwen/all_records.jsonl"
else
    echo "未找到 Qwen 结果文件 (output/qwen/all_records.jsonl)，跳过对比。"
fi

echo ""
echo "========== 流程完成 =========="
echo "Checkpoint: ${LATEST_CKPT}"
echo "推理结果: ${INFER_OUTPUT_DIR}/all_records.jsonl"
