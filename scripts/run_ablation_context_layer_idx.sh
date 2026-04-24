#!/usr/bin/env bash
# =============================================================================
# run_ablation_context_layer_idx.sh —— DCPR Ablation: Context Layer Index
#
# 变动参数：context_layer_idx（从 Transformer 第几层提取 hidden states 作为 h_Q）
# 搜索范围建议：[10, 12, 15, 18, 20, 24]（针对 Qwen2.5-Math-7B，层编号 0-27）
#
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
EXP_NAME="ablation_context_layer_idx"
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dcpr_train.jsonl"
VAL_DATA="data/qwen/dcpr_val.jsonl"
TEST_DATA="data/math_paired.jsonl"
BASE_CHECKPOINT="checkpoints/dcpr_pipeline_test"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# context_layer_idx 搜索列表（请根据实际模型层数调整）
CONTEXT_LAYER_IDX_LIST=(10 12 15 18 20 24)

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${VAL_DATA}" ]; then
    echo "错误: 训练/验证集不存在，请先运行 split_dcpr_dataset.py"
    exit 1
fi

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
for LAYER_IDX in "${CONTEXT_LAYER_IDX_LIST[@]}"; do
    RUN_NAME="${EXP_NAME}_layer${LAYER_IDX}"
    CKPT_DIR="checkpoints/${RUN_NAME}"
    INFER_DIR="output/${RUN_NAME}"
    LATEST_CKPT=""

    echo ""
    echo "============================================================"
    echo "Ablation: context_layer_idx = ${LAYER_IDX}"
    echo "============================================================"

    # ---- 训练 ----
    echo "[1/3] 训练 (context_layer_idx=${LAYER_IDX}) ..."
    python scripts/train_dcpr.py \
        --model_name "${MODEL_NAME}" \
        --train_data_path "${TRAIN_DATA}" \
        --val_data_path "${VAL_DATA}" \
        --context_layer_idx ${LAYER_IDX} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --output_path "${CKPT_DIR}"

    LATEST_CKPT=$(ls -td "${CKPT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    LATEST_CKPT="${LATEST_CKPT%/}"
    if [ -z "${LATEST_CKPT}" ]; then
        echo "警告: 训练未产出 checkpoint，跳过本次推理 (context_layer_idx=${LAYER_IDX})"
        continue
    fi

    # ---- 推理 ----
    echo "[2/3] 推理测试 ..."
    mkdir -p "${INFER_DIR}"
    python src/dcpr_inference_test.py \
        --checkpoint "${LATEST_CKPT}/dcpr_trainable.pt" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${INFER_DIR}" \
        --data_path "${TEST_DATA}" \
        --context_layer_idx ${LAYER_IDX} \
        --device cuda

    # ---- 指标统计 ----
    echo "[3/3] 指标统计 ..."
    echo ""
    echo "--- context_layer_idx=${LAYER_IDX} 结果 ---"
    python scripts/calculate_accuracy.py \
        "${INFER_DIR}/all_records.jsonl"

    if [ -f "${BASE_CHECKPOINT}/dcpr_trainable.pt" ]; then
        echo ""
        echo "--- Delta vs Baseline ---"
        python scripts/calculate_accuracy.py \
            "${INFER_DIR}/all_records.jsonl" \
            --compare-file "output/qwen/all_records.jsonl"
    fi

    echo ""
    echo ">>> context_layer_idx=${LAYER_IDX} 完成 <<<"
    echo ""

done

echo "========== 全部 context_layer_idx 实验完成 =========="
