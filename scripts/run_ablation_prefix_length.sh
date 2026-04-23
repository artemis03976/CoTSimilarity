#!/usr/bin/env bash
# =============================================================================
# run_ablation_prefix_length.sh —— DCPR Ablation: Prefix Length
#
# 变动参数：prefix_length（prefix token 序列长度）
#
# 注意：prefix_length 变化会导致 checkpoint 形状改变，
#       必须重新训练，不能直接复用其他 prefix_length 的 checkpoint。
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
EXP_NAME="ablation_prefix_length"
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dcpr_train.jsonl"
VAL_DATA="data/qwen/dcpr_val.jsonl"
TEST_DATA="data/math_paired.jsonl"
BASE_CHECKPOINT="checkpoints/dcpr_pipeline_test"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# prefix_length 搜索列表
PREFIX_LENGTHS=(5 10 15 20 25 50 100)

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
for PLEN in "${PREFIX_LENGTHS[@]}"; do
    RUN_NAME="${EXP_NAME}_plen${PLEN}"
    CKPT_DIR="checkpoints/${RUN_NAME}"
    INFER_DIR="output/${RUN_NAME}"
    LATEST_CKPT=""

    echo ""
    echo "============================================================"
    echo "Ablation: prefix_length = ${PLEN}"
    echo "============================================================"

    # ---- 训练 ----
    echo "[1/3] 训练 (prefix_length=${PLEN}) ..."
    python scripts/train_dcpr.py \
        --model_name "${MODEL_NAME}" \
        --train_data_path "${TRAIN_DATA}" \
        --val_data_path "${VAL_DATA}" \
        --prefix_length ${PLEN} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --output_path "${CKPT_DIR}"

    LATEST_CKPT=$(ls -td "${CKPT_DIR}"/checkpoint-*/ 2>/dev/null | head -1 | tr -d '/\\')
    if [ -z "${LATEST_CKPT}" ]; then
        echo "警告: 训练未产出 checkpoint，跳过本次推理 (prefix_length=${PLEN})"
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
        --prefix_length ${PLEN} \
        --device cuda

    # ---- 指标统计 ----
    echo "[3/3] 指标统计 ..."
    echo ""
    echo "--- prefix_length=${PLEN} 结果 ---"
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
    echo ">>> prefix_length=${PLEN} 完成 <<<"
    echo ""

done
