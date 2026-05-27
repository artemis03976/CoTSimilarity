#!/usr/bin/env bash
# =============================================================================
# run_ablation_lambda_router.sh —— DSPR Ablation: Lambda Router
#
# Swept parameter: lambda_router (router loss weight)
# Suggested search range: [0.1, 0.3, 0.5, 0.7, 0.9]
#
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXP_NAME="ablation_lambda_router"
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dspr_train.jsonl"
VAL_DATA="data/qwen/dspr_val.jsonl"
TEST_DATA="data/math_paired.jsonl"
BASE_CHECKPOINT="checkpoints/dspr_pipeline_test"   # baseline checkpoint for comparison

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# lambda_router sweep values
LAMBDA_VALUES=(0.1 0.5 1.0 1.5 2.0)

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${VAL_DATA}" ]; then
    echo "Error: train/validation sets do not exist. Run src/dspr_dataset/split_dataset.py first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    RUN_NAME="${EXP_NAME}_lr${LAMBDA}"
    CKPT_DIR="checkpoints/${RUN_NAME}"
    INFER_DIR="output/${RUN_NAME}"
    LATEST_CKPT=""

    echo ""
    echo "============================================================"
    echo "Ablation: lambda_router = ${LAMBDA}"
    echo "============================================================"

    # ---- Training ----
    echo "[1/3] Training (lambda_router=${LAMBDA}) ..."
    python scripts/train_dspr.py \
        --model_name "${MODEL_NAME}" \
        --train_data_path "${TRAIN_DATA}" \
        --val_data_path "${VAL_DATA}" \
        --lambda_router ${LAMBDA} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --output_path "${CKPT_DIR}"

    LATEST_CKPT=$(ls -td "${CKPT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    LATEST_CKPT="${LATEST_CKPT%/}"
    if [ -z "${LATEST_CKPT}" ]; then
        echo "Warning: training did not produce a checkpoint; skipping inference (lambda_router=${LAMBDA})"
        continue
    fi

    # ---- Inference ----
    echo "[2/3] Inference evaluation ..."
    mkdir -p "${INFER_DIR}"
    python scripts/dspr_inference_test.py \
        --checkpoint "${LATEST_CKPT}/dspr_trainable.pt" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${INFER_DIR}" \
        --data_path "${TEST_DATA}" \
        --device cuda

    # ---- Metrics ----
    echo "[3/3] Metric reporting ..."
    echo ""
    echo "--- lambda_router=${LAMBDA} results ---"
    python src/utils/evaluation/calculate_accuracy.py \
        "${INFER_DIR}/all_records.jsonl"

    if [ -f "${BASE_CHECKPOINT}/dspr_trainable.pt" ]; then
        echo ""
        echo "--- Delta vs Baseline ---"
        python src/utils/evaluation/calculate_accuracy.py \
            "${INFER_DIR}/all_records.jsonl" \
            --compare-file "output/qwen/all_records.jsonl"
    fi

    echo ""
    echo ">>> lambda_router=${LAMBDA} complete <<<"
    echo ""

done

echo "========== All lambda_router experiments complete =========="
