#!/usr/bin/env bash
# =============================================================================
# run_ablation_prefix_length.sh —— DSPR Ablation: Prefix Length
#
# Swept parameter: prefix_length (prefix token sequence length)
#
# Note: changing prefix_length changes checkpoint tensor shapes, so each
# setting must be retrained and cannot directly reuse another prefix length.
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXP_NAME="ablation_prefix_length"
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dspr_train.jsonl"
VAL_DATA="data/qwen/dspr_val.jsonl"
TEST_DATA="data/math_paired.jsonl"
BASE_CHECKPOINT="checkpoints/dspr_pipeline_test"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# prefix_length sweep values
PREFIX_LENGTHS=(5 10 15 20 25 50 100)

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
for PLEN in "${PREFIX_LENGTHS[@]}"; do
    RUN_NAME="${EXP_NAME}_plen${PLEN}"
    CKPT_DIR="checkpoints/${RUN_NAME}"
    INFER_DIR="output/${RUN_NAME}"
    LATEST_CKPT=""

    echo ""
    echo "============================================================"
    echo "Ablation: prefix_length = ${PLEN}"
    echo "============================================================"

    # ---- Training ----
    echo "[1/3] Training (prefix_length=${PLEN}) ..."
    python scripts/train.py dspr \
        --model_name "${MODEL_NAME}" \
        --train_data_path "${TRAIN_DATA}" \
        --val_data_path "${VAL_DATA}" \
        --prefix_length ${PLEN} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --output_path "${CKPT_DIR}"

    LATEST_CKPT=$(ls -td "${CKPT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    LATEST_CKPT="${LATEST_CKPT%/}"
    if [ -z "${LATEST_CKPT}" ]; then
        echo "Warning: training did not produce a checkpoint; skipping inference (prefix_length=${PLEN})"
        continue
    fi

    # ---- Inference ----
    echo "[2/3] Inference evaluation ..."
    mkdir -p "${INFER_DIR}"
    python scripts/inference.py \
        --baseline dspr \
        --checkpoint "${LATEST_CKPT}/dspr_trainable.pt" \
        --model_name "${MODEL_NAME}" \
        --output_dir "${INFER_DIR}" \
        --data_path "${TEST_DATA}" \
        --prefix_length ${PLEN} \
        --device cuda

    # ---- Metrics ----
    echo "[3/3] Metric reporting ..."
    echo ""
    echo "--- prefix_length=${PLEN} results ---"
    python src/utils/calculate_accuracy.py \
        "${INFER_DIR}/all_records.jsonl"

    if [ -f "${BASE_CHECKPOINT}/dspr_trainable.pt" ]; then
        echo ""
        echo "--- Delta vs Baseline ---"
        python src/utils/calculate_accuracy.py \
            "${INFER_DIR}/all_records.jsonl" \
            --compare-file "output/qwen-2.5/greedy/all_records.jsonl"
    fi

    echo ""
    echo ">>> prefix_length=${PLEN} complete <<<"
    echo ""

done
