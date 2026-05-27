#!/usr/bin/env bash
# =============================================================================
# run_ablation_context_layer_idx.sh —— DSPR Ablation: Context Layer Index
#
# Swept parameter: context_layer_idx (Transformer layer used to extract h_Q)
# Suggested search range: [10, 12, 15, 18, 20, 24] for Qwen2.5-Math-7B layers 0-27.
#
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXP_NAME="ablation_context_layer_idx"
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dspr_train.jsonl"
VAL_DATA="data/qwen/dspr_val.jsonl"
TEST_DATA="data/math_paired.jsonl"
BASE_CHECKPOINT="checkpoints/dspr_pipeline_test"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# context_layer_idx sweep values; adjust for the actual number of model layers.
CONTEXT_LAYER_IDX_LIST=(10 12 15 18 20 24)

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
for LAYER_IDX in "${CONTEXT_LAYER_IDX_LIST[@]}"; do
    RUN_NAME="${EXP_NAME}_layer${LAYER_IDX}"
    CKPT_DIR="checkpoints/${RUN_NAME}"
    INFER_DIR="output/${RUN_NAME}"
    LATEST_CKPT=""

    echo ""
    echo "============================================================"
    echo "Ablation: context_layer_idx = ${LAYER_IDX}"
    echo "============================================================"

    # ---- Training ----
    echo "[1/3] Training (context_layer_idx=${LAYER_IDX}) ..."
    python scripts/train_dspr.py \
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
        echo "Warning: training did not produce a checkpoint; skipping inference (context_layer_idx=${LAYER_IDX})"
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
        --context_layer_idx ${LAYER_IDX} \
        --device cuda

    # ---- Metrics ----
    echo "[3/3] Metric reporting ..."
    echo ""
    echo "--- context_layer_idx=${LAYER_IDX} results ---"
    python src/utils/calculate_accuracy.py \
        "${INFER_DIR}/all_records.jsonl"

    if [ -f "${BASE_CHECKPOINT}/dspr_trainable.pt" ]; then
        echo ""
        echo "--- Delta vs Baseline ---"
        python src/utils/calculate_accuracy.py \
            "${INFER_DIR}/all_records.jsonl" \
            --compare-file "output/qwen/all_records.jsonl"
    fi

    echo ""
    echo ">>> context_layer_idx=${LAYER_IDX} complete <<<"
    echo ""

done

echo "========== All context_layer_idx experiments complete =========="
