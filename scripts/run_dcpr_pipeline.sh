#!/usr/bin/env bash
# =============================================================================
# run_dcpr_pipeline.sh - Full DCPR pipeline test script
#
# Workflow: data split -> training -> inference -> metric reporting
#
# Prerequisites:
#   - data/qwen/dcpr_train.jsonl
#   - data/qwen/dcpr_val.jsonl
#   - data/math_paired.jsonl for inference evaluation
#
# Outputs:
#   checkpoints/<RUN_NAME>/              training artifacts (dual_prefix + router)
#   output/<RUN_NAME>/all_records.jsonl  DCPR inference results
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUN_NAME="dcpr_pipeline"

# Training
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
TRAIN_DATA="data/qwen/dcpr_train.jsonl"
VAL_DATA="data/qwen/dcpr_val.jsonl"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

# Inference evaluation
TEST_DATA="data/math_paired.jsonl"
INFER_OUTPUT_DIR="output/${RUN_NAME}"

BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=4e-5

# ---------------------------------------------------------------------------
# Step 1: Train DCPR
# ---------------------------------------------------------------------------
echo "========== [1/3] Training DCPR =========="
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
# Step 2: Inference evaluation
# ---------------------------------------------------------------------------
# Use the latest checkpoint.
LATEST_CKPT=$(ls -td "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | head -1)
LATEST_CKPT="${LATEST_CKPT%/}"
if [ -z "${LATEST_CKPT}" ]; then
    echo "Error: no training checkpoint found; training may have failed."
    exit 1
fi

echo "========== [2/3] DCPR inference evaluation =========="
echo "Using checkpoint: ${LATEST_CKPT}"
mkdir -p "${INFER_OUTPUT_DIR}"

python scripts/dcpr_inference_test.py \
    --checkpoint "${LATEST_CKPT}/dcpr_trainable.pt" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${INFER_OUTPUT_DIR}" \
    --data_path "${TEST_DATA}" \
    --device cuda
echo ""

# ---------------------------------------------------------------------------
# Step 3: Metric reporting
# ---------------------------------------------------------------------------
echo "========== [3/3] Metric reporting =========="
echo ""
echo "--- DCPR inference results ---"
python src/utils/evaluation/calculate_accuracy.py \
    "${INFER_OUTPUT_DIR}/all_records.jsonl"

echo ""
echo "--- Compare against baseline if available ---"
if [ -f "output/base_all_records.jsonl" ]; then
    python src/utils/evaluation/calculate_accuracy.py \
        "${INFER_OUTPUT_DIR}/all_records.jsonl" \
        --compare-file "output/qwen/all_records.jsonl"
else
    echo "Qwen result file not found (output/qwen/all_records.jsonl); skipping comparison."
fi

echo ""
echo "========== Pipeline complete =========="
echo "Checkpoint: ${LATEST_CKPT}"
echo "Inference results: ${INFER_OUTPUT_DIR}/all_records.jsonl"
