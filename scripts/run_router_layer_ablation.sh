#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-7B-Instruct}"
PROBLEM_DATASET="${PROBLEM_DATASET:-data/canonical_math_paired.jsonl}"
KFOLD_ROOT="${KFOLD_ROOT:-output/qwen-2.5/kfold}"
FOLD="${FOLD:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/qwen-2.5_router_layer_ablation_seed42}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-20}"
ROUTER_LEARNING_RATE="${ROUTER_LEARNING_RATE:-1e-5}"
ROUTER_WEIGHT_DECAY="${ROUTER_WEIGHT_DECAY:-1e-3}"
ROUTER_DROPOUT="${ROUTER_DROPOUT:-0.10}"
ROUTER_TARGET_SMOOTHING="${ROUTER_TARGET_SMOOTHING:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

if [[ ! -f "${PROBLEM_DATASET}" ]]; then
    echo "Error: canonical source not found: ${PROBLEM_DATASET}" >&2
    exit 1
fi

if [[ -n "${LAYERS:-}" ]]; then
    # shellcheck disable=SC2206
    LAYER_INDICES=(${LAYERS})
else
    NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-}"
    if [[ -z "${NUM_HIDDEN_LAYERS}" ]]; then
        NUM_HIDDEN_LAYERS="$(python - "${MODEL_NAME}" <<'PY'
import sys
from transformers import AutoConfig

model_name = sys.argv[1]
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
value = getattr(config, "num_hidden_layers", None)
if value is None:
    raise SystemExit("Model config has no num_hidden_layers field")
print(int(value))
PY
        )"
    fi

    # Include hidden_states[0] (the embedding output) and every transformer
    # block output through hidden_states[L].
    LAYER_INDICES=()
    for ((layer = 0; layer <= NUM_HIDDEN_LAYERS; layer++)); do
        LAYER_INDICES+=("${layer}")
    done
fi

mkdir -p "${OUTPUT_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Fold: ${FOLD}; seed: ${SEED}"
echo "K-fold root: ${KFOLD_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Layers: ${LAYER_INDICES[*]}"

for LAYER_IDX in "${LAYER_INDICES[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/layer_${LAYER_IDX}"
    CACHE_PATH="${OUTPUT_ROOT}/context_cache_layer_${LAYER_IDX}.pt"

    echo
    echo "============================================================"
    echo "Router layer ablation: context_layer_idx=${LAYER_IDX}"
    echo "Output: ${RUN_DIR}"
    echo "============================================================"

    # A layer run is intentionally resumable: if its final router artifact
    # already exists, leave it untouched and continue to the next layer.
    if [[ -f "${RUN_DIR}/router/router_trainable.pt" && "${FORCE:-0}" != "1" ]]; then
        echo "[skip] existing router checkpoint (set FORCE=1 to retrain)"
        continue
    fi

    python scripts/train.py staged_dspr \
        --run-stage router \
        --model-name "${MODEL_NAME}" \
        --context-layer-idx "${LAYER_IDX}" \
        --router-intermediate-dim "${ROUTER_INTERMEDIATE_DIM:-64}" \
        --router-dropout "${ROUTER_DROPOUT}" \
        --problem-dataset "${PROBLEM_DATASET}" \
        --kfold-root "${KFOLD_ROOT}" \
        --fold "${FOLD}" \
        --router-context-cache-path "${CACHE_PATH}" \
        --output-path "${RUN_DIR}" \
        --router-learning-rate "${ROUTER_LEARNING_RATE}" \
        --router-weight-decay "${ROUTER_WEIGHT_DECAY}" \
        --router-target-smoothing "${ROUTER_TARGET_SMOOTHING}" \
        --router-epochs "${ROUTER_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --logging-steps "${LOGGING_STEPS}" \
        --no-gradient-checkpointing \
        --seed "${SEED}"
done

echo
echo "All router layer runs completed or skipped."
echo "Use scripts/plot_router_layer_ablation.py to summarize:"
echo "  python scripts/plot_router_layer_ablation.py --root \"${OUTPUT_ROOT}\""
