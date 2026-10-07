#!/usr/bin/env bash
# Staged K-fold DSPR training, held-out inference, and OOF metric reporting.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-7B-Instruct}"
PROBLEM_DATASET="${PROBLEM_DATASET:-data/canonical_math_paired.jsonl}"
KFOLD_ROOT="${KFOLD_ROOT:-output/qwen-2.5/kfold}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-qwen_staged_kfold_seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${RUN_NAME}}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-output/${RUN_NAME}}"
GPUS="${GPUS:-auto}"
read -r -a FOLD_IDS <<< "${FOLDS:-${FOLD:-0 1 2 3 4}}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_dspr_pipeline.sh [OPTIONS]
  --gpus IDS                 GPU IDs, e.g. 0 or 0,1; default: auto
  --folds N [N ...]           Fold subset; default: 0 1 2 3 4
  --train-only               Run router -> frozen-router prefix training only
  --eval-only                Evaluate existing prefix checkpoints and report OOF metrics
  --dry-run                  Validate and print commands without training or inference
  --allow-existing           Allow non-empty training and inference output directories
  -h, --help                 Show this help

Configuration can also be set with environment variables:
  PYTHON, MODEL_NAME, PROBLEM_DATASET, KFOLD_ROOT, GPUS, FOLDS, SEED,
  RUN_NAME, CHECKPOINT_DIR, INFER_OUTPUT_DIR, PIPELINE_STAGE (all/train/eval),
  ROUTER_EPOCHS (20), PREFIX_EPOCHS (15), ROUTER_LEARNING_RATE (1e-5),
  PREFIX_LEARNING_RATE (4e-5), ROUTER_WEIGHT_DECAY (1e-3),
  PREFIX_WEIGHT_DECAY (0.0), ROUTER_TARGET_SMOOTHING (0.05),
  PREFIX_TRAJECTORY_SAMPLING (flat/random_one; default: flat),
  BATCH_SIZE (4), GRADIENT_ACCUMULATION_STEPS (4), WARMUP_RATIO (0.05),
  CONTEXT_LAYER_IDX (15), PREFIX_LENGTH (15), ROUTER_INTERMEDIATE_DIM (64),
  ROUTER_DROPOUT (0.10), MAX_SEQ_LENGTH (2048), MAX_NEW_TOKENS (2048),
  ROUTER_EARLY_STOPPING_PATIENCE (3), PREFIX_EARLY_STOPPING_PATIENCE (3),
  EARLY_STOPPING_THRESHOLD (0.0), BASELINE_PATH (optional).
EOF
}

while (( $# )); do
  case "$1" in
    --gpus) GPUS="${2:?--gpus requires a value}"; shift 2 ;;
    --folds)
      shift
      FOLD_IDS=()
      while (( $# )) && [[ "$1" != -* ]]; do
        FOLD_IDS+=("$1")
        shift
      done
      if (( ${#FOLD_IDS[@]} == 0 )); then
        echo "Error: --folds requires at least one fold ID." >&2
        exit 2
      fi
      ;;
    --train-only) PIPELINE_STAGE=train; shift ;;
    --eval-only) PIPELINE_STAGE=eval; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-existing) ALLOW_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${PIPELINE_STAGE}" in
  all|train|eval) ;;
  *) echo "Error: PIPELINE_STAGE must be all, train, or eval." >&2; exit 2 ;;
esac

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --gpus "${GPUS}"
  --folds "${FOLD_IDS[@]}"
  --context-layer-idx "${CONTEXT_LAYER_IDX:-15}"
  --prefix-length "${PREFIX_LENGTH:-15}"
  --router-intermediate-dim "${ROUTER_INTERMEDIATE_DIM:-64}"
  --router-dropout "${ROUTER_DROPOUT:-0.10}"
  --max-seq-length "${MAX_SEQ_LENGTH:-2048}"
)
if [[ "${ALLOW_EXISTING}" == "1" ]]; then COMMON_ARGS+=(--allow-existing); fi
if [[ "${DRY_RUN}" == "1" ]]; then COMMON_ARGS+=(--dry-run); fi

if [[ "${PIPELINE_STAGE}" != "eval" ]]; then
  echo "========== Staged DSPR K-fold training =========="
  "${PYTHON}" scripts/train_kfold.py dspr \
    "${COMMON_ARGS[@]}" \
    --problem-dataset "${PROBLEM_DATASET}" \
    --fold-root "${KFOLD_ROOT}" \
    --output-root "${CHECKPOINT_DIR}" \
    --router-learning-rate "${ROUTER_LEARNING_RATE:-1e-5}" \
    --prefix-learning-rate "${PREFIX_LEARNING_RATE:-4e-5}" \
    --router-weight-decay "${ROUTER_WEIGHT_DECAY:-1e-3}" \
    --prefix-weight-decay "${PREFIX_WEIGHT_DECAY:-0.0}" \
    --router-target-smoothing "${ROUTER_TARGET_SMOOTHING:-0.05}" \
    --router-epochs "${ROUTER_EPOCHS:-20}" \
    --prefix-epochs "${PREFIX_EPOCHS:-15}" \
    --prefix-trajectory-sampling "${PREFIX_TRAJECTORY_SAMPLING:-flat}" \
    --router-early-stopping-patience "${ROUTER_EARLY_STOPPING_PATIENCE:-3}" \
    --prefix-early-stopping-patience "${PREFIX_EARLY_STOPPING_PATIENCE:-3}" \
    --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD:-0.0}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --warmup-ratio "${WARMUP_RATIO:-0.05}" \
    --seed "${SEED}"
  # A training dry run creates no prefix checkpoints to evaluate.
  if [[ "${PIPELINE_STAGE}" == "train" || "${DRY_RUN}" == "1" ]]; then exit 0; fi
fi

echo "========== Held-out K-fold DSPR inference =========="
"${PYTHON}" scripts/evaluate.py dspr run \
  "${COMMON_ARGS[@]}" \
  --checkpoint-root "${CHECKPOINT_DIR}" \
  --test-root "${KFOLD_ROOT}" \
  --id-root "${KFOLD_ROOT}" \
  --output-root "${INFER_OUTPUT_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

echo "========== OOF metric reporting =========="
AGGREGATE_ARGS=(
  --result-root "${INFER_OUTPUT_DIR}"
  --id-root "${KFOLD_ROOT}"
  --raw-data "${PROBLEM_DATASET}"
  --folds "${FOLD_IDS[@]}"
)
if [[ -n "${BASELINE_PATH:-}" ]]; then AGGREGATE_ARGS+=(--baseline "${BASELINE_PATH}"); fi
"${PYTHON}" scripts/evaluate.py dspr aggregate "${AGGREGATE_ARGS[@]}"
echo "Pipeline complete. Checkpoints: ${CHECKPOINT_DIR}; OOF metrics: ${INFER_OUTPUT_DIR}/oof_metrics.json"
