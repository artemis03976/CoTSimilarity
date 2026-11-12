#!/usr/bin/env bash
# Sequential alpha=0/0.5/1 experiments; folds within each alpha run in parallel.
# Usage: GPUS=0,1,2,3,4 bash scripts/run_fixed_alpha_kfold.sh
# DRY_RUN=1 checks commands; RESUME=1 (default) skips completed matching folds.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
REFERENCE_RUN_ROOT="${REFERENCE_RUN_ROOT:-output/qwen_staged_kfold_seed42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/qwen_staged_kfold_seed42}"
KFOLD_ROOT="${KFOLD_ROOT:-output/qwen-2.5/kfold}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/qwen_fixed_alpha_seed42}"
PROBLEM_DATASET="${PROBLEM_DATASET:-data/canonical_math_paired.jsonl}"
read -r -a FOLD_IDS <<< "${FOLDS:-0 1 2 3 4}"

RUN_ARGS=(
  --reference-run-root "${REFERENCE_RUN_ROOT}" --checkpoint-root "${CHECKPOINT_ROOT}"
  --test-root "${KFOLD_ROOT}" --id-root "${KFOLD_ROOT}"
  --gpus "${GPUS}" --folds "${FOLD_IDS[@]}"
)
if [[ "${RESUME:-1}" == 1 ]]; then RUN_ARGS+=(--resume); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then RUN_ARGS+=(--dry-run); fi
if [[ -n "${INFERENCE_SCRIPT:-}" ]]; then RUN_ARGS+=(--inference-script "${INFERENCE_SCRIPT}"); fi
AGGREGATE_ARGS=(--id-root "${KFOLD_ROOT}" --raw-data "${PROBLEM_DATASET}" --folds "${FOLD_IDS[@]}")
if [[ -n "${BASELINE_PATH:-}" ]]; then AGGREGATE_ARGS+=(--baseline "${BASELINE_PATH}"); fi

for ALPHA in 0 0.5 1; do
  RUN_DIR="${OUTPUT_ROOT}/alpha_${ALPHA/./p}"
  echo "Fixed alpha=${ALPHA}: folds=${FOLD_IDS[*]}; GPUs=${GPUS}; results=${RUN_DIR}"
  "${PYTHON}" scripts/evaluate.py dspr run \
    "${RUN_ARGS[@]}" --forced-alpha "${ALPHA}" --output-root "${RUN_DIR}"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then continue; fi
  "${PYTHON}" scripts/evaluate.py dspr aggregate \
    --result-root "${RUN_DIR}" "${AGGREGATE_ARGS[@]}"
done

echo "Fixed-alpha experiments finished: ${OUTPUT_ROOT}"
