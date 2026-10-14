#!/usr/bin/env bash
# Usage: GPUS=0 bash scripts/run_baseline_pipeline.sh spt|lora [training options]
# STAGE=train/eval/all, FOLDS="0 1 2 3 4", DRY_RUN=1 for command validation.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

METHOD="${1:?Specify spt or lora}"
shift
case "${METHOD}" in
  spt) DEFAULT_RUN_NAME=qwen_spt_seed42 ;;
  lora) DEFAULT_RUN_NAME=qwen_lora_qv_r1_seed42 ;;
  *) echo "Unknown baseline: ${METHOD}" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
KFOLD_ROOT="${KFOLD_ROOT:-output/qwen-2.5/kfold}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/${RUN_NAME}}"
STAGE="${STAGE:-all}"
case "${STAGE}" in all|train|eval) ;; *) echo "STAGE must be all, train, or eval" >&2; exit 2 ;; esac
read -r -a FOLD_IDS <<< "${FOLDS:-0 1 2 3 4}"
CHECK_ARGS=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then CHECK_ARGS+=(--dry-run); fi

if [[ "${STAGE}" != eval ]]; then
  "${PYTHON}" scripts/train_kfold.py "${METHOD}" \
    --fold-root "${KFOLD_ROOT}" --output-root "${CHECKPOINT_ROOT}" \
    --gpus "${GPUS}" --folds "${FOLD_IDS[@]}" "${CHECK_ARGS[@]}" "$@"
  if [[ "${STAGE}" == train || "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
fi
"${PYTHON}" scripts/evaluate.py "${METHOD}" run \
  --checkpoint-root "${CHECKPOINT_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --test-root "${KFOLD_ROOT}" --id-root "${KFOLD_ROOT}" \
  --gpus "${GPUS}" --folds "${FOLD_IDS[@]}" "${CHECK_ARGS[@]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
"${PYTHON}" scripts/evaluate.py "${METHOD}" aggregate \
  --result-root "${OUTPUT_ROOT}" --id-root "${KFOLD_ROOT}" \
  --raw-data data/canonical_math_paired.jsonl --folds "${FOLD_IDS[@]}" \
  --base output/qwen-2.5/base/all_records.jsonl \
  --dspr output/qwen_staged_kfold_seed42/oof_all_records.jsonl
