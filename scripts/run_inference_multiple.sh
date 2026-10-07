#!/usr/bin/env bash
set -euo pipefail

# Generate trajectories from the canonical problem dataset. Override these
# variables for a different model, GPU, dataset, or sampling budget, for example:
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_inference_multiple.sh
#
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SAMPLES_PER_PROBLEM="${SAMPLES_PER_PROBLEM:-8}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-data/canonical_math_paired.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen-2.5/multiple_seed42}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Input JSONL does not exist: ${DATA_PATH}" >&2
  exit 1
fi

echo "Model: ${MODEL}"
echo "Input: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${GPU}"
echo "Samples per variant: ${SAMPLES_PER_PROBLEM}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/inference_multiple.py \
  --model "${MODEL}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --sampled-variants simple hard \
  --samples-per-problem "${SAMPLES_PER_PROBLEM}" \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --max-attempt-multiplier 3 \
  --seed "${SEED}" \
  --strict-quality
