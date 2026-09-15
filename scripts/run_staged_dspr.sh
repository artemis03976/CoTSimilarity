#!/usr/bin/env bash
# Two-stage DSPR training: prompt-level router, then trajectory-level prefix.
set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
MATH_PAIRED="data/math_paired.jsonl"
KFOLD_ROOT="data/qwen/kfold"
FOLD=0
OUTPUT_PATH="checkpoints/qwen-2.5_staged_seed42"

python scripts/train.py staged_dspr \
  --model-name "${MODEL_NAME}" \
  --math-paired-path "${MATH_PAIRED}" \
  --kfold-root "${KFOLD_ROOT}" \
  --fold "${FOLD}" \
  --output-path "${OUTPUT_PATH}" \
  --router-learning-rate 1e-5 \
  --prefix-learning-rate 4e-5 \
  --router-weight-decay 1e-3 \
  --prefix-weight-decay 0.0 \
  --router-target-smoothing 0.05 \
  --router-epochs 20 \
  --prefix-epochs 15 \
  --prefix-trajectory-sampling flat \
  --warmup-ratio 0.05 \
  --seed 42
