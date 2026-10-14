#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

python scripts/inference.py \
  --baseline base \
  --backend vllm \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --data-path data/canonical_math_paired.jsonl \
  --output-dir output/qwen-2.5/base \
  --max-new-tokens 2048 \
  --batch-size 128 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  "$@"

# python scripts/inference.py \
#   --baseline base \
#   --backend vllm \
#   --model-name deepseek-ai/deepseek-math-7b-instruct \
#   --data-path data/canonical_math_paired.jsonl \
#   --output-dir output/deepseek/base \
#   --max-new-tokens 2048 \
#   --batch-size 128 \
#   --gpu-memory-utilization 0.90 \
#   --max-model-len 4096 \
#   --max-num-seqs 64 \
#   --max-num-batched-tokens 8192 \
#   "$@"