export CUDA_VISIBLE_DEVICES=7

python scripts/inference_multiple.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --output-dir output/qwen/multiple_seed42 \
  --sampled-variants simple hard \
  --samples-per-problem 50 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed 42