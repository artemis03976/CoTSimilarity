export CUDA_VISIBLE_DEVICES=7

python scripts/inference.py \
  --baseline base \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --output-dir output/qwen-2.5/base \

python scripts/inference.py \
  --baseline base \
  --model-name deepseek-ai/deepseek-math-7b-instruct \
  --output-dir output/deepseek/base \

python scripts/inference.py \
  --baseline dspr \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --checkpoint checkpoints/qwen/dspr/dspr_trainable.pt \
  --output-dir output/qwen-2.5/dspr \
  --context-layer-idx 15 \
  --prefix-length 15 \

python scripts/inference.py \
  --baseline dspr \
  --model-name deepseek-ai/deepseek-math-7b-instruct \
  --checkpoint checkpoints/deepseek/dspr/dspr_trainable.pt \
  --output-dir output/deepseek/dspr \
  --context-layer-idx 15 \
  --prefix-length 15 \
