export CUDA_VISIBLE_DEVICES=7

python src/data_analysis/pertubation_test.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --output_path "output/qwen/all_records_50.jsonl"

python src/data_analysis/pertubation_test.py \
    --model "deepseek-ai/deepseek-math-7b-instruct" \
    --output_path "output/deepseek/all_records_50.jsonl"
