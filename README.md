# StructuredCoT 实验流程

本文档描述了当前的一套端到端流程，覆盖：

- Base Model 数据生成与分析（一次性）
- DSPR 数据构建、划分、训练
- DSPR 与 Base Model 指标对比

---

## 说明

- **1-6 步主要为训练数据生成与预处理准备**，用于生成 DSPR 所需训练数据与中间分析结果。
- 请优先关注 **DSPR 训练与推理**，即重点执行第 **7-9 步**。
- 已有以下文件时，可直接跳过 1-6 步：
  - `data/<model>/dspr_train.jsonl`
  - `data/<model>/dspr_val.jsonl`
  - `data/<model>/dspr_test.jsonl`
  - （可选对比）`output/<model>/all_records.jsonl`
- 新加入的模型需要通过 1-6 步生成对应数据，其中仅第一步生成 base model 的回答需要GPU支持模型推理；第三步刚需批量推理模式以处理数据，需要手动介入
- 请先运行 `run_dspr_pipeline.sh` 中的示例脚本流程，内容是针对 `qwen-2.5-math-7b-instruct` 模型的dspr训练与推理。如无错误，再运行 3 个 ablation study 脚本中的命令，部分必要超参已经在脚本/默认config中配置完毕

---

## 0. 环境准备

```bash
pip install -r requirements.txt
```

### Docker

```bash
# 1) 准备环境变量（可选）
cp .env.example .env

# 2) 构建镜像
docker compose -f docker-compose.gpu.yml build

# 3) 进入容器
docker compose -f docker-compose.gpu.yml run --rm dspr
```

---

## 1. 生成 base model 回答（original 1 条，simple/hard 各 50 条）

```bash
python scripts/pertubation_test.py \
  --model "Qwen/Qwen2.5-Math-7B-Instruct" \
  --data_path "data/math_paired.jsonl" \
  --output_path "output/qwen/all_records_50.jsonl" \
  --n_original 1 \
  --n 50 \
```

输出：`output/qwen/all_records_50.jsonl`

---

## 2. CoT 切分

```bash
python src/data_analysis/cot_segmenter.py \
  --input "output/qwen/all_records_50.jsonl" \
  --output "output/qwen/segmented_records_50.jsonl"
```

---

## 3. DAG 分析（批处理模式）

### 3.1 生成批处理请求文件

```bash
python src/data_analysis/dag_analyzer.py \
  --mode batch \
  --input "output/qwen/segmented_records_50.jsonl" \
  --output-dir "output/qwen/dag_analysis_50" \
  --provider deepseek \
  --model deepseek-chat
```

输出请求文件：`output/qwen/dag_analysis_50/batch/batch_requests.jsonl`

### 3.2 上传到批量推理服务后，合并下载结果

```bash
python src/data_analysis/dag_analyzer.py \
  --mode merge-batch \
  --input "output/qwen/segmented_records_50.jsonl" \
  --batch-results-file "path/to/batch_results.jsonl" \
  --output-dir "output/qwen/dag_analysis_50"
```

输出：`output/qwen/dag_analysis_50/analyzed_records.jsonl`

---

## 4. GED 分析

当前默认流程下，`ged_analysis.py` 会直接从 `--variant-records` 中读取
`original_0` 作为 GED 基准。

示例：

```bash
python src/data_analysis/ged_analysis.py \
  --output-root "output/qwen" \
  --variant-records "output/qwen/dag_analysis_50/analyzed_records.jsonl" \
  --correctness-file "output/qwen/all_records_50.jsonl" \
  --all-results-output "output/qwen/all_ged_results.jsonl"
```

如需兼容旧流程（original 与 variant 分开分析），可额外传：
`--original-records "output/qwen/dag_analysis/analyzed_records.jsonl"`

输出：`output/qwen/all_ged_results.jsonl`

---

## 5. 构建 DSPR 数据集

```bash
python src/dspr_dataset/data_filter.py \
  --input "output/qwen/all_ged_results.jsonl" \
  --output "data/qwen/dspr_dataset.jsonl" \
  --top-k 5 \
  --min-variance 1.0
```

---

## 6. 自动划分 train/val/test（按 problem_id）

```bash
python src/dspr_dataset/split_dataset.py \
  --input "data/qwen/dspr_dataset.jsonl" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

默认输出到同目录：

- `data/qwen/dspr_train.jsonl`
- `data/qwen/dspr_val.jsonl`
- `data/qwen/dspr_test.jsonl`

---

## 7. 训练 DSPR

> 默认应从本步骤开始

```bash
python scripts/train_dspr.py \
  --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
  --train_data_path "data/qwen/dspr_train.jsonl" \
  --val_data_path "data/qwen/dspr_val.jsonl" \
  --output_path "checkpoints/qwen/dspr" \
  --batch_size 4 \
  --num_epochs 10
```

训练产物（默认按 Trainer 保存）：`dspr_trainable.pt`

---

## 8. DSPR 推理评估

```bash
python scripts/dspr_inference_test.py \
  --checkpoint "checkpoints/qwen/dspr/dspr_trainable.pt" \
  --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
  --data_path "data/math_paired.jsonl" \
  --output_dir "output/qwen/dspr" \
  --n 1 \
  --temperature 0.0
```

输出：`output/qwen/dspr/all_records.jsonl`

---

## 9. 统一指标统计与对比

仅统计单个结果：

```bash
python src/utils/evaluation/calculate_accuracy.py "output/qwen/dspr/all_records.jsonl"
```

与 baseline 对比：

```bash
python src/utils/evaluation/calculate_accuracy.py \
  "output/qwen/dspr/all_records.jsonl" \
  --compare-file "output/qwen/all_records_50.jsonl"
```

脚本会输出：

- `First@1`
- `Any@k`
- `AvgSampleAcc`
- `Overall(all variants pass@k)` 及对比增量

---

## 常见文件流向（简表）

1. `all_records_50.jsonl`（Base 回答）
2. `segmented_records_50.jsonl`（CoT 切分）
3. `analyzed_records.jsonl`（DAG 分析结果）
4. `all_ged_results.jsonl`（GED 结果）
5. `dspr_dataset.jsonl`（过滤后训练样本）
6. `dspr_train/val/test.jsonl`（自动划分）
7. `dspr_trainable.pt`（DSPR 可训练参数）
8. `output/.../dspr/all_records.jsonl`（DSPR 推理结果）
