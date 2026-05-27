# StructuredCoT

Official repository for "Dynamic Structural Prefix Routing for Structure-Aware Robust LLM Reasoning". The repository contains the full workflow for generating model responses, extracting reasoning structures as IFD-Graphs, computing graph edit distance (GED), constructing DSPR training data, and evaluating DSPR against baselines.

Most workflows are driven by scripts under `scripts/`, with reusable implementation code under `src/`.

## Repository Layout

```text
.
+-- scripts/
|   +-- pertubation_test.py           # Generate original/simple/hard model responses
|   +-- train_dspr.py                 # Train the DSPR model
|   +-- dspr_inference_test.py        # Evaluate trained DSPR checkpoints
|   +-- train_spt.py                  # Train the Static Prompt Tuning baseline
|   +-- spt_inference_test.py         # Evaluate the SPT baseline
|   +-- run_dspr_pipeline.sh          # Example DSPR train/eval pipeline
|   +-- run_ablation_*.sh             # Ablation for DSPR hyperparameters
|
+-- src/
|   +-- dspr/                         # Core DSPR model implementation
|   +-- dspr_training/                # DSPR dataset wrapper, loss, and Trainer
|   +-- dspr_dataset/                 # DSPR dataset construction utilities
|   +-- spt/                          # Static Prompt Tuning baseline model
|   +-- spt_training/                 # SPT Trainer
|   +-- data_analysis/                # CoT segmentation, DAG analysis, GED analysis
|   +-- utils/                        # Evaluation, sorting, reports, visualization
|
+-- data/                             # Input datasets and derived training splits
+-- output/                           # Generated responses and analysis artifacts
+-- Dockerfile
+-- docker-compose.gpu.yml
+-- requirements.txt
```

## Main Components

`src/dspr/` contains the DSPR model:

- `config.py`: default model, data, training, and hardware settings.
- `model.py`: wraps the frozen base LLM with structural prefix routing.
- `dual_prefix.py`: learnable exploit/explore structural prefixes.
- `router.py`: MLP router that predicts the exploration weight.
- `context_encoder.py`: extracts hidden-state context for routing.

`src/data_analysis/` turns raw generated responses into structural features:

- `cot_segmenter.py`: segments chain-of-thought responses into reasoning units.
- `dag_analyzer.py`: builds prompts or merges batch outputs for DAG extraction.
- `dag_compressor.py`: normalizes and compresses DAG representations.
- `dag_similarity.py`: computes graph-level similarity and GED.
- `ged_analysis.py`: joins DAG similarity with answer correctness to produce records for DSPR dataset construction.
- `llm/`: provider-agnostic API and batch-processing helpers used by DAG analysis.

`src/spt/` and `src/spt_training/` implement the Static Prompt Tuning baseline, which uses the same frozen LLM and DSPR dataset format but replaces dynamic routing with one static learned prefix.

## Environment Setup

Install dependencies directly:

```bash
pip install -r requirements.txt
```

Or use the GPU Docker environment:

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm dspr
```

Most scripts assume they are launched from the repository root. If imports fail in a custom environment, set:

```bash
export PYTHONPATH=src
```

## Data And Artifact Conventions

Common paths used by the current scripts:

- `data/math_paired.jsonl`: source math problems.
- `output/<model>/all_records_50.jsonl`: generated original/simple/hard responses.
- `output/<model>/segmented_records_50.jsonl`: segmented CoT records.
- `output/<model>/dag_analysis_50/analyzed_records.jsonl`: DAG analysis results.
- `output/<model>/all_ged_results.jsonl`: GED and correctness records.
- `data/<model>/dspr_dataset.jsonl`: filtered DSPR dataset.
- `data/<model>/dspr_train.jsonl`, `dspr_val.jsonl`, `dspr_test.jsonl`: DSPR splits.
- `checkpoints/<model>/dspr/dspr_trainable.pt`: trained DSPR parameters.
- `output/<model>/dspr/all_records.jsonl`: DSPR inference records.

Some `output/*/all_records.jsonl` files are expensive experiment artifacts and are intentionally kept in the repository workflow.

## End-To-End DSPR Workflow

The example commands below use Qwen2.5-Math-7B-Instruct and write artifacts under `output/qwen`, `data/qwen`, and `checkpoints/qwen`.

### 1. Generate Base Model Responses

```bash
python scripts/pertubation_test.py \
  --model "Qwen/Qwen2.5-Math-7B-Instruct" \
  --data_path "data/math_paired.jsonl" \
  --output_path "output/qwen/all_records_50.jsonl" \
  --n_original 1 \
  --n 50
```

This produces original, simple, and hard response samples for each problem.

### 2. Segment Chain-Of-Thought Responses

```bash
python src/data_analysis/cot_segmenter.py \
  --input "output/qwen/all_records_50.jsonl" \
  --output "output/qwen/segmented_records_50.jsonl"
```

### 3. Run DAG Analysis

Generate batch requests:

```bash
python src/data_analysis/dag_analyzer.py \
  --mode batch \
  --input "output/qwen/segmented_records_50.jsonl" \
  --output-dir "output/qwen/dag_analysis_50" \
  --provider deepseek \
  --model deepseek-chat
```

After the batch inference service returns results, merge them:

```bash
python src/data_analysis/dag_analyzer.py \
  --mode merge-batch \
  --input "output/qwen/segmented_records_50.jsonl" \
  --batch-results-file "path/to/batch_results.jsonl" \
  --output-dir "output/qwen/dag_analysis_50"
```

The merged DAG records are written to:

```text
output/qwen/dag_analysis_50/analyzed_records.jsonl
```

### 4. Compute GED Records

```bash
python src/data_analysis/ged_analysis.py \
  --output-root "output/qwen" \
  --variant-records "output/qwen/dag_analysis_50/analyzed_records.jsonl" \
  --correctness-file "output/qwen/all_records_50.jsonl" \
  --all-results-output "output/qwen/all_ged_results.jsonl"
```

By default, `ged_analysis.py` uses each problem's `original_0` response as the GED reference graph. For older workflows where original and variant DAG records are stored separately, pass `--original-records` explicitly.

### 5. Build The DSPR Dataset

```bash
python src/dspr_dataset/data_filter.py \
  --input "output/qwen/all_ged_results.jsonl" \
  --output "data/qwen/dspr_dataset.jsonl" \
  --top-k 5 \
  --min-variance 1.0
```

### 6. Split Train/Validation/Test Sets

```bash
python src/dspr_dataset/split_dataset.py \
  --input "data/qwen/dspr_dataset.jsonl" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

Default outputs are written next to the input file:

- `data/qwen/dspr_train.jsonl`
- `data/qwen/dspr_val.jsonl`
- `data/qwen/dspr_test.jsonl`

### 7. Train DSPR

```bash
python scripts/train_dspr.py \
  --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
  --train_data_path "data/qwen/dspr_train.jsonl" \
  --val_data_path "data/qwen/dspr_val.jsonl" \
  --output_path "checkpoints/qwen/dspr" \
  --batch_size 4 \
  --num_epochs 10
```

The trainer saves the trainable DSPR parameters as:

```text
checkpoints/qwen/dspr/dspr_trainable.pt
```

### 8. Evaluate DSPR

```bash
python scripts/dspr_inference_test.py \
  --checkpoint "checkpoints/qwen/dspr/dspr_trainable.pt" \
  --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
  --data_path "data/math_paired.jsonl" \
  --output_dir "output/qwen/dspr" \
  --n 1 \
  --temperature 0.0
```

The inference output is:

```text
output/qwen/dspr/all_records.jsonl
```

### 9. Report Accuracy

Evaluate one result file:

```bash
python src/utils/calculate_accuracy.py "output/qwen/dspr/all_records.jsonl"
```

Compare DSPR against a baseline result file:

```bash
python src/utils/calculate_accuracy.py \
  "output/qwen/dspr/all_records.jsonl" \
  --compare-file "output/qwen/all_records_50.jsonl"
```

## Recommended Reading Order

For new contributors, the fastest way to understand the code is:

1. `scripts/run_dspr_pipeline.sh` for the high-level workflow.
2. `src/dspr/config.py` and `src/dspr/model.py` for the model interface.
3. `src/dspr_training/dataset.py`, `loss.py`, and `trainer.py` for training behavior.
4. `src/data_analysis/ged_analysis.py` and `src/dspr_dataset/data_filter.py` for dataset construction.
5. `scripts/dspr_inference_test.py` and `src/utils/calculate_accuracy.py` for evaluation.
