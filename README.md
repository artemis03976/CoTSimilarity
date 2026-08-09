# StructuredCoT

Official repository for "Dynamic Structural Prefix Routing for Structure-Aware Robust LLM Reasoning". The repository contains the full workflow for generating model responses, extracting reasoning structures as IFD-Graphs, computing graph edit distance (GED), constructing DSPR training data, and evaluating DSPR against baselines.

Most workflows are driven by scripts under `scripts/`, with reusable implementation code under `src/`.

## Repository Layout

```text
.
+-- scripts/
|   +-- inference.py                  # Unified greedy Base/DSPR/SPT/LoRA inference
|   +-- inference_multiple.py         # Batched validated vLLM trajectory sampling
|   +-- pertubation_test.py           # Legacy response-generation entrypoint
|   +-- train_dspr.py                 # Train the DSPR model
|   +-- train_dspr_kfold.py           # Run Qwen DSPR K-fold training across GPUs
|   +-- dspr_inference_test.py        # Evaluate trained DSPR checkpoints
|   +-- evaluate_dspr_kfold.py        # Run held-out K-fold inference across GPUs
|   +-- aggregate_dspr_kfold.py       # Validate, pool, and report OOF predictions
|   +-- train_lora.py                 # Train one parameter-matched LoRA adapter
|   +-- train_lora_kfold.py           # Run LoRA K-fold training across GPUs
|   +-- lora_inference_test.py        # Evaluate one LoRA adapter
|   +-- evaluate_lora_kfold.py        # Run LoRA held-out inference across GPUs
|   +-- aggregate_lora_kfold.py       # Pool LoRA OOF results and compare methods
|   +-- train_spt.py                  # Train the Static Prompt Tuning baseline
|   +-- spt_inference_test.py         # Evaluate the SPT baseline
|   +-- run_dspr_pipeline.sh          # Example DSPR train/eval pipeline
|   +-- run_ablation_*.sh             # Ablation for DSPR hyperparameters
|
+-- src/
|   +-- inference/                    # Shared model adapters, validation, and runners
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

`src/inference/` provides two experiment-facing inference contracts:

- deterministic greedy inference through one model adapter interface for Base, DSPR, SPT, and LoRA;
- batched vLLM sampling with structural CoT validation, bounded resampling, raw-generation retention, and QC summaries.

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
- `output/<model>/multiple_seed42/all_records.jsonl`: accepted original/simple/hard responses.
- `output/<model>/multiple_seed42/raw_generations.jsonl`: all multi-path attempts, including rejected paths.
- `output/<model>/multiple_seed42/generation_qc.json`: decoding and validation audit summary.
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
python scripts/inference_multiple.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --data-path data/math_paired.jsonl \
  --output-dir output/qwen/multiple_seed42 \
  --sampled-variants simple hard \
  --samples-per-problem 50 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --seed 42
```

This produces one greedy original reference and 50 valid sampled paths for
each simple/hard perturbation. The `raw_generations.jsonl` file retains all
accepted and rejected attempts, while `generation_qc.json` reports rejection
reasons and sampling shortfalls. The command fails rather than silently
returning fewer than 50 valid paths.

The repetition validator only rejects a token block with a period of at most
64 tokens when it repeats consecutively five times; reuse of the same phrase
at distant positions is allowed. During generation, the progress display shows
both accepted valid paths and raw attempts against the bounded attempt budget.

### 2. Segment Chain-Of-Thought Responses

```bash
python src/data_analysis/cot_segmenter.py \
  --input "output/qwen/multiple_seed42/all_records.jsonl" \
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
  --correctness-file "output/qwen/multiple_seed42/all_records.jsonl" \
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

### 8. Train Qwen DSPR with the Prepared K-Folds

The five prepared Qwen folds can be trained with seed 42 using the orchestration
script below. Each fold gets an isolated checkpoint directory and `train.log`.
GPU workers dynamically claim the next unfinished fold, so the same command
also works when fewer than five GPUs are available:

```bash
python scripts/train_dspr_kfold.py --gpus 0,1,2,3,4
```

Use `--dry-run` to validate the fold files and print the exact child commands
without loading a model. Outputs default to
`checkpoints/qwen_kfold_seed42/fold_0` through `fold_4`; the complete run
manifest is written to the parent directory.

### 9. Evaluate Qwen DSPR with Held-Out K-Folds

First run one held-out Fold-0 problem as a smoke test. Use a separate output
root so the smoke result cannot overwrite the formal fold output:

```bash
python scripts/evaluate_dspr_kfold.py \
  --folds 0 \
  --problem-id 2 \
  --gpus 0 \
  --output-root output/qwen_kfold_seed42_smoke
```

After the smoke test passes, run all folds. The evaluator reads each fold's
`trainer_state.json`, selects its best validation checkpoint, explicitly uses
prefix length 15, and dynamically assigns pending folds to available GPUs:

```bash
python scripts/evaluate_dspr_kfold.py --gpus 0,1,2,3,4
```

Each fold writes `all_records.jsonl`, `inference.log`, and `run_status.json`
under `output/qwen_kfold_seed42/fold_N`. The parent directory also contains an
`inference_manifest.json` with the exact checkpoint and command for every fold.

Validate and pool the five folds into one OOF result:

```bash
python scripts/aggregate_dspr_kfold.py
```

To add paired bootstrap confidence intervals and exact McNemar tests against a
matched greedy-decoding baseline, first generate one base-Qwen prediction for
every raw problem group. The base model does not require one run per fold
because it has no fold-specific training:

```bash
python scripts/inference.py \
  --baseline base \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --data-path data/math_paired.jsonl \
  --output-dir output/qwen_matched_greedy \
  --max-new-tokens 4096
```

Then pass the resulting 279-problem file to the aggregator:

```bash
python scripts/aggregate_dspr_kfold.py \
  --baseline output/qwen_matched_greedy/all_records.jsonl
```

The aggregator requires exact fold-ID coverage and writes
`oof_all_records.jsonl`, `oof_metrics.json`, `oof_metrics.csv`, and
`oof_alpha_summary.csv` under the K-fold output root.

### 9.1 Train and Evaluate the Parameter-Matched LoRA Baseline

The LoRA baseline uses rank 3 adapters on every `q_proj` and `v_proj`. For
Qwen2.5-Math-7B-Instruct this trains 946,176 parameters, 7.74% fewer than
DSPR's 1,025,537 trainable parameters. The training entrypoint verifies this
budget at runtime and aborts if the gap exceeds 10%.

Validate the fold commands without loading the model, then train all folds:

```bash
python scripts/train_lora_kfold.py --dry-run --gpus 0,1,2,3,4
python scripts/train_lora_kfold.py --gpus 0,1,2,3,4
```

Run one held-out smoke test before starting all-fold inference:

```bash
python scripts/evaluate_lora_kfold.py \
  --folds 0 \
  --problem-id 2 \
  --gpus 0 \
  --output-root output/qwen_lora_qv_r3_seed42_smoke

python scripts/evaluate_lora_kfold.py --gpus 0,1,2,3,4
```

The inference entrypoint runs symbolic-answer evaluator self-tests before it
loads the model. Pool the exact OOF coverage and compare LoRA against both the
matched greedy base model and DSPR:

```bash
python scripts/aggregate_lora_kfold.py \
  --base output/qwen/all_records.jsonl \
  --dspr output/qwen_kfold_seed42/oof_all_records.jsonl
```

The LoRA report writes `oof_all_records.jsonl`, `oof_metrics.json`, and
`oof_metrics.csv` under `output/qwen_lora_qv_r3_seed42`. Paired comparisons
include bootstrap confidence intervals and exact McNemar tests for full,
eligible, and noneligible slices.

### 10. Evaluate a Single DSPR Checkpoint

```bash
python scripts/inference.py \
  --baseline dspr \
  --checkpoint "checkpoints/qwen/dspr/dspr_trainable.pt" \
  --model-name "Qwen/Qwen2.5-Math-7B-Instruct" \
  --data-path "data/math_paired.jsonl" \
  --output-dir "output/qwen/dspr" \
  --prefix-length 50 \
  --max-new-tokens 4096
```

Use the same command with `--baseline spt` and an SPT checkpoint, or
`--baseline lora` and a PEFT adapter directory. The single-path entrypoint is
greedy-only by design: invalid or truncated outputs are recorded in
`generation_qc.json` and are never resampled.

The inference output is:

```text
output/qwen/dspr/all_records.jsonl
```

### 11. Report Accuracy

Evaluate one result file:

```bash
python src/utils/calculate_accuracy.py "output/qwen/dspr/all_records.jsonl"
```

Compare DSPR against a baseline result file:

```bash
python src/utils/calculate_accuracy.py \
  "output/qwen/dspr/all_records.jsonl" \
  --compare-file "output/qwen_matched_greedy/all_records.jsonl"
```

## Recommended Reading Order

For new contributors, the fastest way to understand the code is:

1. `scripts/run_dspr_pipeline.sh` for the high-level workflow.
2. `src/dspr/config.py` and `src/dspr/model.py` for the model interface.
3. `src/dspr_training/dataset.py`, `loss.py`, and `trainer.py` for training behavior.
4. `src/data_analysis/ged_analysis.py` and `src/dspr_dataset/data_filter.py` for dataset construction.
5. `scripts/dspr_inference_test.py` and `src/utils/calculate_accuracy.py` for evaluation.
