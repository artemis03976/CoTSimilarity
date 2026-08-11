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
|   +-- train.py                      # Unified DSPR/SPT/LoRA training entrypoint
|   +-- train_kfold.py                # Unified multi-GPU DSPR/LoRA K-fold training
|   +-- evaluate.py                   # Unified K-fold inference and OOF aggregation
|   +-- run_dspr_pipeline.sh          # Example DSPR train/eval pipeline
|   +-- run_ablation_*.sh             # Ablation for DSPR hyperparameters
|
+-- src/
|   +-- experiment_pipeline/          # Reusable training, K-fold, and OOF capabilities
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

The example commands below use Qwen2.5-Math-7B-Instruct and write new artifacts under `output/qwen-2.5`, `data/qwen`, and `checkpoints/qwen`. Results from the earlier pipeline are archived under `output/qwen_legacy`.

### 1. Generate Base Model Responses

```bash
python scripts/inference_multiple.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --data-path data/math_paired.jsonl \
  --output-dir output/qwen-2.5/multiple_seed42 \
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
64 tokens when it repeats consecutively five times **and** the consecutive loop
spans at least 64 tokens. This avoids rejecting short arithmetic patterns such
as `3 * 3 * 3 * 3 * 3 * 3` or runs of zeros in large integers; reuse of the same
phrase at distant positions is also allowed. During generation, the progress
display shows both accepted valid paths and raw attempts against the bounded
attempt budget.

If validation thresholds change after an expensive vLLM run, rebuild the
accepted pool from the saved raw attempts without loading the model again:

```bash
python scripts/revalidate_multiple.py \
  --raw-path output/qwen-2.5/multiple_seed42/raw_generations.jsonl \
  --records-path output/qwen-2.5/multiple_seed42/all_records.jsonl \
  --output-dir output/qwen-2.5/multiple_seed42_revalidated
```

### 2. Segment CoTs and Prepare DAG Batch Requests

```bash
python src/data_analysis/dag_analyzer.py \
  --mode batch \
  --raw-input "output/qwen-2.5/multiple_seed42/all_records.jsonl" \
  --output-dir "output/qwen-2.5/dag_analysis_50" \
  --provider deepseek \
  --model deepseek-chat
```

This single command segments every CoT, saves the reusable cache at
`output/qwen-2.5/dag_analysis_50/segmented_records.jsonl`, and then creates the
provider-ready file under `dag_analysis_50/batch/`. A cache newer than the raw
input is reused automatically. Pass `--refresh-segment-cache` to regenerate it,
or `--segmented-cache <path>` to choose another cache location.

The standalone `cot_segmenter.py` entrypoint remains available when only
segmentation is needed.

### 3. Merge DAG Batch Results

After the batch inference service returns results, merge them with the same
segmented cache:

```bash
python src/data_analysis/dag_analyzer.py \
  --mode merge-batch \
  --input "output/qwen-2.5/dag_analysis_50/segmented_records.jsonl" \
  --batch-results-file "path/to/batch_results.jsonl" \
  --output-dir "output/qwen-2.5/dag_analysis_50"
```

The merged DAG records are written to:

```text
output/qwen-2.5/dag_analysis_50/analyzed_records.jsonl
```

### 4. Compute GED Records

```bash
python src/data_analysis/ged_analysis.py \
  --output-root "output/qwen-2.5" \
  --variant-records "output/qwen-2.5/dag_analysis_50/analyzed_records.jsonl" \
  --correctness-file "output/qwen-2.5/multiple_seed42/all_records.jsonl" \
  --graph-cache "output/qwen-2.5/ged_graph_cache.pt" \
  --all-results-output "output/qwen-2.5/all_ged_results.jsonl"
```

By default, `ged_analysis.py` uses each problem's `original_0` response as the GED reference graph. For older workflows where original and variant DAG records are stored separately, pass `--original-records` explicitly.

The first run associates every DAG node with its original segmented CoT span,
compresses the graph, and encodes compressed-node text with
`sentence-transformers/all-MiniLM-L6-v2`. The compressed graphs and their shared
embedding matrix are stored together in `ged_graph_cache.pt`. Later runs load
this cache directly and skip graph compression, model initialization, and text
encoding. Use `--rebuild-graph-cache` after changing DAG annotations,
segmentation, compression, or the embedding model. To prepare the cache before
starting the slower GED search, add `--prepare-graph-cache-only`.

Node substitution follows the role-plus-text cost from the paper:
`lambda_role * role_mismatch + lambda_text * (1 - clipped_cosine)`. Both
weights default to `1.0` and can be changed with `--lambda-role` and
`--lambda-text`. The embedding model, optional revision, device, batch size,
and maximum token length are controlled by the corresponding
`--text-similarity-*` and `--embedding-*` arguments. Each result JSONL is
accompanied by an `all_ged_results.config.json` file recording these weights
and the graph-cache metadata.

Each GED record keeps the raw `ged` used by the existing within-problem
selection protocol and also reports `ged_normalized` in `[0, 1]`. The latter
uses the conservative unit-cost upper bound
`|V1| + |V2| + |E1| + |E2|`; `similarity_normalized` is its complement. This
prevents graph size and edge-count differences from producing an invalid
negative similarity while keeping the original raw GED available for audit.
Because the text-aware metric produces fractional costs and changes the raw
GED scale, thresholds calibrated on the earlier role-only GED should be
re-estimated before dataset curation.

### 5. Build The DSPR Dataset

```bash
python src/dspr_dataset/data_filter.py \
  --input "output/qwen-2.5/all_ged_results.jsonl" \
  --output "data/qwen/dspr_dataset.jsonl" \
  --top-k 5 \
  --min-ged-range 3 \
  --eligibility-output "output/qwen-2.5/eligible_problem_ids_ged_range_ge_3.json" \
  --model-name qwen
```

Dataset curation and k-fold eligibility use the same protocol: only correct,
non-timeout trajectories with a valid GED are considered; a problem-variant is
eligible when its within-group GED range is at least 3; Simple selects the five
lowest-GED trajectories and Hard selects the five highest-GED trajectories.
The eligibility JSON written above is consumed directly by
`scripts/build_kfold_splits.py`.

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
python scripts/train.py dspr \
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
python scripts/train_kfold.py dspr --gpus 0,1,2,3,4
```

Use `--dry-run` to validate the fold files and print the exact child commands
without loading a model. Outputs default to
`checkpoints/qwen_kfold_seed42/fold_0` through `fold_4`; the complete run
manifest is written to the parent directory.

### 9. Evaluate Qwen DSPR with Held-Out K-Folds

First run one held-out Fold-0 problem as a smoke test. Use a separate output
root so the smoke result cannot overwrite the formal fold output:

```bash
python scripts/evaluate.py dspr run \
  --folds 0 \
  --problem-id 2 \
  --gpus 0 \
  --output-root output/qwen_kfold_seed42_smoke
```

After the smoke test passes, run all folds. The evaluator reads each fold's
`trainer_state.json`, selects its best validation checkpoint, explicitly uses
prefix length 15, and dynamically assigns pending folds to available GPUs:

```bash
python scripts/evaluate.py dspr run --gpus 0,1,2,3,4
```

Each fold writes `all_records.jsonl`, `inference.log`, and `run_status.json`
under `output/qwen_kfold_seed42/fold_N`. The parent directory also contains an
`inference_manifest.json` with the exact checkpoint and command for every fold.

Validate and pool the five folds into one OOF result:

```bash
python scripts/evaluate.py dspr aggregate
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
python scripts/evaluate.py dspr aggregate \
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
python scripts/train_kfold.py lora --dry-run --gpus 0,1,2,3,4
python scripts/train_kfold.py lora --gpus 0,1,2,3,4
```

Run one held-out smoke test before starting all-fold inference:

```bash
python scripts/evaluate.py lora run \
  --folds 0 \
  --problem-id 2 \
  --gpus 0 \
  --output-root output/qwen_lora_qv_r3_seed42_smoke

python scripts/evaluate.py lora run --gpus 0,1,2,3,4
```

The inference entrypoint runs symbolic-answer evaluator self-tests before it
loads the model. Pool the exact OOF coverage and compare LoRA against both the
matched greedy base model and DSPR:

```bash
python scripts/evaluate.py lora aggregate \
  --base output/qwen-2.5/greedy/all_records.jsonl \
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
  --output-dir "output/qwen-2.5/dspr" \
  --prefix-length 50 \
  --max-new-tokens 4096
```

Use the same command with `--baseline spt` and an SPT checkpoint, or
`--baseline lora` and a PEFT adapter directory. The single-path entrypoint is
greedy-only by design: invalid or truncated outputs are recorded in
`generation_qc.json` and are never resampled.

The inference output is:

```text
output/qwen-2.5/dspr/all_records.jsonl
```

### 11. Report Accuracy

Evaluate one result file:

```bash
python src/utils/calculate_accuracy.py "output/qwen-2.5/dspr/all_records.jsonl"
```

Compare DSPR against a baseline result file:

```bash
python src/utils/calculate_accuracy.py \
  "output/qwen-2.5/dspr/all_records.jsonl" \
  --compare-file "output/qwen_matched_greedy/all_records.jsonl"
```

## Recommended Reading Order

For new contributors, the fastest way to understand the code is:

1. `scripts/run_dspr_pipeline.sh` for the high-level workflow.
2. `src/dspr/config.py` and `src/dspr/model.py` for the model interface.
3. `src/dspr_training/dataset.py`, `loss.py`, and `trainer.py` for training behavior.
4. `src/data_analysis/ged_analysis.py` and `src/dspr_dataset/data_filter.py` for dataset construction.
5. `scripts/inference.py`, `scripts/evaluate.py`, and `src/utils/calculate_accuracy.py` for evaluation.
