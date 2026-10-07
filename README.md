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
|   +-- run_dspr_pipeline.sh          # Staged K-fold DSPR training, inference, and OOF metrics
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
- `scripts/run_dag_analysis.py`: shared input resolution and CLI for DAG extraction.
- `scripts/run_ged_analysis.py`: GED CLI, graph preparation, and worker execution.
- `annotation/`: shared prompts and validation, normal requests, and batch preparation/merging.
- `graph/`: graph construction, compression, and prepared graph/embedding caches.
- `metrics/`: graph edit distance and node-text encoding.
- `measurement/`: joins GED with correctness, saves checkpoints, and exports results.
- `records.py`, `schemas.py`, and `config.py`: record adaptation, sample identities, and configuration.
- `src/utils/io.py`, `artifacts.py`, and `env.py`: common file I/O, content fingerprints, and environment loading.

The processing order remains segmentation, DAG annotation, graph compression and
text encoding, GED comparison, then DSPR dataset filtering. See
[`src/data_analysis/README.md`](src/data_analysis/README.md) for module boundaries.

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

- `data/canonical_math_paired.jsonl`: source math problems.
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

### Staged DSPR training

For the staged training variant, first train the prompt-only router on one
deduplicated example per problem-variant, then train the dual prefix on all
trajectory records while keeping the router fixed:

```bash
python scripts/train.py staged_dspr \
  --problem-dataset data/canonical_math_paired.jsonl \
  --kfold-root output/qwen-2.5/kfold \
  --fold 0 \
  --output-path checkpoints/qwen-2.5_staged_seed42 \
  --router-learning-rate 1e-5 \
  --prefix-learning-rate 4e-5 \
  --router-weight-decay 1e-3 \
  --router-target-smoothing 0.05 \
  --router-epochs 20 \
  --prefix-epochs 15
```

The router artifact is saved as
`<output>/router/router_trainable.pt`; the prefix-stage artifact is saved as
`<output>/prefix/dspr_trainable.pt`. To run the second stage separately, pass
`--run-stage prefix --router-checkpoint <path-to-router_trainable.pt>`. An
optional short joint calibration can be enabled with `--joint-epochs N`; it
uses the small `--joint-router-learning-rate` rather than the main router rate.

The staged pipeline treats `data/canonical_math_paired.jsonl` as the only canonical
source of problem-variant prompts. K-fold `train_ids.json` and `val_ids.json`
select the router examples from that source, while `fold_N/train.jsonl` and
`fold_N/val.jsonl` provide the refined CoT trajectories used only by the prefix
stage. Legacy `dcpr_train.jsonl` and `dcpr_val.jsonl` are not used by this
pipeline. The router context encoder is run once and cached as
`<output>/router_context_cache.pt`; subsequent router epochs use the cached
vectors directly. To share one cache across several folds, pass the same path
with `--router-context-cache-path` to each fold command.

To run a router-only context-layer ablation, use:

```bash
bash scripts/run_router_layer_ablation.sh
```

The script discovers `num_hidden_layers` from the model configuration and
sweeps hidden-state indices `0..L` (including the embedding output at index
0). Set `LAYERS="8 12 15 20"` for a smaller sweep, or override `MODEL_NAME`,
`problem_dataset`, `KFOLD_ROOT`, `FOLD`, and `OUTPUT_ROOT` for another setup. Each
run writes its own router history under `layer_<idx>/router/`. After training,
plot the curves and layer summary with:

```bash
python scripts/plot_router_layer_ablation.py \
  --root checkpoints/qwen-2.5_router_layer_ablation_seed42
```

This produces `router_layer_curves.png`, `router_layer_summary.png`, and CSV/
JSON summaries under `<root>/plots`.

The example commands below use Qwen2.5-Math-7B-Instruct and write new artifacts under `output/qwen-2.5`, `data/qwen`, and `checkpoints/qwen`. Results from the earlier pipeline are archived under `output/qwen_legacy`.

### 1. Generate Base Model Responses

```bash
python scripts/inference_multiple.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --data-path data/canonical_math_paired.jsonl \
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

### 2. Segment CoTs and Run DAG Analysis

Use the unified launcher for either mode. Normal mode supports concurrent
requests; `--concurrency` controls the worker count and `--max-retries` sets
the per-chain retry budget (default 5 retries after the initial request).

```bash
python scripts/run_dag_analysis.py \
  --mode normal \
  --raw-input "output/qwen-2.5/multiple_seed42/all_records.jsonl" \
  --output-dir "output/qwen-2.5/dag_analysis_50" \
  --provider deepseek \
  --model deepseek-chat \
  --concurrency 8 \
  --max-retries 5
```

Batch mode prepares provider-ready requests:

```bash
python scripts/run_dag_analysis.py \
  --mode batch \
  --raw-input "output/qwen-2.5/multiple_seed42/all_records.jsonl" \
  --output-dir "output/qwen-2.5/dag_analysis_50" \
  --provider deepseek \
  --model deepseek-chat
```

This single command segments every CoT, saves the reusable cache at
`output/qwen-2.5/dag_analysis_50/segmented_records.jsonl`, and then creates the
provider-ready file under `dag_analysis_50/batch/`. A cache matching the input
content and segmentation settings is reused automatically.
Pass `--refresh-segment-cache` to regenerate it,
or `--segmented-cache <path>` to choose another cache location.

The standalone `cot_segmenter.py` entrypoint remains available when only
segmentation is needed.

### 3. Merge DAG Batch Results

After the batch inference service returns results, merge them with the same
segmented cache:

```bash
python scripts/run_dag_analysis.py \
  --mode merge-batch \
  --input "output/qwen-2.5/dag_analysis_50/segmented_records.jsonl" \
  --batch-results-file "path/to/batch_results.jsonl" \
  --output-dir "output/qwen-2.5/dag_analysis_50"
```

The merged DAG records are written to:

```text
output/qwen-2.5/dag_analysis_50/analyzed_records.jsonl
```

Normal-mode runs additionally write each provider attempt (including the raw
response and any parse/provider error) to
`output/qwen-2.5/dag_analysis_50/normal/raw_responses.jsonl`.

Known provider variants of the `Conclude` tag are normalized during parsing.
To recover an existing normal-mode run from its saved raw responses without
calling the provider again, use `scripts/recover_dag_raw_responses.py`.

### 4. Compute GED Records

```bash
python scripts/run_ged_analysis.py \
  --output-root "output/qwen-2.5" \
  --variant-records "output/qwen-2.5/dag_analysis_50/analyzed_records.jsonl" \
  --correctness-file "output/qwen-2.5/multiple_seed42/all_records.jsonl" \
  --graph-cache "output/qwen-2.5/ged_graph_cache.pt" \
  --all-results-output "output/qwen-2.5/all_ged_results.jsonl"
```

By default, `run_ged_analysis.py` uses each problem's `original_0` response as the GED reference graph. For older workflows where original and variant DAG records are stored separately, pass `--original-records` explicitly.

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

Each GED record keeps the raw `ged` and also reports `ged_normalized` in
`[0, 1]`, used by the current within-problem selection protocol. The latter
uses the conservative unit-cost upper bound
`|V1| + |V2| + |E1| + |E2|`; `similarity_normalized` is its complement. This
prevents graph size and edge-count differences from producing an invalid
negative similarity while keeping the original raw GED available for audit.
Because the text-aware metric produces fractional costs and changes the raw
GED scale, thresholds calibrated on the earlier role-only GED should be
re-estimated before dataset curation.

### 5. Build The DSPR Dataset

```bash
bash scripts/prepare_qwen_dspr_data.sh
```

This reads `output/qwen-2.5/multiple_n16/ged/all_ged_results.jsonl` and writes
`data/qwen/dspr_dataset.jsonl`, `data/qwen/eligibility.json`, and the five folds
under `output/qwen-2.5/kfold/`. Set `PYTHON`, `GED_RESULTS`, or `DATA_ROOT` to override
the interpreter, input, or output root.

Only correct, non-timeout trajectories with a valid normalized GED are
considered. A problem-variant needs at least three such trajectories and a
normalized GED range of at least 0.10. Simple selects the five lowest values;
Hard selects the five highest when five are available. Eligibility is per
variant, so a single-sided group contributes prefix trajectories only for its
eligible variant.

### 6. K-fold Train/Validation/Test Splits

The problem dataset is `data/canonical_math_paired.jsonl`. The preparation
script assigns every raw problem group to one of five outer test folds with
seed 42, then selects validation groups from the remaining train/validation
pool. With the default eight validation buckets, each run uses approximately
70% of groups for training, 10% for validation, and 20% for testing. All
variants and trajectories of a problem stay together. Assignment balances
problem type, level, and the eligible/noneligible status of the simple and
hard variants. Eligibility controls which trajectories enter prefix training;
the outer test folds retain all problem groups.

Each `output/qwen-2.5/kfold/fold_N/` contains `train.jsonl`, `val.jsonl`,
`test.jsonl`, and the corresponding `train_ids.json`, `val_ids.json`, and
`test_ids.json` manifests. `manifest.json`, `coverage.json`, `fold_assignments.jsonl`, and
`fold_balance.csv` record the protocol and coverage at the K-fold root.

### 7. Train DSPR

Train one fold with the staged pipeline:

```bash
python scripts/train_kfold.py dspr --folds 0 --gpus 0
```

Each fold first trains the router on canonical problem-variant prompts from
`data/canonical_math_paired.jsonl`, selected by its `train_ids.json` and `val_ids.json`.
It then freezes the router and trains the dual prefix on that fold's
`train.jsonl` and `val.jsonl` trajectories. Joint calibration is disabled in
the K-fold entrypoint. Artifacts are saved separately:

```text
checkpoints/qwen_staged_kfold_seed42/fold_0/router/router_trainable.pt
checkpoints/qwen_staged_kfold_seed42/fold_0/prefix/dspr_trainable.pt
```

### 8. Train Qwen DSPR with the Prepared K-Folds

The five prepared Qwen folds can be trained in router -> prefix stages with seed
42 using the orchestration script below. Each fold gets an isolated checkpoint
directory and `train.log`.
GPU workers dynamically claim the next unfinished fold, so the same command
also works when fewer than five GPUs are available:

```bash
python scripts/train_kfold.py dspr --gpus 0,1,2,3,4
```

Use `--dry-run` to validate the fold files and print the exact child commands
without loading a model. Outputs default to
`checkpoints/qwen_staged_kfold_seed42/fold_0` through `fold_4`; the complete run
manifest is written to the parent directory. `run_status.json` records separate
router and prefix progress and early stopping results. Defaults are 20 router
epochs at learning rate `1e-5`, 15 prefix epochs at `4e-5`, and early stopping
patience 3 for each stage. Use `--router-epochs`, `--prefix-epochs`,
`--router-learning-rate`, and `--prefix-learning-rate` to override them.

`scripts/run_dspr_pipeline.sh` is the unified shell entrypoint, replacing the
former separate staged-training script. It runs training, held-out inference,
and OOF metric reporting with the same folds and model settings:

```bash
# Full five-fold pipeline on two GPUs
bash scripts/run_dspr_pipeline.sh --gpus 0,1

# Training only, or validate the training commands without loading a model
bash scripts/run_dspr_pipeline.sh --gpus 0 --train-only
bash scripts/run_dspr_pipeline.sh --gpus 0 --dry-run

# Run just one fold, including inference and metrics for that fold
bash scripts/run_dspr_pipeline.sh --gpus 0 --folds 0

# Evaluate already trained prefix checkpoints
bash scripts/run_dspr_pipeline.sh --gpus 0,1 --eval-only
```

Set `KFOLD_ROOT`, `problem_dataset`, `CHECKPOINT_DIR`, and `INFER_OUTPUT_DIR` to
override data and output locations. Training settings can be configured with
environment variables such as `ROUTER_EPOCHS`, `PREFIX_EPOCHS`,
`ROUTER_LEARNING_RATE`, and `PREFIX_LEARNING_RATE`; see `--help` for the full
list. A training dry run stops before inference because it creates no
checkpoints. To compare OOF results with a baseline, set `BASELINE_PATH`.

### 9. Evaluate Qwen DSPR with Held-Out K-Folds

First run one held-out Fold-0 problem as a smoke test. Use a separate output
root so the smoke result cannot overwrite the formal fold output:

```bash
python scripts/evaluate.py dspr run \
  --folds 0 \
  --problem-id 2 \
  --gpus 0 \
  --output-root output/qwen_staged_kfold_seed42_smoke
```

After the smoke test passes, run all folds. The evaluator reads each fold's
`prefix/checkpoint-*/trainer_state.json`, selects its best prefix validation checkpoint, explicitly uses
prefix length 15, and dynamically assigns pending folds to available GPUs:

```bash
python scripts/evaluate.py dspr run --gpus 0,1,2,3,4
```

Each fold writes `all_records.jsonl`, `inference.log`, and `run_status.json`
under `output/qwen_staged_kfold_seed42/fold_N`. The parent directory also contains an
`inference_manifest.json` with the exact checkpoint and command for every fold.
Inference reads each fold's `test.jsonl` and checks coverage against its
`test_ids.json` manifest.

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
  --data-path data/canonical_math_paired.jsonl \
  --output-dir output/qwen_matched_greedy \
  --max-new-tokens 4096
```

Then pass the resulting baseline file to the aggregator:

```bash
python scripts/evaluate.py dspr aggregate \
  --baseline output/qwen_matched_greedy/all_records.jsonl
```

The aggregator requires exact fold-ID coverage and writes
`oof_all_records.jsonl`, `oof_metrics.json`, `oof_metrics.csv`, and
`oof_alpha_summary.csv` under the K-fold output root.
The OOF problem count is inferred from the selected folds. Use `--folds 0` to
report a single fold and `--expected-problems N` for an explicit count assertion.
Canonical raw data and baseline files can cover a larger universe; metrics use
only the selected held-out IDs.

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
  --dspr output/qwen_staged_kfold_seed42/oof_all_records.jsonl
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
  --data-path "data/canonical_math_paired.jsonl" \
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
4. `src/data_analysis/measurement/` and `src/dspr_dataset/data_filter.py` for dataset construction.
5. `scripts/inference.py`, `scripts/evaluate.py`, and `src/utils/calculate_accuracy.py` for evaluation.
