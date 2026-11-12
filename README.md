# StructuredCoT

Implementation of **Dynamic Structural Prefix Routing for Structure-Aware Robust LLM Reasoning**.
The pipeline generates reasoning traces, builds IFD-Graphs, selects trajectories with graph edit distance (GED), and trains DSPR or its SPT/LoRA baselines.

## Setup and Local Configuration

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
export MODEL_NAME="/path/to/model"
export PROBLEM_DATASET="/path/to/paired-problems.jsonl"
export WORK_DIR="/path/to/your-run"
export GPUS="0,1"
```

Alternatively, use the GPU container:

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm dspr
```

All paths below are placeholders or directories for your own run. Set each artifact-file variable to the file produced by its preceding stage. Configure annotation-provider credentials locally. Use `--help` on an entry point for additional options.

## 1. Generate Base Responses and Sampled Trajectories

Generate one greedy response per Original/Simple/Hard problem for the base comparison:

```bash
python scripts/inference.py \
  --baseline base --backend vllm --model-name "$MODEL_NAME" \
  --data-path "$PROBLEM_DATASET" --output-dir "$WORK_DIR/base" \
  --max-new-tokens 2048 --max-model-len 4096 \
  --batch-size 128 --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 --max-num-batched-tokens 8192
```

Sample multiple Simple/Hard trajectories while retaining one greedy Original anchor:

```bash
python scripts/inference_multiple.py \
  --model "$MODEL_NAME" --data-path "$PROBLEM_DATASET" \
  --output-dir "$WORK_DIR/sampling" \
  --sampled-variants simple hard --samples-per-problem 16 \
  --temperature 0.7 --top-p 0.8 --top-k 20 --seed 42

export BASE_RECORDS="/path/to/base-predictions.jsonl"
export SAMPLED_RECORDS="/path/to/sampled-records.jsonl"
```

Sampling validates responses and retries invalid trajectories within a bounded attempt budget. Greedy inference records one response without resampling. Match generation settings when comparing methods. For base inference, `--batch-size` counts problem groups; `--max-num-seqs` controls vLLM concurrency. Use `--backend transformers` when appropriate.

To revalidate saved sampling attempts without regenerating responses:

```bash
export RAW_ATTEMPTS="/path/to/raw-attempts.jsonl"
python - <<'PY'
import os
from pathlib import Path
from inference.revalidate import revalidate_multiple

revalidate_multiple(
    raw_path=Path(os.environ["RAW_ATTEMPTS"]),
    records_path=Path(os.environ["SAMPLED_RECORDS"]),
    output_dir=Path(os.environ["WORK_DIR"]) / "revalidated",
    samples_per_problem=16,
)
PY
```

## 2. Cheap Screening

Require a valid Original anchor and at least three valid, correct trajectories on each perturbation variant. This stage does not use GED or require the Original answer to be correct.

```bash
python scripts/cheap_screen.py \
  --input "$SAMPLED_RECORDS" --canonical "$PROBLEM_DATASET" \
  --output-dir "$WORK_DIR/screening" --min-correct 3

export CANDIDATE_RECORDS="/path/to/screened-candidates.jsonl"
```

Keep the full sampled dataset for the held-out problem universe; the screened subset controls annotation and supervision preparation.

## 3. Segment CoTs and Annotate DAGs

Create a shared segmentation once, then reuse it for every annotator:

```bash
export SEGMENTED_RECORDS="/path/to/shared-segmentation.jsonl"
python -m data_analysis.cot_segmenter \
  --input "$CANDIDATE_RECORDS" --output "$SEGMENTED_RECORDS"

export ANNOTATOR="deepseek"
python scripts/run_dag_analysis.py \
  --mode normal --annotator "$ANNOTATOR" \
  --input "$SEGMENTED_RECORDS" --output-dir "$WORK_DIR/annotation/$ANNOTATOR" \
  --concurrency 8 --requests-per-minute 60 --max-retries 3 \
  --max-tokens 4096 --thinking off --resume
```

Presets include `deepseek`, `gpt`, `haiku`, and `qwen`. Override `--provider`, `--model`, or `--base-url` for another endpoint. Each annotator needs a separate directory. To combine segmentation and annotation, replace `--input` with `--raw-input` and set `--segmented-cache` explicitly.

For offline batch annotation, prepare requests and then merge the downloaded responses against the same segmentation:

```bash
python scripts/run_dag_analysis.py \
  --mode batch --annotator qwen --input "$SEGMENTED_RECORDS" \
  --output-dir "$WORK_DIR/annotation/qwen" --max-tokens 4096 --thinking off

export BATCH_RESPONSES="/path/to/downloaded-batch-responses.jsonl"
python scripts/run_dag_analysis.py \
  --mode merge-batch --annotator qwen --input "$SEGMENTED_RECORDS" \
  --batch-results-file "$BATCH_RESPONSES" \
  --output-dir "$WORK_DIR/annotation/qwen"
```

Add `--batch-errors-file` when the provider supplies a separate error file. For online runs, `--resume` reuses successful checkpoints; concurrency and request-rate limits control retry pressure.

The shell launcher exposes the same workflow:

```bash
ANNOTATOR="$ANNOTATOR" SEGMENTED_INPUT="$SEGMENTED_RECORDS" \
OUTPUT_DIR="$WORK_DIR/annotation/$ANNOTATOR" MODE=normal \
  bash scripts/run_dag_analysis.sh --resume
```

## 4. Compute GED

Compare each variant with its Original anchor from the same annotator. Keep compression, embedding configuration, costs, and compute budgets fixed across annotators.

```bash
export DAG_RECORDS="/path/to/annotated-records.jsonl"
export GED_RECORDS="/path/to/ged-records.jsonl"
export GRAPH_CACHE="/path/to/graph-cache.pt"

python scripts/run_ged_analysis.py \
  --output-root "$WORK_DIR/ged/$ANNOTATOR" \
  --variant-records "$DAG_RECORDS" --correctness-file "$CANDIDATE_RECORDS" \
  --graph-cache "$GRAPH_CACHE" --all-results-output "$GED_RECORDS" \
  --num-samples 16 --lambda-role 1 --lambda-text 1 \
  --ged-timeout 120 --max-ged-nodes 32 --max-ged-edges 64 \
  --workers 4 --resume
```

Use distinct GED files and caches for each annotator. Pass `--original-records` if Original annotations are stored separately. `--prepare-graph-cache-only` prepares compressed graphs and embeddings; `--rebuild-graph-cache` refreshes them after input or configuration changes. Timeout and size exclusions are recorded separately from successful distances.

Normalized GED is `clip(GED / B, 0, 1)`, where `B` is the total node and edge count of both graphs. Pin the encoder and its revision when comparing runs.

## 5. Select Trajectories and Build K-Folds

`src/dspr_dataset/data_filter.py` is the shared filtering implementation:

| Strategy | Role |
| --- | --- |
| `CheapScreenStrategy` | Screen valid anchors and sufficient correct candidates before GED |
| `EligibilityStrategy` | Require correct, non-timeout candidates with a usable score and sufficient count |
| `GEDRangeStrategy` | Apply the within-variant maximum-minus-minimum score threshold |
| `TopKStrategy` | Optionally retain the lowest Simple or highest Hard scores |

The following configuration retains all qualified trajectories in eligible variants. Set `top_k` explicitly only when a ranking cutoff is part of the intended protocol.

```bash
export ELIGIBILITY_FILE="/path/to/selection-manifest.json"
export TRAINING_RECORDS="/path/to/selected-trajectories.jsonl"
python - <<'PY'
import os
from dspr_dataset.data_filter import load_ged_results, select_trajectories_by_ged
from utils.io import write_json, write_jsonl

source = os.environ["GED_RECORDS"]
selection = select_trajectories_by_ged(
    load_ged_results(source),
    score_field="ged_normalized", min_correct_samples=3,
    min_ged_range=0.10, top_k=None,
    model_name=os.environ["MODEL_NAME"], source_file=source,
)
write_json(os.environ["ELIGIBILITY_FILE"], selection["eligibility"])
write_jsonl(os.environ["TRAINING_RECORDS"], selection["selected"])
PY

python scripts/build_kfold_splits.py \
  --raw-data "$SAMPLED_RECORDS" --eligible "$ELIGIBILITY_FILE" \
  --ged-results "$GED_RECORDS" --output-root "$WORK_DIR/splits" \
  --k 5 --seed 42 --validation-buckets 8
export KFOLD_ROOT="$WORK_DIR/splits/kfold"
```

The builder reuses the saved filtering protocol and verifies eligible IDs. It groups all variants and trajectories of each problem together, balancing type, level, and eligibility. Each outer fold is held out once; validation is selected from the remaining groups. Eligibility controls supervision, while held-out folds cover the full raw problem universe. All methods reuse the same folds. Use a new split directory for a changed dataset or protocol.

## 6. Staged DSPR Pipeline

DSPR first trains a router on deduplicated canonical prompts, then freezes the router and trains dual prefixes on selected trajectories. Router and prefix checkpoints are separate; validation selects checkpoints.

Configure the full K-fold launcher:

```bash
export CHECKPOINT_DIR="$WORK_DIR/models/dspr"
export INFER_OUTPUT_DIR="$WORK_DIR/predictions/dspr"
export BASELINE_PATH="$BASE_RECORDS"
bash scripts/run_dspr_pipeline.sh --gpus "$GPUS"
```

The launcher runs training, held-out inference, and aggregation. It reads the exported `MODEL_NAME`, `PROBLEM_DATASET`, and `KFOLD_ROOT`. Choose a stage or a fold subset as needed:

```bash
bash scripts/run_dspr_pipeline.sh --gpus "$GPUS" --train-only
bash scripts/run_dspr_pipeline.sh --gpus "$GPUS" --eval-only
bash scripts/run_dspr_pipeline.sh --gpus "$GPUS" --folds 0
bash scripts/run_dspr_pipeline.sh --gpus "$GPUS" --dry-run
```

These are alternative invocations. A training dry run validates and prints commands without loading a model. GPU workers claim pending folds; one GPU runs folds sequentially. Configure stage-specific epochs, learning rates, and early stopping through the launcher's environment variables or `--help`.

For a single fold, or independent router/prefix stages:

```bash
python scripts/train.py dspr \
  --model-name "$MODEL_NAME" --problem-dataset "$PROBLEM_DATASET" \
  --kfold-root "$KFOLD_ROOT" --fold 0 \
  --output-path "$WORK_DIR/models/dspr-single" \
  --prefix-length 15 --router-intermediate-dim 64 --router-dropout 0.10 \
  --router-learning-rate 1e-5 --prefix-learning-rate 4e-5 \
  --router-epochs 20 --prefix-epochs 15
```

Use `--run-stage router` for the first stage alone. For the second stage, use `--run-stage prefix --router-checkpoint "$ROUTER_CHECKPOINT"`, where `ROUTER_CHECKPOINT` is your saved router artifact. `--router-context-cache-path` permits reuse of compatible prompt encodings.

## 7. SPT and LoRA Pipelines

Both baselines use the same grouped folds and selected trajectories as DSPR. SPT learns a static prompt; LoRA trains adapters. Use separate model and prediction directories for each method.

```bash
METHOD=spt  # Change to lora for the LoRA pipeline.
python scripts/train_kfold.py "$METHOD" \
  --model-name "$MODEL_NAME" --fold-root "$KFOLD_ROOT" \
  --output-root "$WORK_DIR/models/$METHOD" --gpus "$GPUS"

python scripts/evaluate.py "$METHOD" run \
  --model-name "$MODEL_NAME" --checkpoint-root "$WORK_DIR/models/$METHOD" \
  --test-root "$KFOLD_ROOT" --id-root "$KFOLD_ROOT" \
  --output-root "$WORK_DIR/predictions/$METHOD" --gpus "$GPUS" \
  --max-new-tokens 2048

export DSPR_RECORDS="/path/to/aggregated-dspr-predictions.jsonl"
python scripts/evaluate.py "$METHOD" aggregate \
  --result-root "$WORK_DIR/predictions/$METHOD" --id-root "$KFOLD_ROOT" \
  --raw-data "$PROBLEM_DATASET" --base "$BASE_RECORDS" --dspr "$DSPR_RECORDS"
```

SPT accepts `--prefix-length`; LoRA accepts `--rank`, `--lora-alpha`, and `--target-modules`. Parameter-matched LoRA checks the configured DSPR budget; for another model, set `--dspr-parameter-budget` accordingly. Use `--no-enforce-parameter-budget` for an explicitly unmatched configuration. Add `--folds 0` for one fold or `--dry-run` to inspect training/inference commands. SPT aggregation can additionally take `--lora "$LORA_RECORDS"`.

The baseline shell launcher supports `STAGE=train|eval|all`, `FOLDS`, `GPUS`, and `DRY_RUN`. Its evaluation and comparator settings are defined in the launcher; use the explicit commands above for custom models and comparison files. The training-only shortcut forwards method arguments:

```bash
STAGE=train GPUS="$GPUS" KFOLD_ROOT="$KFOLD_ROOT" \
CHECKPOINT_ROOT="$WORK_DIR/models/$METHOD" \
  bash scripts/run_baseline_pipeline.sh "$METHOD" --model-name "$MODEL_NAME"
```

## 8. Held-Out Evaluation and Aggregation

To run the DSPR stages separately:

```bash
python scripts/train_kfold.py dspr \
  --model-name "$MODEL_NAME" --problem-dataset "$PROBLEM_DATASET" \
  --fold-root "$KFOLD_ROOT" --output-root "$CHECKPOINT_DIR" --gpus "$GPUS"

python scripts/evaluate.py dspr run \
  --model-name "$MODEL_NAME" --checkpoint-root "$CHECKPOINT_DIR" \
  --test-root "$KFOLD_ROOT" --id-root "$KFOLD_ROOT" \
  --output-root "$INFER_OUTPUT_DIR" --gpus "$GPUS" --max-new-tokens 2048

python scripts/evaluate.py dspr aggregate \
  --result-root "$INFER_OUTPUT_DIR" --id-root "$KFOLD_ROOT" \
  --raw-data "$PROBLEM_DATASET" --baseline "$BASE_RECORDS"
```

Evaluation selects validation checkpoints and checks held-out ID coverage. Aggregation combines out-of-fold predictions and reports full, eligible, and noneligible slices; matched comparisons support paired bootstrap intervals and McNemar tests. Use the same `--folds` subset across inference and aggregation. `--expected-problems` optionally asserts the intended test population size.

For fixed-routing controls, reuse the learned run's per-fold checkpoints and recorded inference arguments:

```bash
export LEARNED_RUN_DIR="/path/to/learned-router-predictions"
REFERENCE_RUN_ROOT="$LEARNED_RUN_DIR" CHECKPOINT_ROOT="$CHECKPOINT_DIR" \
KFOLD_ROOT="$KFOLD_ROOT" PROBLEM_DATASET="$PROBLEM_DATASET" GPUS="$GPUS" \
OUTPUT_ROOT="$WORK_DIR/predictions/fixed-alpha" \
  bash scripts/run_fixed_alpha_kfold.sh
```

The launcher runs alpha 0, 0.5, and 1 sequentially, with parallel folds within each condition,
then aggregates each condition. Results use separate `alpha_0`, `alpha_0p5`, and `alpha_1`
directories. Each reference fold must have a successful `run_status.json`; its exact checkpoint
must still exist under `CHECKPOINT_ROOT`. Recorded generation and architecture arguments are
reused. `DRY_RUN=1` checks commands without generation; the default `RESUME=1` skips complete
matching folds and reruns interrupted folds. A failed condition stops the sequence.
Set `BASELINE_PATH` to a matching base prediction file to include base comparisons.
For a single condition, use `scripts/evaluate.py dspr run --forced-alpha 0.5`
with `--reference-run-root "$LEARNED_RUN_DIR"` and the usual checkpoint, fold, and output paths.

For a smoke run, add `--folds 0 --problem-id "$PROBLEM_ID"` to inference and choose a separate run directory. Such a partial run is not a complete fold for aggregation.

Evaluate a single DSPR checkpoint directly:

```bash
export MODEL_CHECKPOINT="/path/to/model-checkpoint"
python scripts/inference.py \
  --baseline dspr --model-name "$MODEL_NAME" --checkpoint "$MODEL_CHECKPOINT" \
  --data-path "$PROBLEM_DATASET" --output-dir "$WORK_DIR/single-checkpoint" \
  --prefix-length 15 --router-intermediate-dim 64 --router-dropout 0.10 \
  --max-new-tokens 2048
```

Match architecture arguments to the saved checkpoint. Use `--baseline spt` with its prompt length, or `--baseline lora` with an adapter directory. Score or compare prediction files with:

```bash
export PREDICTION_RECORDS="/path/to/predictions.jsonl"
python src/utils/calculate_accuracy.py "$PREDICTION_RECORDS"
python src/utils/calculate_accuracy.py "$PREDICTION_RECORDS" --compare-file "$BASE_RECORDS"
```

## 9. Annotator Consistency and Ablation Workflows

For cross-annotator analysis, keep questions, responses, segmentation, and step IDs fixed. Repeat DAG annotation and GED with separate annotator directories and identical GED settings. Use base greedy traces for role/dependency consistency and screened multi-sample traces for selection consistency.

```bash
export BASE_ANALYSIS_ROOT="/path/to/base-analysis-bundle"
export SAMPLE_ANALYSIS_ROOT="/path/to/multi-sample-analysis-bundle"
python scripts/analyze_annotator_reliability.py \
  --base-root "$BASE_ANALYSIS_ROOT" --canonical "$PROBLEM_DATASET" \
  --output-dir "$WORK_DIR/analysis/annotation" --bootstrap-samples 10000 --seed 42

python scripts/analyze_selection_consistency.py \
  --data-root "$SAMPLE_ANALYSIS_ROOT" --canonical "$PROBLEM_DATASET" \
  --eligibility-config "$ELIGIBILITY_FILE" \
  --output-dir "$WORK_DIR/analysis/selection" \
  --report "$WORK_DIR/selection-summary.md" --bootstrap-samples 10000 --seed 42
```

The analysis bundles must follow the input layout expected by each script, including matching records, annotator subdirectories, GED configurations, and failure records. These commands analyze saved artifacts without calling annotators or rerunning GED. Selection consistency uses the shared count/range strategies without a top-k cutoff.

To sweep the router context layer and summarize its training curves:

```bash
MODEL_NAME="$MODEL_NAME" PROBLEM_DATASET="$PROBLEM_DATASET" \
KFOLD_ROOT="$KFOLD_ROOT" OUTPUT_ROOT="$WORK_DIR/router-ablation" \
FOLD=0 LAYERS="8 12 15 20" bash scripts/run_router_layer_ablation.sh

python scripts/plot_router_layer_ablation.py \
  --root "$WORK_DIR/router-ablation" --output-dir "$WORK_DIR/router-plots"
```

Omit `LAYERS` to inspect all model layers. For a fixed-routing inference comparison, add `--forced-alpha` to single-checkpoint DSPR inference and keep all other settings unchanged.

## Code Structure

| Module | Responsibility |
| --- | --- |
| `src/dspr/`, `src/dspr_training/` | Dual prefixes, router, and staged training |
| `src/spt/`, `src/spt_training/` | Static prompt baseline |
| `src/dspr_dataset/` | Shared screening and trajectory-selection strategies |
| `src/data_analysis/` | Segmentation, annotation, graph construction, and GED |
| `src/inference/` | Model adapters, sampling, and response validation |
| `src/experiment_pipeline/` | Training orchestration, held-out evaluation, and aggregation |
| `src/utils/` | Answer evaluation, I/O, and progress reporting |
