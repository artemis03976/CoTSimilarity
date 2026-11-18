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

Answer scoring runs separately from model inference. On the scoring machine, create
a CPU environment with the agreed SymPy 1.14.0 dependencies:

```bash
python -m venv /path/to/scoring-env
export SCORE_PYTHON="/path/to/scoring-env/bin/python"
"$SCORE_PYTHON" -m pip install -r requirements-scoring.txt
```

This environment does not require torch, transformers, vLLM, model weights, or a GPU.
Keep the training/inference environment unchanged. Copy saved predictions and their
generation QC files to the scoring machine, using the same repository code revision.
For K-fold aggregation, also copy the fold ID files and problem dataset; checkpoints
are not needed. Set the path variables below to their locations on that machine.

Training, held-out inference, and aggregation accept `--config FILE.yaml`.
Presets are `configs/dspr.yaml`, `configs/dspr_ged.yaml`, `configs/spt.yaml`, and
`configs/lora.yaml`. Copy a preset for each run and edit its `common`, `training`,
`inference`, and `aggregation` sections. `common` holds shared model, path, and fold
settings; each entrypoint reads its own section. Explicit CLI options override YAML.
Paths are relative to the repository root; YAML values do not expand shell variables.
New runs save effective settings as YAML. Predictions and machine-readable manifests
remain JSON/JSONL.

Install the declared dependencies. GPU auto-detection uses `CUDA_VISIBLE_DEVICES`
or `nvidia-smi`; on a CPU machine, pass `--gpus 0` when inspecting a dry run.

## 1. Generate Base Responses and Sampled Trajectories

Choose `configs/infer_qwen25.yaml` for Qwen2.5-Math-7B-Instruct,
`configs/infer_deepseek.yaml` for DeepSeek-Math-7B-Instruct, or
`configs/infer_phi4.yaml` for Microsoft Phi-4 (14B). Each preset defines the
model and dataset under `common`, with separate output directories and decoding
settings under `greedy` and `sampling`. To switch models, copy the preset and update
the model name and both output directories. Explicit CLI options override the preset.

```bash
CONFIG=configs/infer_qwen25.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 python scripts/inference_multiple.py --config "$CONFIG"
```

Generate one greedy response per Original/Simple/Hard problem for the base comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
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

Inference only checks generation quality. It saves `correct: null`,
`answer_correct: null`, and `answer_evaluation_status: "pending"`. Score the saved
responses on the scoring machine before computing accuracy or running cheap screen:

```bash
"$SCORE_PYTHON" scripts/score_answers.py "$BASE_RECORDS" "$SAMPLED_RECORDS" --workers 4
```

The scorer uses the problem text and reference saved in each prediction file and
replaces only answer scores in place. It records evaluator versions and code hashes
in the accompanying QC metadata. It does not regenerate responses or change quality
flags, cheap-screen artifacts, eligible lists, or folds. Unsupported references are
marked `unsupported_reference` with `answer_correct: null`; under the existing
protocol, their `correct` flag remains false. Pending scores are rejected by accuracy,
aggregation, and screening commands.

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

"$SCORE_PYTHON" scripts/score_answers.py "$WORK_DIR/revalidated"
```

Revalidation selects trajectories using mechanical quality only and resets the
new materialisation to pending scores; score it before further selection.

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
mkdir -p "$WORK_DIR"
export DSPR_CONFIG="$WORK_DIR/dspr.yaml"
cp configs/dspr.yaml "$DSPR_CONFIG"
```

Set `common.model_name`, `problem_dataset`, `fold_root`, `checkpoint_root`, and
`result_root` in that copy to your local paths. Training hyperparameters belong in
`training`; generation settings belong in `inference`. The launcher runs training
and held-out generation, then prints the offline scoring and aggregation commands:

```bash
python scripts/run_pipeline.py dspr --config "$DSPR_CONFIG" --gpus "$GPUS"

# Alternative invocations:
python scripts/run_pipeline.py dspr --config "$DSPR_CONFIG" --gpus "$GPUS" --train-only
python scripts/run_pipeline.py dspr --config "$DSPR_CONFIG" --gpus "$GPUS" --eval-only
python scripts/run_pipeline.py dspr --config "$DSPR_CONFIG" --gpus "$GPUS" --folds 0
python scripts/run_pipeline.py dspr --config "$DSPR_CONFIG" --gpus "$GPUS" --dry-run
```

A dry run validates and prints commands without loading a model. GPU workers claim
pending folds; one GPU runs folds sequentially. All three methods use
`python scripts/run_pipeline.py METHOD --config FILE.yaml`. Experiment settings
come from YAML and CLI options. Invoke it with the Python interpreter of the
training environment; child processes use that same interpreter.

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
Router ID inputs must be JSON manifests containing an `all` list. Missing manifests
are errors; router IDs are never inferred from the filtered prefix trajectories.
An incompatible context cache is also an error: choose a new cache path when
changing its dataset, model, or context layer.

For GED-weighted endpoint preference training, use the dedicated preset. Each sampling
epoch draws one trajectory per problem and variant. Its LM loss uses the frozen
router mixture. The auxiliary loss scores the same answer at both endpoints.
Supply the trajectory GED JSONL through `training.ged_results` or `--ged-results`.

```bash
python scripts/run_pipeline.py dspr --config configs/dspr_ged.yaml \
  --model-name "$MODEL_NAME" --problem-dataset "$PROBLEM_DATASET" \
  --fold-root "$KFOLD_ROOT" --gpus "$GPUS" --ged-results "$GED_RECORDS" \
  --checkpoint-root "$WORK_DIR/models/dspr-ged" \
  --result-root "$WORK_DIR/predictions/dspr-ged"
```

To reuse saved fold routers, add `--router-checkpoint-root "$ROUTER_CHECKPOINT_ROOT"`.
Each source fold must contain `fold_N/router/router_trainable.pt` and match the model,
router architecture and fold assignment. The prefix stage starts with fresh prefixes.
`--folds 0 --train-only` runs the first fold. For single-stage training, use
`scripts/train.py dspr --config configs/dspr_ged.yaml --fold 0 --run-stage prefix
--router-checkpoint "$ROUTER_CHECKPOINT" --output-path "$WORK_DIR/models/dspr-ged/fold_0"`.

Within each split and prompt, usable normalized GED scores receive ascending mean
ranks for ties. With rank `k` among `n` trajectories, Hard uses `k/(n+1)` and Simple
uses `1-k/(n+1)`. Missing GED gives zero auxiliary weight. The preset uses
`ged_preference_weight: 0.1` and an effective batch of 16. Sampling preserves the
optimizer-step budget computed from the full trajectory count. Validation reports
the fraction of scored Simple and Hard trajectories favoring their intended endpoint
and selects checkpoints by routed LM loss.

## 7. SPT and LoRA Pipelines

Both baselines use the same grouped folds and selected trajectories as DSPR. SPT learns a static prompt; LoRA trains adapters. Use separate model and prediction directories for each method.

```bash
METHOD=spt  # Change to lora for the LoRA pipeline.
python scripts/train_kfold.py "$METHOD" \
  --config "configs/$METHOD.yaml" \
  --model-name "$MODEL_NAME" --fold-root "$KFOLD_ROOT" \
  --output-root "$WORK_DIR/models/$METHOD" --gpus "$GPUS"

python scripts/evaluate.py "$METHOD" run \
  --config "configs/$METHOD.yaml" \
  --model-name "$MODEL_NAME" --checkpoint-root "$WORK_DIR/models/$METHOD" \
  --test-root "$KFOLD_ROOT" --id-root "$KFOLD_ROOT" \
  --output-root "$WORK_DIR/predictions/$METHOD" --gpus "$GPUS" \
  --max-new-tokens 2048

# On the scoring machine, after copying the predictions:
"$SCORE_PYTHON" scripts/score_answers.py "$WORK_DIR/predictions/$METHOD"
export DSPR_RECORDS="/path/to/aggregated-dspr-predictions.jsonl"
"$SCORE_PYTHON" scripts/evaluate.py "$METHOD" aggregate \
  --config "configs/$METHOD.yaml" \
  --result-root "$WORK_DIR/predictions/$METHOD" --id-root "$KFOLD_ROOT" \
  --raw-data "$PROBLEM_DATASET" --base "$BASE_RECORDS" --dspr "$DSPR_RECORDS"
```

SPT accepts `--prefix-length`; LoRA accepts `--rank`, `--lora-alpha`, and `--target-modules`. Parameter-matched LoRA checks the configured DSPR budget; for another model, set `--dspr-parameter-budget` accordingly. Use `--no-enforce-parameter-budget` for an explicitly unmatched configuration. Add `--folds 0` for one fold or `--dry-run` to inspect training/inference commands. SPT aggregation can additionally take `--lora "$LORA_RECORDS"`.

The shared pipeline launcher accepts the same `--config`, `--gpus`, `--folds`,
`--stage all|train|eval`, and `--dry-run` options as DSPR. It ends after generation
and prints the separate scoring/aggregation commands. For example:

```bash
python scripts/run_pipeline.py "$METHOD" --config "configs/$METHOD.yaml" \
  --model-name "$MODEL_NAME" --problem-dataset "$PROBLEM_DATASET" \
  --fold-root "$KFOLD_ROOT" --gpus "$GPUS" \
  --checkpoint-root "$WORK_DIR/models/$METHOD" \
  --result-root "$WORK_DIR/predictions/$METHOD" --train-only
```

## 8. Held-Out Evaluation and Aggregation

To run the DSPR stages separately:

```bash
export CHECKPOINT_DIR="$WORK_DIR/models/dspr"
export INFER_OUTPUT_DIR="$WORK_DIR/predictions/dspr"
python scripts/train_kfold.py dspr \
  --model-name "$MODEL_NAME" --problem-dataset "$PROBLEM_DATASET" \
  --fold-root "$KFOLD_ROOT" --output-root "$CHECKPOINT_DIR" --gpus "$GPUS"

python scripts/evaluate.py dspr run \
  --model-name "$MODEL_NAME" --checkpoint-root "$CHECKPOINT_DIR" \
  --test-root "$KFOLD_ROOT" --id-root "$KFOLD_ROOT" \
  --output-root "$INFER_OUTPUT_DIR" --gpus "$GPUS" --max-new-tokens 2048

# On the scoring machine, after copying the predictions:
"$SCORE_PYTHON" scripts/score_answers.py "$INFER_OUTPUT_DIR"
"$SCORE_PYTHON" scripts/evaluate.py dspr aggregate \
  --result-root "$INFER_OUTPUT_DIR" --id-root "$KFOLD_ROOT" \
  --raw-data "$PROBLEM_DATASET" --baseline "$BASE_RECORDS"
```

The `run` action selects validation checkpoints and checks generated response/held-out ID coverage; it does not score answers. The DSPR pipeline launcher also ends after generation. The offline scorer accepts one or more JSONL files or run directories; for a K-fold directory it scores each immediate `fold_*/all_records.jsonl`, with `--folds` available to select a subset. Aggregation combines scored out-of-fold predictions and reports full, eligible, and noneligible slices; matched comparisons support paired bootstrap intervals and McNemar tests. Use the same `--folds` subset across inference, scoring, and aggregation. `--expected-problems` optionally asserts the intended test population size. Re-run aggregation after rescoring folds to refresh existing OOF files and metrics. Comparator files must also have been scored using the agreed environment.

For fixed-routing controls, reuse the learned run's per-fold checkpoints and recorded inference arguments:

```bash
export LEARNED_RUN_DIR="/path/to/learned-router-predictions"
REFERENCE_RUN_ROOT="$LEARNED_RUN_DIR" CHECKPOINT_ROOT="$CHECKPOINT_DIR" \
KFOLD_ROOT="$KFOLD_ROOT" PROBLEM_DATASET="$PROBLEM_DATASET" GPUS="$GPUS" \
OUTPUT_ROOT="$WORK_DIR/predictions/fixed-alpha" \
  bash scripts/run_fixed_alpha_kfold.sh
```

The launcher generates alpha 0, 0.5, and 1 sequentially, with parallel folds within each condition,
then prints the offline scoring/aggregation commands. Results use separate `alpha_0`, `alpha_0p5`, and `alpha_1`
directories. Each reference fold must have a successful `run_status.json`; its exact checkpoint
must still exist under `CHECKPOINT_ROOT`. Recorded generation and architecture arguments are
reused. `DRY_RUN=1` checks commands without generation; the default `RESUME=1` skips complete
matching folds and reruns interrupted folds. A failed condition stops the sequence.
Set `BASELINE_PATH` to a matching base prediction file to include base comparisons.
After transferring the three condition directories to the scoring machine:

```bash
for CONDITION in alpha_0 alpha_0p5 alpha_1; do
  RUN_DIR="$WORK_DIR/predictions/fixed-alpha/$CONDITION"
  "$SCORE_PYTHON" scripts/score_answers.py "$RUN_DIR"
  "$SCORE_PYTHON" scripts/evaluate.py dspr aggregate \
    --result-root "$RUN_DIR" --id-root "$KFOLD_ROOT" \
    --raw-data "$PROBLEM_DATASET" --baseline "$BASE_RECORDS"
done
```

For a single condition, use `scripts/evaluate.py dspr run --forced-alpha 0.5`
with `--reference-run-root "$LEARNED_RUN_DIR"` and the usual checkpoint, fold, and output paths.
Checkpoint selection `best` requires validation evidence or the final saved model;
use `--checkpoint-selection latest` to explicitly select the latest retained step.

For the paired structure probe, prepare the three scored OOF conditions and the matching
base greedy Original graphs. The preparation step selects groups by source and reuses
matching annotations. The annotation request contains the problem and reasoning steps.

```bash
PROBE_DIR="$WORK_DIR/alpha-structure"
python scripts/probe_alpha_structure.py prepare --dataset "$PROBLEM_DATASET" \
  --base-root "$WORK_DIR/base" --fixed-root "$WORK_DIR/predictions/fixed-alpha" \
  --reference-root "$LEARNED_RUN_DIR" --output-dir "$PROBE_DIR"
python scripts/run_dag_analysis.py --input "$PROBE_DIR/pending.jsonl" \
  --output-dir "$PROBE_DIR/annotation" --annotator volcengine --variants simple hard
python scripts/probe_alpha_structure.py measure --output-dir "$PROBE_DIR"
python scripts/probe_alpha_structure.py report --output-dir "$PROBE_DIR"
```

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

Match architecture arguments to the saved checkpoint. Use `--baseline spt` with its prompt length, or `--baseline lora` with an adapter directory. Score first, then compute or compare accuracy from the saved labels:

```bash
export PREDICTION_RECORDS="/path/to/predictions.jsonl"
"$SCORE_PYTHON" scripts/score_answers.py "$PREDICTION_RECORDS"
"$SCORE_PYTHON" src/utils/calculate_accuracy.py "$PREDICTION_RECORDS"
"$SCORE_PYTHON" src/utils/calculate_accuracy.py "$PREDICTION_RECORDS" --compare-file "$BASE_RECORDS"
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

For manual inspection, the GED viewer supports percentile tails and an absolute
threshold through one entrypoint:

```bash
python -m utils.visualization.visualize_ged_similarity \
  --csv "/path/to/ged-comparison-table.csv" \
  --segmented "$SEGMENTED_RECORDS" --analyzed "$DAG_RECORDS" \
  --output "$WORK_DIR/ged-examples.html" --max-ged 0
```

The CSV uses `original_vs_simple_ged` and `original_vs_hard_ged` columns. Omit
`--max-ged` to select the lowest/highest 10%, or set `--low-percentile` and
`--high-percentile` explicitly. Response and DAG viewers remain available through
`python -m utils.visualization.visualize_response --input FILE --output PAGE` and
`python -m utils.visualization.visualize_dag --input FILE --output PAGE`.

## Code Structure

| Module | Responsibility |
| --- | --- |
| `configs/` | YAML presets for DSPR, GED-weighted endpoint preferences, SPT, and LoRA |
| `src/dspr/`, `src/dspr_training/` | Dual prefixes, router, and staged training |
| `src/spt/`, `src/spt_training/` | Static prompt baseline |
| `src/dspr_dataset/` | Shared screening and trajectory-selection strategies |
| `src/data_analysis/` | Segmentation, annotation, graph construction, and GED |
| `src/inference/` | Model adapters, sampling, and response validation |
| `src/experiment_pipeline/` | Training orchestration, held-out evaluation, and aggregation |
| `src/utils/` | Answer evaluation, configuration, I/O, and shared training/progress tools |

All three training methods use `src/utils/datasets.py` for trajectory encoding;
DSPR requests additional router inputs. The router stage reads canonical prompts
through `src/dspr_training/dataset.py`. Training and inference share the math prompt
in `src/utils/prompts.py` and the GPU fold queue in `src/utils/scheduling.py`.
OOF calculations live in `src/experiment_pipeline/metrics.py`; `aggregate.py`
validates input files and exports the results.
