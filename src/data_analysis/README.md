# Data Analysis

The experiment chain is unchanged:

```text
sampled responses -> CoT segmentation -> DAG annotation
-> graph compression -> node-text encoding -> GED -> DSPR filtering
```

```text
data_analysis/
  schemas.py                   Sample IDs and annotation fields
  config.py                    LLM provider and generation settings
  records.py                   Current/legacy record adapters and sample joins
  cot_segmenter.py             Existing segmentation algorithm and standalone CLI
  annotation/
    prompts.py                 Existing annotation prompt
    protocols.py               Native Chat Completions / Messages payloads and response metadata
    parsing.py                 Shared response parser and DAG validator
    client.py                  Provider requests and shared rate limiting
    normal.py                  Concurrent annotation and per-sample recovery
    batch.py                   Batch request preparation and result merging
  graph/
    builder.py                 Problem/step/external graph construction
    compression.py             Existing contraction then parallel folding
    cache.py                   Compressed graphs and node embeddings
  metrics/
    text.py                    Existing encoder, pooling, and normalization
    ged.py                     Existing edit costs, search, and normalization
  measurement/
    analysis.py                Reference selection and per-problem comparisons
    checkpoints.py             Per-problem recovery and completion status
    export.py                  Existing GED CSV layout
```

`scripts/run_ged_analysis.py` owns the GED CLI, graph preparation, and worker
execution.

Generic JSON/JSONL reading and atomic writing live in `utils/io.py`. Content
fingerprints live in `utils/artifacts.py`; dotenv loading lives in `utils/env.py`.
`records.py` only adapts experiment-specific formats. Visualization adapters live
in `utils/visualization/dag_records.py` and use the same annotation validator.

## Entry Points

- `python scripts/run_dag_analysis.py --mode normal ...`: annotate normally;
  add `--resume` to reuse successful sample checkpoints and retry failed samples.
- `python scripts/run_dag_analysis.py --mode batch ...`: prepare requests.
- `python scripts/run_dag_analysis.py --mode merge-batch ...`: merge downloaded
  responses; provider errors, invalid DAGs, and missing results are recorded per
  sample. Preparing and merging batch files requires no API credentials.
- `python scripts/run_ged_analysis.py ...`: compute GED with the existing flags.
- `python scripts/analyze_annotator_reliability.py`: audit the four saved base
  annotators and compare roles, directed dependencies and GED values offline.
  Saves all six pairwise comparisons, problem-cluster bootstrap CIs, coverage,
  failure records, confusion matrices, case figures and a LaTeX table under
  `--output-dir`. Uses the four-way
  common-success set; root and External edges are reported separately and
  both-empty edge sets are undefined. The base greedy data have one response
  per variant; within-variant ranking and eligibility stability are measured
  separately on the n16 candidate pool without a top-k cutoff. See the construction section of
  [the unified Section 4.1 report](../../docs/SECTION_4_1_RESULTS_QWEN25.md#construction-consistency).

### Annotator selection

| `--annotator` | Model | API | Environment key |
| --- | --- | --- | --- |
| `deepseek` | `deepseek-flash` | OpenAI-compatible Chat Completions | `DEEPSEEK_API_KEY` |
| `gpt` | `gpt-6-luna` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `haiku` | `claude-haiku-5-5` | Native Anthropic Messages | `ANTHROPIC_API_KEY` |
| `qwen` | `qwen3.8-flash` | Bailian batch JSONL; optional online Chat Completions | `DASHSCOPE_API_KEY` (online only) |
| `volcengine` | `ep-20261001103240-d9m5m` | Existing Volcengine deployment | `VOLCENGINE_API_KEY` |

Credentials are read from `.env` or exported variables. Each provider also accepts
`<PROVIDER>_BASE_URL`; `LLM_API_KEY` / `LLM_BASE_URL` remain generic fallbacks.
Use `--base-url` for an explicit endpoint override. Models and providers can be
overridden with `--model` / `--provider`.
The transport sends native HTTP requests via `httpx`, so newly released model IDs
do not depend on the model catalog in the previously pinned LiteLLM version.

The four new model presets explicitly **disable thinking** by default. This is
recorded as a run setting, not inferred from earlier provider defaults. Use `--thinking on`
to enable it, and optionally `--reasoning-effort` for GPT, Claude or DeepSeek.
Provider effort scales differ; record the emitted settings when comparing models.
GPT uses `max_completion_tokens`; Claude uses a top-level `system` field and
reads only text blocks, excluding thinking blocks. Neither preset sends the old
`temperature=0, top_p=1` combination. Qwen batch files place `enable_thinking`
directly inside `body`, alongside `model`. No JSON-object response mode is forced:
the existing prompt and validator require a top-level JSON **array** of steps.

`--max-tokens` defaults to 4096; increase it if enabling reasoning or annotating
long chains. `--timeout` (60 seconds) and `--requests-per-minute` (60 shared across
workers) control online requests. Truncations, refusals and malformed DAGs are
recorded as failures and follow the same retry budget.

For raw inputs, preset output defaults to `<raw-input-parent>/dag_analysis/<annotator>/`.
For `--input`, it defaults to `<segmented-input-parent>/<annotator>/`.
`--output-dir` can override this. Outputs and checkpoints from a
different provider/model cannot be reused in that directory. Model response IDs,
reported model versions, token usage and stop reasons are saved with annotations
and raw responses when returned by the provider.

To keep segmentation identical across annotators, use the same existing file:

```bash
export SEGMENTED_INPUT="/path/to/shared-segmentation.jsonl"
python scripts/run_dag_analysis.py --annotator gpt --input "$SEGMENTED_INPUT" --max-retries 3
python scripts/run_dag_analysis.py --annotator haiku --input "$SEGMENTED_INPUT" --max-retries 3
python scripts/run_dag_analysis.py --annotator qwen --input "$SEGMENTED_INPUT"
```

Alternatively pass `--raw-input` to generate a segmentation cache. It is mutually
exclusive with `--input`. Both options apply to all modes. `qwen` defaults to
offline batch preparation; the other annotators default to normal API calls with
checkpoint resume enabled. Select the frozen candidate-pool segmentation for
multi-sample annotation.

Generate the n16 Qwen batch using the frozen shared segmentation:

```bash
python scripts/run_dag_analysis.py --annotator qwen --mode batch \
  --input "$SEGMENTED_INPUT" \
  --model qwen3.8-flash --max-tokens 4096 --thinking off
```

This writes `<segmented-input-parent>/qwen/batch/batch_requests.jsonl`. All stored
samples are included; use the same input for each annotator before GED selection.

After all four n16 GED runs are available, compare their supervision selections:

```bash
python scripts/analyze_selection_consistency.py \
  --data-root /path/to/multi-sample-analysis-bundle \
  --eligibility-config /path/to/selection-manifest.json \
  --canonical /path/to/paired-problems.jsonl \
  --output-dir /path/to/selection-analysis
```

This offline analysis uses the shared strategies in `dspr_dataset/data_filter.py`:
the saved score field, minimum correct count and GED range threshold from
`--eligibility-config`, with `top_k=None`. All qualified candidates in eligible
variants are retained. It compares eligible problem sets and retained path
sets on common and actual candidate pools, including coverage losses;
rank correlations are auxiliary. Outputs are written under `--output-dir`; its per-run report is
`report.md` in that output directory (override with `--report`). The consolidated
interpretation is maintained in
[the unified Section 4.1 report](../../docs/SECTION_4_1_RESULTS_QWEN25.md#selection-stability).
Existing training manifests,
annotation data and GED results are not rewritten.

### Bailian batch workflow

The Qwen command above writes `<output-dir>/batch/batch_requests.jsonl` plus a manifest.
Each row contains `custom_id`, `method: POST`, `url: /v1/chat/completions`, and
`body` with `model: qwen3.8-flash`. Upload this file to the **Beijing** Bailian batch
service and download the result and, if present, error JSONL files. Preparation
and merging are local operations; these commands do not upload or submit jobs.

```bash
# Keep the same SEGMENTED_INPUT and output directory used for preparation.
python scripts/run_dag_analysis.py --annotator qwen --mode merge-batch \
  --input "$SEGMENTED_INPUT" \
  --batch-results-file /path/to/results.jsonl \
  --batch-errors-file /path/to/errors.jsonl
```

Omit `--batch-errors-file` when no error file was produced. If preparation used
custom generation settings (for example `--thinking on --max-tokens 8192`), pass
the same settings when merging. The manifest checks request-file content,
provider/model/settings and the original input identities. Results are joined
by `custom_id`, independent of provider output order. Missing results and
provider errors remain explicit per-sample failures. Final output is
`<output-dir>/analyzed_records.jsonl`, usable by the existing GED pipeline.

OpenAI batch export uses the same standard envelope. Claude `--mode batch` exports
`custom_id` / `params` JSONL for Message Batches; `--mode merge-batch` also accepts
its `result.type` / `result.message` result format. Submission is external.

Provider references (checked 2026-10-21):
[OpenAI GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Claude Haiku 5.5 parameters](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide),
[Bailian batch inference](https://help.aliyun.com/zh/model-studio/batch-inference).

GED also writes `<results-stem>.failures.jsonl` beside the result JSONL. Each
entry identifies the problem, variant, sample and reference, failure reason,
graph sizes, limits and elapsed time when available. Missing DAGs are included;
`kept=true` identifies approximate timeout values retained with `--keep-timeouts`.
Failure details are checkpointed and restored on resume, with counts in the run config.

Known provider expansions of the `Conclude` tag, such as `Draw` and `Draw
intermediate and final conclusions`, are normalized to `Conclude` by the shared
parser. Use normal-mode `--resume` to retry failed samples while reusing successful
checkpoints.

Both annotation modes write `<output-dir>/analyzed_records.jsonl`. Normal mode
also retains its historical `<output-dir>/normal/analyzed_records.jsonl` path.
Normal checkpoints are flushed as samples finish; the final records retain input
order. Each normal-mode provider attempt is recorded separately in
`<output-dir>/normal/raw_responses.jsonl`, including the sample ID, retry
indices, response content, and parse/provider error fields. API credentials are
never written to this file. Records keep stable sample IDs; GED joins verify
response content when it is available in the analyzed source.

## Cache And Recovery

Segmentation caches match input content and segmentation parameters. Graph caches
match DAG sources, graph preprocessing, and encoder settings; changing GED weights
does not invalidate embeddings. Use `--rebuild-graph-cache` to replace an old or
mismatched graph cache. Caches without the new identity metadata need one rebuild.

GED `--resume` checks graph/correctness inputs and measurement parameters. It
restores successful or intentionally empty problems and retries failed problems.
Changed parameters require a separate result/checkpoint path or a run without
`--resume`. Older checkpoints without identity metadata are recomputed.

The prompt, segmentation rules, compression order, encoder behavior, edit costs,
normalization, default `original_0` reference, and downstream filtering defaults
remain unchanged. Unknown invalid annotations are still rejected; only the
explicitly listed provider aliases are normalized.
