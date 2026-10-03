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

- `MODEL=ep-xxxxxxxx bash scripts/run_dag_analysis.sh`: run normal annotation
  with Volcengine, the 51-chain subset, eight workers, and one retry per chain.
  Override `PYTHON`, `PROVIDER`, `MODEL`, `RAW_INPUT`, `OUTPUT_DIR`, `CONCURRENCY`,
  or `MAX_RETRIES` as environment variables; additional CLI arguments such as
  `--resume` are forwarded to the Python entrypoint. Exported `LLM_MODEL` is also
  accepted when `MODEL` is unset. Credentials and base URL use the existing
  `.env` configuration (`VOLCENGINE_API_KEY` / `VOLCENGINE_BASE_URL`, or the
  generic `LLM_API_KEY` / `LLM_BASE_URL`).
- `python scripts/run_dag_analysis.py --mode normal ...`: annotate normally;
  add `--resume` to reuse successful sample checkpoints and retry failed samples.
- `python scripts/run_dag_analysis.py --mode batch ...`: prepare requests.
- `python scripts/run_dag_analysis.py --mode merge-batch ...`: merge downloaded
  responses; provider errors, invalid DAGs, and missing results are recorded per
  sample. Preparing and merging batch files requires no API credentials.
- `python scripts/run_ged_analysis.py ...`: compute GED with the existing flags.

Known provider expansions of the `Conclude` tag, such as `Draw` and `Draw
intermediate and final conclusions`, are normalized to `Conclude` by the shared
parser. Existing normal-mode failures can be recovered without new API calls
with `scripts/recover_dag_raw_responses.py`; the recovery script keeps the
original analyzed file unchanged and validates recovered DAGs against their
segmented steps.

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
