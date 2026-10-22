#!/usr/bin/env bash
# ANNOTATOR=deepseek|gpt|haiku|qwen|volcengine; qwen prepares a Bailian batch file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
ANNOTATOR="${ANNOTATOR:-deepseek}"
RAW_INPUT="${RAW_INPUT:-output/qwen-2.5/base/all_records.jsonl}"
args=(--annotator "${ANNOTATOR}"
      --concurrency "${CONCURRENCY:-8}" --max-retries "${MAX_RETRIES:-3}")
if [[ -n "${SEGMENTED_INPUT:-}" ]]; then
  args+=(--input "${SEGMENTED_INPUT}")
else
  args+=(--raw-input "${RAW_INPUT}")
fi
[[ -n "${MODE:-}" ]] && args+=(--mode "${MODE}")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${MODEL:-}" ]] && args+=(--model "${MODEL}")
[[ -n "${PROVIDER:-}" ]] && args+=(--provider "${PROVIDER}")

# Python loads provider-specific credentials from .env and isolates output by annotator.
exec "${PYTHON}" "${SCRIPT_DIR}/run_dag_analysis.py" "${args[@]}" "$@"
