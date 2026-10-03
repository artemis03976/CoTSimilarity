#!/usr/bin/env bash
# Normal-mode DAG annotation
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
PROVIDER="${PROVIDER:-volcengine}"
MODEL="${MODEL:-${LLM_MODEL:-ep-20261001103240-d9m5m}}"
RAW_INPUT="${RAW_INPUT:-output/qwen-2.5/multiple_n16/pre_eligible/pre_eligible_set.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen-2.5/multiple_n16/dag_analysis}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"

if [[ ! -f "${RAW_INPUT}" ]]; then
  echo "Input JSONL does not exist: ${RAW_INPUT}" >&2
  exit 1
fi

echo "Provider: ${PROVIDER}"
echo "Model: ${MODEL}"
echo "Input: ${RAW_INPUT}"
echo "Output: ${OUTPUT_DIR}"
echo "Concurrency: ${CONCURRENCY}; retries per chain: ${MAX_RETRIES}"

# Credentials and base URL are loaded from .env by the Python configuration.
exec "${PYTHON}" "${SCRIPT_DIR}/run_dag_analysis.py" \
  --mode normal \
  --raw-input "${RAW_INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --max-retries "${MAX_RETRIES}" \
  --resume \
  "$@"
