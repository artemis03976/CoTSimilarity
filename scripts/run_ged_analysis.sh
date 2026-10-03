#!/usr/bin/env bash
# Run GED analysis for the merged multi-path DAG annotations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/qwen-2.5/multiple_n16/ged}"
CORRECTNESS_FILE="${CORRECTNESS_FILE:-output/qwen-2.5/multiple_n16/pre_eligible/pre_eligible_set.jsonl}"
VARIANT_RECORDS="${VARIANT_RECORDS:-output/qwen-2.5/multiple_n16/dag_analysis/analyzed_records.jsonl}"
GRAPH_CACHE="${GRAPH_CACHE:-${OUTPUT_ROOT}/cache/ged_graph_cache.pt}"
ALL_RESULTS_OUTPUT="${ALL_RESULTS_OUTPUT:-${OUTPUT_ROOT}/all_ged_results.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
WORKERS="${WORKERS:-4}"
GED_TIMEOUT="${GED_TIMEOUT:-30}"
MAX_GED_NODES="${MAX_GED_NODES:-32}"
MAX_GED_EDGES="${MAX_GED_EDGES:-64}"

if [[ ! -f "${CORRECTNESS_FILE}" ]]; then
  echo "Correctness input does not exist: ${CORRECTNESS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VARIANT_RECORDS}" ]]; then
  echo "DAG analysis input does not exist: ${VARIANT_RECORDS}" >&2
  exit 1
fi

echo "Output root: ${OUTPUT_ROOT}"
echo "Correctness: ${CORRECTNESS_FILE}"
echo "DAG records: ${VARIANT_RECORDS}"
echo "Results: ${ALL_RESULTS_OUTPUT}"
echo "Workers: ${WORKERS}; samples per variant: ${NUM_SAMPLES}"

ARGS=(
  --output-root "${OUTPUT_ROOT}"
  --variant-records "${VARIANT_RECORDS}"
  --correctness-file "${CORRECTNESS_FILE}"
  --graph-cache "${GRAPH_CACHE}"
  --all-results-output "${ALL_RESULTS_OUTPUT}"
  --num-samples "${NUM_SAMPLES}"
  --workers "${WORKERS}"
  --ged-timeout "${GED_TIMEOUT}"
  --max-ged-nodes "${MAX_GED_NODES}"
  --max-ged-edges "${MAX_GED_EDGES}"
)

# Checkpoints make interrupted GED runs restartable. Set RESUME=0 for a fresh run.
if [[ "${RESUME:-1}" == "1" ]]; then
  ARGS+=(--resume)
fi

exec "${PYTHON}" "${SCRIPT_DIR}/run_ged_analysis.py" "${ARGS[@]}" "$@"
