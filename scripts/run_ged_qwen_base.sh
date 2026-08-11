#!/usr/bin/env bash
# Recompute GED for the archived deterministic Qwen base-model CoTs.
#
# This script does not generate responses or call the DAG analyzer.  It reads
# the legacy one-sample-per-variant DAG output and writes a separate GED result
# file containing both raw and normalized GED values.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Existing deterministic greedy DAG records (279 original + 279 simple + 279 hard).
DAG_RECORDS="${DAG_RECORDS:-${ROOT_DIR}/output/qwen_legacy/dag_analysis/analyzed_records.jsonl}"
# Correctness labels for the same deterministic greedy responses.
CORRECTNESS_FILE="${CORRECTNESS_FILE:-${ROOT_DIR}/output/qwen_legacy/all_records.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/qwen_legacy/ged_base_greedy_normalized}"
GED_OUTPUT="${GED_OUTPUT:-${OUTPUT_DIR}/all_ged_results.jsonl}"
GRAPH_CACHE="${GRAPH_CACHE:-${OUTPUT_DIR}/ged_graph_cache.pt}"

for required_file in "${DAG_RECORDS}" "${CORRECTNESS_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Error: required input does not exist: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo "========== Qwen base greedy GED =========="
echo "DAG records: ${DAG_RECORDS}"
echo "Correctness: ${CORRECTNESS_FILE}"
echo "GED output:  ${GED_OUTPUT}"
echo "Graph cache: ${GRAPH_CACHE}"
echo "Samples per variant: 1"
echo "==========================================="

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/src/data_analysis/ged_analysis.py" \
    --variant-records "${DAG_RECORDS}" \
    --correctness-file "${CORRECTNESS_FILE}" \
    --graph-cache "${GRAPH_CACHE}" \
    --all-results-output "${GED_OUTPUT}" \
    --num-samples 1

echo ""
echo "GED computation complete."
echo "Raw and normalized GED fields are available in: ${GED_OUTPUT}"
