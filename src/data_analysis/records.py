"""Read current and historical records at the pipeline boundary."""

import logging

from utils.artifacts import response_hash
from utils.io import read_jsonl
from .annotation.parsing import extract_dag_from_batch_response, validate_dag
from .schemas import SampleKey, VARIANTS


logger = logging.getLogger(__name__)


def iter_samples(records, variants=VARIANTS):
    seen = set()
    for record in records:
        problem_id = int(record.get("problem_id", record.get("id")))
        for variant in variants:
            entry = record.get(variant) or {}
            for index, sample in enumerate(entry.get("samples", [])):
                key = sample_key(problem_id, variant, index, sample)
                if key in seen:
                    raise ValueError(f"Duplicate sample ID: {key}")
                seen.add(key)
                yield key, entry, sample


def sample_key(problem_id, variant, index, sample):
    key = SampleKey.parse(sample["sample_id"]) if sample.get("sample_id") else SampleKey(int(problem_id), variant, index)
    if key.problem_id != int(problem_id) or key.variant != variant:
        raise ValueError(f"Mismatched sample ID: {key}")
    return key


def attach_step_text(dag, steps):
    texts = {str(step["index"]): step.get("text", "") for step in steps}
    return [
        dict(node, text=texts.get(str(node["step_id"])) or node.get("text") or node.get("analysis", ""))
        for node in dag
    ]


def load_dag_records(path: str) -> dict:
    dags = {}
    for record in read_jsonl(path):
        if record.get("custom_id") is not None:
            key = str(record["custom_id"])
            SampleKey.parse(key)
            dag = extract_dag_from_batch_response(record)
            entries = [(key, dag)] if dag else []
        else:
            entries = []
            for key, _, sample in iter_samples([record]):
                dag = sample.get("dag_analysis")
                if dag is None:
                    continue
                try:
                    validate_dag(dag, sample.get("steps"))
                except ValueError as exc:
                    logger.warning("Skip invalid DAG %s: %s", key, exc)
                    continue
                dag = attach_step_text(dag, sample.get("steps", []))
                if "response" in sample:
                    dag[0]["response_hash"] = response_hash(sample["response"])
                entries.append((str(key), dag))
        for key, dag in entries:
            if key in dags:
                raise ValueError(f"Duplicate DAG sample ID: {key}")
            dags[key] = dag
    return dags


def load_correctness_data(path: str) -> dict:
    correctness = {}
    for record in read_jsonl(path):
        problem_id = int(record["problem_id"])
        if problem_id in correctness:
            raise ValueError(f"Duplicate problem_id: {problem_id}")
        # Keep sample IDs from an earlier stage when records have been reordered.
        for key, _, sample in iter_samples([record]):
            sample["sample_id"] = str(key)
        correctness[problem_id] = {variant: record.get(variant, {}) for variant in VARIANTS}
    return correctness
