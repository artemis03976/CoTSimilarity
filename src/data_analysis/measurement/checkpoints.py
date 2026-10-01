"""Per-problem GED checkpoints, tied to inputs and measurement settings."""

from pathlib import Path

from utils.artifacts import file_identity
from utils.io import read_jsonl, read_manifest, write_json, write_jsonl


RETRYABLE_ERRORS = {"problem_error", "missing_original", "original_compression_error", "graph_error", "error", "invalid", "sample_mismatch"}


def problem_status(results, stats):
    if any(stats.get(name, 0) for name in RETRYABLE_ERRORS):
        return "failed"
    return "completed" if results else "empty"


def write_problem_checkpoint(directory, problem_id, results, stats, signature):
    target = Path(directory) / f"problem_{problem_id}.jsonl"
    write_jsonl(target, results)
    write_json(target.with_suffix(".meta.json"), {
        "signature": signature,
        "status": problem_status(results, stats),
        "stats": stats,
        "output": file_identity(target),
    })


def load_problem_checkpoint(directory, problem_id, signature):
    target = Path(directory) / f"problem_{problem_id}.jsonl"
    metadata = read_manifest(target.with_suffix(".meta.json"))
    if not target.is_file() or not metadata:
        return None
    if metadata.get("signature") != signature:
        raise ValueError("Checkpoint inputs or GED settings changed; use a new output path or run without --resume")
    if metadata.get("status") not in {"completed", "empty"} or metadata.get("output") != file_identity(target):
        return None
    try:
        return list(read_jsonl(target)), metadata.get("stats", {})
    except ValueError:
        return None
