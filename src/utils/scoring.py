"""Offline answer scoring; lightweight score-state helpers also serve inference.

Symbolic evaluation is imported only by the offline scoring entrypoint. Generation,
aggregation, and screening can use the state helpers with any ML environment.
"""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json, write_jsonl

VARIANTS = ("original", "simple", "hard")
SYMPY_VERSION = "1.14.0"


def pending_score() -> dict[str, Any]:
    return {"correct": None, "answer_correct": None, "answer_evaluation_status": "pending"}


def require_scored(sample: dict[str, Any], context: str = "Prediction") -> bool:
    """Read a saved label without treating a pending answer as an incorrect one."""
    if sample.get("answer_evaluation_status") == "pending" or not isinstance(sample.get("correct"), bool):
        raise ValueError(
            f"{context}: answer scoring is incomplete. Run scripts/score_answers.py "
            "on the prediction file with SymPy 1.14.0 before using correctness labels."
        )
    return sample["correct"]


def evaluator_metadata() -> dict[str, Any]:
    """Require the agreed scoring version, without importing the model stack."""
    try:
        installed = version("sympy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install requirements-scoring.txt in the offline scoring environment") from exc
    if installed != SYMPY_VERSION:
        raise RuntimeError(
            f"Scoring requires SymPy {SYMPY_VERSION}; found {installed}. "
            "Use a separate environment with requirements-scoring.txt; "
            "the inference environment does not need to change."
        )
    from . import evaluate  # noqa: F401 — load parser dependencies before touching any file

    root = Path(__file__).parent
    code_paths = [root / "evaluate.py", *sorted((root / "evaluation").glob("*.py")), Path(__file__)]
    return {
        "sympy": installed,
        "parser_backend": "lark",
        "versions": {name: version(name) for name in ("sympy", "lark", "numpy", "regex", "mpmath")},
        "code_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths
        },
        "policy": "Regrade valid saved responses; preserve generation validity, references and membership.",
    }


def prediction_paths(inputs: list[Path], folds: list[int] | None = None) -> list[Path]:
    """Expand only a run's immediate prediction files; never recursively rescore output."""
    paths = []
    for source in inputs:
        source = source.resolve()
        if source.is_file():
            if folds is not None:
                raise ValueError("--folds applies to K-fold run directories, not individual files")
            candidates = [source]
        elif source.is_dir():
            if folds is not None:
                candidates = [source / f"fold_{fold}" / "all_records.jsonl" for fold in folds]
            else:
                fold_dirs = sorted(path for path in source.glob("fold_*") if path.is_dir())
                candidates = [path / "all_records.jsonl" for path in fold_dirs]
                if not candidates:
                    candidates = [source / "all_records.jsonl"]
        else:
            raise FileNotFoundError(source)
        for path in candidates:
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Prediction file not found: {path}")
            if path.suffix != ".jsonl":
                raise ValueError(f"Expected a prediction JSONL file: {path}")
            if "pre_eligible" in path.parts or path.name in {
                "pre_eligible_set.jsonl", "screening_summary.jsonl", "train.jsonl", "val.jsonl", "test.jsonl",
            }:
                raise ValueError(f"Frozen screening/training data must not be rescored here: {path}")
            if path not in paths:
                paths.append(path)
    return paths


def _score_one(task: tuple) -> dict[str, Any]:
    context, variant, problem, reference, reference_type, response, valid = task
    if not valid:
        return {"correct": False, "answer_correct": None, "answer_evaluation_status": "invalid_generation"}
    from .evaluate import UnsupportedReferenceError, answer_check

    try:
        # The legacy extractor prints multi-valued answers. Keep the CLI's progress
        # readable; exceptions still propagate and prevent replacing this file.
        with redirect_stdout(io.StringIO()):
            correct = bool(answer_check(
                problem, response, reference,
                "original" if variant == "original" else "perturb",
                reference_type=reference_type,
            ))
    except UnsupportedReferenceError:
        return {"correct": False, "answer_correct": None, "answer_evaluation_status": "unsupported_reference"}
    except Exception as exc:
        raise RuntimeError(f"{context}: answer evaluation failed: {exc}") from exc
    return {
        "correct": correct, "answer_correct": correct,
        "answer_evaluation_status": "correct" if correct else "incorrect",
    }


def score_file(path: Path, evaluator: dict[str, Any], workers: int) -> dict[str, Any]:
    """Score one complete saved file and atomically replace it after success."""
    from tqdm import tqdm

    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    records = list(read_jsonl(path))
    # Only an all_records file owns its neighbouring generation QC. Explicit
    # OOF/custom JSONL inputs receive their own scoring metadata sidecar.
    qc_path = path.parent / "generation_qc.json" if path.name == "all_records.jsonl" else path.with_suffix(".scoring.json")
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    if qc.get("status") == "running":
        raise ValueError(f"Generation is still running: {qc_path}")
    tasks, samples = [], []
    seen = set()
    for record in records:
        pid = int(record["problem_id"])
        if pid in seen:
            raise ValueError(f"{path}: duplicate problem_id={pid}")
        seen.add(pid)
        for variant in VARIANTS:
            data = record.get(variant)
            context = f"{path}: problem {pid}/{variant}"
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                raise ValueError(f"{context}: missing saved samples")
            if not isinstance(data.get("problem"), str) or data.get("ground_truth") is None:
                raise ValueError(f"{context}: missing saved problem/reference")
            reference_type = data.get("ground_truth_type") or ("solution" if variant == "original" else "answer")
            for index, sample in enumerate(data["samples"]):
                if (not isinstance(sample, dict) or not isinstance(sample.get("response"), str)
                        or not isinstance(sample.get("valid"), bool)):
                    raise ValueError(f"{context}/{index}: missing response or generation validity")
                samples.append(sample)
                tasks.append((f"{context}/{index}", variant, data["problem"], data["ground_truth"],
                              reference_type, sample["response"], sample["valid"]))
    if not tasks:
        raise ValueError(f"No saved responses in {path}")

    print(f"Scoring {path}: {len(records)} groups, {len(tasks)} responses, {workers} workers", flush=True)
    if workers == 1:
        scores = list(tqdm(map(_score_one, tasks), total=len(tasks), desc=path.parent.name, unit="answer"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(tqdm(pool.map(_score_one, tasks, chunksize=8), total=len(tasks),
                               desc=path.parent.name, unit="answer"))
    changed = sum(isinstance(sample.get("correct"), bool) and sample["correct"] != score["correct"]
                  for sample, score in zip(samples, scores))
    for sample, score in zip(samples, scores):
        sample.update(score)
    summary = {
        "records": len(records), "scored_generations": len(samples), "pending_generations": 0,
        "correct_generations": sum(sample["correct"] for sample in samples),
        "ungraded_generations": sum(sample["answer_evaluation_status"] == "unsupported_reference" for sample in samples),
        "fully_correct_triplets": sum(
            all(any(s["correct"] for s in record[variant]["samples"]) for variant in VARIANTS)
            for record in records
        ),
    }
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError(f"Input changed during scoring; refusing to replace {path}")
    write_jsonl(path, records)
    qc.update(summary)
    qc.update(scoring_status="completed", answer_evaluator={
        **evaluator, "updated_at": datetime.now(timezone.utc).isoformat(),
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    write_json(qc_path, qc)
    print(f"Saved in place: {path}; correct={summary['correct_generations']}/{len(samples)}, "
          f"unsupported={summary['ungraded_generations']}, changed existing labels={changed}", flush=True)
    return summary
