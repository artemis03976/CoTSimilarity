"""Data and prompting helpers shared by generation-only inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.io import read_records as read_jsonl, write_json, write_jsonl
from utils.prompts import MATH_SYSTEM_PROMPT


VARIANTS = ("original", "simple", "hard")
DEFAULT_SYSTEM_PROMPT = MATH_SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationResult:
    """One generated continuation and the metadata needed for auditing it."""

    text: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_math_records(
    path: Path,
    problem_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and validate MATH-Perturb triplets in stable file order."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    records = read_jsonl(path)
    if problem_id is not None:
        records = [
            record
            for record in records
            if normalise_problem_id(record.get("problem_id")) == problem_id
        ]
    elif limit is not None:
        records = records[:limit]

    seen: set[int] = set()
    for record in records:
        pid = normalise_problem_id(record.get("problem_id"))
        if pid in seen:
            raise ValueError(f"Duplicate problem_id {pid} in {path}")
        seen.add(pid)
        for variant in VARIANTS:
            variant_data = record.get(variant)
            if not isinstance(variant_data, dict):
                raise ValueError(f"Problem {pid} is missing the {variant!r} object")
            if not isinstance(variant_data.get("problem"), str) or not variant_data["problem"].strip():
                raise ValueError(f"Problem {pid} {variant} has no problem text")
            if ground_truth_for(variant_data) is None:
                raise ValueError(f"Problem {pid} {variant} has no solution, answer, or ground_truth")
    return records


def normalise_problem_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"problem_id must be an integer, got {value!r}") from exc


def ground_truth_for(variant_data: dict[str, Any]) -> Any:
    """Read references from canonical inputs or saved inference records."""
    for field in ("solution", "answer", "ground_truth"):
        value = variant_data.get(field)
        if value is not None:
            return value
    return None


def ground_truth_type_for(variant_data: dict[str, Any], variant: str) -> str:
    """Preserve the distinction between a final answer and a worked solution."""
    for field in ("solution", "answer", "ground_truth"):
        if variant_data.get(field) is not None:
            if field == "ground_truth":
                return variant_data.get("ground_truth_type") or (
                    "solution" if variant == "original" else "answer"
                )
            return "solution" if field == "solution" else "answer"
    raise ValueError(f"No reference found for variant={variant}")
