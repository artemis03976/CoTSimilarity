"""Data, prompting, and evaluation helpers shared by inference pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dspr.config import MATH_SYSTEM_PROMPT
from utils.evaluate import answer_check, test_parse_latex


VARIANTS = ("original", "simple", "hard")
DEFAULT_SYSTEM_PROMPT = MATH_SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationResult:
    """One generated continuation and the metadata needed for auditing it."""

    text: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and fail with an actionable line-level error."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            records.append(record)
    return records


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
            if variant_data.get("solution") is None and variant_data.get("answer") is None:
                raise ValueError(f"Problem {pid} {variant} has no solution or answer")
    return records


def normalise_problem_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"problem_id must be an integer, got {value!r}") from exc


def build_messages(problem: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def build_prompt(tokenizer: Any, problem: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return tokenizer.apply_chat_template(
        build_messages(problem, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def dataset_type_for_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return "original" if variant == "original" else "perturb"


def ground_truth_for(variant_data: dict[str, Any]) -> Any:
    solution = variant_data.get("solution")
    return solution if solution is not None else variant_data.get("answer")


def check_answer_strict(
    problem: str,
    response: str,
    ground_truth: Any,
    variant: str,
) -> bool:
    """Evaluate one answer without silently converting evaluator failures to False."""

    try:
        return bool(
            answer_check(
                problem,
                response,
                str(ground_truth),
                dataset_type_for_variant(variant),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Answer evaluation failed for variant={variant}, problem={problem[:80]!r}"
        ) from exc


def run_evaluator_self_test() -> None:
    """Fail before model loading when symbolic answer validation is degraded."""

    try:
        test_parse_latex()
        integer_ok = answer_check(
            "A synthetic arithmetic problem.",
            "The answer is \\boxed{26}.",
            "26",
            "perturb",
        )
    except Exception as exc:
        raise RuntimeError("Answer evaluator self-test raised an exception") from exc
    if not integer_ok:
        raise RuntimeError("Answer evaluator self-test rejected an exact boxed integer")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
