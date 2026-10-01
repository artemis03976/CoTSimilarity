#!/usr/bin/env python3
"""Cheaply screen multi-path records before DAG/GED annotation.

The screen only uses information already available in ``all_records.jsonl``:
quality validity, answer correctness, and the validity of the greedy original
anchor.  It deliberately does not inspect DAGs or compute GED.  A problem is
eligible for the strict candidate output when its original anchor is valid and
both perturbation variants have at least ``--min-correct`` valid correct
trajectories.  The default of five matches the current downstream GED
eligibility contract.

The script writes three files:

``candidate_records.jsonl``
    Strict pair candidates to pass to DAG analysis.
``pair_candidates.jsonl``
    A wider audit pool requiring only one correct trajectory per variant.
``screening_summary.jsonl``
    One compact diagnostic row per input problem group.

The input records are never modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = ("simple", "hard")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def problem_id(row: dict[str, Any]) -> int:
    try:
        return int(row["problem_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid problem_id in input row: {row.get('problem_id')!r}") from exc


def sample_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize quality/correctness without re-running answer evaluation."""
    valid = [sample for sample in samples if bool(sample.get("valid"))]
    correct = [sample for sample in valid if bool(sample.get("correct"))]
    unknown = [sample for sample in valid if sample.get("answer_correct") is None]
    responses = {
        " ".join(str(sample.get("response") or "").split())
        for sample in valid
    }
    return {
        "sample_count": len(samples),
        "valid_count": len(valid),
        "correct_count": len(correct),
        "incorrect_count": len(valid) - len(correct) - len(unknown),
        "answer_unknown_count": len(unknown),
        "unique_valid_response_count": len(responses),
    }


def classify(original_valid: bool, simple_count: int, hard_count: int, minimum: int) -> str:
    if not original_valid:
        return "invalid_original"
    simple_ok = simple_count >= minimum
    hard_ok = hard_count >= minimum
    if simple_ok and hard_ok:
        return "pair_candidate"
    if simple_ok:
        return "simple_only"
    if hard_ok:
        return "hard_only"
    if simple_count or hard_count:
        return "insufficient_correct"
    return "no_correct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="output/qwen-2.5/multiple_n16/development/all_records.jsonl",
        help="Completed multi-path all_records JSONL",
    )
    parser.add_argument(
        "--canonical",
        default="data/canonical_splits_seed42/development.jsonl",
        help="Canonical development file used only for source metadata",
    )
    parser.add_argument(
        "--output-dir",
        default="output/qwen-2.5/cheap_screen_n16/development",
    )
    parser.add_argument(
        "--min-correct",
        type=int,
        default=5,
        help="Correct valid trajectories required on each variant (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    canonical_path = Path(args.canonical)
    output_dir = Path(args.output_dir)
    if args.min_correct < 1:
        raise ValueError("--min-correct must be positive")

    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    source_by_id: dict[int, str] = {}
    if canonical_path.is_file():
        for row in read_jsonl(canonical_path):
            source_by_id[problem_id(row)] = str(row.get("source") or "unknown")

    seen: set[int] = set()
    summaries: list[dict[str, Any]] = []
    strict_candidates: list[dict[str, Any]] = []
    wide_candidates: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    source_categories: dict[str, Counter[str]] = defaultdict(Counter)
    correct_histogram: Counter[str] = Counter()
    aggregate = Counter()

    for row in records:
        pid = problem_id(row)
        if pid in seen:
            raise ValueError(f"Duplicate problem_id {pid} in {input_path}")
        seen.add(pid)

        original_stats = sample_stats(row.get("original", {}).get("samples", []))
        simple_stats = sample_stats(row.get("simple", {}).get("samples", []))
        hard_stats = sample_stats(row.get("hard", {}).get("samples", []))
        original_valid = original_stats["valid_count"] > 0
        simple_correct = simple_stats["correct_count"]
        hard_correct = hard_stats["correct_count"]
        source = source_by_id.get(pid, "unknown")

        category = classify(original_valid, simple_correct, hard_correct, args.min_correct)
        categories[category] += 1
        source_categories[source][category] += 1
        correct_histogram[f"simple={simple_correct},hard={hard_correct}"] += 1
        aggregate["original_valid"] += int(original_valid)
        original_samples = row.get("original", {}).get("samples", [])
        original_correct = bool(original_samples and original_samples[0].get("correct"))
        aggregate["original_correct"] += int(original_valid and original_correct)
        aggregate["simple_valid_samples"] += simple_stats["valid_count"]
        aggregate["hard_valid_samples"] += hard_stats["valid_count"]
        aggregate["simple_correct_samples"] += simple_correct
        aggregate["hard_correct_samples"] += hard_correct
        aggregate["answer_unknown_samples"] += (
            original_stats["answer_unknown_count"]
            + simple_stats["answer_unknown_count"]
            + hard_stats["answer_unknown_count"]
        )

        summary = {
            "problem_id": pid,
            "source": source,
            "type": row.get("type"),
            "level": row.get("level"),
            "category": category,
            "original_valid": original_valid,
            "simple": simple_stats,
            "hard": hard_stats,
            "pair_candidate_min1": bool(
                original_valid and simple_correct >= 1 and hard_correct >= 1
            ),
            "pair_candidate_min_correct": category == "pair_candidate",
        }
        summaries.append(summary)

        if original_valid and simple_correct >= 1 and hard_correct >= 1:
            wide_candidates.append(row)
        if category == "pair_candidate":
            strict_candidates.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidate_records.jsonl", strict_candidates)
    write_jsonl(output_dir / "pair_candidates.jsonl", wide_candidates)
    write_jsonl(output_dir / "screening_summary.jsonl", summaries)

    manifest = {
        "status": "completed",
        "protocol": "cheap_quality_correctness_screen",
        "input": str(input_path),
        "canonical_metadata": str(canonical_path),
        "min_correct_per_variant": args.min_correct,
        "candidate_definition": (
            "original has at least one valid anchor and both simple and hard "
            f"have at least {args.min_correct} valid correct trajectories"
        ),
        "wide_pair_definition": (
            "original has at least one valid anchor and both simple and hard "
            "have at least one valid correct trajectory"
        ),
        "counts": {
            "input_groups": len(records),
            "original_valid": aggregate["original_valid"],
            "original_invalid": len(records) - aggregate["original_valid"],
            "wide_pair_candidates_min1": len(wide_candidates),
            "strict_pair_candidates": len(strict_candidates),
            "strict_candidate_records_written": len(strict_candidates),
            "answer_unknown_samples": aggregate["answer_unknown_samples"],
        },
        "categories": dict(sorted(categories.items())),
        "source_categories": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(source_categories.items())
        },
        "aggregate_samples": dict(aggregate),
        "correct_count_histogram": dict(
            sorted(correct_histogram.items(), key=lambda item: item[0])
        ),
        "outputs": {
            "candidate_records": str(output_dir / "candidate_records.jsonl"),
            "pair_candidates": str(output_dir / "pair_candidates.jsonl"),
            "screening_summary": str(output_dir / "screening_summary.jsonl"),
        },
        "notes": [
            "This is a cheap pre-DAG screen; it is not formal GED eligibility.",
            "GED range, timeout, and normalized-score conditions are evaluated later.",
            "Input all_records.jsonl was not modified.",
        ],
    }
    (output_dir / "screening_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"counts": manifest["counts"], "categories": manifest["categories"]}, ensure_ascii=False))
    print(f"Strict candidates: {output_dir / 'candidate_records.jsonl'}")
    print(f"Wide candidates:   {output_dir / 'pair_candidates.jsonl'}")
    print(f"Manifest:          {output_dir / 'screening_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
