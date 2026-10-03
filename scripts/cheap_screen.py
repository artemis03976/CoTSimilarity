#!/usr/bin/env python3
"""Cheaply screen multi-path records before DAG/GED annotation.

The screen only uses information already available in ``all_records.jsonl``:
quality validity, answer correctness, and the validity of the greedy original
anchor.  It deliberately does not inspect DAGs or compute GED.  A problem is
pre-eligible when its original anchor is valid and both perturbation variants
have at least ``--min-correct`` valid correct trajectories (default: three).
Thresholds below three are not supported.

The only screened dataset is ``pre_eligible_set.jsonl``. Diagnostic rows and
counts are saved in ``screening_summary.jsonl`` and ``screening_manifest.json``.

The input records are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


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
        default="output/qwen-2.5/multiple_n16/all_records.jsonl",
        help="Completed multi-path all_records JSONL",
    )
    parser.add_argument(
        "--canonical",
        default="data/canonical_math_paired.jsonl",
        help="Canonical data used only for source metadata",
    )
    parser.add_argument(
        "--output-dir",
        default="output/qwen-2.5/multiple_n16/pre_eligible",
    )
    parser.add_argument(
        "--min-correct",
        type=int,
        default=3,
        help="Correct valid trajectories required on each variant (minimum/default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    canonical_path = Path(args.canonical)
    output_dir = Path(args.output_dir)
    if args.min_correct < 3:
        raise ValueError("--min-correct must be at least 3")

    records = list(read_jsonl(input_path))
    if not records:
        raise ValueError(f"No records found in {input_path}")

    source_by_id: dict[int, str] = {}
    if canonical_path.is_file():
        for row in read_jsonl(canonical_path):
            source_by_id[problem_id(row)] = str(row.get("source") or "unknown")

    seen: set[int] = set()
    summaries: list[dict[str, Any]] = []
    pre_eligible: list[dict[str, Any]] = []
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
            "pre_eligible": category == "pair_candidate",
        }
        summaries.append(summary)

        if category == "pair_candidate":
            pre_eligible.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pre_eligible_set.jsonl", pre_eligible)
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
        "counts": {
            "input_groups": len(records),
            "original_valid": aggregate["original_valid"],
            "original_invalid": len(records) - aggregate["original_valid"],
            "pre_eligible_groups": len(pre_eligible),
            "pre_eligible_records_written": len(pre_eligible),
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
            "pre_eligible_set": str(output_dir / "pre_eligible_set.jsonl"),
            "screening_summary": str(output_dir / "screening_summary.jsonl"),
        },
        "notes": [
            "This is a cheap pre-DAG screen; it is not formal GED eligibility.",
            "GED range, timeout, and normalized-score conditions are evaluated later.",
            "Input all_records.jsonl was not modified.",
        ],
    }
    write_json(output_dir / "screening_manifest.json", manifest)

    print(json.dumps({"counts": manifest["counts"], "categories": manifest["categories"]}, ensure_ascii=False))
    print(f"Pre-eligible set: {output_dir / 'pre_eligible_set.jsonl'}")
    print(f"Manifest:          {output_dir / 'screening_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
