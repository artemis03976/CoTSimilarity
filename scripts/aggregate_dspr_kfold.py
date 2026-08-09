#!/usr/bin/env python3
"""Validate, merge, and report DSPR K-fold out-of-fold predictions.

The five test folds are pooled into one 279-problem OOF file.  Metrics are
reported for the full test universe and for variant-specific eligible and
non-eligible slices.  An optional baseline enables paired bootstrap confidence
intervals and exact McNemar tests on matching problem IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


FOLDS = tuple(range(5))
VARIANTS = ("original", "simple", "hard")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected an object in {path} line {line_number}")
            records.append(record)
    return records


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


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def load_id_map(records: list[dict[str, Any]], source: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        pid = normalise_id(record.get("problem_id"))
        if pid in result:
            raise ValueError(f"Duplicate problem_id {pid} in {source}")
        result[pid] = record
    return result


def extract_flags(record: dict[str, Any], variant: str, source: str) -> list[bool]:
    data = record.get(variant)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: problem {record.get('problem_id')} lacks {variant}")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{source}: problem {record.get('problem_id')} {variant} has no samples")
    flags = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("correct"), bool):
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} has invalid correctness"
            )
        flags.append(sample["correct"])
    return flags


def extract_alpha_values(record: dict[str, Any], variant: str, source: str) -> list[float]:
    data = record.get(variant)
    if not isinstance(data, dict) or "alpha" not in data:
        return []
    raw = data["alpha"]
    values = raw if isinstance(raw, list) else [raw]
    result = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} has invalid alpha {value!r}"
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} alpha outside [0,1]: {numeric}"
            )
        result.append(numeric)
    return result


def validate_records(
    records: dict[int, dict[str, Any]],
    expected_ids: set[int],
    source: str,
    expected_samples: int | None,
    require_alpha: bool,
) -> None:
    actual_ids = set(records)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"{source}: ID mismatch; missing={missing}, extra={extra}")
    for pid, record in records.items():
        for variant in VARIANTS:
            flags = extract_flags(record, variant, source)
            if expected_samples is not None and len(flags) != expected_samples:
                raise ValueError(
                    f"{source}: problem {pid} {variant} has {len(flags)} samples; "
                    f"expected {expected_samples}"
                )
            alpha_values = extract_alpha_values(record, variant, source)
            if require_alpha:
                if len(alpha_values) != len(flags):
                    raise ValueError(
                        f"{source}: problem {pid} {variant} alpha count differs from sample count"
                    )


def load_fold_expectations(id_root: Path) -> tuple[dict[int, set[int]], dict[str, set[int]]]:
    fold_ids: dict[int, set[int]] = {}
    eligible = {"simple": set(), "hard": set()}
    all_seen: set[int] = set()
    for fold in FOLDS:
        path = id_root / f"fold_{fold}" / "test_ids.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold test IDs: {path}")
        payload = read_json(path)
        raw_current = [normalise_id(value) for value in payload.get("all", [])]
        current = set(raw_current)
        if not current:
            raise ValueError(f"Empty test ID list: {path}")
        if len(current) != len(raw_current):
            raise ValueError(f"Duplicate test IDs: {path}")
        overlap = all_seen & current
        if overlap:
            raise ValueError(f"Outer test folds overlap at problem IDs: {sorted(overlap)}")
        all_seen.update(current)
        fold_ids[fold] = current
        for variant in ("simple", "hard"):
            raw_variant_ids = [normalise_id(value) for value in payload.get(variant, [])]
            variant_ids = set(raw_variant_ids)
            if len(variant_ids) != len(raw_variant_ids):
                raise ValueError(f"{path}: duplicate {variant} eligible IDs")
            if not variant_ids <= current:
                raise ValueError(f"{path}: {variant} eligible IDs are outside this test fold")
            eligible[variant].update(variant_ids)
    return fold_ids, eligible


def metric_for_variant(
    records: dict[int, dict[str, Any]], ids: Iterable[int], variant: str, source: str
) -> dict[str, Any]:
    selected = sorted(set(ids))
    first_correct = 0
    any_correct = 0
    sample_correct = 0
    sample_total = 0
    for pid in selected:
        flags = extract_flags(records[pid], variant, source)
        first_correct += int(flags[0])
        any_correct += int(any(flags))
        sample_correct += sum(flags)
        sample_total += len(flags)
    total = len(selected)
    return {
        "total": total,
        "first_correct": first_correct,
        "first_accuracy": first_correct / total if total else None,
        "any_correct": any_correct,
        "any_accuracy": any_correct / total if total else None,
        "sample_correct": sample_correct,
        "sample_total": sample_total,
        "sample_accuracy": sample_correct / sample_total if sample_total else None,
    }


def all_variants_metric(
    records: dict[int, dict[str, Any]], ids: Iterable[int], source: str
) -> dict[str, Any]:
    selected = sorted(set(ids))
    correct = sum(
        all(extract_flags(records[pid], variant, source)[0] for variant in VARIANTS)
        for pid in selected
    )
    total = len(selected)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "definition": "all three variants are correct at first@1 for the same problem",
    }


def alpha_summary(records: dict[int, dict[str, Any]], ids: Iterable[int], variant: str) -> dict[str, Any]:
    values: list[float] = []
    for pid in sorted(set(ids)):
        values.extend(extract_alpha_values(records[pid], variant, "DSPR OOF"))
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty list")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def exact_mcnemar_p(baseline_only: int, dspr_only: int) -> float:
    discordant = baseline_only + dspr_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, dspr_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def comparison_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], byteorder="big", signed=False)


def paired_comparison(
    ids: Iterable[int],
    dspr_outcome: Callable[[int], bool],
    baseline_outcome: Callable[[int], bool],
    bootstrap_samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    selected = sorted(set(ids))
    if not selected:
        raise ValueError(f"Cannot compare an empty slice: {label}")
    dspr_values = [int(dspr_outcome(pid)) for pid in selected]
    baseline_values = [int(baseline_outcome(pid)) for pid in selected]
    differences = [new - base for new, base in zip(dspr_values, baseline_values)]
    delta = statistics.fmean(differences)

    rng = random.Random(comparison_seed(seed, label))
    bootstrapped = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(selected)) for _ in selected]
        bootstrapped.append(statistics.fmean(differences[index] for index in indices))
    bootstrapped.sort()

    baseline_only = sum(base == 1 and new == 0 for new, base in zip(dspr_values, baseline_values))
    dspr_only = sum(new == 1 and base == 0 for new, base in zip(dspr_values, baseline_values))
    both_correct = sum(new == 1 and base == 1 for new, base in zip(dspr_values, baseline_values))
    both_wrong = len(selected) - baseline_only - dspr_only - both_correct
    return {
        "total": len(selected),
        "dspr_correct": sum(dspr_values),
        "dspr_accuracy": statistics.fmean(dspr_values),
        "baseline_correct": sum(baseline_values),
        "baseline_accuracy": statistics.fmean(baseline_values),
        "delta": delta,
        "paired_bootstrap_95_ci": [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only_correct": baseline_only,
        "dspr_only_correct": dspr_only,
        "mcnemar_exact_p": exact_mcnemar_p(baseline_only, dspr_only),
    }


def build_comparisons(
    dspr: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"full": {}, "eligible": {}, "noneligible": {}}

    def variant_outcome(records: dict[int, dict[str, Any]], pid: int, variant: str) -> bool:
        return extract_flags(records[pid], variant, "paired comparison")[0]

    for variant in VARIANTS:
        result["full"][variant] = paired_comparison(
            all_ids,
            lambda pid, v=variant: variant_outcome(dspr, pid, v),
            lambda pid, v=variant: variant_outcome(baseline, pid, v),
            bootstrap_samples,
            seed,
            f"full/{variant}",
        )
    result["full"]["all_variants"] = paired_comparison(
        all_ids,
        lambda pid: all(variant_outcome(dspr, pid, variant) for variant in VARIANTS),
        lambda pid: all(variant_outcome(baseline, pid, variant) for variant in VARIANTS),
        bootstrap_samples,
        seed,
        "full/all_variants",
    )
    for variant in ("simple", "hard"):
        result["eligible"][variant] = paired_comparison(
            eligible[variant],
            lambda pid, v=variant: variant_outcome(dspr, pid, v),
            lambda pid, v=variant: variant_outcome(baseline, pid, v),
            bootstrap_samples,
            seed,
            f"eligible/{variant}",
        )
        ineligible = all_ids - eligible[variant]
        result["noneligible"][variant] = paired_comparison(
            ineligible,
            lambda pid, v=variant: variant_outcome(dspr, pid, v),
            lambda pid, v=variant: variant_outcome(baseline, pid, v),
            bootstrap_samples,
            seed,
            f"noneligible/{variant}",
        )
    return result


def write_metrics_csv(path: Path, metrics: dict[str, Any], comparisons: dict[str, Any] | None) -> None:
    fields = [
        "scope",
        "variant",
        "total",
        "dspr_correct",
        "dspr_accuracy",
        "baseline_correct",
        "baseline_accuracy",
        "delta",
        "ci95_low",
        "ci95_high",
        "mcnemar_exact_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scope, variants in metrics.items():
            for variant, metric in variants.items():
                comparison = comparisons.get(scope, {}).get(variant) if comparisons else None
                if variant == "all_variants":
                    correct = metric["correct"]
                    accuracy = metric["accuracy"]
                else:
                    correct = metric["first_correct"]
                    accuracy = metric["first_accuracy"]
                row = {
                    "scope": scope,
                    "variant": variant,
                    "total": metric["total"],
                    "dspr_correct": correct,
                    "dspr_accuracy": accuracy,
                    "baseline_correct": "",
                    "baseline_accuracy": "",
                    "delta": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "mcnemar_exact_p": "",
                }
                if comparison:
                    row.update(
                        {
                            "baseline_correct": comparison["baseline_correct"],
                            "baseline_accuracy": comparison["baseline_accuracy"],
                            "delta": comparison["delta"],
                            "ci95_low": comparison["paired_bootstrap_95_ci"][0],
                            "ci95_high": comparison["paired_bootstrap_95_ci"][1],
                            "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                        }
                    )
                writer.writerow(row)


def write_alpha_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = ["variant", "count", "mean", "std", "median", "min", "max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, summary in summaries.items():
            writer.writerow({"variant": variant, **summary})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="output/qwen_kfold_seed42")
    parser.add_argument("--id-root", default="data/qwen/kfold")
    parser.add_argument("--raw-data", default="data/math_paired.jsonl")
    parser.add_argument("--baseline", default=None, help="Optional matched baseline all_records.jsonl")
    parser.add_argument("--expected-samples", type=int, default=1, help="Use 0 to accept any positive sample count")
    parser.add_argument("--expected-problems", type=int, default=279)
    parser.add_argument("--expected-simple-eligible", type=int, default=221)
    parser.add_argument("--expected-hard-eligible", type=int, default=167)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_samples < 0:
        raise ValueError("--expected-samples cannot be negative")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    result_root = (repo_root / args.result_root).resolve() if not Path(args.result_root).is_absolute() else Path(args.result_root)
    id_root = (repo_root / args.id_root).resolve() if not Path(args.id_root).is_absolute() else Path(args.id_root)
    raw_path = (repo_root / args.raw_data).resolve() if not Path(args.raw_data).is_absolute() else Path(args.raw_data)
    expected_sample_count = args.expected_samples or None

    fold_ids, eligible = load_fold_expectations(id_root)
    raw_records = load_id_map(read_jsonl(raw_path), raw_path)
    all_ids = set(raw_records)
    if len(all_ids) != args.expected_problems:
        raise ValueError(
            f"Raw problem count is {len(all_ids)}; expected {args.expected_problems}. "
            "Check --raw-data or override --expected-problems explicitly."
        )
    expected_all_ids = set().union(*fold_ids.values())
    if expected_all_ids != all_ids:
        missing = sorted(all_ids - expected_all_ids)
        extra = sorted(expected_all_ids - all_ids)
        raise ValueError(f"Fold test universe differs from raw data: missing={missing}, extra={extra}")
    expected_eligible_counts = {
        "simple": args.expected_simple_eligible,
        "hard": args.expected_hard_eligible,
    }
    for variant, expected_count in expected_eligible_counts.items():
        if len(eligible[variant]) != expected_count:
            raise ValueError(
                f"{variant} eligible count is {len(eligible[variant])}; expected {expected_count}. "
                "Check the fold files or override the expected count explicitly."
            )

    merged: dict[int, dict[str, Any]] = {}
    fold_counts: dict[str, int] = {}
    for fold in FOLDS:
        path = result_root / f"fold_{fold}" / "all_records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} result: {path}")
        fold_records = load_id_map(read_jsonl(path), path)
        validate_records(
            fold_records,
            fold_ids[fold],
            str(path),
            expected_sample_count,
            require_alpha=True,
        )
        overlap = set(merged) & set(fold_records)
        if overlap:
            raise ValueError(f"OOF result overlap at IDs: {sorted(overlap)}")
        for pid, record in fold_records.items():
            merged[pid] = {**record, "outer_fold": fold}
        fold_counts[str(fold)] = len(fold_records)

    validate_records(merged, all_ids, "merged DSPR OOF", expected_sample_count, require_alpha=True)

    metrics = {
        "full": {
            **{
                variant: metric_for_variant(merged, all_ids, variant, "DSPR OOF")
                for variant in VARIANTS
            },
            "all_variants": all_variants_metric(merged, all_ids, "DSPR OOF"),
        },
        "eligible": {
            variant: metric_for_variant(merged, eligible[variant], variant, "DSPR OOF")
            for variant in ("simple", "hard")
        },
        "noneligible": {
            variant: metric_for_variant(merged, all_ids - eligible[variant], variant, "DSPR OOF")
            for variant in ("simple", "hard")
        },
    }
    alpha = {variant: alpha_summary(merged, all_ids, variant) for variant in VARIANTS}
    routing_diagnostics = {
        "hard_minus_simple_mean_alpha": alpha["hard"]["mean"] - alpha["simple"]["mean"],
        "simple_minus_original_mean_alpha": alpha["simple"]["mean"] - alpha["original"]["mean"],
    }

    baseline_map: dict[int, dict[str, Any]] | None = None
    comparisons = None
    baseline_path = None
    if args.baseline:
        baseline_path = (repo_root / args.baseline).resolve() if not Path(args.baseline).is_absolute() else Path(args.baseline)
        baseline_map = load_id_map(read_jsonl(baseline_path), baseline_path)
        validate_records(
            baseline_map,
            all_ids,
            str(baseline_path),
            expected_samples=None,
            require_alpha=False,
        )
        comparisons = build_comparisons(
            merged,
            baseline_map,
            all_ids,
            eligible,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )

    merged_path = result_root / "oof_all_records.jsonl"
    metrics_path = result_root / "oof_metrics.json"
    metrics_csv_path = result_root / "oof_metrics.csv"
    alpha_csv_path = result_root / "oof_alpha_summary.csv"
    write_jsonl(merged_path, [merged[pid] for pid in sorted(merged)])
    report = {
        "validation": {
            "status": "passed",
            "raw_problem_count": len(all_ids),
            "oof_problem_count": len(merged),
            "unique_problem_count": len(set(merged)),
            "fold_counts": fold_counts,
            "eligible_counts": {variant: len(ids) for variant, ids in eligible.items()},
            "noneligible_counts": {
                variant: len(all_ids - ids) for variant, ids in eligible.items()
            },
            "expected_samples_per_variant": expected_sample_count,
            "expected_problem_count": args.expected_problems,
            "expected_eligible_counts": expected_eligible_counts,
        },
        "sources": {
            "result_root": str(result_root),
            "id_root": str(id_root),
            "raw_data": str(raw_path),
            "baseline": str(baseline_path) if baseline_path else None,
        },
        "metrics": metrics,
        "alpha_summary": alpha,
        "routing_diagnostics": routing_diagnostics,
        "paired_comparisons": comparisons,
    }
    write_json(metrics_path, report)
    write_metrics_csv(metrics_csv_path, metrics, comparisons)
    write_alpha_csv(alpha_csv_path, alpha)

    print("OOF validation passed")
    print(f"Fold counts: {fold_counts}; total unique problems: {len(merged)}")
    print(
        "Eligible counts: "
        + ", ".join(f"{variant}={len(ids)}" for variant, ids in eligible.items())
    )
    for variant in VARIANTS:
        metric = metrics["full"][variant]
        print(
            f"Full {variant}: {metric['first_correct']}/{metric['total']} "
            f"({metric['first_accuracy'] * 100:.2f}%)"
        )
    for variant in ("simple", "hard"):
        metric = metrics["eligible"][variant]
        print(
            f"Eligible {variant}: {metric['first_correct']}/{metric['total']} "
            f"({metric['first_accuracy'] * 100:.2f}%)"
        )
    if comparisons:
        print("Paired baseline comparisons and bootstrap CIs were written to the metrics files.")
    print(f"Merged OOF: {merged_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
