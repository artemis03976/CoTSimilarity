#!/usr/bin/env python3
"""Shared OOF validation, aggregation, and paired statistics for DSPR, SPT, and LoRA."""

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

from .evaluate import held_out_id_path, parse_fold_list


FOLDS = tuple(range(5))
VARIANTS = ("original", "simple", "hard")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOTS = {
    "dspr": "output/qwen_staged_kfold_seed42",
    "spt": "output/qwen_spt_seed42",
    "lora": "output/qwen_lora_qv_r1_seed42",
}


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


def load_fold_expectations(
    id_root: Path, folds: Iterable[int] = FOLDS,
) -> tuple[dict[int, set[int]], dict[str, set[int]]]:
    fold_ids: dict[int, set[int]] = {}
    eligible = {"simple": set(), "hard": set()}
    all_seen: set[int] = set()
    for fold in folds:
        path = held_out_id_path(id_root, fold)
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


def resolve_eligible_count_expectations(
    eligible: dict[str, set[int]],
    requested: dict[str, int | None],
) -> tuple[dict[str, int], dict[str, str]]:
    """Infer eligibility counts from fold masks, with optional CLI assertions.

    The held-out ID manifest is the authoritative mask used by OOF evaluation. The
    command-line counts are therefore optional guards rather than a second
    source of truth that callers must keep synchronized by hand.
    """

    resolved: dict[str, int] = {}
    sources: dict[str, str] = {}
    for variant in ("simple", "hard"):
        actual_count = len(eligible[variant])
        expected_count = requested.get(variant)
        if expected_count is not None:
            if expected_count < 0:
                raise ValueError(f"Expected {variant} eligibility count cannot be negative")
            if actual_count != expected_count:
                raise ValueError(
                    f"{variant} eligible count is {actual_count}; expected {expected_count}. "
                    "Check the fold files or remove the explicit count to infer it "
                    "from --id-root."
                )
            sources[variant] = "explicit_assertion_verified_against_id_root"
        else:
            sources[variant] = "inferred_from_id_root_test_ids"
        resolved[variant] = actual_count
    return resolved, sources


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


def exact_mcnemar_p(comparator_only: int, method_only: int) -> float:
    discordant = comparator_only + method_only
    if discordant == 0:
        return 1.0
    tail = min(comparator_only, method_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def comparison_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], byteorder="big", signed=False)


def paired_comparison(
    ids: Iterable[int],
    method_outcome: Callable[[int], bool],
    comparator_outcome: Callable[[int], bool],
    bootstrap_samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    selected = sorted(set(ids))
    if not selected:
        raise ValueError(f"Cannot compare an empty slice: {label}")
    method_values = [int(method_outcome(pid)) for pid in selected]
    comparator_values = [int(comparator_outcome(pid)) for pid in selected]
    differences = [new - base for new, base in zip(method_values, comparator_values)]
    delta = statistics.fmean(differences)

    rng = random.Random(comparison_seed(seed, label))
    bootstrapped = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(selected)) for _ in selected]
        bootstrapped.append(statistics.fmean(differences[index] for index in indices))
    bootstrapped.sort()

    comparator_only = sum(base == 1 and new == 0 for new, base in zip(method_values, comparator_values))
    method_only = sum(new == 1 and base == 0 for new, base in zip(method_values, comparator_values))
    both_correct = sum(new == 1 and base == 1 for new, base in zip(method_values, comparator_values))
    both_wrong = len(selected) - comparator_only - method_only - both_correct
    return {
        "total": len(selected),
        "method_correct": sum(method_values),
        "method_accuracy": statistics.fmean(method_values),
        "comparator_correct": sum(comparator_values),
        "comparator_accuracy": statistics.fmean(comparator_values),
        "delta": delta,
        "paired_bootstrap_95_ci": [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "comparator_only_correct": comparator_only,
        "method_only_correct": method_only,
        "mcnemar_exact_p": exact_mcnemar_p(comparator_only, method_only),
    }


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def metric_slices(
    all_ids: set[int], eligible: dict[str, set[int]],
) -> dict[str, dict[str, set[int]]]:
    return {
        "full": {variant: all_ids for variant in (*VARIANTS, "all_variants")},
        "eligible": {variant: eligible[variant] for variant in ("simple", "hard")},
        "noneligible": {variant: all_ids - eligible[variant] for variant in ("simple", "hard")},
    }


def build_metrics(
    records: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    method: str,
) -> dict[str, Any]:
    label = f"{method.upper()} OOF"
    return {
        scope: {
            variant: (
                all_variants_metric(records, ids, label)
                if variant == "all_variants"
                else metric_for_variant(records, ids, variant, label)
            )
            for variant, ids in variants.items()
        }
        for scope, variants in metric_slices(all_ids, eligible).items()
    }


def first_outcome(records: dict[int, dict[str, Any]], pid: int, variant: str) -> bool:
    variants = VARIANTS if variant == "all_variants" else (variant,)
    return all(extract_flags(records[pid], current, "paired comparison")[0] for current in variants)


def comparison_fields(method: str) -> dict[str, str]:
    """Retain existing report column names while sharing all calculations."""
    comparator = "baseline" if method == "dspr" else "comparator"
    return {
        "method_correct": f"{method}_correct",
        "method_accuracy": f"{method}_accuracy",
        "comparator_correct": f"{comparator}_correct",
        "comparator_accuracy": f"{comparator}_accuracy",
        "delta": "delta" if method == "dspr" else f"delta_{method}_minus_comparator",
        "comparator_only_correct": f"{comparator}_only_correct",
        "method_only_correct": f"{method}_only_correct",
    }


def compare_records(
    records: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    bootstrap_samples: int,
    seed: int,
    comparator_name: str,
    method: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = comparison_fields(method)
    # Preserve historical seed labels so refactoring does not change the CIs.
    prefix = "" if method == "dspr" else f"{comparator_name}/"
    for scope, variants in metric_slices(all_ids, eligible).items():
        result[scope] = {}
        for variant, ids in variants.items():
            raw = paired_comparison(
                ids,
                lambda pid, v=variant: first_outcome(records, pid, v),
                lambda pid, v=variant: first_outcome(comparator, pid, v),
                bootstrap_samples,
                seed,
                f"{prefix}{scope}/{variant}",
            )
            result[scope][variant] = {fields.get(key, key): value for key, value in raw.items()}
    return result


def write_metrics_csv(
    path: Path,
    metrics: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    method: str,
) -> None:
    names = comparison_fields(method)
    fields = ([] if method == "dspr" else ["comparator"]) + [
        "scope", "variant", "total", names["method_correct"], names["method_accuracy"],
        names["comparator_correct"], names["comparator_accuracy"], names["delta"],
        "ci95_low", "ci95_high", "mcnemar_exact_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparator_name, scopes in (comparisons or {"": {}}).items():
            for scope, variants in metrics.items():
                for variant, metric in variants.items():
                    correct = "correct" if variant == "all_variants" else "first_correct"
                    accuracy = "accuracy" if variant == "all_variants" else "first_accuracy"
                    row = {
                        "scope": scope, "variant": variant, "total": metric["total"],
                        names["method_correct"]: metric[correct],
                        names["method_accuracy"]: metric[accuracy],
                    }
                    if method != "dspr":
                        row["comparator"] = comparator_name
                    comparison = scopes.get(scope, {}).get(variant)
                    if comparison:
                        row.update({
                            names[key]: comparison[names[key]]
                            for key in ("comparator_correct", "comparator_accuracy", "delta")
                        })
                        row.update({
                            "ci95_low": comparison["paired_bootstrap_95_ci"][0],
                            "ci95_high": comparison["paired_bootstrap_95_ci"][1],
                            "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                        })
                    writer.writerow(row)


def write_alpha_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = ["variant", "count", "mean", "std", "median", "min", "max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, summary in summaries.items():
            writer.writerow({"variant": variant, **summary})


def parse_args(method: str, argv: list[str] | None = None) -> argparse.Namespace:
    if method not in DEFAULT_RESULT_ROOTS:
        raise ValueError(f"Unsupported aggregation method: {method}")
    parser = argparse.ArgumentParser(description=f"Aggregate {method.upper()} K-fold OOF predictions")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOTS[method])
    parser.add_argument("--id-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--raw-data", default="data/canonical_math_paired.jsonl")
    parser.add_argument(
        "--baseline", "--base", dest="base",
        default=None if method == "dspr" else "output/qwen-2.5/base/all_records.jsonl",
        help="Matched base-model all_records.jsonl; use an empty string to skip",
    )
    if method != "dspr":
        parser.add_argument(
            "--dspr", default="output/qwen_staged_kfold_seed42/oof_all_records.jsonl",
            help="DSPR OOF comparator; use an empty string to skip",
        )
    parser.add_argument("--expected-samples", type=int, default=1, help="Use 0 to accept any positive sample count")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Selected folds; default: all five")
    parser.add_argument("--expected-problems", type=int, default=None)
    for variant in ("simple", "hard"):
        parser.add_argument(
            f"--expected-{variant}-eligible", type=int, default=None,
            help="Optional count assertion; otherwise inferred from held-out ID manifests",
        )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(method: str = "dspr", argv: list[str] | None = None) -> int:
    args = parse_args(method, argv)
    if args.expected_samples < 0:
        raise ValueError("--expected-samples cannot be negative")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    result_root = resolve_path(args.result_root)
    id_root = resolve_path(args.id_root)
    raw_path = resolve_path(args.raw_data)
    expected_samples = args.expected_samples or None
    require_alpha = method == "dspr"
    folds = parse_fold_list(args.folds)
    print(f"{method.upper()} aggregation: folds={folds}; results={result_root}", flush=True)
    print("Checking held-out IDs and reference dataset...", flush=True)
    fold_ids, eligible = load_fold_expectations(id_root, folds)
    raw_records = load_id_map(read_jsonl(raw_path), raw_path)
    all_ids = set().union(*fold_ids.values())
    if args.expected_problems is not None and len(all_ids) != args.expected_problems:
        raise ValueError(f"Selected OOF problem count is {len(all_ids)}; expected {args.expected_problems}")
    missing = all_ids - set(raw_records)
    if missing:
        raise ValueError(f"Fold held-out IDs are absent from raw data: {sorted(missing)}")
    expected_eligible, eligible_count_sources = resolve_eligible_count_expectations(
        eligible, {"simple": args.expected_simple_eligible, "hard": args.expected_hard_eligible},
    )

    merged: dict[int, dict[str, Any]] = {}
    fold_counts: dict[str, int] = {}
    for fold in folds:
        path = result_root / f"fold_{fold}" / "all_records.jsonl"
        print(f"[fold {fold}] validating predictions: {path}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} {method.upper()} result: {path}")
        fold_records = load_id_map(read_jsonl(path), path)
        validate_records(fold_records, fold_ids[fold], str(path), expected_samples, require_alpha)
        overlap = set(merged) & set(fold_records)
        if overlap:
            raise ValueError(f"OOF result overlap at IDs: {sorted(overlap)}")
        for pid, record in fold_records.items():
            merged[pid] = {**record, "outer_fold": fold}
        fold_counts[str(fold)] = len(fold_records)
        print(f"[fold {fold}] passed: {len(fold_records)} problems", flush=True)
    validate_records(merged, all_ids, f"merged {method.upper()} OOF", expected_samples, require_alpha)

    print(f"OOF coverage complete: {len(merged)} unique problems. Computing metrics...", flush=True)
    metrics = build_metrics(merged, all_ids, eligible, method)
    comparator_paths = {
        name: resolve_path(value)
        for name, value in (("base", args.base), ("dspr", getattr(args, "dspr", None)))
        if value
    }
    comparisons = {}
    for name, path in comparator_paths.items():
        print(f"Comparing {method.upper()} against {name}: "
              f"{args.bootstrap_samples} paired bootstrap draws per comparison...", flush=True)
        comparator = load_id_map(read_jsonl(path), path)
        comparator = {pid: record for pid, record in comparator.items() if pid in all_ids}
        validate_records(comparator, all_ids, str(path), expected_samples=None, require_alpha=False)
        comparisons[name] = compare_records(
            merged, comparator, all_ids, eligible, args.bootstrap_samples, args.bootstrap_seed, name, method,
        )
        print(f"Comparison against {name} complete.", flush=True)

    # Keep the existing DSPR/base and baseline/multiple-comparator report layouts.
    report = {
        "validation": {
            "status": "passed",
            "raw_problem_count": len(raw_records),
            "oof_problem_count": len(merged),
            "unique_problem_count": len(merged),
            "fold_counts": fold_counts,
            "selected_folds": folds,
            "eligible_counts": {variant: len(ids) for variant, ids in eligible.items()},
            "noneligible_counts": {variant: len(all_ids - ids) for variant, ids in eligible.items()},
            "expected_samples_per_variant": expected_samples,
            "expected_eligible_counts": expected_eligible,
            "eligible_count_sources": eligible_count_sources,
        },
        "sources": {
            "result_root": str(result_root), "id_root": str(id_root), "raw_data": str(raw_path),
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
    }
    if require_alpha:
        alpha = {variant: alpha_summary(merged, all_ids, variant) for variant in VARIANTS}
        report["alpha_summary"] = alpha
        report["routing_diagnostics"] = {
            "hard_minus_simple_mean_alpha": alpha["hard"]["mean"] - alpha["simple"]["mean"],
            "simple_minus_original_mean_alpha": alpha["simple"]["mean"] - alpha["original"]["mean"],
        }
        report["validation"]["expected_problem_count"] = (
            args.expected_problems if args.expected_problems is not None else len(all_ids)
        )
        baseline_path = comparator_paths.get("base")
        report["sources"]["baseline"] = str(baseline_path) if baseline_path else None
        report["paired_comparisons"] = comparisons.get("base")
    else:
        report["method"] = method
        report["sources"]["comparators"] = {name: str(path) for name, path in comparator_paths.items()}

    merged_path = result_root / "oof_all_records.jsonl"
    metrics_path = result_root / "oof_metrics.json"
    csv_path = result_root / "oof_metrics.csv"
    write_jsonl(merged_path, [merged[pid] for pid in sorted(merged)])
    write_json(metrics_path, report)
    write_metrics_csv(csv_path, metrics, comparisons, method)
    if require_alpha:
        write_alpha_csv(result_root / "oof_alpha_summary.csv", report["alpha_summary"])

    print(f"{method.upper()} OOF validation passed")
    print(f"Fold counts: {fold_counts}; total unique problems: {len(merged)}")
    for scope in ("full", "eligible"):
        for variant, metric in metrics[scope].items():
            if variant == "all_variants":
                continue
            accuracy = metric["first_accuracy"]
            formatted = f"{accuracy * 100:.2f}%" if accuracy is not None else "n/a"
            print(f"{scope.title()} {variant}: {metric['first_correct']}/{metric['total']} ({formatted})")
    print(f"Comparators: {list(comparisons)}")
    print(f"Merged OOF: {merged_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Metrics CSV: {csv_path}")
    return 0
