#!/usr/bin/env python3
"""Shared OOF validation, aggregation, and paired statistics for DSPR, SPT, and LoRA."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from utils.config import parse_config_args
from utils.io import read_json, read_records as read_jsonl, write_json, write_jsonl

from .metrics import (
    alpha_summary, build_metrics, compare_records, comparison_fields,
    extract_alpha_values, extract_flags,
)
from .common import (
    DEFAULT_FOLDS, DEFAULT_RESULT_ROOTS, VARIANTS, held_out_id_path, normalise_id,
    parse_fold_list, resolve_path,
)


def load_id_map(records: list[dict[str, Any]], source: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        pid = normalise_id(record.get("problem_id"))
        if pid in result:
            raise ValueError(f"Duplicate problem_id {pid} in {source}")
        result[pid] = record
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
    id_root: Path, folds: Iterable[int] = DEFAULT_FOLDS,
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
    if method == "spt":
        parser.add_argument("--lora", default=None, help="Optional LoRA OOF comparator")
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
    return parse_config_args(parser, argv, method=method, section="aggregation", aliases={
        "fold_root": "id_root", "problem_dataset": "raw_data",
    })


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
        for name, value in (
            ("base", args.base), ("dspr", getattr(args, "dspr", None)),
            ("lora", getattr(args, "lora", None)),
        )
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
