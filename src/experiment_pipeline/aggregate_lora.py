#!/usr/bin/env python3
"""Validate and aggregate LoRA OOF predictions with paired comparators."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

from . import aggregate_dspr as common


FOLDS = common.FOLDS
VARIANTS = common.VARIANTS


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_metrics(
    records: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
) -> dict[str, Any]:
    return {
        "full": {
            **{
                variant: common.metric_for_variant(records, all_ids, variant, "LoRA OOF")
                for variant in VARIANTS
            },
            "all_variants": common.all_variants_metric(records, all_ids, "LoRA OOF"),
        },
        "eligible": {
            variant: common.metric_for_variant(
                records,
                eligible[variant],
                variant,
                "LoRA OOF",
            )
            for variant in ("simple", "hard")
        },
        "noneligible": {
            variant: common.metric_for_variant(
                records,
                all_ids - eligible[variant],
                variant,
                "LoRA OOF",
            )
            for variant in ("simple", "hard")
        },
    }


def first_outcome(
    records: dict[int, dict[str, Any]],
    problem_id: int,
    variant: str,
) -> bool:
    return common.extract_flags(records[problem_id], variant, "paired comparison")[0]


def all_variants_outcome(
    records: dict[int, dict[str, Any]],
    problem_id: int,
) -> bool:
    return all(first_outcome(records, problem_id, variant) for variant in VARIANTS)


def rename_comparison(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": raw["total"],
        "lora_correct": raw["dspr_correct"],
        "lora_accuracy": raw["dspr_accuracy"],
        "comparator_correct": raw["baseline_correct"],
        "comparator_accuracy": raw["baseline_accuracy"],
        "delta_lora_minus_comparator": raw["delta"],
        "paired_bootstrap_95_ci": raw["paired_bootstrap_95_ci"],
        "bootstrap_samples": raw["bootstrap_samples"],
        "both_correct": raw["both_correct"],
        "both_wrong": raw["both_wrong"],
        "comparator_only_correct": raw["baseline_only_correct"],
        "lora_only_correct": raw["dspr_only_correct"],
        "mcnemar_exact_p": raw["mcnemar_exact_p"],
    }


def compare_records(
    lora: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    bootstrap_samples: int,
    seed: int,
    comparator_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"full": {}, "eligible": {}, "noneligible": {}}
    for variant in VARIANTS:
        result["full"][variant] = rename_comparison(
            common.paired_comparison(
                all_ids,
                lambda pid, v=variant: first_outcome(lora, pid, v),
                lambda pid, v=variant: first_outcome(comparator, pid, v),
                bootstrap_samples,
                seed,
                f"{comparator_name}/full/{variant}",
            )
        )
    result["full"]["all_variants"] = rename_comparison(
        common.paired_comparison(
            all_ids,
            lambda pid: all_variants_outcome(lora, pid),
            lambda pid: all_variants_outcome(comparator, pid),
            bootstrap_samples,
            seed,
            f"{comparator_name}/full/all_variants",
        )
    )
    for variant in ("simple", "hard"):
        for scope, selected in (
            ("eligible", eligible[variant]),
            ("noneligible", all_ids - eligible[variant]),
        ):
            result[scope][variant] = rename_comparison(
                common.paired_comparison(
                    selected,
                    lambda pid, v=variant: first_outcome(lora, pid, v),
                    lambda pid, v=variant: first_outcome(comparator, pid, v),
                    bootstrap_samples,
                    seed,
                    f"{comparator_name}/{scope}/{variant}",
                )
            )
    return result


def write_metrics_csv(
    path: Path,
    metrics: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "comparator",
        "scope",
        "variant",
        "total",
        "lora_correct",
        "lora_accuracy",
        "comparator_correct",
        "comparator_accuracy",
        "delta_lora_minus_comparator",
        "ci95_low",
        "ci95_high",
        "mcnemar_exact_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if not comparisons:
            for scope, variants in metrics.items():
                for variant, metric in variants.items():
                    correct_key = "correct" if variant == "all_variants" else "first_correct"
                    accuracy_key = "accuracy" if variant == "all_variants" else "first_accuracy"
                    writer.writerow(
                        {
                            "comparator": "",
                            "scope": scope,
                            "variant": variant,
                            "total": metric["total"],
                            "lora_correct": metric[correct_key],
                            "lora_accuracy": metric[accuracy_key],
                        }
                    )
            return
        for comparator_name, scopes in comparisons.items():
            for scope, variants in scopes.items():
                for variant, comparison in variants.items():
                    writer.writerow(
                        {
                            "comparator": comparator_name,
                            "scope": scope,
                            "variant": variant,
                            "total": comparison["total"],
                            "lora_correct": comparison["lora_correct"],
                            "lora_accuracy": comparison["lora_accuracy"],
                            "comparator_correct": comparison["comparator_correct"],
                            "comparator_accuracy": comparison["comparator_accuracy"],
                            "delta_lora_minus_comparator": comparison[
                                "delta_lora_minus_comparator"
                            ],
                            "ci95_low": comparison["paired_bootstrap_95_ci"][0],
                            "ci95_high": comparison["paired_bootstrap_95_ci"][1],
                            "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="output/qwen_lora_qv_r3_seed42")
    parser.add_argument("--id-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--raw-data", default="data/math_paired.jsonl")
    parser.add_argument(
        "--base",
        default="output/qwen-2.5/greedy/all_records.jsonl",
        help="Matched greedy base-model result; use an empty string to skip",
    )
    parser.add_argument(
        "--dspr",
        default="output/qwen_kfold_seed42/oof_all_records.jsonl",
        help="DSPR OOF result; use an empty string to skip",
    )
    parser.add_argument("--expected-samples", type=int, default=1)
    parser.add_argument("--expected-problems", type=int, default=279)
    parser.add_argument(
        "--expected-simple-eligible",
        type=int,
        default=None,
        help=(
            "Optional consistency assertion. By default the count is inferred "
            "from <id-root>/fold_*/test_ids.json."
        ),
    )
    parser.add_argument(
        "--expected-hard-eligible",
        type=int,
        default=None,
        help=(
            "Optional consistency assertion. By default the count is inferred "
            "from <id-root>/fold_*/test_ids.json."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_samples < 0 or args.bootstrap_samples < 1:
        raise ValueError("Invalid expected sample count or bootstrap count")
    result_root = resolve_path(args.result_root)
    id_root = resolve_path(args.id_root)
    raw_path = resolve_path(args.raw_data)
    expected_samples = args.expected_samples or None
    fold_ids, eligible = common.load_fold_expectations(id_root)
    raw_records = common.load_id_map(common.read_jsonl(raw_path), raw_path)
    all_ids = set(raw_records)
    if len(all_ids) != args.expected_problems:
        raise ValueError(
            f"Raw problem count is {len(all_ids)}; expected {args.expected_problems}"
        )
    if set().union(*fold_ids.values()) != all_ids:
        raise ValueError("Fold test universe differs from raw data")
    expected_eligible, eligible_count_sources = common.resolve_eligible_count_expectations(
        eligible,
        {
            "simple": args.expected_simple_eligible,
            "hard": args.expected_hard_eligible,
        },
    )

    merged: dict[int, dict[str, Any]] = {}
    fold_counts: dict[str, int] = {}
    for fold in FOLDS:
        path = result_root / f"fold_{fold}" / "all_records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} LoRA result: {path}")
        fold_records = common.load_id_map(common.read_jsonl(path), path)
        common.validate_records(
            fold_records,
            fold_ids[fold],
            str(path),
            expected_samples,
            require_alpha=False,
        )
        overlap = set(merged) & set(fold_records)
        if overlap:
            raise ValueError(f"OOF result overlap at IDs: {sorted(overlap)}")
        for problem_id, record in fold_records.items():
            merged[problem_id] = {**record, "outer_fold": fold}
        fold_counts[str(fold)] = len(fold_records)
    common.validate_records(
        merged,
        all_ids,
        "merged LoRA OOF",
        expected_samples,
        require_alpha=False,
    )

    metrics = build_metrics(merged, all_ids, eligible)
    comparator_paths = {
        name: resolve_path(value)
        for name, value in (("base", args.base), ("dspr", args.dspr))
        if value
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for name, path in comparator_paths.items():
        comparator = common.load_id_map(common.read_jsonl(path), path)
        common.validate_records(
            comparator,
            all_ids,
            str(path),
            expected_samples=None,
            require_alpha=False,
        )
        comparisons[name] = compare_records(
            merged,
            comparator,
            all_ids,
            eligible,
            args.bootstrap_samples,
            args.bootstrap_seed,
            name,
        )

    merged_path = result_root / "oof_all_records.jsonl"
    metrics_path = result_root / "oof_metrics.json"
    csv_path = result_root / "oof_metrics.csv"
    common.write_jsonl(merged_path, [merged[problem_id] for problem_id in sorted(merged)])
    report = {
        "method": "parameter_matched_lora",
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
            "expected_samples_per_variant": expected_samples,
            "expected_eligible_counts": expected_eligible,
            "eligible_count_sources": eligible_count_sources,
        },
        "sources": {
            "result_root": str(result_root),
            "id_root": str(id_root),
            "raw_data": str(raw_path),
            "comparators": {name: str(path) for name, path in comparator_paths.items()},
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
    }
    common.write_json(metrics_path, report)
    write_metrics_csv(csv_path, metrics, comparisons)

    print("LoRA OOF validation passed")
    for variant in VARIANTS:
        metric = metrics["full"][variant]
        print(
            f"Full {variant}: {metric['first_correct']}/{metric['total']} "
            f"({metric['first_accuracy'] * 100:.2f}%)"
        )
    print(f"Comparators: {list(comparisons)}")
    print(f"Merged OOF: {merged_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Metrics CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
