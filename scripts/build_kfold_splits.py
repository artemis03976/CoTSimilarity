#!/usr/bin/env python3
"""Build Qwen DSPR K-fold data using the shared GED trajectory selector.

The outer folds cover every raw problem group. Prefix trajectories come only
from dspr_dataset.data_filter, using the saved eligibility protocol and GED
results. Raw sample arrays are never used to reconstruct training examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dspr_dataset.data_filter import VARIANTS, load_filtered_trajectories  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    seen: set[int] = set()
    for record in records:
        pid = normalise_id(record.get("problem_id"))
        if pid in seen:
            raise ValueError(f"Duplicate problem_id in raw dataset: {pid}")
        seen.add(pid)
    if not records:
        raise ValueError(f"Raw dataset is empty: {path}")
    return records


def load_raw_metadata(records: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    return {
        normalise_id(record["problem_id"]): {
            "type": str(record.get("type", "unknown")),
            "level": str(record.get("level", "unknown")),
        }
        for record in records
    }


def build_labels(
    raw_ids: Iterable[int],
    metadata: dict[int, dict[str, str]],
    eligibility: dict[str, set[int]],
) -> dict[int, frozenset[str]]:
    labels: dict[int, frozenset[str]] = {}
    for pid in raw_ids:
        current = {
            f"type={metadata[pid]['type']}",
            f"level={metadata[pid]['level']}",
        }
        for variant in VARIANTS:
            state = "eligible" if pid in eligibility[variant] else "ineligible"
            current.add(f"{variant}_{state}")
        labels[pid] = frozenset(current)
    return labels


def balanced_assign(
    problem_ids: list[int],
    labels: dict[int, frozenset[str]],
    n_buckets: int,
    seed: int,
) -> dict[int, int]:
    """Deterministically assign groups while balancing metadata labels."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    if len(problem_ids) < n_buckets:
        raise ValueError(f"Cannot assign {len(problem_ids)} IDs to {n_buckets} buckets")

    rng = random.Random(seed)
    shuffled = list(problem_ids)
    rng.shuffle(shuffled)
    label_frequency = Counter(label for pid in shuffled for label in labels[pid])
    tie_break = {pid: rng.random() for pid in shuffled}
    shuffled.sort(
        key=lambda pid: (
            min(label_frequency[label] for label in labels[pid]),
            -len(labels[pid]),
            tie_break[pid],
        )
    )

    target_size = len(shuffled) / n_buckets
    target_label = {label: count / n_buckets for label, count in label_frequency.items()}
    bucket_sizes = [0] * n_buckets
    bucket_label_counts = [Counter() for _ in range(n_buckets)]
    assignment: dict[int, int] = {}

    for pid in shuffled:
        pid_labels = labels[pid]
        scores = []
        for bucket in range(n_buckets):
            label_score = sum(
                (bucket_label_counts[bucket][label] + 1) / max(target_label[label], 1.0)
                for label in pid_labels
            )
            size_score = (bucket_sizes[bucket] + 1) / max(target_size, 1.0)
            scores.append((label_score + 0.15 * size_score, bucket_sizes[bucket], bucket))
        _, _, selected_bucket = min(scores)
        assignment[pid] = selected_bucket
        bucket_sizes[selected_bucket] += 1
        bucket_label_counts[selected_bucket].update(pid_labels)
    return assignment


def make_fold_splits(
    raw_ids: list[int],
    outer_assignment: dict[int, int],
    labels: dict[int, frozenset[str]],
    n_folds: int,
    seed: int,
    validation_buckets: int,
) -> dict[int, dict[str, list[int]]]:
    splits: dict[int, dict[str, list[int]]] = {}
    for outer_fold in range(n_folds):
        test_ids = sorted(pid for pid in raw_ids if outer_assignment[pid] == outer_fold)
        train_val_ids = sorted(pid for pid in raw_ids if outer_assignment[pid] != outer_fold)
        internal_assignment = balanced_assign(
            train_val_ids,
            labels,
            validation_buckets,
            seed=seed + 1009 * (outer_fold + 1),
        )
        val_ids = sorted(pid for pid in train_val_ids if internal_assignment[pid] == 0)
        train_ids = sorted(set(train_val_ids) - set(val_ids))
        if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise AssertionError(f"Overlapping split IDs for outer fold {outer_fold}")
        if set(train_ids) | set(val_ids) | set(test_ids) != set(raw_ids):
            raise AssertionError(f"Split IDs do not cover raw dataset for outer fold {outer_fold}")
        splits[outer_fold] = {"train": train_ids, "val": val_ids, "test": test_ids}
    return splits


def split_status(splits: dict[int, dict[str, list[int]]], fold: int) -> dict[int, str]:
    status: dict[int, str] = {}
    for split_name, ids in splits[fold].items():
        for pid in ids:
            if pid in status:
                raise AssertionError(f"Duplicate split assignment for problem {pid}")
            status[pid] = split_name
    return status


def eligible_slices(ids: Iterable[int], eligibility: dict[str, set[int]]) -> dict[str, list[int]]:
    ids = set(ids)
    simple = sorted(ids & eligibility["simple"])
    hard = sorted(ids & eligibility["hard"])
    simple_set, hard_set = set(simple), set(hard)
    return {
        "all": sorted(ids),
        "simple": simple,
        "hard": hard,
        "union": sorted(simple_set | hard_set),
        "intersection": sorted(simple_set & hard_set),
        "simple_only": sorted(simple_set - hard_set),
        "hard_only": sorted(hard_set - simple_set),
    }


def write_balance_csv(
    path: Path,
    splits: dict[int, dict[str, list[int]]],
    metadata: dict[int, dict[str, str]],
    eligibility: dict[str, set[int]],
) -> None:
    fields = [
        "outer_fold",
        "split",
        "raw_count",
        "type_counts",
        "level_counts",
        "simple_eligible",
        "hard_eligible",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold, fold_split in splits.items():
            for split_name, ids in fold_split.items():
                writer.writerow(
                    {
                        "outer_fold": fold,
                        "split": split_name,
                        "raw_count": len(ids),
                        "type_counts": json.dumps(
                            dict(sorted(Counter(metadata[pid]["type"] for pid in ids).items())),
                            ensure_ascii=False,
                        ),
                        "level_counts": json.dumps(
                            dict(sorted(Counter(metadata[pid]["level"] for pid in ids).items())),
                            ensure_ascii=False,
                        ),
                        "simple_eligible": sum(pid in eligibility["simple"] for pid in ids),
                        "hard_eligible": sum(pid in eligibility["hard"] for pid in ids),
                    }
                )


def build_qwen_folds(
    raw_records: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    eligibility: dict[str, set[int]],
    splits: dict[int, dict[str, list[int]]],
    qwen_root: Path,
) -> dict[str, Any]:
    raw_by_id = {normalise_id(record["problem_id"]): record for record in raw_records}
    fold_summaries = []

    for fold, fold_split in splits.items():
        fold_root = qwen_root / f"fold_{fold}"
        status = split_status(splits, fold)
        for split_name, ids in fold_split.items():
            write_json(fold_root / f"{split_name}_ids.json", eligible_slices(ids, eligibility))
        write_jsonl(
            fold_root / "test.jsonl",
            (raw_by_id[pid] for pid in fold_split["test"]),
        )

        train_records = [row for row in training_rows if status[row["problem_id"]] == "train"]
        val_records = [row for row in training_rows if status[row["problem_id"]] == "val"]
        write_jsonl(fold_root / "train.jsonl", train_records)
        write_jsonl(fold_root / "val.jsonl", val_records)

        train_pairs = {(row["problem_id"], row["variant_type"]) for row in train_records}
        val_pairs = {(row["problem_id"], row["variant_type"]) for row in val_records}
        fold_summaries.append(
            {
                "fold": fold,
                "raw_train": len(fold_split["train"]),
                "raw_val": len(fold_split["val"]),
                "raw_test": len(fold_split["test"]),
                "eligible_train_union": len(eligible_slices(fold_split["train"], eligibility)["union"]),
                "eligible_val_union": len(eligible_slices(fold_split["val"], eligibility)["union"]),
                "eligible_test_union": len(eligible_slices(fold_split["test"], eligibility)["union"]),
                "simple_train_pairs": sum(variant == "simple" for _, variant in train_pairs),
                "hard_train_pairs": sum(variant == "hard" for _, variant in train_pairs),
                "simple_val_pairs": sum(variant == "simple" for _, variant in val_pairs),
                "hard_val_pairs": sum(variant == "hard" for _, variant in val_pairs),
                "train_records": len(train_records),
                "val_records": len(val_records),
            }
        )

    return {
        "model": "qwen-2.5",
        "training_records": len(training_rows),
        "training_pairs": len({(row["problem_id"], row["variant_type"]) for row in training_rows}),
        "folds": fold_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, help="Nested raw all_records.jsonl")
    parser.add_argument("--eligible", required=True, help="Single Qwen eligibility JSON manifest")
    parser.add_argument(
        "--ged-results",
        help="GED JSONL source; defaults to the source recorded in --eligible",
    )
    parser.add_argument(
        "--output-root",
        default="output/qwen-2.5",
        help="Qwen output directory; folds are written under <output-root>/kfold",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-buckets",
        type=int,
        default=8,
        help="Internal train/validation buckets; one bucket is used for validation.",
    )
    args = parser.parse_args()
    if args.k < 2:
        raise ValueError("--k must be at least 2")

    raw_path = Path(args.raw_data)
    eligibility_path = Path(args.eligible)
    output_root = Path(args.output_root)
    qwen_root = output_root / "kfold"
    if qwen_root.exists() and any(qwen_root.iterdir()):
        raise FileExistsError(f"Existing folds are preserved: {qwen_root}. Choose a new --output-root.")
    raw_records = load_raw_records(raw_path)
    metadata = load_raw_metadata(raw_records)
    raw_ids = sorted(metadata)
    curation = load_filtered_trajectories(eligibility_path, args.ged_results)
    eligibility_payload = curation["eligibility"]
    eligibility = {
        variant: set(eligibility_payload["problem_ids"][variant]) for variant in VARIANTS
    }
    for variant, ids in eligibility.items():
        unknown = ids - set(raw_ids)
        if unknown:
            raise ValueError(
                f"{variant} contains IDs absent from raw dataset: {sorted(unknown)}"
            )

    labels = build_labels(raw_ids, metadata, eligibility)
    outer_assignment = balanced_assign(raw_ids, labels, args.k, args.seed)
    splits = make_fold_splits(
        raw_ids,
        outer_assignment,
        labels,
        args.k,
        args.seed,
        args.validation_buckets,
    )

    qwen_summary = build_qwen_folds(
        raw_records,
        curation["selected"],
        eligibility,
        splits,
        qwen_root,
    )
    qwen_summary["filter_stats"] = curation["stats"]

    assignments = []
    for pid in raw_ids:
        assignments.append(
            {
                "problem_id": pid,
                **metadata[pid],
                "simple_eligible": pid in eligibility["simple"],
                "hard_eligible": pid in eligibility["hard"],
                "outer_fold": outer_assignment[pid],
                "fold_status": {
                    str(fold): split_status(splits, fold)[pid] for fold in range(args.k)
                },
            }
        )
    write_jsonl(qwen_root / "fold_assignments.jsonl", assignments)
    write_balance_csv(qwen_root / "fold_balance.csv", splits, metadata, eligibility)
    write_json(qwen_root / "coverage.json", qwen_summary)

    manifest = {
        "model": "qwen-2.5",
        "raw_data": str(raw_path),
        "eligibility": str(eligibility_path),
        "kfold_root": str(qwen_root),
        "num_raw_problem_groups": len(raw_ids),
        "k": args.k,
        "seed": args.seed,
        "validation_buckets": args.validation_buckets,
        "validation_fraction_of_train_val": 1 / args.validation_buckets,
        "approximate_proportions": {
            "train": 1 - 1 / args.k - (1 - 1 / args.k) / args.validation_buckets,
            "val": (1 - 1 / args.k) / args.validation_buckets,
            "test": 1 / args.k,
        },
        "eligibility_protocol": {
            key: eligibility_payload.get(key)
            for key in ("score_field", "ged_range_threshold", "min_correct_samples", "top_k", "comparison")
            if key in eligibility_payload
        },
        "eligibility_counts": eligibility_payload.get("counts", {
            "simple": len(eligibility["simple"]),
            "hard": len(eligibility["hard"]),
            "union": len(eligibility["simple"] | eligibility["hard"]),
        }),
        "training": {
            "source": curation["source_file"],
            "filter": "dspr_dataset.data_filter.select_trajectories_by_ged",
            "strategies": ["EligibilityStrategy", "GEDRangeStrategy", "TopKStrategy"],
            "target_alpha": {"simple": 0.0, "hard": 1.0},
        },
        "coverage": qwen_summary,
        "notes": [
            "Outer test folds cover all raw problem groups exactly once.",
            "Prefix trajectories use the shared eligibility, GED min-max and top-k strategies.",
            "Eligibility labels do not change the raw fold universe.",
        ],
    }
    write_json(qwen_root / "manifest.json", manifest)

    print(f"Built {args.k}-fold Qwen assignment for {len(raw_ids)} raw problem groups")
    print(f"Training records: {qwen_summary['training_records']}")
    print(f"Manifest: {qwen_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
