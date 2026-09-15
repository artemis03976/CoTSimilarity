#!/usr/bin/env python3
"""Build shared group-level K-fold splits for the DSPR experiments.

The outer folds are defined over all raw problem groups.  Model-specific
eligibility masks are only used when constructing training supervision and
when reporting eligible/non-eligible evaluation slices; they never change
the raw outer test universe.

For each outer fold, an internal validation subset is selected from the
remaining development groups.  With the default K=5 and validation fraction
1/8 of development, the resulting proportions are approximately 70/10/20
for train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("simple", "hard")
MODELS = ("qwen", "deepseek")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    seen = set()
    for record in records:
        pid = normalise_id(record.get("problem_id"))
        if pid in seen:
            raise ValueError(f"Duplicate problem_id in raw dataset: {pid}")
        seen.add(pid)
    if not records:
        raise ValueError(f"Raw dataset is empty: {path}")
    return records


def load_raw_metadata(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for record in records:
        pid = normalise_id(record.get("problem_id"))
        if pid in metadata:
            raise ValueError(f"Duplicate problem_id in raw dataset: {pid}")
        metadata[pid] = {
            "type": str(record.get("type", "unknown")),
            "level": str(record.get("level", "unknown")),
        }
    return metadata


def load_eligibility(path: Path, raw_ids: set[int]) -> dict[str, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problem_ids = payload.get("problem_ids", {})
    result = {}
    for variant in VARIANTS:
        ids = {normalise_id(value) for value in problem_ids.get(variant, [])}
        unknown = ids - raw_ids
        if unknown:
            raise ValueError(f"{path}: {variant} contains IDs absent from raw dataset: {sorted(unknown)[:10]}")
        result[variant] = ids
    return result


def load_eligibility_payload(path: Path) -> dict[str, Any]:
    """Load the full manifest so the fold manifest records its protocol."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid eligibility JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Eligibility manifest must be an object: {path}")
    return payload


def build_labels(
    raw_ids: Iterable[int],
    metadata: dict[int, dict[str, Any]],
    eligibility: dict[str, dict[str, set[int]]],
) -> dict[int, frozenset[str]]:
    labels = {}
    for pid in raw_ids:
        current = {
            f"type={metadata[pid]['type']}",
            f"level={metadata[pid]['level']}",
        }
        for model in MODELS:
            for variant in VARIANTS:
                if pid in eligibility[model][variant]:
                    current.add(f"{model}_{variant}_eligible")
                else:
                    current.add(f"{model}_{variant}_ineligible")
        labels[pid] = frozenset(current)
    return labels


def balanced_assign(
    problem_ids: list[int],
    labels: dict[int, frozenset[str]],
    n_buckets: int,
    seed: int,
) -> dict[int, int]:
    """Deterministically assign IDs while balancing multi-label counts.

    This is a lightweight iterative/greedy stratifier.  It avoids an extra
    dependency while balancing type, level, and the four model/variant
    eligibility indicators.
    """
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    if len(problem_ids) < n_buckets:
        raise ValueError(f"Cannot assign {len(problem_ids)} IDs to {n_buckets} buckets")

    rng = random.Random(seed)
    shuffled = list(problem_ids)
    rng.shuffle(shuffled)

    label_frequency = Counter(label for pid in shuffled for label in labels[pid])
    tie_break = {pid: rng.random() for pid in shuffled}
    # Place rare and information-rich label combinations first.
    shuffled.sort(
        key=lambda pid: (
            min(label_frequency[label] for label in labels[pid]),
            -len(labels[pid]),
            tie_break[pid],
        )
    )

    target_size = len(shuffled) / n_buckets
    target_label = {
        label: frequency / n_buckets for label, frequency in label_frequency.items()
    }
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
            # Label balance is primary; size keeps buckets close when labels tie.
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
    splits = {}
    for outer_fold in range(n_folds):
        test_ids = sorted(pid for pid in raw_ids if outer_assignment[pid] == outer_fold)
        development_ids = sorted(pid for pid in raw_ids if outer_assignment[pid] != outer_fold)
        internal_assignment = balanced_assign(
            development_ids,
            labels,
            validation_buckets,
            seed=seed + 1009 * (outer_fold + 1),
        )
        # One of 8 internal buckets gives roughly 10% of the full raw data.
        val_ids = sorted(pid for pid in development_ids if internal_assignment[pid] == 0)
        train_ids = sorted(set(development_ids) - set(val_ids))
        if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
            raise AssertionError(f"Overlapping split IDs for outer fold {outer_fold}")
        if set(train_ids) | set(val_ids) | set(test_ids) != set(raw_ids):
            raise AssertionError(f"Split IDs do not cover raw dataset for outer fold {outer_fold}")
        splits[outer_fold] = {"train": train_ids, "val": val_ids, "test": test_ids}
    return splits


def split_status(splits: dict[int, dict[str, list[int]]], fold: int) -> dict[int, str]:
    status = {}
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
    union = sorted(set(simple) | set(hard))
    intersection = sorted(set(simple) & set(hard))
    return {
        "all": sorted(ids),
        "simple": simple,
        "hard": hard,
        "union": union,
        "intersection": intersection,
        "simple_only": sorted(set(simple) - set(hard)),
        "hard_only": sorted(set(hard) - set(simple)),
    }


def build_model_fold_data(
    model: str,
    refined_path: Path,
    model_eligibility: dict[str, set[int]],
    raw_ids: set[int],
    splits: dict[int, dict[str, list[int]]],
    output_root: Path,
) -> dict[str, Any]:
    refined_records = read_jsonl(refined_path)
    eligible_pairs = {(pid, variant) for variant in VARIANTS for pid in model_eligibility[variant]}
    available_pairs = set()
    for record in refined_records:
        pid = normalise_id(record.get("problem_id"))
        variant = str(record.get("variant_type", ""))
        if pid not in raw_ids or variant not in VARIANTS:
            continue
        if pid in model_eligibility[variant]:
            available_pairs.add((pid, variant))
    missing_pairs = sorted(eligible_pairs - available_pairs)

    model_root = output_root / model / "kfold"
    model_root.mkdir(parents=True, exist_ok=True)
    fold_summaries = []

    for fold, fold_split in splits.items():
        status = split_status(splits, fold)
        fold_root = model_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)

        write_json(fold_root / "train_ids.json", eligible_slices(fold_split["train"], model_eligibility))
        write_json(fold_root / "val_ids.json", eligible_slices(fold_split["val"], model_eligibility))
        write_json(fold_root / "test_ids.json", eligible_slices(fold_split["test"], model_eligibility))

        train_records = []
        val_records = []
        skipped_unknown = 0
        skipped_ineligible = 0
        for record in refined_records:
            pid = normalise_id(record.get("problem_id"))
            variant = str(record.get("variant_type", ""))
            if pid not in raw_ids:
                skipped_unknown += 1
                continue
            if variant not in VARIANTS or pid not in model_eligibility[variant]:
                skipped_ineligible += 1
                continue
            if status[pid] == "train":
                train_records.append(record)
            elif status[pid] == "val":
                val_records.append(record)

        write_jsonl(fold_root / "train.jsonl", train_records)
        write_jsonl(fold_root / "val.jsonl", val_records)

        train_pairs = {(normalise_id(row["problem_id"]), row["variant_type"]) for row in train_records}
        val_pairs = {(normalise_id(row["problem_id"]), row["variant_type"]) for row in val_records}
        test_slices = eligible_slices(fold_split["test"], model_eligibility)
        fold_summaries.append(
            {
                "fold": fold,
                "raw_train": len(fold_split["train"]),
                "raw_val": len(fold_split["val"]),
                "raw_test": len(fold_split["test"]),
                "eligible_train_union": len(eligible_slices(fold_split["train"], model_eligibility)["union"]),
                "eligible_val_union": len(eligible_slices(fold_split["val"], model_eligibility)["union"]),
                "eligible_test_union": len(test_slices["union"]),
                "simple_train_pairs": sum(1 for _, variant in train_pairs if variant == "simple"),
                "hard_train_pairs": sum(1 for _, variant in train_pairs if variant == "hard"),
                "simple_val_pairs": sum(1 for _, variant in val_pairs if variant == "simple"),
                "hard_val_pairs": sum(1 for _, variant in val_pairs if variant == "hard"),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "missing_eligible_pairs_total": len(missing_pairs),
                "skipped_unknown_records": skipped_unknown,
                "skipped_ineligible_records": skipped_ineligible,
            }
        )

    write_json(model_root / "coverage.json", {
        "model": model,
        "refined_source": str(refined_path),
        "eligible_pairs": len(eligible_pairs),
        "available_eligible_pairs": len(available_pairs),
        "missing_eligible_pairs": missing_pairs,
        "folds": fold_summaries,
    })
    return {"model": model, "folds": fold_summaries, "missing_pairs": missing_pairs}


def write_balance_csv(
    path: Path,
    splits: dict[int, dict[str, list[int]]],
    metadata: dict[int, dict[str, Any]],
    eligibility: dict[str, dict[str, set[int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "outer_fold",
        "split",
        "raw_count",
        "type_counts",
        "level_counts",
        "qwen_simple_eligible",
        "qwen_hard_eligible",
        "deepseek_simple_eligible",
        "deepseek_hard_eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold, fold_split in splits.items():
            for split_name, ids in fold_split.items():
                type_counts = Counter(metadata[pid]["type"] for pid in ids)
                level_counts = Counter(metadata[pid]["level"] for pid in ids)
                writer.writerow(
                    {
                        "outer_fold": fold,
                        "split": split_name,
                        "raw_count": len(ids),
                        "type_counts": json.dumps(dict(sorted(type_counts.items())), ensure_ascii=False),
                        "level_counts": json.dumps(dict(sorted(level_counts.items())), ensure_ascii=False),
                        **{
                            f"{model}_{variant}_eligible": sum(
                                pid in eligibility[model][variant] for pid in ids
                            )
                            for model in MODELS
                            for variant in VARIANTS
                        },
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", default="data/math_paired.jsonl")
    parser.add_argument("--qwen-eligibility", default="output/qwen-2.5/eligible_problem_ids_ged_range_ge_3.json")
    parser.add_argument("--deepseek-eligibility", default="output/deepseek/eligible_problem_ids_ged_range_ge_3.json")
    parser.add_argument("--qwen-refined", default="data/qwen/dcpr_dataset.jsonl")
    parser.add_argument("--deepseek-refined", default="data/deepseek/dcpr_dataset.jsonl")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-buckets",
        type=int,
        default=8,
        help="Internal development buckets; selecting one gives approximately 10%% of all raw groups.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_data)
    output_root = Path(args.output_root)
    raw_records = load_raw_records(raw_path)
    raw_records_by_id = {
        normalise_id(record["problem_id"]): record for record in raw_records
    }
    metadata = load_raw_metadata(raw_records)
    raw_ids = sorted(metadata)
    eligibility_paths = {
        "qwen": Path(args.qwen_eligibility),
        "deepseek": Path(args.deepseek_eligibility),
    }
    eligibility_payloads = {
        model: load_eligibility_payload(path)
        for model, path in eligibility_paths.items()
    }
    protocol_keys = ("score_field", "ged_range_threshold", "min_correct_samples", "top_k")
    qwen_protocol = tuple(eligibility_payloads["qwen"].get(key) for key in protocol_keys)
    deepseek_protocol = tuple(eligibility_payloads["deepseek"].get(key) for key in protocol_keys)
    if qwen_protocol != deepseek_protocol:
        raise ValueError(
            "Qwen and DeepSeek eligibility manifests use different protocols: "
            f"qwen={qwen_protocol}, deepseek={deepseek_protocol}"
        )
    eligibility = {
        model: load_eligibility(path, set(raw_ids))
        for model, path in eligibility_paths.items()
    }
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

    kfold_root = output_root / "kfold"
    kfold_root.mkdir(parents=True, exist_ok=True)

    # Shared raw test triplets are useful for later inference.  Training
    # supervision remains model-specific and is written below under each
    # model's kfold directory.
    for fold in range(args.k):
        test_ids = splits[fold]["test"]
        write_jsonl(
            kfold_root / f"fold_{fold}" / "test_raw.jsonl",
            [raw_records_by_id[pid] for pid in test_ids],
        )

    assignment_records = []
    for pid in raw_ids:
        fold_status = {str(fold): split_status(splits, fold)[pid] for fold in range(args.k)}
        assignment_records.append(
            {
                "problem_id": pid,
                **metadata[pid],
                "qwen_simple_eligible": pid in eligibility["qwen"]["simple"],
                "qwen_hard_eligible": pid in eligibility["qwen"]["hard"],
                "deepseek_simple_eligible": pid in eligibility["deepseek"]["simple"],
                "deepseek_hard_eligible": pid in eligibility["deepseek"]["hard"],
                "outer_fold": outer_assignment[pid],
                "fold_status": fold_status,
            }
        )
    write_jsonl(kfold_root / "fold_assignments.jsonl", assignment_records)
    write_balance_csv(kfold_root / "fold_balance.csv", splits, metadata, eligibility)

    model_summaries = {}
    for model, refined_arg in (("qwen", args.qwen_refined), ("deepseek", args.deepseek_refined)):
        model_summaries[model] = build_model_fold_data(
            model,
            Path(refined_arg),
            eligibility[model],
            set(raw_ids),
            splits,
            output_root,
        )

    manifest = {
        "raw_data": str(raw_path),
        "num_raw_problem_groups": len(raw_ids),
        "k": args.k,
        "seed": args.seed,
        "validation_buckets": args.validation_buckets,
        "validation_fraction_of_development": 1 / args.validation_buckets,
        "approximate_proportions": {
            "train": 1 - 1 / args.k - (1 - 1 / args.k) / args.validation_buckets,
            "val": (1 - 1 / args.k) / args.validation_buckets,
            "test": 1 / args.k,
        },
        # Keep the selected protocol explicit.  This is read from the
        # manifests rather than hard-coded so normalized-GED and legacy raw-GED
        # runs cannot be mislabeled in the fold metadata.
        "eligibility_protocols": {
            model: {
                "score_field": payload.get("score_field", "ged"),
                "range_threshold": payload.get("ged_range_threshold"),
                "min_correct_samples": payload.get("min_correct_samples", 1),
                "top_k": payload.get("top_k"),
                "comparison": payload.get("comparison"),
            }
            for model, payload in eligibility_payloads.items()
        },
        "eligibility_threshold": eligibility_payloads["qwen"].get("ged_range_threshold"),
        "eligibility_definition": eligibility_payloads["qwen"].get("comparison"),
        "eligibility_sources": {
            "qwen": str(eligibility_paths["qwen"]),
            "deepseek": str(eligibility_paths["deepseek"]),
        },
        "refined_sources": {
            "qwen": str(Path(args.qwen_refined)),
            "deepseek": str(Path(args.deepseek_refined)),
        },
        "model_summaries": model_summaries,
        "notes": [
            "Outer folds are shared across Qwen and DeepSeek and cover all raw problem groups.",
            "Model-specific eligibility is used as a training-supervision/evaluation mask, not to change the outer test universe.",
            "Each outer fold has an internal validation subset selected only from its development groups.",
            "Refined records unavailable for an eligible problem-variant pair are reported in each model coverage.json and are not fabricated.",
        ],
    }
    write_json(kfold_root / "manifest.json", manifest)

    print(f"Built {args.k}-fold shared assignment for {len(raw_ids)} raw problem groups")
    for model in MODELS:
        summary = model_summaries[model]
        print(
            f"{model}: missing eligible pairs in refined source="
            f"{len(summary['missing_pairs'])}"
        )
    print(f"Manifest: {kfold_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
