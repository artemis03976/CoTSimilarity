#!/usr/bin/env python3
"""Create the fixed development/test boundary for the canonical dataset.

This is the *outer* split used by the experiment design.  It is deliberately
not a train/validation/test split: all development problems are later used as
the pool for group-level k-fold cross-validation, while the held-out test
problems are touched only after the screening/GED protocol and model settings
are frozen.  Every row is one original problem group, so original/simple/hard
remain together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("development", "test")
DEFAULT_INPUT = "data/canonical_math_paired.jsonl"
DEFAULT_OUTPUT_DIR = "data/canonical_splits_seed42"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def allocate_counts(size: int, development_ratio: float) -> dict[str, int]:
    """Allocate a source stratum into development and held-out test."""
    if size < 2:
        raise ValueError("Each source stratum must contain at least two rows")
    # Round to nearest integer so the aggregate split stays close to the
    # requested ratio while preserving source stratification.
    development = int(size * development_ratio + 0.5)
    if development <= 0:
        development = 1
    if development >= size:
        development = size - 1
    return {"development": development, "test": size - development}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--development-ratio",
        type=float,
        default=0.8,
        help="Fraction assigned to development; default leaves 20%% held out.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not 0.0 < args.development_ratio < 1.0:
        raise ValueError("--development-ratio must be strictly between 0 and 1")

    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError("canonical dataset is empty")
    problem_ids = [row.get("problem_id") for row in rows]
    hashes = [row.get("original_hash") for row in rows]
    if any(problem_id is None for problem_id in problem_ids):
        raise ValueError("every canonical row must contain problem_id")
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("canonical problem_id values are not unique")
    if len(set(hashes)) != len(hashes):
        raise ValueError("canonical original_hash values are not unique")
    for row in rows:
        if any(not isinstance(row.get(variant), dict) for variant in ("original", "simple", "hard")):
            raise ValueError(f"problem_id={row['problem_id']} is missing a complete variant group")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row.get("source") or "unknown")].append(row)

    rng = random.Random(args.seed)
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    stratum_counts: dict[str, dict[str, int]] = {}
    assignments: dict[int, str] = {}
    for source in sorted(strata):
        source_rows = list(strata[source])
        rng.shuffle(source_rows)
        counts = allocate_counts(len(source_rows), args.development_ratio)
        stratum_counts[source] = {"total": len(source_rows), **counts}
        cursor = 0
        for split in SPLITS:
            selected = source_rows[cursor : cursor + counts[split]]
            cursor += counts[split]
            for row in selected:
                problem_id = int(row["problem_id"])
                assignments[problem_id] = split
                split_rows[split].append(row)

    for split in SPLITS:
        split_rows[split].sort(key=lambda row: int(row["problem_id"]))
    if len(assignments) != len(rows):
        raise AssertionError("not every canonical row received a split")
    split_id_sets = [set(row["problem_id"] for row in split_rows[split]) for split in SPLITS]
    if split_id_sets[0] & split_id_sets[1]:
        raise AssertionError("development and test overlap")
    if set.union(*split_id_sets) != set(problem_ids):
        raise AssertionError("split union does not match canonical dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output_dir / f"{split}.jsonl", split_rows[split])

    source_by_split = {
        split: dict(sorted(Counter(str(row.get("source") or "unknown") for row in split_rows[split]).items()))
        for split in SPLITS
    }
    status_by_split = {
        split: dict(
            sorted(
                Counter(
                    f"{row.get('source', 'unknown')}:{row.get('verification_status', 'unknown')}"
                    for row in split_rows[split]
                ).items()
            )
        )
        for split in SPLITS
    }
    manifest = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "seed": args.seed,
        "development_ratio": args.development_ratio,
        "test_ratio": round(1.0 - args.development_ratio, 10),
        "split_purpose": {
            "development": "all development problems; later eligible subset is used for group-level k-fold",
            "test": "held-out problems; do not use for model selection or threshold tuning",
        },
        "grouping_key": "problem_id / original_hash; original, simple, and hard remain together",
        "stratification": "source",
        "counts": {split: len(split_rows[split]) for split in SPLITS},
        "source_counts": source_by_split,
        "verification_status_counts": status_by_split,
        "source_strata": stratum_counts,
        "problem_ids": {split: [int(row["problem_id"]) for row in split_rows[split]] for split in SPLITS},
        "original_hashes": {split: [row["original_hash"] for row in split_rows[split]] for split in SPLITS},
        "outputs": {split: str(output_dir / f"{split}.jsonl") for split in SPLITS},
        "notes": [
            "All canonical rows, including held-out rows, may undergo cheap screening.",
            "DAG/GED on the held-out split is only used for final frozen-protocol reporting.",
            "The held-out test set is not a second validation set.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "source_counts": source_by_split}, ensure_ascii=False))
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
