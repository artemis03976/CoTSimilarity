#!/usr/bin/env python3
"""Score saved greedy/multiple/K-fold responses in place using SymPy 1.14.0.

Run on the scoring machine after copying inference results. Inputs may be JSONL
files, a single inference directory, or K-fold run directories. No model, GPU,
canonical dataset, training checkpoints, or network connection is required.
Existing OOF/summary files are not refreshed; run aggregate after scoring folds.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.scoring import evaluator_metadata, prediction_paths, score_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Prediction JSONL files or run directories")
    parser.add_argument("--folds", type=int, nargs="+", help="Only these folds of each input run directory")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="CPU scoring workers (default: up to 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.folds is not None and (min(args.folds) < 0 or len(args.folds) != len(set(args.folds))):
        parser.error("--folds must contain distinct nonnegative fold numbers")
    evaluator = evaluator_metadata()
    paths = prediction_paths(args.inputs, args.folds)
    print(f"Offline scorer: SymPy {evaluator['sympy']}; {len(paths)} prediction files", flush=True)
    for path in paths:
        score_file(path, evaluator, args.workers)
    print("Scoring complete. Run aggregate for K-fold OOF results; existing summaries are not rebuilt automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
