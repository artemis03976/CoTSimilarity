#!/usr/bin/env python3
"""Revalidate a completed vLLM multi-path run without loading a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference.quality import CoTValidationConfig
from inference.revalidate import revalidate_multiple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        default="output/qwen/multiple_seed42/raw_generations.jsonl",
    )
    parser.add_argument(
        "--records-path",
        default="output/qwen/multiple_seed42/all_records.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="output/qwen/multiple_seed42_corrected",
    )
    parser.add_argument("--samples-per-problem", type=int, default=None)
    parser.add_argument("--repeat-ngram-size", "--max-repeat-period", type=int, default=64)
    parser.add_argument("--max-ngram-repeats", type=int, default=5)
    parser.add_argument("--min-repeat-span-tokens", type=int, default=64)
    parser.add_argument("--min-response-characters", type=int, default=20)
    parser.add_argument(
        "--require-boxed-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-valid-greedy-anchors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CoTValidationConfig(
        require_boxed_answer=args.require_boxed_answer,
        min_characters=args.min_response_characters,
        repeat_ngram_size=args.repeat_ngram_size,
        max_ngram_repeats=args.max_ngram_repeats,
        min_repeat_span_tokens=args.min_repeat_span_tokens,
    )
    paths = revalidate_multiple(
        Path(args.raw_path),
        Path(args.records_path),
        Path(args.output_dir),
        validation_config=config,
        samples_per_problem=args.samples_per_problem,
        require_valid_greedy_anchors=args.require_valid_greedy_anchors,
    )
    for label, path in zip(("all_records", "raw_generations", "generation_qc"), paths):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
