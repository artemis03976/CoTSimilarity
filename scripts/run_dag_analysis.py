#!/usr/bin/env python3
"""Unified launcher for normal and batch DAG analysis modes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_analysis.cot_segmenter import (  # noqa: E402
    LONG_PARAGRAPH_THRESHOLD,
    MIN_STEP_LENGTH,
    segment_jsonl_file,
)
from data_analysis.dag_batch import merge_batch_results, process_batch_mode  # noqa: E402
from data_analysis.dag_normal import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    process_normal_mode,
)
from data_analysis.llm.config import LLMConfig  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output/dag_analysis"
DEFAULT_VARIANTS = ["original", "simple", "hard"]


def load_records(input_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load segmented records from a JSONL file."""
    records = []
    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    logger.info("Loaded %s records from %s", len(records), input_path)
    return records


def resolve_segmented_input(
    input_path: Optional[str],
    raw_input_path: Optional[str],
    segmented_cache: Optional[str],
    output_dir: Path,
    refresh_cache: bool,
    segment_threshold: int,
    segment_min_step: int,
) -> Path:
    """Resolve an existing segmented file or create/reuse a raw-input cache."""
    if input_path:
        if segmented_cache or refresh_cache:
            raise ValueError(
                "--segmented-cache and --refresh-segment-cache require --raw-input"
            )
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Segmented input does not exist: {path}")
        return path

    if not raw_input_path:
        raise ValueError("Either --input or --raw-input is required")

    raw_path = Path(raw_input_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw input does not exist: {raw_path}")
    cache_path = (
        Path(segmented_cache)
        if segmented_cache
        else output_dir / "segmented_records.jsonl"
    )
    if raw_path.resolve() == cache_path.resolve():
        raise ValueError("Segmented cache must not overwrite the raw input")

    cache_is_fresh = (
        cache_path.is_file()
        and cache_path.stat().st_size > 0
        and cache_path.stat().st_mtime_ns >= raw_path.stat().st_mtime_ns
    )
    if refresh_cache or not cache_is_fresh:
        reason = "explicit refresh" if refresh_cache else "missing or stale cache"
        logger.info("Segmenting raw CoTs (%s): %s", reason, raw_path)
        summary = segment_jsonl_file(
            raw_path,
            cache_path,
            threshold=segment_threshold,
            min_step=segment_min_step,
        )
        logger.info(
            "Segmented %s responses into %s steps (%.1f per response)",
            summary["samples"],
            summary["steps"],
            summary["average_steps"],
        )
        logger.info("Segmented cache saved to: %s", summary["output_path"])
    else:
        logger.info("Using fresh segmented cache: %s", cache_path)
    return cache_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze reasoning chains with LLM")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Existing segmented-records JSONL file")
    input_group.add_argument(
        "--raw-input",
        type=str,
        help="Raw all_records JSONL; segment it before DAG processing",
    )
    parser.add_argument(
        "--segmented-cache",
        type=str,
        help="Cache path used with --raw-input; default: <output-dir>/segmented_records.jsonl",
    )
    parser.add_argument(
        "--refresh-segment-cache",
        action="store_true",
        help="Regenerate the segmented cache even when it is newer than the raw input",
    )
    parser.add_argument(
        "--segment-threshold",
        type=int,
        default=LONG_PARAGRAPH_THRESHOLD,
        help="Long-paragraph threshold forwarded to the CoT segmenter",
    )
    parser.add_argument(
        "--segment-min-step",
        type=int,
        default=MIN_STEP_LENGTH,
        help="Minimum segment length forwarded to the CoT segmenter",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "batch", "merge-batch"],
        default="normal",
        help="Processing mode: normal, batch, or merge-batch",
    )
    parser.add_argument(
        "--concurrency",
        "--workers",
        "--max-workers",
        dest="concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent normal-mode requests (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        "--retry-limit",
        dest="max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for each failed normal-mode chain (default: 5)",
    )
    parser.add_argument(
        "--batch-results-file",
        help="Downloaded batch results JSONL file (for --mode merge-batch)",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        help="LLM provider (deepseek, openai, etc.)",
    )
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument(
        "--include-model-in-batch",
        action="store_true",
        help="Include the configured model in each request body",
    )
    parser.add_argument(
        "--batch-method",
        help="Optional top-level HTTP method, e.g. POST, for OpenAI-compatible batch files",
    )
    parser.add_argument(
        "--batch-url",
        help="Optional top-level endpoint path, e.g. /v1/chat/completions",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Emit enable_thinking=false in every batch request body",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to process (for testing)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Which variants to process",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    try:
        segmented_input = resolve_segmented_input(
            args.input,
            args.raw_input,
            args.segmented_cache,
            output_dir,
            args.refresh_segment_cache,
            args.segment_threshold,
            args.segment_min_step,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    records = load_records(str(segmented_input), args.limit)
    config = LLMConfig(provider=args.provider, model=args.model)

    if args.mode == "normal":
        process_normal_mode(
            records,
            config,
            output_dir,
            args.variants,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    elif args.mode == "batch":
        process_batch_mode(
            records,
            config,
            output_dir,
            args.variants,
            include_model=args.include_model_in_batch,
            batch_method=args.batch_method,
            batch_url=args.batch_url,
            enable_thinking=False if args.disable_thinking else None,
        )
    else:
        if not args.batch_results_file:
            parser.error("--batch-results-file is required for --mode merge-batch")
        merge_batch_results(
            records,
            config,
            output_dir,
            args.batch_results_file,
            variants=args.variants,
        )


if __name__ == "__main__":
    main()
