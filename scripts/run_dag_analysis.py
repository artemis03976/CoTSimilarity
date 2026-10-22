#!/usr/bin/env python3
"""Unified launcher for normal and batch DAG analysis modes."""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_analysis.cot_segmenter import (  # noqa: E402
    LONG_PARAGRAPH_THRESHOLD,
    MIN_STEP_LENGTH,
    segment_jsonl_file,
)
from data_analysis.annotation.batch import merge_batch_results, process_batch_mode  # noqa: E402
from data_analysis.annotation.normal import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    process_normal_mode,
)
from data_analysis.config import ANNOTATORS, LLMConfig  # noqa: E402
from data_analysis.annotation.protocols import annotation_settings  # noqa: E402
from utils.artifacts import file_identity, fingerprint  # noqa: E402
from utils.io import read_jsonl, read_manifest, write_json  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output/dag_analysis"
DEFAULT_VARIANTS = ["original", "simple", "hard"]


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

    manifest_path = cache_path.with_suffix(".config.json")
    signature = fingerprint({
        "input": file_identity(raw_path),
        "threshold": segment_threshold,
        "min_step": segment_min_step,
        "segmenter": file_identity(REPO_ROOT / "src/data_analysis/cot_segmenter.py")["sha256"],
    })
    manifest = read_manifest(manifest_path)
    cache_is_fresh = (
        cache_path.is_file()
        and manifest.get("signature") == signature
        and manifest.get("output") == file_identity(cache_path)
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
        write_json(manifest_path, {"signature": signature, "output": file_identity(cache_path)})
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
        help="Output directory; annotator presets default to a separate directory beside the input",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "batch", "merge-batch"],
        help="Processing mode; qwen defaults to batch, other annotators to normal",
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
    parser.add_argument("--batch-errors-file", help="Optional downloaded batch errors JSONL")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                        help="Reuse successful normal-mode checkpoints (default: enabled for annotator presets)")
    parser.add_argument(
        "--provider",
        help="Override API provider: deepseek, openai, anthropic, dashscope, volcengine",
    )
    parser.add_argument("--annotator", choices=list(ANNOTATORS), help="Named annotator preset")
    parser.add_argument("--model", help="Override the preset's model ID")
    parser.add_argument("--base-url", help="Override the provider API base URL")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Output token budget (includes reasoning where applicable)")
    parser.add_argument("--timeout", type=int, default=60, help="API timeout in seconds")
    parser.add_argument("--requests-per-minute", type=int, default=60, help="Shared normal-mode rate limit")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                        help="Provider reasoning effort; use with --thinking on")
    parser.add_argument(
        "--include-model-in-batch",
        action="store_true",
        default=None,
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
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", choices=["on", "off"],
                          help="Explicit thinking mode (new annotator presets default to off)")
    thinking.add_argument(
        "--disable-thinking",
        dest="thinking", action="store_const", const="off",
        help="Disable thinking using the selected provider's parameter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to process (for testing)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=DEFAULT_VARIANTS,
        help="Which variants to process",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.mode = args.mode or ("batch" if args.annotator == "qwen" else "normal")
    if args.resume is None:
        args.resume = bool(args.annotator and args.mode == "normal")
    if args.resume and args.mode != "normal":
        parser.error("--resume is only supported in normal mode")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.mode == "merge-batch" and not args.batch_results_file:
        parser.error("--batch-results-file is required for --mode merge-batch")
    if args.max_tokens <= 0 or args.timeout <= 0 or args.requests_per_minute <= 0:
        parser.error("--max-tokens, --timeout and --requests-per-minute must be positive")
    settings = {
        "provider": args.provider, "model": args.model, "base_url": args.base_url,
        "max_tokens": args.max_tokens, "timeout": args.timeout,
        "requests_per_minute": args.requests_per_minute,
        "thinking": None if args.thinking is None else args.thinking == "on",
        "reasoning_effort": args.reasoning_effort,
    }
    config = (LLMConfig.for_annotator(args.annotator, **settings) if args.annotator else
              LLMConfig(**{key: value for key, value in settings.items() if value is not None}))
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.annotator:
        parent = Path(args.raw_input).parent / "dag_analysis" if args.raw_input else Path(args.input).parent
        output_dir = parent / args.annotator
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
    try:
        effective = annotation_settings(config)
        for relative in ("normal/analysis.config.json", "batch/batch_requests.config.json"):
            previous = read_manifest(output_dir / relative)
            if previous and (previous.get("model") != config.model or
                             previous.get("provider", config.provider) != config.provider):
                raise ValueError("Output directory belongs to another annotator; use a separate --output-dir")
        if args.mode == "normal" and not config.api_key:
            raise ValueError(f"Set {config.provider.upper()}_API_KEY for normal mode")
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

    records = list(islice(read_jsonl(segmented_input), args.limit))
    logger.info("Loaded %s records from %s", len(records), segmented_input)
    logger.info("Annotator: %s; provider=%s model=%s mode=%s", args.annotator or "custom", config.provider, config.model, args.mode)
    logger.info("Output: %s; generation parameters: %s", output_dir, effective["parameters"])

    if args.mode == "normal":
        process_normal_mode(
            records,
            config,
            output_dir,
            args.variants,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            resume=args.resume,
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
        )
    else:
        merge_batch_results(
            records,
            config,
            output_dir,
            args.batch_results_file,
            variants=args.variants,
            batch_errors_file=args.batch_errors_file,
        )


if __name__ == "__main__":
    main()
