"""LLM-based DAG analysis over segmented reasoning chains.

Input records may either come from cot_segmenter.py and contain per-sample
`steps`, or be raw all_records JSONL passed with ``--raw-input``. In the latter
case this script creates or reuses a persistent segmented cache before DAG
processing. It then asks an external LLM to label each step with dependencies
and stores the resulting DAG annotations back into each sample.

Use normal mode for small runs. Use batch mode for full experiments because it
creates provider-ready request files and avoids many synchronous API calls.
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from data_analysis.cot_segmenter import (
        LONG_PARAGRAPH_THRESHOLD,
        MIN_STEP_LENGTH,
        segment_jsonl_file,
    )
    from data_analysis.llm.config import LLMConfig
    from data_analysis.llm.api_client import LLMClient
    from data_analysis.llm.batch_processor import BatchProcessor
except ModuleNotFoundError:
    from src.data_analysis.cot_segmenter import (
        LONG_PARAGRAPH_THRESHOLD,
        MIN_STEP_LENGTH,
        segment_jsonl_file,
    )
    from src.data_analysis.llm.config import LLMConfig
    from src.data_analysis.llm.api_client import LLMClient
    from src.data_analysis.llm.batch_processor import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output/dag_analysis"


def load_records(input_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load segmented records from JSONL file."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    logger.info(f"Loaded {len(records)} records from {input_path}")
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
    """Resolve an existing segmented input or build/reuse its cache."""
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
            "Segmented %d responses into %d steps (%.1f per response)",
            summary["samples"],
            summary["steps"],
            summary["average_steps"],
        )
        logger.info("Segmented cache saved to: %s", summary["output_path"])
    else:
        logger.info("Using fresh segmented cache: %s", cache_path)
    return cache_path


def process_normal_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: List[str] = ["original", "simple", "hard"]
):
    """Process records with synchronous API calls.

    This mode is easier to debug but slower and more expensive for large all_records_50-style datasets.
    """
    client = LLMClient(config)
    output_file = output_dir / "normal" / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / "logs" / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    error_log.parent.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    with open(output_file, "w", encoding="utf-8") as fout, \
         open(error_log, "w", encoding="utf-8") as ferr:

        for idx, record in enumerate(records):
            logger.info(f"Processing record {idx+1}/{len(records)}: problem_id={record['problem_id']}")

            enriched = record.copy()

            for variant in variants:
                if variant not in record:
                    continue

                entry = record[variant]
                if "samples" not in entry:
                    logger.warning(f"No samples found for {variant} variant")
                    continue

                for sample_idx, sample in enumerate(entry["samples"]):
                    if "steps" not in sample or not sample["steps"]:
                        logger.warning(f"No steps in {variant} sample {sample_idx}")
                        continue

                    start_time = datetime.now()
                    dag, error = client.analyze_reasoning_chain(
                        entry["problem"],
                        sample["steps"]
                    )
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000

                    if error:
                        logger.error(f"Failed {variant} sample {sample_idx}: {error}")
                        sample["dag_analysis"] = None
                        sample["dag_error"] = error
                        total_failed += 1

                        ferr.write(json.dumps({
                            "problem_id": record["problem_id"],
                            "variant": variant,
                            "sample_idx": sample_idx,
                            "error": error,
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False) + "\n")
                        ferr.flush()
                    else:
                        sample["dag_analysis"] = dag
                        sample["dag_metadata"] = {
                            "analyzed_at": datetime.now().isoformat(),
                            "model": config.model,
                            "processing_time_ms": int(processing_time)
                        }
                        total_processed += 1
                        logger.info(f"Success {variant} sample {sample_idx}: {len(dag)} dependencies")

            # Save record immediately
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()

    logger.info(f"Processing complete: {total_processed} succeeded, {total_failed} failed")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Error log: {error_log}")


def process_batch_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: List[str] = ["original", "simple", "hard"]
):
    """Create batch API requests for all requested variants.

    The output is not the final analyzed JSONL. Users must upload the generated
    request file to the provider, download the batch results, and run
    merge-batch mode.
    """
    processor = BatchProcessor(config, output_dir / "batch")

    # Prepare a single batch file containing all variants
    logger.info("Preparing batch requests...")
    batch_file = processor.prepare_batch_requests(records, variants)
    logger.info(f"Created batch file: {batch_file}")

    logger.info("\n" + "="*60)
    logger.info("BATCH MODE: Manual Upload Required")
    logger.info("="*60)
    logger.info(f"Batch request file created:")
    logger.info(f"  - {batch_file}")
    logger.info("\nNext steps:")
    logger.info("1. Upload this file to your LLM provider's batch API")
    logger.info("2. Wait for batch processing to complete")
    logger.info("3. Download the results file")
    logger.info("4. Run: python src/data_analysis/dag_analyzer.py --mode merge-batch --input <segmented_records.jsonl> --batch-results-file <results_file>")
    logger.info("="*60)


def merge_batch_results(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    batch_results_file: str,
):
    """Merge downloaded batch API results back into segmented records."""
    processor = BatchProcessor(config, output_dir / "batch")
    enriched_records = processor.process_batch_results(batch_results_file, records)

    output_file = output_dir / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Merged batch results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze reasoning chains with LLM")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, default=None,
                             help="Existing segmented-records JSONL file")
    input_group.add_argument("--raw-input", type=str, default=None,
                             help="Raw all_records JSONL; segment it before DAG processing")
    parser.add_argument(
        "--segmented-cache",
        type=str,
        default=None,
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
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for results")
    parser.add_argument("--mode", choices=["normal", "batch", "merge-batch"], default="normal",
                       help="Processing mode: normal, batch, or merge-batch")
    parser.add_argument("--batch-results-file", type=str, default=None,
                       help="Downloaded batch results JSONL file (for --mode merge-batch)")
    parser.add_argument("--provider", type=str, default="deepseek",
                       help="LLM provider (deepseek, openai, etc.)")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                       help="Model name")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of records to process (for testing)")
    parser.add_argument("--variants", nargs="+",
                       default=["original", "simple", "hard"],
                       help="Which variants to process")
    args = parser.parse_args()

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

    # Load records
    records = load_records(str(segmented_input), args.limit)

    # Initialize config
    config = LLMConfig(
        provider=args.provider,
        model=args.model
    )

    # Process based on mode
    if args.mode == "normal":
        process_normal_mode(records, config, output_dir, args.variants)
    elif args.mode == "batch":
        process_batch_mode(records, config, output_dir, args.variants)
    elif args.mode == "merge-batch":
        if not args.batch_results_file:
            parser.error("--batch-results-file is required for --mode merge-batch")
        merge_batch_results(records, config, output_dir, args.batch_results_file)


if __name__ == "__main__":
    main()
