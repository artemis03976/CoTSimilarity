"""Main orchestrator script for DAG analysis."""

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
    from data_analysis.llm.config import LLMConfig
    from data_analysis.llm.api_client import LLMClient
    from data_analysis.llm.batch_processor import BatchProcessor
except ModuleNotFoundError:
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


def process_normal_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: List[str] = ["original", "simple", "hard"]
):
    """Process records using normal API mode."""
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
    """Process records using batch inference mode."""
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
    parser.add_argument("--input", type=str, default=None,
                       help="Input JSONL file with segmented records")
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

    if not args.input:
        parser.error("--input is required")

    # Load records
    records = load_records(args.input, args.limit)

    # Initialize config
    config = LLMConfig(
        provider=args.provider,
        model=args.model
    )

    output_dir = Path(args.output_dir)

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
