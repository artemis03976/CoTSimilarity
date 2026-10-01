"""Batch-mode DAG request preparation and result merging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .llm.batch_processor import BatchProcessor
from .llm.config import LLMConfig

logger = logging.getLogger(__name__)
DEFAULT_VARIANTS = ["original", "simple", "hard"]


def process_batch_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: Optional[List[str]] = None,
    include_model: bool = False,
    batch_method: str | None = None,
    batch_url: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Create provider-ready batch requests for all selected chains."""
    processor = BatchProcessor(config, output_dir / "batch")
    batch_file = processor.prepare_batch_requests(
        records,
        variants or DEFAULT_VARIANTS,
        include_model=include_model,
        method=batch_method,
        url=batch_url,
        enable_thinking=enable_thinking,
    )
    logger.info("Created batch file: %s", batch_file)
    logger.info("Upload the file, download provider results, then run merge-batch.")
    return batch_file


def merge_batch_results(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    batch_results_file: str,
    variants: Optional[List[str]] = None,
) -> Path:
    """Merge downloaded batch responses into the original records."""
    processor = BatchProcessor(config, output_dir / "batch")
    enriched_records = processor.process_batch_results(
        batch_results_file,
        records,
        variants=variants or DEFAULT_VARIANTS,
    )
    output_file = output_dir / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for record in enriched_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Merged batch results saved to: %s", output_file)
    return output_file


__all__ = ["merge_batch_results", "process_batch_mode"]
