"""Batch-mode DAG request preparation and result merging."""

from __future__ import annotations

import logging
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import LLMConfig
from utils.artifacts import fingerprint
from utils.io import atomic_text, read_jsonl, read_manifest, write_json, write_jsonl
from .parsing import batch_response_content, parse_dag_response
from ..records import iter_samples
from .prompts import build_prompt
from ..schemas import VARIANTS

logger = logging.getLogger(__name__)
DEFAULT_VARIANTS = list(VARIANTS)


class BatchProcessor:
    """Handle batch inference for cost-optimized processing."""

    def __init__(self, config: LLMConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch_requests(
        self,
        records: List[Dict],
        variants: List[str] = None,
        include_model: bool = False,
        method: str | None = None,
        url: str | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Prepare batch request file in JSONL format.

        All variants are written into a single file, distinguished by custom_id.

        Args:
            records: List of segmented records
            variants: Which variants to process (default: original/simple/hard)
            include_model: Include ``model`` in each request body.  The
                platform's recommended format selects the model outside the
                JSONL file, so this is disabled by default.
            method: Optional top-level HTTP method, such as ``POST``.
            url: Optional top-level endpoint path, such as
                ``/v1/chat/completions``.  These must be provided together.
            enable_thinking: Optional Qwen-compatible thinking switch.  When
                provided, it is emitted in every request body.

        Returns:
            Path to batch request file
        """
        if variants is None:
            variants = ["original", "simple", "hard"]
        if (method is None) != (url is None):
            raise ValueError("method and url must be provided together")

        batch_file = self.output_dir / "batch_requests.jsonl"

        count = 0
        # Use an explicit UTF-8/LF JSONL stream.  Some batch uploaders are
        # stricter than Python's JSON parser and reject files containing a
        # platform-specific BOM or translated CRLF records.
        identities = {}
        tasks = list(iter_samples(records, variants))
        with atomic_text(batch_file) as stream:
            for key, entry, sample in tasks:
                if not sample.get("steps"):
                    continue
                system_prompt, user_prompt = build_prompt(entry["problem"], sample["steps"])
                request_body = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "temperature": self.config.temperature,
                }
                if include_model:
                    request_body["model"] = self.config.model
                if enable_thinking is not None:
                    request_body["enable_thinking"] = enable_thinking
                request = {"custom_id": str(key), "body": request_body}
                if method is not None:
                    request.update(method=method, url=url)
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
                identities[str(key)] = fingerprint({
                    "problem": entry["problem"], "steps": sample["steps"], "response": sample.get("response"),
                })
                count += 1
        write_json(batch_file.with_suffix(".config.json"), {"samples": identities, "model": self.config.model})
        logger.info(f"Prepared {count} batch requests across {variants}: {batch_file}")
        return str(batch_file)

    def process_batch_results(
        self,
        batch_results_file: str,
        original_records: List[Dict],
        variants: List[str] = None,
    ) -> List[Dict]:
        """Process batch results and merge with original records.

        Args:
            batch_results_file: Path to batch results JSONL
            original_records: Original segmented records

        Returns:
            Records with DAG analysis added
        """
        if variants is None:
            variants = ["original", "simple", "hard"]

        results_map = {}
        for result in read_jsonl(batch_results_file):
            custom_id = result.get("custom_id")
            if not custom_id or custom_id in results_map:
                raise ValueError(f"Missing or duplicate batch sample ID: {custom_id}")
            results_map[custom_id] = result

        manifest = read_manifest(self.output_dir / "batch_requests.config.json")
        enriched_records = copy.deepcopy(original_records)
        counts = {"succeeded": 0, "failed": 0, "missing": 0}
        for key, entry, sample in iter_samples(enriched_records, variants):
            if not sample.get("steps"):
                continue
            custom_id = str(key)
            expected = manifest.get("samples", {}).get(custom_id)
            actual = fingerprint({"problem": entry.get("problem", ""), "steps": sample["steps"], "response": sample.get("response")})
            if manifest and expected != actual:
                raise ValueError(f"Segmented input changed since batch preparation: {custom_id}")
            sample["sample_id"] = custom_id
            sample.pop("dag_error", None)
            sample.pop("dag_metadata", None)
            try:
                if custom_id not in results_map:
                    counts["missing"] += 1
                    raise ValueError("Missing batch result")
                content = batch_response_content(results_map[custom_id])
                sample["dag_analysis"] = parse_dag_response(content, sample["steps"])
                sample["dag_metadata"] = {"model": manifest.get("model", self.config.model)}
                counts["succeeded"] += 1
            except ValueError as exc:
                sample["dag_analysis"] = None
                sample["dag_error"] = str(exc)
                counts["failed"] += 1
                logger.warning("Invalid batch result %s: %s", custom_id, exc)
        write_json(self.output_dir / "merge.config.json", {"results": str(Path(batch_results_file).resolve()), **counts})
        return enriched_records


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
    write_jsonl(output_file, enriched_records)
    logger.info("Merged batch results saved to: %s", output_file)
    return output_file


__all__ = ["merge_batch_results", "process_batch_mode"]
