"""Batch-mode DAG request preparation and result merging."""

from __future__ import annotations

import logging
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from ..config import LLMConfig
from utils.artifacts import file_identity, fingerprint
from utils.io import atomic_text, read_jsonl, read_manifest, write_json, write_jsonl
from .parsing import batch_response_body, parse_dag_response
from .protocols import annotation_settings, request_body, response_content
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
        include_model: bool | None = None,
        method: str | None = None,
        url: str | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Prepare batch request file in JSONL format.

        All variants are written into a single file, distinguished by custom_id.

        Args:
            records: List of segmented records
            variants: Which variants to process (default: original/simple/hard)
            include_model: Include ``model`` for platform-specific batch files.
                OpenAI, Bailian and Anthropic always include it automatically.
            method: Optional top-level HTTP method, such as ``POST``.
            url: Optional top-level endpoint path, such as
                ``/v1/chat/completions``.  These must be provided together.
            enable_thinking: Optional override of the provider's thinking mode.

        Returns:
            Path to batch request file
        """
        if variants is None:
            variants = ["original", "simple", "hard"]
        if (method is None) != (url is None):
            raise ValueError("method and url must be provided together")
        config = self.config if enable_thinking is None else replace(self.config, thinking=enable_thinking)
        if config.provider in {"openai", "dashscope"}:
            method, url = method or "POST", url or "/v1/chat/completions"
            include_model = True
        if config.provider == "anthropic" and method is not None:
            raise ValueError("Anthropic batches use custom_id/params, not method/url")

        batch_file = self.output_dir / "batch_requests.jsonl"
        settings = annotation_settings(config)
        previous = read_manifest(batch_file.with_suffix(".config.json"))
        if previous and (previous.get("model") != config.model or
                         previous.get("settings", settings) != settings):
            raise ValueError("Batch annotation settings changed; use a separate --output-dir")

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
                body = request_body(config, system_prompt, user_prompt)
                if not include_model and config.provider != "anthropic":
                    body.pop("model")
                request = {"custom_id": str(key),
                           "params" if config.provider == "anthropic" else "body": body}
                if method is not None:
                    request.update(method=method, url=url)
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
                identities[str(key)] = fingerprint({
                    "problem": entry["problem"], "steps": sample["steps"], "response": sample.get("response"),
                })
                count += 1
        write_json(batch_file.with_suffix(".config.json"), {
            "samples": identities, "model": config.model, "provider": config.provider,
            "settings": settings, "requests": file_identity(batch_file),
        })
        logger.info(f"Prepared {count} batch requests across {variants}: {batch_file}")
        return str(batch_file)

    def process_batch_results(
        self,
        batch_results_file: str,
        original_records: List[Dict],
        variants: List[str] = None,
        batch_errors_file: str | None = None,
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
        for path in filter(None, (batch_results_file, batch_errors_file)):
            for result in read_jsonl(path):
                custom_id = result.get("custom_id")
                if not custom_id or custom_id in results_map:
                    raise ValueError(f"Missing or duplicate batch sample ID: {custom_id}")
                results_map[custom_id] = result

        manifest = read_manifest(self.output_dir / "batch_requests.config.json")
        if manifest.get("settings") and manifest["settings"] != annotation_settings(self.config):
            raise ValueError("Batch annotation settings changed since preparation")
        if (manifest.get("requests") and manifest["requests"]["sha256"] !=
                file_identity(self.output_dir / "batch_requests.jsonl")["sha256"]):
            raise ValueError("Batch request file changed since preparation")
        enriched_records = copy.deepcopy(original_records)
        tasks = [(key, entry, sample) for key, entry, sample in iter_samples(enriched_records, variants)
                 if sample.get("steps")]
        expected_ids = {str(key) for key, _, _ in tasks}
        if manifest and set(manifest.get("samples", {})) != expected_ids:
            raise ValueError("Segmented input sample set changed since batch preparation")
        if results_map.keys() - expected_ids:
            raise ValueError("Batch results contain sample IDs outside the selected input")
        counts = {"succeeded": 0, "failed": 0, "missing": 0}
        for key, entry, sample in tasks:
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
                content, metadata = response_content(batch_response_body(results_map[custom_id]))
                reported_model = metadata.get("response_model")
                if (reported_model and self.config.provider in {"openai", "anthropic", "dashscope"}
                        and reported_model != self.config.model
                        and not reported_model.startswith(self.config.model + "-")):
                    raise ValueError(f"Batch response model mismatch: {reported_model}")
                sample["dag_analysis"] = parse_dag_response(content, sample["steps"])
                sample["dag_metadata"] = {"model": manifest.get("model", self.config.model),
                                          "provider": self.config.provider, **metadata}
                counts["succeeded"] += 1
            except ValueError as exc:
                sample["dag_analysis"] = None
                sample["dag_error"] = str(exc)
                counts["failed"] += 1
                logger.warning("Invalid batch result %s: %s", custom_id, exc)
        write_json(self.output_dir / "merge.config.json", {
            "results": file_identity(batch_results_file),
            "errors": file_identity(batch_errors_file) if batch_errors_file else None,
            "settings": annotation_settings(self.config), **counts,
        })
        logger.info("Batch merge: %s succeeded, %s failed (%s missing)",
                    counts["succeeded"], counts["failed"], counts["missing"])
        return enriched_records


def process_batch_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: Optional[List[str]] = None,
    include_model: bool | None = None,
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
    batch_errors_file: str | None = None,
) -> Path:
    """Merge downloaded batch responses into the original records."""
    processor = BatchProcessor(config, output_dir / "batch")
    enriched_records = processor.process_batch_results(
        batch_results_file,
        records,
        variants=variants or DEFAULT_VARIANTS,
        batch_errors_file=batch_errors_file,
    )
    output_file = output_dir / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_file, enriched_records)
    logger.info("Merged batch results saved to: %s", output_file)
    return output_file


__all__ = ["merge_batch_results", "process_batch_mode"]
