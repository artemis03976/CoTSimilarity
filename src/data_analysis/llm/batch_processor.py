"""Batch inference processor for cost-optimized processing."""

import json
import logging
from pathlib import Path
from typing import List, Dict
from .config import LLMConfig
from .prompt_template import build_prompt

logger = logging.getLogger(__name__)


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
        with open(batch_file, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                for variant in variants:
                    if variant not in record:
                        continue

                    entry = record[variant]
                    if "samples" not in entry:
                        continue

                    for sample_idx, sample in enumerate(entry["samples"]):
                        if "steps" not in sample or not sample["steps"]:
                            continue

                        system_prompt, user_prompt = build_prompt(
                            entry["problem"],
                            sample["steps"]
                        )

                        body = {}
                        if include_model:
                            body["model"] = self.config.model
                        body.update({
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": self.config.max_tokens,
                            "top_p": self.config.top_p,
                            "temperature": self.config.temperature,
                        })
                        if enable_thinking is not None:
                            body["enable_thinking"] = enable_thinking

                        # Keep this aligned with the provider's documented
                        # JSONL shape: one custom_id and one request body per
                        # line.  Endpoint/method metadata is selected by the
                        # batch service and is intentionally omitted.
                        batch_request = {
                            "custom_id": f"{record['problem_id']}_{variant}_{sample_idx}",
                        }
                        if method is not None:
                            batch_request["method"] = method
                            batch_request["url"] = url
                        batch_request["body"] = body

                        f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
                        count += 1

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

        # Load batch results
        results_map = {}
        with open(batch_results_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                response_text = result["response"]["body"]["choices"][0]["message"]["content"]
                results_map[custom_id] = response_text

        # Merge with original records
        enriched_records = []
        for record in original_records:
            enriched = record.copy()

            for variant in variants:
                if variant not in enriched:
                    continue

                samples = enriched[variant].get("samples", [])
                for sample_idx, sample in enumerate(samples):
                    custom_id = f"{record['problem_id']}_{variant}_{sample_idx}"
                    if custom_id in results_map:
                        response_text = results_map[custom_id]
                        try:
                            text = response_text.strip()
                            if text.startswith("```json"):
                                text = text[7:]
                            elif text.startswith("```"):
                                text = text[3:]
                            if text.endswith("```"):
                                text = text[:-3]
                            text = text.strip()

                            dag = json.loads(text)
                            sample["dag_analysis"] = dag
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse DAG for {custom_id}")
                            sample["dag_analysis"] = None
                            sample["dag_error"] = "JSON parse error"

            enriched_records.append(enriched)

        return enriched_records
