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
        variants: List[str] = None
    ) -> str:
        """Prepare batch request file in JSONL format.

        All variants are written into a single file, distinguished by custom_id.

        Args:
            records: List of segmented records
            variants: Which variants to process (default: original/simple/hard)

        Returns:
            Path to batch request file
        """
        if variants is None:
            variants = ["original", "simple", "hard"]

        batch_file = self.output_dir / "batch_requests.jsonl"

        count = 0
        with open(batch_file, "w", encoding="utf-8") as f:
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

                        batch_request = {
                            "custom_id": f"{record['problem_id']}_{variant}_{sample_idx}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.config.model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "temperature": self.config.temperature,
                                "top_p": self.config.top_p,
                                "max_tokens": self.config.max_tokens
                            }
                        }

                        f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
                        count += 1

        logger.info(f"Prepared {count} batch requests across {variants}: {batch_file}")
        return str(batch_file)

    def process_batch_results(
        self,
        batch_results_file: str,
        original_records: List[Dict]
    ) -> List[Dict]:
        """Process batch results and merge with original records.

        Args:
            batch_results_file: Path to batch results JSONL
            original_records: Original segmented records

        Returns:
            Records with DAG analysis added
        """
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

            for variant in ["original", "simple", "hard"]:
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
