"""Normal-mode DAG analysis.

Normal mode sends one request per reasoning chain.  Requests are independent,
so they can be executed concurrently while records are still written in their
original order.  A retry budget is applied to each chain independently; one
failed chain therefore does not cancel the rest of the run.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .client import LLMClient, RateLimiter
from ..config import LLMConfig
from utils.artifacts import file_identity, fingerprint, response_hash
from utils.io import read_manifest, write_json, write_jsonl
from ..records import sample_key
from .parsing import validate_dag
from ..schemas import VARIANTS

logger = logging.getLogger(__name__)

DEFAULT_VARIANTS = list(VARIANTS)
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 5


@dataclass(frozen=True)
class _AnalysisTask:
    record_index: int
    variant: str
    sample_index: int
    problem: str
    steps: List[Dict]
    sample_id: str
    response_hash: Optional[str]


@dataclass(frozen=True)
class _AnalysisResult:
    task: _AnalysisTask
    dag: Optional[List[Dict]]
    error: Optional[str]
    processing_time_ms: int
    raw_responses: List[Dict]


def _build_tasks(records: List[Dict], variants: List[str]) -> List[_AnalysisTask]:
    tasks: List[_AnalysisTask] = []
    seen = set()
    for record_index, record in enumerate(records):
        for variant in variants:
            entry = record.get(variant) or {}
            for index, sample in enumerate(entry.get("samples", [])):
                key = sample_key(record["problem_id"], variant, index, sample)
                if key in seen:
                    raise ValueError(f"Duplicate sample ID: {key}")
                seen.add(key)
                if sample.get("steps"):
                    tasks.append(_AnalysisTask(
                        record_index, variant, index, entry.get("problem", ""),
                        sample["steps"], str(key),
                        response_hash(sample["response"]) if "response" in sample else None,
                    ))
    return tasks


def _analyze_task(
    task: _AnalysisTask,
    config: LLMConfig,
    max_retries: int,
    rate_limiter: Optional[RateLimiter] = None,
) -> _AnalysisResult:
    """Analyze one chain, retrying failed results up to ``max_retries`` times.

    ``LLMClient`` already retries provider-specific API errors.  Limiting that
    client to one attempt here prevents nested retry budgets while this outer
    loop also handles parse, validation, and unexpected failures.
    """
    client = LLMClient(replace(config, max_retries=1), rate_limiter=rate_limiter)
    last_error: Optional[str] = None
    raw_responses: List[Dict] = []
    started = time.perf_counter()

    for retry_index in range(max_retries + 1):
        try:
            dag, error = client.analyze_reasoning_chain(task.problem, task.steps)
        except Exception as exc:  # pragma: no cover - defensive for providers
            dag, error = None, f"Unexpected error: {exc}"
        for api_attempt, record in enumerate(client.take_raw_response_records()):
            raw_responses.append({
                "sample_id": task.sample_id,
                "outer_retry": retry_index,
                "api_attempt": record.get("api_attempt", api_attempt),
                "timestamp": record.get("timestamp", datetime.now().isoformat()),
                "response": record.get("response"),
                "parse_error": record.get("parse_error"),
                "provider_error": record.get("provider_error"),
            })

        if error is None and dag is not None:
            return _AnalysisResult(
                task,
                dag,
                None,
                int((time.perf_counter() - started) * 1000),
                raw_responses,
            )

        last_error = error or "LLM returned no DAG analysis"
        if retry_index < max_retries:
            logger.warning(
                "Retrying %s sample %s (retry %s/%s): %s",
                task.variant,
                task.sample_index,
                retry_index + 1,
                max_retries,
                last_error,
            )
            time.sleep(config.retry_delay)

    return _AnalysisResult(
        task,
        None,
        last_error,
        int((time.perf_counter() - started) * 1000),
        raw_responses,
    )


def process_normal_mode(
    records: List[Dict],
    config: LLMConfig,
    output_dir: Path,
    variants: Optional[List[str]] = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Analyze records concurrently and write the normal-mode JSONL output.

    ``max_retries`` counts retries after the initial request, so the default
    permits at most six provider calls for a single chain.  The output schema
    remains compatible with the previous serial implementation.
    """
    if max_workers is not None:
        concurrency = max_workers
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if not config.api_key:
        raise ValueError("API key is required for normal-mode LLM requests")

    selected_variants = list(variants or DEFAULT_VARIANTS)
    output_file = output_dir / "normal" / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_response_file = output_file.parent / "raw_responses.jsonl"
    error_log = output_dir / "logs" / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    error_log.parent.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(records, selected_variants)
    enriched_records = copy.deepcopy(records)
    total_processed = 0
    total_failed = 0
    checkpoint_file = output_file.parent / "checkpoints.jsonl"
    manifest_file = output_file.parent / "analysis.config.json"
    inputs = sorted([
        {"sample_id": task.sample_id, "problem": task.problem, "steps": task.steps, "response_hash": task.response_hash}
        for task in tasks
    ], key=lambda value: value["sample_id"])
    settings = {
        "inputs": fingerprint(inputs),
        "provider": config.provider, "model": config.model, "base_url": config.base_url,
        "temperature": config.temperature, "top_p": config.top_p, "max_tokens": config.max_tokens,
        "prompt": file_identity(Path(__file__).with_name("prompts.py"))["sha256"],
        "parser": file_identity(Path(__file__).with_name("parsing.py"))["sha256"],
    }
    signature = fingerprint(settings)
    restored = {}
    tasks_by_id = {task.sample_id: task for task in tasks}
    if resume and checkpoint_file.is_file():
        if read_manifest(manifest_file).get("signature") != signature:
            raise ValueError("DAG checkpoint inputs or annotation settings changed; run without --resume")
        with checkpoint_file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    if value.get("signature") == signature and value.get("error") is None:
                        task = tasks_by_id[value["sample_id"]]
                        validate_dag(value["dag"], task.steps)
                        if not isinstance(value["metadata"], dict):
                            raise ValueError("Missing checkpoint metadata")
                        restored[value["sample_id"]] = value
                except (ValueError, KeyError, TypeError):
                    logger.warning("Ignoring an incomplete DAG checkpoint entry")
        write_jsonl(checkpoint_file, restored.values())
    write_json(manifest_file, {"signature": signature, **settings})

    def apply_result(task, value):
        sample = enriched_records[task.record_index][task.variant]["samples"][task.sample_index]
        sample["sample_id"] = task.sample_id
        sample["dag_analysis"] = value["dag"]
        sample.pop("dag_error", None)
        sample.pop("dag_metadata", None)
        if value.get("error") is None:
            sample["dag_metadata"] = value["metadata"]
        else:
            sample["dag_error"] = value["error"]

    pending = []
    for task in tasks:
        if task.sample_id in restored:
            apply_result(task, restored[task.sample_id])
            total_processed += 1
        else:
            pending.append(task)

    logger.info(
        "Processing %s DAG chains with concurrency=%s and max_retries=%s",
        len(tasks),
        concurrency,
        max_retries,
    )
    limiter = RateLimiter(config.requests_per_minute)
    with (
        checkpoint_file.open("a" if resume else "w", encoding="utf-8") as checkpoint,
        raw_response_file.open("a" if resume else "w", encoding="utf-8") as raw_responses,
        error_log.open("w", encoding="utf-8") as ferr,
        ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="dag") as executor,
    ):
        futures = [executor.submit(_analyze_task, task, config, max_retries, limiter) for task in pending]
        for future in as_completed(futures):
            result = future.result()
            task = result.task
            value = {
                "signature": signature,
                "sample_id": task.sample_id,
                "dag": result.dag,
                "error": result.error,
                "metadata": {
                    "analyzed_at": datetime.now().isoformat(), "model": config.model,
                    "processing_time_ms": result.processing_time_ms,
                },
            }
            for record in result.raw_responses:
                raw_responses.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            raw_responses.flush()
            checkpoint.write(json.dumps(value, ensure_ascii=False) + "\n")
            checkpoint.flush()
            apply_result(task, value)
            if result.error is None:
                total_processed += 1
            else:
                total_failed += 1
                ferr.write(
                    json.dumps(
                        {
                            "problem_id": enriched_records[task.record_index]["problem_id"],
                            "variant": task.variant,
                            "sample_idx": task.sample_index,
                            "error": result.error,
                            "timestamp": datetime.now().isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write_jsonl(output_file, enriched_records)
    write_jsonl(output_dir / "analyzed_records.jsonl", enriched_records)
    write_json(manifest_file, {"signature": signature, **settings, "succeeded": total_processed, "failed": total_failed})

    logger.info("Processing complete: %s succeeded, %s failed", total_processed, total_failed)
    logger.info("Results saved to: %s", output_file)
    logger.info("Raw LLM responses: %s", raw_response_file)
    logger.info("Error log: %s", error_log)
    return output_file


__all__ = ["DEFAULT_CONCURRENCY", "DEFAULT_MAX_RETRIES", "process_normal_mode"]
