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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .llm.api_client import LLMClient
from .llm.config import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_VARIANTS = ["original", "simple", "hard"]
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 5


@dataclass(frozen=True)
class _AnalysisTask:
    record_index: int
    variant: str
    sample_index: int
    problem: str
    steps: List[Dict]


@dataclass(frozen=True)
class _AnalysisResult:
    task: _AnalysisTask
    dag: Optional[List[Dict]]
    error: Optional[str]
    processing_time_ms: int


def _build_tasks(records: List[Dict], variants: List[str]) -> List[_AnalysisTask]:
    tasks: List[_AnalysisTask] = []
    for record_index, record in enumerate(records):
        for variant in variants:
            entry = record.get(variant)
            if not entry:
                continue
            samples = entry.get("samples", [])
            for sample_index, sample in enumerate(samples):
                steps = sample.get("steps")
                if steps:
                    tasks.append(
                        _AnalysisTask(
                            record_index=record_index,
                            variant=variant,
                            sample_index=sample_index,
                            problem=entry.get("problem", ""),
                            steps=steps,
                        )
                    )
    return tasks


def _analyze_task(
    task: _AnalysisTask,
    config: LLMConfig,
    max_retries: int,
) -> _AnalysisResult:
    """Analyze one chain, retrying failed results up to ``max_retries`` times.

    ``LLMClient`` already retries provider-specific API errors.  Limiting that
    client to one attempt here prevents nested retry budgets while this outer
    loop also handles parse, validation, and unexpected failures.
    """
    client = LLMClient(replace(config, max_retries=1))
    last_error: Optional[str] = None
    started = time.perf_counter()

    for retry_index in range(max_retries + 1):
        try:
            dag, error = client.analyze_reasoning_chain(task.problem, task.steps)
        except Exception as exc:  # pragma: no cover - defensive for providers
            dag, error = None, f"Unexpected error: {exc}"

        if error is None and dag is not None:
            return _AnalysisResult(
                task,
                dag,
                None,
                int((time.perf_counter() - started) * 1000),
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

    return _AnalysisResult(
        task,
        None,
        last_error,
        int((time.perf_counter() - started) * 1000),
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

    selected_variants = list(variants or DEFAULT_VARIANTS)
    output_file = output_dir / "normal" / "analyzed_records.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "logs" / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    error_log.parent.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(records, selected_variants)
    enriched_records = copy.deepcopy(records)
    total_processed = 0
    total_failed = 0
    results: List[_AnalysisResult] = []

    logger.info(
        "Processing %s DAG chains with concurrency=%s and max_retries=%s",
        len(tasks),
        concurrency,
        max_retries,
    )
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="dag") as executor:
        # executor.map preserves task order, which makes output deterministic.
        results = list(
            executor.map(
                lambda task: _analyze_task(task, config, max_retries),
                tasks,
            )
        )

    with open(error_log, "w", encoding="utf-8") as ferr:
        for result in results:
            task = result.task
            sample = enriched_records[task.record_index][task.variant]["samples"][task.sample_index]
            if result.error is None:
                sample["dag_analysis"] = result.dag
                sample["dag_metadata"] = {
                    "analyzed_at": datetime.now().isoformat(),
                    "model": config.model,
                    "processing_time_ms": result.processing_time_ms,
                }
                total_processed += 1
            else:
                sample["dag_analysis"] = None
                sample["dag_error"] = result.error
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

    with open(output_file, "w", encoding="utf-8") as fout:
        for record in enriched_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Processing complete: %s succeeded, %s failed", total_processed, total_failed)
    logger.info("Results saved to: %s", output_file)
    logger.info("Error log: %s", error_log)
    return output_file


__all__ = ["DEFAULT_CONCURRENCY", "DEFAULT_MAX_RETRIES", "process_normal_mode"]
