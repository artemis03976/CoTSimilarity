"""Experiment runners that convert model continuations into project JSONL records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm import tqdm

from .common import (
    VARIANTS,
    check_answer_strict,
    ground_truth_for,
    ground_truth_type_for,
    normalise_problem_id,
    write_json,
)
from .greedy import GreedyGenerator
from .quality import CoTValidationConfig, validate_cot


def run_greedy_evaluation(
    generator: GreedyGenerator,
    records: list[dict[str, Any]],
    output_dir: Path,
    validation_config: CoTValidationConfig | None = None,
    save_token_ids: bool = False,
    run_metadata: dict[str, Any] | None = None,
    batch_size: int = 128,
) -> tuple[Path, Path]:
    """Run exactly one deterministic continuation for every selected triplet variant."""

    if not records:
        raise ValueError("No evaluation records selected")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validation_config = validation_config or CoTValidationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_records.jsonl"
    qc_path = output_dir / "generation_qc.json"
    fully_correct = 0
    total_generations = 0
    valid_generations = 0
    ungraded_generations = 0
    rejection_reasons: Counter[str] = Counter()
    generate_batch = getattr(generator, "generate_batch", None)
    batched = callable(generate_batch)
    effective_batch_size = batch_size if batched else 1
    batch_results = []
    generated_tokens = 0
    generation_seconds = 0.0
    evaluation_seconds = 0.0
    started = perf_counter()
    metadata = {
        **(run_metadata or {}),
        **getattr(generator, "run_metadata", {}),
        "batch_size": effective_batch_size,
    }

    def write_qc(status: str, completed_records: int) -> None:
        write_json(
            qc_path,
            {
                "status": status,
                "mode": "greedy",
                "baseline": generator.baseline,
                "records": completed_records,
                "expected_records": len(records),
                "generations": total_generations,
                "valid_generations": valid_generations,
                "invalid_generations": total_generations - valid_generations,
                "ungraded_generations": ungraded_generations,
                "rejection_reasons": dict(sorted(rejection_reasons.items())),
                "fully_correct_triplets": fully_correct,
                "run": metadata,
                "performance": {
                    "elapsed_seconds": perf_counter() - started,
                    "generation_seconds": generation_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "generated_tokens": generated_tokens,
                    "generation_tokens_per_second": (
                        generated_tokens / generation_seconds if generation_seconds else 0.0
                    ),
                },
                "validation": {
                    "require_boxed_answer": validation_config.require_boxed_answer,
                    "min_characters": validation_config.min_characters,
                    "repeat_ngram_size": validation_config.repeat_ngram_size,
                    "max_ngram_repeats": validation_config.max_ngram_repeats,
                    "min_repeat_span_tokens": validation_config.min_repeat_span_tokens,
                },
            },
        )

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        write_qc("running", 0)
        progress = tqdm(records, desc=f"Greedy inference ({generator.baseline})")
        for record_index, record in enumerate(progress):
            if batched and record_index % effective_batch_size == 0:
                batch = records[record_index : record_index + effective_batch_size]
                problems = [record[variant]["problem"] for record in batch for variant in VARIANTS]
                generation_started = perf_counter()
                batch_results = generate_batch(problems)
                generation_seconds += perf_counter() - generation_started
                if len(batch_results) != len(problems):
                    raise RuntimeError("Batched generator did not return one result per prompt")
            problem_id = normalise_problem_id(record.get("problem_id"))
            results: dict[str, Any] = {}
            all_correct = True
            for variant_index, variant in enumerate(VARIANTS):
                variant_data = record[variant]
                problem = variant_data["problem"]
                ground_truth = ground_truth_for(variant_data)
                ground_truth_type = ground_truth_type_for(variant_data, variant)
                if batched:
                    offset = (record_index % effective_batch_size) * len(VARIANTS) + variant_index
                    generated = batch_results[offset]
                else:
                    generation_started = perf_counter()
                    generated = generator.generate(problem)
                    generation_seconds += perf_counter() - generation_started
                generated_tokens += len(generated.token_ids)
                evaluation_started = perf_counter()
                validation = validate_cot(
                    generated.text,
                    generated.finish_reason,
                    generated.token_ids,
                    validation_config,
                )
                answer_correct = (
                    check_answer_strict(
                        problem,
                        generated.text,
                        ground_truth,
                        variant,
                        ground_truth_type,
                    )
                    if validation.valid
                    else None
                )
                correct = bool(validation.valid and answer_correct)
                evaluation_status = (
                    "invalid_generation" if not validation.valid
                    else "unsupported_reference" if answer_correct is None
                    else "correct" if answer_correct else "incorrect"
                )
                ungraded_generations += int(validation.valid and answer_correct is None)
                evaluation_seconds += perf_counter() - evaluation_started
                total_generations += 1
                valid_generations += int(validation.valid)
                rejection_reasons.update(validation.reasons)
                sample: dict[str, Any] = {
                    "response": generated.text,
                    "correct": correct,
                    "answer_correct": answer_correct,
                    "answer_evaluation_status": evaluation_status,
                    "valid": validation.valid,
                    "finish_reason": generated.finish_reason,
                    "validation_errors": list(validation.reasons),
                }
                if save_token_ids:
                    sample["token_ids"] = list(generated.token_ids)
                result: dict[str, Any] = {
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "ground_truth_type": ground_truth_type,
                    "samples": [sample],
                }
                if "alpha" in generated.metadata:
                    result["alpha"] = float(generated.metadata["alpha"])
                results[variant] = result
                all_correct = all_correct and correct

            fully_correct += int(all_correct)
            progress.set_postfix(
                problem_id=problem_id,
                status="PASS" if all_correct else "FAIL",
            )
            payload = {
                "problem_id": problem_id,
                "type": record.get("type"),
                "level": record.get("level"),
                **results,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            if (record_index + 1) % effective_batch_size == 0 or record_index + 1 == len(records):
                write_qc("running", record_index + 1)

    write_qc("completed", len(records))
    print(
        f"Generated {generated_tokens} tokens in {generation_seconds:.1f}s "
        f"({generated_tokens / generation_seconds if generation_seconds else 0.0:.1f} tokens/s); "
        f"evaluation: {evaluation_seconds:.1f}s"
    )
    return output_path, qc_path
