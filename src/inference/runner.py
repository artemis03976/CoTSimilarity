"""Experiment runners that convert model continuations into project JSONL records."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .common import (
    VARIANTS,
    check_answer_strict,
    ground_truth_for,
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
) -> tuple[Path, Path]:
    """Run exactly one deterministic continuation for every selected triplet variant."""

    if not records:
        raise ValueError("No evaluation records selected")
    validation_config = validation_config or CoTValidationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_records.jsonl"
    qc_path = output_dir / "generation_qc.json"
    fully_correct = 0
    total_generations = 0
    valid_generations = 0
    rejection_reasons: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        progress = tqdm(records, desc=f"Greedy inference ({generator.baseline})")
        for record in progress:
            problem_id = normalise_problem_id(record.get("problem_id"))
            results: dict[str, Any] = {}
            all_correct = True
            for variant in VARIANTS:
                variant_data = record[variant]
                problem = variant_data["problem"]
                ground_truth = ground_truth_for(variant_data)
                generated = generator.generate(problem)
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
                    )
                    if validation.valid
                    else None
                )
                correct = bool(validation.valid and answer_correct)
                total_generations += 1
                valid_generations += int(validation.valid)
                rejection_reasons.update(validation.reasons)
                sample: dict[str, Any] = {
                    "response": generated.text,
                    "correct": correct,
                    "answer_correct": answer_correct,
                    "valid": validation.valid,
                    "finish_reason": generated.finish_reason,
                    "validation_errors": list(validation.reasons),
                }
                if save_token_ids:
                    sample["token_ids"] = list(generated.token_ids)
                result: dict[str, Any] = {
                    "problem": problem,
                    "ground_truth": ground_truth,
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
            import json

            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    write_json(
        qc_path,
        {
            "status": "completed",
            "mode": "greedy",
            "baseline": generator.baseline,
            "records": len(records),
            "generations": total_generations,
            "valid_generations": valid_generations,
            "invalid_generations": total_generations - valid_generations,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "fully_correct_triplets": fully_correct,
            "run": run_metadata or {},
            "validation": {
                "require_boxed_answer": validation_config.require_boxed_answer,
                "min_characters": validation_config.min_characters,
                "repeat_ngram_size": validation_config.repeat_ngram_size,
                "max_ngram_repeats": validation_config.max_ngram_repeats,
            },
        },
    )
    return output_path, qc_path
