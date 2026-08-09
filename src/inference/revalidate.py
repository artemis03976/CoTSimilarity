"""Offline revalidation and rematerialisation for multi-path generations.

The vLLM sampler keeps every attempt in ``raw_generations.jsonl``.  This
module lets us change mechanical validation thresholds without paying for a
second model run: attempts are revalidated, the first target number of valid
trajectories is selected, and fresh QC/record files are written.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .common import VARIANTS, check_answer_strict, normalise_problem_id, read_jsonl
from .quality import CoTValidationConfig, validate_cot


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def revalidate_multiple(
    raw_path: Path,
    records_path: Path,
    output_dir: Path,
    validation_config: CoTValidationConfig | None = None,
    samples_per_problem: int | None = None,
    sampled_variants: tuple[str, ...] | None = None,
    require_valid_greedy_anchors: bool | None = None,
) -> tuple[Path, Path, Path]:
    """Revalidate a raw multi-path run and write a clean materialisation.

    ``records_path`` supplies problem text, ground truth, and the stable
    record order; all candidate responses and token ids come from
    ``raw_path``.  Original (greedy) anchors are retained exactly once even
    when invalid, matching the online sampler's evaluation contract.  Sampled
    variants retain only revalidated-valid trajectories, capped at the target
    count.
    """

    config = validation_config or CoTValidationConfig()
    source_records = read_jsonl(records_path)
    metadata = {normalise_problem_id(row.get("problem_id")): row for row in source_records}
    if len(metadata) != len(source_records):
        raise ValueError(f"Duplicate problem_id values in {records_path}")

    source_qc = _load_optional_json(raw_path.parent / "generation_qc.json")
    source_sampling = source_qc.get("sampling", {})
    if samples_per_problem is None:
        samples_per_problem = int(source_sampling.get("samples_per_problem", 50))
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be positive")
    if sampled_variants is None:
        sampled_variants = tuple(source_sampling.get("sampled_variants", ("simple", "hard")))
    sampled = set(sampled_variants)
    if not sampled.issubset(set(VARIANTS)):
        raise ValueError(f"Unknown sampled variants: {sorted(sampled - set(VARIANTS))}")
    if require_valid_greedy_anchors is None:
        require_valid_greedy_anchors = bool(source_sampling.get("require_valid_greedy_anchors", True))

    # Correctness is independent of the mechanical quality threshold.  Reuse
    # the already computed labels for trajectories that were accepted by the
    # original run; only newly rescued trajectories need a fresh evaluator
    # call.  This keeps revalidation fast and avoids re-running a relatively
    # expensive symbolic answer parser tens of thousands of times.
    existing_samples: dict[tuple[int, str, int], dict[str, Any]] = {}
    for source in source_records:
        problem_id = normalise_problem_id(source.get("problem_id"))
        for variant in VARIANTS:
            for sample in source[variant].get("samples", []):
                attempt_index = sample.get("attempt_index")
                if attempt_index is not None:
                    existing_samples[(problem_id, variant, int(attempt_index))] = sample

    raw_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    revalidated_raw: list[dict[str, Any]] = []
    attempts = 0
    accepted = 0
    invalid = 0
    rejection_reasons: Counter[str] = Counter()
    by_variant: dict[str, Counter[str]] = {variant: Counter() for variant in VARIANTS}

    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {raw_path} line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Expected an object in {raw_path} line {line_number}")
            problem_id = normalise_problem_id(raw.get("problem_id"))
            variant = str(raw.get("variant"))
            if problem_id not in metadata:
                raise ValueError(f"Raw generation references unknown problem_id {problem_id}")
            if variant not in VARIANTS:
                raise ValueError(f"Raw generation has unknown variant {variant!r}")
            text = str(raw.get("response") or "")
            token_ids = tuple(int(token) for token in (raw.get("token_ids") or ()))
            validation = validate_cot(
                text,
                raw.get("finish_reason"),
                token_ids,
                config,
            )
            quality_accepted = validation.valid
            attempts += 1
            accepted += int(quality_accepted)
            by_variant[variant]["attempts"] += 1
            by_variant[variant]["accepted"] += int(quality_accepted)
            if not quality_accepted:
                invalid += 1
                rejection_reasons.update(validation.reasons)
                by_variant[variant]["invalid"] += 1

            updated = dict(raw)
            updated.update(
                {
                    "valid": quality_accepted,
                    "accepted": quality_accepted,
                    "selected_for_output": False,
                    "validation_errors": list(validation.reasons),
                }
            )
            raw_by_key[(problem_id, variant)].append(updated)
            revalidated_raw.append(updated)

    output_rows: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    invalid_anchors: list[dict[str, Any]] = []
    fully_correct = 0
    total_target = 0

    for source in source_records:
        problem_id = normalise_problem_id(source.get("problem_id"))
        source_variants = {variant: source[variant] for variant in VARIANTS}
        results: dict[str, Any] = {}
        all_correct = True
        for variant in VARIANTS:
            target = samples_per_problem if variant in sampled else 1
            total_target += target
            candidates = raw_by_key.get((problem_id, variant), [])
            selected: list[dict[str, Any]] = []
            for candidate in candidates:
                if variant not in sampled or candidate["valid"]:
                    if variant in sampled and len(selected) >= target:
                        continue
                    candidate["selected_for_output"] = True
                    selected.append(candidate)
                    if len(selected) >= target:
                        break

            metadata_variant = source_variants[variant]
            samples: list[dict[str, Any]] = []
            variant_any_correct = False
            for candidate in selected:
                validation_errors = list(candidate.get("validation_errors") or [])
                valid = bool(candidate["valid"])
                answer_correct: bool | None = None
                correct = False
                if valid:
                    previous = existing_samples.get(
                        (problem_id, variant, int(candidate.get("attempt_index", -1)))
                    )
                    if previous is not None and previous.get("valid"):
                        answer_correct = previous.get("answer_correct")
                        correct = bool(previous.get("correct"))
                    else:
                        answer_correct = check_answer_strict(
                            metadata_variant["problem"],
                            candidate["response"],
                            metadata_variant.get(
                                "ground_truth",
                                metadata_variant.get(
                                    "solution", metadata_variant.get("answer")
                                ),
                            ),
                            variant,
                        )
                        correct = bool(answer_correct)
                sample: dict[str, Any] = {
                    "response": candidate["response"],
                    "correct": correct,
                    "answer_correct": answer_correct,
                    "valid": valid,
                    "finish_reason": candidate.get("finish_reason"),
                    "validation_errors": validation_errors,
                    "attempt_index": candidate.get("attempt_index"),
                    "request_seed": candidate.get("request_seed"),
                }
                if any(
                    "token_ids" in previous_sample
                    for previous_sample in metadata_variant.get("samples", [])
                ):
                    sample["token_ids"] = list(candidate.get("token_ids") or [])
                samples.append(sample)
                variant_any_correct = variant_any_correct or correct

            all_correct = all_correct and variant_any_correct

            results[variant] = {
                "problem": metadata_variant["problem"],
                "ground_truth": metadata_variant.get(
                    "ground_truth", metadata_variant.get("solution", metadata_variant.get("answer"))
                ),
                "sampling_mode": "sample" if variant in sampled else "greedy",
                "samples": samples,
            }

            if len(selected) < target:
                reasons = Counter()
                for candidate in candidates:
                    if not candidate["valid"]:
                        reasons.update(candidate.get("validation_errors") or [])
                shortfalls.append(
                    {
                        "problem_id": problem_id,
                        "variant": variant,
                        "target": target,
                        "accepted": len(selected),
                        "attempts": len(candidates),
                        "rejection_reasons": dict(sorted(reasons.items())),
                    }
                )
            if variant not in sampled and candidates and not candidates[0]["valid"]:
                anchor_reasons = Counter(candidates[0].get("validation_errors") or [])
                invalid_anchors.append(
                    {
                        "problem_id": problem_id,
                        "variant": variant,
                        "rejection_reasons": dict(sorted(anchor_reasons.items())),
                    }
                )

        fully_correct += int(all_correct)
        output_rows.append(
            {
                "problem_id": problem_id,
                "type": source.get("type"),
                "level": source.get("level"),
                **results,
            }
        )

    for raw in revalidated_raw:
        raw["selected_for_output"] = False
    selected_keys = {
        (row["problem_id"], variant, sample.get("attempt_index"))
        for row in output_rows
        for variant in VARIANTS
        for sample in row[variant]["samples"]
    }
    for raw in revalidated_raw:
        if (raw["problem_id"], raw["variant"], raw.get("attempt_index")) in selected_keys:
            raw["selected_for_output"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_records.jsonl"
    raw_output_path = output_dir / "raw_generations.jsonl"
    qc_path = output_dir / "generation_qc.json"
    _write_jsonl(output_path, output_rows)
    _write_jsonl(raw_output_path, revalidated_raw)

    failed = bool(shortfalls) or (require_valid_greedy_anchors and bool(invalid_anchors))
    source_sampling = dict(source_sampling)
    source_sampling["sampled_variants"] = list(sampled_variants)
    source_sampling["samples_per_problem"] = samples_per_problem
    source_sampling["require_valid_greedy_anchors"] = require_valid_greedy_anchors
    qc_payload = {
        "status": "failed" if failed else "completed",
        "mode": "multiple_revalidated",
        "model_name": source_qc.get("model_name", source_sampling.get("model_name")),
        "records": len(source_records),
        "target_accepted_generations": total_target,
        "maximum_attempt_budget": source_qc.get("maximum_attempt_budget"),
        "attempts": attempts,
        "accepted": accepted,
        "invalid_raw_generations": invalid,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "by_variant": {variant: dict(by_variant[variant]) for variant in VARIANTS},
        "shortfalls": shortfalls,
        "invalid_greedy_anchors": invalid_anchors,
        "fully_correct_triplets": fully_correct,
        "sampling": source_sampling,
        "validation": asdict(config),
        "revalidated_from": str(raw_path),
    }
    qc_path.write_text(
        json.dumps(qc_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path, raw_output_path, qc_path
