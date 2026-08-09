"""Batched vLLM trajectory sampling with validation and bounded resampling."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .common import (
    DEFAULT_SYSTEM_PROMPT,
    VARIANTS,
    build_prompt,
    check_answer_strict,
    ground_truth_for,
    normalise_problem_id,
    write_json,
)
from .quality import CoTValidationConfig, ValidationResult, validate_cot


@dataclass(frozen=True)
class MultipleSamplingConfig:
    model_name: str
    sampled_variants: tuple[str, ...] = ("simple", "hard")
    samples_per_problem: int = 50
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    max_attempt_multiplier: int = 3
    request_batch_size: int = 32
    seed: int = 42
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    require_valid_greedy_anchors: bool = True
    strict_quality: bool = True

    def __post_init__(self) -> None:
        if not self.sampled_variants:
            raise ValueError("At least one sampled variant is required")
        unknown = set(self.sampled_variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"Unknown sampled variants: {sorted(unknown)}")
        if len(set(self.sampled_variants)) != len(self.sampled_variants):
            raise ValueError("sampled_variants contains duplicates")
        if self.samples_per_problem < 1:
            raise ValueError("samples_per_problem must be positive")
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive for multi-path sampling")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_tokens < 1 or self.max_attempt_multiplier < 1:
            raise ValueError("token and attempt limits must be positive")
        if self.request_batch_size < 1 or self.tensor_parallel_size < 1:
            raise ValueError("batch and tensor-parallel sizes must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


@dataclass(frozen=True)
class AcceptedTrajectory:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None
    request_seed: int
    attempt_index: int
    validation: ValidationResult


@dataclass
class SamplingTask:
    record_index: int
    problem_id: int
    variant: str
    problem: str
    ground_truth: Any
    prompt: str
    target: int
    resample_invalid: bool
    max_attempts: int
    attempts: int = 0
    request_round: int = 0
    trajectories: list[AcceptedTrajectory] = field(default_factory=list)
    invalid_anchor: bool = False
    validation_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def remaining(self) -> int:
        return max(0, self.target - len(self.trajectories))

    @property
    def can_request(self) -> bool:
        return self.remaining > 0 and self.attempts < self.max_attempts


def stable_request_seed(
    base_seed: int,
    model_name: str,
    problem_id: int,
    variant: str,
    request_round: int,
) -> int:
    material = f"{base_seed}\0{model_name}\0{problem_id}\0{variant}\0{request_round}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


class VLLMMultipleSampler:
    """Collect exactly n structurally valid trajectories for sampled variants."""

    def __init__(
        self,
        config: MultipleSamplingConfig,
        validation_config: CoTValidationConfig | None = None,
        llm: Any | None = None,
        sampling_params_class: Any | None = None,
    ) -> None:
        self.config = config
        self.validation_config = validation_config or CoTValidationConfig()
        if llm is None or sampling_params_class is None:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError("Multi-path inference requires vLLM") from exc
            model_kwargs: dict[str, Any] = {
                "model": config.model_name,
                "trust_remote_code": config.trust_remote_code,
                "dtype": config.dtype,
                "tensor_parallel_size": config.tensor_parallel_size,
                "gpu_memory_utilization": config.gpu_memory_utilization,
            }
            if config.max_model_len is not None:
                model_kwargs["max_model_len"] = config.max_model_len
            llm = LLM(**model_kwargs)
            sampling_params_class = SamplingParams
        self.llm = llm
        self.sampling_params_class = sampling_params_class
        self.tokenizer = llm.get_tokenizer()
        self.stop_token_ids = self._stop_token_ids()
        self._closed = False

    def close(self) -> None:
        """Best-effort shutdown for vLLM and its process group.

        vLLM may initialize distributed state even with one visible GPU.  This
        method is intentionally idempotent so callers can use it from a
        ``finally`` block on both success and quality-contract failures.
        """

        if self._closed:
            return
        self._closed = True
        engine = getattr(self.llm, "llm_engine", None)
        shutdown = getattr(self.llm, "shutdown", None)
        if not callable(shutdown) and engine is not None:
            shutdown = getattr(engine, "shutdown", None)
        if not callable(shutdown) and engine is not None:
            executor = getattr(engine, "model_executor", None)
            shutdown = getattr(executor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                # Do not mask the original generation/validation exception.
                pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    def _stop_token_ids(self) -> list[int]:
        ids: list[int] = []
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            ids.append(eos)
        elif eos is not None:
            ids.extend(int(value) for value in eos)
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk = self.tokenizer.unk_token_id
        if isinstance(im_end, int) and im_end >= 0 and im_end != unk:
            ids.append(im_end)
        return list(dict.fromkeys(ids))

    def _make_sampling_params(self, task: SamplingTask, draw_count: int, seed: int) -> Any:
        common: dict[str, Any] = {
            "n": draw_count,
            "max_tokens": self.config.max_tokens,
            "seed": seed,
            "skip_special_tokens": True,
        }
        if self.stop_token_ids:
            common["stop_token_ids"] = self.stop_token_ids
        if task.resample_invalid:
            common.update(
                {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                    "repetition_penalty": self.config.repetition_penalty,
                }
            )
        else:
            common.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
        return self.sampling_params_class(**common)

    def _build_tasks(self, records: list[dict[str, Any]]) -> list[SamplingTask]:
        sampled = set(self.config.sampled_variants)
        tasks: list[SamplingTask] = []
        for record_index, record in enumerate(records):
            problem_id = normalise_problem_id(record.get("problem_id"))
            for variant in VARIANTS:
                variant_data = record[variant]
                is_sampled = variant in sampled
                target = self.config.samples_per_problem if is_sampled else 1
                tasks.append(
                    SamplingTask(
                        record_index=record_index,
                        problem_id=problem_id,
                        variant=variant,
                        problem=variant_data["problem"],
                        ground_truth=ground_truth_for(variant_data),
                        prompt=build_prompt(
                            self.tokenizer,
                            variant_data["problem"],
                            self.config.system_prompt,
                        ),
                        target=target,
                        resample_invalid=is_sampled,
                        max_attempts=(
                            target * self.config.max_attempt_multiplier if is_sampled else 1
                        ),
                    )
                )
        return tasks

    def _process_request_output(
        self,
        task: SamplingTask,
        request_output: Any,
        requested: int,
        request_seed: int,
        request_round: int,
        raw_handle: Any,
        counters: dict[str, Any],
    ) -> int:
        completions = list(request_output.outputs)
        if len(completions) != requested:
            raise RuntimeError(
                f"vLLM returned {len(completions)} outputs for problem {task.problem_id} "
                f"{task.variant}; requested {requested}"
            )
        accepted_now = 0
        for completion in completions:
            task.attempts += 1
            attempt_index = task.attempts
            text = completion.text
            token_ids = tuple(int(token) for token in completion.token_ids)
            finish_reason = completion.finish_reason
            validation = validate_cot(
                text,
                finish_reason,
                token_ids,
                self.validation_config,
            )
            quality_accepted = validation.valid
            selected_for_output = quality_accepted if task.resample_invalid else True
            if selected_for_output:
                task.trajectories.append(
                    AcceptedTrajectory(
                        text=text,
                        token_ids=token_ids,
                        finish_reason=finish_reason,
                        request_seed=request_seed,
                        attempt_index=attempt_index,
                        validation=validation,
                    )
                )
                if quality_accepted:
                    accepted_now += 1
            if not task.resample_invalid and not validation.valid:
                task.invalid_anchor = True

            counters["attempts"] += 1
            counters["accepted"] += int(quality_accepted)
            counters["by_variant"][task.variant]["attempts"] += 1
            counters["by_variant"][task.variant]["accepted"] += int(quality_accepted)
            if not validation.valid:
                task.validation_reasons.update(validation.reasons)
                counters["invalid"] += 1
                counters["rejection_reasons"].update(validation.reasons)
                counters["by_variant"][task.variant]["invalid"] += 1
            raw_record = {
                "problem_id": task.problem_id,
                "variant": task.variant,
                "sampling_mode": "sample" if task.resample_invalid else "greedy",
                "attempt_index": attempt_index,
                "request_round": request_round,
                "request_seed": request_seed,
                "completion_index": int(completion.index),
                "response": text,
                "token_ids": list(token_ids),
                "finish_reason": finish_reason,
                "stop_reason": completion.stop_reason,
                "valid": validation.valid,
                "accepted": quality_accepted,
                "selected_for_output": selected_for_output,
                "validation_errors": list(validation.reasons),
            }
            raw_handle.write(json.dumps(raw_record, ensure_ascii=False, allow_nan=False) + "\n")
        raw_handle.flush()
        return accepted_now

    def collect(
        self,
        records: list[dict[str, Any]],
        output_dir: Path,
        save_token_ids_in_records: bool = False,
    ) -> tuple[Path, Path, Path]:
        """Generate, validate, evaluate, and persist a complete trajectory pool."""

        if not records:
            raise ValueError("No records selected for multi-path inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "all_records.jsonl"
        raw_path = output_dir / "raw_generations.jsonl"
        qc_path = output_dir / "generation_qc.json"
        tasks = self._build_tasks(records)
        counters: dict[str, Any] = {
            "attempts": 0,
            "accepted": 0,
            "invalid": 0,
            "rejection_reasons": Counter(),
            "by_variant": defaultdict(lambda: Counter()),
        }
        target_total = sum(task.target for task in tasks)
        attempt_budget = sum(task.max_attempts for task in tasks)
        progress = tqdm(total=target_total, desc="Valid trajectories", unit="path")

        with raw_path.open("w", encoding="utf-8", newline="\n") as raw_handle:
            while True:
                pending = [task for task in tasks if task.can_request]
                if not pending:
                    break
                for offset in range(0, len(pending), self.config.request_batch_size):
                    batch = pending[offset : offset + self.config.request_batch_size]
                    prompts: list[str] = []
                    params: list[Any] = []
                    request_details: list[tuple[SamplingTask, int, int, int]] = []
                    for task in batch:
                        draw_count = min(task.remaining, task.max_attempts - task.attempts)
                        request_round = task.request_round
                        request_seed = stable_request_seed(
                            self.config.seed,
                            self.config.model_name,
                            task.problem_id,
                            task.variant,
                            request_round,
                        )
                        prompts.append(task.prompt)
                        params.append(self._make_sampling_params(task, draw_count, request_seed))
                        request_details.append(
                            (task, draw_count, request_seed, request_round)
                        )
                        task.request_round += 1
                    request_outputs = self.llm.generate(
                        prompts,
                        sampling_params=params,
                        use_tqdm=False,
                    )
                    if len(request_outputs) != len(request_details):
                        raise RuntimeError(
                            f"vLLM returned {len(request_outputs)} requests for a batch of "
                            f"{len(request_details)}"
                        )
                    for request_output, detail in zip(request_outputs, request_details):
                        task, requested, request_seed, request_round = detail
                        accepted_now = self._process_request_output(
                            task,
                            request_output,
                            requested,
                            request_seed,
                            request_round,
                            raw_handle,
                            counters,
                        )
                        progress.update(accepted_now)
                        progress.set_postfix(
                            accepted=f"{counters['accepted']}/{target_total}",
                            attempts=f"{counters['attempts']}/{attempt_budget}",
                        )
        has_collection_shortfall = any(
            task.resample_invalid and task.remaining > 0 for task in tasks
        )
        has_invalid_anchor = any(
            not task.resample_invalid and task.invalid_anchor for task in tasks
        )
        progress.set_postfix(
            accepted=f"{counters['accepted']}/{target_total}",
            attempts=f"{counters['attempts']}/{attempt_budget}",
            status=(
                "quality_shortfall"
                if has_collection_shortfall or has_invalid_anchor
                else "complete"
            ),
        )
        progress.close()

        task_map = {(task.record_index, task.variant): task for task in tasks}
        fully_correct = 0
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record_index, record in enumerate(records):
                problem_id = normalise_problem_id(record.get("problem_id"))
                results: dict[str, Any] = {}
                all_correct = True
                for variant in VARIANTS:
                    task = task_map[(record_index, variant)]
                    samples: list[dict[str, Any]] = []
                    for trajectory in task.trajectories:
                        answer_correct = (
                            check_answer_strict(
                                task.problem,
                                trajectory.text,
                                task.ground_truth,
                                variant,
                            )
                            if trajectory.validation.valid
                            else None
                        )
                        correct = bool(trajectory.validation.valid and answer_correct)
                        sample: dict[str, Any] = {
                            "response": trajectory.text,
                            "correct": correct,
                            "answer_correct": answer_correct,
                            "valid": trajectory.validation.valid,
                            "finish_reason": trajectory.finish_reason,
                            "validation_errors": list(trajectory.validation.reasons),
                            "attempt_index": trajectory.attempt_index,
                            "request_seed": trajectory.request_seed,
                        }
                        if save_token_ids_in_records:
                            sample["token_ids"] = list(trajectory.token_ids)
                        samples.append(sample)
                    all_correct = all_correct and any(sample["correct"] for sample in samples)
                    results[variant] = {
                        "problem": task.problem,
                        "ground_truth": task.ground_truth,
                        "sampling_mode": "sample" if task.resample_invalid else "greedy",
                        "samples": samples,
                    }
                fully_correct += int(all_correct)
                payload = {
                    "problem_id": problem_id,
                    "type": record.get("type"),
                    "level": record.get("level"),
                    **results,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

        shortfalls = [
            {
                "problem_id": task.problem_id,
                "variant": task.variant,
                "target": task.target,
                "accepted": len(task.trajectories),
                "attempts": task.attempts,
                "rejection_reasons": dict(sorted(task.validation_reasons.items())),
            }
            for task in tasks
            if task.resample_invalid and task.remaining > 0
        ]
        invalid_anchors = [
            {
                "problem_id": task.problem_id,
                "variant": task.variant,
                "rejection_reasons": dict(sorted(task.validation_reasons.items())),
            }
            for task in tasks
            if not task.resample_invalid and task.invalid_anchor
        ]
        failed = bool(shortfalls) or (
            self.config.require_valid_greedy_anchors and bool(invalid_anchors)
        )
        qc_payload = {
            "status": "failed" if failed else "completed",
            "mode": "multiple",
            "model_name": self.config.model_name,
            "records": len(records),
            "target_accepted_generations": target_total,
            "maximum_attempt_budget": attempt_budget,
            "attempts": counters["attempts"],
            "accepted": counters["accepted"],
            "invalid_raw_generations": counters["invalid"],
            "rejection_reasons": dict(sorted(counters["rejection_reasons"].items())),
            "by_variant": {
                variant: dict(counters["by_variant"][variant]) for variant in VARIANTS
            },
            "shortfalls": shortfalls,
            "invalid_greedy_anchors": invalid_anchors,
            "fully_correct_triplets": fully_correct,
            "sampling": asdict(self.config),
            "validation": asdict(self.validation_config),
        }
        write_json(qc_path, qc_payload)
        if failed and self.config.strict_quality:
            raise RuntimeError(
                "Trajectory collection did not satisfy its quality contract; "
                f"shortfalls={len(shortfalls)}, invalid_greedy_anchors={len(invalid_anchors)}; "
                f"details={shortfalls[:3] + invalid_anchors[:3]}. "
                f"See {qc_path} and {raw_path}. "
                "Use --no-strict-quality only for smoke/debug runs; do not use partial "
                "outputs for formal data construction."
            )
        return output_path, raw_path, qc_path
