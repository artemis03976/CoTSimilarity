#!/usr/bin/env python3
"""Generate a validated multi-path CoT pool with batched vLLM inference.

By default, simple and hard perturbations receive 50 sampled trajectories;
the original problem receives one greedy reference trajectory.  Use
``--sampled-variants`` to change this contract.  Invalid sampled trajectories
remain in ``raw_generations.jsonl`` and are replaced with bounded resampling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference.common import VARIANTS, read_math_records, run_evaluator_self_test
from inference.multiple import MultipleSamplingConfig, VLLMMultipleSampler
from inference.quality import CoTValidationConfig


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-path", "--data_path", default="data/math_paired.jsonl")
    parser.add_argument("--output-dir", "--output_dir", default="output/multiple")
    parser.add_argument("--problem-id", "--id", type=int, default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--sampled-variants",
        nargs="+",
        choices=VARIANTS,
        default=["simple", "hard"],
    )
    parser.add_argument("--samples-per-problem", "--n", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=4096)
    parser.add_argument("--max-attempt-multiplier", type=int, default=3)
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--save-token-ids-in-records", action="store_true")
    parser.add_argument(
        "--require-valid-greedy-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail after writing diagnostics when paths are missing or anchors are invalid. "
            "Use --no-strict-quality only for smoke/debug runs."
        ),
    )

    parser.add_argument("--min-response-characters", type=int, default=20)
    parser.add_argument(
        "--repeat-ngram-size",
        "--max-repeat-period",
        type=int,
        default=64,
        help="Maximum consecutive loop period in tokens; distant phrase reuse is allowed",
    )
    parser.add_argument("--max-ngram-repeats", type=int, default=5)
    parser.add_argument(
        "--min-repeat-span-tokens",
        type=int,
        default=64,
        help="Minimum contiguous token span required before rejecting a periodic loop",
    )
    parser.add_argument(
        "--require-boxed-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_evaluator_self_test()
    records = read_math_records(
        Path(args.data_path),
        problem_id=args.problem_id,
        limit=args.num,
    )
    if not records:
        raise ValueError("No records selected for multi-path inference")
    config_kwargs = {
        "model_name": args.model,
        "sampled_variants": tuple(args.sampled_variants),
        "samples_per_problem": args.samples_per_problem,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "max_attempt_multiplier": args.max_attempt_multiplier,
        "request_batch_size": args.request_batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "require_valid_greedy_anchors": args.require_valid_greedy_anchors,
        "strict_quality": args.strict_quality,
    }
    if args.system_prompt is not None:
        config_kwargs["system_prompt"] = args.system_prompt
    sampling_config = MultipleSamplingConfig(**config_kwargs)
    validation_config = CoTValidationConfig(
        require_boxed_answer=args.require_boxed_answer,
        min_characters=args.min_response_characters,
        repeat_ngram_size=args.repeat_ngram_size,
        max_ngram_repeats=args.max_ngram_repeats,
        min_repeat_span_tokens=args.min_repeat_span_tokens,
    )
    print("Loading vLLM model...")
    sampler = None
    try:
        sampler = VLLMMultipleSampler(sampling_config, validation_config)
        output_path, raw_path, qc_path = sampler.collect(
            records,
            Path(args.output_dir),
            save_token_ids_in_records=args.save_token_ids_in_records,
        )
    finally:
        if sampler is not None:
            sampler.close()
    print(f"Accepted trajectories: {output_path}")
    print(f"Raw generations: {raw_path}")
    print(f"Generation QC: {qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
