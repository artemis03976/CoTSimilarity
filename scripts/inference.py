#!/usr/bin/env python3
"""Run one deterministic greedy CoT per MATH-Perturb variant.

The same entrypoint supports the frozen base model, DSPR, static prompt tuning
(SPT), and PEFT LoRA adapters.  It intentionally exposes no stochastic
decoding mode: each selected problem receives exactly one greedy continuation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference.common import VARIANTS, read_math_records, run_evaluator_self_test
from inference.greedy import BACKENDS, BASELINES, GreedyModelConfig, load_greedy_generator
from inference.quality import CoTValidationConfig
from inference.runner import run_greedy_evaluation


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES, default="base")
    parser.add_argument("--backend", choices=BACKENDS, default="transformers")
    parser.add_argument("--model-name", "--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="DSPR/SPT checkpoint file or LoRA adapter directory; omitted for base",
    )
    parser.add_argument("--data-path", "--data_path", default="data/canonical_math_paired.jsonl")
    parser.add_argument("--output-dir", "--output_dir", default="output/inference")
    parser.add_argument("--problem-id", "--id", type=int, default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens", "--max_new_tokens", type=int, default=None,
        help="Output token limit; default: 2048 for vLLM, 4096 for Transformers",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--save-token-ids", action="store_true")

    vllm_group = parser.add_argument_group("Single-GPU vLLM base inference")
    vllm_group.add_argument(
        "--batch-size", type=int, default=128,
        help="Problem triplets submitted per batch (3 prompts each); vLLM only",
    )
    vllm_group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    vllm_group.add_argument(
        "--max-model-len", type=int, default=None,
        help="Total prompt + output budget; default: max(4096, longest prompt + max-new-tokens)",
    )
    vllm_group.add_argument("--max-num-seqs", type=int, default=64)
    vllm_group.add_argument("--max-num-batched-tokens", type=int, default=8192)
    vllm_group.add_argument(
        "--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True,
    )

    # DSPR/SPT architecture settings must match the training checkpoint.
    parser.add_argument("--context-layer-idx", "--context_layer_idx", type=int, default=15)
    parser.add_argument("--prefix-length", "--prefix_length", type=int, default=50)
    parser.add_argument(
        "--router-intermediate-dim",
        "--router_intermediate_dim",
        type=int,
        default=256,
    )
    parser.add_argument("--router-dropout", "--router_dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", "--max_seq_length", type=int, default=4096)
    parser.add_argument("--forced-alpha", type=float, default=None)

    # Mechanical trajectory validation.  Invalid greedy outputs remain in the
    # evaluation file and count as model outputs; they are never resampled.
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

    # Compatibility with old K-fold orchestration commands.  Non-greedy values
    # are rejected instead of silently changing the evaluation protocol.
    parser.add_argument("--temperature", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.backend == "vllm" else 4096
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.baseline == "base" and args.checkpoint is not None:
        raise ValueError("--checkpoint must be omitted for --baseline base")
    if args.baseline != "base" and args.checkpoint is None:
        raise ValueError(f"--checkpoint is required for --baseline {args.baseline}")
    if args.temperature != 0.0 or args.top_p != 1.0 or args.n != 1:
        raise ValueError(
            "scripts/inference.py is deterministic-only: require temperature=0, top_p=1, n=1"
        )
    if args.max_new_tokens < 1 or args.max_seq_length < 1:
        raise ValueError("token limits must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    run_evaluator_self_test()
    records = read_math_records(
        Path(args.data_path),
        problem_id=args.problem_id,
        limit=args.num,
    )
    if not records:
        raise ValueError("No evaluation records selected")
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint is not None else None
    model_config = GreedyModelConfig(
        baseline=args.baseline,
        model_name=args.model_name,
        checkpoint=checkpoint,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
        context_layer_idx=args.context_layer_idx,
        prefix_length=args.prefix_length,
        router_intermediate_dim=args.router_intermediate_dim,
        router_dropout=args.router_dropout,
        max_seq_length=args.max_seq_length,
        forced_alpha=args.forced_alpha,
        backend=args.backend,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
    validation_config = CoTValidationConfig(
        require_boxed_answer=args.require_boxed_answer,
        min_characters=args.min_response_characters,
        repeat_ngram_size=args.repeat_ngram_size,
        max_ngram_repeats=args.max_ngram_repeats,
        min_repeat_span_tokens=args.min_repeat_span_tokens,
    )
    print(f"Loading {args.baseline} model with {args.backend}...")
    generator = load_greedy_generator(
        model_config,
        problems=[record[variant]["problem"] for record in records for variant in VARIANTS]
        if args.backend == "vllm" else None,
    )
    run_metadata = {
        "backend": args.backend,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "device": args.device,
        "dtype": args.dtype,
        "attention_implementation": (
            args.attention_implementation if args.backend == "transformers" else "auto"
        ),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "system_prompt": args.system_prompt,
        "context_layer_idx": args.context_layer_idx,
        "prefix_length": args.prefix_length,
        "router_intermediate_dim": args.router_intermediate_dim,
        "router_dropout": args.router_dropout,
        "max_seq_length": args.max_seq_length,
        "forced_alpha": args.forced_alpha,
    }
    try:
        output_path, qc_path = run_greedy_evaluation(
            generator,
            records,
            Path(args.output_dir),
            validation_config,
            save_token_ids=args.save_token_ids,
            run_metadata=run_metadata,
            batch_size=args.batch_size,
        )
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    print(f"Predictions: {output_path}")
    print(f"Generation QC: {qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
