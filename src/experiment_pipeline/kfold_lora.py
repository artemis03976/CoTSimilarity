#!/usr/bin/env python3
"""Train parameter-matched Qwen LoRA adapters on the prepared five folds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

from .kfold_dspr import (
    DEFAULT_FOLDS,
    DEFAULT_MODEL_NAME,
    FoldSpec,
    parse_fold_list,
    parse_gpu_list,
    resolve_path,
    run_dynamic_schedule,
    utc_now,
    validate_inputs,
    write_json,
)
from .train_lora import DEFAULT_DSPR_BUDGET, DEFAULT_RANK, DEFAULT_LORA_ALPHA, DEFAULT_RUN_NAME


def build_train_command(
    train_script: Path,
    spec: FoldSpec,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(train_script),
        "lora",
        "--model_name",
        args.model_name,
        "--train_data_path",
        spec.train_data,
        "--val_data_path",
        spec.val_data,
        "--output_path",
        spec.output_path,
        "--rank",
        str(args.rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--target_modules",
        *args.target_modules,
        "--lora_dropout",
        str(args.lora_dropout),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--warmup_steps",
        str(args.warmup_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--max_seq_length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--logging_steps",
        str(args.logging_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--early_stopping_threshold",
        str(args.early_stopping_threshold),
        "--dtype",
        args.dtype,
        "--attention_implementation",
        args.attention_implementation,
        "--dspr_parameter_budget",
        str(args.dspr_parameter_budget),
        "--budget_tolerance",
        str(args.budget_tolerance),
    ]
    command.append(
        "--gradient_checkpointing" if args.gradient_checkpointing else "--no-gradient_checkpointing"
    )
    command.append(
        "--enforce_parameter_budget" if args.enforce_parameter_budget else "--no-enforce_parameter_budget"
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", default="output/qwen-2.5/kfold")
    parser.add_argument(
        "--output-root",
        default=f"checkpoints/{DEFAULT_RUN_NAME}",
    )
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=float, default=15)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dspr-parameter-budget", type=int, default=DEFAULT_DSPR_BUDGET)
    parser.add_argument("--budget-tolerance", type=float, default=0.10)
    parser.add_argument("--enforce-parameter-budget", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = parse_fold_list(args.folds)
    gpu_ids = parse_gpu_list(args.gpus)
    fold_root = resolve_path(REPO_ROOT, args.fold_root).resolve()
    output_root = resolve_path(REPO_ROOT, args.output_root).resolve()
    train_script = resolve_path(REPO_ROOT, args.train_script).resolve()
    validate_inputs(
        train_script,
        fold_root,
        output_root,
        folds,
        allow_existing=args.allow_existing or args.dry_run,
    )

    specs: list[FoldSpec] = []
    for index, fold in enumerate(folds):
        fold_dir = fold_root / f"fold_{fold}"
        fold_output = output_root / f"fold_{fold}"
        spec = FoldSpec(
            fold=fold,
            gpu=gpu_ids[index % len(gpu_ids)],
            train_data=str((fold_dir / "train.jsonl").resolve()),
            val_data=str((fold_dir / "val.jsonl").resolve()),
            output_path=str(fold_output.resolve()),
            log_path=str((fold_output / "train.log").resolve()),
            command=[],
        )
        spec.command = build_train_command(train_script, spec, args)
        specs.append(spec)

    print(f"Model: {args.model_name}")
    print(
        f"LoRA: r={args.rank}, alpha={args.lora_alpha}, "
        f"targets={args.target_modules}; seed={args.seed}"
    )
    print(f"Folds: {folds}; GPUs: {gpu_ids}; output: {output_root}")

    manifest = {
        "created_at": utc_now(),
        "method": "parameter_matched_lora" if args.enforce_parameter_budget else "lora",
        "repo_root": str(REPO_ROOT),
        "model_name": args.model_name,
        "seed": args.seed,
        "fold_root": str(fold_root),
        "output_root": str(output_root),
        "gpus": gpu_ids,
        "folds": [asdict(spec) for spec in specs],
        "hyperparameters": vars(args),
    }
    manifest_path = output_root / "run_manifest.json"
    if args.dry_run:
        print("Dry run: no training processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0

    write_json(manifest_path, manifest)
    results = run_dynamic_schedule(specs, REPO_ROOT, gpu_ids)
    manifest["finished_at"] = utc_now()
    manifest["folds"] = [asdict(spec) for spec in results]
    write_json(manifest_path, manifest)
    failed = [spec for spec in results if spec.status != "succeeded"]
    if len(results) != len(specs) or failed:
        if failed:
            print(
                "Failed folds: " + ", ".join(str(spec.fold) for spec in failed),
                file=sys.stderr,
            )
        return 1
    print(f"All {len(results)} LoRA fold trainings completed successfully.")
    print(f"Run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
