"""Train static prompts on the existing DSPR train/validation folds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .kfold_dspr import (DEFAULT_FOLDS, DEFAULT_MODEL_NAME, FoldSpec, parse_fold_list,
                         parse_gpu_list, resolve_path, run_dynamic_schedule, utc_now,
                         validate_inputs, write_json)


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_train_command(train_script: Path, spec: FoldSpec, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(train_script), "spt",
               "--train_data_path", spec.train_data, "--val_data_path", spec.val_data,
               "--output_path", spec.output_path, "--device", "cuda"]
    for name in ("model_name", "prefix_length", "learning_rate", "weight_decay", "batch_size",
                 "num_epochs", "gradient_accumulation_steps", "max_grad_norm", "max_seq_length",
                 "seed", "logging_steps", "save_total_limit", "trajectory_sampling",
                 "early_stopping_patience", "early_stopping_threshold"):
        command.extend([f"--{name}", str(getattr(args, name))])
    if args.warmup_steps is not None:
        command.extend(["--warmup_steps", str(args.warmup_steps)])
    else:
        command.extend(["--warmup_ratio", str(args.warmup_ratio)])
    command.append("--gradient_checkpointing" if args.gradient_checkpointing else "--no-gradient_checkpointing")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--output-root", default="checkpoints/qwen_spt_seed42")
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=float, default=15)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-steps", type=int, default=None)
    warmup.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--trajectory-sampling", choices=("flat", "random_one"), default="flat")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.warmup_steps is None and args.warmup_ratio is None:
        args.warmup_ratio = 0.05
    return args


def main() -> int:
    args = parse_args()
    folds = parse_fold_list(args.folds)
    gpu_ids = parse_gpu_list(args.gpus)
    fold_root = resolve_path(REPO_ROOT, args.fold_root).resolve()
    output_root = resolve_path(REPO_ROOT, args.output_root).resolve()
    train_script = resolve_path(REPO_ROOT, args.train_script).resolve()
    validate_inputs(train_script, fold_root, output_root, folds,
                    allow_existing=args.allow_existing or args.dry_run)
    specs = []
    for index, fold in enumerate(folds):
        fold_dir = fold_root / f"fold_{fold}"
        output_dir = output_root / f"fold_{fold}"
        spec = FoldSpec(fold=fold, gpu=gpu_ids[index % len(gpu_ids)],
                        train_data=str(fold_dir / "train.jsonl"), val_data=str(fold_dir / "val.jsonl"),
                        output_path=str(output_dir), log_path=str(output_dir / "train.log"), command=[])
        spec.command = build_train_command(train_script, spec, args)
        specs.append(spec)
    print(f"SPT: model={args.model_name}, prefix={args.prefix_length}, seed={args.seed}")
    print(f"Folds: {folds}; GPUs: {gpu_ids}; output: {output_root}")
    manifest = {"created_at": utc_now(), "method": "static_prompt_tuning",
                "model_name": args.model_name, "seed": args.seed, "fold_root": str(fold_root),
                "output_root": str(output_root), "gpus": gpu_ids,
                "folds": [asdict(spec) for spec in specs], "hyperparameters": vars(args)}
    if args.dry_run:
        print("Dry run: no training processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0
    manifest_path = output_root / "run_manifest.json"
    write_json(manifest_path, manifest)
    results = run_dynamic_schedule(specs, REPO_ROOT, gpu_ids)
    manifest.update(finished_at=utc_now(), folds=[asdict(spec) for spec in results])
    write_json(manifest_path, manifest)
    failed = [spec for spec in results if spec.status != "succeeded"]
    if len(results) != len(specs) or failed:
        if failed:
            print("Failed folds: " + ", ".join(str(spec.fold) for spec in failed), file=sys.stderr)
        return 1
    print(f"All {len(results)} SPT fold trainings completed successfully.")
    print(f"Run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
