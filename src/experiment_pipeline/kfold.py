#!/usr/bin/env python3
"""Shared GPU scheduling, progress reporting, and K-fold training for DSPR, SPT, and LoRA.

Each GPU worker claims the next fold when free. Child training remains in
scripts/train.py, with method-specific options assembled by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from utils.training_progress import parse_progress_event

from .train_lora import DEFAULT_DSPR_BUDGET, DEFAULT_RANK, DEFAULT_LORA_ALPHA, DEFAULT_RUN_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOTS = {
    "dspr": "checkpoints/qwen_staged_kfold_seed42",
    "spt": "checkpoints/qwen_spt_seed42",
    "lora": f"checkpoints/{DEFAULT_RUN_NAME}",
}

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_FOLDS = tuple(range(5))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_fold_list(values: list[int] | None) -> list[int]:
    folds = list(DEFAULT_FOLDS) if values is None else values
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError(f"Duplicate fold IDs: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold >= 5]
    if invalid:
        raise ValueError(f"Fold IDs must be in [0, 4], got {invalid}")
    return folds


def parse_gpu_list(value: str | None) -> list[str]:
    """Parse ``--gpus`` while preserving IDs such as ``0`` or ``MIG-...``."""
    if value is None or value.strip().lower() == "auto":
        return detect_gpu_ids()
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    return gpu_ids


def detect_gpu_ids() -> list[str]:
    """Best-effort GPU discovery with a conservative single-GPU fallback."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids:
            return ids

    # Keep the orchestrator usable in a lightweight environment where torch is
    # unavailable.  The actual training child will provide the definitive
    # CUDA error if a GPU is required but missing.
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            check=False,
        )
        count = int(probe.stdout.strip().splitlines()[-1])
        if count > 0:
            return [str(index) for index in range(count)]
    except (OSError, ValueError, IndexError):
        pass

    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            count = len([line for line in probe.stdout.splitlines() if line.strip()])
            if count > 0:
                return [str(index) for index in range(count)]
    except OSError:
        pass

    return ["0"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


@dataclass
class FoldSpec:
    fold: int
    gpu: str
    train_data: str
    val_data: str
    output_path: str
    log_path: str
    command: list[str]
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    global_step: int | None = None
    max_steps: int | None = None
    termination: str | None = None
    best_metric: float | None = None
    best_model_checkpoint: str | None = None
    stage: str | None = None
    stage_results: dict[str, Any] = field(default_factory=dict)


def validate_inputs(
    train_script: Path,
    fold_root: Path,
    output_root: Path,
    folds: Iterable[int],
    allow_existing: bool,
) -> None:
    if not train_script.is_file():
        raise FileNotFoundError(f"Training entrypoint not found: {train_script}")
    if not fold_root.is_dir():
        raise FileNotFoundError(f"Qwen K-fold directory not found: {fold_root}")

    for fold in folds:
        fold_dir = fold_root / f"fold_{fold}"
        for filename in ("train.jsonl", "val.jsonl"):
            path = fold_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing fold input: {path}")

        fold_output = output_root / f"fold_{fold}"
        output_is_nonempty = fold_output.exists() and (
            not fold_output.is_dir() or any(fold_output.iterdir())
        )
        if output_is_nonempty and not allow_existing:
            raise FileExistsError(
                f"Output directory is not empty: {fold_output}. "
                "Choose a new --output-root or pass --allow-existing explicitly."
            )

def run_fold(
    spec: FoldSpec,
    repo_root: Path,
    state_lock: threading.Lock,
    progress_position: int,
) -> FoldSpec:
    """Run one fold and persist a small status file alongside its log."""
    output_path = Path(spec.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env["PYTHONUNBUFFERED"] = "1"
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    with state_lock:
        spec.status = "running"
        spec.started_at = utc_now()
        write_json(output_path / "run_status.json", asdict(spec))

    tqdm.write(f"[fold {spec.fold}] starting on GPU {spec.gpu}")
    started = time.monotonic()
    progress_bar = None
    last_progress_event: dict[str, Any] | None = None
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"Command: {json.dumps(spec.command, ensure_ascii=False)}\n")
        log_handle.write(f"CUDA_VISIBLE_DEVICES={spec.gpu}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            spec.command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            event = parse_progress_event(line)
            if event is None:
                log_handle.write(line)
                log_handle.flush()
                continue

            event_stage = event.get("stage")
            if event_stage != spec.stage or event.get("event") == "start":
                if progress_bar is not None:
                    progress_bar.close()
                    progress_bar = None
                spec.stage = event_stage
                spec.termination = None
            last_progress_event = event
            spec.global_step = int(event.get("step", 0))
            spec.max_steps = int(event.get("total_steps", 0))
            spec.best_metric = event.get("best_metric")
            spec.best_model_checkpoint = event.get("best_model_checkpoint")
            if event.get("event") in {"early_stop", "complete"}:
                spec.termination = (
                    "early_stopping"
                    if event.get("event") == "early_stop"
                    else "completed"
                )
                log_handle.write(f"[progress] {line}")
                log_handle.flush()
                if spec.stage is not None:
                    spec.stage_results[spec.stage] = {**event, "termination": spec.termination}
            if event.get("event") in {"start", "early_stop", "complete"}:
                with state_lock:
                    write_json(output_path / "run_status.json", asdict(spec))

            total_steps = max(int(event.get("total_steps", 0)), 0)
            current_step = max(int(event.get("step", 0)), 0)
            if progress_bar is None and total_steps > 0:
                progress_bar = tqdm(
                    total=total_steps,
                    desc=f"fold {spec.fold} | {spec.stage or 'train'} | GPU {spec.gpu}",
                    unit="step",
                    position=progress_position,
                    leave=False,
                    dynamic_ncols=True,
                )
            if progress_bar is not None:
                progress_bar.update(max(0, min(current_step, progress_bar.total) - progress_bar.n))
                epoch = event.get("epoch")
                postfix = {}
                if epoch is not None:
                    postfix["epoch"] = f"{float(epoch):.2f}"
                if event.get("event") == "early_stop":
                    postfix["status"] = "early stop"
                if postfix:
                    progress_bar.set_postfix(postfix, refresh=False)
        process.stdout.close()
        return_code = process.wait()
        log_handle.write(
            "Child process exited: "
            f"return_code={return_code}, "
            f"termination={spec.termination or 'unknown'}, "
            f"global_step={spec.global_step}, max_steps={spec.max_steps}\n"
        )
        if last_progress_event is not None:
            log_handle.write(
                "Final progress event: "
                f"{json.dumps(last_progress_event, ensure_ascii=False)}\n"
            )
        log_handle.flush()
    if progress_bar is not None:
        progress_bar.close()

    with state_lock:
        spec.status = "succeeded" if return_code == 0 else "failed"
        spec.return_code = return_code
        spec.finished_at = utc_now()
        write_json(output_path / "run_status.json", asdict(spec))

    elapsed = time.monotonic() - started
    tqdm.write(
        f"[fold {spec.fold}] {spec.status} on GPU {spec.gpu} "
        f"(exit={return_code}, termination={spec.termination or 'unknown'}, "
        f"step={spec.global_step}/{spec.max_steps}, elapsed={elapsed / 3600:.2f}h)"
    )
    return spec


def run_dynamic_schedule(specs: list[FoldSpec], repo_root: Path, gpu_ids: list[str]) -> list[FoldSpec]:
    pending: queue.Queue[FoldSpec] = queue.Queue()
    for spec in specs:
        pending.put(spec)

    state_lock = threading.Lock()
    completed: list[FoldSpec] = []
    completed_lock = threading.Lock()

    def worker(gpu: str, progress_position: int) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            result = spec
            try:
                # GPU assignment happens when a worker claims the fold.  This
                # is what prevents a fast worker from colliding with a slower
                # worker on the same GPU.
                spec.gpu = gpu
                result = run_fold(spec, repo_root, state_lock, progress_position)
            except Exception as exc:  # keep other GPUs/folds progressing
                with state_lock:
                    spec.status = "failed"
                    spec.return_code = -1
                    spec.error = repr(exc)
                    spec.finished_at = utc_now()
                    write_json(Path(spec.output_path) / "run_status.json", asdict(spec))
                tqdm.write(f"[fold {spec.fold}] failed before/during launch: {exc}", file=sys.stderr)
                result = spec
            finally:
                with completed_lock:
                    completed.append(result)
                pending.task_done()

    worker_count = min(len(gpu_ids), len(specs))
    threads = [
        threading.Thread(target=worker, args=(gpu, position), name=f"kfold-gpu-{gpu}")
        for position, gpu in enumerate(gpu_ids[:worker_count])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


# Preserve each training entrypoint's option spellings and argument order.
COMMAND_ARGUMENTS = {
    "dspr": """
        run-stage problem-dataset kfold-root fold model_name
        context_layer_idx prefix_length router_intermediate_dim router_dropout
        router_learning_rate prefix_learning_rate router_weight_decay prefix_weight_decay
        router_target_smoothing batch_size router_epochs prefix_epochs
        gradient_accumulation_steps max_grad_norm prefix_train_data_path prefix_val_data_path
        max_seq_length seed device output_path logging_steps save_total_limit emit_progress_events
        prefix_trajectory_sampling router_early_stopping_patience prefix_early_stopping_patience
        early_stopping_threshold
    """.split(),
    "spt": """
        train_data_path val_data_path output_path device emit_progress_events
        model_name prefix_length learning_rate weight_decay batch_size num_epochs
        gradient_accumulation_steps max_grad_norm max_seq_length seed logging_steps
        save_total_limit trajectory_sampling early_stopping_patience early_stopping_threshold
    """.split(),
    "lora": """
        model_name train_data_path val_data_path output_path rank lora_alpha target_modules
        lora_dropout learning_rate weight_decay batch_size num_epochs gradient_accumulation_steps
        max_grad_norm warmup_steps warmup_ratio max_seq_length seed device logging_steps
        save_total_limit early_stopping_patience early_stopping_threshold dtype
        attention_implementation dspr_parameter_budget budget_tolerance emit_progress_events
    """.split(),
}


def build_train_command(
    train_script: Path, spec: FoldSpec, args: argparse.Namespace, method: str = "dspr",
) -> list[str]:
    """Serialize method options into one subprocess command without shell quoting."""
    command = [sys.executable, str(train_script), method]
    values = {
        **vars(args),
        "run_stage": "all", "kfold_root": args.fold_root,
        "fold": spec.fold, "train_data_path": spec.train_data, "val_data_path": spec.val_data,
        "prefix_train_data_path": spec.train_data, "prefix_val_data_path": spec.val_data,
        "output_path": spec.output_path, "device": "cuda", "emit_progress_events": True,
    }
    for option in COMMAND_ARGUMENTS[method]:
        command.append(f"--{option}")
        if option == "emit_progress_events":
            continue
        value = values[option.replace("-", "_")]
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))

    if method == "spt":
        name = "warmup_steps" if args.warmup_steps is not None else "warmup_ratio"
        command.extend([f"--{name}", str(getattr(args, name))])
    command.append("--gradient_checkpointing" if args.gradient_checkpointing else "--no-gradient_checkpointing")
    if method == "dspr":
        if args.router_positive_weight is not None:
            command.extend(["--router_positive_weight", str(args.router_positive_weight)])
        name = "warmup_steps" if args.warmup_ratio is None else "warmup_ratio"
        command.extend([f"--{name}", str(getattr(args, name))])
    elif method == "lora":
        command.append("--enforce_parameter_budget" if args.enforce_parameter_budget else "--no-enforce_parameter_budget")
    return command


def parse_args(method: str = "dspr", argv: list[str] | None = None) -> argparse.Namespace:
    if method not in DEFAULT_OUTPUT_ROOTS:
        raise ValueError(f"Unsupported K-fold method: {method}")
    parser = argparse.ArgumentParser(description=f"Train {method.upper()} on the prepared Qwen folds")
    parser.add_argument("--fold-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOTS[method])
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="Physical GPU IDs, e.g. 0,1; default: auto-detect")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # LoRA's child accepts both values and lets explicit steps take precedence.
    # DSPR/SPT retain their mutually exclusive step/ratio CLI.
    warmup = parser if method == "lora" else parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-steps", type=int, default=0 if method == "lora" else None)
    warmup.add_argument("--warmup-ratio", type=float, default=0.05 if method == "lora" else None)
    if method == "dspr":
        parser.add_argument("--problem-dataset", "--problem_dataset", default="data/canonical_math_paired.jsonl")
        parser.add_argument("--context-layer-idx", type=int, default=15)
        parser.add_argument("--prefix-length", type=int, default=15)
        parser.add_argument("--router-intermediate-dim", type=int, default=64)
        parser.add_argument("--router-dropout", type=float, default=0.10)
        parser.add_argument("--prefix-learning-rate", "--learning-rate", type=float, default=4e-5)
        parser.add_argument("--router-learning-rate", type=float, default=1e-5)
        parser.add_argument("--router-weight-decay", type=float, default=1e-3)
        parser.add_argument("--prefix-weight-decay", type=float, default=0.0)
        parser.add_argument("--router-target-smoothing", type=float, default=0.05)
        parser.add_argument("--router-positive-weight", type=float, default=None)
        parser.add_argument("--router-epochs", type=int, default=20)
        parser.add_argument("--prefix-epochs", "--num-epochs", type=int, default=15)
        parser.add_argument("--prefix-trajectory-sampling", "--trajectory-sampling", choices=("flat", "random_one"), default="flat")
        parser.add_argument("--early-stopping-patience", type=int, default=None)
        parser.add_argument("--router-early-stopping-patience", type=int, default=3)
        parser.add_argument("--prefix-early-stopping-patience", type=int, default=3)
    else:
        parser.add_argument("--learning-rate", type=float, default=4e-5)
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--num-epochs", type=float, default=15)
        parser.add_argument("--early-stopping-patience", type=int, default=3)
        if method == "spt":
            parser.add_argument("--prefix-length", type=int, default=50)
            parser.add_argument("--trajectory-sampling", choices=("flat", "random_one"), default="flat")
        else:
            parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
            parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
            parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
            parser.add_argument("--lora-dropout", type=float, default=0.05)
            parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
            parser.add_argument("--attention-implementation", default="flash_attention_2")
            parser.add_argument("--dspr-parameter-budget", type=int, default=DEFAULT_DSPR_BUDGET)
            parser.add_argument("--budget-tolerance", type=float, default=0.10)
            parser.add_argument("--enforce-parameter-budget", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if method == "dspr":
        if args.warmup_steps is None:
            args.warmup_steps = 100
            if args.warmup_ratio is None:
                args.warmup_ratio = 0.05
        if args.early_stopping_patience is not None:
            args.router_early_stopping_patience = args.early_stopping_patience
            args.prefix_early_stopping_patience = args.early_stopping_patience
    elif method == "spt" and args.warmup_steps is None and args.warmup_ratio is None:
        args.warmup_ratio = 0.05
    return args


def validate_training_args(args: argparse.Namespace, method: str) -> None:
    """Retain staged-DSPR validation; baseline trainers validate their own options."""
    if method != "dspr":
        return
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.prefix_learning_rate is not None and args.prefix_learning_rate <= 0.0:
        raise ValueError("--prefix-learning-rate must be positive")
    if args.router_learning_rate is not None and args.router_learning_rate <= 0.0:
        raise ValueError("--router-learning-rate must be positive")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if min(args.router_early_stopping_patience, args.prefix_early_stopping_patience) < 0:
        raise ValueError("Stage early stopping patience must be non-negative")
    if args.router_epochs <= 0 or args.prefix_epochs <= 0:
        raise ValueError("Stage epoch counts must be positive")
    if args.router_weight_decay < 0 or args.prefix_weight_decay < 0:
        raise ValueError("Weight decay must be non-negative")
    if not 0 <= args.router_target_smoothing < 0.5:
        raise ValueError("--router-target-smoothing must be in [0, 0.5)")
    if args.router_positive_weight is not None and args.router_positive_weight <= 0:
        raise ValueError("--router-positive-weight must be positive")
    if args.early_stopping_threshold < 0.0:
        raise ValueError("--early-stopping-threshold must be non-negative")


def main(method: str = "dspr", argv: list[str] | None = None) -> int:
    args = parse_args(method, argv)
    validate_training_args(args, method)
    folds = parse_fold_list(args.folds)
    gpu_ids = parse_gpu_list(args.gpus)
    fold_root = resolve_path(REPO_ROOT, args.fold_root).resolve()
    output_root = resolve_path(REPO_ROOT, args.output_root).resolve()
    train_script = resolve_path(REPO_ROOT, args.train_script).resolve()
    if method == "dspr":
        args.fold_root = str(fold_root)
        args.problem_dataset = str(resolve_path(REPO_ROOT, args.problem_dataset).resolve())
        if not Path(args.problem_dataset).is_file():
            raise FileNotFoundError(f"Canonical router source not found: {args.problem_dataset}")
    validate_inputs(train_script, fold_root, output_root, folds, allow_existing=args.allow_existing or args.dry_run)

    specs = []
    for index, fold in enumerate(folds):
        fold_dir = fold_root / f"fold_{fold}"
        fold_output = output_root / f"fold_{fold}"
        spec = FoldSpec(
            fold=fold, gpu=gpu_ids[index % len(gpu_ids)],
            train_data=str((fold_dir / "train.jsonl").resolve()),
            val_data=str((fold_dir / "val.jsonl").resolve()),
            output_path=str(fold_output.resolve()), log_path=str((fold_output / "train.log").resolve()),
            command=[],
        )
        spec.command = build_train_command(train_script, spec, args, method)
        specs.append(spec)

    print(f"Method: {method}; model: {args.model_name}; seed: {args.seed}")
    print(f"Folds: {folds}; GPUs: {gpu_ids}; output: {output_root}")
    for spec in specs:
        print(f"[fold {spec.fold}] GPU {spec.gpu} -> {spec.output_path}")
    manifest = {
        "created_at": utc_now(), "model_name": args.model_name, "seed": args.seed,
        "fold_root": str(fold_root), "output_root": str(output_root), "gpus": gpu_ids,
        "folds": [asdict(spec) for spec in specs], "hyperparameters": vars(args),
    }
    if method == "dspr":
        manifest.update(repo_root=str(REPO_ROOT), training_method="staged_dspr")
        manifest["hyperparameters"] = {
            key: value for key, value in vars(args).items()
            if key not in {"folds", "gpus", "dry_run", "allow_existing"}
        }
    elif method == "spt":
        manifest["method"] = "static_prompt_tuning"
    else:
        manifest.update(
            repo_root=str(REPO_ROOT),
            method="parameter_matched_lora" if args.enforce_parameter_budget else "lora",
        )
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
    if len(results) != len(specs):
        missing = sorted(set(folds) - {spec.fold for spec in results})
        print(f"No result was recorded for folds: {missing}", file=sys.stderr)
        return 1
    failed = [spec for spec in results if spec.status != "succeeded"]
    if failed:
        print("Failed folds: " + ", ".join(str(spec.fold) for spec in failed), file=sys.stderr)
        print(f"See per-fold logs under {output_root}", file=sys.stderr)
        return 1
    print(f"All {len(results)} {method.upper()} fold trainings completed successfully.")
    print(f"Run manifest: {manifest_path}")
    return 0
