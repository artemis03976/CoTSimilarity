#!/usr/bin/env python3
"""Train DSPR on the prepared Qwen K-fold splits.

This is an orchestration layer around ``scripts/train.py dspr``.  It keeps the
actual training pipeline in one place and runs one independent process per GPU.
When fewer GPUs than folds are available, each GPU takes the next fold as soon
as its previous fold finishes (a small dynamic work queue).

Examples
--------
Dry-run the five-fold schedule without importing the model::

    python scripts/train_kfold.py dspr --dry-run

Run all folds on five GPUs::

    python scripts/train_kfold.py dspr --gpus 0,1,2,3,4

Run on two GPUs; folds are assigned dynamically::

    python scripts/train_kfold.py dspr --gpus 0,1

The GPU identifiers are physical IDs.  Each child process receives one ID in
``CUDA_VISIBLE_DEVICES`` and therefore sees that GPU as ``cuda:0`` internally.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_FOLDS = tuple(range(5))
PROGRESS_EVENT_PREFIX = "__DSPR_TRAIN_PROGRESS__"


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


def parse_progress_event(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(PROGRESS_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(stripped[len(PROGRESS_EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


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


def build_train_command(
    train_script: Path,
    spec: FoldSpec,
    args: argparse.Namespace,
) -> list[str]:
    """Build one structured subprocess argument list; no shell quoting needed."""
    command = [
        sys.executable,
        str(train_script),
        "dspr",
        "--model_name",
        args.model_name,
        "--context_layer_idx",
        str(args.context_layer_idx),
        "--prefix_length",
        str(args.prefix_length),
        "--router_intermediate_dim",
        str(args.router_intermediate_dim),
        "--router_dropout",
        str(args.router_dropout),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--lambda_router",
        str(args.lambda_router),
        "--train_data_path",
        spec.train_data,
        "--val_data_path",
        spec.val_data,
        "--max_seq_length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--output_path",
        spec.output_path,
        "--logging_steps",
        str(args.logging_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--gradient_checkpointing",
        "--emit_progress_events",
    ]
    if args.warmup_ratio is None:
        command.extend(["--warmup_steps", str(args.warmup_steps)])
    else:
        command.extend(["--warmup_ratio", str(args.warmup_ratio)])
    if args.early_stopping_patience > 0:
        command.extend(
            [
                "--early_stopping_patience",
                str(args.early_stopping_patience),
                "--early_stopping_threshold",
                str(args.early_stopping_threshold),
            ]
        )
    return command


def validate_inputs(
    train_script: Path,
    fold_root: Path,
    output_root: Path,
    folds: Iterable[int],
    allow_existing: bool,
) -> None:
    if not train_script.is_file():
        raise FileNotFoundError(f"DSPR training entrypoint not found: {train_script}")
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", default="data/qwen/kfold", help="Prepared Qwen K-fold data directory")
    parser.add_argument(
        "--output-root",
        default="checkpoints/qwen_kfold_seed42",
        help="Root directory; each fold is written to <output-root>/fold_N",
    )
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Subset of folds, e.g. --folds 0 1")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU IDs, e.g. 0,1,2. Default: auto-detect; use --gpus 0 for one GPU.",
    )
    parser.add_argument("--allow-existing", action="store_true", help="Allow non-empty fold output directories")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the schedule only")

    # These defaults are the paper-facing DSPR settings.  They are forwarded
    # explicitly so a later change to DSPRConfig cannot silently alter a fold.
    parser.add_argument("--context-layer-idx", type=int, default=15)
    parser.add_argument("--prefix-length", type=int, default=15)
    parser.add_argument("--router-intermediate-dim", type=int, default=256)
    parser.add_argument("--router-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup fraction in [0, 1); replaces --warmup-steps when set.",
    )
    parser.add_argument("--lambda-router", type=float, default=0.5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Evaluations without improvement before stopping; 0 disables it.",
    )
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    return parser.parse_args()


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

            total_steps = max(int(event.get("total_steps", 0)), 0)
            current_step = max(int(event.get("step", 0)), 0)
            if progress_bar is None and total_steps > 0:
                progress_bar = tqdm(
                    total=total_steps,
                    desc=f"fold {spec.fold} | GPU {spec.gpu}",
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
        threading.Thread(target=worker, args=(gpu, position), name=f"dSPR-gpu-{gpu}")
        for position, gpu in enumerate(gpu_ids[:worker_count])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


def main() -> int:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_threshold < 0.0:
        raise ValueError("--early-stopping-threshold must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    folds = parse_fold_list(args.folds)
    gpu_ids = parse_gpu_list(args.gpus)
    fold_root = resolve_path(repo_root, args.fold_root).resolve()
    output_root = resolve_path(repo_root, args.output_root).resolve()
    train_script = resolve_path(repo_root, args.train_script).resolve()

    validate_inputs(
        train_script,
        fold_root,
        output_root,
        folds,
        allow_existing=args.allow_existing or args.dry_run,
    )

    specs: list[FoldSpec] = []
    for index, fold in enumerate(folds):
        gpu = gpu_ids[index % len(gpu_ids)]
        fold_dir = fold_root / f"fold_{fold}"
        fold_output = output_root / f"fold_{fold}"
        spec = FoldSpec(
            fold=fold,
            gpu=gpu,
            train_data=str((fold_dir / "train.jsonl").resolve()),
            val_data=str((fold_dir / "val.jsonl").resolve()),
            output_path=str(fold_output.resolve()),
            log_path=str((fold_output / "train.log").resolve()),
            command=[],
        )
        spec.command = build_train_command(train_script, spec, args)
        specs.append(spec)

    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}; folds: {folds}; GPUs: {gpu_ids}")
    print(f"Output root: {output_root}")
    for spec in specs:
        print(f"[fold {spec.fold}] GPU {spec.gpu} -> {spec.output_path}")

    run_manifest = {
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "model_name": args.model_name,
        "seed": args.seed,
        "fold_root": str(fold_root),
        "output_root": str(output_root),
        "gpus": gpu_ids,
        "folds": [asdict(spec) for spec in specs],
        "hyperparameters": {
            key.replace("-", "_"): value
            for key, value in vars(args).items()
            if key not in {"folds", "gpus", "dry_run", "allow_existing"}
        },
    }
    manifest_path = output_root / "run_manifest.json"

    if args.dry_run:
        print("Dry run: no training processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0

    write_json(manifest_path, run_manifest)
    results = run_dynamic_schedule(specs, repo_root, gpu_ids)
    run_manifest["finished_at"] = utc_now()
    run_manifest["folds"] = [asdict(spec) for spec in results]
    write_json(manifest_path, run_manifest)

    failed = [spec for spec in results if spec.status != "succeeded"]
    if len(results) != len(specs):
        missing = sorted(set(folds) - {spec.fold for spec in results})
        print(f"No result was recorded for folds: {missing}", file=sys.stderr)
        return 1
    if failed:
        print("Failed folds: " + ", ".join(str(spec.fold) for spec in failed), file=sys.stderr)
        print(f"See per-fold logs under {output_root}", file=sys.stderr)
        return 1
    print(f"All {len(results)} fold trainings completed successfully.")
    print(f"Run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
