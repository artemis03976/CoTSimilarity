#!/usr/bin/env python3
"""Run held-out DSPR inference for the prepared Qwen K-fold experiment.

The script is an orchestration layer around the shared ``scripts/inference.py``
greedy entrypoint.
It selects the best validation checkpoint for every fold, assigns at most one
fold to each GPU, and validates the generated records before reporting success.

Examples
--------
Run all five folds on five GPUs::

    python scripts/evaluate.py dspr run --gpus 0,1,2,3,4

Run a Fold-0 smoke test on the held-out problem 2::

    python scripts/evaluate.py dspr run \
        --folds 0 --problem-id 2 --gpus 0 \
        --output-root output/qwen_staged_kfold_seed42_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_FOLDS = tuple(range(5))
VARIANTS = ("original", "simple", "hard")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            records.append(record)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def parse_fold_list(values: list[int] | None) -> list[int]:
    folds = list(DEFAULT_FOLDS) if values is None else values
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError(f"Duplicate fold IDs: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold >= len(DEFAULT_FOLDS)]
    if invalid:
        raise ValueError(f"Fold IDs must be in [0, 4], got {invalid}")
    return folds


def parse_gpu_list(value: str | None) -> list[str]:
    if value is None or value.strip().lower() == "auto":
        return detect_gpu_ids()
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU ID")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    return gpu_ids


def detect_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids:
            return list(dict.fromkeys(ids))

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
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )
        if probe.returncode == 0:
            count = len([line for line in probe.stdout.splitlines() if line.strip()])
            if count > 0:
                return [str(index) for index in range(count)]
    except OSError:
        pass
    return ["0"]


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else -1


def checkpoint_file(directory: Path) -> Path:
    return directory / "dspr_trainable.pt"


def eval_loss_for_checkpoint(state: dict[str, Any], step: int) -> float | None:
    matches = []
    for event in state.get("log_history", []):
        if not isinstance(event, dict) or "eval_loss" not in event:
            continue
        event_step = int(event.get("step", -1))
        if event_step <= step:
            try:
                matches.append((event_step, float(event["eval_loss"])))
            except (TypeError, ValueError):
                continue
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def resolve_checkpoint(fold_root: Path, mode: str = "best") -> tuple[Path, dict[str, Any]]:
    """Resolve a trainable checkpoint, repairing stale absolute best paths."""
    # The prefix stage saves both the trained prefixes and its frozen router.
    # Restrict best-checkpoint selection to that stage, whose step counter is
    # independent from the router stage's counter.
    if (fold_root / "prefix").is_dir():
        return resolve_checkpoint(fold_root / "prefix", mode)
    direct = checkpoint_file(fold_root)
    checkpoint_dirs = sorted(
        [path for path in fold_root.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    available = [path for path in checkpoint_dirs if checkpoint_file(path).is_file()]

    if mode == "latest":
        if available:
            selected = available[-1]
            return checkpoint_file(selected), {
                "method": "latest_checkpoint_step",
                "checkpoint_step": checkpoint_step(selected),
            }
        if direct.is_file():
            return direct, {"method": "direct_fold_checkpoint"}
        raise FileNotFoundError(f"No DSPR checkpoint found under {fold_root}")

    state_files = [path / "trainer_state.json" for path in reversed(checkpoint_dirs)]
    for state_path in state_files:
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        best_value = state.get("best_model_checkpoint")
        if not best_value:
            continue
        best_path = Path(str(best_value))
        # ``trainer_state.json`` may have been produced on another OS or the
        # checkpoint tree may have been moved.  Repair both slash styles by
        # retaining only the recorded checkpoint directory name.
        best_dir_name = re.split(r"[\\/]", str(best_value).rstrip("\\/"))[-1]
        candidates = [best_path, fold_root / best_dir_name]
        for candidate in candidates:
            candidate_file = checkpoint_file(candidate)
            if candidate_file.is_file():
                return candidate_file, {
                    "method": "trainer_state.best_model_checkpoint",
                    "trainer_state": str(state_path),
                    "recorded_best_path": str(best_value),
                    "best_metric": state.get("best_metric"),
                    "checkpoint_step": checkpoint_step(candidate),
                }

    # Fallback for older/moved runs whose trainer state lacks a usable best
    # path.  Match each retained checkpoint to the latest eval loss at or
    # before its step, then choose the minimum loss.
    scored: list[tuple[float, int, Path, Path]] = []
    for directory in available:
        state_path = directory / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        loss = eval_loss_for_checkpoint(state, checkpoint_step(directory))
        if loss is not None and math.isfinite(loss):
            scored.append((loss, checkpoint_step(directory), directory, state_path))
    if scored:
        loss, step, selected, state_path = min(scored, key=lambda item: (item[0], item[1]))
        return checkpoint_file(selected), {
            "method": "minimum_retained_eval_loss",
            "trainer_state": str(state_path),
            "best_metric": loss,
            "checkpoint_step": step,
        }

    if direct.is_file():
        return direct, {"method": "direct_fold_checkpoint"}
    if available:
        selected = available[-1]
        return checkpoint_file(selected), {
            "method": "latest_checkpoint_fallback_no_eval_state",
            "checkpoint_step": checkpoint_step(selected),
        }
    raise FileNotFoundError(f"No DSPR checkpoint found under {fold_root}")


@dataclass
class FoldSpec:
    fold: int
    gpu: str
    checkpoint: str
    checkpoint_selection: dict[str, Any]
    data_path: str
    expected_problem_ids: list[int]
    output_dir: str
    log_path: str
    command: list[str]
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


def build_inference_command(
    inference_script: Path,
    spec: FoldSpec,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(inference_script),
        "--baseline",
        "dspr",
        "--checkpoint",
        spec.checkpoint,
        "--model_name",
        args.model_name,
        "--output_dir",
        spec.output_dir,
        "--data_path",
        spec.data_path,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--n",
        str(args.samples_per_problem),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--device",
        "cuda",
        "--context_layer_idx",
        str(args.context_layer_idx),
        "--prefix_length",
        str(args.prefix_length),
        "--router_intermediate_dim",
        str(args.router_intermediate_dim),
        "--router_dropout",
        str(args.router_dropout),
        "--max_seq_length",
        str(args.max_seq_length),
    ]
    if args.problem_id is not None:
        command.extend(["--id", str(args.problem_id)])
    return command


def held_out_id_path(id_root: Path, fold: int) -> Path:
    path = id_root / f"fold_{fold}" / "test_ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold test IDs: {path}")
    return path


def expected_ids_for_fold(id_root: Path, fold: int) -> list[int]:
    path = held_out_id_path(id_root, fold)
    payload = read_json(path)
    ids = [normalise_id(value) for value in payload.get("all", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid or duplicate held-out IDs in {path}")
    return sorted(ids)


def validate_generated_output(spec: FoldSpec, expected_samples: int) -> dict[str, int]:
    path = Path(spec.output_dir) / "all_records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Inference result not found: {path}")
    records = read_jsonl(path)
    ids = [normalise_id(record.get("problem_id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate problem IDs in {path}")
    if set(ids) != set(spec.expected_problem_ids):
        missing = sorted(set(spec.expected_problem_ids) - set(ids))
        extra = sorted(set(ids) - set(spec.expected_problem_ids))
        raise ValueError(f"Fold {spec.fold} output ID mismatch: missing={missing}, extra={extra}")

    for record in records:
        pid = normalise_id(record.get("problem_id"))
        for variant in VARIANTS:
            variant_data = record.get(variant)
            if not isinstance(variant_data, dict):
                raise ValueError(f"Problem {pid} lacks {variant} result")
            samples = variant_data.get("samples")
            if not isinstance(samples, list) or len(samples) != expected_samples:
                raise ValueError(
                    f"Problem {pid} {variant} has {len(samples) if isinstance(samples, list) else 'invalid'} "
                    f"samples; expected {expected_samples}"
                )
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("correct"), bool):
                    raise ValueError(f"Problem {pid} {variant} has an invalid correctness flag")
            alpha = variant_data.get("alpha")
            alpha_values = alpha if isinstance(alpha, list) else [alpha]
            if len(alpha_values) != expected_samples:
                raise ValueError(f"Problem {pid} {variant} alpha count does not match samples")
            for value in alpha_values:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Problem {pid} {variant} has invalid alpha {value!r}") from exc
                if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"Problem {pid} {variant} alpha is outside [0, 1]: {numeric}")
    return {"records": len(records), "variants": len(records) * len(VARIANTS)}


def output_is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="checkpoints/qwen_staged_kfold_seed42")
    parser.add_argument("--test-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--id-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--output-root", default="output/qwen_staged_kfold_seed42")
    parser.add_argument("--inference-script", default="scripts/inference.py")
    parser.add_argument("--checkpoint-selection", choices=("best", "latest"), default="best")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs; default: auto-detect")
    parser.add_argument("--problem-id", type=int, default=None, help="One held-out problem for a smoke test")
    parser.add_argument("--allow-existing", action="store_true", help="Allow and overwrite non-empty fold outputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without inference")

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--context-layer-idx", type=int, default=15)
    parser.add_argument("--prefix-length", type=int, default=15)
    parser.add_argument("--router-intermediate-dim", type=int, default=64)
    parser.add_argument("--router-dropout", type=float, default=0.10)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def run_fold(spec: FoldSpec, repo_root: Path, expected_samples: int, lock: threading.Lock) -> FoldSpec:
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    with lock:
        spec.status = "running"
        spec.started_at = utc_now()
        write_json(output_dir / "run_status.json", asdict(spec))

    print(f"[fold {spec.fold}] starting on GPU {spec.gpu}: {spec.checkpoint}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Command: {json.dumps(spec.command, ensure_ascii=False)}\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={spec.gpu}\n\n")
        handle.flush()
        process = subprocess.Popen(
            spec.command,
            cwd=str(repo_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()

    elapsed = time.monotonic() - started
    validation_error: str | None = None
    if return_code == 0:
        try:
            validate_generated_output(spec, expected_samples)
        except Exception as exc:  # validation failure is a failed fold
            validation_error = str(exc)

    with lock:
        spec.return_code = return_code
        spec.elapsed_seconds = elapsed
        spec.finished_at = utc_now()
        if return_code == 0 and validation_error is None:
            spec.status = "succeeded"
        else:
            spec.status = "failed"
            spec.error = validation_error or f"Inference process exited with code {return_code}"
        write_json(output_dir / "run_status.json", asdict(spec))

    print(
        f"[fold {spec.fold}] {spec.status} on GPU {spec.gpu} "
        f"(exit={return_code}, elapsed={elapsed / 3600:.2f}h)",
        flush=True,
    )
    return spec


def run_dynamic_schedule(
    specs: list[FoldSpec],
    repo_root: Path,
    gpu_ids: list[str],
    expected_samples: int,
) -> list[FoldSpec]:
    pending: queue.Queue[FoldSpec] = queue.Queue()
    for spec in specs:
        pending.put(spec)
    lock = threading.Lock()
    completed: list[FoldSpec] = []
    completed_lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            result = spec
            try:
                spec.gpu = gpu
                result = run_fold(spec, repo_root, expected_samples, lock)
            except Exception as exc:
                spec.status = "failed"
                spec.return_code = -1
                spec.error = repr(exc)
                spec.finished_at = utc_now()
                try:
                    write_json(Path(spec.output_dir) / "run_status.json", asdict(spec))
                except Exception:
                    pass
                print(f"[fold {spec.fold}] failed before/during launch: {exc}", file=sys.stderr, flush=True)
            finally:
                with completed_lock:
                    completed.append(result)
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"dspr-eval-gpu-{gpu}")
        for gpu in gpu_ids[: min(len(gpu_ids), len(specs))]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


def main() -> int:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0 or args.samples_per_problem != 1:
        raise ValueError(
            "Held-out evaluation is deterministic-only: require temperature=0, "
            "top_p=1, samples_per_problem=1"
        )
    folds = parse_fold_list(args.folds)
    if args.problem_id is not None and len(folds) != 1:
        raise ValueError("--problem-id smoke tests require exactly one selected fold")
    gpu_ids = parse_gpu_list(args.gpus)

    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_root = resolve_path(repo_root, args.checkpoint_root).resolve()
    test_root = resolve_path(repo_root, args.test_root).resolve()
    id_root = resolve_path(repo_root, args.id_root).resolve()
    output_root = resolve_path(repo_root, args.output_root).resolve()
    inference_script = resolve_path(repo_root, args.inference_script).resolve()
    if not inference_script.is_file():
        raise FileNotFoundError(f"Inference entrypoint not found: {inference_script}")

    specs: list[FoldSpec] = []
    for index, fold in enumerate(folds):
        fold_checkpoint_root = checkpoint_root / f"fold_{fold}"
        checkpoint, selection = resolve_checkpoint(fold_checkpoint_root, args.checkpoint_selection)
        all_expected_ids = expected_ids_for_fold(id_root, fold)
        if args.problem_id is not None:
            if args.problem_id not in all_expected_ids:
                raise ValueError(f"Problem {args.problem_id} is not in Fold {fold}'s held-out test set")
            expected_ids = [args.problem_id]
        else:
            expected_ids = all_expected_ids

        data_path = test_root / f"fold_{fold}" / "test.jsonl"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} raw test data: {data_path}")
        raw_test_records = read_jsonl(data_path)
        raw_test_ids = [normalise_id(record.get("problem_id")) for record in raw_test_records]
        if len(raw_test_ids) != len(set(raw_test_ids)):
            raise ValueError(f"Duplicate problem IDs in Fold {fold} raw test data: {data_path}")
        if set(raw_test_ids) != set(all_expected_ids):
            missing = sorted(set(all_expected_ids) - set(raw_test_ids))
            extra = sorted(set(raw_test_ids) - set(all_expected_ids))
            raise ValueError(
                f"Fold {fold} raw test data differs from test_ids.json: missing={missing}, extra={extra}"
            )
        output_dir = output_root / f"fold_{fold}"
        if output_is_nonempty(output_dir) and not (args.allow_existing or args.dry_run):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose a new --output-root "
                "or pass --allow-existing explicitly."
            )

        spec = FoldSpec(
            fold=fold,
            gpu=gpu_ids[index % len(gpu_ids)],
            checkpoint=str(checkpoint.resolve()),
            checkpoint_selection=selection,
            data_path=str(data_path.resolve()),
            expected_problem_ids=expected_ids,
            output_dir=str(output_dir.resolve()),
            log_path=str((output_dir / "inference.log").resolve()),
            command=[],
        )
        spec.command = build_inference_command(inference_script, spec, args)
        specs.append(spec)

    print(f"Model: {args.model_name}")
    print(f"Folds: {folds}; GPUs: {gpu_ids}; prefix length: {args.prefix_length}")
    print(f"Output root: {output_root}")
    for spec in specs:
        print(
            f"[fold {spec.fold}] GPU {spec.gpu}; N={len(spec.expected_problem_ids)}; "
            f"checkpoint={spec.checkpoint}"
        )

    manifest = {
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "model_name": args.model_name,
        "checkpoint_root": str(checkpoint_root),
        "test_root": str(test_root),
        "id_root": str(id_root),
        "output_root": str(output_root),
        "gpus": gpu_ids,
        "folds": [asdict(spec) for spec in specs],
        "settings": vars(args),
    }
    manifest_path = output_root / "inference_manifest.json"

    if args.dry_run:
        print("Dry run: no inference processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0

    write_json(manifest_path, manifest)
    results = run_dynamic_schedule(specs, repo_root, gpu_ids, args.samples_per_problem)
    manifest["finished_at"] = utc_now()
    manifest["folds"] = [asdict(spec) for spec in results]
    write_json(manifest_path, manifest)

    if len(results) != len(specs):
        missing = sorted(set(folds) - {spec.fold for spec in results})
        print(f"No result recorded for folds: {missing}", file=sys.stderr)
        return 1
    failed = [spec for spec in results if spec.status != "succeeded"]
    if failed:
        print("Failed folds: " + ", ".join(str(spec.fold) for spec in failed), file=sys.stderr)
        print(f"See inference.log and run_status.json under {output_root}", file=sys.stderr)
        return 1
    print(f"All {len(results)} fold inference jobs completed and passed validation.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
