#!/usr/bin/env python3
"""Shared checkpoint selection and held-out inference for DSPR, SPT, and LoRA."""

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
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm
from utils.inference_progress import parse_progress

from .kfold import (
    DEFAULT_FOLDS, DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_ROOTS as DEFAULT_CHECKPOINT_ROOTS,
    parse_fold_list, parse_gpu_list, resolve_path, utc_now, write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOTS = {
    method: f"output/{Path(path).name}" for method, path in DEFAULT_CHECKPOINT_ROOTS.items()
}
VARIANTS = ("original", "simple", "hard")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
CHECKPOINT_FILES = {"dspr": "dspr_trainable.pt", "spt": "baseline_trainable.pt"}
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")


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


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.match(path.name)
    return int(match.group(1)) if match else -1


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


def output_is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


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
    stage: str = "waiting"
    completed_generations: int = 0
    total_generations: int = 0
    method: str = "dspr"


def run_fold(
    spec: FoldSpec,
    expected_samples: int,
    lock: threading.Lock,
    progress_position: int = 0,
) -> FoldSpec:
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with lock:
        spec.status = "running"
        spec.stage = "starting"
        spec.total_generations = len(spec.expected_problem_ids) * len(VARIANTS)
        spec.started_at = utc_now()
        write_json(output_dir / "run_status.json", asdict(spec))

    tqdm.write(f"[fold {spec.fold}] starting on GPU {spec.gpu}; "
               f"{len(spec.expected_problem_ids)} problems / {spec.total_generations} answers")
    tqdm.write(f"[fold {spec.fold}] checkpoint: {spec.checkpoint}")
    tqdm.write(f"[fold {spec.fold}] log: {spec.log_path}")
    started = time.monotonic()
    error_tail: deque[str] = deque(maxlen=12)
    progress = tqdm(total=spec.total_generations,
                    desc=f"fold {spec.fold} | GPU {spec.gpu}", unit="answer",
                    position=progress_position, dynamic_ncols=True, leave=False)
    progress.set_postfix(stage="starting")
    try:
        return_code = stream_inference(spec, env, progress, lock, error_tail)
    finally:
        progress.close()

    validation_error = None
    if return_code == 0:
        spec.stage = "validating_output"
        tqdm.write(f"[fold {spec.fold}] checking output completeness...")
        try:
            validate_generated_output(spec, expected_samples)
        except Exception as exc:
            validation_error = str(exc)
    with lock:
        spec.return_code = return_code
        spec.elapsed_seconds = time.monotonic() - started
        spec.finished_at = utc_now()
        spec.status = "succeeded" if return_code == 0 and validation_error is None else "failed"
        spec.error = validation_error or (
            None if return_code == 0 else f"Inference exited with code {return_code}"
        )
        spec.stage = "completed" if spec.status == "succeeded" else "failed"
        write_json(output_dir / "run_status.json", asdict(spec))
    tqdm.write(f"[fold {spec.fold}] {spec.status} on GPU {spec.gpu}; "
               f"answers={spec.completed_generations}/{spec.total_generations}; "
               f"elapsed={spec.elapsed_seconds / 60:.1f} min")
    if spec.error:
        tqdm.write(f"[fold {spec.fold}] {spec.error}; log: {spec.log_path}")
        if return_code != 0:
            for line in error_tail:
                tqdm.write(f"[fold {spec.fold}] {line}")
    return spec


def stream_inference(spec, env, progress, lock, error_tail) -> int:
    """Keep full worker output in its log and display structured progress."""
    with Path(spec.log_path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Command: {json.dumps(spec.command, ensure_ascii=False)}\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={spec.gpu}\n\n")
        handle.flush()
        process = subprocess.Popen(
            spec.command,
            cwd=str(REPO_ROOT),
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
            handle.write(line)
            handle.flush()
            event = parse_progress(line)
            if event is None:
                if line.strip():
                    error_tail.append(line.rstrip())
                continue
            stage = event["stage"]
            if stage != spec.stage:
                spec.stage = stage
                tqdm.write(f"[fold {spec.fold}] {stage.replace('_', ' ')}")
                with lock:
                    write_json(Path(spec.output_dir) / "run_status.json", asdict(spec))
            completed = int(event.get("completed", spec.completed_generations))
            spec.completed_generations = completed
            progress.update(max(0, completed - progress.n))
            details = {"stage": stage}
            if "problem_id" in event:
                details.update(problem=event["problem_id"], variant=event["variant"])
            progress.set_postfix(details, refresh=False)
        process.stdout.close()
        return process.wait()


def run_dynamic_schedule(
    specs: list[FoldSpec],
    gpu_ids: list[str],
    expected_samples: int,
) -> list[FoldSpec]:
    pending: queue.Queue[FoldSpec] = queue.Queue()
    for spec in specs:
        pending.put(spec)
    state_lock = threading.Lock()
    completed: list[FoldSpec] = []
    completed_lock = threading.Lock()

    def worker(gpu: str, position: int) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            result = spec
            try:
                spec.gpu = gpu
                result = run_fold(spec, expected_samples, state_lock, position)
            except Exception as exc:
                spec.status = "failed"
                spec.return_code = -1
                spec.error = repr(exc)
                spec.finished_at = utc_now()
                spec.stage = "failed"
                try:
                    write_json(Path(spec.output_dir) / "run_status.json", asdict(spec))
                except OSError:
                    pass
                tqdm.write(f"[fold {spec.fold}] failed: {exc}; log: {spec.log_path}")
                result = spec
            finally:
                with completed_lock:
                    completed.append(result)
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, position), name=f"evaluate-gpu-{gpu}")
        for position, gpu in enumerate(gpu_ids[: min(len(gpu_ids), len(specs))])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


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
            if spec.method == "dspr":
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


def checkpoint_artifact(directory: Path, method: str) -> Path | None:
    if method == "lora":
        if (directory / "adapter_config.json").is_file() and any(
            (directory / filename).is_file() for filename in ADAPTER_FILES
        ):
            return directory
        return None
    path = directory / CHECKPOINT_FILES[method]
    return path if path.is_file() else None


def checkpoint_metadata(path: Path, metadata: dict[str, Any], method: str) -> dict[str, Any]:
    if method != "spt":
        return metadata
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    prompt = checkpoint.get("soft_prompt")
    if not isinstance(prompt, torch.Tensor) or prompt.ndim != 2 or min(prompt.shape) < 1:
        raise ValueError(f"SPT checkpoint has no valid soft prompt: {path}")
    prefix_length = int(prompt.shape[0])
    configured = checkpoint.get("spt_config", {}).get("prefix_length")
    if configured is not None and int(configured) != prefix_length:
        raise ValueError(f"SPT checkpoint prompt shape differs from its config: {path}")
    return {**metadata, "prefix_length": prefix_length, "hidden_size": int(prompt.shape[1])}


def resolve_checkpoint(
    fold_root: Path, mode: str = "best", method: str = "dspr",
) -> tuple[Path, dict[str, Any]]:
    """Select best/latest retained weights, repairing moved trainer-state paths."""
    if method == "dspr" and (fold_root / "prefix").is_dir():
        # Router and prefix stages have independent step counters.
        return resolve_checkpoint(fold_root / "prefix", mode, method)
    directories = sorted(
        (path for path in fold_root.glob("checkpoint-*") if path.is_dir()), key=checkpoint_step,
    )
    available = [path for path in directories if checkpoint_artifact(path, method) is not None]
    direct = checkpoint_artifact(fold_root, method)
    direct_selection = "direct_fold_adapter" if method == "lora" else "direct_fold_checkpoint"

    def selected(directory: Path, metadata: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        path = checkpoint_artifact(directory, method)
        assert path is not None
        return path, checkpoint_metadata(path, metadata, method)

    if mode == "latest":
        if available:
            directory = available[-1]
            return selected(directory, {
                "method": "latest_checkpoint_step", "checkpoint_step": checkpoint_step(directory),
            })
        if direct is not None:
            return selected(fold_root, {"method": direct_selection})
        raise FileNotFoundError(f"No {method.upper()} checkpoint found under {fold_root}")

    state_files = [path / "trainer_state.json" for path in reversed(directories)]
    if method != "dspr":
        state_files.insert(0, fold_root / "trainer_state.json")
    for state_path in state_files:
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        best_value = state.get("best_model_checkpoint")
        if not best_value:
            continue
        best_name = re.split(r"[\\/]", str(best_value).rstrip("\\/"))[-1]
        for candidate in (Path(str(best_value)), fold_root / best_name):
            if checkpoint_artifact(candidate, method) is not None:
                return selected(candidate, {
                    "method": "trainer_state.best_model_checkpoint", "trainer_state": str(state_path),
                    "recorded_best_path": str(best_value), "best_metric": state.get("best_metric"),
                    "checkpoint_step": checkpoint_step(candidate),
                })
    scored = []
    for directory in available:
        state_path = directory / "trainer_state.json"
        if not state_path.is_file():
            continue
        loss = eval_loss_for_checkpoint(read_json(state_path), checkpoint_step(directory))
        if loss is not None and math.isfinite(loss):
            scored.append((loss, checkpoint_step(directory), directory, state_path))
    if scored:
        loss, step, directory, state_path = min(scored, key=lambda item: (item[0], item[1]))
        return selected(directory, {
            "method": "minimum_retained_eval_loss", "trainer_state": str(state_path),
            "best_metric": loss, "checkpoint_step": step,
        })
    if direct is not None:
        return selected(fold_root, {"method": direct_selection})
    if available:
        directory = available[-1]
        return selected(directory, {
            "method": "latest_checkpoint_fallback_no_eval_state", "checkpoint_step": checkpoint_step(directory),
        })
    raise FileNotFoundError(f"No {method.upper()} checkpoint found under {fold_root}")


def build_inference_command(
    inference_script: Path, spec: FoldSpec, args: argparse.Namespace, method: str = "dspr",
) -> list[str]:
    command = [
        sys.executable, str(inference_script), "--baseline", method,
        "--backend", "transformers", "--checkpoint", spec.checkpoint,
        "--model_name", args.model_name, "--data_path", spec.data_path,
        "--output_dir", spec.output_dir, "--temperature", str(args.temperature),
        "--top_p", str(args.top_p), "--n", str(args.samples_per_problem),
        "--max_new_tokens", str(args.max_new_tokens), "--device", "cuda", "--emit-progress-events",
    ]
    if method == "dspr":
        for name in ("context_layer_idx", "prefix_length", "router_intermediate_dim", "router_dropout", "max_seq_length"):
            command.extend([f"--{name}", str(getattr(args, name))])
    elif method == "spt":
        prefix_length = spec.checkpoint_selection["prefix_length"]
        if args.prefix_length is not None and args.prefix_length != prefix_length:
            raise ValueError(f"SPT checkpoint prefix length is {prefix_length}; requested {args.prefix_length}")
        command.extend(["--prefix_length", str(prefix_length)])
    if args.problem_id is not None:
        command.extend(["--id", str(args.problem_id)])
    return command


def parse_args(method: str = "dspr", argv: list[str] | None = None) -> argparse.Namespace:
    if method not in DEFAULT_CHECKPOINT_ROOTS:
        raise ValueError(f"Unsupported evaluation method: {method}")
    parser = argparse.ArgumentParser(description=f"Run held-out {method.upper()} inference")
    if method != "dspr":
        parser.set_defaults(baseline=method)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOTS[method])
    parser.add_argument("--test-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--id-root", default="output/qwen-2.5/kfold")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOTS[method])
    parser.add_argument("--inference-script", default="scripts/inference.py")
    parser.add_argument("--checkpoint-selection", choices=("best", "latest"), default="best")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="Physical GPU IDs, e.g. 0,1; default: auto-detect")
    parser.add_argument("--problem-id", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prefix-length", type=int, default=15 if method == "dspr" else None)
    if method == "dspr":
        parser.add_argument("--context-layer-idx", type=int, default=15)
        parser.add_argument("--router-intermediate-dim", type=int, default=64)
        parser.add_argument("--router-dropout", type=float, default=0.10)
        parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args(argv)


def main(method: str = "dspr", argv: list[str] | None = None) -> int:
    args = parse_args(method, argv)
    if args.temperature != 0.0 or args.top_p != 1.0 or args.samples_per_problem != 1:
        raise ValueError(
            "Held-out evaluation is deterministic-only: require temperature=0, top_p=1, samples_per_problem=1"
        )
    folds = parse_fold_list(args.folds)
    if args.problem_id is not None and len(folds) != 1:
        raise ValueError("--problem-id requires exactly one selected fold")
    print(f"{method.upper()} held-out evaluation: model={args.model_name}; folds={folds}", flush=True)
    print("Resolving GPUs and checkpoints...", flush=True)
    gpu_ids = parse_gpu_list(args.gpus)
    checkpoint_root = resolve_path(REPO_ROOT, args.checkpoint_root).resolve()
    test_root = resolve_path(REPO_ROOT, args.test_root).resolve()
    id_root = resolve_path(REPO_ROOT, args.id_root).resolve()
    output_root = resolve_path(REPO_ROOT, args.output_root).resolve()
    inference_script = resolve_path(REPO_ROOT, args.inference_script).resolve()
    if not inference_script.is_file():
        raise FileNotFoundError(f"Inference entrypoint not found: {inference_script}")
    print(f"GPUs: {gpu_ids}; checkpoint selection: {args.checkpoint_selection}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    specs = []
    for index, fold in enumerate(folds):
        checkpoint, selection = resolve_checkpoint(checkpoint_root / f"fold_{fold}", args.checkpoint_selection, method)
        all_expected_ids = expected_ids_for_fold(id_root, fold)
        if args.problem_id is not None:
            if args.problem_id not in all_expected_ids:
                raise ValueError(f"Problem {args.problem_id} is not held out in Fold {fold}")
            expected_ids = [args.problem_id]
        else:
            expected_ids = all_expected_ids
        data_path = test_root / f"fold_{fold}" / "test.jsonl"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} test data: {data_path}")
        raw_ids = [normalise_id(record.get("problem_id")) for record in read_jsonl(data_path)]
        if len(raw_ids) != len(set(raw_ids)) or set(raw_ids) != set(all_expected_ids):
            raise ValueError(f"Fold {fold} raw test IDs differ from test_ids.json")
        output_dir = output_root / f"fold_{fold}"
        if output_is_nonempty(output_dir) and not (args.allow_existing or args.dry_run):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        spec = FoldSpec(
            fold=fold, gpu=gpu_ids[index % len(gpu_ids)], checkpoint=str(checkpoint.resolve()),
            checkpoint_selection=selection, data_path=str(data_path.resolve()), expected_problem_ids=expected_ids,
            output_dir=str(output_dir.resolve()), log_path=str((output_dir / "inference.log").resolve()),
            command=[], method=method,
        )
        spec.command = build_inference_command(inference_script, spec, args, method)
        specs.append(spec)
        print(f"[fold {fold}] ready: {len(expected_ids)} problems; "
              f"checkpoint={checkpoint}; selection={selection['method']}", flush=True)

    manifest = {
        "created_at": utc_now(), "repo_root": str(REPO_ROOT), "method": method, "model_name": args.model_name,
        "checkpoint_root": str(checkpoint_root), "test_root": str(test_root), "id_root": str(id_root),
        "output_root": str(output_root), "gpus": gpu_ids, "folds": [asdict(spec) for spec in specs],
        "settings": vars(args),
    }
    if args.dry_run:
        print("Dry run: no inference processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0
    manifest_path = output_root / "inference_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Starting {len(specs)} folds on {min(len(gpu_ids), len(specs))} GPUs. Manifest: {manifest_path}", flush=True)
    results = run_dynamic_schedule(specs, gpu_ids, args.samples_per_problem)
    manifest.update(finished_at=utc_now(), folds=[asdict(spec) for spec in results])
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
    print(f"All {len(results)} {method.upper()} fold inference jobs passed validation.")
    print(f"Manifest: {manifest_path}")
    return 0
