#!/usr/bin/env python3
"""Run held-out LoRA inference through the shared greedy entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_dspr_kfold import (
    DEFAULT_FOLDS,
    DEFAULT_MODEL_NAME,
    checkpoint_step,
    eval_loss_for_checkpoint,
    expected_ids_for_fold,
    output_is_nonempty,
    parse_fold_list,
    parse_gpu_list,
    read_json,
    read_jsonl,
    resolve_path,
    utc_now,
    write_json,
)


VARIANTS = ("original", "simple", "hard")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def has_adapter(directory: Path) -> bool:
    return (directory / "adapter_config.json").is_file() and any(
        (directory / filename).is_file() for filename in ADAPTER_FILES
    )


def resolve_adapter_checkpoint(
    fold_root: Path,
    mode: str = "best",
) -> tuple[Path, dict[str, Any]]:
    """Resolve a PEFT adapter directory and repair moved trainer-state paths."""
    checkpoint_dirs = sorted(
        [path for path in fold_root.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    available = [path for path in checkpoint_dirs if has_adapter(path)]
    if mode == "latest":
        if available:
            selected = available[-1]
            return selected, {
                "method": "latest_checkpoint_step",
                "checkpoint_step": checkpoint_step(selected),
            }
        if has_adapter(fold_root):
            return fold_root, {"method": "direct_fold_adapter"}
        raise FileNotFoundError(f"No LoRA adapter found under {fold_root}")

    state_files = [fold_root / "trainer_state.json"] + [
        directory / "trainer_state.json" for directory in reversed(checkpoint_dirs)
    ]
    for state_path in state_files:
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        best_value = state.get("best_model_checkpoint")
        if not best_value:
            continue
        best_name = re.split(r"[\\/]", str(best_value).rstrip("\\/"))[-1]
        for candidate in (Path(str(best_value)), fold_root / best_name):
            if candidate.is_dir() and has_adapter(candidate):
                return candidate, {
                    "method": "trainer_state.best_model_checkpoint",
                    "trainer_state": str(state_path),
                    "recorded_best_path": str(best_value),
                    "best_metric": state.get("best_metric"),
                    "checkpoint_step": checkpoint_step(candidate),
                }

    scored: list[tuple[float, int, Path, Path]] = []
    for directory in available:
        state_path = directory / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        loss = eval_loss_for_checkpoint(state, checkpoint_step(directory))
        if loss is not None:
            scored.append((loss, checkpoint_step(directory), directory, state_path))
    if scored:
        loss, step, selected, state_path = min(scored, key=lambda item: (item[0], item[1]))
        return selected, {
            "method": "minimum_retained_eval_loss",
            "trainer_state": str(state_path),
            "best_metric": loss,
            "checkpoint_step": step,
        }
    if has_adapter(fold_root):
        return fold_root, {"method": "direct_fold_adapter"}
    if available:
        selected = available[-1]
        return selected, {
            "method": "latest_checkpoint_fallback_no_eval_state",
            "checkpoint_step": checkpoint_step(selected),
        }
    raise FileNotFoundError(f"No LoRA adapter found under {fold_root}")


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
        "lora",
        "--checkpoint",
        spec.checkpoint,
        "--model_name",
        args.model_name,
        "--data_path",
        spec.data_path,
        "--output_dir",
        spec.output_dir,
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
    ]
    if args.problem_id is not None:
        command.extend(["--id", str(args.problem_id)])
    return command


def validate_generated_output(spec: FoldSpec, expected_samples: int) -> dict[str, int]:
    path = Path(spec.output_dir) / "all_records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Inference result not found: {path}")
    records = read_jsonl(path)
    ids = [int(record.get("problem_id")) for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(spec.expected_problem_ids):
        missing = sorted(set(spec.expected_problem_ids) - set(ids))
        extra = sorted(set(ids) - set(spec.expected_problem_ids))
        raise ValueError(f"Fold {spec.fold} output ID mismatch: missing={missing}, extra={extra}")
    for record in records:
        problem_id = int(record["problem_id"])
        for variant in VARIANTS:
            data = record.get(variant)
            samples = data.get("samples") if isinstance(data, dict) else None
            if not isinstance(samples, list) or len(samples) != expected_samples:
                raise ValueError(
                    f"Problem {problem_id} {variant} sample count is invalid; "
                    f"expected {expected_samples}"
                )
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("correct"), bool):
                    raise ValueError(
                        f"Problem {problem_id} {variant} has invalid correctness"
                    )
    return {"records": len(records), "variants": len(records) * len(VARIANTS)}


def run_fold(
    spec: FoldSpec,
    expected_samples: int,
    lock: threading.Lock,
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
        spec.started_at = utc_now()
        write_json(output_dir / "run_status.json", asdict(spec))

    started = time.monotonic()
    with Path(spec.log_path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Command: {json.dumps(spec.command, ensure_ascii=False)}\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={spec.gpu}\n\n")
        handle.flush()
        process = subprocess.Popen(
            spec.command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()

    validation_error = None
    if return_code == 0:
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
        write_json(output_dir / "run_status.json", asdict(spec))
    print(f"[fold {spec.fold}] {spec.status} on GPU {spec.gpu}", flush=True)
    return spec


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

    def worker(gpu: str) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            try:
                spec.gpu = gpu
                result = run_fold(spec, expected_samples, state_lock)
            except Exception as exc:
                spec.status = "failed"
                spec.return_code = -1
                spec.error = repr(exc)
                spec.finished_at = utc_now()
                result = spec
            finally:
                with completed_lock:
                    completed.append(result)
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"lora-eval-gpu-{gpu}")
        for gpu in gpu_ids[: min(len(gpu_ids), len(specs))]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="checkpoints/qwen_lora_qv_r3_seed42")
    parser.add_argument("--test-root", default="data/kfold")
    parser.add_argument("--id-root", default="data/qwen/kfold")
    parser.add_argument("--output-root", default="output/qwen_lora_qv_r3_seed42")
    parser.add_argument("--inference-script", default="scripts/inference.py")
    parser.add_argument("--checkpoint-selection", choices=("best", "latest"), default="best")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--problem-id", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.temperature != 0.0 or args.top_p != 1.0 or args.samples_per_problem != 1:
        raise ValueError(
            "Held-out evaluation is deterministic-only: require temperature=0, "
            "top_p=1, samples_per_problem=1"
        )
    folds = parse_fold_list(args.folds)
    if args.problem_id is not None and len(folds) != 1:
        raise ValueError("--problem-id requires exactly one selected fold")
    gpu_ids = parse_gpu_list(args.gpus)
    checkpoint_root = resolve_path(REPO_ROOT, args.checkpoint_root).resolve()
    test_root = resolve_path(REPO_ROOT, args.test_root).resolve()
    id_root = resolve_path(REPO_ROOT, args.id_root).resolve()
    output_root = resolve_path(REPO_ROOT, args.output_root).resolve()
    inference_script = resolve_path(REPO_ROOT, args.inference_script).resolve()
    if not inference_script.is_file():
        raise FileNotFoundError(f"Inference entrypoint not found: {inference_script}")

    specs: list[FoldSpec] = []
    for index, fold in enumerate(folds):
        checkpoint, selection = resolve_adapter_checkpoint(
            checkpoint_root / f"fold_{fold}",
            args.checkpoint_selection,
        )
        all_expected_ids = expected_ids_for_fold(id_root, fold)
        if args.problem_id is not None:
            if args.problem_id not in all_expected_ids:
                raise ValueError(f"Problem {args.problem_id} is not held out in Fold {fold}")
            expected_ids = [args.problem_id]
        else:
            expected_ids = all_expected_ids
        data_path = test_root / f"fold_{fold}" / "test_raw.jsonl"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing Fold {fold} test data: {data_path}")
        raw_ids = [int(record["problem_id"]) for record in read_jsonl(data_path)]
        if len(raw_ids) != len(set(raw_ids)) or set(raw_ids) != set(all_expected_ids):
            raise ValueError(f"Fold {fold} raw test IDs differ from test_ids.json")
        output_dir = output_root / f"fold_{fold}"
        if output_is_nonempty(output_dir) and not (args.allow_existing or args.dry_run):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
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

    manifest = {
        "created_at": utc_now(),
        "method": "parameter_matched_lora",
        "model_name": args.model_name,
        "checkpoint_root": str(checkpoint_root),
        "test_root": str(test_root),
        "id_root": str(id_root),
        "output_root": str(output_root),
        "gpus": gpu_ids,
        "folds": [asdict(spec) for spec in specs],
        "settings": vars(args),
    }
    if args.dry_run:
        print("Dry run: no inference processes started.")
        for spec in specs:
            print(json.dumps(spec.command, ensure_ascii=False))
        return 0

    manifest_path = output_root / "inference_manifest.json"
    write_json(manifest_path, manifest)
    results = run_dynamic_schedule(specs, gpu_ids, args.samples_per_problem)
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
    print(f"All {len(results)} LoRA fold inference jobs passed validation.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
