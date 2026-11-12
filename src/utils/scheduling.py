"""GPU discovery and a shared fold queue for training and generation."""

import os
import queue
import subprocess
import threading


def run_folds(specs, gpu_ids, run_fold, on_error):
    """One worker per GPU claims folds in order; failures leave other folds running.

    run_fold receives the spec, a shared state_lock and a stable progress_position.
    Each caller supplies its own progress handling and failure/status recording.
    """
    pending = queue.Queue()
    for spec in specs:
        pending.put(spec)
    state_lock = threading.Lock()
    completed = []

    def worker(gpu, position):
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            # Assign on claim, so two workers never use the same GPU concurrently.
            spec.gpu = gpu
            result = spec
            try:
                result = run_fold(spec, state_lock=state_lock, progress_position=position)
            except Exception as exc:
                with state_lock:
                    on_error(spec, exc)
            finally:
                with state_lock:
                    completed.append(result)
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, position), name=f"fold-gpu-{gpu}")
        for position, gpu in enumerate(gpu_ids[:min(len(gpu_ids), len(specs))])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(completed, key=lambda item: item.fold)


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
    """Use the explicit CUDA mask or the GPU indices reported by nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True,
    )
    gpu_ids = output.split()
    if not gpu_ids:
        raise RuntimeError("No GPUs available; set --gpus explicitly for a dry run")
    return gpu_ids
