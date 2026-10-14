"""Run held-out SPT inference using the shared baseline evaluator."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .evaluate_dspr import checkpoint_step, eval_loss_for_checkpoint, read_json
from . import evaluate_lora


CHECKPOINT_FILENAME = "baseline_trainable.pt"


def checkpoint_with_metadata(path: Path, metadata: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Infer architecture from the selected prompt, including older checkpoints."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    prompt = checkpoint.get("soft_prompt")
    if not isinstance(prompt, torch.Tensor) or prompt.ndim != 2 or min(prompt.shape) < 1:
        raise ValueError(f"SPT checkpoint has no valid soft prompt: {path}")
    prefix_length = int(prompt.shape[0])
    configured = checkpoint.get("spt_config", {}).get("prefix_length")
    if configured is not None and int(configured) != prefix_length:
        raise ValueError(f"SPT checkpoint prompt shape differs from its config: {path}")
    return path, {**metadata, "prefix_length": prefix_length, "hidden_size": int(prompt.shape[1])}


def resolve_spt_checkpoint(fold_root: Path, mode: str = "best") -> tuple[Path, dict[str, Any]]:
    """Resolve the best validation soft prompt, including moved checkpoints."""
    directories = sorted((path for path in fold_root.glob("checkpoint-*") if path.is_dir()),
                         key=checkpoint_step)
    available = [path for path in directories if (path / CHECKPOINT_FILENAME).is_file()]
    direct = fold_root / CHECKPOINT_FILENAME
    if mode == "latest":
        if available:
            selected = available[-1]
            return checkpoint_with_metadata(selected / CHECKPOINT_FILENAME, {"method": "latest_checkpoint_step",
                                                     "checkpoint_step": checkpoint_step(selected)})
        if direct.is_file():
            return checkpoint_with_metadata(direct, {"method": "direct_fold_checkpoint"})
        raise FileNotFoundError(f"No SPT checkpoint found under {fold_root}")
    state_files = [fold_root / "trainer_state.json"] + [path / "trainer_state.json" for path in reversed(directories)]
    for state_path in state_files:
        if not state_path.is_file():
            continue
        state = read_json(state_path)
        best_value = state.get("best_model_checkpoint")
        if not best_value:
            continue
        best_name = re.split(r"[\\/]", str(best_value).rstrip("\\/"))[-1]
        for candidate in (Path(str(best_value)), fold_root / best_name):
            path = candidate / CHECKPOINT_FILENAME
            if path.is_file():
                return checkpoint_with_metadata(path, {"method": "trainer_state.best_model_checkpoint", "trainer_state": str(state_path),
                              "recorded_best_path": str(best_value), "best_metric": state.get("best_metric"),
                              "checkpoint_step": checkpoint_step(candidate)})
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
        return checkpoint_with_metadata(directory / CHECKPOINT_FILENAME, {"method": "minimum_retained_eval_loss",
                                                "trainer_state": str(state_path), "best_metric": loss,
                                                "checkpoint_step": step})
    if direct.is_file():
        return checkpoint_with_metadata(direct, {"method": "direct_fold_checkpoint"})
    if available:
        selected = available[-1]
        return checkpoint_with_metadata(selected / CHECKPOINT_FILENAME, {"method": "latest_checkpoint_fallback_no_eval_state",
                                                 "checkpoint_step": checkpoint_step(selected)})
    raise FileNotFoundError(f"No SPT checkpoint found under {fold_root}")


def main() -> int:
    return evaluate_lora.main(baseline_name="spt", checkpoint_resolver=resolve_spt_checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
