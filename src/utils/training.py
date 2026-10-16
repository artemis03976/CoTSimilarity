"""Shared Trainer callbacks, log persistence, and optimizer-step budgeting."""

from __future__ import annotations

import json
import math
import os
import sys

from transformers import TrainerCallback

from .training_progress import PROGRESS_EVENT_PREFIX, TRAIN_LOG_PREFIX


class ProgressEventCallback(TrainerCallback):
    """Emit machine-readable optimizer-step progress for the K-fold parent."""

    def __init__(self, stage: str | None = None):
        self.stage = stage

    def _emit(self, event: str, state) -> None:
        payload = {
            "event": event,
            "step": int(state.global_step),
            "total_steps": int(state.max_steps),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "best_metric": getattr(state, "best_metric", None),
            "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
        }
        if self.stage is not None:
            payload["stage"] = self.stage
        print(
            PROGRESS_EVENT_PREFIX + json.dumps(payload, separators=(",", ":")),
            file=sys.stdout,
            flush=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._emit("start", state)

    def on_step_end(self, args, state, control, **kwargs):
        self._emit("step", state)

    def on_train_end(self, args, state, control, **kwargs):
        event = "early_stop" if state.global_step < state.max_steps else "complete"
        self._emit(event, state)


class TrainingHistoryCallback(TrainerCallback):
    """Persist every Trainer log and expose it to the K-fold parent process."""

    def __init__(self, write_train_log: bool, resampled_dataset=None):
        self.write_train_log = write_train_log
        self.resampled_dataset = resampled_dataset
        self.history_handle = None
        self.train_log_handle = None

    def on_train_begin(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.resampled_dataset is not None:
            self.resampled_dataset.set_epoch(0)
        self.history_handle = open(
            os.path.join(output_dir, "training_history.jsonl"),
            "w",
            encoding="utf-8",
        )
        if self.write_train_log:
            self.train_log_handle = open(
                os.path.join(output_dir, "train.log"),
                "w",
                encoding="utf-8",
            )

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.resampled_dataset is not None:
            epoch = int(state.epoch) if state.epoch is not None else 0
            self.resampled_dataset.set_epoch(epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "logs": logs,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        if self.history_handle is not None:
            self.history_handle.write(line + "\n")
            self.history_handle.flush()

        # The parent K-fold process preserves this stdout line in fold_N/train.log.
        output_line = TRAIN_LOG_PREFIX + line
        print(output_line, file=sys.stdout, flush=True)
        if self.train_log_handle is not None:
            self.train_log_handle.write(line + "\n")
            self.train_log_handle.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for handle in (self.history_handle, self.train_log_handle):
            if handle is not None:
                handle.close()
        self.history_handle = None
        self.train_log_handle = None


def update_steps_per_epoch(dataset_size: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    """Match Trainer's update-step rounding for a finite map-style dataset."""
    if dataset_size <= 0:
        raise ValueError("Training dataset must contain at least one record")
    if batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    dataloader_batches = math.ceil(dataset_size / batch_size)
    return max(math.ceil(dataloader_batches / gradient_accumulation_steps), 1)
