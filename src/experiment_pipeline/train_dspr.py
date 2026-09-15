"""
Training script for DSPR model.
"""

import argparse
import inspect
import json
import math
import os
import random
import sys
from dataclasses import fields
import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from dspr.config import DSPRConfig
from dspr.model import DSPRModel
from dspr_training.dataset import DSPRDataset, ProblemResampledDSPRDataset
from dspr_training.trainer import DSPRTrainer, compute_dspr_metrics


PROGRESS_EVENT_PREFIX = "__DSPR_TRAIN_PROGRESS__"
TRAIN_LOG_PREFIX = "__DSPR_TRAIN_LOG__"


class ProgressEventCallback(TrainerCallback):
    """Emit machine-readable optimizer-step progress for the K-fold parent."""

    @staticmethod
    def _emit(event: str, state) -> None:
        payload = {
            "event": event,
            "step": int(state.global_step),
            "total_steps": int(state.max_steps),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "best_metric": getattr(state, "best_metric", None),
            "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
        }
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_steps_per_epoch(dataset_size: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    """Match Trainer's update-step rounding for a finite map-style dataset."""
    if dataset_size <= 0:
        raise ValueError("Training dataset must contain at least one record")
    if batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    dataloader_batches = math.ceil(dataset_size / batch_size)
    return max(math.ceil(dataloader_batches / gradient_accumulation_steps), 1)


def parse_args():
    default_config = DSPRConfig()

    parser = argparse.ArgumentParser(description="Train DSPR model with configurable hyperparameters.")
    parser.add_argument("--model_name", type=str, default=default_config.model_name)
    parser.add_argument("--context_layer_idx", type=int, default=default_config.context_layer_idx)
    parser.add_argument("--prefix_length", type=int, default=default_config.prefix_length)
    parser.add_argument("--router_intermediate_dim", type=int, default=default_config.router_intermediate_dim)
    parser.add_argument("--router_dropout", type=float, default=default_config.router_dropout)
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate)
    parser.add_argument(
        "--prefix_learning_rate",
        type=float,
        default=None,
        help="Dual-prefix learning rate; defaults to --learning_rate.",
    )
    parser.add_argument(
        "--router_learning_rate",
        type=float,
        default=None,
        help="Router learning rate; defaults to --learning_rate.",
    )
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=default_config.num_epochs)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=default_config.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=default_config.max_grad_norm)
    parser.add_argument("--warmup_steps", type=int, default=default_config.warmup_steps)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Warmup fraction in [0, 1). When set, it replaces --warmup_steps.",
    )
    parser.add_argument("--lambda_router", type=float, default=default_config.lambda_router)
    parser.add_argument("--train_data_path", type=str, default=default_config.train_data_path)
    parser.add_argument("--val_data_path", type=str, default=default_config.val_data_path)
    parser.add_argument("--max_seq_length", type=int, default=default_config.max_seq_length)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--device", type=str, default=default_config.device)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=default_config.gradient_checkpointing,
    )
    parser.add_argument("--output_path", type=str, default=default_config.checkpoint_dir)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Number of evaluations without improvement before stopping; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum eval-loss improvement counted by early stopping.",
    )
    parser.add_argument(
        "--emit_progress_events",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trajectory_sampling",
        choices=("flat", "random_one"),
        default="flat",
        help="Use all flattened trajectories or resample one trajectory per problem each sampling epoch.",
    )
    parser.add_argument("--resume_trainable_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive")
    if args.prefix_learning_rate is not None and args.prefix_learning_rate <= 0.0:
        raise ValueError("--prefix_learning_rate must be positive")
    if args.router_learning_rate is not None and args.router_learning_rate <= 0.0:
        raise ValueError("--router_learning_rate must be positive")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    if args.early_stopping_threshold < 0.0:
        raise ValueError("--early_stopping_threshold must be non-negative")

    config_field_names = {f.name for f in fields(DSPRConfig)}
    config_kwargs = {k: v for k, v in vars(args).items() if k in config_field_names}
    config = DSPRConfig(**config_kwargs)
    prefix_learning_rate = (
        config.learning_rate
        if args.prefix_learning_rate is None
        else args.prefix_learning_rate
    )
    router_learning_rate = (
        config.learning_rate
        if args.router_learning_rate is None
        else args.router_learning_rate
    )

    # Set seed for reproducibility
    set_seed(config.seed)

    # Initialize model
    print("Loading DSPR model...")
    model = DSPRModel(config)

    # Load datasets
    print("Loading datasets...")
    flat_train_dataset = DSPRDataset(config.train_data_path, model.tokenizer, config.max_seq_length)
    train_dataset = flat_train_dataset
    resampled_dataset = None
    flat_steps_per_epoch = update_steps_per_epoch(
        len(flat_train_dataset),
        config.batch_size,
        config.gradient_accumulation_steps,
    )
    if args.trajectory_sampling == "random_one":
        resampled_dataset = ProblemResampledDSPRDataset(
            config.train_data_path,
            model.tokenizer,
            config.max_seq_length,
            seed=config.seed,
        )
        train_dataset = resampled_dataset
        print(
            "Trajectory sampling: random_one "
            f"({resampled_dataset.num_problems} problems, "
            f"{resampled_dataset.num_trajectories} flat trajectories; "
            "one trajectory per problem per sampling epoch)",
            flush=True,
        )
    else:
        print(
            f"Trajectory sampling: flat ({len(flat_train_dataset)} trajectories)",
            flush=True,
        )
    val_dataset = DSPRDataset(config.val_data_path, model.tokenizer, config.max_seq_length)

    eval_strategy = "epoch" if len(val_dataset) > 0 else "no"
    save_strategy = "epoch" if len(val_dataset) > 0 else "no"
    if args.early_stopping_patience > 0 and eval_strategy == "no":
        raise ValueError("Early stopping requires a non-empty validation dataset")

    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else 0.0
    warmup_steps = 0 if args.warmup_ratio is not None else config.warmup_steps
    max_steps = -1
    if resampled_dataset is not None:
        # The resampled dataloader has one item per problem, but the run keeps
        # the same optimizer update budget as flat training.
        max_steps = flat_steps_per_epoch * math.ceil(config.num_epochs)
        if len(val_dataset) > 0:
            # Evaluate/save at the same step cadence as one flat-data epoch.
            eval_strategy = "steps"
            save_strategy = "steps"
        print(
            f"Preserving flat training budget: {max_steps} optimizer steps "
            f"({flat_steps_per_epoch} steps per flat-data epoch)",
            flush=True,
        )
    training_arg_kwargs = dict(
        output_dir=args.output_path,
        learning_rate=prefix_learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=flat_steps_per_epoch if resampled_dataset is not None else None,
        save_steps=flat_steps_per_epoch if resampled_dataset is not None else 500,
        save_strategy=save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        disable_tqdm=args.emit_progress_events,
        fp16=(config.device.startswith("cuda") and torch.cuda.is_available()),
        gradient_checkpointing=False,
        label_names=["labels", "target_alpha"],
        seed=config.seed,
    )
    # Transformers renamed this argument in recent releases.  Accept both the
    # experiment environment's older version and current local installations.
    strategy_argument = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_arg_kwargs[strategy_argument] = eval_strategy
    training_args = TrainingArguments(**training_arg_kwargs)

    callbacks = [
        TrainingHistoryCallback(
            write_train_log=not args.emit_progress_events,
            resampled_dataset=resampled_dataset,
        )
    ]
    if args.emit_progress_events:
        callbacks.append(ProgressEventCallback())
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    # Initialize trainer
    trainer = DSPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_dspr_metrics if len(val_dataset) > 0 else None,
        lambda_router=config.lambda_router,
        prefix_learning_rate=prefix_learning_rate,
        router_learning_rate=router_learning_rate,
        callbacks=callbacks,
    )

    print(
        "Learning rates: "
        f"prefix={prefix_learning_rate:g}, router={router_learning_rate:g}",
        flush=True,
    )

    if args.resume_trainable_checkpoint:
        print(f"Loading trainable checkpoint from {args.resume_trainable_checkpoint}")
        trainer.load_trainable_checkpoint(args.resume_trainable_checkpoint)

    # Train
    print("Starting training...", flush=True)
    train_output = trainer.train()
    termination = (
        "early_stopping"
        if trainer.state.global_step < trainer.state.max_steps
        else "completed"
    )
    print(
        "Training finished: "
        f"termination={termination}, "
        f"global_step={trainer.state.global_step}, "
        f"max_steps={trainer.state.max_steps}, "
        f"best_metric={getattr(trainer.state, 'best_metric', None)}, "
        f"best_model_checkpoint={getattr(trainer.state, 'best_model_checkpoint', None)}",
        flush=True,
    )
    print(f"Training metrics: {train_output.metrics}", flush=True)


if __name__ == "__main__":
    main()
