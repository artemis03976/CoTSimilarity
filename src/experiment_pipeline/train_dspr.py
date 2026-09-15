"""
Training script for DSPR model.
"""

import argparse
import inspect
import json
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
from dspr_training.dataset import DSPRDataset
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

    def __init__(self, write_train_log: bool):
        self.write_train_log = write_train_log
        self.history_handle = None
        self.train_log_handle = None

    def on_train_begin(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
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


def parse_args():
    default_config = DSPRConfig()

    parser = argparse.ArgumentParser(description="Train DSPR model with configurable hyperparameters.")
    parser.add_argument("--model_name", type=str, default=default_config.model_name)
    parser.add_argument("--context_layer_idx", type=int, default=default_config.context_layer_idx)
    parser.add_argument("--prefix_length", type=int, default=default_config.prefix_length)
    parser.add_argument("--router_intermediate_dim", type=int, default=default_config.router_intermediate_dim)
    parser.add_argument("--router_dropout", type=float, default=default_config.router_dropout)
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate)
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
    parser.add_argument("--resume_trainable_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    if args.early_stopping_threshold < 0.0:
        raise ValueError("--early_stopping_threshold must be non-negative")

    config_field_names = {f.name for f in fields(DSPRConfig)}
    config_kwargs = {k: v for k, v in vars(args).items() if k in config_field_names}
    config = DSPRConfig(**config_kwargs)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Initialize model
    print("Loading DSPR model...")
    model = DSPRModel(config)

    # Load datasets
    print("Loading datasets...")
    train_dataset = DSPRDataset(config.train_data_path, model.tokenizer, config.max_seq_length)
    val_dataset = DSPRDataset(config.val_data_path, model.tokenizer, config.max_seq_length)

    eval_strategy = "epoch" if len(val_dataset) > 0 else "no"
    save_strategy = "epoch" if len(val_dataset) > 0 else "no"
    if args.early_stopping_patience > 0 and eval_strategy == "no":
        raise ValueError("Early stopping requires a non-empty validation dataset")

    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else 0.0
    warmup_steps = 0 if args.warmup_ratio is not None else config.warmup_steps
    training_arg_kwargs = dict(
        output_dir=args.output_path,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
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

    callbacks = [TrainingHistoryCallback(write_train_log=not args.emit_progress_events)]
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
        callbacks=callbacks,
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
