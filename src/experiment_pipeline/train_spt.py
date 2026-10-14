"""Train one static prompt on the same trajectories as the DSPR prefix stage."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import fields
from pathlib import Path

from spt.config import SPTConfig


def parse_args() -> argparse.Namespace:
    config = SPTConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=config.model_name)
    parser.add_argument("--prefix_length", type=int, default=config.prefix_length)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_epochs", type=float, default=config.num_epochs)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup_steps", type=int, default=None)
    warmup.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--train_data_path", default=config.train_data_path)
    parser.add_argument("--val_data_path", default=config.val_data_path)
    parser.add_argument("--max_seq_length", type=int, default=config.max_seq_length)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                        default=config.gradient_checkpointing)
    parser.add_argument("--output_path", default="checkpoints_baseline")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--trajectory_sampling", choices=("flat", "random_one"), default="flat")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--resume_trainable_checkpoint", default=None)
    args = parser.parse_args()
    if args.warmup_steps is None and args.warmup_ratio is None:
        args.warmup_ratio = 0.05
    return args


def main() -> int:
    args = parse_args()
    if min(args.prefix_length, args.batch_size, args.gradient_accumulation_steps,
           args.max_seq_length, args.num_epochs) <= 0:
        raise ValueError("Prompt length, batch size, accumulation, sequence length and epochs must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_threshold < 0:
        raise ValueError("Early-stopping patience and threshold must be non-negative")
    if args.warmup_ratio is not None and not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    import numpy as np
    import torch
    from transformers import EarlyStoppingCallback, TrainingArguments, default_data_collator

    from dspr_training.dataset import DSPRDataset, ProblemResampledDSPRDataset
    from experiment_pipeline.train_dspr import ProgressEventCallback, TrainingHistoryCallback, update_steps_per_epoch
    from spt import StaticPromptModel
    from spt_training import BaselineTrainer

    config_fields = {field.name for field in fields(SPTConfig)}
    config = SPTConfig(**{key: value for key, value in vars(args).items() if key in config_fields
                          and value is not None})
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading StaticPromptModel baseline...")
    model = StaticPromptModel(config)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} (soft prompt only)")

    flat_train = DSPRDataset(config.train_data_path, model.tokenizer, config.max_seq_length)
    flat_steps = update_steps_per_epoch(len(flat_train), config.batch_size, config.gradient_accumulation_steps)
    train_dataset = flat_train
    resampled = None
    if args.trajectory_sampling == "random_one":
        resampled = ProblemResampledDSPRDataset(config.train_data_path, model.tokenizer,
                                               config.max_seq_length, seed=config.seed)
        train_dataset = resampled
    val_dataset = DSPRDataset(config.val_data_path, model.tokenizer, config.max_seq_length)
    has_validation = len(val_dataset) > 0
    if args.early_stopping_patience > 0 and not has_validation:
        raise ValueError("Early stopping requires a non-empty validation dataset")
    eval_strategy = ("steps" if resampled is not None else "epoch") if has_validation else "no"
    kwargs = dict(
        output_dir=args.output_path,
        learning_rate=config.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        max_steps=flat_steps * math.ceil(config.num_epochs) if resampled is not None else -1,
        warmup_steps=args.warmup_steps or 0,
        warmup_ratio=args.warmup_ratio or 0.0,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy=eval_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=config.device.startswith("cuda"),
        report_to="none",
        disable_tqdm=True,
        fp16=(config.device.startswith("cuda") and torch.cuda.is_available()),
        gradient_checkpointing=False,
        label_names=["labels"],
        seed=config.seed,
    )
    if resampled is not None and has_validation:
        kwargs.update(eval_steps=flat_steps, save_steps=flat_steps)
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = eval_strategy
    if config.device == "cpu":
        kwargs["use_cpu" if "use_cpu" in parameters else "no_cuda"] = True
    training_args = TrainingArguments(**kwargs)
    callbacks = [TrainingHistoryCallback(write_train_log=False, resampled_dataset=resampled),
                 ProgressEventCallback(stage="spt")]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience,
                                              early_stopping_threshold=args.early_stopping_threshold))
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "static_prompt_tuning",
        "model_name": config.model_name,
        "prefix_length": config.prefix_length,
        "trainable_parameters": trainable_params,
        "train_records": len(flat_train),
        "validation_records": len(val_dataset),
        "sampling_records": len(train_dataset),
        "training_arguments": vars(args),
    }
    (output_path / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trainer = BaselineTrainer(model=model, args=training_args, train_dataset=train_dataset,
                              eval_dataset=val_dataset if has_validation else None,
                              data_collator=default_data_collator, callbacks=callbacks)
    if args.resume_trainable_checkpoint:
        trainer.load_trainable_checkpoint(args.resume_trainable_checkpoint)
    trainer.train()
    trainer.save_model(str(output_path))
    trainer.save_state()
    print(f"Saved best/final soft prompt and trainer state under {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
