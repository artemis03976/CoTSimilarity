"""
Training script for StaticPromptModel baseline.

Uses the same DSPR dataset but trains only a single soft prompt
with LM loss (no router, no dual prefix).
"""

import argparse
import sys
from pathlib import Path
import random
from dataclasses import fields
import numpy as np
import torch
from transformers import TrainingArguments, default_data_collator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dspr_training.dataset import DSPRDataset
from spt import SPTConfig, StaticPromptModel
from spt_training import BaselineTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    default_config = SPTConfig()

    parser = argparse.ArgumentParser(description="Train StaticPromptModel baseline.")
    parser.add_argument("--model_name", type=str, default=default_config.model_name)
    parser.add_argument("--prefix_length", type=int, default=default_config.prefix_length)
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate)
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size)
    parser.add_argument("--num_epochs", type=int, default=default_config.num_epochs)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=default_config.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=default_config.max_grad_norm)
    parser.add_argument("--warmup_steps", type=int, default=default_config.warmup_steps)
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
    parser.add_argument("--output_path", type=str, default="checkpoints_baseline")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_trainable_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config_field_names = {f.name for f in fields(SPTConfig)}
    config_kwargs = {k: v for k, v in vars(args).items() if k in config_field_names}
    config = SPTConfig(**config_kwargs)

    set_seed(config.seed)

    print("Loading StaticPromptModel baseline...")
    model = StaticPromptModel(config)

    # Print trainable parameter count for comparison with DSPR
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} (soft_prompt only)")

    print("Loading datasets...")
    train_dataset = DSPRDataset(config.train_data_path, model.tokenizer, config.max_seq_length)
    val_dataset = DSPRDataset(config.val_data_path, model.tokenizer, config.max_seq_length)

    eval_strategy = "epoch" if len(val_dataset) > 0 else "no"
    save_strategy = "epoch" if len(val_dataset) > 0 else "no"
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        fp16=(config.device.startswith("cuda") and torch.cuda.is_available()),
        gradient_checkpointing=False,
        label_names=["labels"],
        seed=config.seed,
    )

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    if args.resume_trainable_checkpoint:
        print(f"Loading trainable checkpoint from {args.resume_trainable_checkpoint}")
        trainer.load_trainable_checkpoint(args.resume_trainable_checkpoint)

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
