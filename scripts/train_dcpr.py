"""
Training script for DCPR model.
"""

import argparse
import sys
import random
from dataclasses import fields
import numpy as np
import torch
from transformers import TrainingArguments, default_data_collator
sys.path.append('src')

from dcpr.config import DCPRConfig
from dcpr.model import DCPRModel
from dcpr_training.dataset import DCPRDataset
from dcpr_training.trainer import DCPRTrainer, compute_dcpr_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    default_config = DCPRConfig()

    parser = argparse.ArgumentParser(description="Train DCPR model with configurable hyperparameters.")
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
    parser.add_argument("--resume_trainable_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config_field_names = {f.name for f in fields(DCPRConfig)}
    config_kwargs = {k: v for k, v in vars(args).items() if k in config_field_names}
    config = DCPRConfig(**config_kwargs)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Initialize model
    print("Loading DCPR model...")
    model = DCPRModel(config)

    # Load datasets
    print("Loading datasets...")
    train_dataset = DCPRDataset(config.train_data_path, model.tokenizer, config.max_seq_length)
    val_dataset = DCPRDataset(config.val_data_path, model.tokenizer, config.max_seq_length)

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
        label_names=["labels", "target_alpha"],
        seed=config.seed,
    )

    # Initialize trainer
    trainer = DCPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_dcpr_metrics if len(val_dataset) > 0 else None,
        lambda_router=config.lambda_router,
    )

    if args.resume_trainable_checkpoint:
        print(f"Loading trainable checkpoint from {args.resume_trainable_checkpoint}")
        trainer.load_trainable_checkpoint(args.resume_trainable_checkpoint)

    # Train
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
