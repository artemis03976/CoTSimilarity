#!/usr/bin/env python3
"""Train a parameter-matched LoRA baseline on one prepared Qwen fold."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_DSPR_BUDGET = 337_025
DEFAULT_RANK = 1
DEFAULT_LORA_ALPHA = 2
DEFAULT_RUN_NAME = "qwen_lora_qv_r1_seed42"
DEFAULT_HIDDEN_SIZE = 3584
DEFAULT_KV_SIZE = 512
DEFAULT_LAYERS = 28


def expected_qv_lora_parameters(
    rank: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    kv_size: int = DEFAULT_KV_SIZE,
    layers: int = DEFAULT_LAYERS,
) -> int:
    """Return the LoRA A/B parameter count for q_proj and v_proj."""
    if rank < 1 or hidden_size < 1 or kv_size < 1 or layers < 1:
        raise ValueError("LoRA rank and model dimensions must be positive")
    per_layer = rank * ((hidden_size + hidden_size) + (hidden_size + kv_size))
    return layers * per_layer


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--val_data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=float, default=15)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attention_implementation", default="flash_attention_2")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--emit_progress_events", "--emit-progress-events", action="store_true",
        help="Send optimizer-step progress to the K-fold parent process",
    )
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--dspr_parameter_budget", type=int, default=DEFAULT_DSPR_BUDGET)
    parser.add_argument("--budget_tolerance", type=float, default=0.10)
    parser.add_argument(
        "--enforce_parameter_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable explicitly to train a larger, unmatched LoRA baseline",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rank < 1:
        raise ValueError("--rank must be positive")
    if args.lora_alpha < 1:
        raise ValueError("--lora_alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora_dropout must be in [0, 1)")
    if args.dspr_parameter_budget < 1 or not 0.0 <= args.budget_tolerance < 1.0:
        raise ValueError("Invalid parameter-budget settings")
    if args.warmup_steps < 0 or not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("Invalid warmup settings")
    if args.early_stopping_patience < 0 or args.early_stopping_threshold < 0:
        raise ValueError("Invalid early-stopping settings")

    try:
        import numpy as np
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training dependencies are missing. Install requirements.txt, including peft."
        ) from exc

    from dspr_training.dataset import DSPRDataset

    class LoRADataset(DSPRDataset):
        """Reuse DSPR tokenization while dropping routing-only tensors."""

        def __getitem__(self, index):
            item = super().__getitem__(index)
            for key in (
                "prompt_input_ids",
                "prompt_attention_mask",
                "target_alpha",
                "variant_type",
            ):
                item.pop(key, None)
            return item

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading base model for parameter-matched LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if args.device.startswith("cpu") else getattr(torch, args.dtype),
        trust_remote_code=True,
        attn_implementation=(
            "eager" if args.device.startswith("cpu") and args.attention_implementation == "flash_attention_2"
            else args.attention_implementation
        ),
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    budget_gap = (trainable_parameters - args.dspr_parameter_budget) / args.dspr_parameter_budget
    if args.enforce_parameter_budget and abs(budget_gap) > args.budget_tolerance:
        raise ValueError(
            "LoRA trainable-parameter budget is outside tolerance: "
            f"LoRA={trainable_parameters:,}, DSPR={args.dspr_parameter_budget:,}, "
            f"relative_gap={budget_gap:.2%}, tolerance={args.budget_tolerance:.2%}"
        )

    print(
        f"Trainable parameters: {trainable_parameters:,}; "
        f"DSPR reference: {args.dspr_parameter_budget:,}; gap: {budget_gap:.2%}"
    )
    model.print_trainable_parameters()

    train_dataset = LoRADataset(args.train_data_path, tokenizer, args.max_seq_length)
    val_dataset = LoRADataset(args.val_data_path, tokenizer, args.max_seq_length)
    eval_strategy = "epoch" if len(val_dataset) > 0 else "no"
    output_path = Path(args.output_path).resolve()
    budget_manifest = {
        "method": "parameter_matched_lora" if args.enforce_parameter_budget else "lora",
        "model_name": args.model_name,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(args.target_modules),
        "bias": "none",
        "quantized": False,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "dspr_reference_parameters": args.dspr_parameter_budget,
        "relative_budget_gap": budget_gap,
        "budget_tolerance": args.budget_tolerance,
        "parameter_budget_enforced": args.enforce_parameter_budget,
        "train_records": len(train_dataset),
        "validation_records": len(val_dataset),
        "training_arguments": vars(args),
    }
    write_json(output_path / "budget_manifest.json", budget_manifest)

    training_kwargs = dict(
        output_dir=str(output_path),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy=eval_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        disable_tqdm=args.emit_progress_events,
        fp16=(args.device.startswith("cuda") and args.dtype == "float16" and torch.cuda.is_available()),
        bf16=(args.device.startswith("cuda") and args.dtype == "bfloat16" and torch.cuda.is_available()),
        gradient_checkpointing=False,
        label_names=["labels"],
        seed=args.seed,
    )
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    strategy_argument = "eval_strategy" if "eval_strategy" in supported_args else "evaluation_strategy"
    training_kwargs[strategy_argument] = eval_strategy
    if args.device.startswith("cpu"):
        training_kwargs["use_cpu" if "use_cpu" in supported_args else "no_cuda"] = True
    training_args = TrainingArguments(**training_kwargs)
    callbacks = []
    if args.emit_progress_events:
        from utils.training import ProgressEventCallback

        callbacks.append(ProgressEventCallback(stage="lora"))
    if eval_strategy != "no" and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    print("Starting LoRA training...")
    trainer.train()
    trainer.save_model(str(output_path))
    trainer.save_state()
    print(f"Saved best/final adapter and state under {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
