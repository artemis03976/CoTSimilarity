import os
from typing import Any

import numpy as np
import torch
from transformers import Trainer

from .loss import DCPRLoss


class DCPRTrainer(Trainer):
    """HuggingFace Trainer for DCPR model."""

    def __init__(self, *args, lambda_router: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = DCPRLoss(lambda_router=lambda_router)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_alpha = inputs.pop("target_alpha")
        inputs.pop("variant_type", None)

        lm_loss, alpha = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            prompt_input_ids=inputs["prompt_input_ids"],
            prompt_attention_mask=inputs["prompt_attention_mask"],
        )

        total_loss, _ = self.loss_fn(lm_loss, alpha, target_alpha)
        if not return_outputs:
            return total_loss

        outputs = {"logits": alpha}
        return total_loss, outputs

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save only trainable DCPR parameters to avoid huge checkpoints of frozen base LLM.
        torch.save(
            {
                "dual_prefix_state_dict": model_to_save.dual_prefix.state_dict(),
                "router_state_dict": model_to_save.router.state_dict(),
                "training_args": self.args.to_dict(),
            },
            os.path.join(output_dir, "dcpr_trainable.pt"),
        )

    def load_trainable_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.args.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.dual_prefix.load_state_dict(ckpt["dual_prefix_state_dict"])
        model_to_load.router.load_state_dict(ckpt["router_state_dict"])


def compute_dcpr_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute router alpha diagnostics on validation set."""
    predictions, label_ids = eval_pred

    alpha = predictions
    if isinstance(alpha, tuple):
        alpha = alpha[0]
    alpha = np.asarray(alpha).reshape(-1)

    target_alpha = label_ids
    if isinstance(target_alpha, tuple):
        target_alpha = target_alpha[-1]
    target_alpha = np.asarray(target_alpha).reshape(-1)

    simple_mask = target_alpha < 0.5
    hard_mask = ~simple_mask

    alpha_simple_mean = float(alpha[simple_mask].mean()) if simple_mask.any() else 0.0
    alpha_hard_mean = float(alpha[hard_mask].mean()) if hard_mask.any() else 0.0
    alpha_gap = float(alpha_hard_mean - alpha_simple_mean)

    return {
        "alpha_simple_mean": alpha_simple_mean,
        "alpha_hard_mean": alpha_hard_mean,
        "alpha_gap": alpha_gap,
    }
