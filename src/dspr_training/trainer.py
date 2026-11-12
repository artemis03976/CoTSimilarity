import os
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from utils.training import stage_optimizer


logger = logging.getLogger(__name__)


def _empty_diagnostic_accumulator() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "lm_loss_sum": 0.0,
        "router_loss_sum": 0.0,
        "alpha": [],
        "target_alpha": [],
    }


def _binary_router_metrics(alpha: np.ndarray, target_alpha: np.ndarray) -> dict[str, float]:
    """Compute threshold and ranking diagnostics for the DSPR router."""
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    target_alpha = np.asarray(target_alpha, dtype=np.float64).reshape(-1)
    if alpha.size != target_alpha.size:
        raise ValueError(
            f"alpha and target_alpha must have the same size, got {alpha.size} and {target_alpha.size}"
        )

    simple_mask = target_alpha < 0.5
    hard_mask = ~simple_mask
    simple_count = int(simple_mask.sum())
    hard_count = int(hard_mask.sum())

    alpha_simple_mean = float(alpha[simple_mask].mean()) if simple_count else 0.0
    alpha_hard_mean = float(alpha[hard_mask].mean()) if hard_count else 0.0
    metrics = {
        "mean_alpha": float(alpha.mean()) if alpha.size else 0.0,
        "alpha_simple_mean": alpha_simple_mean,
        "alpha_hard_mean": alpha_hard_mean,
        "alpha_gap": float(alpha_hard_mean - alpha_simple_mean),
    }

    recalls: list[float] = []
    if simple_count:
        simple_recall = float((alpha[simple_mask] < 0.5).mean())
        metrics["router_simple_recall"] = simple_recall
        recalls.append(simple_recall)
    if hard_count:
        hard_recall = float((alpha[hard_mask] >= 0.5).mean())
        metrics["router_hard_recall"] = hard_recall
        recalls.append(hard_recall)
    if len(recalls) == 2:
        metrics["router_balanced_accuracy"] = float(np.mean(recalls))

    # Mann-Whitney formulation of ROC AUC, with average ranks for ties.  This
    # avoids adding a scikit-learn dependency to the training environment.
    if simple_count and hard_count:
        order = np.argsort(alpha, kind="mergesort")
        ranks = np.empty(alpha.size, dtype=np.float64)
        sorted_alpha = alpha[order]
        start = 0
        while start < alpha.size:
            end = start + 1
            while end < alpha.size and sorted_alpha[end] == sorted_alpha[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        hard_rank_sum = float(ranks[hard_mask].sum())
        metrics["router_auc"] = float(
            (hard_rank_sum - hard_count * (hard_count + 1) / 2.0)
            / (hard_count * simple_count)
        )

    return metrics


class DSPRTrainer(Trainer):
    """HuggingFace Trainer for DSPR model."""

    def __init__(self, *args, prefix_routing: str = "router", **kwargs):
        self.prefix_routing = prefix_routing
        self._train_diagnostics = _empty_diagnostic_accumulator()
        self._eval_diagnostics = _empty_diagnostic_accumulator()
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        return stage_optimizer(self, self.model.dual_prefix, 'prefix')

    @staticmethod
    def _accumulate_diagnostics(
        accumulator: dict[str, Any],
        loss_components: dict[str, float],
        alpha: torch.Tensor,
        target_alpha: torch.Tensor,
    ) -> None:
        alpha_values = alpha.detach().float().reshape(-1).cpu().numpy()
        target_values = target_alpha.detach().float().reshape(-1).cpu().numpy()
        sample_count = int(target_values.size)
        if sample_count == 0:
            return

        accumulator["sample_count"] += sample_count
        accumulator["lm_loss_sum"] += loss_components["lm_loss"] * sample_count
        accumulator["router_loss_sum"] += loss_components["router_loss"] * sample_count
        accumulator["alpha"].append(alpha_values)
        accumulator["target_alpha"].append(target_values)

    @staticmethod
    def _summarize_diagnostics(
        accumulator: dict[str, Any], *, router_diagnostics: bool = True,
    ) -> dict[str, float]:
        sample_count = accumulator["sample_count"]
        if sample_count == 0:
            return {}

        alpha = np.concatenate(accumulator["alpha"])
        target_alpha = np.concatenate(accumulator["target_alpha"])
        if not router_diagnostics:
            # These are supplied labels, not router predictions. Do not report
            # tautological router accuracy/AUC for label-routed training.
            return {
                "lm_loss": float(accumulator["lm_loss_sum"] / sample_count),
                "prefix_alpha_mean": float(alpha.mean()),
            }
        return {
            "lm_loss": float(accumulator["lm_loss_sum"] / sample_count),
            "router_loss": float(accumulator["router_loss_sum"] / sample_count),
            **_binary_router_metrics(alpha, target_alpha),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_alpha = inputs.pop("target_alpha")
        inputs.pop("variant_type", None)
        # Validation always uses the deployed soft mixture. Only training
        # bypasses the frozen router with the per-example binary labels.
        label_routing = self.prefix_routing == "label" and model.training

        lm_loss, alpha = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            prompt_input_ids=inputs["prompt_input_ids"],
            prompt_attention_mask=inputs["prompt_attention_mask"],
            forced_alpha=target_alpha if label_routing else None,
        )

        # The prefix stage optimizes LM loss only; router BCE is diagnostic.
        with torch.no_grad():
            router_loss = 0.0 if label_routing else F.binary_cross_entropy(alpha, target_alpha).item()
        loss_components = {"lm_loss": lm_loss.item(), "router_loss": router_loss}
        accumulator = self._train_diagnostics if model.training else self._eval_diagnostics
        self._accumulate_diagnostics(
            accumulator,
            loss_components,
            alpha,
            target_alpha,
        )
        if not return_outputs:
            return lm_loss

        outputs = {"logits": alpha}
        return lm_loss, outputs

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save only trainable DSPR parameters to avoid huge checkpoints of frozen base LLM.
        torch.save(
            {
                "dual_prefix_state_dict": self.model.dual_prefix.state_dict(),
                "router_state_dict": self.model.router.state_dict(),
                "training_args": self.args.to_dict(),
                "prefix_routing": self.prefix_routing,
                "prefix_validation_routing": "router",
            },
            os.path.join(output_dir, "dspr_trainable.pt"),
        )

    def load_trainable_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.args.device)
        self.model.dual_prefix.load_state_dict(ckpt["dual_prefix_state_dict"])
        self.model.router.load_state_dict(ckpt["router_state_dict"])

    def _load_best_model(self):
        checkpoint = os.path.join(self.state.best_model_checkpoint, "dspr_trainable.pt")
        logger.info("Loading best DSPR checkpoint from %s", checkpoint)
        self.load_trainable_checkpoint(checkpoint)

    def log(self, logs: dict[str, float], start_time=None) -> None:
        """Attach interval-level training diagnostics to regular Trainer logs."""
        logs = dict(logs)
        if "loss" in logs or "train_loss" in logs:
            logs.update(self._summarize_diagnostics(
                self._train_diagnostics, router_diagnostics=self.prefix_routing == "router",
            ))
            self._train_diagnostics = _empty_diagnostic_accumulator()
            if self.optimizer is not None:
                logs["prefix_learning_rate"] = float(self.optimizer.param_groups[0]["lr"])

        if start_time is None:
            super().log(logs)
        else:
            super().log(logs, start_time)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Add decomposed validation losses to the normal evaluation result."""
        self._eval_diagnostics = _empty_diagnostic_accumulator()
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        diagnostics = self._summarize_diagnostics(self._eval_diagnostics)
        for name, value in diagnostics.items():
            output.metrics.setdefault(f"{metric_key_prefix}_{name}", value)
        return output


def compute_dspr_metrics(eval_pred: Any) -> dict[str, float]:
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

    return _binary_router_metrics(alpha, target_alpha)
