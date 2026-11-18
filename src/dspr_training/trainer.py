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
        "preference_loss_sum": 0.0,
        "endpoint_counts": np.zeros(2, dtype=int),
        "endpoint_wins": np.zeros(2, dtype=int),
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

    def __init__(self, *args, ged_preference_weight: float = 0.0, **kwargs):
        self.ged_preference_weight = ged_preference_weight
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
        accumulator: dict[str, Any],
    ) -> dict[str, float]:
        sample_count = accumulator["sample_count"]
        if sample_count == 0:
            return {}

        alpha = np.concatenate(accumulator["alpha"])
        target_alpha = np.concatenate(accumulator["target_alpha"])
        preference_metrics = {}
        if accumulator["endpoint_counts"].sum():
            preference_metrics["ged_preference_loss"] = accumulator["preference_loss_sum"] / sample_count
            for i, variant in enumerate(("simple", "hard")):
                n = accumulator["endpoint_counts"][i]
                if n:
                    preference_metrics[f"{variant}_endpoint_win_rate"] = float(accumulator["endpoint_wins"][i] / n)
        return {
            **preference_metrics,
            "lm_loss": float(accumulator["lm_loss_sum"] / sample_count),
            "router_loss": float(accumulator["router_loss_sum"] / sample_count),
            **_binary_router_metrics(alpha, target_alpha),
        }

    def _ged_prefix_loss(self, model, inputs, target_alpha, accumulator):
        """Train at the routed mixture and score the same response at both endpoints."""
        self.model.router.eval()
        self.model.frozen_llm.eval()
        weights = inputs.pop("ged_weight").reshape(-1)
        # The dataset is right-padded. Trim padding shared by the entire batch.
        for mask, keys in (("attention_mask", ("input_ids", "labels", "attention_mask")),
                           ("prompt_attention_mask", ("prompt_input_ids", "prompt_attention_mask"))):
            length = int(inputs[mask].sum(1).max().item())
            for key in keys:
                inputs[key] = inputs[key][:, :length]
        token_counts = inputs["labels"].ne(-100).sum(1)
        mixed_nll, alpha = model(**inputs, response_nll=True)
        lm_loss = (mixed_nll * token_counts).sum() / token_counts.sum().clamp_min(1)
        loss = lm_loss
        scored = (weights > 0) & (token_counts > 0)
        if scored.any():
            endpoint_inputs = {key: inputs[key][scored] for key in ("input_ids", "attention_mask", "labels")}
            nll0, _ = model(**endpoint_inputs, forced_alpha=0.0, response_nll=True)
            nll1, _ = model(**endpoint_inputs, forced_alpha=1.0, response_nll=True)
            targets = target_alpha.reshape(-1)[scored]
            # Positive when the intended endpoint gives the higher mean log probability.
            gap = (2 * targets - 1) * (nll0 - nll1)
            weighted_sum = (weights[scored] * F.softplus(-gap)).sum()
            if model.training:
                loss = lm_loss + self.ged_preference_weight * weighted_sum / weights.numel()
            accumulator["preference_loss_sum"] += weighted_sum.detach().item()
            labels = targets.detach().long().cpu().numpy()
            wins = gap.detach().gt(0).cpu().numpy()
            accumulator["endpoint_counts"] += np.bincount(labels, minlength=2)
            accumulator["endpoint_wins"] += np.bincount(labels[wins], minlength=2)
        return loss, lm_loss, alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target_alpha = inputs.pop("target_alpha")
        inputs.pop("variant_type", None)
        accumulator = self._train_diagnostics if model.training else self._eval_diagnostics
        if self.ged_preference_weight:
            loss, lm_loss, alpha = self._ged_prefix_loss(model, inputs, target_alpha, accumulator)
        else:
            lm_loss, alpha = model(**inputs)
            loss = lm_loss

        # Router BCE remains diagnostic in the prefix stage.
        with torch.no_grad(), torch.autocast(device_type=alpha.device.type, enabled=False):
            router_loss = F.binary_cross_entropy(alpha.float(), target_alpha.float()).item()
        loss_components = {"lm_loss": lm_loss.item(), "router_loss": router_loss}
        self._accumulate_diagnostics(
            accumulator,
            loss_components,
            alpha,
            target_alpha,
        )
        if not return_outputs:
            return loss

        outputs = {"logits": alpha}
        return loss, outputs

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save only trainable DSPR parameters to avoid huge checkpoints of frozen base LLM.
        torch.save(
            {
                "dual_prefix_state_dict": self.model.dual_prefix.state_dict(),
                "router_state_dict": self.model.router.state_dict(),
                "training_args": self.args.to_dict(),
                "ged_preference_weight": self.ged_preference_weight,
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
            logs.update(self._summarize_diagnostics(self._train_diagnostics))
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
