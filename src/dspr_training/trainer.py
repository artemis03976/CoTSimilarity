import os
import logging
from typing import Any

import numpy as np
import torch
from transformers import Trainer

from .loss import DSPRLoss


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

    def __init__(
        self,
        *args,
        lambda_router: float = 0.1,
        prefix_learning_rate: float | None = None,
        router_learning_rate: float | None = None,
        prefix_weight_decay: float | None = None,
        router_weight_decay: float | None = None,
        **kwargs,
    ):
        self._train_diagnostics = _empty_diagnostic_accumulator()
        self._eval_diagnostics = _empty_diagnostic_accumulator()
        super().__init__(*args, **kwargs)
        self.loss_fn = DSPRLoss(lambda_router=lambda_router)
        self.prefix_learning_rate = (
            self.args.learning_rate if prefix_learning_rate is None else prefix_learning_rate
        )
        self.router_learning_rate = (
            self.args.learning_rate if router_learning_rate is None else router_learning_rate
        )
        self.prefix_weight_decay = (
            self.args.weight_decay if prefix_weight_decay is None else prefix_weight_decay
        )
        self.router_weight_decay = (
            self.args.weight_decay if router_weight_decay is None else router_weight_decay
        )

    def create_optimizer(self):
        """Build AdamW groups with independent prefix and router rates."""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        model_to_group = opt_model.module if hasattr(opt_model, "module") else opt_model
        prefix_parameter_ids = {id(parameter) for parameter in model_to_group.dual_prefix.parameters()}
        router_parameter_ids = {id(parameter) for parameter in model_to_group.router.parameters()}
        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        grouped_parameters = []
        group_specs = (
            ("prefix", prefix_parameter_ids, self.prefix_learning_rate),
            ("router", router_parameter_ids, self.router_learning_rate),
        )
        assigned_parameter_ids = prefix_parameter_ids | router_parameter_ids
        for group_name, parameter_ids, learning_rate in group_specs:
            for use_decay in (True, False):
                parameters = [
                    parameter
                    for name, parameter in opt_model.named_parameters()
                    if parameter.requires_grad
                    and id(parameter) in parameter_ids
                    and (name in decay_parameters) == use_decay
                ]
                if parameters:
                    grouped_parameters.append(
                        {
                            "params": parameters,
                            "lr": learning_rate,
                            "weight_decay": (
                                self.prefix_weight_decay if use_decay else 0.0
                            )
                            if group_name == "prefix"
                            else (
                                self.router_weight_decay if use_decay else 0.0
                            ),
                            "group_name": group_name,
                        }
                    )

        other_parameters = [
            parameter
            for parameter in opt_model.parameters()
            if parameter.requires_grad and id(parameter) not in assigned_parameter_ids
        ]
        if other_parameters:
            grouped_parameters.append(
                {
                    "params": other_parameters,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                    "group_name": "other",
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        self.optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
        return self.optimizer

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
    def _summarize_diagnostics(accumulator: dict[str, Any]) -> dict[str, float]:
        sample_count = accumulator["sample_count"]
        if sample_count == 0:
            return {}

        alpha = np.concatenate(accumulator["alpha"])
        target_alpha = np.concatenate(accumulator["target_alpha"])
        return {
            "lm_loss": float(accumulator["lm_loss_sum"] / sample_count),
            "router_loss": float(accumulator["router_loss_sum"] / sample_count),
            **_binary_router_metrics(alpha, target_alpha),
        }

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

        total_loss, loss_components = self.loss_fn(lm_loss, alpha, target_alpha)
        accumulator = self._train_diagnostics if model.training else self._eval_diagnostics
        self._accumulate_diagnostics(
            accumulator,
            loss_components,
            alpha,
            target_alpha,
        )
        if not return_outputs:
            return total_loss

        outputs = {"logits": alpha}
        return total_loss, outputs

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save only trainable DSPR parameters to avoid huge checkpoints of frozen base LLM.
        torch.save(
            {
                "dual_prefix_state_dict": model_to_save.dual_prefix.state_dict(),
                "router_state_dict": model_to_save.router.state_dict(),
                "training_args": self.args.to_dict(),
                "dspr_learning_rates": {
                    "prefix": self.prefix_learning_rate,
                    "router": self.router_learning_rate,
                },
                "dspr_weight_decays": {
                    "prefix": self.prefix_weight_decay,
                    "router": self.router_weight_decay,
                },
            },
            os.path.join(output_dir, "dspr_trainable.pt"),
        )

    def load_trainable_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.args.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.dual_prefix.load_state_dict(ckpt["dual_prefix_state_dict"])
        model_to_load.router.load_state_dict(ckpt["router_state_dict"])

    def _load_best_model(self):
        """Load the project-specific lightweight checkpoint selected by Trainer."""
        best_checkpoint = self.state.best_model_checkpoint
        if best_checkpoint:
            trainable_checkpoint = os.path.join(best_checkpoint, "dspr_trainable.pt")
            if os.path.isfile(trainable_checkpoint):
                logger.info("Loading best DSPR checkpoint from %s", trainable_checkpoint)
                self.load_trainable_checkpoint(trainable_checkpoint)
                return
        return super()._load_best_model()

    def log(self, logs: dict[str, float], start_time=None) -> None:
        """Attach interval-level training diagnostics to regular Trainer logs."""
        logs = dict(logs)
        if "loss" in logs or "train_loss" in logs:
            logs.update(self._summarize_diagnostics(self._train_diagnostics))
            self._train_diagnostics = _empty_diagnostic_accumulator()
            if self.optimizer is not None:
                group_rates = {
                    group.get("group_name"): float(group["lr"])
                    for group in self.optimizer.param_groups
                }
                if "prefix" in group_rates:
                    logs["prefix_learning_rate"] = group_rates["prefix"]
                if "router" in group_rates:
                    logs["router_learning_rate"] = group_rates["router"]

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
