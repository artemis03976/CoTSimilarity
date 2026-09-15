"""Prompt-level trainer used by the first stage of DSPR training."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from .trainer import _binary_router_metrics


logger = logging.getLogger(__name__)


def compute_router_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute problem-level router diagnostics from alpha predictions."""
    predictions, label_ids = eval_pred
    alpha = predictions[0] if isinstance(predictions, tuple) else predictions
    target = label_ids[-1] if isinstance(label_ids, tuple) else label_ids
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    clipped = np.clip(alpha, 1e-7, 1.0 - 1e-7)
    router_loss = -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    return {
        'router_loss': float(router_loss),
        **_binary_router_metrics(alpha, target),
    }


class RouterPromptTrainer(Trainer):
    """Train only the DSPR router from unique prompt-level labels."""

    def __init__(
        self,
        *args,
        target_smoothing: float = 0.0,
        positive_weight: float | None = None,
        **kwargs,
    ):
        if not 0.0 <= target_smoothing < 0.5:
            raise ValueError('target_smoothing must be in [0, 0.5)')
        if positive_weight is not None and positive_weight <= 0.0:
            raise ValueError('positive_weight must be positive')
        self.target_smoothing = float(target_smoothing)
        self.positive_weight = positive_weight
        self._train_alpha: list[np.ndarray] = []
        self._train_target: list[np.ndarray] = []
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Build AdamW groups containing only trainable router parameters."""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        model_to_group = opt_model.module if hasattr(opt_model, 'module') else opt_model
        router_parameter_ids = {id(parameter) for parameter in model_to_group.router.parameters()}
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        grouped_parameters = []
        for use_decay in (True, False):
            parameters = [
                parameter
                for name, parameter in opt_model.named_parameters()
                if parameter.requires_grad
                and id(parameter) in router_parameter_ids
                and (name in decay_parameters) == use_decay
            ]
            if parameters:
                grouped_parameters.append(
                    {
                        'params': parameters,
                        'lr': self.args.learning_rate,
                        'weight_decay': self.args.weight_decay if use_decay else 0.0,
                        'group_name': 'router',
                    }
                )
        if not grouped_parameters:
            raise ValueError('Router stage has no trainable router parameters')

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs.pop('params', None)
        optimizer_kwargs.pop('model', None)
        self.optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target = inputs.pop('target_alpha').float().reshape(-1, 1)
        context = inputs.pop('context_embeddings', None)
        if context is None:
            context = model.context_encoder(
                inputs['prompt_input_ids'], inputs['prompt_attention_mask']
            )
        router_dtype = next(model.router.parameters()).dtype
        alpha = model.router(context.to(dtype=router_dtype)).float()

        smoothed_target = target * (1.0 - 2.0 * self.target_smoothing) + self.target_smoothing
        per_example = F.binary_cross_entropy(alpha, smoothed_target, reduction='none')
        if self.positive_weight is None:
            loss = per_example.mean()
        else:
            weights = torch.where(
                target >= 0.5,
                torch.full_like(target, float(self.positive_weight)),
                torch.ones_like(target),
            )
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1.0)

        if model.training:
            self._train_alpha.append(alpha.detach().cpu().numpy())
            self._train_target.append(target.detach().cpu().numpy())
        if not return_outputs:
            return loss
        return loss, {'logits': alpha}

    def log(self, logs: dict[str, float], start_time=None) -> None:
        logs = dict(logs)
        if ('loss' in logs or 'train_loss' in logs) and self._train_alpha:
            alpha = np.concatenate(self._train_alpha).reshape(-1)
            target = np.concatenate(self._train_target).reshape(-1)
            logs.update(_binary_router_metrics(alpha, target))
            logs['router_loss'] = float(logs.get('loss', logs.get('train_loss', 0.0)))
            logs['router_learning_rate'] = float(
                self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 0.0
            )
            self._train_alpha = []
            self._train_target = []

        if start_time is None:
            super().log(logs)
        else:
            super().log(logs, start_time)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        config = model_to_save.config
        config_payload = asdict(config) if is_dataclass(config) else None
        torch.save(
            {
                'router_state_dict': model_to_save.router.state_dict(),
                'model_config': config_payload,
                'training_args': self.args.to_dict(),
                'target_smoothing': self.target_smoothing,
                'positive_weight': self.positive_weight,
            },
            os.path.join(output_dir, 'router_trainable.pt'),
        )

    def load_router_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        state_dict = checkpoint.get('router_state_dict', checkpoint)
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.router.load_state_dict(state_dict)

    def _load_best_model(self):
        best_checkpoint = self.state.best_model_checkpoint
        if best_checkpoint:
            router_checkpoint = os.path.join(best_checkpoint, 'router_trainable.pt')
            if os.path.isfile(router_checkpoint):
                logger.info('Loading best router checkpoint from %s', router_checkpoint)
                self.load_router_checkpoint(router_checkpoint)
                return
        return super()._load_best_model()
