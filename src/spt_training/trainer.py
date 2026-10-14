import logging
import os
from dataclasses import asdict

import torch
from transformers import Trainer


logger = logging.getLogger(__name__)


class BaselineTrainer(Trainer):
    """HuggingFace Trainer for StaticPromptModel baseline.

    Unlike DSPRTrainer, this uses only the LM cross-entropy loss
    (no router loss) since there is no routing mechanism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss returns a mean LM loss and does not normalize it by the
        # accumulated token count supplied by recent Trainer versions.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Drop DSPR-specific fields that come from the shared dataset
        inputs.pop("target_alpha", None)
        inputs.pop("variant_type", None)
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)

        lm_loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )

        if not return_outputs:
            return lm_loss
        return lm_loss, {"logits": lm_loss.new_zeros((inputs["input_ids"].shape[0], 1))}

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        torch.save(
            {
                "soft_prompt": model_to_save.soft_prompt.detach().cpu(),
                "spt_config": asdict(model_to_save.config),
                "training_args": self.args.to_dict(),
            },
            os.path.join(output_dir, "baseline_trainable.pt"),
        )

    def load_trainable_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.args.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            model_to_load.soft_prompt.copy_(ckpt["soft_prompt"])

    def _load_best_model(self):
        """Restore the lightweight prompt checkpoint selected by validation loss."""
        best_checkpoint = self.state.best_model_checkpoint
        if best_checkpoint:
            checkpoint_path = os.path.join(best_checkpoint, "baseline_trainable.pt")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Best SPT checkpoint is missing: {checkpoint_path}")
            logger.info("Loading best SPT checkpoint from %s", checkpoint_path)
            self.load_trainable_checkpoint(checkpoint_path)
            return
        return super()._load_best_model()
