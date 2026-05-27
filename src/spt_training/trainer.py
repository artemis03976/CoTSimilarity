import os

import torch
from transformers import Trainer


class BaselineTrainer(Trainer):
    """HuggingFace Trainer for StaticPromptModel baseline.

    Unlike DCPRTrainer, this uses only the LM cross-entropy loss
    (no router loss) since there is no routing mechanism.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Drop DCPR-specific fields that come from the shared dataset
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
        return lm_loss, {"logits": torch.tensor(0.0)}

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        torch.save(
            {
                "soft_prompt": model_to_save.soft_prompt.data,
                "training_args": self.args.to_dict(),
            },
            os.path.join(output_dir, "baseline_trainable.pt"),
        )

    def load_trainable_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.args.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.soft_prompt.data.copy_(ckpt["soft_prompt"])
