import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DCPRConfig


class StaticPromptModel(nn.Module):
    """Single static soft prompt baseline — no router, no dual prefix.

    Uses the same frozen LLM and prefix length as DCPR, but replaces the
    dual-prefix + router mechanism with a single learnable prefix that is
    shared across all inputs regardless of difficulty.
    """
    supports_gradient_checkpointing = True

    def __init__(self, config: DCPRConfig):
        super().__init__()
        self.config = config

        # Load frozen LLM
        self.frozen_llm, self.tokenizer = self._load_frozen_llm()

        llm_hidden_dim = self.frozen_llm.config.hidden_size

        # Single learnable soft prompt (same size as each of DCPR's dual prefixes)
        self.soft_prompt = nn.Parameter(
            torch.randn(config.prefix_length, llm_hidden_dim) * 0.02
        )

        # Freeze base LLM
        for param in self.frozen_llm.parameters():
            param.requires_grad = False

    def _load_frozen_llm(self):
        """Load frozen LLM (identical to DCPRModel)."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)

        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        }
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.eval()
        return model, tokenizer

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) — optional, for training
        Returns:
            lm_loss if labels provided, else logits
        """
        batch_size = input_ids.shape[0]
        inputs_embeds, extended_attention_mask = self._build_prefixed_inputs(
            input_ids, attention_mask
        )

        # Adjust labels for prefix
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, self.config.prefix_length), -100,
                device=labels.device, dtype=labels.dtype,
            )
            extended_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            extended_labels = None

        outputs = self.frozen_llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            use_cache=False,
        )

        if labels is not None:
            return outputs.loss
        else:
            return outputs.logits

    def _build_prefixed_inputs(self, input_ids, attention_mask):
        """Prepend the single soft prompt to token embeddings."""
        batch_size = input_ids.shape[0]

        # Expand soft prompt for the batch
        prefix = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Get token embeddings
        token_embeds = self.frozen_llm.get_input_embeddings()(input_ids)

        # Cast prefix to match token embedding dtype
        if prefix.dtype != token_embeds.dtype:
            prefix = prefix.to(dtype=token_embeds.dtype)

        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.config.prefix_length,
            device=attention_mask.device, dtype=attention_mask.dtype,
        )
        extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_attention_mask

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        """Generate with single soft prompt prefix.

        Returns:
            new_token_ids: generated continuation tokens, shape (batch, new_len)
        """
        inputs_embeds, extended_attention_mask = self._build_prefixed_inputs(
            input_ids, attention_mask
        )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        batch_size = input_ids.size(0)
        prefix_dummy_ids = torch.full(
            (batch_size, self.config.prefix_length),
            fill_value=pad_token_id if pad_token_id is not None else 0,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        extended_input_ids = torch.cat([prefix_dummy_ids, input_ids], dim=1)
        input_len = extended_input_ids.shape[1]

        outputs = self.frozen_llm.generate(
            input_ids=extended_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        new_token_ids = outputs[:, input_len:]
        return new_token_ids

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Compatibility hook for HuggingFace Trainer."""
        if hasattr(self.frozen_llm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.frozen_llm.gradient_checkpointing_enable()
            else:
                self.frozen_llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
        self.frozen_llm.config.use_cache = False

    def gradient_checkpointing_disable(self):
        """Compatibility hook for HuggingFace Trainer."""
        if hasattr(self.frozen_llm, "gradient_checkpointing_disable"):
            self.frozen_llm.gradient_checkpointing_disable()
