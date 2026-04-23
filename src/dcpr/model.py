import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .context_encoder import ContextEncoder
from .router import UCBRouter
from .dual_prefix import DualCognitivePrefix
from .config import DCPRConfig


class DCPRModel(nn.Module):
    """Dynamic Cognitive Prefix Routing model."""
    supports_gradient_checkpointing = True

    def __init__(self, config: DCPRConfig):
        super().__init__()
        self.config = config

        # Load frozen LLM
        self.frozen_llm, self.tokenizer = self._load_frozen_llm()

        llm_hidden_dim = self.frozen_llm.config.hidden_size
        self.context_encoder = ContextEncoder(self.frozen_llm, layer_idx=config.context_layer_idx)
        self.router = UCBRouter(llm_hidden_dim, config.router_intermediate_dim, config.router_dropout)
        self.dual_prefix = DualCognitivePrefix(config.prefix_length, llm_hidden_dim)

        # Freeze base LLM
        for param in self.frozen_llm.parameters():
            param.requires_grad = False

    def _load_frozen_llm(self):
        """Load frozen LLM"""
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

    def forward(self, input_ids, attention_mask, labels=None, prompt_input_ids=None, prompt_attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) - optional, for training
            prompt_input_ids: (batch_size, seq_len) - optional, prompt-only input for context encoder
            prompt_attention_mask: (batch_size, seq_len) - optional, prompt-only attention mask
        Returns:
            (lm_loss, alpha) if labels provided, else (logits, alpha)
        """
        batch_size = input_ids.shape[0]
        inputs_embeds, extended_attention_mask, alpha = self._build_prefixed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )

        # 7. Adjust labels for prefix (if provided)
        if labels is not None:
            prefix_labels = torch.full((batch_size, self.config.prefix_length), -100, device=labels.device, dtype=labels.dtype)
            extended_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            extended_labels = None

        # 8. Forward through frozen LLM
        outputs = self.frozen_llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            use_cache=False
        )

        if labels is not None:
            return outputs.loss, alpha
        else:
            return outputs.logits, alpha

    def _build_prefixed_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ):
        """Build prefix-augmented embeddings and masks for generation."""
        batch_size = input_ids.shape[0]
        if prompt_input_ids is None:
            prompt_input_ids = input_ids
        if prompt_attention_mask is None:
            prompt_attention_mask = attention_mask

        h_Q = self.context_encoder(prompt_input_ids, prompt_attention_mask)
        router_dtype = next(self.router.parameters()).dtype
        if h_Q.dtype != router_dtype:
            h_Q = h_Q.to(dtype=router_dtype)
        alpha = self.router(h_Q)

        P_final = self.dual_prefix(alpha, batch_size)
        prompt_embeds = self.frozen_llm.get_input_embeddings()(input_ids)
        if P_final.dtype != prompt_embeds.dtype:
            P_final = P_final.to(dtype=prompt_embeds.dtype)

        inputs_embeds = torch.cat([P_final, prompt_embeds], dim=1)
        prefix_mask = torch.ones(
            batch_size,
            self.config.prefix_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, extended_attention_mask, alpha

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate with DCPR prefix routing via transformers.generate.

        Returns:
            new_token_ids: generated continuation tokens, shape (batch, new_len)
            alpha: router outputs, shape (batch, 1)
        """
        inputs_embeds, extended_attention_mask, alpha = self._build_prefixed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
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
        return new_token_ids, alpha

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
