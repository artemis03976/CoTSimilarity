import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """Extracts hidden state h_Q from frozen LLM's last layer, last token position."""

    def __init__(self, frozen_model, layer_idx=-1):
        super().__init__()
        self.frozen_model = frozen_model
        self.layer_idx = layer_idx

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            h_Q: (batch_size, hidden_dim) - context vector from last token
        """
        with torch.no_grad():
            outputs = self.frozen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = outputs.hidden_states[self.layer_idx]
            last_token_indices = attention_mask.long().sum(dim=1) - 1
            last_token_indices = torch.clamp(last_token_indices, min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            h_Q = hidden_states[batch_indices, last_token_indices, :]
        return h_Q
