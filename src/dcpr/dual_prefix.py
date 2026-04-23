import torch
import torch.nn as nn

class DualCognitivePrefix(nn.Module):
    """Manages P_exploit and P_explore soft prompts with interpolation."""

    def __init__(self, prefix_length=20, hidden_dim=4096, init_from_vocab=None):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim

        if init_from_vocab is not None:
            self.P_exploit = nn.Parameter(init_from_vocab.clone())
            self.P_explore = nn.Parameter(init_from_vocab.clone())
        else:
            self.P_exploit = nn.Parameter(torch.randn(prefix_length, hidden_dim) * 0.02)
            self.P_explore = nn.Parameter(torch.randn(prefix_length, hidden_dim) * 0.02)

    def forward(self, alpha, batch_size):
        """
        Args:
            alpha: (batch_size, 1) - exploration weight
            batch_size: int
        Returns:
            P_final: (batch_size, prefix_length, hidden_dim)
        """
        alpha = alpha.unsqueeze(-1)  # (batch_size, 1, 1)

        P_exploit = self.P_exploit.unsqueeze(0).expand(batch_size, -1, -1)
        P_explore = self.P_explore.unsqueeze(0).expand(batch_size, -1, -1)

        P_final = (1 - alpha) * P_exploit + alpha * P_explore
        return P_final
