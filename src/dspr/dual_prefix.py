import torch
import torch.nn as nn

class DualStructuralPrefix(nn.Module):
    """Manage the reuse and adapt soft prompts.

    P_reuse: encode conservative reuse behavior, while
    P_adapt: encourage more divergent reasoning. 
    The router's alpha chooses a convex mixture between them for each input.
    """

    def __init__(self, prefix_length=20, hidden_dim=4096, init_from_vocab=None):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim

        if init_from_vocab is not None:
            # Optional warm start: initialize both prefixes from existing token embeddings.
            self.P_reuse = nn.Parameter(init_from_vocab.clone())
            self.P_adapt = nn.Parameter(init_from_vocab.clone())
        else:
            self.P_reuse = nn.Parameter(torch.randn(prefix_length, hidden_dim) * 0.02)
            self.P_adapt = nn.Parameter(torch.randn(prefix_length, hidden_dim) * 0.02)

    def forward(self, alpha, batch_size):
        """
        Args:
            alpha: (batch_size, 1) - exploration weight
            batch_size: int
        Returns:
            P_final: (batch_size, prefix_length, hidden_dim)
        """
        alpha = alpha.unsqueeze(-1)  # (batch_size, 1, 1)

        P_reuse = self.P_reuse.unsqueeze(0).expand(batch_size, -1, -1)
        P_adapt = self.P_adapt.unsqueeze(0).expand(batch_size, -1, -1)

        # alpha=0 uses the reuse prefix; alpha=1 uses the adapt prefix.
        P_final = (1 - alpha) * P_reuse + alpha * P_adapt
        return P_final
