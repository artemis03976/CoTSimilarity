import torch
import torch.nn as nn

class UCBRouter(nn.Module):
    """Predict exploration weight alpha in [0, 1] from context vector h_Q.

    The router is deliberately small because the paper's intervention is meant
    to be a lightweight prefix-routing layer over a frozen LLM.
    """

    def __init__(self, hidden_dim=4096, intermediate_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, context_vector):
        """
        Args:
            context_vector: (batch_size, hidden_dim)
        Returns:
            alpha: (batch_size, 1) - exploration weight ∈ [0,1]
        """
        return self.mlp(context_vector)
