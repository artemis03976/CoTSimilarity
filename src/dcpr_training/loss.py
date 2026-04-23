import torch
import torch.nn as nn

class DCPRLoss(nn.Module):
    """Joint loss: L_total = L_LLM_Gen + λ * L_Router"""

    def __init__(self, lambda_router=0.1):
        super().__init__()
        self.lambda_router = lambda_router
        self.bce_loss = nn.BCELoss()

    def forward(self, lm_loss, predicted_alpha, target_alpha):
        """
        Args:
            lm_loss: Language modeling loss (scalar)
            predicted_alpha: Router output (batch_size, 1)
            target_alpha: Ground truth (batch_size, 1) - 0.0 for simple, 1.0 for hard
        Returns:
            total_loss, loss_dict
        """
        router_loss = self.bce_loss(predicted_alpha, target_alpha)
        total_loss = lm_loss + self.lambda_router * router_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'router_loss': router_loss.item(),
            'mean_alpha': predicted_alpha.mean().item()
        }
