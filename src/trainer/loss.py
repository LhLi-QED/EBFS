import torch
import torch.nn as nn
from typing import Optional, Tuple


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


class ProverPolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps_low: float = 0.2, clip_eps_high: float = 0.2, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss

        self.neg_eps_low = 0.1
        self.neg_eps_high = 1.2
        self.pos_eps_low = 0.2
        self.pos_eps_high = 1.0

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropys: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entropy_weights = entropys.detach() * action_mask
        neg_entropy = entropy_weights[advantages <= 0.0]
        pos_entropy = entropy_weights[advantages > 0.0]


        neg_entropy = 1 - (neg_entropy - self.neg_eps_low) / (self.neg_eps_high - self.neg_eps_low)
        neg_entropy[neg_entropy <= 0.0] = 0.0
        neg_entropy[neg_entropy >= 1.0] = 0.0


        pos_entropy = (pos_entropy - self.pos_eps_low) / (self.pos_eps_high - self.pos_eps_low)
        pos_entropy[pos_entropy >= 1.0] = 1.0
        pos_entropy[pos_entropy <= 0.0] = 0.0

        entropy_weights[advantages <= 0.0] = neg_entropy
        entropy_weights[advantages > 0.0] = pos_entropy

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages * entropy_weights
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages * entropy_weights
        loss = -torch.min(surr1, surr2)
        loss = (
            masked_mean(loss, entropy_weights > 0.0, dim=None)
            if self.token_level_loss
            else masked_mean(loss, entropy_weights > 0.0, dim=-1).mean()
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), entropy_weights > 0.0, dim=None)
        return loss, clip_ratio


