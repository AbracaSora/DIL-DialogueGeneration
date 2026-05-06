from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def discounted_returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    # rewards: [B, T]
    ret = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(0), device=rewards.device)
    for t in reversed(range(rewards.size(1))):
        running = rewards[:, t] + gamma * running
        ret[:, t] = running
    return ret


@dataclass
class PgLossOutput:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    total_loss: torch.Tensor


class ValueBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    entropies: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
    gamma: float = 1.0,
    baseline: Optional[torch.Tensor] = None,
) -> PgLossOutput:
    """
    log_probs/rewards/entropies 形状均为 [B, T]。
    """
    returns = discounted_returns(rewards, gamma=gamma)
    adv = returns - baseline if baseline is not None else returns
    policy_loss = -(log_probs * adv.detach()).mean()

    if entropies is None:
        entropy = torch.tensor(0.0, device=log_probs.device)
    else:
        entropy = entropies.mean()

    total_loss = policy_loss - entropy_coef * entropy
    return PgLossOutput(policy_loss=policy_loss, entropy=entropy, total_loss=total_loss)
