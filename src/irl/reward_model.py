from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """
    将 (state_text, action_text) 映射到连续向量。
    这里用简化的句向量平均池化，可被更强编码器替换。
    """

    def __init__(self, embedding: nn.Embedding, pad_id: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.pad_id = pad_id

    def encode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)  # [B, T, D]
        mask = (ids != self.pad_id).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / denom

    def forward(self, state_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        s = self.encode_ids(state_ids)
        a = self.encode_ids(action_ids)
        return torch.cat([s, a], dim=-1)


class MaxEntRewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class AirlRewardModel(nn.Module):
    """
    AIRL 常用形式:
      f(s,a,s') = g(s,a) + gamma * h(s') - h(s)
    """

    def __init__(self, sa_dim: int, s_dim: int, hidden_dim: int = 256, gamma: float = 0.99) -> None:
        super().__init__()
        self.gamma = gamma
        self.g = nn.Sequential(
            nn.Linear(sa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.h = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sa_feat: torch.Tensor, s_feat: torch.Tensor, s_next_feat: torch.Tensor) -> torch.Tensor:
        g_val = self.g(sa_feat).squeeze(-1)
        h_s = self.h(s_feat).squeeze(-1)
        h_sp = self.h(s_next_feat).squeeze(-1)
        return g_val + self.gamma * h_sp - h_s


@dataclass
class RewardBatch:
    state_ids: torch.Tensor
    action_ids: torch.Tensor
    next_state_ids: torch.Tensor
