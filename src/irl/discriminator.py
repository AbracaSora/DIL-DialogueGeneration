from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AirlDiscriminator(nn.Module):
    """
    D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + pi(a|s))
    输入 log_pi(a|s) 以保持数值稳定。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, f_sa: torch.Tensor, log_pi: torch.Tensor) -> torch.Tensor:
        # log D = f - logsumexp(f, log_pi)
        denom = torch.logsumexp(torch.stack([f_sa, log_pi], dim=-1), dim=-1)
        return torch.exp(f_sa - denom)


def airl_disc_loss(
    f_expert: torch.Tensor,
    logp_expert: torch.Tensor,
    f_policy: torch.Tensor,
    logp_policy: torch.Tensor,
) -> torch.Tensor:
    d_exp = torch.sigmoid(f_expert - logp_expert)
    d_pol = torch.sigmoid(f_policy - logp_policy)
    loss_exp = F.binary_cross_entropy(d_exp, torch.ones_like(d_exp))
    loss_pol = F.binary_cross_entropy(d_pol, torch.zeros_like(d_pol))
    return loss_exp + loss_pol


def gail_disc_loss(logits_expert: torch.Tensor, logits_policy: torch.Tensor) -> torch.Tensor:
    loss_exp = F.binary_cross_entropy_with_logits(logits_expert, torch.ones_like(logits_expert))
    loss_pol = F.binary_cross_entropy_with_logits(logits_policy, torch.zeros_like(logits_policy))
    return loss_exp + loss_pol
