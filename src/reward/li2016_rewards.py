from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F


DEFAULT_DULL_RESPONSES = [
    "i don't know",
    "i don't know what you are talking about",
    "i have no idea",
    "i don't care",
    "i am not sure",
    "you don't know",
    "whatever",
    "i don't understand",
]


@dataclass
class RewardWeights:
    lambda1: float = 0.25
    lambda2: float = 0.25
    lambda3: float = 0.5


def _avg_logprob(model_log_probs: torch.Tensor, token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # model_log_probs: [B, T, V], token_ids: [B, T]
    tgt = token_ids[:, 1:]
    lp = model_log_probs[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id).float()
    denom = torch.clamp(mask.sum(dim=-1), min=1.0)
    return (lp * mask).sum(dim=-1) / denom


def ease_of_answering_reward(
    logp_dull_responses: torch.Tensor,
) -> torch.Tensor:
    """
    对应 Li2016 r1:
      r1 = - average_s( 1/N_s * log p_seq2seq(s | a) )
    输入为 [B, num_dull] 的平均log概率，输出 [B]。
    """
    return -logp_dull_responses.mean(dim=-1)


def information_flow_reward(prev_self_repr: torch.Tensor, cur_self_repr: torch.Tensor) -> torch.Tensor:
    """
    对应 Li2016 r2:
      r2 = -log cos(h_{p_i}, h_{p_{i+1}})
    """
    cos = F.cosine_similarity(prev_self_repr, cur_self_repr, dim=-1).clamp(min=1e-6, max=1.0)
    return -torch.log(cos)


def semantic_coherence_reward(forward_avg_logp: torch.Tensor, backward_avg_logp: torch.Tensor) -> torch.Tensor:
    """
    对应 Li2016 r3:
      r3 = 1/N_a log p(a|q,p) + 1/N_q log p_backward(q|a)
    """
    return forward_avg_logp + backward_avg_logp


def combine_li2016_rewards(
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    weights: RewardWeights,
) -> torch.Tensor:
    return weights.lambda1 * r1 + weights.lambda2 * r2 + weights.lambda3 * r3


def build_dull_matrix(
    dull_responses: Iterable[str],
    vocab_encode_fn,
    max_len: int,
    device: torch.device,
) -> List[torch.Tensor]:
    mats: List[torch.Tensor] = []
    for s in dull_responses:
        ids = torch.tensor(vocab_encode_fn(s, max_len=max_len), dtype=torch.long, device=device)
        mats.append(ids.unsqueeze(0))
    return mats
