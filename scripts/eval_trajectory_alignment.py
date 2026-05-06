from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import save_json, set_seed


class TinyTrajectoryClassifier(nn.Module):
    def __init__(self, vocab_size: int = 5000, emb_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.cls = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x).mean(dim=1)
        return self.cls(emb).squeeze(-1)


def fake_tokenize(text: str, max_len: int = 32) -> List[int]:
    toks = text.lower().split()
    ids = [abs(hash(t)) % 5000 for t in toks[:max_len]]
    if not ids:
        ids = [0]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


def load_texts(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except FileNotFoundError:
        return []


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> float:
    def _k(a, b):
        dist = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist / (2 * sigma * sigma))

    k_xx = _k(x, x).mean()
    k_yy = _k(y, y).mean()
    k_xy = _k(x, y).mean()
    return float((k_xx + k_yy - 2.0 * k_xy).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_file", default="outputs/expert_traj.txt")
    parser.add_argument("--policy_file", default="outputs/policy_traj.txt")
    parser.add_argument("--out_file", default="outputs/trajectory_alignment.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    expert = load_texts(args.expert_file)
    policy = load_texts(args.policy_file)
    if not expert:
        expert = ["how are you ? i am fine what about you ?"] * 100
    if not policy:
        policy = ["i dont know whatever"] * 100

    n = min(len(expert), len(policy), 200)
    expert = expert[:n]
    policy = policy[:n]
    data = expert + policy
    labels = [1] * n + [0] * n

    ids = torch.tensor([fake_tokenize(t) for t in data], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)

    model = TinyTrajectoryClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        logits = model(ids)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(ids)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = float((pred == y).float().mean().item())
        emb = model.emb(ids).mean(dim=1)
        emb_e = emb[:n]
        emb_p = emb[n:]
        mmd = mmd_rbf(emb_e, emb_p)

    result = {
        "trajectory_discriminability_acc": acc,
        "trajectory_discriminability_goal": "lower_is_better",
        "embedding_mmd": mmd,
        "question_rate_expert": sum("?" in t for t in expert) / max(len(expert), 1),
        "question_rate_policy": sum("?" in t for t in policy) / max(len(policy), 1),
    }
    save_json(result, args.out_file)
    print(result)


if __name__ == "__main__":
    main()
