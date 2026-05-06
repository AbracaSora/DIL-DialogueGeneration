from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.rl.policy_gradient import reinforce_loss
from src.train_common import (
    build_dataloaders,
    build_model,
    cfg_device,
    maybe_load_checkpoint,
    prepare_run_dir,
    save_checkpoint,
)
from src.utils import load_yaml, save_json, set_seed


def sequence_log_prob(model, src_ids: torch.Tensor, sampled_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    out = model(src_ids, sampled_ids)
    log_probs = torch.log_softmax(out.logits, dim=-1)
    tgt = sampled_ids[:, 1:]
    lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id).float()
    return (lp * mask).sum(dim=-1) / torch.clamp(mask.sum(dim=-1), min=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mmi_pg.yaml")
    parser.add_argument("--init_ckpt", default="outputs/seq2seq_baseline/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg_device(cfg)
    train_loader, _, vocab = build_dataloaders(cfg)
    model = build_model(cfg, vocab).to(device)
    maybe_load_checkpoint(args.init_ckpt, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    run_dir = prepare_run_dir(cfg)

    metrics = {"epochs": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        ep_reward = []
        ep_loss = []
        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            sampled = model.generate(src_ids, max_len=cfg["data"]["max_tgt_len"], sample=True)

            forward_lp = sequence_log_prob(model, src_ids, sampled, vocab.pad_id)
            # 简化版 backward: 反向条件概率近似为把 sampled 当输入重算 src 概率的负值。
            backward_lp = sequence_log_prob(model, sampled, src_ids[:, : sampled.size(1)], vocab.pad_id)
            reward = forward_lp + backward_lp

            logp = sequence_log_prob(model, src_ids, sampled, vocab.pad_id).unsqueeze(1)
            pg = reinforce_loss(log_probs=logp, rewards=reward.unsqueeze(1), entropy_coef=0.0, gamma=1.0)

            optimizer.zero_grad(set_to_none=True)
            pg.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            ep_reward.append(float(reward.mean().item()))
            ep_loss.append(float(pg.total_loss.item()))

        mean_reward = sum(ep_reward) / max(len(ep_reward), 1)
        mean_loss = sum(ep_loss) / max(len(ep_loss), 1)
        metrics["epochs"].append({"epoch": epoch, "reward": mean_reward, "loss": mean_loss})
        print(f"[mmi_pg] epoch={epoch} reward={mean_reward:.4f} loss={mean_loss:.4f}")

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            meta={"cfg": cfg, "vocab_itos": vocab.itos, "epoch": epoch, "reward": mean_reward},
        )

    save_json(metrics, run_dir / "metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
