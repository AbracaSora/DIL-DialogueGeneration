from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.irl.reward_model import AirlRewardModel, MaxEntRewardModel, TrajectoryEncoder
from src.train_common import (
    build_dataloaders,
    build_model,
    cfg_device,
    maybe_load_checkpoint,
    prepare_run_dir,
    save_checkpoint,
)
from src.utils import load_yaml, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/irl_airl.yaml")
    parser.add_argument("--policy_ckpt", default="outputs/li2016_rl/last.pt")
    parser.add_argument("--reward_ckpt", default="outputs/airl_reward/reward_model.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg_device(cfg)
    train_loader, _, vocab = build_dataloaders(cfg)

    policy = build_model(cfg, vocab).to(device)
    maybe_load_checkpoint(args.policy_ckpt, policy)

    reward_state = torch.load(args.reward_ckpt, map_location=device) if Path(args.reward_ckpt).exists() else None
    embedding = nn.Embedding(len(vocab.itos), cfg["model"]["embed_dim"], padding_idx=vocab.pad_id).to(device)
    encoder = TrajectoryEncoder(embedding, vocab.pad_id).to(device)
    sa_dim = cfg["model"]["embed_dim"] * 2
    s_dim = cfg["model"]["embed_dim"]
    if cfg["irl"]["mode"] == "maxent":
        reward_model = MaxEntRewardModel(sa_dim, cfg["irl"]["hidden_dim"]).to(device)
    else:
        reward_model = AirlRewardModel(sa_dim, s_dim, cfg["irl"]["hidden_dim"]).to(device)

    if reward_state is not None:
        encoder.load_state_dict(reward_state["encoder"])
        reward_model.load_state_dict(reward_state["reward_model"])
    reward_model.eval()
    encoder.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["train"]["lr"])
    alpha = cfg["irl"]["alpha"]
    run_dir = prepare_run_dir({"output_root": cfg["output_root"], "experiment_name": "hybrid_policy"})
    metrics = {"epochs": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        losses = []
        avg_reward = []
        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            sampled = policy.generate(src_ids, max_len=cfg["data"]["max_tgt_len"], sample=True)
            out = policy(src_ids, sampled)
            if not torch.isfinite(out.loss):
                continue

            with torch.no_grad():
                sa = encoder(src_ids, sampled)
                s_feat = encoder.encode_ids(src_ids)
                if cfg["irl"]["mode"] == "maxent":
                    r_irl = reward_model(sa)
                else:
                    r_irl = reward_model(sa, s_feat, s_feat)

                sampled_texts = [vocab.decode(x.tolist()) for x in sampled]
                # baseline 奖励近似：含问号/交互倾向给正分，含dull给负分
                r_base = []
                for t in sampled_texts:
                    score = 0.2 if "?" in t else 0.0
                    if "i don't know" in t or "whatever" in t:
                        score -= 1.0
                    r_base.append(score)
                r_base = torch.tensor(r_base, dtype=torch.float, device=device)
                total_reward = alpha * r_irl + (1.0 - alpha) * r_base
                total_reward = torch.clamp(total_reward, min=-5.0, max=5.0)

            scale = torch.clamp(-total_reward.mean() + 1.0, min=0.1, max=10.0)
            loss = out.loss * scale
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            losses.append(float(loss.item()))
            avg_reward.append(float(total_reward.mean().item()))

        m_loss = sum(losses) / max(len(losses), 1)
        m_rew = sum(avg_reward) / max(len(avg_reward), 1)
        metrics["epochs"].append({"epoch": epoch, "hybrid_loss": m_loss, "hybrid_reward": m_rew})
        print(f"[hybrid] epoch={epoch} loss={m_loss:.4f} reward={m_rew:.4f}")

        save_checkpoint(
            run_dir / "last.pt",
            policy,
            optimizer,
            meta={"cfg": cfg, "vocab_itos": vocab.itos, "epoch": epoch, "hybrid_reward": m_rew},
        )

    save_json(metrics, run_dir / "metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
