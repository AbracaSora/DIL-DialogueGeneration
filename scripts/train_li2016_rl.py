from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.rl.dialogue_env import TwoAgentDialogueEnv, distinct_n, repetition_ratio
from src.reward.li2016_rewards import RewardWeights
from src.train_common import (
    build_dataloaders,
    build_model,
    cfg_device,
    maybe_load_checkpoint,
    prepare_run_dir,
    save_checkpoint,
)
from src.utils import load_yaml, save_json, set_seed


def build_reward_fn(dull_list: List[str], weights: RewardWeights):
    def _reward(state_text: str, prev_turn: str, action_text: str) -> float:
        lower = action_text.lower().strip()
        # r1: ease of answering (反 dull)
        r1 = 1.0
        if any(d in lower for d in dull_list):
            r1 = -1.0

        # r2: information flow (同一agent连续重复的近似惩罚)
        rep = repetition_ratio(prev_turn.lower(), lower)
        r2 = -rep

        # r3: semantic coherence (和状态有一定词面相关)
        state_tokens = set(state_text.lower().split())
        act_tokens = set(lower.split())
        overlap = len(state_tokens & act_tokens) / max(len(state_tokens | act_tokens), 1)
        r3 = overlap

        return weights.lambda1 * r1 + weights.lambda2 * r2 + weights.lambda3 * r3

    return _reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/li2016_rl.yaml")
    parser.add_argument("--init_ckpt", default="outputs/mmi_pg/last.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg_device(cfg)
    train_loader, _, vocab = build_dataloaders(cfg)
    model = build_model(cfg, vocab).to(device)
    maybe_load_checkpoint(args.init_ckpt, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    run_dir = prepare_run_dir(cfg)

    rl_cfg = cfg["rl"]
    weights = RewardWeights(rl_cfg["lambda1"], rl_cfg["lambda2"], rl_cfg["lambda3"])
    reward_fn = build_reward_fn(rl_cfg["dull_responses"], weights)
    env = TwoAgentDialogueEnv(
        model=model,
        vocab=vocab,
        reward_fn=reward_fn,
        max_turns=rl_cfg["max_sim_turns"],
        max_decode_len=rl_cfg["max_decode_len"],
    )

    metrics = {"epochs": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        rewards = []
        turns = []
        all_texts = []

        for batch in train_loader:
            src_texts = []
            src_ids = batch["src_ids"]
            for i in range(src_ids.size(0)):
                src_texts.append(vocab.decode(src_ids[i].tolist()))

            batch_loss = torch.tensor(0.0, device=device)
            for msg in src_texts[: min(4, len(src_texts))]:
                episode = env.rollout(msg, device=device)
                ep_reward = episode.total_reward
                rewards.append(ep_reward)
                turns.append(episode.num_turns)
                all_texts.extend([s.action_text for s in episode.steps])

                # 用生成语句做行为克隆式加权训练，近似 policy gradient 更新方向
                for step in episode.steps:
                    state_ids = torch.tensor(
                        [vocab.encode(step.state_text, max_len=cfg["data"]["max_src_len"], add_bos_eos=True)],
                        dtype=torch.long,
                        device=device,
                    )
                    action_ids = torch.tensor(
                        [vocab.encode(step.action_text, max_len=cfg["data"]["max_tgt_len"], add_bos_eos=True)],
                        dtype=torch.long,
                        device=device,
                    )
                    out = model(state_ids, action_ids)
                    batch_loss = batch_loss + (-ep_reward) * out.loss

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

        avg_reward = sum(rewards) / max(len(rewards), 1)
        avg_turn = sum(turns) / max(len(turns), 1)
        d1 = distinct_n(all_texts, n=1)
        d2 = distinct_n(all_texts, n=2)
        metrics["epochs"].append(
            {
                "epoch": epoch,
                "avg_reward": avg_reward,
                "avg_turns": avg_turn,
                "distinct1": d1,
                "distinct2": d2,
            }
        )
        print(f"[li2016_rl] epoch={epoch} reward={avg_reward:.4f} turns={avg_turn:.2f} d1={d1:.4f} d2={d2:.4f}")
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            meta={"cfg": cfg, "vocab_itos": vocab.itos, "epoch": epoch, "avg_reward": avg_reward},
        )

    save_json(metrics, run_dir / "metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
