from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.rl.dialogue_env import TwoAgentDialogueEnv
from src.train_common import build_dataloaders, build_model, cfg_device, maybe_load_checkpoint
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/li2016_rl.yaml")
    parser.add_argument("--ckpt", default="outputs/li2016_rl/last.pt")
    parser.add_argument("--message", default="how are you")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = cfg_device(cfg)
    _, _, vocab = build_dataloaders(cfg)
    model = build_model(cfg, vocab).to(device)
    maybe_load_checkpoint(args.ckpt, model)

    env = TwoAgentDialogueEnv(
        model=model,
        vocab=vocab,
        reward_fn=lambda s, p, a: 0.0,
        max_turns=cfg["rl"]["max_sim_turns"],
        max_decode_len=cfg["rl"]["max_decode_len"],
    )
    ep = env.rollout(args.message, device=device)
    for i, step in enumerate(ep.steps, start=1):
        print(f"turn={i} state='{step.state_text}'")
        print(f"  action='{step.action_text}' reward={step.reward:.4f}")
    print(f"total_reward={ep.total_reward:.4f}")


if __name__ == "__main__":
    main()
