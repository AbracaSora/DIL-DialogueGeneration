from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.irl.discriminator import airl_disc_loss
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


def sample_pairs(vocab, batch_size: int, max_len: int):
    experts = []
    policies = []
    templates = [
        ("how are you", "i am fine how are you"),
        ("what is your name", "my name is bot"),
        ("do you like music", "yes i like music"),
        ("where are you going", "i am going home"),
    ]
    bad_templates = [
        ("how are you", "i do not know"),
        ("what is your name", "whatever"),
        ("do you like music", "i do not care"),
        ("where are you going", "i have no idea"),
    ]
    for _ in range(batch_size):
        s, a = random.choice(templates)
        sb, ab = random.choice(bad_templates)
        experts.append((vocab.encode(s, max_len=max_len, add_bos_eos=True), vocab.encode(a, max_len=max_len, add_bos_eos=True)))
        policies.append(
            (vocab.encode(sb, max_len=max_len, add_bos_eos=True), vocab.encode(ab, max_len=max_len, add_bos_eos=True))
        )
    return experts, policies


def to_tensor(pairs, device):
    s_list = [torch.tensor(x[0], dtype=torch.long) for x in pairs]
    a_list = [torch.tensor(x[1], dtype=torch.long) for x in pairs]
    s = torch.nn.utils.rnn.pad_sequence(s_list, batch_first=True, padding_value=0).to(device)
    a = torch.nn.utils.rnn.pad_sequence(a_list, batch_first=True, padding_value=0).to(device)
    return s, a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/irl_airl.yaml")
    parser.add_argument("--policy_ckpt", default="outputs/li2016_rl/last.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg_device(cfg)

    _, _, vocab = build_dataloaders(cfg)
    policy = build_model(cfg, vocab).to(device)
    maybe_load_checkpoint(args.policy_ckpt, policy)
    policy.eval()

    embedding = nn.Embedding(len(vocab.itos), cfg["model"]["embed_dim"], padding_idx=vocab.pad_id).to(device)
    encoder = TrajectoryEncoder(embedding=embedding, pad_id=vocab.pad_id).to(device)
    sa_dim = cfg["model"]["embed_dim"] * 2
    s_dim = cfg["model"]["embed_dim"]

    mode = cfg["irl"]["mode"]
    if mode == "maxent":
        reward_model = MaxEntRewardModel(input_dim=sa_dim, hidden_dim=cfg["irl"]["hidden_dim"]).to(device)
    else:
        reward_model = AirlRewardModel(sa_dim=sa_dim, s_dim=s_dim, hidden_dim=cfg["irl"]["hidden_dim"]).to(device)

    optimizer = torch.optim.Adam(list(embedding.parameters()) + list(reward_model.parameters()), lr=cfg["train"]["lr"])
    run_dir = prepare_run_dir(cfg)
    metrics = {"mode": mode, "epochs": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        losses = []
        for _ in range(cfg["irl"]["disc_steps_per_epoch"]):
            experts, policies = sample_pairs(vocab, cfg["irl"]["expert_batch_size"], cfg["data"]["max_tgt_len"])
            s_exp, a_exp = to_tensor(experts, device)
            s_pol, a_pol = to_tensor(policies, device)

            sa_exp = encoder(s_exp, a_exp)
            sa_pol = encoder(s_pol, a_pol)
            s_exp_feat = encoder.encode_ids(s_exp)
            s_pol_feat = encoder.encode_ids(s_pol)

            if mode == "maxent":
                f_exp = reward_model(sa_exp)
                f_pol = reward_model(sa_pol)
            else:
                f_exp = reward_model(sa_exp, s_exp_feat, s_exp_feat)
                f_pol = reward_model(sa_pol, s_pol_feat, s_pol_feat)

            # 简化 log pi: 使用常数近似，后续可替换为真实策略log prob。
            logp_exp = torch.zeros_like(f_exp)
            logp_pol = torch.zeros_like(f_pol)
            loss = airl_disc_loss(f_exp, logp_exp, f_pol, logp_pol)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        mean_loss = sum(losses) / max(len(losses), 1)
        metrics["epochs"].append({"epoch": epoch, "disc_loss": mean_loss})
        print(f"[irl_reward] epoch={epoch} disc_loss={mean_loss:.4f}")

    ckpt = {
        "encoder": encoder.state_dict(),
        "reward_model": reward_model.state_dict(),
        "vocab_itos": vocab.itos,
        "cfg": cfg,
    }
    torch.save(ckpt, run_dir / "reward_model.pt")
    save_json(metrics, run_dir / "metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
