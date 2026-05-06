from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.train_common import build_dataloaders, build_model, cfg_device, prepare_run_dir, save_checkpoint
from src.utils import load_yaml, save_json, set_seed


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            out = model(src_ids, tgt_ids)
            losses.append(float(out.loss.item()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seq2seq.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg_device(cfg)

    train_loader, valid_loader, vocab = build_dataloaders(cfg)
    model = build_model(cfg, vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    run_dir = prepare_run_dir(cfg)
    best_loss = float("inf")
    metrics = {"epochs": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(src_ids, tgt_ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            train_losses.append(float(out.loss.item()))

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        valid_loss = evaluate(model, valid_loader, device)
        metrics["epochs"].append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"[seq2seq] epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        last_ckpt = run_dir / "last.pt"
        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            meta={"cfg": cfg, "vocab_itos": vocab.itos, "valid_loss": valid_loss, "epoch": epoch},
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_ckpt = run_dir / "best.pt"
            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                meta={"cfg": cfg, "vocab_itos": vocab.itos, "valid_loss": valid_loss, "epoch": epoch},
            )

    save_json(metrics, run_dir / "metrics.json")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
