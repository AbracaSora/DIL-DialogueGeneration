from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DialoguePairDataset, collate_batch, load_or_build_vocab
from src.models.seq2seq_attn import Seq2SeqAttn
from src.utils import ensure_dir, resolve_device


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, object]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    vocab = load_or_build_vocab(data_cfg["train_path"], max_size=data_cfg["max_vocab_size"])

    train_ds = DialoguePairDataset.from_jsonl(
        data_cfg["train_path"], vocab, max_src_len=data_cfg["max_src_len"], max_tgt_len=data_cfg["max_tgt_len"]
    )
    valid_ds = DialoguePairDataset.from_jsonl(
        data_cfg["valid_path"], vocab, max_src_len=data_cfg["max_src_len"], max_tgt_len=data_cfg["max_tgt_len"]
    )

    if len(train_ds) == 0:
        train_ds = DialoguePairDataset.synthetic(2000, vocab, data_cfg["max_src_len"], data_cfg["max_tgt_len"])
    if len(valid_ds) == 0:
        valid_ds = DialoguePairDataset.synthetic(200, vocab, data_cfg["max_src_len"], data_cfg["max_tgt_len"], seed=7)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
    )
    return train_loader, valid_loader, vocab


def build_model(cfg: Dict, vocab) -> Seq2SeqAttn:
    model_cfg = cfg["model"]
    return Seq2SeqAttn(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def maybe_load_checkpoint(path: str | Path, model: torch.nn.Module) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    ckpt = torch.load(p, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return True


def prepare_run_dir(cfg: Dict) -> Path:
    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["experiment_name"])
    return run_dir


def cfg_device(cfg: Dict) -> torch.device:
    return resolve_device(cfg.get("device", "cpu"))
