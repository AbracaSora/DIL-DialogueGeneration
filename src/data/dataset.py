from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


def simple_tokenize(text: str) -> List[str]:
    return text.strip().lower().split()


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 30000) -> "Vocab":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(simple_tokenize(t))

        base_tokens = [PAD, UNK, BOS, EOS]
        keep = [w for w, _ in counter.most_common(max(0, max_size - len(base_tokens)))]
        itos = base_tokens + keep
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_len: int, add_bos_eos: bool = True) -> List[int]:
        tokens = simple_tokenize(text)
        if add_bos_eos:
            tokens = [BOS] + tokens[: max_len - 2] + [EOS]
        else:
            tokens = tokens[:max_len]
        return [self.stoi.get(tok, self.stoi[UNK]) for tok in tokens]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        words: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            w = self.itos[idx]
            if skip_special and w in {PAD, BOS, EOS}:
                continue
            words.append(w)
        return " ".join(words).strip()

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]


class DialoguePairDataset(Dataset):
    """
    训练样本格式为 jsonl，每行:
      {"context": ["u_{t-2}", "u_{t-1}"], "target": "u_t"}
    """

    def __init__(
        self,
        rows: List[Tuple[str, str]],
        vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 32,
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 32,
    ) -> "DialoguePairDataset":
        rows: List[Tuple[str, str]] = []
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                context_turns = obj.get("context", [])
                if isinstance(context_turns, list):
                    src = " ".join(context_turns[-2:])
                else:
                    src = str(context_turns)
                tgt = str(obj.get("target", ""))
                rows.append((src, tgt))
        return cls(rows=rows, vocab=vocab, max_src_len=max_src_len, max_tgt_len=max_tgt_len)

    @classmethod
    def synthetic(
        cls,
        size: int,
        vocab: Vocab,
        max_src_len: int = 64,
        max_tgt_len: int = 32,
        seed: int = 42,
    ) -> "DialoguePairDataset":
        random.seed(seed)
        templates = [
            ("how are you", "i am fine how about you"),
            ("what is your name", "my name is bot what is yours"),
            ("do you like music", "yes i like music what do you like"),
            ("where are you going", "i am going home do you want to come"),
            ("why are you here", "i am here to help you"),
            ("how old are you", "i am sixteen why are you asking"),
        ]
        rows = [random.choice(templates) for _ in range(size)]
        return cls(rows=rows, vocab=vocab, max_src_len=max_src_len, max_tgt_len=max_tgt_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text, tgt_text = self.rows[idx]
        src_ids = self.vocab.encode(src_text, self.max_src_len, add_bos_eos=True)
        tgt_ids = self.vocab.encode(tgt_text, self.max_tgt_len, add_bos_eos=True)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    src_list = [b["src_ids"] for b in batch]
    tgt_list = [b["tgt_ids"] for b in batch]

    src_len = torch.tensor([x.size(0) for x in src_list], dtype=torch.long)
    tgt_len = torch.tensor([x.size(0) for x in tgt_list], dtype=torch.long)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)

    return {"src_ids": src_pad, "tgt_ids": tgt_pad, "src_len": src_len, "tgt_len": tgt_len}


def load_or_build_vocab(
    train_path: str | Path,
    max_size: int = 30000,
    fallback_size: int = 5000,
) -> Vocab:
    p = Path(train_path)
    texts: List[str] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            context_turns = obj.get("context", [])
            if isinstance(context_turns, list):
                texts.extend([str(x) for x in context_turns])
            else:
                texts.append(str(context_turns))
            texts.append(str(obj.get("target", "")))
    if not texts:
        # 空仓库也能快速跑通脚本。
        texts = [f"token_{i}" for i in range(fallback_size)]
    return Vocab.build(texts=texts, max_size=max_size)
