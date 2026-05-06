from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


if sys.platform.startswith("win"):
    # Windows + conda/pytorch 环境下常见 OpenMP 重复加载问题。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    parent = data.get("extends")
    if parent:
        base = load_yaml(str(Path(parent)))
        return deep_update(base, data)
    return data


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
