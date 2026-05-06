from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.dialogue_env import distinct_n
from src.utils import save_json


def load_lines(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="outputs/generated.txt")
    parser.add_argument("--out_file", default="outputs/dialogue_metrics.json")
    args = parser.parse_args()

    texts = load_lines(args.pred_file)
    result = {
        "count": len(texts),
        "distinct1": distinct_n(texts, n=1),
        "distinct2": distinct_n(texts, n=2),
        "avg_len_tokens": (sum(len(t.split()) for t in texts) / max(len(texts), 1)),
    }
    save_json(result, args.out_file)
    print(result)


if __name__ == "__main__":
    main()
