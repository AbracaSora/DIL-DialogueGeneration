from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F


@dataclass
class DialogueStep:
    state_text: str
    action_text: str
    reward: float


@dataclass
class DialogueEpisode:
    steps: List[DialogueStep]

    @property
    def total_reward(self) -> float:
        return float(sum(s.reward for s in self.steps))

    @property
    def num_turns(self) -> int:
        return len(self.steps)


class TwoAgentDialogueEnv:
    """
    轻量版双agent模拟器:
    - 两个agent共享同一策略模型(与 Li2016 的自博弈思想一致)
    - 状态为最近两轮文本拼接
    """

    def __init__(
        self,
        model,
        vocab,
        reward_fn: Callable[[str, str, str], float],
        max_turns: int = 5,
        max_decode_len: int = 24,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.max_turns = max_turns
        self.max_decode_len = max_decode_len

    @torch.no_grad()
    def rollout(self, initial_message: str, device: torch.device) -> DialogueEpisode:
        history: List[str] = [initial_message]
        steps: List[DialogueStep] = []

        for _ in range(self.max_turns):
            state_text = " ".join(history[-2:])
            src_ids = torch.tensor(
                [self.vocab.encode(state_text, max_len=64, add_bos_eos=True)],
                dtype=torch.long,
                device=device,
            )
            gen_ids = self.model.generate(src_ids, max_len=self.max_decode_len, sample=True)
            action_text = self.vocab.decode(gen_ids[0].tolist(), skip_special=True).strip()
            if not action_text:
                action_text = "..."
            prev_turn = history[-1] if history else ""
            reward = self.reward_fn(state_text, prev_turn, action_text)
            steps.append(DialogueStep(state_text=state_text, action_text=action_text, reward=reward))
            history.append(action_text)
        return DialogueEpisode(steps=steps)


def distinct_n(texts: Sequence[str], n: int = 1) -> float:
    all_ngrams: List[tuple] = []
    for t in texts:
        toks = t.split()
        if len(toks) < n:
            continue
        all_ngrams.extend(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def repetition_ratio(text_a: str, text_b: str) -> float:
    a = set(text_a.split())
    b = set(text_b.split())
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)
