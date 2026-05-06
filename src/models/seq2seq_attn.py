from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Seq2SeqOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, enc_outputs: torch.Tensor, dec_hidden: torch.Tensor, enc_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_outputs: [B, S, H], dec_hidden: [B, H]
        scores = self.v(torch.tanh(self.w_enc(enc_outputs) + self.w_dec(dec_hidden).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(enc_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn


class Seq2SeqAttn(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = BahdanauAttention(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(src_ids))
        enc_outputs, enc_hidden = self.encoder(emb)
        enc_mask = (src_ids != self.pad_id).long()
        return enc_outputs, enc_hidden, enc_mask

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> Seq2SeqOutput:
        enc_outputs, enc_hidden, enc_mask = self.encode(src_ids)
        batch_size, tgt_len = tgt_ids.size()

        logits = []
        dec_hidden = enc_hidden[:1]  # [1, B, H]
        prev_token = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            token_emb = self.embedding(prev_token).unsqueeze(1)  # [B, 1, E]
            context, _ = self.attn(enc_outputs, dec_hidden.squeeze(0), enc_mask)
            dec_in = torch.cat([token_emb, context.unsqueeze(1)], dim=-1)
            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
            step_logits = self.out_proj(torch.cat([dec_out.squeeze(1), context], dim=-1))
            logits.append(step_logits)
            prev_token = tgt_ids[:, t]

        logits_tensor = torch.stack(logits, dim=1)  # [B, T-1, V]
        target = tgt_ids[:, 1:]
        loss = F.cross_entropy(logits_tensor.reshape(-1, logits_tensor.size(-1)), target.reshape(-1), ignore_index=self.pad_id)
        return Seq2SeqOutput(logits=logits_tensor, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_len: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        sample: bool = True,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        enc_outputs, enc_hidden, enc_mask = self.encode(src_ids)
        batch_size = src_ids.size(0)
        dec_hidden = enc_hidden[:1]
        prev_token = torch.full((batch_size,), self.bos_id, device=src_ids.device, dtype=torch.long)

        out_tokens = [prev_token]
        for _ in range(max_len - 1):
            token_emb = self.embedding(prev_token).unsqueeze(1)
            context, _ = self.attn(enc_outputs, dec_hidden.squeeze(0), enc_mask)
            dec_in = torch.cat([token_emb, context.unsqueeze(1)], dim=-1)
            dec_out, dec_hidden = self.decoder(dec_in, dec_hidden)
            logits = self.out_proj(torch.cat([dec_out.squeeze(1), context], dim=-1))
            logits = logits / max(temperature, 1e-6)

            if top_k > 0:
                v, i = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.zeros_like(logits).scatter(-1, i, torch.softmax(v, dim=-1))
            else:
                probs = torch.softmax(logits, dim=-1)

            if sample:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(probs, dim=-1)
            out_tokens.append(next_token)
            prev_token = next_token

        out = torch.stack(out_tokens, dim=1)
        if was_training:
            self.train()
        return out
