"""Masked bidirectional transformer for iterative decoding of VQ-VAE token sequences.

SoundStorm/MaskGIT paradigm: train with random masking, generate via iterative
confidence-ordered unmasking. Full bidirectional context at every generation step.

Vocab: 0=stall, 1-1024=motion, 1025=MASK
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SoundStormBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, scale, shift, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask,
        )
        x = x + self.drop(attn_out)
        x = x * (1.0 + scale) + shift
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


class SoundStormTransformer(nn.Module):

    MASK_TOKEN_ID = 1025

    def __init__(
        self,
        vocab_size: int = 1025,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        cond_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.cond_dropout = cond_dropout

        self.tok_embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2 * n_layers),
        )

        self.layers = nn.ModuleList([
            SoundStormBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, condition, padding_mask=None):
        """
        tokens:       (B, T) int — may contain MASK_TOKEN_ID at masked positions
        condition:    (B, cond_dim) float
        padding_mask: (B, T) bool — True=valid, False=pad
        Returns:      (B, T, vocab_size) logits
        """
        B, T = tokens.shape

        pos = torch.arange(T, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        x = self.drop(x)

        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=condition.device) > self.cond_dropout).float()
            cond_input = condition * keep
        else:
            cond_input = condition

        film = self.cond_proj(cond_input).view(B, self.n_layers, 2, self.d_model)

        attn_key_mask = None
        if padding_mask is not None:
            attn_key_mask = ~padding_mask

        for i, layer in enumerate(self.layers):
            scale = film[:, i, 0].unsqueeze(1)
            shift = film[:, i, 1].unsqueeze(1)
            x = layer(x, scale, shift, key_padding_mask=attn_key_mask)

        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        seq_len: int,
        n_rounds: int = 16,
        temperature: float = 2.5,
        temp_floor: float = 0.5,
        cfg_scale: float = 0.0,
        top_p: float = 0.95,
        random_order_rounds: int = 3,
        initial_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Iterative masked decoding. Returns (B, seq_len) token ids.

        If initial_tokens is provided, non-MASK positions are treated as
        pre-revealed seeds. First `random_order_rounds` use random reveal
        ordering to avoid cold-start collapse.
        """
        B = condition.shape[0]
        device = condition.device

        if initial_tokens is not None:
            tokens = initial_tokens.clone().to(device)
            is_masked = tokens == self.MASK_TOKEN_ID
        else:
            tokens = torch.full(
                (B, seq_len), self.MASK_TOKEN_ID, dtype=torch.long, device=device,
            )
            is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        for r in range(n_rounds):
            frac = (r + 1) / n_rounds
            temp = temperature - (temperature - temp_floor) * frac

            logits = self.forward(tokens, condition)

            if cfg_scale > 0:
                logits_uncond = self.forward(tokens, torch.zeros_like(condition))
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            probs = torch.softmax(logits / max(temp, 0.01), dim=-1)

            sampled = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            confidence = torch.zeros(B, seq_len, device=device)

            for b in range(B):
                masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                p_batch = probs[b, masked_positions]
                if top_p < 1.0:
                    sorted_p, sorted_idx = p_batch.sort(dim=-1, descending=True)
                    cumsum = sorted_p.cumsum(dim=-1)
                    remove = cumsum - sorted_p > top_p
                    sorted_p[remove] = 0.0
                    p_batch.scatter_(1, sorted_idx, sorted_p)
                    p_batch = p_batch / p_batch.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                toks = torch.multinomial(p_batch, 1).squeeze(-1)
                sampled[b, masked_positions] = toks
                confidence[b, masked_positions] = torch.gather(
                    probs[b, masked_positions], 1, toks.unsqueeze(-1),
                ).squeeze(-1)

            if r < n_rounds - 1:
                target_frac = math.cos(frac * math.pi / 2)
            else:
                target_frac = 0.0

            use_random_order = r < random_order_rounds

            for b in range(B):
                masked_idx = is_masked[b].nonzero(as_tuple=True)[0]
                n_masked_b = len(masked_idx)
                if n_masked_b == 0:
                    continue

                n_to_keep_masked = max(0, int(target_frac * seq_len))
                n_to_reveal = n_masked_b - n_to_keep_masked

                if n_to_reveal <= 0:
                    continue

                if use_random_order:
                    perm = torch.randperm(n_masked_b, device=device)
                    reveal_idx = masked_idx[perm[:n_to_reveal]]
                else:
                    conf_vals = confidence[b, masked_idx]
                    _, top_order = conf_vals.sort(descending=True)
                    reveal_idx = masked_idx[top_order[:n_to_reveal]]

                tokens[b, reveal_idx] = sampled[b, reveal_idx]
                is_masked[b, reveal_idx] = False

        if is_masked.any():
            for b in range(B):
                remaining = is_masked[b].nonzero(as_tuple=True)[0]
                if len(remaining) > 0:
                    tokens[b, remaining] = sampled[b, remaining]

        return tokens

    @torch.no_grad()
    def generate_refine(
        self,
        condition: torch.Tensor,
        initial_tokens: torch.Tensor,
        n_rounds: int = 12,
        mask_ratio: float = 0.4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Iterative refinement: start with a full sequence, repeatedly mask
        and re-predict subsets. Avoids the cold-start problem."""
        B, T = initial_tokens.shape
        device = condition.device
        tokens = initial_tokens.clone().to(device)

        for r in range(n_rounds):
            temp = temperature * (1.0 - 0.3 * r / max(n_rounds - 1, 1))

            n_mask = max(1, int(mask_ratio * T * (1.0 - 0.5 * r / n_rounds)))
            masked = tokens.clone()
            for b in range(B):
                idx = torch.randperm(T, device=device)[:n_mask]
                masked[b, idx] = self.MASK_TOKEN_ID

            logits = self.forward(masked, condition)

            if cfg_scale > 0:
                logits_u = self.forward(masked, torch.zeros_like(condition))
                logits = logits_u + cfg_scale * (logits - logits_u)

            probs = torch.softmax(logits / max(temp, 0.01), dim=-1)

            for b in range(B):
                mask_pos = (masked[b] == self.MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                p_batch = probs[b, mask_pos]
                if top_p < 1.0:
                    sorted_p, sorted_idx = p_batch.sort(dim=-1, descending=True)
                    cumsum = sorted_p.cumsum(dim=-1)
                    remove = cumsum - sorted_p > top_p
                    sorted_p[remove] = 0.0
                    p_batch.scatter_(1, sorted_idx, sorted_p)
                    p_batch = p_batch / p_batch.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                new_toks = torch.multinomial(p_batch, 1).squeeze(-1)
                tokens[b, mask_pos] = new_toks

        return tokens
