"""
Autoregressive transformer for mouse trajectory token sequences.

Generates sequences of VQ-VAE motion tokens conditioned on trajectory parameters
(log_distance, log_duration, cos_angle, sin_angle).

Architecture: 4 layers, 256 dim, 4 heads, 1024 FFN, dropout 0.1
Context window: 256 tokens
Vocabulary: 1025 (token 0 = stall, tokens 1-1024 = motion)

Conditioning: FiLM-style - MLP maps conditions to scale+shift at each layer.
Endpoint: per-layer injection of (remaining_dx, remaining_dy, remaining_frac).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,  # 1 stall + 1024 motion
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,  # (log_dist, log_dur, cos_a, sin_a)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Token embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Condition projection (FiLM: condition → scale + shift for each layer)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2 * n_layers),  # scale + shift per layer
        )

        # Endpoint conditioning: (remaining_dx_norm, remaining_dy_norm, remaining_frac) → d_model
        self.endpoint_proj = nn.Linear(3, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        # Pre-compute causal mask (avoids per-forward allocation)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,        # (B, T) token indices
        condition: torch.Tensor,      # (B, 4) trajectory conditions
        endpoint_info: torch.Tensor | None = None,  # (B, T, 3) per-step endpoint info
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # Embeddings
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        x = self.dropout(x)

        # Condition → FiLM params
        film = self.cond_proj(condition)  # (B, d_model * 2 * n_layers)
        film = film.view(B, self.n_layers, 2, self.d_model)  # (B, L, 2, D)

        # Endpoint conditioning (optional)
        if endpoint_info is not None:
            ep_embed = self.endpoint_proj(endpoint_info)  # (B, T, D)
            x = x + ep_embed

        # Causal mask (sliced from pre-computed buffer)
        mask = self.causal_mask[:T, :T]

        # Transformer layers with FiLM conditioning
        for i, layer in enumerate(self.layers):
            scale = film[:, i, 0, :].unsqueeze(1)  # (B, 1, D)
            shift = film[:, i, 1, :].unsqueeze(1)  # (B, 1, D)
            x = layer(x, mask, scale, shift)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,     # (B, T, D)
        mask: torch.Tensor,   # (T, T) causal mask
        scale: torch.Tensor,  # (B, 1, D) FiLM scale
        shift: torch.Tensor,  # (B, 1, D) FiLM shift
    ) -> torch.Tensor:
        # Self-attention with causal mask
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # FiLM conditioning
        x = x * (1 + scale) + shift

        # Feed-forward
        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x
