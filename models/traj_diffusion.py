"""
Trajectory Diffusion Model.

Non-autoregressive: generates all (dx, dy) timesteps simultaneously via
denoising diffusion. Bidirectional Transformer denoiser with FiLM conditioning.
Predicts x_0 directly (not noise).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, scale, shift):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = x * (1 + scale) + shift
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


class TrajectoryDiffusionModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        n_diff_steps: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_diff_steps = n_diff_steps

        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2 * n_layers),
        )

        self.layers = nn.ModuleList([
            DiffusionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)
        self.dropout_layer = nn.Dropout(dropout)

        betas = cosine_beta_schedule(n_diff_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_omab * noise

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t.

        x_noisy: (B, T, 2)
        t: (B,) diffusion timestep indices
        condition: (B, 4) — (log_dist, log_dur, cos_a, sin_a)
        padding_mask: (B, T) — True for VALID positions
        Returns: (B, T, 2) predicted x_0
        """
        B, T, _ = x_noisy.shape
        pos = torch.arange(T, device=x_noisy.device).unsqueeze(0)
        x = self.input_proj(x_noisy) + self.pos_embed(pos)

        t_emb = self.time_embed(t)
        x = x + t_emb.unsqueeze(1)
        x = self.dropout_layer(x)

        film = self.cond_proj(condition)
        film = film.view(B, self.n_layers, 2, self.d_model)

        kpm = ~padding_mask if padding_mask is not None else None

        for i, layer in enumerate(self.layers):
            scale = film[:, i, 0].unsqueeze(1)
            shift = film[:, i, 1].unsqueeze(1)
            x = layer(x, kpm, scale, shift)

        x = self.norm(x)
        return self.output_proj(x)

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        seq_len: int,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling. Returns (B, seq_len, 2) predicted x_0."""
        B = condition.shape[0]
        device = condition.device

        x = torch.randn(B, seq_len, 2, device=device)
        mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        timesteps = torch.linspace(
            self.n_diff_steps - 1, 0, n_steps + 1, device=device,
        ).long()

        for i in range(n_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = t_now.expand(B)
            x_0_pred = self(x, t_batch, condition, mask)

            if t_next > 0:
                ab_now = self.alpha_bar[t_now]
                ab_next = self.alpha_bar[t_next]

                pred_noise = (
                    x - torch.sqrt(ab_now) * x_0_pred
                ) / torch.sqrt(1.0 - ab_now).clamp(min=1e-8)

                sigma = eta * torch.sqrt(
                    (1 - ab_next) / (1 - ab_now) * (1 - ab_now / ab_next)
                )

                x = (
                    torch.sqrt(ab_next) * x_0_pred
                    + torch.sqrt((1 - ab_next - sigma ** 2).clamp(min=0)) * pred_noise
                    + sigma * torch.randn_like(x)
                )
            else:
                x = x_0_pred

        return x
