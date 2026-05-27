"""CANDI: Hybrid discrete-continuous diffusion for mouse trajectory generation.

Two coupled channels in a single Transformer denoiser:
- Continuous: Gaussian diffusion on (dx, dy) displacements
- Discrete: Absorbing-state masking on stall/no-stall labels

The discrete channel produces exact-zero stalls. The continuous channel
produces smooth displacements. The shared backbone learns the coupling.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((t / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999).float()


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CANDIBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(d_model, d_model * 2)

    def forward(self, x, cond_emb, key_padding_mask=None):
        scale, shift = self.film(cond_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        x = x * (1.0 + scale) + shift
        x = x + self.ff(self.norm2(x))
        return x


class CANDIModel(nn.Module):

    STALL_MASK = -1.0

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        n_diffusion_steps: int = 1000,
        cond_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_steps = n_diffusion_steps
        self.cond_dropout = cond_dropout

        self.input_proj = nn.Linear(4, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            CANDIBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.cont_head = nn.Linear(d_model, 2)
        self.disc_head = nn.Linear(d_model, 1)

        betas = cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_1mab", torch.sqrt(1 - alpha_bar))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def q_continuous(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_ab[t].view(-1, 1, 1)
        n = self.sqrt_1mab[t].view(-1, 1, 1)
        return s * x0 + n * noise, noise

    def q_discrete(self, stall, t):
        mask_prob = 1.0 - self.sqrt_ab[t].view(-1, 1)
        mask = torch.rand_like(stall.float()) < mask_prob
        out = stall.clone().float()
        out[mask] = self.STALL_MASK
        return out, mask

    def forward(self, dxdy_noisy, stall_state, mask_flag, t, cond, pad_mask=None):
        B, T = dxdy_noisy.shape[:2]
        inp = torch.cat([
            dxdy_noisy,
            stall_state.unsqueeze(-1),
            mask_flag.unsqueeze(-1).float(),
        ], dim=-1)

        x = self.input_proj(inp) + self.pos_embed(torch.arange(T, device=inp.device))

        t_emb = self.time_embed(t)
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=cond.device) > self.cond_dropout).float()
            cond = cond * keep
        c_emb = self.cond_embed(cond)
        combined = t_emb + c_emb

        kpm = ~pad_mask if pad_mask is not None else None
        for layer in self.layers:
            x = layer(x, combined, key_padding_mask=kpm)

        x = self.norm(x)
        return self.cont_head(x), self.disc_head(x).squeeze(-1)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        seq_len: int,
        n_steps: int = 50,
        eta: float = 0.0,
        cfg_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = cond.shape[0]
        dev = cond.device

        xt = torch.randn(B, seq_len, 2, device=dev)
        stall_s = torch.full((B, seq_len), self.STALL_MASK, device=dev)
        mflag = torch.ones(B, seq_len, device=dev)

        step_size = self.n_steps // n_steps
        times = list(range(self.n_steps - 1, -1, -step_size))
        if times[-1] != 0:
            times.append(0)

        for i, tv in enumerate(times):
            t = torch.full((B,), tv, dtype=torch.long, device=dev)
            dp, sl = self.forward(xt, stall_s, mflag, t, cond)

            if cfg_scale > 0:
                dp_u, sl_u = self.forward(
                    xt, stall_s, mflag, t, torch.zeros_like(cond),
                )
                dp = dp_u + cfg_scale * (dp - dp_u)
                sl = sl_u + cfg_scale * (sl - sl_u)

            frac = 1.0 - tv / self.n_steps
            if frac > 0.3:
                conf = torch.abs(sl)
                thresh = max(0.5, 3.0 * (1.0 - frac))
                reveal = (conf > thresh) & (mflag > 0.5)
                stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
                mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

            if tv > 0:
                nt = times[i + 1] if i + 1 < len(times) else 0
                ab_t = self.alpha_bar[tv]
                ab_n = self.alpha_bar[nt] if nt > 0 else torch.ones(1, device=dev)
                eps = (xt - ab_t.sqrt() * dp) / self.sqrt_1mab[tv].clamp(min=1e-8)
                if eta > 0 and nt > 0:
                    sig = eta * ((1 - ab_n) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_n).sqrt()
                    noise = torch.randn_like(xt) * sig
                else:
                    sig = torch.zeros(1, device=dev)
                    noise = 0.0
                dir_coef = (1 - ab_n - sig ** 2).clamp(min=0).sqrt()
                xt = ab_n.sqrt() * dp + dir_coef * eps + noise
            else:
                xt = dp

        sp = torch.sigmoid(sl)
        final_stall = torch.where(mflag > 0.5, (sp > 0.5).float(), stall_s)
        out = xt.clone()
        out[final_stall > 0.5] = 0.0
        return out, final_stall
