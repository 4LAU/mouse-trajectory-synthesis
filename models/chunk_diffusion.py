"""Chunk-level diffusion model for mouse trajectory generation.

Generates 25-step chunks via DDPM, sequenced autoregressively.
Predicts (dx, dy, stall_logit) jointly. Uses cosine schedule, x_0 prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_unet import sinusoidal_embedding, FiLMConditioner, FiLMBlock


class ContextEncoder(nn.Module):
    """Encode 5-step context (dx, dy, stall) into a fixed-size embedding."""

    def __init__(self, context_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(3, context_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(context_dim, context_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx: (B, 5, 3) -> (B, context_dim)"""
        x = ctx.transpose(1, 2)  # (B, 3, 5)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, context_dim)
        return x


class ChunkUNet(nn.Module):
    """1D U-Net for 25-step chunk diffusion.

    Input:  x_t (B, 25, 3)  — noisy chunk (dx, dy, stall_logit)
    Cond:   c   (B, cond_dim) — concatenated condition
    Time:   t   (B,)         — diffusion timestep
    Output: x_0 (B, 25, 3)  — predicted clean chunk
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_dim: int = 42,
        time_dim: int = 64,
        film_dim: int = 128,
        encoder_channels: tuple = (48, 96, 192),
        kernel_sizes: tuple = (5, 3, 3),
    ):
        super().__init__()
        self.time_dim = time_dim

        c1, c2, c3 = encoder_channels
        k1, k2, k3 = kernel_sizes

        self.film = FiLMConditioner(cond_dim, time_dim, film_dim)

        # Encoder: 25 -> 12 -> 6 -> 3
        self.enc1a = FiLMBlock(in_channels, c1, kernel_size=k1, film_dim=film_dim)
        self.enc1b = FiLMBlock(c1, c1, kernel_size=3, film_dim=film_dim)

        self.enc2a = FiLMBlock(c1, c2, kernel_size=k2, film_dim=film_dim)
        self.enc2b = FiLMBlock(c2, c2, kernel_size=3, film_dim=film_dim)

        self.enc3a = FiLMBlock(c2, c3, kernel_size=k3, film_dim=film_dim)
        self.enc3b = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)

        # Bottleneck
        self.bot_a = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)
        self.bot_b = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)

        # Decoder
        self.dec3 = FiLMBlock(c3 * 2, c3, kernel_size=k3, film_dim=film_dim)
        self.dec2 = FiLMBlock(c3 + c2, c2, kernel_size=k2, film_dim=film_dim)
        self.dec1 = FiLMBlock(c2 + c1, c1, kernel_size=k1, film_dim=film_dim)

        self.out_conv = nn.Conv1d(c1, in_channels, kernel_size=1)

    def forward(self, x_t, t, condition):
        """
        x_t:       (B, 25, 3)
        t:         (B,) diffusion timestep (integer)
        condition: (B, cond_dim)
        Returns:   (B, 25, 3) predicted x_0
        """
        t_embed = sinusoidal_embedding(t.float(), self.time_dim)
        film_embed = self.film(condition, t_embed)

        x = x_t.transpose(1, 2)  # (B, 3, 25)

        h1 = self.enc1b(self.enc1a(x, film_embed), film_embed)  # (B, c1, 25)
        h1_down = F.avg_pool1d(h1, 2)  # (B, c1, 12)

        h2 = self.enc2b(self.enc2a(h1_down, film_embed), film_embed)  # (B, c2, 12)
        h2_down = F.avg_pool1d(h2, 2)  # (B, c2, 6)

        h3 = self.enc3b(self.enc3a(h2_down, film_embed), film_embed)  # (B, c3, 6)
        h3_down = F.avg_pool1d(h3, 2)  # (B, c3, 3)

        bot = self.bot_b(self.bot_a(h3_down, film_embed), film_embed)

        d3 = F.interpolate(bot, size=h3.shape[-1], mode="nearest")
        d3 = self.dec3(torch.cat([d3, h3], dim=1), film_embed)

        d2 = F.interpolate(d3, size=h2.shape[-1], mode="nearest")
        d2 = self.dec2(torch.cat([d2, h2], dim=1), film_embed)

        d1 = F.interpolate(d2, size=h1.shape[-1], mode="nearest")
        d1 = self.dec1(torch.cat([d1, h1], dim=1), film_embed)

        v = self.out_conv(d1)
        return v.transpose(1, 2)  # (B, 25, 3)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


class ChunkDiffusionModel(nn.Module):
    """Full chunk diffusion: U-Net + context encoder + diffusion schedule."""

    def __init__(
        self,
        n_diff_steps: int = 200,
        context_dim: int = 32,
        global_cond_dim: int = 4,
        local_cond_dim: int = 6,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_diff_steps = n_diff_steps
        self.cond_dropout = cond_dropout

        total_cond_dim = global_cond_dim + local_cond_dim + context_dim
        self.context_encoder = ContextEncoder(context_dim)
        self.unet = ChunkUNet(in_channels=3, cond_dim=total_cond_dim)

        betas = cosine_beta_schedule(n_diff_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_omab * noise

    def forward(self, chunk_x0, t, context, global_cond, local_cond, noise=None):
        """Training forward pass.

        chunk_x0:    (B, 25, 3) — clean chunk (dx, dy, stall_logit)
        t:           (B,) — diffusion timestep
        context:     (B, 5, 3) — previous chunk tail
        global_cond: (B, 4) — (log_dist, log_dur, cos_a, sin_a)
        local_cond:  (B, 6) — (rem_dx, rem_dy, rem_frac, progress, cum_dx, cum_dy)
        """
        x_t = self.q_sample(chunk_x0, t, noise)

        ctx_embed = self.context_encoder(context)

        if self.training and self.cond_dropout > 0:
            mask = torch.rand(global_cond.shape[0], 1, device=global_cond.device) > self.cond_dropout
            global_cond = global_cond * mask
            local_cond = local_cond * mask
            ctx_embed = ctx_embed * mask

        condition = torch.cat([global_cond, local_cond, ctx_embed], dim=-1)
        x_0_pred = self.unet(x_t, t, condition)
        return x_0_pred

    @torch.no_grad()
    def ddim_sample(self, context, global_cond, local_cond,
                    n_steps=50, eta=0.3, cfg_scale=0.0):
        """DDIM sampling for a single chunk. Returns (B, 25, 3)."""
        B = global_cond.shape[0]
        device = global_cond.device

        x = torch.randn(B, 25, 3, device=device)
        ctx_embed = self.context_encoder(context)

        condition = torch.cat([global_cond, local_cond, ctx_embed], dim=-1)

        if cfg_scale > 0:
            uncond = torch.zeros_like(condition)

        timesteps = torch.linspace(
            self.n_diff_steps - 1, 0, n_steps + 1, device=device
        ).long()

        for i in range(n_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t_now.expand(B)

            x_0_pred = self.unet(x, t_batch, condition)

            if cfg_scale > 0:
                x_0_uncond = self.unet(x, t_batch, uncond)
                x_0_pred = x_0_uncond + cfg_scale * (x_0_pred - x_0_uncond)

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
