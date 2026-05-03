"""1D Temporal U-Net with FiLM conditioning for flow matching and diffusion.

Architecture:
  - Input:  x_t (B, 192, C) noisy trajectory (C=2 positions, C=3 positions+timing)
  - Cond:   c (B, 4) [log_dist, log_duration, cos_angle, sin_angle]
  - Time:   t (B,) flow interpolation parameter in [0, 1]
  - Output: v_theta (B, 192, C) predicted velocity field

~2M parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar time values.

    Args:
        t: (B,) tensor of time values
        dim: embedding dimension (must be even)

    Returns:
        (B, dim) embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLMConditioner(nn.Module):
    """Produces per-layer FiLM scale and shift from condition + time embedding."""

    def __init__(self, cond_dim: int, time_dim: int, d_model: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.d_model = d_model

    def forward(self, c: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Returns (B, d_model) embedding."""
        return self.mlp(torch.cat([c, t_embed], dim=-1))


class FiLMBlock(nn.Module):
    """Conv1D block with GroupNorm + SiLU + FiLM modulation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, film_dim: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film_proj = nn.Linear(film_dim, 2 * out_ch)

    def forward(self, x: torch.Tensor, film_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) conv feature map
            film_embed: (B, film_dim) conditioning embedding
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        # FiLM modulation
        params = self.film_proj(film_embed)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each (B, C)
        scale = scale.unsqueeze(-1)  # (B, C, 1)
        shift = shift.unsqueeze(-1)  # (B, C, 1)
        return scale * x + shift


class TemporalUNet(nn.Module):
    """1D U-Net for flow matching on trajectories.

    Input:  x_t (B, 192, C)  - noisy trajectory (C=in_channels)
    Cond:   c   (B, 4)       - movement conditions
    Time:   t   (B,)         - flow time in [0, 1]
    Output: v   (B, 192, C)  - predicted velocity field
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_dim: int = 4,
        time_dim: int = 64,
        film_dim: int = 128,
        encoder_channels: tuple = (64, 128, 256),
        kernel_sizes: tuple = (7, 5, 3),
    ):
        super().__init__()
        self.time_dim = time_dim
        self.in_channels = in_channels

        c1, c2, c3 = encoder_channels
        k1, k2, k3 = kernel_sizes

        # FiLM conditioner
        self.film = FiLMConditioner(cond_dim, time_dim, film_dim)

        # Encoder
        self.enc1a = FiLMBlock(in_channels, c1, kernel_size=k1, film_dim=film_dim)
        self.enc1b = FiLMBlock(c1, c1, kernel_size=3, film_dim=film_dim)

        self.enc2a = FiLMBlock(c1, c2, kernel_size=k2, film_dim=film_dim)
        self.enc2b = FiLMBlock(c2, c2, kernel_size=3, film_dim=film_dim)

        self.enc3a = FiLMBlock(c2, c3, kernel_size=k3, film_dim=film_dim)
        self.enc3b = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)

        # Bottleneck
        self.bot_a = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)
        self.bot_b = FiLMBlock(c3, c3, kernel_size=3, film_dim=film_dim)

        # Decoder (skip connections double input channels)
        self.dec3 = FiLMBlock(c3 * 2, c3, kernel_size=k3, film_dim=film_dim)
        self.dec2 = FiLMBlock(c3 + c2, c2, kernel_size=k2, film_dim=film_dim)
        self.dec1 = FiLMBlock(c2 + c1, c1, kernel_size=k1, film_dim=film_dim)

        # Output projection
        self.out_conv = nn.Conv1d(c1, in_channels, kernel_size=1)


    def forward(
        self, x_t: torch.Tensor, t_flow: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t:       (B, 192, C) noisy trajectory
            t_flow:    (B,) flow time in [0, 1]
            condition: (B, 4) movement conditions

        Returns:
            v_theta: (B, 192, C) predicted velocity
        """
        # Time embedding
        t_embed = sinusoidal_embedding(t_flow, self.time_dim)  # (B, 64)
        film_embed = self.film(condition, t_embed)  # (B, 128)

        # Transpose to channels-first: (B, 192, C) -> (B, C, 192)
        x = x_t.transpose(1, 2)

        # Encoder
        h1 = self.enc1b(self.enc1a(x, film_embed), film_embed)  # (B, 64, 192)
        h1_down = F.avg_pool1d(h1, 2)  # (B, 64, 96)

        h2 = self.enc2b(self.enc2a(h1_down, film_embed), film_embed)  # (B, 128, 96)
        h2_down = F.avg_pool1d(h2, 2)  # (B, 128, 48)

        h3 = self.enc3b(self.enc3a(h2_down, film_embed), film_embed)  # (B, 256, 48)
        h3_down = F.avg_pool1d(h3, 2)  # (B, 256, 24)

        # Bottleneck
        bot = self.bot_b(self.bot_a(h3_down, film_embed), film_embed)  # (B, 256, 24)

        # Decoder with skip connections
        d3 = F.interpolate(bot, size=h3.shape[-1], mode="nearest")  # (B, 256, 48)
        d3 = torch.cat([d3, h3], dim=1)  # (B, 512, 48)
        d3 = self.dec3(d3, film_embed)  # (B, 256, 48)

        d2 = F.interpolate(d3, size=h2.shape[-1], mode="nearest")  # (B, 256, 96)
        d2 = torch.cat([d2, h2], dim=1)  # (B, 384, 96)
        d2 = self.dec2(d2, film_embed)  # (B, 128, 96)

        d1 = F.interpolate(d2, size=h1.shape[-1], mode="nearest")  # (B, 128, 192)
        d1 = torch.cat([d1, h1], dim=1)  # (B, 192, 192)
        d1 = self.dec1(d1, film_embed)  # (B, 64, 192)

        # Output
        v = self.out_conv(d1)  # (B, C, 192)

        # Transpose back: (B, C, 192) -> (B, 192, C)
        return v.transpose(1, 2)
