"""
ZIMT: Zero-Inflated Mouse Trajectory Generator.

At each timestep the model makes two decisions:
  1. Gate: P(stall | context) — binary, produces exact (0, 0)
  2. MDN: mixture of bivariate Gaussians for (dx, dy) if moving

Architecture: causal Transformer + FiLM conditioning + dual output heads.
Input per step: (dx_prev, dy_prev, stall_prev, remaining_dx, remaining_dy, remaining_frac)
Condition (FiLM): (log_dist, log_dur, cos_angle, sin_angle)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ZIMTModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        cond_dim: int = 4,
        n_components: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_components = n_components
        M = n_components

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2 * n_layers),
        )

        self.layers = nn.ModuleList([
            ZIMTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.gate_head = nn.Linear(d_model, 1)
        # Per component: mu_x, mu_y, log_sigma_x, log_sigma_y, atanh_rho, logit_pi
        self.mdn_head = nn.Linear(d_model, M * 6)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.mdn_head.bias)

    def forward(
        self,
        input_seq: torch.Tensor,   # (B, T, input_dim)
        condition: torch.Tensor,    # (B, 4)
    ) -> dict:
        B, T, _ = input_seq.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=input_seq.device).unsqueeze(0)
        x = self.input_proj(input_seq) + self.pos_embed(pos)
        x = self.dropout(x)

        film = self.cond_proj(condition)
        film = film.view(B, self.n_layers, 2, self.d_model)

        mask = self.causal_mask[:T, :T]

        for i, layer in enumerate(self.layers):
            scale = film[:, i, 0, :].unsqueeze(1)
            shift = film[:, i, 1, :].unsqueeze(1)
            x = layer(x, mask, scale, shift)

        x = self.norm(x)

        gate_logit = self.gate_head(x).squeeze(-1)  # (B, T)

        raw_mdn = self.mdn_head(x)  # (B, T, M*6)
        M = self.n_components
        mu = raw_mdn[:, :, :M * 2].reshape(B, T, M, 2)
        log_sigma = raw_mdn[:, :, M * 2:M * 4].reshape(B, T, M, 2)
        atanh_rho = raw_mdn[:, :, M * 4:M * 5]
        logit_pi = raw_mdn[:, :, M * 5:M * 6]

        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        rho = torch.tanh(atanh_rho)
        pi = torch.softmax(logit_pi, dim=-1)

        return {
            "gate_logit": gate_logit,
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
            "pi": pi,
            "logit_pi": logit_pi,
        }


class ZIMTBlock(nn.Module):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, scale, shift):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = x * (1 + scale) + shift
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


def zimt_loss(params, target_dxdy, target_stall, mask, gate_pos_weight=5.0):
    """
    Joint NLL for zero-inflated mixture model.

    L = -sum_t [ z_t * log(sig(g_t)) + (1-z_t) * (log(1-sig(g_t)) + log_MDN) ]
    """
    gate_logit = params["gate_logit"]  # (B, T)
    mu = params["mu"]                  # (B, T, M, 2)
    sigma = params["sigma"]            # (B, T, M, 2)
    rho = params["rho"]                # (B, T, M)
    pi = params["pi"]                  # (B, T, M)

    # Gate loss: BCE with class imbalance weighting
    pos_weight = torch.tensor(gate_pos_weight, device=gate_logit.device)
    gate_bce = nn.functional.binary_cross_entropy_with_logits(
        gate_logit, target_stall.float(), pos_weight=pos_weight, reduction="none",
    )
    gate_loss = (gate_bce * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    # MDN NLL for non-stall steps only
    motion_mask = mask & (~target_stall.bool())
    n_motion = motion_mask.float().sum().clamp(min=1.0)

    dx = target_dxdy[:, :, 0:1] - mu[:, :, :, 0]  # (B, T, M)
    dy = target_dxdy[:, :, 1:2] - mu[:, :, :, 1]
    sx = sigma[:, :, :, 0]
    sy = sigma[:, :, :, 1]

    z = (dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * dx * dy / (sx * sy)
    denom = 1 - rho ** 2 + 1e-8

    log_norm = -math.log(2 * math.pi) - torch.log(sx) - torch.log(sy) - 0.5 * torch.log(denom)
    log_exp = -0.5 * z / denom
    log_prob_component = log_norm + log_exp

    log_pi = torch.log(pi + 1e-8)
    log_prob_mixture = torch.logsumexp(log_pi + log_prob_component, dim=-1)  # (B, T)

    nll = -log_prob_mixture
    mdn_loss = (nll * motion_mask.float()).sum() / n_motion

    return gate_loss + mdn_loss, gate_loss.detach(), mdn_loss.detach()


def jerk_loss(target_dxdy, mask):
    """Minimum-jerk regularization: penalize squared jerk (third derivative of position)."""
    speed = torch.sqrt((target_dxdy ** 2).sum(dim=-1) + 1e-8)  # (B, T)
    accel = speed[:, 1:] - speed[:, :-1]  # (B, T-1)
    jerk = accel[:, 1:] - accel[:, :-1]   # (B, T-2)
    jerk_mask = mask[:, 2:].float()
    return (jerk ** 2 * jerk_mask).sum() / jerk_mask.sum().clamp(min=1.0)


def sample_step(params, temperature=1.0, gate_bias=0.0):
    """
    Sample (dx, dy, is_stall) from model output for the last timestep.

    params: dict from forward(), shapes (1, T, ...)
    Returns: (dx, dy, is_stall)
    """
    gate_logit = params["gate_logit"][0, -1]
    pi = params["pi"][0, -1]            # (M,)
    mu = params["mu"][0, -1]            # (M, 2)
    sigma = params["sigma"][0, -1]      # (M, 2)
    rho = params["rho"][0, -1]          # (M,)

    stall_prob = torch.sigmoid(gate_logit + gate_bias)
    is_stall = torch.bernoulli(stall_prob).item() > 0.5

    if is_stall:
        return 0.0, 0.0, True

    if temperature != 1.0:
        logit_pi = params["logit_pi"][0, -1]
        pi = torch.softmax(logit_pi / temperature, dim=0)
        sigma = sigma * temperature

    comp_idx = torch.multinomial(pi, 1).item()

    mu_x = mu[comp_idx, 0].item()
    mu_y = mu[comp_idx, 1].item()
    sx = sigma[comp_idx, 0].item()
    sy = sigma[comp_idx, 1].item()
    r = rho[comp_idx].item()

    z1 = torch.randn(1).item()
    z2 = torch.randn(1).item()
    dx = mu_x + sx * z1
    dy = mu_y + sy * (r * z1 + math.sqrt(max(1 - r * r, 1e-8)) * z2)

    return dx, dy, False
