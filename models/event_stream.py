"""Event-stream generative model for WS7.

Generates the physical mouse event stream (dt, dx, dy) rather than the
smoothed 125Hz signal. The human recording artifacts (exact-zero stalls,
sub-pixel steps, residual grid structure) emerge when the generated events
are pushed through the same standard resample the human data went through.

Architecture extends CANDI's proven hybrid: one non-autoregressive Transformer
with FiLM conditioning on (log_dist, log_dur, cos, sin), three coupled heads:
- dx, dy: categorical over 128 classes (displacements -63..63 plus PAD),
  trained with absorbing-state masking, sampled by confidence-based reveal.
  PAD is a real predictable class, so the model chooses its own event count.
- dt: continuous flow matching on z-scored log(dt_ms).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.candi import CANDIBlock, SinusoidalEmbedding, cosine_beta_schedule

VOCAB_MAX = 63
N_CLASSES = 2 * VOCAB_MAX + 1 + 1  # 127 displacement classes + PAD = 128
PAD_CLASS = 2 * VOCAB_MAX + 1      # 127
MASK_TOKEN = N_CLASSES             # 128, input-only absorbing state


def disp_to_class(d: torch.Tensor) -> torch.Tensor:
    """Displacement in [-63, 63] -> class index in [0, 126]."""
    return (d + VOCAB_MAX).long()


def class_to_disp(c: torch.Tensor) -> torch.Tensor:
    """Class index -> displacement. PAD maps to 0 (callers truncate at PAD)."""
    d = c - VOCAB_MAX
    return torch.where(c >= PAD_CLASS, torch.zeros_like(d), d)


class EventStreamModel(nn.Module):

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

        self.dx_embed = nn.Embedding(N_CLASSES + 1, d_model)  # +1 for MASK
        self.dy_embed = nn.Embedding(N_CLASSES + 1, d_model)
        self.dt_proj = nn.Linear(1, d_model)
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
        self.dt_head = nn.Linear(d_model, 1)
        self.dx_head = nn.Linear(d_model, N_CLASSES)
        self.dy_head = nn.Linear(d_model, N_CLASSES)

        betas = cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bar))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def q_flow(self, x0: torch.Tensor, t_cont: torch.Tensor, noise=None):
        """Flow matching on the (B, T) continuous dt channel."""
        if noise is None:
            noise = torch.randn_like(x0)
        t = t_cont.view(-1, 1)
        x_t = (1 - t) * x0 + t * noise
        velocity = noise - x0
        return x_t, noise, velocity

    def q_mask(self, tokens: torch.Tensor, t_int: torch.Tensor):
        """Absorbing-state masking. tokens: (B, T) class indices."""
        mask_prob = 1.0 - self.sqrt_ab[t_int].view(-1, 1)
        mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
        out = tokens.clone()
        out[mask] = MASK_TOKEN
        return out, mask

    def forward(self, dt_noisy, dx_tok, dy_tok, t, cond):
        B, T = dt_noisy.shape
        x = (
            self.dt_proj(dt_noisy.unsqueeze(-1))
            + self.dx_embed(dx_tok)
            + self.dy_embed(dy_tok)
            + self.pos_embed(torch.arange(T, device=dt_noisy.device))
        )

        t_emb = self.time_embed(t)
        if self.training and self.cond_dropout > 0:
            keep = (torch.rand(B, 1, device=cond.device) > self.cond_dropout).float()
            cond = cond * keep
        combined = t_emb + self.cond_embed(cond)

        for layer in self.layers:
            x = layer(x, combined)

        x = self.norm(x)
        return (
            self.dt_head(x).squeeze(-1),
            self.dx_head(x),
            self.dy_head(x),
        )

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        seq_len: int,
        n_steps: int = 100,
        temperature: float = 1.0,
    ):
        """Joint sampling: flow ODE on dt, confidence reveal on dx/dy tokens.

        Returns (dt_z, dx_cls, dy_cls): (B, T) z-scored log-dt and class indices.
        """
        B = cond.shape[0]
        dev = cond.device

        dt_z = torch.randn(B, seq_len, device=dev)
        dx_tok = torch.full((B, seq_len), MASK_TOKEN, dtype=torch.long, device=dev)
        dy_tok = torch.full((B, seq_len), MASK_TOKEN, dtype=torch.long, device=dev)

        step = 1.0 / n_steps
        dx_logits = dy_logits = None

        for i in range(n_steps):
            t_cont = 1.0 - i * step
            t_scaled = torch.full((B,), t_cont * (self.n_steps - 1), device=dev)
            v_pred, dx_logits, dy_logits = self.forward(dt_z, dx_tok, dy_tok, t_scaled, cond)

            dt_z = dt_z - step * v_pred

            # Reveal fraction tracks the training mask schedule: at time t the
            # model saw ~sqrt_ab[t] of tokens unmasked, so keep sampling in
            # that regime. Tokens are SAMPLED from the predicted distribution
            # (argmax collapses to the per-position mode, which telescopes to
            # near-zero net displacement); selection order is by confidence
            # of the sampled token.
            t_next = max(t_cont - step, 0.0)
            n_target = int(round(
                float(self.sqrt_ab[int(t_next * (self.n_steps - 1))]) * seq_len
            ))
            for tok, logits in ((dx_tok, dx_logits), (dy_tok, dy_logits)):
                masked = tok == MASK_TOKEN
                n_new = n_target - int(seq_len - masked[0].sum().item())
                if n_new <= 0:
                    continue
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, seq_len)
                conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
                order = conf.argsort(dim=-1, descending=True)
                reveal = torch.zeros_like(masked)
                reveal.scatter_(1, order[:, :n_new], True)
                reveal &= masked
                tok[reveal] = sampled[reveal]

        # Sample anything still masked from the final distribution
        for tok, logits in ((dx_tok, dx_logits), (dy_tok, dy_logits)):
            still = tok == MASK_TOKEN
            if still.any():
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, seq_len)
                tok[still] = sampled[still]

        return dt_z, dx_tok, dy_tok
