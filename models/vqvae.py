"""
VQ-VAE for mouse trajectory displacement tokens.

Quantizes (dx, dy) displacement pairs into 1024 motion tokens.
Token 0 is hardcoded to (0, 0) - the stall token (NOT learned).

Architecture:
  Encoder: MLP (2 → 64 → 128 → 64)
  Codebook: 1024 learned entries (64-dim)
  Decoder: MLP (64 → 128 → 64 → 2)

Loss: MSE reconstruction + commitment (β=0.25) + EMA codebook updates
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization with EMA updates (van den Oord et al., 2017)."""

    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.ema_decay = ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_codes, 1.0 / n_codes)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor):
        """
        z: (B, D) latent vectors
        Returns: (z_q, loss, indices)
        """
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e^T
        d = (z.pow(2).sum(1, keepdim=True)
             + self.embedding.weight.pow(2).sum(1)
             - 2 * z @ self.embedding.weight.t())

        # Nearest codebook entry
        indices = d.argmin(dim=1)  # (B,)
        z_q = self.embedding(indices)  # (B, D)

        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)

        # Straight-through estimator: gradients flow through z_q as if it were z
        z_q_st = z + (z_q - z).detach()

        loss = codebook_loss + self.beta * commitment_loss

        # EMA codebook updates (during training)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.n_codes).float()
                cluster_size = one_hot.sum(0)
                embed_sum = one_hot.t() @ z

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay
                )
                self.ema_embed_sum.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed = (self.ema_cluster_size + 1e-5) / (n + self.n_codes * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

        return z_q_st, loss, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert token indices to codebook vectors."""
        return self.embedding(indices)


class MotionVQVAE(nn.Module):
    """VQ-VAE for (dx, dy) mouse displacement pairs.

    Token 0 is the stall token (hardcoded to zero displacement).
    Tokens 1-1024 are learned motion tokens.
    """

    def __init__(self, n_codes: int = 1024, code_dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim

        # Encoder: (dx, dy) → latent
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, code_dim),
        )

        # Vector quantizer (for non-stall tokens)
        self.vq = VectorQuantizer(n_codes, code_dim, beta=beta)

        # Decoder: latent → (dx, dy)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, dxdy: torch.Tensor):
        """
        dxdy: (B, 2) displacement pairs (NON-STALL only during training)
        Returns: (reconstructed, vq_loss, indices)
        """
        z = self.encoder(dxdy)
        z_q, vq_loss, indices = self.vq(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, vq_loss, indices

    def encode(self, dxdy: torch.Tensor) -> torch.Tensor:
        """Encode displacements to token indices. Returns (B,) int tensor."""
        z = self.encoder(dxdy)
        _, _, indices = self.vq(z)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to (dx, dy). Returns (B, 2) tensor."""
        z_q = self.vq.decode_indices(indices)
        return self.decoder(z_q)

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary: 1 stall token + n_codes motion tokens."""
        return 1 + self.n_codes
