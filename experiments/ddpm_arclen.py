"""DDPM with DDIM sampling for mouse trajectory generation.

Approach:
  Denoising Diffusion Probabilistic Model trained on arc-length-resampled
  trajectories.  Deterministic DDIM reverse sampling (eta=0) by default.
  Optional stochastic sampling with AR(1) temporally correlated noise.

Expected AUC: ~0.93 (deterministic, eta=0, full pool, n=2000)

Key insight:
  DDPM and CFM produce nearly identical results (both yield smooth
  conditional-mean paths). The curvature gap is a fundamental limitation
  of continuous diffusion sampling.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.temporal_unet import TemporalUNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_parser.add_argument("--checkpoint", default=None)
_parser.add_argument("--steps", type=int, default=100)
_parser.add_argument("--eta", type=float, default=0.0)
_parser.add_argument("--rho", type=float, default=0.95)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = Path(_args.checkpoint) if _args.checkpoint else _DATA_DIR / "ddpm_best.pt"
_N_STEPS = _args.steps
_ETA = _args.eta
_RHO = _args.rho

_DEVICE = get_device()

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
_duration = DurationModel(_DATA_DIR)


# ---------------------------------------------------------------------------
# Load model and noise schedule
# ---------------------------------------------------------------------------
_checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _checkpoint["config"]
_N_PTS = _cfg["n_points"]
_T = _cfg["timesteps"]

_model = TemporalUNet(
    in_channels=_cfg.get("in_channels", 3),
    cond_dim=_cfg.get("cond_dim", 4),
).to(_DEVICE)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.train(False)

# Noise schedule
_schedule = {k: v.float().to(_DEVICE) for k, v in _checkpoint["schedule"].items()}
_alpha_bar = _schedule["alpha_bar"]

# DDIM timestep pairs
_strided = np.linspace(0, _T - 1, _N_STEPS + 1).astype(int)
_ddim_pairs = [
    (int(_strided[i + 1]), int(_strided[i]))
    for i in range(len(_strided) - 1)
]

# Pre-compute normalized timestep tensors (avoids per-step allocation)
_t_norms = {t_curr: torch.tensor([t_curr / _T], dtype=torch.float32, device=_DEVICE)
             for t_curr, _ in _ddim_pairs}

print(
    f"[ddpm_arclen] Loaded {_CKPT_PATH.name}  "
    f"epoch={_checkpoint['epoch']}  val_loss={_checkpoint['val_loss']:.6f}  "
    f"n_points={_N_PTS}  T={_T}  eta={_ETA}  rho={_RHO}  steps={_N_STEPS}"
)


# ---------------------------------------------------------------------------
# AR(1) correlated noise
# ---------------------------------------------------------------------------

def _make_correlated_noise(shape, device):
    """Generate AR(1) temporally correlated noise for smooth perturbations."""
    noise = torch.randn(shape, device=device)
    coeff = math.sqrt(1 - _RHO ** 2)
    for t in range(1, shape[1]):
        noise[:, t] = _RHO * noise[:, t - 1] + coeff * noise[:, t]
    return noise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Generate a trajectory via DDPM reverse diffusion with DDIM sampling."""
    dx_total = end_x - start_x
    dy_total = end_y - start_y
    total_dist = math.hypot(dx_total, dy_total)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    log_dist = math.log(total_dist)
    total_duration = _duration.sample(log_dist)
    log_dur = math.log(max(total_duration, 0.01))

    dx_norm = dx_total / total_dist
    dy_norm = dy_total / total_dist

    condition = torch.tensor(
        [[log_dist, log_dur, dx_norm, dy_norm]],
        dtype=torch.float32,
        device=_DEVICE,
    )

    start_pt = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=_DEVICE)
    end_pt = torch.tensor([float(dx_norm), float(dy_norm), 1.0], dtype=torch.float32, device=_DEVICE)

    # Start from pure noise
    x = torch.randn(1, _N_PTS, 3, device=_DEVICE)

    with torch.no_grad():
        for t_curr, t_prev in reversed(_ddim_pairs):
            noise_pred = _model(x, _t_norms[t_curr], condition)

            alpha_t = _alpha_bar[t_curr]
            alpha_prev = _alpha_bar[t_prev]

            # Estimate x_0
            x_0_hat = (x - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            # DDIM stochastic component
            if t_prev > 0 and _ETA > 0:
                sigma_sq = (1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev)
                sigma = _ETA * sigma_sq.clamp(min=0.0).sqrt()
            else:
                sigma = torch.tensor(0.0, dtype=torch.float32, device=_DEVICE)

            sqrt_alpha_prev = alpha_prev.sqrt()
            dir_coeff = (1.0 - alpha_prev - sigma * sigma).clamp(min=0.0).sqrt()

            x = sqrt_alpha_prev * x_0_hat + dir_coeff * noise_pred

            # Temporally correlated noise injection
            if t_prev > 0 and sigma.item() > 0:
                x = x + sigma * _make_correlated_noise(x.shape, device=_DEVICE)

            # Inpaint endpoints
            x[:, 0, :] = start_pt
            x[:, -1, :] = end_pt

    x_out = x[0].cpu().numpy()

    # Convert to real coordinates
    real_x = x_out[:, 0] * total_dist + start_x
    real_y = x_out[:, 1] * total_dist + start_y

    # Timing from model's t channel
    t_norm_out = np.clip(x_out[:, 2], 0.0, 1.0)
    t_norm_out = np.maximum.accumulate(t_norm_out)
    t_norm_out[0] = 0.0
    t_norm_out[-1] = 1.0
    timestamps = t_norm_out * total_duration

    # Snap endpoints
    if len(timestamps) >= 2:
        gap = max(0.004, (timestamps[-1] - timestamps[-2]) * 0.5)
        real_x[-1] = end_x
        real_y[-1] = end_y
        timestamps[-1] = timestamps[-2] + gap

    return [
        (float(real_x[i]), float(real_y[i]), float(timestamps[i]))
        for i in range(len(timestamps))
    ]
