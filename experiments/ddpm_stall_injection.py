"""DDPM path + empirical stall injection with deceleration ramps.

Approach:
  1. Generate smooth DDPM path via deterministic DDIM (eta=0)
  2. Arc-length parameterize the path
  3. Apply bell-shaped speed profile with Gaussian deceleration dips
  4. Apply heading changes at stall positions
  5. Insert zero-displacement stalls at dip minima
  6. Sample at 125 Hz with monotonic timestamps

Expected AUC: measured (this experiment proved post-hoc curvature injection
is a dead end - acceleration artifacts from the deceleration ramps exactly
offset curvature/angular_velocity gains).

Key insight:
  Curvature = |v x a| / |v|^3.  At zero speed, numerator is 0 so stalls
  contribute zero curvature.  Curvature comes from LOW-SPEED deceleration
  ramps where heading is changing.  Post-hoc injection of these ramps onto
  DDPM paths creates acceleration artifacts that the RF classifier detects.
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
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = Path(_args.checkpoint) if _args.checkpoint else _DATA_DIR / "ddpm_best.pt"

_DEVICE = get_device()

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
_duration = DurationModel(_DATA_DIR)
_rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# Load DDPM model
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

_schedule = {k: v.float().to(_DEVICE) for k, v in _checkpoint["schedule"].items()}
_alpha_bar = _schedule["alpha_bar"]

_N_STEPS = 100

_strided = np.linspace(0, _T - 1, _N_STEPS + 1).astype(int)
_ddim_pairs = [(int(_strided[i + 1]), int(_strided[i]))
               for i in range(len(_strided) - 1)]

# Pre-compute normalized timestep tensors (avoids per-step allocation)
_t_norms = {t_curr: torch.tensor([t_curr / _T], dtype=torch.float32, device=_DEVICE)
             for t_curr, _ in _ddim_pairs}

_HZ = 125.0
_DT = 1.0 / _HZ

print(f"[ddpm_stall_injection] DDPM + decel ramps + stalls + heading changes")


def _generate_ddpm_path(log_dist, log_dur, dx_norm, dy_norm, total_dist):
    """Generate DDPM path and return (x_real, y_real) arrays."""
    condition = torch.tensor(
        [[log_dist, log_dur, dx_norm, dy_norm]],
        dtype=torch.float32, device=_DEVICE,
    )
    start_pt = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=_DEVICE)
    end_pt = torch.tensor([float(dx_norm), float(dy_norm), 1.0], dtype=torch.float32, device=_DEVICE)

    x = torch.randn(1, _N_PTS, 3, device=_DEVICE)

    with torch.no_grad():
        for t_curr, t_prev in reversed(_ddim_pairs):
            noise_pred = _model(x, _t_norms[t_curr], condition)
            alpha_t = _alpha_bar[t_curr]
            alpha_prev = _alpha_bar[t_prev]
            x_0_hat = (x - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            sqrt_alpha_prev = alpha_prev.sqrt()
            sqrt_1_minus = (1.0 - alpha_prev).clamp(min=0.0).sqrt()
            x = sqrt_alpha_prev * x_0_hat + sqrt_1_minus * noise_pred
            x[:, 0, :] = start_pt
            x[:, -1, :] = end_pt

    x_out = x[0].cpu().numpy()
    real_x = x_out[:, 0] * total_dist
    real_y = x_out[:, 1] * total_dist
    return real_x, real_y


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
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

    # Step 1: Generate DDPM path (spatial only)
    ddpm_x, ddpm_y = _generate_ddpm_path(
        log_dist, log_dur, dx_norm, dy_norm, total_dist
    )
    ddpm_x += start_x
    ddpm_y += start_y
    ddpm_x[0], ddpm_y[0] = start_x, start_y
    ddpm_x[-1], ddpm_y[-1] = end_x, end_y

    # Arc-length parameterization
    ds = np.sqrt(np.diff(ddpm_x)**2 + np.diff(ddpm_y)**2)
    s_cum = np.concatenate([[0], np.cumsum(ds)])
    s_total = s_cum[-1]
    if s_total < 1e-6:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    # Step 2: Build speed profile with bell shape + stall dips
    n_output = max(5, int(round(total_duration * _HZ)))

    # Bell-shaped base speed profile (peak at ~35%)
    t_frac = np.linspace(0, 1, n_output)
    peak_frac = 0.35
    sigma_l = peak_frac * 0.55
    sigma_r = (1 - peak_frac) * 0.55
    sigma = np.where(t_frac < peak_frac, sigma_l, sigma_r)
    speed_profile = np.exp(-0.5 * ((t_frac - peak_frac) / sigma) ** 2)

    # Step 3: Add stall dips
    n_stalls = max(1, int(_rng.normal(4, 1.5)))
    n_stalls = min(n_stalls, max(1, n_output // 10))

    stall_fracs = np.sort(_rng.uniform(0.08, 0.92, size=n_stalls))
    for i in range(1, len(stall_fracs)):
        if stall_fracs[i] - stall_fracs[i-1] < 0.08:
            stall_fracs[i] = stall_fracs[i-1] + 0.08
    stall_fracs = stall_fracs[stall_fracs < 0.92]

    dip_width = 0.04
    for sf in stall_fracs:
        dip = np.exp(-0.5 * ((t_frac - sf) / dip_width) ** 2)
        speed_profile *= (1.0 - 0.98 * dip)

    speed_profile = np.maximum(speed_profile, 1e-4)
    ds_profile = speed_profile * _DT
    s_profile = np.concatenate([[0], np.cumsum(ds_profile)])
    s_profile = s_profile / s_profile[-1] * s_total

    # Interpolate DDPM path at these arc-length positions
    x_out = np.interp(s_profile[:n_output], s_cum, ddpm_x)
    y_out = np.interp(s_profile[:n_output], s_cum, ddpm_y)

    # Step 4: Apply heading changes at stall positions
    for sf in stall_fracs:
        idx = int(round(sf * (n_output - 1)))
        idx = max(1, min(idx, n_output - 2))
        heading_change = float(_rng.normal(0, 0.20))
        pivot_x, pivot_y = x_out[idx], y_out[idx]
        cos_hc, sin_hc = math.cos(heading_change), math.sin(heading_change)

        for j in range(idx + 1, n_output):
            dx = x_out[j] - pivot_x
            dy = y_out[j] - pivot_y
            x_out[j] = pivot_x + dx * cos_hc - dy * sin_hc
            y_out[j] = pivot_y + dx * sin_hc + dy * cos_hc

        x_out[-1] = end_x
        y_out[-1] = end_y

    # Step 5: Insert zero-displacement stalls at dip minima
    points = []
    accumulated_stall_time = 0.0

    stall_lens = {}
    for sf in stall_fracs:
        idx = int(round(sf * (n_output - 1)))
        idx = max(1, min(idx, n_output - 2))
        stall_lens[idx] = int(_rng.integers(1, 5))

    t_base = np.arange(n_output) * _DT

    for i in range(n_output):
        t_shifted = t_base[i] + accumulated_stall_time
        points.append((float(x_out[i]), float(y_out[i]), t_shifted))

        if i in stall_lens:
            slen = stall_lens[i]
            sx, sy = float(x_out[i]), float(y_out[i])
            st = t_shifted
            for _ in range(slen):
                st += _DT
                points.append((sx, sy, st))
            accumulated_stall_time += slen * _DT

    # Snap endpoint
    points[-1] = (end_x, end_y, points[-1][2])

    # Ensure monotonic timestamps
    result = [points[0]]
    for p in points[1:]:
        if p[2] <= result[-1][2]:
            result.append((p[0], p[1], result[-1][2] + _DT))
        else:
            result.append(p)

    result[0] = (start_x, start_y, 0.0)
    return result
