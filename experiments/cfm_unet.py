"""Conditional Flow Matching with Euler ODE integration.

Approach:
  CFM trains a velocity field v(x, t) on 2-channel (x, y) position data.
  At inference, Euler integration with 20 ODE steps produces spatial paths.
  Timing is assigned via uniform resampling at the estimated duration.
  Endpoints are corrected via smoothstep easing over the last 25%.

Expected AUC: ~0.99 (full pool)

Note:
  This 2-channel checkpoint generates only positions, not timing.
  The original 3-channel (position + timing) training was never completed.
  As a result, CFM performs worse than DDPM on the adversarial evaluation.
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
_parser.add_argument("--ode-steps", type=int, default=20)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = Path(_args.checkpoint) if _args.checkpoint else _DATA_DIR / "cfm_best.pt"
_N_ODE_STEPS = _args.ode_steps

_DEVICE = get_device()

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
_duration = DurationModel(_DATA_DIR)


# ---------------------------------------------------------------------------
# Load model (2-channel: x, y positions only)
# ---------------------------------------------------------------------------
_checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _checkpoint["config"]
_N_PTS = _cfg["n_points"]

_model = TemporalUNet(
    in_channels=_cfg.get("in_channels", 2),
    cond_dim=_cfg.get("cond_dim", 4),
).to(_DEVICE)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.train(False)

_IN_CH = _cfg.get("in_channels", 2)

print(
    f"[cfm_unet] Loaded {_CKPT_PATH.name}  "
    f"in_channels={_IN_CH}  n_points={_N_PTS}  ode_steps={_N_ODE_STEPS}"
)

# Pre-compute ODE time step tensors (avoids per-step allocation)
_dt_ode = 1.0 / _N_ODE_STEPS
_t_tensors = [torch.tensor([step * _dt_ode], dtype=torch.float32, device=_DEVICE)
              for step in range(_N_ODE_STEPS)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Generate a trajectory via Euler ODE integration of the learned velocity field."""
    dx_total = end_x - start_x
    dy_total = end_y - start_y
    total_dist = math.hypot(dx_total, dy_total)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    log_dist = math.log(total_dist)
    total_duration = _duration.sample(log_dist)
    log_dur = math.log(max(total_duration, 0.01))

    cos_a = dx_total / total_dist
    sin_a = dy_total / total_dist

    condition = torch.tensor(
        [[log_dist, log_dur, cos_a, sin_a]],
        dtype=torch.float32,
        device=_DEVICE,
    )

    # CFM operates in raw pixel coordinates (origin-translated, not distance-normalized)
    x = torch.randn(1, _N_PTS, _IN_CH, device=_DEVICE)

    with torch.no_grad():
        for step in range(_N_ODE_STEPS):
            v = _model(x, _t_tensors[step], condition)
            x = x + v * _dt_ode

    x_out = x[0].cpu().numpy()

    # Positions are origin-translated raw pixel coordinates
    real_x = x_out[:, 0] + start_x
    real_y = x_out[:, 1] + start_y

    # Trim to physical trajectory length
    n_pts = max(5, min(_N_PTS, round(total_duration * 125)))
    real_x = real_x[:n_pts]
    real_y = real_y[:n_pts]

    # Endpoint correction: smoothstep ease over last 25%
    correction_start = int(n_pts * 0.75)
    if correction_start < n_pts - 1:
        for i in range(correction_start, n_pts):
            alpha = (i - correction_start) / (n_pts - 1 - correction_start)
            ease = alpha * alpha * (3 - 2 * alpha)
            real_x[i] = real_x[i] * (1 - ease) + end_x * ease
            real_y[i] = real_y[i] * (1 - ease) + end_y * ease

    timestamps = np.linspace(0.0, total_duration, n_pts)

    return [
        (float(real_x[i]), float(real_y[i]), float(timestamps[i]))
        for i in range(n_pts)
    ]
