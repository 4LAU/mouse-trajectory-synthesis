"""ZIMT: Zero-Inflated Mouse Trajectory Generator.

Generates trajectories autoregressively with a binary stall gate and
MDN displacement head on a Transformer backbone.

AUC: ~0.76 (Phase 2, n=200, vs DDPM ~0.93)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.zimt import ZIMTModel, sample_step

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_parser.add_argument("--checkpoint", default=None)
_parser.add_argument("--temperature", type=float, default=0.85)
_parser.add_argument("--gate-bias", type=float, default=-1.0)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = (
    Path(_args.checkpoint) if _args.checkpoint
    else _DATA_DIR / "zimt_best.pt"
)
_TEMPERATURE = _args.temperature
_GATE_BIAS = _args.gate_bias

_DEVICE = get_device()
_HZ = 125.0
_DT = 1.0 / _HZ

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
_duration = DurationModel(_DATA_DIR)

# ---------------------------------------------------------------------------
# Load ZIMT
# ---------------------------------------------------------------------------
_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _ckpt["config"]

_model = ZIMTModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.train(False)

_MAX_SEQ = _cfg["max_seq_len"]

print(
    f"[zimt] ZIMT ({_cfg['d_model']}d, {_cfg['n_layers']}L, "
    f"{_cfg['n_components']}K MDN) | "
    f"ep {_ckpt.get('epoch', '?')}, "
    f"val_loss={_ckpt.get('val_loss', 0):.4f}, "
    f"phase={_ckpt.get('phase', '?')}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Generate a trajectory via autoregressive sampling with stall gate."""
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

    n_target = max(5, int(round(total_duration * _HZ)))
    n_target = min(n_target, _MAX_SEQ - 2)

    # Autoregressive generation in normalized space (training data has unit distance)
    # Input features: (dx_prev, dy_prev, stall_prev, remaining_dx, remaining_dy, remaining_frac)
    input_buf = torch.zeros(1, n_target, 6, device=_DEVICE)
    generated_dxdy = []
    cum_dx, cum_dy = 0.0, 0.0

    with torch.no_grad():
        for step in range(n_target):
            if step > 0:
                ddx, ddy = generated_dxdy[step - 1]
                input_buf[0, step, 0] = ddx
                input_buf[0, step, 1] = ddy
                input_buf[0, step, 2] = 1.0 if (ddx == 0 and ddy == 0) else 0.0

            # Remaining displacement in normalized space (matches training)
            input_buf[0, step, 3] = cos_a - cum_dx
            input_buf[0, step, 4] = sin_a - cum_dy
            input_buf[0, step, 5] = 1.0 - step / n_target

            params = _model(input_buf[:, :step + 1], condition)
            dx, dy, is_stall = sample_step(
                params, temperature=_TEMPERATURE, gate_bias=_GATE_BIAS,
            )

            generated_dxdy.append((dx, dy))
            cum_dx += dx
            cum_dy += dy

    # Build positions — scale from normalized space to pixels
    positions = [(start_x, start_y)]
    cx, cy = start_x, start_y
    for ddx, ddy in generated_dxdy:
        cx += ddx * total_dist
        cy += ddy * total_dist
        positions.append((cx, cy))

    # Endpoint correction: scale final 20%
    n = len(positions)
    actual_end_x, actual_end_y = positions[-1]
    err_x = end_x - actual_end_x
    err_y = end_y - actual_end_y

    n_correct = max(3, n // 5)
    for i in range(n_correct):
        idx = n - n_correct + i
        frac = (i + 1) / n_correct
        px, py = positions[idx]
        positions[idx] = (px + err_x * frac, py + err_y * frac)

    result = [
        (float(positions[i][0]), float(positions[i][1]), i * _DT)
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    if len(result) < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, _DT)]

    return result
