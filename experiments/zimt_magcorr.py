"""ZIMT with magnitude-weighted endpoint correction.

Replaces ZIMT's linear last-20% endpoint correction with
magnitude-weighted correction spread across ALL steps. Each moving step
absorbs correction proportional to its displacement magnitude, so
faster segments absorb more error. This avoids the artificial velocity
peak at ~90% that the linear correction creates.

Also uses uniform timestamps (no donor time warp, which worsens AUC).
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

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_parser.add_argument("--checkpoint", default=None)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = (
    Path(_args.checkpoint) if _args.checkpoint
    else _DATA_DIR / "zimt_best.pt"
)

_TEMPERATURE = float(os.environ.get("ZIMT_TEMPERATURE", "1.0"))
_GATE_BIAS = float(os.environ.get("ZIMT_GATE_BIAS", "-1.0"))

_DEVICE = get_device()
_HZ = 125.0
_DT = 1.0 / _HZ

_duration = DurationModel(_DATA_DIR)

_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _ckpt["config"]
_model = ZIMTModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.train(False)
_MAX_SEQ = _cfg["max_seq_len"]

print(
    f"[zimt_magcorr] ZIMT ({_cfg['d_model']}d, {_cfg['n_layers']}L, "
    f"{_cfg['n_components']}K MDN) | "
    f"temp={_TEMPERATURE}, gate_bias={_GATE_BIAS}"
)


def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
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

    input_buf = torch.zeros(1, n_target, _cfg["input_dim"], device=_DEVICE)
    generated_dxdy = []
    cum_dx, cum_dy = 0.0, 0.0

    with torch.no_grad():
        for step in range(n_target):
            if step > 0:
                ddx, ddy = generated_dxdy[step - 1]
                input_buf[0, step, 0] = ddx
                input_buf[0, step, 1] = ddy
                input_buf[0, step, 2] = 1.0 if (ddx == 0 and ddy == 0) else 0.0

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

    # Build positions in pixel space
    positions_x = [start_x]
    positions_y = [start_y]
    cx, cy = start_x, start_y
    for ddx, ddy in generated_dxdy:
        cx += ddx * total_dist
        cy += ddy * total_dist
        positions_x.append(cx)
        positions_y.append(cy)

    n = len(positions_x)

    # Magnitude-weighted endpoint correction
    step_mags = []
    for i in range(1, n):
        dx = positions_x[i] - positions_x[i - 1]
        dy = positions_y[i] - positions_y[i - 1]
        step_mags.append(math.hypot(dx, dy))

    err_x = end_x - positions_x[-1]
    err_y = end_y - positions_y[-1]

    if abs(err_x) > 0.01 or abs(err_y) > 0.01:
        moving_mask = [m > 0.3 for m in step_mags]
        total_moving_mag = sum(m for m, moving in zip(step_mags, moving_mask) if moving)

        if total_moving_mag > 0.1:
            cum_corr_x, cum_corr_y = 0.0, 0.0
            for i in range(len(step_mags)):
                if moving_mask[i]:
                    weight = step_mags[i] / total_moving_mag
                    cum_corr_x += err_x * weight
                    cum_corr_y += err_y * weight
                positions_x[i + 1] += cum_corr_x
                positions_y[i + 1] += cum_corr_y

    # Uniform timestamps
    timestamps = [i * _DT for i in range(n)]

    result = [
        (float(positions_x[i]), float(positions_y[i]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    if len(result) < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, _DT)]

    return result
