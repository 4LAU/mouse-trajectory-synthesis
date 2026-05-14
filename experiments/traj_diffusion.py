"""Trajectory Diffusion Model — non-autoregressive generation.

Generates all (dx, dy) timesteps simultaneously via DDIM sampling.
Post-processes with stall thresholding and magnitude-weighted endpoint correction.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.traj_diffusion import TrajectoryDiffusionModel

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_parser.add_argument("--checkpoint", default=None)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = (
    Path(_args.checkpoint) if _args.checkpoint
    else Path("training/diffusion_best.pt")
)

if not _CKPT_PATH.exists():
    raise FileNotFoundError(f"Diffusion checkpoint not found: {_CKPT_PATH}")

_DEVICE = get_device()
_HZ = 125.0
_DT = 1.0 / _HZ
_STALL_THRESHOLD = 0.05
_DDIM_STEPS = 50

_duration = DurationModel(_DATA_DIR)

_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _ckpt["config"]
_data_scale = _ckpt["data_scale"]
_data_std = _ckpt["data_std"]

_model = TrajectoryDiffusionModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_MAX_SEQ = _cfg["max_seq_len"]

print(
    f"[traj_diffusion] Diffusion ({_cfg['d_model']}d, {_cfg['n_layers']}L) | "
    f"T={_MAX_SEQ}, ddim_steps={_DDIM_STEPS}, stall_thresh={_STALL_THRESHOLD}"
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
    n_target = min(n_target, _MAX_SEQ)

    with torch.no_grad():
        x_0 = _model.ddim_sample(condition, n_target, n_steps=_DDIM_STEPS)

    dxdy = x_0[0].cpu()  # (n_target, 2)
    dxdy = dxdy / _data_scale  # undo normalization

    # Stall thresholding: near-zero displacements become exact stalls
    magnitudes = torch.sqrt((dxdy ** 2).sum(dim=-1))
    stall_mask = magnitudes < _STALL_THRESHOLD
    dxdy[stall_mask] = 0.0

    # Build pixel-space trajectory
    positions_x = [start_x]
    positions_y = [start_y]
    cx, cy = start_x, start_y
    for i in range(n_target):
        cx += dxdy[i, 0].item() * total_dist
        cy += dxdy[i, 1].item() * total_dist
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

    if err_x * err_x + err_y * err_y > 0.01:
        moving = [m > 0.3 for m in step_mags]
        total_moving = sum(m for m, mv in zip(step_mags, moving) if mv)
        if total_moving > 0.1:
            cum_cx, cum_cy = 0.0, 0.0
            for i in range(len(step_mags)):
                if moving[i]:
                    w = step_mags[i] / total_moving
                    cum_cx += err_x * w
                    cum_cy += err_y * w
                positions_x[i + 1] += cum_cx
                positions_y[i + 1] += cum_cy

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
