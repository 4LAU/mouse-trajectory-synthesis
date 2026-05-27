"""CANDI hybrid discrete-continuous diffusion experiment.

Supports both Cartesian (dx,dy) and polar (speed, delta_heading) checkpoints.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.candi import CANDIModel

_DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))
_DEVICE = get_device()
_HZ = 125.0

_ckpt_name = os.environ.get("CANDI_CKPT", "candi_best.pt")
_ckpt_path = _TRAIN_DIR / _ckpt_name
_ckpt = torch.load(_ckpt_path, map_location=_DEVICE, weights_only=False)
_cfg = _ckpt["config"]
_data_scale = _ckpt["data_scale"]
_POLAR = _ckpt.get("polar", False)

_model = CANDIModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_duration = DurationModel(_TRAIN_DIR)

_N_SAMPLE_STEPS = int(os.environ.get("CANDI_STEPS", "50"))
_ETA = float(os.environ.get("CANDI_ETA", "0.0"))
_CFG = float(os.environ.get("CANDI_CFG", "2.0"))
_N_CANDIDATES = int(os.environ.get("CANDI_CANDIDATES", "1"))

_n_params = sum(p.numel() for p in _model.parameters())
_mode = "polar" if _POLAR else "cartesian"
print(f"[candi] {_n_params:,} params, mode={_mode}, "
      f"steps={_N_SAMPLE_STEPS}, eta={_ETA}, cfg={_CFG}, "
      f"candidates={_N_CANDIDATES}")


def _decode_cartesian(raw_np, stall_np):
    dxdy_np = raw_np / _data_scale
    dxdy_np[stall_np > 0.5] = 0.0
    return np.cumsum(dxdy_np[:, 0]), np.cumsum(dxdy_np[:, 1])


def _decode_polar(raw_np, stall_np):
    spd_scale, dh_scale = float(_data_scale[0]), float(_data_scale[1])
    speed = np.maximum(raw_np[:, 0] / spd_scale, 0.0)
    dheading = raw_np[:, 1] / dh_scale
    speed[stall_np > 0.5] = 0.0
    dheading[stall_np > 0.5] = 0.0
    heading = np.cumsum(dheading)
    return np.cumsum(speed * np.cos(heading)), np.cumsum(speed * np.sin(heading))


def _build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist, dx, dy,
                      start_x, start_y, end_x, end_y):
    target_dx = dx / total_dist if total_dist > 0 else 0.0
    target_dy = dy / total_dist if total_dist > 0 else 0.0
    err_x = target_dx - cum_x[-1]
    err_y = target_dy - cum_y[-1]

    if err_x * err_x + err_y * err_y > 1e-8:
        moving = stall_np < 0.5
        if moving.sum() > 0:
            magnitudes = np.sqrt(np.diff(np.concatenate([[0], cum_x])) ** 2 +
                                 np.diff(np.concatenate([[0], cum_y])) ** 2)
            mag_moving = magnitudes * moving
            total_mag = mag_moving.sum()
            weights = (mag_moving / total_mag if total_mag > 1e-8
                       else moving.astype(np.float64) / moving.sum())
            cum_x = cum_x + err_x * np.cumsum(weights)
            cum_y = cum_y + err_y * np.cumsum(weights)
        else:
            frac = np.linspace(0, 1, seq_len)
            cum_x = cum_x + err_x * frac
            cum_y = cum_y + err_y * frac

    out_x = cum_x * total_dist + start_x
    out_y = cum_y * total_dist + start_y

    dt = 1.0 / _HZ
    result: Trajectory = [(start_x, start_y, 0.0)]
    for i in range(seq_len):
        result.append((float(out_x[i]), float(out_y[i]), (i + 1) * dt))
    result[-1] = (end_x, end_y, result[-1][2])
    return result


def _score_trajectory(traj: Trajectory) -> float:
    if len(traj) < 5:
        return 0.0
    pts = np.array(traj)
    dt = np.maximum(np.diff(pts[:, 2]), 1e-8)
    vx = np.diff(pts[:, 0]) / dt
    vy = np.diff(pts[:, 1]) / dt
    speed = np.sqrt(vx ** 2 + vy ** 2)
    if len(speed) < 3:
        return 0.0
    jerk_x = np.diff(np.diff(vx) / dt[1:]) / dt[2:]
    jerk_y = np.diff(np.diff(vy) / dt[1:]) / dt[2:]
    jerk_mag = np.sqrt(jerk_x ** 2 + jerk_y ** 2)
    heading = np.arctan2(vy, vx)
    moving = speed > 1e-3
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    mv2 = moving[1:] & moving[:-1]
    av_std = np.std(dh[mv2]) if mv2.sum() > 2 else 1.0
    return -(np.mean(jerk_mag) * 0.001 + av_std)


def _generate_single(cond, seq_len, total_dist, dx, dy,
                     start_x, start_y, end_x, end_y):
    decode = _decode_polar if _POLAR else _decode_cartesian
    with torch.no_grad():
        raw, stall = _model.sample(
            cond, seq_len,
            n_steps=_N_SAMPLE_STEPS, eta=_ETA, cfg_scale=_CFG,
        )
    raw_np = raw[0].cpu().numpy()
    stall_np = stall[0].cpu().numpy()
    cum_x, cum_y = decode(raw_np, stall_np)
    return _build_trajectory(cum_x, cum_y, stall_np, seq_len, total_dist, dx, dy,
                             start_x, start_y, end_x, end_y)


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    dx = end_x - start_x
    dy = end_y - start_y
    total_dist = math.hypot(dx, dy)

    if total_dist < 1.0:
        return [(start_x, start_y, 0.0), (end_x, end_y, 0.008)]

    log_dist = math.log(total_dist)
    angle = math.atan2(dy, dx)
    duration = _duration.sample(log_dist)
    log_dur = math.log(duration)
    seq_len = max(5, min(int(round(duration * _HZ)), _cfg["max_seq_len"]))

    cond = torch.tensor(
        [[log_dist, log_dur, math.cos(angle), math.sin(angle)]],
        dtype=torch.float32, device=_DEVICE,
    )

    if _N_CANDIDATES <= 1:
        return _generate_single(cond, seq_len, total_dist, dx, dy,
                                start_x, start_y, end_x, end_y)

    best_traj = None
    best_score = float("-inf")
    for _ in range(_N_CANDIDATES):
        traj = _generate_single(cond, seq_len, total_dist, dx, dy,
                                start_x, start_y, end_x, end_y)
        score = _score_trajectory(traj)
        if score > best_score:
            best_score = score
            best_traj = traj
    return best_traj
