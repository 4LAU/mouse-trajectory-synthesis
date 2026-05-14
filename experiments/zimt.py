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
_parser.add_argument("--temperature", type=float, default=1.0)
_parser.add_argument("--gate-bias", type=float, default=-1.0)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_CKPT_PATH = (
    Path(_args.checkpoint) if _args.checkpoint
    else _DATA_DIR / "zimt_best.pt"
)
_TEMPERATURE = float(os.environ.get("ZIMT_TEMPERATURE", _args.temperature))
_GATE_BIAS = float(os.environ.get("ZIMT_GATE_BIAS", _args.gate_bias))

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
_USE_VELOCITY_GUIDE = _cfg.get("input_dim", 6) == 7
_POLAR = _ckpt.get("polar", False)

_TIME_WARP = os.environ.get("ZIMT_TIME_WARP", "donor")  # "off", "template", "donor"
_WARP_TEMPLATE = np.load(_DATA_DIR / "velocity_template.npy")

_VELOCITY_TEMPLATE = None
if _USE_VELOCITY_GUIDE:
    _VELOCITY_TEMPLATE = torch.from_numpy(_WARP_TEMPLATE).float().to(_DEVICE)

# Load pool for donor velocity profiles
_DONOR_POOL = None
if _TIME_WARP == "donor":
    _pool_offsets_path = Path("training/full_pool_offsets.npy")
    if _pool_offsets_path.exists():
        _DONOR_OFFSETS = np.load("training/full_pool_offsets.npy")
        _DONOR_META = np.load("training/full_pool_meta.npy")
        _DONOR_FLAT = np.load("training/pool_flat_i16.npy", mmap_mode="r")
        _DONOR_T = np.load("training/pool_t_rel_f32.npy", mmap_mode="r")
        _DONOR_LOG_DIST = _DONOR_META[:, 0]
        _DONOR_POOL = True
        print(f"[zimt] Donor pool: {len(_DONOR_OFFSETS)-1:,} trajectories")
    else:
        print("[zimt] No donor pool found, falling back to uniform timestamps")
        _TIME_WARP = "off"

_mode = "polar" if _POLAR else f"{_cfg.get('input_dim', 6)}d"
print(
    f"[zimt] ZIMT ({_cfg['d_model']}d, {_cfg['n_layers']}L, "
    f"{_cfg['n_components']}K MDN, {_mode}) | "
    f"ep {_ckpt.get('epoch', '?')}, "
    f"val_loss={_ckpt.get('val_loss', 0):.4f}, "
    f"phase={_ckpt.get('phase', '?')}"
)


_donor_rng = np.random.default_rng()


def _get_donor_velocity_profile(log_dist, n_output_bins):
    """Sample a real trajectory with similar distance from the pool, return its velocity profile."""
    dist_diff = np.abs(_DONOR_LOG_DIST - log_dist)
    candidates = np.where(dist_diff < 0.3)[0]
    if len(candidates) < 10:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) < 5:
        candidates = np.arange(len(_DONOR_LOG_DIST))

    chosen = candidates[_donor_rng.integers(0, len(candidates))]
    lo, hi = int(_DONOR_OFFSETS[chosen]), int(_DONOR_OFFSETS[chosen + 1])
    xy = np.array(_DONOR_FLAT[lo:hi], dtype=np.float64)

    if len(xy) < 3:
        return _WARP_TEMPLATE

    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))

    # Resample to n_output_bins
    profile = np.interp(
        np.linspace(0, 1, n_output_bins),
        np.linspace(0, 1, len(speeds)),
        speeds,
    )
    total = profile.sum()
    if total < 1e-8:
        return _WARP_TEMPLATE
    return profile / total


def _time_warp_timestamps(positions, total_duration, velocity_profile=None):
    """Redistribute timestamps so velocity profile matches the given profile."""
    n = len(positions)
    step_dists = []
    for i in range(1, n):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        step_dists.append(math.hypot(dx, dy))

    profile = velocity_profile if velocity_profile is not None else _WARP_TEMPLATE
    n_bins = len(profile)
    target_dt = []
    for i in range(len(step_dists)):
        progress = i / max(len(step_dists) - 1, 1)
        bin_idx = min(int(progress * (n_bins - 1)), n_bins - 1)
        target_speed = max(profile[bin_idx], 1e-6)
        target_dt.append(step_dists[i] / target_speed)

    dt_sum = sum(target_dt)
    if dt_sum < 1e-8:
        return [i * (total_duration / max(n - 1, 1)) for i in range(n)]

    timestamps = [0.0]
    cum = 0.0
    for dt in target_dt:
        cum += dt
        timestamps.append(total_duration * cum / dt_sum)

    return timestamps


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
    input_buf = torch.zeros(1, n_target, _cfg["input_dim"], device=_DEVICE)
    generated_dxdy = []
    cum_dx, cum_dy = 0.0, 0.0
    running_angle = math.atan2(dy_total, dx_total)

    with torch.no_grad():
        for step in range(n_target):
            if step > 0:
                ddx, ddy = generated_dxdy[step - 1]
                if _POLAR:
                    speed = math.hypot(ddx, ddy)
                    if step >= 2:
                        prev_ddx, prev_ddy = generated_dxdy[step - 2]
                        prev_angle = math.atan2(prev_ddy, prev_ddx) if math.hypot(prev_ddx, prev_ddy) > 1e-12 else running_angle
                        cur_angle = math.atan2(ddy, ddx) if speed > 1e-12 else prev_angle
                        dangle = math.atan2(math.sin(cur_angle - prev_angle), math.cos(cur_angle - prev_angle))
                    else:
                        dangle = 0.0
                    input_buf[0, step, 0] = speed
                    input_buf[0, step, 1] = dangle
                else:
                    input_buf[0, step, 0] = ddx
                    input_buf[0, step, 1] = ddy
                input_buf[0, step, 2] = 1.0 if (ddx == 0 and ddy == 0) else 0.0

            input_buf[0, step, 3] = cos_a - cum_dx
            input_buf[0, step, 4] = sin_a - cum_dy
            input_buf[0, step, 5] = 1.0 - step / n_target

            if _USE_VELOCITY_GUIDE:
                progress = step / n_target
                bin_idx = min(int(progress * (len(_VELOCITY_TEMPLATE) - 1)), len(_VELOCITY_TEMPLATE) - 1)
                input_buf[0, step, 6] = _VELOCITY_TEMPLATE[bin_idx]

            params = _model(input_buf[:, :step + 1], condition)

            if _POLAR:
                speed_raw, dangle_raw, is_stall = sample_step(
                    params, temperature=_TEMPERATURE, gate_bias=_GATE_BIAS,
                )
                if is_stall:
                    dx, dy = 0.0, 0.0
                else:
                    speed = abs(speed_raw)
                    running_angle += dangle_raw
                    dx = speed * math.cos(running_angle)
                    dy = speed * math.sin(running_angle)
            else:
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

    if _TIME_WARP == "donor" and _DONOR_POOL and n > 3:
        donor_profile = _get_donor_velocity_profile(log_dist, n - 1)
        timestamps = _time_warp_timestamps(positions, total_duration, velocity_profile=donor_profile)
    elif _TIME_WARP == "template" and n > 3:
        timestamps = _time_warp_timestamps(positions, total_duration)
    else:
        timestamps = [i * _DT for i in range(n)]

    result = [
        (float(positions[i][0]), float(positions[i][1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    if len(result) < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, _DT)]

    return result
