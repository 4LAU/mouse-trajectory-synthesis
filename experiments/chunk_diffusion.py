"""Chunk-level diffusion: generate trajectories by sequencing 25-step chunks.

Each chunk is generated via DDPM (DDIM sampling), conditioned on the previous
chunk's tail and global/local trajectory metadata. Stalls are modeled jointly
as the 3rd channel (stall logit).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.chunk_diffusion import ChunkDiffusionModel

CHUNK_SIZE = 25
CONTEXT_SIZE = 5
STRIDE = 20

_DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
_DEVICE = get_device()
_HZ = 125.0
_DT = 1.0 / _HZ

# Duration model
_duration = DurationModel(_DATA_DIR)

# Load checkpoint
_CKPT_PATH = Path(os.environ.get(
    "CHUNK_CKPT", "training/chunk_diffusion_best.pt"
))
_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_cfg = _ckpt["config"]
_data_scale = _ckpt["data_scale"]

_model = ChunkDiffusionModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

# DDIM parameters
_N_DDIM_STEPS = int(os.environ.get("CHUNK_DDIM_STEPS", "50"))
_ETA = float(os.environ.get("CHUNK_ETA", "0.3"))
_CFG_SCALE = float(os.environ.get("CHUNK_CFG", "0.0"))

# Donor pool for velocity profiles
_pool_offsets = np.load("training/full_pool_offsets.npy")
_pool_meta = np.load("training/full_pool_meta.npy")
_pool_flat = np.load("training/pool_flat_i16.npy", mmap_mode="r")
_pool_t_rel = np.load("training/pool_t_rel_f32.npy", mmap_mode="r")
_pool_log_dist = _pool_meta[:, 0]
_donor_rng = np.random.default_rng()

print(f"[chunk_diffusion] {_cfg['n_diff_steps']} steps, "
      f"scale={_data_scale:.2f}, eta={_ETA}, "
      f"ep={_ckpt.get('epoch', '?')}, val={_ckpt.get('val_loss', 0):.6f}")


def _get_donor_velocity_profile(log_dist, n_bins):
    dist_diff = np.abs(_pool_log_dist - log_dist)
    candidates = np.where(dist_diff < 0.3)[0]
    if len(candidates) < 10:
        candidates = np.where(dist_diff < 1.0)[0]
    if len(candidates) < 5:
        candidates = np.arange(len(_pool_log_dist))

    chosen = candidates[_donor_rng.integers(0, len(candidates))]
    lo, hi = int(_pool_offsets[chosen]), int(_pool_offsets[chosen + 1])
    xy = np.array(_pool_flat[lo:hi], dtype=np.float64)
    if len(xy) < 3:
        return np.ones(n_bins) / n_bins

    diffs = np.diff(xy, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))
    profile = np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(speeds)), speeds)
    total = profile.sum()
    if total < 1e-8:
        return np.ones(n_bins) / n_bins
    return profile / total


def _time_warp_timestamps(positions, total_duration, velocity_profile):
    n = len(positions)
    step_dists = []
    for i in range(1, n):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        step_dists.append(math.hypot(dx, dy))

    n_bins = len(velocity_profile)
    target_dt = []
    for i in range(len(step_dists)):
        progress = i / max(len(step_dists) - 1, 1)
        bin_idx = min(int(progress * (n_bins - 1)), n_bins - 1)
        target_speed = max(velocity_profile[bin_idx], 1e-6)
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
    cos_a = dx_total / total_dist
    sin_a = dy_total / total_dist

    n_steps = max(5, int(round(total_duration * _HZ)))
    n_steps = min(n_steps, 256)
    n_chunks = max(1, int(math.ceil(n_steps / STRIDE)))

    global_cond = torch.tensor(
        [[log_dist, log_dur, cos_a, sin_a]], dtype=torch.float32, device=_DEVICE
    )

    context = torch.zeros(1, CONTEXT_SIZE, 3, device=_DEVICE)
    all_dxdy = []
    cum_dx_norm = 0.0
    cum_dy_norm = 0.0

    with torch.no_grad():
        for k in range(n_chunks):
            is_last = (k == n_chunks - 1)
            rem_dx = (cos_a - cum_dx_norm) * _data_scale
            rem_dy = (sin_a - cum_dy_norm) * _data_scale
            rem_frac = 1.0 - (k * STRIDE) / n_steps
            progress = k / max(n_chunks - 1, 1)

            local_cond = torch.tensor(
                [[rem_dx, rem_dy, rem_frac, progress,
                  cum_dx_norm * _data_scale, cum_dy_norm * _data_scale]],
                dtype=torch.float32, device=_DEVICE,
            )

            chunk = _model.ddim_sample(
                context, global_cond, local_cond,
                n_steps=_N_DDIM_STEPS, eta=_ETA, cfg_scale=_CFG_SCALE,
            )  # (1, 25, 3)

            chunk_np = chunk[0].cpu().numpy()  # (25, 3)

            # Extract stall decisions
            stall_probs = 1.0 / (1.0 + np.exp(-chunk_np[:, 2]))
            stalls = stall_probs > 0.5

            # Get displacements, force zero on stalls
            dxdy = chunk_np[:, :2] / _data_scale
            dxdy[stalls] = 0.0

            # Determine how many steps to use from this chunk
            if is_last:
                useful = n_steps - k * STRIDE
            else:
                useful = min(STRIDE, n_steps - k * STRIDE)
            useful = max(1, min(useful, CHUNK_SIZE))

            all_dxdy.extend(dxdy[:useful].tolist())
            for j in range(useful):
                cum_dx_norm += dxdy[j, 0]
                cum_dy_norm += dxdy[j, 1]

            # Build context for next chunk from the tail of this chunk
            ctx_start = max(0, CHUNK_SIZE - CONTEXT_SIZE)
            new_ctx = chunk_np[ctx_start:CHUNK_SIZE]  # up to 5 steps
            context = torch.zeros(1, CONTEXT_SIZE, 3, device=_DEVICE)
            n_ctx = len(new_ctx)
            context[0, CONTEXT_SIZE - n_ctx:] = torch.from_numpy(new_ctx).to(_DEVICE)

    # Build positions in pixel space
    positions = [(start_x, start_y)]
    cx, cy = start_x, start_y
    for ddx, ddy in all_dxdy:
        cx += ddx * total_dist
        cy += ddy * total_dist
        positions.append((cx, cy))

    n = len(positions)
    if n < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, _DT)]

    # Magnitude-weighted endpoint correction
    actual_end_x, actual_end_y = positions[-1]
    err_x = end_x - actual_end_x
    err_y = end_y - actual_end_y

    if err_x * err_x + err_y * err_y > 0.01:
        mags = []
        for i in range(1, n):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            mags.append(math.hypot(dx, dy))

        total_mag = sum(mags)
        if total_mag > 1e-8:
            cum_frac = 0.0
            for i in range(1, n):
                cum_frac += mags[i - 1] / total_mag
                px, py = positions[i]
                positions[i] = (px + err_x * cum_frac, py + err_y * cum_frac)

    positions[0] = (start_x, start_y)
    positions[-1] = (end_x, end_y)

    # Timestamps via donor velocity profile
    donor_profile = _get_donor_velocity_profile(log_dist, n - 1)
    timestamps = _time_warp_timestamps(positions, total_duration, donor_profile)

    result = [
        (float(positions[i][0]), float(positions[i][1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)
    result[-1] = (end_x, end_y, result[-1][2])

    return result
