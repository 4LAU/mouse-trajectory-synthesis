"""ZIMT with guided MDN sampling (no endpoint correction).

The key insight: ZIMT's 0.878 AUC ceiling comes from its endpoint
correction, not its architecture. The linear correction in the last 20%
creates an artificial velocity peak at ~90% of duration (human peak is
~35%). This is the #1 discriminative feature (time_to_peak_velocity).

Fix: instead of generating freely then correcting, GUIDE each MDN sample
toward the endpoint. At each step, shift the MDN component means by a
guidance vector proportional to the remaining displacement. The guidance
strength increases as remaining steps decrease, creating natural
deceleration toward the endpoint.

This is analogous to classifier-free guidance in diffusion models, but
applied to an autoregressive MDN.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.zimt import ZIMTModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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
_GUIDE_STRENGTH = float(os.environ.get("ZIMT_GUIDE_STRENGTH", "0.3"))

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
    f"[zimt_guided] ZIMT ({_cfg['d_model']}d, {_cfg['n_layers']}L, "
    f"{_cfg['n_components']}K MDN) | "
    f"guide={_GUIDE_STRENGTH}, temp={_TEMPERATURE}, gate_bias={_GATE_BIAS}"
)


def _guided_sample_step(params, remaining_dx, remaining_dy, remaining_frac,
                        temperature=1.0, gate_bias=0.0, guide_strength=0.3):
    """Sample from MDN with guidance toward endpoint.

    The guidance shifts MDN component means toward the "ideal" next step
    (remaining_displacement / remaining_steps). The shift strength increases
    as remaining_frac decreases (quadratic schedule).
    """
    gate_logit = params["gate_logit"][0, -1]
    pi = params["pi"][0, -1]
    mu = params["mu"][0, -1].clone()
    sigma = params["sigma"][0, -1]
    rho = params["rho"][0, -1]

    stall_prob = torch.sigmoid(gate_logit + gate_bias)
    is_stall = torch.bernoulli(stall_prob).item() > 0.5

    if is_stall:
        return 0.0, 0.0, True

    if temperature != 1.0:
        logit_pi = params["logit_pi"][0, -1]
        pi = torch.softmax(logit_pi / temperature, dim=0)
        sigma = sigma * temperature

    # Guidance: compute ideal next step displacement
    remaining_steps = max(remaining_frac * _MAX_SEQ, 1.0)
    ideal_dx = remaining_dx / remaining_steps
    ideal_dy = remaining_dy / remaining_steps

    # Guidance strength: quadratic increase as we approach endpoint
    # At frac=1.0 (start): strength ≈ 0
    # At frac=0.0 (end): strength = guide_strength
    alpha = guide_strength * (1.0 - remaining_frac) ** 2

    # Shift all component means toward ideal step
    mu[:, 0] = mu[:, 0] * (1.0 - alpha) + ideal_dx * alpha
    mu[:, 1] = mu[:, 1] * (1.0 - alpha) + ideal_dy * alpha

    comp_idx = torch.multinomial(pi, 1).item()

    mu_x = mu[comp_idx, 0].item()
    mu_y = mu[comp_idx, 1].item()
    sx = sigma[comp_idx, 0].item()
    sy = sigma[comp_idx, 1].item()
    r = rho[comp_idx].item()

    z1 = torch.randn(1).item()
    z2 = torch.randn(1).item()
    dx = mu_x + sx * z1
    dy = mu_y + sy * (r * z1 + math.sqrt(max(1 - r * r, 1e-8)) * z2)

    return dx, dy, False


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

            remaining_dx = cos_a - cum_dx
            remaining_dy = sin_a - cum_dy
            remaining_frac = 1.0 - step / n_target

            input_buf[0, step, 3] = remaining_dx
            input_buf[0, step, 4] = remaining_dy
            input_buf[0, step, 5] = remaining_frac

            params = _model(input_buf[:, :step + 1], condition)

            dx, dy, is_stall = _guided_sample_step(
                params,
                remaining_dx=remaining_dx,
                remaining_dy=remaining_dy,
                remaining_frac=remaining_frac,
                temperature=_TEMPERATURE,
                gate_bias=_GATE_BIAS,
                guide_strength=_GUIDE_STRENGTH,
            )

            generated_dxdy.append((dx, dy))
            cum_dx += dx
            cum_dy += dy

    # Build positions — scale from normalized to pixel space
    positions = [(start_x, start_y)]
    cx, cy = start_x, start_y
    for ddx, ddy in generated_dxdy:
        cx += ddx * total_dist
        cy += ddy * total_dist
        positions.append((cx, cy))

    # NO endpoint correction — guidance should have landed us close
    # Just snap the final point exactly
    positions[-1] = (end_x, end_y)

    n = len(positions)
    timestamps = [i * _DT for i in range(n)]

    result = [
        (float(positions[i][0]), float(positions[i][1]), timestamps[i])
        for i in range(n)
    ]
    result[0] = (start_x, start_y, 0.0)

    if len(result) < 3:
        return [(start_x, start_y, 0.0), (end_x, end_y, _DT)]

    return result
