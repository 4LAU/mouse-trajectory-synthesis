"""VQ-VAE + autoregressive transformer trajectory generator.

Approach:
  1. VQ-VAE tokenizes (dx, dy) displacements into 1024 motion tokens
  2. GRPO-finetuned transformer autoregressively generates token sequences
     conditioned on (log_dist, log_dur, cos_angle, sin_angle)
  3. Classifier-free guidance (CFG scale 3.0) on endpoint conditioning
  4. Cumulative displacements compose the trajectory

Expected AUC: ~0.892 (full pool, n=2000)

Key insight:
  Discrete tokenization sidesteps the blurriness problem of continuous
  generative models.  GRPO finetuning + CFG gives the best results of
  all approaches tested.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.vqvae import MotionVQVAE
from models.trajectory_transformer import TrajectoryTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
_parser.add_argument("--vqvae-checkpoint", default=None)
_parser.add_argument("--transformer-checkpoint", default=None)
_args, _ = _parser.parse_known_args()

_DATA_DIR = Path(_args.data_dir)
_VQVAE_PATH = (
    Path(_args.vqvae_checkpoint) if _args.vqvae_checkpoint
    else _DATA_DIR / "vqvae_best.pt"
)
_TF_PATH = (
    Path(_args.transformer_checkpoint) if _args.transformer_checkpoint
    else _DATA_DIR / "trajectory_transformer_best.pt"
)

_DEVICE = get_device()
_HZ = 125.0
_DT = 1.0 / _HZ

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
_duration = DurationModel(_DATA_DIR)


# ---------------------------------------------------------------------------
# Load VQ-VAE
# ---------------------------------------------------------------------------
_vqvae_ckpt = torch.load(_VQVAE_PATH, map_location="cpu", weights_only=False)
_vqvae_cfg = _vqvae_ckpt["config"]
_norm_mean = np.array(_vqvae_ckpt["norm_mean"], dtype=np.float32)
_norm_std = np.array(_vqvae_ckpt["norm_std"], dtype=np.float32)
_clip_lo = np.array(_vqvae_ckpt["clip_lo"], dtype=np.float32)
_clip_hi = np.array(_vqvae_ckpt["clip_hi"], dtype=np.float32)

_vqvae = MotionVQVAE(
    n_codes=_vqvae_cfg["n_codes"],
    code_dim=_vqvae_cfg["code_dim"],
).to(_DEVICE)
_vqvae.load_state_dict(_vqvae_ckpt["model_state_dict"])
_vqvae.train(False)

# ---------------------------------------------------------------------------
# Load Transformer
# ---------------------------------------------------------------------------
_tf_ckpt = torch.load(_TF_PATH, map_location="cpu", weights_only=False)
_tf_cfg = _tf_ckpt["config"]

_transformer = TrajectoryTransformer(
    vocab_size=_tf_cfg["vocab_size"],
    d_model=_tf_cfg["d_model"],
    n_heads=_tf_cfg["n_heads"],
    n_layers=_tf_cfg["n_layers"],
    d_ff=_tf_cfg["d_ff"],
    max_seq_len=_tf_cfg["max_seq_len"],
    cond_dim=_tf_cfg["cond_dim"],
).to(_DEVICE)
_transformer.load_state_dict(_tf_ckpt["model_state_dict"], strict=False)
_transformer.train(False)

_MAX_SEQ = _tf_cfg["max_seq_len"]

print(
    f"[vqvae_ar_transformer] VQ-VAE ({_vqvae_cfg['n_codes']} codes) + "
    f"Transformer (ep {_tf_ckpt.get('epoch', '?')}, "
    f"val_loss={_tf_ckpt.get('val_loss', 0):.4f}, val_acc={_tf_ckpt.get('val_acc', 0):.4f})"
)

# ---------------------------------------------------------------------------
# Pre-decode all 1024 motion tokens for speed
# ---------------------------------------------------------------------------
_TOKEN_DXDY: dict[int, tuple[float, float]] = {}
with torch.no_grad():
    _all_idx = torch.arange(1024, dtype=torch.long, device=_DEVICE)
    _all_dxdy_norm = _vqvae.decode(_all_idx).cpu().numpy()
    _all_dxdy = _all_dxdy_norm * _norm_std + _norm_mean
    _TOKEN_DXDY[0] = (0.0, 0.0)  # stall token
    for _i in range(1024):
        _TOKEN_DXDY[_i + 1] = (float(_all_dxdy[_i, 0]), float(_all_dxdy[_i, 1]))

del _all_idx, _all_dxdy_norm, _all_dxdy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CFG_SCALE = 3.0


def generate_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> Trajectory:
    """Generate a trajectory via autoregressive token sampling with CFG."""
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

    expected_speed = total_dist / total_duration
    expected_dx = cos_a * expected_speed * _DT * 0.8
    expected_dy = sin_a * expected_speed * _DT * 0.8
    start_dxdy = np.clip(
        np.array([[expected_dx, expected_dy]], dtype=np.float32),
        _clip_lo, _clip_hi,
    )
    start_normed = (start_dxdy - _norm_mean) / _norm_std
    with torch.no_grad():
        start_token = _vqvae.encode(
            torch.tensor(start_normed, dtype=torch.float32, device=_DEVICE)
        ).item() + 1

    generated = torch.tensor([[start_token]], dtype=torch.long, device=_DEVICE)

    with torch.no_grad():
        for _ in range(n_target - 2):
            T = generated.shape[1]
            ep_info = torch.zeros(1, T, 3, device=_DEVICE)
            _cum_dx, _cum_dy = 0.0, 0.0
            for t in range(T):
                tok = generated[0, t].item()
                tdx, tdy = _TOKEN_DXDY.get(int(tok), (0.0, 0.0))
                _cum_dx += tdx
                _cum_dy += tdy
                ep_info[0, t, 0] = (dx_total - _cum_dx) / max(total_dist, 1.0)
                ep_info[0, t, 1] = (dy_total - _cum_dy) / max(total_dist, 1.0)
                ep_info[0, t, 2] = 1.0 - (t + 1) / n_target

            if _CFG_SCALE > 1.0:
                logits_cond = _transformer(generated, condition, ep_info)
                ep_info_null = torch.zeros_like(ep_info)
                logits_uncond = _transformer(generated, condition, ep_info_null)
                next_logits = (
                    logits_uncond[0, -1, :]
                    + _CFG_SCALE * (logits_cond[0, -1, :] - logits_uncond[0, -1, :])
                )
            else:
                logits = _transformer(generated, condition, ep_info)
                next_logits = logits[0, -1, :]

            probs = torch.softmax(next_logits, dim=0)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(0)
            cutoff = (cumsum > 0.95).nonzero()
            if len(cutoff) > 0:
                k = max(5, cutoff[0].item() + 1)
            else:
                k = len(probs)
            top_vals = sorted_probs[:k]
            top_idx = sorted_idx[:k]
            top_vals = top_vals / top_vals.sum()
            sampled_pos = torch.multinomial(top_vals, 1)
            next_token = top_idx[sampled_pos]

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if generated.shape[1] >= _MAX_SEQ - 2:
                break

    token_seq = generated[0].cpu().numpy()

    positions = [(start_x, start_y)]
    cx, cy = start_x, start_y
    for tok in token_seq:
        dx, dy = _TOKEN_DXDY.get(int(tok), (0.0, 0.0))
        cx += dx
        cy += dy
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
