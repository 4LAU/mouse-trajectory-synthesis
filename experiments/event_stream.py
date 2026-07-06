"""WS7 event-stream model experiment.

Generates (dt, dx, dy) event sequences, integrates them into positions at
cumulative timestamps, and returns the raw event trajectory. evaluate.py
applies the standard 125Hz resample during feature extraction, so the human
recording artifacts appear by construction. Pure T3: no post-processing.

Env knobs:
  EVENT_CKPT   checkpoint name in training/ (default event_stream_best.pt)
  EVENT_STEPS  sampler steps (default 100)
  EVENT_TEMP   softmax temperature for token confidence (default 1.0)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

from experiments._common import DurationModel, Trajectory, get_device
from models.event_stream import PAD_CLASS, VOCAB_MAX, EventStreamModel

torch.manual_seed(42)

_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))
_DEVICE = get_device()

_ckpt_name = os.environ.get("EVENT_CKPT", "event_stream_best.pt")
_ckpt = torch.load(_TRAIN_DIR / _ckpt_name, map_location=_DEVICE, weights_only=False)
_cfg = _ckpt["config"]
_DT_MEAN = float(_ckpt["dt_mean"])
_DT_STD = float(_ckpt["dt_std"])

_model = EventStreamModel(**_cfg).to(_DEVICE)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_duration = DurationModel(_TRAIN_DIR, std_mult=float(os.environ.get("EVENT_DUR_STD", "0.7")))
_N_STEPS = int(os.environ.get("EVENT_STEPS", "100"))
_TEMP = float(os.environ.get("EVENT_TEMP", "1.0"))
_EVAL_BATCH = int(os.environ.get("EVENT_EVAL_BATCH", "256"))

print(f"[event_stream] ckpt={_ckpt_name} epoch={_ckpt.get('epoch')} "
      f"steps={_N_STEPS} temp={_TEMP}")


def _decode(dt_z, dx_cls, dy_cls, sx, sy) -> Trajectory | None:
    # Truncate at the first PAD on either head
    pad = (dx_cls >= PAD_CLASS) | (dy_cls >= PAD_CLASS)
    n = int(np.argmax(pad)) if pad.any() else len(dx_cls)
    if n < 2:
        return None

    dx = dx_cls[:n].astype(np.float64) - VOCAB_MAX
    dy = dy_cls[:n].astype(np.float64) - VOCAB_MAX
    dt_ms = np.exp(dt_z[:n] * _DT_STD + _DT_MEAN)
    dt_s = np.clip(dt_ms, 0.1, 1000.0) / 1000.0

    x = np.concatenate([[sx], sx + np.cumsum(dx)])
    y = np.concatenate([[sy], sy + np.cumsum(dy)])
    t = np.concatenate([[0.0], np.cumsum(dt_s)])
    return list(zip(x.tolist(), y.tolist(), t.tolist()))


def generate_paths(specs: list) -> list:
    results: list = [None] * len(specs)
    pending = []
    for idx, (sx, sy, ex, ey) in enumerate(specs):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            results[idx] = [(sx, sy, 0.0), (ex, ey, 0.008)]
            continue
        log_dist = math.log(dist)
        angle = math.atan2(ey - sy, ex - sx)
        log_dur = math.log(_duration.sample(log_dist))
        pending.append({
            "idx": idx, "sx": sx, "sy": sy,
            "cond": [log_dist, log_dur, math.cos(angle), math.sin(angle)],
        })

    seq_len = _cfg["max_seq_len"]
    for c0 in range(0, len(pending), _EVAL_BATCH):
        chunk = pending[c0:c0 + _EVAL_BATCH]
        cond = torch.tensor([it["cond"] for it in chunk],
                            dtype=torch.float32, device=_DEVICE)
        with torch.no_grad():
            dt_z, dx_tok, dy_tok = _model.sample(
                cond, seq_len, n_steps=_N_STEPS, temperature=_TEMP,
            )
        dt_np = dt_z.float().cpu().numpy()
        dx_np = dx_tok.cpu().numpy()
        dy_np = dy_tok.cpu().numpy()
        for k, it in enumerate(chunk):
            results[it["idx"]] = _decode(dt_np[k], dx_np[k], dy_np[k],
                                         it["sx"], it["sy"])
    return results


def generate_path(sx, sy, ex, ey) -> Trajectory | None:
    return generate_paths([(sx, sy, ex, ey)])[0]
