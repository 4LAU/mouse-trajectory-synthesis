"""ZIMT with rejection sampling — generate N candidates, keep the most human-like.

For each query, generates N candidate trajectories and selects the one whose
kinematic features are closest to the human population statistics (measured
by normalized L2 distance from the human feature mean).

N candidates costs N× generation time but requires no retraining.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from experiments._common import Trajectory
from experiments.zimt_magcorr import generate_path as _base_generate
from features import extract_features, FEATURE_NAMES

_N_CANDIDATES = int(os.environ.get("ZIMT_N_CANDIDATES", "10"))

_DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
_human_features = np.load(_DATA_DIR / "human_eval_features.npy")
_h_mean = _human_features.mean(axis=0)
_h_std = np.maximum(_human_features.std(axis=0), 1e-8)

print(f"[zimt_reject] N_CANDIDATES={_N_CANDIDATES}")


def _score(traj: Trajectory) -> float:
    """Lower score = more human-like."""
    feats = extract_features(traj)
    if feats is None:
        return float("inf")
    diff = (feats - _h_mean) / _h_std
    return float(np.sum(diff ** 2))


def generate_path(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
) -> Trajectory:
    best_traj = None
    best_score = float("inf")

    for _ in range(_N_CANDIDATES):
        traj = _base_generate(start_x, start_y, end_x, end_y)
        s = _score(traj)
        if s < best_score:
            best_score = s
            best_traj = traj

    return best_traj
