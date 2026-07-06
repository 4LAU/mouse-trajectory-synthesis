"""Shared utilities for experiment modules."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

Trajectory = list[tuple[float, float, float]]


class DurationModel:
    """Binned empirical duration model from training conditions.

    Default samples a Gaussian per log-distance bin. DUR_EMPIRICAL=1 draws an
    actual training log-duration from the bin (plus a 0.02 jitter) instead,
    which keeps the conditional skew and tails the Gaussian throws away. Both
    are the same existing duration-conditioning prior, only the fit differs.
    """

    def __init__(self, data_dir: str | Path, n_bins: int = 60, std_mult: float = 0.7):
        import os
        conditions = np.load(Path(data_dir) / "train_conditions.npy")
        train_log_dist = conditions[:, 0]
        train_log_dur = conditions[:, 1]

        self._n_bins = n_bins
        self._std_mult = std_mult
        self._empirical = os.environ.get("DUR_EMPIRICAL", "0") == "1"
        self._d_edges = np.linspace(
            train_log_dist.min(), train_log_dist.max(), n_bins + 1
        )
        self._dur_mean = np.zeros(n_bins)
        self._dur_std = np.zeros(n_bins)
        self._bin_durs: list[np.ndarray | None] = [None] * n_bins
        for b in range(n_bins):
            m = (train_log_dist >= self._d_edges[b]) & (
                train_log_dist < self._d_edges[b + 1]
            )
            if m.sum() >= 3:
                self._dur_mean[b] = train_log_dur[m].mean()
                self._dur_std[b] = train_log_dur[m].std()
                if self._empirical:
                    durs = train_log_dur[m]
                    if len(durs) > 20000:
                        durs = np.random.default_rng(b).choice(durs, 20000, replace=False)
                    self._bin_durs[b] = durs.copy()
            else:
                self._dur_mean[b] = np.median(train_log_dur)
                self._dur_std[b] = 0.12

        del conditions, train_log_dist, train_log_dur
        self._rng = np.random.default_rng()

    def sample(self, log_dist: float) -> float:
        bin_idx = int(
            np.clip(
                np.searchsorted(self._d_edges[1:], log_dist), 0, self._n_bins - 1
            )
        )
        if self._empirical and self._bin_durs[bin_idx] is not None:
            log_d = float(self._rng.choice(self._bin_durs[bin_idx]))
            log_d += float(self._rng.normal(0.0, 0.02 * self._std_mult))
        else:
            std = max(float(self._dur_std[bin_idx]), 0.05)
            log_d = float(self._rng.normal(float(self._dur_mean[bin_idx]), std * self._std_mult))
        return float(np.clip(math.exp(log_d), 0.05, 4.0))


def get_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
