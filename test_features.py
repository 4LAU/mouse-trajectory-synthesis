"""Sanity tests for extract_features()."""

import math
import numpy as np
import pytest
from features import extract_features


def test_straight_line():
    """Straight line: path_efficiency ~1.0, curvature ~0.0."""
    traj = [(i * 10.0, 0.0, i * 8.0) for i in range(10)]
    feats = extract_features(traj)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (18,)
    assert feats[9] == pytest.approx(1.0, abs=0.01)   # path_efficiency
    assert feats[11] == pytest.approx(0.0, abs=0.01)   # curvature_mean


def test_circular_trajectory():
    """Points around a circle should have positive curvature."""
    n = 20
    traj = [
        (100 * math.cos(2 * math.pi * i / n),
         100 * math.sin(2 * math.pi * i / n),
         i * 8.0)
        for i in range(n)
    ]
    feats = extract_features(traj)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (18,)
    assert feats[11] > 0  # curvature_mean


def test_too_few_points_returns_none():
    """Fewer than 5 points should return None."""
    traj = [(0, 0, 0), (1, 1, 8), (2, 2, 16), (3, 3, 24)]
    assert extract_features(traj) is None
