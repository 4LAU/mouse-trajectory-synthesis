from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import skew, wasserstein_distance

Trajectory = List[Tuple[float, float, float]]
FEATURE_NAMES = [
    "mean_velocity",
    "std_velocity",
    "max_velocity",
    "velocity_skewness",
    "mean_acceleration",
    "std_acceleration",
    "max_acceleration",
    "mean_jerk",
    "std_jerk",
    "path_efficiency",
    "max_deviation",
    "curvature_mean",
    "curvature_std",
    "num_direction_changes",
    "movement_duration",
    "time_to_peak_velocity",
    "angular_velocity_mean",
    "angular_velocity_std",
]


def resample_trajectory(trajectory: Sequence[Tuple[float, float, float]], hz: float = 125.0) -> Trajectory:
    if len(trajectory) < 2:
        return list(trajectory)
    pts = np.asarray(trajectory, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    t = np.maximum.accumulate(pts[:, 2])
    duration = t[-1] - t[0]
    if duration <= 0:
        return [(float(x[-1]), float(y[-1]), float(t[-1]))]

    step = 1.0 / hz
    target_t = np.arange(t[0], t[-1], step, dtype=np.float64)
    if target_t.size == 0 or target_t[-1] < t[-1]:
        target_t = np.append(target_t, t[-1])

    target_x = np.interp(target_t, t, x)
    target_y = np.interp(target_t, t, y)
    return list(zip(target_x.tolist(), target_y.tolist(), target_t.tolist()))


def extract_features(trajectory: Sequence[Tuple[float, float, float]]) -> np.ndarray | None:
    pts = np.asarray(trajectory, dtype=np.float64)
    if len(pts) < 5:
        return None

    x = pts[:, 0]
    y = pts[:, 1]
    t = pts[:, 2]
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.maximum(np.diff(t), 1e-6)
    ds = np.sqrt(dx ** 2 + dy ** 2)

    vx = dx / dt
    vy = dy / dt
    speed = ds / dt
    dv = np.diff(speed)
    dt2 = np.maximum(dt[:-1], 1e-6)
    acc = dv / dt2 if len(dv) else np.array([], dtype=np.float64)

    if len(acc) > 1:
        jerk = np.diff(acc) / np.maximum(dt2[:-1], 1e-6)
    else:
        jerk = np.array([], dtype=np.float64)

    d_straight = np.hypot(x[-1] - x[0], y[-1] - y[0])
    d_traveled = np.sum(ds)
    path_efficiency = d_straight / max(d_traveled, 1e-6)

    if d_straight > 1e-6:
        line_dx = x[-1] - x[0]
        line_dy = y[-1] - y[0]
        perp_dist = np.abs(line_dy * (x - x[0]) - line_dx * (y - y[0])) / d_straight
        max_dev = float(np.max(perp_dist))
    else:
        max_dev = 0.0

    if len(acc):
        ax_comp = np.diff(vx) / dt2
        ay_comp = np.diff(vy) / dt2
        speed_mid = np.maximum(speed[:-1], 1e-6)
        cross = np.abs(vx[:-1] * ay_comp - vy[:-1] * ax_comp)
        curvature = np.clip(cross / (speed_mid ** 3), 0, 1e6)
        curvature_mean = float(np.mean(curvature))
        curvature_std = float(np.std(curvature))
    else:
        curvature_mean = 0.0
        curvature_std = 0.0

    angles = np.arctan2(dy, dx)
    angle_diff = np.diff(angles)
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    sign_changes = float(np.sum(np.diff(np.sign(angle_diff)) != 0)) if len(angle_diff) > 1 else 0.0

    duration = float(t[-1] - t[0])
    peak_idx = int(np.argmax(speed))
    time_to_peak = (t[min(peak_idx, len(t) - 1)] - t[0]) / max(duration, 1e-6)
    omega = angle_diff / dt[:-1] if len(angle_diff) else np.array([0.0], dtype=np.float64)
    omega = np.clip(omega, -1e6, 1e6)

    return np.asarray(
        [
            float(np.mean(speed)),
            float(np.std(speed)),
            float(np.max(speed)),
            float(skew(speed)) if len(speed) > 2 and np.std(speed) > 1e-10 else 0.0,
            float(np.mean(acc)) if len(acc) else 0.0,
            float(np.std(acc)) if len(acc) else 0.0,
            float(np.max(np.abs(acc))) if len(acc) else 0.0,
            float(np.mean(jerk)) if len(jerk) else 0.0,
            float(np.std(jerk)) if len(jerk) else 0.0,
            float(path_efficiency),
            max_dev,
            curvature_mean,
            curvature_std,
            sign_changes,
            duration,
            float(time_to_peak),
            float(np.mean(np.abs(omega))),
            float(np.std(omega)),
        ],
        dtype=np.float64,
    )


def extract_feature_matrix(
    trajectories: Iterable[Sequence[Tuple[float, float, float]]],
    hz: float = 125.0,
) -> np.ndarray:
    rows = []
    for trajectory in trajectories:
        features = extract_features(resample_trajectory(trajectory, hz=hz))
        if features is not None and not np.any(np.isnan(features)):
            rows.append(features)
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, len(FEATURE_NAMES)))


def normalized_wasserstein_by_feature(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = []
    for idx in range(len(FEATURE_NAMES)):
        left_col = left[:, idx]
        right_col = right[:, idx]
        std = np.std(left_col)
        if std < 1e-10:
            distances.append(0.0)
            continue
        distances.append(float(wasserstein_distance(left_col / std, right_col / std)))
    return np.asarray(distances, dtype=np.float64)
