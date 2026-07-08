"""Extract repo-standard 18-dim features from the AdSERP external dataset.

Standalone script under external_validation/ - does not modify or reimplement
any existing repo code. Segmentation mirrors the CANONICAL rule in setup_data.py
(_segment_movements / build_demo_pool), which is what actually built the
training pool - NOT the path-length-based rule in segment_count_adserp.py
(that script was a feasibility probe only). Per task spec, where the two
differ, the canonical setup_data.py rule wins:

  - split on inter-point gap > PAUSE_THRESHOLD_S (0.200s). setup_data.py also
    splits on Balabit button Pressed/Released events, which have no analogue
    in AdSERP's (mousemove, click) stream - AdSERP has no explicit
    press/release rows, so only the pause-gap split applies here.
  - a movement is valid iff raw point count >= MIN_POINTS (5) - counted
    BEFORE any de-duplication (setup_data.py never deduplicates consecutive
    identical coordinates, unlike segment_count_adserp.py).
  - the distance gate uses the STRAIGHT-LINE distance between the first and
    last point of the movement (hypot(ex-sx, ey-sy)), in
    [MIN_DISTANCE_PX, MAX_DISTANCE_PX] = [20, 5000] px - NOT total path
    length traveled, which is what segment_count_adserp.py checked instead.

Feature extraction itself is untouched: every valid movement's raw
(x, y, t_seconds) point list is passed straight into
features.extract_feature_matrix (features.py), the same function evaluate.py
and regenerate_human_features.py call. That function internally applies the
125Hz resample_trajectory step and the 18-feature extractor - no math is
reimplemented here.

Output:
  external_data/adserp_features_all.npy       (N, 18) float64, cached
  external_data/adserp_features_2000.npy      2000-row sample, rng seed 42,
                                               NaN/inf rows dropped before sampling
  external_validation/adserp_features_meta.json  counts, params, per-user tallies
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features import extract_feature_matrix, FEATURE_NAMES  # noqa: E402

EXT_DATA_DIR = REPO_ROOT / "external_data"  # raw datasets + cached .npy
OUT_DIR = Path(__file__).resolve().parent  # external_validation/ - meta.json goes here
DATA_DIR = EXT_DATA_DIR / "adserp" / "mouse-movement-data"

# Canonical constants, copied verbatim from setup_data.py (repo root).
PAUSE_THRESHOLD_S = 0.200
MIN_POINTS = 5
MIN_DISTANCE_PX = 20.0
MAX_DISTANCE_PX = 5000.0

VALID_EVENTS = {"mousemove", "click"}

Trajectory = list  # list[(x, y, t)]


def parse_session(path: Path):
    """Return sorted list of (x, y, t_seconds) for mousemove/click rows."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        for row in reader:
            if len(row) < 5:
                continue
            ts, x, y, event, _xpath = row[0], row[1], row[2], row[3], row[4]
            if event not in VALID_EVENTS:
                continue
            try:
                t_s = float(ts) / 1000.0
                xf = float(x)
                yf = float(y)
            except ValueError:
                continue
            rows.append((xf, yf, t_s))
    rows.sort(key=lambda r: r[2])
    return rows


def segment_movements(points: list) -> list:
    """Canonical segmentation: split on gap > PAUSE_THRESHOLD_S only (no
    press/release analogue available in AdSERP). Distance gate uses
    straight-line distance, min points counted pre-dedup - matches
    setup_data.py._segment_movements exactly, minus the Balabit-specific
    button-state resets which AdSERP's event stream has no equivalent for."""
    trajectories: list = []
    current: Trajectory = []

    def maybe_commit(pts: Trajectory) -> None:
        if len(pts) < MIN_POINTS:
            return
        sx, sy, _ = pts[0]
        ex, ey, _ = pts[-1]
        dist = math.hypot(ex - sx, ey - sy)
        if MIN_DISTANCE_PX <= dist <= MAX_DISTANCE_PX:
            trajectories.append(list(pts))

    for x, y, t in points:
        if current and (t - current[-1][2]) > PAUSE_THRESHOLD_S:
            maybe_commit(current)
            current = []
        current.append((x, y, t))
    maybe_commit(current)
    return trajectories


def user_id_from_filename(path: Path) -> str:
    return path.stem.split("-")[0].lstrip("p")


def main() -> None:
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Found {len(files)} session files under {DATA_DIR}")

    all_trajectories: list = []
    per_user_counts: dict = defaultdict(int)
    per_user_files: dict = defaultdict(int)

    for path in files:
        user = user_id_from_filename(path)
        per_user_files[user] += 1
        points = parse_session(path)
        if not points:
            continue
        movements = segment_movements(points)
        all_trajectories.extend(movements)
        per_user_counts[user] += len(movements)

    print(f"Segmented {len(all_trajectories)} valid movements "
          f"(canonical setup_data.py rule) across {len(per_user_files)} users")

    print("Extracting features via features.extract_feature_matrix "
          "(125Hz resample + 18-feature extractor, unmodified)...")
    feats = extract_feature_matrix(all_trajectories)
    n_movements = len(all_trajectories)
    n_extracted = len(feats)
    print(f"Feature matrix: {feats.shape} "
          f"({n_extracted}/{n_movements} movements yielded finite features; "
          f"extract_feature_matrix already drops NaN rows and rows with <5 "
          f"resampled points internally)")

    # Extra safety net: confirm no residual non-finite values before saving.
    finite_mask = np.all(np.isfinite(feats), axis=1)
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite:
        print(f"WARNING: {n_nonfinite} rows with non-finite features slipped "
              f"through extract_feature_matrix's own filter - dropping them.")
        feats = feats[finite_mask]

    all_path = EXT_DATA_DIR / "adserp_features_all.npy"
    np.save(all_path, feats)
    print(f"Saved {all_path} {feats.shape}")

    n_sample = 2000
    if len(feats) < n_sample:
        raise RuntimeError(
            f"Only {len(feats)} valid feature rows, need {n_sample} to sample.")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(feats), size=n_sample, replace=False)
    sample = feats[sample_idx]
    sample_path = EXT_DATA_DIR / "adserp_features_2000.npy"
    np.save(sample_path, sample)
    print(f"Saved {sample_path} {sample.shape} "
          f"(numpy default_rng(seed=42) draw of {n_sample} from {len(feats)})")

    meta = {
        "feature_names": FEATURE_NAMES,
        "n_files": len(files),
        "n_users": len(per_user_files),
        "n_movements_segmented": n_movements,
        "n_feature_rows_final": int(len(feats)),
        "n_nonfinite_dropped": n_nonfinite,
        "n_sampled_2000": n_sample,
        "sample_rng": "np.random.default_rng(seed=42)",
        "segmentation_params": {
            "source": "setup_data.py canonical rule (Balabit-derived), "
                      "NOT segment_count_adserp.py's path-length probe",
            "pause_threshold_s": PAUSE_THRESHOLD_S,
            "min_points": MIN_POINTS,
            "min_distance_px_straight_line": MIN_DISTANCE_PX,
            "max_distance_px_straight_line": MAX_DISTANCE_PX,
            "dedup_consecutive_identical_points": False,
            "timestamp_unit_conversion": "epoch_ms / 1000 -> seconds",
            "deviation_note": (
                "AdSERP has no discrete button press/release events (only "
                "mousemove/click rows), so the Balabit press/release segment "
                "reset in setup_data.py has no applicable analogue here; "
                "only the pause-gap split was applied."
            ),
        },
        "per_user_valid_movements": dict(per_user_counts),
        "per_user_file_counts": dict(per_user_files),
    }
    meta_path = OUT_DIR / "adserp_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
