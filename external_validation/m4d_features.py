"""Extract repo-standard 18-dim features from the M4D web-bot-detection
dataset's HUMAN-labeled sessions (bot sessions excluded entirely).

Standalone script under external_validation/ - does not modify or reimplement
any existing repo code. Two things are reused, kept deliberately separate:

  - Parsing (which files, which JSON keys, which bracket formats) is copied
    from segment_count_m4d.py's feasibility probe: phase1 sessions live one
    JSON file per folder ("[x,y]" coordinate brackets, small relative-ms
    timestamps); phase2 sessions live as JSON-lines rows ("m(x,y)" bracket
    format, epoch-ms timestamps). Human-only session ids are selected via the
    annotation splits exactly as that probe does (phase1: dedup of mirrored
    folders -> 50 unique human sessions; phase2: annotation-labeled base
    session ids -> 44 usable, keyed against the json dump).
  - Segmentation is the CANONICAL rule from setup_data.py (_segment_movements
    / build_demo_pool), copied verbatim from adserp_features.py - NOT
    segment_count_m4d.py's own segment_and_score rule. Per task spec, where
    the two differ, the canonical setup_data.py rule wins:

      - split on inter-point gap > PAUSE_THRESHOLD_S (0.200s) only (M4D has
        no button press/release events, same situation as AdSERP).
      - a movement is valid iff raw point count >= MIN_POINTS (5), counted
        BEFORE any de-duplication (setup_data.py never deduplicates
        consecutive identical coordinates; segment_count_m4d.py's probe did
        dedup before counting - that dedup step is deliberately NOT applied
        here).
      - the distance gate uses the STRAIGHT-LINE distance between the first
        and last point of the movement (hypot(ex-sx, ey-sy)), in
        [MIN_DISTANCE_PX, MAX_DISTANCE_PX] = [20, 5000] px - NOT total path
        length traveled, which is what segment_count_m4d.py's probe checked.

Feature extraction itself is untouched: every valid movement's raw
(x, y, t_seconds) point list is passed straight into
features.extract_feature_matrix (features.py), the same function evaluate.py,
regenerate_human_features.py, and adserp_features.py call. That function
internally applies the 125Hz resample_trajectory step and the 18-feature
extractor - no math is reimplemented here.

Output:
  external_data/m4d_features_all.npy       (N, 18) float64, cached
  external_data/m4d_features_2000.npy      2000-row sample, rng seed 42,
                                            NaN/inf rows dropped before sampling
  external_validation/m4d_features_meta.json  counts, params, per-session tallies
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features import extract_feature_matrix, FEATURE_NAMES  # noqa: E402

EXT_DATA_DIR = REPO_ROOT / "external_data"  # raw dataset + cached .npy
OUT_DIR = Path(__file__).resolve().parent  # external_validation/ - meta.json goes here
M4D = EXT_DATA_DIR / "m4d"

# Canonical constants, copied verbatim from setup_data.py (repo root), same
# as adserp_features.py.
PAUSE_THRESHOLD_S = 0.200
MIN_POINTS = 5
MIN_DISTANCE_PX = 20.0
MAX_DISTANCE_PX = 5000.0

ANNOT_RE_PLAIN = re.compile(r"([0-9a-z]{20,30})\s+(human|advanced_bot|moderate_bot)")
ANNOT_RE_SUFFIX = re.compile(r"([0-9a-z]{20,30}(?:_\d+)?)\s+(human|advanced_bot|moderate_bot)")

Trajectory = list  # list[(x, y, t)]


# ---------------------------------------------------------------------------
# Parsing (copied from segment_count_m4d.py's feasibility probe)
# ---------------------------------------------------------------------------

def collect_phase1_human_ids():
    files = [
        "phase1/annotations/humans_and_advanced_bots/train",
        "phase1/annotations/humans_and_advanced_bots/test",
        "phase1/annotations/humans_and_moderate_bots/train",
        "phase1/annotations/humans_and_moderate_bots/test",
    ]
    humans, bots = set(), set()
    for rel in files:
        path = M4D / rel
        data = path.read_text(encoding="utf-8", errors="replace")
        for sid, label in ANNOT_RE_PLAIN.findall(data):
            if label == "human":
                humans.add(sid)
            else:
                bots.add(sid)
    return humans, bots


def collect_phase2_human_ids():
    files = [
        "phase2/annotations/humans_and_advanced_bots/humans_and_advanced_bots",
        "phase2/annotations/humans_and_moderate_and_advanced_bots/humans_and_moderate_and_advanced_bots",
    ]
    human_entries, bot_entries = set(), set()
    for rel in files:
        path = M4D / rel
        data = path.read_text(encoding="utf-8", errors="replace")
        for sid, label in ANNOT_RE_SUFFIX.findall(data):
            if label == "human":
                human_entries.add(sid)
            else:
                bot_entries.add(sid)
    base_ids = set(s.rsplit("_", 1)[0] for s in human_entries)
    return base_ids, human_entries, bot_entries


def parse_phase1_session(session_id: str, folder_hint: str):
    """Load one phase1 session's mouse_movements.json ("[x,y]" format).
    Returns (points, shortfall) where points is a list of (x, y, t_seconds)
    NOT yet sorted."""
    path = (M4D / "phase1" / "data" / "mouse_movements" / folder_hint /
            session_id / "mouse_movements.json")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    times_raw = [t for t in obj["mousemove_times"].split(",") if t.strip() != ""]
    times = [float(t) for t in times_raw]
    coords = re.findall(r"\[(-?[\d.]+),(-?[\d.]+)\]", obj["mousemove_total_behaviour"])
    n = min(len(times), len(coords))
    shortfall = max(len(times), len(coords)) - n
    points = [(float(coords[i][0]), float(coords[i][1]), times[i] / 1000.0) for i in range(n)]
    return points, shortfall


def parse_phase2_session(row: dict):
    """Parse one phase2 aggregate JSON row ("m(x,y)" format, epoch-ms times).
    Returns (points, shortfall), points as (x, y, t_seconds) NOT yet sorted."""
    times_raw = [t for t in row["mousemove_times"].split(",") if t.strip() != ""]
    times = [float(t) for t in times_raw]
    coords = re.findall(r"m\((-?[\d.]+),(-?[\d.]+)\)", row["mousemove_total_behaviour"])
    n = min(len(times), len(coords))
    shortfall = max(len(times), len(coords)) - n
    points = [(float(coords[i][0]), float(coords[i][1]), times[i] / 1000.0) for i in range(n)]
    return points, shortfall


# ---------------------------------------------------------------------------
# Canonical segmentation (copied verbatim from adserp_features.py, which
# mirrors setup_data.py._segment_movements)
# ---------------------------------------------------------------------------

def segment_movements(points: list) -> list:
    """Canonical segmentation: split on gap > PAUSE_THRESHOLD_S only (no
    press/release analogue available in M4D, same as AdSERP). Distance gate
    uses straight-line distance, min points counted pre-dedup - matches
    setup_data.py._segment_movements exactly, minus the Balabit-specific
    button-state resets which M4D's event stream has no equivalent for."""
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


def main() -> None:
    all_trajectories: list = []
    per_session_counts: dict = defaultdict(int)
    sessions_parsed = 0
    parse_failures: dict = defaultdict(int)
    phase1_shortfall_total = 0
    phase2_shortfall_total = 0

    # ---- phase1 ----
    p1_humans, p1_bots = collect_phase1_human_ids()
    print(f"[phase1] annotation-confirmed unique human sessions: {len(p1_humans)} "
          f"(bot entries seen: {len(p1_bots)}, excluded)")

    for sid in sorted(p1_humans):
        folder_hint = "humans_and_advanced_bots"
        candidate = M4D / "phase1" / "data" / "mouse_movements" / folder_hint / sid
        if not candidate.is_dir():
            folder_hint = "humans_and_moderate_bots"
        try:
            points, shortfall = parse_phase1_session(sid, folder_hint)
            phase1_shortfall_total += shortfall
            sessions_parsed += 1
        except Exception as e:  # noqa: BLE001
            parse_failures[f"phase1:{type(e).__name__}"] += 1
            print(f"  [FAIL] phase1 session {sid}: {e}", file=sys.stderr)
            continue
        points.sort(key=lambda p: p[2])
        movements = segment_movements(points)
        all_trajectories.extend(movements)
        per_session_counts[f"phase1:{sid}"] = len(movements)

    # ---- phase2 ----
    p2_base_ids, p2_human_entries, p2_bot_entries = collect_phase2_human_ids()
    print(f"[phase2] annotation-confirmed human sub-session rows: {len(p2_human_entries)} "
          f"-> {len(p2_base_ids)} unique base session_ids (bot entries seen: "
          f"{len(p2_bot_entries)}, excluded)")

    p2_json_path = (M4D / "phase2" / "data" / "mouse_movements" / "humans" /
                     "mouse_movements_humans.json")
    p2_rows_by_sid = {}
    p2_total_rows = 0
    with open(p2_json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p2_total_rows += 1
            row = json.loads(line)
            p2_rows_by_sid[row["session_id"]] = row
    unlabeled_in_dump = set(p2_rows_by_sid.keys()) - p2_base_ids
    print(f"[phase2] json dump has {p2_total_rows} total session rows; "
          f"{len(unlabeled_in_dump)} present in dump but never annotation-labeled "
          f"human -> excluded")

    for sid in sorted(p2_base_ids):
        row = p2_rows_by_sid.get(sid)
        if row is None:
            parse_failures["phase2:missing_from_json_dump"] += 1
            print(f"  [FAIL] phase2 session {sid}: annotated human but absent "
                  f"from json dump", file=sys.stderr)
            continue
        try:
            points, shortfall = parse_phase2_session(row)
            phase2_shortfall_total += shortfall
            sessions_parsed += 1
        except Exception as e:  # noqa: BLE001
            parse_failures[f"phase2:{type(e).__name__}"] += 1
            print(f"  [FAIL] phase2 session {sid}: {e}", file=sys.stderr)
            continue
        points.sort(key=lambda p: p[2])
        movements = segment_movements(points)
        all_trajectories.extend(movements)
        per_session_counts[f"phase2:{sid}"] = len(movements)

    n_movements = len(all_trajectories)
    print(f"\nSegmented {n_movements} valid movements (canonical setup_data.py "
          f"rule) across {sessions_parsed} human sessions "
          f"(phase1={len(p1_humans)}, phase2={len(p2_base_ids)}); "
          f"parse failures: {dict(parse_failures) if parse_failures else 'none'}")
    print(f"Timestamp/coord count shortfall (events dropped by positional zip): "
          f"phase1={phase1_shortfall_total}, phase2={phase2_shortfall_total}")

    print("Extracting features via features.extract_feature_matrix "
          "(125Hz resample + 18-feature extractor, unmodified)...")
    feats = extract_feature_matrix(all_trajectories)
    n_extracted = len(feats)
    n_dropped_by_extractor = n_movements - n_extracted
    print(f"Feature matrix: {feats.shape} "
          f"({n_extracted}/{n_movements} movements yielded finite features; "
          f"extract_feature_matrix already drops NaN rows and rows with <5 "
          f"resampled points internally -> {n_dropped_by_extractor} dropped here)")

    # Extra safety net: confirm no residual non-finite values before saving.
    finite_mask = np.all(np.isfinite(feats), axis=1)
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite:
        print(f"WARNING: {n_nonfinite} rows with non-finite features slipped "
              f"through extract_feature_matrix's own filter - dropping them.")
        feats = feats[finite_mask]

    total_dropped = n_movements - len(feats)
    print(f"Movement counts -> all: {n_movements}, valid (finite features): "
          f"{len(feats)}, dropped: {total_dropped} "
          f"(extractor internal: {n_dropped_by_extractor}, residual non-finite: "
          f"{n_nonfinite})")

    all_path = EXT_DATA_DIR / "m4d_features_all.npy"
    np.save(all_path, feats)
    print(f"Saved {all_path} {feats.shape}")

    n_sample = 2000
    if len(feats) < n_sample:
        raise RuntimeError(
            f"Only {len(feats)} valid feature rows, need {n_sample} to sample.")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(feats), size=n_sample, replace=False)
    sample = feats[sample_idx]
    sample_path = EXT_DATA_DIR / "m4d_features_2000.npy"
    np.save(sample_path, sample)
    print(f"Saved {sample_path} {sample.shape} "
          f"(numpy default_rng(seed=42) draw of {n_sample} from {len(feats)})")

    meta = {
        "feature_names": FEATURE_NAMES,
        "n_phase1_human_sessions": len(p1_humans),
        "n_phase2_human_base_sessions": len(p2_base_ids),
        "n_sessions_parsed": sessions_parsed,
        "parse_failures": dict(parse_failures),
        "phase1_timestamp_coord_shortfall": phase1_shortfall_total,
        "phase2_timestamp_coord_shortfall": phase2_shortfall_total,
        "n_movements_segmented": n_movements,
        "n_dropped_by_extractor": int(n_dropped_by_extractor),
        "n_nonfinite_dropped": n_nonfinite,
        "n_feature_rows_final": int(len(feats)),
        "n_movements_dropped_total": int(total_dropped),
        "n_sampled_2000": n_sample,
        "sample_rng": "np.random.default_rng(seed=42)",
        "segmentation_params": {
            "source": "setup_data.py canonical rule (Balabit-derived), "
                      "NOT segment_count_m4d.py's own segment_and_score probe",
            "pause_threshold_s": PAUSE_THRESHOLD_S,
            "min_points": MIN_POINTS,
            "min_distance_px_straight_line": MIN_DISTANCE_PX,
            "max_distance_px_straight_line": MAX_DISTANCE_PX,
            "dedup_consecutive_identical_points": False,
            "timestamp_unit_conversion": "epoch_ms / 1000 -> seconds "
                                         "(phase1 relative ms, phase2 epoch ms; "
                                         "gap math identical either way)",
            "deviation_note": (
                "M4D has no discrete button press/release events (only "
                "mousemove coordinate/time streams), so the Balabit "
                "press/release segment reset in setup_data.py has no "
                "applicable analogue here; only the pause-gap split was "
                "applied. segment_count_m4d.py's own probe deduplicated "
                "consecutive identical coordinates before counting MIN_POINTS "
                "and gated on total path length; the canonical rule used here "
                "does neither."
            ),
        },
        "per_session_valid_movements": dict(per_session_counts),
    }
    meta_path = OUT_DIR / "m4d_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
