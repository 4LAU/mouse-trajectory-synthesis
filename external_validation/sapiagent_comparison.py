"""Can the headline's RF judge tell our humans from SapiAgent's synthetics?

SapiAgent (github.com/margitantal68/sapiagent, Antal & Fejer, TF/Keras) is a
prior-art mouse-trajectory generator: a convolutional autoencoder trained on
SapiMouse recordings, run in "unsupervised" mode (its own default: settings.py
KEY='fcn', TRAINING_TYPE='unsupervised', 100 epochs). This script does not
retrain or reimplement SapiAgent - it loads the CSV artifacts their own
pipeline already produced, reshapes them into our (x, y, t) trajectory
format, applies our canonical segmentation acceptance gates, and scores them
with the exact detector this repo's headline uses.

How the SapiAgent artifacts were produced (short-path venv required - a
straight pip install of tensorflow inside this repo's own deeply-nested
scratchpad path hits a Windows long-path OSError):
    1. py -3.12 -m venv C:/Users/<user>/AppData/Local/Temp/sapi/venv
    2. that venv's pip install tensorflow-cpu pandas numpy scikit-learn
       matplotlib requests seaborn (seaborn is an undeclared dependency of
       their plots.py, imported by autoencoder_training.py)
    3. git clone https://github.com/margitantal68/sapiagent into
       C:/Users/<user>/AppData/Local/Temp/sapi/sapiagent
    4. download http://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip,
       unzip into sapiagent/sapimouse/ (the zip nests an extra sapimouse/
       folder - flatten it so sapiagent/sapimouse/user1/... etc.)
    5. inside sapiagent/: python create_sapimouse_actions.py (segments raw
       sessions into fixed 128-step dx,dy actions; also writes
       statistics/actions_start_stop_{1min,3min}.csv, one row per action,
       recording that action's real start point, true point count and true
       elapsed time, in the same row order as the action files)
    6. python create_equidistant_actions.py (builds the equidistant curves
       used as model input/output)
    7. python autoencoder_training.py (trains the default fcn autoencoder,
       unsupervised, 100 epochs, on actions_3min_dx_dy.csv - unmodified
       default settings.py, ~13 minutes on CPU for this dataset size)
    8. python generate_autoencoder_actions.py (runs the trained autoencoder
       over equidistant_actions/equidistant_1min.csv, writing
       generated_actions/generated_fcn_dx_dy_mse_unsupervised.csv - 128 dx
       values then 128 dy values per row, zero-padded, same row order as
       statistics/actions_start_stop_1min.csv)
No SapiAgent code or data is copied into this repository; this script only
reads the two CSVs described above from wherever SAPIAGENT_DIR points.

Timing reconstruction (disclose plainly, this is not an artifact of our
harness): SapiAgent generates spatial paths only - dx, dy displacements at
128 fixed steps - with no timing model at all. To score it with a feature
set that includes velocity/acceleration/jerk, we have to invent timestamps.
For each generated action we take its real total elapsed time and its real
point count from actions_start_stop_1min.csv (both true recordings of the
source SapiMouse action, not fabricated), trim the generated 128-step
output to that true point count, cumsum the trimmed dx,dy from the action's
real start point to get (x, y), and spread the real total elapsed time
evenly across the points to get t. The total duration per action is real;
the shape of the time axis inside that duration (uniform spacing) is not -
real mouse movement accelerates and decelerates, so this fabrication
particularly distorts jerk and acceleration features, and to a lesser
extent velocity ones. Curvature and the pure-geometry features (path
efficiency, max deviation, direction changes) are unaffected: with a
constant per-step dt, the dt terms cancel algebraically inside the
curvature formula, and the shape features never reference time at all. A
secondary AUC restricted to those timing-free features is reported below
for this reason.

Segmentation: each SapiAgent action already corresponds to a single
uninterrupted movement (no pauses to split on), so the canonical
setup_data.py rule reduces to its two static gates - length >= MIN_POINTS
(5) and straight-line start-end distance in [MIN_DISTANCE_PX,
MAX_DISTANCE_PX] = [20, 5000] px - evaluated on the reconstructed points,
not the original SapiMouse endpoints.

Detector: RF-OOB with n_estimators=100, oob_score=True, same as evaluate.py
and validate_adserp.py. Since SapiAgent has only one generated set (unlike
our three headline seeds, which each pick from a different candidate pool),
"seeds 42/43/44" here sweep the RF's own random_state as a detector-stability
check on the one fixed (SapiAgent, human) pair, matching the detector-seed
sweep already used elsewhere in this project (see EXPERIMENTS.md, July 8).

Run:
    .venv/Scripts/python.exe external_validation/sapiagent_comparison.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features import extract_feature_matrix, FEATURE_NAMES  # noqa: E402
from setup_data import _segment_movements  # noqa: E402

EXT_DIR = Path(__file__).resolve().parent
EXT_DATA_DIR = REPO_ROOT / "external_data"

# Where the SapiAgent pipeline (see module docstring) was run. Override with
# the SAPIAGENT_DIR env var if reproducing elsewhere.
SAPIAGENT_DIR = Path(os.environ.get(
    "SAPIAGENT_DIR", r"C:\Users\aaron\AppData\Local\Temp\sapi\sapiagent"))
GENERATED_CSV = SAPIAGENT_DIR / "generated_actions" / \
    "generated_fcn_dx_dy_mse_unsupervised.csv"
START_STOP_CSV = SAPIAGENT_DIR / "statistics" / "actions_start_stop_1min.csv"
SAPIMOUSE_RAW_DIR = SAPIAGENT_DIR / "sapimouse"

# Canonical constants, copied verbatim from setup_data.py.
PAUSE_THRESHOLD_S = 0.200
MIN_POINTS = 5
MIN_DISTANCE_PX = 20.0
MAX_DISTANCE_PX = 5000.0
MAX_LEN = 128  # SapiAgent's fixed step count (settings.py FEATURES)

DETECTOR_SEEDS = [42, 43, 44]
SANITY_TOLERANCE = 1e-3
EXPECTED_HEADLINE_SEED42 = 0.5095

# Timing-free subset: pure path/shape geometry, unaffected by the evenly
# spread fabricated timestamps (see module docstring). Everything else
# (velocity, acceleration, jerk, duration, time-to-peak, angular velocity)
# depends on the assumed-uniform per-step dt in some way.
TIMING_FREE_FEATURES = [
    "path_efficiency", "max_deviation", "curvature_mean",
    "curvature_std", "num_direction_changes",
]
TIMING_FREE_IDX = [FEATURE_NAMES.index(f) for f in TIMING_FREE_FEATURES]


# ---------------------------------------------------------------------------
# SapiAgent generated-action transform
# ---------------------------------------------------------------------------

def load_sapiagent_trajectories() -> tuple[list, dict]:
    """Reconstruct (x, y, t) trajectories from SapiAgent's generated dx,dy
    rows plus the matching real start point / length / elapsed time, then
    apply the two static canonical acceptance gates. Returns (accepted
    trajectories in original row order, stats dict)."""
    gen = np.loadtxt(GENERATED_CSV, delimiter=",")  # (N, 256)
    meta_rows = []
    with open(START_STOP_CSV, newline="") as fh:
        reader = csv.DictReader(fh)  # startx,starty,stopx,stopy,length,time,userid
        for row in reader:
            meta_rows.append(row)

    if len(gen) != len(meta_rows):
        raise RuntimeError(
            f"row count mismatch: generated={len(gen)} vs meta={len(meta_rows)}; "
            "these must be produced from the same equidistant_1min.csv in "
            "the same order for row-alignment to hold")

    n_total = len(gen)
    n_degenerate = 0
    n_pass_points = 0
    n_pass_both = 0
    trajectories = []

    for i in range(n_total):
        length_i = int(meta_rows[i]["length"])
        time_ms = float(meta_rows[i]["time"])
        n = min(length_i, MAX_LEN)
        if n < 1 or time_ms <= 0:
            n_degenerate += 1
            continue

        dx = gen[i, 0:n]
        dy = gen[i, 128:128 + n]
        x0 = float(meta_rows[i]["startx"])
        y0 = float(meta_rows[i]["starty"])

        xs = x0 + np.concatenate([[0.0], np.cumsum(dx)])
        ys = y0 + np.concatenate([[0.0], np.cumsum(dy)])
        n_points = n + 1
        if n_points < MIN_POINTS:
            continue
        n_pass_points += 1

        dt_step = (time_ms / 1000.0) / n  # seconds, fabricated even spacing
        ts = np.arange(n_points) * dt_step

        sx, sy = xs[0], ys[0]
        ex, ey = xs[-1], ys[-1]
        dist = math.hypot(ex - sx, ey - sy)
        if not (MIN_DISTANCE_PX <= dist <= MAX_DISTANCE_PX):
            continue
        n_pass_both += 1

        trajectories.append(list(zip(xs.tolist(), ys.tolist(), ts.tolist())))

    stats = {
        "n_total_generated_actions": n_total,
        "n_degenerate_skipped": n_degenerate,
        "n_pass_min_points_gate": n_pass_points,
        "n_pass_both_gates": n_pass_both,
        "pass_rate": n_pass_both / n_total if n_total else 0.0,
    }
    return trajectories, stats


def sample_2000(trajectories: list, seed: int = 42) -> list:
    if len(trajectories) <= 2000:
        return trajectories
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(trajectories), size=2000, replace=False)
    return [trajectories[i] for i in idx]


# ---------------------------------------------------------------------------
# SapiMouse-only human control (secondary, held-out human recordings)
# ---------------------------------------------------------------------------

def parse_sapimouse_session(path: Path) -> list:
    """Return (t_seconds, button, state, x, y) 5-tuples, the format
    setup_data.py's _segment_movements expects, from a raw SapiMouse session
    CSV (columns: client timestamp[ms], button, state, x, y)."""
    events = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return events
            for row in reader:
                if len(row) < 5:
                    continue
                ts, button, state, x, y = row[:5]
                try:
                    events.append((float(ts) / 1000.0, button, state,
                                   float(x), float(y)))
                except ValueError:
                    continue
    except OSError:
        pass
    return events


def build_sapimouse_human_pool() -> list:
    """Segment every raw SapiMouse session (1min and 3min) with the
    unmodified, imported setup_data._segment_movements - these are real
    human recordings, held out in the sense that they are parsed fresh from
    the raw session files rather than pulled from this repo's training-pool
    cache (which carries no per-source-dataset labels to filter by)."""
    all_trajectories = []
    session_files = sorted(SAPIMOUSE_RAW_DIR.rglob("*.csv"))
    for path in session_files:
        events = parse_sapimouse_session(path)
        if not events:
            continue
        all_trajectories.extend(_segment_movements(events))
    return all_trajectories


# ---------------------------------------------------------------------------
# Detector (copied verbatim from evaluate.py's RF-OOB block)
# ---------------------------------------------------------------------------

def rf_oob_auc(X, y, seed: int) -> float:
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=seed)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    return float(roc_auc_score(y, oob_proba))


def run_suite(class0: np.ndarray, class1: np.ndarray, seeds: list) -> dict:
    """class0 = label 0, class1 = label 1 (synthetic/generated = 1,
    matching evaluate.py's human=0, synthetic=1 convention)."""
    for name, arr in [("class0", class0), ("class1", class1)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} has non-finite values, aborting")
    n_use = min(len(class0), len(class1))
    c0 = class0[:n_use]
    c1 = class1[:n_use]
    X = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    return {
        "n_class0_total": int(len(class0)),
        "n_class1_total": int(len(class1)),
        "n_use_per_class": int(n_use),
        "rf_oob_auc_per_seed": {s: rf_oob_auc(X, y, s) for s in seeds},
    }


def main() -> None:
    results: dict = {}

    print("=== Loading human_eval_features.npy (headline human class) ===")
    human_eval = np.load(REPO_ROOT / "data" / "human_eval_features.npy")
    print(f"human_eval_features.npy: {human_eval.shape}")

    # --- Sanity gate: reproduce the published headline through this
    # script's own rf_oob_auc before trusting anything else it computes.
    print("\n=== SANITY GATE: human_eval vs cached synth_features_seed42 ===")
    synth42_path = EXT_DATA_DIR / "synth_features_seed42.npy"
    synth42 = np.load(synth42_path)
    sanity_X = np.vstack([human_eval[:2000], synth42[:2000]])
    sanity_y = np.concatenate([np.zeros(2000), np.ones(2000)])
    sanity_auc = rf_oob_auc(sanity_X, sanity_y, seed=42)
    delta = abs(sanity_auc - EXPECTED_HEADLINE_SEED42)
    sanity = {
        "rf_oob_auc": sanity_auc,
        "published_expected": EXPECTED_HEADLINE_SEED42,
        "delta_from_published": delta,
        "within_tolerance": delta <= SANITY_TOLERANCE,
        "tolerance": SANITY_TOLERANCE,
    }
    results["sanity_gate"] = sanity
    print(json.dumps(sanity, indent=2))
    if not sanity["within_tolerance"]:
        print("\n*** STOP: sanity gate does NOT reproduce the published "
              f"headline. Got {sanity_auc:.4f}, expected "
              f"{EXPECTED_HEADLINE_SEED42:.4f} (tolerance {SANITY_TOLERANCE}). "
              "Halting before trusting the SapiAgent comparison. ***")
        with open(EXT_DIR / "sapiagent_results.json", "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)
    print(f"Sanity OK: RF-OOB {sanity_auc:.4f} vs published "
          f"{EXPECTED_HEADLINE_SEED42:.4f} (delta {delta:.5f})")

    # --- Load and transform SapiAgent's generated actions
    print("\n=== Loading SapiAgent generated actions and applying "
          "canonical segmentation gates ===")
    print(f"SAPIAGENT_DIR = {SAPIAGENT_DIR}")
    trajectories, gate_stats = load_sapiagent_trajectories()
    print(json.dumps(gate_stats, indent=2))
    sampled = sample_2000(trajectories, seed=42)
    print(f"Using {len(sampled)} SapiAgent trajectories "
          f"({'first N in row order' if len(trajectories) <= 2000 else 'seed-42 sample of 2000'})")

    print("Extracting 18-dim features via features.extract_feature_matrix "
          "(unmodified)...")
    sapiagent_feats = extract_feature_matrix(sampled)
    print(f"Feature matrix: {sapiagent_feats.shape} "
          f"({len(sapiagent_feats)}/{len(sampled)} yielded finite features)")
    results["gate_stats"] = gate_stats
    results["n_sapiagent_features"] = int(len(sapiagent_feats))

    # --- Main comparison: SapiAgent vs our headline human_eval class
    print("\n=== MAIN: SapiAgent vs human_eval_features (18 features, "
          "RF-OOB detector-seed sweep) ===")
    main_result = run_suite(human_eval, sapiagent_feats, DETECTOR_SEEDS)
    results["main_18feat"] = main_result
    print(json.dumps(main_result, indent=2))

    # --- Secondary: timing-free feature subset only
    print("\n=== SECONDARY: timing-free feature subset "
          f"({TIMING_FREE_FEATURES}) ===")
    human_eval_tf = human_eval[:, TIMING_FREE_IDX]
    sapiagent_tf = sapiagent_feats[:, TIMING_FREE_IDX]
    timing_free_result = run_suite(human_eval_tf, sapiagent_tf, DETECTOR_SEEDS)
    timing_free_result["features_used"] = TIMING_FREE_FEATURES
    results["timing_free_subset"] = timing_free_result
    print(json.dumps(timing_free_result, indent=2))

    # --- Secondary: SapiMouse-only held-out human control
    print("\n=== SECONDARY: SapiMouse-only human control ===")
    if not SAPIMOUSE_RAW_DIR.exists():
        print(f"SKIPPED: {SAPIMOUSE_RAW_DIR} not found (raw SapiMouse "
              "session data not available at scoring time)")
        results["sapimouse_only_control"] = {"skipped": True,
                                              "reason": "raw sapimouse dir not found"}
    else:
        sapimouse_traj = build_sapimouse_human_pool()
        print(f"Segmented {len(sapimouse_traj)} valid human movements from "
              f"raw SapiMouse sessions (canonical setup_data.py rule)")
        sapimouse_feats_all = extract_feature_matrix(sapimouse_traj)
        finite_mask = np.all(np.isfinite(sapimouse_feats_all), axis=1)
        sapimouse_feats_all = sapimouse_feats_all[finite_mask]
        print(f"Feature matrix: {sapimouse_feats_all.shape}")
        if len(sapimouse_feats_all) < 2000:
            print(f"SKIPPED: only {len(sapimouse_feats_all)} valid SapiMouse "
                  "human feature rows, need 2000")
            results["sapimouse_only_control"] = {
                "skipped": True,
                "reason": f"only {len(sapimouse_feats_all)} valid rows",
            }
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(sapimouse_feats_all), size=2000, replace=False)
            sapimouse_2000 = sapimouse_feats_all[idx]

            print("-- dataset shift: SapiMouse-humans vs our human_eval "
                  "(humans vs humans) --")
            shift_result = run_suite(sapimouse_2000, human_eval, DETECTOR_SEEDS)
            print(json.dumps(shift_result, indent=2))

            print("-- SapiAgent vs SapiMouse-only humans (main, secondary "
                  "human reference) --")
            sapimouse_main = run_suite(sapimouse_2000, sapiagent_feats, DETECTOR_SEEDS)
            print(json.dumps(sapimouse_main, indent=2))

            results["sapimouse_only_control"] = {
                "skipped": False,
                "n_segmented_movements": len(sapimouse_traj),
                "n_valid_feature_rows": int(len(sapimouse_feats_all)),
                "dataset_shift_vs_human_eval": shift_result,
                "sapiagent_vs_sapimouse_humans": sapimouse_main,
            }

    out_path = EXT_DIR / "sapiagent_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n=== SUMMARY ===")
    print(f"Sanity: {sanity_auc:.4f} (published {EXPECTED_HEADLINE_SEED42:.4f})")
    print(f"Gate pass rate: {gate_stats['pass_rate']:.1%} "
          f"({gate_stats['n_pass_both_gates']}/{gate_stats['n_total_generated_actions']})")
    seed42_main = main_result["rf_oob_auc_per_seed"][42]
    seed42_tf = timing_free_result["rf_oob_auc_per_seed"][42]
    print(f"Main (18-feat, seed 42): RF-OOB {seed42_main:.4f}")
    print(f"Timing-free subset (seed 42): RF-OOB {seed42_tf:.4f}")
    print(f"Our headline (selected synthetics vs human_eval, seed 42): "
          f"{EXPECTED_HEADLINE_SEED42:.4f}")


if __name__ == "__main__":
    main()
