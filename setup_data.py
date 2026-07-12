"""Download model checkpoints and evaluation data for MIME-mouse.

Default mode downloads release assets from GitHub into ./data/.
Build mode (--build-demo-pool) downloads the Balabit Mouse Dynamics Challenge
dataset, segments it into individual mouse movements, and saves a demo pool.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEASE_BASE = (
    "https://github.com/4LAU/MIME-mouse/releases/latest/download/"
)

RELEASE_ASSETS = [
    "ddpm_best.pt",
    "cfm_best.pt",
    "vqvae_best.pt",
    "trajectory_transformer_best.pt",
    "human_eval_features.npy",
    "human_distances.npy",
    "train_conditions.npy",
]

OPTIONAL_ASSETS = [
    "demo_pool.npz",
]

# Everything needed to reproduce the headline 0.504 result. The cached
# candidate pools and winning picks land in the repo root (where the replay
# commands expect them); the event-stream checkpoint lands in training/ so
# the sampler can regenerate pools from scratch. Replaying the cached pools
# needs no GPU and never loads the checkpoint.
REPRO_ASSETS_ROOT = [
    "pool_s42_k16.npz",
    "pool_s43_k16.npz",
    "pool_s44_k16.npz",
    "pool_s42_k16_picks_trust33_f20d85_r30_rf.npy",
    "pool_s43_k16_picks_trust33_f20d85_r30_rf.npy",
    "pool_s44_k16_picks_trust33_f20d85_r30_rf.npy",
]

REPRO_ASSETS_TRAINING = [
    "event_polar_4m_fc_v2.pt",
]

# Balabit dataset
BALABIT_REPO = "https://github.com/balabit/Mouse-Dynamics-Challenge.git"
PAUSE_THRESHOLD_S = 0.200  # seconds - split movements on pauses longer than this
MIN_POINTS = 5
MIN_DISTANCE_PX = 20.0
MAX_DISTANCE_PX = 5000.0
DEMO_POOL_SIZE = 50_000
DEMO_POOL_SEED = 42

Trajectory = List[Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, *, force: bool = False) -> bool:
    """Download a single file with progress indication.

    Returns True on success, False on failure.  Skips if *dest* exists and
    *force* is False.
    """
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already exists")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "setup_data/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                last_print = 0.0

                with open(tmp_path, "wb") as fp:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        fp.write(chunk)
                        downloaded += len(chunk)

                        now = time.monotonic()
                        if now - last_print > 1.0 or not chunk:
                            if total:
                                pct = downloaded / total * 100
                                mb = downloaded / (1024 * 1024)
                                print(
                                    f"  [download] {dest.name}: "
                                    f"{mb:.1f} MB ({pct:.0f}%)",
                                    end="\r",
                                )
                            else:
                                mb = downloaded / (1024 * 1024)
                                print(
                                    f"  [download] {dest.name}: {mb:.1f} MB",
                                    end="\r",
                                )
                            last_print = now

            print()  # newline after progress
            tmp_path.rename(dest)
            return True

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 0:
                print(f"\n  [retry] {dest.name}: {exc}")
            else:
                print(f"\n  [FAILED] {dest.name}: {exc}")
                return False

    return False


# ---------------------------------------------------------------------------
# Default mode: download release assets
# ---------------------------------------------------------------------------


def download_assets(data_dir: Path, *, force: bool = False) -> None:
    """Download model checkpoints and evaluation data from the GitHub release."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading release assets into {data_dir}/\n")

    failures: list[str] = []

    for name in RELEASE_ASSETS:
        url = RELEASE_BASE + name
        ok = _download_file(url, data_dir / name, force=force)
        if not ok:
            failures.append(name)

    # Optional assets - don't count as failures
    for name in OPTIONAL_ASSETS:
        url = RELEASE_BASE + name
        ok = _download_file(url, data_dir / name, force=force)
        if not ok and not (data_dir / name).exists():
            print(
                f"\n  {name} is not available in the release."
                f"\n  To build it from the Balabit dataset, run:"
                f"\n    python setup_data.py --build-demo-pool\n"
            )

    # Reproduce bundle for the 0.504 headline (cached pools + picks +
    # event-stream checkpoint). See README "Reproduce the current results".
    print("\nDownloading the 0.504 reproduce bundle...\n")
    for name in REPRO_ASSETS_ROOT:
        if not _download_file(RELEASE_BASE + name, Path(name), force=force):
            failures.append(name)
    for name in REPRO_ASSETS_TRAINING:
        dest = Path("training") / name
        if not _download_file(RELEASE_BASE + name, dest, force=force):
            failures.append(name)

    if failures:
        print(f"\nFailed to download: {', '.join(failures)}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    print("\nAll required assets downloaded successfully.")


# ---------------------------------------------------------------------------
# Build mode: construct demo_pool.npz from Balabit dataset
# ---------------------------------------------------------------------------


def _parse_balabit_file(path: Path) -> list:
    """Parse a Balabit session CSV into a list of (t, button, state, x, y)."""
    events = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0] == "record timestamp":
                    continue
                try:
                    _, client_ts, button, state, x, y = row[:6]
                    events.append((float(client_ts), button, state, float(x), float(y)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass
    return events


def _segment_movements(events: list) -> List[Trajectory]:
    """Segment raw mouse events into individual point-to-point movements."""
    trajectories: List[Trajectory] = []
    current: Trajectory = []

    def maybe_commit(points: Trajectory) -> None:
        if len(points) < MIN_POINTS:
            return
        sx, sy, _ = points[0]
        ex, ey, _ = points[-1]
        dist = math.hypot(ex - sx, ey - sy)
        if MIN_DISTANCE_PX <= dist <= MAX_DISTANCE_PX:
            trajectories.append(list(points))

    for t, _button, state, x, y in events:
        if state in ("Pressed", "Released"):
            maybe_commit(current)
            current = []
            continue

        if state != "Move":
            continue

        if current and (t - current[-1][2]) > PAUSE_THRESHOLD_S:
            maybe_commit(current)
            current = []

        current.append((x, y, t))

    maybe_commit(current)
    return trajectories


def _load_all_balabit(root: Path) -> List[Trajectory]:
    """Load and segment all session files under a Balabit clone."""
    trajectories: List[Trajectory] = []
    session_files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not p.suffix
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    print(f"  Found {len(session_files)} session files")
    for i, path in enumerate(session_files):
        trajs = _segment_movements(_parse_balabit_file(path))
        trajectories.extend(trajs)
        if (i + 1) % 50 == 0 or i == len(session_files) - 1:
            print(
                f"  Parsed {i + 1}/{len(session_files)} files, "
                f"{len(trajectories)} movements so far",
                end="\r",
            )
    print()
    return trajectories


def build_demo_pool(data_dir: Path, *, force: bool = False) -> None:
    """Download Balabit dataset, segment movements, save demo_pool.npz."""
    out_path = data_dir / "demo_pool.npz"
    if out_path.exists() and not force:
        print(f"  {out_path} already exists (use --force to rebuild)")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # Clone Balabit dataset into a temp directory
    clone_dir = data_dir / "_balabit_clone"
    if not clone_dir.exists():
        print("Cloning Balabit Mouse Dynamics Challenge dataset...")
        subprocess.run(
            ["git", "clone", "--depth", "1", BALABIT_REPO, str(clone_dir)],
            check=True,
        )

    training_dir = clone_dir / "training_files"
    if not training_dir.exists():
        print("ERROR: Clone appears incomplete (missing training_files/).")
        print(f"Delete {clone_dir} and try again.")
        sys.exit(1)

    # Parse and segment
    print("Parsing and segmenting trajectories...")
    all_trajs = _load_all_balabit(clone_dir)
    print(f"Total segmented movements: {len(all_trajs)}")

    if len(all_trajs) < DEMO_POOL_SIZE:
        print(
            f"WARNING: Only {len(all_trajs)} movements found "
            f"(requested {DEMO_POOL_SIZE}). Using all."
        )
        sample = all_trajs
    else:
        rng = np.random.default_rng(DEMO_POOL_SEED)
        indices = rng.choice(len(all_trajs), size=DEMO_POOL_SIZE, replace=False)
        sample = [all_trajs[i] for i in indices]

    print(f"Selected {len(sample)} trajectories for demo pool")

    # Build flat arrays
    flat_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    meta_rows: list[np.ndarray] = []
    offsets = [0]

    for traj in sample:
        pts = np.asarray(traj, dtype=np.float64)
        xy = pts[:, :2]
        t = pts[:, 2]

        # Compute metadata
        dx = xy[-1, 0] - xy[0, 0]
        dy = xy[-1, 1] - xy[0, 1]
        dist = math.hypot(dx, dy)
        log_dist = float(np.log(max(dist, 1e-8)))
        angle = math.atan2(dy, dx)

        flat_parts.append(xy)
        # Store time relative to trajectory start
        t_parts.append(t - t[0])
        meta_rows.append(np.array([log_dist, math.cos(angle), math.sin(angle)]))
        offsets.append(offsets[-1] + len(pts))

    flat = np.concatenate(flat_parts, axis=0).astype(np.float32)
    t_arr = np.concatenate(t_parts, axis=0).astype(np.float32)
    meta = np.stack(meta_rows).astype(np.float32)
    off_arr = np.array(offsets, dtype=np.int64)

    np.savez_compressed(
        out_path,
        flat=flat,
        offsets=off_arr,
        meta=meta,
        t=t_arr,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {out_path} ({size_mb:.1f} MB)")
    print(f"  flat:    {flat.shape}")
    print(f"  offsets: {off_arr.shape}")
    print(f"  meta:    {meta.shape}  [log_dist, cos_angle, sin_angle]")
    print(f"  t:       {t_arr.shape}")

    # Clean up the clone
    print(f"\nCleaning up {clone_dir}...")
    shutil.rmtree(clone_dir, ignore_errors=True)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model checkpoints and evaluation data, "
        "or build a demo pool from the Balabit dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Target directory for downloaded files (default: ./data)",
    )
    parser.add_argument(
        "--build-demo-pool",
        action="store_true",
        help="Build demo_pool.npz from the Balabit Mouse Dynamics Challenge dataset "
        "instead of downloading release assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / rebuild files even if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)

    if args.build_demo_pool:
        build_demo_pool(data_dir, force=args.force)
    else:
        download_assets(data_dir, force=args.force)


if __name__ == "__main__":
    main()
