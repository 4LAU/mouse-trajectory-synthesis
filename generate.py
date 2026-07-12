"""Generate a human-like mouse trajectory between two screen points.

This is the front door to the trained MIME model. Give it a start point and
an end point in pixels; it returns a trajectory that begins exactly at the
start, ends exactly at the end, and moves like a person in between.

Usage:
    python generate.py 200 600 1500 300
    python generate.py 200 600 1500 300 --n 5 --seed 7 --format csv
    python generate.py 200 600 1500 300 --plot out.png

Output (default JSON, one object per point, the shape a replay client wants):
    [{"x": 200, "y": 600, "delay": 0}, {"x": 235, "y": 602, "delay": 12.2}, ...]

`delay` is the wait in milliseconds before moving to that point. Feed the
points to any input-automation layer as-is.

From Python:
    from generate import generate
    traj = generate(200, 600, 1500, 300)          # one (m, 3) array of x, y, t_seconds
    trajs = generate(200, 600, 1500, 300, n=5)    # list of five

Needs `training/event_polar_4m_fc_v2.pt` and `training/train_conditions.npy`;
`python setup_data.py` downloads both. Runs on CPU in about a second per
trajectory.

The raw model output heads the right way and covers roughly the right
distance, but lands near the target rather than on it (the detectors it was
evaluated against score how movement looks, not where it stops). The final
landing is therefore a small rotate-and-scale of the whole path around the
start point, typically a few percent, which leaves its character intact.
Pass land=False (or --no-land) to see the raw output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Single-shot sampling recipe. This is the STABLE decoder (confidence order),
# not the high-diversity gumbel recipe the README uses for step 4. That recipe
# deliberately scatters samples wide because a downstream selection step then
# cherry-picks the good ones out of a large pool; run one-shot with no
# selection, it heads the wrong way about half the time. Confidence order
# heads toward the target reliably, which is what a caller of this script
# wants. setdefault so the environment can still override any knob.
_RECIPE = {
    "EVENT_CKPT": "event_polar_4m_fc_v2.pt",
    "EVENT_ORDER": "conf",
    "EVENT_CHOICE_TEMP": "0",
    "EVENT_SNAP": "2.5",
    "EVENT_DUR_STD": "1.0",
    "DUR_EMPIRICAL": "1",
}
for _k, _v in _RECIPE.items():
    os.environ.setdefault(_k, _v)

# How many candidates to draw per requested trajectory. The model naturally
# lands 10-30% off the target; we keep the candidate whose raw endpoint is
# closest, so the landing correction stays small and the path's character is
# barely touched. Override with MIME_OVERSAMPLE=1 to disable.
_OVERSAMPLE = max(1, int(os.environ.get("MIME_OVERSAMPLE", "6")))

_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))


def _check_assets() -> None:
    missing = [
        p.name
        for p in (_TRAIN_DIR / os.environ["EVENT_CKPT"], _TRAIN_DIR / "train_conditions.npy")
        if not p.exists()
    ]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)} in {_TRAIN_DIR}/.\n"
            "Run `python setup_data.py` first to download the model and its data."
        )


def _land_on_target(traj: np.ndarray, ex: float, ey: float) -> np.ndarray:
    """Rotate and scale a path around its start so its last point is (ex, ey).

    A similarity transform: multiply each point's offset from the start by the
    complex ratio (desired end vector) / (raw end vector). Preserves every
    turn angle and every relative timing; only the overall heading and length
    change, both by whatever small amount the raw path missed by.
    """
    start = traj[0, :2]
    raw_vec = complex(*(traj[-1, :2] - start))
    if abs(raw_vec) < 1e-9:
        return traj
    ratio = complex(ex - start[0], ey - start[1]) / raw_vec
    offsets = (traj[:, 0] - start[0]) + 1j * (traj[:, 1] - start[1])
    moved = offsets * ratio
    out = traj.copy()
    out[:, 0] = start[0] + moved.real
    out[:, 1] = start[1] + moved.imag
    out[-1, :2] = [ex, ey]  # kill floating-point residue on the endpoint
    return out


def generate(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    *,
    n: int = 1,
    seed: int | None = None,
    land: bool = True,
):
    """Generate trajectories from (start_x, start_y) to (end_x, end_y).

    Returns one (m, 3) float array of columns [x, y, t_seconds] when n == 1,
    or a list of n such arrays. With land=True (default) each path ends
    exactly on the target; with land=False you get the raw model output.
    """
    _check_assets()
    if seed is not None:
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

    # Imported here, not at module top, so that --help and asset checks do not
    # pay the cost of loading torch and the checkpoint. The module prints a
    # one-line config banner to stdout at import; send it to stderr so it never
    # lands in the JSON/CSV a caller is parsing from stdout.
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        from experiments.event_stream_polar import generate_paths

    # Draw _OVERSAMPLE candidates per requested output in one batched call.
    k = _OVERSAMPLE
    specs = [(start_x, start_y, end_x, end_y)] * (n * k)
    raw = generate_paths(specs)

    target = np.array([end_x, end_y])
    out = []
    for g in range(n):
        best, best_miss = None, np.inf
        for t in raw[g * k : (g + 1) * k]:
            arr = np.asarray(t, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            miss = float(np.hypot(*(arr[-1, :2] - target)))
            if miss < best_miss:
                best, best_miss = arr, miss
        if best is None:
            continue
        out.append(_land_on_target(best, end_x, end_y) if land else best)

    if not out:
        raise RuntimeError("model returned no usable trajectory; try a different seed")
    return out[0] if n == 1 else out


def _as_points(traj: np.ndarray) -> list[dict]:
    """Convert an (m, 3) [x, y, t_seconds] array to per-point delay records."""
    t_ms = traj[:, 2] * 1000.0
    delays = np.diff(t_ms, prepend=t_ms[0])
    return [
        {"x": round(float(x), 2), "y": round(float(y), 2), "delay": round(float(d), 2)}
        for x, y, d in zip(traj[:, 0], traj[:, 1], delays)
    ]


def _print_csv(trajs: list[np.ndarray]) -> None:
    print("traj,x,y,delay_ms")
    for i, traj in enumerate(trajs):
        for p in _as_points(traj):
            print(f"{i},{p['x']},{p['y']},{p['delay']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a human-like mouse trajectory between two points.",
    )
    parser.add_argument("start_x", type=float)
    parser.add_argument("start_y", type=float)
    parser.add_argument("end_x", type=float)
    parser.add_argument("end_y", type=float)
    parser.add_argument("--n", type=int, default=1, help="How many to generate (default 1)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Output format (default json)"
    )
    parser.add_argument(
        "--no-land",
        action="store_true",
        help="Return raw model output instead of landing exactly on the target",
    )
    parser.add_argument("--plot", metavar="PATH", help="Also save a PNG picture of the paths")
    args = parser.parse_args(argv)

    result = generate(
        args.start_x, args.start_y, args.end_x, args.end_y,
        n=args.n, seed=args.seed, land=not args.no_land,
    )
    trajs = result if isinstance(result, list) else [result]

    if args.format == "csv":
        _print_csv(trajs)
    else:
        payload = [_as_points(t) for t in trajs]
        print(json.dumps(payload[0] if args.n == 1 else payload))

    if args.plot:
        _save_plot(trajs, args.start_x, args.start_y, args.end_x, args.end_y, args.plot)
        print(f"Saved {args.plot}", file=sys.stderr)


def _save_plot(trajs, sx, sy, ex, ey, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#4040E0", lw=1.0, alpha=0.5)
        ax.plot(traj[:, 0], traj[:, 1], ".", color="#4040E0", ms=3)
    ax.plot([sx], [sy], "o", color="#2CB25C", ms=10, label="start")
    ax.plot([ex], [ey], "X", color="#E04040", ms=11, label="end")
    ax.invert_yaxis()  # screen coordinates: y grows downward
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.set_title(f"({sx:.0f}, {sy:.0f}) to ({ex:.0f}, {ey:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
