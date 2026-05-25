"""Generate updated figures for README."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def fig_auc_progression():
    """Bar chart of AUC by architecture family."""
    labels = [
        "Parametric\n(sigma-log)",
        "CFM",
        "Stall\ninjection",
        "VQ-VAE +\nTransformer",
        "DDPM",
        "ZIMT\n(magcorr)",
        "Corpus\nrotate",
        "Corpus\nreplay",
    ]
    aucs = [0.998, 0.919, 0.93, 0.890, 0.862, 0.864, 0.686, 0.52]
    is_generative = [True, True, True, True, True, True, False, False]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#4040E0" if g else "#40D8D8" for g in is_generative]
    bars = ax.bar(labels, aucs, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, auc in zip(bars, aucs):
        label = f"{auc:.3f}" if auc < 1 else f"~{auc:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=0.75, color="#E04040", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(labels) - 0.5, 0.76, "Target (0.75)", color="#E04040",
            fontsize=9, ha="right", fontstyle="italic")
    ax.axhline(y=0.50, color="#E04040", linestyle="--", linewidth=1.5, alpha=0.4)
    ax.text(len(labels) - 0.5, 0.51, "Theoretical min (0.50)", color="#E04040",
            fontsize=9, ha="right", fontstyle="italic", alpha=0.6)

    gen_patch = plt.Rectangle((0, 0), 1, 1, fc="#4040E0")
    rep_patch = plt.Rectangle((0, 0), 1, 1, fc="#40D8D8")
    ax.legend([gen_patch, rep_patch], ["Generative", "Replay-based"],
              loc="upper right", fontsize=10)

    ax.set_ylabel("AUC (lower = more human-like)", fontsize=12)
    ax.set_title("AUC by Architecture Family (lower = more human-like)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "auc_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_trajectory_overlay():
    """Overlay of human, ZIMT, and corpus_rotate trajectories for the same query."""
    from experiments._common import Trajectory
    from experiments.zimt_magcorr import generate_path as zimt_generate
    from experiments.corpus_rotate import generate_path as rotate_generate

    np.random.seed(42)
    human_feats = np.load("data/human_eval_features.npy")
    distances = np.load("data/human_distances.npy")

    start_x, start_y = 150.0, 640.0
    end_x, end_y = 350.0, 950.0

    zimt_traj = zimt_generate(start_x, start_y, end_x, end_y)
    rotate_traj = rotate_generate(start_x, start_y, end_x, end_y)

    zx = [p[0] for p in zimt_traj]
    zy = [p[1] for p in zimt_traj]
    rx = [p[0] for p in rotate_traj]
    ry = [p[1] for p in rotate_traj]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rx, ry, color="#40D8D8", linewidth=2.0, label="Corpus Rotate", alpha=0.85)
    ax.plot(zx, zy, color="#E08040", linewidth=2.0, label="ZIMT (magcorr)", alpha=0.85)

    ax.plot(start_x, start_y, "ko", markersize=10, zorder=5, label="Start")
    ax.plot(end_x, end_y, "kX", markersize=12, zorder=5, label="End")

    ax.set_xlabel("x (px)", fontsize=12)
    ax.set_ylabel("y (px)", fontsize=12)
    ax.set_title("Generated Trajectories", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "trajectory_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_feature_distributions():
    """Violin plots comparing human vs ZIMT vs corpus_rotate feature distributions."""
    from features import extract_features, FEATURE_NAMES
    from experiments.zimt_magcorr import generate_path as zimt_generate
    from experiments.corpus_rotate import generate_path as rotate_generate

    human_feats = np.load("data/human_eval_features.npy")
    distances = np.load("data/human_distances.npy")

    n_samples = 500
    rng = np.random.default_rng(42)

    display_features = [
        ("mean_velocity", "Mean Velocity"),
        ("curvature_mean", "Curvature Mean"),
        ("angular_velocity_mean", "Angular Vel Mean"),
        ("path_efficiency", "Path Efficiency"),
        ("num_direction_changes", "Num Dir Changes"),
    ]
    feat_indices = [FEATURE_NAMES.index(f[0]) for f in display_features]

    def _generate_features(gen_fn, n):
        feats = []
        for i in range(n):
            idx = rng.integers(0, len(distances))
            dist = distances[idx]
            angle = rng.uniform(0, 2 * np.pi)
            sx, sy = 500.0, 500.0
            ex = sx + dist * np.cos(angle)
            ey = sy + dist * np.sin(angle)
            traj = gen_fn(sx, sy, ex, ey)
            f = extract_features(traj)
            if f is not None:
                feats.append(f)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n}")
        return np.array(feats)

    print(f"Generating {n_samples} ZIMT trajectories...")
    zimt_feats = _generate_features(zimt_generate, n_samples)
    print(f"Generating {n_samples} corpus_rotate trajectories...")
    rotate_feats = _generate_features(rotate_generate, n_samples)

    fig, axes = plt.subplots(1, len(display_features), figsize=(18, 4))

    for ax_i, (feat_name, display_name) in enumerate(display_features):
        fi = feat_indices[ax_i]
        ax = axes[ax_i]

        h_vals = human_feats[:, fi]
        z_vals = zimt_feats[:, fi]
        r_vals = rotate_feats[:, fi]

        all_vals = np.concatenate([h_vals, z_vals, r_vals])
        p95 = np.percentile(all_vals, 95)
        p5 = np.percentile(all_vals, 5)
        h_clip = np.clip(h_vals, p5, p95)
        z_clip = np.clip(z_vals, p5, p95)
        r_clip = np.clip(r_vals, p5, p95)

        data = [h_clip, z_clip, r_clip]
        positions = [1, 2, 3]
        colors = ["#4488CC", "#E08040", "#40D8D8"]

        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Human", "ZIMT", "Rotate"], fontsize=8)
        ax.set_title(display_name, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Feature Distribution Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["auc", "trajectory", "features", "all"],
                        default="all")
    args = parser.parse_args()

    if args.figure in ("auc", "all"):
        fig_auc_progression()
    if args.figure in ("trajectory", "all"):
        fig_trajectory_overlay()
    if args.figure in ("features", "all"):
        fig_feature_distributions()
