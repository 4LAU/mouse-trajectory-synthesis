"""Generate updated figures for README."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Headline result for the event-stream family: the confirmed three-seed
# n=2000 RF OOB AUC of the set-level reselection recipe (33-feature RF judge).
# Three-seed mean 0.504 (0.5095 / 0.5030 / 0.4993), chance level on the
# primary detector. The 18-feature judge lands at 0.491 across the same seeds.
EVENT_STREAM_AUC = 0.504


def fig_auc_progression():
    """Bar chart of AUC by architecture family."""
    labels = [
        "Parametric\n(sigma-log)",
        "CFM",
        "Stall\ninjection",
        "VQ-VAE +\nTransformer",
        "DDPM",
        "ZIMT\n(magcorr)",
        "Event-stream\n(polar)",
        "Corpus\nrotate",
        "Corpus\nreplay",
    ]
    aucs = [0.998, 0.919, 0.93, 0.890, 0.862, 0.864, EVENT_STREAM_AUC, 0.686, 0.52]
    is_generative = [True, True, True, True, True, True, True, False, False]

    fig, ax = plt.subplots(figsize=(15, 5))

    # Highlight the event-stream bar as the result the whole project builds toward.
    colors = []
    for i, g in enumerate(is_generative):
        if labels[i].startswith("Event-stream"):
            colors.append("#2CB25C")
        elif g:
            colors.append("#4040E0")
        else:
            colors.append("#40D8D8")
    bars = ax.bar(labels, aucs, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, auc in zip(bars, aucs):
        label = f"{auc:.3f}" if auc < 1 else f"~{auc:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=0.50, color="#E04040", linestyle="--", linewidth=1.6, alpha=0.85)
    ax.text(len(labels) - 0.5, 0.515, "Indistinguishable from human (0.50)",
            color="#E04040", fontsize=9.5, ha="right", fontstyle="italic")

    gen_patch = plt.Rectangle((0, 0), 1, 1, fc="#4040E0")
    win_patch = plt.Rectangle((0, 0), 1, 1, fc="#2CB25C")
    rep_patch = plt.Rectangle((0, 0), 1, 1, fc="#40D8D8")
    ax.legend([win_patch, gen_patch, rep_patch],
              ["Event-stream (this work)", "Earlier generative", "Replay-based"],
              loc="upper right", fontsize=10)

    ax.set_ylabel("Detector AUC (lower is more human-like)", fontsize=12)
    ax.set_title("Detector AUC by architecture family (lower is more human-like)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "auc_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# Record-progression milestones: best detector AUC achieved to date, in
# chronological order. Dates come from EXPERIMENTS.md headings (or, for the
# three earliest entries, the "Initial release" commit date, since the
# journal did not start dating individual entries until 2026-05-13 and these
# three predate that). Only points that actually LOWER the running-best AUC
# are included:
#   - "## VQ-VAE + Transformer (v145-v146)" (undated, part of the initial
#     2026-05-03 log): retrained VQ-VAE reaches 0.890, explicitly "still
#     worse than DDPM baseline (0.86)" (EXPERIMENTS.md line ~404) -> not a
#     record, excluded.
#   - "## ZIMT Endpoint Correction & Guided Sampling (2026-05-13)": ZIMT
#     magcorr reaches 0.864, added after DDPM's 0.862 was already the
#     standing record -> not a record, excluded.
AUC_TIMELINE = [
    ("2026-05-03", 0.998, "Sigma-lognormal baseline"),
    # "## Generative Research: CFM and Baselines" (undated; initial log).
    ("2026-05-03", 0.919, "CFM flow matching"),
    # "## Stall Injection Experiments (v143)" / VQ-VAE summary table both
    # cite DDPM v135 eta=0 as "Best baseline" at 0.862 (undated; initial log).
    ("2026-05-03", 0.862, "DDPM diffusion"),
    # "## Updated Scoreboard (2026-05-27)": CANDI polar DDIM (30ep, CFG=0)
    # at 0.852, the first generative result under the DDPM 0.862 record.
    ("2026-05-27", 0.852, "CANDI polar diffusion"),
    # "## Pure model + empirical duration prior: confirmed at 3 seeds
    # (July 6, 13:25)": event-stream model, no selection, 0.652 +/- 0.003.
    ("2026-07-06 13:25", 0.652, "Event-stream (pure)"),
    # "## New honest best: 0.568 +/- 0.010 at three seeds (July 6, 19:40)".
    ("2026-07-06 19:40", 0.568, "+ SIR selection"),
    # "## Protected three-seed confirmation, both judge widths
    # (July 6, 23:47)": headline decision, 33-dim judge, set-level
    # reselection, 0.504 three-seed confirmed.
    ("2026-07-06 23:47", 0.504, "Set-level reselection"),
]


def fig_auc_timeline():
    """Record-progression chart: best detector AUC achieved to date."""
    dates = [datetime.strptime(d, "%Y-%m-%d %H:%M") if " " in d
             else datetime.strptime(d, "%Y-%m-%d")
             for d, _, _ in AUC_TIMELINE]
    aucs = [a for _, a, _ in AUC_TIMELINE]
    labels = [l for _, _, l in AUC_TIMELINE]

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_facecolor("white")

    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.7, zorder=0)
    ax.step(dates, aucs, where="post", color="#4040E0", linewidth=2.2,
            zorder=3)
    ax.scatter(dates[:-1], aucs[:-1], color="#4040E0", s=45, zorder=4,
               edgecolor="white", linewidth=0.8)
    ax.scatter([dates[-1]], [aucs[-1]], color="#2CB25C", s=110, zorder=5,
               edgecolor="white", linewidth=1.0)

    ax.axhline(y=0.50, color="#E04040", linestyle="--", linewidth=1.6,
               alpha=0.85, zorder=2)
    ax.text(dates[0], 0.515,
            "Chance (0.50): detector cannot tell synthetic from human",
            color="#E04040", fontsize=9.5, ha="left", fontstyle="italic")

    # Manual per-point offsets (in points) to keep annotations legible and
    # collision-free given the two dense clusters (May 3 and July 6).
    offsets = [
        (10, -16),   # Sigma-lognormal baseline (below, clear of the title)
        (10, 12),    # CFM flow matching
        (10, -20),   # DDPM diffusion
        (10, -18),   # CANDI polar diffusion
        (-100, 22),  # Event-stream (pure)
        (-100, -22), # + SIR selection
    ]
    for i, (d, auc, label) in enumerate(AUC_TIMELINE[:-1]):
        dx, dy = offsets[i]
        ax.annotate(f"{label} {auc:.3f}", xy=(dates[i], aucs[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, color="#333333",
                    ha="left" if dx >= 0 else "right")

    # Final point: placed well clear of the axis and the two preceding
    # labels, in the open space above the July cluster.
    ax.annotate("Set-level selection 0.504 (tuning) /\n0.513 (out-of-sample)",
                xy=(dates[-1], aucs[-1]), xytext=(-70, 175),
                textcoords="offset points", fontsize=9.5, color="#333333",
                ha="right", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#999999",
                                 linewidth=0.8, shrinkA=2, shrinkB=6))

    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Detector AUC (lower is more human-like)", fontsize=11.5)
    ax.set_title("Best detector AUC over the project (lower is more human-like)",
                 fontsize=14, fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "auc_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


POOL_FILE = "pool_s42_k16.npz"
PICKS_FILE = "pool_s42_k16_picks_trust33_f20d85_r30_rf.npy"


def _load_selected_pool():
    """Selected set for seed 42: the exact trajectories behind the 0.504 result.

    Returns (trajs, feats): one raw trajectory and one 18-feature row per spec,
    taken from the cached candidate pool via the winning trust-loop picks.
    No GPU work; the pool was generated once and cached.
    """
    d = np.load(POOL_FILE, allow_pickle=True)
    picks = np.load(PICKS_FILE).astype(int)
    picks = picks[picks >= 0]
    trajs = [np.asarray(d["trajs"][ci], dtype=np.float64) for ci in picks]
    feats = d["X"][picks]
    return trajs, feats


def fig_trajectory_overlay():
    """Held-out human paths next to generated paths of matched lengths."""
    from features import FEATURE_NAMES

    gen_trajs, gen_feats = _load_selected_pool()
    eff_i = FEATURE_NAMES.index("path_efficiency")

    positions = np.load("training/test_positions.npy", mmap_mode="r")
    n_real = np.load("training/test_n_real.npy")
    conditions = np.load("training/test_conditions.npy", mmap_mode="r")

    def gen_distance(traj):
        return float(np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1]))

    # Four generated paths spanning short to long movements. Within each
    # distance band take the path of median directness, so the examples are
    # typical of the selected set rather than tail cases.
    dists = np.array([gen_distance(t) for t in gen_trajs])
    lens = np.array([len(t) for t in gen_trajs])
    chosen_gen = []
    for q in (25, 50, 75, 92):
        target = np.percentile(dists, q)
        band = np.where((np.abs(dists - target) < 0.12 * target)
                        & (lens >= 15))[0]
        band = np.array([i for i in band if i not in chosen_gen])
        effs = gen_feats[band, eff_i]
        idx = int(band[np.argsort(effs)[len(effs) // 2]])
        chosen_gen.append(idx)

    # Match each generated path with a held-out human path of similar
    # distance and typical directness.
    rng = np.random.default_rng(7)
    hum_pool = rng.choice(len(n_real), size=20000, replace=False)
    hum_pool = hum_pool[n_real[hum_pool] >= 40]
    hum_dist = np.exp(np.asarray(conditions[hum_pool, 0], dtype=np.float64))
    chosen_hum = []
    for gi in chosen_gen:
        cand = hum_pool[np.abs(hum_dist - dists[gi]) < 0.12 * dists[gi]]
        cand = np.array([h for h in cand if h not in chosen_hum])
        effs = []
        for h in cand[:200]:
            L = int(n_real[h])
            p = np.asarray(positions[h, :L], dtype=np.float64)
            seg = np.hypot(*np.diff(p, axis=0).T).sum()
            effs.append(1.0 / max(seg, 1e-9))
        chosen_hum.append(int(cand[:200][np.argsort(effs)[len(effs) // 2]]))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for col in range(4):
        hi = chosen_hum[col]
        L = int(n_real[hi])
        scale = float(np.exp(conditions[hi, 0]))
        hp = np.asarray(positions[hi, :L], dtype=np.float64) * scale

        gi = chosen_gen[col]
        gp = gen_trajs[gi][:, :2] - gen_trajs[gi][0, :2]

        for row, (p, color, name) in enumerate(
                [(hp, "#4488CC", "Human (held out)"),
                 (gp, "#2CB25C", "Generated (selected)")]):
            ax = axes[row, col]
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.6, alpha=0.9)
            ax.plot(p[0, 0], p[0, 1], "ko", markersize=6, zorder=5)
            ax.plot(p[-1, 0], p[-1, 1], "kX", markersize=8, zorder=5)
            d = np.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1])
            ax.set_title(f"{name}, {d:.0f} px", fontsize=10)
            # Square, centered limits so every panel gets the same box.
            cx = (p[:, 0].min() + p[:, 0].max()) / 2
            cy = (p[:, 1].min() + p[:, 1].max()) / 2
            r = max(p[:, 0].max() - p[:, 0].min(),
                    p[:, 1].max() - p[:, 1].min()) / 2 * 1.15 + 1.0
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("Held-out human vs generated trajectories (event-stream model, "
                 "set-level selection, seed 42)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES_DIR / "trajectory_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_feature_distributions():
    """Violin plots: human eval features vs the selected generated set."""
    from features import FEATURE_NAMES

    human_feats = np.load("data/human_eval_features.npy")
    _, gen_feats = _load_selected_pool()

    display_features = [
        ("mean_velocity", "Mean Velocity"),
        ("curvature_mean", "Curvature Mean"),
        ("angular_velocity_mean", "Angular Vel Mean"),
        ("path_efficiency", "Path Efficiency"),
        ("num_direction_changes", "Num Dir Changes"),
    ]
    feat_indices = [FEATURE_NAMES.index(f[0]) for f in display_features]

    fig, axes = plt.subplots(1, len(display_features), figsize=(18, 4))

    for ax_i, (feat_name, display_name) in enumerate(display_features):
        fi = feat_indices[ax_i]
        ax = axes[ax_i]

        h_vals = human_feats[:, fi]
        g_vals = gen_feats[:, fi]

        all_vals = np.concatenate([h_vals, g_vals])
        p95 = np.percentile(all_vals, 95)
        p5 = np.percentile(all_vals, 5)
        h_clip = np.clip(h_vals, p5, p95)
        g_clip = np.clip(g_vals, p5, p95)

        data = [h_clip, g_clip]
        positions = [1, 2]
        colors = ["#4488CC", "#2CB25C"]

        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Human", "Generated"], fontsize=9)
        ax.set_title(display_name, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Feature distributions: human eval set vs selected generated set "
                 "(seed 42, n=2000 each)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure",
                        choices=["auc", "trajectory", "features", "timeline", "all"],
                        default="all")
    args = parser.parse_args()

    if args.figure in ("auc", "all"):
        fig_auc_progression()
    if args.figure in ("trajectory", "all"):
        fig_trajectory_overlay()
    if args.figure in ("features", "all"):
        fig_feature_distributions()
    if args.figure in ("timeline", "all"):
        fig_auc_timeline()
