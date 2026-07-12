"""Autoresearch experiment runner.

Runs evaluate.py with different inference settings, logs results,
and identifies the best configuration.

Usage:
    python autoresearch.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
EVAL_CMD = [PYTHON, "evaluate.py", "--experiment", "experiments.candi"]
RESULTS_FILE = Path("autoresearch_results.json")


def run_experiment(env_overrides: dict, label: str) -> dict | None:
    env = os.environ.copy()
    env.update(env_overrides)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"  Settings: {env_overrides}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        EVAL_CMD, env=env, capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    print(output, flush=True)

    auc = None
    for line in output.split("\n"):
        if line.strip().startswith("val_auc:"):
            try:
                auc = float(line.strip().split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if auc is None:
        print(f"  FAILED: could not parse AUC from output", flush=True)
        return None

    record = {
        "label": label,
        "auc": auc,
        "settings": env_overrides,
        "elapsed": round(elapsed, 1),
    }
    print(f"\n  RESULT: AUC = {auc:.4f} ({elapsed:.0f}s)", flush=True)
    return record


def load_results() -> list:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(results: list):
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def main():
    results = load_results()
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Previous best AUC: {best_auc:.4f}" if results else "No previous results")

    experiments = [
        # Baseline with best known checkpoint
        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.5 additive"),

        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "rotate", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.5 rotate"),

        # Guide strength sweep
        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.4",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.4 additive"),

        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.6",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.6 additive"),

        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.35",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.35 additive"),

        # Stochastic sampling
        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.3", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.5 additive eta=0.3"),

        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.5", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.5 additive eta=0.5"),

        # More steps
        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "100",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr guide=0.5 additive 100steps"),

        # Compare with original x0 checkpoint
        ({"CANDI_CKPT": "candi_polar_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.5",
          "CANDI_CORRECT": "rotate", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0-orig guide=0.5 rotate (baseline)"),

        # x0+corr with no guide (pure model)
        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.0",
          "CANDI_CORRECT": "additive", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr no-guide additive"),

        ({"CANDI_CKPT": "candi_polar_x0_corr_best.pt",
          "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.0",
          "CANDI_CORRECT": "rotate", "CANDI_STEPS": "50",
          "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1"},
         "x0+corr no-guide rotate"),
    ]

    # Skip already-run experiments
    done_labels = {r["label"] for r in results}
    remaining = [(s, l) for s, l in experiments if l not in done_labels]

    print(f"\n{len(remaining)} experiments to run, {len(done_labels)} already done\n")

    for settings, label in remaining:
        record = run_experiment(settings, label)
        if record:
            results.append(record)
            save_results(results)
            if record["auc"] < best_auc:
                best_auc = record["auc"]
                print(f"  *** NEW BEST: {best_auc:.4f} ***", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["auc"])
    for r in sorted_results:
        marker = " <-- BEST" if r["auc"] == sorted_results[0]["auc"] else ""
        print(f"  {r['auc']:.4f}  {r['label']}{marker}")
    print(f"\nBest AUC: {sorted_results[0]['auc']:.4f} ({sorted_results[0]['label']})")


if __name__ == "__main__":
    main()
