"""Round 6 sweep: frequency-dependent perpendicular scaling + flow noise.

Hypothesis: curvature comes from high-freq perpendicular variation while
max_deviation comes from low-freq. PERP_HP > 1.0 amplifies high-freq
wiggles (curvature) while PERP_SCALE still compresses low-freq (deviation).
This decouples the tradeoff that blocked all previous interventions.

Also tests FLOW_NOISE (stochastic ODE sampling for more diverse samples).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_round6_results.json")

BASE = {
    "CANDI_CKPT": "candi_polar_flow_best.pt",
    "CANDI_CFG": "0.0",
    "CANDI_STEPS": "200",
    "CANDI_ETA": "0.0",
    "CANDI_CANDIDATES": "1",
    "CANDI_GUIDE": "0.3",
    "CANDI_CORRECT": "rotate",
    "CANDI_SPEED_SKEW": "0.3",
    "CANDI_PERP_SCALE": "0.7",
}


def run(env_overrides: dict, label: str) -> dict | None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env.update(env_overrides)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "evaluate.py", "--experiment", "experiments.candi",
         "--n-synthetic", "100"],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if line.strip():
            print(f"  {line.strip()}", flush=True)

    auc = None
    for line in output.split("\n"):
        if line.strip().startswith("val_auc:"):
            try:
                auc = float(line.strip().split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if auc is None:
        print(f"  FAILED: no AUC parsed", flush=True)
        return None

    record = {"label": label, "auc": auc, "settings": env_overrides,
              "elapsed": round(elapsed, 1)}
    print(f"\n  >> AUC = {auc:.4f} ({elapsed:.0f}s)", flush=True)
    return record


def main():
    results = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    done = {r["label"] for r in results}
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Previous best: {best_auc:.4f}" if results else "No previous results")

    experiments = [
        # === Freq-dependent perp: amplify high-freq wiggles ===
        # PERP_SCALE=0.7 for low-freq, PERP_HP controls high-freq
        ({**BASE, "CANDI_PERP_HP": "1.0"},
         "perp_hp=1.0"),  # high-freq unscaled, low-freq compressed
        ({**BASE, "CANDI_PERP_HP": "1.3"},
         "perp_hp=1.3"),
        ({**BASE, "CANDI_PERP_HP": "1.5"},
         "perp_hp=1.5"),
        ({**BASE, "CANDI_PERP_HP": "2.0"},
         "perp_hp=2.0"),

        # === Different cutoff windows ===
        ({**BASE, "CANDI_PERP_HP": "1.3", "CANDI_PERP_HP_WIN": "11"},
         "perp_hp=1.3+win=11"),
        ({**BASE, "CANDI_PERP_HP": "1.3", "CANDI_PERP_HP_WIN": "31"},
         "perp_hp=1.3+win=31"),

        # === PERP_HP + higher base PERP_SCALE ===
        ({**BASE, "CANDI_PERP_SCALE": "0.8", "CANDI_PERP_HP": "1.3"},
         "perp=0.8+hp=1.3"),
        ({**BASE, "CANDI_PERP_SCALE": "0.6", "CANDI_PERP_HP": "1.5"},
         "perp=0.6+hp=1.5"),

        # === Flow noise (stochastic ODE sampling) ===
        ({**BASE, "CANDI_FLOW_NOISE": "0.1"},
         "flow_noise=0.1"),
        ({**BASE, "CANDI_FLOW_NOISE": "0.3"},
         "flow_noise=0.3"),
        ({**BASE, "CANDI_FLOW_NOISE": "0.5"},
         "flow_noise=0.5"),

        # === Flow noise + perp_hp combo ===
        ({**BASE, "CANDI_FLOW_NOISE": "0.3", "CANDI_PERP_HP": "1.3"},
         "flow_noise=0.3+perp_hp=1.3"),
    ]

    remaining = [(s, l) for s, l in experiments if l not in done]
    print(f"\n{len(remaining)} experiments, {len(done)} already done\n")

    for settings, label in remaining:
        record = run(settings, label)
        if record:
            results.append(record)
            RESULTS_FILE.write_text(json.dumps(results, indent=2))
            if record["auc"] < best_auc:
                best_auc = record["auc"]
                print(f"  *** NEW BEST: {best_auc:.4f} ***", flush=True)

    print(f"\n{'='*60}")
    print("RESULTS (sorted by AUC, lower=better)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda r: r["auc"]):
        print(f"  {r['auc']:.4f}  {r['label']}")


if __name__ == "__main__":
    main()
