"""Round 4 sweep: DH_AMP for curvature + best combos from rounds 2-3.

Diagnostic findings:
- curvature_mean: synth 0.058 vs human 0.112 (median) — need ~2x increase
- angular_velocity_std: synth 31.5 vs human 45.5 — need ~45% increase
- angular_velocity_mean: synth 15.9 vs human 22.5 — need ~40% increase
- velocity_skewness: synth 1.50 vs human 1.04 — need reduction
- DH_AMP scales heading changes directly (curvature ~ |dheading|/speed)
- JITTER was harmful (0.611 vs 0.533 baseline)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
RESULTS_FILE = Path("sweep_round4_results.json")

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
        # === DH_AMP sweep (curvature & angular velocity) ===
        ({**BASE, "CANDI_DH_AMP": "0.3"},
         "dh_amp=0.3"),
        ({**BASE, "CANDI_DH_AMP": "0.5"},
         "dh_amp=0.5"),
        ({**BASE, "CANDI_DH_AMP": "1.0"},
         "dh_amp=1.0"),

        # === DH_AMP + reduced skew (fix both curvature and velocity_skewness) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_DH_AMP": "0.5"},
         "skew=0.2+dh_amp=0.5"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.25", "CANDI_DH_AMP": "0.5"},
         "skew=0.25+dh_amp=0.5"),

        # === DH_OU (correlated heading noise, alternative to jitter) ===
        ({**BASE, "CANDI_DH_OU_SIGMA": "0.3"},
         "dh_ou=0.3"),
        ({**BASE, "CANDI_DH_OU_SIGMA": "0.5"},
         "dh_ou=0.5"),

        # === Reduced skew alone (velocity_skewness too high) ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.2"},
         "skew=0.2+perp=0.7"),
        ({**BASE, "CANDI_SPEED_SKEW": "0.15"},
         "skew=0.15+perp=0.7"),

        # === DH_AMP + DH_OU (amplify + add correlated noise) ===
        ({**BASE, "CANDI_DH_AMP": "0.3", "CANDI_DH_OU_SIGMA": "0.3"},
         "dh_amp=0.3+dh_ou=0.3"),

        # === Kitchen sink: reduced skew + DH_AMP + DH_OU ===
        ({**BASE, "CANDI_SPEED_SKEW": "0.2", "CANDI_DH_AMP": "0.3",
          "CANDI_DH_OU_SIGMA": "0.3"},
         "skew=0.2+dh_amp=0.3+dh_ou=0.3"),

        # === Perp sweep with DH_AMP ===
        ({**BASE, "CANDI_DH_AMP": "0.5", "CANDI_PERP_SCALE": "0.6"},
         "dh_amp=0.5+perp=0.6"),
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
