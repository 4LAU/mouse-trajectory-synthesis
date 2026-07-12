"""Autoresearch sweep for residual velocity post-processing.

Targets the speed block correlation gap: human data has mean_acc proportional
to mean_vel (r=0.99+) because fast movements don't fully decelerate.
The model generates symmetric bell-shaped speed profiles (mean_acc ≈ 0).

RESIDUAL_VEL adds a speed ramp to the last fraction of moving timesteps,
proportional to mean_speed, creating the asymmetry that produces the correlation.
"""
from autoresearch import run_experiment, load_results, save_results


def main():
    results = load_results()
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Current best AUC: {best_auc:.4f}")

    base = {
        "CANDI_CKPT": "candi_polar_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1",
        "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
        "CANDI_JITTER": "0.005",
        "CANDI_SMOOTH_DH": "0.0", "CANDI_SMOOTH_POS": "0",
        "CANDI_SPEED_JITTER": "0.0",
        "CANDI_OU_SIGMA": "0.0", "CANDI_OU_THETA": "5.0",
        "CANDI_DH_OU_SIGMA": "0.0", "CANDI_DH_OU_THETA": "3.0",
        "CANDI_SHARPEN": "0.0",
        "CANDI_FEAT_GUIDE": "0.0",
        "CANDI_ACC_SCALE": "0.0",
        "CANDI_PERP_SCALE": "1.0",
        "CANDI_DUR_STD": "0.7",
    }

    experiments = []

    # --- Phase 1: RESIDUAL_VEL strength sweep (fixed frac=0.25) ---
    for rv in ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5"]:
        s = {**base, "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_FRAC": "0.25"}
        experiments.append((s, f"residual_vel={rv} frac=0.25"))

    # --- Phase 2: RESIDUAL_FRAC sweep (with best strength candidates) ---
    for rv in ["0.3", "0.5", "0.7"]:
        for frac in ["0.1", "0.15", "0.3", "0.4"]:
            s = {**base, "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_FRAC": frac}
            experiments.append((s, f"residual_vel={rv} frac={frac}"))

    # --- Phase 3: Combine with best known post-processing ---
    for rv in ["0.3", "0.5"]:
        s = {**base, "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_FRAC": "0.25",
             "CANDI_OU_SIGMA": "0.5", "CANDI_OU_THETA": "5.0"}
        experiments.append((s, f"residual_vel={rv}+OU=0.5"))

        s = {**base, "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_FRAC": "0.25",
             "CANDI_PERP_SCALE": "1.2"}
        experiments.append((s, f"residual_vel={rv}+perp=1.2"))

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

    print(f"\n{'='*60}")
    print("RESIDUAL VELOCITY RESULTS")
    print(f"{'='*60}")
    rv_results = [r for r in results if "residual_vel" in r["label"]]
    rv_results.sort(key=lambda r: r["auc"])
    for r in rv_results:
        print(f"  {r['auc']:.4f}  {r['label']}")
    if rv_results:
        print(f"\nBest residual_vel AUC: {rv_results[0]['auc']:.4f}")


if __name__ == "__main__":
    main()
