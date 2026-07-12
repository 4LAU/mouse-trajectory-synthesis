"""Autoresearch sweep for speed profile shape interventions.

Tests approaches to fix shape features and angular velocity:
1. MIN_SPEED: Speed floor (fixes angular velocity without mean_acc artifact)
2. SPEED_SKEW: Time-warps speed profile to shift peak earlier (human TTPV=0.345)
3. SPEED_REPLACE: Replaces speed profile with physics-based asymmetric shape
Plus combinations.
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
        "CANDI_RESIDUAL_VEL": "0.0",
        "CANDI_SPEED_SKEW": "0.0",
        "CANDI_SPEED_REPLACE": "",
        "CANDI_MIN_SPEED": "0.0",
    }

    experiments = []

    # --- Phase 0: MIN_SPEED sweep (fixes angular velocity cleanly) ---
    for ms in ["0.01", "0.02", "0.05", "0.1", "0.15", "0.2"]:
        s = {**base, "CANDI_MIN_SPEED": ms}
        experiments.append((s, f"min_speed={ms}"))

    # --- Phase 1: SPEED_SKEW sweep ---
    for sk in ["0.2", "0.4", "0.6", "0.8", "1.0", "1.5"]:
        s = {**base, "CANDI_SPEED_SKEW": sk}
        experiments.append((s, f"skew={sk}"))

    # --- Phase 2: MIN_SPEED + SPEED_SKEW combos ---
    for ms in ["0.05", "0.1"]:
        for sk in ["0.4", "0.6", "0.8"]:
            s = {**base, "CANDI_MIN_SPEED": ms, "CANDI_SPEED_SKEW": sk}
            experiments.append((s, f"min_speed={ms}+skew={sk}"))

    # --- Phase 3: SPEED_REPLACE modes ---
    for mode in ["beta", "asym_mj"]:
        s = {**base, "CANDI_SPEED_REPLACE": mode}
        experiments.append((s, f"replace={mode}"))

    # --- Phase 4: SPEED_REPLACE + noise ---
    for mode in ["beta", "asym_mj"]:
        for sj in ["0.05", "0.1", "0.2"]:
            s = {**base, "CANDI_SPEED_REPLACE": mode, "CANDI_SPEED_JITTER": sj}
            experiments.append((s, f"replace={mode}+sj={sj}"))

    # --- Phase 5: Best combos ---
    for ms in ["0.05", "0.1"]:
        for mode in ["beta", "asym_mj"]:
            s = {**base, "CANDI_MIN_SPEED": ms, "CANDI_SPEED_REPLACE": mode,
                 "CANDI_SPEED_JITTER": "0.1"}
            experiments.append((s, f"min_speed={ms}+replace={mode}+sj=0.1"))

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
    print("SHAPE INTERVENTION RESULTS")
    print(f"{'='*60}")
    shape_results = [r for r in results
                     if any(k in r["label"] for k in ["skew=", "replace=", "min_speed="])]
    shape_results.sort(key=lambda r: r["auc"])
    for r in shape_results:
        print(f"  {r['auc']:.4f}  {r['label']}")
    if shape_results:
        print(f"\nBest shape AUC: {shape_results[0]['auc']:.4f}")


if __name__ == "__main__":
    main()
