"""Autoresearch wave 9: Acceleration scaling + broader sharpening.

Key insight: the cross-trajectory correlation gap (human r=+0.99, synth r=-0.17
for speed-derived features) is the #1 detection signal. Acceleration scaling
amplifies speed deviations proportional to trajectory speed level, making fast
trajectories jerkier — creating positive speed-acceleration correlation.
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
        "CANDI_SMOOTH_DH": "0.0", "CANDI_SMOOTH_POS": "0",
        "CANDI_SPEED_JITTER": "0.0",
        "CANDI_OU_SIGMA": "0.0", "CANDI_OU_THETA": "5.0",
        "CANDI_DH_OU_SIGMA": "0.0", "CANDI_DH_OU_THETA": "3.0",
        "CANDI_SHARPEN": "0.0",
        "CANDI_FEAT_GUIDE": "0.0", "CANDI_FEAT_EFF_TARGET": "0.84",
        "CANDI_ACC_SCALE": "0.0", "CANDI_ACC_MODE": "speed",
        "CANDI_PERP_SCALE": "1.0",
        "CANDI_DUR_STD": "0.7",
    }

    experiments = []

    # --- PRIORITY 1: Acceleration scaling sweep (novel approach) ---
    for acc in ["0.1", "0.2", "0.3", "0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc}
        experiments.append((s, f"acc_scale={acc}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 1b: Acc scaling with distance mode ---
    for acc in ["0.1", "0.2", "0.3"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc, "CANDI_ACC_MODE": "dist"}
        experiments.append((s, f"acc_dist={acc}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 2: Acc scaling + OU combo ---
    for acc in ["0.2", "0.3", "0.5"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": "0.5",
             "CANDI_ACC_SCALE": acc}
        experiments.append((s, f"acc={acc}+OU=0.5+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 3: Broader sharpening values ---
    for sh in ["0.2", "0.3", "0.5", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_SHARPEN": sh}
        experiments.append((s, f"sharpen={sh}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 4: Acc scaling + sharpening combo ---
    for acc, sh in [("0.3", "0.1"), ("0.3", "0.3"), ("0.5", "0.1")]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc, "CANDI_SHARPEN": sh}
        experiments.append((s, f"acc={acc}+sh={sh}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 5: Duration std multiplier ---
    for dur_std in ["0.85", "1.0", "1.2"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_DUR_STD": dur_std}
        experiments.append((s, f"dur_std={dur_std}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 6: Perpendicular scale (amplify curvature) ---
    for ps in ["1.5", "2.0", "3.0", "5.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_PERP_SCALE": ps}
        experiments.append((s, f"perp={ps}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 7: Kitchen sink combos ---
    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_OU_SIGMA": "0.5",
         "CANDI_ACC_SCALE": "0.3",
         "CANDI_SHARPEN": "0.1"}
    experiments.append((s, "acc=0.3+OU=0.5+sh=0.1+hj=0.005 rotate"))

    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_ACC_SCALE": "0.3",
         "CANDI_DUR_STD": "1.0"}
    experiments.append((s, "acc=0.3+dur=1.0+hj=0.005 rotate"))

    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_ACC_SCALE": "0.3",
         "CANDI_PERP_SCALE": "2.0"}
    experiments.append((s, "acc=0.3+perp=2.0+hj=0.005 rotate"))

    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_ACC_SCALE": "0.3",
         "CANDI_PERP_SCALE": "2.0",
         "CANDI_DUR_STD": "1.0"}
    experiments.append((s, "acc=0.3+perp=2.0+dur=1.0+hj=0.005 rotate"))

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
    print("ALL RESULTS (sorted, top 20)")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["auc"])
    for r in sorted_results[:20]:
        marker = " <-- BEST" if r["auc"] == sorted_results[0]["auc"] else ""
        print(f"  {r['auc']:.4f}  {r['label']}{marker}")
    print(f"\nBest AUC: {sorted_results[0]['auc']:.4f} ({sorted_results[0]['label']})")


if __name__ == "__main__":
    main()
