"""Autoresearch wave 8: Feature guidance, OU combos, heading OU, sharpening.

Priority order:
1. OU speed + heading jitter combo (highest priority)
2. Feature guidance during DDIM (novel approach)
3. OU heading noise
4. Sharpening
5. Combined approaches
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
    }

    experiments = []

    # --- PRIORITY 1: OU speed + heading jitter combos ---
    # NOTE: sigma=0.5/0.8/1.0 + hj=0.005 already covered in wave 7
    # Only test sigma=0.3 + hj combo here
    for ou_sigma in ["0.3"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": ou_sigma}
        experiments.append((s, f"OU={ou_sigma}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 2: Feature guidance sweep ---
    for fg in ["0.5", "1.0", "2.0", "5.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_FEAT_GUIDE": fg}
        experiments.append((s, f"feat_guide={fg}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 3: Feature guidance + OU speed ---
    for fg in ["1.0", "2.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": "0.5",
             "CANDI_FEAT_GUIDE": fg}
        experiments.append((s, f"feat={fg}+OU=0.5+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 4: OU heading noise ---
    for dh_sigma in ["0.02", "0.05", "0.1", "0.2"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_DH_OU_SIGMA": dh_sigma, "CANDI_DH_OU_THETA": "3.0"}
        experiments.append((s, f"DhOU={dh_sigma}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 5: Sharpening ---
    for sharpen in ["0.02", "0.05", "0.1"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_SHARPEN": sharpen}
        experiments.append((s, f"sharpen={sharpen}+hj=0.005 guide=0.3 rotate"))

    # --- PRIORITY 6: Kitchen sink combo ---
    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_OU_SIGMA": "0.5",
         "CANDI_DH_OU_SIGMA": "0.05",
         "CANDI_FEAT_GUIDE": "1.0"}
    experiments.append((s, f"feat=1+OU=0.5+DhOU=0.05+hj=0.005 rotate"))

    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_OU_SIGMA": "0.5",
         "CANDI_DH_OU_SIGMA": "0.05",
         "CANDI_SHARPEN": "0.05",
         "CANDI_FEAT_GUIDE": "1.0"}
    experiments.append((s, f"feat=1+OU=0.5+DhOU=0.05+sh=0.05+hj=0.005 rotate"))

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
