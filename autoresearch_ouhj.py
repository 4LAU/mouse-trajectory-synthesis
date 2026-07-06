"""Quick OU+hj synergy test, then done with post-processing exploration."""
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
    for sigma in ["0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "5.0"}
        experiments.append((s, f"OU sigma={sigma}+hj=0.005 theta=5 guide=0.3 rotate"))

    done_labels = {r["label"] for r in results}
    remaining = [(s, l) for s, l in experiments if l not in done_labels]
    print(f"\n{len(remaining)} experiments to run\n")

    for settings, label in remaining:
        record = run_experiment(settings, label)
        if record:
            results.append(record)
            save_results(results)
            if record["auc"] < best_auc:
                best_auc = record["auc"]
                print(f"  *** NEW BEST: {best_auc:.4f} ***", flush=True)

    sorted_results = sorted(results, key=lambda r: r["auc"])
    print(f"\nBest AUC: {sorted_results[0]['auc']:.4f} ({sorted_results[0]['label']})")


if __name__ == "__main__":
    main()
