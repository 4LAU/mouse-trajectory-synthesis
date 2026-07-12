"""Autoresearch wave 7: OU speed modulation sweep.

Tests Ornstein-Uhlenbeck temporally-correlated speed perturbation
to create speed-acceleration correlation matching human data.
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
        "CANDI_OU_THETA": "5.0",
    }

    experiments = []

    # --- OU sigma sweep at theta=5 (correlation time = 0.2s = 25 samples) ---
    for sigma in ["0.3", "0.5", "0.8", "1.0", "1.5", "2.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.0", "CANDI_OU_SIGMA": sigma}
        experiments.append((s, f"OU sigma={sigma} theta=5 guide=0.3 rotate"))

    # --- Longer correlation time (theta=2, corr_time=0.5s=62 samples) ---
    for sigma in ["0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.0",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "2.0"}
        experiments.append((s, f"OU sigma={sigma} theta=2 guide=0.3 rotate"))

    # --- Shorter correlation time (theta=10, corr_time=0.1s=12 samples) ---
    for sigma in ["0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.0",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "10.0"}
        experiments.append((s, f"OU sigma={sigma} theta=10 guide=0.3 rotate"))

    # --- Best OU + heading jitter combo ---
    for sigma in ["0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "5.0"}
        experiments.append((s, f"OU sigma={sigma}+hj=0.005 theta=5 guide=0.3 rotate"))

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
