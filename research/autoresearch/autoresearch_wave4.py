"""Autoresearch wave 4: speed jitter + combined jitter experiments.

Speed jitter adds multiplicative noise to speed profile, making fast
movements noisier (matching human acceleration/jerk correlations).
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
        "CANDI_JITTER": "0.0",
    }

    experiments = []

    # --- Speed jitter only (guide=0.3, rotate) ---
    for sj in ["0.02", "0.05", "0.08", "0.1", "0.15", "0.2", "0.3"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_SPEED_JITTER": sj}
        experiments.append((s, f"x0-orig speed_jitter={sj} guide=0.3 rotate"))

    # --- Combined heading + speed jitter ---
    # Use best heading jitter from wave 3 (will know after it finishes)
    # For now test combinations around the sweet spot
    for hj in ["0.005", "0.01", "0.02"]:
        for sj in ["0.05", "0.1", "0.15"]:
            s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": hj, "CANDI_SPEED_JITTER": sj}
            experiments.append((s, f"x0-orig hj={hj}+sj={sj} guide=0.3 rotate"))

    # --- Speed jitter with no guide ---
    for sj in ["0.05", "0.1", "0.15"]:
        s = {**base, "CANDI_GUIDE": "0.0", "CANDI_CORRECT": "rotate",
             "CANDI_SPEED_JITTER": sj}
        experiments.append((s, f"x0-orig speed_jitter={sj} no-guide rotate"))

    # --- Speed jitter with additive correction ---
    for sj in ["0.05", "0.1", "0.15"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "additive",
             "CANDI_SPEED_JITTER": sj}
        experiments.append((s, f"x0-orig speed_jitter={sj} guide=0.3 additive"))

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
