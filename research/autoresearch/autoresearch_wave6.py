"""Autoresearch wave 6: scorer-based multi-candidate selection.

Uses trained logistic regression scorer to pick most human-like
trajectory from N candidates. Tests with best jitter settings.
"""
from autoresearch import run_experiment, load_results, save_results


def main():
    results = load_results()
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Current best AUC: {best_auc:.4f}")

    base = {
        "CANDI_CKPT": "candi_polar_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0",
        "CANDI_SMOOTH_DH": "0.0", "CANDI_SMOOTH_POS": "0",
        "CANDI_SPEED_JITTER": "0.0",
    }

    experiments = []

    # --- Multi-candidate with best jitter (hj=0.005) ---
    for n_cand in ["3", "5", "8"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005", "CANDI_CANDIDATES": n_cand}
        experiments.append((s, f"x0-orig hj=0.005 {n_cand}cand guide=0.3 rotate"))

    # --- Multi-candidate without jitter ---
    for n_cand in ["3", "5", "8"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.0", "CANDI_CANDIDATES": n_cand}
        experiments.append((s, f"x0-orig no-jitter {n_cand}cand guide=0.3 rotate"))

    # --- Multi-candidate with different jitter values ---
    for hj in ["0.003", "0.007"]:
        for n_cand in ["3", "5"]:
            s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": hj, "CANDI_CANDIDATES": n_cand}
            experiments.append((s, f"x0-orig hj={hj} {n_cand}cand guide=0.3 rotate"))

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
