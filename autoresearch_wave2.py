"""Autoresearch wave 2: post-processing + x0-orig parameter sweep.

Focus on x0-orig model (candi_polar_best.pt) since x0+corr is dead.
Tests heading smoothing, position smoothing, guide tuning, and combinations.
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
    }

    experiments = []

    # --- x0-orig guide sweep ---
    for guide in ["0.3", "0.4", "0.6", "0.7", "1.0"]:
        for correct in ["additive", "rotate"]:
            s = {**base, "CANDI_GUIDE": guide, "CANDI_CORRECT": correct}
            experiments.append((s, f"x0-orig guide={guide} {correct}"))

    # x0-orig no-guide baselines
    for correct in ["additive", "rotate"]:
        s = {**base, "CANDI_GUIDE": "0.0", "CANDI_CORRECT": correct}
        experiments.append((s, f"x0-orig no-guide {correct}"))

    # x0-orig additive guide=0.5 (was only tested with rotate)
    s = {**base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": "additive"}
    experiments.append((s, "x0-orig guide=0.5 additive"))

    # --- Heading smoothing on x0-orig ---
    for alpha in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
        for correct in ["additive", "rotate"]:
            s = {**base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": correct,
                 "CANDI_SMOOTH_DH": alpha}
            experiments.append((s, f"x0-orig smooth_dh={alpha} {correct} guide=0.5"))

    # --- Position smoothing on x0-orig ---
    for wl in ["7", "11", "15", "21"]:
        s = {**base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": "rotate",
             "CANDI_SMOOTH_POS": wl}
        experiments.append((s, f"x0-orig smooth_pos={wl} rotate guide=0.5"))

    # --- Combined smoothing ---
    for alpha in ["0.2", "0.3"]:
        for wl in ["7", "11"]:
            for correct in ["additive", "rotate"]:
                s = {**base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": correct,
                     "CANDI_SMOOTH_DH": alpha, "CANDI_SMOOTH_POS": wl}
                experiments.append((s, f"x0-orig dh={alpha}+pos={wl} {correct} guide=0.5"))

    # --- Stochastic on x0-orig ---
    for eta in ["0.2", "0.3", "0.5"]:
        s = {**base, "CANDI_GUIDE": "0.5", "CANDI_CORRECT": "rotate",
             "CANDI_ETA": eta}
        experiments.append((s, f"x0-orig eta={eta} rotate guide=0.5"))

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
