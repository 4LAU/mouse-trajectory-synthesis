"""Autoresearch wave 3: jitter experiments on x0-orig model.

Speed-proportional heading jitter to make fast movements jerkier,
matching human correlation structure where high speed → high acc/jerk.
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

    # --- Jitter sweep with best settings (guide=0.3, rotate) ---
    for jitter in ["0.005", "0.01", "0.02", "0.03", "0.05", "0.08", "0.1", "0.15"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": jitter}
        experiments.append((s, f"x0-orig jitter={jitter} guide=0.3 rotate"))

    # --- Jitter with no guide (in case guide + jitter interact badly) ---
    for jitter in ["0.01", "0.03", "0.05", "0.1"]:
        s = {**base, "CANDI_GUIDE": "0.0", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": jitter}
        experiments.append((s, f"x0-orig jitter={jitter} no-guide rotate"))

    # --- Jitter with additive correction ---
    for jitter in ["0.01", "0.03", "0.05", "0.1"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "additive",
             "CANDI_JITTER": jitter}
        experiments.append((s, f"x0-orig jitter={jitter} guide=0.3 additive"))

    # --- Jitter + stochastic sampling ---
    for jitter in ["0.02", "0.05"]:
        for eta in ["0.2", "0.3"]:
            s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": jitter, "CANDI_ETA": eta}
            experiments.append((s, f"x0-orig jitter={jitter} eta={eta} guide=0.3 rotate"))

    # --- Multi-candidate with jitter ---
    for jitter in ["0.02", "0.05"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": jitter, "CANDI_CANDIDATES": "3"}
        experiments.append((s, f"x0-orig jitter={jitter} 3cand guide=0.3 rotate"))

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
