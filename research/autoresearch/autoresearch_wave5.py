"""Autoresearch wave 5: focused jitter fine-tuning.

Based on findings:
- Heading jitter=0.005 is optimal (0.782), higher values degrade spatial features
- Speed jitter alone doesn't help (wrong noise structure)
- Test fine granularity around hj=0.005 and small combined jitter
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
    }

    experiments = []

    # --- Fine-grained heading jitter sweep ---
    for hj in ["0.001", "0.002", "0.003", "0.004", "0.006", "0.007", "0.008"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": hj}
        experiments.append((s, f"x0-orig hj={hj} guide=0.3 rotate"))

    # --- Heading jitter with different guide values ---
    for hj in ["0.003", "0.005", "0.007"]:
        for guide in ["0.2", "0.4", "0.5"]:
            s = {**base, "CANDI_GUIDE": guide, "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": hj}
            experiments.append((s, f"x0-orig hj={hj} guide={guide} rotate"))

    # --- Combined small heading + small speed jitter ---
    for hj in ["0.003", "0.005"]:
        for sj in ["0.02", "0.03", "0.05"]:
            s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": hj, "CANDI_SPEED_JITTER": sj}
            experiments.append((s, f"x0-orig hj={hj}+sj={sj} guide=0.3 rotate"))

    # --- Heading jitter with additive correction ---
    for hj in ["0.003", "0.005", "0.007"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "additive",
             "CANDI_JITTER": hj}
        experiments.append((s, f"x0-orig hj={hj} guide=0.3 additive"))

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
