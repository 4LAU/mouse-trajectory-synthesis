"""Autoresearch sweep for flow matching CANDI model.

Tests guide strengths, sampling steps, jitter, FEAT_GUIDE, and combinations
with the flow matching checkpoint.
"""
from autoresearch import run_experiment, load_results, save_results


def main():
    results = load_results()
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Current best AUC: {best_auc:.4f}")

    base = {
        "CANDI_CKPT": "candi_polar_flow_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1",
    }

    experiments = []

    # --- Phase 0: No guide baseline ---
    experiments.append(({**base, "CANDI_GUIDE": "0.0"}, "flow no-guide"))

    # --- Phase 1: Guide strength sweep (rotate) ---
    for g in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]:
        s = {**base, "CANDI_GUIDE": g, "CANDI_CORRECT": "rotate"}
        experiments.append((s, f"flow guide={g} rotate"))

    # --- Phase 2: Best guide + jitter ---
    for g in ["0.3", "0.4"]:
        for j in ["0.003", "0.005", "0.01"]:
            s = {**base, "CANDI_GUIDE": g, "CANDI_CORRECT": "rotate",
                 "CANDI_JITTER": j}
            experiments.append((s, f"flow guide={g} rotate jitter={j}"))

    # --- Phase 3: Sampling steps ---
    for g in ["0.3", "0.4"]:
        for steps in ["20", "100", "200"]:
            s = {**base, "CANDI_GUIDE": g, "CANDI_CORRECT": "rotate",
                 "CANDI_STEPS": steps}
            experiments.append((s, f"flow guide={g} rotate steps={steps}"))

    # --- Phase 4: Best combos ---
    for g in ["0.3", "0.4"]:
        s = {**base, "CANDI_GUIDE": g, "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005", "CANDI_STEPS": "100"}
        experiments.append((s, f"flow guide={g} rotate jitter=0.005 steps=100"))

    # --- Phase 5: FEAT_GUIDE (path efficiency guidance, never tested before) ---
    for fg in ["0.5", "1.0", "2.0", "5.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_FEAT_GUIDE": fg, "CANDI_FEAT_EFF_TARGET": "0.84"}
        experiments.append((s, f"flow guide=0.3 rotate feat_guide={fg} eff=0.84"))

    # --- Phase 6: FEAT_GUIDE with jitter ---
    for fg in ["1.0", "2.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_FEAT_GUIDE": fg, "CANDI_FEAT_EFF_TARGET": "0.84"}
        experiments.append((s, f"flow guide=0.3 rotate jitter=0.005 feat_guide={fg}"))

    # --- Phase 7: FEAT_GUIDE on DDPM baseline (test on known-good model) ---
    ddpm_base = {
        "CANDI_CKPT": "candi_polar_best.pt",
        "CANDI_CFG": "0.0", "CANDI_STEPS": "50",
        "CANDI_ETA": "0.0", "CANDI_CANDIDATES": "1",
        "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
        "CANDI_JITTER": "0.005",
    }
    for fg in ["0.5", "1.0", "2.0", "5.0"]:
        s = {**ddpm_base, "CANDI_FEAT_GUIDE": fg, "CANDI_FEAT_EFF_TARGET": "0.84"}
        experiments.append((s, f"ddpm guide=0.3 rotate jitter=0.005 feat_guide={fg}"))

    # --- Phase 8: Probabilistic residual velocity (fixes mean_jerk/mean_acc gap) ---
    for rv in ["0.3", "0.5"]:
        for rp in ["0.05", "0.10", "0.20"]:
            s = {**base, "CANDI_GUIDE": "0.0",
                 "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_PROB": rp,
                 "CANDI_RESIDUAL_FRAC": "0.25"}
            experiments.append((s, f"flow rv={rv} rp={rp}"))
    # residual velocity + guide
    for rv in ["0.3", "0.5"]:
        for rp in ["0.05", "0.10"]:
            s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
                 "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_PROB": rp,
                 "CANDI_RESIDUAL_FRAC": "0.25"}
            experiments.append((s, f"flow guide=0.3 rotate rv={rv} rp={rp}"))

    # --- Phase 9: Residual velocity on DDPM baseline ---
    for rv in ["0.3", "0.5"]:
        for rp in ["0.05", "0.10"]:
            s = {**ddpm_base, "CANDI_RESIDUAL_VEL": rv, "CANDI_RESIDUAL_PROB": rp,
                 "CANDI_RESIDUAL_FRAC": "0.25"}
            experiments.append((s, f"ddpm guide=0.3 rv={rv} rp={rp}"))

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
    print("ALL RESULTS (sorted by AUC)")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["auc"])
    for r in sorted_results[:20]:
        print(f"  {r['auc']:.4f}  {r['label']}")
    if sorted_results:
        print(f"\nBest AUC: {sorted_results[0]['auc']:.4f} ({sorted_results[0]['label']})")


if __name__ == "__main__":
    main()
