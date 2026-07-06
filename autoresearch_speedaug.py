"""Autoresearch sweep for speed-aug retrained model.

Evaluates candi_polar_speedaug_best.pt with various post-processing configs
to find optimal settings for the new model.
"""
from autoresearch import run_experiment, load_results, save_results


def main():
    results = load_results()
    best_auc = min((r["auc"] for r in results), default=1.0)
    print(f"Current best AUC: {best_auc:.4f}")

    base = {
        "CANDI_CKPT": "candi_polar_speedaug_best.pt",
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

    # --- Baseline: no post-processing ---
    s = {**base}
    experiments.append((s, "speedaug baseline (no PP)"))

    # --- Best known settings from original model ---
    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005"}
    experiments.append((s, "speedaug guide=0.3 rotate jitter=0.005"))

    # --- Guide strength sweep ---
    for g in ["0.1", "0.2", "0.3", "0.5"]:
        s = {**base, "CANDI_GUIDE": g, "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005"}
        experiments.append((s, f"speedaug guide={g} rotate jitter=0.005"))

    # --- Jitter sweep ---
    for j in ["0.0", "0.003", "0.005", "0.01"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": j}
        experiments.append((s, f"speedaug guide=0.3 rotate jitter={j}"))

    # --- OU speed modulation (top performers from original) ---
    for sigma in ["0.5", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "5.0"}
        experiments.append((s, f"speedaug OU={sigma}+hj=0.005 guide=0.3 rotate"))

    # --- Stochastic sampling (eta > 0) ---
    for eta in ["0.1", "0.3", "0.5", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005", "CANDI_ETA": eta}
        experiments.append((s, f"speedaug eta={eta}+hj=0.005 guide=0.3 rotate"))

    # --- PERP_SCALE (path efficiency) ---
    for ps in ["1.2", "1.5", "2.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_PERP_SCALE": ps}
        experiments.append((s, f"speedaug perp={ps}+hj=0.005 guide=0.3 rotate"))

    # --- No CFG (CFG=0 baseline) ---
    s = {**base, "CANDI_CFG": "0.0", "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005"}
    experiments.append((s, "speedaug cfg=0 guide=0.3 rotate jitter=0.005"))

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
    print("SPEEDAUG RESULTS")
    print(f"{'='*60}")
    speedaug_results = [r for r in results if "speedaug" in r["label"]]
    speedaug_results.sort(key=lambda r: r["auc"])
    for r in speedaug_results:
        print(f"  {r['auc']:.4f}  {r['label']}")
    if speedaug_results:
        print(f"\nBest speedaug AUC: {speedaug_results[0]['auc']:.4f}")


if __name__ == "__main__":
    main()
