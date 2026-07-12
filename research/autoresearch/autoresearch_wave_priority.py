"""Autoresearch priority sweep: highest-impact experiments first.

Combines the most promising experiments from waves 8+9, ordered by
expected impact. Runs ACC_SCALE and PERP_SCALE first (targeting the
#1 and #2 detection signals), then combos, then remaining exploratory.
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
        "CANDI_ACC_SCALE": "0.0", "CANDI_ACC_MODE": "speed",
        "CANDI_PERP_SCALE": "1.0",
        "CANDI_DUR_STD": "0.7",
    }

    experiments = []

    # ============================================================
    # TIER 1: ACC_SCALE sweep (targets speed-acc correlation gap)
    # ============================================================
    for acc in ["0.1", "0.2", "0.3", "0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc}
        experiments.append((s, f"acc_scale={acc}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 1: PERP_SCALE sweep (targets path_efficiency gap)
    # ============================================================
    for ps in ["1.5", "2.0", "3.0", "5.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_PERP_SCALE": ps}
        experiments.append((s, f"perp={ps}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 2: ACC + PERP combos
    # ============================================================
    for acc, ps in [("0.3", "2.0"), ("0.3", "3.0"), ("0.5", "2.0")]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc, "CANDI_PERP_SCALE": ps}
        experiments.append((s, f"acc={acc}+perp={ps}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 2: ACC + OU combos (two mechanisms for speed variation)
    # ============================================================
    for acc in ["0.2", "0.3", "0.5"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": "0.5",
             "CANDI_ACC_SCALE": acc}
        experiments.append((s, f"acc={acc}+OU=0.5+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 2: ACC + sharpening combos
    # ============================================================
    for acc, sh in [("0.3", "0.1"), ("0.3", "0.3"), ("0.5", "0.1")]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc, "CANDI_SHARPEN": sh}
        experiments.append((s, f"acc={acc}+sh={sh}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: Broader sharpening sweep
    # ============================================================
    for sh in ["0.2", "0.3", "0.5", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_SHARPEN": sh}
        experiments.append((s, f"sharpen={sh}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: Feature guidance (gradient-based path_efficiency opt)
    # ============================================================
    for fg in ["0.5", "1.0", "2.0", "5.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_FEAT_GUIDE": fg}
        experiments.append((s, f"feat_guide={fg}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: DH OU (heading noise with temporal correlation)
    # ============================================================
    for dh_sigma in ["0.02", "0.05", "0.1", "0.2"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_DH_OU_SIGMA": dh_sigma, "CANDI_DH_OU_THETA": "3.0"}
        experiments.append((s, f"DhOU={dh_sigma}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: ACC_SCALE distance mode
    # ============================================================
    for acc in ["0.1", "0.2", "0.3"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_ACC_SCALE": acc, "CANDI_ACC_MODE": "dist"}
        experiments.append((s, f"acc_dist={acc}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: Duration std multiplier
    # ============================================================
    for dur_std in ["0.85", "1.0", "1.2"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_DUR_STD": dur_std}
        experiments.append((s, f"dur_std={dur_std}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: OU + heading jitter (wave 8 leftover)
    # ============================================================
    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.005",
         "CANDI_OU_SIGMA": "0.3"}
    experiments.append((s, f"OU=0.3+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: Feature guidance + OU speed combos
    # ============================================================
    for fg in ["1.0", "2.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": "0.5",
             "CANDI_FEAT_GUIDE": fg}
        experiments.append((s, f"feat={fg}+OU=0.5+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: Sharpening (wave 8 leftover)
    # ============================================================
    for sharpen in ["0.02", "0.05", "0.1"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_SHARPEN": sharpen}
        experiments.append((s, f"sharpen={sharpen}+hj=0.005 guide=0.3 rotate"))

    # ============================================================
    # TIER 3: OU+hj combos (missing from wave 7)
    # ============================================================
    for sigma in ["0.5", "0.8", "1.0"]:
        s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
             "CANDI_JITTER": "0.005",
             "CANDI_OU_SIGMA": sigma, "CANDI_OU_THETA": "5.0"}
        experiments.append((s, f"OU sigma={sigma}+hj=0.005 theta=5 guide=0.3 rotate"))

    # Missing theta=10 experiment
    s = {**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
         "CANDI_JITTER": "0.0",
         "CANDI_OU_SIGMA": "1.0", "CANDI_OU_THETA": "10.0"}
    experiments.append((s, "OU sigma=1.0 theta=10 guide=0.3 rotate"))

    # ============================================================
    # TIER 4: Kitchen sink combos
    # ============================================================
    kitchen_sinks = [
        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_OU_SIGMA": "0.5", "CANDI_ACC_SCALE": "0.3",
          "CANDI_SHARPEN": "0.1"},
         "acc=0.3+OU=0.5+sh=0.1+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_ACC_SCALE": "0.3", "CANDI_DUR_STD": "1.0"},
         "acc=0.3+dur=1.0+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_ACC_SCALE": "0.3", "CANDI_PERP_SCALE": "2.0"},
         "acc=0.3+perp=2.0+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_ACC_SCALE": "0.3", "CANDI_PERP_SCALE": "2.0",
          "CANDI_DUR_STD": "1.0"},
         "acc=0.3+perp=2.0+dur=1.0+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_OU_SIGMA": "0.5", "CANDI_DH_OU_SIGMA": "0.05",
          "CANDI_FEAT_GUIDE": "1.0"},
         "feat=1+OU=0.5+DhOU=0.05+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_OU_SIGMA": "0.5", "CANDI_DH_OU_SIGMA": "0.05",
          "CANDI_SHARPEN": "0.05", "CANDI_FEAT_GUIDE": "1.0"},
         "feat=1+OU=0.5+DhOU=0.05+sh=0.05+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_ACC_SCALE": "0.3", "CANDI_PERP_SCALE": "2.0",
          "CANDI_OU_SIGMA": "0.5"},
         "acc=0.3+perp=2.0+OU=0.5+hj=0.005 rotate"),

        ({**base, "CANDI_GUIDE": "0.3", "CANDI_CORRECT": "rotate",
          "CANDI_JITTER": "0.005",
          "CANDI_ACC_SCALE": "0.3", "CANDI_PERP_SCALE": "2.0",
          "CANDI_OU_SIGMA": "0.5", "CANDI_SHARPEN": "0.1",
          "CANDI_DUR_STD": "1.0"},
         "full_combo: acc=0.3+perp=2.0+OU=0.5+sh=0.1+dur=1.0+hj=0.005"),
    ]
    experiments.extend(kitchen_sinks)

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
    print("ALL RESULTS (sorted, top 30)")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r["auc"])
    for r in sorted_results[:30]:
        marker = " <-- BEST" if r["auc"] == sorted_results[0]["auc"] else ""
        print(f"  {r['auc']:.4f}  {r['label']}{marker}")
    print(f"\nBest AUC: {sorted_results[0]['auc']:.4f} ({sorted_results[0]['label']})")


if __name__ == "__main__":
    main()
