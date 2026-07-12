"""Compute seed-level statistics from recorded per-seed AUC values.

The headline result is a detector AUC of 0.504 (mean of tuning seeds
42/43/44). Six later out-of-sample seeds (45-50) were run on another
machine; their per-seed AUCs are recorded in EXPERIMENTS.md, but the
per-seed feature matrices are not in the release bundle, so a
refit/bootstrap (as external_validation/headline_statistics_45_46.py
does for seeds 45/46) is not possible for seeds 47-50 here. This script
therefore computes seed-level statistics from the recorded per-seed
values, citing the EXPERIMENTS.md entries of July 9 and July 11 as the
source of the values.
"""

import json
import math
import statistics


# Recorded tuning RF-OOB AUCs for seeds 42-44 (EXPERIMENTS.md, July 9).
TUNING_RF_OOB = {42: 0.5095, 43: 0.5030, 44: 0.4993}

# Recorded out-of-sample RF-OOB AUCs for seeds 45-50 (EXPERIMENTS.md, July 9 and July 11).
OOS_RF_OOB = {45: 0.5148, 46: 0.5190, 47: 0.5221, 48: 0.5114, 49: 0.5033, 50: 0.5087}

# Recorded out-of-sample raw neural-network AUCs (3-fold, never part of any selection loop).
OOS_RAW_NN = {45: 0.5033, 46: 0.5077, 47: 0.5063, 48: 0.5174, 49: 0.5036, 50: 0.5080}  # 3-fold, never part of any selection loop

# Recorded out-of-sample gradient-boosted-model AUCs.
OOS_GBM = {45: 0.5211, 46: 0.5189, 47: 0.5262, 48: 0.5271, 49: 0.5362, 50: 0.5106}  # 5-fold


T_CRITICAL_95_DF_5 = 2.570582


def _summary(values, include_ci=False):
    """Return summary statistics for a mapping of seed to AUC."""
    observations = list(values.values())
    result = {
        "n": len(observations),
        "mean": statistics.mean(observations),
        "sd": statistics.stdev(observations),
        "min": min(observations),
        "max": max(observations),
    }
    if include_ci:
        mean = result["mean"]
        margin = T_CRITICAL_95_DF_5 * result["sd"] / math.sqrt(result["n"])
        lower = mean - margin
        upper = mean + margin
        result["ci95"] = {
            "df": result["n"] - 1,
            "t_critical": T_CRITICAL_95_DF_5,
            "lower": lower,
            "upper": upper,
        }
        result["ci_excludes_0_50"] = lower > 0.50 or upper < 0.50
    return result


def _rounded(value):
    return round(value, 10)


def _rounded_summary(summary):
    """Round computed floating-point values for stable, readable JSON."""
    rounded = dict(summary)
    for key in ("mean", "sd", "min", "max"):
        rounded[key] = _rounded(rounded[key])
    if "ci95" in rounded:
        rounded["ci95"] = dict(rounded["ci95"])
        for key in ("lower", "upper"):
            rounded["ci95"][key] = _rounded(rounded["ci95"][key])
    return rounded


def main():
    summaries = {
        "tuning_rf_oob": _rounded_summary(_summary(TUNING_RF_OOB)),
        "oos_rf_oob": _rounded_summary(_summary(OOS_RF_OOB, include_ci=True)),
        "oos_raw_nn": _rounded_summary(_summary(OOS_RAW_NN, include_ci=True)),
        "oos_gbm": _rounded_summary(_summary(OOS_GBM, include_ci=True)),
    }
    nine_seed_rf = _summary({**TUNING_RF_OOB, **OOS_RF_OOB})
    summaries["nine_seed_rf"] = {
        "n": nine_seed_rf["n"],
        "mean": _rounded(nine_seed_rf["mean"]),
        "sd": _rounded(nine_seed_rf["sd"]),
    }

    rf = summaries["oos_rf_oob"]
    sanity_ok = (
        round(rf["mean"], 4) == 0.5132
        and round(rf["sd"], 4) == 0.0069
        and round(rf["ci95"]["lower"], 3) == 0.506
        and round(rf["ci95"]["upper"], 3) == 0.520
        and rf["ci_excludes_0_50"]
    )
    if not sanity_ok:
        print("BLOCKED: OOS RF sanity values disagree with the requested checks.")
        print(rf)
        return

    output = {"groups": summaries}
    with open("external_validation/oos_seed_statistics.json", "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print("group           n   mean      sd      min      max       95% CI             excludes 0.50")
    for name in ("tuning_rf_oob", "oos_rf_oob", "oos_raw_nn", "oos_gbm"):
        summary = summaries[name]
        if "ci95" in summary:
            ci = "[%.3f, %.3f]" % (summary["ci95"]["lower"], summary["ci95"]["upper"])
            excludes = str(summary["ci_excludes_0_50"])
        else:
            ci = "-"
            excludes = "-"
        print(
            "%-15s %d   %.4f   %.4f   %.4f   %.4f   %-18s %s"
            % (
                name,
                summary["n"],
                summary["mean"],
                summary["sd"],
                summary["min"],
                summary["max"],
                ci,
                excludes,
            )
        )
    print(
        "%-15s %d   %.4f   %.4f"
        % ("nine_seed_rf", summaries["nine_seed_rf"]["n"], summaries["nine_seed_rf"]["mean"], summaries["nine_seed_rf"]["sd"])
    )


if __name__ == "__main__":
    main()
