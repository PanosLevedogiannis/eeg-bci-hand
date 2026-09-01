"""
Group-level statistical test: is classification accuracy significantly
above chance (50%) ACROSS subjects? This is the question a supervisor
asks that per-subject permutation tests don't directly answer — eleven
subjects each individually beating chance could still, in principle, be
a fluke; a one-sample Wilcoxon signed-rank test against the chance
reference (treating each subject's accuracy as one independent
observation) is the standard non-parametric group-level test for this.

Uses the per-subject CV accuracies already computed by
classify_all_subjects.py (eeg_data/exports/classification_summary.json).
LDA is used as the single consistent classifier for the formal test
(picking each subject's best-of-4 model would inflate significance via
selection bias); best-of-4 is reported alongside as a supplementary,
more optimistic figure.
"""

import json
import numpy as np
from scipy.stats import wilcoxon, ttest_1samp

SUMMARY_PATH = "eeg_data/exports/classification_summary.json"
CHANCE = 0.5

EXCLUDED = ["S08"]            # at-chance, recommended exclude (see QC report)
BORDERLINE = ["S07"]          # borderline — reported both ways


def run_test(accs: dict, label: str):
    subs = sorted(accs)
    vals = np.array([accs[s] for s in subs])
    diffs = vals - CHANCE

    print(f"\n  {label}  (n={len(subs)})")
    print(f"  Subjects: {subs}")
    print(f"  Accuracies: {[f'{v*100:.1f}%' for v in vals]}")
    print(f"  Mean: {vals.mean()*100:.1f}%   Median: {np.median(vals)*100:.1f}%   "
          f"SD: {vals.std(ddof=1)*100:.1f}%")

    if len(subs) < 2 or np.all(diffs == 0):
        print("  Not enough variation for a test.")
        return None

    try:
        w_stat, p_w = wilcoxon(diffs, alternative="greater")
    except ValueError as e:
        print(f"  Wilcoxon failed: {e}")
        w_stat, p_w = np.nan, np.nan

    t_stat, p_t = ttest_1samp(vals, CHANCE, alternative="greater")

    print(f"  Wilcoxon signed-rank (vs chance=50%, one-sided): W={w_stat:.1f}, p={p_w:.4f} "
          f"{'*** significant' if p_w < 0.05 else 'not significant'}")
    print(f"  Paired t-test (supplementary, parametric):       t={t_stat:.2f}, p={p_t:.4f}")

    return {"subjects": subs, "accuracies": vals.tolist(), "mean": float(vals.mean()),
            "median": float(np.median(vals)), "wilcoxon_W": float(w_stat),
            "wilcoxon_p": float(p_w), "ttest_t": float(t_stat), "ttest_p": float(p_t)}


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    lda_accs  = {s: r["all_results"]["LDA"]["mean"] for s, r in summary.items() if "all_results" in r}
    best_accs = {s: r["accuracy_mean"] for s, r in summary.items() if "accuracy_mean" in r}

    results = {}
    print("=" * 70)
    print("  GROUP-LEVEL STATISTICAL TEST: accuracy vs chance, across subjects")
    print("=" * 70)

    results["lda_all11"] = run_test(lda_accs, "LDA accuracy — ALL 11 subjects")

    kept = {s: v for s, v in lda_accs.items() if s not in EXCLUDED}
    results["lda_kept10"] = run_test(kept, "LDA accuracy — excluding S08 (at-chance fault)")

    kept_strict = {s: v for s, v in lda_accs.items() if s not in EXCLUDED + BORDERLINE}
    results["lda_kept9_strict"] = run_test(kept_strict, "LDA accuracy — excluding S08 + S07 (strict)")

    results["best_all11"] = run_test(best_accs, "Best-of-4 accuracy — ALL 11 subjects (supplementary, optimistic)")

    out_path = "eeg_data/exports/group_level_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
