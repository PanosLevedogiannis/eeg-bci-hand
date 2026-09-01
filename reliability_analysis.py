"""
=============================================================================
  Reliability Validation — inMoov EEG MI BCI Pipeline
=============================================================================
  Adds the statistical rigor a thesis defense needs on top of classify.py:

    1. Permutation test (1000 shuffles) — null distribution vs real accuracy,
       using the SAME leave-one-run-out CV as the real fit (not a different
       scheme — otherwise the null distribution wouldn't be comparable).
    2. Cohen's Kappa — chance-corrected agreement (kappa > 0.2 above chance,
       > 0.4 moderate, > 0.6 good).
    3. Per-class F1 (MI vs REST) — catches a classifier that just always
       predicts one class.
    4. Run-by-run consistency — leave-one-run-out CV doubles as both the
       evaluation scheme AND the run-trend diagnostic: train on 3 runs,
       test on the held-out run, rotate across all 4 runs. A declining
       trend points at fatigue/drift within the session.
    5. Leave-one-run-out CV (not random k-fold) — standard in BCI literature
       because random k-fold lets temporally-adjacent trials leak across
       train/test, inflating accuracy. Run boundaries are recovered from
       epochs.selection (original chronological event index // 80trials/run)
       since each run was recorded as a contiguous block of 40 MI + 40 REST.

  Reuses the existing pipeline rather than reinventing it:
    - erd_analysis.find_fif / load_and_preprocess  (raw QC + epoching)
    - classify.prepare_data                        (8-30Hz filter + crop + X/y)

  Usage:
    python reliability_analysis.py --dataset "/path/to/inMoov_Dataset" --perms 1000
=============================================================================
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

mne.set_log_level("ERROR")

from erd_analysis import find_fif, load_and_preprocess
from classify import prepare_data, N_CSP_COMPONENTS

DATASET_DIR  = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset 4/inMoov_Dataset"
FIG_DIR      = "eeg_data/figures"
EXPORT_PATH  = "eeg_data/exports/reliability_summary.json"
N_PERMS      = 1000
TRIALS_PER_RUN = 80   # protocol: 4 runs x (40 MI + 40 REST)

# Verdict thresholds
ACC_PASS   = 0.60
KAPPA_PASS = 0.2
P_SIG      = 0.05
TREND_THRESHOLD = 0.05   # total run1->run4 predicted change (accuracy fraction)

DARK_BG, COL_PANEL, COL_BORDER, COL_GREY = "#0A0C14", "#12162A", "#283256", "#788092"
COL_MI, COL_REST, COL_SIG, COL_WARN = "#00DC82", "#3C8CFF", "#FF5A50", "#FFD700"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": COL_PANEL,
    "axes.edgecolor": COL_BORDER, "axes.labelcolor": "#C8CDE8",
    "axes.titlecolor": "#F0F5FF", "xtick.color": COL_GREY,
    "ytick.color": COL_GREY, "text.color": "#F0F5FF",
    "grid.color": COL_BORDER, "grid.linewidth": 0.5,
    "font.family": "monospace", "figure.dpi": 120,
})


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {name}")


# ─────────────────────────────────────────────
#  LEAVE-ONE-RUN-OUT CV
# ─────────────────────────────────────────────

def get_run_ids(epochs: mne.Epochs) -> np.ndarray:
    """
    epochs.selection holds each kept epoch's index into the ORIGINAL
    chronological event list (0..319) — unaffected by artifact rejection.
    Each run is a contiguous block of 80 trials (40 MI + 40 REST), recorded
    in order, so original_index // 80 recovers the run number even when
    rejection has dropped a different number of trials per run.
    """
    return (epochs.selection // TRIALS_PER_RUN) + 1   # 1..4


def make_loro_splits(run_ids: np.ndarray):
    runs = sorted(set(run_ids.tolist()))
    splits = []
    for r in runs:
        test_idx  = np.where(run_ids == r)[0]
        train_idx = np.where(run_ids != r)[0]
        splits.append((train_idx, test_idx))
    return splits, runs


def build_csp_lda():
    return Pipeline([
        ("csp", CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])


# ─────────────────────────────────────────────
#  PER-SUBJECT RELIABILITY ANALYSIS
# ─────────────────────────────────────────────

def classify_trend(slope_per_run: float) -> str:
    total_change = slope_per_run * 3   # run1 -> run4 span
    if total_change > TREND_THRESHOLD:
        return "improving"
    if total_change < -TREND_THRESHOLD:
        return "declining"
    return "stable"


def classify_verdict(acc: float, kappa: float, p_value: float) -> str:
    if p_value >= P_SIG:
        return "FAIL"            # not statistically distinguishable from chance
    if acc > ACC_PASS and kappa > KAPPA_PASS:
        return "PASS"
    return "BORDERLINE"          # significant, but below the practical accuracy/kappa bar


def run_subject(subj_dir: str, subj_id: str, n_perms: int) -> dict:
    fif_path = find_fif(subj_dir)
    epochs, qc = load_and_preprocess(fif_path, subj_id)

    run_ids = get_run_ids(epochs)
    X, y, label_map = prepare_data(epochs)
    assert len(run_ids) == len(y), "run_ids / X length mismatch — rejection desync"

    splits, runs = make_loro_splits(run_ids)
    pipe = build_csp_lda()

    # --- Leave-one-run-out: per-run accuracy + pooled out-of-fold predictions ---
    y_pred  = np.zeros_like(y)
    run_acc = {}
    for (train_idx, test_idx), r in zip(splits, runs):
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_pred[test_idx] = pred
        run_acc[r] = float(accuracy_score(y[test_idx], pred))

    overall_acc = float(accuracy_score(y, y_pred))
    kappa = float(cohen_kappa_score(y, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], zero_division=0)
    f1_per_class = {label_map[0]: float(f1[0]), label_map[1]: float(f1[1])}

    # --- Permutation test, SAME leave-one-run-out splits ---
    real_score, perm_scores, p_value = permutation_test_score(
        pipe, X, y, cv=splits, n_permutations=n_perms,
        scoring="accuracy", random_state=42, n_jobs=-1)
    pct95 = float(np.percentile(perm_scores, 95))

    # --- Run trend ---
    run_nums = np.array(sorted(run_acc))
    accs_in_order = np.array([run_acc[r] for r in run_nums])
    slope, intercept = np.polyfit(run_nums, accs_in_order, 1)
    trend = classify_trend(float(slope))

    verdict = classify_verdict(overall_acc, kappa, float(p_value))

    print(f"  {subj_id}: acc={overall_acc*100:.1f}%  kappa={kappa:.3f}  "
          f"F1[{label_map[0]}]={f1_per_class[label_map[0]]:.3f}  "
          f"F1[{label_map[1]}]={f1_per_class[label_map[1]]:.3f}  "
          f"p={p_value:.4f}  trend={trend}  -> {verdict}")

    plot_run_trend(subj_id, run_nums, accs_in_order, trend, real_score, p_value)

    return {
        "subject": subj_id,
        "qc": qc,
        "accuracy": overall_acc,
        "kappa": kappa,
        "f1_per_class": f1_per_class,
        "label_map": label_map,
        "run_accuracy": {str(int(r)): a for r, a in zip(run_nums, accs_in_order)},
        "run_trend": trend,
        "run_trend_slope_per_run": float(slope),
        "permutation": {
            "n_perms": n_perms,
            "real_score": float(real_score),
            "p_value": float(p_value),
            "null_mean": float(perm_scores.mean()),
            "null_95th_pct": pct95,
            "above_95th_pct": bool(real_score > pct95),
        },
        "verdict": verdict,
    }


# ─────────────────────────────────────────────
#  PER-SUBJECT FIGURE — RUN-BY-RUN TREND
# ─────────────────────────────────────────────

def plot_run_trend(subj_id, run_nums, accs, trend, real_score, p_value):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    trend_col = {"improving": COL_MI, "declining": COL_SIG, "stable": COL_REST}[trend]

    ax.plot(run_nums, accs * 100, "o-", color=trend_col, lw=2.2, ms=9,
            label=f"Run accuracy (trend: {trend})")
    slope, intercept = np.polyfit(run_nums, accs, 1)
    fit_x = np.array([run_nums[0], run_nums[-1]])
    ax.plot(fit_x, (slope * fit_x + intercept) * 100, "--", color=COL_WARN,
            lw=1.3, alpha=0.8, label="Linear fit")

    ax.axhline(50, color=COL_GREY, lw=1.0, ls=":", alpha=0.7, label="Chance (50%)")
    ax.axhline(real_score * 100, color="white", lw=1.0, ls="-.", alpha=0.5,
              label=f"Overall LORO acc: {real_score*100:.1f}%")

    sig_str = f"p={p_value:.4f}" if p_value >= 0.0001 else "p<0.0001"
    ax.text(0.97, 0.04, sig_str, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=COL_MI if p_value < 0.05 else COL_SIG,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COL_PANEL,
                      edgecolor=COL_BORDER, alpha=0.9))

    ax.set_xticks(run_nums)
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(20, 100)
    ax.set_title(f"Run-by-Run Consistency (Leave-One-Run-Out) — {subj_id}")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    savefig(fig, f"reliability_{subj_id}_run_trend.png")


# ─────────────────────────────────────────────
#  SUMMARY FIGURE — ALL SUBJECTS
# ─────────────────────────────────────────────

def plot_summary(results: dict):
    subs = list(results.keys())
    n = len(subs)
    x = np.arange(n)

    accs    = [results[s]["accuracy"] * 100 for s in subs]
    kappas  = [results[s]["kappa"] for s in subs]
    pvals   = [results[s]["permutation"]["p_value"] for s in subs]
    verdict_col = {"PASS": COL_MI, "BORDERLINE": COL_WARN, "FAIL": COL_SIG}
    cols    = [verdict_col[results[s]["verdict"]] for s in subs]

    f1_labels = results[subs[0]]["label_map"]
    f1_a = [results[s]["f1_per_class"][f1_labels[0]] for s in subs]
    f1_b = [results[s]["f1_per_class"][f1_labels[1]] for s in subs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Reliability Summary — All Subjects (Leave-One-Run-Out CV)", fontsize=13)

    ax = axes[0]
    ax.bar(x, accs, color=cols, alpha=0.9)
    ax.axhline(50, color="red", lw=1.2, ls="--", alpha=0.7, label="Chance")
    ax.axhline(ACC_PASS * 100, color=COL_WARN, lw=1.0, ls=":", alpha=0.7, label="PASS bar (60%)")
    ax.set_xticks(x); ax.set_xticklabels(subs)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("LORO Accuracy\n(green=PASS, yellow=BORDERLINE, red=FAIL)")
    ax.set_ylim(30, 100); ax.grid(True, axis="y"); ax.legend(fontsize=7)
    for i, (a, s) in enumerate(zip(accs, subs)):
        ax.text(i, a + 1.5, f"{a:.0f}%", ha="center", fontsize=7.5, color="white")

    ax = axes[1]
    ax.bar(x, kappas, color=cols, alpha=0.9)
    ax.axhline(0.2, color=COL_WARN, lw=1.0, ls=":", alpha=0.7, label="κ=0.2 (above chance)")
    ax.axhline(0.4, color="orange", lw=1.0, ls=":", alpha=0.7, label="κ=0.4 (moderate)")
    ax.axhline(0.6, color=COL_MI,  lw=1.0, ls=":", alpha=0.7, label="κ=0.6 (good)")
    ax.set_xticks(x); ax.set_xticklabels(subs)
    ax.set_ylabel("Cohen's Kappa"); ax.set_title("Chance-Corrected Agreement")
    ax.grid(True, axis="y"); ax.legend(fontsize=7)

    ax = axes[2]
    w = 0.35
    ax.bar(x - w/2, f1_a, w, color=COL_MI,   alpha=0.85, label=f1_labels[0])
    ax.bar(x + w/2, f1_b, w, color=COL_REST, alpha=0.85, label=f1_labels[1])
    ax.set_xticks(x); ax.set_xticklabels(subs)
    ax.set_ylabel("F1-score"); ax.set_title("Per-Class F1 (checks classifier bias)")
    ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "reliability_summary.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--perms", default=N_PERMS, type=int)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)

    subject_dirs = sorted(glob.glob(os.path.join(args.dataset, "S*")))
    if not subject_dirs:
        print(f"✗ No subjects found in {args.dataset}")
        return

    print(f"\n{'='*70}")
    print(f"  Reliability Analysis  |  {len(subject_dirs)} subject(s)  |  "
          f"{args.perms} permutations  |  Leave-One-Run-Out CV")
    print(f"{'='*70}\n")

    results = {}
    for subj_dir in subject_dirs:
        subj_id = os.path.basename(subj_dir)
        try:
            results[subj_id] = run_subject(subj_dir, subj_id, args.perms)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[subj_id] = {"error": str(e)}

    ok_results = {s: r for s, r in results.items() if "error" not in r}
    if len(ok_results) >= 2:
        plot_summary(ok_results)

    with open(EXPORT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Final clean summary table ---
    print(f"\n{'='*100}")
    print("  RELIABILITY SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Subj':<6} {'Acc%':>6} {'Kappa':>7} {'F1-MI':>7} {'F1-REST':>8} "
          f"{'p-value':>9} {'Trend':>11} {'Verdict':>11}")
    print(f"  {'-'*96}")
    for subj, r in results.items():
        if "error" in r:
            print(f"  {subj:<6}  FAILED: {r['error']}")
            continue
        lm = r["label_map"]
        f1_mi   = r["f1_per_class"].get("MI",   list(r["f1_per_class"].values())[0])
        f1_rest = r["f1_per_class"].get("REST", list(r["f1_per_class"].values())[-1])
        print(f"  {subj:<6} {r['accuracy']*100:>5.1f}% {r['kappa']:>7.3f} "
              f"{f1_mi:>7.3f} {f1_rest:>8.3f} {r['permutation']['p_value']:>9.4f} "
              f"{r['run_trend']:>11} {r['verdict']:>11}")
    print(f"{'='*100}\n")
    print(f"Saved -> {EXPORT_PATH}")


if __name__ == "__main__":
    main()
