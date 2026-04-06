"""
=============================================================================
  Multi-Subject Validation — PhysioNet EEG Motor Imagery Dataset
=============================================================================
  Runs the full pipeline on N PhysioNet subjects and produces:
    - Per-subject accuracy table (all classifiers)
    - Mean ± std across subjects
    - Summary bar chart saved to eeg_data/figures/
    - JSON report saved to eeg_data/models/

  This demonstrates that the pipeline generalises across subjects —
  a key requirement for a BCI thesis.

  Usage:
    python multi_subject_analysis.py
=============================================================================
"""

import numpy as np
import json
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mne
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from load_data   import load_physionet
from preprocess  import run_preprocessing, DATA_DIR
from classify    import run_classification

FIG_DIR   = os.path.join(DATA_DIR, "figures")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

SUBJECTS      = list(range(1, 11))   # PhysioNet subjects 1-10
PHYSIONET_RUNS = [6, 10, 14]         # left/right fist imagery

DARK_BG   = "#0A0C14"
COL_PANEL = "#12162A"
COL_OPEN  = "#00DC82"
COL_CLOSE = "#FF5A50"
COL_BLUE  = "#3C8CFF"
COL_CYAN  = "#00D2C8"
COL_GREY  = "#788092"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   COL_PANEL,
    "axes.edgecolor":   "#283256",
    "axes.labelcolor":  "#C8CDE8",
    "axes.titlecolor":  "#F0F5FF",
    "xtick.color":      COL_GREY,
    "ytick.color":      COL_GREY,
    "text.color":       "#F0F5FF",
    "grid.color":       "#283256",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "figure.dpi":       120,
})

CLASSIFIER_COLORS = {
    "LDA":            COL_BLUE,
    "SVM (RBF)":      COL_OPEN,
    "SVM (Linear)":   COL_CYAN,
    "Riemannian MDM": "#FFD700",
}


# ─────────────────────────────────────────────
#  RUN ONE SUBJECT
# ─────────────────────────────────────────────

def run_subject(subject_num: int):
    subject_id = f"P{subject_num:03d}"
    print(f"\n{'─'*50}")
    print(f"  Subject {subject_id}  ({SUBJECTS.index(subject_num)+1}/{len(SUBJECTS)})")
    print(f"{'─'*50}")

    try:
        raw    = load_physionet(subject=subject_num, runs=PHYSIONET_RUNS)
        epochs = run_preprocessing(raw, subject_id=subject_id, source="physionet")
        _, results = run_classification(epochs, subject_id=subject_id,
                                        source="physionet")
        return subject_id, results, None
    except Exception as e:
        print(f"  ✗ Subject {subject_id} failed: {e}")
        return subject_id, None, str(e)


# ─────────────────────────────────────────────
#  SUMMARY FIGURE
# ─────────────────────────────────────────────

def plot_summary(all_results: dict):
    """
    Two-panel figure:
      Left:  per-subject accuracy for each classifier (line plot)
      Right: mean ± std bar chart across subjects
    """
    valid = {sid: r for sid, r in all_results.items() if r is not None}
    if not valid:
        print("  No valid results to plot.")
        return

    classifiers = list(next(iter(valid.values())).keys())
    subject_ids = list(valid.keys())
    n_sub       = len(subject_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Multi-Subject Validation — PhysioNet EEG Motor Imagery\n"
        f"Subjects: {subject_ids[0]}–{subject_ids[-1]}  |  "
        f"Task: Left vs Right Fist Imagery  |  Pipeline: CSP + classifiers",
        fontsize=12, y=1.02
    )

    # ── Left panel: per-subject lines ──
    x = np.arange(n_sub)
    for clf_name in classifiers:
        accs = [valid[sid][clf_name]["mean"] * 100 for sid in subject_ids]
        col  = CLASSIFIER_COLORS.get(clf_name, COL_GREY)
        ax1.plot(x, accs, "o-", color=col, linewidth=1.8,
                 markersize=6, label=clf_name, alpha=0.9)

    ax1.axhline(50, color=COL_GREY, linewidth=0.8, linestyle=":",
                alpha=0.7, label="Chance (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subject_ids, rotation=45, fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(40, 105)
    ax1.set_title("Per-Subject Accuracy", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(True, alpha=0.4)

    # ── Right panel: mean ± std bars ──
    means = []
    stds  = []
    cols  = []
    for clf_name in classifiers:
        accs = [valid[sid][clf_name]["mean"] * 100 for sid in subject_ids]
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        cols.append(CLASSIFIER_COLORS.get(clf_name, COL_GREY))

    xi = np.arange(len(classifiers))
    bars = ax2.bar(xi, means, yerr=stds, color=cols, alpha=0.85,
                   capsize=6, error_kw={"linewidth": 1.5, "color": "white"},
                   width=0.55)

    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + 1.5,
                 f"{mean:.1f}%", ha="center", va="bottom",
                 fontsize=10, color="white", fontweight="bold")

    ax2.axhline(50, color=COL_GREY, linewidth=0.8, linestyle=":",
                alpha=0.7, label="Chance (50%)")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(classifiers, rotation=15, fontsize=9)
    ax2.set_ylabel("Mean Accuracy (%)")
    ax2.set_ylim(40, 110)
    ax2.set_title(f"Mean ± Std  (N={n_sub} subjects)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, axis="y")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "05_multi_subject_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Figure saved → {path}")


# ─────────────────────────────────────────────
#  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary_table(all_results: dict):
    valid = {sid: r for sid, r in all_results.items() if r is not None}
    if not valid:
        return

    classifiers = list(next(iter(valid.values())).keys())
    col_w = 18

    print(f"\n{'='*70}")
    print(f"  MULTI-SUBJECT RESULTS SUMMARY")
    print(f"{'='*70}")

    header = f"  {'Subject':<10}" + "".join(f"{c:>{col_w}}" for c in classifiers)
    print(header)
    print(f"  {'─'*8}" + ("─" * col_w) * len(classifiers))

    all_accs = {clf: [] for clf in classifiers}
    for sid, results in valid.items():
        row = f"  {sid:<10}"
        for clf in classifiers:
            acc = results[clf]["mean"] * 100
            all_accs[clf].append(acc)
            row += f"{acc:>{col_w-1}.1f}%"
        best = max(classifiers, key=lambda c: results[c]["mean"])
        print(row + f"   ← best: {best}")

    print(f"  {'─'*8}" + ("─" * col_w) * len(classifiers))
    mean_row = f"  {'MEAN':<10}"
    std_row  = f"  {'STD':<10}"
    for clf in classifiers:
        m = np.mean(all_accs[clf])
        s = np.std(all_accs[clf])
        mean_row += f"{m:>{col_w-1}.1f}%"
        std_row  += f"{s:>{col_w-1}.1f}%"
    print(mean_row)
    print(std_row)
    print(f"{'='*70}\n")

    # Overall best classifier
    best_clf = max(classifiers, key=lambda c: np.mean(all_accs[c]))
    best_mean = np.mean(all_accs[best_clf])
    best_std  = np.std(all_accs[best_clf])
    print(f"  ★  Best classifier: {best_clf}")
    print(f"  ★  Mean accuracy  : {best_mean:.1f}% ± {best_std:.1f}%")
    print(f"  ★  Subjects above chance (>55%): "
          f"{sum(1 for a in all_accs[best_clf] if a > 55)} / {len(valid)}\n")

    return all_accs


# ─────────────────────────────────────────────
#  SAVE JSON REPORT
# ─────────────────────────────────────────────

def save_report(all_results: dict, all_accs: dict):
    valid    = {sid: r for sid, r in all_results.items() if r is not None}
    failed   = [sid for sid, r in all_results.items() if r is None]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp":       timestamp,
        "dataset":         "PhysioNet EEG Motor Imagery",
        "task":            "Left vs Right Fist Imagery (Runs 6,10,14)",
        "n_subjects":      len(valid),
        "subjects":        list(valid.keys()),
        "failed_subjects": failed,
        "per_subject":     {
            sid: {clf: {"mean": r[clf]["mean"], "std": r[clf]["std"]}
                  for clf in r}
            for sid, r in valid.items()
        },
        "aggregate": {
            clf: {
                "mean": float(np.mean(all_accs[clf])),
                "std":  float(np.std(all_accs[clf])),
                "min":  float(np.min(all_accs[clf])),
                "max":  float(np.max(all_accs[clf])),
            }
            for clf in all_accs
        },
    }

    path = os.path.join(MODEL_DIR, f"multi_subject_report_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report saved → {path}")
    return report


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_multi_subject_analysis(subjects: list = SUBJECTS):
    print(f"\n{'='*55}")
    print(f"  Multi-Subject Analysis  |  {len(subjects)} subjects")
    print(f"  PhysioNet runs: {PHYSIONET_RUNS}")
    print(f"{'='*55}")

    all_results = {}
    for subj_num in subjects:
        sid, results, error = run_subject(subj_num)
        all_results[sid] = results

    # Print summary table
    all_accs = print_summary_table(all_results)

    if all_accs:
        # Plot comparison figure
        plot_summary(all_results)

        # Save JSON report
        save_report(all_results, all_accs)

    print("  Multi-subject analysis complete.\n")
    return all_results


if __name__ == "__main__":
    run_multi_subject_analysis(subjects=SUBJECTS)
