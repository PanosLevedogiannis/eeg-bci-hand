"""
=============================================================================
  Multi-Subject Analysis — OpenBCI Cyton recordings
=============================================================================
  Τρέχει το pipeline σε όλα τα FIF αρχεία του eeg_data/ και παράγει:
    - Πίνακας accuracy ανά subject και classifier
    - Mean ± std across subjects
    - Summary bar chart → eeg_data/figures/multi_subject_comparison.png
    - JSON report       → eeg_data/models/multi_subject_report.json

  Usage:
    python multi_subject_analysis.py

  Απαιτεί: τουλάχιστον ένα *_raw.fif στο eeg_data/
=============================================================================
"""

import numpy as np
import json, os, sys, glob, warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mne
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from preprocess import apply_filters, apply_car, apply_ica, extract_epochs, save_epochs, DATA_DIR
from classify   import run_classification

FIG_DIR   = os.path.join(DATA_DIR, "figures")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DARK_BG   = "#0A0C14"
COL_PANEL = "#12162A"
COL_MI    = "#00DC82"
COL_REST  = "#FF5A50"
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
    "SVM (RBF)":      COL_MI,
    "SVM (Linear)":   COL_CYAN,
    "Riemannian MDM": "#FFD700",
}


# ─────────────────────────────────────────────
#  FIND ALL FIF RECORDINGS
# ─────────────────────────────────────────────

def find_recordings() -> list:
    """Returns list of (subject_id, fif_path) sorted by subject_id."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_raw.fif")),
                   key=os.path.getmtime)
    recordings = []
    for f in files:
        basename = os.path.basename(f)
        # e.g. EEG-RAW_S01_20260424_141809_raw.fif → subject = S01
        parts = basename.replace("EEG-RAW_", "").split("_")
        subject_id = parts[0]
        recordings.append((subject_id, f))
    return recordings


# ─────────────────────────────────────────────
#  RUN ONE SUBJECT
# ─────────────────────────────────────────────

def run_subject(subject_id: str, fif_path: str, idx: int, total: int):
    print(f"\n{'─'*55}")
    print(f"  Subject {subject_id}  ({idx}/{total})  ← {os.path.basename(fif_path)}")
    print(f"{'─'*55}")

    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

        # Remap numeric markers → MI / REST
        raw.annotations.rename({"1": "MI", "2": "REST"})

        raw_filt  = apply_filters(raw)
        raw_car   = apply_car(raw_filt)
        raw_clean = apply_ica(raw_car)

        events, _ = mne.events_from_annotations(
            raw_clean, event_id={"MI": 1, "REST": 2}, verbose=False
        )

        if len(events) == 0:
            raise ValueError("No MI/REST events found in annotations")

        epochs = extract_epochs(raw_clean, events, {"MI": 1, "REST": 2})
        save_epochs(epochs, subject_id=subject_id, source="cyton")

        _, results = run_classification(epochs, subject_id=subject_id,
                                        source="cyton")
        return subject_id, results, None

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return subject_id, None, str(e)


# ─────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary_table(all_results: dict):
    valid = {sid: r for sid, r in all_results.items() if r is not None}
    if not valid:
        print("  No valid results.")
        return {}

    classifiers = list(next(iter(valid.values())).keys())
    col_w = 18

    print(f"\n{'='*70}")
    print(f"  MULTI-SUBJECT RESULTS — MI vs REST (Cyton)")
    print(f"{'='*70}")

    header = f"  {'Subject':<12}" + "".join(f"{c:>{col_w}}" for c in classifiers)
    print(header)
    print(f"  {'─'*10}" + ("─" * col_w) * len(classifiers))

    all_accs = {clf: [] for clf in classifiers}
    for sid, results in valid.items():
        row = f"  {sid:<12}"
        for clf in classifiers:
            acc = results[clf]["mean"] * 100
            all_accs[clf].append(acc)
            row += f"{acc:>{col_w-1}.1f}%"
        best = max(classifiers, key=lambda c: results[c]["mean"])
        print(row + f"   ← {best}")

    print(f"  {'─'*10}" + ("─" * col_w) * len(classifiers))
    mean_row = f"  {'MEAN':<12}"
    std_row  = f"  {'STD':<12}"
    for clf in classifiers:
        m = np.mean(all_accs[clf])
        s = np.std(all_accs[clf])
        mean_row += f"{m:>{col_w-1}.1f}%"
        std_row  += f"{s:>{col_w-1}.1f}%"
    print(mean_row)
    print(std_row)
    print(f"{'='*70}\n")

    best_clf  = max(classifiers, key=lambda c: np.mean(all_accs[c]))
    best_mean = np.mean(all_accs[best_clf])
    best_std  = np.std(all_accs[best_clf])
    n_above   = sum(1 for a in all_accs[best_clf] if a > 55)
    print(f"  ★  Best classifier : {best_clf}")
    print(f"  ★  Mean accuracy   : {best_mean:.1f}% ± {best_std:.1f}%")
    print(f"  ★  Above chance (>55%): {n_above} / {len(valid)} subjects\n")

    return all_accs


# ─────────────────────────────────────────────
#  SUMMARY FIGURE
# ─────────────────────────────────────────────

def plot_summary(all_results: dict, all_accs: dict):
    valid = {sid: r for sid, r in all_results.items() if r is not None}
    if not valid:
        return

    classifiers = list(next(iter(valid.values())).keys())
    subject_ids = list(valid.keys())
    n_sub       = len(subject_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Multi-Subject Analysis — MI vs REST  |  OpenBCI Cyton  |  N={n_sub} subjects\n"
        "Pipeline: Bandpass + CAR + ICA + CSP + classifiers",
        fontsize=11, y=1.02
    )

    # Per-subject lines
    x = np.arange(n_sub)
    for clf_name in classifiers:
        accs = [valid[sid][clf_name]["mean"] * 100 for sid in subject_ids]
        col  = CLASSIFIER_COLORS.get(clf_name, COL_GREY)
        ax1.plot(x, accs, "o-", color=col, linewidth=1.8,
                 markersize=7, label=clf_name, alpha=0.9)

    ax1.axhline(50, color=COL_GREY, lw=0.8, ls=":", alpha=0.7, label="Chance (50%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subject_ids, rotation=45, fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(30, 105)
    ax1.set_title("Per-Subject Accuracy", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # Mean ± std bars
    means = [np.mean(all_accs[c]) for c in classifiers]
    stds  = [np.std(all_accs[c])  for c in classifiers]
    cols  = [CLASSIFIER_COLORS.get(c, COL_GREY) for c in classifiers]

    xi   = np.arange(len(classifiers))
    bars = ax2.bar(xi, means, yerr=stds, color=cols, alpha=0.85,
                   capsize=6, error_kw={"lw": 1.5, "color": "white"}, width=0.55)

    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + s + 1.5,
                 f"{m:.1f}%", ha="center", va="bottom",
                 fontsize=10, color="white", fontweight="bold")

    ax2.axhline(50, color=COL_GREY, lw=0.8, ls=":", alpha=0.7, label="Chance (50%)")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(classifiers, rotation=15, fontsize=9)
    ax2.set_ylabel("Mean Accuracy (%)")
    ax2.set_ylim(30, 110)
    ax2.set_title(f"Mean ± Std  (N={n_sub})", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, axis="y")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "multi_subject_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Figure saved → {path}")


# ─────────────────────────────────────────────
#  SAVE JSON REPORT
# ─────────────────────────────────────────────

def save_report(all_results: dict, all_accs: dict):
    valid     = {sid: r for sid, r in all_results.items() if r is not None}
    failed    = [sid for sid, r in all_results.items() if r is None]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp":        timestamp,
        "dataset":          "OpenBCI Cyton — MI vs REST",
        "n_subjects":       len(valid),
        "subjects":         list(valid.keys()),
        "failed_subjects":  failed,
        "per_subject": {
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


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    recordings = find_recordings()

    if not recordings:
        print(f"\n  ✗ No *_raw.fif files found in {DATA_DIR}/")
        print(f"    Run eeg_mi_paradigm.py first to record data.\n")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Multi-Subject Analysis  |  {len(recordings)} recording(s) found")
    print(f"{'='*55}")
    for sid, path in recordings:
        print(f"  {sid:<12} ← {os.path.basename(path)}")

    all_results = {}
    for i, (sid, path) in enumerate(recordings, 1):
        sid_out, results, error = run_subject(sid, path, i, len(recordings))
        all_results[sid_out] = results

    all_accs = print_summary_table(all_results)

    if all_accs:
        plot_summary(all_results, all_accs)
        save_report(all_results, all_accs)

    print("  Multi-subject analysis complete.\n")
