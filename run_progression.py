"""
=============================================================================
  Run-by-Run Progression Analysis — inMoov Dataset
=============================================================================
  Splits each session into 4 runs using break markers (code 40/41).
  Per run computes:
    - Epoch retention
    - Peak Mu ERD (C3 + C4)
    - CSP+LDA accuracy (5-fold CV)

  Shows whether subjects improve, plateau, or fatigue across runs.

  Usage:
    python run_progression.py
    python run_progression.py --dataset /path/to/inMoov_Dataset
=============================================================================
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, glob, argparse, warnings
warnings.filterwarnings("ignore")

from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset 2"
FIG_DIR     = "eeg_data/figures"
os.makedirs(FIG_DIR, exist_ok=True)

EVENT_ID = {"MI": 1, "REST": 2}
TMIN, TMAX = -2.0, 6.0
BASELINE   = (-1.5, -0.5)

DARK_BG  = "#0A0C14"
COL_PANEL= "#12162A"
COL_GREY = "#788092"

SUBJECT_COLORS = {
    "S01": "#00DC82",
    "S02": "#3C8CFF",
    "S03": "#FF5A50",
    "S04": "#FFD700",
}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor": COL_PANEL,
    "axes.edgecolor":  "#283256", "axes.labelcolor": "#C8CDE8",
    "axes.titlecolor": "#F0F5FF", "xtick.color":     COL_GREY,
    "ytick.color":     COL_GREY,  "text.color":      "#F0F5FF",
    "grid.color":      "#283256", "grid.linewidth":  0.5,
    "font.family":     "monospace","figure.dpi":      120,
})


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def find_fif(subject_dir):
    fifs = glob.glob(os.path.join(subject_dir, "*_raw.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_raw.fif in {subject_dir}")
    return sorted(fifs)[-1]


def load_raw(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw, _ = mne.set_eeg_reference(raw, "average", projection=False,
                                   verbose=False)
    return raw


# ─────────────────────────────────────────────
#  SPLIT INTO RUNS
# ─────────────────────────────────────────────

def get_run_boundaries(raw):
    """
    Use break_start (code 40) annotations to find run end times.
    Returns list of (t_start, t_end) tuples for each run.
    """
    break_starts = []
    for desc, onset in zip(raw.annotations.description,
                           raw.annotations.onset):
        if desc == "40":
            break_starts.append(float(onset))

    # Find baseline end (code 13 = eyes_closed_end)
    session_start = 0.0
    for desc, onset in zip(raw.annotations.description,
                           raw.annotations.onset):
        if desc == "13":
            session_start = float(onset)
            break

    t_end = raw.times[-1]

    if not break_starts:
        # No breaks found — split evenly into 4
        total = t_end - session_start
        boundaries = [(session_start + i * total / 4,
                       session_start + (i+1) * total / 4)
                      for i in range(4)]
        return boundaries

    # Build run intervals: [session_start .. break1] [break_end .. break2] ...
    break_ends = []
    for desc, onset in zip(raw.annotations.description,
                           raw.annotations.onset):
        if desc == "41":
            break_ends.append(float(onset))

    boundaries = []
    run_start = session_start
    for bs, be in zip(break_starts, break_ends):
        boundaries.append((run_start, bs))
        run_start = be
    boundaries.append((run_start, t_end))   # last run

    return boundaries


def epochs_for_run(raw, t_start, t_end):
    """Extract and epoch the segment [t_start, t_end]."""
    raw_crop = raw.copy().crop(tmin=t_start, tmax=t_end)

    events, _ = mne.events_from_annotations(
        raw_crop, event_id={"1": 1, "2": 2}, verbose=False)

    if len(events) == 0:
        return None

    epochs = mne.Epochs(
        raw_crop, events, EVENT_ID,
        tmin=TMIN, tmax=TMAX,
        baseline=None,
        reject=dict(eeg=200e-6),
        preload=True, verbose=False,
    )
    if len(epochs) == 0:
        epochs = mne.Epochs(
            raw_crop, events, EVENT_ID,
            tmin=TMIN, tmax=TMAX,
            baseline=None, reject=None,
            preload=True, verbose=False,
        )
    return epochs


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def peak_mu_erd(epochs, ch):
    if ch not in epochs.ch_names:
        return np.nan
    sfreq = epochs.info["sfreq"]
    times = epochs.times
    nyq   = sfreq / 2.0
    b, a  = butter(4, [8/nyq, 12/nyq], btype="band")
    ep    = epochs["MI"]
    if len(ep) == 0:
        return np.nan
    idx   = ep.ch_names.index(ch)
    data  = ep.get_data()[:, idx, :]
    filt  = np.array([filtfilt(b, a, t) for t in data])
    power = filt ** 2
    ksz   = int(sfreq * 0.25)
    smooth= np.array([np.convolve(p, np.ones(ksz)/ksz, "same") for p in power])
    bl    = (times >= BASELINE[0]) & (times <= BASELINE[1])
    bl_m  = smooth[:, bl].mean(axis=1, keepdims=True)
    erd   = 100.0 * (smooth - bl_m) / (bl_m + 1e-12)
    t_mask= (times >= 0) & (times <= 4.0)
    return float(erd.mean(axis=0)[t_mask].min())


def classify_run(epochs):
    """5-fold CV CSP+LDA. Returns (mean_acc, std_acc) or (nan, nan)."""
    n_mi   = len(epochs["MI"])
    n_rest = len(epochs["REST"])
    if n_mi < 5 or n_rest < 5:
        return np.nan, np.nan

    t_mask = (epochs.times >= 0) & (epochs.times <= 4.0)
    ep_f   = epochs.copy().filter(8, 30, method="fir", verbose=False)
    X      = ep_f.get_data()[:, :, t_mask]
    y      = (epochs.events[:, 2] == 1).astype(int)

    n_splits = min(5, min(n_mi, n_rest))
    pipe = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


# ─────────────────────────────────────────────
#  PER-SUBJECT ANALYSIS
# ─────────────────────────────────────────────

def analyse_subject(subj_dir):
    subj = os.path.basename(subj_dir)
    fif  = find_fif(subj_dir)
    raw  = load_raw(fif)
    boundaries = get_run_boundaries(raw)

    print(f"  {subj}: {len(boundaries)} runs detected")

    runs = []
    for i, (t0, t1) in enumerate(boundaries, 1):
        ep = epochs_for_run(raw, t0, t1)
        if ep is None:
            print(f"    Run {i}: no epochs found, skipping")
            continue

        n_mi     = len(ep["MI"])   if "MI"   in ep.event_id else 0
        n_rest   = len(ep["REST"]) if "REST" in ep.event_id else 0
        n_total  = n_mi + n_rest
        ret      = n_total / 80 * 100   # 80 = 40 MI + 40 REST per run

        erd_c3   = peak_mu_erd(ep, "C3")
        erd_c4   = peak_mu_erd(ep, "C4")
        acc, std = classify_run(ep)

        print(f"    Run {i}: MI={n_mi}, REST={n_rest}, "
              f"ret={ret:.0f}%, ERD_C3={erd_c3:.1f}%, "
              f"acc={acc*100:.1f}%" if not np.isnan(acc) else
              f"    Run {i}: MI={n_mi}, REST={n_rest}, ret={ret:.0f}%")

        runs.append({
            "run": i, "n_mi": n_mi, "n_rest": n_rest,
            "retention": ret,
            "erd_c3": erd_c3, "erd_c4": erd_c4,
            "acc": acc, "acc_std": std,
        })

    return subj, runs


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

def plot_all_subjects(all_results):
    """
    3-row figure: ERD magnitude, epoch retention, accuracy — per run.
    One line per subject.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Run-by-Run Progression (Run 1 → 4)", fontsize=14, y=1.01)

    run_labels = ["Run 1", "Run 2", "Run 3", "Run 4"]
    x = np.arange(1, 5)

    metrics = [
        (0, "erd_c3", "Peak Mu ERD — C3 (% change)", True,  "ERD (%)"),
        (1, "retention", "Epoch Retention per Run (%)",  False, "Retention (%)"),
        (2, "acc",    "CSP+LDA Accuracy per Run (%)",   False, "Accuracy (%)"),
    ]

    for ax_idx, key, title, invert, ylabel in metrics:
        ax = axes[ax_idx]
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        for subj, runs in all_results.items():
            col   = SUBJECT_COLORS.get(subj, "#FFFFFF")
            r_x   = [r["run"] for r in runs]
            if key == "erd_c3":
                vals = [-r["erd_c3"] for r in runs]   # flip: bigger = more ERD
            elif key == "acc":
                vals = [r["acc"] * 100 if not np.isnan(r["acc"]) else np.nan
                        for r in runs]
            else:
                vals = [r[key] for r in runs]

            ax.plot(r_x, vals, "o-", color=col, lw=2.2,
                    markersize=7, label=subj)
            ax.fill_between(r_x, vals,
                            alpha=0.08, color=col)

        if key == "acc":
            ax.axhline(50, color="red", lw=1.0, ls="--",
                       alpha=0.6, label="Chance")
            ax.set_ylim(30, 100)
        if key == "erd_c3":
            ax.axhline(0, color=COL_GREY, lw=0.8, ls=":")

        ax.set_xticks(x)
        ax.set_xticklabels(run_labels)
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Run")
    fig.tight_layout()
    savefig(fig, "run_progression.png")


def plot_per_subject(all_results):
    """
    One subplot per subject, shows all 3 metrics together with dual axis.
    """
    subjects = list(all_results.keys())
    n = len(subjects)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("Per-Subject Run Progression", fontsize=13, y=1.01)

    for i, subj in enumerate(subjects):
        ax   = axes[i]
        col  = SUBJECT_COLORS.get(subj, "#FFFFFF")
        runs = all_results[subj]

        rx       = [r["run"]      for r in runs]
        erd_vals = [-r["erd_c3"]  for r in runs]
        ret_vals = [r["retention"] for r in runs]
        acc_vals = [r["acc"] * 100 if not np.isnan(r["acc"]) else np.nan
                    for r in runs]

        ax2 = ax.twinx()
        ax2.set_facecolor(COL_PANEL)

        l1, = ax.plot(rx, erd_vals, "o-", color="#FF5A50", lw=2.0,
                      markersize=7, label="Mu ERD C3 (mag)")
        l2, = ax.plot(rx, [-r["erd_c4"] for r in runs], "s--",
                      color="#FFA040", lw=1.5, markersize=6, label="Mu ERD C4")
        l3, = ax2.plot(rx, acc_vals, "^-", color=col, lw=2.2,
                       markersize=8, label="Accuracy %")
        l4  = ax2.bar(rx, ret_vals, alpha=0.15, color=col, label="Retention %")

        ax.axhline(0, color=COL_GREY, lw=0.8, ls=":")
        ax2.axhline(50, color="red", lw=1.0, ls="--", alpha=0.5)

        ax.set_title(subj, fontsize=12, color=col)
        ax.set_xlabel("Run")
        ax.set_ylabel("ERD magnitude (%)", color="#FF5A50")
        ax2.set_ylabel("Accuracy / Retention (%)", color=col)
        ax.set_xticks(rx)
        ax.set_xticklabels([f"R{r}" for r in rx])
        ax.grid(True)

        lines  = [l1, l2, l3]
        labels = [l.get_label() for l in lines] + ["Retention %"]
        ax.legend(lines + [l4], labels, fontsize=7, loc="upper left")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    savefig(fig, "run_progression_per_subject.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    args = parser.parse_args()

    subject_dirs = sorted(glob.glob(os.path.join(args.dataset, "S*")))
    if not subject_dirs:
        print(f"✗ No subject folders found in {args.dataset}")
        return

    print(f"\n{'='*60}")
    print(f"  Run-by-Run Progression  |  {len(subject_dirs)} subject(s)")
    print(f"{'='*60}\n")

    all_results = {}
    for subj_dir in subject_dirs:
        subj, runs = analyse_subject(subj_dir)
        if runs:
            all_results[subj] = runs

    print(f"\n─── Plotting ───────────────────────────────────────────")
    plot_all_subjects(all_results)
    plot_per_subject(all_results)

    # Console table
    print(f"\n{'='*65}")
    print(f"  {'Sub':<5} {'Run':>4} {'MI':>4} {'REST':>5} "
          f"{'Ret%':>6} {'ERD_C3':>8} {'ERD_C4':>8} {'Acc%':>8}")
    print(f"  {'─'*60}")
    for subj, runs in all_results.items():
        for r in runs:
            acc_str = f"{r['acc']*100:>7.1f}%" if not np.isnan(r['acc']) else "     n/a"
            print(f"  {subj:<5} {r['run']:>4}  {r['n_mi']:>3}  {r['n_rest']:>4}  "
                  f"{r['retention']:>5.0f}%  "
                  f"{r['erd_c3']:>7.1f}%  "
                  f"{r['erd_c4']:>7.1f}%  {acc_str}")

    print(f"\n  Figures → {FIG_DIR}/")
    print(f"    run_progression.png")
    print(f"    run_progression_per_subject.png")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
