"""
=============================================================================
  Full EEG Analysis — inMoov Dataset  (4 subjects, MI vs REST)
=============================================================================
  Produces per-subject and between-subject figures + a summary report.

  Per subject (eeg_data/figures/<subject>_*.png):
    1. ERD/ERS time course     — mu + beta, C3/C4/Cz/FCz
    2. Time-frequency (Morlet) — C3 + C4
    3. Topomaps                — mu band at 0.5 / 1.5 / 2.5 / 3.5 s
    4. PSD comparison          — all channels, MI vs REST

  Between subjects:
    5. Classification summary  — CSP+LDA 10-fold CV accuracy per subject
    6. Grand average ERD       — all 4 subjects overlaid

  Usage:
    python full_analysis.py
    python full_analysis.py --dataset /path/to/inMoov_Dataset
=============================================================================
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import butter, filtfilt, welch
import os
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

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

TMIN     = -2.0
TMAX     =  6.0
BASELINE = (-1.5, -0.5)

BANDS = {
    "Mu (8–12 Hz)":    (8,  12),
    "Beta (13–30 Hz)": (13, 30),
}

CHANNELS_OF_INTEREST = ["C3", "C4", "Cz", "FCz"]

DARK_BG   = "#0A0C14"
COL_PANEL = "#12162A"
COL_MI    = "#00DC82"
COL_REST  = "#3C8CFF"
COL_GREY  = "#788092"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    COL_PANEL,
    "axes.edgecolor":    "#283256",
    "axes.labelcolor":   "#C8CDE8",
    "axes.titlecolor":   "#F0F5FF",
    "xtick.color":       COL_GREY,
    "ytick.color":       COL_GREY,
    "text.color":        "#F0F5FF",
    "grid.color":        "#283256",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "figure.dpi":        120,
})


def savefig(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {name}")


# ─────────────────────────────────────────────
#  LOAD + PREPROCESS
# ─────────────────────────────────────────────

def find_fif(subject_dir: str) -> str:
    fifs = glob.glob(os.path.join(subject_dir, "*_raw.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_raw.fif in {subject_dir}")
    return sorted(fifs)[-1]


def load_and_preprocess(fif_path: str) -> mne.Epochs:
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw, _ = mne.set_eeg_reference(raw, "average", projection=False, verbose=False)

    events, _ = mne.events_from_annotations(raw, event_id={"1": 1, "2": 2},
                                             verbose=False)
    epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                        baseline=None, reject=dict(eeg=200e-6),
                        preload=True, verbose=False)
    if len(epochs) == 0:
        epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                            baseline=None, reject=None, preload=True,
                            verbose=False)

    return epochs, raw.info


# ─────────────────────────────────────────────
#  HELPER — ERD computation
# ─────────────────────────────────────────────

def compute_erd(epochs, ch_name, flo, fhi):
    sfreq = epochs.info["sfreq"]
    times = epochs.times
    nyq   = sfreq / 2.0
    b, a  = butter(4, [flo / nyq, fhi / nyq], btype="band")
    out   = {}
    for label in ("MI", "REST"):
        ep   = epochs[label]
        idx  = ep.ch_names.index(ch_name)
        data = ep.get_data()[:, idx, :]
        filt = np.array([filtfilt(b, a, t) for t in data])
        power = filt ** 2
        ksz   = int(sfreq * 0.25)
        kern  = np.ones(ksz) / ksz
        smooth = np.array([np.convolve(p, kern, mode="same") for p in power])
        bl    = (times >= BASELINE[0]) & (times <= BASELINE[1])
        bl_m  = smooth[:, bl].mean(axis=1, keepdims=True)
        erd   = 100.0 * (smooth - bl_m) / (bl_m + 1e-12)
        out[label] = (erd.mean(axis=0), erd.std(axis=0) / np.sqrt(len(erd)))
    return times, out


# ─────────────────────────────────────────────
#  PLOT 1 — ERD/ERS time course
# ─────────────────────────────────────────────

def plot_erd(epochs, subject_id):
    chs    = [c for c in CHANNELS_OF_INTEREST if c in epochs.ch_names]
    n_b, n_c = len(BANDS), len(chs)
    fig, axes = plt.subplots(n_b, n_c, figsize=(4.5*n_c, 4*n_b), sharex=True)
    axes = np.array(axes).reshape(n_b, n_c)
    fig.suptitle(
        f"ERD/ERS — {subject_id}  "
        f"(MI n={len(epochs['MI'])}, REST n={len(epochs['REST'])})",
        fontsize=13, y=1.01)

    for r, (bname, (flo, fhi)) in enumerate(BANDS.items()):
        for c, ch in enumerate(chs):
            ax = axes[r, c]
            times, res = compute_erd(epochs, ch, flo, fhi)
            for label, col, ls in [("MI", COL_MI, "-"), ("REST", COL_REST, "--")]:
                m, s = res[label]
                ax.plot(times, m, color=col, lw=1.8, ls=ls, label=label)
                ax.fill_between(times, m-s, m+s, color=col, alpha=0.18)
            ax.axhline(0, color=COL_GREY, lw=0.8, ls=":")
            ax.axvline(0, color="white",  lw=1.2, ls="--", alpha=0.8)
            ax.axvspan(TMIN, BASELINE[1], color="#1A2040", alpha=0.3)
            ax.set_title(f"{bname} — {ch}", fontsize=10)
            ax.set_ylabel("ERD/ERS (%)")
            ax.grid(True)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    for c in range(n_c):
        axes[-1, c].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, f"{subject_id}_1_erd.png")


# ─────────────────────────────────────────────
#  PLOT 2 — Time-frequency spectrogram
# ─────────────────────────────────────────────

def plot_tf(epochs, subject_id):
    chs      = [c for c in ["C3", "C4"] if c in epochs.ch_names]
    freqs    = np.arange(4, 40, 0.5)
    n_cycles = freqs / 2.0
    n_c      = len(chs)

    fig, axes = plt.subplots(2, n_c, figsize=(6*n_c, 9),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(2, n_c)
    fig.suptitle(f"Time-Frequency (Morlet) — {subject_id}", fontsize=13, y=1.01)

    for col, ch in enumerate(chs):
        for row, (label, col_str) in enumerate([("MI", COL_MI), ("REST", COL_REST)]):
            ax  = axes[row, col]
            ep  = epochs[label].pick_channels([ch], ordered=False).copy()
            tfr = mne.time_frequency.tfr_morlet(
                ep, freqs=freqs, n_cycles=n_cycles,
                return_itc=False, average=True, verbose=False)
            tfr.apply_baseline(BASELINE, mode="logratio", verbose=False)
            power = tfr.data[0]
            vmax  = np.percentile(np.abs(power), 97)
            norm  = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(power, aspect="auto", origin="lower",
                           extent=[epochs.times[0], epochs.times[-1],
                                   freqs[0], freqs[-1]],
                           cmap="RdBu_r", norm=norm)
            ax.axvline(0,  color="white",  lw=1.5, ls="--", alpha=0.9)
            ax.axhline(8,  color="yellow", lw=0.7, ls=":",  alpha=0.6)
            ax.axhline(12, color="yellow", lw=0.7, ls=":",  alpha=0.6)
            ax.axhline(13, color="cyan",   lw=0.7, ls="-.", alpha=0.5)
            ax.axhline(30, color="cyan",   lw=0.7, ls="-.", alpha=0.5)
            ax.set_title(f"{label}  ({ch})", color=col_str, fontsize=11)
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="dB (baseline)", shrink=0.85)
        axes[-1, col].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, f"{subject_id}_2_tf.png")


# ─────────────────────────────────────────────
#  PLOT 3 — Topomaps at peak ERD times
# ─────────────────────────────────────────────

def plot_topomaps(epochs, raw_info, subject_id):
    freqs     = np.arange(8, 31, 1)
    n_cycles  = freqs / 2.0
    times_plot = [0.5, 1.5, 2.5, 3.5]
    labels    = [("MI", COL_MI), ("REST", COL_REST)]

    fig, axes = plt.subplots(len(labels), len(times_plot),
                             figsize=(3.5*len(times_plot), 5*len(labels)))
    axes = np.array(axes).reshape(len(labels), len(times_plot))
    fig.suptitle(f"Mu+Beta Topomaps — {subject_id}", fontsize=13, y=1.01)

    for row, (label, col_str) in enumerate(labels):
        ep  = epochs[label].copy()
        tfr = mne.time_frequency.tfr_morlet(
            ep, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        tfr.apply_baseline(BASELINE, mode="logratio", verbose=False)

        for col, t in enumerate(times_plot):
            ax = axes[row, col]
            tfr.plot_topomap(
                tmin=t - 0.25, tmax=t + 0.25,
                fmin=8, fmax=30,
                axes=ax, show=False, colorbar=False,
                cmap="RdBu_r", vlim=(-0.5, 0.5),
            )
            if row == 0:
                ax.set_title(f"t = {t}s", fontsize=10)
            if col == 0:
                ax.set_ylabel(label, color=col_str, fontsize=11)

    fig.tight_layout()
    savefig(fig, f"{subject_id}_3_topomaps.png")


# ─────────────────────────────────────────────
#  PLOT 4 — PSD comparison MI vs REST
# ─────────────────────────────────────────────

def plot_psd(epochs, subject_id):
    sfreq  = epochs.info["sfreq"]
    chs    = epochs.ch_names
    n_ch   = len(chs)

    # Use 0–4s imagery window
    t_mask = (epochs.times >= 0) & (epochs.times <= 4.0)
    n_cols = 4
    n_rows = int(np.ceil(n_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5*n_cols, 3.5*n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle(f"PSD: MI vs REST (0–4s imagery window) — {subject_id}",
                 fontsize=13, y=1.01)

    for i, ch in enumerate(chs):
        ax   = axes[i]
        idx  = epochs.ch_names.index(ch)
        mi_data   = epochs["MI"].get_data()[:, idx, :][:, t_mask]
        rest_data = epochs["REST"].get_data()[:, idx, :][:, t_mask]

        freqs_psd, psd_mi   = welch(mi_data,   fs=sfreq, nperseg=int(sfreq*2),
                                    axis=-1)
        _,          psd_rest = welch(rest_data, fs=sfreq, nperseg=int(sfreq*2),
                                    axis=-1)

        mask = freqs_psd <= 40
        f    = freqs_psd[mask]

        mi_mean   = 10 * np.log10(psd_mi[:,   mask].mean(axis=0) + 1e-30)
        rest_mean = 10 * np.log10(psd_rest[:, mask].mean(axis=0) + 1e-30)
        mi_sem    = psd_mi[:,   mask].std(axis=0) / np.sqrt(len(psd_mi))
        rest_sem  = psd_rest[:, mask].std(axis=0) / np.sqrt(len(psd_rest))
        mi_sem_db   = 10 * np.log10(psd_mi[:, mask].mean(axis=0) + mi_sem   + 1e-30) - mi_mean
        rest_sem_db = 10 * np.log10(psd_rest[:, mask].mean(axis=0) + rest_sem + 1e-30) - rest_mean

        ax.plot(f, mi_mean,   color=COL_MI,   lw=1.6, label="MI")
        ax.fill_between(f, mi_mean - mi_sem_db, mi_mean + mi_sem_db,
                        color=COL_MI, alpha=0.2)
        ax.plot(f, rest_mean, color=COL_REST, lw=1.6, ls="--", label="REST")
        ax.fill_between(f, rest_mean - rest_sem_db, rest_mean + rest_sem_db,
                        color=COL_REST, alpha=0.2)

        ax.axvspan(8,  12, color="yellow", alpha=0.07, label="Mu")
        ax.axvspan(13, 30, color="cyan",   alpha=0.05, label="Beta")
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    savefig(fig, f"{subject_id}_4_psd.png")


# ─────────────────────────────────────────────
#  CLASSIFICATION — CSP + LDA (10-fold CV)
# ─────────────────────────────────────────────

def classify_subject(epochs) -> tuple:
    """
    Returns (mean_acc, std_acc).
    Uses 8-30 Hz band-filtered epochs, CSP(4) + LDA, 10-fold stratified CV.
    """
    sfreq = epochs.info["sfreq"]
    t_mask = (epochs.times >= 0) & (epochs.times <= 4.0)

    # Band-filter 8-30 Hz for CSP
    ep_filtered = epochs.copy().filter(8, 30, method="fir", verbose=False)
    X = ep_filtered.get_data()[:, :, t_mask]       # (trials, ch, time)
    y = (epochs.events[:, 2] == 1).astype(int)     # 1=MI, 0=REST

    pipe = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])
    cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


# ─────────────────────────────────────────────
#  PLOT 5 — Summary across subjects
# ─────────────────────────────────────────────

def plot_summary(results: dict):
    """
    results: {subject_id: {"retention": float, "n_mi": int, "n_rest": int,
                           "acc": float, "acc_std": float,
                           "peak_erd_c3": float, "peak_erd_c4": float}}
    """
    subjects = list(results.keys())
    n        = len(subjects)
    accs     = [results[s]["acc"]       for s in subjects]
    acc_stds = [results[s]["acc_std"]   for s in subjects]
    rets     = [results[s]["retention"] for s in subjects]
    erd_c3   = [results[s]["peak_erd_c3"] for s in subjects]
    erd_c4   = [results[s]["peak_erd_c4"] for s in subjects]

    x = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Between-Subject Summary — MI vs REST", fontsize=14)

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, [a*100 for a in accs], color=COL_MI, alpha=0.8,
                  yerr=[s*100 for s in acc_stds], capsize=5,
                  error_kw=dict(ecolor=COL_GREY, lw=1.5))
    ax.axhline(50, color="red", lw=1.2, ls="--", alpha=0.7, label="Chance (50%)")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("CSP+LDA (10-fold CV)")
    ax.set_ylim(30, 100); ax.grid(True, axis="y"); ax.legend(fontsize=8)
    for i, (acc, std) in enumerate(zip(accs, acc_stds)):
        ax.text(i, acc*100 + std*100 + 1.5, f"{acc*100:.1f}%",
                ha="center", fontsize=9, color="white")

    # Epoch retention
    ax = axes[1]
    ax.bar(x, rets, color=COL_REST, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("Retention (%)"); ax.set_title("Epoch Retention (200µV threshold)")
    ax.set_ylim(0, 105); ax.grid(True, axis="y")
    for i, r in enumerate(rets):
        ax.text(i, r + 1.5, f"{r:.0f}%", ha="center", fontsize=9, color="white")

    # Peak ERD
    ax = axes[2]
    w  = 0.35
    ax.bar(x - w/2, [-e for e in erd_c3], w, color="#FF5A50", alpha=0.85,
           label="C3 (contralateral L)")
    ax.bar(x + w/2, [-e for e in erd_c4], w, color="#FFA040", alpha=0.85,
           label="C4 (contralateral R)")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("Peak ERD magnitude (%, Mu)")
    ax.set_title("Peak Mu ERD (0–4s, MI trials)")
    ax.grid(True, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "summary.png")


def plot_grand_average_erd(subject_data: dict):
    """Overlay all subjects' mu ERD on C3 and C4 in one plot."""
    colors = ["#00DC82", "#3C8CFF", "#FF5A50", "#FFD700"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.suptitle("Grand Average — Mu ERD/ERS  (all subjects, MI only)",
                 fontsize=13, y=1.01)

    for col, ch in enumerate(["C3", "C4"]):
        ax = axes[col]
        for i, (subj, epochs) in enumerate(subject_data.items()):
            if ch not in epochs.ch_names:
                continue
            times, res = compute_erd(epochs, ch, 8, 12)
            m, s = res["MI"]
            ax.plot(times, m, color=colors[i % len(colors)],
                    lw=2.0, label=subj)
            ax.fill_between(times, m-s, m+s,
                            color=colors[i % len(colors)], alpha=0.1)
        ax.axhline(0, color=COL_GREY, lw=0.8, ls=":")
        ax.axvline(0, color="white",  lw=1.2, ls="--", alpha=0.8,
                   label="Cue onset")
        ax.axvspan(TMIN, BASELINE[1], color="#1A2040", alpha=0.3)
        ax.set_title(ch, fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ERD/ERS (%)")
        ax.grid(True)
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    savefig(fig, "grand_average_erd.png")


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
    print(f"  Full Analysis  |  {len(subject_dirs)} subject(s)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output:  {FIG_DIR}/")
    print(f"{'='*60}")

    subject_data = {}
    summary      = {}

    for subj_dir in subject_dirs:
        subj = os.path.basename(subj_dir)
        print(f"\n─── {subj} ──────────────────────────────────────")
        try:
            fif_path       = find_fif(subj_dir)
            epochs, raw_info = load_and_preprocess(fif_path)

            n_mi   = len(epochs["MI"])
            n_rest = len(epochs["REST"])
            n_tot  = 320   # 4 runs × 40 trials × 2 classes
            ret    = 100 * len(epochs) / n_tot

            print(f"  Trials: MI={n_mi}, REST={n_rest}  |  "
                  f"Retention: {ret:.0f}%")

            print(f"  [1/4] ERD/ERS time course...")
            plot_erd(epochs, subj)

            print(f"  [2/4] Time-frequency spectrograms...")
            plot_tf(epochs, subj)

            print(f"  [3/4] Topomaps...")
            plot_topomaps(epochs, raw_info, subj)

            print(f"  [4/4] PSD comparison...")
            plot_psd(epochs, subj)

            print(f"  [+]   Classification (CSP+LDA)...")
            acc, acc_std = classify_subject(epochs)
            print(f"        Accuracy: {acc*100:.1f}% ± {acc_std*100:.1f}%")

            # Peak ERD in mu band, C3 and C4, during 0-4s MI
            peak_erd = {}
            for ch in ["C3", "C4"]:
                if ch in epochs.ch_names:
                    times, res = compute_erd(epochs, ch, 8, 12)
                    t_mask = (times >= 0) & (times <= 4.0)
                    peak_erd[ch] = float(res["MI"][0][t_mask].min())
                else:
                    peak_erd[ch] = 0.0

            summary[subj] = {
                "retention":    ret,
                "n_mi":         n_mi,
                "n_rest":       n_rest,
                "acc":          acc,
                "acc_std":      acc_std,
                "peak_erd_c3":  peak_erd.get("C3", 0.0),
                "peak_erd_c4":  peak_erd.get("C4", 0.0),
            }
            subject_data[subj] = epochs

        except Exception as e:
            import traceback
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()

    # ── Between-subject plots ──
    if subject_data:
        print(f"\n─── Between-subject ────────────────────────────────")
        print(f"  [5/6] Summary figure...")
        plot_summary(summary)
        print(f"  [6/6] Grand average ERD...")
        plot_grand_average_erd(subject_data)

    # ── Console summary table ──
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Subject':<8} {'MI':>4} {'REST':>5} {'Ret%':>6} "
          f"{'Acc%':>8} {'C3_ERD':>8} {'C4_ERD':>8}")
    print(f"  {'─'*55}")
    for subj, r in summary.items():
        print(f"  {subj:<8} {r['n_mi']:>4} {r['n_rest']:>5} "
              f"{r['retention']:>5.0f}%  "
              f"{r['acc']*100:>6.1f}%  "
              f"{r['peak_erd_c3']:>7.1f}%  "
              f"{r['peak_erd_c4']:>7.1f}%")

    print(f"\n  Figures saved to {FIG_DIR}/")
    print(f"  Per subject: <ID>_1_erd, _2_tf, _3_topomaps, _4_psd")
    print(f"  Summary:     summary.png,  grand_average_erd.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
