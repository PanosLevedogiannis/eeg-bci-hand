"""
=============================================================================
  EEG Preprocessing Pipeline — Step 3: Visualization
=============================================================================
  Plots:
    1. Raw vs filtered signal comparison
    2. ERD/ERS time course  (C3, C4 — key channels)
    3. Time-frequency spectrogram  (Morlet wavelets)
    4. Topomap at peak ERD  (spatial distribution)
    5. Epoch comparison: OPEN vs CLOSE

  Input:  epochs_<subject>_<source>-epo.fif  (from preprocess.py)
  Output: PNG figures saved to  eeg_data/figures/

  Usage:
    python visualize.py
=============================================================================
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_epochs, DATA_DIR

mne.set_log_level("WARNING")

FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  COLOUR THEME  (matches paradigm app)
# ─────────────────────────────────────────────
DARK_BG   = "#0A0C14"
COL_PANEL = "#12162A"
COL_OPEN  = "#00DC82"
COL_CLOSE = "#FF5A50"
COL_BLUE  = "#3C8CFF"
COL_CYAN  = "#00D2C8"
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
    print(f"  ✓ Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 1 — Raw signal overview
# ─────────────────────────────────────────────

def plot_raw_overview(epochs: mne.Epochs, n_epochs: int = 3):
    """
    Show a few raw epochs for OPEN and CLOSE side by side.
    Good for a quick sanity check — do the signals look reasonable?
    """
    print("  [1/4] Plotting raw epoch overview...")

    ep_open  = epochs["T1"] if "T1" in epochs.event_id else epochs["OPEN"]
    ep_close = epochs["T2"] if "T2" in epochs.event_id else epochs["CLOSE"]
    times    = epochs.times

    # Pick C3 and C4 if available, else first 2 channels
    ch_names = epochs.ch_names
    c3 = ch_names.index("C3") if "C3" in ch_names else 0
    c4 = ch_names.index("C4") if "C4" in ch_names else 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 7),
                             sharex=True, sharey=True)
    fig.suptitle("Raw Epoch Overview — C3 & C4", fontsize=14, y=1.01)

    pairs = [
        (ep_open,  c3, COL_OPEN,  "OPEN  — C3",  axes[0, 0]),
        (ep_open,  c4, COL_CYAN,  "OPEN  — C4",  axes[0, 1]),
        (ep_close, c3, COL_CLOSE, "CLOSE — C3",  axes[1, 0]),
        (ep_close, c4, COL_BLUE,  "CLOSE — C4",  axes[1, 1]),
    ]

    for ep, ch_idx, col, title, ax in pairs:
        data = ep.get_data()[:n_epochs, ch_idx, :] * 1e6  # V → µV
        for i, trial in enumerate(data):
            ax.plot(times, trial, color=col, alpha=0.6, linewidth=0.8)
        ax.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.7,
                   label="Cue onset")
        ax.axvspan(-1, 0, color="#1A2040", alpha=0.5, label="Baseline")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "01_raw_epochs.png")


# ─────────────────────────────────────────────
#  PLOT 2 — ERD/ERS Time Course
# ─────────────────────────────────────────────

def plot_erd_ers(epochs: mne.Epochs):
    """
    Event-Related Desynchronization / Synchronization.

    Method: band-power in mu (8-12Hz) and beta (13-30Hz) bands,
    averaged across trials, normalized to baseline (−1s to 0s).

    ERD: power DECREASE during imagery  (negative % change)
    ERS: power INCREASE after imagery   (positive % change)

    This is THE key biomarker of motor imagery — you should see
    clear ERD at C3/C4 during the imagery window [0, +4s].
    """
    print("  [2/4] Computing ERD/ERS...")

    bands = {"Mu (8–12 Hz)": (8, 12), "Beta (13–30 Hz)": (13, 30)}
    sfreq = epochs.info["sfreq"]
    times = epochs.times

    ch_names = epochs.ch_names
    c3 = "C3" if "C3" in ch_names else ch_names[0]
    c4 = "C4" if "C4" in ch_names else ch_names[1]
    channels_of_interest = [c3, c4]

    label1 = "T1" if "T1" in epochs.event_id else "OPEN"
    label2 = "T2" if "T2" in epochs.event_id else "CLOSE"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("ERD/ERS Time Course — Mu & Beta Bands", fontsize=14)

    for row, (band_name, (flo, fhi)) in enumerate(bands.items()):
        for col, ch_name in enumerate(channels_of_interest):
            ax = axes[row, col]

            for label, color, ls in [(label1, COL_OPEN,  "-"),
                                      (label2, COL_CLOSE, "--")]:
                ep  = epochs[label]
                idx = ep.ch_names.index(ch_name)
                data = ep.get_data()[:, idx, :]  # (trials, timepoints)

                # Band-pass filter each trial in the band of interest
                from scipy.signal import butter, filtfilt
                nyq = sfreq / 2
                b, a = butter(4, [flo/nyq, fhi/nyq], btype="band")
                filtered = np.array([filtfilt(b, a, trial) for trial in data])

                # Instantaneous power (signal²), smoothed
                power = filtered ** 2
                kernel_size = int(sfreq * 0.25)  # 250ms smoothing
                kernel = np.ones(kernel_size) / kernel_size
                smooth_power = np.array([np.convolve(p, kernel, mode="same")
                                         for p in power])

                # Baseline normalization (−1s to 0s)
                baseline_mask = (times >= -1.0) & (times <= 0.0)
                baseline_mean = smooth_power[:, baseline_mask].mean(axis=1, keepdims=True)
                erd = 100 * (smooth_power - baseline_mean) / (baseline_mean + 1e-10)

                mean_erd = erd.mean(axis=0)
                sem_erd  = erd.std(axis=0) / np.sqrt(len(erd))

                ax.plot(times, mean_erd, color=color, linewidth=1.8,
                        linestyle=ls, label=label)
                ax.fill_between(times,
                                mean_erd - sem_erd,
                                mean_erd + sem_erd,
                                color=color, alpha=0.15)

            ax.axhline(0,  color=COL_GREY,  linewidth=0.8, linestyle=":")
            ax.axvline(0,  color="white",   linewidth=1.2, linestyle="--",
                       alpha=0.8, label="Cue onset")
            ax.axvspan(-1, 0, color="#1A2040", alpha=0.4, label="Baseline")
            ax.set_title(f"{band_name} — {ch_name}", fontsize=11)
            ax.set_ylabel("ERD/ERS (%)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "02_erd_ers.png")


# ─────────────────────────────────────────────
#  PLOT 3 — Time-Frequency Spectrogram
# ─────────────────────────────────────────────

def plot_spectrogram(epochs: mne.Epochs):
    """
    Time-frequency representation using Morlet wavelets.
    Shows how power evolves across frequencies over time.
    You should see a 'dip' in mu/beta power during [0, +4s].
    """
    print("  [3/4] Computing time-frequency spectrogram...")

    freqs = np.arange(4, 40, 1)   # 4–40 Hz
    n_cycles = freqs / 2.0         # wavelet cycles scales with freq

    label1 = "T1" if "T1" in epochs.event_id else "OPEN"
    label2 = "T2" if "T2" in epochs.event_id else "CLOSE"

    ch_names = epochs.ch_names
    c3 = "C3" if "C3" in ch_names else ch_names[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Time-Frequency Power — {c3}", fontsize=14)

    for ax, label, title, col in [
        (axes[0], label1, f"OPEN  ({label1})",  COL_OPEN),
        (axes[1], label2, f"CLOSE ({label2})", COL_CLOSE),
    ]:
        ep  = epochs[label].pick_channels([c3])
        tfr = mne.time_frequency.tfr_morlet(
            ep, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False
        )

        # Baseline normalization (dB)
        tfr.apply_baseline(baseline=(-1.0, 0.0), mode="logratio",
                           verbose=False)

        power_data = tfr.data[0]   # shape: (freqs, times)
        vmax = np.percentile(np.abs(power_data), 95)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            power_data,
            aspect     = "auto",
            origin     = "lower",
            extent     = [epochs.times[0], epochs.times[-1],
                          freqs[0], freqs[-1]],
            cmap       = "RdBu_r",
            norm       = norm,
        )

        ax.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.9)
        ax.axhline(8,  color="yellow", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="Mu start (8 Hz)")
        ax.axhline(13, color="yellow", linewidth=0.8, linestyle="-.", alpha=0.6,
                   label="Beta start (13 Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, color=col, fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, label="Power (dB, baseline-norm)")

    fig.tight_layout()
    savefig(fig, "03_spectrogram.png")


# ─────────────────────────────────────────────
#  PLOT 4 — Topomap at Peak ERD
# ─────────────────────────────────────────────

def plot_topomap(epochs: mne.Epochs):
    """
    Spatial distribution of mu-band power during motor imagery.
    Shows WHERE on the scalp the ERD is strongest.
    For hand imagery you should see contralateral C3/C4 dominance.
    """
    print("  [4/4] Plotting topomaps...")

    freqs    = np.arange(8, 13, 1)   # mu band
    n_cycles = freqs / 2.0
    times_plot = [0.5, 1.5, 2.5, 3.5]  # seconds during imagery

    label1 = "T1" if "T1" in epochs.event_id else "OPEN"
    label2 = "T2" if "T2" in epochs.event_id else "CLOSE"

    fig, axes = plt.subplots(2, len(times_plot), figsize=(14, 6))
    fig.suptitle("Mu Band Topomap During Motor Imagery", fontsize=14)

    for row, (label, title, col) in enumerate([
        (label1, "OPEN",  COL_OPEN),
        (label2, "CLOSE", COL_CLOSE),
    ]):
        ep  = epochs[label]
        tfr = mne.time_frequency.tfr_morlet(
            ep, freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False,
        )
        tfr.apply_baseline((-1.0, 0.0), mode="logratio", verbose=False)

        for col_idx, t in enumerate(times_plot):
            ax = axes[row, col_idx]
            tfr.plot_topomap(
                tmin=t - 0.25, tmax=t + 0.25,
                fmin=8, fmax=13,
                axes=ax,
                show=False,
                colorbar=False,
                cmap="RdBu_r",
                vlim=(-1, 1),
            )
            if row == 0:
                ax.set_title(f"t = {t}s", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, color=col, fontsize=11)

    fig.tight_layout()
    savefig(fig, "04_topomap.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_visualization(epochs: mne.Epochs = None,
                      subject_id: str = "P001",
                      source: str = "physionet"):
    """Run all 4 visualization plots."""

    if epochs is None:
        epochs = load_epochs(subject_id, source)

    print(f"\n{'='*55}")
    print(f"  Visualization  |  {subject_id}  |  {source}")
    print(f"  Output → {FIG_DIR}/")
    print(f"{'='*55}")

    plot_raw_overview(epochs)
    plot_erd_ers(epochs)
    plot_spectrogram(epochs)
    plot_topomap(epochs)

    print(f"\n  All plots saved to {FIG_DIR}/")
    print("  Step 3 complete. Run classify.py next.\n")


if __name__ == "__main__":
    run_visualization(subject_id="P001", source="physionet")