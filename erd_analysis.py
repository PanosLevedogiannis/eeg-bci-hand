"""
=============================================================================
  ERD/ERS Time-Frequency Analysis — inMoov Dataset
=============================================================================
  Generates ERD/ERS and time-frequency plots for the shared S01/S02 data.

  Task:    Motor Imagery (right hand) vs Rest
  Markers: code 1 = MI, code 2 = REST (embedded in FIF via LSL)
  Bands:   Mu 8–12 Hz,  Beta 13–30 Hz  (key motor cortex rhythms)

  Output:  eeg_data/figures/erd_<subject>.png
           eeg_data/figures/tf_<subject>.png
           eeg_data/figures/erd_comparison.png

  Usage:
    python erd_analysis.py
    python erd_analysis.py --dataset /path/to/inMoov_Dataset
=============================================================================
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import butter, filtfilt
import os
import glob
import argparse

mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset"
FIG_DIR     = "eeg_data/figures"
os.makedirs(FIG_DIR, exist_ok=True)

EVENT_ID = {"MI": 1, "REST": 2}

TMIN     = -2.0   # 2s pre-cue baseline (wider window for TF edge effects)
TMAX     =  6.0   # 6s post-cue (captures post-imagery beta rebound)
BASELINE = (-1.5, -0.5)   # mid-baseline, avoids edge artifacts

BANDS = {
    "Mu (8–12 Hz)":   (8,  12),
    "Beta (13–30 Hz)":(13, 30),
}

CHANNELS_OF_INTEREST = ["C3", "C4", "Cz", "FCz"]

# ─────────────────────────────────────────────
#  COLOUR THEME
# ─────────────────────────────────────────────

DARK_BG   = "#0A0C14"
COL_PANEL = "#12162A"
COL_MI    = "#00DC82"    # green  — Motor Imagery
COL_REST  = "#3C8CFF"    # blue   — Rest
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


def use_print_theme():
    """White-background palette for figures that go into the printed thesis."""
    global DARK_BG, COL_PANEL, COL_MI, COL_REST, COL_GREY
    DARK_BG, COL_PANEL = "white", "white"
    COL_MI, COL_REST, COL_GREY = "#1B7F4F", "#2A6FB5", "#555555"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111", "xtick.color": "#333333",
        "ytick.color": "#333333", "text.color": "#111111",
        "grid.color": "#CCCCCC", "grid.linewidth": 0.6,
        "font.family": "serif", "savefig.dpi": 300,
    })


def savefig(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def find_fif(subject_dir: str) -> str:
    fifs = glob.glob(os.path.join(subject_dir, "*_raw.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_raw.fif found in {subject_dir}")
    return sorted(fifs)[-1]   # take most recent if multiple


# ADS1299 full-scale rail at gain 24 (OpenBCI Cyton default) — physically
# impossible to exceed this in real signal; sustained presence means
# amplifier clipping (bad electrode/reference contact), not brain activity.
ADS1299_RAIL_UV = 187500.0


def check_raw_signal_quality(raw: mne.io.Raw) -> dict:
    """
    Sanity-check the RAW signal before filtering/referencing hides problems.
    Bandpass + CAR remove large offsets and saturation transients, which is
    why epoch-rejection-based retention can look "excellent" even when the
    raw acquisition was faulty (rail-clipping, dead reference electrode).
    """
    data_uv = raw.get_data() * 1e6
    rail_pct_per_ch = 100 * np.mean(np.abs(data_uv) >= ADS1299_RAIL_UV - 1, axis=1)
    max_rail_pct = float(rail_pct_per_ch.max())

    # Per-channel: does this channel ever cross to the other sign?
    eps = 50.0  # uV — "never crosses zero" tolerance
    ch_one_sided = (data_uv.max(axis=1) <= eps) | (data_uv.min(axis=1) >= -eps)
    n_one_sided  = int(ch_one_sided.sum())
    n_ch         = data_uv.shape[0]
    # Flag only when most channels are affected — a chronic per-session offset
    # (bad reference/bias contact), not a couple of channels with real signal.
    one_sided    = n_one_sided >= int(round(0.75 * n_ch))

    return {
        "max_rail_pct": max_rail_pct,
        "rail_warning": max_rail_pct >= 1.0,
        "one_sided": one_sided,
        "n_one_sided_ch": n_one_sided,
    }


def check_physiological_spectrum(raw: mne.io.Raw) -> dict:
    """
    After CAR+filter: does the signal still look like real EEG (1/f decay)
    or like residual noise/offset (flat spectrum)? Run on filtered+referenced
    raw, channels C3/C4.
    """
    picks = [ch for ch in ("C3", "C4") if ch in raw.ch_names]
    psd = raw.compute_psd(fmin=1, fmax=40, picks=picks, verbose=False)
    freqs = psd.freqs
    data = psd.get_data().mean(axis=0)
    p_low  = data[(freqs >= 3) & (freqs <= 5)].mean()
    p_high = data[(freqs >= 25) & (freqs <= 35)].mean()
    ratio_1f = float(p_low / p_high)
    return {
        "ratio_1f": ratio_1f,
        "physiological": ratio_1f > 3.0,
    }


def load_and_preprocess(fif_path: str, subject_id: str) -> tuple:
    print(f"\n  Loading  {os.path.basename(fif_path)}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    print(f"  Channels: {raw.ch_names}")
    print(f"  Duration: {raw.times[-1]/60:.1f} min  |  "
          f"Sfreq: {raw.info['sfreq']:.0f} Hz")

    # --- Raw QC (must run BEFORE filtering — that's what hides the fault) ---
    qc = check_raw_signal_quality(raw)
    flags = []
    if qc["rail_warning"]:
        flags.append(f"RAIL-CLIP {qc['max_rail_pct']:.1f}%")
    if qc["one_sided"]:
        flags.append(f"ONE-SIDED-OFFSET ({qc['n_one_sided_ch']}/{len(raw.ch_names)} ch)")
    if flags:
        print(f"  ⚠ Raw signal fault: {', '.join(flags)}")

    # 1. Bandpass 1–40 Hz + notch 50 Hz
    raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    # 2. Common Average Reference
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average",
                                   projection=False, verbose=False)

    phys = check_physiological_spectrum(raw)
    qc.update(phys)
    if not phys["physiological"]:
        print(f"  ⚠ Post-filter spectrum looks non-physiological "
              f"(1/f ratio={phys['ratio_1f']:.1f}, expected >3) — verify visually")
    else:
        print(f"  ✓ Post-filter spectrum is physiological (1/f ratio={phys['ratio_1f']:.1f})")

    # 3. Extract events from annotations (codes 1 and 2)
    events, _ = mne.events_from_annotations(
        raw,
        event_id={"1": 1, "2": 2},
        verbose=False,
    )

    counts = {name: int(np.sum(events[:, 2] == eid))
              for name, eid in EVENT_ID.items()}
    print(f"  Trials: {counts}")

    # 4. Epoch (wide window for TF edge-effect buffer)
    epochs = mne.Epochs(
        raw,
        events     = events,
        event_id   = EVENT_ID,
        tmin       = TMIN,
        tmax       = TMAX,
        baseline   = None,    # manual baseline in analysis
        reject     = dict(eeg=200e-6),  # 200µV threshold
        preload    = True,
        verbose    = False,
    )

    n_total = len(events)
    n_kept  = len(epochs)
    if n_kept == 0:
        print("  ⚠ All epochs rejected at 200µV — retrying without rejection")
        epochs = mne.Epochs(
            raw, events=events, event_id=EVENT_ID,
            tmin=TMIN, tmax=TMAX, baseline=None,
            reject=None, preload=True, verbose=False,
        )
        n_kept = len(epochs)

    retention_pct = 100 * n_kept / max(1, n_total)
    print(f"  Epochs: {n_kept}/{n_total} kept  "
          f"({n_total - n_kept} dropped, "
          f"{retention_pct:.1f}% retained)")

    qc["n_kept"]  = n_kept
    qc["n_total"] = n_total
    qc["retention_pct"] = retention_pct
    return epochs, qc


# ─────────────────────────────────────────────
#  ERD/ERS TIME COURSE
# ─────────────────────────────────────────────

def compute_erd(epochs: mne.Epochs, ch_name: str,
                flo: float, fhi: float) -> tuple:
    """
    Returns (times, mean_mi, sem_mi, mean_rest, sem_rest) in % change from baseline.
    Method: bandpass → square → 250ms smooth → baseline normalise.
    """
    sfreq = epochs.info["sfreq"]
    times = epochs.times

    nyq  = sfreq / 2.0
    b, a = butter(4, [flo / nyq, fhi / nyq], btype="band")

    results = {}
    for label in ("MI", "REST"):
        ep   = epochs[label]
        idx  = ep.ch_names.index(ch_name)
        data = ep.get_data()[:, idx, :]   # (trials, time)

        filtered = np.array([filtfilt(b, a, trial) for trial in data])
        power    = filtered ** 2

        ksz    = int(sfreq * 0.25)        # 250ms boxcar smooth
        kernel = np.ones(ksz) / ksz
        smooth = np.array([np.convolve(p, kernel, mode="same") for p in power])

        bl_mask  = (times >= BASELINE[0]) & (times <= BASELINE[1])
        bl_mean  = smooth[:, bl_mask].mean(axis=1, keepdims=True)
        erd      = 100.0 * (smooth - bl_mean) / (bl_mean + 1e-12)

        results[label] = (erd.mean(axis=0), erd.std(axis=0) / np.sqrt(len(erd)))

    return (times,
            results["MI"][0],  results["MI"][1],
            results["REST"][0], results["REST"][1])


def plot_erd_ers(epochs: mne.Epochs, subject_id: str):
    print(f"  [1/2] ERD/ERS time course...")

    channels = [c for c in CHANNELS_OF_INTEREST if c in epochs.ch_names]
    n_bands  = len(BANDS)
    n_ch     = len(channels)

    fig, axes = plt.subplots(n_bands, n_ch,
                             figsize=(4.5 * n_ch, 4 * n_bands),
                             sharex=True)
    if n_bands == 1:
        axes = [axes]
    if n_ch == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(f"ERD/ERS Time Course — {subject_id}  "
                 f"(MI vs Rest, n={len(epochs['MI'])}+{len(epochs['REST'])})",
                 fontsize=13, y=1.01)

    for row, (band_name, (flo, fhi)) in enumerate(BANDS.items()):
        for col, ch in enumerate(channels):
            ax = axes[row][col]
            t, mi_m, mi_s, re_m, re_s = compute_erd(epochs, ch, flo, fhi)

            ax.plot(t, mi_m, color=COL_MI,   lw=1.8, label="MI")
            ax.fill_between(t, mi_m - mi_s, mi_m + mi_s,
                            color=COL_MI, alpha=0.18)

            ax.plot(t, re_m, color=COL_REST, lw=1.8, ls="--", label="REST")
            ax.fill_between(t, re_m - re_s, re_m + re_s,
                            color=COL_REST, alpha=0.18)

            ax.axhline(0,    color=COL_GREY, lw=0.8, ls=":")
            ax.axvline(0,    color="white",  lw=1.2, ls="--", alpha=0.8,
                       label="Cue onset")
            ax.axvspan(TMIN, BASELINE[1], color="#1A2040", alpha=0.35,
                       label="Baseline")

            ax.set_title(f"{band_name} — {ch}", fontsize=10)
            ax.set_ylabel("ERD/ERS (%)")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

        axes[-1][col].set_xlabel("Time (s)")

    fig.tight_layout()
    savefig(fig, f"erd_{subject_id}.png")


# ─────────────────────────────────────────────
#  TIME-FREQUENCY SPECTROGRAM
# ─────────────────────────────────────────────

def plot_tf(epochs: mne.Epochs, subject_id: str):
    print(f"  [2/2] Time-frequency spectrograms...")

    channels  = [c for c in ["C3", "C4"] if c in epochs.ch_names]
    freqs     = np.arange(4, 40, 0.5)
    n_cycles  = freqs / 2.0

    n_ch = len(channels)
    fig, axes = plt.subplots(2, n_ch, figsize=(6 * n_ch, 9),
                             sharex=True, sharey=True)
    if n_ch == 1:
        axes = [[axes[0]], [axes[1]]]

    fig.suptitle(f"Time-Frequency Power (Morlet) — {subject_id}",
                 fontsize=13, y=1.01)

    for col, ch in enumerate(channels):
        for row, (label, title, col_str) in enumerate([
            ("MI",   f"MI  ({ch})",   COL_MI),
            ("REST", f"REST  ({ch})", COL_REST),
        ]):
            ax  = axes[row][col]
            ep  = epochs[label].pick_channels([ch], ordered=False).copy()

            tfr = mne.time_frequency.tfr_morlet(
                ep, freqs=freqs, n_cycles=n_cycles,
                return_itc=False, average=True, verbose=False,
            )
            tfr.apply_baseline(baseline=BASELINE, mode="logratio",
                               verbose=False)

            power = tfr.data[0]   # (freqs, times)
            vmax  = np.percentile(np.abs(power), 97)
            norm  = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = ax.imshow(
                power,
                aspect = "auto",
                origin = "lower",
                extent = [epochs.times[0], epochs.times[-1],
                          freqs[0], freqs[-1]],
                cmap   = "RdBu_r",
                norm   = norm,
            )
            ax.axvline(0,  color="white",  lw=1.5, ls="--", alpha=0.9,
                       label="Cue onset")
            ax.axhline(8,  color="yellow", lw=0.8, ls=":",  alpha=0.65)
            ax.axhline(12, color="yellow", lw=0.8, ls=":",  alpha=0.65,
                       label="Mu 8–12 Hz")
            ax.axhline(13, color="cyan",   lw=0.8, ls="-.", alpha=0.5)
            ax.axhline(30, color="cyan",   lw=0.8, ls="-.", alpha=0.5,
                       label="Beta 13–30 Hz")
            ax.set_title(title, color=col_str, fontsize=11)
            ax.set_ylabel("Frequency (Hz)")
            ax.legend(fontsize=7, loc="upper right")
            fig.colorbar(im, ax=ax, label="Power (dB rel. baseline)",
                         shrink=0.85)

    for col in range(n_ch):
        axes[-1][col].set_xlabel("Time (s)")

    fig.tight_layout()
    savefig(fig, f"tf_{subject_id}.png")


# ─────────────────────────────────────────────
#  BETWEEN-SUBJECT COMPARISON
# ─────────────────────────────────────────────

def plot_comparison(subject_data: dict):
    """
    Side-by-side ERD/ERS comparison for all subjects, mu band, C3+C4.
    subject_data: {subject_id: epochs}
    """
    if len(subject_data) < 2:
        return

    print(f"  [3/3] Between-subject comparison...")
    subjects = list(subject_data.keys())
    n_subj   = len(subjects)
    channels = ["C3", "C4"]
    flo, fhi = BANDS["Mu (8–12 Hz)"]

    fig, axes = plt.subplots(n_subj, len(channels),
                             figsize=(10, 4 * n_subj),
                             sharex=True, sharey=True)
    if n_subj == 1:
        axes = [axes]

    fig.suptitle("Mu ERD/ERS — Between-Subject Comparison",
                 fontsize=13, y=1.01)

    for row, subj in enumerate(subjects):
        epochs = subject_data[subj]
        chs    = [c for c in channels if c in epochs.ch_names]
        for col, ch in enumerate(chs):
            ax = axes[row][col]
            t, mi_m, mi_s, re_m, re_s = compute_erd(epochs, ch, flo, fhi)

            ax.plot(t, mi_m, color=COL_MI,   lw=2.0, label="MI")
            ax.fill_between(t, mi_m - mi_s, mi_m + mi_s,
                            color=COL_MI, alpha=0.2)
            ax.plot(t, re_m, color=COL_REST, lw=2.0, ls="--", label="REST")
            ax.fill_between(t, re_m - re_s, re_m + re_s,
                            color=COL_REST, alpha=0.2)

            ax.axhline(0, color=COL_GREY, lw=0.8, ls=":")
            ax.axvline(0, color="white",  lw=1.2, ls="--", alpha=0.8)
            ax.axvspan(TMIN, BASELINE[1], color="#1A2040", alpha=0.3)

            ax.set_title(f"{subj}  —  {ch}", fontsize=11)
            ax.set_ylabel("ERD/ERS (%, Mu)")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc="upper right")

        axes[-1][col].set_xlabel("Time (s)")

    fig.tight_layout()
    savefig(fig, "erd_comparison.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR,
                        help="Path to inMoov_Dataset folder")
    parser.add_argument("--print-theme", action="store_true",
                        help="White background for the printed thesis")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Only these subject IDs (default: all)")
    args = parser.parse_args()

    if args.print_theme:
        use_print_theme()

    dataset_dir = args.dataset

    if not os.path.isdir(dataset_dir):
        print(f"✗ Dataset not found: {dataset_dir}")
        print("  Pass --dataset /path/to/inMoov_Dataset")
        return

    subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, "S*")))
    if args.subjects:
        wanted = set(args.subjects)
        subject_dirs = [d for d in subject_dirs if os.path.basename(d) in wanted]
    if not subject_dirs:
        print(f"✗ No subject folders (S01, S02, ...) found in {dataset_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  ERD/ERS Analysis  |  {len(subject_dirs)} subject(s)")
    print(f"  Output → {FIG_DIR}/")
    print(f"{'='*60}")

    subject_data = {}
    qc_data = {}
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        print(f"\n─── {subject_id} ─────────────────────────────")
        try:
            fif_path = find_fif(subj_dir)
            epochs, qc = load_and_preprocess(fif_path, subject_id)
            plot_erd_ers(epochs, subject_id)
            plot_tf(epochs, subject_id)
            subject_data[subject_id] = epochs
            qc_data[subject_id] = qc
        except Exception as e:
            print(f"  ✗ Failed for {subject_id}: {e}")
            import traceback; traceback.print_exc()

    if len(subject_data) >= 2:
        plot_comparison(subject_data)

    print(f"\n{'='*60}")
    print(f"  Done. Figures saved to {FIG_DIR}/")
    print(f"  Files:")
    for subj in subject_data:
        print(f"    erd_{subj}.png  |  tf_{subj}.png")
    if len(subject_data) >= 2:
        print(f"    erd_comparison.png")

    print(f"\n{'='*60}")
    print(f"  RAW SIGNAL QC SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Subject':8s} {'Retain%':>8s} {'RailClip%':>10s} {'OneSided':>9s} {'1/f ratio':>10s} {'Physio':>7s}")
    for subj, qc in qc_data.items():
        print(f"  {subj:8s} {qc['retention_pct']:7.1f}% {qc['max_rail_pct']:9.1f}% "
              f"{str(qc['one_sided']):>9s} {qc['ratio_1f']:9.1f} "
              f"{'OK' if qc['physiological'] else 'CHECK':>7s}")
    faulty = [s for s, qc in qc_data.items() if qc["rail_warning"] or qc["one_sided"] or not qc["physiological"]]
    if faulty:
        print(f"\n  ⚠ Subjects with raw-acquisition faults (verify before trusting retention%): {', '.join(faulty)}")
    print(f"{'='*60}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
