"""
=============================================================================
  Load & inspect a recorded EEG session
=============================================================================
  Loads the FIF file saved by eeg_mi_paradigm.py and prints a full report:
  channels, duration, markers, signal quality per channel.

  Usage:
    python load_recording.py                        # latest FIF in eeg_data/
    python load_recording.py eeg_data/EEG-RAW_S01_*.fif
=============================================================================
"""

import os
import sys
import glob
import numpy as np
import mne

mne.set_log_level("WARNING")

DATA_DIR = "eeg_data"

# Marker code → human label (matches eeg_mi_paradigm.py LSL_MARKERS)
MARKER_LABELS = {
    10: "baseline_eyes_open_start",
    11: "baseline_eyes_open_end",
    12: "baseline_eyes_closed_start",
    13: "baseline_eyes_closed_end",
    20: "trial_start",
     1: "cue_MI",
     2: "cue_REST",
    30: "imagery_start",
    31: "imagery_end",
    40: "break_start",
    41: "break_end",
    99: "session_end",
}


# ─────────────────────────────────────────────
#  FIND FILE
# ─────────────────────────────────────────────

def find_fif(path: str = None) -> str:
    if path:
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No FIF found: {path}")
        return files[-1]

    pattern = os.path.join(DATA_DIR, "*_raw.fif")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No FIF files in {DATA_DIR}/")
    return files[-1]


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def load_fif(filepath: str) -> mne.io.Raw:
    print(f"\n{'='*60}")
    print(f"  Loading: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    return raw


# ─────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────

def print_report(raw: mne.io.Raw):
    dur = raw.times[-1]
    print(f"\n  {'─'*50}")
    print(f"  Channels  : {raw.info['nchan']}  →  {raw.ch_names}")
    print(f"  Sample rate: {raw.info['sfreq']:.0f} Hz")
    print(f"  Duration  : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Samples   : {len(raw.times)}")

    # ── Markers ──────────────────────────────
    ann = raw.annotations
    print(f"\n  Markers ({len(ann)} total):")
    if len(ann) == 0:
        print("    (none)")
    else:
        from collections import Counter
        counts = Counter(a["description"] for a in ann)
        for desc, n in sorted(counts.items()):
            label = MARKER_LABELS.get(int(desc), desc) if desc.isdigit() else desc
            print(f"    [{desc:>3}]  {label:<35}  ×{n}")

    # ── Signal quality ────────────────────────
    print(f"\n  Signal quality (µV RMS per channel):")
    data_uv = raw.get_data() * 1e6
    for i, ch in enumerate(raw.ch_names):
        rms    = np.sqrt(np.mean(data_uv[i] ** 2))
        peak   = np.max(np.abs(data_uv[i]))
        status = "✓" if 1 < rms < 100 else "⚠"
        print(f"    {status}  {ch:<6}  RMS={rms:6.1f} µV   peak={peak:7.1f} µV")

    print(f"\n  {'─'*50}")
    print(f"  Ready for preprocess.py\n")

    return raw


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    fif  = find_fif(path)
    raw  = load_fif(fif)
    print_report(raw)
