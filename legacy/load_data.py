"""
=============================================================================
  EEG Preprocessing Pipeline — Step 1: Data Loading
=============================================================================
  Supports:
    A) PhysioNet EEG Motor Imagery Dataset  (for testing NOW)
    B) OpenBCI Cyton CSV export             (for your real recordings)

  PhysioNet dataset info:
    - 109 subjects, 64 channels, 160 Hz
    - Tasks: rest, left/right fist, both fists/feet
    - We use runs 6,10,14 → left fist vs right fist (closest to open/close)
    - Download: automatically via mne.datasets.eegbci

  Usage:
    python 01_load_data.py
=============================================================================
"""

import mne
import os

mne.set_log_level("WARNING")   # keep console clean

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_DIR    = "eeg_data"
SUBJECT     = 1          # PhysioNet subject number (1–109)
# Runs 6,10,14 = "imagine opening/closing left vs right fist"
PHYSIONET_RUNS = [6, 10, 14]

# OpenBCI export path (set this when you have real data)
OPENBCI_CSV = "eeg_data/OpenBCI-RAW-session1.csv"

# Channel names for your 8-channel Cyton montage
CYTON_CHANNELS = ["C3", "C4", "FC3", "FC4", "CP3", "CP4", "Cz", "FCz"]
CYTON_SFREQ    = 250   # Hz — default OpenBCI sample rate


# ─────────────────────────────────────────────
#  LOADER A — PhysioNet
# ─────────────────────────────────────────────

def load_physionet(subject: int = 1, runs: list = None):
    """
    Download and load PhysioNet EEG Motor Imagery data.
    First run downloads automatically to ~/mne_data/.
    Returns: mne.io.Raw object
    """
    runs = runs or PHYSIONET_RUNS
    print(f"\n{'='*55}")
    print(f"  Loading PhysioNet — Subject {subject:03d}, Runs {runs}")
    print(f"{'='*55}")

    raw_files = mne.datasets.eegbci.load_data(
        subjects    = [subject],
        runs        = runs,
        path        = os.path.expanduser("~/mne_data"),
        verbose     = False,
        update_path = True,
    )

    # Load and concatenate runs
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False)
            for f in raw_files]
    raw  = mne.concatenate_raws(raws)

    # Standardise channel names (e.g. "Fc5." → "FC5")
    mne.datasets.eegbci.standardize(raw)

    # Set montage (10-20 system electrode positions)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

    print(f"  ✓ Loaded:  {len(raw.ch_names)} channels  |  "
          f"{raw.info['sfreq']:.0f} Hz  |  "
          f"{raw.times[-1]:.1f}s total")
    print(f"  ✓ Events:  {set(raw.annotations.description)}")
    return raw


# ─────────────────────────────────────────────
#  LOADER B — OpenBCI Cyton CSV
# ─────────────────────────────────────────────

def load_openbci_csv(filepath: str = OPENBCI_CSV,
                     channels: list = None,
                     sfreq: float   = CYTON_SFREQ):
    """
    Load an OpenBCI Cyton CSV export.

    OpenBCI CSV format (default GUI export):
      Row 0: header comment lines starting with %
      Columns: Sample Index, EXG Ch0..7, Accel X/Y/Z, Timestamp
      Units: microvolts (already scaled)

    Returns: mne.io.Raw object
    """
    channels = channels or CYTON_CHANNELS
    print(f"\n{'='*55}")
    print(f"  Loading OpenBCI CSV: {filepath}")
    print(f"{'='*55}")

    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        print(f"    Export from OpenBCI GUI:  Session → Save Data")
        print(f"    Place the CSV in:         {DATA_DIR}/")
        return None

    import pandas as pd

    # Skip comment lines (start with %)
    with open(filepath) as f:
        skip = sum(1 for line in f if line.startswith("%"))

    df = pd.read_csv(filepath, skiprows=skip)

    # OpenBCI column names: " EXG Channel 0" ... " EXG Channel 7"
    eeg_cols = [c for c in df.columns if "EXG" in c or "Channel" in c][:8]
    data_uv  = df[eeg_cols].values.T          # shape: (n_channels, n_samples)
    data_v   = data_uv * 1e-6                 # convert µV → V for MNE

    # Build MNE info object
    info = mne.create_info(
        ch_names = channels[:data_v.shape[0]],
        sfreq    = sfreq,
        ch_types = "eeg",
    )
    raw = mne.io.RawArray(data_v, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

    print(f"  ✓ Loaded:  {raw.info['nchan']} channels  |  "
          f"{raw.info['sfreq']:.0f} Hz  |  "
          f"{raw.times[-1]:.1f}s total")
    print(f"  ✓ Channels: {raw.ch_names}")
    return raw


# ─────────────────────────────────────────────
#  MARKER ALIGNMENT — match paradigm JSON → MNE annotations
# ─────────────────────────────────────────────

def inject_markers_from_json(raw: mne.io.Raw, json_path: str):
    """
    Load trial markers saved by eeg_mi_paradigm.py and inject them
    as MNE annotations into the raw object.

    The JSON stores Unix timestamps; we align them to the EEG
    recording start time (raw.info['meas_date']).

    Returns: raw with annotations added
    """
    import json

    with open(json_path) as f:
        log = json.load(f)

    rec_start = raw.info["meas_date"].timestamp() if raw.info["meas_date"] else 0.0

    onsets, durations, descriptions = [], [], []
    for trial in log["trials"]:
        onset_rel = trial["onset_unix"] - rec_start
        if onset_rel < 0:
            print(f"  ⚠ Trial {trial['trial_number']} onset before recording start, skipping.")
            continue
        onsets.append(onset_rel)
        durations.append(4.0)              # imagery window duration
        descriptions.append(trial["label"])  # "OPEN" or "CLOSE"

    annotations = mne.Annotations(
        onset       = onsets,
        duration    = durations,
        description = descriptions,
    )
    raw.set_annotations(raw.annotations + annotations)
    print(f"  ✓ Injected {len(onsets)} trial markers into raw")
    return raw


# ─────────────────────────────────────────────
#  QUICK INFO PRINT
# ─────────────────────────────────────────────

def print_raw_info(raw: mne.io.Raw):
    print(f"\n  {'─'*45}")
    print(f"  Channels  : {raw.info['nchan']}")
    print(f"  Sfreq     : {raw.info['sfreq']} Hz")
    print(f"  Duration  : {raw.times[-1]:.1f} s  "
          f"({raw.times[-1]/60:.1f} min)")
    print(f"  Annotations: {set(raw.annotations.description)}")
    print(f"  {'─'*45}\n")


# ─────────────────────────────────────────────
#  MAIN — test both loaders
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── A: PhysioNet (runs now, no hardware needed) ──
    raw_physionet = load_physionet(subject=SUBJECT, runs=PHYSIONET_RUNS)
    print_raw_info(raw_physionet)

    # ── B: OpenBCI (will work once you have recordings) ──
    raw_openbci = load_openbci_csv(OPENBCI_CSV)
    if raw_openbci:
        print_raw_info(raw_openbci)
    else:
        print("  → OpenBCI loader ready, waiting for real recordings.\n")

    print("  Step 1 complete. Run preprocess.py next.\n")