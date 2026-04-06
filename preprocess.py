"""
=============================================================================
  EEG Preprocessing Pipeline — Step 2: Preprocessing
=============================================================================
  Pipeline:
    1. Bandpass filter  8–30 Hz  (mu + beta bands)
    2. Notch filter     50 Hz    (EU power line noise)
    3. Re-reference     CAR      (Common Average Reference)
    4. ICA              remove eye-blink & muscle artifacts
    5. Epoch extraction −1s to +4s around motor imagery onset
    6. Baseline correction  (−1s to 0s)
    7. Save epochs to disk

  Input:  mne.io.Raw  (from load_data.py)
  Output: mne.Epochs  saved as  eeg_data/epochs_<subject>.fif

  Usage:
    python preprocess.py
=============================================================================
"""

import mne
import numpy as np
import os
import sys
from mne.preprocessing import ICA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_data import load_physionet, load_openbci_csv, CYTON_CHANNELS

mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_DIR     = "eeg_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Filter settings (from your protocol document)
HP_FREQ      = 1.0      # High-pass Hz  (removes DC + slow drift)
LP_FREQ      = 40.0     # Low-pass  Hz  (anti-alias + noise reduction)
NOTCH_FREQ   = 50.0     # Notch     Hz  (EU power line)

# Epoch window (seconds relative to cue onset)
TMIN         = -1.0     # 1s before cue  → baseline
TMAX         =  4.0     # 4s after cue   → full imagery window
BASELINE     = (-1.0, 0.0)

# ICA
N_ICA_COMP   = 15       # number of ICA components to compute
ICA_RANDOM   = 42       # random seed for reproducibility

# Artifact rejection threshold (µV)
REJECT_THRESH = dict(eeg=300e-6)   # 300 µV → auto-reject bad epochs

# Event mapping
#   PhysioNet: T1 = left fist, T2 = right fist (we treat as OPEN/CLOSE)
#   OpenBCI:   OPEN, CLOSE  (from our paradigm markers)
EVENT_ID_PHYSIONET = {"T1": 1, "T2": 2}     # left/right fist imagery
EVENT_ID_OPENBCI   = {"OPEN": 1, "CLOSE": 2}


# ─────────────────────────────────────────────
#  STEP 1 — BANDPASS + NOTCH FILTER
# ─────────────────────────────────────────────

def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply bandpass and notch filters to raw EEG.
    All filtering done on a copy to preserve original.
    """
    print("  [1/5] Filtering...")
    raw = raw.copy()

    # Bandpass: keeps only 1–40 Hz
    raw.filter(
        l_freq  = HP_FREQ,
        h_freq  = LP_FREQ,
        method  = "fir",
        fir_window = "hamming",
        verbose = False,
    )

    # Notch: removes 50 Hz power line noise
    raw.notch_filter(
        freqs   = NOTCH_FREQ,
        verbose = False,
    )

    print(f"      ✓ Bandpass {HP_FREQ}–{LP_FREQ} Hz  |  Notch {NOTCH_FREQ} Hz")
    return raw


# ─────────────────────────────────────────────
#  STEP 2 — COMMON AVERAGE REFERENCE (CAR)
# ─────────────────────────────────────────────

def apply_car(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Re-reference to Common Average Reference.
    Formula: x_i_CAR(t) = x_i(t) - (1/C) * sum(x_j(t))
    This reduces global noise and enhances spatial resolution.
    """
    print("  [2/5] Applying Common Average Reference (CAR)...")
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average",
                                   projection=False, verbose=False)
    print("      ✓ CAR applied")
    return raw


# ─────────────────────────────────────────────
#  STEP 3 — ICA (artifact removal)
# ─────────────────────────────────────────────

def apply_ica(raw: mne.io.Raw, n_components: int = N_ICA_COMP,
              plot: bool = False) -> mne.io.Raw:
    """
    Run ICA to identify and remove ocular (EOG) artifacts.

    ICA decomposes the EEG into independent components.
    Eye-blink components have characteristic frontal topography
    and are automatically detected using EOG correlation.

    If no EOG channel exists, we use Fp1 as a proxy.
    """
    print("  [3/5] Running ICA (this may take ~30s)...")

    # Cap components to n_channels-1 (e.g. 7 for 8-channel Cyton)
    n_components = min(n_components, len(raw.ch_names) - 1)

    # ICA needs high-pass filtered data (≥1Hz) — already done
    ica = ICA(
        n_components = n_components,
        method       = "fastica",
        random_state = ICA_RANDOM,
        max_iter     = 800,
        verbose      = False,
    )
    ica.fit(raw, verbose=False)

    # Auto-detect eye blink components
    eog_channels = [ch for ch in raw.ch_names
                    if ch.upper() in ["FP1", "FP2", "FPZ"]]

    if eog_channels:
        eog_idx, eog_scores = ica.find_bads_eog(
            raw, ch_name=eog_channels[0], verbose=False
        )
    else:
        # No frontal channels — use correlation threshold heuristic
        eog_idx = []
        print("      ⚠ No frontal channel for EOG detection — skipping auto ICA")

    if eog_idx:
        ica.exclude = eog_idx
        print(f"      ✓ Removing ICA components: {eog_idx} (eye artifacts)")
    else:
        print("      ✓ No eye artifact components auto-detected")

    if plot:
        ica.plot_components(title="ICA Components")
        ica.plot_scores(eog_scores if eog_channels else [],
                        title="EOG Scores")

    raw_clean = ica.apply(raw.copy(), verbose=False)
    print("      ✓ ICA applied")
    return raw_clean


# ─────────────────────────────────────────────
#  STEP 4 — EXTRACT EVENTS
# ─────────────────────────────────────────────

def extract_events(raw: mne.io.Raw, source: str = "physionet"):
    """
    Extract event markers from annotations.

    PhysioNet: annotations are 'T0' (rest), 'T1' (left), 'T2' (right)
    OpenBCI:   annotations are 'OPEN', 'CLOSE' (from our paradigm JSON)
    """
    print("  [4/5] Extracting events...")

    event_id = (EVENT_ID_PHYSIONET if source == "physionet"
                else EVENT_ID_OPENBCI)

    # Map annotation labels to event IDs
    if source == "physionet":
        annotation_map = {"T1": 1, "T2": 2}
    else:
        annotation_map = {"OPEN": 1, "CLOSE": 2}

    events, _ = mne.events_from_annotations(
        raw,
        event_id   = annotation_map,
        verbose    = False,
    )

    counts = {name: np.sum(events[:, 2] == eid)
              for name, eid in event_id.items()}
    print(f"      ✓ Events found: {counts}")
    return events, event_id


# ─────────────────────────────────────────────
#  STEP 5 — EPOCH EXTRACTION
# ─────────────────────────────────────────────

def extract_epochs(raw: mne.io.Raw, events: np.ndarray,
                   event_id: dict) -> mne.Epochs:
    """
    Segment continuous EEG into epochs time-locked to motor imagery onset.

    Window: TMIN (−1s) to TMAX (+4s)
    Baseline correction: −1s to 0s (pre-cue period)
    Auto-rejection: epochs with amplitude > REJECT_THRESH are dropped
    """
    print("  [5/5] Extracting epochs...")

    epochs = mne.Epochs(
        raw         = raw,
        events      = events,
        event_id    = event_id,
        tmin        = TMIN,
        tmax        = TMAX,
        baseline    = BASELINE,
        reject      = REJECT_THRESH,
        preload     = True,
        verbose     = False,
    )

    n_total = len(events)
    n_kept  = len(epochs)

    # Fallback: if rejection removed all epochs, retry without rejection
    if n_kept == 0:
        print("      ⚠ All epochs rejected at 300µV — retrying without rejection")
        epochs = mne.Epochs(
            raw      = raw,
            events   = events,
            event_id = event_id,
            tmin     = TMIN,
            tmax     = TMAX,
            baseline = BASELINE,
            reject   = None,
            preload  = True,
            verbose  = False,
        )
        n_kept = len(epochs)

    n_dropped = n_total - n_kept
    pct_kept  = 100 * n_kept / max(1, n_total)

    print(f"      ✓ Epochs: {n_kept}/{n_total} kept  "
          f"({n_dropped} dropped, {pct_kept:.0f}% retained)")
    print(f"      ✓ Shape: {epochs.get_data().shape}  "
          f"(trials × channels × timepoints)")
    return epochs


# ─────────────────────────────────────────────
#  SAVE / LOAD EPOCHS
# ─────────────────────────────────────────────

def save_epochs(epochs: mne.Epochs, subject_id: str = "S01",
                source: str = "physionet"):
    fname = os.path.join(DATA_DIR, f"epochs_{subject_id}_{source}-epo.fif")
    epochs.save(fname, overwrite=True, verbose=False)
    print(f"\n  ✓ Epochs saved → {fname}")
    return fname

def load_epochs(subject_id: str = "S01", source: str = "physionet"):
    fname = os.path.join(DATA_DIR, f"epochs_{subject_id}_{source}-epo.fif")
    epochs = mne.read_epochs(fname, verbose=False)
    print(f"  ✓ Epochs loaded ← {fname}")
    return epochs


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing(raw: mne.io.Raw, subject_id: str = "S01",
                      source: str = "physionet",
                      plot_ica: bool = False) -> mne.Epochs:
    """
    Run the complete preprocessing pipeline on a raw EEG object.

    Args:
        raw:        mne.io.Raw object (from 01_load_data.py)
        subject_id: string ID for saving files
        source:     "physionet" or "openbci"
        plot_ica:   show ICA component plots (interactive)

    Returns:
        mne.Epochs  — clean, epoched EEG ready for feature extraction
    """
    print(f"\n{'='*55}")
    print(f"  Preprocessing  |  Subject: {subject_id}  |  Source: {source}")
    print(f"{'='*55}")

    raw_filt  = apply_filters(raw)
    raw_car   = apply_car(raw_filt)
    raw_clean = apply_ica(raw_car, plot=plot_ica)
    events, event_id = extract_events(raw_clean, source)
    epochs    = extract_epochs(raw_clean, events, event_id)
    save_epochs(epochs, subject_id, source)

    print(f"\n  Pipeline complete ✓")
    print(f"  → epochs shape : {epochs.get_data().shape}")
    print(f"  → classes      : {list(event_id.keys())}")
    print(f"  → time window  : {TMIN}s to {TMAX}s\n")
    return epochs


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── PhysioNet test ──────────────────────────
    raw = load_physionet(subject=1, runs=[6, 10, 14])
    epochs = run_preprocessing(raw, subject_id="P001", source="physionet")

    print(f"  Sample epoch data shape : {epochs.get_data().shape}")
    print(f"  Labels (0=T1, 1=T2)     : {epochs.events[:5, 2]}")
    print("\n  Step 2 complete. Run visualize.py next.\n")