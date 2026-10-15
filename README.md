# Robotic Arm Control through a Brain–Computer Interface Using EEG Motor Imagery

*Έλεγχος Ρομποτικού Βραχίονα μέσω Brain–Computer Interface με χρήση EEG και Νοερών Κινήσεων*

**Diploma Thesis — Department of Electrical and Computer Engineering, University of the Peloponnese**  
**Supervisor:** Assoc. Prof. Athanasios Koutras

**▶ Live site: https://panoslevedogiannis.github.io/eeg-bci-hand/** — the protocol, the results for all 14 participants, and an interactive 3D model of the hand.

---

## Overview

A Brain-Computer Interface (BCI) that decodes motor imagery EEG and controls a 5-finger robotic hand. The system classifies imagined right-hand movement against rest from scalp EEG and sends servo commands to an InMoov i2 robotic hand via Arduino.

**Task:** Two-class motor imagery — imagine closing your right hand vs rest.

**Dataset:** 14 participants recorded for this thesis, 320 trials each (4,480 total). Evaluation uses this primary dataset exclusively.

**Results:** mean accuracy **60.8%** (10-fold CV) and **59.1%** (Leave-One-Run-Out), both significant above chance (Wilcoxon, *p* < 0.01).

**Main methodological finding:** the choice of validation scheme materially changes the conclusions. Under LORO — where the model is tested on a run it has never seen — only **6 of 14** participants decode reliably, while 10-fold cross-validation would suggest nearly all of them succeed. Since any practical BCI must generalise to a new session, LORO is the scheme with operational meaning.

---

## System Architecture

```
OpenBCI Cyton (8-ch, 250 Hz)
        │
        ▼  BrainFlow streaming
┌────────────────────────────────────┐
│  8–30 Hz + notch 50 Hz + CAR       │  preprocessing
│  Sliding window: 1000ms / 125ms    │
└────────────────────────────────────┘
        │
        ▼  CSP (4 comp, Ledoit-Wolf) + LDA / SVM / Riemannian MDM
┌────────────────────────────────────┐
│  Offline-trained classifier        │
│  Majority vote over 7 predictions  │
└────────────────────────────────────┘
        │  Serial  "mid\n" / "min\n"
        ▼
   Arduino + Adafruit PCA9685 (I2C)
        │  50 Hz PWM
        ▼
   InMoov i2 Hand (5 servos)
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| EEG amplifier | OpenBCI Cyton (8-channel, 250 Hz) |
| Electrode montage | C3, C4, FC3, FC4, CP3, CP4, Cz, FCz (motor cortex) |
| Microcontroller | Arduino (any model with I2C) |
| PWM driver | Adafruit PCA9685 16-channel servo driver (I2C) |
| Robotic hand | InMoov i2 — 5 servos (thumb, index, middle, ring, wrist/pinky) |
| Computer | macOS / Linux — Python 3.10+ |

---

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `eeg_mi_paradigm.py` | Graz-BCI recording paradigm (Pygame), 4 runs × 80 trials, LSL markers |
| 2 | `erd_analysis.py` | Raw-signal QC, then bandpass 1–40 Hz + notch 50 Hz + CAR + epoching + amplitude rejection (200 µV); ERD/ERS figures |
| 3 | `classify_all_subjects.py` | CSP + LDA / SVM / Riemannian MDM per participant, 10-fold CV (`classify.py`) |
| 4 | `reliability_analysis.py` | Leave-One-Run-Out CV, 1,000-permutation tests, run-by-run trend |
| 5 | `group_level_stats.py` | Wilcoxon signed-rank against chance, 10-fold and LORO |
| 6 | `replay_gui.py` / `replay_demo.py` | Train on runs 1–3, replay run 4 through the classifier to the hand |
| 7 | `realtime_gui.py` | Live Cyton EEG → classifier → Arduino (not part of the reported evaluation) |

Every analysis script reads the same dataset folder: pass `--dataset /path/to/inMoov_Dataset`
or set `INMOOV_DATASET` once.

Earlier scripts (PhysioNet pipeline, first-cohort analyses) are kept in [`legacy/`](legacy/).

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd eeg-bci-hand

# Create virtual environment
python3 -m venv eeg_bci_env
source eeg_bci_env/bin/activate        # Windows: eeg_bci_env\Scripts\activate

# Install dependencies
pip install -r requirements_pipeline.txt
```

**Key dependencies:**

| Package | Purpose |
|---------|---------|
| `mne` | EEG processing, epoching, CSP |
| `scikit-learn` | CSP, LDA, SVM classifiers |
| `pyriemann` | Riemannian geometry (MDM classifier) |
| `brainflow` | OpenBCI real-time streaming |
| `pyserial` | Arduino serial communication |
| `pygame` | Data collection GUI |
| `matplotlib`, `scipy`, `numpy`, `joblib` | Core utilities |

---

## Reproducing the results

```bash
source eeg_bci_env/bin/activate
export INMOOV_DATASET=/path/to/inMoov_Dataset      # folder of S01/ … S14/

python classify_all_subjects.py                    # 10-fold, all four classifiers
python reliability_analysis.py --perms 1000        # leave-one-run-out + permutation tests
python group_level_stats.py                        # group-level Wilcoxon (both schemes)
python render_thesis_figures.py                    # print-ready figures
```

Results are written to `eeg_data/exports/`, figures to `eeg_data/figures/`.

---

## Data Collection (with OpenBCI Cyton)

```bash
python eeg_mi_paradigm.py
```

Set `cyton_port` in `SessionConfig` first. `simulate=True` there records from BrainFlow's
synthetic board and must never be used for a real session.

Each session starts with an eyes-open and an eyes-closed baseline (a beep marks its end), then 4 runs of
80 trials (40 MI + 40 REST in random order) with a mandatory break between runs:

| Phase | Duration | Screen |
|-------|----------|--------|
| Rest | 1.5–2.5 s (jittered) | Blank |
| Prepare | 2.0 s | Fixation cross |
| Cue | 2.25 s | "MI" or "REST" |
| Imagery | 4.0 s | Fixation cross only — marker 1 (MI) or 2 (REST) at its start |
| Feedback | 1.5 s | "You may blink" |

Markers go to an LSL stream and are saved with the EEG as FIF annotations.

**Electrode Placement (10-20 system):**

```
     FC3  FCz  FC4
     C3   Cz   C4
     CP3       CP4

Reference: right earlobe / mastoid
Ground:    Fpz or AFz
```

---

## Real-Time BCI — Graphical Monitor

### Simulate mode (no EEG hardware needed — Arduino is real)

```bash
python realtime_gui.py --simulate
```

Uses BrainFlow's synthetic board for EEG. The hand physically opens and closes every 3 seconds. Useful for testing the full pipeline and demonstrating the hardware.

### Full hardware mode

```bash
# 1. Find your serial ports
ls /dev/cu.*        # macOS
ls /dev/ttyUSB*     # Linux

# 2. Update ports at the top of realtime_gui.py
ARDUINO_PORT = "/dev/cu.usbmodem1101"
CYTON_PORT   = "/dev/cu.usbserial-XXXX"

# 3. Run with the model classify_all_subjects.py saved for that participant
python realtime_gui.py --subject S11
```

The class names are read from the report saved next to the model, so MI always closes
the hand and REST always opens it.

**The GUI shows:**
- Live scrolling EEG waveform (C3 = blue, C4 = orange)
- Classification result — OPEN / CLOSE with confidence bar (green / red)
- Hand state — updates when a command is sent to the Arduino
- Cue countdown (simulate mode: "IMAGINE: OPEN — 2.4s")

**Real-time parameters:**

| Parameter | Value |
|-----------|-------|
| Sampling rate | 250 Hz |
| Filter | 8–30 Hz Butterworth (order 4) + 50 Hz notch, causal, over a 4 s buffer |
| Units | BrainFlow µV converted to V, as in training |
| Window | 1000 ms (last second of the filtered buffer) |
| Step | 125 ms |
| Vote buffer | 7 predictions |
| Confidence threshold | 60% |
| Command hold | 2 s minimum between hand changes |

### Arduino serial protocol (PCA9685)

| Python sends | Arduino does | Arduino replies |
|---|---|---|
| `"mid\n"` | Open hand (all fingers extend) | `"OPEN"` |
| `"min\n"` | Close hand (all fingers curl) | `"CLOSED"` |

Fingers move sequentially with 100 ms delay to prevent power surge.

---

## Demo — recorded EEG driving the real hand

Two scripts replay recorded trials through the classifier and drive the physical hand. Both train on runs 1–3 and replay **only run 4**, so every trial the hand reacts to is data the model has never seen. This is a genuine generalisation test, not playback of a stored movement sequence — the classifier decides on the spot and its mistakes are visible.

```bash
# Presentation GUI: scrolling EEG + instruction + verdict + hand state
python replay_gui.py --subject S11

python replay_gui.py --subject S11 --trials 12    # short version
python replay_gui.py --subject S11 --no-arduino   # screen only, no hardware
python replay_gui.py --subject S02                # a participant who fails

# Terminal version, prints a per-trial table
python replay_demo.py --subject S11
```

`space` pauses, `q` quits.

The GUI shows the EEG of the trial being classified, then the instruction the participant was actually given, and only *then* the classifier's decision — so the prediction is never revealed before the evidence.

**Suggested sequence for a demo:** run S11, then S02. The first tracks the instruction closely; the second moves the hand almost at random. The contrast makes the LORO finding visible in a way no table does.

**Result for S11:** 58/80 correct on the held-out run = **72.5%**.

---

## Classification Results — own dataset (N = 14)

### Classifier comparison (10-fold CV)

| Classifier | Mean ± Std | Range |
|------------|------------|-------|
| SVM (Linear) | **61.1% ± 7.7%** | 50.3–73.8% |
| LDA | 60.8% ± 7.6% | 49.9–75.6% |
| SVM (RBF) | 60.5% ± 7.4% | 49.0–72.3% |
| Riemannian MDM | 60.5% ± 7.4% | 51.4–72.8% |

All four perform equivalently. The spread between classifiers (0.6 pp) is far smaller than the spread between participants (σ ≈ 7.5), which matches the literature: once CSP has done the feature extraction, the choice of classifier matters much less than signal quality and user aptitude.

### Leave-One-Run-Out — the stricter test

Trained on runs 1–3, tested on run 4. Verdict is PASS when *p* < 0.05, accuracy > 60% and κ > 0.2.

| | Participants | Accuracy |
|---|---|---|
| **PASS** | S01, S04, S05, S11, S12, S13 | 62.3–74.7% |
| **BORDERLINE** | S03, S06, S10 | 55.6–57.8% |
| **FAIL** | S02, S07, S08, S09, S14 | 46.7–53.2% |

Group-level Wilcoxon against chance stays significant under LORO (*W* = 94, *p* = 0.003), so the effect is real at population level even though it does not appear in every individual.

**Why this matters:** S02 scores 58.2% under 10-fold but 46.7% under LORO — below chance, *p* = 0.77. Per-class F1 exposes the mechanism (MI 0.28 vs REST 0.58): the classifier simply predicts *rest* almost always. Random 10-fold let it see every run during training and hid a complete failure to generalise.

> 5 of 14 participants (36%) fail to decode reliably — above the 15–30% BCI-illiteracy rate reported in the literature.

---

## Project Structure

```
eeg-bci-hand/
├── eeg_mi_paradigm.py          # Data collection — Graz-BCI paradigm (Pygame)
│
├── erd_analysis.py             # Preprocessing + ERD/ERS + raw-signal QC
├── classify.py                 # CSP + LDA / SVM / Riemannian MDM
├── classify_all_subjects.py    # Batch classification over the cohort
├── reliability_analysis.py     # Leave-One-Run-Out CV + permutation tests
├── group_level_stats.py        # Group-level Wilcoxon
│
├── replay_gui.py               # DEMO — recorded EEG → classifier → hand (GUI)
├── replay_demo.py              # DEMO — same, terminal output
├── realtime_gui.py             # Live BCI from a streaming Cyton board
│
├── render_thesis_figures.py    # Print-ready figures from saved results
├── prepare_appendix_figures.py # Downsample per-participant figures
├── generate_report.py          # PDF report for the supervisor
│
├── legacy/                     # Earlier scripts, not used for the results
├── site/                       # Source of the thesis site (published from docs/)
├── docs/                       # Published site (GitHub Pages)
├── arduino/
│   └── InMoov_EEG_Control/
│       └── InMoov_EEG_Control.ino   # PCA9685 sketch, 9600 baud
└── eeg_data/
    ├── figures/thesis/         # The two group-level figures used above
    └── models/                 # Per-participant 10-fold reports (JSON)
```

### Data availability

The participant recordings are **not** included in this repository. They are personal data collected under informed consent for this study and were not released publicly. The analysis scripts take a `--dataset` argument pointing at a local folder of `SXX/` subdirectories, each containing the raw `.fif` and its marker files.

---

## Scientific Background

| Concept | Role in this project |
|---------|---------------------|
| **ERD/ERS** (Event-Related Desynchronization/Synchronization) | Motor imagery suppresses mu/beta power at C3 or C4 depending on hand |
| **CSP** (Common Spatial Patterns) | Optimal linear spatial filter for two-class ERD lateralization |
| **LDA** | Lightweight, regularized linear classifier — best on small EEG datasets |
| **Riemannian MDM** | Classifies covariance matrices on the SPD manifold — no hyperparameters |
| **Graz-BCI protocol** | Standard cue-based motor imagery paradigm (Pfurtscheller & Neuper, 2001) |

**Key references:**
- Pfurtscheller & Neuper (2001). Motor imagery and direct brain-computer communication. *Proc. IEEE*.
- Blankertz et al. (2008). The BCI competition 2003. *IEEE TBME*.
- Barachant et al. (2012). Multiclass BCI classification by Riemannian geometry. *IEEE TBME*.
- Schalk et al. (2004). BCI2000 / PhysioNet EEG dataset. *IEEE TBME*.

---

## Status

- [x] Offline pipeline — preprocess, ERD/ERS, classify
- [x] Data collection paradigm — Graz-BCI, 4 runs × 80 trials
- [x] **14 participants recorded** — 4,480 trials total
- [x] Raw-signal quality control (rail-clipping and reference-offset detection)
- [x] ERD/ERS analysis confirming genuine mu-band desynchronisation
- [x] Classifier comparison — LDA, SVM (RBF + linear), Riemannian MDM
- [x] Leave-One-Run-Out validation with 1,000-permutation significance tests
- [x] Arduino sketch — PCA9685 I2C, 5 servos
- [x] **Replay demo — unseen recorded trials driving the physical hand (72.5%)**
- [x] Real-time GUI checked offline against the training pipeline (same features, same class mapping)
- [ ] Closed-loop live session with real-time visual feedback to the user
- [ ] Cross-session validation (models trained one day, tested another)

---

*Thesis: "Robotic Arm Control through a Brain–Computer Interface Using EEG Motor Imagery"*  
*Supervisor: Assoc. Prof. Athanasios Koutras*
