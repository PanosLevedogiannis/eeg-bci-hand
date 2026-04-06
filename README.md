# EEG-Based Control of the InMoov i2 Robotic Hand

**BSc Thesis — Department of Electronics Engineering**  
**Supervisor:** Asst. Prof. Athanasios Koutras

---

## Overview

A Brain-Computer Interface (BCI) that decodes motor imagery EEG signals and controls a 5-finger robotic hand in real time. The system classifies imagined hand movements (open vs close) from scalp EEG and sends servo commands to an InMoov i2 robotic hand via Arduino.

**Task:** Two-class motor imagery — imagine opening vs closing your hand.  
**Result:** 74.8% ± 20.2% mean accuracy across 10 subjects (PhysioNet benchmark), 8/10 subjects above chance.

---

## System Architecture

```
OpenBCI Cyton (8-ch, 250 Hz)
        │
        ▼  BrainFlow streaming
┌────────────────────────────────────┐
│  IIR Bandpass 8–30 Hz + CAR        │  preprocessing
│  Sliding window: 1000ms / 125ms    │
└────────────────────────────────────┘
        │
        ▼  CSP + LDA (best: 74.8% mean)
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
| 1 | `load_data.py` | Load PhysioNet benchmark or OpenBCI recordings |
| 2 | `preprocess.py` | Bandpass 1–40 Hz + notch + CAR + ICA + epoching |
| 3 | `visualize.py` | ERD/ERS maps, spectrograms, topomaps |
| 4 | `classify.py` | CSP + LDA / SVM / Riemannian MDM (10-fold CV) |
| 5 | `realtime_gui.py` | Live EEG → classify → Arduino + graphical monitor |

**Supporting scripts:**

| Script | Description |
|--------|-------------|
| `run_all.py` | Runs steps 1–4 for one subject |
| `multi_subject_analysis.py` | Batch benchmark on PhysioNet subjects 1–N |
| `eeg_mi_paradigm.py` | Graz-BCI data collection paradigm (Pygame GUI) |

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
| `mne` | EEG processing, epoching, ICA |
| `scikit-learn` | CSP, LDA, SVM classifiers |
| `pyriemann` | Riemannian geometry (MDM classifier) |
| `brainflow` | OpenBCI real-time streaming |
| `pyserial` | Arduino serial communication |
| `pygame` | Data collection GUI |
| `matplotlib`, `scipy`, `numpy`, `joblib` | Core utilities |

---

## Quickstart — No Hardware Needed

The pipeline auto-downloads the PhysioNet EEG Motor Imagery dataset (~60 MB per subject).

```bash
source eeg_bci_env/bin/activate

# Run full pipeline for subject 1 (download → preprocess → visualize → classify)
python run_all.py

# Different subject
python run_all.py --subject 5

# Benchmark across 10 subjects
python multi_subject_analysis.py
```

**Output:**
- `eeg_data/figures/` — ERD/ERS, spectrogram, topomap, multi-subject comparison
- `eeg_data/models/` — trained classifier (`.joblib`) + report (`.json`)

---

## Data Collection (with OpenBCI Cyton)

```bash
python eeg_mi_paradigm.py
```

The Graz-BCI paradigm guides the subject through 40 trials per class:

| Phase | Duration | Description |
|-------|----------|-------------|
| Fixation cross | 2 s | Prepare, stay still |
| Cue arrow | 1 s | LEFT = imagine opening, RIGHT = imagine closing |
| Imagery window | 4 s | Maintain imagined movement |
| Rest | 2–3 s (jittered) | Relax |

Trial markers saved to `eeg_data/session_log_<timestamp>.json` for offline alignment.

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

# 3. Run
python realtime_gui.py
```

**The GUI shows:**
- Live scrolling EEG waveform (C3 = blue, C4 = orange)
- Classification result — OPEN / CLOSE with confidence bar (green / red)
- Hand state — updates when a command is sent to the Arduino
- Cue countdown (simulate mode: "IMAGINE: OPEN — 2.4s")

**Real-time parameters:**

| Parameter | Value |
|-----------|-------|
| Sampling rate | 250 Hz |
| Bandpass filter | 8–30 Hz (IIR Butterworth, order 4) |
| Window | 1000 ms |
| Step | 125 ms |
| Vote buffer | 7 predictions |
| Confidence threshold | 60% |
| Command hold | 2 s |

### Arduino serial protocol (PCA9685)

| Python sends | Arduino does | Arduino replies |
|---|---|---|
| `"mid\n"` | Open hand (all fingers extend) | `"OPEN"` |
| `"min\n"` | Close hand (all fingers curl) | `"CLOSED"` |

Fingers move sequentially with 100 ms delay to prevent power surge.

---

## Classification Results

### Single subject — PhysioNet P001

| Classifier | Accuracy (10-fold CV) |
|------------|----------------------|
| LDA | **98.0%** ± 6.0% |
| SVM (Linear) | 98.0% ± 6.0% |
| SVM (RBF) | 95.5% ± 9.1% |
| Riemannian MDM | 98.0% ± 6.0% |

### Multi-subject benchmark — PhysioNet (N = 10)

| Classifier | Mean ± Std | Min | Max |
|------------|------------|-----|-----|
| **LDA** | **74.8% ± 20.2%** | 43.5% | 98.0% |
| SVM (Linear) | 71.9% ± 20.1% | 43.5% | 100.0% |
| SVM (RBF) | 71.7% ± 19.9% | 38.5% | 97.5% |
| Riemannian MDM | 59.2% ± 23.0% | 27.0% | 98.0% |

- **8/10 subjects** above chance (>55%) with LDA
- Chance level: 50% (binary classification)

> **Note:** PhysioNet uses 64 channels at 160 Hz. With the 8-channel OpenBCI Cyton on real subject data, expected accuracy is 65–80% — still well above chance for BCI control.

---

## Project Structure

```
eeg-bci-hand/
├── run_all.py                  # Master script: steps 1–4
├── load_data.py                # Step 1 — data loading
├── preprocess.py               # Step 2 — preprocessing + ICA
├── visualize.py                # Step 3 — figures
├── classify.py                 # Step 4 — CSP + classifiers
├── realtime_gui.py             # Step 5 — live BCI + graphical monitor
├── eeg_mi_paradigm.py          # Data collection GUI (Graz-BCI)
├── multi_subject_analysis.py   # Multi-subject benchmark
├── requirements_pipeline.txt   # Python dependencies
├── arduino/
│   └── InMoov_EEG_Control/
│       └── InMoov_EEG_Control.ino   # PCA9685 Arduino sketch
└── eeg_data/
    ├── figures/                # Generated plots (ERD, topomap, ...)
    └── models/                 # Trained classifiers + JSON reports
```

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

- [x] Offline pipeline — load, preprocess, visualize, classify
- [x] PhysioNet 10-subject benchmark (LDA 74.8% mean)
- [x] Real-time BCI with sliding window + majority vote
- [x] Graphical monitor — live EEG + prediction display
- [x] Simulate mode — full pipeline demo without EEG hardware
- [x] Arduino sketch — PCA9685 I2C, 5-finger sequential control
- [x] Data collection paradigm — Graz-BCI, 40 trials/class
- [ ] Real EEG session with OpenBCI Cyton (dongle pending)
- [ ] Subject-specific model training on personal data
- [ ] Live closed-loop demo validation

---

*Thesis: "EEG-Based Control of the InMoov i2 Robotic Hand"*  
*Supervisor: Asst. Prof. Athanasios Koutras*
