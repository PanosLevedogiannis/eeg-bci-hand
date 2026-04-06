"""
=============================================================================
  EEG Preprocessing Pipeline — Step 4: Classification
=============================================================================
  Pipeline:
    1. CSP (Common Spatial Patterns) — spatial filtering
    2. Log-variance feature extraction
    3. LDA classifier  (fast baseline)
    4. SVM classifier  (better accuracy)
    5. Cross-validation & results report

  Input:  epochs_<subject>_<source>-epo.fif  (from preprocess.py)
  Output: trained model saved to  eeg_data/models/

  Usage:
    python classify.py
=============================================================================
"""

import numpy as np
import os
import json
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

import mne
from mne.decoding import CSP

mne.set_log_level("ERROR")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_epochs, DATA_DIR

MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Time window for classification (seconds within epoch)
# Use only the imagery window, NOT the baseline
T_CLASS_START = 0.5    # 500ms after cue (avoid early artifacts)
T_CLASS_END   = 3.5    # 3.5s (avoid end-of-epoch artifacts)

# Narrow bandpass for classification (mu + beta only)
# Research: 8-30 Hz outperforms 1-40 Hz by 3-8% for CSP-based MI classification
CLF_LFREQ = 8.0
CLF_HFREQ = 30.0

# CSP
# With 8 channels, 4 components (2 per class) reduces overfitting vs 6
N_CSP_COMPONENTS = 4

# Cross-validation
N_FOLDS = 10           # 10-fold stratified CV


# ─────────────────────────────────────────────
#  STEP 1 — PREPARE DATA MATRIX
# ─────────────────────────────────────────────

def prepare_data(epochs: mne.Epochs):
    """
    Extract X (data matrix) and y (labels) from epochs.
    Crops to the classification window [T_CLASS_START, T_CLASS_END].

    Returns:
        X: np.ndarray  shape (n_trials, n_channels, n_times)
        y: np.ndarray  shape (n_trials,)   0=OPEN/T1, 1=CLOSE/T2
    """
    print("  [1/4] Preparing data matrix...")

    # Narrow-band filter for classification: 8-30 Hz (mu + beta)
    # Applied here so visualize.py still sees the full 1-40 Hz spectrum
    epochs_filt = epochs.copy().filter(
        l_freq=CLF_LFREQ, h_freq=CLF_HFREQ, method="fir", verbose=False
    )

    # Crop to classification window
    epochs_crop = epochs_filt.crop(tmin=T_CLASS_START, tmax=T_CLASS_END)

    X = epochs_crop.get_data()   # (trials, channels, timepoints)
    y = epochs_crop.events[:, 2] # event codes

    # Remap event codes to 0/1 — use event_id dict to guarantee correct mapping
    unique = np.unique(y)
    y_bin  = (y == unique[1]).astype(int)
    code_to_name = {v: k for k, v in epochs.event_id.items()}
    label_map = {0: code_to_name[int(unique[0])], 1: code_to_name[int(unique[1])]}

    print(f"      ✓ X shape : {X.shape}  (trials × channels × timepoints)")
    print(f"      ✓ y shape : {y_bin.shape}")
    print(f"      ✓ Classes : {label_map}")
    print(f"      ✓ Balance : {np.sum(y_bin==0)} class-0  |  "
          f"{np.sum(y_bin==1)} class-1")
    return X, y_bin, label_map


# ─────────────────────────────────────────────
#  STEP 2 — BUILD PIPELINES
# ─────────────────────────────────────────────

def build_pipelines():
    """
    Build scikit-learn pipelines:
      CSP → LogVar features → Classifier

    CSP finds spatial filters that maximise variance for one class
    while minimising it for the other — perfect for ERD lateralisation.
    """
    print("  [2/4] Building classification pipelines...")

    csp = CSP(
        n_components  = N_CSP_COMPONENTS,
        reg           = "ledoit_wolf",   # regularization (good for small N)
        log           = True,            # log-variance features
        norm_trace    = False,
    )

    pipelines = {
        "LDA": Pipeline([
            ("csp",     csp),
            ("scaler",  StandardScaler()),
            ("clf",     LDA()),
        ]),
        "SVM (RBF)": Pipeline([
            ("csp",     CSP(n_components=N_CSP_COMPONENTS,
                            reg="ledoit_wolf", log=True)),
            ("scaler",  StandardScaler()),
            ("clf",     SVC(kernel="rbf", C=1.0, gamma="scale",
                            probability=True)),
        ]),
        "SVM (Linear)": Pipeline([
            ("csp",     CSP(n_components=N_CSP_COMPONENTS,
                            reg="ledoit_wolf", log=True)),
            ("scaler",  StandardScaler()),
            ("clf",     SVC(kernel="linear", C=1.0, probability=True)),
        ]),
    }

    # Riemannian MDM: classifies covariance matrices directly on the SPD manifold.
    # No hyperparameters → no overfitting risk; often +2-5% over CSP with small N.
    if HAS_PYRIEMANN:
        pipelines["Riemannian MDM"] = Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("clf", MDM(metric="riemann")),
        ])

    print(f"      ✓ {len(pipelines)} pipelines ready: {list(pipelines.keys())}")
    return pipelines


# ─────────────────────────────────────────────
#  STEP 3 — CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate_all(X, y, pipelines):
    """
    Run stratified k-fold cross-validation for all classifiers.
    StratifiedKFold ensures each fold has balanced class distribution.
    n_splits is capped by the smallest class size to avoid CV errors
    when some subjects have fewer clean epochs.
    """
    min_class = int(min(np.sum(y == c) for c in np.unique(y)))
    n_splits  = min(N_FOLDS, min_class)
    if n_splits < 2:
        raise ValueError(
            f"Too few epochs per class ({min_class}) for cross-validation. "
            f"Need at least 2."
        )
    print(f"  [3/4] Cross-validation ({n_splits}-fold"
          f"{' ← adapted' if n_splits < N_FOLDS else ''})...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        results[name] = {
            "mean":   float(np.mean(scores)),
            "std":    float(np.std(scores)),
            "scores": scores.tolist(),
        }
        bar = "█" * int(results[name]["mean"] * 20)
        print(f"      {name:<15} {results[name]['mean']*100:.1f}% ± "
              f"{results[name]['std']*100:.1f}%  {bar}")

    best_name = max(results, key=lambda k: results[k]["mean"])
    print(f"\n      ★ Best: {best_name}  "
          f"({results[best_name]['mean']*100:.1f}%)")
    return results, best_name


# ─────────────────────────────────────────────
#  STEP 4 — TRAIN FINAL MODEL + SAVE
# ─────────────────────────────────────────────

def train_and_save(X, y, pipelines, best_name, subject_id,
                   label_map, cv_results):
    """
    Train the best pipeline on ALL data and save it.
    This is the model that will be used for real-time control.
    """
    print(f"  [4/4] Training final model: {best_name}...")

    best_pipe = pipelines[best_name]
    best_pipe.fit(X, y)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        MODEL_DIR, f"model_{subject_id}_{timestamp}.joblib"
    )
    joblib.dump(best_pipe, model_path)
    print(f"      ✓ Model saved → {model_path}")

    # Save results JSON
    report = {
        "subject_id":   subject_id,
        "timestamp":    timestamp,
        "n_trials":     len(y),
        "label_map":    {str(k): v for k, v in label_map.items()},
        "t_window":     [T_CLASS_START, T_CLASS_END],
        "n_csp":        N_CSP_COMPONENTS,
        "cv_folds":     N_FOLDS,
        "best_model":   best_name,
        "results":      cv_results,
    }
    report_path = os.path.join(
        MODEL_DIR, f"report_{subject_id}_{timestamp}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"      ✓ Report saved → {report_path}")
    return best_pipe, model_path


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PRINT
# ─────────────────────────────────────────────

def print_confusion_matrix(y, y_pred, label_map):
    cm    = confusion_matrix(y, y_pred)
    names = [label_map[0], label_map[1]]

    print(f"\n  Confusion Matrix ({N_FOLDS}-fold CV predictions):")
    print(f"  {'':>10}  {'Pred ' + names[0]:>14}  {'Pred ' + names[1]:>14}")
    print(f"  {'True ' + names[0]:>10}  {cm[0,0]:>14}  {cm[0,1]:>14}")
    print(f"  {'True ' + names[1]:>10}  {cm[1,0]:>14}  {cm[1,1]:>14}")

    cr = classification_report(y, y_pred,
                                target_names=names,
                                digits=3)
    print(f"\n{cr}")


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────

def run_classification(epochs: mne.Epochs = None,
                       subject_id: str = "P001",
                       source: str = "physionet"):

    if epochs is None:
        epochs = load_epochs(subject_id, source)

    print(f"\n{'='*55}")
    print(f"  Classification  |  {subject_id}  |  {source}")
    print(f"  Window: [{T_CLASS_START}, {T_CLASS_END}]s  |  "
          f"CSP components: {N_CSP_COMPONENTS}")
    print(f"{'='*55}")

    X, y, label_map = prepare_data(epochs)
    pipelines        = build_pipelines()
    cv_results, best = cross_validate_all(X, y, pipelines)
    model, path      = train_and_save(X, y, pipelines, best,
                                      subject_id, label_map, cv_results)

    # Run cross_val_predict once for the best model (reuses the same CV split)
    min_class = int(min(np.sum(y == c) for c in np.unique(y)))
    n_splits  = min(N_FOLDS, min_class)
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred    = cross_val_predict(pipelines[best], X, y, cv=cv)
    print_confusion_matrix(y, y_pred, label_map)

    print(f"\n  Classification complete ✓")
    print(f"  → Best model : {best}")
    print(f"  → Accuracy   : {cv_results[best]['mean']*100:.1f}% ± "
          f"{cv_results[best]['std']*100:.1f}%")
    print(f"  → Saved to   : {path}")
    print(f"\n  Next: run realtime.py for live EEG → Arduino control\n")

    return model, cv_results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_classification(subject_id="P001", source="physionet")