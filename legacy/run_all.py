"""
=============================================================================
  EEG Motor Imagery Pipeline — Master Script
=============================================================================
  Runs the complete offline pipeline for a single subject:
    1. load_data.py    → load PhysioNet or OpenBCI data
    2. preprocess.py   → filter, CAR, ICA, epoch extraction
    3. visualize.py    → ERD/ERS, spectrograms, topomaps
    4. classify.py     → CSP + LDA/SVM/Riemannian + cross-validation

  For multi-subject validation:
    python multi_subject_analysis.py   (subjects 1–10, summary figure + JSON)

  For real-time BCI demo (requires hardware):
    python realtime.py                 (OpenBCI Cyton + Arduino)

  Usage:
    python run_all.py
    python run_all.py --subject 3      (run a different subject)

  Or step by step:
    python load_data.py
    python preprocess.py
    python visualize.py
    python classify.py
=============================================================================
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_data    import load_physionet
from preprocess   import run_preprocessing
from visualize    import run_visualization
from classify     import run_classification

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

PHYSIONET_RUNS = [6, 10, 14]   # left vs right fist imagery
SOURCE         = "physionet"   # "physionet" or "openbci"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG MI Pipeline — InMoov i2 Robotic Hand"
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="PhysioNet subject number 1-109 (default: 1)"
    )
    args = parser.parse_args()

    subject_num = args.subject
    subject_id  = f"P{subject_num:03d}"

    print("\n" + "="*55)
    print("  EEG MI Pipeline — InMoov i2 Robotic Hand")
    print(f"  Subject: {subject_id}  |  Source: {SOURCE}")
    print("="*55 + "\n")

    # 1. Load
    raw = load_physionet(subject=subject_num, runs=PHYSIONET_RUNS)

    # 2. Preprocess → epochs
    epochs = run_preprocessing(raw, subject_id=subject_id, source=SOURCE)

    # 3. Visualize → figures
    run_visualization(epochs, subject_id=subject_id, source=SOURCE)

    # 4. Classify → trained model
    model, results = run_classification(epochs, subject_id=subject_id,
                                        source=SOURCE)

    best = max(results, key=lambda k: results[k]["mean"])
    print("="*55)
    print("  Pipeline complete!")
    print(f"  Best classifier : {best}")
    print(f"  Accuracy        : {results[best]['mean']*100:.1f}% "
          f"± {results[best]['std']*100:.1f}%")
    print(f"  Figures  → eeg_data/figures/")
    print(f"  Model    → eeg_data/models/")
    print("="*55 + "\n")