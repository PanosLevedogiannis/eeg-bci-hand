"""
Run CSP + LDA/SVM/Riemannian-MDM classification (MI vs REST) on every
subject in the inMoov dataset, and report accuracy alongside the raw-signal
QC flags from erd_analysis.py — to check whether the S04/S07/S08/S10
acquisition fault (chronic one-sided offset) actually hurts classification,
or whether CAR+bandpass recovers usable signal regardless.

Usage:
    python classify_all_subjects.py --dataset "/path/to/inMoov_Dataset"
"""

import argparse
import glob
import os
import json
from datetime import datetime

from erd_analysis import find_fif, load_and_preprocess
from classify import run_classification

DATASET_DIR = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset"
RESULTS_PATH = "eeg_data/exports/classification_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    args = parser.parse_args()

    subject_dirs = sorted(glob.glob(os.path.join(args.dataset, "S*")))
    if not subject_dirs:
        print(f"No subject folders found in {args.dataset}")
        return

    summary = {}
    for subj_dir in subject_dirs:
        subj = os.path.basename(subj_dir)
        print(f"\n{'#'*60}\n#  {subj}\n{'#'*60}")
        try:
            fif_path = find_fif(subj_dir)
            epochs, qc = load_and_preprocess(fif_path, subj)
            _, cv_results = run_classification(epochs=epochs, subject_id=subj,
                                               source="inmoov")
            best = max(cv_results, key=lambda k: cv_results[k]["mean"])
            summary[subj] = {
                "qc": qc,
                "best_model": best,
                "accuracy_mean": cv_results[best]["mean"],
                "accuracy_std": cv_results[best]["std"],
                "all_results": {k: {"mean": v["mean"], "std": v["std"]}
                                for k, v in cv_results.items()},
            }
        except Exception as e:
            print(f"  FAILED for {subj}: {e}")
            import traceback; traceback.print_exc()
            summary[subj] = {"error": str(e)}

    # ---- Final comparison table: faulty vs clean ----
    print(f"\n{'='*78}")
    print("  CLASSIFICATION vs RAW-SIGNAL QC — FINAL COMPARISON")
    print(f"{'='*78}")
    print(f"  {'Subject':8s} {'Best Acc':>10s} {'RailClip%':>10s} {'OneSided':>9s} {'Fault?':>8s}")

    faulty_accs, clean_accs = [], []
    for subj, r in summary.items():
        if "error" in r:
            print(f"  {subj:8s}  FAILED: {r['error']}")
            continue
        qc = r["qc"]
        is_faulty = qc["rail_warning"] or qc["one_sided"]
        acc = r["accuracy_mean"] * 100
        print(f"  {subj:8s} {acc:9.1f}% {qc['max_rail_pct']:9.1f}% "
              f"{str(qc['one_sided']):>9s} {'FAULT' if is_faulty else 'clean':>8s}")
        (faulty_accs if is_faulty else clean_accs).append(acc)

    if faulty_accs:
        print(f"\n  Faulty subjects   mean accuracy: {sum(faulty_accs)/len(faulty_accs):.1f}%  (n={len(faulty_accs)})")
    if clean_accs:
        print(f"  Clean subjects    mean accuracy: {sum(clean_accs)/len(clean_accs):.1f}%  (n={len(clean_accs)})")
    print(f"{'='*78}\n")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved full results → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
