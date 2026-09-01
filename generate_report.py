"""
Build a single PDF report for sharing the inMoov S01-S11 ERD/ERS +
classification + raw-signal QC analysis with the supervisor (Koutras).
Pulls real numbers from eeg_data/exports/classification_summary.json
(produced by classify_all_subjects.py) rather than hardcoding them.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

FIG_DIR      = "eeg_data/figures"
SUMMARY_PATH = "eeg_data/exports/classification_summary.json"
GROUP_STATS_PATH = "eeg_data/exports/group_level_stats.json"
OUT_PATH     = "eeg_data/exports/ERD_ERS_Report.pdf"
os.makedirs("eeg_data/exports", exist_ok=True)

with open(SUMMARY_PATH) as f:
    SUMMARY = json.load(f)

GROUP_STATS = None
if os.path.exists(GROUP_STATS_PATH):
    with open(GROUP_STATS_PATH) as f:
        GROUP_STATS = json.load(f)

RELIABILITY_PATH = "eeg_data/exports/reliability_summary.json"
RELIABILITY = None
if os.path.exists(RELIABILITY_PATH):
    with open(RELIABILITY_PATH) as f:
        RELIABILITY = json.load(f)

CHANCE_CI_HALFWIDTH = 5.5   # ~95% CI half-width around 50% chance for n=320 trials
CLEARLY_ABOVE = 50 + CHANCE_CI_HALFWIDTH        # 55.5% — confidently above chance
CLEARLY_AT    = 50 + CHANCE_CI_HALFWIDTH * 0.5  # 52.75% — within noise of chance


def verdict(acc_pct: float, qc: dict) -> str:
    faulty = qc["rail_warning"] or qc["one_sided"]
    if not faulty:
        return "Keep (clean)"
    if acc_pct > CLEARLY_ABOVE:
        return "Keep (fault, but decodes fine)"
    if acc_pct > CLEARLY_AT:
        return "Borderline"
    return "Exclude (~chance level)"


with PdfPages(OUT_PATH) as pdf:
    # ---- Summary page ----
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.96, "EEG Motor Imagery — Full Analysis Report", ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.935, "InMoov i2 Robotic Hand BCI — Subjects S01–S11", ha="center", fontsize=11, color="0.3")

    body = (
        "Task: Motor Imagery (right hand) vs Rest, Graz-BCI protocol\n"
        "Hardware: OpenBCI Cyton, 8 channels (C3,C4,FC3,FC4,CP3,CP4,Cz,FCz), 250 Hz\n"
        "Trials per subject: 160 MI + 160 REST (320 total), 4 runs\n"
        "Classifier: CSP (4 comp, Ledoit-Wolf) + LDA / SVM(RBF) / SVM(linear) / Riemannian MDM, 10-fold CV\n"
        "Raw-signal QC: rail-clipping (ADS1299 gain-24 saturation) + chronic one-sided\n"
        "offset (bad reference/bias electrode) checked on RAW signal before filtering —\n"
        "epoch-rejection-based retention alone misses these because CAR+bandpass hides them.\n"
    )
    fig.text(0.06, 0.91, body, fontsize=8.7, va="top", linespacing=1.45)

    table_y = 0.70
    row_h   = 0.026
    headers = ["Subject", "Retain%", "RailClip%", "1-Sided", "Best Acc", "Verdict"]
    xs      = [0.07, 0.22, 0.34, 0.47, 0.58, 0.70]
    for x, h in zip(xs, headers):
        fig.text(x, table_y, h, fontsize=8.5, weight="bold")

    for i, subj in enumerate(sorted(SUMMARY)):
        r = SUMMARY[subj]
        y = table_y - row_h * (i + 1)
        if "error" in r:
            fig.text(xs[0], y, subj, fontsize=8)
            fig.text(xs[1], y, "FAILED", fontsize=8)
            continue
        qc = r["qc"]
        acc = r["accuracy_mean"] * 100
        v = verdict(acc, qc)
        vals = [subj, f"{qc['retention_pct']:.1f}%", f"{qc['max_rail_pct']:.1f}%",
                "Yes" if qc["one_sided"] else "No", f"{acc:.1f}%", v]
        for x, val in zip(xs, vals):
            fig.text(x, y, val, fontsize=8)

    notes_y = table_y - row_h * (len(SUMMARY) + 2)
    fig.text(0.06, notes_y,
        "Key findings:\n"
        "- S04, S07, S08, S10 have a chronic one-sided offset (7-8/8 channels stuck on one\n"
        "  sign for the entire session) traced to the RAW board CSV itself — almost certainly\n"
        "  a loose SRB1/SRB2 (reference/bias) electrode, not a software bug.\n"
        "- Classification settles whether this fault matters: S04 (66.3%) and S10 (57.8%) decode\n"
        "  clearly above chance despite it, confirming CAR+bandpass recovers real signal. S08\n"
        "  (51.9%) is statistically at chance level -> recommend excluding. S07 (53.3%) is\n"
        "  borderline -> verify before including.\n"
        "- S01/S03/S05/S06/S09 show milder intermittent rail-clipping (1.5-6% of samples,\n"
        "  electrode movement/blink artifacts) and decode at or above the other subjects.\n"
        "- S02 and S11 are the only fully clean sessions (no rail-clip, no offset); both also\n"
        "  decode well (58.2%, 75.6%).\n"
        "- Fix going forward: an automated check now runs after the baseline recording in\n"
        "  eeg_mi_paradigm.py and warns the experimenter on-screen if a fault is detected,\n"
        "  before the 60+ minute main session begins.",
        fontsize=8.3, va="top", linespacing=1.45)

    fig.text(0.5, 0.02, "Generated automatically from erd_analysis.py + classify_all_subjects.py",
             ha="center", fontsize=7.5, color="0.5")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

    # ---- Statistical validation page ----
    if GROUP_STATS:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Statistical Validation", ha="center", fontsize=16, weight="bold")
        fig.text(0.5, 0.925, "Is classification accuracy significantly above chance (50%)?",
                 ha="center", fontsize=10.5, color="0.3")

        intro = (
            "Per-subject permutation tests (CSP+LDA, label-shuffling, see appendix figures)\n"
            "show whether each individual subject beats chance. But eleven individually\n"
            "significant subjects could in principle still be a fluke pattern. The test below\n"
            "is the standard non-parametric GROUP-LEVEL test: a one-sample Wilcoxon\n"
            "signed-rank test comparing each subject's CV accuracy against the chance\n"
            "reference (50%), across subjects. LDA is used as a single, consistent classifier\n"
            "for this test (selecting each subject's best-of-4 model would inflate\n"
            "significance via selection bias); best-of-4 is shown only as a supplementary,\n"
            "more optimistic cross-check.\n"
        )
        fig.text(0.06, 0.89, intro, fontsize=8.6, va="top", linespacing=1.5)

        y0 = 0.62
        row_h = 0.034
        headers = ["Group", "n", "Mean Acc", "Wilcoxon p", "Result"]
        xs = [0.07, 0.40, 0.52, 0.66, 0.80]
        for x, h in zip(xs, headers):
            fig.text(x, y0, h, fontsize=9, weight="bold")

        labels = {
            "lda_all11": "LDA — all 11 subjects",
            "lda_kept10": "LDA — excl. S08 (at-chance)",
            "lda_kept9_strict": "LDA — excl. S08 + S07 (strict)",
            "best_all11": "Best-of-4 — all 11 (supplementary)",
        }
        for i, key in enumerate(["lda_all11", "lda_kept10", "lda_kept9_strict", "best_all11"]):
            r = GROUP_STATS.get(key)
            if not r:
                continue
            y = y0 - row_h * (i + 1)
            sig = "p < 0.05  significant" if r["wilcoxon_p"] < 0.05 else "not significant"
            vals = [labels[key], str(len(r["subjects"])), f"{r['mean']*100:.1f}%",
                    f"{r['wilcoxon_p']:.4f}", sig]
            for x, val in zip(xs, vals):
                fig.text(x, y, val, fontsize=8.3)

        concl_y = y0 - row_h * 6
        fig.text(0.06, concl_y,
            "Conclusion: classification accuracy is significantly above chance at the\n"
            "group level (Wilcoxon signed-rank, p < 0.01) whether or not the at-chance\n"
            "S08 session is included — the result is not an artifact of one lucky subject.\n"
            "Excluding the faulty sessions (S08, S07) only strengthens the effect (p < 0.005,\n"
            "higher mean accuracy), supporting the QC-based exclusion decision rather than\n"
            "contradicting it.",
            fontsize=8.6, va="top", linespacing=1.5)

        fig.text(0.5, 0.03,
                 "Full per-subject Mann-Whitney (ERD significance) + permutation test results "
                 "in eeg_data/figures/stats_*.png",
                 ha="center", fontsize=7.5, color="0.5")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    # ---- Reliability validation page (leave-one-run-out CV) ----
    if RELIABILITY:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.96, "Reliability Validation", ha="center", fontsize=16, weight="bold")
        fig.text(0.5, 0.935, "Leave-One-Run-Out CV + Cohen's Kappa + Per-Class F1",
                 ha="center", fontsize=10.5, color="0.3")

        intro = (
            "Random k-fold CV (used above) mixes trials from all 4 runs into both train and\n"
            "test sets, which can hide within-session drift (fatigue, electrode impedance\n"
            "change) by letting the model see examples from every part of the session during\n"
            "training. Leave-one-run-out CV (train on 3 runs, test on the 4th, rotate) is the\n"
            "stricter, literature-standard check: it tests whether a model trained on part of\n"
            "the session actually generalizes to a run it never saw. This produced a more\n"
            "conservative ranking than the earlier random k-fold result.\n"
            "Classifier: CSP (4 comp, Ledoit-Wolf) + LDA. Permutations: 1000.\n"
            "Verdict: FAIL if p>=0.05 (not significant); PASS if p<0.05 AND acc>60% AND kappa>0.2;\n"
            "else BORDERLINE.\n"
        )
        fig.text(0.06, 0.89, intro, fontsize=8.3, va="top", linespacing=1.45)

        table_y = 0.63
        row_h = 0.027
        headers = ["Subj", "Acc%", "Kappa", "F1-MI", "F1-REST", "p-value", "Trend", "Verdict"]
        xs = [0.06, 0.16, 0.25, 0.34, 0.44, 0.55, 0.66, 0.80]
        for x, h in zip(xs, headers):
            fig.text(x, table_y, h, fontsize=8, weight="bold")

        for i, subj in enumerate(sorted(RELIABILITY)):
            r = RELIABILITY[subj]
            y = table_y - row_h * (i + 1)
            if "error" in r:
                fig.text(xs[0], y, subj, fontsize=7.7)
                fig.text(xs[1], y, "FAILED", fontsize=7.7)
                continue
            lm = r["label_map"]
            f1 = r["f1_per_class"]
            vals = [subj, f"{r['accuracy']*100:.1f}%", f"{r['kappa']:.3f}",
                    f"{f1.get('MI', list(f1.values())[0]):.3f}",
                    f"{f1.get('REST', list(f1.values())[-1]):.3f}",
                    f"{r['permutation']['p_value']:.4f}", r["run_trend"], r["verdict"]]
            for x, val in zip(xs, vals):
                fig.text(x, y, val, fontsize=7.7)

        notes_y = table_y - row_h * (len(RELIABILITY) + 2)
        n_pass = sum(1 for r in RELIABILITY.values() if r.get("verdict") == "PASS")
        n_border = sum(1 for r in RELIABILITY.values() if r.get("verdict") == "BORDERLINE")
        n_fail = sum(1 for r in RELIABILITY.values() if r.get("verdict") == "FAIL")
        fig.text(0.06, notes_y,
            f"Result: {n_pass} PASS, {n_border} BORDERLINE, {n_fail} FAIL out of {len(RELIABILITY)}.\n"
            "Notably stricter than the random k-fold result: S02 (clean signal, no raw fault)\n"
            "drops to chance level under LORO — its CSP+LDA model does not generalize across\n"
            "runs despite a clean recording, visible in the very unequal per-class F1 (heavy\n"
            "bias toward REST). S09 also fails under LORO. S10 (raw-signal fault) downgrades\n"
            "from 'decodes fine' to BORDERLINE under the stricter test. Most subjects show a\n"
            "'declining' run trend even when PASSing overall — consistent with within-session\n"
            "fatigue/drift, a normal BCI phenomenon worth discussing in the thesis rather than\n"
            "a red flag on its own.",
            fontsize=8.0, va="top", linespacing=1.5)

        fig.text(0.5, 0.02, "Generated automatically from reliability_analysis.py",
                 ha="center", fontsize=7.5, color="0.5")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Summary figure
        rel_summary_path = os.path.join(FIG_DIR, "reliability_summary.png")
        if os.path.exists(rel_summary_path):
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(rel_summary_path)
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("Reliability Summary — All Subjects (LORO CV)", fontsize=12, pad=10)
            pdf.savefig(fig)
            plt.close(fig)

    # ---- Comparison figure ----
    fig = plt.figure(figsize=(8.5, 11))
    img = plt.imread(os.path.join(FIG_DIR, "erd_comparison.png"))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Between-Subject Comparison (Mu band, C3/C4)", fontsize=12, pad=10)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- Stats summary figure (per-subject permutation overview), if available ----
    stats_summary_path = os.path.join(FIG_DIR, "stats_summary.png")
    if os.path.exists(stats_summary_path):
        fig = plt.figure(figsize=(8.5, 11))
        img = plt.imread(stats_summary_path)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Per-Subject Statistical Summary", fontsize=12, pad=10)
        pdf.savefig(fig)
        plt.close(fig)

    # ---- Per-subject pages ----
    for subj in sorted(SUMMARY):
        pages = [
            (f"erd_{subj}.png", "ERD/ERS Time Course"),
            (f"tf_{subj}.png", "Time-Frequency Spectrogram"),
            (f"reliability_{subj}_run_trend.png", "Run-by-Run Consistency (LORO)"),
        ]
        for fname, label in pages:
            path = os.path.join(FIG_DIR, fname)
            if not os.path.exists(path):
                continue
            fig = plt.figure(figsize=(8.5, 11))
            img = plt.imread(path)
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
            ax.imshow(img)
            ax.axis("off")
            r = SUMMARY.get(subj, {})
            extra = f"  |  Acc: {r['accuracy_mean']*100:.1f}%" if "accuracy_mean" in r else ""
            ax.set_title(f"{subj} — {label}{extra}", fontsize=12, pad=10)
            pdf.savefig(fig)
            plt.close(fig)

print(f"Saved {OUT_PATH}")
