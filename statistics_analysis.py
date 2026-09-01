"""
=============================================================================
  Statistical Validation — inMoov Dataset
=============================================================================
  1. Mann-Whitney U test  — MI vs REST band power per channel
     (+ FDR correction across channels)
  2. Permutation test     — classifier accuracy vs chance distribution
     (1000 permutations, CSP+LDA)

  Output:
    eeg_data/figures/stats_<subject>_wilcoxon.png
    eeg_data/figures/stats_<subject>_permutation.png
    eeg_data/figures/stats_summary.png
    Console: p-values table + effect sizes

  Usage:
    python statistics_analysis.py
=============================================================================
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, glob, argparse, warnings
warnings.filterwarnings("ignore")

from scipy.signal import welch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, permutation_test_score

mne.set_log_level("WARNING")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset 2"
FIG_DIR     = "eeg_data/figures"
os.makedirs(FIG_DIR, exist_ok=True)

EVENT_ID   = {"MI": 1, "REST": 2}
TMIN, TMAX = -2.0, 6.0
BASELINE   = (-1.5, -0.5)
N_PERMS    = 1000

BANDS = {"Mu (8–12 Hz)": (8, 12), "Beta (13–30 Hz)": (13, 30)}

DARK_BG  = "#0A0C14"; COL_PANEL = "#12162A"; COL_GREY = "#788092"
COL_MI   = "#00DC82"; COL_REST  = "#3C8CFF"; COL_SIG  = "#FF5A50"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor": COL_PANEL,
    "axes.edgecolor":  "#283256", "axes.labelcolor": "#C8CDE8",
    "axes.titlecolor": "#F0F5FF", "xtick.color":     COL_GREY,
    "ytick.color":     COL_GREY,  "text.color":      "#F0F5FF",
    "grid.color":      "#283256", "grid.linewidth":  0.5,
    "font.family":     "monospace","figure.dpi":      120,
})


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"    ✓ {name}")


# ─────────────────────────────────────────────
#  LOAD + PREPROCESS
# ─────────────────────────────────────────────

def find_fif(subject_dir):
    fifs = glob.glob(os.path.join(subject_dir, "*_raw.fif"))
    if not fifs:
        raise FileNotFoundError(f"No *_raw.fif in {subject_dir}")
    return sorted(fifs)[-1]


def load_epochs(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw, _ = mne.set_eeg_reference(raw, "average", projection=False,
                                   verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id={"1": 1, "2": 2},
                                             verbose=False)
    epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                        baseline=None, reject=dict(eeg=200e-6),
                        preload=True, verbose=False)
    if len(epochs) == 0:
        epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX,
                            baseline=None, reject=None, preload=True,
                            verbose=False)
    return epochs


# ─────────────────────────────────────────────
#  1. MANN-WHITNEY U TEST
# ─────────────────────────────────────────────

def band_power_per_trial(epochs, flo, fhi):
    """
    Returns array (n_trials, n_channels) of mean band power
    during the 0–4s imagery window.
    """
    sfreq  = epochs.info["sfreq"]
    times  = epochs.times
    t_mask = (times >= 0) & (times <= 4.0)
    data   = epochs.get_data()[:, :, t_mask]   # (trials, ch, time)

    power = np.zeros((data.shape[0], data.shape[1]))
    for trial_idx in range(data.shape[0]):
        for ch_idx in range(data.shape[1]):
            freqs, psd = welch(data[trial_idx, ch_idx], fs=sfreq,
                               nperseg=int(sfreq * 2))
            band_mask  = (freqs >= flo) & (freqs <= fhi)
            power[trial_idx, ch_idx] = psd[band_mask].mean()
    return power


def cohens_d(a, b):
    """Effect size: Cohen's d."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)


def run_wilcoxon(epochs, subject_id):
    """
    Mann-Whitney U test: MI vs REST band power, per channel.
    FDR correction (Benjamini-Hochberg) across channels × bands.
    """
    channels = epochs.ch_names
    n_ch     = len(channels)

    all_pvals = []
    all_stats = []   # (band, ch, U, p_raw, d, mi_mean, rest_mean)

    for band_name, (flo, fhi) in BANDS.items():
        bp_mi   = band_power_per_trial(epochs["MI"],   flo, fhi)
        bp_rest = band_power_per_trial(epochs["REST"], flo, fhi)

        for ch_idx, ch in enumerate(channels):
            U, p = mannwhitneyu(bp_mi[:, ch_idx], bp_rest[:, ch_idx],
                                alternative="less")  # MI < REST (ERD hypothesis)
            d = cohens_d(bp_mi[:, ch_idx], bp_rest[:, ch_idx])
            all_pvals.append(p)
            all_stats.append((band_name, ch, U, p, d,
                              bp_mi[:, ch_idx].mean(),
                              bp_rest[:, ch_idx].mean()))

    # FDR correction
    reject, pvals_fdr, _, _ = multipletests(all_pvals, method="fdr_bh")

    print(f"\n  {subject_id} — Mann-Whitney U (MI < REST band power, FDR corrected)")
    print(f"  {'Band':<18} {'Ch':<6} {'p_raw':>8} {'p_FDR':>8} "
          f"{'sig':>5} {'d':>7} {'MI_pwr':>10} {'REST_pwr':>10}")
    print(f"  {'─'*75}")

    results = []
    for i, (band_name, ch, U, p_raw, d, mi_m, re_m) in enumerate(all_stats):
        sig  = "***" if pvals_fdr[i] < 0.001 else \
               "**"  if pvals_fdr[i] < 0.01  else \
               "*"   if pvals_fdr[i] < 0.05  else "ns"
        print(f"  {band_name:<18} {ch:<6} {p_raw:>8.4f} {pvals_fdr[i]:>8.4f} "
              f"{sig:>5} {d:>7.3f} {mi_m*1e12:>10.3f} {re_m*1e12:>10.3f}")
        results.append({
            "band": band_name, "ch": ch,
            "p_raw": p_raw, "p_fdr": pvals_fdr[i],
            "sig": sig, "d": d,
            "reject": reject[i],
            "mi_mean": mi_m, "rest_mean": re_m,
        })

    _plot_wilcoxon(results, subject_id, channels)
    return results


def _plot_wilcoxon(results, subject_id, channels):
    bands     = list(BANDS.keys())
    n_b, n_ch = len(bands), len(channels)

    fig, axes = plt.subplots(1, n_b, figsize=(6 * n_b, 5))
    if n_b == 1:
        axes = [axes]
    fig.suptitle(f"Mann-Whitney U: MI vs REST Band Power — {subject_id}\n"
                 f"(* p<0.05  ** p<0.01  *** p<0.001, FDR corrected)",
                 fontsize=12, y=1.02)

    band_results = {b: [r for r in results if r["band"] == b] for b in bands}

    for ax, band_name in zip(axes, bands):
        br    = band_results[band_name]
        chs   = [r["ch"]   for r in br]
        d_vals= [abs(r["d"]) for r in br]
        sigs  = [r["sig"]  for r in br]
        cols  = [COL_SIG if r["reject"] else COL_GREY for r in br]

        bars = ax.barh(chs, d_vals, color=cols, alpha=0.85)

        for bar, sig, d in zip(bars, sigs, d_vals):
            if sig != "ns":
                ax.text(d + 0.02, bar.get_y() + bar.get_height()/2,
                        sig, va="center", fontsize=11, color="white")

        ax.axvline(0.2, color="yellow", lw=1.0, ls=":", alpha=0.6,
                   label="Small effect (d=0.2)")
        ax.axvline(0.5, color="orange", lw=1.0, ls=":", alpha=0.6,
                   label="Medium effect (d=0.5)")
        ax.axvline(0.8, color=COL_SIG,  lw=1.0, ls=":", alpha=0.6,
                   label="Large effect (d=0.8)")
        ax.set_title(band_name, fontsize=11)
        ax.set_xlabel("|Cohen's d|  (effect size)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, axis="x")

    fig.tight_layout()
    savefig(fig, f"stats_{subject_id}_wilcoxon.png")


# ─────────────────────────────────────────────
#  2. PERMUTATION TEST
# ─────────────────────────────────────────────

def run_permutation_test(epochs, subject_id, n_perms=N_PERMS):
    """
    Permutation test on CSP+LDA classifier.
    Shuffles labels n_perms times and compares real accuracy
    against the null distribution.
    """
    print(f"\n  {subject_id} — Permutation test (n={n_perms}, CSP+LDA)...")

    t_mask = (epochs.times >= 0) & (epochs.times <= 4.0)
    ep_f   = epochs.copy().filter(8, 30, method="fir", verbose=False)
    X      = ep_f.get_data()[:, :, t_mask]
    y      = (epochs.events[:, 2] == 1).astype(int)

    pipe   = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])
    cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    score, perm_scores, p_value = permutation_test_score(
        pipe, X, y, cv=cv,
        n_permutations=n_perms,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    print(f"    Real accuracy:  {score*100:.1f}%")
    print(f"    Chance (mean):  {perm_scores.mean()*100:.1f}%")
    print(f"    p-value:        {p_value:.4f}  "
          f"{'✓ significant' if p_value < 0.05 else '✗ not significant'}")

    _plot_permutation(score, perm_scores, p_value, subject_id)
    return score, perm_scores, p_value


def _plot_permutation(score, perm_scores, p_value, subject_id):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Permutation Test — {subject_id}", fontsize=13)

    ax.hist(perm_scores * 100, bins=40, color=COL_REST, alpha=0.75,
            label=f"Null distribution (n={len(perm_scores)})")
    ax.axvline(score * 100, color=COL_MI, lw=2.5, ls="-",
               label=f"Real accuracy: {score*100:.1f}%")
    ax.axvline(50, color="red", lw=1.2, ls="--", alpha=0.6,
               label="Chance (50%)")

    sig_str = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
    color   = COL_MI if p_value < 0.05 else COL_SIG
    ax.text(0.97, 0.92, sig_str,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COL_PANEL,
                      edgecolor=color, alpha=0.9))

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    savefig(fig, f"stats_{subject_id}_permutation.png")


# ─────────────────────────────────────────────
#  SUMMARY FIGURE
# ─────────────────────────────────────────────

def plot_stats_summary(summary):
    """
    summary: {subj: {"score": float, "p_perm": float,
                      "sig_channels_mu": int, "sig_channels_beta": int}}
    """
    subjects = list(summary.keys())
    n        = len(subjects)
    x        = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Statistical Summary — All Subjects", fontsize=13)

    # Classifier accuracy + p-value significance
    ax = axes[0]
    accs = [summary[s]["score"] * 100 for s in subjects]
    cols = [COL_MI if summary[s]["p_perm"] < 0.05 else COL_GREY
            for s in subjects]
    ax.bar(x, accs, color=cols, alpha=0.85)
    ax.axhline(50, color="red", lw=1.2, ls="--", alpha=0.7, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Classifier Accuracy\n(green = p<0.05)")
    ax.set_ylim(30, 100); ax.grid(True, axis="y"); ax.legend(fontsize=8)
    for i, (acc, s) in enumerate(zip(accs, subjects)):
        p = summary[s]["p_perm"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(i, acc + 1.5, f"{acc:.1f}%\n{sig}", ha="center",
                fontsize=8, color="white")

    # Significant channels — mu band
    ax = axes[1]
    sig_mu   = [summary[s]["sig_channels_mu"]   for s in subjects]
    sig_beta = [summary[s]["sig_channels_beta"]  for s in subjects]
    w = 0.35
    ax.bar(x - w/2, sig_mu,   w, color="#FFD700", alpha=0.85, label="Mu (8–12 Hz)")
    ax.bar(x + w/2, sig_beta, w, color="#00D2C8", alpha=0.85, label="Beta (13–30 Hz)")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("# Significant channels (FDR p<0.05)")
    ax.set_title("Significant ERD Channels\n(Mann-Whitney, FDR corrected)")
    ax.set_ylim(0, 9); ax.grid(True, axis="y"); ax.legend(fontsize=8)

    # Permutation p-values
    ax = axes[2]
    p_vals = [summary[s]["p_perm"] for s in subjects]
    cols2  = [COL_MI if p < 0.05 else COL_GREY for p in p_vals]
    ax.bar(x, [-np.log10(p + 1e-4) for p in p_vals], color=cols2, alpha=0.85)
    ax.axhline(-np.log10(0.05), color="red", lw=1.2, ls="--",
               label="p=0.05 threshold")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("-log10(p)"); ax.set_title("Permutation Test p-value\n(higher = more significant)")
    ax.grid(True, axis="y"); ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "stats_summary.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--perms",   default=N_PERMS, type=int)
    args = parser.parse_args()

    subject_dirs = sorted(glob.glob(os.path.join(args.dataset, "S*")))
    if not subject_dirs:
        print(f"✗ No subjects found in {args.dataset}")
        return

    print(f"\n{'='*65}")
    print(f"  Statistical Analysis  |  {len(subject_dirs)} subject(s)")
    print(f"  Permutations: {args.perms}")
    print(f"{'='*65}")

    summary = {}

    for subj_dir in subject_dirs:
        subj = os.path.basename(subj_dir)
        print(f"\n─── {subj} ──────────────────────────────────────")
        try:
            fif    = find_fif(subj_dir)
            epochs = load_epochs(fif)
            print(f"  Trials: MI={len(epochs['MI'])}, REST={len(epochs['REST'])}")

            # 1. Mann-Whitney
            wilcox_results = run_wilcoxon(epochs, subj)

            sig_mu   = sum(1 for r in wilcox_results
                          if r["band"] == "Mu (8–12 Hz)"    and r["reject"])
            sig_beta = sum(1 for r in wilcox_results
                          if r["band"] == "Beta (13–30 Hz)" and r["reject"])

            # 2. Permutation test
            score, perm_scores, p_perm = run_permutation_test(
                epochs, subj, n_perms=args.perms)

            summary[subj] = {
                "score":            score,
                "p_perm":           p_perm,
                "sig_channels_mu":  sig_mu,
                "sig_channels_beta":sig_beta,
            }

        except Exception as e:
            import traceback
            print(f"  ✗ {e}"); traceback.print_exc()

    if len(summary) > 1:
        print(f"\n─── Summary figure ────────────────────────────────")
        plot_stats_summary(summary)

    # Final table
    print(f"\n{'='*65}")
    print(f"  STATISTICAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Subj':<6} {'Acc%':>6} {'p_perm':>8} {'Sig?':>6} "
          f"{'SigCh_Mu':>10} {'SigCh_Beta':>11}")
    print(f"  {'─'*55}")
    for subj, r in summary.items():
        sig = "✓" if r["p_perm"] < 0.05 else "✗"
        print(f"  {subj:<6} {r['score']*100:>5.1f}% {r['p_perm']:>8.4f} "
              f"{sig:>6} {r['sig_channels_mu']:>10} {r['sig_channels_beta']:>11}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
