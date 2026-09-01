"""
=============================================================================
  Thesis Figures — print-ready renders from saved analysis results
=============================================================================
  The analysis scripts write dark-themed figures for on-screen review. A
  printed thesis needs the opposite: white background, dark ink, serif
  labels to match the body text.

  Reads the JSON summaries (no recomputation) and writes light-theme
  figures to eeg_data/figures/thesis/.

  Usage:
    python render_thesis_figures.py
=============================================================================
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE      = os.path.dirname(os.path.abspath(__file__))
EXPORTS   = os.path.join(HERE, "eeg_data", "exports")
OUT_DIR   = os.path.join(HERE, "eeg_data", "figures", "thesis")

RELI_JSON = os.path.join(EXPORTS, "reliability_summary.json")
CLF_JSON  = os.path.join(EXPORTS, "classification_summary.json")

# Print palette: colour-blind safe and legible in greyscale.
C_PASS, C_BORD, C_FAIL = "#1B7F4F", "#C88A00", "#B03030"
C_MI, C_REST, C_REF     = "#2A6FB5", "#7A9FC4", "#555555"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "axes.titlecolor": "#111111", "xtick.color": "#333333",
    "ytick.color": "#333333", "text.color": "#111111",
    "grid.color": "#CCCCCC", "grid.linewidth": 0.6,
    "font.family": "serif", "font.size": 10,
    "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#999999", "savefig.dpi": 300,
})

VERDICT_COL = {"PASS": C_PASS, "BORDERLINE": C_BORD, "FAIL": C_FAIL}


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


def fig_loro_summary(reli):
    subs = sorted(reli)
    acc  = [reli[s]["accuracy"] * 100 for s in subs]
    kap  = [reli[s]["kappa"] for s in subs]
    cols = [VERDICT_COL[reli[s]["verdict"]] for s in subs]
    x    = np.arange(len(subs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.bar(x, acc, color=cols, edgecolor="#222222", linewidth=0.5)
    ax.axhline(50, color=C_FAIL, lw=1.2, ls="--", label="Επίπεδο τύχης (50%)")
    ax.axhline(60, color=C_BORD, lw=1.2, ls=":", label="Όριο PASS (60%)")
    for xi, a in zip(x, acc):
        ax.text(xi, a + 0.8, f"{a:.0f}", ha="center", fontsize=8)
    ax.set_ylim(30, 88)
    ax.set_ylabel("Ακρίβεια (%)")
    ax.set_title("Ακρίβεια LORO ανά συμμετέχοντα")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=VERDICT_COL[v], edgecolor="#222222", label=v)
               for v in ("PASS", "BORDERLINE", "FAIL")]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", ncol=2)

    ax = axes[1]
    ax.bar(x, kap, color=cols, edgecolor="#222222", linewidth=0.5)
    ax.axhline(0.0, color="#222222", lw=0.9)
    ax.axhline(0.2, color=C_BORD, lw=1.2, ls=":", label="κ=0.2 (πάνω από τύχη)")
    ax.axhline(0.4, color=C_PASS, lw=1.2, ls="--", label="κ=0.4 (μέτρια συμφωνία)")
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Συμφωνία διορθωμένη ως προς την τύχη")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    w   = 0.38
    f1a = [reli[s]["f1_per_class"]["MI"]   for s in subs]
    f1b = [reli[s]["f1_per_class"]["REST"] for s in subs]
    ax.bar(x - w/2, f1a, w, color=C_MI,   edgecolor="#222222", linewidth=0.5, label="MI")
    ax.bar(x + w/2, f1b, w, color=C_REST, edgecolor="#222222", linewidth=0.5, label="REST")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1 ανά κλάση (έλεγχος μεροληψίας)")
    ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=45, fontsize=8)

    fig.tight_layout()
    save(fig, "fig_loro_summary.png")


def fig_kfold_vs_loro(reli, clf):
    """The methodological core: how much each subject loses under LORO."""
    subs = sorted(set(reli) & set(clf))
    kf   = np.array([clf[s]["all_results"]["LDA"]["mean"] * 100 for s in subs])
    lo   = np.array([reli[s]["accuracy"] * 100 for s in subs])
    cols = [VERDICT_COL[reli[s]["verdict"]] for s in subs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    lim = [40, 80]
    ax.plot(lim, lim, ls="--", color=C_REF, lw=1.2, label="Ταύτιση (y = x)")
    ax.axhline(50, color=C_FAIL, lw=1.0, ls=":", alpha=0.8)
    ax.axvline(50, color=C_FAIL, lw=1.0, ls=":", alpha=0.8)
    ax.scatter(kf, lo, c=cols, s=95, edgecolor="#222222", linewidth=0.7, zorder=3)
    for s, a, b in zip(subs, kf, lo):
        ax.annotate(s, (a, b), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Ακρίβεια 10-fold CV (%)")
    ax.set_ylabel("Ακρίβεια LORO CV (%)")
    ax.set_title("Κάθε σημείο κάτω από τη διαγώνιο\nυπερεκτιμήθηκε από το 10-fold")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    order = np.argsort(lo - kf)
    x = np.arange(len(subs))
    for i, idx in enumerate(order):
        ax.plot([i, i], [kf[idx], lo[idx]], color="#999999", lw=1.2, zorder=1)
    ax.scatter(x, kf[order], marker="o", s=60, color=C_REST,
               edgecolor="#222222", linewidth=0.6, zorder=3, label="10-fold CV")
    ax.scatter(x, lo[order], marker="D", s=55,
               c=[cols[i] for i in order], edgecolor="#222222",
               linewidth=0.6, zorder=3, label="LORO CV")
    ax.axhline(50, color=C_FAIL, lw=1.2, ls="--", label="Επίπεδο τύχης")
    ax.set_xticks(x)
    ax.set_xticklabels([subs[i] for i in order], rotation=45, fontsize=8)
    ax.set_ylabel("Ακρίβεια (%)")
    ax.set_title("Μεταβολή ανά συμμετέχοντα\n(ταξινομημένη κατά μέγεθος πτώσης)")
    ax.set_xlim(-0.8, len(subs) - 0.2)
    ax.legend(fontsize=9, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout()
    save(fig, "fig_kfold_vs_loro.png")

    drop = kf - lo
    print(f"    mean drop {drop.mean():+.1f} pp   worst {subs[int(np.argmax(drop))]} "
          f"{drop.max():+.1f} pp")


def fig_run_trend(reli, examples=("S11", "S02")):
    """Contrast a subject that generalises with one that does not."""
    avail = [s for s in examples if s in reli]
    fig, axes = plt.subplots(1, len(avail), figsize=(6.2 * len(avail), 4.4),
                             squeeze=False)

    for ax, s in zip(axes[0], avail):
        r    = reli[s]
        runs = sorted(r["run_accuracy"], key=int)
        vals = [r["run_accuracy"][k] * 100 for k in runs]
        col  = VERDICT_COL[r["verdict"]]

        ax.plot(range(1, len(runs) + 1), vals, "o-", color=col, lw=2, ms=8,
                markeredgecolor="#222222", markeredgewidth=0.6)
        ax.axhline(50, color=C_FAIL, lw=1.2, ls="--", label="Επίπεδο τύχης")

        slope = r["run_trend_slope_per_run"] * 100
        xs    = np.arange(1, len(runs) + 1)
        fit   = np.polyfit(xs, vals, 1)
        ax.plot(xs, np.polyval(fit, xs), ls=":", color=C_REF, lw=1.5,
                label=f"Τάση: {slope:+.1f} pp/run")

        ax.set_xticks(xs)
        ax.set_xlabel("Run")
        ax.set_ylabel("Ακρίβεια (%)")
        ax.set_ylim(30, 95)
        ax.set_title(f"{s} — {r['verdict']}  "
                     f"(συνολικά {r['accuracy']*100:.1f}%, p={r['permutation']['p_value']:.3f})")
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    save(fig, "fig_run_trend.png")


def main():
    reli = json.load(open(RELI_JSON))
    clf  = json.load(open(CLF_JSON))
    print(f"  {len(reli)} subjects in reliability, {len(clf)} in classification\n")

    fig_loro_summary(reli)
    fig_kfold_vs_loro(reli, clf)
    fig_run_trend(reli)

    print(f"\n  Thesis figures → {os.path.relpath(OUT_DIR, HERE)}/")


if __name__ == "__main__":
    main()
