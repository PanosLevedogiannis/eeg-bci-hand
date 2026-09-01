"""
=============================================================================
  Replay Demo GUI — recorded EEG → classifier → InMoov hand, on screen
=============================================================================
  Presentation version of replay_demo.py. Plays the EEG of one trial at a
  time, reveals what the classifier decided, and drives the real hand over
  serial — so an audience sees the signal and the movement together.

  The model is trained on runs 1-3 and only run 4 is replayed, so every
  trial shown is data the classifier has never seen.

  Usage:
    python replay_gui.py --subject S11
    python replay_gui.py --subject S11 --trials 12       # short demo
    python replay_gui.py --subject S11 --no-arduino      # screen only
    python replay_gui.py --subject S11 --record demo.mp4 # offline video
    python replay_gui.py --subject S02                   # a failing subject

  Keys:  space = pause/resume     q = quit
=============================================================================
"""

import argparse
import os
import queue
import sys
import threading

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erd_analysis import find_fif, load_and_preprocess
from classify import prepare_data, N_CSP_COMPONENTS
from reliability_analysis import get_run_ids

DATASET_DIR  = "/Users/panoslevedogiannis/Downloads/inMoov_Dataset__LAST"
ARDUINO_PORT = "/dev/cu.usbmodem1101"
ARDUINO_BAUD = 9600

FPS          = 25
SCROLL_S     = 2.4      # seconds spent drawing one trial's EEG
REVEAL_S     = 1.6      # seconds the verdict stays up before the next trial

BG, PANEL    = "#0f0f1a", "#16162a"
C_MI, C_REST = "#00c896", "#3c8cff"
C_OK, C_BAD  = "#00c896", "#e84040"
C_IDLE, C_DIM = "#555577", "#8888aa"
C_C3, C_C4   = "#38bdf8", "#fb923c"


# ─────────────────────────────────────────────
#  ARDUINO (background thread — serial must not stall the animation)
# ─────────────────────────────────────────────

class HandDriver:
    def __init__(self, port, baud=ARDUINO_BAUD, enabled=True):
        self.q = queue.Queue()
        self.ser = None
        if not enabled:
            return
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            import time as _t
            _t.sleep(2)                 # board resets when the port opens
            self.ser.reset_input_buffer()
            print(f"  Arduino connected on {port}")
        except Exception as e:
            print(f"  Arduino unavailable ({e}) — screen-only demo")
            self.ser = None
            return
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            cmd = self.q.get()
            if cmd is None:
                break
            try:
                self.ser.write(cmd)
                self.ser.readline()
            except Exception:
                pass

    def send(self, label):
        if self.ser is not None:
            self.q.put(b"min\n" if label == "MI" else b"mid\n")

    def close(self):
        if self.ser is not None:
            self.send("REST")
            self.q.put(None)
            try:
                self.ser.close()
            except Exception:
                pass


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────

def load_trials(subject, dataset, test_run):
    epochs, _ = load_and_preprocess(find_fif(os.path.join(dataset, subject)), subject)
    X, y, label_map = prepare_data(epochs)          # 8-30 Hz, cropped
    runs = get_run_ids(epochs)

    train, test = runs != test_run, runs == test_run
    if test.sum() == 0:
        raise SystemExit(f"Run {test_run} has no surviving epochs for {subject}")

    clf = Pipeline([
        ("csp", CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    clf.fit(X[train], y[train])

    # Wider, less aggressively filtered window purely for display.
    disp = epochs.get_data(copy=True)[test] * 1e6   # → µV
    ch   = epochs.ch_names
    idx  = [ch.index(c) for c in ("C3", "C4") if c in ch]

    preds = clf.predict(X[test])
    probs = (clf.predict_proba(X[test])
             if hasattr(clf, "predict_proba") else None)

    return dict(disp=disp[:, idx, :], times=epochs.times, y=y[test],
                preds=preds, probs=probs, label_map=label_map,
                n_train=int(train.sum()), ch=[ch[i] for i in idx])


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

def build_gui(subject, n_trials, n_train, test_run):
    fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
    fig.canvas.manager.set_window_title(f"EEG → InMoov — replay {subject}")

    gs = GridSpec(3, 2, figure=fig, height_ratios=[2.4, 1.7, 0.5],
                  hspace=0.62, wspace=0.16,
                  left=0.07, right=0.96, top=0.83, bottom=0.06)

    ax_eeg = fig.add_subplot(gs[0, :])
    ax_eeg.set_facecolor(PANEL)
    ax_eeg.set_ylabel("µV", color=C_DIM, fontsize=9)
    ax_eeg.set_xlabel("χρόνος από το cue (s)", color=C_DIM, fontsize=8)
    ax_eeg.tick_params(colors=C_DIM, labelsize=8)
    for sp in ax_eeg.spines.values():
        sp.set_color("#333355")
    l_c3, = ax_eeg.plot([], [], color=C_C3, lw=1.0, label="C3")
    l_c4, = ax_eeg.plot([], [], color=C_C4, lw=1.0, label="C4")
    cue_line = ax_eeg.axvline(0, color="#dddd55", lw=1.2, ls="--", alpha=0.8)
    ax_eeg.legend(loc="upper right", fontsize=8, framealpha=0.25,
                  labelcolor="white", facecolor=PANEL, edgecolor="#333355")

    # what the subject was told to do
    ax_true = fig.add_subplot(gs[1, 0]); ax_true.axis("off")
    ax_true.set_title("Οδηγία στον συμμετέχοντα", color=C_DIM, fontsize=11, pad=10)
    true_bg = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.92,
                                      boxstyle="round,pad=0.02",
                                      facecolor=C_IDLE, edgecolor="none",
                                      transform=ax_true.transAxes)
    ax_true.add_patch(true_bg)
    true_txt = ax_true.text(0.5, 0.48, "—", ha="center", va="center",
                            fontsize=30, fontweight="bold", color="white",
                            transform=ax_true.transAxes)

    # what the classifier said + hand
    ax_pred = fig.add_subplot(gs[1, 1]); ax_pred.axis("off")
    ax_pred.set_title("Απόφαση ταξινομητή → χέρι", color=C_DIM, fontsize=11, pad=10)
    pred_bg = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.92,
                                      boxstyle="round,pad=0.02",
                                      facecolor=C_IDLE, edgecolor="none",
                                      transform=ax_pred.transAxes)
    ax_pred.add_patch(pred_bg)
    pred_txt = ax_pred.text(0.5, 0.62, "—", ha="center", va="center",
                            fontsize=28, fontweight="bold", color="white",
                            transform=ax_pred.transAxes)
    hand_txt = ax_pred.text(0.5, 0.24, "", ha="center", va="center",
                            fontsize=14, color="white",
                            transform=ax_pred.transAxes)

    ax_bot = fig.add_subplot(gs[2, :]); ax_bot.axis("off"); ax_bot.set_facecolor(BG)
    hist_txt  = ax_bot.text(0.5, 0.80, "", ha="center", va="center",
                            fontsize=15, color=C_DIM, family="monospace",
                            transform=ax_bot.transAxes)
    stats_txt = ax_bot.text(0.5, 0.15, "", ha="center", va="center",
                            fontsize=12, color="white",
                            transform=ax_bot.transAxes)

    fig.text(0.5, 0.95, f"Replay {subject} — run {test_run}, δεδομένα εκτός εκπαίδευσης",
             ha="center", fontsize=14, fontweight="bold", color="white")
    fig.text(0.5, 0.905,
             f"Το μοντέλο εκπαιδεύτηκε σε {n_train} trials από τα runs 1-3 "
             f"και δοκιμάζεται σε {n_trials} trials που δεν έχει ξαναδεί",
             ha="center", fontsize=9.5, color=C_DIM)

    return fig, dict(ax_eeg=ax_eeg, l_c3=l_c3, l_c4=l_c4, cue_line=cue_line,
                     true_bg=true_bg, true_txt=true_txt,
                     pred_bg=pred_bg, pred_txt=pred_txt, hand_txt=hand_txt,
                     hist_txt=hist_txt, stats_txt=stats_txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S11")
    ap.add_argument("--dataset", default=DATASET_DIR)
    ap.add_argument("--port", default=ARDUINO_PORT)
    ap.add_argument("--no-arduino", action="store_true")
    ap.add_argument("--test-run", type=int, default=4)
    ap.add_argument("--trials", type=int, default=None,
                    help="replay only the first N trials")
    ap.add_argument("--record", metavar="FILE.mp4", default=None,
                    help="render to a video file instead of showing a window")
    args = ap.parse_args()

    matplotlib.use("Agg" if args.record else "MacOSX")

    d = load_trials(args.subject, args.dataset, args.test_run)
    n = len(d["y"]) if args.trials is None else min(args.trials, len(d["y"]))
    lm = d["label_map"]

    scroll_f, reveal_f = int(SCROLL_S * FPS), int(REVEAL_S * FPS)
    per_trial = scroll_f + reveal_f

    fig, el = build_gui(args.subject, n, d["n_train"], args.test_run)
    hand = HandDriver(args.port, enabled=not args.no_arduino and not args.record)

    t = d["times"]
    el["ax_eeg"].set_xlim(t[0], t[-1])
    lim = float(np.percentile(np.abs(d["disp"][:n]), 99.5)) * 1.15
    el["ax_eeg"].set_ylim(-lim, lim)

    state = dict(correct=0, done=0, hist=[], sent=-1, paused=False)

    def on_key(ev):
        if ev.key == " ":
            state["paused"] = not state["paused"]
        elif ev.key == "q":
            plt.close(fig)
    if not args.record:
        fig.canvas.mpl_connect("key_press_event", on_key)

    def animate(f):
        if state["paused"]:
            return []
        trial, phase = divmod(f, per_trial)
        if trial >= n:
            return []

        true_name = lm[int(d["y"][trial])]
        pred_name = lm[int(d["preds"][trial])]
        ok = pred_name == true_name

        if phase < scroll_f:                      # drawing the EEG
            k = max(2, int(len(t) * (phase + 1) / scroll_f))
            el["l_c3"].set_data(t[:k], d["disp"][trial, 0, :k])
            if d["disp"].shape[1] > 1:
                el["l_c4"].set_data(t[:k], d["disp"][trial, 1, :k])

            el["true_bg"].set_facecolor(C_MI if true_name == "MI" else C_REST)
            el["true_txt"].set_text("ΦΑΝΤΑΣΙΑ ΚΙΝΗΣΗΣ" if true_name == "MI" else "ΗΡΕΜΙΑ")
            el["true_txt"].set_fontsize(19 if true_name == "MI" else 26)
            el["pred_bg"].set_facecolor(C_IDLE)
            el["pred_txt"].set_text("...")
            el["hand_txt"].set_text("ανάλυση σήματος")

        else:                                      # verdict + hand
            if state["sent"] != trial:
                hand.send(pred_name)
                state["sent"] = trial
                state["correct"] += int(ok)
                state["done"] += 1
                state["hist"].append("O" if ok else "X")

            el["pred_bg"].set_facecolor(C_OK if ok else C_BAD)
            el["pred_txt"].set_text(
                "ΦΑΝΤΑΣΙΑ ΚΙΝΗΣΗΣ" if pred_name == "MI" else "ΗΡΕΜΙΑ")
            el["pred_txt"].set_fontsize(17 if pred_name == "MI" else 24)
            el["hand_txt"].set_text(
                ("ΧΕΡΙ: ΚΛΕΙΣΤΟ" if pred_name == "MI" else "ΧΕΡΙ: ΑΝΟΙΧΤΟ")
                + ("   ✓ σωστό" if ok else "   ✗ λάθος"))

        done = state["done"]
        el["hist_txt"].set_text("".join(state["hist"][-40:]))
        el["stats_txt"].set_text(
            f"trial {min(trial + 1, n)}/{n}     σωστά {state['correct']}/{done}"
            f"     ακρίβεια {state['correct'] / done * 100:.1f}%" if done else
            f"trial {trial + 1}/{n}")

        return []

    frames = per_trial * n + FPS
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000 // FPS, blit=False,
                                   repeat=False)

    if args.record:
        out = args.record
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=FPS, bitrate=2400)
        else:
            # ffmpeg is not installed; Pillow ships with matplotlib and can
            # write GIF, so the demo still produces a file.
            out = os.path.splitext(out)[0] + ".gif"
            writer = animation.PillowWriter(fps=min(FPS, 15))
            print("  ffmpeg not found — writing GIF instead "
                  "(brew install ffmpeg for mp4)")
        print(f"  rendering {n} trials → {out} ...")
        anim.save(out, writer=writer)
        print(f"  saved {out}")
    else:
        print(f"\n  Replaying {n} unseen trials — space = pause, q = quit\n")
        plt.show()

    hand.close()
    if state["done"]:
        print(f"\n  Τελικό: {state['correct']}/{state['done']} = "
              f"{state['correct'] / state['done'] * 100:.1f}%")


if __name__ == "__main__":
    main()
