"""
=============================================================================
  EEG Real-Time BCI — Graphical Monitor
=============================================================================
  Live EEG signal + classification result + hand state in one window.

  Usage:
    python realtime_gui.py --simulate   # synthetic EEG, real Arduino
    python realtime_gui.py              # real Cyton + Arduino
=============================================================================
"""

import argparse
import threading
import collections
import time
import sys
import os
import glob

import numpy as np

import matplotlib
matplotlib.use("MacOSX")          # macOS native — change to TkAgg if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import joblib
import serial
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import DATA_DIR

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODEL_DIR    = os.path.join(DATA_DIR, "models")
ARDUINO_PORT = "/dev/cu.usbmodem1101"
CYTON_PORT   = "/dev/cu.usbserial-YYYY"
SUBJECT_ID   = "P001"

SFREQ        = 250
CLF_LFREQ    = 8.0
CLF_HFREQ    = 30.0
FILTER_ORDER = 4

WINDOW_MS    = 1000
STEP_MS      = 125
VOTE_SIZE    = 7
MIN_VOTE_PCT = 0.60
HOLD_S       = 2.0
SIM_INTERVAL = 3.0      # seconds per state in simulate mode

DISPLAY_SECS    = 4     # EEG window to show on screen
WINDOW_SAMPLES  = int(SFREQ * WINDOW_MS / 1000)
STEP_SAMPLES    = int(SFREQ * STEP_MS  / 1000)
DISPLAY_SAMPLES = SFREQ * DISPLAY_SECS

# ─────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────

BG       = "#0f0f1a"
PANEL    = "#16162a"
OPEN_COL = "#00c896"    # green
CLOSE_COL= "#e84040"    # red
IDLE_COL = "#555577"
CH0_COL  = "#38bdf8"    # sky blue  (C3)
CH1_COL  = "#fb923c"    # orange    (C4)

# ─────────────────────────────────────────────
#  SHARED STATE (worker → GUI)
# ─────────────────────────────────────────────

class State:
    def __init__(self):
        self._lock      = threading.Lock()
        self.running    = True
        self.pred       = None       # 0 = OPEN, 1 = CLOSE
        self.confidence = 0.0
        self.hand       = None       # last confirmed command
        self.cue        = ""         # simulate cue text
        self.n_preds    = 0
        self.elapsed    = 0.0
        self._ch0       = collections.deque(maxlen=DISPLAY_SAMPLES)
        self._ch1       = collections.deque(maxlen=DISPLAY_SAMPLES)

    def push_eeg(self, ch0_arr, ch1_arr):
        with self._lock:
            self._ch0.extend(ch0_arr)
            self._ch1.extend(ch1_arr)

    def snapshot(self):
        with self._lock:
            return dict(
                pred       = self.pred,
                confidence = self.confidence,
                hand       = self.hand,
                cue        = self.cue,
                n_preds    = self.n_preds,
                elapsed    = self.elapsed,
                ch0        = np.array(self._ch0, dtype=float),
                ch1        = np.array(self._ch1, dtype=float),
            )

# ─────────────────────────────────────────────
#  SIGNAL PROCESSING
# ─────────────────────────────────────────────

def build_iir():
    nyq = SFREQ / 2.0
    return butter(FILTER_ORDER, [CLF_LFREQ / nyq, CLF_HFREQ / nyq],
                  btype="band", output="sos")

def preprocess_window(eeg, sos):
    filtered = np.array([sosfiltfilt(sos, ch) for ch in eeg])
    filtered -= filtered.mean(axis=0, keepdims=True)
    return filtered[np.newaxis, :, :]

# ─────────────────────────────────────────────
#  ARDUINO
# ─────────────────────────────────────────────

def connect_arduino(port, baud=9600):
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"  ✓ Arduino on {port}  ({line})")
        return ser
    except serial.SerialException as e:
        print(f"  ✗ Arduino not found: {e}")
        print(f"    → Continuing without Arduino")
        return None

def send_command(ser, cmd):
    if ser is None:
        return
    ser.write((b"mid\n" if cmd == "O" else b"min\n"))
    ser.readline()

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────

def load_model(subject_id):
    pattern = os.path.join(MODEL_DIR, f"model_{subject_id}_*.joblib")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model for '{subject_id}' in {MODEL_DIR}")
    model = joblib.load(files[-1])
    print(f"  ✓ Model loaded  ← {os.path.basename(files[-1])}")
    return model

# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────

def worker(state, model, arduino, sos, board, eeg_channels, simulate):
    vote_buffer = collections.deque(maxlen=VOTE_SIZE)
    last_cmd    = None
    last_cmd_t  = 0.0
    start_t     = time.time()

    time.sleep(WINDOW_SAMPLES / SFREQ + 0.3)   # initial buffer fill

    while state.running:
        t0  = time.time()
        raw = board.get_current_board_data(WINDOW_SAMPLES)

        if raw.shape[1] < WINDOW_SAMPLES:
            time.sleep(0.01)
            continue

        # Push latest step to EEG display buffer (raw µV)
        step = raw[eeg_channels[:2], -STEP_SAMPLES:]
        state.push_eeg(step[0], step[1])

        # ── Classification ─────────────────────────
        if simulate:
            elapsed_total = time.time() - start_t
            pred  = int(elapsed_total // SIM_INTERVAL) % 2
            # Time remaining in current cue
            remaining = SIM_INTERVAL - (elapsed_total % SIM_INTERVAL)
            label = "OPEN" if pred == 0 else "CLOSE"
            with state._lock:
                state.cue = f"IMAGINE:  {label}   ({remaining:.1f}s)"
        else:
            eeg_data = raw[eeg_channels, :]
            X    = preprocess_window(eeg_data, sos)
            pred = int(model.predict(X)[0])
            with state._lock:
                state.cue = ""

        vote_buffer.append(pred)

        with state._lock:
            state.n_preds += 1
            state.elapsed  = time.time() - start_t

        # ── Majority vote & command ─────────────────
        if len(vote_buffer) == VOTE_SIZE:
            votes      = list(vote_buffer)
            n_open     = votes.count(0)
            n_close    = votes.count(1)
            winner     = 0 if n_open >= n_close else 1
            confidence = max(n_open, n_close) / VOTE_SIZE
            now        = time.time()

            with state._lock:
                state.pred       = winner
                state.confidence = confidence

            if confidence >= MIN_VOTE_PCT:
                if winner != last_cmd or (now - last_cmd_t) > HOLD_S:
                    cmd = "O" if winner == 0 else "C"
                    send_command(arduino, cmd)
                    last_cmd   = winner
                    last_cmd_t = now
                    with state._lock:
                        state.hand = winner

        sleep_t = max(0.0, STEP_SAMPLES / SFREQ - (time.time() - t0))
        time.sleep(sleep_t)

    # Reset hand on exit
    send_command(arduino, "O")
    time.sleep(0.5)
    if arduino:
        arduino.close()
    board.stop_stream()
    board.release_session()

# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

def build_gui(state_ref, simulate):
    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    fig.canvas.manager.set_window_title("EEG BCI — Real-Time Monitor")

    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[2, 2.5, 1],
                  hspace=0.45, wspace=0.35,
                  left=0.08, right=0.95, top=0.93, bottom=0.08)

    # ── EEG panel (top, full width) ───────────────
    ax_eeg = fig.add_subplot(gs[0, :])
    ax_eeg.set_facecolor(PANEL)
    ax_eeg.set_title("Live EEG  (bandpass 8–30 Hz)", color="white",
                     fontsize=10, pad=6)
    ax_eeg.set_xlim(0, DISPLAY_SAMPLES)
    ax_eeg.set_ylim(-200, 200)
    ax_eeg.set_ylabel("µV", color="#aaaaaa", fontsize=9)
    ax_eeg.set_xlabel(f"← {DISPLAY_SECS}s rolling window", color="#666688", fontsize=8)
    ax_eeg.tick_params(colors="#666688", labelsize=8)
    ax_eeg.set_xticks([])
    for sp in ax_eeg.spines.values():
        sp.set_color("#333355")

    line0, = ax_eeg.plot([], [], color=CH0_COL, lw=0.9, label="Ch 1 (C3)", alpha=0.95)
    line1, = ax_eeg.plot([], [], color=CH1_COL, lw=0.9, label="Ch 2 (C4)", alpha=0.95)
    ax_eeg.legend(loc="upper right", framealpha=0.2, labelcolor="white",
                  fontsize=8, facecolor=PANEL, edgecolor="#333355")
    ax_eeg.axhline(0, color="#333355", lw=0.5)

    # ── Prediction panel (middle left) ────────────
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_pred.set_facecolor(PANEL)
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    ax_pred.axis("off")
    ax_pred.set_title("Classification", color="white", fontsize=10, pad=6)

    pred_bg  = mpatches.FancyBboxPatch((0.05, 0.35), 0.9, 0.5,
                                        boxstyle="round,pad=0.02",
                                        facecolor=IDLE_COL, edgecolor="none",
                                        transform=ax_pred.transAxes, zorder=1)
    ax_pred.add_patch(pred_bg)

    pred_txt = ax_pred.text(0.5, 0.62, "—", ha="center", va="center",
                             fontsize=26, fontweight="bold", color="white",
                             transform=ax_pred.transAxes, zorder=2)

    conf_bg  = mpatches.Rectangle((0.1, 0.12), 0.8, 0.14,
                                   facecolor="#2a2a44", edgecolor="none",
                                   transform=ax_pred.transAxes, zorder=1)
    conf_bar = mpatches.Rectangle((0.1, 0.12), 0.0,  0.14,
                                   facecolor=IDLE_COL, edgecolor="none",
                                   transform=ax_pred.transAxes, zorder=2)
    ax_pred.add_patch(conf_bg)
    ax_pred.add_patch(conf_bar)

    conf_txt = ax_pred.text(0.5, 0.04, "Confidence: —", ha="center", va="center",
                             fontsize=9, color="#888899",
                             transform=ax_pred.transAxes)

    # ── Hand state panel (middle right) ───────────
    ax_hand = fig.add_subplot(gs[1, 1])
    ax_hand.set_facecolor(PANEL)
    ax_hand.set_xlim(0, 1)
    ax_hand.set_ylim(0, 1)
    ax_hand.axis("off")
    ax_hand.set_title("Hand State", color="white", fontsize=10, pad=6)

    hand_bg  = mpatches.FancyBboxPatch((0.05, 0.25), 0.9, 0.65,
                                        boxstyle="round,pad=0.02",
                                        facecolor=IDLE_COL, edgecolor="none",
                                        transform=ax_hand.transAxes, zorder=1)
    ax_hand.add_patch(hand_bg)

    hand_icon = ax_hand.text(0.5, 0.62, "—", ha="center", va="center",
                              fontsize=40, transform=ax_hand.transAxes, zorder=2)
    hand_lbl  = ax_hand.text(0.5, 0.32, "WAITING", ha="center", va="center",
                              fontsize=13, fontweight="bold", color="white",
                              transform=ax_hand.transAxes, zorder=2)

    # ── Cue / stats panel (bottom, full width) ────
    ax_bot = fig.add_subplot(gs[2, :])
    ax_bot.set_facecolor(BG)
    ax_bot.axis("off")

    cue_txt   = ax_bot.text(0.5, 0.65, "", ha="center", va="center",
                             fontsize=14, color="#dddd55", fontweight="bold",
                             transform=ax_bot.transAxes)
    stats_txt = ax_bot.text(0.5, 0.1, "", ha="center", va="center",
                             fontsize=8, color="#555577",
                             transform=ax_bot.transAxes)

    # ── Title ─────────────────────────────────────
    mode_label = " [SIMULATE]" if simulate else ""
    fig.text(0.5, 0.97, f"EEG BCI — Real-Time Monitor{mode_label}",
             ha="center", fontsize=13, fontweight="bold", color="white")

    elements = dict(
        line0=line0, line1=line1,
        pred_bg=pred_bg, pred_txt=pred_txt,
        conf_bar=conf_bar, conf_txt=conf_txt,
        hand_bg=hand_bg, hand_icon=hand_icon, hand_lbl=hand_lbl,
        cue_txt=cue_txt, stats_txt=stats_txt,
        ax_eeg=ax_eeg,
    )
    return fig, elements


def make_animate(state_ref, elements, sos):
    ax_eeg   = elements["ax_eeg"]

    def animate(_frame):
        s = state_ref.snapshot()

        # ── EEG waveform ──────────────────────────
        ch0 = s["ch0"]
        ch1 = s["ch1"]
        n   = len(ch0)
        if n > 1:
            # Bandpass-filter display buffer
            try:
                ch0_f = sosfiltfilt(sos, ch0)
                ch1_f = sosfiltfilt(sos, ch1)
            except Exception:
                ch0_f, ch1_f = ch0, ch1

            # Auto-scale EEG axis to data
            peak = max(np.percentile(np.abs(ch0_f), 99),
                       np.percentile(np.abs(ch1_f), 99), 10)
            ax_eeg.set_ylim(-peak * 1.3, peak * 1.3)

            x = np.arange(n)
            elements["line0"].set_data(x, ch0_f)
            elements["line1"].set_data(x, ch1_f)
            ax_eeg.set_xlim(0, max(n, DISPLAY_SAMPLES))

        # ── Prediction & confidence ───────────────
        pred = s["pred"]
        conf = s["confidence"]

        if pred is None:
            elements["pred_txt"].set_text("—")
            elements["pred_bg"].set_facecolor(IDLE_COL)
            elements["conf_bar"].set_width(0)
            elements["conf_txt"].set_text("Confidence: —")
        else:
            label = "OPEN" if pred == 0 else "CLOSE"
            color = OPEN_COL if pred == 0 else CLOSE_COL
            elements["pred_txt"].set_text(label)
            elements["pred_bg"].set_facecolor(color + "55")
            elements["conf_bar"].set_width(0.8 * conf)
            elements["conf_bar"].set_facecolor(color)
            elements["conf_txt"].set_text(f"Confidence:  {conf*100:.0f}%")

        # ── Hand state ────────────────────────────
        hand = s["hand"]
        if hand is None:
            elements["hand_icon"].set_text("- -")
            elements["hand_lbl"].set_text("WAITING")
            elements["hand_bg"].set_facecolor(IDLE_COL)
        else:
            icon  = "( )" if hand == 0 else "[|]"   # open fingers vs closed fist
            label = "OPEN" if hand == 0 else "CLOSE"
            color = OPEN_COL if hand == 0 else CLOSE_COL
            elements["hand_icon"].set_text(icon)
            elements["hand_lbl"].set_text(label)
            elements["hand_bg"].set_facecolor(color + "55")

        # ── Cue (simulate mode) ───────────────────
        elements["cue_txt"].set_text(s["cue"])

        # ── Stats ─────────────────────────────────
        elements["stats_txt"].set_text(
            f"Predictions: {s['n_preds']}   |   "
            f"Elapsed: {s['elapsed']:.0f}s   |   "
            f"Window: {WINDOW_MS}ms / step {STEP_MS}ms / vote {VOTE_SIZE}"
        )

        return list(elements.values())

    return animate


# ─────────────────────────────────────────────
#  BOARD SETUP
# ─────────────────────────────────────────────

def setup_board(simulate):
    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    if simulate:
        board_id = BoardIds.SYNTHETIC_BOARD.value
        print(f"\n  [SIMULATE] BrainFlow synthetic board — Arduino is REAL\n")
    else:
        board_id = BoardIds.CYTON_BOARD.value
        params.serial_port = CYTON_PORT
        print(f"\n  Connecting to Cyton on {CYTON_PORT}...")

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    label = "synthetic" if simulate else "Cyton"
    print(f"  ✓ Streaming ({label})  —  {len(eeg_channels)} EEG ch @ {SFREQ} Hz")
    return board, eeg_channels


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG BCI GUI")
    parser.add_argument("--simulate", action="store_true",
                        help="Use synthetic board (Arduino still real)")
    parser.add_argument("--subject", type=str, default=SUBJECT_ID)
    args = parser.parse_args()

    if not HAS_BRAINFLOW:
        print("✗ BrainFlow not installed.  pip install brainflow")
        return

    print(f"\n{'='*55}")
    print(f"  EEG BCI GUI  |  Subject: {args.subject}")
    print(f"{'='*55}")

    model   = load_model(args.subject)
    arduino = connect_arduino(ARDUINO_PORT)
    sos     = build_iir()
    board, eeg_channels = setup_board(args.simulate)

    state = State()

    # Start worker thread
    t = threading.Thread(
        target=worker,
        args=(state, model, arduino, sos, board, eeg_channels, args.simulate),
        daemon=True,
    )
    t.start()

    # Build GUI
    fig, elements = build_gui(state, args.simulate)
    animate_fn    = make_animate(state, elements, sos)

    ani = animation.FuncAnimation(
        fig, animate_fn,
        interval=50,        # 20 fps
        blit=False,
        cache_frame_data=False,
    )

    def on_close(_event):
        state.running = False

    fig.canvas.mpl_connect("close_event", on_close)

    print(f"\n  Close the window to stop.\n")
    plt.show()

    state.running = False
    t.join(timeout=3)
    print(f"  ✓ Done\n")


if __name__ == "__main__":
    main()
