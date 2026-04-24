"""
=============================================================================
  EEG Motor Imagery Data Collection Paradigm
  Graz-BCI Protocol — Movement vs Rest (2-Class)
=============================================================================
  Thesis: EEG-Based Control of the InMoov i2 Robotic Hand
  Supervisor: Asst. Prof. Athanasios Koutras
  Hardware: OpenBCI Cyton (8-channel)

  Classification task: Motor Imagery vs Rest
    - Class 1 (MI):   Imagine closing the right hand
    - Class 0 (REST): Relax, no movement imagery

  Protocol changes (v2 — per supervisor feedback):
    1. Task changed to MI vs Rest (from open/close same hand)
    2. Imagery phase shows fixation cross only — no visual hand icons
    3. LSL markers for sample-accurate alignment with OpenBCI recording
    4. Randomized inter-trial interval (1.5–2.5s)
    5. Session split into 4 runs × 20 trials with mandatory breaks
    6. Beep (1kHz, 70ms) before each cue
    7. Baseline recording at session start (2min eyes open + 1min eyes closed)

  Requirements:
      pip install pygame numpy pylsl

  Usage:
      python eeg_mi_paradigm.py

  Controls:
      SPACE   — Start session / Advance / Continue after break
      ESC     — Quit at any time
      P       — Pause / Resume during session
=============================================================================
"""

import pygame
import numpy as np
import json
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    from pylsl import StreamInfo, StreamOutlet
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print("⚠ pylsl not found — LSL markers disabled. Install with: pip install pylsl")

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False
    print("⚠ brainflow not found — EEG recording disabled. Install with: pip install brainflow")

import threading
import csv

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class SessionConfig:
    subject_id:       str   = "S01"
    trials_per_class: int   = 20          # 20 MI + 20 REST per run
    n_runs:           int   = 1           # 1 run × 40 trials = 40 total
    classes:          List[str] = field(default_factory=lambda: ["MI", "REST"])

    # Timing (seconds) — Graz-BCI protocol
    t_rest_min:  float = 1.5   # jittered rest (prevents CNV anticipatory potentials)
    t_rest_max:  float = 2.5
    t_prepare:   float = 2.0   # fixation cross — focus
    t_cue:       float = 2.25  # brief cue, then disappears (Graz standard)
    t_imagery:   float = 4.0   # motor imagery window — fixation cross only
    t_feedback:  float = 1.5   # blink freely

    # Break between runs
    t_break_min: float = 60.0  # seconds (subject presses SPACE when ready)

    # Baseline at session start
    t_baseline_eyes_open:  float = 30.0
    t_baseline_eyes_closed: float = 30.0

    # Beep (played before cue)
    beep_freq:   int   = 1000   # Hz
    beep_dur_ms: int   = 70     # ms

    # Display
    screen_w:    int = 1280
    screen_h:    int = 800
    fps:         int = 60

    output_dir:  str = "eeg_data"

    # BrainFlow / Cyton
    cyton_port:  str = "/dev/cu.usbserial-DQ007OFI"
    simulate:    bool = False   # True = synthetic board (no hardware needed)


CFG = SessionConfig()

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
C = {
    "bg":           (10,  12,  20),
    "panel":        (18,  22,  38),
    "border":       (40,  50,  90),
    "accent_blue":  (60, 140, 255),
    "accent_cyan":  (0,  210, 200),
    "accent_mi":    (0,  220, 130),    # green — MI
    "accent_rest":  (100, 120, 200),   # blue-grey — REST
    "white":        (240, 245, 255),
    "grey":         (120, 130, 160),
    "dim":          (50,  60,  90),
    "black":        (0,   0,   0),
    "warning":      (255, 180,  40),
}

# LSL marker codes
LSL_MARKERS = {
    "baseline_eyes_open_start":  10,
    "baseline_eyes_open_end":    11,
    "baseline_eyes_closed_start":12,
    "baseline_eyes_closed_end":  13,
    "trial_start":               20,
    "cue_MI":                    1,
    "cue_REST":                  2,
    "imagery_start":             30,
    "imagery_end":               31,
    "break_start":               40,
    "break_end":                 41,
    "session_end":               99,
}

# ─────────────────────────────────────────────
#  LSL OUTLET
# ─────────────────────────────────────────────

def create_lsl_outlet():
    if not HAS_LSL:
        return None
    info = StreamInfo(
        name="EEG_BCI_Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="int32",
        source_id="eeg_mi_paradigm",
    )
    outlet = StreamOutlet(info)
    print("✓ LSL marker stream created: EEG_BCI_Markers")
    return outlet

_active_recorder = None   # set when EEGRecorder starts

def push_marker(outlet, code: int):
    if outlet is not None:
        outlet.push_sample([code])
    if _active_recorder is not None:
        _active_recorder.log_marker(code)

# ─────────────────────────────────────────────
#  EEG RECORDER (BrainFlow)
# ─────────────────────────────────────────────

class EEGRecorder:
    """
    Connects to OpenBCI Cyton via BrainFlow and records all EEG samples
    in a background thread. Saves an OpenBCI-compatible CSV on stop().
    """
    SFREQ = 250

    def __init__(self, port: str, simulate: bool = False):
        if not HAS_BRAINFLOW:
            self.board = None
            print("⚠ BrainFlow not available — EEG not recorded")
            return

        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()

        if simulate:
            board_id = BoardIds.SYNTHETIC_BOARD.value
            print("  [BrainFlow] Synthetic board (no hardware)")
        else:
            board_id = BoardIds.CYTON_BOARD.value
            params.serial_port = port

        self.board        = BoardShim(board_id, params)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.all_data     = []
        self._running     = False
        self._thread      = None

        self._markers = []   # list of (unix_time, code)

        self.board.prepare_session()
        self.board.start_stream()
        print(f"  ✓ BrainFlow streaming — {len(self.eeg_channels)} EEG ch @ {self.SFREQ} Hz")

    def log_marker(self, code: int):
        self._markers.append((time.time(), code))

    def start(self):
        if self.board is None:
            return
        global _active_recorder
        _active_recorder = self
        self._running = True
        self._thread  = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def _record_loop(self):
        while self._running:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                self.all_data.append(data)
            time.sleep(0.04)   # ~25 Hz poll

    def stop(self, subject_id: str, output_dir: str) -> str | None:
        if self.board is None:
            return None
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Drain remaining buffer
        data = self.board.get_board_data()
        if data.shape[1] > 0:
            self.all_data.append(data)
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception:
            pass

        if not self.all_data:
            print("  ⚠ No EEG data collected")
            return None

        import numpy as np
        all_data = np.concatenate(self.all_data, axis=1)
        return self._save_csv(all_data, subject_id, output_dir)

    def _save_csv(self, data, subject_id: str, output_dir: str) -> str:
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(output_dir, f"EEG-RAW_{subject_id}_{timestamp}.csv")

        n_samples  = data.shape[1]
        timestamps = data[-2] if data.shape[0] > 2 else np.arange(n_samples) / self.SFREQ

        # Build marker column: align each marker to nearest sample by timestamp
        marker_col = np.zeros(n_samples, dtype=int)
        half_sample = 0.5 / self.SFREQ
        for marker_time, code in self._markers:
            diffs = np.abs(timestamps - marker_time)
            idx   = int(np.argmin(diffs))
            if diffs[idx] < half_sample * 20:   # within 20 samples tolerance
                marker_col[idx] = code

        with open(fname, "w", newline="") as f:
            f.write("%OpenBCI Raw EXG Data\n")
            f.write(f"%Number of channels = {len(self.eeg_channels)}\n")
            f.write(f"%Sample Rate = {self.SFREQ} Hz\n")
            f.write(f"%Subject = {subject_id}\n")
            header = (["Sample Index"] +
                      [f"EXG Channel {i}" for i in range(len(self.eeg_channels))] +
                      ["Timestamp", "Marker"])
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(n_samples):
                row = ([i] +
                       [f"{data[ch, i]:.6f}" for ch in self.eeg_channels] +
                       [f"{timestamps[i]:.6f}", marker_col[i]])
                writer.writerow(row)

        print(f"  ✓ EEG saved → {fname}  ({n_samples} samples, {len(self._markers)} markers)")
        self._save_fif(data, timestamps, subject_id, output_dir)
        return fname

    def _save_fif(self, data, timestamps, subject_id: str, output_dir: str):
        try:
            import mne
            import numpy as np
        except ImportError:
            print("  ⚠ mne not installed — skipping FIF save")
            return

        ch_names = ["C3", "C4", "FC3", "FC4", "CP3", "CP4", "Cz", "FCz"]
        n_ch     = len(self.eeg_channels)
        ch_names = ch_names[:n_ch]

        eeg_data = data[self.eeg_channels, :] * 1e-6   # µV → V for MNE

        info = mne.create_info(
            ch_names = ch_names,
            sfreq    = self.SFREQ,
            ch_types = "eeg",
        )
        info.set_montage("standard_1020", on_missing="ignore", verbose=False)

        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        # Add markers as annotations
        if self._markers and len(timestamps) > 0:
            t0 = timestamps[0]
            onsets      = [t - t0 for t, _ in self._markers]
            durations   = [0.0]   * len(self._markers)
            descriptions= [str(c) for _, c in self._markers]
            raw.set_annotations(mne.Annotations(onsets, durations, descriptions))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fif_path  = os.path.join(output_dir, f"EEG-RAW_{subject_id}_{timestamp}_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)
        print(f"  ✓ FIF saved  → {fif_path}")


# ─────────────────────────────────────────────
#  BEEP GENERATOR
# ─────────────────────────────────────────────

def make_beep(freq=1000, duration_ms=70, volume=0.4, sample_rate=44100):
    """Generate a short sine wave beep as a pygame Sound."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    # Stereo
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TrialMarker:
    trial_number:   int
    run_number:     int
    label:          str       # "MI" or "REST"
    onset_time_s:   float     # seconds since session start
    onset_unix:     float
    lsl_code:       int
    phase_onsets:   dict = field(default_factory=dict)

@dataclass
class SessionLog:
    subject_id:     str
    date:           str
    start_time:     str
    task:           str = "MI_vs_REST"
    config:         dict = field(default_factory=dict)
    baseline:       dict = field(default_factory=dict)
    trials:         List[dict] = field(default_factory=list)
    breaks:         List[dict] = field(default_factory=list)
    end_time:       str = ""
    total_duration: float = 0.0
    lsl_available:  bool = False

# ─────────────────────────────────────────────
#  UTILITY — drawing helpers
# ─────────────────────────────────────────────

def draw_text(surf, text, font, color, cx, cy, anchor="center"):
    rendered = font.render(text, True, color)
    r = rendered.get_rect()
    if anchor == "center":  r.center = (cx, cy)
    elif anchor == "left":  r.midleft = (cx, cy)
    elif anchor == "right": r.midright = (cx, cy)
    surf.blit(rendered, r)
    return r

def draw_rounded_rect(surf, color, rect, radius=12, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

def draw_fixation_cross(surf, cx, cy, size=40, thick=6, color=None):
    color = color or C["white"]
    pygame.draw.rect(surf, color, (cx - thick//2, cy - size, thick, size*2))
    pygame.draw.rect(surf, color, (cx - size, cy - thick//2, size*2, thick))

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def alpha_surface(w, h, color, alpha):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color, alpha))
    return s

# ─────────────────────────────────────────────
#  EEG WAVEFORM — decorative animated background
# ─────────────────────────────────────────────

class EEGDecoration:
    def __init__(self, screen_w, screen_h):
        self.w = screen_w
        self.h = screen_h
        self.channels = 4
        self.phase = [random.uniform(0, 6.28) for _ in range(self.channels)]
        self.amp   = [random.uniform(8, 20)   for _ in range(self.channels)]
        self.freq  = [random.uniform(0.8, 2.2) for _ in range(self.channels)]
        self.noise = [[random.gauss(0, 3) for _ in range(screen_w//4)]
                      for _ in range(self.channels)]

    def update(self, dt):
        for i in range(self.channels):
            self.phase[i] += dt * self.freq[i]
            self.noise[i].pop(0)
            self.noise[i].append(random.gauss(0, 3))

    def draw(self, surf, alpha=35):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        for ch in range(self.channels):
            y_base = int(self.h * 0.15 + ch * self.h * 0.06)
            pts = []
            step = 4
            for xi, x in enumerate(range(0, self.w, step)):
                t = x / self.w
                y_wave = self.amp[ch] * np.sin(self.phase[ch] + t * 12)
                ni = min(xi, len(self.noise[ch]) - 1)
                y = y_base + int(y_wave + self.noise[ch][ni])
                pts.append((x, y))
            if len(pts) > 2:
                pygame.draw.lines(overlay, (*C["accent_blue"], alpha), False, pts, 1)
        surf.blit(overlay, (0, 0))

# ─────────────────────────────────────────────
#  PROGRESS BAR
# ─────────────────────────────────────────────

class ProgressBar:
    def __init__(self, x, y, w, h, color):
        self.rect  = pygame.Rect(x, y, w, h)
        self.color = color
        self.value = 0.0

    def draw(self, surf):
        draw_rounded_rect(surf, C["panel"], self.rect, radius=6,
                          border=1, border_color=C["border"])
        if self.value > 0:
            fill = pygame.Rect(self.rect.x, self.rect.y,
                               int(self.rect.w * self.value), self.rect.h)
            draw_rounded_rect(surf, self.color, fill, radius=6)

# ─────────────────────────────────────────────
#  SCREENS — base class
# ─────────────────────────────────────────────

class Screen:
    def __init__(self, app):
        self.app  = app
        self.surf = app.screen
        self.W    = app.W
        self.H    = app.H

    def handle_event(self, event): pass
    def update(self, dt):          pass
    def draw(self):                pass

# ─────────────────────────────────────────────
#  INTRO SCREEN
# ─────────────────────────────────────────────

class IntroScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.page    = 0
        self.fade_in = 0.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if self.page < 2:
                    self.page += 1
                    self.fade_in = 0.0
                else:
                    self.app.goto("subject")
            elif event.key == pygame.K_ESCAPE:
                self.app.running = False

    def update(self, dt):
        self.fade_in = min(1.0, self.fade_in + dt * 1.8)

    def draw(self):
        s = self.surf
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=25)
        W, H = self.W, self.H

        if self.page == 0:
            draw_text(s, "EEG MOTOR IMAGERY", self.app.font_title, C["accent_blue"], W//2, H//2 - 120)
            draw_text(s, "DATA COLLECTION PARADIGM", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 68)
            pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 40), (W//2 + 260, H//2 - 40), 1)
            draw_text(s, "Graz-BCI Protocol  ·  Motor Imagery vs Rest", self.app.font_body, C["grey"], W//2, H//2 - 12)
            draw_text(s, "InMoov i2 Robotic Hand Control", self.app.font_body, C["grey"], W//2, H//2 + 18)

            # Task illustration
            box_mi   = pygame.Rect(W//2 - 200, H//2 + 60, 160, 80)
            box_rest = pygame.Rect(W//2 + 40,  H//2 + 60, 160, 80)
            draw_rounded_rect(s, C["panel"], box_mi,   radius=10, border=1, border_color=C["accent_mi"])
            draw_rounded_rect(s, C["panel"], box_rest, radius=10, border=1, border_color=C["accent_rest"])
            draw_text(s, "MOTOR IMAGERY", self.app.font_small_bold, C["accent_mi"],   W//2 - 120, H//2 + 88)
            draw_text(s, "Imagine closing fist", self.app.font_small, C["grey"],      W//2 - 120, H//2 + 112)
            draw_text(s, "REST",         self.app.font_small_bold, C["accent_rest"],  W//2 + 120, H//2 + 88)
            draw_text(s, "Relax, clear mind", self.app.font_small, C["grey"],         W//2 + 120, H//2 + 112)

            draw_text(s, "SPACE to continue", self.app.font_small, C["dim"], W//2, H - 50)

        elif self.page == 1:
            draw_text(s, "SESSION PROTOCOL", self.app.font_heading, C["accent_cyan"], W//2, 80)
            pygame.draw.line(s, C["border"], (W//2 - 300, 110), (W//2 + 300, 110), 1)

            phases = [
                ("BASELINE",      "3 min",                   "Eyes open (2min) + eyes closed (1min)",    C["accent_blue"]),
                ("REST",          f"{CFG.t_rest_min:.1f}–{CFG.t_rest_max:.1f}s", "Black screen — relax completely (jittered)", C["grey"]),
                ("PREPARE",       f"{CFG.t_prepare:.0f}s",   "Fixation cross — focus attention",          C["accent_blue"]),
                ("CUE",           f"{CFG.t_cue:.2f}s",       "MI or REST label + beep — then disappears", C["warning"]),
                ("MOTOR IMAGERY", f"{CFG.t_imagery:.0f}s",   "Fixation cross only — imagine or rest",     C["accent_cyan"]),
                ("FEEDBACK",      f"{CFG.t_feedback:.1f}s",  "Blink freely — prepare for next trial",     C["accent_mi"]),
            ]

            for i, (name, dur, desc, col) in enumerate(phases):
                y = 145 + i * 68
                box = pygame.Rect(W//2 - 340, y - 18, 680, 52)
                draw_rounded_rect(s, C["panel"], box, radius=10, border=1, border_color=C["border"])
                bar = pygame.Rect(W//2 - 340, y - 18, 5, 52)
                draw_rounded_rect(s, col, bar, radius=2)
                draw_text(s, name, self.app.font_small_bold, col,        W//2 - 290, y + 8, "left")
                draw_text(s, dur,  self.app.font_small_bold, C["white"], W//2 + 310, y + 8, "right")
                draw_text(s, desc, self.app.font_small,      C["grey"],  W//2 - 120, y + 8, "left")

            total_per_run = CFG.trials_per_class * 2
            draw_text(s, f"{CFG.n_runs} runs  ×  {total_per_run} trials  =  "
                         f"{CFG.n_runs * total_per_run} total  ·  mandatory 60s break between runs",
                      self.app.font_body, C["white"], W//2, 570)
            draw_text(s, "SPACE to continue", self.app.font_small, C["dim"], W//2, H - 50)

        elif self.page == 2:
            draw_text(s, "IMAGERY INSTRUCTIONS", self.app.font_heading, C["accent_cyan"], W//2, 75)
            pygame.draw.line(s, C["border"], (W//2 - 300, 108), (W//2 + 300, 108), 1)

            tips = [
                ("Kinesthetic imagery only",
                 "Feel the sensation of your fist closing — muscles contracting, fingers curling."),
                ("During MI cue",
                 "Imagine closing your RIGHT hand as strongly as possible for the full 4 seconds."),
                ("During REST cue",
                 "Clear your mind completely. No movement, no imagery. Just breathe."),
                ("Cue disappears after 1.25s",
                 "The label vanishes — only the fixation cross remains. Keep imagining until it ends."),
                ("No artifacts",
                 "Avoid jaw clenching, blinking, or eye movements during the imagery window."),
            ]

            for i, (title, body) in enumerate(tips):
                y = 148 + i * 88
                box = pygame.Rect(W//2 - 360, y - 12, 720, 68)
                num_col = [C["accent_cyan"], C["accent_mi"], C["accent_rest"],
                           C["warning"], C["accent_blue"]][i]
                draw_rounded_rect(s, C["panel"], box, radius=10, border=1, border_color=C["border"])
                draw_text(s, f"{i+1}", self.app.font_small_bold, num_col, W//2 - 330, y + 22)
                draw_text(s, title, self.app.font_small_bold, C["white"],  W//2 - 295, y + 8,  "left")
                draw_text(s, body,  self.app.font_small,      C["grey"],   W//2 - 295, y + 34, "left")

            draw_text(s, "SPACE — Begin Session", self.app.font_body, C["accent_cyan"], W//2, H - 50)

# ─────────────────────────────────────────────
#  SUBJECT SCREEN
# ─────────────────────────────────────────────

class SubjectScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.subject_id = ""
        self.trials_str = str(CFG.trials_per_class)
        self.active     = "sid"
        self.error      = ""

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.running = False
            elif event.key == pygame.K_TAB:
                self.active = "trials" if self.active == "sid" else "sid"
            elif event.key == pygame.K_RETURN:
                self._submit()
            elif event.key == pygame.K_BACKSPACE:
                if self.active == "sid":
                    self.subject_id = self.subject_id[:-1]
                else:
                    self.trials_str = self.trials_str[:-1]
            else:
                ch = event.unicode
                if self.active == "sid" and ch.isprintable() and len(self.subject_id) < 10:
                    self.subject_id += ch.upper()
                elif self.active == "trials" and ch.isdigit() and len(self.trials_str) < 3:
                    self.trials_str += ch

    def _submit(self):
        if not self.subject_id.strip():
            self.error = "Subject ID cannot be empty."
            return
        try:
            t = int(self.trials_str)
            if t < 5 or t > 100:
                raise ValueError
        except ValueError:
            self.error = "Trials per class per run must be 5–100."
            return
        CFG.subject_id       = self.subject_id.strip()
        CFG.trials_per_class = t
        self.app.goto("baseline")

    def draw(self):
        s = self.surf
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=20)
        W, H = self.W, self.H

        draw_text(s, "SESSION SETUP", self.app.font_heading, C["accent_cyan"], W//2, 100)
        pygame.draw.line(s, C["border"], (W//2 - 220, 130), (W//2 + 220, 130), 1)

        fields = [
            ("Subject ID",               self.subject_id, "sid",    H//2 - 70),
            ("Trials per class per run", self.trials_str, "trials", H//2 + 40),
        ]

        for label, val, key, y in fields:
            active = self.active == key
            col = C["accent_cyan"] if active else C["grey"]
            draw_text(s, label, self.app.font_small_bold, col, W//2, y - 22)
            box = pygame.Rect(W//2 - 180, y, 360, 50)
            draw_rounded_rect(s, C["panel"], box, radius=8, border=2, border_color=col)
            display = val + ("|" if active and int(time.time() * 2) % 2 == 0 else "")
            draw_text(s, display or " ", self.app.font_body, C["white"], W//2, y + 25)

        if self.error:
            draw_text(s, self.error, self.app.font_small, C["warning"], W//2, H//2 + 150)

        try:
            t = int(self.trials_str) if self.trials_str else 0
            total = t * 2 * CFG.n_runs
            info = f"{CFG.n_runs} runs  ×  {t*2} trials  =  {total} total trials"
        except:
            info = ""
        draw_text(s, info, self.app.font_small, C["grey"], W//2, H//2 + 185)
        draw_text(s, "TAB to switch field  ·  ENTER to start", self.app.font_small, C["dim"], W//2, H - 50)

# ─────────────────────────────────────────────
#  BASELINE SCREEN
# ─────────────────────────────────────────────

class BaselineScreen(Screen):
    """
    2 min eyes open + 1 min eyes closed baseline recording.
    LSL markers sent at start/end of each phase.
    """
    PHASE_EYES_OPEN   = "eyes_open"
    PHASE_EYES_CLOSED = "eyes_closed"
    PHASE_DONE        = "done"

    def __init__(self, app):
        super().__init__(app)
        self.phase        = self.PHASE_EYES_OPEN
        self.elapsed      = 0.0
        self.log          = {}
        self.marker_sent  = False   # marker fires after first blank frame renders

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.running = False

    def update(self, dt):
        self.elapsed += dt
        dur = (CFG.t_baseline_eyes_open if self.phase == self.PHASE_EYES_OPEN
               else CFG.t_baseline_eyes_closed)
        if self.elapsed >= dur:
            self.elapsed = 0.0
            if self.phase == self.PHASE_EYES_OPEN:
                push_marker(self.app.lsl_outlet, LSL_MARKERS["baseline_eyes_open_end"])
                push_marker(self.app.lsl_outlet, LSL_MARKERS["baseline_eyes_closed_start"])
                self.log["eyes_open_end_unix"]    = time.time()
                self.log["eyes_closed_start_unix"] = time.time()
                self.phase = self.PHASE_EYES_CLOSED
                print("  Baseline: eyes closed started")
            elif self.phase == self.PHASE_EYES_CLOSED:
                push_marker(self.app.lsl_outlet, LSL_MARKERS["baseline_eyes_closed_end"])
                self.log["eyes_closed_end_unix"] = time.time()
                self.app.beep.play()
                self.phase = self.PHASE_DONE
                print("  Baseline complete")
                self.app.session_log.baseline = self.log
                self.app.goto("session")

    def draw(self):
        s = self.surf
        s.fill(C["bg"])
        W, H = self.W, self.H

        # Send marker AFTER first frame is drawn — screen is visible before marker fires
        if not self.marker_sent:
            pygame.display.flip()   # ensure frame is on screen
            push_marker(self.app.lsl_outlet, LSL_MARKERS["baseline_eyes_open_start"])
            self.log["eyes_open_start_unix"] = time.time()
            self.marker_sent = True
            print("  Baseline: eyes open started")

        draw_text(s, "BASELINE RECORDING", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 180)
        pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 148), (W//2 + 260, H//2 - 148), 1)

        if self.phase == self.PHASE_EYES_OPEN:
            dur = CFG.t_baseline_eyes_open
            remaining = dur - self.elapsed
            draw_fixation_cross(s, W//2, H//2 - 30, size=50, thick=8, color=C["white"])
            draw_text(s, "EYES OPEN", self.app.font_cue, C["accent_blue"], W//2, H//2 + 70)
            draw_text(s, "Look at the fixation cross. Remain still.", self.app.font_body, C["grey"], W//2, H//2 + 120)
            draw_text(s, f"{remaining:.0f}s remaining", self.app.font_small, C["dim"], W//2, H//2 + 160)

        elif self.phase == self.PHASE_EYES_CLOSED:
            dur = CFG.t_baseline_eyes_closed
            remaining = dur - self.elapsed
            s.fill(C["black"])
            draw_text(s, "EYES CLOSED", self.app.font_cue, C["grey"], W//2, H//2 - 20)
            draw_text(s, "Close your eyes. Relax completely.", self.app.font_body, C["grey"], W//2, H//2 + 50)
            draw_text(s, f"{remaining:.0f}s remaining", self.app.font_small, C["dim"], W//2, H//2 + 100)

        # Progress bar
        dur = (CFG.t_baseline_eyes_open if self.phase == self.PHASE_EYES_OPEN
               else CFG.t_baseline_eyes_closed)
        prog_w = int(W * min(1.0, self.elapsed / dur))
        pygame.draw.rect(s, C["accent_blue"], (0, H - 8, prog_w, 8))

        draw_text(s, "ESC to quit", self.app.font_small, C["dim"], W//2, H - 20)

# ─────────────────────────────────────────────
#  BREAK SCREEN
# ─────────────────────────────────────────────

class BreakScreen(Screen):
    """Mandatory rest between runs. Subject presses SPACE when ready (min 60s)."""

    def __init__(self, app, run_number, total_runs):
        super().__init__(app)
        self.run_number  = run_number
        self.total_runs  = total_runs
        self.elapsed     = 0.0
        self.ready       = False
        self.break_start = time.time()
        push_marker(self.app.lsl_outlet, LSL_MARKERS["break_start"])
        print(f"  Break after run {run_number}")

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.running = False
            elif event.key == pygame.K_SPACE and self.ready:
                push_marker(self.app.lsl_outlet, LSL_MARKERS["break_end"])
                dur = time.time() - self.break_start
                self.app.session_log.breaks.append({
                    "after_run": self.run_number,
                    "duration_s": round(dur, 2),
                })
                self.app.goto("session")

    def update(self, dt):
        self.elapsed += dt
        if self.elapsed >= CFG.t_break_min:
            self.ready = True

    def draw(self):
        s = self.surf
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=20)
        W, H = self.W, self.H

        draw_text(s, f"RUN {self.run_number} COMPLETE", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 140)
        pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 110), (W//2 + 260, H//2 - 110), 1)

        draw_text(s, "REST PERIOD", self.app.font_cue, C["white"], W//2, H//2 - 40)

        if not self.ready:
            remaining = CFG.t_break_min - self.elapsed
            draw_text(s, f"Minimum break: {remaining:.0f}s", self.app.font_body, C["grey"], W//2, H//2 + 50)
            draw_text(s, "Relax. Stretch if needed.", self.app.font_small, C["dim"], W//2, H//2 + 90)
            prog_w = int(W * self.elapsed / CFG.t_break_min)
            pygame.draw.rect(s, C["accent_blue"], (0, H - 8, prog_w, 8))
        else:
            draw_text(s, f"{self.total_runs - self.run_number} run(s) remaining",
                      self.app.font_body, C["grey"], W//2, H//2 + 50)
            draw_text(s, "SPACE when ready to continue", self.app.font_body, C["accent_cyan"], W//2, H//2 + 100)

# ─────────────────────────────────────────────
#  SESSION SCREEN
# ─────────────────────────────────────────────

class SessionScreen(Screen):
    """Main paradigm — runs one run of trials."""

    PHASE_REST     = "rest"
    PHASE_PREPARE  = "prepare"
    PHASE_CUE      = "cue"
    PHASE_IMAGERY  = "imagery"
    PHASE_FEEDBACK = "feedback"
    PHASE_DONE     = "done"

    def __init__(self, app):
        super().__init__(app)
        self.reset()

    def reset(self):
        self.run_number    = getattr(self.app, "current_run", 1)
        total_per_run      = CFG.trials_per_class * len(CFG.classes)
        labels             = CFG.classes * CFG.trials_per_class
        random.shuffle(labels)
        self.trial_sequence     = labels
        self.current_trial      = 0
        self.total_trials       = total_per_run
        self.phase              = self.PHASE_REST
        self.phase_elapsed      = 0.0
        self.current_rest_dur   = random.uniform(CFG.t_rest_min, CFG.t_rest_max)
        self.session_start      = time.time()
        self.paused             = False
        self.markers: List[TrialMarker] = []
        self.current_marker: Optional[TrialMarker] = None
        self.phase_bar          = ProgressBar(40, self.H - 30, self.W - 80, 8, C["accent_cyan"])
        self.beep_played        = False

    def session_elapsed(self):
        return time.time() - self.session_start

    def current_label(self):
        if self.current_trial < len(self.trial_sequence):
            return self.trial_sequence[self.current_trial]
        return None

    def phase_duration(self):
        return {
            self.PHASE_REST:     self.current_rest_dur,
            self.PHASE_PREPARE:  CFG.t_prepare,
            self.PHASE_CUE:      CFG.t_cue,
            self.PHASE_IMAGERY:  CFG.t_imagery,
            self.PHASE_FEEDBACK: CFG.t_feedback,
        }.get(self.phase, 1.0)

    def next_phase(self):
        order = [self.PHASE_REST, self.PHASE_PREPARE, self.PHASE_CUE,
                 self.PHASE_IMAGERY, self.PHASE_FEEDBACK]

        if self.phase == self.PHASE_FEEDBACK:
            if self.current_marker:
                self.markers.append(self.current_marker)
                self.app.session_log.trials.append(asdict(self.current_marker))
            self.current_trial += 1
            if self.current_trial >= self.total_trials:
                self.phase = self.PHASE_DONE
                self._finish_run()
                return
            self.phase = self.PHASE_REST
            self.current_rest_dur = random.uniform(CFG.t_rest_min, CFG.t_rest_max)
            self.beep_played = False
        else:
            idx = order.index(self.phase)
            self.phase = order[idx + 1]

        self.phase_elapsed = 0.0

        # LSL marker at imagery onset
        if self.phase == self.PHASE_IMAGERY:
            label     = self.current_label()
            lsl_code  = LSL_MARKERS["cue_MI"] if label == "MI" else LSL_MARKERS["cue_REST"]
            push_marker(self.app.lsl_outlet, LSL_MARKERS["imagery_start"])
            push_marker(self.app.lsl_outlet, lsl_code)
            self.current_marker = TrialMarker(
                trial_number = self.current_trial + 1,
                run_number   = self.run_number,
                label        = label,
                onset_time_s = self.session_elapsed(),
                onset_unix   = time.time(),
                lsl_code     = lsl_code,
                phase_onsets = {"imagery": self.session_elapsed()},
            )

        # LSL marker at imagery end
        if self.phase == self.PHASE_FEEDBACK:
            push_marker(self.app.lsl_outlet, LSL_MARKERS["imagery_end"])

        # Trial start marker
        if self.phase == self.PHASE_REST:
            push_marker(self.app.lsl_outlet, LSL_MARKERS["trial_start"])

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._save_log()
                self.app.running = False
            elif event.key == pygame.K_p and self.phase != self.PHASE_DONE:
                self.paused = not self.paused

    def update(self, dt):
        if self.paused or self.phase == self.PHASE_DONE:
            return
        self.phase_elapsed += dt
        dur = self.phase_duration()
        self.phase_bar.value = min(1.0, self.phase_elapsed / dur)

        pass  # beep only at end of baseline eyes closed

        if self.phase_elapsed >= dur:
            self.next_phase()

    def _finish_run(self):
        self._save_log()
        run = getattr(self.app, "current_run", 1)
        if run < CFG.n_runs:
            self.app.current_run = run + 1
            self.app.goto("break")
        else:
            push_marker(self.app.lsl_outlet, LSL_MARKERS["session_end"])
            self.app.goto("done")

    def _save_log(self):
        self.app.session_log.end_time       = datetime.now().strftime("%H:%M:%S")
        self.app.session_log.total_duration = self.session_elapsed()
        self.app.session_log.lsl_available  = HAS_LSL
        os.makedirs(CFG.output_dir, exist_ok=True)
        fname = os.path.join(
            CFG.output_dir,
            f"markers_{CFG.subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(fname, "w") as f:
            json.dump(asdict(self.app.session_log), f, indent=2)
        print(f"\n  Markers saved → {fname}")

    # ── Drawing ──────────────────────────────────────────────────

    def draw(self):
        s = self.surf
        W, H = self.W, self.H
        label = self.current_label()

        if self.phase == self.PHASE_DONE:
            return  # handled by _finish_run

        if self.phase == self.PHASE_REST:
            s.fill(C["black"])
        else:
            s.fill(C["bg"])
            self.app.eeg_deco.draw(s, alpha=18)

        if self.paused:
            self._draw_paused(); return

        # ── Phase content ──────────────────────────

        if self.phase == self.PHASE_REST:
            pass  # black screen

        elif self.phase == self.PHASE_PREPARE:
            # Fixation cross only
            draw_fixation_cross(s, W//2, H//2, size=45, thick=7, color=C["white"])

        elif self.phase == self.PHASE_CUE:
            col = C["accent_mi"] if label == "MI" else C["accent_rest"]
            draw_text(s, label, self.app.font_cue, col, W//2, H//2)

        elif self.phase == self.PHASE_IMAGERY:
            # FIXATION CROSS ONLY — no hand icons, no label (prevents visual ERPs)
            draw_fixation_cross(s, W//2, H//2, size=50, thick=8, color=C["white"])

        elif self.phase == self.PHASE_FEEDBACK:
            draw_text(s, "+", self.app.font_cue, C["dim"], W//2, H//2 - 10)
            draw_text(s, "You may blink", self.app.font_body, C["grey"], W//2, H//2 + 60)

        # ── HUD + progress bar ──
        self._draw_hud(s, W, H, label)
        self.phase_bar.draw(s)

    def _draw_hud(self, s, W, H, label):
        mi_count   = sum(1 for m in self.markers if m.label == "MI")
        rest_count = sum(1 for m in self.markers if m.label == "REST")

        hud_rect = pygame.Rect(0, 0, W, 52)
        pygame.draw.rect(s, C["panel"], hud_rect)
        pygame.draw.line(s, C["border"], (0, 52), (W, 52), 1)

        draw_text(s, f"Subject: {CFG.subject_id}  |  Run {self.run_number}/{CFG.n_runs}",
                  self.app.font_small_bold, C["grey"], 20, 26, "left")
        draw_text(s, f"Trial  {self.current_trial + 1} / {self.total_trials}",
                  self.app.font_small_bold, C["white"], W//2, 26)

        elapsed = int(self.session_elapsed())
        draw_text(s, f"{elapsed//60:02d}:{elapsed%60:02d}",
                  self.app.font_small_bold, C["grey"], W - 20, 26, "right")

        draw_text(s, f"MI: {mi_count}   REST: {rest_count}",
                  self.app.font_small, C["grey"], 20, H - 45, "left")

        phase_names = {
            self.PHASE_REST:     "REST",
            self.PHASE_PREPARE:  "PREPARE",
            self.PHASE_CUE:      "CUE",
            self.PHASE_IMAGERY:  "MOTOR IMAGERY",
            self.PHASE_FEEDBACK: "FEEDBACK",
        }
        pcol = {
            self.PHASE_REST:     C["grey"],
            self.PHASE_PREPARE:  C["accent_blue"],
            self.PHASE_CUE:      C["warning"],
            self.PHASE_IMAGERY:  C["accent_cyan"],
            self.PHASE_FEEDBACK: C["accent_mi"],
        }.get(self.phase, C["white"])

        remaining = max(0.0, self.phase_duration() - self.phase_elapsed)
        pname = phase_names.get(self.phase, "")
        draw_text(s, f"{pname}  {remaining:.1f}s",
                  self.app.font_small_bold, pcol, W - 20, H - 45, "right")

        prog_w = int(W * self.current_trial / max(1, self.total_trials))
        pygame.draw.rect(s, C["accent_blue"], (0, 51, prog_w, 3))

        if not self.paused:
            draw_text(s, "P — pause  ·  ESC — quit", self.app.font_small, C["dim"], W - 20, H - 20, "right")

        if HAS_LSL:
            draw_text(s, "LSL", self.app.font_small, C["accent_mi"], 20, H - 20, "left")
        else:
            draw_text(s, "LSL: OFF", self.app.font_small, C["warning"], 20, H - 20, "left")

    def _draw_paused(self):
        overlay = alpha_surface(self.W, self.H, C["bg"], 200)
        self.surf.blit(overlay, (0, 0))
        draw_text(self.surf, "PAUSED", self.app.font_heading, C["warning"], self.W//2, self.H//2 - 20)
        draw_text(self.surf, "Press  P  to resume", self.app.font_body, C["grey"], self.W//2, self.H//2 + 40)

# ─────────────────────────────────────────────
#  DONE SCREEN
# ─────────────────────────────────────────────

class DoneScreen(Screen):
    def __init__(self, app):
        super().__init__(app)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.running = False

    def draw(self):
        s = self.surf
        W, H = self.W, self.H
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=30)

        draw_text(s, "SESSION COMPLETE", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 160)
        pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 128), (W//2 + 260, H//2 - 128), 1)

        trials = self.app.session_log.trials
        mi_count   = sum(1 for t in trials if t["label"] == "MI")
        rest_count = sum(1 for t in trials if t["label"] == "REST")
        dur = int(self.app.session_log.total_duration)

        stats = [
            ("Total Trials",   str(len(trials)),            C["white"]),
            ("MI trials",      str(mi_count),               C["accent_mi"]),
            ("REST trials",    str(rest_count),             C["accent_rest"]),
            ("Runs completed", str(CFG.n_runs),             C["accent_blue"]),
            ("Duration",       f"{dur//60:02d}:{dur%60:02d}", C["accent_cyan"]),
        ]

        for i, (label, val, col) in enumerate(stats):
            y = H//2 - 80 + i * 58
            box = pygame.Rect(W//2 - 220, y - 18, 440, 48)
            draw_rounded_rect(s, C["panel"], box, radius=8, border=1, border_color=C["border"])
            draw_text(s, label, self.app.font_small,      C["grey"], W//2 - 80, y + 6)
            draw_text(s, val,   self.app.font_small_bold, col,       W//2 + 80, y + 6)

        draw_text(s, "Markers saved to  eeg_data/",
                  self.app.font_small, C["accent_mi"], W//2, H//2 + 215)
        draw_text(s, "ESC to exit", self.app.font_small, C["dim"], W//2, H - 50)

# ─────────────────────────────────────────────
#  APPLICATION
# ─────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.display.set_caption("EEG Motor Imagery Paradigm — InMoov i2  [v2]")

        self.W = CFG.screen_w
        self.H = CFG.screen_h
        self.screen  = pygame.display.set_mode((self.W, self.H))
        self.clock   = pygame.time.Clock()
        self.running = True

        self._load_fonts()
        self.eeg_deco    = EEGDecoration(self.W, self.H)
        self.beep        = make_beep(CFG.beep_freq, CFG.beep_dur_ms)
        self.lsl_outlet  = create_lsl_outlet()
        self.current_run = 1

        self.recorder = EEGRecorder(CFG.cyton_port, simulate=CFG.simulate)
        self.recorder.start()

        self.session_log = SessionLog(
            subject_id = CFG.subject_id,
            date       = datetime.now().strftime("%Y-%m-%d"),
            start_time = datetime.now().strftime("%H:%M:%S"),
            config     = asdict(CFG),
        )

        self.screens: dict = {}
        self.goto("intro")

    def _load_fonts(self):
        def try_font(names, size, bold=False):
            for name in names:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f: return f
                except: pass
            return pygame.font.SysFont("monospace", size, bold=bold)

        mono  = ["JetBrains Mono", "Fira Code", "Consolas", "Courier New", "monospace"]
        clean = ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]

        self.font_title      = try_font(clean, 52, bold=True)
        self.font_heading    = try_font(clean, 34, bold=True)
        self.font_cue        = try_font(clean, 72, bold=True)
        self.font_body       = try_font(clean, 22)
        self.font_small      = try_font(mono,  17)
        self.font_small_bold = try_font(mono,  17, bold=True)

    def goto(self, name: str):
        constructors = {
            "intro":    IntroScreen,
            "subject":  SubjectScreen,
            "baseline": BaselineScreen,
            "session":  SessionScreen,
            "break":    lambda app: BreakScreen(app, self.current_run - 1, CFG.n_runs),
            "done":     DoneScreen,
        }
        self.screens[name] = constructors[name](self)
        self.current_screen = self.screens[name]

    def run(self):
        while self.running:
            dt = self.clock.tick(CFG.fps) / 1000.0
            dt = min(dt, 0.1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.current_screen.handle_event(event)

            self.eeg_deco.update(dt)
            self.current_screen.update(dt)
            self.current_screen.draw()
            pygame.display.flip()

        pygame.quit()
        self.recorder.stop(CFG.subject_id, CFG.output_dir)
        print("\nParadigm closed.")

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    App().run()
