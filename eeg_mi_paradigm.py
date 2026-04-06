"""
=============================================================================
  EEG Motor Imagery Data Collection Paradigm
  Graz-BCI Protocol — Hand Open / Close (2-Class)
=============================================================================
  Thesis: EEG-Based Control of the InMoov i2 Robotic Hand
  Supervisor: Asst. Prof. Athanasios Koutras
  Hardware: OpenBCI Cyton (8-channel)

  Requirements:
      pip install pygame numpy

  Usage:
      python eeg_mi_paradigm.py

  Controls:
      SPACE   — Start session / Advance through intro screens
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

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class SessionConfig:
    subject_id: str         = "S01"
    trials_per_class: int   = 40          # 40 open + 40 close = 80 total
    classes: List[str]      = field(default_factory=lambda: ["OPEN", "CLOSE"])

    # Timing (seconds) — Graz-BCI protocol
    t_rest:      float = 2.0   # Rest min (jittered to t_rest_max — prevents anticipatory ERPs)
    t_rest_max:  float = 3.0   # Rest max (uniform random between t_rest and t_rest_max)
    t_prepare:   float = 2.0   # Fixation cross
    t_cue:       float = 1.0   # Arrow / label cue
    t_imagery:   float = 4.0   # Motor imagery window
    t_feedback:  float = 2.0   # Blink / relax

    # Display
    screen_w:    int = 1280
    screen_h:    int = 800
    fps:         int = 60

    output_dir:  str = "eeg_data"


CFG = SessionConfig()

# ─────────────────────────────────────────────
#  COLOUR PALETTE  — dark neuroscience theme
# ─────────────────────────────────────────────
C = {
    "bg":           (10,  12,  20),
    "panel":        (18,  22,  38),
    "border":       (40,  50,  90),
    "accent_blue":  (60, 140, 255),
    "accent_cyan":  (0,  210, 200),
    "accent_open":  (0,  220, 130),   # green  → hand OPEN
    "accent_close": (255, 90,  80),   # red    → hand CLOSE
    "white":        (240, 245, 255),
    "grey":         (120, 130, 160),
    "dim":          (50,  60,  90),
    "black":        (0,   0,   0),
    "warning":      (255, 180,  40),
}

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TrialMarker:
    trial_number:   int
    label:          str          # "OPEN" or "CLOSE"
    onset_time_s:   float        # seconds since session start
    onset_unix:     float        # unix timestamp
    phase_onsets:   dict = field(default_factory=dict)

@dataclass
class SessionLog:
    subject_id:     str
    date:           str
    start_time:     str
    config:         dict
    trials:         List[dict] = field(default_factory=list)
    end_time:       str = ""
    total_duration: float = 0.0

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
    pygame.draw.rect(surf, color, (cx - size,  cy - thick//2, size*2, thick))

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def alpha_surface(w, h, color, alpha):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color, alpha))
    return s

# ─────────────────────────────────────────────
#  EEG WAVEFORM — decorative animated display
# ─────────────────────────────────────────────

class EEGDecoration:
    """Animated fake EEG waveforms as background decoration."""
    def __init__(self, screen_w, screen_h):
        self.w = screen_w
        self.h = screen_h
        self.channels = 4
        self.phase = [random.uniform(0, 6.28) for _ in range(self.channels)]
        self.amp   = [random.uniform(8, 20)   for _ in range(self.channels)]
        self.freq  = [random.uniform(0.8, 2.2) for _ in range(self.channels)]
        self.noise = [[random.gauss(0, 3) for _ in range(screen_w//4)] for _ in range(self.channels)]

    def update(self, dt):
        for i in range(self.channels):
            self.phase[i] += dt * self.freq[i]
            # scroll noise
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
        self.value = 0.0   # 0.0 → 1.0

    def draw(self, surf, label=""):
        draw_rounded_rect(surf, C["panel"], self.rect, radius=6,
                          border=1, border_color=C["border"])
        if self.value > 0:
            fill = pygame.Rect(self.rect.x, self.rect.y,
                               int(self.rect.w * self.value), self.rect.h)
            draw_rounded_rect(surf, self.color, fill, radius=6)

# ─────────────────────────────────────────────
#  HAND ICON — simple vector drawing
# ─────────────────────────────────────────────

def draw_hand_icon(surf, cx, cy, state="open", size=80, color=None):
    """Draw a schematic open or closed hand."""
    color = color or C["white"]
    palm_w = int(size * 0.55)
    palm_h = int(size * 0.45)
    palm_rect = pygame.Rect(cx - palm_w//2, cy - palm_h//4, palm_w, palm_h)
    draw_rounded_rect(surf, color, palm_rect, radius=10)

    if state == "open":
        # 4 extended fingers
        finger_w  = int(size * 0.10)
        finger_h  = int(size * 0.42)
        offsets   = [-int(size*0.195), -int(size*0.065),
                      int(size*0.065),  int(size*0.195)]
        for ox in offsets:
            r = pygame.Rect(cx + ox - finger_w//2,
                            cy - palm_h//4 - finger_h + 4,
                            finger_w, finger_h)
            draw_rounded_rect(surf, color, r, radius=finger_w//2)
        # thumb
        thumb_w, thumb_h = int(size*0.11), int(size*0.30)
        tr = pygame.Rect(cx - palm_w//2 - thumb_w + 4,
                         cy - palm_h//4 + 4,
                         thumb_w, thumb_h)
        draw_rounded_rect(surf, color, tr, radius=thumb_w//2)
    else:
        # closed fist — knuckles only
        knuckle_r = int(size * 0.075)
        kx_vals = [cx - int(size*0.18), cx - int(size*0.06),
                   cx + int(size*0.06), cx + int(size*0.18)]
        ky = cy - palm_h//4 + 2
        for kx in kx_vals:
            pygame.draw.circle(surf, lerp_color(color, C["bg"], 0.3), (kx, ky), knuckle_r)
        # thumb tucked
        pygame.draw.circle(surf, color,
                           (cx - palm_w//2 + int(size*0.07), cy), int(size*0.09))

# ─────────────────────────────────────────────
#  SCREENS
# ─────────────────────────────────────────────

class Screen:
    def __init__(self, app):
        self.app = app
        self.surf = app.screen
        self.W    = app.W
        self.H    = app.H

    def handle_event(self, event): pass
    def update(self, dt):          pass
    def draw(self):                pass


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
        alpha = int(self.fade_in * 255)

        if self.page == 0:
            # Title page
            draw_text(s, "EEG MOTOR IMAGERY", self.app.font_title, C["accent_blue"],  W//2, H//2 - 120)
            draw_text(s, "DATA COLLECTION PARADIGM", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 68)

            # Divider line
            pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 40), (W//2 + 260, H//2 - 40), 1)

            draw_text(s, "Graz-BCI Protocol  ·  2-Class Hand Motor Imagery", self.app.font_body,
                      C["grey"], W//2, H//2 - 12)
            draw_text(s, "InMoov i2 Robotic Hand Control", self.app.font_body, C["grey"], W//2, H//2 + 18)

            draw_hand_icon(s, W//2 - 70, H//2 + 100, "open",  70, C["accent_open"])
            draw_hand_icon(s, W//2 + 70, H//2 + 100, "close", 70, C["accent_close"])

            draw_text(s, "OPEN", self.app.font_small, C["accent_open"],  W//2 - 70, H//2 + 155)
            draw_text(s, "CLOSE", self.app.font_small, C["accent_close"], W//2 + 70, H//2 + 155)

            draw_text(s, "SPACE to continue", self.app.font_small, C["dim"], W//2, H - 50)

        elif self.page == 1:
            # Protocol overview
            draw_text(s, "SESSION PROTOCOL", self.app.font_heading, C["accent_cyan"], W//2, 80)
            pygame.draw.line(s, C["border"], (W//2 - 300, 110), (W//2 + 300, 110), 1)

            phases = [
                ("REST",         f"{CFG.t_rest:.0f}s",     "Black screen — relax completely",         C["grey"]),
                ("PREPARE",      f"{CFG.t_prepare:.0f}s",  "Fixation cross — focus your attention",    C["accent_blue"]),
                ("CUE",          f"{CFG.t_cue:.0f}s",      "OPEN or CLOSE label appears",              C["warning"]),
                ("MOTOR IMAGERY",f"{CFG.t_imagery:.0f}s",  "Imagine the FEELING of the hand movement", C["accent_cyan"]),
                ("FEEDBACK",     f"{CFG.t_feedback:.0f}s", "Blink freely — prepare for next trial",    C["accent_open"]),
            ]

            total_trial = CFG.t_rest + CFG.t_prepare + CFG.t_cue + CFG.t_imagery + CFG.t_feedback
            total_trials = CFG.trials_per_class * 2
            total_min = total_trials * total_trial / 60

            for i, (name, dur, desc, col) in enumerate(phases):
                y = 155 + i * 78
                box = pygame.Rect(W//2 - 340, y - 22, 680, 56)
                draw_rounded_rect(s, C["panel"], box, radius=10,
                                  border=1, border_color=C["border"])
                # color bar
                bar = pygame.Rect(W//2 - 340, y - 22, 5, 56)
                draw_rounded_rect(s, col, bar, radius=2)

                draw_text(s, name, self.app.font_small_bold, col,         W//2 - 290, y + 6, "left")
                draw_text(s, dur,  self.app.font_small_bold, C["white"],  W//2 + 310, y + 6, "right")
                draw_text(s, desc, self.app.font_small,      C["grey"],   W//2 - 130, y + 6, "left")

            y_info = 570
            draw_text(s, f"Total trials: {total_trials}  ·  "
                         f"{CFG.trials_per_class} per class  ·  "
                         f"≈ {total_min:.0f} minutes",
                      self.app.font_body, C["white"], W//2, y_info)

            draw_text(s, "SPACE to continue", self.app.font_small, C["dim"], W//2, H - 50)

        elif self.page == 2:
            # KMI instructions
            draw_text(s, "IMAGERY INSTRUCTIONS", self.app.font_heading, C["accent_cyan"], W//2, 75)
            pygame.draw.line(s, C["border"], (W//2 - 300, 108), (W//2 + 300, 108), 1)

            tips = [
                ("KINESTHETIC imagery only",
                 "Feel the sensation — muscles stretching, fingers moving. Not watching yourself."),
                ("OPEN cue",
                 "Imagine your palm spreading wide, fingers extending fully outward."),
                ("CLOSE cue",
                 "Imagine your fist clenching tight, all fingers curling into your palm."),
                ("NO artifacts",
                 "Avoid jaw clenching, blinking, or eye movements during the imagery window."),
                ("Stay relaxed",
                 "Only your imagination should be active. Body remains completely still."),
            ]

            for i, (title, body) in enumerate(tips):
                y = 148 + i * 88
                box = pygame.Rect(W//2 - 360, y - 12, 720, 68)
                num_col = [C["accent_cyan"], C["accent_open"], C["accent_close"],
                           C["warning"], C["accent_blue"]][i]
                draw_rounded_rect(s, C["panel"], box, radius=10,
                                  border=1, border_color=C["border"])
                draw_text(s, f"{i+1}", self.app.font_small_bold, num_col, W//2 - 330, y + 22)
                draw_text(s, title, self.app.font_small_bold, C["white"],  W//2 - 295, y + 8,  "left")
                draw_text(s, body,  self.app.font_small,      C["grey"],   W//2 - 295, y + 34, "left")

            draw_text(s, "SPACE — Begin Session", self.app.font_body, C["accent_cyan"], W//2, H - 50)


class SubjectScreen(Screen):
    """Simple subject ID entry screen."""
    def __init__(self, app):
        super().__init__(app)
        self.subject_id = ""
        self.trials_str = str(CFG.trials_per_class)
        self.active     = "sid"   # which field is active
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
            if t < 5 or t > 200:
                raise ValueError
        except ValueError:
            self.error = "Trials per class must be between 5 and 200."
            return
        CFG.subject_id      = self.subject_id.strip()
        CFG.trials_per_class = t
        self.app.goto("session")

    def draw(self):
        s = self.surf
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=20)
        W, H = self.W, self.H

        draw_text(s, "SESSION SETUP", self.app.font_heading, C["accent_cyan"], W//2, 100)
        pygame.draw.line(s, C["border"], (W//2 - 220, 130), (W//2 + 220, 130), 1)

        fields = [
            ("Subject ID",          self.subject_id,  "sid",    H//2 - 60),
            ("Trials per class",    self.trials_str,  "trials", H//2 + 50),
        ]

        for label, val, key, y in fields:
            active = self.active == key
            col = C["accent_cyan"] if active else C["grey"]
            draw_text(s, label, self.app.font_small_bold, col, W//2, y - 20)
            box = pygame.Rect(W//2 - 180, y, 360, 50)
            draw_rounded_rect(s, C["panel"], box, radius=8,
                              border=2, border_color=col)
            display = val + ("|" if active and int(time.time() * 2) % 2 == 0 else "")
            draw_text(s, display or " ", self.app.font_body, C["white"], W//2, y + 25)

        if self.error:
            draw_text(s, self.error, self.app.font_small, C["accent_close"], W//2, H//2 + 140)

        # Info box
        total_trial = CFG.t_rest + CFG.t_prepare + CFG.t_cue + CFG.t_imagery + CFG.t_feedback
        try:
            t = int(self.trials_str) if self.trials_str else 0
            mins = t * 2 * total_trial / 60
            info = f"Total: {t*2} trials  ·  ≈ {mins:.0f} min"
        except:
            info = ""
        draw_text(s, info, self.app.font_small, C["grey"], W//2, H//2 + 175)

        draw_text(s, "TAB to switch  ·  ENTER to start", self.app.font_small, C["dim"], W//2, H - 50)


class SessionScreen(Screen):
    """Main paradigm display — runs trials."""

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
        total = CFG.trials_per_class * len(CFG.classes)
        labels = CFG.classes * CFG.trials_per_class
        random.shuffle(labels)
        self.trial_sequence = labels
        self.current_trial  = 0
        self.total_trials   = total
        self.phase          = self.PHASE_REST
        self.phase_elapsed  = 0.0
        self.current_rest_duration = random.uniform(CFG.t_rest, CFG.t_rest_max)
        self.session_start  = time.time()
        self.paused         = False
        self.markers: List[TrialMarker] = []
        self.current_marker: Optional[TrialMarker] = None
        self.phase_bar      = ProgressBar(40, self.H - 30, self.W - 80, 8, C["accent_cyan"])

        self.log = SessionLog(
            subject_id = CFG.subject_id,
            date       = datetime.now().strftime("%Y-%m-%d"),
            start_time = datetime.now().strftime("%H:%M:%S"),
            config     = asdict(CFG),
        )
        os.makedirs(CFG.output_dir, exist_ok=True)

    def session_elapsed(self):
        return time.time() - self.session_start

    def current_label(self):
        if self.current_trial < len(self.trial_sequence):
            return self.trial_sequence[self.current_trial]
        return None

    def phase_duration(self):
        return {
            self.PHASE_REST:     self.current_rest_duration,
            self.PHASE_PREPARE:  CFG.t_prepare,
            self.PHASE_CUE:      CFG.t_cue,
            self.PHASE_IMAGERY:  CFG.t_imagery,
            self.PHASE_FEEDBACK: CFG.t_feedback,
        }.get(self.phase, 1.0)

    def next_phase(self):
        order = [self.PHASE_REST, self.PHASE_PREPARE, self.PHASE_CUE,
                 self.PHASE_IMAGERY, self.PHASE_FEEDBACK]
        idx = order.index(self.phase) if self.phase in order else -1

        if self.phase == self.PHASE_FEEDBACK:
            # Log completed trial
            if self.current_marker:
                self.markers.append(self.current_marker)
                self.log.trials.append(asdict(self.current_marker))
            self.current_trial += 1
            if self.current_trial >= self.total_trials:
                self.phase = self.PHASE_DONE
                self._save_log()
                return
            self.phase = self.PHASE_REST
            self.current_rest_duration = random.uniform(CFG.t_rest, CFG.t_rest_max)
        else:
            self.phase = order[idx + 1]

        self.phase_elapsed = 0.0

        # Create marker at imagery onset
        if self.phase == self.PHASE_IMAGERY:
            self.current_marker = TrialMarker(
                trial_number = self.current_trial + 1,
                label        = self.current_label(),
                onset_time_s = self.session_elapsed(),
                onset_unix   = time.time(),
                phase_onsets = {"imagery": self.session_elapsed()},
            )

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
        if self.phase_elapsed >= dur:
            self.next_phase()

    def _save_log(self):
        self.log.end_time       = datetime.now().strftime("%H:%M:%S")
        self.log.total_duration = self.session_elapsed()
        fname = os.path.join(
            CFG.output_dir,
            f"markers_{CFG.subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(fname, "w") as f:
            json.dump(asdict(self.log), f, indent=2)
        print(f"\n✓ Markers saved → {fname}")

    # ── Drawing ──────────────────────────────────

    def draw(self):
        s = self.surf
        W, H = self.W, self.H
        label = self.current_label()

        if self.phase == self.PHASE_DONE:
            self._draw_done(); return

        # Background
        if self.phase == self.PHASE_REST:
            s.fill(C["black"])
        else:
            s.fill(C["bg"])
            self.app.eeg_deco.draw(s, alpha=18)

        if self.paused:
            self._draw_paused()
            return

        # ── Phase-specific content ──
        if self.phase == self.PHASE_REST:
            pass  # black screen

        elif self.phase == self.PHASE_PREPARE:
            draw_fixation_cross(s, W//2, H//2, size=45, thick=7, color=C["white"])

        elif self.phase == self.PHASE_CUE:
            col  = C["accent_open"] if label == "OPEN" else C["accent_close"]
            draw_fixation_cross(s, W//2, H//2, size=45, thick=7, color=C["dim"])
            # Arrow
            if label == "OPEN":
                self._draw_arrow_up(s, W//2, H//2 - 140, col)
            else:
                self._draw_arrow_down(s, W//2, H//2 - 140, col)
            draw_text(s, label, self.app.font_cue, col, W//2, H//2 + 20)

        elif self.phase == self.PHASE_IMAGERY:
            col = C["accent_open"] if label == "OPEN" else C["accent_close"]
            state = "open" if label == "OPEN" else "close"
            draw_hand_icon(s, W//2, H//2 - 30, state, size=120, color=col)
            draw_text(s, label, self.app.font_cue, col, W//2, H//2 + 100)
            draw_text(s, "imagine the sensation", self.app.font_small, C["grey"], W//2, H//2 + 148)

        elif self.phase == self.PHASE_FEEDBACK:
            # Smiley / blink indicator
            draw_text(s, "😌", self.app.font_emoji, C["white"], W//2, H//2 - 20)
            draw_text(s, "You may blink", self.app.font_body, C["grey"], W//2, H//2 + 60)

        # ── HUD ──
        self._draw_hud(s, W, H, label)

        # ── Phase progress bar ──
        self.phase_bar.draw(s)

    def _draw_hud(self, s, W, H, label):
        # Trial counter
        done   = self.current_trial
        total  = self.total_trials
        opens  = sum(1 for m in self.markers if m.label == "OPEN")
        closes = sum(1 for m in self.markers if m.label == "CLOSE")

        # Top bar
        hud_rect = pygame.Rect(0, 0, W, 52)
        pygame.draw.rect(s, C["panel"], hud_rect)
        pygame.draw.line(s, C["border"], (0, 52), (W, 52), 1)

        draw_text(s, f"Subject: {CFG.subject_id}", self.app.font_small_bold,
                  C["grey"], 20, 26, "left")
        draw_text(s, f"Trial  {done + 1} / {total}", self.app.font_small_bold,
                  C["white"], W//2, 26)

        elapsed = int(self.session_elapsed())
        draw_text(s, f"{elapsed//60:02d}:{elapsed%60:02d}",
                  self.app.font_small_bold, C["grey"], W - 20, 26, "right")

        # Class counters bottom-left
        draw_text(s, f"✓ OPEN: {opens}",  self.app.font_small, C["accent_open"],  20, H - 55, "left")
        draw_text(s, f"✓ CLOSE: {closes}", self.app.font_small, C["accent_close"], 20, H - 35, "left")

        # Phase label bottom-right
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
            self.PHASE_FEEDBACK: C["accent_open"],
        }.get(self.phase, C["white"])

        pname = phase_names.get(self.phase, "")
        remaining = max(0.0, self.phase_duration() - self.phase_elapsed)
        draw_text(s, f"{pname}  {remaining:.1f}s",
                  self.app.font_small_bold, pcol, W - 20, H - 45, "right")

        # Overall progress bar (thin, top)
        prog_w = int(W * done / max(1, self.total_trials))
        pygame.draw.rect(s, C["accent_blue"], (0, 51, prog_w, 3))

        # P = pause hint
        if self.paused:
            draw_text(s, "PAUSED — P to resume", self.app.font_small, C["warning"], W//2, H - 45)
        else:
            draw_text(s, "P — pause  ·  ESC — quit", self.app.font_small, C["dim"], W - 20, H - 20, "right")

    def _draw_arrow_up(self, surf, cx, cy, color):
        pts = [(cx, cy - 50), (cx - 40, cy + 20), (cx - 15, cy + 20),
               (cx - 15, cy + 50), (cx + 15, cy + 50),
               (cx + 15, cy + 20), (cx + 40, cy + 20)]
        pygame.draw.polygon(surf, color, pts)

    def _draw_arrow_down(self, surf, cx, cy, color):
        pts = [(cx, cy + 50), (cx - 40, cy - 20), (cx - 15, cy - 20),
               (cx - 15, cy - 50), (cx + 15, cy - 50),
               (cx + 15, cy - 20), (cx + 40, cy - 20)]
        pygame.draw.polygon(surf, color, pts)

    def _draw_paused(self):
        overlay = alpha_surface(self.W, self.H, C["bg"], 200)
        self.surf.blit(overlay, (0, 0))
        draw_text(self.surf, "PAUSED", self.app.font_heading, C["warning"], self.W//2, self.H//2 - 20)
        draw_text(self.surf, "Press  P  to resume", self.app.font_body, C["grey"], self.W//2, self.H//2 + 40)

    def _draw_done(self):
        s = self.surf
        W, H = self.W, self.H
        s.fill(C["bg"])
        self.app.eeg_deco.draw(s, alpha=30)

        draw_text(s, "SESSION COMPLETE", self.app.font_heading, C["accent_cyan"], W//2, H//2 - 160)
        pygame.draw.line(s, C["border"], (W//2 - 260, H//2 - 128), (W//2 + 260, H//2 - 128), 1)

        opens  = sum(1 for m in self.markers if m.label == "OPEN")
        closes = sum(1 for m in self.markers if m.label == "CLOSE")
        dur    = int(self.session_elapsed())

        stats = [
            ("Total Trials",     str(len(self.markers)),          C["white"]),
            ("OPEN trials",      str(opens),                       C["accent_open"]),
            ("CLOSE trials",     str(closes),                      C["accent_close"]),
            ("Duration",         f"{dur//60:02d}:{dur%60:02d}",   C["accent_blue"]),
        ]

        for i, (label, val, col) in enumerate(stats):
            y = H//2 - 80 + i * 65
            box = pygame.Rect(W//2 - 220, y - 20, 440, 52)
            draw_rounded_rect(s, C["panel"], box, radius=8,
                              border=1, border_color=C["border"])
            draw_text(s, label, self.app.font_small,      C["grey"],  W//2 - 80, y + 6)
            draw_text(s, val,   self.app.font_small_bold, col,        W//2 + 80, y + 6)

        draw_text(s, "✓ Markers saved to  eeg_data/",
                  self.app.font_small, C["accent_open"], W//2, H//2 + 200)
        draw_text(s, "ESC to exit", self.app.font_small, C["dim"], W//2, H - 50)


# ─────────────────────────────────────────────
#  APPLICATION
# ─────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("EEG Motor Imagery Paradigm — InMoov i2")
        self.W = CFG.screen_w
        self.H = CFG.screen_h
        self.screen  = pygame.display.set_mode((self.W, self.H))
        self.clock   = pygame.time.Clock()
        self.running = True

        self._load_fonts()
        self.eeg_deco = EEGDecoration(self.W, self.H)

        self.screens = {}
        self.current_screen: Optional[Screen] = None
        self.goto("intro")

    def _load_fonts(self):
        # Use system monospace for a technical feel; fallback gracefully
        def try_font(names, size, bold=False):
            for name in names:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f: return f
                except: pass
            return pygame.font.SysFont("monospace", size, bold=bold)

        mono_names  = ["JetBrains Mono", "Fira Code", "Consolas",
                       "Courier New", "monospace"]
        clean_names = ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]

        self.font_title      = try_font(clean_names, 52, bold=True)
        self.font_heading    = try_font(clean_names, 34, bold=True)
        self.font_cue        = try_font(clean_names, 72, bold=True)
        self.font_body       = try_font(clean_names, 22)
        self.font_small      = try_font(mono_names,  17)
        self.font_small_bold = try_font(mono_names,  17, bold=True)
        self.font_emoji      = try_font(["Segoe UI Emoji", "Apple Color Emoji",
                                         "Noto Color Emoji", "sans-serif"], 64)

    def goto(self, name: str):
        constructors = {
            "intro":   IntroScreen,
            "subject": SubjectScreen,
            "session": SessionScreen,
        }
        if name not in self.screens or name == "session":
            self.screens[name] = constructors[name](self)
        self.current_screen = self.screens[name]

    def run(self):
        while self.running:
            dt = self.clock.tick(CFG.fps) / 1000.0
            dt = min(dt, 0.1)   # clamp to avoid huge jumps

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
        print("\nParadigm closed.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EEG Motor Imagery Paradigm — InMoov i2 Robotic Hand")
    print("  Graz-BCI Protocol | 2-Class: OPEN / CLOSE")
    print("=" * 60)
    print(f"  Trials per class : {CFG.trials_per_class}")
    print(f"  Trial duration   : {CFG.t_rest + CFG.t_prepare + CFG.t_cue + CFG.t_imagery + CFG.t_feedback:.1f}s")
    print(f"  Output directory : {CFG.output_dir}/")
    print("  Controls: SPACE = advance  |  P = pause  |  ESC = quit")
    print("=" * 60 + "\n")
    App().run()
    