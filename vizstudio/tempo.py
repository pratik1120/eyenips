"""The beat clock — musical time as drive sources.

This is the first piece of the "Music Director": instead of reacting to raw
loudness, Eyenips keeps a tempo + beat grid and exposes *musical time* the same
way it exposes audio bands and LFOs. Any knob can then be driven by **Bar / 1-2
/ 1-4 / 1-8 note** (smooth, tempo-locked motion) or **Beat / Bar pulse** (a hit
that fires on each beat / downbeat) — so a sine breathes over exactly one bar and
a flash lands on every kick, locked to the song rather than to wall-clock seconds.

Tempo comes from three places, most reliable first:
  • a BPM you type, or **Tap** to set,
  • **Auto** — follow the audio beat detector (BPM from the gaps between kicks,
    plus a gentle phase nudge so the grid stays locked),
  • **Set downbeat** to mark where bar 1 begins.
"""

from __future__ import annotations

import math
import threading
import time

# musical-time drive sources (shown in every knob's "drive" menu)
TEMPO_IDS = ["bar", "half", "quarter", "eighth", "beatpulse", "barpulse"]
TEMPO_LABELS = {
    "bar": "Bar", "half": "1/2 note", "quarter": "1/4 note",
    "eighth": "1/8 note", "beatpulse": "Beat pulse", "barpulse": "Bar pulse",
}
BEATS_PER_BAR = 4


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def phases_from_beats(beats):
    """Musical phases from a continuous beat count. Shared by the wall-clock
    tempo engine and the song-position clock (analyzed track), so Bar/Beat lock
    to whichever is driving."""
    qp = beats % 1.0                    # within a beat
    hp = (beats / 2.0) % 1.0            # within a 1/2 note
    bp = (beats / BEATS_PER_BAR) % 1.0  # within a bar
    ep = (beats * 2.0) % 1.0            # within a 1/8 note

    def s(p):
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * p)
    return {"bar": s(bp), "half": s(hp), "quarter": s(qp), "eighth": s(ep),
            "beatpulse": 1.0 - qp, "barpulse": 1.0 - bp}


class TempoEngine:
    """Holds BPM + a downbeat anchor and turns wall-clock time into musical
    phases. Thread-safe: the UI sets tempo, the render loop reads phases()."""

    def __init__(self, bpm=120.0):
        self._lock = threading.Lock()
        self._bpm = float(bpm)
        self._t0 = time.perf_counter()   # time of a downbeat (beat 0 of a bar)
        self._taps = []                  # recent tap times (manual tap tempo)
        self._beats = []                 # recent detected-beat times (auto)
        self.auto = False

    # ---- tempo control (UI thread) -------------------------------------
    def set_bpm(self, bpm):
        """Change tempo while preserving the current beat phase (no jump)."""
        now = time.perf_counter()
        with self._lock:
            beats = (now - self._t0) * self._bpm / 60.0
            self._bpm = max(40.0, min(300.0, float(bpm)))
            self._t0 = now - beats * 60.0 / self._bpm

    def bpm(self):
        with self._lock:
            return self._bpm

    def align(self):
        """Mark NOW as a downbeat (bar 1, beat 0)."""
        with self._lock:
            self._t0 = time.perf_counter()

    def tap(self):
        """Tap tempo: average the gaps between recent taps -> BPM, and put the
        downbeat on the latest tap."""
        now = time.perf_counter()
        with self._lock:
            self._taps = [t for t in self._taps if now - t < 2.5] + [now]
            if len(self._taps) >= 2:
                gaps = [b - a for a, b in zip(self._taps, self._taps[1:])]
                avg = sum(gaps) / len(gaps)
                if 0.2 <= avg <= 1.5:                 # 40..300 BPM
                    self._bpm = 60.0 / avg
            self._t0 = now                            # downbeat on this tap

    def set_auto(self, on):
        with self._lock:
            self.auto = bool(on)
            if not on:
                self._beats = []

    # ---- auto: follow the audio beat detector (render thread) ----------
    def on_beat(self, now):
        """Called on each detected beat onset (auto mode). Estimates BPM from the
        gaps between recent beats and gently nudges the grid into phase."""
        with self._lock:
            self._beats = [t for t in self._beats if now - t < 3.0] + [now]
            if len(self._beats) >= 4:
                gaps = [b - a for a, b in zip(self._beats, self._beats[1:])]
                med = _median(gaps)
                if 0.25 <= med <= 1.0:                # 60..240 BPM
                    self._bpm = self._bpm * 0.6 + (60.0 / med) * 0.4
            # PLL nudge: pull the grid so this beat sits on a beat boundary,
            # keeping the integer beat count (so bars stay continuous).
            beats = (now - self._t0) * self._bpm / 60.0
            err = beats - round(beats)                # -0.5..0.5 beats off
            self._t0 += err * (60.0 / self._bpm) * 0.3

    # ---- per-frame musical phases (render thread) ----------------------
    def beats_now(self):
        now = time.perf_counter()
        with self._lock:
            return (now - self._t0) * self._bpm / 60.0

    def phases(self):
        return phases_from_beats(self.beats_now())

    # ---- persistence ---------------------------------------------------
    def to_dict(self):
        with self._lock:
            return {"bpm": self._bpm, "auto": self.auto}

    def load_dict(self, data):
        if not data:
            return
        with self._lock:
            self._bpm = max(40.0, min(300.0, float(data.get("bpm", self._bpm))))
            self.auto = bool(data.get("auto", self.auto))
