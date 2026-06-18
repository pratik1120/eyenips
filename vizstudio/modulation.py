"""Modulation engine — the routing spine.

The app already lets a numeric knob be "driven" by an audio band (bass, beat,
kick, ...). This adds the *other* kind of driver: free-running shapes — **LFOs**
that wobble a value over time on a sine / triangle / saw / square / random, at a
rate you set. Audio bands and LFOs are exposed through ONE lookup (`signals`),
so to the rest of the app an LFO is just another drive source — a knob doesn't
care whether its driver is the kick drum or LFO 2.

This is deliberately the single seam every input flows through: later sources
(MIDI controllers, OSC, an Ableton-Link beat clock) register here the same way,
so "route anything to anything" is built once and everything plugs into it.
"""

from __future__ import annotations

import math
import threading

# the wave shapes an LFO can take (all normalized to a 0..1 output)
LFO_SHAPES = ["sine", "triangle", "saw", "ramp-down", "square", "random"]
N_LFOS = 4
# stable source ids + friendly labels (shown in every knob's "drive" dropdown)
LFO_IDS = [f"lfo{i + 1}" for i in range(N_LFOS)]
LFO_LABELS = {f"lfo{i + 1}": f"LFO {i + 1}" for i in range(N_LFOS)}
# the audio bands that also appear in the unified signal table
AUDIO_BANDS = ("volume", "bass", "mid", "treble", "beat", "kick", "snare", "hihat")


def _hash01(n):
    """Deterministic pseudo-random in [0,1) from an integer — for sample&hold,
    so 'random' LFOs need no per-frame state (and survive save/restore)."""
    x = math.sin(n * 12.9898) * 43758.5453
    return x - math.floor(x)


def _wave(shape, phase):
    """One LFO cycle -> [0,1]. `phase` is the cycle position in [0,1)."""
    if shape == "sine":
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)   # starts at 0, smooth
    if shape == "triangle":
        return 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
    if shape == "saw":
        return phase
    if shape == "ramp-down":
        return 1.0 - phase
    if shape == "square":
        return 1.0 if phase < 0.5 else 0.0
    return phase


class ModEngine:
    """Owns the LFOs and merges them with the audio into one signal lookup.

    Thread-safe: the UI thread edits LFOs while the render thread reads them.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # each LFO: shape, rate (Hz), depth (0..1 output amplitude), phase offset
        self.lfos = [dict(shape="sine", rate=0.5, depth=1.0, phase=0.0)
                     for _ in range(N_LFOS)]

    # ---- live editing (UI thread) --------------------------------------
    def set_lfo(self, idx, shape=None, rate=None, depth=None, phase=None):
        if not (0 <= idx < N_LFOS):
            return
        with self._lock:
            l = self.lfos[idx]
            if shape is not None:
                l["shape"] = shape
            if rate is not None:
                l["rate"] = float(rate)
            if depth is not None:
                l["depth"] = float(depth)
            if phase is not None:
                l["phase"] = float(phase)

    def get_lfo(self, idx):
        with self._lock:
            return dict(self.lfos[idx])

    # ---- per-frame (render thread) -------------------------------------
    def signals(self, t, feats):
        """A {source_name: value} dict combining the audio bands with the LFOs,
        so knob resolution can look up either kind of driver by name."""
        d = {}
        for s in AUDIO_BANDS:
            d[s] = feats.get(s) if feats is not None else 0.0
        with self._lock:
            snap = [dict(l) for l in self.lfos]
        for i, l in enumerate(snap):
            rate = l["rate"]
            if l["shape"] == "random":
                w = _hash01(int(math.floor(t * rate)))      # sample & hold
            else:
                phase = (t * rate + l["phase"]) % 1.0
                w = _wave(l["shape"], phase)
            d[LFO_IDS[i]] = l["depth"] * w
        return d

    # ---- persistence (part of a project / session) ---------------------
    def to_dict(self):
        with self._lock:
            return {"lfos": [dict(l) for l in self.lfos]}

    def load_dict(self, data):
        if not data:
            return
        lfos = data.get("lfos") or []
        with self._lock:
            for i in range(min(len(lfos), N_LFOS)):
                src, l = lfos[i], self.lfos[i]
                l["shape"] = src.get("shape", l["shape"])
                l["rate"] = float(src.get("rate", l["rate"]))
                l["depth"] = float(src.get("depth", l["depth"]))
                l["phase"] = float(src.get("phase", l["phase"]))
