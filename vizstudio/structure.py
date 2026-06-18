"""Offline song analysis — the brain of the Music Director.

Given a track's samples, this produces a *map of the song*: a tempo, a beat grid,
a smooth **intensity** curve (energy normalized over the whole song, so it ramps
through build-ups and peaks on the drop), a **build** signal (how fast intensity
is rising — high during build-ups), and a list of **drop** times (where it goes
from quiet to slamming).

It's deliberately lean numpy DSP — no librosa, no ML, no "AI". The artist still
designs every visual; this just lets Eyenips read the song's shape so the show
can follow it. The engine samples `Structure.at(t)` each frame at the current
playback position and exposes intensity / build / drop as ordinary drive sources.
"""

from __future__ import annotations

import numpy as np

ANALYZE_SR = 22050          # decimate to this for speed (envelopes don't need hi-fi)
HOP = 512                   # frames hop (~23 ms at 22050)
WIN = 1024

# drive sources the Music Director exposes (joined into every knob's drive menu)
DIRECTOR_IDS = ["intensity", "build", "drop"]
DIRECTOR_LABELS = {"intensity": "Intensity", "build": "Build", "drop": "Drop"}


def _frames(x, win, hop):
    """Stack overlapping frames -> (n_frames, win) without copying much."""
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.zeros((0, win), np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _smooth(x, n):
    """Centered moving average over `n` samples (odd-ish), edge-safe."""
    n = max(1, int(n))
    if n <= 1 or len(x) == 0:
        return x.astype(np.float32)
    k = np.ones(n, np.float32) / n
    return np.convolve(x, k, mode="same").astype(np.float32)


def _norm_pct(x, lo=5, hi=95):
    """Normalize to 0..1 by the 5th/95th percentiles (robust to outliers)."""
    if len(x) == 0:
        return x
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    if b - a < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _estimate_bpm(flux, rate, lo=70.0, hi=180.0):
    """Autocorrelation of the onset envelope -> tempo (BPM) in [lo, hi]."""
    if len(flux) < 8:
        return 120.0
    x = flux - flux.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lag_lo = int(rate * 60.0 / hi)
    lag_hi = int(rate * 60.0 / lo)
    lag_hi = min(lag_hi, len(ac) - 1)
    if lag_hi <= lag_lo:
        return 120.0
    best = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
    return float(60.0 * rate / best)


def _beat_times(flux, rate, bpm, duration):
    """A simple beat grid: phase-align a pulse train at `bpm` to the onsets."""
    period = 60.0 / bpm
    step = period * rate                        # samples per beat in env terms
    if step < 1 or len(flux) == 0:
        return []
    # try a handful of phase offsets, keep the one that lands on the most energy
    offsets = np.linspace(0, step, 16, endpoint=False)
    best_off, best_score = 0.0, -1.0
    n_beats = int(len(flux) / step)
    for off in offsets:
        idx = (off + step * np.arange(n_beats)).astype(int)
        idx = idx[idx < len(flux)]
        score = float(flux[idx].sum())
        if score > best_score:
            best_score, best_off = score, off
    t0 = best_off / rate
    return [t0 + i * period for i in range(int(duration / period))]


def _find_drops(inten_short, hop_t):
    """Drop = energy crossing UP through ~0.55 after a quieter stretch, staying
    high. Returns a list of times (seconds), min ~6 s apart."""
    drops = []
    pre = int(round(3.0 / hop_t))               # look ~3 s before / after
    post = int(round(3.0 / hop_t))
    gap = int(round(6.0 / hop_t))
    last = -gap
    for k in range(1, len(inten_short)):
        if inten_short[k - 1] < 0.55 <= inten_short[k] and k - last >= gap:
            before = inten_short[max(0, k - pre):k].mean() if k > 0 else 1.0
            after = inten_short[k:k + post].mean() if k < len(inten_short) else 0.0
            if before < 0.45 and after > 0.6:
                drops.append(k * hop_t)
                last = k
    return drops


class Structure:
    """The analyzed song map; `at(t)` samples it at a playback time (seconds)."""

    def __init__(self, bpm, beat_times, intensity, build, hop_t, drops, duration):
        self.bpm = float(bpm)
        self.beat_times = list(beat_times)
        self.intensity = intensity              # np.float32 array, 0..1
        self.build = build                      # np.float32 array, 0..1
        self.hop_t = float(hop_t)               # seconds per intensity sample
        self.drops = list(drops)                # seconds
        self.duration = float(duration)

    def at(self, t):
        """(intensity, build) linearly interpolated at time `t`."""
        if self.hop_t <= 0 or len(self.intensity) == 0:
            return 0.0, 0.0
        f = t / self.hop_t
        k = int(f)
        if k < 0:
            return float(self.intensity[0]), float(self.build[0])
        if k >= len(self.intensity) - 1:
            return float(self.intensity[-1]), float(self.build[-1])
        a = f - k
        i = self.intensity[k] * (1 - a) + self.intensity[k + 1] * a
        b = self.build[k] * (1 - a) + self.build[k + 1] * a
        return float(i), float(b)

    def to_dict(self):
        return {"bpm": self.bpm, "duration": self.duration,
                "drops": self.drops, "beats": len(self.beat_times)}


def analyze(samples, sr):
    """Analyze a track. `samples`: (n,) or (n, ch) float. Returns a Structure."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr > ANALYZE_SR:                         # crude decimation (fine for envelopes)
        factor = int(round(sr / ANALYZE_SR))
        x = x[::factor]
        sr = sr / factor
    duration = len(x) / sr
    rate = sr / HOP                             # frames per second

    fr = _frames(x, WIN, HOP)
    if len(fr) == 0:
        return Structure(120.0, [], np.zeros(1, np.float32), np.zeros(1, np.float32),
                         HOP / sr, [], duration)
    win = np.hanning(WIN).astype(np.float32)
    mag = np.abs(np.fft.rfft(fr * win, axis=1))

    # onset envelope (spectral flux) -> tempo + beat grid
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux]).astype(np.float32)
    flux = _norm_pct(_smooth(flux, 3))
    bpm = _estimate_bpm(flux, rate)
    beats = _beat_times(flux, rate, bpm, duration)

    # energy -> intensity (macro, ~1.5 s smooth) + build (rising slope)
    energy = np.sqrt((fr ** 2).mean(axis=1) + 1e-9).astype(np.float32)
    inten = _norm_pct(_smooth(energy, int(round(1.5 * rate))))
    inten_short = _norm_pct(_smooth(energy, int(round(0.4 * rate))))
    slope = np.gradient(_smooth(inten, int(round(0.8 * rate)))) * rate
    build = _norm_pct(np.maximum(0.0, slope))
    drops = _find_drops(inten_short, HOP / sr)

    return Structure(bpm, beats, inten, build, HOP / sr, drops, duration)
