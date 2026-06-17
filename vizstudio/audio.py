"""Audio engine: capture -> FFT -> features (volume/bass/mid/treble/beat).

One background thread continuously grabs audio (from system loopback, the mic,
or a playing file), runs an FFT, and updates a small `AudioFeatures` snapshot
under a lock. The render loop reads that snapshot each frame.

Every numeric knob in any effect can be "driven" by one of these features, so
this is the single place the whole app gets its reactivity from.

Capture deps are optional and imported lazily: if `soundcard` / `sounddevice`
/ `soundfile` aren't installed (or there's no device), the engine quietly
falls back to silence and the app still runs.
"""

from __future__ import annotations

import threading
import numpy as np

SAMPLE_RATE = 44100
BLOCK = 2048  # frames per analysis block


class AudioFeatures:
    """A normalized 0..1 snapshot of the current sound. `beat` is a bool.

    kick/snare/hihat are narrow drum-tuned bands for more musical reactions
    than the broad bass/mid/treble.
    """

    __slots__ = ("volume", "bass", "mid", "treble", "beat",
                 "kick", "snare", "hihat", "spectrum")

    def __init__(self):
        self.volume = 0.0
        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.beat = False
        self.kick = 0.0
        self.snare = 0.0
        self.hihat = 0.0
        self.spectrum = np.zeros(64, dtype=np.float32)

    def get(self, source):
        """Look up a feature by name (used to drive knobs)."""
        if source == "volume": return self.volume
        if source == "bass":   return self.bass
        if source == "mid":    return self.mid
        if source == "treble": return self.treble
        if source == "beat":   return 1.0 if self.beat else 0.0
        if source == "kick":   return self.kick
        if source == "snare":  return self.snare
        if source == "hihat":  return self.hihat
        return 0.0


def _band_energy(mag, freqs, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    if not np.any(sel):
        return 0.0
    return float(np.sqrt(np.mean(mag[sel] ** 2)))


class AudioEngine:
    """Owns the capture thread and the latest feature snapshot.

    mode: "system" (speaker loopback), "mic", "file", or "none".
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._features = AudioFeatures()
        self._mode = "none"
        self._file_path = None
        self._thread = None
        self._stop = threading.Event()
        self._gain = 1.0
        self.status = "idle"

        # smoothing + beat-detection state
        # order: vol, bass, mid, treble, kick, snare, hihat
        self._smooth = np.zeros(7, dtype=np.float32)
        self._bass_history = np.zeros(43, dtype=np.float32)  # ~1s at this block size
        self._beat_cooldown = 0
        self._window = np.hanning(BLOCK).astype(np.float32)
        self._freqs = np.fft.rfftfreq(BLOCK, 1.0 / SAMPLE_RATE)

    # ---- public control -------------------------------------------------
    def set_gain(self, g):
        self._gain = float(g)

    def current_file(self):
        """The audio file currently loaded (File mode), or None."""
        return self._file_path if self._mode == "file" else None

    def features(self):
        with self._lock:
            f = self._features
            snap = AudioFeatures()
            snap.volume, snap.bass, snap.mid, snap.treble = f.volume, f.bass, f.mid, f.treble
            snap.kick, snap.snare, snap.hihat = f.kick, f.snare, f.hihat
            snap.beat, snap.spectrum = f.beat, f.spectrum
            return snap

    def set_mode(self, mode, file_path=None):
        """Switch audio source. Restarts the capture thread."""
        self.stop()
        self._mode = mode
        self._file_path = file_path
        if mode == "none":
            self.status = "silent"
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    # ---- analysis -------------------------------------------------------
    def _analyze(self, mono):
        if mono.shape[0] < BLOCK:
            mono = np.pad(mono, (0, BLOCK - mono.shape[0]))
        else:
            mono = mono[:BLOCK]
        mono = mono * self._window * self._gain
        mag = np.abs(np.fft.rfft(mono))
        freqs = self._freqs

        vol = float(np.sqrt(np.mean((mono) ** 2)) * 4.0)
        bass = _band_energy(mag, freqs, 20, 250)
        mid = _band_energy(mag, freqs, 250, 2000)
        treble = _band_energy(mag, freqs, 2000, 8000)
        # drum-tuned bands
        kick = _band_energy(mag, freqs, 40, 120)       # bass-drum thump
        snare = _band_energy(mag, freqs, 180, 520)     # snare body
        hihat = _band_energy(mag, freqs, 8000, 16000)  # cymbals / hats

        # crude per-band normalization to land in ~0..1, then smooth
        raw = np.array([vol, bass / 60.0, mid / 30.0, treble / 20.0,
                        kick / 55.0, snare / 26.0, hihat / 9.0], dtype=np.float32)
        raw = np.clip(raw, 0.0, 1.0)
        # fast attack, slow release.  Drum bands (kick/snare/hihat) use a
        # snappier release so hits read as punchy hits, not sustained levels.
        for i in range(7):
            up = 0.6
            down = 0.15 if i < 4 else 0.3
            a = up if raw[i] > self._smooth[i] else down
            self._smooth[i] += (raw[i] - self._smooth[i]) * a

        # beat detection on bass energy vs rolling average
        self._bass_history = np.roll(self._bass_history, -1)
        self._bass_history[-1] = self._smooth[1]
        avg = float(np.mean(self._bass_history)) + 1e-4
        beat = False
        if self._beat_cooldown > 0:
            self._beat_cooldown -= 1
        elif self._smooth[1] > avg * 1.4 and self._smooth[1] > 0.15:
            beat = True
            self._beat_cooldown = 6

        spectrum = np.interp(
            np.linspace(0, len(mag) - 1, 64),
            np.arange(len(mag)),
            np.clip(mag / 50.0, 0, 1),
        ).astype(np.float32)

        with self._lock:
            f = self._features
            f.volume, f.bass, f.mid, f.treble = (
                float(self._smooth[0]), float(self._smooth[1]),
                float(self._smooth[2]), float(self._smooth[3]))
            f.kick, f.snare, f.hihat = (
                float(self._smooth[4]), float(self._smooth[5]),
                float(self._smooth[6]))
            f.beat = beat
            f.spectrum = spectrum

    # ---- capture backends ----------------------------------------------
    def _run(self):
        try:
            if self._mode in ("system", "mic"):
                self._run_soundcard()
            elif self._mode == "file":
                self._run_file()
        except Exception as e:  # never let audio crash the app
            self.status = f"audio off ({type(e).__name__}: {e})"

    def _run_soundcard(self):
        import soundcard as sc
        if self._mode == "system":
            spk = sc.default_speaker()
            mic = sc.get_microphone(spk.name, include_loopback=True)
            self.status = f"system: {spk.name}"
        else:
            mic = sc.default_microphone()
            self.status = f"mic: {mic.name}"
        with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
            while not self._stop.is_set():
                data = rec.record(numframes=BLOCK)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                self._analyze(mono.astype(np.float32))

    def _run_file(self):
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(self._file_path, dtype="float32", always_2d=True)
        self.status = f"file: {self._file_path.split('/')[-1].split(chr(92))[-1]}"
        idx = 0
        n = data.shape[0]
        ev = threading.Event()

        def callback(outdata, frames, time_info, status):
            nonlocal idx
            end = idx + frames
            chunk = data[idx:end]
            if chunk.shape[0] < frames:
                outdata[:chunk.shape[0]] = chunk
                outdata[chunk.shape[0]:] = 0
                raise sd.CallbackStop
            else:
                outdata[:] = chunk
            mono = chunk.mean(axis=1)
            self._analyze(mono.astype(np.float32))
            idx = end

        with sd.OutputStream(samplerate=sr, channels=data.shape[1],
                             blocksize=BLOCK, callback=callback,
                             finished_callback=ev.set):
            while not self._stop.is_set() and not ev.is_set():
                ev.wait(0.1)
