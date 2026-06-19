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
import time
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
        self._sys_device = None      # which output to loopback (None = default)
        self._thread = None
        self._stop = threading.Event()
        self._gain = 1.0
        self.status = "idle"
        # file-playback position (so the Music Director can sync its song map)
        self._file_pos = 0          # frames played
        self._file_sr = SAMPLE_RATE
        self._file_dur = 0.0

        # smoothing + beat-detection state
        # order: vol, bass, mid, treble, kick, snare, hihat
        self._smooth = np.zeros(7, dtype=np.float32)
        self._bass_history = np.zeros(43, dtype=np.float32)  # ~1s at this block size
        self._beat_cooldown = 0
        # adaptive gain: a decaying running peak per band, so a loud (rock) and a
        # quiet master both use the full 0..1 range. Fixed divisors used to pin
        # loud tracks at 1.0 -> every band looked "constant" except the hi-hat.
        self._peak = np.full(7, 1e-4, dtype=np.float32)
        self._peak_decay = 0.992          # ~4 s release at this block rate
        self._band_ema = np.zeros(7, dtype=np.float32)  # short avg, for transients
        self._window = np.hanning(BLOCK).astype(np.float32)
        self._freqs = np.fft.rfftfreq(BLOCK, 1.0 / SAMPLE_RATE)

    # ---- public control -------------------------------------------------
    def set_gain(self, g):
        self._gain = float(g)

    def current_file(self):
        """The audio file currently loaded (File mode), or None."""
        return self._file_path if self._mode == "file" else None

    def position(self):
        """Seconds into the loaded file (File mode), else 0.0."""
        if self._mode == "file" and self._file_sr:
            return self._file_pos / self._file_sr
        return 0.0

    def duration(self):
        return self._file_dur

    def features(self):
        with self._lock:
            f = self._features
            snap = AudioFeatures()
            snap.volume, snap.bass, snap.mid, snap.treble = f.volume, f.bass, f.mid, f.treble
            snap.kick, snap.snare, snap.hihat = f.kick, f.snare, f.hihat
            snap.beat, snap.spectrum = f.beat, f.spectrum
            return snap

    def list_outputs(self):
        """Output devices available for system-sound (loopback) capture.

        The 'System' source visualizes whatever is playing on the PC; on a
        machine with several outputs the user has to point us at the one their
        music is actually using, or the screen stays black."""
        try:
            import soundcard as sc
            return [s.name for s in sc.all_speakers()]
        except Exception:
            return []

    def set_mode(self, mode, file_path=None, device=None):
        """Switch audio source. Restarts the capture thread.

        `device` (system mode only) is the output-device name to loop back;
        None means the current default speaker."""
        self.stop()
        self._mode = mode
        self._file_path = file_path
        if device is not None:
            self._sys_device = device
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
        # window only here; sensitivity (gain) is applied AFTER the AGC below,
        # otherwise dividing by the running peak would cancel it out entirely.
        mono = mono * self._window
        mag = np.abs(np.fft.rfft(mono))
        freqs = self._freqs

        vol = float(np.sqrt(np.mean(mono ** 2)))
        bass = _band_energy(mag, freqs, 20, 250)
        mid = _band_energy(mag, freqs, 250, 2000)
        treble = _band_energy(mag, freqs, 2000, 8000)
        # drum-tuned bands
        kick = _band_energy(mag, freqs, 40, 120)       # bass-drum thump
        snare = _band_energy(mag, freqs, 180, 520)     # snare body
        hihat = _band_energy(mag, freqs, 8000, 16000)  # cymbals / hats

        e = np.array([vol, bass, mid, treble, kick, snare, hihat], dtype=np.float32)
        # --- adaptive gain (AGC): normalize each band by its own recent peak so
        # the full 0..1 range is used no matter how loud the master is. ----------
        self._peak = np.maximum(e, self._peak * self._peak_decay)
        if vol < 5e-4:                                   # near-silence: don't amp hiss
            level = np.zeros(7, dtype=np.float32)
        else:
            level = np.clip(e / (self._peak + 1e-9), 0.0, 1.0)

        # --- transient "punch": on a brick-walled master the level barely moves,
        # so emphasize each band's RISE above its own short average. Drum bands
        # lean on this so kicks/snares/hats still pulse out of the wall of sound.
        self._band_ema += (level - self._band_ema) * 0.25
        punch = np.clip((level - self._band_ema) * 3.0, 0.0, 1.0)
        raw = level.copy()
        for i in (4, 5, 6):                              # kick, snare, hihat
            raw[i] = max(level[i] * 0.4, punch[i])
        raw = np.clip(raw * self._gain, 0.0, 1.0)        # Sensitivity acts here

        # fast attack, slow release.  Drum bands (kick/snare/hihat) use a
        # snappier release so hits read as punchy hits, not sustained levels.
        for i in range(7):
            up = 0.6
            down = 0.15 if i < 4 else 0.3
            a = up if raw[i] > self._smooth[i] else down
            self._smooth[i] += (raw[i] - self._smooth[i]) * a

        # beat detection on the kick/bass (whichever is stronger) vs its rolling
        # average — rock beats live in the kick, EDM in the sub-bass.
        beat_src = max(self._smooth[1], self._smooth[4])
        self._bass_history = np.roll(self._bass_history, -1)
        self._bass_history[-1] = beat_src
        avg = float(np.mean(self._bass_history)) + 1e-4
        beat = False
        if self._beat_cooldown > 0:
            self._beat_cooldown -= 1
        elif beat_src > avg * 1.4 and beat_src > 0.15:
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

    def _pick_speaker(self, sc):
        """Resolve the chosen output device (for system loopback), tolerant of
        slightly-mismatched names; fall back to the default speaker."""
        name = self._sys_device
        if name:
            try:
                return sc.get_speaker(name)
            except Exception:
                for s in sc.all_speakers():
                    if name.lower() in s.name.lower() or s.name.lower() in name.lower():
                        return s
        return sc.default_speaker()

    def _run_soundcard(self):
        import soundcard as sc
        if self._mode == "system":
            spk = self._pick_speaker(sc)
            mic = sc.get_microphone(spk.name, include_loopback=True)
            base = f"system: {spk.name}"
        else:
            mic = sc.default_microphone()
            base = f"mic: {mic.name}"
        self.status = base
        last_status = 0.0
        with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
            while not self._stop.is_set():
                data = rec.record(numframes=BLOCK)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                self._analyze(mono.astype(np.float32))
                # silence-aware status: connected-but-quiet must not look broken
                now = time.perf_counter()
                if now - last_status > 0.4:
                    last_status = now
                    playing = float(np.abs(data).max()) > 1e-3
                    self.status = (f"{base}  ▶ playing" if playing
                                   else f"{base}  … silent (play something)")

    def _run_file(self):
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(self._file_path, dtype="float32", always_2d=True)
        self.status = f"file: {self._file_path.split('/')[-1].split(chr(92))[-1]}"
        idx = 0
        n = data.shape[0]
        self._file_sr = sr
        self._file_dur = n / sr
        self._file_pos = 0
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
            self._file_pos = idx        # expose playback position

        with sd.OutputStream(samplerate=sr, channels=data.shape[1],
                             blocksize=BLOCK, callback=callback,
                             finished_callback=ev.set):
            while not self._stop.is_set() and not ev.is_set():
                ev.wait(0.1)
