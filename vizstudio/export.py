"""Offline MP4 export.

Renders the active effect frame-by-frame, *in sync with an audio file*, using
the exact knobs and audio-bindings the user currently has set, then muxes the
audio in via ffmpeg. This is an offline (faster- or slower-than-real-time)
render, so it's frame-accurate and deterministic - not a screen grab.

Why a file is required: to bake audio into the MP4 we need the samples. "System"
and "Mic" sources are live and have nothing to embed, so export always works
from an audio file (the one you loaded, or one you pick at export time).
"""

import os
import subprocess
import tempfile

import numpy as np

from .audio import AudioEngine, AudioFeatures, BLOCK
from .media import MediaSource


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _extract_audio(path):
    """Pull a video/clip's audio to a temp mono wav (for analysis AND muxing).
    Returns the temp path, or None if there's no usable audio."""
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [_ffmpeg_exe(), "-y", "-nostdin", "-loglevel", "error",
           "-i", path, "-vn", "-ac", "1", "-ar", "44100", wav]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.getsize(wav) > 1024:
            return wav
    except Exception:
        pass
    try:
        os.remove(wav)
    except Exception:
        pass
    return None


class _VideoFeeder:
    """Feeds export video frames through MediaSource's CV (resize + flow + blob
    tracking) so the engine's live _update_media / _update_video run UNCHANGED
    during a render-through-video. We just swap it in for engine.media."""

    def __init__(self, w, h):
        self._ms = MediaSource()
        self._ms.set_target(w, h)
        self._frame = None
        self.status = "video export"

    def push(self, bgr):
        self._frame = self._ms._resize_rgb(bgr)
        if self._ms._analyze:
            try:
                self._ms._compute_analysis(bgr)
            except Exception:
                pass

    def frame(self):
        return self._frame

    def flow(self):
        return self._ms.flow()

    def blobs(self):
        return self._ms.blobs()

    def set_analyze(self, on):
        self._ms.set_analyze(on)

    def stop(self):
        pass


def export_mp4(engine, audio_path, out_path, fps=30, seconds=None, progress=None):
    """Render <seconds> (or the whole file) to out_path with audio.

    engine    : the live Engine (we reuse its canvas, palette, effect, params)
    audio_path: wav/mp3/flac/... to analyze AND embed
    progress  : optional callback(frame, total, message)
    Returns (ok: bool, message: str).
    """
    import soundfile as sf

    # H.264 / yuv420p needs even dimensions; force them just in case.
    W, H = engine.w - (engine.w % 2), engine.h - (engine.h % 2)
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    total_samples = mono.shape[0]
    duration = total_samples / sr
    if seconds:
        duration = min(duration, float(seconds))
    total_frames = max(1, int(duration * fps))

    # a dedicated analyzer so smoothing/beat state is clean and matches this sr
    analyzer = AudioEngine()
    analyzer._freqs = np.fft.rfftfreq(BLOCK, 1.0 / sr)
    analyzer.set_gain(engine.audio._gain if engine.audio else 1.0)

    ff = _ffmpeg_exe()
    cmd = [
        ff, "-y", "-nostdin", "-loglevel", "warning", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",                      # video frames from our stdin pipe
        "-ss", "0", "-t", f"{duration:.3f}", "-i", audio_path,  # audio track
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",      # moov atom up front -> plays everywhere
        "-shortest", out_path,
    ]
    # IMPORTANT: ffmpeg's log goes to a real temp file, NOT a pipe. A pipe would
    # fill its ~64KB OS buffer on a long export, block ffmpeg, and deadlock us.
    import tempfile
    log = tempfile.TemporaryFile()

    def _log_tail():
        try:
            log.seek(0)
            return log.read().decode("utf-8", "ignore")[-800:]
        except Exception:
            return ""

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=log)
    except Exception as e:
        return False, f"could not start ffmpeg: {e}"

    effect = engine.effect
    ctx = engine.ctx
    if effect:
        effect.reset()
    engine._clear_canvas()
    engine._upload_palette(force=True)

    try:
        for f in range(total_frames):
            t = f / fps
            s = int(t * sr)
            window = mono[s:s + BLOCK]
            if window.shape[0] < BLOCK:
                window = np.pad(window, (0, BLOCK - window.shape[0]))
            analyzer._analyze(window.astype(np.float32))
            feats = analyzer.features()

            ctx.time = t
            ctx.dt = 1.0 / fps
            ctx.frame = f
            ctx.audio = feats
            ctx.p = engine._resolve(feats)
            engine._upload_palette()

            if ctx.p.get("trails"):
                engine.postfx.decay(float(ctx.p.get("trail_length", 0.9)))
            else:
                engine._clear_canvas()
            engine._update_media()      # so texture/warp effects sample media
            if effect:
                effect.render(ctx)
            engine.postfx.apply(ctx.p, ctx.time)
            engine._composite_media()   # optional background blend

            img = engine.canvas.to_numpy()
            np.clip(img, 0.0, 1.0, out=img)
            # (W,H,3) GPU layout -> (H,W,3) top-left-origin video frame
            frame = np.flipud(np.transpose(img, (1, 0, 2)))
            frame = np.ascontiguousarray((frame * 255).astype(np.uint8))
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                proc.wait()
                return False, f"ffmpeg stopped early:\n{_log_tail()}"

            if progress and (f % 5 == 0 or f == total_frames - 1):
                progress(f + 1, total_frames, f"Rendering frame {f + 1}/{total_frames}")
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()   # signals end-of-stream; ffmpeg finalizes moov
            except Exception:
                pass
        proc.wait()
        if effect:
            effect.reset()      # restore clean state for the live view
        engine._clear_canvas()

    tail = _log_tail()
    log.close()
    if proc.returncode != 0:
        return False, f"ffmpeg exit {proc.returncode}:\n{tail}"
    return True, f"Saved {out_path}  ({total_frames} frames @ {fps}fps)"


def export_video(engine, video_path, out_path, seconds=None, progress=None):
    """Render-through-video: run the active effect OVER an input video, baking in
    content-aware augmentation (blobs / flow), and write a new MP4 with the
    original audio. The effect reads ctx.media / ctx.blobs / ctx.flow per frame,
    exactly like the live view — we just drive the engine from the file.

    Returns (ok, message).
    """
    import cv2

    W, H = engine.w - (engine.w % 2), engine.h - (engine.h % 2)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "could not open the video"
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if vfps < 1:
        vfps = 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (nframes / vfps) if nframes > 0 else (float(seconds) if seconds else 0.0)
    if seconds:
        dur = min(dur, float(seconds)) if dur > 0 else float(seconds)
    total_frames = int(dur * vfps) if dur > 0 else 0
    maxf = total_frames if total_frames > 0 else 10 ** 9

    # audio: extract once -> analyze per frame AND mux into the output
    wav = _extract_audio(video_path)
    mono, sr, analyzer = None, 44100, None
    if wav:
        try:
            import soundfile as sf
            data, sr = sf.read(wav, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            analyzer = AudioEngine()
            analyzer._freqs = np.fft.rfftfreq(BLOCK, 1.0 / sr)
            analyzer.set_gain(engine.audio._gain if engine.audio else 1.0)
        except Exception:
            mono = None

    ff = _ffmpeg_exe()
    cmd = [ff, "-y", "-nostdin", "-loglevel", "warning", "-nostats",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{vfps:g}", "-i", "-"]
    if wav:
        cmd += ["-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "veryfast", "-movflags", "+faststart", out_path]

    log = tempfile.TemporaryFile()

    def _log_tail():
        try:
            log.seek(0)
            return log.read().decode("utf-8", "ignore")[-800:]
        except Exception:
            return ""

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=log)
    except Exception as e:
        cap.release()
        return False, f"could not start ffmpeg: {e}"

    feeder = _VideoFeeder(engine.w, engine.h)
    feeder.set_analyze(True)
    orig_media = engine.media
    engine.media = feeder                    # drive the engine from the file
    effect = engine.effect
    ctx = engine.ctx
    if effect:
        effect.reset()
    engine._clear_canvas()
    engine._upload_palette(force=True)
    silent = AudioFeatures()
    f = 0
    try:
        while f < maxf:
            ok, bgr = cap.read()
            if not ok:
                break
            feeder.push(bgr)
            t = f / vfps
            if mono is not None:
                s = int(t * sr)
                window = mono[s:s + BLOCK]
                if window.shape[0] < BLOCK:
                    window = np.pad(window, (0, BLOCK - window.shape[0]))
                analyzer._analyze(window.astype(np.float32))
                feats = analyzer.features()
            else:
                feats = silent

            ctx.time = t
            ctx.dt = 1.0 / vfps
            ctx.frame = f
            ctx.audio = feats
            ctx.p = engine._resolve(feats)
            engine._upload_palette()

            if ctx.p.get("trails"):
                engine.postfx.decay(float(ctx.p.get("trail_length", 0.9)))
            else:
                engine._clear_canvas()
            engine._update_media()           # upload this frame -> ctx.media
            engine._update_video()           # flow + blobs from this frame
            if effect:
                effect.render(ctx)
            engine.postfx.apply(ctx.p, ctx.time)
            engine._composite_media()        # honors media_blend if the user set it

            img = engine.canvas.to_numpy()
            np.clip(img, 0.0, 1.0, out=img)
            frame = np.ascontiguousarray(
                (np.flipud(np.transpose(img, (1, 0, 2))) * 255).astype(np.uint8))
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                proc.wait()
                return False, f"ffmpeg stopped early:\n{_log_tail()}"
            f += 1
            if progress and (f % 5 == 0 or f == total_frames):
                denom = total_frames if total_frames > 0 else f
                progress(f, max(1, denom), f"Rendering frame {f}"
                         + (f"/{total_frames}" if total_frames > 0 else ""))
    finally:
        engine.media = orig_media            # restore the live source
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.wait()
        if effect:
            effect.reset()
        engine._clear_canvas()
        if wav:
            try:
                os.remove(wav)
            except Exception:
                pass

    tail = _log_tail()
    log.close()
    if proc.returncode != 0:
        return False, f"ffmpeg exit {proc.returncode}:\n{tail}"
    return True, f"Saved {out_path}  ({f} frames @ {vfps:g}fps, from your video)"
