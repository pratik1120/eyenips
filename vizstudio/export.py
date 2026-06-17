"""Offline MP4 export.

Renders the active effect frame-by-frame, *in sync with an audio file*, using
the exact knobs and audio-bindings the user currently has set, then muxes the
audio in via ffmpeg. This is an offline (faster- or slower-than-real-time)
render, so it's frame-accurate and deterministic - not a screen grab.

Why a file is required: to bake audio into the MP4 we need the samples. "System"
and "Mic" sources are live and have nothing to embed, so export always works
from an audio file (the one you loaded, or one you pick at export time).
"""

import subprocess
import numpy as np

from .audio import AudioEngine, BLOCK


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


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
