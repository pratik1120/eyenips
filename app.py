"""Eyenips - entry point.

  python app.py

Initializes the GPU, discovers every effect in effects/, opens the control
panel (its own thread) and the render window (main thread).
"""

import os
import sys

# When frozen, vizstudio ships as LOOSE source next to the exe (not inside the
# PyInstaller archive) so Taichi can read each @ti.kernel's source at runtime
# (inspect.getsource fails on archived modules). Put the exe dir on sys.path so
# `import vizstudio` finds that loose copy — must run before importing vizstudio.
if getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(sys.executable))

import taichi as ti

from vizstudio.engine import Engine
from vizstudio.audio import AudioEngine
from vizstudio.media import MediaSource
from vizstudio.registry import discover
from vizstudio.ui import ControlPanel
from vizstudio import paths

# Effects live as loose, updatable files next to the exe (or in the project in
# dev) — resolved frozen-safely so the packaged app finds them too.
EFFECTS_DIR = paths.effects_dir()

# The canvas is now drawn *inside* the app window, so keep it a size that
# converts to an image smoothly each frame.
WIDTH, HEIGHT = 1024, 640


def _init_taichi():
    """Bring up the GPU backend, but never let a machine without a usable GPU
    (no CUDA/Vulkan, old drivers) crash the app — fall back to CPU."""
    try:
        ti.init(arch=ti.gpu)
        return "gpu"
    except Exception as e:
        paths.safe_print(f"[taichi] GPU unavailable ({e}); falling back to CPU.")
        ti.init(arch=ti.cpu)
        return "cpu"


def main(prefer=None):
    """prefer: optional effect name to start on (else the first discovered)."""
    backend = _init_taichi()

    effect_classes, errors = discover(EFFECTS_DIR)
    for fn, msg in errors:
        paths.safe_print(f"[skip] {fn} failed to load:\n{msg}")
    if not effect_classes:
        paths.safe_print("No effects found in effects/. Add an Effect subclass there.")
        return

    # pick the start effect: the requested one, else the first *visual* effect
    # (never the black "Blank (for shapes)" base, or the preview looks empty).
    start = next((c for c in effect_classes if c.name == prefer), None)
    if start is None:
        start = next((c for c in effect_classes
                      if not c.name.startswith("Blank")), effect_classes[0])

    audio = AudioEngine()
    media = MediaSource()
    engine = Engine(WIDTH, HEIGHT, audio=audio, media=media)
    engine.set_effect_catalog(effect_classes)   # shapes can show any effect by name
    engine.set_effect(start())

    # Start reacting to whatever's playing on the PC immediately (WinAmp-style).
    # Falls back to silent if loopback capture isn't available.
    audio.set_mode("system")

    paths.safe_print(f"=== Eyenips === (render backend: {backend})")
    paths.safe_print(f"Effects: {', '.join(c.name for c in effect_classes)}")
    paths.safe_print("Everything lives in one window: preview left, controls right.")

    # Single-threaded: Tkinter MUST run on the main thread. We build the one
    # window here and let the render loop draw frames into it + pump the UI.
    panel = ControlPanel(engine, effect_classes, effects_dir=EFFECTS_DIR)
    panel.build()
    try:
        engine.run(on_frame=panel.pump, display=panel.show_frame)
    finally:
        media.stop()   # release the camera / video file
        panel.close()


if __name__ == "__main__":
    main()
