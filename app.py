"""Eyenips - entry point.

  python app.py

Initializes the GPU, discovers every effect in effects/, opens the control
panel (its own thread) and the render window (main thread).
"""

import os
import sys

import taichi as ti

from vizstudio.engine import Engine
from vizstudio.audio import AudioEngine
from vizstudio.media import MediaSource
from vizstudio.registry import discover
from vizstudio.ui import ControlPanel

HERE = os.path.dirname(os.path.abspath(__file__))
EFFECTS_DIR = os.path.join(HERE, "effects")

# The canvas is now drawn *inside* the app window, so keep it a size that
# converts to an image smoothly each frame.
WIDTH, HEIGHT = 1024, 640


def main(prefer=None):
    """prefer: optional effect name to start on (else the first discovered)."""
    ti.init(arch=ti.gpu)

    effect_classes, errors = discover(EFFECTS_DIR)
    for fn, msg in errors:
        print(f"[skip] {fn} failed to load:\n{msg}", file=sys.stderr)
    if not effect_classes:
        print("No effects found in effects/. Add an Effect subclass there.")
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

    print("=== Eyenips ===")
    print(f"Effects: {', '.join(c.name for c in effect_classes)}")
    print("Everything lives in one window: preview on the left, controls on the right.")

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
