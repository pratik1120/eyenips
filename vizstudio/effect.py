"""The Effect base class - the one thing a plugin author subclasses.

A minimal effect looks like:

    import taichi as ti
    from vizstudio import Effect, Slider, ColorPalette

    class MyEffect(Effect):
        name = "My Effect"
        params = [
            Slider("speed", 0.1, 5.0, default=1.0),
            ColorPalette(),
        ]

        def setup(self, ctx):
            ...   # allocate Taichi fields once

        def render(self, ctx):
            ...   # draw into ctx.canvas each frame

Read knob values from `ctx.p["speed"]` (already resolved, incl. audio drive).
Read audio from `ctx.audio.bass` etc. Sample colors from `ctx.palette` (a
256x3 Taichi field) inside your kernels.
"""

from __future__ import annotations


class Context:
    """Everything an effect needs each frame. Created and owned by the engine."""

    def __init__(self, width, height, canvas, palette):
        self.width = width
        self.height = height
        self.canvas = canvas        # ti.Vector.field(3) shape (W, H) - draw here
        self.palette = palette      # ti.Vector.field(3) shape (256,) - color LUT
        self.time = 0.0             # seconds since start (scaled by nothing)
        self.dt = 0.0               # seconds since last frame
        self.frame = 0
        self.audio = None           # AudioFeatures (volume/bass/mid/treble/beat)
        self.p = {}                 # resolved param values {name: value}
        # media input (camera/image/video) - effects can SAMPLE these:
        self.media = None           # ti.Vector.field(3) shape (W,H): current frame
        self.media_motion = None    # ti.field(f32) shape (W,H): camera motion 0..1
        self.has_media = False      # is a media source currently producing frames?


class Effect:
    name = "Untitled Effect"
    description = ""
    author = ""
    params = []  # list of vizstudio Param objects

    def setup(self, ctx):
        """Called once when the effect becomes active. Allocate fields here."""

    def render(self, ctx):
        """Called every frame. Draw into ctx.canvas."""
        raise NotImplementedError

    def reset(self):
        """Optional: called when the user hits Reset. Re-seed state."""
