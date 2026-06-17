"""Text templates the Create Effect window writes into effects/*.py.

Kept here (not inline in the UI) so they're easy to read and tweak.
"""

# The live expression effect's display name (the Expression tab edits it).
EXPR_EFFECT_NAME = "Custom (expression)"


def _slug(name):
    s = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
    return s or "my_effect"


def expression_file(name, bright, hue, palette="plasma", source="paint"):
    """A standalone effect file built from two formulas (Expression/Build Save)."""
    return (
        '"""An effect you made in the Create Effect window."""\n'
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "\n"
        "from vizstudio.exprfx import ExpressionEffectBase\n"
        "\n"
        "\n"
        "class CustomEffect(ExpressionEffectBase):\n"
        f"    name = {name!r}\n"
        f"    BRIGHT = {bright!r}\n"
        f"    HUE = {hue!r}\n"
        f"    PALETTE = {palette!r}\n"
        f"    SOURCE = {source!r}\n"
    )


# The scaffold the Code tab starts from: a complete, working effect.
CODE_TEMPLATE = '''\
"""My custom effect. Edit the math below, press Reload to preview live,
then Save. Everything (knobs, audio, colors, post-FX) is already wired up -
you only write the drawing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
from vizstudio import Effect, Slider, ColorPalette


@ti.data_oriented
class MyEffect(Effect):
    name = "My Effect"          # <- shows up in the effect dropdown

    params = [                  # <- these become sliders/pickers automatically
        Slider("speed", 0.0, 4.0, default=1.0, drive=("volume", 1.0)),
        Slider("scale", 1.0, 20.0, default=6.0),
        ColorPalette(default="plasma"),
    ]

    def setup(self, ctx):       # runs once - grab what you need
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette

    @ti.kernel
    def _draw(self, t: ti.f32, scale: ti.f32):
        for i, j in self.canvas:
            x = i / self.w
            y = j / self.h
            # --- your math here ---
            v = ti.sin(x * scale + t) + ti.sin(y * scale - t)
            c = v * 0.25 + 0.5
            c = c - ti.floor(c)
            ci = ti.cast(c * 255.0, ti.i32) % 256
            self.canvas[i, j] = self.palette[ci]

    def render(self, ctx):      # runs every frame
        self._draw(ctx.time * float(ctx.p["speed"]), float(ctx.p["scale"]))
'''
