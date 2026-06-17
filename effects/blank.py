"""Blank - a black base, so the Shapes overlay can be used on its own.

Shapes are a LAYER that interacts with whatever effect is running (see the
✨ Shapes panel). Pick this when you want *just* shapes on a clean background:
'Glow / Fill / Outline / Ripples' shapes paint on the black, and the
interaction modes (Reveal / Hide / Warp / Tint) have a calm canvas to act on.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

from vizstudio import Effect, ColorPalette


@ti.data_oriented
class Blank(Effect):
    name = "Blank (for shapes)"
    description = "A black canvas — pick this to use the Shapes overlay on its own."

    params = [ColorPalette(default="rainbow")]

    def setup(self, ctx):
        self.canvas = ctx.canvas

    def render(self, ctx):
        self.canvas.fill(0)
