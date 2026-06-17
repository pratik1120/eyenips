"""Plasma - a classic per-pixel field effect.

Included to prove the plugin system handles totally different effect *styles*:
Liquid Fractal moves particles; Plasma colors every pixel from a math field.
Same base class, same auto-UI, same audio-drive, same color/post-FX systems.

A nice template if you want to write your own shader-like effect.
"""

# Let this file find the vizstudio package whether it's run directly (IDE
# "Run" button) or imported by the app. Harmless either way.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

from vizstudio import Effect, Slider, ColorPalette


@ti.data_oriented
class Plasma(Effect):
    name = "Plasma"
    description = "Smooth flowing plasma colored by your palette."
    author = "Eyenips"

    params = [
        Slider("speed", 0.0, 4.0, default=0.8, drive=("volume", 1.0),
               help="Animation speed. (reacts to volume)"),
        Slider("scale", 1.0, 20.0, default=6.0, help="Number of waves across the screen."),
        Slider("warp", 0.0, 3.0, default=1.0, drive=("bass", 1.0),
               help="How much the waves bend each other. (reacts to bass)"),
        ColorPalette(default="plasma"),
    ]

    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette

    @ti.kernel
    def _draw(self, t: ti.f32, scale: ti.f32, warp: ti.f32):
        for i, j in self.canvas:
            u = i / self.w * scale
            v = j / self.h * scale
            val = ti.sin(u + t)
            val += ti.sin(v * 0.8 - t * 1.1)
            val += ti.sin((u + v) * 0.5 + t * 0.7)
            val += warp * ti.sin(ti.sqrt(u * u + v * v) * 1.3 - t)
            c = (val * 0.25 + 0.5)  # roughly 0..1
            c = c - ti.floor(c)
            ci = ti.cast(c * 255.0, ti.i32) % 256
            self.canvas[i, j] = self.palette[ci]

    def render(self, ctx):
        p = ctx.p
        # plasma owns every pixel, so wipe trails for a clean field
        self._draw(ctx.time * float(p["speed"]), float(p["scale"]), float(p["warp"]))


if __name__ == "__main__":
    # Convenience: running this effect file directly just launches the whole
    # studio (starting on this effect). The real entry point is ../app.py.
    import app
    app.main(prefer=Plasma.name)
