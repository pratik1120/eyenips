"""Liquid Fractal - a GPU flow-field particle effect.

This is the original demo, rebuilt as a plugin so you can see the shape every
effect takes:
  * subclass Effect
  * declare params (these auto-build the control panel)
  * allocate Taichi fields in setup()
  * draw into ctx.canvas in render()

Read knobs from ctx.p[...]; sample colors from ctx.palette (the 256-entry LUT
the user chose). Any numeric knob can be made audio-reactive from the UI.
"""

# Let this file find the vizstudio package whether it's run directly (IDE
# "Run" button) or imported by the app. Harmless either way.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, ColorPalette

N = 280_000  # particle count (fixed once allocated)


@ti.data_oriented
class LiquidFractal(Effect):
    name = "Liquid Fractal"
    description = "Half a million particles flowing through a fractal field."
    author = "Eyenips"

    params = [
        # drive=(...) gives each knob a default audio binding so the effect
        # reacts to music out of the box. The user can change these in the UI.
        Slider("speed", 0.1, 5.0, default=0.8, drive=("bass", 0.9),
               help="Overall motion speed. (reacts to bass)"),
        IntSlider("particle_size", 1, 8, default=2, drive=("beat", 0.8),
                  help="Splat size of each particle. (pulses on the beat)"),
        Slider("swirl", 0.0, 2.0, default=0.6, drive=("treble", 0.7),
               help="Spiral / vortex strength. (reacts to treble)"),
        Slider("flow_scale", 0.002, 0.03, default=0.01, audio=False,
               help="Zoom of the flow field. Low = big smooth swirls."),
        Slider("randomness", 0.0, 1.0, default=0.2, help="Jitter / chaos."),
        Slider("intensity", 0.1, 2.0, default=0.5, drive=("volume", 1.0),
               help="Brightness of each particle. (reacts to volume)"),
        ColorPalette(),
    ]

    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.px = ti.field(ti.f32, N)
        self.py = ti.field(ti.f32, N)
        self.vx = ti.field(ti.f32, N)
        self.vy = ti.field(ti.f32, N)
        self.col = ti.field(ti.f32, N)
        self._seed()

    @ti.kernel
    def _seed(self):
        for i in self.px:
            self.px[i] = ti.random() * self.w
            self.py[i] = ti.random() * self.h
            self.vx[i] = 0.0
            self.vy[i] = 0.0
            self.col[i] = ti.random()

    def reset(self):
        self._seed()

    @ti.kernel
    def _step(self, t: ti.f32, speed: ti.f32, swirl: ti.f32, scale: ti.f32,
              rnd: ti.f32, size: ti.i32, intensity: ti.f32):
        cx = self.w * 0.5
        cy = self.h * 0.5
        for i in self.px:
            x = self.px[i]
            y = self.py[i]

            fx = ti.sin(x * scale + t * 0.5) * ti.cos(y * scale * 0.6) + ti.sin(t * 0.3) * 0.5
            fy = ti.cos(x * scale * 0.8 - t * 0.4) * ti.sin(y * scale) + ti.cos(t * 0.25) * 0.5

            dx = x - cx
            dy = y - cy
            dist = ti.sqrt(dx * dx + dy * dy) + 1e-3
            ang = ti.atan2(dy, dx)
            spiral = ti.sin(dist * 0.002 - t * 0.2) * swirl
            sx = (-dy / dist) * spiral
            sy = (dx / dist) * spiral

            jx = (ti.random() - 0.5) * rnd
            jy = (ti.random() - 0.5) * rnd

            self.vx[i] = self.vx[i] * 0.92 + (fx + sx + jx) * 0.3
            self.vy[i] = self.vy[i] * 0.92 + (fy + sy + jy) * 0.3
            self.px[i] = self.px[i] + self.vx[i] * speed
            self.py[i] = self.py[i] + self.vy[i] * speed

            # wrap
            if self.px[i] < 0: self.px[i] += self.w
            if self.px[i] >= self.w: self.px[i] -= self.w
            if self.py[i] < 0: self.py[i] += self.h
            if self.py[i] >= self.h: self.py[i] -= self.h

            self.col[i] = self.col[i] + 0.003
            if self.col[i] > 1.0:
                self.col[i] -= 1.0

            # splat with palette color
            ci = ti.cast(self.col[i] * 255.0, ti.i32) % 256
            rgb = self.palette[ci] * intensity
            xi = ti.cast(self.px[i], ti.i32)
            yi = ti.cast(self.py[i], ti.i32)
            for ox in range(size):
                for oy in range(size):
                    xx = xi + ox
                    yy = yi + oy
                    if 0 <= xx < self.w and 0 <= yy < self.h:
                        self.canvas[xx, yy] += rgb

    def render(self, ctx):
        p = ctx.p
        self._step(ctx.time, float(p["speed"]), float(p["swirl"]),
                   float(p["flow_scale"]), float(p["randomness"]),
                   int(p["particle_size"]), float(p["intensity"]))


if __name__ == "__main__":
    # Convenience: running this effect file directly just launches the whole
    # studio (starting on this effect). The real entry point is ../app.py.
    import app
    app.main(prefer=LiquidFractal.name)
