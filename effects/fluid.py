"""Liquid Light — an interactive fluid you stir with your mouse, to the music.

This is the "real-deal interactive art" piece: a genuine fluid simulation
(Navier–Stokes, stable-fluids: semi-Lagrangian advection + Jacobi pressure
projection) running on the GPU. It's not a canned effect — it's smoke / ink /
fire you actually push around.

  • **Your mouse is a hand in the fluid.** Move it to stir; hold the button to
    paint glowing dye and shove the flow where you drag. Drag fast = hard shove.
  • **The music lives in it too.** Every beat puffs a plume of colored dye that
    rises and curls; bass pushes the throttle; the drop detonates the whole tank.
  • Smoke **rises** (buoyancy), swirls, and slowly fades — so it always looks alive
    even before you touch it.

Reliable (just the mouse, no webcam), mesmerizing, and endlessly different. Pure
physics, no generative AI. Colors come from your chosen palette.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, ColorPalette

GRID_SCALE = 4          # sim grid = canvas / this (fluid sims are heavy)


@ti.data_oriented
class Fluid(Effect):
    name = "Liquid Light"
    description = ("A real fluid you stir with your mouse while the music puffs "
                   "glowing dye into it. Interactive smoke / ink. No AI.")
    author = "Eyenips"

    params = [
        Slider("stir_force", 0.0, 1.0, default=0.35, audio=False,
               help="How hard your mouse shoves the fluid when you drag."),
        Slider("dye_amount", 0.0, 4.0, default=2.0, audio=False,
               help="How much glowing ink your mouse paints (hold the button)."),
        Slider("music_push", 0.0, 1.0, default=0.5, drive=("bass", 0.6),
               help="How forcefully the music puffs dye in. Bass drives it."),
        Slider("beat_puff", 0.0, 1.0, default=0.6, audio=False,
               help="Size of the colored plume each beat injects."),
        Slider("detonate", 0.0, 1.0, default=0.0, drive=("drop", 1.0),
               help="The drop blows the whole tank outward. Driven by Drop — crank "
                    "it by hand to blast on demand."),
        Slider("swirl", 0.0, 1.0, default=0.35, drive=("treble", 0.4),
               help="Curl / turbulence — amplifies the rolling swirls so it billows "
                    "like real smoke. Treble feeds it."),
        Slider("buoyancy", 0.0, 1.0, default=0.45, audio=False,
               help="How strongly the smoke RISES — 0 = weightless ink."),
        Slider("fade", 0.95, 0.999, default=0.99, audio=False,
               help="How long the smoke lingers before dissolving."),
        IntSlider("quality", 8, 40, default=24, audio=False,
                  help="Incompressibility solve iterations — higher = crisper, "
                       "more billowing flow (a little heavier)."),
        Slider("glow", 0.3, 3.0, default=1.4, drive=("volume", 0.4),
               help="Brightness of the glowing fluid."),
        ColorPalette(default="fire"),
    ]

    # ---- lifecycle ------------------------------------------------------
    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.gw = max(16, self.w // GRID_SCALE)
        self.gh = max(16, self.h // GRID_SCALE)
        self.vel = ti.Vector.field(2, ti.f32, shape=(self.gw, self.gh))
        self.vel_tmp = ti.Vector.field(2, ti.f32, shape=(self.gw, self.gh))
        self.dye = ti.Vector.field(3, ti.f32, shape=(self.gw, self.gh))
        self.dye_tmp = ti.Vector.field(3, ti.f32, shape=(self.gw, self.gh))
        self.pr = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.pr_tmp = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.div = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.curl = ti.field(ti.f32, shape=(self.gw, self.gh))
        self._cphase = 0.0
        self._clear()

    def reset(self):
        self._clear()

    @ti.kernel
    def _clear(self):
        for i, j in self.vel:
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            self.dye[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.pr[i, j] = 0.0

    # ---- fluid primitives ----------------------------------------------
    @ti.func
    def _samp(self, f: ti.template(), x, y):
        x = ti.min(ti.max(x, 0.0), self.gw - 1.0)
        y = ti.min(ti.max(y, 0.0), self.gh - 1.0)
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.gw - 1, x0 + 1)
        y1 = ti.min(self.gh - 1, y0 + 1)
        ax = x - x0
        ay = y - y0
        return (f[x0, y0] * (1 - ax) * (1 - ay) + f[x1, y0] * ax * (1 - ay)
                + f[x0, y1] * (1 - ax) * ay + f[x1, y1] * ax * ay)

    @ti.kernel
    def _advect(self, src: ti.template(), dst: ti.template(), diss: ti.f32):
        for i, j in src:
            v = self.vel[i, j]
            dst[i, j] = self._samp(src, i - v[0], j - v[1]) * diss

    @ti.kernel
    def _curl(self):
        for i, j in self.curl:
            r = self.vel[ti.min(i + 1, self.gw - 1), j][1]
            l = self.vel[ti.max(i - 1, 0), j][1]
            t = self.vel[i, ti.min(j + 1, self.gh - 1)][0]
            b = self.vel[i, ti.max(j - 1, 0)][0]
            self.curl[i, j] = 0.5 * ((r - l) - (t - b))   # dVy/dx - dVx/dy

    @ti.kernel
    def _vort_confine(self, eps: ti.f32):
        for i, j in self.vel:
            gx = 0.5 * (ti.abs(self.curl[ti.min(i + 1, self.gw - 1), j])
                        - ti.abs(self.curl[ti.max(i - 1, 0), j]))
            gy = 0.5 * (ti.abs(self.curl[i, ti.min(j + 1, self.gh - 1)])
                        - ti.abs(self.curl[i, ti.max(j - 1, 0)]))
            ln = ti.sqrt(gx * gx + gy * gy) + 1e-5
            nx = gx / ln
            ny = gy / ln
            w = self.curl[i, j]
            self.vel[i, j] += ti.Vector([ny * w, -nx * w]) * eps   # push along the swirl

    @ti.kernel
    def _clamp_vel(self, vmax: ti.f32):
        for i, j in self.vel:
            v = self.vel[i, j]
            s = ti.sqrt(v[0] * v[0] + v[1] * v[1])
            if s > vmax:
                self.vel[i, j] = v * (vmax / s)

    @ti.kernel
    def _buoyancy(self, amt: ti.f32):
        for i, j in self.vel:
            d = self.dye[i, j]
            self.vel[i, j][1] += (d[0] + d[1] + d[2]) * 0.333 * amt

    @ti.kernel
    def _divergence(self):
        for i, j in self.div:
            l = self.vel[ti.max(i - 1, 0), j][0]
            r = self.vel[ti.min(i + 1, self.gw - 1), j][0]
            b = self.vel[i, ti.max(j - 1, 0)][1]
            t = self.vel[i, ti.min(j + 1, self.gh - 1)][1]
            self.div[i, j] = 0.5 * ((r - l) + (t - b))

    @ti.kernel
    def _pjacobi(self, src: ti.template(), dst: ti.template()):
        for i, j in dst:
            l = src[ti.max(i - 1, 0), j]
            r = src[ti.min(i + 1, self.gw - 1), j]
            b = src[i, ti.max(j - 1, 0)]
            t = src[i, ti.min(j + 1, self.gh - 1)]
            dst[i, j] = (l + r + b + t - self.div[i, j]) * 0.25

    @ti.kernel
    def _subgrad(self):
        for i, j in self.vel:
            l = self.pr[ti.max(i - 1, 0), j]
            r = self.pr[ti.min(i + 1, self.gw - 1), j]
            b = self.pr[i, ti.max(j - 1, 0)]
            t = self.pr[i, ti.min(j + 1, self.gh - 1)]
            self.vel[i, j][0] -= 0.5 * (r - l)
            self.vel[i, j][1] -= 0.5 * (t - b)
            if i == 0 or i == self.gw - 1 or j == 0 or j == self.gh - 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])   # solid walls

    @ti.kernel
    def _splat(self, cx: ti.f32, cy: ti.f32, r: ti.f32, fx: ti.f32, fy: ti.f32,
               dr: ti.f32, dg: ti.f32, db: ti.f32, fmul: ti.f32, dmul: ti.f32):
        r2 = r * r
        force = ti.Vector([fx, fy])
        col = ti.Vector([dr, dg, db])
        for i, j in self.vel:
            dx = ti.cast(i, ti.f32) - cx
            dy = ti.cast(j, ti.f32) - cy
            d2 = dx * dx + dy * dy
            if d2 < r2 * 5.0:
                w = ti.exp(-d2 / r2)
                self.vel[i, j] += force * (w * fmul)
                nd = self.dye[i, j] + col * (w * dmul)
                self.dye[i, j] = ti.min(nd, ti.Vector([4.0, 4.0, 4.0]))

    @ti.kernel
    def _detonate(self, cx: ti.f32, cy: ti.f32, strength: ti.f32):
        for i, j in self.vel:
            dx = ti.cast(i, ti.f32) - cx
            dy = ti.cast(j, ti.f32) - cy
            d = ti.sqrt(dx * dx + dy * dy) + 1e-3
            self.vel[i, j] += ti.Vector([dx / d, dy / d]) * strength

    @ti.kernel
    def _render(self, gain: ti.f32):
        gwf = self.gw - 1.0
        ghf = self.gh - 1.0
        for i, j in self.canvas:
            c = self._samp(self.dye, ti.cast(i, ti.f32) * gwf / self.w,
                           ti.cast(j, ti.f32) * ghf / self.h) * gain
            # filmic-ish rolloff so dense smoke glows instead of clipping flat
            self.canvas[i, j] = ti.Vector([1.0 - ti.exp(-c[0]),
                                           1.0 - ti.exp(-c[1]),
                                           1.0 - ti.exp(-c[2])])

    # ---- helpers --------------------------------------------------------
    def _next_color(self, speed=0.013):
        """Cycle a color out of the user's palette for the next injection."""
        self._cphase = (self._cphase + speed) % 1.0
        ci = int(self._cphase * 255) % 256
        c = self.palette[ci]
        return float(c[0]), float(c[1]), float(c[2])

    # ---- per frame ------------------------------------------------------
    def render(self, ctx):
        p = ctx.p
        a = ctx.audio
        gw, gh = self.gw, self.gh

        # --- inject from the MOUSE ---
        pt = getattr(ctx, "pointer", None)
        if pt is not None and pt.active:
            cx, cy = pt.x * gw, pt.y * gh
            sf = float(p["stir_force"]) * gw
            fx, fy = pt.dx * sf, pt.dy * sf
            if pt.down:
                r, g, b = self._next_color()
                self._splat(cx, cy, gh * 0.05, fx, fy, r, g, b,
                            1.0, float(p["dye_amount"]))
            else:                                   # hover = gentle stir, no ink
                self._splat(cx, cy, gh * 0.05, fx * 0.5, fy * 0.5, 0, 0, 0, 0.7, 0.0)

        # --- inject from the MUSIC ---
        push = float(p["music_push"])
        if a is not None and a.beat:
            r, g, b = self._next_color(0.07)
            bx = random.uniform(0.2, 0.8) * gw
            self._splat(bx, gh * 0.12, gh * (0.05 + 0.08 * float(p["beat_puff"])),
                        0.0, (3.0 + 12.0 * push), r, g, b, 1.0, 1.6)
        elif a is not None and push > 0.001:        # steady gentle feed from the music
            r, g, b = self._next_color(0.02)
            self._splat(random.uniform(0.3, 0.7) * gw, gh * 0.1, gh * 0.04,
                        0.0, 4.0 * push * float(a.bass), r, g, b, 1.0, 0.5 * push)

        det = float(p["detonate"])
        if det > 0.5:
            self._detonate(gw * 0.5, gh * 0.45, det * 6.0)

        # --- simulate: swirl -> buoyancy -> advect vel -> project -> advect dye ---
        sw = float(p["swirl"])
        if sw > 0.001:
            self._curl()
            self._vort_confine(sw * 0.35)
        self._buoyancy(float(p["buoyancy"]) * 0.6)
        self._clamp_vel(self.gh * 0.5)            # stability cap
        self._advect(self.vel, self.vel_tmp, 0.999)
        self.vel, self.vel_tmp = self.vel_tmp, self.vel
        self._divergence()
        self.pr.fill(0.0)
        iters = int(p["quality"])
        for _ in range(iters):
            self._pjacobi(self.pr, self.pr_tmp)
            self.pr, self.pr_tmp = self.pr_tmp, self.pr
        self._subgrad()
        self._advect(self.dye, self.dye_tmp, float(p["fade"]))
        self.dye, self.dye_tmp = self.dye_tmp, self.dye

        self._render(float(p["glow"]))


if __name__ == "__main__":
    import app
    app.main(prefer=Fluid.name)
