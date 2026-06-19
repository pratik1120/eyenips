"""The Living Visual — music as a living organism (emergent, no AI).

The signature effect: not a pattern that *reacts* to sound, but a creature that
*lives* on it. Under the hood it's a Gray–Scott reaction–diffusion system — two
chemicals (U, V) diffusing and reacting on a GPU grid — but the beauty comes from
how it's grown and lit:

  • a randomized **genome** per birth gives every generation different feed/kill
    *gradients across space*, so spots, mazes, cells and worms COEXIST in one
    frame — no two organisms are even the same species,
  • it's rendered as a 3-D **height-field**: real surface normals, diffuse light
    and a moving specular highlight, so it glistens like living tissue / liquid
    metal instead of flat coral,
  • **iridescent** color — the hue shifts with the direction of growth (oil-slick
    / soap-film shimmer) on top of your palette,
  • it **metamorphoses on the drop** — re-rolling its genome into a new species.

Music is its metabolism: the **beat** is a heartbeat (seeds new growth), **bass**
feeds it, **highs** agitate it into filigree, and **Life** (driven by the Music
Director's Intensity) is its lifeforce — born sparse, blooming at the peak, dying
in the quiet. Loud moments leave permanent **scars** it carries to the end.

Every play is a one-of-a-kind life that can never be reproduced. Pure simulation —
no generative AI.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, ColorPalette

# Gray–Scott diffusion rates (classic stable regime, 9-point Laplacian, dt = 1).
DU = 0.16
DV = 0.08
GRID_SCALE = 2          # sim grid = canvas / this (bolder forms + speed)
TWO_PI = 6.28318


@ti.data_oriented
class LivingVisual(Effect):
    name = "Living Organism"
    description = ("A reaction–diffusion lifeform: lit like 3-D tissue, a new "
                   "randomized species every birth, metamorphosing on the drop. "
                   "Emergent, never the same twice. No AI.")
    author = "Eyenips"

    params = [
        IntSlider("organism_speed", 1, 16, default=8, drive=("intensity", 0.5),
                  help="How fast it lives (sim steps/frame). Quickens with energy."),
        Slider("feed", 0.012, 0.060, default=0.036, drive=("bass", 0.4),
               help="Appetite (feed rate). Bass feeds it — more bass, more growth."),
        Slider("kill", 0.045, 0.070, default=0.062, audio=False,
               help="What dissolves it. With feed + Diversity this sets the FORMS: "
                    "coral, cells, mitosis, mazes, worms."),
        Slider("diversity", 0.0, 1.0, default=0.6, audio=False,
               help="How many different patterns coexist in ONE frame — feed/kill "
                    "vary across space. High = a whole ecosystem at once."),
        Slider("life", 0.0, 1.0, default=0.30, drive=("intensity", 0.7),
               help="Lifeforce (driven by Intensity): born → blooms at the peak → "
                    "dies in the quiet. Set the base to 0 for a true birth→death arc."),
        Slider("seed_beat", 0.0, 1.0, default=0.55, audio=False,
               help="Heartbeat: how much new growth each beat seeds."),
        Slider("metamorph", 0.0, 1.0, default=0.0, drive=("drop", 1.0),
               help="Metamorphosis. Driven by Drop, so it transforms into a NEW "
                    "species on the drop. Crank it by hand to mutate on demand."),
        Slider("agitate", 0.0, 1.0, default=0.30, drive=("hihat", 0.6),
               help="Fine detail / shimmer from the highs (hi-hats, cymbals)."),
        Slider("memory", 0.0, 1.0, default=0.40, audio=False,
               help="How deeply loud moments SCAR it — permanent marks it carries."),
        Slider("relief", 0.0, 1.0, default=0.60, audio=False,
               help="3-D depth. Lights the organism as a height-field — flat coral "
                    "becomes glistening sculpted tissue."),
        Slider("shine", 0.0, 1.0, default=0.35, audio=False,
               help="Wet/metallic specular highlight that slides across it."),
        Slider("iridescence", 0.0, 1.0, default=0.50, audio=False,
               help="Oil-slick shimmer: hue shifts with the direction of growth."),
        Slider("glow", 0.3, 3.0, default=1.5, drive=("volume", 0.5),
               help="Brightness of the living tissue."),
        ColorPalette(default="fire"),
    ]

    # ---- lifecycle ------------------------------------------------------
    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.gw = max(8, self.w // GRID_SCALE)
        self.gh = max(8, self.h // GRID_SCALE)
        self.U = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.V = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.Un = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.Vn = ti.field(ti.f32, shape=(self.gw, self.gh))
        self.scar = ti.field(ti.f32, shape=(self.gw, self.gh))
        self._meta_cool = 0
        self._birth()

    def _roll_genome(self):
        """The hidden DNA — randomized so every generation is a new species."""
        self.gen_fx = random.uniform(-1, 1)      # feed-gradient direction
        self.gen_fy = random.uniform(-1, 1)
        self.gen_kx = random.uniform(-1, 1)      # kill-gradient direction
        self.gen_ky = random.uniform(-1, 1)
        self.gen_light = random.uniform(0, TWO_PI)
        self.gen_hue = random.random()           # palette rotation
        self.gen_irid = random.uniform(0.3, 1.0)
        self.gen_style = random.randint(0, 2)    # embryo shape

    def _birth(self):
        self._roll_genome()
        self._init()
        self._seed_embryo()

    def _seed_embryo(self):
        """The starting germ cells — shape depends on the genome."""
        gw, gh = self.gw, self.gh
        if self.gen_style == 0:                  # scattered blobs
            for _ in range(6):
                self._seed(random.random() * gw, random.random() * gh, gh * 0.05, 0.5)
        elif self.gen_style == 1:                # a ring (radial symmetry)
            for a in range(12):
                ang = a / 12.0 * TWO_PI
                self._seed(gw * 0.5 + math.cos(ang) * gw * 0.3,
                           gh * 0.5 + math.sin(ang) * gh * 0.3, gh * 0.04, 0.5)
        else:                                    # dense dust (colonies)
            for _ in range(14):
                self._seed(random.random() * gw, random.random() * gh, gh * 0.03, 0.5)

    def reset(self):
        self._birth()

    @ti.kernel
    def _init(self):
        for i, j in self.U:
            self.U[i, j] = 1.0
            self.V[i, j] = 0.0
            self.scar[i, j] = 0.0

    @ti.kernel
    def _seed(self, cx: ti.f32, cy: ti.f32, r: ti.f32, amt: ti.f32):
        r2 = r * r
        for i, j in self.V:
            dx = ti.cast(i, ti.f32) - cx
            dy = ti.cast(j, ti.f32) - cy
            if dx * dx + dy * dy < r2:
                self.V[i, j] = ti.min(1.0, self.V[i, j] + amt)
                self.U[i, j] = ti.max(0.0, self.U[i, j] - amt)

    @ti.kernel
    def _agitate(self, amt: ti.f32):
        for i, j in self.V:
            if ti.random() < 0.02:
                self.V[i, j] = ti.min(1.0, self.V[i, j] + ti.random() * amt)

    @ti.kernel
    def _scar_add(self, amt: ti.f32):
        for i, j in self.scar:
            self.scar[i, j] = ti.min(0.05, self.scar[i, j] + self.V[i, j] * amt)

    @ti.kernel
    def _step(self, U: ti.template(), V: ti.template(),
              Un: ti.template(), Vn: ti.template(),
              F0: ti.f32, k0: ti.f32, Fx: ti.f32, Fy: ti.f32,
              kx: ti.f32, ky: ti.f32, fbias: ti.f32):
        gw, gh = self.gw, self.gh
        for i, j in U:
            im = (i - 1 + gw) % gw
            ip = (i + 1) % gw
            jm = (j - 1 + gh) % gh
            jp = (j + 1) % gh
            lu = ((U[im, j] + U[ip, j] + U[i, jm] + U[i, jp]) * 0.2
                  + (U[im, jm] + U[ip, jm] + U[im, jp] + U[ip, jp]) * 0.05
                  - U[i, j])
            lv = ((V[im, j] + V[ip, j] + V[i, jm] + V[i, jp]) * 0.2
                  + (V[im, jm] + V[ip, jm] + V[im, jp] + V[ip, jp]) * 0.05
                  - V[i, j])
            # feed/kill vary across space -> many morphologies coexist
            nx = ti.cast(i, ti.f32) / gw - 0.5
            ny = ti.cast(j, ti.f32) / gh - 0.5
            f = ti.min(0.10, ti.max(0.008, F0 + Fx * nx + Fy * ny
                                    + self.scar[i, j] * fbias))
            k = ti.min(0.075, ti.max(0.038, k0 + kx * nx + ky * ny))
            u = U[i, j]
            v = V[i, j]
            uvv = u * v * v
            nu = u + DU * lu - uvv + f * (1.0 - u)
            nv = v + DV * lv + uvv - (k + f) * v
            Un[i, j] = ti.min(ti.max(nu, 0.0), 1.0)
            Vn[i, j] = ti.min(ti.max(nv, 0.0), 1.0)

    @ti.func
    def _sampleV(self, gx, gy):
        gx = ti.min(ti.max(gx, 0.0), self.gw - 1.0)
        gy = ti.min(ti.max(gy, 0.0), self.gh - 1.0)
        x0 = ti.cast(ti.floor(gx), ti.i32)
        y0 = ti.cast(ti.floor(gy), ti.i32)
        x1 = ti.min(self.gw - 1, x0 + 1)
        y1 = ti.min(self.gh - 1, y0 + 1)
        ax = gx - x0
        ay = gy - y0
        return (self.V[x0, y0] * (1 - ax) * (1 - ay)
                + self.V[x1, y0] * ax * (1 - ay)
                + self.V[x0, y1] * (1 - ax) * ay
                + self.V[x1, y1] * ax * ay)

    @ti.kernel
    def _render(self, gain: ti.f32, relief_k: ti.f32, shine: ti.f32,
                irid: ti.f32, hue_rot: ti.f32, la: ti.f32):
        gwf = self.gw - 1.0
        ghf = self.gh - 1.0
        L = ti.Vector([ti.cos(la) * 0.7, ti.sin(la) * 0.7, 0.65]).normalized()
        Hh = (L + ti.Vector([0.0, 0.0, 1.0])).normalized()
        for i, j in self.canvas:
            fx = ti.cast(i, ti.f32) * gwf / self.w
            fy = ti.cast(j, ti.f32) * ghf / self.h
            v = self._sampleV(fx, fy)
            # height-field gradient -> surface normal (the 3-D look)
            gx = self._sampleV(fx + 1.0, fy) - self._sampleV(fx - 1.0, fy)
            gy = self._sampleV(fx, fy + 1.0) - self._sampleV(fx, fy - 1.0)
            n = ti.Vector([-gx * relief_k, -gy * relief_k, 1.0]).normalized()
            diff = ti.max(0.0, n.dot(L))
            spec = ti.pow(ti.max(0.0, n.dot(Hh)), 24.0) * shine
            # iridescent color: hue shifts with growth direction + concentration
            ang = ti.atan2(gy, gx) / TWO_PI
            hue = v * 2.0 + hue_rot + ang * irid
            hue = hue - ti.floor(hue)
            ci = ti.cast(hue * 255.0, ti.i32) % 256
            col = self.palette[ci]
            b = ti.pow(ti.min(1.0, v * 3.2), 0.7)
            shade = 0.28 + 0.72 * diff
            self.canvas[i, j] = (col * b * shade * gain
                                 + ti.Vector([1.0, 1.0, 1.0]) * spec * b)

    # ---- per frame ------------------------------------------------------
    def render(self, ctx):
        p = ctx.p

        # metamorphosis: re-roll the genome on a drop (or a hand-cranked knob)
        self._meta_cool = max(0, self._meta_cool - 1)
        if float(p["metamorph"]) > 0.55 and self._meta_cool == 0:
            self._roll_genome()
            self._meta_cool = 90                 # ~3 s lockout between morphs

        F = float(p["feed"])
        life = float(p["life"])
        k = float(p["kill"]) + (1.0 - life) * 0.012   # quiet -> recede / die
        fbias = float(p["memory"])
        div = float(p["diversity"])
        f_amp = div * 0.024
        k_amp = div * 0.014
        steps = max(1, int(p["organism_speed"]))

        for _ in range(steps):
            self._step(self.U, self.V, self.Un, self.Vn,
                       F, k, self.gen_fx * f_amp, self.gen_fy * f_amp,
                       self.gen_kx * k_amp, self.gen_ky * k_amp, fbias)
            self.U, self.Un = self.Un, self.U
            self.V, self.Vn = self.Vn, self.V

        ag = float(p["agitate"])
        if ag > 0.001:
            self._agitate(ag * 0.15)

        a = ctx.audio
        if a is not None and a.beat and life > 0.05:
            r = self.gh * (0.04 + 0.05 * life)
            for _ in range(1 + int(2 * life)):
                self._seed(random.random() * self.gw, random.random() * self.gh,
                           r, float(p["seed_beat"]))
            self._scar_add(fbias * 0.02)

        la = self.gen_light + ctx.time * 0.15        # the highlight slides slowly
        self._render(float(p["glow"]), float(p["relief"]) * 8.0,
                     float(p["shine"]), float(p["iridescence"]) * self.gen_irid,
                     self.gen_hue, la)


if __name__ == "__main__":
    import app
    app.main(prefer=LivingVisual.name)
