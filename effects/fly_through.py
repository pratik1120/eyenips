"""Fly Through Your Song — a first-person flight at speed through your music.

The fast, crisp, made-to-be-filmed counterpart to the Living Organism. You're
hurtling forward through an endless tunnel whose walls ARE your song: the recent
energy is laid down along the corridor, and every beat drops a ring of light that
comes rushing at you. Bass flexes the tunnel open, treble twists it, the drop
slams the throttle. Pure analytic per-pixel rendering (polar coordinates, no
raymarch loop) so it stays razor-sharp at high frame-rates.

It's "your song" because the corridor is literally written from the audio as you
fly: a ring buffer of energy you travel through, beats burned in as bright gates.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, ColorPalette

TWO_PI = 6.28318
RINGS = 512             # length of the corridor's energy buffer


@ti.data_oriented
class FlyThrough(Effect):
    name = "Fly Through Your Song"
    description = ("A first-person flight through a tunnel built from your song — "
                   "beats rush at you as rings of light. Fast, crisp, made to film.")
    author = "Eyenips"

    params = [
        Slider("speed", 0.2, 6.0, default=1.6, drive=("bass", 0.5),
               help="Flight speed. Bass pushes the throttle."),
        Slider("beat_surge", 0.0, 3.0, default=1.2, audio=False,
               help="How hard each beat kicks you forward (the lurch on the kick)."),
        Slider("tunnel_width", 0.2, 1.6, default=0.7, drive=("bass", 0.3),
               help="How open the corridor is. Bass flexes the walls."),
        Slider("twist", -1.0, 1.0, default=0.15, drive=("treble", 0.4),
               help="Spiral / corkscrew of the tunnel. Treble winds it up."),
        IntSlider("sides", 0, 8, default=0, audio=False,
                  help="Cross-section: 0 = round tunnel, 3+ = star / flower bore."),
        Slider("edge", 0.0, 0.4, default=0.12, audio=False,
               help="Sharpness of the star points (with Sides)."),
        Slider("ring_density", 2.0, 30.0, default=11.0, audio=False,
               help="How tightly packed the rings are along the corridor."),
        Slider("color_scroll", 0.0, 2.0, default=0.5, drive=("intensity", 0.4),
               help="How fast color cycles down the tunnel."),
        Slider("fog", 0.0, 0.5, default=0.12, audio=False,
               help="Depth haze — how quickly the far end fades to the vanishing point."),
        Slider("glow", 0.3, 3.0, default=1.5, drive=("volume", 0.5),
               help="Overall brightness of the walls."),
        ColorPalette(default="plasma"),
    ]

    # ---- lifecycle ------------------------------------------------------
    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.R = RINGS
        self.rings = ti.field(ti.f32, shape=self.R)   # the song, written along the corridor
        self.forward = 0.0
        self._head = 0
        self._fill_rings(0.3)

    def reset(self):
        self.forward = 0.0
        self._head = 0
        self._fill_rings(0.3)

    @ti.kernel
    def _fill_rings(self, v: ti.f32):
        for i in self.rings:
            self.rings[i] = v

    @ti.kernel
    def _render(self, fwd: ti.f32, tk: ti.f32, twist: ti.f32, rden: ti.f32,
                cscroll: ti.f32, glow: ti.f32, fog: ti.f32, sides: ti.i32,
                samp: ti.f32):
        cx = self.w * 0.5
        cy = self.h * 0.5
        inv = 1.0 / cy
        for i, j in self.canvas:
            x = (ti.cast(i, ti.f32) - cx) * inv
            y = (ti.cast(j, ti.f32) - cy) * inv
            ang = ti.atan2(y, x)
            r = ti.sqrt(x * x + y * y)
            if sides > 0:                                   # star / flower cross-section
                r = r * (1.0 + samp * ti.cos(sides * ang))
            r = ti.max(r, 1e-3)
            pdepth = tk / r                                 # perspective depth (fixed: for fog)
            coord = pdepth + fwd                            # scrolls forward = flying in
            ring = coord * rden
            ri = ti.cast(ti.floor(ring), ti.i32)
            wall = self.rings[ri % self.R]                  # the song at this distance
            frac = ring - ti.floor(ring)
            groove = 0.5 + 0.5 * ti.cos(frac * TWO_PI)      # the rings streaking past
            hue = ang / TWO_PI + twist * coord + coord * cscroll
            hue = hue - ti.floor(hue)
            ci = ti.cast(hue * 255.0, ti.i32) % 256
            depthfade = ti.exp(-pdepth * fog)               # far end fades out (vanishing point)
            b = (0.12 + wall) * groove * depthfade * glow
            self.canvas[i, j] = self.palette[ci] * b

    # ---- per frame ------------------------------------------------------
    def render(self, ctx):
        p = ctx.p
        a = ctx.audio
        dt = ctx.dt if ctx.dt and ctx.dt > 0 else (1.0 / 60.0)

        beat = bool(a.beat) if a is not None else False
        speed = float(p["speed"]) * (1.0 + (float(p["beat_surge"]) if beat else 0.0))
        rden = float(p["ring_density"])

        # advance the camera; wrap on a full ring-buffer cycle so it never drifts
        period = self.R / max(1.0, rden)
        self.forward = (self.forward + speed * dt * 6.0) % period

        # write the song into the corridor at the rings we just flew past
        newhead = int(self.forward * rden)
        steps = newhead - self._head
        if steps < 0:
            steps += self.R                                 # wrapped around the buffer
        energy = 0.30
        if a is not None:
            energy = 0.25 + 0.9 * float(a.bass)
        if beat:
            energy = 1.6                                    # beats = bright gates
        for s in range(min(steps, 24)):
            self.rings[(self._head + 1 + s) % self.R] = energy
        self._head = newhead

        # tunnel_width -> how far the walls sit (bigger = wider bore)
        tk = float(p["tunnel_width"])
        self._render(self.forward, tk, float(p["twist"]), rden,
                     float(p["color_scroll"]), float(p["glow"]), float(p["fog"]),
                     int(p["sides"]), float(p["edge"]))


if __name__ == "__main__":
    import app
    app.main(prefer=FlyThrough.name)
