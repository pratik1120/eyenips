"""Effect Lab — a generative effect engine: every number is a new effect.

The look isn't one formula with different knobs — it's a library of structurally
DIFFERENT generators (archetypes): plasma, moving lines, grids, rings, spirals,
Voronoi cells, Julia fractals, moiré, turbulent cloud, oscilloscope waveforms,
halftone dots, topographic contours. Each **Recipe** (one integer) randomly:

  • picks an archetype (sometimes blends a second one a different way),
  • decides symmetry — usually NONE, occasionally mirror/kaleidoscope,
  • picks a render *style* (smooth blobs / line-art / posterized flats),
  • picks color mapping, domain-warp, and motion.

So Recipe 7 might be scrolling diagonal lines, Recipe 8 a pulsing Julia set, and
Recipe 9 drifting cells — genuinely different effects, not variations of one. The
seed space is millions deep. Spin **Recipe** / hit **🎲 Randomize** to explore,
tweak, and **Save** the keepers.

It's also a build-your-own VIDEO effect: load a clip and turn on **Over video**,
**Flow warp** (footage motion bends the art) or **Blob sparks**. Pure math +
OpenCV motion; no AI.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random as _random

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, Toggle, ColorPalette

TWO_PI = 6.28318
N_ARCH = 12


@ti.data_oriented
class EffectLab(Effect):
    name = "Effect Lab"
    description = ("A generative engine: every Recipe is a structurally different "
                   "effect (lines, cells, fractals, clouds, waveforms…), audio-"
                   "reactive, doubling as a build-your-own video effect. No AI.")
    author = "Eyenips"
    uses_video = True

    params = [
        IntSlider("recipe", 0, 99999, default=7, audio=False,
                  help="THE knob: each number is a different effect (millions). "
                       "Spin it, or 🎲 Randomize, to discover."),
        Slider("speed", 0.0, 3.0, default=1.0, drive=("bass", 0.4),
               help="Animation speed."),
        Slider("warp", 0.0, 2.0, default=1.0, audio=False,
               help="Extra liquid domain-warp on top of the recipe's own."),
        Slider("detail", 0.4, 2.5, default=1.0, drive=("treble", 0.5),
               help="Pattern fineness. Treble adds detail."),
        Slider("react", 0.0, 1.5, default=0.6, drive=("volume", 0.8),
               help="How much the music brightens / pumps it."),
        Slider("glow", 0.3, 3.0, default=1.4, drive=("volume", 0.3),
               help="Overall brightness."),
        Slider("video_mix", 0.0, 1.0, default=0.0, label="Over video",
               help="Blend a loaded VIDEO under the art (0 = pure generative)."),
        Toggle("flow_warp", default=False,
               help="The video's MOTION bends the art (optical-flow warp)."),
        Toggle("blob_sparks", default=False,
               help="Sparks latch onto the things that MOVE in the video."),
        Slider("spark_size", 0.2, 2.0, default=0.6, audio=False,
               help="Size of the blob sparks."),
        ColorPalette(default="rainbow"),
    ]

    # ---- lifecycle ------------------------------------------------------
    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.media = ctx.media
        self.flow = ctx.flow
        self.blobs = ctx.blobs
        self._seed = None
        self._R = self._recipe(7)

    def _recipe(self, seed):
        """One integer -> a full structural recipe. Symmetry is usually OFF and
        the archetype is chosen freely, so recipes look categorically different."""
        rng = _random.Random((int(seed) * 2654435761) & 0xFFFFFFFF)
        ch = rng.choice
        return dict(
            archA=rng.randrange(N_ARCH),
            archB=rng.randrange(N_ARCH),
            mixmode=ch([0, 0, 0, 1, 2, 3, 4]),          # mostly a single archetype
            sym=ch([0, 0, 0, 0, 0, 1, 2, 3]),           # mostly NO symmetry
            nfold=float(ch([3, 4, 5, 6, 8])),
            style=ch([0, 0, 1, 1, 2]),                  # smooth / line-art / flats
            colmode=ch([0, 0, 0, 1, 2, 3]),
            f1=rng.uniform(0.6, 4.0), f2=rng.uniform(0.6, 4.0), f3=rng.uniform(0.5, 3.0),
            ph=rng.uniform(0.0, TWO_PI),
            warp=ch([0.0, 0.0, 0.0, 0.4, 0.8, 1.2]) * rng.uniform(0.6, 1.4),
            fw1=rng.uniform(1.0, 4.0), fw2=rng.uniform(1.0, 4.0),
            colscale=rng.uniform(0.2, 1.6), coloff=rng.uniform(0.0, 1.0),
            bands=float(rng.randint(2, 7)),
            rot=rng.uniform(0.0, TWO_PI),
            spd=rng.uniform(0.3, 1.6) * ch([-1.0, 1.0]),
            contrast=rng.uniform(5.0, 16.0),
            levels=float(rng.randint(2, 6)))

    # ---- noise / helpers ------------------------------------------------
    @ti.func
    def _h21(self, px, py):
        h = ti.sin(px * 127.1 + py * 311.7) * 43758.5453
        return h - ti.floor(h)

    @ti.func
    def _vnoise(self, x, y):
        ix = ti.floor(x)
        iy = ti.floor(y)
        fx = x - ix
        fy = y - iy
        ux = fx * fx * (3.0 - 2.0 * fx)
        uy = fy * fy * (3.0 - 2.0 * fy)
        a = self._h21(ix, iy)
        b = self._h21(ix + 1.0, iy)
        c = self._h21(ix, iy + 1.0)
        d = self._h21(ix + 1.0, iy + 1.0)
        return (a * (1 - ux) * (1 - uy) + b * ux * (1 - uy)
                + c * (1 - ux) * uy + d * ux * uy)

    @ti.func
    def _fbm(self, x, y):
        s = 0.0
        amp = 0.5
        fr = 1.0
        for _k in range(4):
            s += amp * self._vnoise(x * fr, y * fr)
            fr *= 2.0
            amp *= 0.5
        return s

    @ti.func
    def _voronoi(self, x, y, t):
        gx = ti.floor(x)
        gy = ti.floor(y)
        fx = x - gx
        fy = y - gy
        md = 8.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                hx = self._h21(gx + ox, gy + oy)
                hy = self._h21(gx + ox + 17.0, gy + oy + 9.0)
                px = ox + 0.5 + 0.4 * ti.sin(t + TWO_PI * hx)
                py = oy + 0.5 + 0.4 * ti.sin(t * 1.1 + TWO_PI * hy)
                d = ti.sqrt((px - fx) ** 2 + (py - fy) ** 2)
                md = ti.min(md, d)
        return md

    @ti.func
    def _julia(self, u, v, f1, f2, ph, t):
        zx = u * 1.6
        zy = v * 1.6
        cx = 0.7885 * ti.cos(t * 0.15 + ph)
        cy = 0.7885 * ti.sin(t * 0.15 + ph * 0.7)
        it = 0.0
        done = 0
        for _k in range(28):
            if done == 0:
                nx = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = nx
                if zx * zx + zy * zy > 4.0:
                    done = 1
                else:
                    it += 1.0
        return it / 28.0

    # ---- the archetype library (each returns roughly -1..1) -------------
    @ti.func
    def _arch(self, mode, u, v, t, f1, f2, f3, ph):
        val = 0.0
        if mode == 0:                                   # plasma blobs
            val = (ti.sin(u * f1 + t) + ti.sin(v * f2 + t * 0.7 + ph)
                   + ti.sin((u + v) * f3 - t * 0.5)) * 0.33
        elif mode == 1:                                 # directional lines
            d = u * ti.cos(ph) + v * ti.sin(ph)
            val = ti.sin(d * f1 * 4.0 - t * 2.0)
        elif mode == 2:                                 # grid / checker
            val = ti.sin(u * f1 * 4.0 + t) * ti.sin(v * f2 * 4.0 - t)
        elif mode == 3:                                 # concentric rings
            val = ti.sin(ti.sqrt(u * u + v * v) * f1 * 5.0 - t * 2.0)
        elif mode == 4:                                 # spiral
            r = ti.sqrt(u * u + v * v)
            a = ti.atan2(v, u)
            val = ti.sin(r * f1 * 4.0 + a * ti.round(f2 * 2.0 + 1.0) - t * 2.0)
        elif mode == 5:                                 # voronoi cells
            val = self._voronoi(u * f1 + 8.0, v * f1 + 8.0, t) * 2.0 - 1.0
        elif mode == 6:                                 # julia fractal
            val = self._julia(u, v, f1, f2, ph, t) * 2.0 - 1.0
        elif mode == 7:                                 # moiré interference
            d1 = ti.sin((u * ti.cos(ph) + v * ti.sin(ph)) * f1 * 5.0)
            d2 = ti.sin((u * ti.cos(ph + 1.3) + v * ti.sin(ph + 1.3)) * f2 * 5.0)
            val = d1 * d2
        elif mode == 8:                                 # turbulent cloud
            val = self._fbm(u * f1 + t * 0.1, v * f1 - t * 0.05) * 2.0 - 1.0
        elif mode == 9:                                 # oscilloscope waveforms
            wv = (ti.sin(u * f1 * 4.0 + t * 2.0) + ti.sin(u * f2 * 7.0 - t)) * 0.25
            val = 1.0 - ti.min(1.0, ti.abs(v - wv) * 9.0)
            val = val * 2.0 - 1.0
        elif mode == 10:                                # halftone dots
            cell = 5.0 * f1
            fu = u * cell - ti.floor(u * cell) - 0.5
            fv = v * cell - ti.floor(v * cell) - 0.5
            sz = 0.34 + 0.22 * ti.sin(t + ti.floor(u * cell) + ti.floor(v * cell))
            val = (sz - ti.sqrt(fu * fu + fv * fv)) * 5.0
        else:                                           # topographic contours
            val = ti.sin(self._fbm(u * f1 + t * 0.1, v * f1) * 12.0)
        return val

    @ti.kernel
    def _render(self, t: ti.f32, archA: ti.i32, archB: ti.i32, mixmode: ti.i32,
                sym: ti.i32, nfold: ti.f32, style: ti.i32, colmode: ti.i32,
                f1: ti.f32, f2: ti.f32, f3: ti.f32, ph: ti.f32, warp: ti.f32,
                fw1: ti.f32, fw2: ti.f32, colscale: ti.f32, coloff: ti.f32,
                bands: ti.f32, rot: ti.f32, detail: ti.f32, contrast: ti.f32,
                levels: ti.f32, glow: ti.f32, react: ti.f32, vol: ti.f32,
                bass: ti.f32, flow_amt: ti.f32):
        ca = ti.cos(rot)
        sa = ti.sin(rot)
        for i, j in self.canvas:
            u = (ti.cast(i, ti.f32) / self.w - 0.5) * 2.0
            v = (ti.cast(j, ti.f32) / self.h - 0.5) * 2.0
            if flow_amt > 0.0:
                fl = self.flow[i, j]
                u += fl[0] * flow_amt * 8.0
                v += fl[1] * flow_amt * 8.0
            ru = u * ca - v * sa
            rv = u * sa + v * ca
            u, v = ru, rv
            if sym == 1:                                # mirror x
                u = ti.abs(u)
            elif sym == 2:                              # kaleidoscope
                ang = ti.atan2(v, u)
                rad = ti.sqrt(u * u + v * v)
                seg = TWO_PI / nfold
                ang = ti.abs((ang - ti.floor(ang / seg) * seg) - seg * 0.5)
                u, v = ti.cos(ang) * rad, ti.sin(ang) * rad
            elif sym == 3:                              # 4-way mirror
                u = ti.abs(u)
                v = ti.abs(v)
            d = detail * (1.0 + 0.3 * bass)
            wu = u * d + warp * ti.sin(v * fw1 + t * 0.3)
            wv = v * d + warp * ti.cos(u * fw2 - t * 0.2)
            val = self._arch(archA, wu, wv, t, f1, f2, f3, ph)
            if mixmode > 0:
                vb = self._arch(archB, wv, wu, t * 0.8, f2, f3, f1, ph + 1.7)
                if mixmode == 1:
                    val = (val + vb) * 0.5
                elif mixmode == 2:
                    val = val * vb
                elif mixmode == 3:
                    val = ti.max(val, vb)
                else:
                    val = val if vb > 0.0 else vb       # mask
            vn = val * 0.5 + 0.5                         # -> 0..1

            # render style: how the field becomes brightness
            b = 0.0
            if style == 0:                              # smooth
                b = 0.3 + 0.7 * vn
            elif style == 1:                            # line-art (zero-crossings)
                b = 1.0 - ti.min(1.0, ti.abs(val) * contrast)
            else:                                       # posterized flats
                b = ti.floor(vn * levels) / levels

            # color mapping
            hue = vn * colscale + coloff + t * 0.02
            if colmode == 1:
                hue = ti.atan2(v, u) / TWO_PI + coloff + t * 0.02
            elif colmode == 2:
                hue = ti.floor(vn * bands) / bands + coloff
            elif colmode == 3:
                hue = ti.sqrt(u * u + v * v) * colscale + coloff
            hue = hue - ti.floor(hue)
            ci = ti.cast(hue * 255.0, ti.i32) % 256
            bright = b * glow * (1.0 - react + react * (0.4 + vol))
            self.canvas[i, j] = self.palette[ci] * bright

    @ti.kernel
    def _over_video(self, mix: ti.f32):
        for i, j in self.canvas:
            self.canvas[i, j] += self.media[i, j] * mix

    @ti.kernel
    def _sparks(self, n: ti.i32, size: ti.f32, glow: ti.f32, pulse: ti.f32):
        for i, j in self.canvas:
            p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            acc = ti.Vector([0.0, 0.0, 0.0])
            for b in range(n):
                c = ti.Vector([self.blobs[b, 0] * self.w, self.blobs[b, 1] * self.h])
                br = self.blobs[b, 2] * self.h * size * (1.0 + pulse) * 0.5 + 3.0
                dd = (p - c).norm()
                acc += self.palette[(b * 50) % 256] * ti.exp(-(dd * dd) / (br * br))
            self.canvas[i, j] += acc * glow

    # ---- per frame ------------------------------------------------------
    def render(self, ctx):
        p = ctx.p
        seed = int(p["recipe"])
        if seed != self._seed:
            self._R = self._recipe(seed)
            self._seed = seed
        R = self._R
        a = ctx.audio
        vol = float(a.volume) if a is not None else 0.0
        bass = float(a.bass) if a is not None else 0.0
        flow_amt = 1.0 if (p.get("flow_warp") and getattr(ctx, "has_video", False)) else 0.0
        t = ctx.time * float(p["speed"]) * R["spd"]

        self._render(
            t, R["archA"], R["archB"], R["mixmode"], R["sym"], R["nfold"],
            R["style"], R["colmode"], R["f1"], R["f2"], R["f3"], R["ph"],
            R["warp"] * float(p["warp"]), R["fw1"], R["fw2"], R["colscale"],
            R["coloff"], R["bands"], R["rot"], float(p["detail"]), R["contrast"],
            R["levels"], float(p["glow"]), float(p["react"]), vol, bass, flow_amt)

        if float(p["video_mix"]) > 0.001 and getattr(ctx, "has_media", False):
            self._over_video(float(p["video_mix"]))
        if p.get("blob_sparks") and getattr(ctx, "has_video", False):
            n = int(getattr(ctx, "n_blobs", 0))
            if n > 0:
                pulse = float(p["react"]) if (a is not None and a.beat) else 0.0
                self._sparks(n, float(p["spark_size"]), float(p["glow"]), pulse)


if __name__ == "__main__":
    import app
    app.main(prefer=EffectLab.name)
