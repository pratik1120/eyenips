"""Video Lab — a generative engine for INTERACTIVE VIDEO effects.

Effect Lab generates visualizers; this generates *video effects*. The trick to
making them feel genuinely random (not "effect A glued to effect B") is that a
Recipe is NOT a pick of one transform + one overlay — it's a continuous,
high-dimensional **pipeline**. Every operator is always present to some random
degree (most near zero), composed together:

  warp the sampling of the footage:  flow-melt · swirl · wave · noise-warp ·
                                      kaleidoscope · mirror · tile  (each 0..1)
  restyle the color:                  chromatic · neon-edges · posterize ·
                                      duotone · invert            (each 0..1)
  augment from motion (overlays):     blob halos · sparks · trails ·
                                      constellation · flow-glow   (each 0..1)

So one Recipe is "70% melt + a little swirl + 30% kaleidoscope, colors 20%
duotone, trails at 0.6" and the next is something else entirely — millions of
content-aware video effects that don't read as combinations. Keep the **Footage**
knob up to preserve the clip's essence; **🎲 Randomize** to surf; **Export VIDEO**
the keepers. Pure math + OpenCV motion; no AI.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random as _random

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, ColorPalette
from vizstudio import labkit

TWO_PI = 6.28318

# the operators whose strengths a Recipe randomizes (order matters for packing)
_WARP = ["melt", "swirl", "wave", "noise", "kale", "mirror", "tile"]
_COLOR = ["chroma", "edge", "poster", "duo", "invert"]
_OVER = ["halo", "spark", "trail", "link", "flow"]
_OPS = _WARP + _COLOR + _OVER


@ti.data_oriented
class VideoLab(Effect):
    name = "Video Lab"
    description = ("A generative engine for content-aware VIDEO effects — each "
                   "Recipe is a continuous blend of many operators, so they feel "
                   "truly random, not combinations. Load a video. No AI.")
    author = "Eyenips"
    uses_video = True
    editable_kit = "video_lab"     # the Edit-equations editor edits its recipe

    params = [
        IntSlider("recipe", 0, 99999, default=7, audio=False,
                  help="THE knob: each number is a different VIDEO effect "
                       "(millions). Spin it, or 🎲 Randomize. Load a video first."),
        Slider("strength", 0.0, 1.5, default=0.8, drive=("bass", 0.4),
               help="How hard the whole pipeline hits the footage."),
        Slider("footage", 0.0, 1.0, default=0.5, audio=False,
               help="Keep the ESSENCE: how much original video shows through "
                    "(1 = subtle, 0 = fully restyled)."),
        Slider("react", 0.0, 1.5, default=0.6, drive=("volume", 0.8),
               help="How much the music pumps the effect."),
        Slider("glow", 0.3, 3.0, default=1.3, drive=("volume", 0.3),
               help="Brightness of added light."),
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
        self._kit = labkit.load()
        self._R = self._recipe(7)

    def set_kit(self, kit):
        """Called by the Edit-equations editor — swaps in edited operator pool /
        recipe ranges and re-rolls the current recipe next frame."""
        self._kit = kit
        self._seed = None

    def current_kit(self):
        return self._kit

    def _recipe(self, seed):
        """Each Recipe has a distinct CHARACTER: a few dominant operators plus a
        couple of subtle accents, the rest off. Continuous weights (so it's never
        binary on/off) but sparse (so it's never a muddy kitchen-sink) — that
        balance is what makes recipes feel genuinely random and distinct. The
        operator pool, sparsity and ranges are the editable Lab Kit."""
        vk = self._kit["video_lab"]
        rc = vk["recipe"]
        ops_on = [o for o in _OPS if vk["operators"].get(o, True)] or list(_OPS)
        over_on = [o for o in _OVER if o in ops_on]
        rng = _random.Random((int(seed) * 2654435761) & 0xFFFFFFFF)
        R = {o: 0.0 for o in _OPS}

        def rint(key):
            return rng.randint(int(rc[key][0]), int(rc[key][1]))

        def runi(key):
            return rng.uniform(rc[key][0], rc[key][1])

        ndom = min(len(ops_on), rint("dominant"))               # dominant character
        for o in rng.sample(ops_on, ndom):
            R[o] = runi("dominant_weight")
        rest = [o for o in ops_on if R[o] == 0.0]
        nacc = min(len(rest), rint("accent"))                   # subtle accents
        for o in rng.sample(rest, nacc):
            R[o] = runi("accent_weight")
        if over_on and rng.random() < 0.6 and not any(R[o] > 0.1 for o in _OVER):
            R[rng.choice(over_on)] = runi("dominant_weight")    # usually some motion overlay
        R["fold"] = runi("fold")
        R["tiles"] = runi("tiles")
        R["woff"] = runi("woff")
        R["levels"] = float(rint("levels"))
        R["colscale"] = runi("colscale")
        R["coloff"] = runi("coloff")
        R["fw1"] = runi("fw1")
        R["fw2"] = runi("fw2")
        R["edgeg"] = runi("edgeg")
        R["ovsize"] = runi("ovsize")
        R["disp"] = runi("disp")
        R["spd"] = rng.choice([-1.0, 1.0]) * runi("spd")
        return R

    # ---- samplers / noise ----------------------------------------------
    @ti.func
    def _sm(self, x, y):
        x = ti.min(ti.max(x, 0.0), self.w - 1.0)
        y = ti.min(ti.max(y, 0.0), self.h - 1.0)
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.w - 1, x0 + 1)
        y1 = ti.min(self.h - 1, y0 + 1)
        ax = x - x0
        ay = y - y0
        return (self.media[x0, y0] * (1 - ax) * (1 - ay)
                + self.media[x1, y0] * ax * (1 - ay)
                + self.media[x0, y1] * (1 - ax) * ay
                + self.media[x1, y1] * ax * ay)

    @ti.func
    def _lum(self, c):
        return (c[0] + c[1] + c[2]) * 0.3333

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
        return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy

    @ti.func
    def _sobel(self, i, j):
        x0 = ti.max(i - 1, 0)
        x1 = ti.min(i + 1, self.w - 1)
        y0 = ti.max(j - 1, 0)
        y1 = ti.min(j + 1, self.h - 1)
        gx = ((self._lum(self.media[x1, y1]) + 2 * self._lum(self.media[x1, j])
               + self._lum(self.media[x1, y0]))
              - (self._lum(self.media[x0, y1]) + 2 * self._lum(self.media[x0, j])
                 + self._lum(self.media[x0, y0])))
        gy = ((self._lum(self.media[x0, y1]) + 2 * self._lum(self.media[i, y1])
               + self._lum(self.media[x1, y1]))
              - (self._lum(self.media[x0, y0]) + 2 * self._lum(self.media[i, y0])
                 + self._lum(self.media[x1, y0])))
        return ti.sqrt(gx * gx + gy * gy)

    # ---- the pipeline (continuous blend of every operator) -------------
    @ti.kernel
    def _render(self, t: ti.f32, footage: ti.f32, glow: ti.f32, react: ti.f32,
                vol: ti.f32, hasm: ti.i32, disp: ti.f32, fold: ti.f32, tiles: ti.f32,
                woff: ti.f32, levels: ti.f32, colscale: ti.f32, coloff: ti.f32,
                fw1: ti.f32, fw2: ti.f32, edgeg: ti.f32,
                w_melt: ti.f32, w_swirl: ti.f32, w_wave: ti.f32, w_noise: ti.f32,
                w_kale: ti.f32, w_mirror: ti.f32, w_tile: ti.f32, w_chroma: ti.f32,
                w_edge: ti.f32, w_poster: ti.f32, w_duo: ti.f32, w_invert: ti.f32):
        for i, j in self.canvas:
            x = ti.cast(i, ti.f32)
            y = ti.cast(j, ti.f32)
            if hasm == 0:                                # no video: alive placeholder
                u = x / self.w
                vv = y / self.h
                s = 0.5 + 0.4 * ti.sin(u * 7.0 + t) + 0.4 * ti.sin(vv * 7.0 - t)
                hue = s * 0.3 + t * 0.05
                self.canvas[i, j] = (
                    self.palette[ti.cast((hue - ti.floor(hue)) * 255, ti.i32) % 256] * 0.4)
            else:
                u = x / self.w - 0.5                     # centered normalized coords
                v = y / self.h - 0.5
                fl = self.flow[i, j]
                # --- warp the sampling (all continuous) ---
                u += fl[0] * disp * w_melt                # flow-melt
                v += fl[1] * disp * w_melt
                if w_swirl > 0.0:                         # swirl (stronger near center)
                    swa = w_swirl / (ti.sqrt(u * u + v * v) + 0.25)
                    cs = ti.cos(swa)
                    sn = ti.sin(swa)
                    u, v = u * cs - v * sn, u * sn + v * cs
                u += ti.sin(v * fw1 + t) * 0.12 * w_wave  # wave
                v += ti.cos(u * fw2 - t) * 0.12 * w_wave
                if w_noise > 0.0:                         # noise warp
                    nx = self._vnoise(u * 3.0 + t * 0.1, v * 3.0)
                    ny = self._vnoise(u * 3.0 + 7.0, v * 3.0 - t * 0.1)
                    u += (nx - 0.5) * 0.5 * w_noise
                    v += (ny - 0.5) * 0.5 * w_noise
                if w_kale > 0.0:                          # kaleidoscope (lerped)
                    ang = ti.atan2(v, u)
                    rad = ti.sqrt(u * u + v * v)
                    seg = TWO_PI / fold
                    fa = ti.abs((ang - ti.floor(ang / seg) * seg) - seg * 0.5)
                    u += (ti.cos(fa) * rad - u) * w_kale
                    v += (ti.sin(fa) * rad - v) * w_kale
                u += (ti.abs(u) - u) * w_mirror           # mirror (lerped)
                if w_tile > 0.0:                          # tile / mosaic (lerped)
                    tu = (u + 0.5) * tiles
                    tv = (v + 0.5) * tiles
                    u += ((tu - ti.floor(tu)) - 0.5 - u) * w_tile
                    v += ((tv - ti.floor(tv)) - 0.5 - v) * w_tile

                sx = (u + 0.5) * self.w
                sy = (v + 0.5) * self.h
                out = self._sm(sx, sy)

                # --- restyle the color (all continuous) ---
                if w_chroma > 0.0:                        # chromatic split
                    ox = woff * w_chroma
                    cr = self._sm(sx + ox, sy)[0]
                    cb = self._sm(sx - ox, sy)[2]
                    out = ti.Vector([out[0] * (1 - w_chroma) + cr * w_chroma, out[1],
                                     out[2] * (1 - w_chroma) + cb * w_chroma])
                if w_edge > 0.0:                          # neon edges added
                    e = self._sobel(i, j) * edgeg
                    hue = e * colscale + coloff
                    ec = self.palette[ti.cast((hue - ti.floor(hue)) * 255, ti.i32) % 256]
                    out += ec * (ti.min(1.0, e) * w_edge)
                if w_poster > 0.0:                        # posterize
                    lv = ti.max(2.0, levels)
                    po = ti.Vector([ti.floor(out[0] * lv) / lv, ti.floor(out[1] * lv) / lv,
                                    ti.floor(out[2] * lv) / lv])
                    out = out * (1 - w_poster) + po * w_poster
                if w_duo > 0.0:                           # duotone
                    lu = self._lum(out) * colscale + coloff
                    du = self.palette[ti.cast((lu - ti.floor(lu)) * 255, ti.i32) % 256]
                    out = out * (1 - w_duo) + du * w_duo
                out = out * (1 - w_invert) + (ti.Vector([1.0, 1.0, 1.0]) - out) * w_invert

                base = self.media[i, j]
                out = out * (1.0 - footage) + base * footage     # keep the essence
                self.canvas[i, j] = out * glow * (1.0 - react + react * (0.4 + vol))

    # ---- overlays: every type summed by its own weight (not one-of) ----
    @ti.kernel
    def _overlays(self, n: ti.i32, w_halo: ti.f32, w_spark: ti.f32, w_trail: ti.f32,
                  w_link: ti.f32, w_flow: ti.f32, size: ti.f32, glow: ti.f32,
                  pulse: ti.f32, flowamt: ti.f32):
        for i, j in self.canvas:
            p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            acc = ti.Vector([0.0, 0.0, 0.0])
            for b in range(n):
                c = ti.Vector([self.blobs[b, 0] * self.w, self.blobs[b, 1] * self.h])
                br = self.blobs[b, 2] * self.h * size * (1.0 + pulse) + 4.0
                d = (p - c).norm()
                col = self.palette[(b * 47) % 256]
                sig = 0.3 * br + 2.0
                halo = ti.exp(-((d - br) * (d - br)) / (2.0 * sig * sig))
                spark = ti.exp(-(d * d) / (br * br))
                vel = ti.Vector([self.blobs[b, 3] * self.w, self.blobs[b, 4] * self.h]) * 1.6
                ab = -vel
                ll = ab.dot(ab) + 1e-3
                tt = ti.min(1.0, ti.max(0.0, (p - c).dot(ab) / ll))
                dt = (p - (c + ab * tt)).norm()
                trail = ti.exp(-(dt * dt) / 36.0) * (1.0 - tt)
                acc += col * (halo * w_halo + spark * w_spark + trail * w_trail)
            if w_link > 0.0:
                lk = 0.0
                for b in range(n - 1):
                    a = ti.Vector([self.blobs[b, 0] * self.w, self.blobs[b, 1] * self.h])
                    c = ti.Vector([self.blobs[b + 1, 0] * self.w,
                                   self.blobs[b + 1, 1] * self.h])
                    ab = c - a
                    ll = ab.dot(ab) + 1e-3
                    tt = ti.min(1.0, ti.max(0.0, (p - a).dot(ab) / ll))
                    dd = (p - (a + ab * tt)).norm()
                    lk += ti.exp(-(dd * dd) / 9.0)
                acc += self.palette[150] * (lk * w_link)
            if w_flow > 0.0:
                fl = self.flow[i, j]
                m = ti.sqrt(fl[0] * fl[0] + fl[1] * fl[1])
                ang = ti.atan2(fl[1], fl[0]) / TWO_PI + 0.5
                ci = ti.cast(ang * 255.0, ti.i32) % 256
                acc += self.palette[ci] * (ti.min(1.0, m * flowamt) * w_flow)
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
        st = float(p["strength"])
        glow = float(p["glow"])
        react = float(p["react"])
        hasm = 1 if getattr(ctx, "has_media", False) else 0
        t = ctx.time * R["spd"]

        def ww(k):                                        # operator weight × strength (clamped)
            return min(1.0, R[k] * st)

        self._render(
            t, float(p["footage"]), glow, react, vol, hasm, R["disp"], R["fold"],
            R["tiles"], R["woff"], R["levels"], R["colscale"], R["coloff"],
            R["fw1"], R["fw2"], R["edgeg"],
            ww("melt"), ww("swirl"), ww("wave"), ww("noise"), ww("kale"),
            ww("mirror"), ww("tile"), ww("chroma"), ww("edge"), ww("poster"),
            ww("duo"), ww("invert"))

        if getattr(ctx, "has_video", False):
            n = int(getattr(ctx, "n_blobs", 0))
            wl = ww("link")
            wf = ww("flow")
            if n > 0 or wf > 0.0:
                pulse = react if (a is not None and a.beat) else 0.0
                self._overlays(n, ww("halo"), ww("spark"), ww("trail"), wl, wf,
                               R["ovsize"], glow * (0.6 + st), pulse, 6.0 * (0.4 + st))


if __name__ == "__main__":
    import app
    app.main(prefer=VideoLab.name)
