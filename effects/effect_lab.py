"""Effect Lab — a generative effect engine: every number is a new effect.

The look isn't one formula with different knobs — it's a library of structurally
DIFFERENT generators (archetypes): plasma, moving lines, grids, rings, spirals,
Voronoi cells, Julia fractals, moiré, turbulent cloud, oscilloscope waveforms,
halftone dots, topographic contours. Each **Recipe** (one integer) randomly:

  • picks an archetype (sometimes blends a second one a different way),
  • decides symmetry — usually NONE, occasionally mirror/kaleidoscope,
  • picks a render *style* (smooth blobs / line-art / posterized flats),
  • picks color mapping, domain-warp, and motion.

The archetype EQUATIONS and the recipe RANGES aren't locked in — they live in the
editable **Lab Kit** (see `vizstudio/labkit.py`) and the in-app *Edit equations*
editor, so you can tweak them, add your own generators, and Reset to defaults.
Each archetype is a live-compiled formula referencing u, v, t, f1, f2, f3, ph and
the helpers vor(), julia(), fbm(), vnoise().

It's also a build-your-own VIDEO effect: load a clip and turn on **Over video**,
**Flow warp** (footage motion bends the art) or **Blob sparks**. Pure math +
OpenCV motion; no AI.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random as _random

import taichi as ti

from vizstudio import Effect, Slider, IntSlider, Toggle, ColorPalette
from vizstudio import labkit
from vizstudio.exprutil import exec_with_source

TWO_PI = 6.28318


# The fixed scaffolding of the render kernel: helper building blocks + the
# symmetry / style / colour / warp logic. Only the `_arch` dispatch (the
# archetype equations) is generated from the Lab Kit and spliced in at {ARCH}.
_KERNEL_SRC = '''
import taichi as ti
TWO_PI = 6.28318

@ti.func
def _h21(px, py):
    h = ti.sin(px * 127.1 + py * 311.7) * 43758.5453
    return h - ti.floor(h)

@ti.func
def _vnoise(x, y):
    ix = ti.floor(x)
    iy = ti.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _h21(ix, iy)
    b = _h21(ix + 1.0, iy)
    c = _h21(ix, iy + 1.0)
    d = _h21(ix + 1.0, iy + 1.0)
    return (a * (1 - ux) * (1 - uy) + b * ux * (1 - uy)
            + c * (1 - ux) * uy + d * ux * uy)

@ti.func
def _fbm(x, y):
    s = 0.0
    amp = 0.5
    fr = 1.0
    for _k in range(4):
        s += amp * _vnoise(x * fr, y * fr)
        fr *= 2.0
        amp *= 0.5
    return s

@ti.func
def _vor(x, y, t):
    gx = ti.floor(x)
    gy = ti.floor(y)
    fx = x - gx
    fy = y - gy
    md = 8.0
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            hx = _h21(gx + ox, gy + oy)
            hy = _h21(gx + ox + 17.0, gy + oy + 9.0)
            px = ox + 0.5 + 0.4 * ti.sin(t + TWO_PI * hx)
            py = oy + 0.5 + 0.4 * ti.sin(t * 1.1 + TWO_PI * hy)
            dd = ti.sqrt((px - fx) ** 2 + (py - fy) ** 2)
            md = ti.min(md, dd)
    return md

@ti.func
def _julia(u, v, f1, f2, ph, t):
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

@ti.func
def _arch(mode, u, v, t, f1, f2, f3, ph):
{ARCH}

@ti.kernel
def _k(canvas: ti.template(), palette: ti.template(), flow: ti.template(),
       W: ti.i32, H: ti.i32,
       t: ti.f32, archA: ti.i32, archB: ti.i32, mixmode: ti.i32,
       sym: ti.i32, nfold: ti.f32, style: ti.i32, colmode: ti.i32,
       f1: ti.f32, f2: ti.f32, f3: ti.f32, ph: ti.f32, warp: ti.f32,
       fw1: ti.f32, fw2: ti.f32, colscale: ti.f32, coloff: ti.f32,
       bands: ti.f32, rot: ti.f32, detail: ti.f32, contrast: ti.f32,
       levels: ti.f32, glow: ti.f32, react: ti.f32, vol: ti.f32,
       bass: ti.f32, flow_amt: ti.f32):
    ca = ti.cos(rot)
    sa = ti.sin(rot)
    for i, j in canvas:
        u = (ti.cast(i, ti.f32) / W - 0.5) * 2.0
        v = (ti.cast(j, ti.f32) / H - 0.5) * 2.0
        if flow_amt > 0.0:
            fl = flow[i, j]
            u += fl[0] * flow_amt * 8.0
            v += fl[1] * flow_amt * 8.0
        ru = u * ca - v * sa
        rv = u * sa + v * ca
        u, v = ru, rv
        if sym == 1:
            u = ti.abs(u)
        elif sym == 2:
            ang = ti.atan2(v, u)
            rad = ti.sqrt(u * u + v * v)
            seg = TWO_PI / nfold
            ang = ti.abs((ang - ti.floor(ang / seg) * seg) - seg * 0.5)
            u, v = ti.cos(ang) * rad, ti.sin(ang) * rad
        elif sym == 3:
            u = ti.abs(u)
            v = ti.abs(v)
        d = detail * (1.0 + 0.3 * bass)
        wu = u * d + warp * ti.sin(v * fw1 + t * 0.3)
        wv = v * d + warp * ti.cos(u * fw2 - t * 0.2)
        val = _arch(archA, wu, wv, t, f1, f2, f3, ph)
        if mixmode > 0:
            vb = _arch(archB, wv, wu, t * 0.8, f2, f3, f1, ph + 1.7)
            if mixmode == 1:
                val = (val + vb) * 0.5
            elif mixmode == 2:
                val = val * vb
            elif mixmode == 3:
                val = ti.max(val, vb)
            else:
                val = val if vb > 0.0 else vb
        vn = val * 0.5 + 0.5
        b = 0.0
        if style == 0:
            b = 0.3 + 0.7 * vn
        elif style == 1:
            b = 1.0 - ti.min(1.0, ti.abs(val) * contrast)
        else:
            b = ti.floor(vn * levels) / levels
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
        canvas[i, j] = palette[ci] * bright
'''


@ti.data_oriented
class EffectLab(Effect):
    name = "Effect Lab"
    description = ("A generative engine: every Recipe is a structurally different "
                   "effect (lines, cells, fractals, clouds, waveforms…), audio-"
                   "reactive, doubling as a build-your-own video effect. The "
                   "equations are editable (Edit equations). No AI.")
    author = "Eyenips"
    uses_video = True
    # the editor finds the live instance through this to edit equations/recipe
    editable_kit = "effect_lab"

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
        self._kit = labkit.load()
        self._kernel = None
        self._good_kernel = None
        self._n_arch = 1
        self.error = ""
        self._dirty = True
        self._R = self._recipe(7)

    # ---- the editable kit (equations + recipe ranges) -------------------
    def set_kit(self, kit):
        """Called by the Edit-equations editor when the user applies changes.
        Recompiles the archetypes and re-rolls the current recipe next frame."""
        self._kit = kit
        self._dirty = True
        self._seed = None              # force the recipe to re-roll on the new kit

    def current_kit(self):
        return self._kit

    def _build_kernel(self, archetypes):
        """Compile the archetype equations from the kit into a live GPU kernel,
        the same way the Create-Effect editor compiles formulas. Raises
        ValueError (naming the offending archetype) on a bad formula."""
        body = labkit.archetype_source(None, archetypes)
        src = _KERNEL_SRC.replace("{ARCH}", body)
        ns = {}
        exec_with_source(src, ns, tag="vizstudio-effectlab")
        return ns["_k"]

    def _recompile(self):
        self._dirty = False
        arch = self._kit["effect_lab"]["archetypes"]
        self._n_arch = max(1, len(arch))
        try:
            self._kernel = self._build_kernel(arch)
            self._good_kernel = self._kernel
            self.error = ""
        except (ValueError, Exception) as e:
            self.error = str(e)
            self._kernel = self._good_kernel       # keep the last working one

    def _recipe(self, seed):
        """One integer -> a full structural recipe, using the kit's editable
        ranges/weights. Symmetry is usually OFF and the archetype is chosen
        freely, so recipes look categorically different."""
        rc = self._kit["effect_lab"]["recipe"]
        n = self._n_arch
        rng = _random.Random((int(seed) * 2654435761) & 0xFFFFFFFF)
        ch = rng.choice

        def ru(key):
            lo, hi = rc[key]
            return rng.uniform(lo, hi)

        return dict(
            archA=rng.randrange(n),
            archB=rng.randrange(n),
            mixmode=ch(rc["mix_weights"]),
            sym=ch(rc["sym_weights"]),
            nfold=float(ch(rc["nfold_choices"])),
            style=ch(rc["style_weights"]),
            colmode=ch(rc["colmode_weights"]),
            f1=ru("f1"), f2=ru("f2"), f3=ru("f3"),
            ph=rng.uniform(0.0, TWO_PI),
            warp=ch(rc["warp_choices"]) * rng.uniform(0.6, 1.4),
            fw1=ru("fw1"), fw2=ru("fw2"),
            colscale=ru("colscale"), coloff=ru("coloff"),
            bands=float(rng.randint(int(rc["bands"][0]), int(rc["bands"][1]))),
            rot=rng.uniform(0.0, TWO_PI),
            spd=rng.uniform(rc["spd"][0], rc["spd"][1]) * ch([-1.0, 1.0]),
            contrast=rng.uniform(rc["contrast"][0], rc["contrast"][1]),
            levels=float(rng.randint(int(rc["levels"][0]), int(rc["levels"][1]))))

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
        if self._dirty:
            self._recompile()
        p = ctx.p
        seed = int(p["recipe"])
        if seed != self._seed:
            self._R = self._recipe(seed)
            self._seed = seed
        if self._kernel is None:
            self.canvas.fill(0)       # no valid equations yet; UI shows self.error
            return
        R = self._R
        a = ctx.audio
        vol = float(a.volume) if a is not None else 0.0
        bass = float(a.bass) if a is not None else 0.0
        flow_amt = 1.0 if (p.get("flow_warp") and getattr(ctx, "has_video", False)) else 0.0
        t = ctx.time * float(p["speed"]) * R["spd"]

        try:
            self._kernel(
                self.canvas, self.palette, self.flow, self.w, self.h,
                t, R["archA"], R["archB"], R["mixmode"], R["sym"], R["nfold"],
                R["style"], R["colmode"], R["f1"], R["f2"], R["f3"], R["ph"],
                R["warp"] * float(p["warp"]), R["fw1"], R["fw2"], R["colscale"],
                R["coloff"], R["bands"], R["rot"], float(p["detail"]), R["contrast"],
                R["levels"], float(p["glow"]), float(p["react"]), vol, bass, flow_amt)
            self._good_kernel = self._kernel
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self._kernel = self._good_kernel
            if self._kernel is None:
                self.canvas.fill(0)
            return

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
