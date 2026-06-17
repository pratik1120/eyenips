"""ExpressionEffectBase - an effect whose visuals are two math formulas.

The user supplies a `BRIGHT` formula (how bright each pixel is) and a `HUE`
formula (which color from the palette). Per pixel we evaluate both and write
`palette[hue] * bright`. Formulas can reference position, time, and audio, so
this single effect covers an enormous range with zero real coding.

Both the live editor and every "Save as effect" file are just thin subclasses
that set `name`, `BRIGHT`, and `HUE`. The kernel is rebuilt from those strings.
"""

import taichi as ti

from .effect import Effect
from .params import Slider, ColorPalette
from .exprutil import translate, VARS, exec_with_source


@ti.data_oriented
class ExpressionEffectBase(Effect):
    name = "Expression Effect"
    description = "Visuals defined by two math formulas."

    # Subclasses / the live editor override these.
    BRIGHT = "sin(x*scale + t) + sin(y*scale - t) + bass*4"
    HUE = "x*0.5 + y*0.3 + t*0.05 + bass*0.5"
    PALETTE = "plasma"   # saved block/formula effects bake their colors here
    SOURCE = "paint"     # "paint" (palette) | "texture" (show media) | "warp" (distort media)

    VARS = VARS

    @property
    def params(self):
        # a property (not a class list) so a saved subclass can pick its colors
        # just by setting PALETTE = "fire" etc.
        return [
            Slider("scale", 1.0, 30.0, default=8.0, audio=False,
                   help="Spatial frequency knob (use it as 'scale' in formulas)."),
            ColorPalette(default=self.PALETTE),
        ]

    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas
        self.palette = ctx.palette
        self.mediaf = ctx.media          # media + motion fields the kernel samples
        self.motionf = ctx.media_motion
        self.bright_expr = self.BRIGHT
        self.hue_expr = self.HUE
        self.source = self.SOURCE
        self.error = ""
        self._kernel = None
        self._good_kernel = None
        self._dirty = True

    def set_formulas(self, bright, hue, source=None):
        """Called live from the editor. Recompiles on the next frame."""
        self.bright_expr = bright
        self.hue_expr = hue
        if source is not None:
            self.source = source
        self._dirty = True

    # ---- kernel (re)building -------------------------------------------
    def _build_kernel(self, bright_src, hue_src, source):
        # how the per-pixel result is produced from the two formulas + media
        if source == "texture":      # show the media, modulated by `bright`
            out = "        col = mediaf[i, j] * bright\n"
        elif source == "warp":       # displace the media by the formulas
            out = ("        col = _smp(mediaf, W, H, x + bright*0.15, y + hue*0.15)\n")
        else:                         # "paint": color from the palette
            out = ("        hh = hue - ti.floor(hue)\n"
                   "        ci = ti.cast(hh * 255.0, ti.i32) % 256\n"
                   "        col = palette[ci] * bright\n")
        src = (
            "import taichi as ti\n"
            "@ti.func\n"
            "def _smp(media, W, H, u, v):\n"          # bilinear media sample
            "    fx = u * (W - 1)\n"
            "    fy = v * (H - 1)\n"
            "    x0 = ti.cast(ti.floor(fx), ti.i32)\n"
            "    y0 = ti.cast(ti.floor(fy), ti.i32)\n"
            "    ax = fx - x0\n"
            "    ay = fy - y0\n"
            "    x0 = ti.max(0, ti.min(W - 1, x0))\n"
            "    y0 = ti.max(0, ti.min(H - 1, y0))\n"
            "    x1 = ti.min(W - 1, x0 + 1)\n"
            "    y1 = ti.min(H - 1, y0 + 1)\n"
            "    a = media[x0, y0] * (1 - ax) + media[x1, y0] * ax\n"
            "    b = media[x0, y1] * (1 - ax) + media[x1, y1] * ax\n"
            "    return a * (1 - ay) + b * ay\n"
            "@ti.kernel\n"
            "def _k(canvas: ti.template(), palette: ti.template(),\n"
            "       mediaf: ti.template(), motionf: ti.template(),\n"
            "       W: ti.i32, H: ti.i32, t: ti.f32, scale: ti.f32,\n"
            "       bass: ti.f32, mid: ti.f32, treble: ti.f32,\n"
            "       vol: ti.f32, beat: ti.f32,\n"
            "       kick: ti.f32, snare: ti.f32, hihat: ti.f32):\n"
            "    for i, j in canvas:\n"
            "        x = i / W\n"
            "        y = j / H\n"
            "        dx = x - 0.5\n"
            "        dy = y - 0.5\n"
            "        r = ti.sqrt(dx*dx + dy*dy)\n"
            "        theta = ti.atan2(dy, dx)\n"
            "        mc = mediaf[i, j]\n"             # media-derived variables
            "        tex = (mc[0] + mc[1] + mc[2]) * 0.33333\n"
            "        texr = mc[0]\n"
            "        texg = mc[1]\n"
            "        texb = mc[2]\n"
            "        motion = motionf[i, j]\n"
            f"        bright = ti.cast({bright_src}, ti.f32)\n"
            f"        hue = ti.cast({hue_src}, ti.f32)\n"
            + out +
            "        canvas[i, j] = col\n"
        )
        ns = {}
        exec_with_source(src, ns, tag="vizstudio-expr")
        return ns["_k"]

    def _recompile(self):
        self._dirty = False
        try:
            b = translate(self.bright_expr, self.VARS)
            h = translate(self.hue_expr, self.VARS)
            self._kernel = self._build_kernel(b, h, self.source)
            self.error = ""
        except ValueError as e:
            self.error = str(e)
            self._kernel = self._good_kernel  # keep last working one

    def render(self, ctx):
        if self._dirty:
            self._recompile()
        if self._kernel is None:
            return  # nothing valid yet; UI shows self.error
        a = ctx.audio
        bass = a.bass if a else 0.0
        mid = a.mid if a else 0.0
        treble = a.treble if a else 0.0
        vol = a.volume if a else 0.0
        beat = 1.0 if (a and a.beat) else 0.0
        kick = a.kick if a else 0.0
        snare = a.snare if a else 0.0
        hihat = a.hihat if a else 0.0
        try:
            self._kernel(self.canvas, self.palette, self.mediaf, self.motionf,
                         self.w, self.h, ctx.time, float(ctx.p.get("scale", 8.0)),
                         bass, mid, treble, vol, beat, kick, snare, hihat)
            self._good_kernel = self._kernel  # it ran -> it's good
        except Exception as e:
            # a formula that compiles textually but Taichi rejects
            self.error = f"{type(e).__name__}: {e}"
            self._kernel = self._good_kernel
            if self._kernel is None:
                self.canvas.fill(0)
