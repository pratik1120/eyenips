"""ShapesCompositor - draws the shape "elements" as a LAYER on top of the
active effect, so the shapes *interact* with the effect's pixels.

This is the whole point of shapes: they are not their own effect. Each frame,
after the effect (and post-FX / media) have drawn into the canvas, this takes a
snapshot of that image and, per pixel, lets every shape mask / cut / warp /
tint / overlay it:

    Show effect     paint a chosen effect inside the shape, over the main one
    Hide effect     the shape removes the effect inside it (a hole)
    Warp effect     the shape bends the effect like a lens (magnify / push)
    Tint effect     the shape colors the effect passing through it
    Glow/Fill/Outline/Ripples   the shape adds its own light/color on top

One data-driven kernel loops over a fixed array of shape rows (see
vizstudio/shapes.py `encode`), so adding or dragging a shape only rewrites
numbers - nothing recompiles.
"""

import taichi as ti

from .shapes import MAX_SHAPES, NF

TAU = 6.283185307179586


@ti.data_oriented
class ShapesCompositor:
    def __init__(self, canvas, palette):
        self.canvas = canvas
        self.palette = palette          # stable 256x3 color LUT (shared w/ engine)
        self.w, self.h = canvas.shape[0], canvas.shape[1]
        self.src = ti.Vector.field(3, ti.f32, shape=(self.w, self.h))  # snapshot
        self.data = ti.field(ti.f32, shape=(MAX_SHAPES, NF))
        # buffers a secondary effect renders into, so a "window" shape can show
        # a DIFFERENT effect than the primary one (slots 1..3).
        self.buf1 = ti.Vector.field(3, ti.f32, shape=(self.w, self.h))
        self.buf2 = ti.Vector.field(3, ti.f32, shape=(self.w, self.h))
        self.buf3 = ti.Vector.field(3, ti.f32, shape=(self.w, self.h))

    def buffer(self, slot):
        """The Taichi field a secondary effect (slot 1..3) renders into."""
        return (self.buf1, self.buf2, self.buf3)[slot - 1]

    def upload(self, np_rows):
        """np_rows: (MAX_SHAPES, NF) float array from shapes.encode()."""
        self.data.from_numpy(np_rows)

    @ti.func
    def _src_at(self, slot, i, j):
        """The effect-source pixel for a window shape: slot 0 = primary
        (the canvas snapshot), 1..3 = a secondary effect's buffer. Clamped to
        [0,1] so accumulating effects (trails) don't blow the window out."""
        res = self.src[i, j]
        if slot == 1:
            res = self.buf1[i, j]
        elif slot == 2:
            res = self.buf2[i, j]
        elif slot == 3:
            res = self.buf3[i, j]
        return ti.Vector([ti.min(ti.max(res[0], 0.0), 1.0),
                          ti.min(ti.max(res[1], 0.0), 1.0),
                          ti.min(ti.max(res[2], 0.0), 1.0)])

    @ti.func
    def _smp(self, u, v):
        """Bilinear sample of the pre-shapes snapshot at normalized (u, v)."""
        fx = u * (self.w - 1)
        fy = v * (self.h - 1)
        x0 = ti.cast(ti.floor(fx), ti.i32)
        y0 = ti.cast(ti.floor(fy), ti.i32)
        ax = fx - x0
        ay = fy - y0
        x0 = ti.max(0, ti.min(self.w - 1, x0))
        y0 = ti.max(0, ti.min(self.h - 1, y0))
        x1 = ti.min(self.w - 1, x0 + 1)
        y1 = ti.min(self.h - 1, y0 + 1)
        a = self.src[x0, y0] * (1 - ax) + self.src[x1, y0] * ax
        b = self.src[x0, y1] * (1 - ax) + self.src[x1, y1] * ax
        return a * (1 - ay) + b * ay

    @ti.func
    def _dist(self, s, x, y):
        """Signed distance from (x, y) to shape s's edge (<0 inside)."""
        geom = self.data[s, 1]
        lobes = self.data[s, 2]
        amp = self.data[s, 3]
        cx = self.data[s, 4]
        cy = self.data[s, 5]
        size = self.data[s, 6]
        rot = self.data[s, 7]
        dx = x - cx
        dy = y - cy
        d = 0.0
        if geom > 0.5:                      # box (rotatable square)
            a = rot * TAU
            c = ti.cos(a)
            sn = ti.sin(a)
            u = dx * c + dy * sn
            v = -dx * sn + dy * c
            d = ti.max(ti.abs(u) - size, ti.abs(v) - size)
        else:                               # round: circle / polygon / star
            rr = ti.sqrt(dx * dx + dy * dy)
            rad = size * (1.0 + amp * ti.cos(lobes * ti.atan2(dy, dx) - rot * TAU))
            d = rr - rad
        return d

    @ti.func
    def _react(self, s, bass, beat, vol, treble, kick, snare, hihat):
        react = ti.cast(self.data[s, 9], ti.i32)
        av = 0.0
        if react == 1:
            av = bass
        elif react == 2:
            av = beat
        elif react == 3:
            av = vol
        elif react == 4:
            av = treble
        elif react == 5:
            av = kick
        elif react == 6:
            av = snare
        elif react == 7:
            av = hihat
        return 1.0 + av * self.data[s, 10]

    @ti.kernel
    def composite(self, t: ti.f32, gain: ti.f32,
                  bass: ti.f32, beat: ti.f32, vol: ti.f32, treble: ti.f32,
                  kick: ti.f32, snare: ti.f32, hihat: ti.f32):
        for i, j in self.canvas:
            x = i / self.w
            y = j / self.h
            # the MAIN effect (Effects panel) is always the full-screen base;
            # every shape only changes pixels INSIDE itself, on top of it.
            base = self.src[i, j]

            for s in range(MAX_SHAPES):
                if self.data[s, 0] > 0.5:
                    mode = ti.cast(self.data[s, 8], ti.i32)
                    d = self._dist(s, x, y)
                    speed = self.data[s, 11]
                    amount = self.data[s, 12]
                    hue = self.data[s, 13]
                    rfac = self._react(s, bass, beat, vol, treble, kick, snare, hihat)
                    c = ti.min(1.0, ti.max(0.0, -d * 40.0) * rfac)   # inside-ness
                    ci = ti.cast((hue - ti.floor(hue)) * 255.0, ti.i32) % 256
                    col = self.palette[ci]

                    if mode == 0:               # Show: paint the chosen effect inside
                        slot = ti.cast(self.data[s, 14], ti.i32)
                        base = base * (1.0 - c) + self._src_at(slot, i, j) * c
                    elif mode == 1:             # Hide: cut a hole in the main effect
                        base = base * (1.0 - c * ti.min(1.0, amount))
                    elif mode == 2:             # Warp: lens (magnify the main effect)
                        k = c * amount * 0.6
                        su = x - (x - self.data[s, 4]) * k
                        sv = y - (y - self.data[s, 5]) * k
                        base = base * (1.0 - c) + self._smp(su, sv) * c
                    elif mode == 3:             # Tint: color the main effect inside
                        fac = (1.0 - c) + c * col * (0.4 + amount)
                        base = base * fac
                    else:                       # self-draw overlays (4..7)
                        b = 0.0
                        if mode == 4:           # Glow
                            b = ti.exp(-ti.abs(d) * 12.0)
                        elif mode == 5:         # Fill
                            b = ti.min(1.0, ti.max(0.0, -d * 60.0))
                        elif mode == 6:         # Outline
                            b = ti.max(0.0, 1.0 - ti.abs(d) * 70.0)
                        else:                   # Ripples (emitter)
                            b = (0.5 + 0.5 * ti.sin(d * 36.0 - t * speed)) \
                                * ti.exp(-ti.abs(d) * 5.0)
                        base = base + col * (b * rfac * amount * gain)

            self.canvas[i, j] = base

    def run(self, t, gain, feats):
        """Snapshot the canvas (the main effect), then composite shapes over it."""
        self.src.copy_from(self.canvas)
        bass = feats.bass if feats else 0.0
        beat = 1.0 if (feats and feats.beat) else 0.0
        vol = feats.volume if feats else 0.0
        treble = feats.treble if feats else 0.0
        kick = feats.kick if feats else 0.0
        snare = feats.snare if feats else 0.0
        hihat = feats.hihat if feats else 0.0
        self.composite(t, gain,
                       bass, beat, vol, treble, kick, snare, hihat)
