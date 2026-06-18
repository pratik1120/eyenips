"""Post-processing stack: global look toggles applied AFTER any effect renders.

Because these run on the final canvas, every effect - including ones users
write later - gets "grainy / fluid / flicker / trails / fade" for free. The
effect author does nothing; the user just flips a switch.

The master params declared here always appear in the control panel, on top of
whatever knobs the active effect declares.
"""

import taichi as ti

from .params import Toggle, Slider


# These show up in the UI for EVERY effect (the "look" section).
def global_params():
    return [
        Toggle("trails", default=True, help="Let previous frames linger (motion smear)."),
        Slider("trail_length", 0.0, 0.99, default=0.9, audio=False,
               help="How long trails persist. Higher = longer smear."),
        Toggle("fluid", default=False, help="Blur the image into a liquid, flowing look."),
        Slider("fluid_amount", 0.0, 1.0, default=0.5, audio=True,
               help="Strength of the liquid blur."),
        Toggle("grainy", default=False, help="Add film-grain / sandy texture."),
        Slider("grain_amount", 0.0, 1.0, default=0.3, audio=True,
               help="Amount of grain noise."),
        Toggle("flicker", default=False, help="Randomly pulse the brightness."),
        Slider("flicker_amount", 0.0, 1.0, default=0.3, audio=True,
               help="How violent the flicker is."),
        Toggle("fade_in", default=True,
               help="Ramp up from black when the effect starts."),
        Slider("brightness", 0.1, 3.0, default=1.0, audio=True,
               help="Master brightness."),
        Toggle("feedback", default=False,
               help="Feed the previous frame back in, zoomed/spun — infinite "
                    "tunnels, spirals and echoes. Drive zoom on the bass!"),
        Slider("fb_decay", 0.0, 0.98, default=0.85, audio=True,
               help="How strongly the previous frame persists. Higher = longer, "
                    "brighter echoes (too high can bloom to white)."),
        Slider("fb_zoom", 0.80, 1.25, default=1.02, audio=True,
               help="Scale the fed-back frame each step. >1 tunnels outward, "
                    "<1 sucks inward, 1.0 holds."),
        Slider("fb_rotate", -8.0, 8.0, default=0.0, audio=True,
               help="Spin the fed-back frame each step (degrees) — spirals."),
    ]


@ti.data_oriented
class PostFX:
    def __init__(self, width, height, canvas):
        self.w = width
        self.h = height
        self.canvas = canvas
        self.buffer = ti.Vector.field(3, ti.f32, shape=(width, height))

    @ti.kernel
    def decay(self, factor: ti.f32):
        for i, j in self.canvas:
            self.canvas[i, j] *= factor

    @ti.kernel
    def blur(self, amount: ti.f32):
        # 3x3 box blur into buffer, then lerp back by `amount`
        for i, j in self.canvas:
            acc = ti.Vector([0.0, 0.0, 0.0])
            cnt = 0.0
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    x = i + di
                    y = j + dj
                    if 0 <= x < self.w and 0 <= y < self.h:
                        acc += self.canvas[x, y]
                        cnt += 1.0
            self.buffer[i, j] = acc / cnt
        for i, j in self.canvas:
            self.canvas[i, j] = self.canvas[i, j] * (1 - amount) + self.buffer[i, j] * amount

    @ti.kernel
    def grain(self, amount: ti.f32):
        for i, j in self.canvas:
            n = (ti.random() - 0.5) * amount
            self.canvas[i, j] += ti.Vector([n, n, n])

    @ti.kernel
    def scale(self, mult: ti.f32):
        for i, j in self.canvas:
            self.canvas[i, j] *= mult

    def apply(self, p, elapsed):
        """Run the enabled passes. `p` is the resolved param dict."""
        if p.get("fluid") and p.get("fluid_amount", 0) > 0:
            self.blur(float(p["fluid_amount"]))
        if p.get("grainy") and p.get("grain_amount", 0) > 0:
            self.grain(float(p["grain_amount"]))

        mult = float(p.get("brightness", 1.0))
        if p.get("flicker") and p.get("flicker_amount", 0) > 0:
            import random
            mult *= 1.0 - random.random() * float(p["flicker_amount"])
        if p.get("fade_in"):
            mult *= min(1.0, elapsed / 2.0)  # 2-second ramp from black
        if abs(mult - 1.0) > 1e-3:
            self.scale(mult)


@ti.data_oriented
class Feedback:
    """Frame feedback: blends the PREVIOUS composited frame back into this one,
    transformed (zoom + rotate around the center) and decayed. That recursion is
    what makes the classic infinite-tunnel / spiral / echo looks. It runs as a
    post-pass on the final canvas (not by relying on an effect leaving pixels
    untouched), so it works on ANY effect, including overwriting ones."""

    def __init__(self, width, height, canvas):
        self.w = width
        self.h = height
        self.canvas = canvas
        self.prev = ti.Vector.field(3, ti.f32, shape=(width, height))

    @ti.func
    def _sample(self, u, v):
        """Bilinear sample of the previous frame at normalized (u, v); anything
        outside [0,1] reads as black so the tunnel fades into darkness."""
        out = ti.Vector([0.0, 0.0, 0.0])
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            fx = u * (self.w - 1)
            fy = v * (self.h - 1)
            x0 = ti.cast(ti.floor(fx), ti.i32)
            y0 = ti.cast(ti.floor(fy), ti.i32)
            ax = fx - x0
            ay = fy - y0
            x1 = ti.min(self.w - 1, x0 + 1)
            y1 = ti.min(self.h - 1, y0 + 1)
            a = self.prev[x0, y0] * (1 - ax) + self.prev[x1, y0] * ax
            b = self.prev[x0, y1] * (1 - ax) + self.prev[x1, y1] * ax
            out = a * (1 - ay) + b * ay
        return out

    @ti.kernel
    def apply(self, zoom: ti.f32, angle: ti.f32, decay: ti.f32):
        ca = ti.cos(angle)
        sa = ti.sin(angle)
        inv = 1.0 / ti.max(zoom, 1e-3)
        for i, j in self.canvas:
            du = i / (self.w - 1) - 0.5      # output pixel offset from center
            dv = j / (self.h - 1) - 0.5
            # sample the previous frame at the inverse transform (rotate -angle,
            # shrink by 1/zoom) so zoom>1 pushes content outward (a tunnel).
            su = (ca * du + sa * dv) * inv + 0.5
            sv = (-sa * du + ca * dv) * inv + 0.5
            self.canvas[i, j] += self._sample(su, sv) * decay

    @ti.kernel
    def save(self):
        for i, j in self.canvas:
            self.prev[i, j] = self.canvas[i, j]

    @ti.kernel
    def clear(self):
        for i, j in self.prev:
            self.prev[i, j] = ti.Vector([0.0, 0.0, 0.0])
