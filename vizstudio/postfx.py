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
