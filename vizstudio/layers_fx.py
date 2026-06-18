"""Layer compositing — blend a stack of effects like Photoshop layers.

The main effect (the Effect panel) is the bottom of the stack, rendered into the
canvas as always. Each *additional* layer is a full effect with its own knobs,
look and palette, rendered into its own buffer (exactly like the per-shape
secondary effects) and then BLENDED onto the canvas here with a blend mode and
opacity. So "run Plasma, then Screen a Liquid-Fractal on top at 60%%" becomes a
two-row stack — and because every layer is just another effect instance, all the
existing machinery (audio/LFO drive, post-FX, palettes) works per layer for free.
"""

import taichi as ti

# blend modes, in the order shown in the UI. Index = kernel id.
LAYER_BLENDS = ["Normal", "Add", "Screen", "Multiply", "Lighten", "Difference"]
LAYER_BLEND_IDS = {name: i for i, name in enumerate(LAYER_BLENDS)}
# how many effect layers can sit ON TOP of the main effect at once
MAX_LAYERS = 4


@ti.data_oriented
class LayerCompositor:
    """Blends one layer buffer onto the canvas with a mode + opacity."""

    def __init__(self, canvas):
        self.canvas = canvas

    @ti.func
    def _clamp(self, v):
        return ti.Vector([ti.min(ti.max(v[0], 0.0), 1.0),
                          ti.min(ti.max(v[1], 0.0), 1.0),
                          ti.min(ti.max(v[2], 0.0), 1.0)])

    @ti.kernel
    def blend(self, src: ti.template(), mode: ti.i32, opacity: ti.f32):
        for i, j in self.canvas:
            b = self.canvas[i, j]          # base (may exceed 1 from additive fx)
            t = src[i, j]
            tc = self._clamp(t)
            bc = self._clamp(b)
            out = tc                       # 0 Normal
            if mode == 1:                  # Add
                out = b + tc
            elif mode == 2:                # Screen
                out = ti.Vector([1.0 - (1.0 - bc[0]) * (1.0 - tc[0]),
                                 1.0 - (1.0 - bc[1]) * (1.0 - tc[1]),
                                 1.0 - (1.0 - bc[2]) * (1.0 - tc[2])])
            elif mode == 3:                # Multiply
                out = ti.Vector([bc[0] * tc[0], bc[1] * tc[1], bc[2] * tc[2]])
            elif mode == 4:                # Lighten
                out = ti.Vector([ti.max(b[0], tc[0]), ti.max(b[1], tc[1]),
                                 ti.max(b[2], tc[2])])
            elif mode == 5:                # Difference
                out = ti.Vector([ti.abs(bc[0] - tc[0]), ti.abs(bc[1] - tc[1]),
                                 ti.abs(bc[2] - tc[2])])
            self.canvas[i, j] = b * (1.0 - opacity) + out * opacity
