"""Shape "elements": shared metadata + numeric encoding for the shapes overlay.

Shapes are *placed objects* (a circle/star/box at an x,y) that sit as a LAYER on
top of whatever effect is running and *interact* with it — masking, cutting,
warping, or tinting the effect's pixels (see MODE_CHOICES). They're drawn by one
data-driven Taichi compositor (vizstudio/shapes_fx.py) that loops over a fixed
array of shape "slots": each shape is just a row of numbers, so moving one
(dragging it on the canvas) only changes a number — no recompile, smooth drag.

This module is the shared vocabulary between that compositor and the Shapes
panel: the shape names, the interaction modes, the "Reacts to" sources, and
`encode()`, which turns one shape dict into the float row the kernel reads.
"""

# name -> (geom, lobes, wobble-amplitude)
#   geom 0 = round (circle / polygon / star via a wobbling radius)
#   geom 1 = box   (rotatable square)
SHAPES = {
    "Circle":   (0, 0, 0.00),
    "Square":   (1, 4, 0.00),
    "Triangle": (0, 3, 0.26),
    "Pentagon": (0, 5, 0.13),
    "Hexagon":  (0, 6, 0.07),
    "Star":     (0, 5, 0.42),
    "Flower":   (0, 6, 0.32),
    "Diamond":  (0, 2, 0.40),
}
SHAPE_CHOICES = list(SHAPES)

# what the shape DOES to the effect underneath it. The first four are
# "interactions" (they read & change the effect's pixels); the last four are
# self-draw overlays (the shape adds its own light/color on top of the effect).
MODE_CHOICES = [
    "Show effect",       # 0: paint the chosen effect inside the shape (over main)
    "Hide effect",       # 1: shape punches a hole in the effect (cutout)
    "Warp effect",       # 2: shape bends the effect like a lens
    "Tint effect",       # 3: shape colors the effect passing through it
    "Glow",              # 4: add a soft colored halo over the effect
    "Fill",              # 5: solid shape color over the effect
    "Outline",           # 6: shape-colored edge over the effect
    "Ripples",           # 7: rings radiating from the edge, over the effect
]
MODE_IDS = {name: i for i, name in enumerate(MODE_CHOICES)}
SHOW_ID = 0
# modes <= this index are "interactions"; above are self-draw overlays
LAST_INTERACTION_ID = 3
# modes that show an effect *inside* the shape (so they can pick WHICH effect)
WINDOW_MODES = {"Show effect"}
# how many DISTINCT secondary effects shapes may show at once (+ the primary)
MAX_FX_SLOTS = 3
PRIMARY = "Primary"


def is_window(mode):
    return mode in WINDOW_MODES

# friendly "Reacts to" -> kernel id (and the audio band it reads)
REACT_CHOICES = ["Nothing", "Bass", "Beat", "Volume", "Treble",
                 "Kick", "Snare", "Hi-hat"]
REACT_IDS = {name: i for i, name in enumerate(REACT_CHOICES)}

# how many shapes can exist at once (one kernel array row each)
MAX_SHAPES = 24
# columns of one encoded shape row (must match the kernel's reads)
NF = 15


def default_shape(**kw):
    """A fresh shape dict with sensible defaults; override via kwargs."""
    s = dict(shape="Circle", x=0.5, y=0.5, size=0.18, rotation=0.0,
             mode="Show effect", react="Nothing", strength=1.5, speed=2.0,
             amount=1.0, hue=0.0, effect=PRIMARY)
    s.update(kw)
    return s


def encode(shape, slot=0):
    """Turn one shape dict into a length-NF float row for the kernel.

    slot: which effect source a window-shape shows — 0 = the primary effect,
    1..MAX_FX_SLOTS = a secondary effect buffer (resolved by the engine).
    """
    geom, lobes, amp = SHAPES.get(shape.get("shape", "Circle"), SHAPES["Circle"])
    return [
        1.0,                                              # 0 active
        float(geom),                                      # 1 geom (0 round/1 box)
        float(lobes),                                     # 2 lobes
        float(amp),                                       # 3 wobble amplitude
        float(shape.get("x", 0.5)),                       # 4 center x  (0..1)
        float(shape.get("y", 0.5)),                       # 5 center y  (0..1)
        float(shape.get("size", 0.18)),                   # 6 size
        float(shape.get("rotation", 0.0)),                # 7 rotation (turns)
        float(MODE_IDS.get(shape.get("mode", "Show effect"), 0)),  # 8 mode id
        float(REACT_IDS.get(shape.get("react", "Nothing"), 0)),      # 9 react id
        float(shape.get("strength", 1.5)),                # 10 react strength
        float(shape.get("speed", 2.0)),                   # 11 emitter speed
        float(shape.get("amount", 1.0)),                  # 12 amount/strength
        float(shape.get("hue", 0.0)),                     # 13 color (0..1 in palette)
        float(slot),                                      # 14 effect-source slot
    ]
