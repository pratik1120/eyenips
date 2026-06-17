"""Color palettes -> a 256-entry RGB lookup table (LUT).

Everything color-related collapses to "build a 256x3 array". The engine uploads
that array to the GPU once whenever the user changes colors, and every effect
samples it. That's why arbitrary multi-color gradients are free for all effects.
"""

from __future__ import annotations

import numpy as np

N = 256  # LUT resolution


def _hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
    except ValueError:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _gradient(stops):
    """stops: list of (pos0..1, (r,g,b)). Returns (256,3) float32 LUT."""
    stops = sorted(stops, key=lambda s: s[0])
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    t = np.linspace(0.0, 1.0, N)
    lut = np.empty((N, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(t, xs, cols[:, c])
    return lut


_NAMED = {
    "rainbow": lambda: _hsv_sweep(),
    "fire":    lambda: _gradient([(0, (0, 0, 0)), (0.4, (0.6, 0, 0)),
                                  (0.7, (1, 0.5, 0)), (1, (1, 1, 0.6))]),
    "ocean":   lambda: _gradient([(0, (0, 0.02, 0.1)), (0.5, (0, 0.4, 0.6)),
                                  (1, (0.6, 0.95, 1))]),
    "plasma":  lambda: _gradient([(0, (0.05, 0, 0.3)), (0.4, (0.6, 0, 0.5)),
                                  (0.7, (1, 0.4, 0.2)), (1, (1, 1, 0.4))]),
    "mono":    lambda: _gradient([(0, (0, 0, 0)), (1, (1, 1, 1))]),
    "sunset":  lambda: _gradient([(0, (0.1, 0, 0.2)), (0.4, (0.8, 0.2, 0.3)),
                                  (0.7, (1, 0.5, 0.2)), (1, (1, 0.85, 0.5))]),
    "ice":     lambda: _gradient([(0, (0, 0, 0.1)), (0.5, (0.2, 0.6, 0.9)),
                                  (1, (0.9, 1, 1))]),
}


def _hsv_sweep():
    t = np.linspace(0.0, 1.0, N, endpoint=False)
    lut = np.empty((N, 3), dtype=np.float32)
    for i, h in enumerate(t):
        lut[i] = _hsv(h, 0.85, 1.0)
    return lut


def _hsv(h, s, v):
    h = (h % 1.0) * 6.0
    c = v * s
    x = c * (1 - abs(h % 2 - 1))
    m = v - c
    if h < 1:   r, g, b = c, x, 0
    elif h < 2: r, g, b = x, c, 0
    elif h < 3: r, g, b = 0, c, x
    elif h < 4: r, g, b = 0, x, c
    elif h < 5: r, g, b = x, 0, c
    else:       r, g, b = c, 0, x
    return (r + m, g + m, b + m)


def build_lut(spec):
    """spec: {"named": str, "custom": [hex,...]}  ->  (256,3) float32.

    If the user picked 2+ custom colors, build a gradient through them.
    Otherwise use the named palette.
    """
    custom = [c for c in spec.get("custom", []) if c]
    if len(custom) >= 2:
        cols = [_hex_to_rgb(c) for c in custom]
        stops = [(i / (len(cols) - 1), tuple(col)) for i, col in enumerate(cols)]
        return _gradient(stops)
    if len(custom) == 1:
        # single accent color over black
        return _gradient([(0, (0, 0, 0)), (1, tuple(_hex_to_rgb(custom[0])))])
    return _NAMED.get(spec.get("named", "rainbow"), _hsv_sweep)()
