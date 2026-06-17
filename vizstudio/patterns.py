"""Friendly, named "pattern blocks" -> formulas (the Scratch-like layer).

A true beginner shouldn't type math or see the word "sin". Instead they pick a
named shape ("Circles", "Waves", "Spiral"...), set Size and Speed, and choose
what it Reacts to (Bass / Beat / Volume...). Those choices are composed here
into the same expression strings the ExpressionEffect already understands, so
the whole "Build" tab is just a friendly front-end over the formula engine.

Templates are written in the expression language (which uses sin/etc. and the
variables x, y, r, theta, t, bass, beat, vol, treble). `{sz}` and `{sp}` are
filled with the slider values.
"""

# name -> brightness template. {sz}=Size, {sp}=Speed.
PATTERNS = {
    "Circles (rings)":  "sin(r*{sz} - t*{sp})",
    "Waves":            "sin(x*{sz} + t*{sp})",
    "Vertical waves":   "sin(y*{sz} + t*{sp})",
    "Diagonal waves":   "sin((x+y)*{sz} + t*{sp})",
    "Spiral":           "sin(theta*{sz} + r*{sz} - t*{sp})",
    "Starburst":        "sin(theta*{sz} - t*{sp})",
    "Ripples":          "sin(r*{sz}*3 - t*{sp}*2)",
    "Checkerboard":     "sin(x*{sz} + t*{sp}) * sin(y*{sz} - t*{sp})",
    "Tunnel":           "sin(1/(r+0.15)*{sz} - t*{sp})",
}
PATTERN_CHOICES = ["Off"] + list(PATTERNS)

# friendly "Reacts to" -> audio variable name (None = no reaction)
AUDIO_MAP = {"Nothing": None, "Bass": "bass", "Beat": "beat",
             "Volume": "vol", "Treble": "treble", "Highs": "treble",
             "Kick": "kick", "Snare": "snare", "Hi-hat": "hihat"}
REACT_CHOICES = ["Nothing", "Bass", "Beat", "Volume", "Treble",
                 "Kick", "Snare", "Hi-hat"]

# how the colors move
HUE_MODES = {
    "Rainbow drift":    "x*0.5 + y*0.3 + t*0.1",
    "Spin":             "theta*0.5 + t*0.1",
    "Match the shape":  "r*0.6 + t*0.05",
    "Pulse with bass":  "t*0.05 + bass*0.6",
    "Still":            "x*0.4 + y*0.4",
}
HUE_CHOICES = list(HUE_MODES)

COMBINE_CHOICES = ["Add (blend)", "Multiply (mask)"]


def _num(v):
    """Format a slider value as a short, unambiguous literal."""
    return f"{float(v):.3f}".rstrip("0").rstrip(".") or "0"


def build_formulas(layers, combine="Add (blend)", hue_mode="Rainbow drift"):
    """Compose friendly layer choices into (brightness, hue) expression strings.

    layers: list of dicts with keys:
        pattern, size, speed, reverse(bool), react, react_strength, amount.
    """
    parts = []
    for L in layers:
        pat = L.get("pattern", "Off")
        if pat == "Off" or pat not in PATTERNS:
            continue
        sp = float(L.get("speed", 2)) * (-1.0 if L.get("reverse") else 1.0)
        expr = PATTERNS[pat].format(sz=_num(L.get("size", 8)), sp=_num(sp))
        audio = AUDIO_MAP.get(L.get("react", "Nothing"))
        if audio:
            expr = f"({expr}) * (1 + {audio}*{_num(L.get('react_strength', 1.5))})"
        amount = float(L.get("amount", 1.0))
        if abs(amount - 1.0) > 1e-3:
            expr = f"({expr}) * {_num(amount)}"
        parts.append(expr)

    if not parts:
        bright = "0.0"
    elif combine.startswith("Multiply"):
        bright = "*".join(f"({p})" for p in parts)
    else:  # Add (blend) - average so the range stays sensible
        scale = _num(1.4 / len(parts))
        bright = "(" + " + ".join(f"({p})" for p in parts) + f") * {scale}"

    hue = HUE_MODES.get(hue_mode, HUE_MODES["Rainbow drift"])
    return bright, hue
