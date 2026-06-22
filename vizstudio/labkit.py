"""Lab Kit — the editable 'genome' behind Effect Lab and Video Lab.

The generative labs don't hide their math: the archetype EQUATIONS and the
recipe RANGES that turn one number into millions of effects live here as plain
data, so the in-app editor can show them, let you tweak/add to them, and reset
back to these defaults.

* Effect Lab archetypes are real formula strings (compiled live through the same
  expression engine the Create-Effect editor uses). You can edit them and add
  your own — referencing u, v, t, f1, f2, f3, ph, the usual math functions, and
  the helper building blocks vor(), julia(), fbm(), vnoise(), round().
* Recipe ranges/weights control how a Recipe number is rolled into a concrete
  effect (frequency ranges, symmetry odds, colour modes, operator sparsity …).

The user's edits are saved to ~/.eyenips/lab_kit.json; DEFAULTS below is the
pristine copy the Reset button restores.
"""

import copy
import json
import os

from .exprutil import translate

# helper building blocks an archetype formula may call (friendly -> generated)
ARCH_FUNCS = {"vor": "_vor", "julia": "_julia", "fbm": "_fbm",
              "vnoise": "_vnoise", "round": "ti.round"}
# variables an archetype formula may reference
ARCH_VARS = ["u", "v", "t", "f1", "f2", "f3", "ph"]


# --------------------------------------------------------------------------
# DEFAULTS — the pristine kit. Edit the app, not this, to customise; Reset
# restores exactly this.
# --------------------------------------------------------------------------
DEFAULTS = {
    "effect_lab": {
        "archetypes": [
            {"name": "plasma",
             "formula": "(sin(u*f1+t) + sin(v*f2+t*0.7+ph) + sin((u+v)*f3-t*0.5))*0.33"},
            {"name": "lines",
             "formula": "sin((u*cos(ph)+v*sin(ph))*f1*4 - t*2)"},
            {"name": "grid",
             "formula": "sin(u*f1*4+t)*sin(v*f2*4-t)"},
            {"name": "rings",
             "formula": "sin(sqrt(u*u+v*v)*f1*5 - t*2)"},
            {"name": "spiral",
             "formula": "sin(sqrt(u*u+v*v)*f1*4 + atan2(v,u)*round(f2*2+1) - t*2)"},
            {"name": "voronoi",
             "formula": "vor(u*f1+8, v*f1+8, t)*2 - 1"},
            {"name": "julia",
             "formula": "julia(u,v,f1,f2,ph,t)*2 - 1"},
            {"name": "moire",
             "formula": ("sin((u*cos(ph)+v*sin(ph))*f1*5) * "
                         "sin((u*cos(ph+1.3)+v*sin(ph+1.3))*f2*5)")},
            {"name": "cloud",
             "formula": "fbm(u*f1+t*0.1, v*f1-t*0.05)*2 - 1"},
            {"name": "waveform",
             "formula": ("(1 - min(1, abs(v - (sin(u*f1*4+t*2)+sin(u*f2*7-t))*0.25)*9))"
                         "*2 - 1")},
            {"name": "halftone",
             "formula": ("(0.34 + 0.22*sin(t + floor(u*5*f1) + floor(v*5*f1)) "
                         "- sqrt((u*5*f1-floor(u*5*f1)-0.5)**2 + "
                         "(v*5*f1-floor(v*5*f1)-0.5)**2)) * 5")},
            {"name": "contour",
             "formula": "sin(fbm(u*f1+t*0.1, v*f1)*12)"},
        ],
        # recipe ranges/weights — how one Recipe number becomes an effect
        "recipe": {
            "f1": [0.6, 4.0], "f2": [0.6, 4.0], "f3": [0.5, 3.0],
            "warp_choices": [0.0, 0.0, 0.0, 0.4, 0.8, 1.2],
            "fw1": [1.0, 4.0], "fw2": [1.0, 4.0],
            "colscale": [0.2, 1.6], "coloff": [0.0, 1.0],
            "bands": [2, 7], "contrast": [5.0, 16.0], "levels": [2, 6],
            "spd": [0.3, 1.6],
            "sym_weights": [0, 0, 0, 0, 0, 1, 2, 3],     # mostly NO symmetry
            "nfold_choices": [3, 4, 5, 6, 8],
            "style_weights": [0, 0, 1, 1, 2],            # smooth / line / flats
            "colmode_weights": [0, 0, 0, 1, 2, 3],
            "mix_weights": [0, 0, 0, 1, 2, 3, 4],        # mostly single archetype
        },
    },
    "video_lab": {
        # which operators are in the random pool (the math itself is built in)
        "operators": {
            "melt": True, "swirl": True, "wave": True, "noise": True,
            "kale": True, "mirror": True, "tile": True,
            "chroma": True, "edge": True, "poster": True, "duo": True,
            "invert": True,
            "halo": True, "spark": True, "trail": True, "link": True, "flow": True,
        },
        "recipe": {
            "dominant": [2, 4], "accent": [0, 2],
            "dominant_weight": [0.5, 1.0], "accent_weight": [0.1, 0.35],
            "fold": [3.0, 8.0], "tiles": [2.0, 9.0], "woff": [4.0, 18.0],
            "levels": [2, 6], "colscale": [0.3, 1.5], "coloff": [0.0, 1.0],
            "fw1": [2.0, 6.0], "fw2": [2.0, 6.0], "edgeg": [3.0, 9.0],
            "ovsize": [0.5, 1.5], "disp": [0.12, 0.4], "spd": [0.4, 1.5],
        },
    },
}

# operator groups (warp / colour / overlay) — for the editor's layout & the
# recipe's "make sure there's some motion overlay" rule
VIDEO_GROUPS = {
    "warp": ["melt", "swirl", "wave", "noise", "kale", "mirror", "tile"],
    "color": ["chroma", "edge", "poster", "duo", "invert"],
    "over": ["halo", "spark", "trail", "link", "flow"],
}


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def _kit_dir():
    from . import paths
    return paths.user_data_dir()


def _kit_path():
    return os.path.join(_kit_dir(), "lab_kit.json")


def defaults():
    """A deep copy of the pristine kit (safe to mutate)."""
    return copy.deepcopy(DEFAULTS)


def _merge(base, over):
    """Deep-merge user overrides onto defaults so a partial/old saved file still
    gets any new default keys (forward-compatible)."""
    out = copy.deepcopy(base)
    if not isinstance(over, dict):
        return out
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load():
    """Return the effective kit: user edits merged over the defaults. Never
    raises — a corrupt/missing file just yields the defaults."""
    try:
        with open(_kit_path(), "r", encoding="utf-8") as f:
            user = json.load(f)
        return _merge(DEFAULTS, user)
    except Exception:
        return defaults()


def save(kit):
    """Persist the full kit to ~/.eyenips/lab_kit.json. Returns (ok, message)."""
    try:
        os.makedirs(_kit_dir(), exist_ok=True)
        with open(_kit_path(), "w", encoding="utf-8") as f:
            json.dump(kit, f, indent=2)
        return True, _kit_path()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def reset():
    """Delete the user kit so the labs fall back to DEFAULTS. Returns (ok, msg)."""
    try:
        p = _kit_path()
        if os.path.exists(p):
            os.remove(p)
        return True, "reset to defaults"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validate_formula(expr):
    """Return "" if `expr` is a legal archetype formula, else an error string."""
    try:
        translate(expr, ARCH_VARS, ARCH_FUNCS)
        return ""
    except ValueError as e:
        return str(e)


def archetype_source(name_to_gen, archetypes):
    """Build the body of the dynamic `_arch` dispatch from the archetype list:
    one `if/elif mode == k:` branch per archetype, each assigning the translated
    formula. Raises ValueError (with the archetype name) on a bad formula."""
    lines = ["    val = 0.0"]
    for k, a in enumerate(archetypes):
        try:
            code = translate(a.get("formula", "0"), ARCH_VARS, ARCH_FUNCS)
        except ValueError as e:
            raise ValueError(f"archetype '{a.get('name', k)}': {e}")
        kw = "if" if k == 0 else "elif"
        lines.append(f"    {kw} mode == {k}:")
        lines.append(f"        val = ti.cast({code}, ti.f32)")
    lines.append("    return val")
    return "\n".join(lines)
