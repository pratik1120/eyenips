"""Turn a user's friendly math expression into safe Taichi kernel source.

A non-coder types something like:   sin(x*8 + t) + bass*3
We:
  * reject anything that isn't plain math (no '=', ';', quotes, attribute
    access, unknown names) - this both prevents code injection and gives a
    clear "unknown name: foo" message for typos, and
  * rewrite the friendly function names (sin, cos, ...) to their Taichi
    equivalents (ti.sin, ...).

The result is a string that can be dropped straight into a generated kernel.
"""

import re
import linecache
from itertools import count

_exec_counter = count()


def exec_with_source(src, namespace, tag="vizstudio-dynamic"):
    """exec() code while keeping its source discoverable.

    Taichi's @ti.kernel reads a function's source via inspect.getsource(), which
    fails for plain exec()'d code ("OSError: Cannot find source code"). Registering
    the source in linecache under a unique filename makes inspection work, so
    kernels built at runtime (live formulas, the code editor) compile correctly.
    Returns the synthetic filename used.
    """
    fname = f"<{tag}-{next(_exec_counter)}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    exec(compile(src, fname, "exec"), namespace)
    return fname

# Variables the user may reference (provided by the kernel at each pixel).
VARS = ["x", "y", "r", "theta", "t", "scale",
        "bass", "mid", "treble", "vol", "beat",
        "kick", "snare", "hihat",          # drum-tuned audio bands
        # media (camera/image/video) sampled at this pixel:
        "tex", "texr", "texg", "texb", "motion"]

# Friendly function name -> Taichi function.
FUNCS = {
    "sin": "ti.sin", "cos": "ti.cos", "tan": "ti.tan",
    "asin": "ti.asin", "acos": "ti.acos", "atan2": "ti.atan2",
    "sqrt": "ti.sqrt", "exp": "ti.exp", "log": "ti.log",
    "floor": "ti.floor", "ceil": "ti.ceil", "abs": "ti.abs",
    "min": "ti.min", "max": "ti.max",
}
CONSTS = {"pi": "3.141592653589793"}

# Only these characters may appear (identifiers handled separately below).
_LEGAL_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyz"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ_+-*/%(),. \t")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def translate(expr, allowed_vars=None, extra_funcs=None):
    """Return Taichi source for `expr`. Raises ValueError on anything illegal.

    `extra_funcs` maps friendly helper names to their generated names (e.g.
    {"vor": "_vor"}), so callers like the Lab Kit editor can expose extra
    building-block functions on top of the standard math vocabulary."""
    allowed = set(allowed_vars or VARS)
    funcs = FUNCS if not extra_funcs else {**FUNCS, **extra_funcs}
    expr = (expr or "").strip()
    if not expr:
        return "0.0"

    bad = [c for c in expr if c not in _LEGAL_CHARS]
    if bad:
        raise ValueError(f"illegal character: '{bad[0]}'")

    def repl(m):
        name = m.group(0)
        if name in funcs:
            return funcs[name]
        if name in CONSTS:
            return CONSTS[name]
        if name in allowed:
            return name
        raise ValueError(f"unknown name: '{name}'  "
                         f"(allowed: {', '.join(sorted(allowed))})")

    return _IDENT.sub(repl, expr)


def cheat_sheet():
    return (
        "Variables you can use:\n"
        "  x, y      position 0..1 (left/bottom = 0)\n"
        "  r, theta  distance & angle from center\n"
        "  t         time in seconds\n"
        "  scale     the Scale knob\n"
        "  bass mid treble vol   audio levels 0..1\n"
        "  beat      1 on a beat, else 0\n"
        "  kick snare hihat   drum-band hits 0..1\n"
        "  tex       camera/image brightness here 0..1\n"
        "  texr texg texb   media color channels\n"
        "  motion    how much the camera moved here 0..1\n"
        "Functions:\n"
        "  sin cos tan sqrt abs floor ceil exp log min max atan2 pi\n"
        "Math: + - * / %  ** (power)\n"
        "Example:  sin(x*scale + t) + bass*4"
    )
