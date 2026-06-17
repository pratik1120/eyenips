"""Project files (.viz) — save / load the whole creative state.

A project is just JSON: the active effect + its knobs, every shape and each
shape's effect knobs, the colors, audio bindings, theme and window layout. The
panel builds the dict (see ControlPanel._capture_state) and applies it back
(ControlPanel._apply_state); this module only does the file I/O so saving,
presets, the autosaved session, and undo/redo all share one format.
"""

import json
import os

VERSION = 1
EXT = ".viz"


def save(path, state):
    """Write `state` (a plain dict) to `path` as a .viz project."""
    out = dict(state)
    out["version"] = VERSION
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)        # atomic-ish: never leave a half-written project


def load(path):
    """Read a .viz project back into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_presets(folder):
    """Preset names (without extension) in `folder`, sorted."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder)
                  if f.endswith(EXT))
