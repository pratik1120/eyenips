"""Plugin discovery: find every Effect subclass in the `effects/` folder.

This is the "open-source / drop-a-file" door. Any `.py` in effects/ that
defines an `Effect` subclass shows up automatically - no registration, no
imports to edit. That's how a power user adds total freedom without touching
the core.
"""

from __future__ import annotations

import importlib.util
import os
import traceback

from .effect import Effect


def discover(effects_dir):
    """Return a list of (Effect subclass) found in effects_dir, plus errors.

    Returns (effects, errors) where errors is a list of (filename, message).
    """
    effects = []
    errors = []
    if not os.path.isdir(effects_dir):
        return effects, errors

    for fn in sorted(os.listdir(effects_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(effects_dir, fn)
        modname = "vizeffect_" + os.path.splitext(fn)[0]
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            errors.append((fn, traceback.format_exc()))
            continue

        for attr in vars(mod).values():
            if (isinstance(attr, type) and issubclass(attr, Effect)
                    and attr is not Effect
                    # only classes DEFINED in this file, not imported bases
                    # (e.g. ExpressionEffectBase) that happen to be in scope
                    and getattr(attr, "__module__", None) == mod.__name__):
                effects.append(attr)

    return effects, errors
