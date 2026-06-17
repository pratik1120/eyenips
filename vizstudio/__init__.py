"""Eyenips - a beginner-friendly, plugin-based audio visualization studio.

Public API for effect authors:

    from vizstudio import Effect, Slider, IntSlider, Toggle, Choice, ColorPalette

Drop a new effect file in the `effects/` folder, subclass `Effect`, declare
your `params`, and it shows up in the app automatically.
"""

from .params import Slider, IntSlider, Toggle, Choice, ColorPalette, Param
from .effect import Effect, Context

__all__ = [
    "Effect",
    "Context",
    "Param",
    "Slider",
    "IntSlider",
    "Toggle",
    "Choice",
    "ColorPalette",
]
