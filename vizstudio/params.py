"""Self-describing parameters.

An effect declares a list of these. From that declaration the app builds the
entire control panel automatically (sliders, toggles, dropdowns, color
pickers) AND lets numeric knobs be "driven" by the audio (bass / mids /
treble / volume / beat). The same declaration is the single source of truth
for both the UI and the values the effect reads at render time.

Effect authors never touch the UI. They just say what knobs exist.
"""

from __future__ import annotations

# The audio features a numeric param can be bound ("driven") by.
AUDIO_SOURCES = ["none", "volume", "bass", "mid", "treble", "beat",
                 "kick", "snare", "hihat"]


class Param:
    """Base class. Not used directly - use one of the concrete kinds below."""

    kind = "param"

    def __init__(self, name, label=None, default=None, help=None):
        self.name = name
        self.label = label or name.replace("_", " ").title()
        self.default = default
        self.help = help or ""

    def coerce(self, value):
        """Normalize a raw value coming from the UI / a preset file."""
        return value


class Slider(Param):
    """A floating-point knob with a min/max range. Can be audio-driven."""

    kind = "slider"

    def __init__(self, name, lo, hi, default=None, step=None, label=None,
                 help=None, audio=True, drive=None):
        super().__init__(name, label, default if default is not None else lo, help)
        self.lo = float(lo)
        self.hi = float(hi)
        self.step = step if step is not None else (hi - lo) / 200.0
        self.audio = audio  # may this knob be driven by audio?
        # optional default audio binding, e.g. drive=("bass", 0.8) so the knob
        # reacts to the music out of the box. User can still change it.
        self.drive_source = "none"
        self.drive_amount = 0.5
        if drive:
            self.drive_source, self.drive_amount = drive[0], float(drive[1])

    def coerce(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.default
        return max(self.lo, min(self.hi, v))


class IntSlider(Slider):
    """An integer knob. Same as Slider but values are whole numbers."""

    kind = "int"

    def __init__(self, name, lo, hi, default=None, label=None, help=None,
                 audio=True, drive=None):
        super().__init__(name, lo, hi, default, step=1, label=label, help=help,
                         audio=audio, drive=drive)

    def coerce(self, value):
        return int(round(super().coerce(value)))


class Toggle(Param):
    """An on/off switch."""

    kind = "toggle"

    def __init__(self, name, default=False, label=None, help=None):
        super().__init__(name, label, bool(default), help)

    def coerce(self, value):
        return bool(value)


class Choice(Param):
    """A dropdown of named options."""

    kind = "choice"

    def __init__(self, name, options, default=None, label=None, help=None):
        super().__init__(name, label, default if default is not None else options[0], help)
        self.options = list(options)

    def coerce(self, value):
        return value if value in self.options else self.default


class ColorPalette(Param):
    """A color choice. Offers named gradients plus up to a few custom colors.

    Resolves (in the engine) to a 256-entry RGB lookup table the GPU samples,
    so any effect gets arbitrary multi-color gradients for free.
    """

    kind = "palette"

    NAMED = ["rainbow", "fire", "ocean", "plasma", "mono", "sunset", "ice"]

    def __init__(self, name="palette", default="rainbow", label=None, help=None):
        super().__init__(name, label or "Colors", default, help)

    def coerce(self, value):
        # value is a dict: {"named": str, "custom": [hex, ...]}  OR a bare name
        if isinstance(value, str):
            return {"named": value, "custom": []}
        if isinstance(value, dict):
            return {"named": value.get("named", self.default),
                    "custom": list(value.get("custom", []))}
        return {"named": self.default, "custom": []}
