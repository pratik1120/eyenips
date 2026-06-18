"""The auto-pilot — choreography rules that make the show run itself.

Phase 2 turned the song into drive sources (intensity / build / drop) you can map
to any knob by hand. This is the hands-off layer on top: a couple of opt-in rules
that read the same analysis and steer the show automatically while you just design
the looks.

  • Auto-intensity master: brightness breathes with the song's energy and any
    feedback intensifies on the drop — zero wiring, press play.
  • Auto-switch: change palette or effect ON each drop, or EVERY N bars.

Everything is opt-in and saved with the project. It runs on the render thread via
apply(), which the engine calls each frame after resolving the params.
"""

from __future__ import annotations

ACTIONS = ["Off", "Switch palette", "Switch effect"]
TRIGGERS = ["on drop", "every N bars"]


class Director:
    def __init__(self):
        self.auto_intensity = False
        self.action = "Off"          # one of ACTIONS
        self.trigger = "on drop"     # one of TRIGGERS
        self.every_bars = 8
        # event-detection state (render thread)
        self._last_pos = 0.0
        self._last_block = None

    # ---- per-frame (render thread) -------------------------------------
    def apply(self, engine, p, dsig):
        """Steer the show for this frame. `p` is the resolved param dict (we may
        nudge it in place); `dsig` carries intensity/build/drop."""
        inten = float(dsig.get("intensity", 0.0))

        # (a) auto-intensity master — universal, safe, no wiring
        if self.auto_intensity:
            if "brightness" in p:
                p["brightness"] = min(3.0, p["brightness"] * (0.55 + 0.9 * inten))
            if p.get("feedback") and "fb_decay" in p:
                p["fb_decay"] = min(0.97, p["fb_decay"] + 0.25 * inten)

        # (b) discrete auto-switch on drop / every N bars
        if self.action != "Off" and self._fired(engine):
            self._do_action(engine)

    def _fired(self, engine):
        """True exactly on the frame a trigger condition is newly met."""
        if self.trigger == "on drop":
            pos = engine.audio.position() if engine.audio else 0.0
            st = engine.structure
            crossed = False
            if engine._has_song_map() and st is not None:
                if pos < self._last_pos:          # song restarted / seeked back
                    self._last_pos = 0.0
                for d in st.drops:
                    if self._last_pos < d <= pos:
                        crossed = True
                self._last_pos = pos
            return crossed
        # every N bars
        beats = engine._musical_beats()
        block = int(beats // (4 * max(1, self.every_bars)))
        first = self._last_block is None
        changed = (not first) and block != self._last_block
        self._last_block = block
        return changed

    def _do_action(self, engine):
        if self.action == "Switch palette":
            self._next_palette(engine)
        elif self.action == "Switch effect":
            self._next_effect(engine)

    def _next_palette(self, engine):
        from .params import ColorPalette
        for pr in engine.params:
            if isinstance(pr, ColorPalette):
                cur = engine.store.values.get(pr.name) or {}
                names = ColorPalette.NAMED
                i = names.index(cur["named"]) if cur.get("named") in names else -1
                engine.store.set(pr.name, {"named": names[(i + 1) % len(names)],
                                           "custom": list(cur.get("custom", []))})
                break

    def _next_effect(self, engine):
        classes = [c for c in engine.effect_catalog.values()
                   if not c.name.startswith("Blank")]
        if not classes or engine.effect is None:
            return
        names = [c.name for c in classes]
        i = names.index(engine.effect.name) if engine.effect.name in names else -1
        engine.request_effect(classes[(i + 1) % len(classes)])

    # ---- persistence ---------------------------------------------------
    def to_dict(self):
        return {"auto_intensity": self.auto_intensity, "action": self.action,
                "trigger": self.trigger, "every_bars": self.every_bars}

    def load_dict(self, data):
        if not data:
            return
        self.auto_intensity = bool(data.get("auto_intensity", self.auto_intensity))
        self.action = data.get("action", self.action)
        self.trigger = data.get("trigger", self.trigger)
        self.every_bars = int(data.get("every_bars", self.every_bars))
