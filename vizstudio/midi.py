"""MIDI input -> modulation sources (optional).

A hardware knob or fader is just another driver. Each of the 8 MIDI slots tracks
one CC controller (value 0..127 -> 0..1) and shows up in every knob's 'drive'
menu as MIDI 1..8 — exactly like an LFO or an audio band, because they all flow
through the same signal table (see ModEngine). Uses `mido` (+ `python-rtmidi`)
if installed; without them the MIDI panel shows 'unavailable' and the rest of the
app runs untouched.

Assign a slot to a controller with **Learn**: click Learn, wiggle the knob, and
the next CC that arrives is captured into that slot.
"""

from __future__ import annotations

import threading

N_MIDI = 8
MIDI_IDS = [f"midi{i + 1}" for i in range(N_MIDI)]
MIDI_LABELS = {f"midi{i + 1}": f"MIDI {i + 1}" for i in range(N_MIDI)}


class MidiEngine:
    """Owns the (optional) input port and maps 8 slots to CC controllers.

    Thread-safe: rtmidi delivers messages on its own thread (via a callback),
    the UI edits slots, and the render loop reads `values()` each frame.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cc = {}                    # cc number -> latest value 0..1
        self._slots = [None] * N_MIDI    # slot index -> cc number (or None)
        self._learn = -1                 # slot awaiting the next CC, or -1
        self._port = None
        self._port_name = None
        self.status = "off"

    # ---- availability / ports ------------------------------------------
    def available(self):
        try:
            import mido  # noqa: F401
            return True
        except Exception:
            return False

    def ports(self):
        try:
            import mido
            return list(mido.get_input_names())
        except Exception:
            return []

    # ---- connection ----------------------------------------------------
    def open(self, name):
        """Open `name` and stream its CC messages into the slots. mido calls our
        callback on rtmidi's own thread, so there's no loop to manage."""
        self.close()
        if not name:
            return False
        try:
            import mido
            self._port = mido.open_input(name, callback=self._cb)
        except Exception as e:
            self.status = f"error: {type(e).__name__}"
            self._port = None
            return False
        self._port_name = name
        self.status = f"on · {name}"
        return True

    def close(self):
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self.status = "off"

    def is_open(self):
        return self._port is not None

    def _cb(self, msg):
        if getattr(msg, "type", None) == "control_change":
            self._on_cc(msg.control, msg.value / 127.0)

    # ---- message handling (also the test seam) -------------------------
    def _on_cc(self, cc, value01):
        with self._lock:
            self._cc[cc] = float(value01)
            if self._learn >= 0:
                self._slots[self._learn] = int(cc)
                self._learn = -1

    # ---- slots / learn (UI thread) -------------------------------------
    def learn(self, slot):
        with self._lock:
            self._learn = slot

    def cancel_learn(self):
        with self._lock:
            self._learn = -1

    def clear_slot(self, slot):
        with self._lock:
            self._slots[slot] = None

    def slot_info(self, slot):
        with self._lock:
            cc = self._slots[slot]
            val = self._cc.get(cc, 0.0) if cc is not None else 0.0
            return dict(cc=cc, value=val, learning=(self._learn == slot))

    # ---- per-frame signal table ----------------------------------------
    def values(self):
        """{midi1..midi8: 0..1} from each slot's assigned CC (0 if unassigned)."""
        out = {}
        with self._lock:
            for i, cc in enumerate(self._slots):
                out[MIDI_IDS[i]] = self._cc.get(cc, 0.0) if cc is not None else 0.0
        return out

    # ---- persistence ---------------------------------------------------
    def to_dict(self):
        with self._lock:
            return {"port": self._port_name, "slots": list(self._slots)}

    def load_dict(self, data):
        if not data:
            return
        with self._lock:
            slots = data.get("slots") or []
            for i in range(min(len(slots), N_MIDI)):
                self._slots[i] = slots[i]
        self._port_name = data.get("port")     # remembered; UI offers Connect
