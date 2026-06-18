"""MIDI input -> modulation sources (optional).

A hardware knob or fader is just another driver. Each of the 8 MIDI slots tracks
one CC controller (value 0..127 -> 0..1) and shows up in every knob's 'drive'
menu as MIDI 1..8 — exactly like an LFO or an audio band, because they all flow
through the same signal table (see ModEngine). Uses `mido` with whatever backend
loads — `python-rtmidi` if available, else `pygame` (PortMidi, which has wheels
everywhere); without any backend the MIDI panel shows 'unavailable' and the rest
of the app runs untouched.

Assign a slot to a controller with **Learn**: click Learn, wiggle the knob, and
the next CC that arrives is captured into that slot.
"""

from __future__ import annotations

import threading
import time

N_MIDI = 8
MIDI_IDS = [f"midi{i + 1}" for i in range(N_MIDI)]
MIDI_LABELS = {f"midi{i + 1}": f"MIDI {i + 1}" for i in range(N_MIDI)}

# mido needs a backend to reach hardware. python-rtmidi is best (low latency)
# but has no prebuilt wheel on some Pythons; pygame ships PortMidi and wheels
# everywhere, so we fall back to it. We pick the first that actually loads.
_BACKENDS = ["mido.backends.rtmidi", "mido.backends.pygame", "mido.backends.portmidi"]
_mido = None          # cached working module, or None
_mido_probed = False


def _get_mido():
    """The mido module with a backend that actually loads on this machine, or
    None. Probed once and cached (switching backends mid-run is messy)."""
    global _mido, _mido_probed
    if _mido_probed:
        return _mido
    _mido_probed = True
    try:
        import mido
    except Exception:
        return None
    for be in _BACKENDS:
        try:
            mido.set_backend(be, load=True)
            mido.get_input_names()          # probe: backend loads + can enumerate
            _mido = mido
            return _mido
        except Exception:
            continue
    return None


class MidiEngine:
    """Owns the (optional) input port and maps 8 slots to CC controllers.

    Thread-safe: a small poll thread drains the port (works with any backend),
    the UI edits slots, and the render loop reads `values()` each frame.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cc = {}                    # cc number -> latest value 0..1
        self._slots = [None] * N_MIDI    # slot index -> cc number (or None)
        self._learn = -1                 # slot awaiting the next CC, or -1
        self._port = None
        self._port_name = None
        self._stop = threading.Event()
        self._thread = None
        self.status = "off"

    # ---- availability / ports ------------------------------------------
    def available(self):
        return _get_mido() is not None

    def ports(self):
        mido = _get_mido()
        if mido is None:
            return []
        try:
            return list(mido.get_input_names())
        except Exception:
            return []

    # ---- connection ----------------------------------------------------
    def open(self, name):
        """Open `name` and poll its CC messages into the slots. Polling (rather
        than a callback) works across every mido backend, not just rtmidi."""
        self.close()
        mido = _get_mido()
        if mido is None or not name:
            return False
        try:
            self._port = mido.open_input(name)
        except Exception as e:
            self.status = f"error: {type(e).__name__}"
            self._port = None
            return False
        self._port_name = name
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.status = f"on · {name}"
        return True

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        self.status = "off"

    def is_open(self):
        return self._port is not None

    def _run(self):
        """Drain pending messages a few hundred times a second."""
        while not self._stop.is_set():
            port = self._port
            if port is None:
                break
            try:
                for msg in port.iter_pending():
                    self._cb(msg)
            except Exception:
                break
            time.sleep(0.003)

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
