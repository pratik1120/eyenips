"""The engine: owns the canvas, the colors, the audio, the post-FX, and the
render loop. It ties an Effect together with everything it needs.

Single-threaded by design: the render loop runs on the main thread and, each
frame, hands the rendered image to a `display` callback (the Tk preview) and
calls an `on_frame` callback (to pump the Tk controls). So the whole app -
visuals and controls - lives in one window. The UI only mutates a thread-safe
ParamStore; the engine reads it each frame and is the only thing that talks
to Taichi.
"""

import math
import threading
import time

import numpy as np
import taichi as ti

from . import color
from . import shapes as shapelib
from .effect import Context
from .params import Slider, ColorPalette
from .postfx import PostFX, Feedback, global_params
from .media import media_params, BLEND_IDS, MAX_BLOBS
from .shapes_fx import ShapesCompositor
from .modulation import ModEngine
from .midi import MidiEngine
from .tempo import TempoEngine, phases_from_beats
from .director import Director
from .layers_fx import LayerCompositor, LAYER_BLEND_IDS, MAX_LAYERS


class ParamStore:
    """Thread-safe bag of current knob values + per-knob audio bindings."""

    def __init__(self):
        self._lock = threading.Lock()
        self.values = {}          # name -> value
        self.audio_src = {}       # name -> "none"/"bass"/... (sliders only)
        self.audio_amt = {}       # name -> 0..1
        self.params = {}          # name -> Param object (meta)

    def load(self, params):
        with self._lock:
            for p in params:
                self.params[p.name] = p
                self.values.setdefault(p.name, p.coerce(p.default))
                if isinstance(p, Slider):
                    self.audio_src.setdefault(p.name, p.drive_source)
                    self.audio_amt.setdefault(p.name, p.drive_amount)

    def set(self, name, value):
        with self._lock:
            p = self.params.get(name)
            self.values[name] = p.coerce(value) if p else value

    def set_audio(self, name, source=None, amount=None):
        with self._lock:
            if source is not None:
                self.audio_src[name] = source
            if amount is not None:
                self.audio_amt[name] = float(amount)

    def snapshot(self):
        with self._lock:
            return (dict(self.values), dict(self.audio_src), dict(self.audio_amt))

    def to_dict(self):
        """JSON-serializable copy of this store's knob values + audio bindings."""
        with self._lock:
            return {"values": dict(self.values),
                    "audio_src": dict(self.audio_src),
                    "audio_amt": dict(self.audio_amt)}

    def load_dict(self, d):
        """Apply saved values/bindings onto an already-loaded store."""
        with self._lock:
            for name, val in (d.get("values") or {}).items():
                p = self.params.get(name)
                self.values[name] = p.coerce(val) if p else val
            self.audio_src.update(d.get("audio_src") or {})
            self.audio_amt.update({k: float(v) for k, v in (d.get("audio_amt") or {}).items()})


@ti.data_oriented
class MediaCompositor:
    """Owns the media frame + camera-motion fields, computes motion, and (for
    the optional 'Background' mode) blends media with the canvas."""

    def __init__(self, canvas, media, motion, prev, mask):
        self.canvas = canvas
        self.media = media
        self.motion = motion     # ti.field(f32): camera-motion magnitude 0..1
        self.prev = prev         # ti.field(f32): previous-frame luma
        self.mask = mask         # ti.field(f32): subject (person) mask 0..1
        self.w, self.h = canvas.shape[0], canvas.shape[1]

    @ti.func
    def _smp(self, u, v):
        """Bilinear sample of the media at normalized (u, v), clamped to edges."""
        fx = u * (self.w - 1)
        fy = v * (self.h - 1)
        x0 = ti.cast(ti.floor(fx), ti.i32)
        y0 = ti.cast(ti.floor(fy), ti.i32)
        ax = fx - x0
        ay = fy - y0
        x0 = ti.max(0, ti.min(self.w - 1, x0))
        y0 = ti.max(0, ti.min(self.h - 1, y0))
        x1 = ti.min(self.w - 1, x0 + 1)
        y1 = ti.min(self.h - 1, y0 + 1)
        a = self.media[x0, y0] * (1 - ax) + self.media[x1, y0] * ax
        b = self.media[x0, y1] * (1 - ax) + self.media[x1, y1] * ax
        return a * (1 - ay) + b * ay

    @ti.kernel
    def update_motion(self, gain: ti.f32, decay: ti.f32):
        for i, j in self.media:
            m = self.media[i, j]
            luma = (m[0] + m[1] + m[2]) * 0.33333
            d = ti.abs(luma - self.prev[i, j]) * gain
            self.motion[i, j] = ti.max(self.motion[i, j] * decay, ti.min(d, 1.0))
            self.prev[i, j] = luma

    @ti.kernel
    def composite(self, mode: ti.i32, a: ti.f32, br: ti.f32):
        one = ti.Vector([1.0, 1.0, 1.0])
        for i, j in self.canvas:
            E = self.canvas[i, j]
            M = self.media[i, j] * br
            Ec = ti.Vector([ti.min(ti.max(E[0], 0.0), 1.0),
                            ti.min(ti.max(E[1], 0.0), 1.0),
                            ti.min(ti.max(E[2], 0.0), 1.0)])
            out = E
            if mode == 1:          # Behind: media shows, effect glows on top
                out = M * a + E
            elif mode == 2:        # Tint: effect colors/lights the media
                fac = ti.Vector([(1.0 - a) + a * Ec[0],
                                 (1.0 - a) + a * Ec[1],
                                 (1.0 - a) + a * Ec[2]])
                out = M * fac
            elif mode == 3:        # Screen: lighten media by the effect
                sc = one - (one - M) * (one - Ec)
                out = M * (1.0 - a) + sc * a
            elif mode == 4:        # Warp: distort the media using the effect's colors
                u = i / self.w + (Ec[0] - 0.5) * a * 0.4
                v = j / self.h + (Ec[1] - 0.5) * a * 0.4
                out = self._smp(u, v) * br
            self.canvas[i, j] = out

    @ti.kernel
    def overlay_subject(self, opacity: ti.f32, feather: ti.f32):
        """Paste the masked SUBJECT (the clean media frame) in FRONT of whatever
        is on the canvas, so the effect/layers play BEHIND the person. The mask
        is a soft alpha; `feather` controls edge hardness (0 = crisp, 1 = soft)."""
        k = 1.0 / (feather * 0.9 + 0.05)        # 0 -> hard edge, 1 -> soft edge
        for i, j in self.canvas:
            a = (self.mask[i, j] - 0.5) * k + 0.5
            a = ti.min(ti.max(a, 0.0), 1.0) * opacity
            self.canvas[i, j] = self.canvas[i, j] * (1.0 - a) + self.media[i, j] * a


class Pointer:
    """The live mouse over the preview, so effects can be interactive. Coords are
    normalized 0..1 with y UP (matching the canvas); dx/dy are per-frame velocity."""
    __slots__ = ("x", "y", "dx", "dy", "down", "active", "_px", "_py")

    def __init__(self):
        self.x = self.y = 0.5
        self.dx = self.dy = 0.0
        self.down = False
        self.active = False
        self._px = self._py = 0.5

    def tick(self):
        """Called once per frame by the engine to compute velocity from motion."""
        self.dx = self.x - self._px
        self.dy = self.y - self._py
        self._px = self.x
        self._py = self.y


@ti.data_oriented
class Presenter:
    """Packs the final float canvas into an image-oriented uint8 RGB buffer on
    the GPU — transpose (x,y)->(row,col), vertical flip, clamp to [0,1] and
    scale to bytes, all in one kernel. The UI then reads back a buffer that's a
    quarter of the float canvas' size and needs zero per-pixel CPU work, so the
    preview blit stops being the bottleneck. buf is laid out (H rows, W cols, 3)
    exactly like the image the UI used to build on the CPU."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.w, self.h = canvas.shape[0], canvas.shape[1]
        self.buf = ti.field(ti.u8, shape=(self.h, self.w, 3))

    @ti.kernel
    def pack(self):
        for x, y in self.canvas:
            row = self.h - 1 - y                  # flip: canvas y-up -> image top-down
            v = self.canvas[x, y]
            for c in ti.static(range(3)):
                u = ti.min(ti.max(v[c], 0.0), 1.0)
                self.buf[row, x, c] = ti.cast(u * 255.0 + 0.5, ti.u8)


class Engine:
    def __init__(self, width=1280, height=800, audio=None, media=None):
        self.w = width
        self.h = height
        self.audio = audio
        self.media = media
        self.canvas = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.palette = ti.Vector.field(3, ti.f32, shape=256)
        self.media_field = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.motion_field = ti.field(ti.f32, shape=(width, height))
        self._prev_luma = ti.field(ti.f32, shape=(width, height))
        self.mask_field = ti.field(ti.f32, shape=(width, height))   # subject mask
        self.postfx = PostFX(width, height, self.canvas)
        # GPU "present" path: pack the canvas to a uint8 image the UI can blit
        # directly. If it can't be built on this backend we fall back to the old
        # float readback in run(), so the app still works no matter what.
        try:
            self.presenter = Presenter(self.canvas)
        except Exception:
            self.presenter = None
        self.media_fx = MediaCompositor(self.canvas, self.media_field,
                                        self.motion_field, self._prev_luma,
                                        self.mask_field)
        # frame feedback (infinite tunnels / spirals / echoes) — a post-pass
        # that recursively blends the previous composited frame back in.
        self.feedback = Feedback(width, height, self.canvas)
        # shapes are a LAYER over the active effect (mask / warp / tint / overlay)
        self.shapes_fx = ShapesCompositor(self.canvas, self.palette)
        # modulation: LFOs (and MIDI) that can drive any knob — merged with the
        # audio bands into one signal table each frame.
        self.mods = ModEngine()
        self.midi = MidiEngine()      # optional hardware controllers (CC -> drive)
        self.tempo = TempoEngine()    # beat clock -> musical-time drive sources
        self._prev_beat = False       # rising-edge detect to feed the beat clock
        # Music Director: an offline song analysis (intensity / build / drops)
        # that drives the show from the track's structure.
        self.structure = None
        self._structure_path = None
        self._structure_status = ""
        self._live_inten = 0.0        # live-audio intensity fallback (no song map)
        self.director = Director()    # auto-pilot: choreography rules on the song
        self._shapes_np = np.zeros((shapelib.MAX_SHAPES, shapelib.NF), np.float32)
        self._shapes_count = 0
        # per-shape "show a different effect": name->class catalog + the live
        # secondary effect instances (slot 1..MAX_FX_SLOTS), each a full effect
        # with its OWN params, look/post-FX and palette, rendering into a buffer.
        self.effect_catalog = {}      # effect name -> class
        self._fx_wanted = {}          # slot -> (shape_id, effect_name)
        self._fx_active = {}          # slot -> dict(slot,id,name,inst,ctx,store,...)
        self._fx_dirty = False
        # each slot gets its own palette LUT so its colors are independent
        self._fx_pal = [ti.Vector.field(3, ti.f32, shape=256)
                        for _ in range(shapelib.MAX_FX_SLOTS)]
        # remember each shape-effect's knob values across slot renumbering
        self._fx_store_cache = {}     # (shape_id, effect_name) -> ParamStore
        # --- layer stack: extra effects blended ON TOP of the main effect ---
        # each layer is a full effect rendered into its own buffer, then blended
        # onto the canvas (mode + opacity). Mirrors the secondary-effect setup.
        self.layer_fx = LayerCompositor(self.canvas)
        self._layer_bufs = [ti.Vector.field(3, ti.f32, shape=(width, height))
                            for _ in range(MAX_LAYERS)]
        self._layer_pals = [ti.Vector.field(3, ti.f32, shape=256)
                            for _ in range(MAX_LAYERS)]
        self._layers_meta = []        # live list: dict(id, effect, blend, opacity, visible)
        self._layers_active = []      # built layer slots (dicts), bottom -> top
        self._layers_key = None       # structural key -> when to rebuild
        self._layers_dirty = False
        self._layer_store_cache = {}  # (layer_id, effect_name) -> ParamStore
        self.ctx = Context(width, height, self.canvas, self.palette)
        # media fields are stable refs effects can sample any time
        self.ctx.media = self.media_field
        self.ctx.media_motion = self.motion_field
        # video analysis fields (optical flow + tracked blobs) for "video effects"
        self.flow_field = ti.Vector.field(2, ti.f32, shape=(width, height))
        self.blob_field = ti.field(ti.f32, shape=(MAX_BLOBS, 5))
        self.ctx.flow = self.flow_field
        self.ctx.blobs = self.blob_field
        self.ctx.max_blobs = MAX_BLOBS
        # live pointer (the mouse over the preview) -> interactive effects
        self.pointer = Pointer()
        self.ctx.pointer = self.pointer
        self._media_cleared = True
        self.store = ParamStore()
        if self.media is not None:
            self.media.set_target(width, height)

        self.effect = None
        self.params = []           # merged: global + effect params
        self.effect_version = 0    # bumped on every effect swap (UI watches this)
        self._pending_effect = None
        self._pending_reset = False
        self._pending_export = None
        self._export_cancel = False    # set by cancel_export() to stop a render
        self._pending_expr = None
        self._pending_shapes = False  # shape array changed -> re-upload
        self.effect_error = ""     # last error from a (user) effect, for the UI
        self._last_palette = None
        self._start = time.perf_counter()
        self._last = self._start
        self.paused = False
        self.running = True

    # ---- effect management ---------------------------------------------
    def set_effect(self, effect):
        """Make `effect` active. Never crashes the app: a broken user effect is
        caught, reported via self.effect_error, and the previous effect kept."""
        prev = self.effect
        prev_params, prev_store = self.params, self.store
        self.params = global_params() + media_params() + list(effect.params)
        self.store = ParamStore()
        self.store.load(self.params)
        self._last_palette = None
        self._upload_palette(force=True)
        try:
            effect.setup(self.ctx)
        except Exception as e:
            self.effect_error = f"setup failed: {type(e).__name__}: {e}"
            # roll back to the previous effect so the app keeps running
            self.effect = prev
            self.params, self.store = prev_params, prev_store
            if prev is not None:
                self._upload_palette(force=True)
            return False
        self.effect = effect
        self.effect_error = getattr(effect, "error", "")
        self._clear_canvas()
        self.feedback.clear()        # don't carry the old effect's echoes over
        self._start = time.perf_counter()
        self.effect_version += 1
        return True

    def request_effect(self, effect_cls_or_instance):
        """Swap effect on the engine thread. Accepts a class OR an instance
        (the live code editor builds instances)."""
        self._pending_effect = effect_cls_or_instance

    def request_reset(self):
        self._pending_reset = True

    def request_export(self, config):
        """config: dict(audio_path, out_path, fps, seconds, progress)."""
        self._export_cancel = False
        self._pending_export = config

    def cancel_export(self):
        """Ask the running export to stop after the current frame (safe to call
        any time; the export loop checks this flag each frame)."""
        self._export_cancel = True

    def apply_expression(self, bright, hue, source=None):
        """Live-update the active expression effect's formulas (engine thread)."""
        self._pending_expr = (bright, hue, source)

    def set_effect_catalog(self, classes):
        """Tell the engine which effects exist (name -> class), so a shape can
        show a secondary effect by name. Safe to call any time."""
        self.effect_catalog = {c.name: c for c in classes}

    # ---- save / load (engine-owned state) ------------------------------
    def export_state(self):
        """The engine's slice of a project: the active effect, its knobs, and
        every per-shape effect's knobs."""
        fx = []
        for (sid, name), store in self._fx_store_cache.items():
            d = store.to_dict(); d["id"] = sid; d["effect"] = name
            fx.append(d)
        layers = []
        for m in self._layers_meta:
            d = dict(id=m.get("id"), effect=m.get("effect"),
                     blend=m.get("blend", "Normal"), opacity=m.get("opacity", 1.0),
                     visible=m.get("visible", True))
            store = self._layer_store_cache.get((m.get("id"), m.get("effect")))
            if store is not None:
                d["params"] = store.to_dict()
            layers.append(d)
        return {
            "effect": self.effect.name if self.effect else None,
            "params": self.store.to_dict(),
            "shape_fx": fx,
            "mods": self.mods.to_dict(),
            "midi": self.midi.to_dict(),
            "tempo": self.tempo.to_dict(),
            "director": self.director.to_dict(),
            "layers": layers,
        }

    def import_state(self, data):
        """Restore the engine's slice of a project (effect + knobs + shape-fx
        knob caches). Runs on the render thread (single-threaded app)."""
        name = data.get("effect")
        cls = self.effect_catalog.get(name)
        if cls is not None and self.set_effect(cls()):
            self.store.load_dict(data.get("params") or {})
            self._upload_palette(force=True)
        self.mods.load_dict(data.get("mods"))
        self.midi.load_dict(data.get("midi"))
        self.tempo.load_dict(data.get("tempo"))
        self.director.load_dict(data.get("director"))
        # rebuild the per-shape effect knob caches so reconcile restores them
        cache = {}
        for entry in data.get("shape_fx") or []:
            sid, ename = entry.get("id"), entry.get("effect")
            ecls = self.effect_catalog.get(ename)
            if ecls is None:
                continue
            inst = ecls()
            store = ParamStore()
            store.load(global_params() + list(inst.params))
            store.load_dict(entry)
            cache[(sid, ename)] = store
        self._fx_store_cache = cache
        self._fx_wanted = {}        # force reconcile to rebuild instances
        self._fx_active = {}
        self._fx_dirty = True
        # restore the layer stack (each layer's effect + its own knobs)
        lcache = {}
        meta = []
        for entry in data.get("layers") or []:
            lid, ename = entry.get("id"), entry.get("effect")
            ecls = self.effect_catalog.get(ename)
            if ecls is None:
                continue
            inst = ecls()
            store = ParamStore()
            store.load(global_params() + list(inst.params))
            store.load_dict(entry.get("params") or {})
            lcache[(lid, ename)] = store
            meta.append(dict(id=lid, effect=ename, blend=entry.get("blend", "Normal"),
                             opacity=entry.get("opacity", 1.0),
                             visible=entry.get("visible", True)))
        self._layer_store_cache = lcache
        self._layers_meta = meta
        self._layers_active = []
        self._layers_key = None     # force reconcile to rebuild instances
        self._layers_dirty = True

    def request_shapes(self, shape_dicts):
        """Set the shapes overlay (drawn on top of whatever effect is active).
        Each "Show effect" shape that shows a non-Primary effect gets its OWN
        render slot (so two shapes can show the same effect tuned differently),
        up to MAX_FX_SLOTS. Encodes (cheap) and flags work for the render thread."""
        shape_dicts = shape_dicts or []
        wanted = {}                       # slot -> (shape_id, effect_name)
        slot_by_index = {}                # shape index -> slot
        next_slot = 1
        for idx, s in enumerate(shape_dicts):
            if not shapelib.is_window(s.get("mode", "")):
                continue
            name = s.get("effect", shapelib.PRIMARY)
            if name in (shapelib.PRIMARY, "", None) or name not in self.effect_catalog:
                continue
            if next_slot > shapelib.MAX_FX_SLOTS:
                continue
            wanted[next_slot] = (s.get("id"), name)
            slot_by_index[idx] = next_slot
            next_slot += 1

        self._shapes_np[:] = 0.0
        n = 0
        for k, s in enumerate(shape_dicts[:shapelib.MAX_SHAPES]):
            self._shapes_np[k] = shapelib.encode(s, slot_by_index.get(k, 0))
            n += 1
        self._shapes_count = n
        if wanted != self._fx_wanted:
            self._fx_wanted = wanted
            self._fx_dirty = True
        self._pending_shapes = True

    def shape_fx_for(self, shape_id):
        """The live secondary-effect slot for a shape id (or None) — the Shape
        FX panel reads its store + params from here."""
        for d in self._fx_active.values():
            if d["id"] == shape_id:
                return d
        return None

    def _clear_canvas(self):
        self.canvas.fill(0)

    # ---- color ----------------------------------------------------------
    def _upload_palette(self, force=False):
        # find the (first) palette param's current spec
        spec = None
        for p in self.params:
            if isinstance(p, ColorPalette):
                spec = self.store.values.get(p.name)
                break
        if spec is None:
            spec = {"named": "rainbow", "custom": []}
        key = (spec.get("named"), tuple(spec.get("custom", [])))
        if force or key != self._last_palette:
            lut = color.build_lut(spec)
            self.palette.from_numpy(lut.astype(np.float32))
            self._last_palette = key

    # ---- per-frame param resolution ------------------------------------
    def _resolve_store(self, store, signals):
        """Resolve a store's knob values for this frame, applying each driven
        knob's modulation. `signals` is the unified {source: value} table (audio
        bands + LFOs), so a knob can be driven by the kick OR by LFO 2 alike."""
        values, srcs, amts = store.snapshot()
        out = {}
        for name, val in values.items():
            p = store.params.get(name)
            if isinstance(p, Slider) and signals is not None:
                src = srcs.get(name, "none")
                if src != "none":
                    drive = signals.get(src, 0.0) * (p.hi - p.lo) * amts.get(name, 0.5)
                    val = max(p.lo, min(p.hi, val + drive))
            out[name] = val
        return out

    def _resolve(self, signals):
        return self._resolve_store(self.store, signals)

    # ---- Music Director: song analysis -> drive sources ----------------
    def analyze_track(self, path):
        """Analyze a music file in the background -> self.structure, and set the
        beat clock to the detected tempo. Safe to call from the UI thread."""
        if not path:
            return
        self._structure_status = "analyzing…"
        self.structure = None
        self._structure_path = None

        def work():
            try:
                import soundfile as sf
                from .structure import analyze
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                st = analyze(data, sr)
                self.structure = st
                self._structure_path = path
                self.tempo.set_bpm(st.bpm)
                self._structure_status = (f"BPM {st.bpm:.0f} · {len(st.drops)} drops "
                                          f"· {st.duration:.0f}s")
            except Exception as e:
                self.structure = None
                self._structure_status = f"analysis failed: {type(e).__name__}"
        threading.Thread(target=work, daemon=True).start()

    def _has_song_map(self):
        """True when an analysis matches the file currently playing."""
        return (self.structure is not None and self.audio is not None
                and self.audio.current_file() == self._structure_path)

    def _musical_beats(self):
        """Continuous beat count — from the analyzed song's playback position
        when we have a map for the playing track, else the wall-clock engine."""
        if self._has_song_map() and self.structure.beat_times:
            pos = self.audio.position()
            beat0 = self.structure.beat_times[0]
            return (pos - beat0) * self.structure.bpm / 60.0
        return self.tempo.beats_now()

    def _musical_signals(self):
        """Bar/beat phases — locked to the song when we have a map, else clock."""
        if self._has_song_map() and self.structure.beat_times:
            return phases_from_beats(self._musical_beats())
        return self.tempo.phases()

    def _director_signals(self, feats):
        """intensity / build / drop. From the song map at the playback position
        (anticipatory) when available, else a live-audio fallback."""
        if self._has_song_map():
            pos = self.audio.position()
            inten, build = self.structure.at(pos)
            drop = 0.0
            for d in self.structure.drops:
                if 0.0 <= pos - d <= 2.0:                  # 2 s decay after a drop
                    drop = max(drop, 1.0 - (pos - d) / 2.0)
            return {"intensity": float(inten), "build": float(build), "drop": float(drop)}
        # live fallback: smoothed volume + its rise (no look-ahead, no drops)
        vol = float(feats.volume) if feats is not None else 0.0
        prev = self._live_inten
        self._live_inten += (vol - self._live_inten) * 0.05
        return {"intensity": self._live_inten,
                "build": min(1.0, max(0.0, (self._live_inten - prev) * 20.0)),
                "drop": 0.0}

    # ---- secondary effects (shapes showing a different effect) ----------
    def _reconcile_secondary(self):
        """Make self._fx_active match self._fx_wanted: build new secondary
        effect instances (each a full effect with its own params/look/palette),
        drop unused ones. Runs on the render thread."""
        self._fx_dirty = False
        new_active = {}
        for slot, (sid, name) in self._fx_wanted.items():
            cur = self._fx_active.get(slot)
            if cur is not None and cur["id"] == sid and cur["name"] == name:
                new_active[slot] = cur          # reuse — same shape+effect here
                continue
            cls = self.effect_catalog.get(name)
            if cls is None:
                continue
            buf = self.shapes_fx.buffer(slot)
            pal = self._fx_pal[slot - 1]
            inst = cls()
            # full param set: global look (trails/grain/...) + the effect's knobs
            params = global_params() + list(inst.params)
            key = (sid, name)
            store = self._fx_store_cache.get(key)
            if store is None:               # first time -> defaults; else keep edits
                store = ParamStore(); store.load(params)
                self._fx_store_cache[key] = store
            ctx = Context(self.w, self.h, buf, pal)
            self._wire_inputs(ctx)
            try:
                inst.setup(ctx)
            except Exception as e:
                self.effect_error = f"shape effect '{name}': {type(e).__name__}: {e}"
                continue
            new_active[slot] = dict(slot=slot, id=sid, name=name, inst=inst, ctx=ctx,
                                    store=store, params=params, buf=buf, pal=pal,
                                    postfx=PostFX(self.w, self.h, buf), palkey=None)
        # clear any buffer no longer backed by an effect (avoid stale art)
        for slot in range(1, shapelib.MAX_FX_SLOTS + 1):
            if slot not in new_active:
                self.shapes_fx.buffer(slot).fill(0)
        # prune knob caches for shape-effects that no longer exist
        live = {(sid, name) for sid, name in self._fx_wanted.values()}
        self._fx_store_cache = {k: v for k, v in self._fx_store_cache.items() if k in live}
        self._fx_active = new_active

    def _upload_store_palette(self, d):
        """Upload a slot's own palette LUT from its ColorPalette knob."""
        spec = None
        for p in d["params"]:
            if isinstance(p, ColorPalette):
                spec = d["store"].values.get(p.name)
                break
        if spec is None:
            spec = {"named": "rainbow", "custom": []}
        key = (spec.get("named"), tuple(spec.get("custom", [])))
        if key != d["palkey"]:
            d["pal"].from_numpy(color.build_lut(spec).astype(np.float32))
            d["palkey"] = key

    def _wire_inputs(self, ctx):
        """Give a sub-context (a layer / secondary effect) the SAME input fields
        the main context has, so a video effect works there too (it reads
        ctx.media / ctx.flow / ctx.blobs / ctx.pointer just like the main one)."""
        ctx.media = self.media_field
        ctx.media_motion = self.motion_field
        ctx.flow = self.flow_field
        ctx.blobs = self.blob_field
        ctx.max_blobs = MAX_BLOBS
        ctx.pointer = self.pointer

    def _render_secondary(self, feats, signals):
        """Render each secondary effect (its own knobs + look post-FX + palette)
        into its buffer so window-shapes can sample it."""
        for d in self._fx_active.values():
            self._upload_store_palette(d)
            c = d["ctx"]
            c.time = self.ctx.time; c.dt = self.ctx.dt; c.frame = self.ctx.frame
            c.audio = feats
            c.has_media = self.ctx.has_media
            c.has_video = self.ctx.has_video; c.n_blobs = self.ctx.n_blobs
            c.p = self._resolve_store(d["store"], signals)
            if c.p.get("trails"):
                d["postfx"].decay(float(c.p.get("trail_length", 0.9)))
            else:
                d["buf"].fill(0)
            try:
                d["inst"].render(c)
                d["postfx"].apply(c.p, c.time)
            except Exception as e:
                self.effect_error = f"shape effect '{d['name']}': {type(e).__name__}: {e}"

    # ---- layer stack (effects blended over the main effect) -------------
    def request_layers(self, layers):
        """Set the layer stack (bottom -> top), each: dict(id, effect, blend,
        opacity, visible). Cheap: swaps the live meta list and only flags a
        rebuild when the set of (id, effect) actually changes, so dragging an
        opacity slider doesn't re-instantiate anything."""
        layers = [l for l in (layers or []) if l.get("effect")][:MAX_LAYERS]
        self._layers_meta = layers          # atomic ref swap (read on render thread)
        key = tuple((l.get("id"), l.get("effect")) for l in layers)
        if key != self._layers_key:
            self._layers_key = key
            self._layers_dirty = True

    def layer_fx_for(self, layer_id):
        """The live layer slot for an id (or None) — the Layer FX editor reads
        its store + params from here."""
        for d in self._layers_active:
            if d["id"] == layer_id:
                return d
        return None

    def _reconcile_layers(self):
        """Make self._layers_active match the current meta: build new layer
        effect instances (each a full effect with its own params/look/palette),
        reuse unchanged ones, drop the rest. Runs on the render thread."""
        self._layers_dirty = False
        by_key = {(d["id"], d["name"]): d for d in self._layers_active}
        new_active = []
        used_bufs = 0
        for m in self._layers_meta:
            lid, name = m.get("id"), m.get("effect")
            cur = by_key.get((lid, name))
            if cur is not None:
                new_active.append(cur)          # reuse — same layer+effect
                used_bufs += 1
                continue
            cls = self.effect_catalog.get(name)
            if cls is None or used_bufs >= MAX_LAYERS:
                continue
            buf = self._layer_bufs[used_bufs]
            pal = self._layer_pals[used_bufs]
            inst = cls()
            params = global_params() + list(inst.params)
            key = (lid, name)
            store = self._layer_store_cache.get(key)
            if store is None:
                store = ParamStore(); store.load(params)
                self._layer_store_cache[key] = store
            ctx = Context(self.w, self.h, buf, pal)
            self._wire_inputs(ctx)
            try:
                inst.setup(ctx)
            except Exception as e:
                self.effect_error = f"layer '{name}': {type(e).__name__}: {e}"
                continue
            new_active.append(dict(id=lid, name=name, inst=inst, ctx=ctx, store=store,
                                   params=params, buf=buf, pal=pal,
                                   postfx=PostFX(self.w, self.h, buf), palkey=None))
            used_bufs += 1
        # clear buffers no longer backed by a layer (avoid stale art)
        for k in range(len(new_active), MAX_LAYERS):
            self._layer_bufs[k].fill(0)
        # prune knob caches for layers that no longer exist
        live = {(l.get("id"), l.get("effect")) for l in self._layers_meta}
        self._layer_store_cache = {k: v for k, v in self._layer_store_cache.items()
                                   if k in live}
        self._layers_active = new_active

    def _render_and_blend_layers(self, feats, signals):
        """Render each layer (own knobs + look post-FX + palette) into its buffer
        and blend it onto the canvas with its mode + opacity, bottom -> top."""
        if not self._layers_active:
            return
        meta_by_id = {m.get("id"): m for m in self._layers_meta}
        for d in self._layers_active:
            m = meta_by_id.get(d["id"], {})
            if not m.get("visible", True):
                continue
            self._upload_store_palette(d)
            c = d["ctx"]
            c.time = self.ctx.time; c.dt = self.ctx.dt; c.frame = self.ctx.frame
            c.audio = feats
            c.has_media = self.ctx.has_media
            c.has_video = self.ctx.has_video; c.n_blobs = self.ctx.n_blobs
            c.p = self._resolve_store(d["store"], signals)
            if c.p.get("trails"):
                d["postfx"].decay(float(c.p.get("trail_length", 0.9)))
            else:
                d["buf"].fill(0)
            try:
                d["inst"].render(c)
                d["postfx"].apply(c.p, c.time)
            except Exception as e:
                self.effect_error = f"layer '{d['name']}': {type(e).__name__}: {e}"
                continue
            self.layer_fx.blend(d["buf"],
                                LAYER_BLEND_IDS.get(m.get("blend", "Normal"), 0),
                                float(m.get("opacity", 1.0)))

    # ---- main loop ------------------------------------------------------
    def run(self, on_frame=None, display=None, target_fps=60):
        """Single-threaded render loop.

        We no longer open Taichi's own window. Instead each frame is handed to
        `display(img)` (the Tk preview) and `on_frame()` pumps the UI, so the
        whole app lives in ONE window. `target_fps` caps the loop so we don't
        spin the CPU at 100%."""
        frame_budget = 1.0 / max(1, target_fps)
        while self.running:
            # apply UI-requested swaps on THIS (GPU-owning) thread
            if self._pending_effect is not None:
                req = self._pending_effect
                self._pending_effect = None
                inst = req() if isinstance(req, type) else req
                self.set_effect(inst)
            if self._pending_expr is not None and hasattr(self.effect, "set_formulas"):
                b, h, src = self._pending_expr
                self._pending_expr = None
                self.effect.set_formulas(b, h, src)
            if self._pending_shapes:
                self.shapes_fx.upload(self._shapes_np)
                self._pending_shapes = False
            if self._fx_dirty:
                self._reconcile_secondary()
            if self._layers_dirty:
                self._reconcile_layers()
            if self._pending_reset:
                self._pending_reset = False
                if self.effect:
                    self.effect.reset()
                self._clear_canvas()
                self.feedback.clear()
            if self._pending_export is not None:
                cfg = self._pending_export
                self._pending_export = None
                self._do_export(cfg)
                self._last = time.perf_counter()  # don't count export time as dt

            now = time.perf_counter()
            self.ctx.dt = now - self._last
            self.ctx.time = now - self._start
            self._last = now

            feats = self.audio.features() if self.audio else None
            self.ctx.audio = feats
            # feed the beat clock from the audio beat detector (auto mode only),
            # on the rising edge of a detected beat.
            beat_now = bool(feats.beat) if feats is not None else False
            if self.tempo.auto and beat_now and not self._prev_beat:
                self.tempo.on_beat(time.perf_counter())
            self._prev_beat = beat_now

            # one signal table per frame: audio bands + LFOs + MIDI + musical
            # time (bar/beat) + the Music Director (intensity/build/drop).
            dsig = self._director_signals(feats)
            signals = self.mods.signals(self.ctx.time, feats)
            signals.update(self.midi.values())
            signals.update(self._musical_signals())
            signals.update(dsig)
            self.ctx.p = self._resolve(signals)
            # auto-pilot: steer params + fire scene changes from the song map
            self.director.apply(self, self.ctx.p, dsig)
            self._upload_palette()
            self.pointer.tick()        # per-frame mouse velocity for interactive fx

            if not self.paused:
                self._update_media()   # refresh media+motion so effects can sample
                self._update_video()   # optical flow + blobs for video effects
                self._update_subject() # person mask for 'effect behind subject'

                # trails: fade old frame instead of full clear
                if self.ctx.p.get("trails"):
                    self.postfx.decay(float(self.ctx.p.get("trail_length", 0.9)))
                else:
                    self._clear_canvas()

                if self.effect:
                    try:
                        self.effect.render(self.ctx)
                        # surface any error the effect recorded internally
                        self.effect_error = getattr(self.effect, "error", "")
                    except Exception as e:
                        # a user effect threw - report it, don't kill the app
                        self.effect_error = f"{type(e).__name__}: {e}"
                self.postfx.apply(self.ctx.p, self.ctx.time)
                self._render_and_blend_layers(feats, signals)  # stack over main fx
                self._apply_feedback()                          # tunnels / spirals / echoes
                self._composite_media()
                self._composite_subject()  # subject in front of the effect/layers
                if self._fx_active:        # shapes showing other effects -> draw them
                    self._render_secondary(feats, signals)
                self._composite_shapes()   # shapes interact with the final image
                self.ctx.frame += 1

            if display is not None:
                if self.presenter is not None:
                    # GPU packs canvas -> image-oriented uint8; UI blits as-is.
                    try:
                        self.presenter.pack()
                        display(self.presenter.buf.to_numpy())
                    except Exception:
                        self.presenter = None      # never again; use fallback
                if self.presenter is None:         # fallback: float readback
                    img = self.canvas.to_numpy()
                    np.clip(img, 0.0, 1.0, out=img)
                    display(img)
            if on_frame is not None:
                on_frame()

            # frame cap so the loop doesn't peg a CPU core
            spare = frame_budget - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

        self.running = False
        if self.audio:
            self.audio.stop()
        self.midi.close()

    def _update_media(self):
        """Upload the latest media frame and update the camera-motion field, so
        effects can SAMPLE ctx.media / ctx.media_motion this frame."""
        self.ctx.has_media = False
        if self.media is None:
            return
        frame = self.media.frame()
        if frame is None or frame.shape[:2] != (self.h, self.w):
            # no live media -> clear the fields once so stale frames don't linger
            if not self._media_cleared:
                self.media_field.fill(0)
                self.motion_field.fill(0)
                self._prev_luma.fill(0)
                self._media_cleared = True
            return
        # (H,W,3) image -> field[x,y] in the canvas' (x right, y up) orientation
        arr = np.ascontiguousarray(
            np.transpose(np.flipud(frame), (1, 0, 2)).astype(np.float32))
        self.media_field.from_numpy(arr)
        self.media_fx.update_motion(6.0, 0.85)
        self.ctx.has_media = True
        self._media_cleared = False

    def _update_video(self):
        """Run/stop video analysis and upload flow + tracked blobs so a video
        effect can read ctx.flow / ctx.blobs. Only active when the current effect
        opts in (`uses_video = True`) and a video/camera is producing frames."""
        self.ctx.has_video = False
        self.ctx.n_blobs = 0
        if self.media is None:
            return
        wants = self.ctx.has_media and bool(getattr(self.effect, "uses_video", False))
        self.media.set_analyze(wants)
        if not wants:
            return
        flow = self.media.flow()
        if flow is not None and flow.shape == (self.h, self.w, 2):
            f = np.flipud(flow).copy()
            f[..., 1] = -f[..., 1]              # y flips with flipud -> negate it
            self.flow_field.from_numpy(np.ascontiguousarray(
                np.transpose(f, (1, 0, 2)).astype(np.float32)))
        else:
            self.flow_field.fill(0)
        blobs, n = self.media.blobs()
        conv = blobs.copy()                     # image coords (y down) -> canvas (y up)
        conv[:, 1] = 1.0 - conv[:, 1]
        conv[:, 4] = -conv[:, 4]
        self.blob_field.from_numpy(np.ascontiguousarray(conv.astype(np.float32)))
        self.ctx.n_blobs = int(n)
        self.ctx.has_video = True

    def _update_subject(self):
        """Turn person segmentation on/off and upload the subject mask, so the
        SUBJECT of the video/camera can be composited in FRONT of the effect
        ('Effect behind subject'). Only active when the user enables it AND media
        is producing frames — otherwise segmentation costs nothing."""
        self.ctx.has_subject = False
        if self.media is None:
            return
        mode = self.ctx.p.get("subject_front", "Off")
        wants = self.ctx.has_media and mode in ("Person", "Motion")
        if hasattr(self.media, "set_mask"):
            self.media.set_mask(mode.lower() if wants else "off")
        if not wants:
            return
        m = self.media.mask() if hasattr(self.media, "mask") else None
        if m is not None and m.shape == (self.h, self.w):
            # (H,W) image mask -> field[x,y] in the canvas' (x right, y up) frame
            self.mask_field.from_numpy(
                np.ascontiguousarray(np.flipud(m).T.astype(np.float32)))
            self.ctx.has_subject = True

    def _composite_subject(self):
        """Paste the masked subject over the composited frame (effect behind)."""
        if not getattr(self.ctx, "has_subject", False):
            return
        self.media_fx.overlay_subject(
            float(self.ctx.p.get("subject_strength", 1.0)),
            float(self.ctx.p.get("subject_feather", 0.4)))

    def _composite_shapes(self):
        """Draw the shape elements as a layer that interacts with the effect.
        Skipped entirely when there are no shapes (zero cost)."""
        if self._shapes_count == 0:
            return
        self.shapes_fx.run(self.ctx.time,
                           float(self.ctx.p.get("shapes_gain", 1.0)),
                           self.ctx.audio)

    def reset_render_state(self):
        """Clear ALL transient visual accumulation so a fresh render (e.g. an
        export) starts from a clean first frame instead of inheriting the live
        session's mid-animation buildup: the main effect, the feedback buffer
        (tunnels/spirals/echoes), and every layer / secondary effect's own state
        and trail buffer. Call after the layer/secondary stacks are reconciled."""
        if self.effect:
            try:
                self.effect.reset()
            except Exception:
                pass
        self._clear_canvas()
        self.feedback.clear()
        for d in self._layers_active:
            try:
                d["inst"].reset()
            except Exception:
                pass
            d["buf"].fill(0)
        for d in self._fx_active.values():
            try:
                d["inst"].reset()
            except Exception:
                pass
            d["buf"].fill(0)

    def _apply_feedback(self):
        """Recursively blend the previous composited frame back in (zoom/spin +
        decay) for tunnels / spirals / echoes. Cheap no-op when off."""
        p = self.ctx.p
        if not p.get("feedback"):
            return
        self.feedback.apply(float(p.get("fb_zoom", 1.0)),
                            math.radians(float(p.get("fb_rotate", 0.0))),
                            float(p.get("fb_decay", 0.85)))
        self.feedback.save()      # this frame becomes next frame's feedback source

    def _composite_media(self):
        """OPTIONAL 'Background' mode: blend the media behind/over the effect.
        Media is already uploaded by _update_media()."""
        if self.media is None:
            return
        blend = self.ctx.p.get("media_blend", "Off")
        if blend == "Off" or not self.ctx.has_media:
            return
        self.media_fx.composite(BLEND_IDS.get(blend, 0),
                                float(self.ctx.p.get("media_opacity", 1.0)),
                                float(self.ctx.p.get("media_brightness", 1.0)))

    def _do_export(self, cfg):
        # pause live audio so it doesn't fight the offline analyzer / CPU
        was_mode = self.audio._mode if self.audio else "none"
        was_file = self.audio.current_file() if self.audio else None
        if self.audio:
            self.audio.stop()
        if cfg.get("video_path"):
            # render-through-video: run the effect OVER the input clip
            from .export import export_video
            was_media = self.media._mode if self.media else "off"
            was_media_path = getattr(self.media, "_path", None) if self.media else None
            if self.media:
                self.media.stop()                  # free the file + CPU
            ok, msg = export_video(
                self, cfg["video_path"], cfg["out_path"],
                seconds=cfg.get("seconds"), progress=cfg.get("progress"))
            if self.media and was_media in ("camera", "video"):
                self.media.set_mode(was_media, was_media_path)
        else:
            from .export import export_mp4
            ok, msg = export_mp4(
                self, cfg["audio_path"], cfg["out_path"],
                fps=cfg.get("fps", 30), seconds=cfg.get("seconds"),
                progress=cfg.get("progress"))
        cfg.get("done", lambda *_: None)(ok, msg)
        if self.audio and was_mode != "none":
            self.audio.set_mode(was_mode, was_file)
