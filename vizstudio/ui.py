"""The control panel - auto-generated from whatever params the active effect
(plus the global look section) declares.

The author of an effect writes zero UI code. They list params; this file turns
each one into the right widget, and wires numeric ones up to an audio-drive
dropdown so "make particle size follow the bass" is a menu pick, not code.

Runs in its own thread. Only touches the engine's ParamStore and AudioEngine
(both thread-safe) - never Taichi.
"""

from __future__ import annotations

import os
import re
import json
import time

import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog

from .params import Slider, IntSlider, Toggle, Choice, ColorPalette, AUDIO_SOURCES
from .effect import Effect
from .registry import discover
from .exprutil import cheat_sheet, translate, exec_with_source
from . import labkit
from . import paths
from .exprfx import ExpressionEffectBase
from .builder_templates import CODE_TEMPLATE, expression_file, EXPR_EFFECT_NAME
from . import patterns
from . import shapes
from . import project
from .modulation import LFO_IDS, LFO_LABELS, LFO_SHAPES, N_LFOS
from .layers_fx import LAYER_BLENDS, MAX_LAYERS
from .midi import MIDI_IDS, MIDI_LABELS, N_MIDI
from .tempo import TEMPO_IDS, TEMPO_LABELS
from .structure import DIRECTOR_IDS, DIRECTOR_LABELS
from .director import ACTIONS, TRIGGERS

_DRIVE_LABELS = {"none": "—", "volume": "Vol", "bass": "Bass",
                 "mid": "Mid", "treble": "Treble", "beat": "Beat",
                 "kick": "Kick", "snare": "Snare", "hihat": "HiHat"}
_DRIVE_LABELS.update(LFO_LABELS)
_DRIVE_LABELS.update(MIDI_LABELS)
_DRIVE_LABELS.update(TEMPO_LABELS)
_DRIVE_LABELS.update(DIRECTOR_LABELS)
# every knob's "drive" menu: audio bands + the Music Director (intensity/build/
# drop) + musical time (tempo) + LFOs + MIDI
DRIVE_SOURCES = (list(AUDIO_SOURCES) + list(DIRECTOR_IDS) + list(TEMPO_IDS)
                 + list(LFO_IDS) + list(MIDI_IDS))

# panels that float in their own window, and panels hidden in the *default*
# layout (still one click away in the Panels menu). Keeping defaults lean —
# left: Audio / Media / Shapes; right: Effect / Layers / Parameters.
_FLOATING_PANELS = ("create", "shapefx", "layerfx")
_DEFAULT_HIDDEN_PANELS = _FLOATING_PANELS + ("mod", "midi", "director", "export")

# UI color themes. Keys: bg (window), panel (boxes), fg (text), accent
# (highlights), btn (buttons), entry/entry_fg (text inputs).
THEMES = {
    "Eyenips":  dict(bg="#000000", panel="#0b0b0b", fg="#ffffff", accent="#e81ce8",
                     btn="#181018", entry="#141014", entry_fg="#ffffff", meter_bg="#000"),
    "Dark":     dict(bg="#1e1e1e", panel="#252526", fg="#e0e0e0", accent="#3fd0ff",
                     btn="#3a3a3c", entry="#333333", entry_fg="#ffffff", meter_bg="#111"),
    "Light":    dict(bg="#f3f3f3", panel="#ffffff", fg="#1a1a1a", accent="#0a84ff",
                     btn="#e6e6e6", entry="#ffffff", entry_fg="#000000", meter_bg="#222"),
    "Midnight": dict(bg="#0b1021", panel="#141a33", fg="#cdd6f4", accent="#89b4fa",
                     btn="#1e2745", entry="#1e2745", entry_fg="#ffffff", meter_bg="#05060f"),
    "Neon":     dict(bg="#0a0a0a", panel="#141414", fg="#d6ffd6", accent="#39ff14",
                     btn="#1a1a1a", entry="#161616", entry_fg="#b6ffb6", meter_bg="#000"),
    "Sunset":   dict(bg="#2a1726", panel="#3a2236", fg="#ffe6d9", accent="#ff8a5c",
                     btn="#4a2c44", entry="#3a2236", entry_fg="#ffffff", meter_bg="#1a0e18"),
}


class ControlPanel:
    def __init__(self, engine, effect_classes, effects_dir=None):
        self.engine = engine
        self.effect_classes = effect_classes
        self.effects_dir = effects_dir
        self.root = None
        self._effect_version = -1
        self._closed = False
        self._pump_count = 0
        self._loaded_audio = None  # last audio file chosen (for export default)
        self._builder = None       # the Create Effect Toplevel (built lazily)
        self._last_effect_error = ""
        self.shape_items = []       # Shapes-panel state (panel built lazily)
        self.shape_sel = None
        self._img_rect = None       # on-screen preview-image rect (for placing)
        self._photo = None          # reused preview PhotoImage (in-place paste)
        self._fps = 0.0             # smoothed frames/sec, shown in the toolbar
        self._fps_t = None          # last show_frame timestamp
        self._fps_n = 0
        self.fps_lbl = None         # toolbar FPS readout (built in build())
        self._media_path = None     # last image/video path (for project restore)
        self._project_path = None   # current .viz project file (Save vs Save As)
        self._out_win = None        # fullscreen output Toplevel (projector/2nd screen)
        self._out_label = None
        self._out_photo = None
        self._tip_win = None
        # Writable user data lives in ~/.eyenips (NOT the install dir, which is
        # read-only in a Program Files install) so sessions/presets persist and
        # survive updates. Starter presets ship bundled and are read read-only.
        udir = paths.user_data_dir()
        try:
            os.makedirs(udir, exist_ok=True)
        except Exception:
            udir = os.getcwd()                      # last-ditch: stay runnable
        self._session_path = os.path.join(udir, "session.viz")
        self._presets_dir = os.path.join(udir, "presets")     # where SAVES go
        # bundled, read-only starter presets (shipped with the app); fall back to
        # a local presets/ folder in older/dev layouts
        self._builtin_presets_dir = paths.find("starter_presets", "presets")
        self._welcome_flag = os.path.join(udir, ".welcomed")  # first-run greeting

    def build(self):
        """Create the ONE app window (preview + docked controls). Does NOT
        block; call pump() each frame and show_frame(img) to draw the visual.
        Everything is on the main thread (Tkinter requires it)."""
        self.root = tk.Tk()
        self.root.title("Eyenips")
        try:                                   # window/taskbar icon (frozen-safe)
            ico = paths.find(os.path.join("assets", "eyenips.png"))
            if os.path.exists(ico):
                self._icon_img = tk.PhotoImage(file=ico)
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass
        self.root.geometry("1560x900")
        self.root.minsize(1100, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._photo = None
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")   # 'clam' lets us color ttk widgets
        except tk.TclError:
            pass

        # --- toolbar (title, transport, theme) ---
        self.toolbar = tk.Frame(self.root)
        self.toolbar.pack(fill="x")
        tk.Label(self.toolbar, text="Eyenips", font=("", 13, "bold")).pack(
            side="left", padx=10, pady=6)
        self.pause_btn = tk.Button(self.toolbar, text="⏸ Pause", width=8,
                                   command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=4)
        self._tip(self.pause_btn, "Pause / resume the visual")
        reset_btn = tk.Button(self.toolbar, text="⟳ Reset", command=self._reset)
        reset_btn.pack(side="left", padx=4)
        self._tip(reset_btn, "Re-seed the current effect")
        fs_btn = tk.Button(self.toolbar, text="⛶ Output", command=self._toggle_output)
        fs_btn.pack(side="left", padx=4)
        self._tip(fs_btn, "Open the visual full-screen on a projector / second "
                  "monitor (F11). Esc or double-click to exit.")
        self.theme_var = tk.StringVar(value="Eyenips")
        # live FPS / frame-time readout (a small "pro tool" signal), right side
        self.fps_lbl = tk.Label(self.toolbar, text="—", font=("", 9), fg="#888")
        self.fps_lbl.pack(side="right", padx=10)
        self._tip(self.fps_lbl, "Live preview frame rate (frames per second / "
                  "milliseconds per frame).")
        # the menus live IN the toolbar (themeable), not a native menubar
        # (Windows won't recolor a native menubar).

        # --- split: left dock | preview (center) | right dock ---
        # A symmetric layout: dockable panels on BOTH sides, the live visual in
        # the middle. Drag the two outer sashes to resize the docks.
        main = tk.PanedWindow(self.root, orient="horizontal", sashwidth=6)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=400)
        left.pack_propagate(False)
        self.dock_left = tk.PanedWindow(left, orient="vertical", sashwidth=6,
                                        showhandle=False)
        self.dock_left.pack(fill="both", expand=True)
        main.add(left, minsize=300)

        center = tk.Frame(main, bg="#000")
        self.preview = tk.Label(center, bg="#000", text="starting…", fg="#888")
        self.preview.pack(fill="both", expand=True)
        # click / drag the preview to place the selected shape
        self.preview.bind("<Button-1>", self._place_selected)
        self.preview.bind("<B1-Motion>", self._place_selected)
        # feed the mouse to the engine so effects can be INTERACTIVE (e.g. stir
        # the fluid). add="+" so this coexists with shape-placing above.
        self.preview.bind("<Motion>", lambda e: self._update_pointer(e, down=False), add="+")
        self.preview.bind("<Button-1>", lambda e: self._update_pointer(e, down=True), add="+")
        self.preview.bind("<B1-Motion>", lambda e: self._update_pointer(e, down=True), add="+")
        self.preview.bind("<ButtonRelease-1>", lambda e: self._update_pointer(e, down=False), add="+")
        self.preview.bind("<Leave>", lambda e: self._pointer_leave(), add="+")
        main.add(center, stretch="always", minsize=380)

        right = tk.Frame(main, width=400)
        right.pack_propagate(False)
        self.dock_right = tk.PanedWindow(right, orient="vertical", sashwidth=6,
                                         showhandle=False)
        self.dock_right.pack(fill="both", expand=True)
        main.add(right, minsize=300)

        # let the engine resolve a shape's chosen effect by name
        self.engine.set_effect_catalog(self.effect_classes)

        self._init_panels()
        self._build_menu()
        self._build_all_panels()

        # status bar along the bottom (current effect · messages)
        self.statusbar = tk.Label(self.root, text="Ready", anchor="w",
                                  font=("", 9), padx=8, pady=2)
        self.statusbar.pack(fill="x", side="bottom")

        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-o>", lambda e: self._open_project())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-y>", lambda e: self._redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self._redo())
        self.root.bind("<F11>", lambda e: self._toggle_output())
        self.apply_theme("Eyenips")

        # restore the last session (effect, knobs, shapes, colors, layout)
        self._restore_session()
        self.apply_theme(self.theme_var.get())
        self._init_history()       # baseline for undo/redo = the restored state

        # First launch (no "seen" marker yet): greet the user with a 3-step start.
        if not os.path.exists(self._welcome_flag):
            self.root.after(250, lambda: self._show_welcome(first_run=True))
        # quietly check for a newer release (stays silent unless one exists)
        self.root.after(1500, lambda: self._check_updates(manual=False))

    # ---- dockable panels -----------------------------------------------
    def _init_panels(self):
        # builder(parent) builds that panel's content into `parent`.
        # side = which dock it lives in. Each side gets a couple of fixed
        # panels plus one "always"-stretch panel that fills the leftover height,
        # so the two docks stay balanced.
        self.panels = [
            # --- left dock: inputs + the shapes editor ---
            dict(key="audio",  title="Audio",          builder=self._build_audio_section,
                 side="left",  floating=False, visible=True,  stretch="never",  minsize=140),
            dict(key="mod",    title="🎛 Modulation & Tempo", builder=self._build_mod_panel,
                 side="left",  floating=False, visible=False, stretch="never",  minsize=120),
            dict(key="midi",   title="🎹 MIDI",          builder=self._build_midi_panel,
                 side="left",  floating=False, visible=False, stretch="never",  minsize=120),
            dict(key="director", title="🎬 Music Director", builder=self._build_director_panel,
                 side="left",  floating=False, visible=False, stretch="never",  minsize=150),
            dict(key="media",  title="Media",          builder=self._build_media_section,
                 side="left",  floating=False, visible=True,  stretch="never",  minsize=110),
            dict(key="shapes", title="✨ Shapes (elements)", builder=self._build_shapes_panel,
                 side="left",  floating=False, visible=True,  stretch="always", minsize=260),
            # --- right dock: effect + layers + parameters + export ---
            dict(key="effect", title="Effect",         builder=self._build_effect_picker,
                 side="right", floating=False, visible=True,  stretch="never",  minsize=64),
            dict(key="layers", title="🧱 Layers",       builder=self._build_layers_panel,
                 side="right", floating=False, visible=True,  stretch="never",  minsize=120),
            dict(key="params", title="Parameters",     builder=self._build_params_panel,
                 side="right", floating=False, visible=True,  stretch="always", minsize=180),
            dict(key="export", title="Export MP4",     builder=self._build_export_section,
                 side="right", floating=False, visible=False, stretch="never",  minsize=90),
            # --- on-demand floating ---
            dict(key="shapefx", title="⚙ Shape FX", builder=self._build_shapefx_panel,
                 side="right", floating=True,  visible=False, stretch="always", minsize=320),
            dict(key="layerfx", title="⚙ Layer FX", builder=self._build_layerfx_panel,
                 side="right", floating=True,  visible=False, stretch="always", minsize=320),
            dict(key="create", title="✎ Create Effect", builder=self._build_create_panel,
                 side="right", floating=True,  visible=False, stretch="always", minsize=360),
        ]
        for p in self.panels:
            p["wrap"] = None
            p["toplevel"] = None
            p["menu_var"] = tk.BooleanVar(value=p["visible"])

    def _panel(self, key):
        return next(p for p in self.panels if p["key"] == key)

    def _build_menu(self):
        # Menubuttons in the (themeable) toolbar instead of a native menubar,
        # so the whole top bar follows the selected theme. We keep references
        # to the dropdown menus so apply_theme() can recolor them too.
        self._menus = []

        prj_btn = tk.Menubutton(self.toolbar, text="Project ▾")
        prj = tk.Menu(prj_btn, tearoff=0)
        prj_btn.config(menu=prj)
        prj_btn.pack(side="left", padx=(8, 2))
        self._project_menu = prj
        self._menus.append(prj)
        self._rebuild_project_menu()

        pm_btn = tk.Menubutton(self.toolbar, text="Panels ▾")
        pm = tk.Menu(pm_btn, tearoff=0)
        pm_btn.config(menu=pm)
        for p in self.panels:
            pm.add_checkbutton(label=p["title"], variable=p["menu_var"],
                               command=lambda p=p: self._set_visible(p, p["menu_var"].get()))
        pm.add_separator()
        pm.add_command(label="Reset layout", command=self._reset_layout)
        pm.add_command(label="Quit", command=self._on_close)
        pm_btn.pack(side="left", padx=(16, 2))
        self._menus.append(pm)

        tm_btn = tk.Menubutton(self.toolbar, text="Theme ▾")
        tm = tk.Menu(tm_btn, tearoff=0)
        tm_btn.config(menu=tm)
        for name in THEMES:
            tm.add_radiobutton(label=name, variable=self.theme_var, value=name,
                               command=lambda n=name: self.apply_theme(n))
        tm_btn.pack(side="left", padx=2)
        self._menus.append(tm)

        out_btn = tk.Menubutton(self.toolbar, text="Output ▾")
        out = tk.Menu(out_btn, tearoff=0)
        out_btn.config(menu=out)
        out_btn.pack(side="left", padx=2)
        self._output_menu = out
        self._menus.append(out)
        self._rebuild_output_menu()

        help_btn = tk.Menubutton(self.toolbar, text="Help ▾")
        hm = tk.Menu(help_btn, tearoff=0)
        help_btn.config(menu=hm)
        help_btn.pack(side="left", padx=2)
        hm.add_command(label="Quick start…", command=lambda: self._show_welcome())
        hm.add_command(label="Check for updates…", command=lambda: self._check_updates(manual=True))
        hm.add_command(label="About Eyenips…", command=self._show_about)
        self._menus.append(hm)

    def _check_updates(self, manual=False):
        """Ask (off-thread) whether a newer release exists. `manual` also reports
        'you're up to date'; the silent startup check stays quiet unless newer."""
        import vizstudio
        from . import updatecheck

        def done(res):
            # hop back onto the Tk thread before touching widgets
            try:
                self.root.after(0, lambda: self._update_result(res, manual))
            except Exception:
                pass
        updatecheck.check_async(vizstudio.__version__, done)

    def _update_result(self, res, manual):
        if not res:
            if manual:
                self._set_status("You're on the latest version. ✓")
            return
        win = tk.Toplevel(self.root)
        win.title("Update available")
        win.resizable(False, False)
        tk.Label(win, text=f"Eyenips {res['version']} is available",
                 font=("", 13, "bold")).pack(padx=24, pady=(16, 4))
        if res.get("notes"):
            tk.Label(win, text=res["notes"], fg="#666", wraplength=360,
                     justify="left").pack(padx=24, pady=(0, 8))
        bar = tk.Frame(win)
        bar.pack(padx=24, pady=14)

        def _download():
            if res.get("url"):
                import webbrowser
                webbrowser.open(res["url"])
            win.destroy()
        tk.Button(bar, text="Download", command=_download).pack(side="left", padx=4)
        tk.Button(bar, text="Later", command=win.destroy).pack(side="left", padx=4)

    _WELCOME_STEPS = [
        ("🔊", "It's already listening",
         "Eyenips reacts to whatever's playing on your PC (the “System” source). "
         "Just play music — or load a song file in the Media panel. The visuals "
         "move with the sound."),
        ("🎨", "Pick an effect, then roll the dice",
         "Choose one from the Effect dropdown. Try “Effect Lab” and hit "
         "🎲 Randomize — every click is a brand-new effect from millions. Tweak "
         "the knobs in Parameters."),
        ("🎬", "Turn a video into an effect",
         "Load a video (Media panel) and pick “Video Lab” or “Effect Lab”. The "
         "effect becomes content-aware. Toggle “Effect behind subject” to put the "
         "effect behind a person — then 🎬 Export it."),
        ("✎", "Make it yours",
         "“✎ Edit equations” opens the actual math behind the labs — edit it, add "
         "your own, or Reset. It's all pure math; no AI. Have fun!"),
    ]

    def _show_welcome(self, first_run=False):
        win = tk.Toplevel(self.root)
        win.title("Welcome to Eyenips")
        win.resizable(False, False)
        win.transient(self.root)
        tk.Label(win, text="Welcome to Eyenips 🎶", font=("", 17, "bold")).pack(
            padx=28, pady=(18, 2))
        tk.Label(win, text="A no-code music visualizer. Three quick steps:",
                 fg="#888").pack(pady=(0, 10))
        body = tk.Frame(win)
        body.pack(fill="x", padx=26)
        for icon, title, text in self._WELCOME_STEPS:
            row = tk.Frame(body)
            row.pack(fill="x", pady=6, anchor="w")
            tk.Label(row, text=icon, font=("", 18)).pack(side="left", padx=(0, 12), anchor="n")
            col = tk.Frame(row)
            col.pack(side="left", fill="x", expand=True)
            tk.Label(col, text=title, font=("", 11, "bold"), anchor="w").pack(anchor="w")
            tk.Label(col, text=text, fg="#555", justify="left", wraplength=440,
                     anchor="w").pack(anchor="w")

        def _dismiss():
            try:
                with open(self._welcome_flag, "w", encoding="utf-8") as f:
                    f.write("seen")          # don't auto-show next launch
            except Exception:
                pass
            win.destroy()

        bar = tk.Frame(win)
        bar.pack(fill="x", padx=26, pady=16)
        tk.Label(bar, text="(Reopen any time from Help → Quick start)",
                 fg="#999").pack(side="left")
        tk.Button(bar, text="Let's go!", command=_dismiss).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", _dismiss)

    def _show_about(self):
        import vizstudio
        backend = "?"
        try:
            import taichi as ti
            backend = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
        except Exception:
            pass
        win = tk.Toplevel(self.root)
        win.title("About Eyenips")
        win.resizable(False, False)
        tk.Label(win, text="Eyenips", font=("", 16, "bold")).pack(padx=24, pady=(16, 2))
        tk.Label(win, text=f"version {vizstudio.__version__}").pack()
        tk.Label(win, text=f"render backend: {backend}", fg="#888").pack(pady=(0, 8))
        tk.Label(win, justify="center", fg="#666", wraplength=320,
                 text=("A no-code music-visualization studio.\n"
                       "Pure math + audio/video analysis — no generative AI.")).pack(
            padx=24, pady=(0, 6))
        tk.Label(win, fg="#888", wraplength=320, justify="center",
                 text=f"Your data: {paths.user_data_dir()}").pack(padx=24)
        tk.Button(win, text="Close", command=win.destroy).pack(pady=14)

    def _rebuild_project_menu(self):
        m = getattr(self, "_project_menu", None)
        if m is None:
            return
        m.delete(0, "end")
        m.add_command(label="Open project…   (Ctrl+O)", command=self._open_project)
        m.add_command(label="Save project   (Ctrl+S)", command=lambda: self._save_project())
        m.add_command(label="Save project as…", command=lambda: self._save_project(True))
        m.add_separator()
        m.add_command(label="Save as preset…", command=self._save_preset)
        presets = sorted(set(project.list_presets(self._presets_dir))
                         | set(project.list_presets(self._builtin_presets_dir)))
        if presets:
            sub = tk.Menu(m, tearoff=0)
            for name in presets:
                sub.add_command(
                    label=name,
                    command=lambda n=name: self._open_project(self._preset_path(n)))
            m.add_cascade(label="Load preset", menu=sub)
            self._menus.append(sub)
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _build_all_panels(self):
        for p in self.panels:
            if p["floating"]:
                self._make_floating(p)
                if not p["visible"]:
                    p["toplevel"].withdraw()
            elif p["visible"]:
                self._make_docked(p)

    def _dock_for(self, p):
        """The PanedWindow this panel docks into (left or right)."""
        return self.dock_left if p.get("side") == "left" else self.dock_right

    def _dock_panes(self, dock):
        return [str(x) for x in dock.panes()]

    def _make_docked(self, p):
        dock = self._dock_for(p)
        wrap = tk.Frame(dock, bd=1, relief="groove")
        hdr = tk.Frame(wrap); hdr.pack(fill="x")
        tk.Label(hdr, text=p["title"], font=("", 9, "bold")).pack(side="left", padx=6, pady=1)
        tk.Button(hdr, text="✕", width=2, command=lambda: self._set_visible(p, False)
                  ).pack(side="right", padx=1)
        tk.Button(hdr, text="⧉", width=2, command=lambda: self._set_floating(p, True)
                  ).pack(side="right", padx=1)
        body = tk.Frame(wrap); body.pack(fill="both", expand=True)
        p["builder"](body)
        p["wrap"] = wrap
        self._dock_insert(p, wrap)

    def _dock_insert(self, p, wrap):
        # keep panel order: insert before the next docked, present panel ON THE
        # SAME SIDE so each dock preserves the declared order.
        dock = self._dock_for(p)
        panes = self._dock_panes(dock)
        after = None
        seen = False
        for q in self.panels:
            if q is p:
                seen = True
                continue
            if seen and not q["floating"] and q.get("side") == p.get("side") \
                    and q["wrap"] is not None and str(q["wrap"]) in panes:
                after = q["wrap"]; break
        if after is not None:
            dock.add(wrap, before=after, stretch=p["stretch"], minsize=p["minsize"])
        else:
            dock.add(wrap, stretch=p["stretch"], minsize=p["minsize"])

    def _make_floating(self, p):
        tl = tk.Toplevel(self.root)
        tl.title(p["title"])
        tl.geometry("520x560")
        tl.protocol("WM_DELETE_WINDOW", lambda: self._set_visible(p, False))
        hdr = tk.Frame(tl); hdr.pack(fill="x")
        tk.Button(hdr, text="⧉ Dock", command=lambda: self._set_floating(p, False)
                  ).pack(side="right", padx=4, pady=2)
        body = tk.Frame(tl); body.pack(fill="both", expand=True)
        p["builder"](body)
        p["toplevel"] = tl

    def _set_visible(self, p, val):
        p["visible"] = bool(val)
        p["menu_var"].set(bool(val))
        if p["floating"]:
            if p["toplevel"] is None:
                self._make_floating(p)
            (p["toplevel"].deiconify() if val else p["toplevel"].withdraw())
        else:
            dock = self._dock_for(p)
            if val:
                if p["wrap"] is None:
                    self._make_docked(p)
                elif str(p["wrap"]) not in self._dock_panes(dock):
                    self._dock_insert(p, p["wrap"])
            elif p["wrap"] is not None and str(p["wrap"]) in self._dock_panes(dock):
                dock.forget(p["wrap"])
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _set_floating(self, p, val):
        # changing container requires a rebuild (Tk can't reparent widgets)
        if p["wrap"] is not None:
            dock = self._dock_for(p)
            if str(p["wrap"]) in self._dock_panes(dock):
                dock.forget(p["wrap"])
            p["wrap"].destroy(); p["wrap"] = None
        if p["toplevel"] is not None:
            p["toplevel"].destroy(); p["toplevel"] = None
        p["floating"] = bool(val)
        if val:
            self._make_floating(p)
            if not p["visible"]:
                p["toplevel"].withdraw()
        elif p["visible"]:
            self._make_docked(p)
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _reset_layout(self):
        for p in self.panels:
            if p["wrap"] is not None:
                dock = self._dock_for(p)
                if str(p["wrap"]) in self._dock_panes(dock):
                    dock.forget(p["wrap"])
                p["wrap"].destroy(); p["wrap"] = None
            if p["toplevel"] is not None:
                p["toplevel"].destroy(); p["toplevel"] = None
        for p in self.panels:
            p["floating"] = p["key"] in _FLOATING_PANELS
            p["visible"] = p["key"] not in _DEFAULT_HIDDEN_PANELS
            p["menu_var"].set(p["visible"])
        self._build_all_panels()
        self.apply_theme(self.theme_var.get())

    # ---- modulation panel (LFOs) ---------------------------------------
    def _build_mod_panel(self, parent):
        """Editor for the LFOs. Route one to any knob via that knob's 'drive'
        menu (the LFO 1–4 entries). Edits go straight to the engine's ModEngine,
        which is read on the render thread — no rebuild, smooth while dragging."""
        self._build_tempo_controls(parent)
        tk.Label(parent, text="LFOs — free-running shapes you can route to ANY "
                 "knob: open a knob's 'drive' menu and pick LFO 1–4.", fg="#888",
                 wraplength=360, justify="left").pack(anchor="w", padx=6, pady=(2, 4))
        for i in range(N_LFOS):
            self._build_lfo_row(parent, i)

    def _build_tempo_controls(self, parent):
        """The beat clock: set/tap a tempo, mark the downbeat, or follow the
        audio beat. Bar/beat phases then appear in every knob's drive menu."""
        box = tk.LabelFrame(parent, text="🥁 Tempo / beat clock", padx=6, pady=3)
        box.pack(fill="x", padx=6, pady=(2, 4))

        r = tk.Frame(box); r.pack(fill="x")
        tk.Label(r, text="BPM").pack(side="left")
        self.bpm_var = tk.StringVar(value=f"{self.engine.tempo.bpm():.0f}")
        e = tk.Entry(r, textvariable=self.bpm_var, width=6)
        e.pack(side="left", padx=3)
        e.bind("<Return>", self._set_bpm_from_entry)
        tk.Button(r, text="Set", width=3, command=self._set_bpm_from_entry).pack(side="left")
        tk.Button(r, text="Tap", width=4, command=self._tap_tempo).pack(side="left", padx=3)
        self.bpm_live = tk.Label(r, text="", fg="#888")
        self.bpm_live.pack(side="left", padx=4)

        r2 = tk.Frame(box); r2.pack(fill="x")
        tk.Button(r2, text="◉ Set downbeat", command=self.engine.tempo.align).pack(side="left")
        self.tempo_auto = tk.BooleanVar(value=self.engine.tempo.auto)
        tk.Checkbutton(r2, text="Auto (follow the beat)", variable=self.tempo_auto,
                       command=lambda: self.engine.tempo.set_auto(self.tempo_auto.get())
                       ).pack(side="left", padx=6)
        tk.Label(box, text="Drive any knob by Bar / 1-4 note / Beat pulse… from its "
                 "'drive' menu — locked to the song, not to seconds.", fg="#888",
                 wraplength=360, justify="left").pack(anchor="w")

    def _set_bpm_from_entry(self, *_a):
        try:
            self.engine.tempo.set_bpm(float(self.bpm_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _tap_tempo(self):
        self.engine.tempo.tap()
        if getattr(self, "bpm_var", None) is not None:
            self.bpm_var.set(f"{self.engine.tempo.bpm():.0f}")

    def _load_tempo(self):
        """Reflect a restored tempo in the panel widgets."""
        if getattr(self, "bpm_var", None) is not None:
            self.bpm_var.set(f"{self.engine.tempo.bpm():.0f}")
        if getattr(self, "tempo_auto", None) is not None:
            self.tempo_auto.set(self.engine.tempo.auto)

    # ---- Music Director panel ------------------------------------------
    def _build_director_panel(self, parent):
        """Analyze the loaded track and (optionally) let it run the show."""
        head = tk.Frame(parent, padx=8, pady=4); head.pack(side="top", fill="x")
        tk.Label(head, text="Eyenips reads the song's shape — beats, builds, "
                 "drops. Play an audio File, then drive any knob with Intensity / "
                 "Build / Drop from its 'drive' menu, or hand it the auto-pilot.",
                 wraplength=360, justify="left", font=("", 9, "bold")).pack(anchor="w")

        box = tk.Frame(parent, padx=8); box.pack(side="top", fill="x", pady=(0, 2))
        tk.Button(box, text="🔎 Analyze current track",
                  command=self._analyze_current).pack(side="left")
        self.director_status = tk.Label(parent, text="", fg="#888", wraplength=360,
                                        justify="left", padx=8)
        self.director_status.pack(side="top", fill="x", anchor="w")

        auto = tk.LabelFrame(parent, text="Auto-pilot (it runs the show)",
                             padx=6, pady=3)
        auto.pack(side="top", fill="x", padx=6, pady=4)
        self.dir_auto_inten = tk.BooleanVar(value=self.engine.director.auto_intensity)
        tk.Checkbutton(auto, text="Auto-intensity — brightness & feedback follow the "
                       "song's energy", variable=self.dir_auto_inten,
                       command=self._apply_director, wraplength=320, justify="left",
                       anchor="w").pack(anchor="w")

        r = tk.Frame(auto); r.pack(fill="x", pady=(2, 0))
        tk.Label(r, text="Switch:").pack(side="left")
        self.dir_action = tk.StringVar(value=self.engine.director.action)
        ca = ttk.Combobox(r, textvariable=self.dir_action, width=13, state="readonly",
                          values=ACTIONS)
        ca.pack(side="left", padx=3)
        ca.bind("<<ComboboxSelected>>", lambda e: self._apply_director())
        self.dir_trigger = tk.StringVar(value=self.engine.director.trigger)
        ct = ttk.Combobox(r, textvariable=self.dir_trigger, width=11, state="readonly",
                          values=TRIGGERS)
        ct.pack(side="left", padx=3)
        ct.bind("<<ComboboxSelected>>", lambda e: self._apply_director())
        self.dir_bars = tk.IntVar(value=self.engine.director.every_bars)
        tk.Spinbox(r, from_=1, to=64, width=4, textvariable=self.dir_bars,
                   command=self._apply_director).pack(side="left", padx=3)
        tk.Label(r, text="bars").pack(side="left")
        self._refresh_director()

    def _analyze_current(self):
        path = self.engine.audio.current_file() if self.engine.audio else None
        if not path:
            self._set_status("Load an audio File first (Audio panel).", error=True)
            return
        self.engine.analyze_track(path)
        self._refresh_director()

    def _apply_director(self):
        d = self.engine.director
        if getattr(self, "dir_auto_inten", None) is None:
            return
        d.auto_intensity = bool(self.dir_auto_inten.get())
        d.action = self.dir_action.get()
        d.trigger = self.dir_trigger.get()
        try:
            d.every_bars = max(1, int(self.dir_bars.get()))
        except (ValueError, tk.TclError):
            pass

    def _refresh_director(self):
        if getattr(self, "director_status", None) is not None:
            try:
                self.director_status.config(
                    text=self.engine._structure_status or "No track analyzed yet.")
            except tk.TclError:
                pass

    def _load_director(self):
        d = self.engine.director
        if getattr(self, "dir_auto_inten", None) is not None:
            self.dir_auto_inten.set(d.auto_intensity)
            self.dir_action.set(d.action)
            self.dir_trigger.set(d.trigger)
            self.dir_bars.set(d.every_bars)
        self._refresh_director()

    def _build_lfo_row(self, parent, i):
        cfg = self.engine.mods.get_lfo(i)
        box = tk.LabelFrame(parent, text=f"LFO {i + 1}", padx=6, pady=2)
        box.pack(fill="x", padx=6, pady=2)

        r1 = tk.Frame(box); r1.pack(fill="x")
        tk.Label(r1, text="Shape", width=6, anchor="w").pack(side="left")
        shp = tk.StringVar(value=cfg["shape"])
        cb = ttk.Combobox(r1, textvariable=shp, width=10, state="readonly",
                          values=LFO_SHAPES)
        cb.pack(side="left", padx=3)
        cb.bind("<<ComboboxSelected>>",
                lambda e, idx=i, v=shp: self.engine.mods.set_lfo(idx, shape=v.get()))

        r2 = tk.Frame(box); r2.pack(fill="x")
        tk.Label(r2, text="Rate", width=6, anchor="w").pack(side="left")
        tk.Scale(r2, from_=0.01, to=10.0, resolution=0.01, orient="horizontal",
                 length=180, showvalue=True,
                 command=lambda v, idx=i: self.engine.mods.set_lfo(idx, rate=float(v))
                 ).pack(side="left", fill="x", expand=True)
        tk.Label(r2, text="Hz").pack(side="left")
        # set initial position without firing the command's float() on a string
        r2.winfo_children()[1].set(cfg["rate"])

        r3 = tk.Frame(box); r3.pack(fill="x")
        tk.Label(r3, text="Depth", width=6, anchor="w").pack(side="left")
        sc = tk.Scale(r3, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
                      length=180, showvalue=True,
                      command=lambda v, idx=i: self.engine.mods.set_lfo(idx, depth=float(v)))
        sc.pack(side="left", fill="x", expand=True)
        sc.set(cfg["depth"])

    # ---- MIDI panel: hardware controllers -> drive sources --------------
    def _build_midi_panel(self, parent):
        """Connect a MIDI controller; map up to 8 of its knobs/faders to the
        MIDI 1–8 drive sources (which then appear in every knob's drive menu)."""
        self._midi_rows = []
        if not self.engine.midi.available():
            tk.Label(parent, text="MIDI unavailable. Install it with:\n\n    pip "
                     "install mido pygame\n\nthen reopen Eyenips — your "
                     "controller's knobs become MIDI 1–8 in every 'drive' menu.",
                     fg="#888", justify="left", wraplength=360,
                     padx=8, pady=8).pack(anchor="w")
            return

        top = tk.Frame(parent, padx=8, pady=4); top.pack(side="top", fill="x")
        tk.Label(top, text="Port").pack(side="left")
        self.midi_port = tk.StringVar(value=self.engine.midi._port_name or "")
        self.midi_port_cb = ttk.Combobox(top, textvariable=self.midi_port, width=18,
                                         state="readonly", values=self.engine.midi.ports())
        self.midi_port_cb.pack(side="left", padx=3)
        tk.Button(top, text="⟳", width=2, command=self._refresh_midi_ports).pack(side="left")
        tk.Button(top, text="Connect", command=self._midi_connect).pack(side="left", padx=3)
        self.midi_status = tk.Label(parent, text=self.engine.midi.status, fg="#888",
                                    padx=8, anchor="w")
        self.midi_status.pack(side="top", fill="x")
        tk.Label(parent, text="Click Learn, then wiggle a knob to bind it. Each "
                 "slot is MIDI 1–8 in every 'drive' menu.", fg="#888",
                 wraplength=360, justify="left", padx=8).pack(side="top", anchor="w")

        rows = tk.Frame(parent, padx=6); rows.pack(side="top", fill="x")
        for i in range(N_MIDI):
            r = tk.Frame(rows); r.pack(fill="x", pady=1)
            tk.Label(r, text=f"MIDI {i + 1}", width=7, anchor="w").pack(side="left")
            info = tk.Label(r, text="—", width=16, anchor="w", fg="#888")
            info.pack(side="left")
            tk.Button(r, text="Learn", width=6,
                      command=lambda idx=i: self._midi_learn(idx)).pack(side="left", padx=2)
            tk.Button(r, text="✕", width=2,
                      command=lambda idx=i: (self.engine.midi.clear_slot(idx),
                                             self._refresh_midi())).pack(side="left")
            self._midi_rows.append(info)
        self._refresh_midi()

    def _refresh_midi_ports(self):
        if hasattr(self, "midi_port_cb"):
            self.midi_port_cb.config(values=self.engine.midi.ports())

    def _midi_connect(self):
        ok = self.engine.midi.open(self.midi_port.get())
        self._set_status("MIDI connected" if ok else "MIDI connect failed", error=not ok)
        if hasattr(self, "midi_status"):
            self.midi_status.config(text=self.engine.midi.status)

    def _midi_learn(self, slot):
        self.engine.midi.learn(slot)
        self._refresh_midi()

    def _refresh_midi(self):
        """Update the 8 slot read-outs (assigned CC + live value / 'learning')."""
        rows = getattr(self, "_midi_rows", None)
        if not rows:
            return
        for i, lbl in enumerate(rows):
            info = self.engine.midi.slot_info(i)
            try:
                if info["learning"]:
                    lbl.config(text="move a control…", fg="#06c")
                elif info["cc"] is None:
                    lbl.config(text="(unassigned)", fg="#888")
                else:
                    bar = "█" * int(info["value"] * 8)
                    lbl.config(text=f"CC {info['cc']:>3}  {info['value']:.2f} {bar}",
                               fg=(self._theme or {}).get("fg", "#888"))
            except tk.TclError:
                pass

    def _load_midi(self, data):
        """After a project load: reflect restored slot mappings; reconnect the
        remembered port if it's still present."""
        if not self.engine.midi.available():
            return
        name = (data or {}).get("port")
        if name and name in self.engine.midi.ports():
            self.engine.midi.open(name)
        if hasattr(self, "midi_port"):
            self.midi_port.set(self.engine.midi._port_name or "")
        if hasattr(self, "midi_status"):
            self.midi_status.config(text=self.engine.midi.status)
        self._refresh_midi()

    def _build_params_panel(self, parent):
        self.param_host = tk.Frame(parent)
        self.param_host.pack(fill="both", expand=True, padx=4, pady=4)
        self._build_params()

    def _build_create_panel(self, parent):
        cnb = ttk.Notebook(parent)
        cnb.pack(fill="both", expand=True, padx=4, pady=4)
        self._build_build_tab(cnb)     # the easiest, no-typing one first
        self._build_expr_tab(cnb)
        self._build_code_tab(cnb)

    # ---- live preview (Taichi canvas -> embedded image) ----------------
    def _tick_fps(self):
        """Measure the real end-to-end frame interval and show a smoothed FPS /
        ms readout in the toolbar (updated a few times a second, not every
        frame, so the text itself never costs anything noticeable)."""
        now = time.perf_counter()
        last, self._fps_t = self._fps_t, now
        if last is None:
            return
        dt = now - last
        if dt > 0:
            inst = 1.0 / dt
            # exponential moving average: smooth but still responsive
            self._fps = inst if self._fps <= 0 else self._fps * 0.9 + inst * 0.1
        self._fps_n += 1
        if self._fps_n % 12 == 0 and self.fps_lbl is not None:
            ms = 1000.0 / self._fps if self._fps > 0 else 0.0
            try:
                self.fps_lbl.config(text=f"{self._fps:4.0f} fps · {ms:4.1f} ms")
            except tk.TclError:
                pass

    def show_frame(self, img):
        """Draw one engine frame into the preview label.

        `img` is normally the GPU-packed (H,W,3) uint8 image — already
        top-left-origin, so we just wrap, resize-to-fit and blit. (If the engine
        falls back to its float path, img is (W,H,3) float and we transpose it
        here.) We reuse one PhotoImage and paste into it, so a steady frame
        size does no per-frame allocation — the old churn that made the preview
        feel less than smooth.

        Records the on-screen image rectangle in self._img_rect, so
        click-to-place maps screen coords back to the canvas at any size."""
        if self._closed or self.root is None:
            return
        self._tick_fps()
        from PIL import Image, ImageTk
        if img.dtype == np.uint8 and img.ndim == 3:
            frame = img                            # GPU-packed, ready to show
        else:                                      # float fallback (W,H,3)
            frame = np.ascontiguousarray(
                (np.flipud(np.transpose(img, (1, 0, 2))) * 255).astype(np.uint8))
        try:
            base = Image.fromarray(frame, "RGB")   # unscaled (canvas W×H)
            iw, ih = base.size
            lw, lh = self.preview.winfo_width(), self.preview.winfo_height()
            if lw > 1 and lh > 1:
                scale = min(lw / iw, lh / ih)      # fit, keep aspect
                dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
                im = base.resize((dw, dh), Image.BILINEAR) if (dw, dh) != (iw, ih) else base
                self._img_rect = ((lw - dw) // 2, (lh - dh) // 2, dw, dh)
            else:
                im = base
                self._img_rect = (0, 0, iw, ih)
            photo = self._photo
            if photo is None or (photo.width(), photo.height()) != im.size:
                photo = ImageTk.PhotoImage(im)     # (re)allocate only on resize
                self.preview.configure(image=photo, text="")
                self._photo = photo
            else:
                photo.paste(im)                    # in place: no allocation/GC
            if self._out_win is not None:
                self._draw_output(base, Image, ImageTk)
        except tk.TclError:
            self._closed = True
            self.engine.running = False

    def _draw_output(self, base, Image, ImageTk):
        """Scale the frame to fill the output monitor (letterboxed), reusing one
        PhotoImage so the projector path is allocation-free too."""
        lbl = self._out_label
        try:
            ow, oh = lbl.winfo_width(), lbl.winfo_height()
            if ow <= 1 or oh <= 1:
                return
            iw, ih = base.size
            scale = min(ow / iw, oh / ih)
            dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
            im = base.resize((dw, dh), Image.BILINEAR)
            photo = self._out_photo
            if photo is None or (photo.width(), photo.height()) != (dw, dh):
                photo = ImageTk.PhotoImage(im)
                lbl.configure(image=photo)
                self._out_photo = photo
            else:
                photo.paste(im)
        except tk.TclError:
            self._close_output()

    # ---- fullscreen / second-monitor output ----------------------------
    def _list_monitors(self):
        """Each connected monitor as (x, y, w, h). Falls back to the primary
        screen if the Windows API isn't available."""
        mons = []
        try:
            import ctypes
            from ctypes import wintypes
            CB = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(wintypes.RECT), ctypes.c_void_p)

            def _cb(_hmon, _hdc, lprect, _data):
                r = lprect.contents
                mons.append((int(r.left), int(r.top),
                             int(r.right - r.left), int(r.bottom - r.top)))
                return 1
            ctypes.windll.user32.EnumDisplayMonitors(None, None, CB(_cb), None)
        except Exception:
            pass
        if not mons:
            mons = [(0, 0, self.root.winfo_screenwidth(),
                     self.root.winfo_screenheight())]
        return mons

    def _open_output(self, rect=None):
        """Open a borderless full-screen output covering `rect` (x,y,w,h);
        default = the monitor the control window is on, else the primary."""
        self._close_output()
        if rect is None:
            mons = self._list_monitors()
            rect = self._pick_monitor(mons)
        x, y, w, h = rect
        win = tk.Toplevel(self.root)
        win.title("Eyenips — Output")
        win.configure(bg="#000")
        win.geometry(f"{w}x{h}+{x}+{y}")
        try:
            win.overrideredirect(True)             # borderless, exact monitor fill
            win.attributes("-topmost", True)
        except tk.TclError:
            pass
        lbl = tk.Label(win, bg="#000")
        lbl.pack(fill="both", expand=True)
        win.bind("<Escape>", lambda e: self._close_output())
        win.bind("<F11>", lambda e: self._close_output())
        win.bind("<Double-Button-1>", lambda e: self._close_output())
        win.protocol("WM_DELETE_WINDOW", self._close_output)
        win.focus_force()
        self._out_win, self._out_label = win, lbl
        self._set_status("Output on — Esc / F11 / double-click to exit")
        self._rebuild_output_menu()

    def _pick_monitor(self, mons):
        """The monitor the control window mostly sits on (so F11 sends output to
        the *other* screen if you dragged controls there)."""
        try:
            cx = self.root.winfo_rootx() + self.root.winfo_width() // 2
            cy = self.root.winfo_rooty() + self.root.winfo_height() // 2
            for (x, y, w, h) in mons:
                if x <= cx < x + w and y <= cy < y + h:
                    # if there's another monitor, prefer it for output
                    others = [m for m in mons if m != (x, y, w, h)]
                    return others[0] if others else (x, y, w, h)
        except tk.TclError:
            pass
        return mons[-1]

    def _close_output(self):
        if self._out_win is not None:
            try:
                self._out_win.destroy()
            except tk.TclError:
                pass
        self._out_win = self._out_label = self._out_photo = None
        self._set_status("Output off")
        self._rebuild_output_menu()

    def _toggle_output(self):
        if self._out_win is not None:
            self._close_output()
        else:
            self._open_output()

    def _rebuild_output_menu(self):
        m = getattr(self, "_output_menu", None)
        if m is None:
            return
        m.delete(0, "end")
        on = self._out_win is not None
        m.add_command(label=("● Output ON  (F11 to toggle)" if on
                             else "Output OFF  (F11 to toggle)"),
                      command=self._toggle_output)
        m.add_separator()
        for i, rect in enumerate(self._list_monitors(), 1):
            x, y, w, h = rect
            m.add_command(label=f"Fullscreen on Monitor {i}  ({w}×{h})",
                          command=lambda r=rect: self._open_output(r))
        if on:
            m.add_separator()
            m.add_command(label="Close output", command=self._close_output)
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _toggle_pause(self):
        self.engine.paused = not self.engine.paused
        if hasattr(self, "pause_btn"):
            self.pause_btn.config(text="▶ Play" if self.engine.paused else "⏸ Pause")

    def pump(self):
        """Service the UI once. Called from the render loop every frame."""
        if self._closed or self.root is None:
            return
        try:
            # rebuild knobs if the engine swapped effects
            if self.engine.effect_version != self._effect_version:
                if self.engine.effect:
                    self.effect_var.set(self.engine.effect.name)
                    self._set_status(f"Effect: {self.engine.effect.name}")
                self._build_params()
                # newly created knob widgets need the current theme applied
                self.apply_theme(self.theme_var.get())
            self._pump_count += 1
            self._draw_meter()  # cheap; gives instant capture feedback
            # surface a live effect (formula/code) error into the builder
            err = self.engine.effect_error or ""
            if err != self._last_effect_error:
                self._last_effect_error = err
                if err:
                    for name in ("expr_status", "code_status"):
                        w = getattr(self, name, None)
                        if w is not None:
                            try:
                                w.config(text=f"⚠ {err}", fg="#a00")
                            except tk.TclError:
                                pass
            # refresh audio/media status a few times a second, not every frame
            if self._pump_count % 10 == 0:
                if self.engine.audio:
                    self.audio_status.config(text=self.engine.audio.status)
                if self.engine.media and hasattr(self, "media_status"):
                    self.media_status.config(text=self.engine.media.status)
                # keep the Media 'Show' dropdown in sync with the store
                if getattr(self, "media_show", None) is not None:
                    v = self.engine.store.values.get("media_blend", "Off")
                    if v != self.media_show.get():
                        self.media_show.set(v)
            # live MIDI read-outs (assigned CC + value / learning) when shown
            if self._pump_count % 6 == 0 and getattr(self, "_midi_rows", None):
                self._refresh_midi()
            # live tempo read-out (BPM drifts in auto mode)
            if self._pump_count % 6 == 0 and getattr(self, "bpm_live", None) is not None:
                try:
                    self.bpm_live.config(text=f"now {self.engine.tempo.bpm():.1f}")
                except tk.TclError:
                    pass
            # Music Director status (background analysis finishing)
            if self._pump_count % 12 == 0 and getattr(self, "director_status", None) is not None:
                self._refresh_director()
            # coalesce changes into undo steps (~twice a second)
            if self._pump_count % 30 == 0:
                self._maybe_snapshot()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True
            self.engine.running = False

    def _on_close(self):
        self._save_session()       # remember everything for next launch
        self._hide_tip()
        self._close_output()
        self._closed = True
        self.engine.running = False

    def close(self):
        if self.root is not None and not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self._closed = True

    # ---- helpers --------------------------------------------------------
    def _wheel_scroll(self, canvas):
        """Scroll `canvas` with the mouse wheel only while the pointer is over
        it. Scoping with Enter/Leave (instead of a permanent bind_all) keeps
        the Build list and the Controls list from fighting over the wheel."""
        def on_wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # ---- theming --------------------------------------------------------
    def apply_theme(self, name):
        th = THEMES.get(name, THEMES["Eyenips"])
        self._theme = th
        if self.theme_var.get() != name:
            self.theme_var.set(name)
        st = self.style
        st.configure("TNotebook", background=th["bg"], borderwidth=0)
        st.configure("TNotebook.Tab", background=th["panel"], foreground=th["fg"],
                     padding=(12, 5))
        st.map("TNotebook.Tab",
               background=[("selected", th["accent"])],
               foreground=[("selected", th["bg"])])
        st.configure("TCombobox", fieldbackground=th["entry"], background=th["btn"],
                     foreground=th["entry_fg"], arrowcolor=th["fg"])
        st.map("TCombobox",
               fieldbackground=[("readonly", th["entry"])],
               foreground=[("readonly", th["entry_fg"])])
        self.root.configure(bg=th["bg"])
        self._recolor(self.root, th)
        # the dropdown menus aren't in the widget tree - recolor them directly
        for m in getattr(self, "_menus", []):
            try:
                m.configure(bg=th["panel"], fg=th["fg"], activebackground=th["accent"],
                            activeforeground=th["bg"], selectcolor=th["accent"])
            except tk.TclError:
                pass
        # floating panels are separate toplevels - recolor them too
        for p in getattr(self, "panels", []):
            tl = p.get("toplevel")
            if tl is not None:
                try:
                    tl.configure(bg=th["bg"])
                    self._recolor(tl, th)
                except tk.TclError:
                    pass

    def _recolor(self, w, th):
        # the output window must stay pure black; tooltips theme themselves
        if w is getattr(self, "_out_win", None) or w is getattr(self, "_tip_win", None):
            return
        cls = w.winfo_class()
        try:
            if w is getattr(self, "preview", None):
                w.configure(bg="#000", fg=th["fg"])
            elif w is getattr(self, "meter", None):
                w.configure(bg=th["meter_bg"])
            elif cls == "Panedwindow":
                w.configure(bg=th["bg"])
            elif cls == "Labelframe":
                w.configure(bg=th["panel"], fg=th["fg"])
            elif cls in ("Frame", "Canvas"):
                w.configure(bg=th["panel"])
            elif cls == "Label":
                w.configure(bg=th["panel"], fg=th["fg"])
            elif cls in ("Button", "Menubutton"):
                w.configure(bg=th["btn"], fg=th["fg"], activebackground=th["accent"],
                            activeforeground=th["bg"], highlightbackground=th["panel"])
            elif cls == "Scale":
                w.configure(bg=th["panel"], fg=th["fg"], troughcolor=th["entry"],
                            activebackground=th["accent"], highlightbackground=th["panel"])
            elif cls in ("Checkbutton", "Radiobutton"):
                w.configure(bg=th["panel"], fg=th["fg"], selectcolor=th["entry"],
                            activebackground=th["panel"], activeforeground=th["fg"],
                            highlightbackground=th["panel"])
            elif cls in ("Entry", "Text"):
                w.configure(bg=th["entry"], fg=th["entry_fg"], insertbackground=th["fg"])
        except tk.TclError:
            pass
        for c in w.winfo_children():
            self._recolor(c, th)

    # ---- media (camera / image / video) --------------------------------
    def _build_media_section(self, parent):
        if self.engine.media is None:
            tk.Label(parent, text="(media unavailable — install opencv-python)",
                     fg="#888").pack(padx=8, pady=8)
            return
        box = tk.Frame(parent, padx=8, pady=4)
        box.pack(fill="both", expand=True)

        self.media_mode = tk.StringVar(value="off")
        row = tk.Frame(box); row.pack(fill="x")
        for label, mode in [("Off", "off"), ("Camera", "camera"),
                            ("Image", "image"), ("Video", "video")]:
            tk.Radiobutton(row, text=label, variable=self.media_mode, value=mode,
                           command=self._on_media_mode).pack(side="left")

        b = tk.Frame(box); b.pack(fill="x", pady=(4, 0))
        tk.Button(b, text="Load image…", command=lambda: self._pick_media("image")).pack(side="left")
        tk.Button(b, text="Load video…", command=lambda: self._pick_media("video")).pack(side="left", padx=6)

        # how the media SHOWS (it's invisible when this is Off — the #1 gotcha)
        srow = tk.Frame(box); srow.pack(fill="x", pady=(4, 0))
        tk.Label(srow, text="Show:").pack(side="left")
        self.media_show = tk.StringVar(value=self.engine.store.values.get("media_blend", "Off"))
        ssb = ttk.Combobox(srow, textvariable=self.media_show, width=9, state="readonly",
                           values=["Off", "Behind", "Tint", "Screen", "Warp"])
        ssb.pack(side="left", padx=3)
        ssb.bind("<<ComboboxSelected>>",
                 lambda e: self.engine.store.set("media_blend", self.media_show.get()))
        tk.Label(srow, text="Behind = plain backdrop", fg="#888").pack(side="left")

        self.media_status = tk.Label(box, text="off", fg="#666",
                                     wraplength=380, justify="left")
        self.media_status.pack(anchor="w", pady=(2, 0))
        tk.Label(box, text="Use it ANY effect (Liquid Fractal, Plasma, ...) via the "
                 "Media blend knob below — set it to Warp to make that effect "
                 "distort your media. For full control (texture / react to motion) "
                 "build an effect in ✎ Create Effect with Output = Texture/Warp.",
                 fg="#888", wraplength=380, justify="left").pack(anchor="w")

    def _on_media_mode(self):
        mode = self.media_mode.get()
        if mode in ("image", "video"):
            self._pick_media(mode)
        else:
            self.engine.media.set_mode(mode)
            self._media_on_hint(mode)

    def _pick_media(self, kind):
        if kind == "image":
            ft = [("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        else:
            ft = [("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All", "*.*")]
        path = filedialog.askopenfilename(title=f"Choose {kind}", filetypes=ft)
        if not path:
            return
        self.media_mode.set(kind)
        self._media_path = path
        self.engine.media.set_mode(kind, path)
        self._media_on_hint(kind)

    def _media_on_hint(self, mode):
        """When a media source is turned on, MAKE IT VISIBLE. Media is invisible
        when 'Show' (media_blend) is Off, which reads as 'media is broken' — so if
        the current effect doesn't already use the media as a texture, flip Show
        to Behind so the camera/image/video appears right away."""
        if mode == "off":
            return
        src = getattr(self.engine.effect, "source", None)
        uses_media = src in ("texture", "warp")     # effect already draws the media
        if not uses_media and self.engine.store.values.get("media_blend", "Off") == "Off":
            self.engine.store.set("media_blend", "Behind")
            if getattr(self, "media_show", None) is not None:
                self.media_show.set("Behind")
            self.media_status.config(
                text=f"{self.engine.media.status} — showing as backdrop. Change "
                     "'Show' to mix it with the effect.", fg="#06c")
        else:
            self.media_status.config(text=self.engine.media.status, fg="#06c")

    # ---- panel content builders (each takes its panel body as `parent`) --
    def _build_audio_section(self, parent):
        box = tk.Frame(parent, padx=8, pady=4)
        box.pack(fill="both", expand=True)

        self.audio_mode = tk.StringVar(value="system")
        row = tk.Frame(box); row.pack(fill="x")
        for label, mode in [("System", "system"), ("Mic", "mic"),
                            ("File", "file"), ("Off", "none")]:
            tk.Radiobutton(row, text=label, variable=self.audio_mode, value=mode,
                           command=self._on_audio_mode).pack(side="left")

        # System = visualize whatever's playing on the PC (Spotify, YouTube, a
        # DAW…). With several outputs we MUST point at the right one or it's
        # black; this picker is what makes the loopback path actually usable.
        sysrow = tk.Frame(box); sysrow.pack(fill="x", pady=(2, 0))
        tk.Label(sysrow, text="System out:").pack(side="left")
        outs = self.engine.audio.list_outputs() if self.engine.audio else []
        self.sys_device_var = tk.StringVar(value=outs[0] if outs else "")
        self.sys_device_menu = ttk.Combobox(
            sysrow, textvariable=self.sys_device_var, values=outs,
            state="readonly", width=22)
        self.sys_device_menu.pack(side="left", padx=(3, 0))
        self.sys_device_menu.bind("<<ComboboxSelected>>", self._on_sys_device)
        tk.Label(box, text="“System” shows whatever is playing on this PC. "
                           "Pick the output your music actually uses.",
                 fg="#888", wraplength=240, justify="left", font=("", 7)
                 ).pack(anchor="w")

        tk.Button(box, text="Load audio file…", command=self._pick_file).pack(
            anchor="w", pady=(4, 0))

        tk.Label(box, text="Sensitivity").pack(anchor="w")
        self.gain = tk.DoubleVar(value=1.0)
        tk.Scale(box, variable=self.gain, from_=0.1, to=5.0, resolution=0.1,
                 orient="horizontal", length=160,
                 command=lambda v: self.engine.audio and self.engine.audio.set_gain(float(v))
                 ).pack(fill="x")

        # live level meter - the diagnostic. If these bars don't move when
        # music plays, capture is the problem (not the visuals/bindings).
        self._meter_labels = ["Vol", "Bass", "Mid", "Treble", "Kick", "Snare", "Hat"]
        self.meter = tk.Canvas(box, height=54, bg="#111", highlightthickness=0)
        self.meter.pack(fill="x", pady=(4, 0))

        self.audio_status = tk.Label(box, text="off", fg="#666")
        self.audio_status.pack(anchor="w")

    def _draw_meter(self):
        f = self.engine.audio.features() if self.engine.audio else None
        c = self.meter
        c.delete("all")
        vals = ([f.volume, f.bass, f.mid, f.treble, f.kick, f.snare, f.hihat]
                if f else [0] * 7)
        beat = f.beat if f else False
        w = max(self.meter.winfo_width(), 60)   # follow the panel's real width
        n = len(vals)
        bw = w / n
        for i, (lab, v) in enumerate(zip(self._meter_labels, vals)):
            x0 = i * bw + 3
            x1 = (i + 1) * bw - 3
            h = max(2, min(1.0, v) * 40)
            col = "#3fd0ff" if not (beat and i == 1) else "#ff5050"
            if i >= 4:                       # drum bands in a warmer colour
                col = "#ffd24a"
            c.create_rectangle(x0, 48 - h, x1, 48, fill=col, outline="")
            c.create_text((x0 + x1) / 2, 52, text=lab, fill="#888", font=("", 7))

    def _build_effect_picker(self, parent):
        box = tk.Frame(parent, padx=8, pady=4)
        box.pack(fill="both", expand=True)
        names = [c.name for c in self.effect_classes]
        self.effect_var = tk.StringVar(
            value=self.engine.effect.name if self.engine.effect else (names[0] if names else ""))
        self.effect_menu = ttk.Combobox(box, textvariable=self.effect_var, values=names,
                                        state="readonly")
        self.effect_menu.pack(fill="x")
        self.effect_menu.bind("<<ComboboxSelected>>", self._on_effect_change)

        btns = tk.Frame(box); btns.pack(fill="x", pady=(4, 0))
        tk.Button(btns, text="Reset effect", command=self._reset).pack(side="left")
        tk.Button(btns, text="🎲 Randomize", command=self._randomize_recipe).pack(side="left", padx=6)
        tk.Button(btns, text="✎ Edit equations…", command=self._open_lab_editor).pack(side="left", padx=6)
        tk.Button(btns, text="✨ Shapes…", command=self._open_shapes).pack(side="left", padx=6)
        tk.Button(btns, text="🧱 Layers…", command=self._open_layers).pack(side="left", padx=6)
        tk.Button(btns, text="✎ Create Effect…", command=self._goto_create).pack(side="left", padx=6)

    def _randomize_recipe(self):
        """Roll a new Recipe for a generative effect (Effect Lab) — a fresh effect
        from its millions, one click. No-op (with a hint) for fixed effects."""
        import random
        target = next((p for p in self.engine.params
                       if isinstance(p, IntSlider) and p.name in ("recipe", "seed")), None)
        if target is None:
            self._set_status("This effect has no Recipe to randomize "
                             "(try the Effect Lab).")
            return
        val = random.randint(int(target.lo), int(target.hi))
        self.engine.store.set(target.name, val)
        self._build_params()                      # reflect the new value on the knob
        self.apply_theme(self.theme_var.get())    # fresh widgets need the theme re-applied
        self._set_status(f"🎲 Recipe {val}")

    # ---- Lab Kit editor: edit the generative equations & recipe ranges ---
    _EL_PAIR = [("f1", "freq 1"), ("f2", "freq 2"), ("f3", "freq 3"),
                ("fw1", "warp freq 1"), ("fw2", "warp freq 2"),
                ("colscale", "colour scale"), ("coloff", "colour offset"),
                ("bands", "colour bands"), ("contrast", "line contrast"),
                ("levels", "poster levels"), ("spd", "speed")]
    _EL_LIST = [("mix_weights", "blend modes (0=single,1=avg,2=mult,3=max,4=mask)"),
                ("sym_weights", "symmetry (0=none,1=mirror,2=kaleido,3=4-way)"),
                ("nfold_choices", "kaleidoscope folds"),
                ("style_weights", "render style (0=smooth,1=line,2=flats)"),
                ("colmode_weights", "colour mode (0=value,1=angle,2=banded,3=radius)"),
                ("warp_choices", "domain-warp amounts")]
    _VL_PAIR = [("dominant", "dominant ops (count)"), ("accent", "accent ops (count)"),
                ("dominant_weight", "dominant strength"), ("accent_weight", "accent strength"),
                ("fold", "kaleido folds"), ("tiles", "mosaic tiles"), ("woff", "split offset"),
                ("levels", "poster levels"), ("colscale", "colour scale"),
                ("coloff", "colour offset"), ("fw1", "wave freq 1"), ("fw2", "wave freq 2"),
                ("edgeg", "edge gain"), ("ovsize", "overlay size"), ("disp", "displacement"),
                ("spd", "speed")]

    def _open_lab_editor(self):
        """Open the equation/recipe editor for the current generative lab."""
        eff = self.engine.effect
        kind = getattr(eff, "editable_kit", None)
        if not kind:
            self._set_status("Only the generative labs (Effect Lab / Video Lab) "
                             "have editable equations — pick one first.")
            return
        if getattr(self, "_lab_win", None) is not None:
            try:
                self._lab_win.destroy()
            except Exception:
                pass
        win = tk.Toplevel(self.root)
        self._lab_win = win
        self._lab_kind = kind
        win.title(f"Edit equations — {eff.name}")
        win.geometry("760x680")
        self._lab_body = tk.Frame(win)
        self._lab_body.pack(fill="both", expand=True)

        bar = tk.Frame(win, padx=8, pady=6)
        bar.pack(fill="x", side="bottom")
        self._lab_status = tk.Label(bar, text="", fg="#888", anchor="w",
                                    wraplength=440, justify="left")
        self._lab_status.pack(side="left", fill="x", expand=True)
        tk.Button(bar, text="Reset to defaults", command=self._lab_reset).pack(side="right")
        tk.Button(bar, text="✓ Apply", command=self._lab_apply).pack(side="right", padx=6)

        self._lab_render(eff.current_kit())

    def _lab_render(self, kit):
        """(Re)build the editor widgets from a kit dict (used on open + Reset)."""
        for c in self._lab_body.winfo_children():
            c.destroy()
        # scrollable inner frame
        canvas = tk.Canvas(self._lab_body, highlightthickness=0)
        sb = tk.Scrollbar(self._lab_body, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._wheel_scroll(canvas)
        if self._lab_kind == "effect_lab":
            self._build_effectlab_editor(inner, kit)
        else:
            self._build_videolab_editor(inner, kit)

    def _lab_pair_row(self, parent, label, pair):
        """A 'lo .. hi' range editor row. Returns (lo_var, hi_var)."""
        row = tk.Frame(parent)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, width=24, anchor="w").pack(side="left")
        lo = tk.StringVar(value=str(pair[0]))
        hi = tk.StringVar(value=str(pair[1]))
        tk.Entry(row, textvariable=lo, width=8).pack(side="left")
        tk.Label(row, text="…").pack(side="left", padx=2)
        tk.Entry(row, textvariable=hi, width=8).pack(side="left")
        return lo, hi

    def _build_effectlab_editor(self, inner, kit):
        el = kit["effect_lab"]
        tk.Label(inner, text="ARCHETYPE EQUATIONS", font=("", 10, "bold")).pack(
            anchor="w", padx=8, pady=(8, 0))
        tk.Label(inner, justify="left", fg="#666", wraplength=700,
                 text=("Each is a formula in u, v (position −1..1), t (time), f1/f2/f3 "
                       "(recipe frequencies) and ph (phase). Functions: sin cos tan sqrt "
                       "abs floor ceil exp log min max atan2 round, plus building blocks "
                       "vor(x,y,t), julia(u,v,f1,f2,ph,t), fbm(x,y), vnoise(x,y). Range "
                       "≈ −1..1.")).pack(anchor="w", padx=8)
        self._el_rows = []
        self._el_host = tk.Frame(inner)
        self._el_host.pack(fill="x", padx=8, pady=2)
        for a in el["archetypes"]:
            self._el_add_row(a.get("name", ""), a.get("formula", ""))
        tk.Button(inner, text="+ Add archetype",
                  command=lambda: self._el_add_row("new", "sin(u*f1+t)*cos(v*f2-t)")).pack(
            anchor="w", padx=8, pady=(2, 8))

        tk.Label(inner, text="RECIPE RANGES", font=("", 10, "bold")).pack(
            anchor="w", padx=8, pady=(6, 0))
        tk.Label(inner, fg="#666", text="How a Recipe number is rolled into an effect.").pack(
            anchor="w", padx=8)
        self._el_pair_vars = {}
        pf = tk.Frame(inner)
        pf.pack(fill="x", padx=8, pady=2)
        for key, lab in self._EL_PAIR:
            self._el_pair_vars[key] = self._lab_pair_row(pf, lab, el["recipe"][key])
        self._el_list_vars = {}
        lf = tk.Frame(inner)
        lf.pack(fill="x", padx=8, pady=2)
        for key, lab in self._EL_LIST:
            row = tk.Frame(lf)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lab, width=40, anchor="w").pack(side="left")
            var = tk.StringVar(value=", ".join(str(x) for x in el["recipe"][key]))
            tk.Entry(row, textvariable=var, width=26).pack(side="left", fill="x", expand=True)
            self._el_list_vars[key] = var

    def _el_add_row(self, name, formula):
        row = tk.Frame(self._el_host)
        row.pack(fill="x", pady=1)
        nv = tk.StringVar(value=name)
        fv = tk.StringVar(value=formula)
        tk.Entry(row, textvariable=nv, width=12).pack(side="left")
        tk.Entry(row, textvariable=fv, width=58).pack(side="left", fill="x", expand=True, padx=2)
        rec = {"name": nv, "formula": fv, "frame": row}
        tk.Button(row, text="✕", width=2,
                  command=lambda: self._el_del_row(rec)).pack(side="left")
        self._el_rows.append(rec)

    def _el_del_row(self, rec):
        if len(self._el_rows) <= 1:
            self._lab_status.config(text="Keep at least one archetype.", fg="#a60")
            return
        rec["frame"].destroy()
        self._el_rows.remove(rec)

    def _build_videolab_editor(self, inner, kit):
        vl = kit["video_lab"]
        tk.Label(inner, text="OPERATORS IN THE RANDOM POOL", font=("", 10, "bold")).pack(
            anchor="w", padx=8, pady=(8, 0))
        tk.Label(inner, fg="#666", wraplength=700, justify="left",
                 text=("Tick the operators a Recipe may draw from. The operator MATH is "
                       "built in; uncheck any you don't want appearing.")).pack(
            anchor="w", padx=8)
        self._vl_op_vars = {}
        for grp, title in (("warp", "Warp the footage"), ("color", "Restyle colour"),
                           ("over", "Motion overlays")):
            f = tk.LabelFrame(inner, text=title, padx=6, pady=2)
            f.pack(fill="x", padx=8, pady=3)
            for o in labkit.VIDEO_GROUPS[grp]:
                var = tk.BooleanVar(value=bool(vl["operators"].get(o, True)))
                tk.Checkbutton(f, text=o, variable=var).pack(side="left")
                self._vl_op_vars[o] = var

        tk.Label(inner, text="RECIPE RANGES", font=("", 10, "bold")).pack(
            anchor="w", padx=8, pady=(8, 0))
        self._vl_pair_vars = {}
        pf = tk.Frame(inner)
        pf.pack(fill="x", padx=8, pady=2)
        for key, lab in self._VL_PAIR:
            self._vl_pair_vars[key] = self._lab_pair_row(pf, lab, vl["recipe"][key])

    # ---- read / validate / apply ---------------------------------------
    def _parse_num(self, s):
        s = s.strip()
        f = float(s)
        return int(f) if f == int(f) and "." not in s and "e" not in s.lower() else f

    def _collect_effectlab_kit(self):
        kit = labkit.defaults()
        archs = []
        for rec in self._el_rows:
            name = rec["name"].get().strip() or "arch"
            formula = rec["formula"].get().strip()
            if not formula:
                continue
            err = labkit.validate_formula(formula)
            if err:
                raise ValueError(f"'{name}': {err}")
            archs.append({"name": name, "formula": formula})
        if not archs:
            raise ValueError("need at least one archetype with a formula")
        kit["effect_lab"]["archetypes"] = archs
        rc = kit["effect_lab"]["recipe"]
        for key, (lo, hi) in self._el_pair_vars.items():
            rc[key] = [self._parse_num(lo.get()), self._parse_num(hi.get())]
        for key, var in self._el_list_vars.items():
            nums = [self._parse_num(x) for x in var.get().split(",") if x.strip()]
            if not nums:
                raise ValueError(f"'{key}' needs at least one value")
            rc[key] = nums
        return kit

    def _collect_videolab_kit(self):
        kit = labkit.defaults()
        for o, var in self._vl_op_vars.items():
            kit["video_lab"]["operators"][o] = bool(var.get())
        if not any(kit["video_lab"]["operators"].values()):
            raise ValueError("enable at least one operator")
        rc = kit["video_lab"]["recipe"]
        for key, (lo, hi) in self._vl_pair_vars.items():
            rc[key] = [self._parse_num(lo.get()), self._parse_num(hi.get())]
        return kit

    def _lab_apply(self):
        eff = self.engine.effect
        if getattr(eff, "editable_kit", None) != self._lab_kind:
            self._lab_status.config(
                text=f"Switch back to the {self._lab_kind.replace('_',' ')} effect to apply.",
                fg="#a60")
            return
        try:
            if self._lab_kind == "effect_lab":
                kit = self._collect_effectlab_kit()
            else:
                kit = self._collect_videolab_kit()
        except (ValueError, Exception) as e:
            self._lab_status.config(text=f"✗ {e}", fg="#c00")
            return
        eff.set_kit(kit)
        # surface a compile error from the next render (Effect Lab) if any
        ok, msg = labkit.save(kit)
        err = getattr(eff, "error", "")
        if err:
            self._lab_status.config(text=f"⚠ applied, but: {err}", fg="#a60")
        elif ok:
            self._lab_status.config(text="✓ Applied & saved.", fg="#080")
        else:
            self._lab_status.config(text=f"Applied (save failed: {msg})", fg="#a60")

    def _lab_reset(self):
        ok, msg = labkit.reset()
        kit = labkit.defaults()
        eff = self.engine.effect
        if getattr(eff, "editable_kit", None) == self._lab_kind:
            eff.set_kit(kit)
        self._lab_render(kit)
        self._lab_status.config(text="↺ Reset to defaults." if ok else f"Reset: {msg}",
                                fg="#080" if ok else "#a60")

    def _build_export_section(self, parent):
        box = tk.Frame(parent, padx=8, pady=4)
        box.pack(fill="both", expand=True)

        row = tk.Frame(box); row.pack(fill="x")
        tk.Label(row, text="FPS").pack(side="left")
        self.fps_var = tk.StringVar(value="30")
        tk.Entry(row, textvariable=self.fps_var, width=5).pack(side="left", padx=(2, 10))
        tk.Label(row, text="Seconds (blank = full song)").pack(side="left")
        self.secs_var = tk.StringVar(value="")
        tk.Entry(row, textvariable=self.secs_var, width=6).pack(side="left", padx=2)

        brow = tk.Frame(box); brow.pack(fill="x", pady=(4, 0))
        tk.Button(brow, text="Export MP4 (audio)…", command=self._on_export).pack(side="left")
        tk.Button(brow, text="🎬 Export VIDEO…", command=self._on_export_video).pack(
            side="left", padx=6)
        self.stop_export_btn = tk.Button(brow, text="■ Stop", command=self._stop_export,
                                         state="disabled")
        self.stop_export_btn.pack(side="left", padx=6)
        self.export_status = tk.Label(box, text="“Export MP4” renders the effect to your "
                                      "audio. “Export VIDEO” runs the effect OVER a video "
                                      "clip (e.g. Video Lab) and bakes in the result. Use "
                                      "“■ Stop” to end a render early (keeps what's done).",
                                      fg="#888", wraplength=380, justify="left")
        self.export_status.pack(anchor="w")

    def _stop_export(self):
        if self.engine:
            self.engine.cancel_export()
            self.export_status.config(text="Stopping…", fg="#a60")

    def _export_running(self, on):
        """Enable the Stop button only while a render is in flight."""
        if getattr(self, "stop_export_btn", None) is not None:
            try:
                self.stop_export_btn.config(state="normal" if on else "disabled")
            except tk.TclError:
                pass

    def _on_export_video(self):
        """Render-through-video: run the current effect over an input clip."""
        video_path = (self.engine.media._path if self.engine.media
                      and self.engine.media._mode == "video" else None)
        video_path = video_path or getattr(self, "_media_path", None)
        if not (video_path and os.path.exists(video_path)):
            video_path = filedialog.askopenfilename(
                title="Choose the video to run the effect over",
                filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All", "*.*")])
        if not video_path:
            self.export_status.config(text="Video export cancelled (no clip chosen).", fg="#a00")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save augmented video as", defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")])
        if not out_path:
            return
        try:
            seconds = float(self.secs_var.get()) if self.secs_var.get().strip() else None
        except ValueError:
            seconds = None

        def progress(done, total, msg):
            self.export_status.config(text=f"{msg}  ({100*done//total}%)", fg="#06c")
            try:
                self.root.update()
            except tk.TclError:
                pass

        def finished(ok, msg):
            self.export_status.config(text=msg, fg="#070" if ok else "#a00")
            self._export_running(False)

        self.export_status.config(text="Starting video export…", fg="#06c")
        self._export_running(True)
        self.engine.request_export({
            "video_path": video_path, "out_path": out_path,
            "seconds": seconds, "progress": progress, "done": finished,
        })

    def _on_export(self):
        # 1) need an audio file to embed (system/mic are live - nothing to bake)
        audio_path = self.engine.audio.current_file() if self.engine.audio else None
        audio_path = audio_path or self._loaded_audio
        if not audio_path:
            audio_path = filedialog.askopenfilename(
                title="Choose the audio to put in the video",
                filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")])
        if not audio_path:
            self.export_status.config(text="Export cancelled (no audio chosen).", fg="#a00")
            return

        # 2) where to save
        out_path = filedialog.asksaveasfilename(
            title="Save video as", defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")])
        if not out_path:
            return

        try:
            fps = max(1, int(float(self.fps_var.get())))
        except ValueError:
            fps = 30
        try:
            seconds = float(self.secs_var.get()) if self.secs_var.get().strip() else None
        except ValueError:
            seconds = None

        def progress(done, total, msg):
            self.export_status.config(text=f"{msg}  ({100*done//total}%)", fg="#06c")
            try:
                self.root.update()
            except tk.TclError:
                pass

        def finished(ok, msg):
            self.export_status.config(text=msg, fg="#070" if ok else "#a00")
            self._export_running(False)

        self.export_status.config(text="Starting export…", fg="#06c")
        self._export_running(True)
        self.engine.request_export({
            "audio_path": audio_path, "out_path": out_path,
            "fps": fps, "seconds": seconds,
            "progress": progress, "done": finished,
        })

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")])
        if path:
            self._loaded_audio = path
            self.audio_mode.set("file")
            self.engine.audio.set_mode("file", path)
            self.engine.analyze_track(path)     # Music Director: map the song
            self._refresh_director()

    def _on_audio_mode(self):
        mode = self.audio_mode.get()
        if mode == "file":
            self._pick_file()
        elif mode == "system":
            dev = getattr(self, "sys_device_var", None)
            self.engine.audio.set_mode("system", device=dev.get() if dev else None)
        else:
            self.engine.audio.set_mode(mode)

    def _on_sys_device(self, _evt=None):
        """Chose a different output to loop back; re-open if System is live."""
        if self.audio_mode.get() == "system":
            self.engine.audio.set_mode("system", device=self.sys_device_var.get())

    def _on_effect_change(self, _evt=None):
        name = self.effect_var.get()
        for c in self.effect_classes:
            if c.name == name:
                self.engine.request_effect(c)
                self._effect_empty_hint(c)
                break

    def _effect_empty_hint(self, cls):
        """Nudge the user when a video effect has no footage to act on — the #1
        'I picked it and nothing happens' confusion. Otherwise just name it."""
        if getattr(cls, "uses_video", False):
            mode = getattr(self.engine.media, "_mode", "off") if self.engine.media else "off"
            if mode not in ("video", "camera", "image"):
                self._set_status("🎬 This is a video effect — load a video or camera "
                                 "in the Media panel to see it act on footage "
                                 "(it still works as a generative visual without one).")
                return
        self._set_status(f"Effect: {cls.name}")

    def _reset(self):
        if self.engine.effect:
            # reset runs on the engine thread via a flag to stay GPU-safe
            self.engine.request_reset()

    # ---- Create Effect (its own panel) ---------------------------------
    def _goto_create(self):
        p = self._panel("create")
        self._set_visible(p, True)
        if p["floating"] and p["toplevel"] is not None:
            try:
                p["toplevel"].deiconify(); p["toplevel"].lift()
            except tk.TclError:
                pass

    def _switch_to(self, name):
        """Make the effect with this name active (live)."""
        for c in self.effect_classes:
            if c.name == name:
                self.engine.request_effect(c)
                self.effect_var.set(name)
                return True
        return False

    # ---- projects: save / load / session / presets ---------------------
    def _capture_state(self, with_layout=True):
        """The whole creative state as a plain dict (engine + UI parts)."""
        st = self.engine.export_state()
        st["shapes"] = self._collect_shape_dicts()
        st["theme"] = self.theme_var.get()
        st["audio"] = {
            "mode": self.audio_mode.get() if hasattr(self, "audio_mode") else "system",
            "gain": float(self.gain.get()) if hasattr(self, "gain") else 1.0,
            "file": self._loaded_audio,
            "device": self.sys_device_var.get() if hasattr(self, "sys_device_var") else "",
        }
        if hasattr(self, "media_mode"):
            st["media"] = {"mode": self.media_mode.get(), "path": self._media_path}
        if with_layout:
            st["layout"] = self._capture_layout()
        return st

    def _capture_layout(self):
        try:
            geo = self.root.geometry()
        except tk.TclError:
            geo = None
        return {"geometry": geo,
                "panels": {p["key"]: {"visible": p["visible"]} for p in self.panels}}

    def _apply_state(self, data, with_layout=True):
        """Restore a project dict. Order matters: engine first, then the shape
        rows, then audio/media, then (safe) layout."""
        if not data:
            return
        th = data.get("theme")
        if th in THEMES:
            self.theme_var.set(th)
        # effect + primary knobs + per-shape effect knob caches
        self.engine.import_state(data)
        if self.engine.effect:
            self.effect_var.set(self.engine.effect.name)
        self._build_params()                       # reflect loaded primary knobs
        self._load_shapes(data.get("shapes") or [])
        self._load_layers(data.get("layers") or [])
        self._load_midi(data.get("midi") or {})
        self._load_tempo()
        self._load_director()
        self._apply_audio(data.get("audio") or {})
        self._apply_media(data.get("media") or {})
        if with_layout and data.get("layout"):
            self._apply_layout(data["layout"])
        self.apply_theme(self.theme_var.get())

    def _load_shapes(self, shape_dicts):
        for it in list(self.shape_items):
            try:
                it["frame"].destroy()
            except tk.TclError:
                pass
        self.shape_items = []
        for sd in shape_dicts:
            fields = {k: v for k, v in sd.items() if k != "id"}
            self._add_shape(id=sd.get("id"), **fields)
        self._push_shapes()
        self._refresh_shape_params()

    def _apply_audio(self, a):
        mode = a.get("mode", "system")
        gain = float(a.get("gain", 1.0))
        f = a.get("file")
        dev = a.get("device") or ""
        if hasattr(self, "audio_mode"):
            self.audio_mode.set(mode)
        if hasattr(self, "gain"):
            self.gain.set(gain)
        # restore the saved output device if it's still present
        if dev and hasattr(self, "sys_device_menu"):
            if dev in self.sys_device_menu.cget("values"):
                self.sys_device_var.set(dev)
        if self.engine.audio:
            self.engine.audio.set_gain(gain)
            if mode == "file" and f and os.path.exists(f):
                self._loaded_audio = f
                self.engine.audio.set_mode("file", f)
                self.engine.analyze_track(f)        # re-map the song for the Director
            elif mode == "system":
                self.engine.audio.set_mode("system", device=self.sys_device_var.get()
                                           if hasattr(self, "sys_device_var") else None)
            else:
                self.engine.audio.set_mode("none" if mode == "file" else mode)

    def _apply_media(self, m):
        if not (hasattr(self, "media_mode") and self.engine.media):
            return
        mode = m.get("mode", "off")
        path = m.get("path")
        if mode in ("image", "video") and path and os.path.exists(path):
            self.media_mode.set(mode); self._media_path = path
            self.engine.media.set_mode(mode, path)
        else:
            self.media_mode.set(mode if mode in ("off", "camera") else "off")
            self.engine.media.set_mode(mode if mode in ("off", "camera") else "off")

    def _apply_layout(self, layout):
        geo = layout.get("geometry")
        if geo:
            try:
                self.root.geometry(geo)
            except tk.TclError:
                pass
        # _set_visible preserves widgets (no rebuild), so shapes survive
        for p in self.panels:
            cfg = (layout.get("panels") or {}).get(p["key"])
            if cfg is not None and "visible" in cfg:
                self._set_visible(p, bool(cfg["visible"]))

    def _save_project(self, save_as=False):
        path = self._project_path
        if save_as or not path:
            os.makedirs(self._presets_dir, exist_ok=True)
            path = filedialog.asksaveasfilename(
                title="Save project", defaultextension=project.EXT,
                initialdir=self._presets_dir,
                filetypes=[("Eyenips project", "*" + project.EXT), ("All", "*.*")])
            if not path:
                return
        try:
            project.save(path, self._capture_state())
            self._project_path = path
            self._set_status(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"Save failed: {e}", error=True)

    def _open_project(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(
                title="Open project", initialdir=self._presets_dir,
                filetypes=[("Eyenips project", "*" + project.EXT), ("All", "*.*")])
        if not path:
            return
        try:
            self._apply_state(project.load(path))
            self._project_path = path
            self._set_status(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"Open failed: {e}", error=True)

    def _preset_path(self, name):
        """Resolve a preset name to a file: a user-saved copy wins over the
        bundled starter of the same name."""
        user = os.path.join(self._presets_dir, name + project.EXT)
        if os.path.exists(user):
            return user
        return os.path.join(self._builtin_presets_dir, name + project.EXT)

    def _save_preset(self):
        os.makedirs(self._presets_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save preset", defaultextension=project.EXT,
            initialdir=self._presets_dir,
            filetypes=[("Eyenips preset", "*" + project.EXT)])
        if not path:
            return
        try:
            project.save(path, self._capture_state())
            self._set_status(f"Preset saved: {os.path.splitext(os.path.basename(path))[0]}")
            self._rebuild_project_menu()
        except Exception as e:
            self._set_status(f"Preset failed: {e}", error=True)

    def _save_session(self):
        try:
            project.save(self._session_path, self._capture_state())
        except Exception:
            pass

    def _restore_session(self):
        if os.path.exists(self._session_path):
            try:
                self._apply_state(project.load(self._session_path))
                return
            except Exception as e:
                paths.safe_print(f"[session] could not restore: {e}")
        # fresh launch: start the default source so "System" works on day one
        # (loopback is silent until something plays — no harm, instant payoff)
        if self.engine.audio and self.audio_mode.get() != "none":
            self._on_audio_mode()

    # ---- undo / redo (coalesced state history) -------------------------
    def _init_history(self):
        s = self._capture_state(with_layout=False)
        self._hist = [s]
        self._hist_i = 0
        self._hist_last = json.dumps(s, sort_keys=True, default=str)
        self._hist_freeze = 0          # suppress capture right after an undo/redo
        self._restoring = False

    def _maybe_snapshot(self):
        """Poll for a settled change and push it as one undo step."""
        if getattr(self, "_restoring", False) or not getattr(self, "_hist", None):
            return
        if self._pump_count < self._hist_freeze:
            return
        s = self._capture_state(with_layout=False)
        js = json.dumps(s, sort_keys=True, default=str)
        if js == self._hist_last:
            return
        self._hist = self._hist[:self._hist_i + 1]      # drop redo branch
        self._hist.append(s)
        if len(self._hist) > 60:                         # cap memory
            self._hist.pop(0)
        self._hist_i = len(self._hist) - 1
        self._hist_last = js

    def _undo(self):
        if not getattr(self, "_hist", None) or self._hist_i <= 0:
            self._set_status("Nothing to undo")
            return
        self._hist_i -= 1
        self._restore_history("Undo")

    def _redo(self):
        if not getattr(self, "_hist", None) or self._hist_i >= len(self._hist) - 1:
            self._set_status("Nothing to redo")
            return
        self._hist_i += 1
        self._restore_history("Redo")

    def _restore_history(self, label):
        self._restoring = True
        target = self._hist[self._hist_i]
        self._apply_state(target, with_layout=False)
        self._hist_last = json.dumps(target, sort_keys=True, default=str)
        self._hist_freeze = self._pump_count + 45     # let it settle (~0.75s)
        self._restoring = False
        self._set_status(f"{label}  ({self._hist_i + 1}/{len(self._hist)})")

    def _set_status(self, text, error=False):
        w = getattr(self, "statusbar", None)
        if w is not None:
            try:
                th = getattr(self, "_theme", None) or {}
                w.config(text=text, fg=("#ff6b6b" if error else th.get("fg", "#888")))
            except tk.TclError:
                pass

    # ---- Shapes (elements): an overlay that interacts with the active effect -
    def _open_shapes(self, *_a, **_k):
        """Show the Shapes editor. Shapes layer over WHATEVER effect is active,
        so this never changes the effect. Adds a starter shape if empty."""
        p = self._panel("shapes")
        self._set_visible(p, True)
        if p["floating"] and p["toplevel"] is not None:
            try:
                p["toplevel"].deiconify(); p["toplevel"].lift()
            except tk.TclError:
                pass
        if not self.shape_items:               # first open: show one so it's not empty
            self._add_shape(shape="Circle", mode="Show effect")

    def _build_shapes_panel(self, parent):
        self.shape_items = []
        self._shape_id = 0
        self.shape_sel = tk.StringVar(value="")

        head = tk.Frame(parent, padx=8, pady=4); head.pack(side="top", fill="x")
        tk.Label(head, text="The main effect (Effects panel) always fills the "
                 "screen. Shapes sit ON TOP and only change pixels inside "
                 "themselves. Select one (◉), then CLICK the preview to place it — "
                 "drag to move. 'Show effect' paints a chosen effect inside the "
                 "shape; 'Warp' bends the main effect; 'Hide' cuts a hole; 'Tint' "
                 "colors it.", wraplength=360, justify="left",
                 font=("", 9, "bold")).pack(anchor="w")

        # Pack the action row + status at the BOTTOM first, so the "Add shape"
        # button is always visible no matter how short the panel is; the
        # scrollable list then fills whatever height is left.
        self.shapes_status = tk.Label(parent, text="", fg="#888",
                                      wraplength=360, justify="left")
        self.shapes_status.pack(side="bottom", anchor="w", padx=8, pady=(0, 4))
        btns = tk.Frame(parent, padx=8, pady=4); btns.pack(side="bottom", fill="x")
        tk.Button(btns, text="➕ Add shape", command=lambda: self._add_shape()).pack(side="left")
        tk.Button(btns, text="🗑 Clear all", command=self._clear_shapes).pack(side="left", padx=6)
        tk.Button(btns, text="⚙ Shape FX", command=self._open_shapefx).pack(side="left")

        listwrap = tk.Frame(parent); listwrap.pack(side="top", fill="both",
                                                   expand=True, padx=4)
        canvas = tk.Canvas(listwrap, highlightthickness=0)
        sb = tk.Scrollbar(listwrap, orient="vertical", command=canvas.yview)
        self.shapes_host = tk.Frame(canvas)
        win = canvas.create_window((0, 0), window=self.shapes_host, anchor="nw")
        self.shapes_host.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._wheel_scroll(canvas)
        # no shapes by default — the overlay stays off until the user adds one

    def _add_shape(self, user=True, id=None, **d):
        # reuse a saved id (so its effect-knob cache matches) or mint a new one
        if id:
            sid = id
            try:
                self._shape_id = max(self._shape_id, int(str(id).lstrip("s")))
            except ValueError:
                pass
        else:
            self._shape_id += 1
            sid = f"s{self._shape_id}"
        box = tk.LabelFrame(self.shapes_host, padx=6, pady=3)
        box.pack(fill="x", padx=4, pady=2)
        it = {"frame": box, "id": sid}

        top = tk.Frame(box); top.pack(fill="x")
        tk.Radiobutton(top, variable=self.shape_sel, value=sid,
                       command=self._refresh_shape_params).pack(side="left")
        it["title"] = tk.Label(top, text="Shape")
        it["title"].pack(side="left")
        it["shape"] = tk.StringVar(value=d.get("shape", "Circle"))
        cb = ttk.Combobox(top, textvariable=it["shape"], width=9, state="readonly",
                          values=shapes.SHAPE_CHOICES)
        cb.pack(side="left", padx=3); cb.bind("<<ComboboxSelected>>", lambda e: self._push_shapes())
        it["mode"] = tk.StringVar(value=d.get("mode", "Show effect"))
        cs = ttk.Combobox(top, textvariable=it["mode"], width=14, state="readonly",
                          values=shapes.MODE_CHOICES)
        cs.pack(side="left", padx=3)
        cs.bind("<<ComboboxSelected>>",
                lambda e: (self._push_shapes(), self._refresh_shape_params()))
        tk.Button(top, text="✕", width=2, fg="#a00",
                  command=lambda: self._remove_shape(it)).pack(side="right")

        # which effect a "Show effect" shape paints inside it: Primary (the main
        # effect) or any other effect. Ignored by non-window modes.
        rsrc = tk.Frame(box); rsrc.pack(fill="x")
        tk.Label(rsrc, text="Shows:").pack(side="left")
        it["effect"] = tk.StringVar(value=d.get("effect", shapes.PRIMARY))
        eff_vals = [shapes.PRIMARY] + [c.name for c in self.effect_classes]
        ce = ttk.Combobox(rsrc, textvariable=it["effect"], state="readonly",
                          values=eff_vals)
        ce.pack(side="left", fill="x", expand=True, padx=3)
        ce.bind("<<ComboboxSelected>>",
                lambda e: (self._push_shapes(), self._refresh_shape_params()))
        tk.Button(rsrc, text="⚙", width=2,
                  command=lambda: (self.shape_sel.set(sid), self._open_shapefx())
                  ).pack(side="left", padx=2)

        r2 = tk.Frame(box); r2.pack(fill="x")
        it["x"] = self._shape_slider(r2, "X", 0.0, 1.0, d.get("x", 0.5), 0.01)
        it["y"] = self._shape_slider(r2, "Y", 0.0, 1.0, d.get("y", 0.5), 0.01)
        it["size"] = self._shape_slider(r2, "Size", 0.02, 0.6, d.get("size", 0.18), 0.01)
        it["rotation"] = self._shape_slider(r2, "Rotate", 0.0, 1.0, d.get("rotation", 0.0), 0.01)

        r3 = tk.Frame(box); r3.pack(fill="x")
        it["hue"] = self._shape_slider(r3, "Color", 0.0, 1.0, d.get("hue", 0.0), 0.01)
        tk.Label(r3, text="Reacts:").pack(side="left")
        it["react"] = tk.StringVar(value=d.get("react", "Nothing"))
        cr = ttk.Combobox(r3, textvariable=it["react"], width=8, state="readonly",
                          values=shapes.REACT_CHOICES)
        cr.pack(side="left", padx=3); cr.bind("<<ComboboxSelected>>", lambda e: self._push_shapes())

        r4 = tk.Frame(box); r4.pack(fill="x")
        it["strength"] = self._shape_slider(r4, "Strength", 0, 4, d.get("strength", 1.5), 0.1)
        it["speed"] = self._shape_slider(r4, "Speed", 0, 6, d.get("speed", 2.0), 0.1)
        it["amount"] = self._shape_slider(r4, "Amount", 0, 2, d.get("amount", 1.0), 0.05)

        self.shape_items.append(it)
        self.shape_sel.set(sid)        # newly added = selected for placement
        self._renumber_shapes()
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())
        self._push_shapes()
        return it

    # ---- Shape FX panel: edit the effect a selected shape shows -----------
    def _build_shapefx_panel(self, parent):
        head = tk.Frame(parent, padx=8, pady=4); head.pack(side="top", fill="x")
        self.shapefx_label = tk.Label(head, text="", font=("", 9, "bold"),
                                      wraplength=360, justify="left")
        self.shapefx_label.pack(anchor="w")
        tk.Label(head, text="These are the full knobs (look, grain, colors, audio "
                 "drive) for the effect this shape shows — just like a normal effect.",
                 fg="#888", wraplength=360, justify="left").pack(anchor="w")
        self.shapefx_host = tk.Frame(parent)
        self.shapefx_host.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        self._refresh_shape_params()

    def _open_shapefx(self):
        p = self._panel("shapefx")
        self._set_visible(p, True)
        if p["floating"] and p["toplevel"] is not None:
            try:
                p["toplevel"].deiconify(); p["toplevel"].lift()
            except tk.TclError:
                pass
        self._refresh_shape_params()

    def _refresh_shape_params(self, _retries=8):
        """Populate the Shape FX panel for the selected shape. ALWAYS shows the
        shape's own controls; if it also shows a secondary effect, that effect's
        full knobs are shown below."""
        host = getattr(self, "shapefx_host", None)
        if host is None:
            return
        for c in host.winfo_children():
            c.destroy()
        it = self._selected_shape()
        if it is None:
            self.shapefx_label.config(text="Shape FX")
            tk.Label(host, text="Select a shape (◉) in the Shapes panel to edit it "
                     "here.", fg="#888", wraplength=360, justify="left",
                     padx=8, pady=8).pack(anchor="w")
            return

        n = self.shape_items.index(it) + 1 if it in self.shape_items else "?"
        self.shapefx_label.config(
            text=f"Shape {n}: {it['shape'].get()} · {it['mode'].get()}")

        # one scroll for the WHOLE panel, so nothing hides below the fold
        canvas = tk.Canvas(host, highlightthickness=0)
        bar = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self._wheel_scroll(canvas)

        shows_effect = (shapes.is_window(it["mode"].get())
                        and it["effect"].get() != shapes.PRIMARY)

        # the EFFECT's knobs FIRST (that's the point of this panel)
        if shows_effect:
            d = self.engine.shape_fx_for(it["id"])
            if d is None:                        # effect still spinning up
                if _retries > 0:
                    self.root.after(120, lambda: self._refresh_shape_params(_retries - 1))
                tk.Label(inner, text="Starting this shape's effect…", fg="#888",
                         padx=8, pady=6).pack(anchor="w")
            else:
                efr = tk.LabelFrame(inner, text=f"Effect: {d['name']}  (grain, "
                                    "speed, colors, look, audio)", padx=4, pady=2)
                efr.pack(side="top", fill="x", padx=4, pady=(2, 4))
                self._params_grid(efr, d["params"], d["store"], scroll=False)
        elif shapes.is_window(it["mode"].get()):
            tk.Label(inner, text="Tip: set 'Shows' to an effect to give this shape "
                     "its own full effect knobs (grain, speed, colors, look, audio).",
                     fg="#06c", wraplength=360, justify="left", padx=8,
                     pady=4).pack(anchor="w")

        # the shape's own geometry/appearance (below the effect knobs)
        props = tk.LabelFrame(inner, text="This shape (position, size, …)",
                              padx=6, pady=4)
        props.pack(side="top", fill="x", padx=4, pady=(2, 4))
        self._build_shape_props(props, it)

        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _build_shape_props(self, parent, it):
        """Full-size editor for a shape's own knobs (shares the Tk vars with the
        compact Shapes-panel row, so both views stay in sync)."""
        def combo(label, var, values, refresh=False):
            r = tk.Frame(parent); r.pack(fill="x", pady=1)
            tk.Label(r, text=label, width=8, anchor="w").pack(side="left")
            cb = ttk.Combobox(r, textvariable=var, state="readonly", values=values)
            cb.pack(side="left", fill="x", expand=True)
            cb.bind("<<ComboboxSelected>>", lambda e: (
                self._push_shapes(),
                self._refresh_shape_params() if refresh else None))

        combo("Shape", it["shape"], shapes.SHAPE_CHOICES)
        combo("Mode", it["mode"], shapes.MODE_CHOICES, refresh=True)
        combo("Shows", it["effect"], [shapes.PRIMARY] + [c.name for c in self.effect_classes],
              refresh=True)
        self._prop_slider(parent, "X", it["x"], 0.0, 1.0, 0.01)
        self._prop_slider(parent, "Y", it["y"], 0.0, 1.0, 0.01)
        self._prop_slider(parent, "Size", it["size"], 0.02, 0.6, 0.01)
        self._prop_slider(parent, "Rotate", it["rotation"], 0.0, 1.0, 0.01)
        self._prop_slider(parent, "Color", it["hue"], 0.0, 1.0, 0.01)
        combo("Reacts", it["react"], shapes.REACT_CHOICES)
        self._prop_slider(parent, "Strength", it["strength"], 0.0, 4.0, 0.1)
        self._prop_slider(parent, "Speed", it["speed"], 0.0, 6.0, 0.1)
        self._prop_slider(parent, "Amount", it["amount"], 0.0, 2.0, 0.05)

    def _prop_slider(self, parent, label, var, lo, hi, res):
        row = tk.Frame(parent); row.pack(fill="x", pady=1)
        tk.Label(row, text=label, width=8, anchor="w").pack(side="left")
        tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", showvalue=True,
                 command=lambda v: self._push_shapes_debounced()).pack(
                     side="left", fill="x", expand=True)

    def _shape_slider(self, parent, label, lo, hi, default, res):
        tk.Label(parent, text=label).pack(side="left")
        var = tk.DoubleVar(value=default)
        tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", length=58, showvalue=False,
                 command=lambda v: self._push_shapes_debounced()).pack(
                     side="left", fill="x", expand=True, padx=(2, 5))
        return var

    def _remove_shape(self, it):
        try:
            it["frame"].destroy()
        except tk.TclError:
            pass
        if it in self.shape_items:
            self.shape_items.remove(it)
        self._renumber_shapes()
        self._push_shapes()

    def _clear_shapes(self):
        for it in list(self.shape_items):
            try:
                it["frame"].destroy()
            except tk.TclError:
                pass
        self.shape_items = []
        self._push_shapes()

    def _renumber_shapes(self):
        for i, it in enumerate(self.shape_items, 1):
            try:
                it["title"].config(text=f" Shape {i}")
            except tk.TclError:
                pass

    def _collect_shape_dicts(self):
        return [{"id": it["id"], "shape": it["shape"].get(), "mode": it["mode"].get(),
                 "react": it["react"].get(), "effect": it["effect"].get(),
                 "x": it["x"].get(), "y": it["y"].get(),
                 "size": it["size"].get(), "rotation": it["rotation"].get(),
                 "hue": it["hue"].get(), "strength": it["strength"].get(),
                 "speed": it["speed"].get(), "amount": it["amount"].get()}
                for it in self.shape_items]

    def _push_shapes(self):
        """Send the current shape list to the engine (stored + applied live)."""
        dicts = self._collect_shape_dicts()
        self.engine.request_shapes(dicts)
        if hasattr(self, "shapes_status"):
            n = len(self.shape_items)
            # warn if more distinct secondary effects are requested than slots
            distinct = {d["effect"] for d in dicts
                        if shapes.is_window(d["mode"]) and d["effect"] != shapes.PRIMARY}
            if len(distinct) > shapes.MAX_FX_SLOTS:
                self.shapes_status.config(
                    text=f"⚠ Up to {shapes.MAX_FX_SLOTS} different effects at once — "
                         f"extra ones fall back to Primary.", fg="#a60")
            else:
                self.shapes_status.config(
                    text=f"{n} shape{'' if n == 1 else 's'}. Tip: select one and click "
                         "the preview to place it.", fg="#888")

    def _push_shapes_debounced(self):
        prev = getattr(self, "_shapes_after", None)
        if prev:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass
        self._shapes_after = self.root.after(60, self._push_shapes)

    def _selected_shape(self):
        sid = self.shape_sel.get() if getattr(self, "shape_sel", None) else ""
        return next((it for it in getattr(self, "shape_items", []) if it["id"] == sid), None)

    # ---- Layers: effects stacked & blended over the main effect ----------
    def _open_layers(self, *_a, **_k):
        """Show the Layers panel; add a starter layer if empty."""
        p = self._panel("layers")
        self._set_visible(p, True)
        if p["floating"] and p["toplevel"] is not None:
            try:
                p["toplevel"].deiconify(); p["toplevel"].lift()
            except tk.TclError:
                pass
        if not getattr(self, "layer_items", None):
            self._add_layer()

    def _build_layers_panel(self, parent):
        self.layer_items = []
        self._layer_id = 0
        self.layer_sel = tk.StringVar(value="")

        head = tk.Frame(parent, padx=8, pady=4); head.pack(side="top", fill="x")
        tk.Label(head, text="Stack extra effects ON TOP of the main effect. Each "
                 "layer is a full effect with its own knobs — pick its blend mode "
                 "and opacity, and ⚙ to edit it. The bottom row draws on top.",
                 wraplength=360, justify="left", font=("", 9, "bold")).pack(anchor="w")

        self.layers_status = tk.Label(parent, text="", fg="#888",
                                      wraplength=360, justify="left")
        self.layers_status.pack(side="bottom", anchor="w", padx=8, pady=(0, 4))
        btns = tk.Frame(parent, padx=8, pady=4); btns.pack(side="bottom", fill="x")
        tk.Button(btns, text="➕ Add layer", command=lambda: self._add_layer()).pack(side="left")
        tk.Button(btns, text="🗑 Clear all", command=self._clear_layers).pack(side="left", padx=6)
        tk.Button(btns, text="⚙ Layer FX", command=self._open_layerfx).pack(side="left")

        listwrap = tk.Frame(parent); listwrap.pack(side="top", fill="both",
                                                   expand=True, padx=4)
        canvas = tk.Canvas(listwrap, highlightthickness=0)
        sb = tk.Scrollbar(listwrap, orient="vertical", command=canvas.yview)
        self.layers_host = tk.Frame(canvas)
        win = canvas.create_window((0, 0), window=self.layers_host, anchor="nw")
        self.layers_host.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._wheel_scroll(canvas)

    def _add_layer(self, id=None, **d):
        # reuse a saved id (so its effect-knob cache matches) or mint a new one
        if id:
            lid = id
            try:
                self._layer_id = max(self._layer_id, int(str(id).lstrip("L")))
            except ValueError:
                pass
        else:
            self._layer_id += 1
            lid = f"L{self._layer_id}"
        eff_vals = [c.name for c in self.effect_classes]
        # default to a concrete visual effect, not Blank or an empty Custom one
        default_eff = (next((n for n in ("Plasma", "Liquid Fractal") if n in eff_vals), None)
                       or next((n for n in eff_vals if not n.startswith(("Blank", "Custom"))),
                               eff_vals[0] if eff_vals else ""))
        box = tk.LabelFrame(self.layers_host, padx=6, pady=3)
        box.pack(fill="x", padx=4, pady=2)
        it = {"frame": box, "id": lid}

        top = tk.Frame(box); top.pack(fill="x")
        it["title"] = tk.Label(top, text="Layer")
        it["title"].pack(side="left")
        it["visible"] = tk.BooleanVar(value=bool(d.get("visible", True)))
        tk.Checkbutton(top, text="on", variable=it["visible"],
                       command=self._push_layers).pack(side="left", padx=2)
        tk.Button(top, text="✕", width=2, fg="#a00",
                  command=lambda: self._remove_layer(it)).pack(side="right")
        tk.Button(top, text="⚙", width=2,
                  command=lambda: (self.layer_sel.set(lid), self._open_layerfx())
                  ).pack(side="right", padx=2)

        r1 = tk.Frame(box); r1.pack(fill="x")
        tk.Label(r1, text="Effect").pack(side="left")
        it["effect"] = tk.StringVar(value=d.get("effect", default_eff))
        ce = ttk.Combobox(r1, textvariable=it["effect"], state="readonly", values=eff_vals)
        ce.pack(side="left", fill="x", expand=True, padx=3)
        ce.bind("<<ComboboxSelected>>",
                lambda e: (self._push_layers(), self._refresh_layer_params()))

        r2 = tk.Frame(box); r2.pack(fill="x")
        tk.Label(r2, text="Blend").pack(side="left")
        it["blend"] = tk.StringVar(value=d.get("blend", "Normal"))
        cb = ttk.Combobox(r2, textvariable=it["blend"], width=11, state="readonly",
                          values=LAYER_BLENDS)
        cb.pack(side="left", padx=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._push_layers())
        tk.Label(r2, text="Opacity").pack(side="left", padx=(8, 0))
        it["opacity"] = tk.DoubleVar(value=float(d.get("opacity", 1.0)))
        tk.Scale(r2, variable=it["opacity"], from_=0.0, to=1.0, resolution=0.01,
                 orient="horizontal", showvalue=False,
                 command=lambda v: self._push_layers_debounced()).pack(
                     side="left", fill="x", expand=True, padx=(2, 4))

        self.layer_items.append(it)
        self.layer_sel.set(lid)
        self._renumber_layers()
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())
        self._push_layers()
        return it

    def _remove_layer(self, it):
        try:
            it["frame"].destroy()
        except tk.TclError:
            pass
        if it in self.layer_items:
            self.layer_items.remove(it)
        self._renumber_layers()
        self._push_layers()
        self._refresh_layer_params()

    def _clear_layers(self):
        for it in list(getattr(self, "layer_items", [])):
            try:
                it["frame"].destroy()
            except tk.TclError:
                pass
        self.layer_items = []
        self._push_layers()
        self._refresh_layer_params()

    def _renumber_layers(self):
        for i, it in enumerate(self.layer_items, 1):
            try:
                it["title"].config(text=f" Layer {i}")
            except tk.TclError:
                pass

    def _collect_layer_dicts(self):
        return [{"id": it["id"], "effect": it["effect"].get(),
                 "blend": it["blend"].get(), "opacity": float(it["opacity"].get()),
                 "visible": bool(it["visible"].get())}
                for it in self.layer_items]

    def _push_layers(self):
        """Send the current layer stack to the engine (stored + applied live)."""
        dicts = self._collect_layer_dicts()
        self.engine.request_layers(dicts)
        if hasattr(self, "layers_status"):
            n = len(self.layer_items)
            extra = max(0, n - MAX_LAYERS)
            if extra:
                self.layers_status.config(
                    text=f"⚠ Up to {MAX_LAYERS} layers at once — {extra} extra ignored.",
                    fg="#a60")
            else:
                self.layers_status.config(
                    text=f"{n} layer{'' if n == 1 else 's'} over the main effect.",
                    fg="#888")

    def _push_layers_debounced(self):
        prev = getattr(self, "_layers_after", None)
        if prev:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass
        self._layers_after = self.root.after(60, self._push_layers)

    def _selected_layer(self):
        lid = self.layer_sel.get() if getattr(self, "layer_sel", None) else ""
        return next((it for it in getattr(self, "layer_items", []) if it["id"] == lid), None)

    def _load_layers(self, layer_dicts):
        for it in list(getattr(self, "layer_items", [])):
            try:
                it["frame"].destroy()
            except tk.TclError:
                pass
        self.layer_items = []
        for d in layer_dicts:
            self._add_layer(id=d.get("id"),
                            **{k: d[k] for k in ("effect", "blend", "opacity", "visible")
                               if k in d})
        self._push_layers()
        self._refresh_layer_params()

    # ---- Layer FX editor: full knobs for a selected layer's effect -------
    def _build_layerfx_panel(self, parent):
        head = tk.Frame(parent, padx=8, pady=4); head.pack(side="top", fill="x")
        self.layerfx_label = tk.Label(head, text="", font=("", 9, "bold"),
                                      wraplength=360, justify="left")
        self.layerfx_label.pack(anchor="w")
        tk.Label(head, text="The full knobs (look, grain, colors, audio/LFO drive) "
                 "for this layer's effect — exactly like a normal effect.",
                 fg="#888", wraplength=360, justify="left").pack(anchor="w")
        self.layerfx_host = tk.Frame(parent)
        self.layerfx_host.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        self._refresh_layer_params()

    def _open_layerfx(self):
        p = self._panel("layerfx")
        self._set_visible(p, True)
        if p["floating"] and p["toplevel"] is not None:
            try:
                p["toplevel"].deiconify(); p["toplevel"].lift()
            except tk.TclError:
                pass
        self._refresh_layer_params()

    def _refresh_layer_params(self, _retries=8):
        """Populate the Layer FX panel with the selected layer's effect knobs."""
        host = getattr(self, "layerfx_host", None)
        if host is None:
            return
        for c in host.winfo_children():
            c.destroy()
        it = self._selected_layer()
        if it is None:
            self.layerfx_label.config(text="Layer FX")
            tk.Label(host, text="Add a layer (🧱 Layers) and click its ⚙ to edit "
                     "its effect here.", fg="#888", wraplength=360, justify="left",
                     padx=8, pady=8).pack(anchor="w")
            return
        n = self.layer_items.index(it) + 1 if it in self.layer_items else "?"
        self.layerfx_label.config(text=f"Layer {n}: {it['effect'].get()}")
        d = self.engine.layer_fx_for(it["id"])
        if d is None:                              # effect still spinning up
            if _retries > 0:
                self.root.after(120, lambda: self._refresh_layer_params(_retries - 1))
            tk.Label(host, text="Starting this layer's effect…", fg="#888",
                     padx=8, pady=6).pack(anchor="w")
            return
        self._params_grid(host, d["params"], d["store"], scroll=True)
        if getattr(self, "_theme", None):
            self.apply_theme(self.theme_var.get())

    def _update_pointer(self, event, down=None):
        """Feed the mouse over the preview into the engine pointer (normalized,
        y-up to match the canvas), so interactive effects can read it."""
        rect = getattr(self, "_img_rect", None)
        pt = getattr(self.engine, "pointer", None)
        if not rect or pt is None:
            return
        offx, offy, dw, dh = rect
        pt.x = min(1.0, max(0.0, (event.x - offx) / dw))
        pt.y = min(1.0, max(0.0, 1.0 - (event.y - offy) / dh))
        pt.active = True
        if down is not None:
            pt.down = down

    def _pointer_leave(self):
        pt = getattr(self.engine, "pointer", None)
        if pt is not None:
            pt.active = False
            pt.down = False

    def _place_selected(self, event):
        """Click/drag on the preview -> move the selected shape there. Acts only
        when a shape is selected in the Shapes panel (otherwise clicks do
        nothing), so it works over whatever effect is running."""
        it = self._selected_shape()
        if it is None:
            return
        rect = getattr(self, "_img_rect", None)
        if not rect:
            return
        offx, offy, dw, dh = rect             # the on-screen image rectangle
        sx = (event.x - offx) / dw
        sy = 1.0 - (event.y - offy) / dh      # display is flipped vertically
        it["x"].set(round(min(1.0, max(0.0, sx)), 3))
        it["y"].set(round(min(1.0, max(0.0, sy)), 3))
        self._push_shapes()

    # --- Build tab (the easiest, no-typing one) ---
    def _build_build_tab(self, nb):
        tab = tk.Frame(nb); nb.add(tab, text="🧩 Build (easiest)")
        tk.Label(tab, text="Stack as many pattern blocks as you like. No typing — "
                 "pick a shape and slide.", font=("", 10, "bold"),
                 wraplength=560, justify="left").pack(anchor="w", padx=8, pady=(8, 4))

        # --- scrollable list of blocks (grows without limit) ---
        listwrap = tk.Frame(tab)
        listwrap.pack(fill="both", expand=True, padx=4)
        canvas = tk.Canvas(listwrap, highlightthickness=0, height=320)
        sb = tk.Scrollbar(listwrap, orient="vertical", command=canvas.yview)
        self.blocks_host = tk.Frame(canvas)
        win = canvas.create_window((0, 0), window=self.blocks_host, anchor="nw")
        self.blocks_host.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._wheel_scroll(canvas)

        self.build_rows = []
        tk.Button(tab, text="➕ Add block", command=lambda: (self._add_block_row(), self._apply_build())
                  ).pack(anchor="w", padx=8, pady=(2, 0))

        # --- mix & color ---
        opt = tk.LabelFrame(tab, text="Mix & color", padx=6, pady=4)
        opt.pack(fill="x", padx=8, pady=6)
        r1 = tk.Frame(opt); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Blend:").pack(side="left")
        self.build_combine = tk.StringVar(value=patterns.COMBINE_CHOICES[0])
        c1 = ttk.Combobox(r1, textvariable=self.build_combine, width=14,
                          state="readonly", values=patterns.COMBINE_CHOICES)
        c1.pack(side="left", padx=4); c1.bind("<<ComboboxSelected>>", lambda e: self._apply_build())
        tk.Label(r1, text="  Color motion:").pack(side="left")
        self.build_hue = tk.StringVar(value=patterns.HUE_CHOICES[0])
        c2 = ttk.Combobox(r1, textvariable=self.build_hue, width=15,
                          state="readonly", values=patterns.HUE_CHOICES)
        c2.pack(side="left", padx=4); c2.bind("<<ComboboxSelected>>", lambda e: self._apply_build())

        r2 = tk.Frame(opt); r2.pack(fill="x", pady=2)
        tk.Label(r2, text="Colors:").pack(side="left")
        self.build_palette = tk.StringVar(value="plasma")
        c3 = ttk.Combobox(r2, textvariable=self.build_palette, width=12,
                          state="readonly", values=ColorPalette.NAMED)
        c3.pack(side="left", padx=4); c3.bind("<<ComboboxSelected>>", lambda e: self._apply_build())

        r3 = tk.Frame(opt); r3.pack(fill="x", pady=2)
        tk.Label(r3, text="Output:").pack(side="left")
        self.build_output = tk.StringVar(value=self.BUILD_OUTPUTS[0])
        c4 = ttk.Combobox(r3, textvariable=self.build_output, width=22,
                          state="readonly", values=self.BUILD_OUTPUTS)
        c4.pack(side="left", padx=4); c4.bind("<<ComboboxSelected>>", lambda e: self._apply_build())
        self.build_motion = tk.BooleanVar(value=False)
        tk.Checkbutton(r3, text="Glow where I move (camera)", variable=self.build_motion,
                       command=self._apply_build).pack(side="left", padx=6)

        br = tk.Frame(tab); br.pack(fill="x", padx=8, pady=6)
        tk.Button(br, text="▶ Apply (live)", command=self._apply_build).pack(side="left")
        tk.Label(br, text="  Name:").pack(side="left")
        self.build_name = tk.Entry(br, width=18)
        self.build_name.insert(0, "My Block Effect")
        self.build_name.pack(side="left")
        tk.Button(br, text="💾 Save as effect", command=self._save_build).pack(side="left", padx=6)

        self.build_status = tk.Label(tab, text="Tip: add Circles+Bass, then a Spiral+Beat.",
                                     fg="#070", wraplength=560, justify="left")
        self.build_status.pack(anchor="w", padx=8)

        # start with one nice reactive block
        self._add_block_row(pattern="Circles (rings)", react="Bass")

    def _add_block_row(self, pattern="Off", react="Nothing"):
        box = tk.LabelFrame(self.blocks_host, padx=6, pady=3)
        box.pack(fill="x", padx=4, pady=2)
        row = {"frame": box}

        top = tk.Frame(box); top.pack(fill="x")
        row["title"] = tk.Label(top, text="Block")
        row["title"].pack(side="left")
        tk.Label(top, text=" Shape:").pack(side="left")
        row["pattern"] = tk.StringVar(value=pattern)
        cb = ttk.Combobox(top, textvariable=row["pattern"], width=15, state="readonly",
                          values=patterns.PATTERN_CHOICES)
        cb.pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: self._apply_build())
        tk.Button(top, text="✕", width=2, fg="#a00",
                  command=lambda: self._remove_block(row)).pack(side="right")

        mid = tk.Frame(box); mid.pack(fill="x")
        row["size"] = self._mini_slider(mid, "Size", 1, 30, 8, 1)
        row["speed"] = self._mini_slider(mid, "Speed", 0, 6, 2.0, 0.1)
        row["reverse"] = tk.BooleanVar(value=False)
        tk.Checkbutton(mid, text="Reverse", variable=row["reverse"],
                       command=self._apply_build).pack(side="left", padx=4)

        bot = tk.Frame(box); bot.pack(fill="x")
        tk.Label(bot, text="Reacts to:").pack(side="left")
        row["react"] = tk.StringVar(value=react)
        cbr = ttk.Combobox(bot, textvariable=row["react"], width=9, state="readonly",
                           values=patterns.REACT_CHOICES)
        cbr.pack(side="left", padx=4); cbr.bind("<<ComboboxSelected>>", lambda e: self._apply_build())
        row["react_strength"] = self._mini_slider(bot, "Strength", 0, 4, 1.5, 0.1)
        row["amount"] = self._mini_slider(bot, "Amount", 0, 2, 1.0, 0.05)

        self.build_rows.append(row)
        self._renumber_blocks()
        if getattr(self, "_theme", None):   # theme the new widgets (post-startup)
            self.apply_theme(self.theme_var.get())
        return row

    def _mini_slider(self, parent, label, lo, hi, default, res, length=110):
        tk.Label(parent, text=label).pack(side="left")
        var = tk.DoubleVar(value=default)
        tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", length=length, showvalue=False,
                 command=lambda v: self._apply_build_debounced()).pack(side="left", padx=(2, 6))
        return var

    def _remove_block(self, row):
        try:
            row["frame"].destroy()
        except tk.TclError:
            pass
        if row in self.build_rows:
            self.build_rows.remove(row)
        self._renumber_blocks()
        self._apply_build()

    def _renumber_blocks(self):
        for i, r in enumerate(self.build_rows, 1):
            try:
                r["title"].config(text=f"Block {i}")
            except tk.TclError:
                pass

    # Output label -> ExpressionEffect SOURCE mode
    BUILD_OUTPUTS = ["Paint colors", "Texture the media", "Warp the media"]
    _OUTPUT_SRC = {"Paint colors": "paint", "Texture the media": "texture",
                   "Warp the media": "warp"}

    def _collect_build(self):
        layers = [{"pattern": r["pattern"].get(), "size": r["size"].get(),
                   "speed": r["speed"].get(), "reverse": r["reverse"].get(),
                   "react": r["react"].get(),
                   "react_strength": r["react_strength"].get(),
                   "amount": r["amount"].get()}
                  for r in self.build_rows]
        bright, hue = patterns.build_formulas(layers, self.build_combine.get(),
                                              self.build_hue.get())
        src = self._OUTPUT_SRC.get(self.build_output.get(), "paint")
        if src == "texture":               # show the media, modulated by the blocks
            bright = f"0.45 + 0.6*({bright})"
        if src in ("paint", "texture") and self.build_motion.get():
            bright = f"({bright}) + motion*1.5"   # interactive camera glow
        if src == "warp":                  # blocks displace the media
            hue = bright
        return bright, hue, src

    def _apply_build(self):
        bright, hue, src = self._collect_build()
        if self.engine.effect is None or self.engine.effect.name != EXPR_EFFECT_NAME:
            self._switch_to(EXPR_EFFECT_NAME)
        # colors: set the expression effect's palette knob (once it's active)
        self.engine.store.set("palette", {"named": self.build_palette.get(), "custom": []})
        self.engine.apply_expression(bright, hue, src)
        hint = {"texture": "Showing your media, shaped by the blocks.",
                "warp": "Warping your media with the blocks.",
                "paint": "Live — tweak blocks and watch the preview."}[src]
        if src != "paint" and not (self.engine.media and self.engine.media._mode != "off"):
            hint = "Pick a Media source (camera/image/video) on the Controls tab to see this."
        self.build_status.config(text=hint, fg="#070")

    def _apply_build_debounced(self):
        # sliders fire rapidly; wait for a short idle so we recompile once, not 60x/sec
        prev = getattr(self, "_build_after", None)
        if prev:
            try:
                self.root.after_cancel(prev)
            except Exception:
                pass
        self._build_after = self.root.after(150, self._apply_build)

    def _save_build(self):
        bright, hue, src = self._collect_build()
        name = self.build_name.get().strip() or "My Block Effect"
        self._write_effect(name, expression_file(name, bright, hue,
                                                 self.build_palette.get(), src),
                           self.build_status)

    # --- Expression tab ---
    def _build_expr_tab(self, nb):
        tab = tk.Frame(nb); nb.add(tab, text="Expression (no code)")

        tk.Label(tab, text="Make an effect from two math formulas. No coding.",
                 font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        defaults_b = ExpressionEffectBase.BRIGHT
        defaults_h = ExpressionEffectBase.HUE

        tk.Label(tab, text="Brightness  =").pack(anchor="w", padx=8)
        self.expr_bright = tk.Entry(tab, width=70)
        self.expr_bright.insert(0, defaults_b)
        self.expr_bright.pack(fill="x", padx=8)

        tk.Label(tab, text="Color (hue)  =").pack(anchor="w", padx=8, pady=(6, 0))
        self.expr_hue = tk.Entry(tab, width=70)
        self.expr_hue.insert(0, defaults_h)
        self.expr_hue.pack(fill="x", padx=8)

        orow = tk.Frame(tab); orow.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(orow, text="Output:").pack(side="left")
        self.expr_output = tk.StringVar(value=self.BUILD_OUTPUTS[0])
        ocb = ttk.Combobox(orow, textvariable=self.expr_output, width=22,
                           state="readonly", values=self.BUILD_OUTPUTS)
        ocb.pack(side="left", padx=4); ocb.bind("<<ComboboxSelected>>", lambda e: self._apply_expr())
        tk.Label(orow, text="(Texture/Warp use camera/image/video; sample with "
                 "tex, texr/g/b, motion)", fg="#888").pack(side="left", padx=4)

        row = tk.Frame(tab); row.pack(fill="x", padx=8, pady=6)
        tk.Button(row, text="▶ Apply (live preview)", command=self._apply_expr).pack(side="left")
        tk.Label(row, text="   Name:").pack(side="left")
        self.expr_name = tk.Entry(row, width=22)
        self.expr_name.insert(0, "My Formula Effect")
        self.expr_name.pack(side="left")
        tk.Button(row, text="💾 Save as effect", command=self._save_expr).pack(side="left", padx=6)

        self.expr_status = tk.Label(tab, text="", fg="#070", wraplength=580, justify="left")
        self.expr_status.pack(anchor="w", padx=8)

        cheat = tk.Label(tab, text=cheat_sheet(), justify="left", fg="#333",
                         font=("Consolas", 9), bg="#f4f4f4", relief="groove")
        cheat.pack(fill="both", expand=True, padx=8, pady=8)

    def _apply_expr(self):
        b = self.expr_bright.get()
        h = self.expr_hue.get()
        try:                       # validate up front for a friendly message
            translate(b, ExpressionEffectBase.VARS)
            translate(h, ExpressionEffectBase.VARS)
        except ValueError as e:
            self.expr_status.config(text=f"⚠ {e}", fg="#a00")
            return
        if self.engine.effect is None or self.engine.effect.name != EXPR_EFFECT_NAME:
            self._switch_to(EXPR_EFFECT_NAME)
        src = self._OUTPUT_SRC.get(self.expr_output.get(), "paint")
        self.engine.apply_expression(b, h, src)
        self.expr_status.config(text="Applied - watch the main window.", fg="#070")

    def _save_expr(self):
        b, h = self.expr_bright.get(), self.expr_hue.get()
        try:
            translate(b, ExpressionEffectBase.VARS)
            translate(h, ExpressionEffectBase.VARS)
        except ValueError as e:
            self.expr_status.config(text=f"⚠ {e}", fg="#a00")
            return
        name = self.expr_name.get().strip() or "My Formula Effect"
        src = self._OUTPUT_SRC.get(self.expr_output.get(), "paint")
        self._write_effect(name, expression_file(name, b, h, "plasma", src),
                           self.expr_status)

    # --- Code tab ---
    def _build_code_tab(self, nb):
        tab = tk.Frame(nb); nb.add(tab, text="Code (full control)")

        tk.Label(tab, text="A complete working effect. Edit the math, Reload to preview.",
                 font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        ef = tk.Frame(tab); ef.pack(fill="both", expand=True, padx=8)
        self.code_text = tk.Text(ef, wrap="none", font=("Consolas", 10),
                                 undo=True, height=24)
        ys = tk.Scrollbar(ef, command=self.code_text.yview)
        self.code_text.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.code_text.pack(side="left", fill="both", expand=True)
        self.code_text.insert("1.0", CODE_TEMPLATE)

        row = tk.Frame(tab); row.pack(fill="x", padx=8, pady=6)
        tk.Button(row, text="▶ Reload (live preview)", command=self._reload_code).pack(side="left")
        tk.Button(row, text="💾 Save as effect", command=self._save_code).pack(side="left", padx=6)

        self.code_status = tk.Label(tab, text="", fg="#070", wraplength=580, justify="left")
        self.code_status.pack(anchor="w", padx=8)

    def _compile_user_code(self, src, status):
        """exec user code, return an Effect subclass or None (and show errors)."""
        # Seed __file__/__name__ so the template's path-bootstrap line (which
        # references __file__) works during in-memory live reload, not just when
        # the file is later saved and run for real.
        base = self.effects_dir or os.getcwd()
        ns = {"__file__": os.path.join(base, "_live_effect.py"),
              "__name__": "vizstudio_live_effect"}
        try:
            # exec_with_source keeps the code inspectable so @ti.kernel works
            exec_with_source(src, ns, tag="vizstudio-usercode")
        except Exception as e:
            status.config(text=f"⚠ {type(e).__name__}: {e}", fg="#a00")
            return None
        cls = next((v for v in ns.values()
                    if isinstance(v, type) and issubclass(v, Effect) and v is not Effect), None)
        if cls is None:
            status.config(text="⚠ No effect found. Keep the 'class ...(Effect):' line.", fg="#a00")
        return cls

    def _reload_code(self):
        cls = self._compile_user_code(self.code_text.get("1.0", "end"), self.code_status)
        if cls is None:
            return
        try:
            inst = cls()
        except Exception as e:
            self.code_status.config(text=f"⚠ {type(e).__name__}: {e}", fg="#a00")
            return
        self.engine.request_effect(inst)   # live preview (errors surface via pump)
        self.code_status.config(text=f"Loaded '{cls.name}' - watch the main window.", fg="#070")

    def _save_code(self):
        src = self.code_text.get("1.0", "end")
        cls = self._compile_user_code(src, self.code_status)
        if cls is None:
            return
        self._write_effect(cls.name, src, self.code_status)

    # --- shared: write a file into effects/ and refresh the dropdown ---
    def _write_effect(self, name, source, status):
        if not self.effects_dir:
            status.config(text="⚠ effects folder unknown; cannot save.", fg="#a00")
            return
        slug = re.sub(r"\W+", "_", name).strip("_").lower() or "my_effect"
        path = os.path.join(self.effects_dir, slug + ".py")
        if os.path.exists(path):
            from tkinter import messagebox
            if not messagebox.askyesno("Overwrite?", f"{slug}.py already exists. Replace it?"):
                return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(source)
        except Exception as e:
            status.config(text=f"⚠ could not save: {e}", fg="#a00")
            return
        n = self._refresh_effects()
        status.config(text=f"✓ Saved to effects/{slug}.py  ({n} effects available). "
                           f"Pick '{name}' in the Effect dropdown.", fg="#070")

    def _refresh_effects(self):
        """Re-scan effects/ and update the main dropdown. Returns the count."""
        fx, errs = discover(self.effects_dir)
        if fx:
            self.effect_classes = fx
            if hasattr(self, "effect_menu"):
                self.effect_menu["values"] = [c.name for c in fx]
        return len(fx)

    # ---- the auto-generated knobs --------------------------------------
    def _build_params(self):
        self._params_grid(self.param_host, self.engine.params, self.engine.store)
        self._effect_version = self.engine.effect_version

    def _params_grid(self, host, params, store, scroll=True):
        """Build knob widgets for `params`, bound to `store`. With scroll=True
        wraps them in their own scroll canvas (main Parameters panel); with
        scroll=False builds them straight into `host` (the caller scrolls)."""
        for child in host.winfo_children():
            child.destroy()
        if scroll:
            canvas = tk.Canvas(host, highlightthickness=0)
            bar = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
            inner = tk.Frame(canvas)
            inner.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=bar.set)
            canvas.pack(side="left", fill="both", expand=True)
            bar.pack(side="right", fill="y")
            self._wheel_scroll(canvas)
        else:
            inner = host
        for p in params:
            self._build_one(inner, p, store)

    def _build_one(self, parent, p, store):
        if isinstance(p, Toggle):
            var = tk.BooleanVar(value=store.values.get(p.name, p.default))
            cb = tk.Checkbutton(parent, text=p.label, variable=var,
                                command=lambda: store.set(p.name, var.get()))
            cb.pack(anchor="w", pady=2)
            self._tip(cb, p.help)
            return

        if isinstance(p, Choice):
            tk.Label(parent, text=p.label).pack(anchor="w")
            var = tk.StringVar(value=store.values.get(p.name, p.default))
            cb = ttk.Combobox(parent, textvariable=var, values=p.options, state="readonly")
            cb.pack(fill="x")
            cb.bind("<<ComboboxSelected>>", lambda e: store.set(p.name, var.get()))
            self._tip(cb, p.help)
            return

        if isinstance(p, ColorPalette):
            self._build_palette(parent, p, store)
            return

        if isinstance(p, Slider):  # includes IntSlider
            self._build_slider(parent, p, store)

    def _build_slider(self, parent, p, store):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=(4, 0))
        lbl = tk.Label(frame, text=p.label); lbl.pack(side="left")
        res = 1 if isinstance(p, IntSlider) else max((p.hi - p.lo) / 200.0, 1e-5)
        var = tk.DoubleVar(value=store.values.get(p.name, p.default))
        scale = tk.Scale(parent, variable=var, from_=p.lo, to=p.hi, resolution=res,
                         orient="horizontal", length=400, showvalue=True,
                         command=lambda v: store.set(p.name, float(v)))
        scale.pack(fill="x")
        self._tip(lbl, p.help)
        self._tip(scale, p.help)

        if not p.audio:
            return
        # audio-drive row: source dropdown + amount
        drive = tk.Frame(parent); drive.pack(fill="x")
        tk.Label(drive, text="  drive:", fg="#446").pack(side="left")
        src = tk.StringVar(value=store.audio_src.get(p.name, "none"))
        menu = ttk.Combobox(drive, textvariable=src, width=7, state="readonly",
                            values=[_DRIVE_LABELS[s] for s in DRIVE_SOURCES])
        menu.set(_DRIVE_LABELS[src.get()])
        menu.pack(side="left")

        amt = tk.DoubleVar(value=store.audio_amt.get(p.name, 0.5))

        def on_src(_e=None):
            label = menu.get()
            chosen = next(s for s, l in _DRIVE_LABELS.items() if l == label)
            store.set_audio(p.name, source=chosen)
        menu.bind("<<ComboboxSelected>>", on_src)

        tk.Scale(drive, variable=amt, from_=0.0, to=1.0, resolution=0.01,
                 orient="horizontal", length=160, showvalue=False,
                 command=lambda v: store.set_audio(p.name, amount=float(v))
                 ).pack(side="left", padx=4)

    def _build_palette(self, parent, p, store):
        box = tk.LabelFrame(parent, text=p.label, padx=6, pady=4)
        box.pack(fill="x", pady=4)
        spec = store.values.get(p.name, {"named": p.default, "custom": []})
        named = tk.StringVar(value=spec.get("named", p.default))
        custom = list(spec.get("custom", []))

        def push():
            store.set(p.name, {"named": named.get(), "custom": list(custom)})

        cb = ttk.Combobox(box, textvariable=named, values=ColorPalette.NAMED, state="readonly")
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: push())

        swatch_row = tk.Frame(box); swatch_row.pack(fill="x", pady=4)

        def redraw_swatches():
            for c in swatch_row.winfo_children():
                c.destroy()
            for i, col in enumerate(custom):
                tk.Label(swatch_row, bg=col, width=3, relief="raised").pack(side="left", padx=1)
            tk.Button(swatch_row, text="+ color", command=add_color).pack(side="left", padx=4)
            if custom:
                tk.Button(swatch_row, text="clear", command=clear_colors).pack(side="left")

        def add_color():
            res = colorchooser.askcolor(title="Pick a color")
            if res and res[1]:
                custom.append(res[1])
                push(); redraw_swatches()

        def clear_colors():
            custom.clear()
            push(); redraw_swatches()

        redraw_swatches()
        tk.Label(box, text="Pick 2+ colors for a custom gradient (overrides the named one).",
                 fg="#888", wraplength=380, justify="left").pack(anchor="w")

    def _tip(self, widget, text):
        """Show `text` in a small popup after the pointer rests on `widget`."""
        if not text:
            return

        def schedule(_e=None):
            cancel()
            widget._tip_after = widget.after(500, show)

        def show():
            self._hide_tip()
            try:
                x = widget.winfo_rootx() + 14
                y = widget.winfo_rooty() + widget.winfo_height() + 4
            except tk.TclError:
                return
            th = getattr(self, "_theme", None) or {}
            tw = tk.Toplevel(self.root)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            try:
                tw.attributes("-topmost", True)
            except tk.TclError:
                pass
            tk.Label(tw, text=text, justify="left", wraplength=320,
                     bg=th.get("panel", "#222"), fg=th.get("fg", "#eee"),
                     relief="solid", bd=1, padx=6, pady=3, font=("", 8)).pack()
            self._tip_win = tw

        def cancel(_e=None):
            a = getattr(widget, "_tip_after", None)
            if a is not None:
                try:
                    widget.after_cancel(a)
                except tk.TclError:
                    pass
                widget._tip_after = None
            self._hide_tip()

        widget.bind("<Enter>", schedule, add="+")
        widget.bind("<Leave>", cancel, add="+")
        widget.bind("<Button>", cancel, add="+")

    def _hide_tip(self):
        w = getattr(self, "_tip_win", None)
        if w is not None:
            try:
                w.destroy()
            except tk.TclError:
                pass
            self._tip_win = None

