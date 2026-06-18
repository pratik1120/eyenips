# Eyenips

A beginner-friendly, audio-reactive visualization studio — think *WinAmp
visualizations meets TouchDesigner*. Pick an effect, turn knobs, choose
colors, flip on grain / fluid / flicker, and bind any knob to the music. When
you want more, drop a Python file in `effects/` and it shows up automatically.

> Status: **working skeleton.** The engine, plugin system, auto-generated
> controls, audio reactivity, color system, and post-FX are all in place, with
> two example effects (Liquid Fractal, Plasma).

## Run it

```bash
pip install -r requirements.txt
python app.py
```

**One window** opens with the live visual **in the center** and a **dock of
panels on each side** — Audio, Media, ✨ Shapes on the **left**; Effect,
Parameters, Export on the **right**. Each panel can be:

- **resized** — drag the sashes between panels (give Parameters more room), or
  the two outer sashes to widen either dock,
- **detached** — click **⧉** to pop it into its own floating window, **⧉ Dock**
  to put it back (it returns to its original side),
- **hidden / shown** — the **✕** on a panel, or the top **Panels** menu.

The top menu bar has **Project** (Open / Save / Save As / presets), **Panels**
(toggle each + Reset layout), **Theme** (Dark / Light / Midnight / Neon /
Sunset), and **Output** (full-screen / second-monitor). The toolbar has
Pause / Reset / **⛶ Output**, a live **FPS / ms-per-frame** readout on the
right, and a **status bar** along the bottom shows the current effect and the
last action. Hover any knob for a **tooltip** explaining it. **✎ Create Effect**
opens as its own (floating) panel. `Esc` closes.

The preview is **GPU-accelerated end-to-end**: the engine packs each finished
frame into a display-ready image on the GPU (clamp, flip and 8-bit conversion in
one kernel), so the readback is a quarter of the data and the preview reuses a
single image buffer instead of re-allocating one every frame — smooth, low-jitter
playback with the frame rate shown live in the toolbar.

### Full-screen / second-monitor output

Send the pure visual (no controls) to a projector or second screen while the
controls stay on your main display — the VJ/performance setup:

- **⛶ Output** in the toolbar, **F11**, or **Output → …** opens a borderless
  full-screen window of the live visual.
- **Output ▾** lists every connected monitor — pick which one to fill. With two
  screens, F11 sends the output to the *other* monitor automatically.
- **Esc**, **F11**, or **double-click** on the output exits it; the controls
  never leave your main window.

### Projects, presets & undo

Your whole setup is saved state, not throwaway:

- **It remembers itself.** Close and reopen the app and you're back exactly
  where you left off — effect, every knob, all shapes (and each shape's own
  effect + knobs), colors, audio bindings, theme, and which panels were open.
- **Save / load projects** (`.viz` files): **Project → Save** (`Ctrl+S`),
  **Save As**, **Open** (`Ctrl+O`). A project captures everything above.
- **Presets:** **Project → Save as preset…** drops a `.viz` into `presets/`;
  it then shows up under **Project → Load preset** for one-click recall.
- **Undo / Redo:** `Ctrl+Z` / `Ctrl+Y` (or `Ctrl+Shift+Z`). Changes are
  coalesced into sensible steps, so a slider drag is one undo, not a hundred.

In the panels you can:
- Switch **effect** from the dropdown.
- Choose an **audio source**: System (whatever's playing on your PC), Mic,
  a music **File**, or Off.
- Tweak the effect's own knobs (speed, size, swirl, …).
- Set the **look**: trails, fluid blur, grain, flicker, fade-in, brightness,
  and **feedback** (tunnels / spirals — see below).
- Pick **colors**: a named palette, or 2+ custom colors for your own gradient.
- **Drive any knob with a signal** — next to each slider, set `drive:` to an
  audio band (Bass / Mid / Treble / Vol / Beat, or the drum-tuned **Kick /
  Snare / Hi-hat**) **or an LFO** (see below), and dial the amount. e.g.
  particle size kicking on the kick drum, or swirl breathing on a slow sine.
  The live meter shows all seven audio bands.

If the audio libraries aren't installed or there's no device, the app still
runs — audio just stays "off".

### Modulation — make anything move, with or without sound

The **🎛 Modulation (LFOs)** panel gives you four **LFOs** — free-running shapes
(*sine, triangle, saw, ramp-down, square, random*) with a **Rate** (Hz) and
**Depth**. They show up in *every knob's* `drive:` menu as **LFO 1–4**, right
alongside the audio bands — so the exact same "route a source to a knob" gesture
drives a parameter off a slow pulse, a sharp square, or random steps instead of
(or as well as) the music. Set a square LFO on a color knob for a strobing
palette, a slow sine on zoom for a breathing image — no audio required.

This is the **routing spine**: audio bands, LFOs **and MIDI** are one unified
signal table, so they all share the exact same "drive" menu (future inputs like
OSC / Ableton Link will join it too). LFO settings are saved in your
project/session.

### MIDI control

Open the **🎹 MIDI** panel (Panels menu), pick your controller's **Port** and
**Connect**. Each of **MIDI 1–8** can track one knob/fader: click **Learn**, then
wiggle a control and it's captured (its CC + live value show in the row). Those
slots appear in *every knob's* `drive:` menu as **MIDI 1–8**, so a hardware fader
can ride brightness, a knob can sweep the swirl — hands-on, no AI. Mappings (and
the port) are saved with your project. MIDI needs `mido` + a backend
(`pip install mido pygame`); without them the panel says so and the rest of the
app is unaffected.

### Layers — stack & blend effects

The **🧱 Layers** panel stacks extra effects *on top of* the main effect, like
Photoshop layers. Each layer is a **full effect with its own knobs**:

- Pick its **blend mode** — *Normal, Add, Screen, Multiply, Lighten,
  Difference* — and its **opacity**, and toggle it on/off.
- Hit **⚙** to open **Layer FX** and edit that layer's *complete* knobs (look,
  grain, colors, and its own audio/LFO drive) — exactly like a normal effect.
- Stack up to **4 layers** above the base. So run **Plasma**, **Screen** a
  **Liquid Fractal** over it at 60%, then **Add** a third on the beat.

The main effect is always the bottom of the stack, so a single-effect setup is
unchanged until you add a layer. The whole stack (each layer's effect, blend,
opacity and knobs) is saved in your project/session.

### Feedback — tunnels, spirals & echoes

Flip on **feedback** (in the look knobs) to feed the **previous frame** back into
the current one, transformed and decayed — the classic infinite-tunnel look:

- **fb_zoom** — `>1` tunnels outward, `<1` sucks inward, `1.0` holds.
- **fb_rotate** — spins the echo each step, turning tunnels into **spirals**.
- **fb_decay** — how long echoes persist (too high blooms to white).

It runs as a post-pass on the final frame, so it works on **any** effect (not
just additive ones). And because these are ordinary knobs, they're **drivable** —
put **fb_zoom on the bass** and the tunnel pumps with the kick. The buffer resets
when you switch effects or hit Reset.

### Camera / image / video — as the *material* the effects act on

In the **Media (camera / image / video)** section pick a source: your **webcam**,
an **image**, or a **video** file (loops). Then in **✎ Create Effect** set
**Output** to control how the effect uses it:

- **Texture the media** — the effect plays *on* the media: your blocks/formulas
  brighten, reveal, and modulate the actual image/video pixels.
- **Warp the media** — the equations *distort* the media (ripples, swirls,
  displacement driven by the pattern and the music).
- **Paint colors** — the original behavior (palette colors, media ignored).

In formulas (Expression/Code) the media is available per-pixel as **`tex`**
(brightness), **`texr/texg/texb`** (colors), and **`motion`** (how much the
camera moved here) — so `motion*8` makes your movement glow: **interactive art**,
not a backdrop. In the Build tab, tick **"Glow where I move"** for the same thing
with no typing.

**Works on the built-in effects too:** set the **Media blend** knob to **Warp**
and *any* effect (Liquid Fractal, Plasma, …) distorts your media — no Create
Effect needed. **Behind / Tint / Screen** give the plain-backdrop mixes. These
are normal knobs, so they're **audio-drivable** (e.g. Media opacity ← Beat).

Needs `opencv-python`; without it the Media section is hidden and everything
else works.

### Shapes (elements) — objects that *interact* with your effect

Shapes are a **layer that sits on top of whatever effect is running and changes
its pixels** — they are not a separate effect. Click **✨ Shapes…** next to the
effect dropdown to open the **Shapes** panel, then drop elements on the visual:

- **Add** as many shapes as you like — *Circle, Square, Triangle, Pentagon,
  Hexagon, Star, Flower, Diamond*.
- **Place them by clicking the preview.** Select a shape (the ◉ button on its
  row), then click anywhere on the live visual to move it there — **drag** to
  slide it around. Or fine-tune with the **X / Y** sliders. **Size** and
  **Rotate** each one.
The **main effect (Effects panel) always fills the screen** — shapes sit on top
and only change the pixels *inside* themselves. (Want shapes alone on black?
set the main effect to **Blank (for shapes)**.)

- Choose how the shape **interacts with the effect**:
  - **Show effect** — paints an effect *inside* the shape, on top of the main
    one. Its **Shows:** dropdown picks which: **Primary** (the main effect) or
    **another effect**. Different shapes can show different effects, so you can
    drop a Liquid-Fractal star onto a Plasma background, or a Plasma circle and
    a Liquid-Fractal star side by side (up to **3** shapes showing their own
    effect at once). The main effect keeps running everywhere else.
- **⚙ Shape FX panel** — click the **⚙** on any shape (or **⚙ Shape FX** in the
  Shapes panel) to open the full editor for the **selected** shape. It's always
  available, whatever the shape does:
  - the shape's own controls (shape, mode, **Shows**, position, size, rotate,
    color, react, strength, speed, amount) in roomy full-size form, and
  - **if the shape Shows an effect**, that effect's *complete* knobs too —
    speed/scale/…, the look (trails, **grain**, fluid, flicker, fade, brightness),
    its own **colors**, and per-knob **audio drive** — exactly like a normal
    effect, but just for that shape. Two shapes showing the same effect can be
    tuned independently, and your tweaks stick.
  - **Hide effect** — the shape cuts a hole in the effect.
  - **Warp effect** — the shape bends the effect like a lens.
  - **Tint effect** — the shape colors the effect passing through it.
  - **Glow / Fill / Outline / Ripples** — the shape adds its *own* light/color
    on top of the effect (ripples radiate from its edge).
- Give it a **Color** (a position in the current palette) and make it **React to**
  Bass / Beat / Volume / Treble / **Kick / Snare / Hi-hat** with a **Strength** —
  e.g. a Reveal window that grows on the kick, or a Warp lens that pulses on the beat.

So you can run **Plasma** (or Liquid Fractal, or your own effect) and punch a
star-shaped window into it, warp it through a circle, or cut a hole on every
beat. Each shape is a *signed-distance field* composited by one data-driven
kernel, so adding and dragging is instant (nothing recompiles), and your shapes
stay put when you switch effects. Want shapes on a clean background? Pick the
**Blank (for shapes)** effect.

### Make your own effect (no setup, in-app)

Click **✎ Create Effect…** next to the effect dropdown. Three ways, easiest
first — all preview live and **Save** a real file into `effects/`:

- **🧩 Build (easiest, no typing):** stack **as many pattern blocks as you like**
  (➕ Add / ✕ remove), picked from plain names — *Circles, Waves, Spiral,
  Starburst, Ripples, Checkerboard, Tunnel*. Each block has **Size, Speed,
  Reverse, Reacts to (Bass/Beat/Volume/Treble), React strength,** and **Amount**.
  Then choose how blocks **blend**, the **colors**, and how they **move**. No math,
  no code. (Under the hood it writes the same formulas the engine runs.)
- **Expression (no code):** type two math formulas — one for *brightness*, one
  for *color* — e.g. `sin(x*scale + t) + bass*4`. A cheat-sheet lists every
  variable (`x y r theta t scale bass mid treble vol beat`) and function
  (`sin cos sqrt …`). Input is validated, so typos and anything unsafe are
  rejected with a clear message. Hit **Apply** to watch it live, **Save** to keep it.
- **Code (full control):** a built-in editor pre-filled with a complete, working
  effect (not a blank page). Edit the math, **Reload** to preview, **Save** when
  happy. A broken effect shows its error instead of crashing the app.

Saved effects appear in the dropdown immediately and behave like any built-in.

### Export an MP4 (with the audio baked in)

In the **Export MP4** panel: set FPS (and optionally a length in seconds),
click **Export MP4…**, pick the audio file to embed, and choose where to save.

This is an *offline* render: for every video frame it re-derives that moment's
audio features with the same analyzer, applies your current effect + knobs +
colors, and pipes the frames to ffmpeg alongside the audio track — so the
result is frame-accurate and perfectly synced (not a screen recording).

Because the audio has to be embedded, export needs an **audio file** (the one
you loaded, or one you pick). "System"/"Mic" are live and have nothing to bake
in. ffmpeg ships with the `imageio-ffmpeg` dependency — nothing to install
separately.

## How it's built

```
app.py                  entry point (init GPU, discover effects, open windows)
vizstudio/
  params.py     Slider / IntSlider / Toggle / Choice / ColorPalette  (self-describing knobs)
  effect.py     Effect base class + Context (what every effect receives)
  color.py      palette specs -> a 256-entry RGB lookup table the GPU samples
  audio.py      capture (system/mic/file) -> FFT -> bass/mid/treble/volume/beat
  media.py      camera / image / video input -> a frame the engine composites
  midi.py       MIDI input (optional): maps controllers to the MIDI 1-8 drivers
  postfx.py     global look: trails, fluid, grain, flicker, fade, brightness +
                Feedback (recursive frame feedback: tunnels / spirals / echoes)
  engine.py     owns canvas + colors + audio + media + post-FX, runs the loop;
                Presenter packs each frame to a display-ready uint8 image on GPU
  export.py     offline frame-accurate MP4 render, audio muxed via ffmpeg
  exprutil.py   validate + translate user math formulas to safe Taichi source
  exprfx.py     ExpressionEffectBase - an effect defined by two formulas
  patterns.py   named "pattern blocks" (Circles/Waves/...) -> formulas
  shapes.py     shape "elements" metadata + numeric encoding (mode/shape/react)
  shapes_fx.py  ShapesCompositor: draws shapes as a layer that masks/warps/tints
                the active effect (reveal / hide / warp / tint / glow)
  modulation.py LFOs + the unified signal table (audio bands + LFOs + MIDI) any
                knob can be driven by — the routing spine for all inputs
  layers_fx.py  LayerCompositor: blends a stack of effects over the main effect
                (Normal / Add / Screen / Multiply / Lighten / Difference)
  builder_templates.py   the .py templates the Create Effect window writes
  ui.py         control panel + the in-app Create Effect window
  registry.py   finds every Effect subclass defined in effects/
  project.py    save / load the whole state to .viz files (projects, presets, session)
effects/
  liquid_fractal.py   particle flow-field effect
  plasma.py           per-pixel plasma effect
  blank.py            black base, for using the Shapes overlay on its own
```

The core idea: **an effect declares its knobs, and everything else is
automatic.** The UI, the audio-driving, the colors, and the post-FX all read
from that one declaration — so a beginner sees only knobs, and a power user
gets full freedom by writing a new effect file.

## Write your own effect

See [CUSTOMIZATION.md](CUSTOMIZATION.md). Short version: copy
[effects/plasma.py](effects/plasma.py), rename the class, change the math,
restart. It appears in the dropdown.

## Validate without a GPU/window

```bash
python _smoketest.py   # runs every effect through the pipeline on the CPU backend
```

## Roadmap (not built yet)

- Save / load presets (`.json`) and per-effect favorites.
- Live in-app code editor (the old prototype had one) wired to the plugin system.
- More built-in effects (waveform, spectrum bars, tunnel, kaleidoscope).
- Fullscreen / second-monitor output; GIF export.
- Bloom and feedback post-FX passes.
- Record live system/mic audio to a file so it can be exported too.
