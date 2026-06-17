# Writing your own effect

An effect is one Python file in `effects/`. Drop it in, restart `app.py`, and
it appears in the effect dropdown. No registration, no editing the core.

## The smallest possible effect

```python
import taichi as ti
from vizstudio import Effect, Slider, ColorPalette

@ti.data_oriented
class MyEffect(Effect):
    name = "My Effect"

    params = [
        Slider("speed", 0.0, 4.0, default=1.0),
        ColorPalette(),
    ]

    def setup(self, ctx):
        self.w, self.h = ctx.width, ctx.height
        self.canvas = ctx.canvas      # draw here: ti.Vector.field(3) shape (W, H)
        self.palette = ctx.palette    # color LUT: ti.Vector.field(3) shape (256,)

    @ti.kernel
    def _draw(self, t: ti.f32):
        for i, j in self.canvas:
            c = (ti.sin(i * 0.01 + t) * 0.5 + 0.5)
            ci = ti.cast(c * 255.0, ti.i32) % 256
            self.canvas[i, j] = self.palette[ci]

    def render(self, ctx):
        self._draw(ctx.time * float(ctx.p["speed"]))
```

That's a complete, audio-reactive, color-customizable effect.

## What you declare: params

Each param becomes a control automatically. Numeric ones can be driven by audio.

| Param | Widget | Read it as |
| --- | --- | --- |
| `Slider("speed", 0.1, 5.0, default=1.0)` | slider + audio-drive | `ctx.p["speed"]` (float) |
| `IntSlider("count", 1, 8, default=2)` | slider (whole numbers) | `ctx.p["count"]` (int) |
| `Toggle("mirror", default=False)` | checkbox | `ctx.p["mirror"]` (bool) |
| `Choice("mode", ["a","b"])` | dropdown | `ctx.p["mode"]` (str) |
| `ColorPalette()` | palette + color pickers | sample `ctx.palette` in a kernel |

Pass `audio=False` to a slider if it shouldn't be audio-drivable (e.g. a
structural "scale" knob). Pass `help="..."` to document a knob.

## What you receive: `ctx` (the Context)

| Field | What it is |
| --- | --- |
| `ctx.canvas` | the RGB image to draw into, `ti.Vector.field(3)` shape `(W, H)` |
| `ctx.palette` | the chosen colors as a 256-entry LUT, `ti.Vector.field(3)` |
| `ctx.p` | resolved knob values **with audio already mixed in** |
| `ctx.audio` | raw features: `ctx.audio.bass/.mid/.treble/.volume` (0..1), `.beat` (bool) |
| `ctx.time` | seconds since the effect started |
| `ctx.dt` | seconds since last frame |
| `ctx.frame` | frame counter |
| `ctx.width`, `ctx.height` | canvas size |

You usually only need `ctx.p`, `ctx.canvas`, `ctx.palette`, and `ctx.time`.

## The three methods

- `setup(self, ctx)` — runs once when the effect is selected. Allocate your
  Taichi fields here and stash `ctx.canvas` / `ctx.palette`.
- `render(self, ctx)` — runs every frame. Copy the `ctx.p` values you need into
  kernel arguments and launch your kernel(s) to draw into `ctx.canvas`.
- `reset(self)` — optional. Re-seed your state (called on `r` / Reset).

## Two drawing styles

- **Particles** (like `effects/liquid_fractal.py`): keep particle fields, move
  them, and *add* color into the canvas. Pair with **trails** for smear.
- **Per-pixel** (like `effects/plasma.py`): loop over `ctx.canvas` and *set*
  each pixel from a math field. Effectively a tiny fragment shader in Python.

## Colors: always go through the palette

Don't hard-code RGB. Map some value to `0..255` and read `ctx.palette[idx]`.
Then the user's color choices (named gradients or custom multi-color gradients)
"just work" with your effect, for free.

## The look (grain / fluid / flicker / fade) is not your job

Those are global post-FX applied after you render, available on every effect.
Don't implement them yourself — just draw your raw visual and let the user
toggle the look.

## Running

The real entry point is `python app.py` (from the project root). Your effect is
discovered automatically — you don't run effect files to "use" them.

If you press your IDE's **Run** button while an effect file is open, the two
convenience blocks in the example effects (a `sys.path` line at the top and an
`if __name__ == "__main__"` launcher at the bottom) make that just open the
studio on your effect. Copy them into your own effect if you want the same.

## Gotchas

- Put `@ti.data_oriented` on your class so its methods can be `@ti.kernel`.
- Don't add `from __future__ import annotations` — it breaks Taichi's reading
  of `ti.f32` kernel argument types.
- Field sizes (e.g. particle count) are fixed once allocated in `setup`. To make
  count adjustable, re-allocate on `reset()`.
```
