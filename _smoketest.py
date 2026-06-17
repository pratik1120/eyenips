"""Headless validation: run the engine pipeline a few frames with NO window
and NO audio device, on the CPU backend. Exercises params -> resolve ->
effect.render -> postfx for every discovered effect.
"""
import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from vizstudio.engine import Engine
from vizstudio.registry import discover

HERE = os.path.dirname(os.path.abspath(__file__))
fx, errs = discover(os.path.join(HERE, "effects"))
for f, m in errs:
    print("LOAD ERROR", f, m.splitlines()[-1])
assert fx, "no effects discovered"
print("effects:", [c.name for c in fx])

eng = Engine(320, 200, audio=None)


class FakeFeats:
    def get(self, s):
        return {"bass": 0.8, "volume": 0.5}.get(s, 0.3)


for cls in fx:
    eng.set_effect(cls())
    # turn on a bunch of post-FX + an audio drive to exercise paths
    eng.store.set("grainy", True)
    eng.store.set("fluid", True)
    eng.store.set("flicker", True)
    if "speed" in eng.store.params:
        eng.store.set_audio("speed", source="bass", amount=0.7)
    for frame in range(5):
        eng.ctx.time = frame * 0.05
        eng.ctx.audio = FakeFeats()
        eng.ctx.p = eng._resolve(eng.ctx.audio)
        eng._upload_palette()
        if eng.ctx.p.get("trails"):
            eng.postfx.decay(float(eng.ctx.p.get("trail_length", 0.9)))
        else:
            eng._clear_canvas()
        eng.effect.render(eng.ctx)
        eng.postfx.apply(eng.ctx.p, eng.ctx.time)
    img = eng.canvas.to_numpy()
    np.clip(img, 0, 1, out=img)
    print(f"OK {cls.name:16s} frames=5 img range [{img.min():.3f},{img.max():.3f}] mean={img.mean():.4f}")

print("SMOKETEST PASSED")
