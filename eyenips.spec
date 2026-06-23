# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Eyenips (Windows, one-dir, windowed).

    pip install pyinstaller
    pyinstaller eyenips.spec --noconfirm

Output: dist/Eyenips/Eyenips.exe  (plus _internal/, effects/, presets/).
See BUILD.md for the full pipeline (installer + release manifest).

Design notes:
  * taichi / mediapipe / opencv ship native libs + data files, so we collect_all
    them rather than relying on auto-detection.
  * The selfie segmenter model goes into _internal/vizstudio/models, where
    vizstudio.paths.find() looks for it when frozen.
  * effects/ and presets/ are copied LOOSE next to the exe (not into _internal)
    so vizstudio.paths.effects_dir() (= <exe dir>/effects) finds them AND so you
    can ship content updates without rebuilding the whole exe.
"""

import os
import shutil

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe", "taichi", "cv2"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] collect_all({pkg}) skipped: {e}")

# the person-segmentation model -> _internal/vizstudio/models (paths.find finds it)
_model = os.path.join("vizstudio", "models", "selfie_segmenter.tflite")
if os.path.exists(_model):
    datas += [(_model, os.path.join("vizstudio", "models"))]

# optional/odd-to-detect deps that the app imports lazily
hiddenimports += [
    "soundcard", "soundfile", "sounddevice", "imageio_ffmpeg",
    "mido", "mido.backends.pygame", "pygame",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "unittest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

_icon = os.path.join("installer", "eyenips.ico")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Eyenips",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # windowed (no console) — safe_print handles None streams
    icon=_icon if os.path.exists(_icon) else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Eyenips",
)

# --- place loose, updatable content NEXT TO the exe (not inside _internal) ---
# effects/ (updatable), starter_presets/ (read-only shipped starters). User-saved
# presets live in %USERPROFILE%\.eyenips and are never bundled.
_dist_root = os.path.join(DISTPATH, "Eyenips")
for _folder in ("effects", "starter_presets", "assets"):
    _src = os.path.abspath(_folder)
    _dst = os.path.join(_dist_root, _folder)
    if os.path.isdir(_src):
        shutil.rmtree(_dst, ignore_errors=True)
        shutil.copytree(_src, _dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"[spec] bundled loose {_folder}/ -> {_dst}")
