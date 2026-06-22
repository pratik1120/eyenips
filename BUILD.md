# Building & shipping Eyenips (Windows `.exe`)

Three pieces: **package** (PyInstaller) → **installer** (Inno Setup) → **updates**
(a tiny JSON manifest the app checks on launch).

## 0. Prereqs
```bash
pip install pyinstaller
```
Install **Inno Setup** (free): https://jrsoftware.org/isdl.php
Use the **same Python env that runs the app** so taichi/mediapipe/opencv match.

## 1. Package the exe
```bash
pyinstaller eyenips.spec --noconfirm
```
Output: `dist/Eyenips/`
```
dist/Eyenips/
  Eyenips.exe         ← the app (windowed, no console)
  _internal/          ← Python, taichi, mediapipe, opencv, the .tflite model
  effects/            ← LOOSE, updatable effect files (next to the exe)
  starter_presets/    ← read-only starter presets shipped with the app
```
Smoke-test it: run `dist/Eyenips/Eyenips.exe`. It should open, list effects, and
react to audio. (Test on a machine **without** a good GPU too — it falls back to
CPU. And confirm it runs where there's no console window.)

Why effects/ + presets/ sit loose next to the exe: `vizstudio/paths.py` resolves
them relative to the exe, so you can **ship content updates by replacing those
files** — no full rebuild. User data (presets you save, sessions, lab kits) lives
in `%USERPROFILE%\.eyenips` and survives every update/uninstall.

## 2. Build the installer
```bash
iscc installer\eyenips.iss
```
Output: `dist_installer/Eyenips-Setup-0.1.0.exe`

It's a **per-user** install (`%LOCALAPPDATA%\Programs\Eyenips`) — no admin prompt,
and the app can self-update its own `effects/` without elevation.

> Unsigned exes trip Windows SmartScreen ("unknown publisher"). Users click
> *More info → Run anyway*. A code-signing certificate (~$100+/yr) removes it —
> skip until you have real users.

## 3. Wire up updates
The app checks `vizstudio/updatecheck.py:DEFAULT_MANIFEST_URL` on launch (and from
**Help → Check for updates**). It's silent unless a newer version exists, and
never errors if the URL is missing/offline — so you can ship before it exists.

Because this repo is **private**, host the manifest at a **public** URL — a public
`eyenips-releases` repo, a gist, or your site. Default points at:
```
https://raw.githubusercontent.com/pratik1120/eyenips-releases/main/latest.json
```
`latest.json` (see `installer/latest.example.json`):
```json
{
  "version": "0.2.0",
  "url": "https://github.com/.../releases/download/v0.2.0/Eyenips-Setup-0.2.0.exe",
  "notes": "What's new in this release."
}
```
The app compares `version` to its own `vizstudio.__version__` and, if newer, shows
a dialog with a **Download** button that opens `url`.

## Releasing a new version — checklist
1. Bump `__version__` in `vizstudio/__init__.py` **and** `MyAppVersion` in
   `installer/eyenips.iss` (keep them in sync).
2. `pyinstaller eyenips.spec --noconfirm`
3. `iscc installer\eyenips.iss`
4. Upload `dist_installer/Eyenips-Setup-X.Y.Z.exe` somewhere public (e.g. a
   GitHub Release).
5. Update the public `latest.json` (`version` + `url`).
6. Existing users get the "update available" prompt on their next launch.

### Content-only updates (cheap)
New/changed **effects** or **presets** don't need a new installer: drop the
updated `.py` / `.viz` files into the user's `effects/` or `presets/` folder.
(Only ship effect files from your own trusted source — they execute as code.)

## Optional polish before a public release
- An app icon: add `installer/eyenips.ico` (the spec + iss already reference it).
- Bundle third-party license notices (MediaPipe/Taichi/OpenCV are required when
  you redistribute) — drop a `THIRD_PARTY_LICENSES.txt` next to the exe.
