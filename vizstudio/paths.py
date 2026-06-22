"""Filesystem locations that work in BOTH a dev checkout and a frozen .exe.

A PyInstaller build changes what `__file__` means, and a `--windowed` exe has no
console at all (`sys.stdout` / `sys.stderr` are `None`), so dev-only assumptions
silently break — or crash — once packaged. Routing paths and startup messages
through here keeps both modes working, and keeps the "effects live as loose,
updatable files next to the exe" design intact.
"""

import os
import sys


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def app_dir():
    """Where the app lives: the folder containing the .exe when frozen (so loose
    effects/ and other updatable content sit next to it), else the project root."""
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bundle_dir():
    """Where read-only BUNDLED resources are unpacked: PyInstaller's _MEIPASS for
    a one-file build, else the app dir."""
    return getattr(sys, "_MEIPASS", app_dir())


def effects_dir():
    """The loose, updatable effects folder (next to the exe / in the project)."""
    return os.path.join(app_dir(), "effects")


def user_data_dir():
    """Per-user writable data (lab kits, presets, sessions). Lives in the home
    folder, OUTSIDE the install dir, so it survives reinstalls and updates."""
    return os.path.join(os.path.expanduser("~"), ".eyenips")


def find(*relpath_candidates):
    """First existing path among relative candidates, searched under both the
    bundle dir and the app dir. If none exist, returns the bundle-dir form of the
    first candidate (best guess) so callers can still produce a clear error."""
    bases = []
    for b in (bundle_dir(), app_dir()):
        if b not in bases:
            bases.append(b)
    for rel in relpath_candidates:
        for b in bases:
            p = os.path.join(b, rel)
            if os.path.exists(p):
                return p
    return os.path.join(bundle_dir(), relpath_candidates[0])


def safe_print(*args, **kwargs):
    """print() that never crashes a windowed exe (where stdout/stderr are None).
    Silently does nothing when there's no console stream to write to."""
    kwargs.pop("file", None)
    stream = sys.stderr or sys.stdout
    if stream is None:
        return
    try:
        print(*args, file=stream, **kwargs)
    except Exception:
        pass
