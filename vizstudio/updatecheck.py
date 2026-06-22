"""Lightweight 'is there a newer version?' check.

On launch (and from Help -> Check for updates) the app fetches a tiny JSON
manifest and compares its version to the running one. No server, no auto-install
— just a heads-up with a download link. Everything here is best-effort and never
raises into the app: no network, a private/missing manifest, or bad JSON simply
means 'no update info', so it can ship before the manifest even exists.

Manifest format (host it at a PUBLIC url — e.g. a public 'eyenips-releases'
repo, a gist, or your site; the app's repo is private so raw access needs this):

    { "version": "0.2.0",
      "url": "https://.../download/Eyenips-Setup-0.2.0.exe",
      "notes": "What's new in this release (optional)." }
"""

import json
import threading
import urllib.request

# Where the app looks for the release manifest. Point this at YOUR public file.
DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/pratik1120/eyenips-releases/main/latest.json")


def _parse(v):
    """'v0.2.1' -> (0, 2, 1). Tolerant of leading 'v' and trailing junk."""
    out = []
    for part in str(v).strip().lstrip("vV").split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        out.append(int(digits) if digits else 0)
    return tuple(out)


def is_newer(remote, local):
    """True if version string `remote` is strictly newer than `local`."""
    return _parse(remote) > _parse(local)


def fetch(url, timeout=4):
    req = urllib.request.Request(url, headers={"User-Agent": "Eyenips"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check(current_version, url=DEFAULT_MANIFEST_URL):
    """Return {version, url, notes} if a newer release exists, else None.
    Never raises — any failure (offline, 404, bad JSON) yields None."""
    try:
        data = fetch(url)
        remote = data.get("version")
        if remote and is_newer(remote, current_version):
            return {"version": remote,
                    "url": data.get("url", ""),
                    "notes": data.get("notes", "")}
    except Exception:
        return None
    return None


def check_async(current_version, callback, url=DEFAULT_MANIFEST_URL):
    """Run check() on a daemon thread and call callback(result) when done
    (result is the dict or None). The callback runs OFF the UI thread — marshal
    back to Tk with root.after(0, ...)."""
    def run():
        callback(check(current_version, url))
    threading.Thread(target=run, daemon=True).start()
