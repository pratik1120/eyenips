"""Media input: live camera, a still image, or a video file.

A background thread grabs frames (camera/video) and keeps the latest one,
resized to the canvas size, as a float32 RGB array. The engine uploads that
into a Taichi field each frame and composites it with the rendered effect.

Like the audio engine, capture is optional and defensive: if OpenCV isn't
installed or a device/file can't be opened, media just stays off and the app
keeps running.
"""

import threading
import time

import numpy as np

from .params import Choice, Slider

# Video analysis: the most a "video effect" can track at once. Each blob is a
# moving region: [x, y, radius, vx, vy] in 0..1 image coords (y DOWN). The engine
# flips these into the canvas' y-up frame before handing them to effects.
MAX_BLOBS = 16


def media_params():
    """Compositing controls. Real params, so they auto-build UI AND can be
    audio-driven (e.g. camera opacity following the bass)."""
    return [
        Choice("media_blend", ["Off", "Behind", "Tint", "Screen", "Warp"],
               default="Off", label="Media blend",
               help="How camera/image/video mixes with ANY effect. Warp = the "
                    "effect distorts the media (works on Liquid Fractal, Plasma, ...)."),
        Slider("media_opacity", 0.0, 1.0, default=1.0, label="Media opacity",
               help="Mix strength (and the Warp distortion amount)."),
        Slider("media_brightness", 0.0, 2.0, default=1.0, label="Media brightness",
               help="Brighten/darken the media."),
    ]

# blend-mode name -> int the compositor kernel understands
BLEND_IDS = {"Off": 0, "Behind": 1, "Tint": 2, "Screen": 3, "Warp": 4}


class MediaSource:
    """Latest-frame provider. mode: "off" | "camera" | "image" | "video"."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None          # (H, W, 3) float32 0..1, already at target size
        self._mode = "off"
        self._path = None           # current image/video path (for restore/export)
        self._thread = None
        self._stop = threading.Event()
        self._target = (640, 400)   # (W, H) the engine wants
        self.status = "off"
        # --- video analysis (the "video to effects" core): optical flow + blob
        # tracking, computed in the grab thread so the render loop stays light. ---
        self._analyze = False
        self._prev_gray = None
        self._flow = None               # (H, W, 2) float32, normalized motion
        self._blobs = np.zeros((MAX_BLOBS, 5), np.float32)
        self._n_blobs = 0
        self._blob_prev = []            # last frame's centroids (for velocity)
        self._cv_res = (192, 108)       # analyze cheap, upscale flow to target

    # ---- control --------------------------------------------------------
    def set_target(self, w, h):
        self._target = (int(w), int(h))

    def has_frame(self):
        return self._frame is not None

    def frame(self):
        with self._lock:
            return self._frame

    # ---- video analysis -------------------------------------------------
    def set_analyze(self, on):
        """Enable/disable flow + blob tracking (only while a video effect wants
        it, so plain playback pays nothing)."""
        on = bool(on)
        if on and not self._analyze:
            self._prev_gray = None       # restart cleanly
            self._blob_prev = []
        self._analyze = on
        if not on:
            with self._lock:
                self._flow = None
                self._n_blobs = 0

    def flow(self):
        with self._lock:
            return self._flow

    def blobs(self):
        with self._lock:
            return self._blobs.copy(), self._n_blobs

    def _compute_analysis(self, bgr):
        """Dense optical flow + connected-component blob tracking on the video,
        at low res. Pure OpenCV — no AI. Blobs are MOVING regions (frame
        differencing), each tagged with a velocity matched from last frame."""
        import cv2
        cw, ch = self._cv_res
        small = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev = self._prev_gray

        # --- optical flow (the motion field effects can sample) ---
        if prev is not None and prev.shape == gray.shape:
            fl = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 21, 3, 7, 1.5, 0)
            fl[..., 0] /= cw
            fl[..., 1] /= ch
            fl = cv2.GaussianBlur(fl, (0, 0), 2)
            np.clip(fl, -0.25, 0.25, out=fl)
        else:
            fl = np.zeros((ch, cw, 2), np.float32)

        # --- blobs: moving regions -> tracked points with size + velocity ---
        blobs = np.zeros((MAX_BLOBS, 5), np.float32)
        n = 0
        if prev is not None and prev.shape == gray.shape:
            diff = cv2.absdiff(gray, prev)
            _, m = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            ncc, _lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
            min_area = cw * ch * 0.002
            items = []
            for k in range(1, ncc):                      # 0 = background
                area = stats[k, cv2.CC_STAT_AREA]
                if area >= min_area:
                    r = min(0.30, ((area / 3.14159) ** 0.5) / ch)   # disc radius, capped
                    items.append((area, cents[k][0] / cw, cents[k][1] / ch, r))
            items.sort(reverse=True)                     # biggest movers first
            items = items[:MAX_BLOBS]
            for idx, (area, x, y, r) in enumerate(items):
                vx = vy = 0.0
                bd = 0.15 ** 2                           # max match distance²
                for px, py in self._blob_prev:
                    d = (px - x) ** 2 + (py - y) ** 2
                    if d < bd:
                        bd = d
                        vx, vy = x - px, y - py
                blobs[idx] = (x, y, r, vx, vy)
                n += 1
            self._blob_prev = [(it[1], it[2]) for it in items]

        self._prev_gray = gray
        w, h = self._target
        flow = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        with self._lock:
            self._flow = flow
            self._blobs = blobs
            self._n_blobs = n

    def set_mode(self, mode, path=None, cam_index=0):
        self.stop()
        self._mode = mode
        self._path = path
        with self._lock:
            self._frame = None
        if mode == "off":
            self.status = "off"
            return
        if mode == "image":
            self._load_image(path)
            return
        # camera / video run in a thread
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(mode, path, cam_index), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    # ---- helpers --------------------------------------------------------
    def _resize_rgb(self, bgr):
        import cv2
        w, h = self._target
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0)

    def _publish(self, rgb):
        with self._lock:
            self._frame = rgb

    def _load_image(self, path):
        try:
            import cv2
            bgr = cv2.imread(path)
            if bgr is None:
                self.status = "image: could not open"
                return
            self._publish(self._resize_rgb(bgr))
            self.status = f"image: {path.split('/')[-1].split(chr(92))[-1]}"
        except Exception as e:
            self.status = f"image off ({type(e).__name__})"

    # ---- capture loop ---------------------------------------------------
    def _run(self, mode, path, cam_index):
        try:
            import cv2
            if mode == "camera":
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # DSHOW = fast on Windows
                self.status = f"camera #{cam_index}"
            else:
                cap = cv2.VideoCapture(path)
                self.status = f"video: {path.split('/')[-1].split(chr(92))[-1]}"
            if not cap.isOpened():
                self.status = f"{mode}: could not open"
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay = 1.0 / src_fps if (mode == "video" and src_fps > 1) else 0.0
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok:
                    if mode == "video":          # loop the file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                self._publish(self._resize_rgb(bgr))
                if self._analyze:
                    try:
                        self._compute_analysis(bgr)
                    except Exception:
                        pass            # never let CV crash the capture loop
                if delay:
                    time.sleep(delay)
            cap.release()
        except Exception as e:
            self.status = f"{mode} off ({type(e).__name__}: {e})"
