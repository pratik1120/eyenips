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
        # --- "Become the Visual": use the camera as YOU (interactive art) ---
        Choice("mirror", ["Off", "Become the effect", "Push by motion", "Both"],
               default="Off", label="🪞 Become the Visual",
               help="Use your CAMERA as YOU. 'Become the effect' renders the effect "
                    "INSIDE your silhouette (you're made of fire / plasma); 'Push by "
                    "motion' makes your movement shove the visual around. Camera on."),
        Slider("mirror_push", 0.0, 1.0, default=0.4, label="Motion push",
               help="How hard your movement pushes the visual (Push by motion)."),
        Slider("mirror_bg", 0.0, 1.0, default=0.0, label="Show me (bg)",
               help="How much of the camera shows behind your silhouette (0 = black)."),
    ]

# blend-mode name -> int the compositor kernel understands
BLEND_IDS = {"Off": 0, "Behind": 1, "Tint": 2, "Screen": 3, "Warp": 4}


class MediaSource:
    """Latest-frame provider. mode: "off" | "camera" | "image" | "video"."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None          # (H, W, 3) float32 0..1, already at target size
        self._mode = "off"
        self._thread = None
        self._stop = threading.Event()
        self._target = (640, 400)   # (W, H) the engine wants
        self.status = "off"
        # "Become the Visual": silhouette (background subtraction) + optical flow,
        # computed here in the grab thread so the render loop stays light.
        self._mask = None           # (H, W) float32 0..1 — the person
        self._flow = None           # (H, W, 2) float32 — normalized motion vectors
        self._interactive = False
        self._mog = None            # cv2 background subtractor (lazy)
        self._prev_gray = None
        self._cv_res = (192, 108)   # compute CV cheap, then upscale

    # ---- control --------------------------------------------------------
    def set_target(self, w, h):
        self._target = (int(w), int(h))

    def has_frame(self):
        return self._frame is not None

    def frame(self):
        with self._lock:
            return self._frame

    # ---- "Become the Visual": silhouette + optical flow ----------------
    def set_interactive(self, on):
        """Turn silhouette/flow computation on or off (only on while a mirror
        mode is active, so we don't pay for CV when it's unused)."""
        on = bool(on)
        if on and not self._interactive:
            self._prev_gray = None          # restart flow cleanly
        self._interactive = on
        if not on:
            with self._lock:
                self._mask = None
                self._flow = None

    def reset_background(self):
        """Re-learn the empty scene (step out of frame, click, step back in)."""
        self._mog = None

    def mask(self):
        with self._lock:
            return self._mask

    def flow(self):
        with self._lock:
            return self._flow

    def _compute_cv(self, bgr):
        """Silhouette (MOG2 background subtraction) + dense optical flow, at a
        small resolution, upscaled to the target. Pure OpenCV — no AI."""
        import cv2
        cw, ch = self._cv_res
        small = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._mog is None:
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=250, varThreshold=24, detectShadows=False)
        fg = self._mog.apply(small)
        k3 = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg = cv2.GaussianBlur(fg, (9, 9), 0)
        mask_small = fg.astype(np.float32) / 255.0

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            fl = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
            fl[..., 0] /= cw                 # normalize to fractions of the frame
            fl[..., 1] /= ch
        else:
            fl = np.zeros((ch, cw, 2), np.float32)
        self._prev_gray = gray

        w, h = self._target
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        with self._lock:
            self._mask = mask
            self._flow = flow

    def set_mode(self, mode, path=None, cam_index=0):
        self.stop()
        self._mode = mode
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
                if self._interactive:
                    try:
                        self._compute_cv(bgr)
                    except Exception:
                        pass            # never let CV crash the capture loop
                if delay:
                    time.sleep(delay)
            cap.release()
        except Exception as e:
            self.status = f"{mode} off ({type(e).__name__}: {e})"
