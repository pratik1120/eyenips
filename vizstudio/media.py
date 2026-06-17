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

    # ---- control --------------------------------------------------------
    def set_target(self, w, h):
        self._target = (int(w), int(h))

    def has_frame(self):
        return self._frame is not None

    def frame(self):
        with self._lock:
            return self._frame

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
                if delay:
                    time.sleep(delay)
            cap.release()
        except Exception as e:
            self.status = f"{mode} off ({type(e).__name__}: {e})"
