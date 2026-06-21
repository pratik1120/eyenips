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
        Choice("subject_front", ["Off", "Person", "Motion"], default="Off",
               label="Effect behind subject",
               help="Keep the SUBJECT in FRONT while the effect plays BEHIND "
                    "them (the TouchDesigner look). Person = body segmentation "
                    "(clean edges, best for people, any motion). Motion = cuts "
                    "out whatever MOVES (no model, ultra-light & smooth, great "
                    "for a static camera; a still subject fades). No generative "
                    "AI either way."),
        Slider("subject_strength", 0.0, 1.0, default=1.0, label="Subject strength",
               help="How solidly the subject sits in front of the effect."),
        Slider("subject_feather", 0.0, 1.0, default=0.4, label="Subject edge",
               help="Soften (towards 1) or sharpen (towards 0) the cut-out edge "
                    "around the subject."),
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
        # --- subject cutout (effect-behind-the-subject): a foreground mask,
        # computed in the grab thread when an effect/composite wants it. Two
        # modes: "person" (MediaPipe body segmentation) or "motion" (background
        # subtraction — cuts out what moves). Both are temporally smoothed. ---
        self._mask_mode = "off"         # "off" | "person" | "motion"
        self._want_mask = False
        self._mask = None               # (H, W) float32 0..1 at target size
        self._mask_ema = None           # smoothed mask (kills frame-to-frame flicker)
        self._seg = None                # lazy MediaPipe ImageSegmenter
        self._seg_fail = False
        self.seg_status = ""
        self._bgsub = None              # lazy OpenCV background subtractor (motion)

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

    # ---- subject cutout (effect behind the subject) ---------------------
    def set_mask(self, mode):
        """Pick the cutout mode: "off" | "person" | "motion". Anything else is
        treated as off, so plain playback pays nothing."""
        mode = (mode or "off").lower()
        if mode not in ("person", "motion"):
            mode = "off"
        if mode != self._mask_mode:           # switching modes -> drop history
            self._bgsub = None
            self._mask_ema = None
        self._mask_mode = mode
        self._want_mask = mode != "off"
        if not self._want_mask:
            with self._lock:
                self._mask = None

    def mask(self):
        with self._lock:
            return self._mask

    def _get_segmenter(self):
        """Lazily build MediaPipe's selfie ImageSegmenter. A discriminative
        sensor (like optical flow) — it labels pixels, it doesn't generate art.
        If MediaPipe / the model isn't available, the feature just stays off."""
        if self._seg is not None:
            return self._seg
        if self._seg_fail:
            return None
        try:
            import os
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            model = os.path.join(os.path.dirname(__file__),
                                 "models", "selfie_segmenter.tflite")
            opts = vision.ImageSegmenterOptions(
                base_options=mpp.BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True)
            self._seg = vision.ImageSegmenter.create_from_options(opts)
            self.seg_status = "subject mask: on"
        except Exception as e:
            self._seg_fail = True
            self.seg_status = f"subject mask off ({type(e).__name__})"
            return None
        return self._seg

    def _person_mask(self, bgr):
        """MediaPipe body segmentation -> soft mask (h, w) float32 0..1, or None
        if the model isn't available."""
        seg = self._get_segmenter()
        if seg is None:
            return None
        import cv2
        import mediapipe as mp
        w, h = self._target
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = seg.segment(mp_img)
        m = res.confidence_masks[0].numpy_view()      # (h, w) float32 0..1
        return cv2.GaussianBlur(m, (0, 0), 1.5)       # soften the cut-out edge

    def _motion_mask(self, bgr):
        """Background-subtraction cutout -> soft mask of whatever MOVES. Pure
        OpenCV, no model — ultra-light and temporally smooth. Best with a static
        camera; a subject that stops moving slowly fades back in."""
        import cv2
        w, h = self._target
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if self._bgsub is None:
            self._bgsub = cv2.createBackgroundSubtractorMOG2(
                history=160, varThreshold=24, detectShadows=False)
        fg = self._bgsub.apply(small)                 # 0/255 foreground
        k3 = np.ones((3, 3), np.uint8)
        k11 = np.ones((11, 11), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)            # drop speckle
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k11, iterations=2)  # fill body
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)
        m = fg.astype(np.float32) / 255.0
        return cv2.GaussianBlur(m, (0, 0), 3.0)       # soft, blobby edge

    def _compute_mask(self, bgr):
        """Compute the active-mode foreground mask, smooth it across frames to
        kill flicker, and store it at target size (H, W) float32 0..1."""
        if self._mask_mode == "person":
            m = self._person_mask(bgr)
        elif self._mask_mode == "motion":
            m = self._motion_mask(bgr)
        else:
            return
        if m is None:
            return
        # temporal EMA: blend with the last mask so edges don't shimmer/flicker.
        prev = self._mask_ema
        if prev is not None and prev.shape == m.shape:
            m = prev * 0.55 + m * 0.45
        self._mask_ema = m
        with self._lock:
            self._mask = m.astype(np.float32)

    def set_mode(self, mode, path=None, cam_index=0):
        self.stop()
        self._mode = mode
        self._path = path
        self._bgsub = None              # new clip -> relearn the background
        self._mask_ema = None
        with self._lock:
            self._frame = None
            self._mask = None
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
                if self._want_mask:
                    try:
                        self._compute_mask(bgr)
                    except Exception:
                        pass            # segmentation is best-effort
                if delay:
                    time.sleep(delay)
            cap.release()
        except Exception as e:
            self.status = f"{mode} off ({type(e).__name__}: {e})"
