"""
Reusable webcam capture abstraction (Milestone 2).

Wraps cv2.VideoCapture with:
  - explicit V4L2 backend + MJPG fourcc (cheap UVC webcams like the Logitech C510
    often cap out at ~5 FPS at 720p in the default YUYV format; MJPG is required
    to hit 30 FPS at 720p on this class of hardware),
  - descriptive failures instead of a bare `False` return,
  - a rolling FPS counter,
  - "latest frame wins" semantics: callers should always ask for the newest frame
    rather than queuing every one (Section 25 — avoid unbounded latency buildup).

This module has no dependency on the face pipeline so it can be exercised and
benchmarked standalone.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised with an actionable message — never a bare 'camera failed'."""


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    # Many cheap UVC webcams (measured on a Logitech C510) throttle their real
    # frame rate to well under the requested value in low light: auto-exposure
    # extends exposure time per frame, which caps FPS regardless of the
    # requested setting. Setting manual_exposure_value forces a short, fixed
    # exposure so the sensor can actually deliver the requested frame rate, at
    # the cost of a darker/noisier image in low light. Leave None to keep the
    # camera's default auto-exposure behavior.
    manual_exposure_value: int | None = None


@dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: float  # time.monotonic() at the moment of capture
    frame_index: int


class FpsCounter:
    """Rolling FPS over a sliding window (default: last 2 seconds of samples)."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._timestamps.append(now)
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span


class CameraCapture:
    """Opens a V4L2 camera device and yields frames.

    Usage:
        with CameraCapture(CameraConfig(device_index=0)) as cam:
            frame = cam.read()
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._cap: cv2.VideoCapture | None = None
        self._fps_counter = FpsCounter()
        self._frame_index = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera at /dev/video{self.config.device_index}.\n"
                "Check:\n"
                f"  1. The device exists: ls -la /dev/video{self.config.device_index}\n"
                "  2. Your user has permission (member of the 'video' group, or an "
                "ACL entry — see `getfacl`).\n"
                "  3. No other application is holding the device open."
            )

        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # UVC exposure controls are stateful on the device itself and persist
        # across process restarts / open-close cycles — measured directly: a
        # manual exposure value set in one process was still in effect after
        # closing and reopening the device from a fresh process. Always set the
        # mode explicitly rather than relying on whatever a previous session
        # left behind, or behavior becomes non-reproducible.
        #
        # IMPORTANT (empirically determined on this system, do not trust the
        # commonly-quoted 0.25/0.75 OpenCV convention — it does NOT hold for
        # this camera/driver/OpenCV build): passing 1 selects true manual mode;
        # passing 3 (V4L2's V4L2_EXPOSURE_APERTURE_PRIORITY) is what actually
        # re-enables the hardware auto-exposure algorithm. Verified by reading
        # back actual frame brightness, not just the FPS side effect — 0.75
        # silently failed to apply and left the device in whatever exposure
        # mode a *previous* process had set, producing a fast but pitch-black
        # image. If you port this to different hardware, re-verify empirically
        # rather than trusting either convention blindly.
        if self.config.manual_exposure_value is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.config.manual_exposure_value)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        # NOTE: do NOT force CAP_PROP_BUFFERSIZE=1 here. Measured on the Logitech
        # C510: forcing a 1-frame driver buffer serialized each read behind a full
        # USB transfer and cut throughput from ~14 FPS to ~7.7 FPS. "Latest frame
        # wins" is instead implemented one layer up, in ThreadedCameraStream, via a
        # background thread that drains the driver at full speed and hands the
        # consumer only the newest frame — see that class for why.

        self._cap = cap

    def read(self) -> Frame:
        if self._cap is None:
            raise CameraError("CameraCapture.read() called before open().")
        ok, image = self._cap.read()
        if not ok or image is None:
            raise CameraError(
                f"Camera read failed on /dev/video{self.config.device_index}.\n"
                "The device may have been unplugged, or another process took "
                "exclusive control of it. Re-open the capture to recover."
            )
        now = time.monotonic()
        self._fps_counter.tick(now)
        self._frame_index += 1
        return Frame(image=image, timestamp=now, frame_index=self._frame_index)

    def actual_format(self) -> dict:
        if self._cap is None:
            raise CameraError("Camera not open.")
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_requested": self.config.fps,
            "fps_reported_by_driver": self._cap.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc_str,
        }

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ThreadedCameraStream:
    """Latest-frame-wins wrapper around CameraCapture (Section 25).

    A background thread continuously reads from the camera at whatever rate the
    hardware actually sustains and stores only the single newest frame. The
    consumer (the face pipeline's capture step) calls `get_latest()` and never
    blocks on — or falls behind — the camera; if the pipeline is momentarily slow,
    old frames are silently dropped rather than queued, which is the correct
    trade-off for real-time video (bounded latency over completeness).
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self._camera = CameraCapture(config)
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._error: CameraError | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._camera.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._camera.read()
            except CameraError as e:
                with self._lock:
                    self._error = e
                return
            with self._lock:
                self._latest = frame

    def get_latest(self) -> Frame:
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._latest is None:
                raise CameraError("No frame captured yet — camera still starting up.")
            return self._latest

    @property
    def fps(self) -> float:
        return self._camera.fps

    @property
    def capture(self):
        """The live cv2.VideoCapture, for components that must drive camera
        controls (see shared/utils/exposure.py). None before start()."""
        return self._camera._cap

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._camera.close()

    def __enter__(self) -> "ThreadedCameraStream":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
