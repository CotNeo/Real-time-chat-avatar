"""
Face-metered auto exposure.

The camera's own auto-exposure meters the whole scene. With a window or lamp
behind the user it exposes for that bright background and leaves the face in
shadow — measured on this setup, whole-frame brightness looks acceptable while
the face itself sits far below usable. Everything downstream then degrades:
the swap model receives a dark, noisy 128px crop and returns a washed-out face.

This controller ignores the background entirely and drives the camera's manual
exposure until the *face region* hits a target brightness, which is what video
call software does. The background is allowed to blow out — that is the correct
trade, because the face is the only part the pipeline reconstructs.

Measured exposure response on the Logitech C510 (this room, backlit):

    exposure   frame brightness   fps
       200          17.9         33.1
       500          46.3         22.5
       800          68.2         15.2
      1200          87.9         11.0
      1600         101.5          9.0
      2000         106.9          7.1

Note the cost: brightness is bought with exposure time, and exposure time caps
frame rate. That trade is deliberate and is surfaced to the caller rather than
hidden — a well-exposed face at 11 FPS produces a far better swap than a dark
one at 15 FPS, because input quality bounds output quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ExposureConfig:
    # Mean brightness (0-255) to aim for inside the face box. ~120 is a
    # well-lit face: bright enough for the swap model, short of clipping.
    target_face_brightness: float = 118.0
    # Stop adjusting inside this band so the controller doesn't hunt.
    tolerance: float = 10.0
    min_exposure: int = 150
    max_exposure: int = 1600
    # Fraction of the error corrected per step. Low enough that exposure walks
    # smoothly instead of visibly pumping between frames.
    gain: float = 0.35
    # Only act every Nth processed frame: each change needs a moment to take
    # effect in the sensor, and reacting to every frame just oscillates.
    interval_frames: int = 8


class FaceExposureController:
    """Drives camera exposure from face-region brightness.

    Usage: call `update()` once per processed frame with the frame, the face
    bbox (or None) and the live cv2.VideoCapture. It self-throttles.
    """

    def __init__(self, config: ExposureConfig | None = None) -> None:
        self.config = config or ExposureConfig()
        self._frames_since_update = 0
        self._exposure: int | None = None
        self.last_face_brightness: float | None = None
        self.enabled = True

    @staticmethod
    def face_brightness(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
        """Mean luma of the face box, clipped to the frame.

        Uses the Y channel rather than a mean over BGR: perceived exposure is
        luminance, and averaging raw channels lets a strong colour cast (warm
        indoor light) skew the reading.
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_bgr.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        region = frame_bgr[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean())

    def update(
        self,
        capture,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> bool:
        """Returns True when the exposure was actually changed this call."""
        if not self.enabled or capture is None or bbox is None:
            return False

        self._frames_since_update += 1
        if self._frames_since_update < self.config.interval_frames:
            return False
        self._frames_since_update = 0

        brightness = self.face_brightness(frame_bgr, bbox)
        if brightness is None:
            return False
        self.last_face_brightness = brightness

        error = self.config.target_face_brightness - brightness
        if abs(error) <= self.config.tolerance:
            return False

        if self._exposure is None:
            # Take manual control on first correction. Read the current value
            # so the first step continues from where auto-exposure left off
            # rather than jumping to an arbitrary starting point.
            current = capture.get(cv2.CAP_PROP_EXPOSURE)
            self._exposure = int(current) if current and current > 0 else 500
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual on this driver

        # Brightness responds roughly proportionally to exposure time, so scale
        # the current value by the relative error instead of stepping by a
        # fixed amount — that keeps the step size sensible at both ends of the
        # range.
        ratio = self.config.target_face_brightness / max(brightness, 1.0)
        target = self._exposure * ratio
        new_exposure = int(self._exposure + (target - self._exposure) * self.config.gain)
        new_exposure = max(self.config.min_exposure, min(self.config.max_exposure, new_exposure))

        if new_exposure == self._exposure:
            return False

        self._exposure = new_exposure
        capture.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
        return True

    @property
    def exposure(self) -> int | None:
        return self._exposure

    def release(self, capture) -> None:
        """Hand control back to the camera's own auto-exposure.

        Worth doing on shutdown: exposure is a stateful UVC device setting that
        survives process exit (documented in shared/utils/camera.py), so
        leaving it pinned would affect every other app that opens this camera
        next.
        """
        if capture is not None and self._exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = aperture priority/auto
        self._exposure = None
