"""Section 24 unit tests for face-metered exposure control. The controller
logic is pure arithmetic plus a few cv2 setter calls, so it is tested against a
fake capture rather than real hardware; the exposure/brightness/fps response of
the actual camera is recorded in docs/PROGRESS.md."""
from __future__ import annotations

import cv2
import numpy as np

from shared.utils.exposure import ExposureConfig, FaceExposureController


class FakeCapture:
    """Records cv2 property writes so tests can assert on them."""

    def __init__(self, exposure=500.0):
        self.props = {cv2.CAP_PROP_EXPOSURE: exposure}
        self.writes = []

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        self.props[prop] = value
        self.writes.append((prop, value))
        return True


def _frame_with_face(face_value: int, bg_value: int, size=200):
    """A bright background with a darker face box — the backlit case."""
    frame = np.full((size, size, 3), bg_value, dtype=np.uint8)
    frame[50:150, 50:150] = face_value
    return frame


BBOX = (50, 50, 150, 150)


def _settled(controller, capture, frame, bbox, steps=40):
    """Run enough frames for the interval throttle to allow several updates."""
    for _ in range(steps):
        controller.update(capture, frame, bbox)


def test_face_brightness_measures_the_face_not_the_background():
    frame = _frame_with_face(face_value=40, bg_value=240)
    measured = FaceExposureController.face_brightness(frame, BBOX)
    # Must reflect the dark face (~40), not the bright surroundings.
    assert measured is not None
    assert abs(measured - 40) < 3


def test_dark_face_raises_exposure():
    controller = FaceExposureController()
    capture = FakeCapture(exposure=400)
    frame = _frame_with_face(face_value=35, bg_value=240)  # backlit, face dark
    _settled(controller, capture, frame, BBOX)
    assert controller.exposure is not None
    assert controller.exposure > 400


def test_bright_face_lowers_exposure():
    controller = FaceExposureController()
    capture = FakeCapture(exposure=900)
    frame = _frame_with_face(face_value=220, bg_value=220)  # face blown out
    _settled(controller, capture, frame, BBOX)
    assert controller.exposure is not None
    assert controller.exposure < 900


def test_face_already_at_target_leaves_exposure_alone():
    config = ExposureConfig(target_face_brightness=118.0, tolerance=10.0)
    controller = FaceExposureController(config)
    capture = FakeCapture(exposure=600)
    frame = _frame_with_face(face_value=118, bg_value=118)
    _settled(controller, capture, frame, BBOX)
    # Never took manual control, never wrote anything.
    assert controller.exposure is None
    assert capture.writes == []


def test_exposure_is_clamped_to_configured_range():
    config = ExposureConfig(min_exposure=200, max_exposure=900)
    controller = FaceExposureController(config)
    capture = FakeCapture(exposure=800)
    frame = _frame_with_face(face_value=5, bg_value=250)  # pathologically dark face
    _settled(controller, capture, frame, BBOX, steps=200)
    assert controller.exposure <= 900


def test_no_bbox_means_no_adjustment():
    controller = FaceExposureController()
    capture = FakeCapture()
    frame = _frame_with_face(face_value=20, bg_value=240)
    for _ in range(40):
        controller.update(capture, frame, None)
    assert capture.writes == []


def test_updates_are_throttled_between_frames():
    """Each exposure change needs time to take effect; reacting every frame
    would oscillate rather than settle."""
    config = ExposureConfig(interval_frames=8)
    controller = FaceExposureController(config)
    capture = FakeCapture(exposure=400)
    frame = _frame_with_face(face_value=30, bg_value=240)
    for _ in range(7):
        assert controller.update(capture, frame, BBOX) is False
    assert controller.update(capture, frame, BBOX) is True


def test_release_restores_automatic_exposure():
    """Exposure survives process exit on UVC devices, so shutdown must hand it
    back or every other app inherits our manual value."""
    controller = FaceExposureController()
    capture = FakeCapture(exposure=400)
    frame = _frame_with_face(face_value=30, bg_value=240)
    _settled(controller, capture, frame, BBOX)
    controller.release(capture)
    assert capture.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 3
    assert controller.exposure is None
