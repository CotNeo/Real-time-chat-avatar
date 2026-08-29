"""Section 24 unit test: the overlay drawer must not crash on the shapes it
will actually receive, with or without a detected face — a plausible bug here
is an off-by-one in unpacking a landmark point or an empty-list edge case."""
from __future__ import annotations

import numpy as np

from services.face.detector import DetectedFace
from services.face.overlay import draw_detection_overlay


def test_overlay_with_no_faces_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_detection_overlay(frame, [], fps=15.0, detect_ms=12.3)
    assert out.shape == (480, 640, 3)


def test_overlay_with_one_face_draws_without_crashing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face = DetectedFace(
        bbox=(100, 100, 300, 300),
        score=0.93,
        landmarks=np.array([[150, 150], [250, 150], [200, 200], [160, 260], [240, 260]]),
    )
    out = draw_detection_overlay(frame, [face], fps=15.0, detect_ms=12.3)
    # The green bbox line should have changed some pixels along its border.
    assert out[100, 200].tolist() == [0, 255, 0]


def test_overlay_with_multiple_faces_does_not_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = [
        DetectedFace(bbox=(10, 10, 60, 60), score=0.7, landmarks=np.zeros((5, 2))),
        DetectedFace(bbox=(400, 300, 500, 420), score=0.85, landmarks=np.zeros((5, 2))),
    ]
    out = draw_detection_overlay(frame, faces, fps=15.0, detect_ms=20.0)
    assert out.shape == (480, 640, 3)
