"""Section 24 unit tests for FaceSwapEngine's error handling and tracking-
interval bookkeeping — the parts that don't need the real ~265MB model or a
GPU. The model's actual output quality (and the real, load-bearing low-light
finding) is covered by scripts/benchmark/benchmark_face_swap.py against real
hardware — see docs/PROGRESS.md, Milestone 5."""
from __future__ import annotations

import numpy as np
import pytest

from services.face.detector import DetectedFace
from services.face.engine import FaceEngineError
from services.face.swapper import FaceSwapEngine
from shared.schemas.identity import IdentitySession


class FakeDetector:
    """Counts calls so tests can assert the detection-interval skip logic
    actually skips, rather than just trusting it does."""

    def __init__(self, faces):
        self.faces = faces
        self.call_count = 0

    def detect(self, image):
        self.call_count += 1
        return self.faces, 1.0


def _make_face():
    return DetectedFace(
        bbox=(10, 10, 100, 100), score=0.9,
        landmarks=np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32),
    )


def test_process_frame_before_load_raises():
    engine = FaceSwapEngine(detector=FakeDetector([]))
    with pytest.raises(FaceEngineError, match="before load"):
        engine.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))


def test_process_frame_without_identity_raises():
    engine = FaceSwapEngine(detector=FakeDetector([]))
    engine._swapper = object()  # bypass load() — only testing the identity guard
    with pytest.raises(FaceEngineError, match="No identity loaded"):
        engine.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))


def test_load_identity_rejects_unusable_session():
    engine = FaceSwapEngine(detector=FakeDetector([]))
    unusable = IdentitySession(
        session_id="s", accepted_images=[], rejected_images=[],
        aggregated_embedding=None, created_at=0.0,
    )
    with pytest.raises(FaceEngineError, match="zero accepted"):
        engine.load_identity(unusable)


def test_process_frame_with_no_face_passes_through_original_image():
    detector = FakeDetector([])
    engine = FaceSwapEngine(detector=detector)
    engine._swapper = object()  # process_frame shouldn't even reach the swapper
    engine._source_embedding = np.ones(512, dtype=np.float32)

    frame = np.full((100, 100, 3), 77, dtype=np.uint8)
    result = engine.process_frame(frame)

    assert not result.face_detected
    assert result.bbox is None
    assert np.array_equal(result.output_image, frame)


def test_detection_interval_skips_redetection_on_subsequent_frames():
    face = _make_face()
    detector = FakeDetector([face])
    engine = FaceSwapEngine(detector=detector)
    engine.detection_interval = 3

    class RecordingSwapper:
        def get(self, img, target_face, source_face, paste_back=True):
            return img

    engine._swapper = RecordingSwapper()
    engine._source_embedding = np.ones(512, dtype=np.float32)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ran_detection = []
    for _ in range(6):
        result = engine.process_frame(frame)
        ran_detection.append(result.detection_ran_this_frame)

    # With interval=3: detect, skip, skip, detect, skip, skip
    assert ran_detection == [True, False, False, True, False, False]
    assert detector.call_count == 2


def test_reset_clears_tracking_state_forcing_redetection():
    face = _make_face()
    detector = FakeDetector([face])
    engine = FaceSwapEngine(detector=detector)
    engine.detection_interval = 5

    class PassthroughSwapper:
        def get(self, img, target_face, source_face, paste_back=True):
            return img

    engine._swapper = PassthroughSwapper()
    engine._source_embedding = np.ones(512, dtype=np.float32)

    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    engine.process_frame(frame)  # detects
    engine.process_frame(frame)  # would normally skip
    assert detector.call_count == 1

    engine.reset()
    result = engine.process_frame(frame)
    assert result.detection_ran_this_frame is True
    assert detector.call_count == 2
