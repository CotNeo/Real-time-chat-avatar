"""Section 24 unit tests for the parts of the face detector that don't need
real hardware/a downloaded model. The hardware-dependent path (does it
actually detect a face, does it actually run on CUDA) is exercised by
scripts/benchmark/benchmark_face_detection.py against the real camera and
model instead — see docs/PROGRESS.md, Milestone 3, for those results."""
from __future__ import annotations

from pathlib import Path

import pytest

from services.face.detector import FaceDetector, FaceDetectorError


def test_load_raises_descriptive_error_when_model_missing(tmp_path):
    missing_model = tmp_path / "does-not-exist.onnx"
    detector = FaceDetector(model_path=missing_model)
    with pytest.raises(FaceDetectorError, match="not found"):
        detector.load()


def test_detect_before_load_raises_descriptive_error():
    detector = FaceDetector()
    with pytest.raises(FaceDetectorError, match="before load"):
        detector.detect(None)
