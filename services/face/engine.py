"""
FaceEngine interface (Section 17).

Any face pipeline implementation — Mode A real-time swap (Milestone 5, first),
or Mode B reenactment (later) — implements this. The rest of the app (API,
runtime loop, benchmarking) depends only on this interface, never on a
specific model, so swapping the backing model is a one-file change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from shared.schemas.identity import IdentitySession


@dataclass
class FaceFrameResult:
    """Everything downstream (rendering, metrics, virtual camera output) needs
    from processing one frame. `output_image` is always populated (falls back
    to the untouched input frame when no face is found), so callers never have
    to special-case a None frame — see Section 20: no face detected must
    degrade gracefully, not crash the pipeline."""

    output_image: np.ndarray
    face_detected: bool
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) in source-frame pixels
    landmarks: np.ndarray | None
    detection_ran_this_frame: bool  # False when a tracked/interpolated box was used
    timings_ms: dict[str, float]  # e.g. {"detect": 4.2, "inference": 18.1, "blend": 2.0}


class FaceEngineError(RuntimeError):
    """Raised with an actionable message (Section 20) — e.g. GPU OOM, model
    missing, CUDA init failure. Never a bare exception."""


class FaceEngine(ABC):
    @abstractmethod
    def load_identity(self, identity: IdentitySession) -> None:
        """Bind this engine to a reference identity built by the identity
        pipeline (Section 5). Must raise FaceEngineError with a clear message
        if `identity.is_usable` is False rather than silently no-op'ing."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> FaceFrameResult:
        """Process one BGR frame from the live camera. Must be safe to call at
        the camera's native frame rate — internal detection-interval skipping
        (Section 6) is this method's responsibility, not the caller's."""

    @abstractmethod
    def reset(self) -> None:
        """Clear tracking state (previous bbox/landmarks/pose) e.g. after a
        camera disconnect/reconnect or when starting a new session, without
        needing to reload the identity."""

    @abstractmethod
    def warm_up(self) -> None:
        """Run one dummy inference at load time so the first *real* frame
        doesn't pay CUDA context / cuDNN algo-selection latency (this
        materially affects the P95 latency numbers Section 8 asks us to
        measure honestly — the first inference after loading a model is not
        representative of steady-state performance)."""
