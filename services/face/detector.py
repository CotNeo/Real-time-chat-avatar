"""
Milestone 3 — SCRFD face detection (via InsightFace's model_zoo, loading only
the detection ONNX file, not the full FaceAnalysis app with its recognition/
landmark/age/gender models — Section 3: "select the smallest practical
combination", and Milestone 4's identity/recognition step is a separate
concern that will load its own model from this same downloaded pack later).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "buffalo_l" / "det_10g.onnx"
)


class FaceDetectorError(RuntimeError):
    """Actionable message (Section 20) — never a bare exception three layers
    inside onnxruntime/insightface."""


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in source-image pixels
    score: float  # detection confidence, 0-1
    landmarks: np.ndarray  # 5x2 float array: left eye, right eye, nose, mouth L, mouth R


class FaceDetector:
    """Thin, swappable wrapper. FaceEngine implementations (Milestone 5+) use
    this internally rather than talking to insightface directly, so the
    detector backend can change without touching the engine's blending logic."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        det_thresh: float = 0.5,
        input_size: tuple[int, int] = (640, 640),
        use_gpu: bool = True,
    ) -> None:
        self.model_path = model_path
        self.det_thresh = det_thresh
        self.input_size = input_size
        self.use_gpu = use_gpu
        self._model = None
        self._actual_providers: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise FaceDetectorError(
                f"Face detection model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-detection"
            )

        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()

        from insightface.model_zoo import model_zoo

        requested_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        model = model_zoo.get_model(str(self.model_path), providers=requested_providers)
        if model is None:
            raise FaceDetectorError(
                f"insightface did not recognize {self.model_path} as a usable model. "
                "The download may be corrupt — re-run "
                "`python scripts/models.py install face-detection`."
            )

        model.prepare(ctx_id=0 if self.use_gpu else -1, input_size=self.input_size, det_thresh=self.det_thresh)
        self._model = model

        # Verify — don't trust — that CUDA is actually in use, the same
        # discipline as scripts/setup/verify_cuda.py. insightface has already
        # bitten this project once with a silent CPU fallback (see
        # scripts/setup/install_face_deps.sh's writeup); check explicitly here
        # too rather than assuming the fix holds forever.
        self._actual_providers = model.session.get_providers()
        if self.use_gpu and "CUDAExecutionProvider" not in self._actual_providers:
            raise FaceDetectorError(
                f"Requested CUDA for face detection but the session is actually using "
                f"{self._actual_providers}.\n"
                "This project has hit this exact failure mode before from a dependency "
                "conflict (see requirements/face.txt) — check: "
                "python -c \"import onnxruntime as ort; print(ort.get_available_providers())\" "
                "and confirm CUDAExecutionProvider is listed; if not, re-run "
                "scripts/setup/install_face_deps.sh."
            )

    @property
    def actual_providers(self) -> list[str]:
        return self._actual_providers

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[DetectedFace], float]:
        """Returns (faces sorted by confidence desc, inference_ms)."""
        if self._model is None:
            raise FaceDetectorError("FaceDetector.detect() called before load().")

        start = time.perf_counter()
        bboxes, kpss = self._model.detect(frame_bgr, input_size=self.input_size)
        elapsed_ms = (time.perf_counter() - start) * 1000

        faces = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i]
            landmarks = kpss[i] if kpss is not None else np.zeros((5, 2), dtype=np.float32)
            faces.append(
                DetectedFace(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(score),
                    landmarks=landmarks,
                )
            )
        return faces, elapsed_ms
