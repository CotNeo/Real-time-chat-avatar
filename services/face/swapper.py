"""
Milestone 5 — Mode A real-time face swap (Section 4):

    Reference Identity + Live Camera Face -> Identity Transfer -> Blending

Uses `inswapper_128` (128x128 input/output, ~277 MB fp16 ONNX) — the de facto
standard lightweight real-time face-swap model (the same one roop/ReActor/
FaceFusion build on), chosen specifically for RTX-2060-class real-time budgets
per Section 4: prioritize speed/stability/identity similarity over perfect
image quality, and don't start with heavy diffusion models.

Provenance note (Section 23): InsightFace's own team no longer officially
hosts or maintains this model (they now point users at their commercial
product instead) — there is no clean official license file the way
`buffalo_l` has one. This file was downloaded from `Gourieff/ReActor` on
Hugging Face, a long-standing, widely-used community mirror (2+ years old,
scanned "Safe" by Hugging Face) — not an official, clearly-licensed source.
Documented honestly in `models/registry.yaml`. This project uses it only for
strictly local, single-user, consensual avatar experimentation on the
operator's own likeness (Section 21) — never redistributed.

Blending/masking logic is NOT reimplemented here — it's delicate, well-tested
code (soft-edged mask, eroded/blurred seams, inverse-affine paste-back) that
every major face-swap tool shares verbatim from InsightFace's own
`model_zoo.inswapper.INSwapper.get()`. Reusing it beats a naive rectangular
paste or a fresh (and likely buggier) reimplementation — Section 3's "select
the smallest practical combination" cuts both ways: don't add unnecessary
packages, but don't discard a well-tested one already in the dependency tree
either.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from services.face.detector import FaceDetector, FaceDetectorError
from services.face.engine import FaceEngine, FaceEngineError, FaceFrameResult
from shared.schemas.identity import IdentitySession

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWAP_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "inswapper" / "inswapper_128_fp16.onnx"
)


class FaceSwapEngine(FaceEngine):
    """Mode A implementation of the FaceEngine interface. Detection is
    delegated to a shared FaceDetector instance (Milestone 3) rather than
    owning its own — the live API already has one loaded, no reason to load
    SCRFD twice and double its VRAM footprint."""

    def __init__(
        self,
        detector: FaceDetector,
        model_path: Path = DEFAULT_SWAP_MODEL_PATH,
        use_gpu: bool = True,
    ) -> None:
        self._detector = detector
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._swapper = None
        self._source_embedding: np.ndarray | None = None
        self.actual_providers: list[str] = []
        # Tracking state (Section 6): reuse the last bbox/landmarks for a few
        # frames instead of re-running detection on every single one.
        self._last_faces = []
        self._frames_since_detect = 0
        self.detection_interval = 3

    def load(self) -> None:
        if not self.model_path.exists():
            raise FaceEngineError(
                f"Face swap model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-swap"
            )

        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()

        import onnxruntime
        from insightface.model_zoo.inswapper import INSwapper

        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        session = onnxruntime.InferenceSession(str(self.model_path), providers=requested)
        self._swapper = INSwapper(model_file=str(self.model_path), session=session)
        self.actual_providers = session.get_providers()

        if self.use_gpu and "CUDAExecutionProvider" not in self.actual_providers:
            raise FaceEngineError(
                f"Requested CUDA for face swap but got {self.actual_providers}. "
                "See services/face/detector.py's identical check for the known "
                "cause (docs/PROGRESS.md, Milestone 3)."
            )

    def load_identity(self, identity: IdentitySession) -> None:
        if not identity.is_usable:
            raise FaceEngineError(
                "Cannot load an identity session with zero accepted reference "
                "images — upload at least one valid reference photo via "
                "POST /identity first."
            )
        self._source_embedding = identity.aggregated_embedding
        self.reset()

    def warm_up(self) -> None:
        if self._swapper is None:
            raise FaceEngineError("FaceSwapEngine.warm_up() called before load().")
        # Bug found and fixed during Milestone 5 testing (docs/PROGRESS.md): an
        # all-zeros dummy embedding projects through the model's internal emap
        # matrix to an all-zero latent, whose norm is exactly 0 — the
        # subsequent `latent /= norm` inside insightface's own INSwapper.get()
        # then divides 0/0, raising "invalid value encountered in divide" and
        # producing a NaN-filled warm-up output. Harmless in isolation (this
        # call's result is discarded), but confusing to debug later, and it's
        # trivially avoided with any non-degenerate unit vector instead.
        dummy_target = SimpleNamespace(kps=_dummy_landmarks())
        dummy_vector = np.ones(512, dtype=np.float32)
        dummy_vector /= np.linalg.norm(dummy_vector)
        dummy_source = SimpleNamespace(normed_embedding=dummy_vector)
        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        self._swapper.get(dummy_frame, dummy_target, dummy_source, paste_back=True)

    def reset(self) -> None:
        self._last_faces = []
        self._frames_since_detect = 0

    def process_frame(self, frame: np.ndarray) -> FaceFrameResult:
        if self._swapper is None:
            raise FaceEngineError("FaceSwapEngine.process_frame() called before load().")
        if self._source_embedding is None:
            raise FaceEngineError(
                "No identity loaded — call load_identity() before process_frame()."
            )

        timings: dict[str, float] = {}
        detect_start = time.perf_counter()
        detection_ran = self._frames_since_detect == 0
        if detection_ran:
            faces, _ = self._detector.detect(frame)
            self._last_faces = faces
            self._frames_since_detect = 1
        else:
            faces = self._last_faces
            self._frames_since_detect = (self._frames_since_detect + 1) % max(
                1, self.detection_interval
            )
        timings["detect"] = (time.perf_counter() - detect_start) * 1000

        if not faces:
            return FaceFrameResult(
                output_image=frame,
                face_detected=False,
                bbox=None,
                landmarks=None,
                detection_ran_this_frame=detection_ran,
                timings_ms=timings,
            )

        face = faces[0]  # largest/highest-confidence face only — single-user product
        infer_start = time.perf_counter()
        target_face = SimpleNamespace(kps=face.landmarks)
        source_face = SimpleNamespace(normed_embedding=self._source_embedding)
        output_image = self._swapper.get(frame, target_face, source_face, paste_back=True)
        timings["inference"] = (time.perf_counter() - infer_start) * 1000

        return FaceFrameResult(
            output_image=output_image,
            face_detected=True,
            bbox=face.bbox,
            landmarks=face.landmarks,
            detection_ran_this_frame=detection_ran,
            timings_ms=timings,
        )


def _dummy_landmarks() -> np.ndarray:
    return np.array(
        [[96, 96], [160, 96], [128, 128], [100, 160], [156, 160]], dtype=np.float32
    )
