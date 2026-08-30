"""
Milestone 4 — reference identity pipeline (Section 5):

    Reference Images -> Face Detection -> Quality Validation -> Face Alignment
    -> Identity Encoding -> Embedding Aggregation -> Avatar Identity Session

Reuses the SCRFD detector (Milestone 3) and the same downloaded `buffalo_l`
pack's recognition model (w600k_r50.onnx, ArcFace) — no second model download
needed. Reference images themselves are never retained (Section 5/21): only
the derived embedding and a per-image report survive past this function call.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align

from services.face.detector import DetectedFace, FaceDetector, FaceDetectorError
from shared.schemas.identity import (
    IdentitySession,
    ReferenceImageProblem,
    ReferenceImageResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOGNITION_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "buffalo_l" / "w600k_r50.onnx"
)
GENDERAGE_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "buffalo_l" / "genderage.onnx"
)


# Plain-language verdicts for the UI (Section 14 asks for reference thumbnails,
# and a bare 0-1 number is not something a person can act on when choosing
# which photo to swap out).
def _verdict_for(quality: float) -> str:
    if quality >= 0.85:
        return "Good"
    if quality >= 0.70:
        return "Usable"
    return "Weak — try a sharper, more front-facing photo"


PROBLEM_EXPLANATIONS = {
    "no_face_detected": "No face found in this photo",
    "multiple_faces": "More than one face — use a photo with just the person",
    "resolution_too_low": "Photo is too small — use a larger one",
    "too_blurry": "Too blurry — use a sharper photo",
    "excessive_occlusion": "Face is turned away or partly covered",
}

# Thresholds are deliberately simple, documented heuristics for an MVP, not
# trained classifiers — Section 31: don't over-engineer. Revisit with real
# data if false accepts/rejects turn out to be a problem in practice.
MIN_IMAGE_DIMENSION_PX = 200
BLUR_VARIANCE_THRESHOLD = 60.0  # Laplacian variance on the aligned 112x112 crop
MIN_EYE_DISTANCE_RATIO = 0.15  # of bbox width — a crude occlusion/bad-landmark proxy
MAX_REFERENCE_IMAGES = 5


class IdentityEncoderError(RuntimeError):
    """Actionable message (Section 20)."""


class GenderEstimator:
    """Reports the apparent gender of a reference face.

    Not a control knob — face swap has no separate gender setting, the
    apparent gender rides along with whichever identity is uploaded. This
    exists so the UI can tell the user what their chosen photos will actually
    produce, and warn when a set mixes genders (which averages into a muddled
    identity). Uses `genderage.onnx` from the buffalo_l pack already
    downloaded — no extra model.
    """

    def __init__(self, model_path: Path = GENDERAGE_MODEL_PATH, use_gpu: bool = True) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise IdentityEncoderError(
                f"Gender model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-detection"
            )
        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()
        from insightface.model_zoo import model_zoo

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        model = model_zoo.get_model(str(self.model_path), providers=providers)
        if model is None:
            raise IdentityEncoderError(f"Could not load {self.model_path}.")
        model.prepare(ctx_id=0 if self.use_gpu else -1)
        self._model = model

    def estimate(self, image_bgr, face) -> str | None:
        if self._model is None:
            return None
        from insightface.app.common import Face

        try:
            obj = Face(
                bbox=np.array(face.bbox, dtype=np.float32),
                kps=face.landmarks,
                det_score=face.score,
            )
            gender, _age = self._model.get(image_bgr, obj)
            return "female" if int(gender) == 0 else "male"
        except Exception:  # noqa: BLE001 - advisory only, never fail the upload
            return None


class IdentityEncoder:
    """Wraps the ArcFace recognition model. Kept separate from FaceDetector —
    Milestone 3's detector answers "is there a face and where", this answers
    "what identity does that face encode" — different lifecycles (this one
    isn't needed at all in Mode A's live loop after the identity is loaded)."""

    def __init__(self, model_path: Path = RECOGNITION_MODEL_PATH, use_gpu: bool = True) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None
        self.actual_providers: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise IdentityEncoderError(
                f"Recognition model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-detection\n"
                "(the same buffalo_l pack includes the recognition model)."
            )

        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()

        from insightface.model_zoo import model_zoo

        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        model = model_zoo.get_model(str(self.model_path), providers=requested)
        if model is None:
            raise IdentityEncoderError(
                f"insightface did not recognize {self.model_path} as a usable model."
            )
        model.prepare(ctx_id=0 if self.use_gpu else -1)
        self._model = model
        self.actual_providers = model.session.get_providers()

        if self.use_gpu and not any(
            p in self.actual_providers
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        ):
            raise IdentityEncoderError(
                f"Requested CUDA for identity encoding but got {self.actual_providers}. "
                "See services/face/detector.py's identical check for the known cause "
                "(this project has hit onnxruntime CUDA fallback twice already — "
                "search docs/PROGRESS.md, Milestone 3)."
            )

    def encode(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """`aligned_face_bgr` must already be a 112x112 ArcFace-aligned crop —
        see `align_face` below. Returns a raw (not yet L2-normalized) embedding."""
        if self._model is None:
            raise IdentityEncoderError("IdentityEncoder.encode() called before load().")
        return self._model.get_feat(aligned_face_bgr).flatten()


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Standard ArcFace 5-point alignment (Section 5's "Face Alignment" step)."""
    return face_align.norm_crop(image_bgr, landmark=landmarks, image_size=image_size)


def _blur_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _eye_distance_ratio(face: DetectedFace) -> float:
    left_eye, right_eye = face.landmarks[0], face.landmarks[1]
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    bbox_width = max(1, face.bbox[2] - face.bbox[0])
    return eye_dist / bbox_width


def process_reference_image(
    filename: str,
    image_bgr: np.ndarray | None,
    detector: FaceDetector,
    encoder: IdentityEncoder,
    gender_estimator: "GenderEstimator | None" = None,
) -> ReferenceImageResult:
    """Runs the full per-image pipeline (Section 5's validation checklist).
    Never raises for a bad image — a rejected image is a normal, expected
    result, reported cleanly (Section 20), not an exception."""
    if image_bgr is None:
        return ReferenceImageResult(
            filename=filename, accepted=False,
            problems=[ReferenceImageProblem.NO_FACE_DETECTED],
            verdict=PROBLEM_EXPLANATIONS["no_face_detected"],
        )

    h, w = image_bgr.shape[:2]
    if min(h, w) < MIN_IMAGE_DIMENSION_PX:
        return ReferenceImageResult(
            filename=filename, accepted=False,
            problems=[ReferenceImageProblem.RESOLUTION_TOO_LOW],
            verdict=PROBLEM_EXPLANATIONS["resolution_too_low"],
        )

    faces, _ = detector.detect(image_bgr)
    if len(faces) == 0:
        return ReferenceImageResult(
            filename=filename, accepted=False,
            problems=[ReferenceImageProblem.NO_FACE_DETECTED],
            verdict=PROBLEM_EXPLANATIONS["no_face_detected"],
        )
    if len(faces) > 1:
        return ReferenceImageResult(
            filename=filename, accepted=False,
            problems=[ReferenceImageProblem.MULTIPLE_FACES],
            verdict=PROBLEM_EXPLANATIONS["multiple_faces"],
        )

    face = faces[0]
    problems = []
    if _eye_distance_ratio(face) < MIN_EYE_DISTANCE_RATIO:
        # Heuristic, not a trained occlusion classifier — see module docstring.
        problems.append(ReferenceImageProblem.EXCESSIVE_OCCLUSION)

    aligned = align_face(image_bgr, face.landmarks)
    blur = _blur_variance(aligned)
    if blur < BLUR_VARIANCE_THRESHOLD:
        problems.append(ReferenceImageProblem.TOO_BLURRY)

    if problems:
        return ReferenceImageResult(
            filename=filename,
            accepted=False,
            problems=problems,
            verdict=PROBLEM_EXPLANATIONS.get(problems[0].value, "Unusable"),
        )

    embedding = encoder.encode(aligned)
    normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
    quality_score = min(1.0, face.score)
    gender = (
        gender_estimator.estimate(image_bgr, face) if gender_estimator is not None else None
    )
    return ReferenceImageResult(
        filename=filename,
        accepted=True,
        quality_score=quality_score,
        embedding=normalized,
        verdict=_verdict_for(quality_score),
        gender=gender,
    )


def build_identity_session(
    images: list[tuple[str, np.ndarray | None]],
    detector: FaceDetector,
    encoder: IdentityEncoder,
    gender_estimator: "GenderEstimator | None" = None,
) -> IdentitySession:
    """Section 5's full pipeline entrypoint: Upload -> Process -> Session
    Identity. `images` is (filename, decoded BGR array or None if decode
    failed) — decode failures are the caller's concern (Milestone 14's upload
    endpoint), reported here as NO_FACE_DETECTED-equivalent rejections."""
    accepted: list[ReferenceImageResult] = []
    rejected: list[ReferenceImageResult] = []

    for filename, image_bgr in images[:MAX_REFERENCE_IMAGES]:
        result = process_reference_image(
            filename, image_bgr, detector, encoder, gender_estimator
        )
        (accepted if result.accepted else rejected).append(result)

    aggregated = None
    if accepted:
        stacked = np.stack([r.embedding for r in accepted])
        mean_embedding = stacked.mean(axis=0)
        aggregated = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

    return IdentitySession(
        session_id=str(uuid.uuid4()),
        accepted_images=accepted,
        rejected_images=rejected,
        aggregated_embedding=aggregated,
        created_at=time.time(),
    )
