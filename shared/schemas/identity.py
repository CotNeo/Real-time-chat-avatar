"""
Data contracts for the reference-identity pipeline (Section 5) and voice
profiles (Section 7). Kept separate from shared/schemas/config.py: config is
static/loaded-once, these are runtime session data that flows between services.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReferenceImageProblem(str, Enum):
    """Every way an uploaded reference image can fail validation (Section 5).
    Reported to the user by name, never as a bare rejection."""

    NO_FACE_DETECTED = "no_face_detected"
    MULTIPLE_FACES = "multiple_faces"
    RESOLUTION_TOO_LOW = "resolution_too_low"
    TOO_BLURRY = "too_blurry"
    EXCESSIVE_OCCLUSION = "excessive_occlusion"


@dataclass
class ReferenceImageResult:
    """Outcome of validating+processing one uploaded reference image."""

    filename: str
    accepted: bool
    problems: list[ReferenceImageProblem] = field(default_factory=list)
    quality_score: float | None = None  # 0-1, only set if accepted
    embedding: object | None = None  # np.ndarray once computed; typed loosely to
    # avoid importing numpy into a schemas module that other layers may want to
    # keep dependency-light.
    # Plain-language outcome for the UI. A raw 0-1 score tells someone
    # choosing photos nothing actionable; "Good" / "Usable" / a named problem
    # does.
    verdict: str | None = None
    # Apparent gender of the reference face ("female"/"male"/None). Reported
    # because averaging embeddings across genders produces a muddled identity
    # — the UI warns when a set is mixed.
    gender: str | None = None


@dataclass
class IdentitySession:
    """The 'Avatar Identity Session' produced at the end of the reference
    pipeline (Section 5: Upload -> Process -> Session Identity -> Runtime ->
    Cleanup). Reference images themselves are NOT retained here by design
    (Section 5/21) — only the derived embedding and enough metadata to explain
    to the user what was used."""

    session_id: str
    accepted_images: list[ReferenceImageResult]
    rejected_images: list[ReferenceImageResult]
    aggregated_embedding: object | None  # np.ndarray, mean/weighted of accepted embeddings
    created_at: float  # time.time()

    @property
    def is_usable(self) -> bool:
        return self.aggregated_embedding is not None and len(self.accepted_images) > 0

    @property
    def gender_summary(self) -> str | None:
        """"female" / "male" when the accepted references agree, "mixed" when
        they don't. A mixed set averages incompatible faces together and the
        result reads as neither — worth telling the user rather than letting
        them wonder why the output looks off."""
        genders = {r.gender for r in self.accepted_images if r.gender}
        if not genders:
            return None
        if len(genders) > 1:
            return "mixed"
        return genders.pop()


@dataclass
class VoiceProfile:
    id: str
    display_name: str
    engine: str  # e.g. "rvc", "seed-vc" — which VoiceEngine backend serves it
    model_path: str
    sample_rate: int = 16000
