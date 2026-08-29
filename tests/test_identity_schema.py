"""Section 24 unit tests: reference-identity validation contracts."""
from __future__ import annotations

import time

from shared.schemas.identity import (
    IdentitySession,
    ReferenceImageProblem,
    ReferenceImageResult,
)


def test_session_with_no_accepted_images_is_not_usable():
    session = IdentitySession(
        session_id="s1",
        accepted_images=[],
        rejected_images=[
            ReferenceImageResult(
                filename="blurry.jpg",
                accepted=False,
                problems=[ReferenceImageProblem.TOO_BLURRY],
            )
        ],
        aggregated_embedding=None,
        created_at=time.time(),
    )
    assert session.is_usable is False


def test_session_with_accepted_image_and_embedding_is_usable():
    session = IdentitySession(
        session_id="s2",
        accepted_images=[
            ReferenceImageResult(filename="front.jpg", accepted=True, quality_score=0.9)
        ],
        rejected_images=[],
        aggregated_embedding=object(),  # stand-in for a real np.ndarray
        created_at=time.time(),
    )
    assert session.is_usable is True


def test_session_with_embedding_but_zero_accepted_images_is_not_usable():
    """Guards against a bug where an embedding survives even though every
    image that produced it was later marked rejected (e.g. by a stricter
    downstream quality re-check)."""
    session = IdentitySession(
        session_id="s3",
        accepted_images=[],
        rejected_images=[
            ReferenceImageResult(
                filename="two_faces.jpg",
                accepted=False,
                problems=[ReferenceImageProblem.MULTIPLE_FACES],
            )
        ],
        aggregated_embedding=object(),
        created_at=time.time(),
    )
    assert session.is_usable is False
