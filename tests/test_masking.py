"""Section 24 unit tests for the face-shape masking and colour-transfer logic
(services/face/masking.py). Pure geometry/statistics — no GPU or model needed;
the landmark model itself is exercised in the live pipeline and benchmarks."""
from __future__ import annotations

import cv2
import numpy as np

from services.face.masking import LandmarkMasker, match_color


def _identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _square_contour(cx=128, cy=128, r=60) -> np.ndarray:
    """Stand-in for a face outline: points around a square."""
    return np.array(
        [
            [cx - r, cy - r], [cx, cy - r], [cx + r, cy - r],
            [cx + r, cy], [cx + r, cy + r], [cx, cy + r],
            [cx - r, cy + r], [cx - r, cy],
        ],
        dtype=np.float32,
    )


def test_mask_is_filled_inside_contour_and_empty_far_outside():
    mask = LandmarkMasker.build_mask(_square_contour(), _identity_affine(), size=256)
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert mask[128, 128] > 0.99  # centre of the face is fully included
    assert mask[5, 5] < 0.01  # far corner is fully excluded


def test_mask_edge_is_feathered_not_binary():
    """A hard-edged mask is exactly what makes a swap look pasted on; the
    boundary must be a gradient, so the mask needs intermediate values."""
    mask = LandmarkMasker.build_mask(_square_contour(), _identity_affine(), size=256)
    intermediate = np.count_nonzero((mask > 0.05) & (mask < 0.95))
    assert intermediate > 500


def test_mask_respects_the_affine_transform():
    """Landmarks arrive in frame coordinates and must be mapped into aligned
    space — a translation in the matrix has to move the mask."""
    shift = np.array([[1.0, 0.0, -60.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mask = LandmarkMasker.build_mask(_square_contour(), shift, size=256)
    # Face centred at x=128 shifted left by 60 should now cover x≈68.
    assert mask[128, 68] > 0.9
    assert mask[128, 200] < 0.1


def test_color_match_shifts_source_toward_reference():
    source = np.full((64, 64, 3), 60, dtype=np.uint8)  # dark
    reference = np.full((64, 64, 3), 190, dtype=np.uint8)  # bright
    out = match_color(source, reference)
    # Should land near the reference's brightness, well away from where it started.
    assert abs(float(out.mean()) - 190) < 12
    assert float(out.mean()) > 150


def test_color_match_is_stable_when_source_already_matches():
    image = np.random.default_rng(0).integers(40, 200, (64, 64, 3), dtype=np.uint8)
    out = match_color(image, image)
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 3


def test_color_match_uses_only_masked_pixels_for_statistics():
    """Statistics must come from inside the face, not the background — a bright
    wall behind the person must not drag the correction with it."""
    source = np.full((64, 64, 3), 100, dtype=np.uint8)
    reference = np.full((64, 64, 3), 100, dtype=np.uint8)
    reference[:, 32:] = 250  # bright "background" on the right half

    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:, :32] = 1.0  # only the left half (the "face") counts

    out = match_color(source, reference, mask)
    # Reference *inside the mask* is 100, same as source, so barely any shift.
    assert abs(float(out.mean()) - 100) < 15


def test_color_match_returns_source_unchanged_for_empty_mask():
    source = np.full((32, 32, 3), 123, dtype=np.uint8)
    reference = np.full((32, 32, 3), 20, dtype=np.uint8)
    out = match_color(source, reference, np.zeros((32, 32), dtype=np.float32))
    assert np.array_equal(out, source)


def test_occlusion_and_contour_masks_multiply_to_exclude_both():
    """The engine combines the two masks by multiplication: a pixel is painted
    only if it is BOTH inside the face contour AND not covered by something.
    This guards the combination rule itself, which is easy to get wrong (an
    OR/max would paint over a hand)."""
    contour = np.zeros((64, 64), dtype=np.float32)
    contour[16:48, 16:48] = 1.0  # face region

    visible = np.ones((64, 64), dtype=np.float32)
    visible[30:40, 20:44] = 0.0  # a "hand" across part of the face

    combined = contour * visible

    assert combined[20, 20] == 1.0  # face, unobstructed -> painted
    assert combined[35, 30] == 0.0  # face, but covered -> NOT painted
    assert combined[5, 5] == 0.0  # outside the face -> not painted
