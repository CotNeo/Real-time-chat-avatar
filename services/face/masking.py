"""
Face-shaped masking and color matching — the two things that separate a
"pasted rectangle" look from a swap that reads as real.

Both address problems visible in the Milestone 5 output:

1. **Shape.** The swap/enhance paste-back used the full aligned square as its
   mask, eroded and blurred. That covers forehead, hair edges and background
   corners that are not the person's face, so the boundary shows. This module
   builds the mask from the 106-point face contour instead, so the composite
   follows the real jaw/cheek/brow outline.

2. **Colour.** The generated face carries the reference identity's skin tone
   and the model's own lighting, which rarely matches the user's actual room
   light. Even a perfect shape leaves a visible patch if the tone is off —
   this is usually what makes a swap obvious at the jawline and neck. The
   colour transfer here rescales the swapped face's per-channel statistics to
   the original face's, inside the mask only.

The landmark model (`2d106det.onnx`) ships in the same `buffalo_l` pack
already downloaded for detection and recognition — no extra model download,
and measured at 1.8 ms/frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDMARK_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "buffalo_l" / "2d106det.onnx"
)
DEFAULT_OCCLUDER_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "masking" / "dfl_xseg.onnx"
)
OCCLUDER_SIZE = 256


class FaceMaskerError(RuntimeError):
    """Actionable message (Section 20)."""


class LandmarkMasker:
    """Produces a face-shaped soft mask in the aligned working space."""

    def __init__(
        self, model_path: Path = DEFAULT_LANDMARK_MODEL_PATH, use_gpu: bool = True
    ) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._model = None
        self.actual_providers: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise FaceMaskerError(
                f"Landmark model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-detection\n"
                "(the buffalo_l pack includes the 106-point landmark model)."
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
            raise FaceMaskerError(f"Could not load landmark model {self.model_path}.")
        model.prepare(ctx_id=0 if self.use_gpu else -1)
        self._model = model
        self.actual_providers = model.session.get_providers()

    def landmarks_106(
        self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], kps: np.ndarray
    ) -> np.ndarray | None:
        """106 dense landmarks in frame coordinates, or None if unavailable."""
        if self._model is None:
            raise FaceMaskerError("LandmarkMasker.landmarks_106() called before load().")
        from insightface.app.common import Face

        face = Face(
            bbox=np.array(bbox, dtype=np.float32),
            kps=kps,
            det_score=1.0,
        )
        try:
            return self._model.get(frame_bgr, face)
        except Exception:  # noqa: BLE001 - a mask failure must not kill the frame
            return None

    @staticmethod
    def build_mask(
        landmarks_frame: np.ndarray,
        affine_matrix: np.ndarray,
        size: int,
        feather: float = 0.06,
        expand: float = 1.04,
    ) -> np.ndarray:
        """Convex hull of the face contour, warped into aligned space, softened.

        `expand` grows the hull slightly around its centroid so the mask edge
        sits just outside the facial boundary rather than clipping it;
        `feather` is the blur radius as a fraction of `size`, which is what
        makes the transition invisible rather than a hard cut.
        """
        points = np.hstack(
            [landmarks_frame, np.ones((landmarks_frame.shape[0], 1), dtype=np.float32)]
        )
        aligned_points = points @ affine_matrix.T  # (N, 2) in aligned space

        centroid = aligned_points.mean(axis=0)
        aligned_points = centroid + (aligned_points - centroid) * expand

        hull = cv2.convexHull(aligned_points.astype(np.int32))
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        blur = int(size * feather) | 1  # force odd
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        return mask.astype(np.float32) / 255.0


class OcclusionMasker:
    """Excludes anything in front of the face — a hand, a mug, a microphone —
    from the region the swap is painted into.

    Without this, the swap covers whatever crosses the face, so raising a hand
    smears a generated face over it. This was the concrete failure the user
    reported, and the landmark contour mask cannot fix it: landmarks describe
    where the face *is*, not what is in front of it.

    Model: DeepFaceLab's XSeg (`dfl_xseg.onnx`, 70 MB). Benchmarked against
    BiSeNet face parsing on the same synthetically-occluded frame — XSeg was
    both faster (44.5 ms vs 55.6 ms) and far better at the actual job:
    occluded-region coverage dropped 0.128 for XSeg versus 0.009 for BiSeNet,
    which mostly classified a skin-toned occluder as skin. Measured at 45.8 ms
    on CUDA; note some ConvTranspose nodes fall back to CPU (asymmetric
    padding is unsupported on the CUDA EP), yet mixed CUDA/CPU is still far
    ahead of CPU-only at 107.7 ms — so do not "fix" the warnings by forcing
    CPU.
    """

    def __init__(
        self, model_path: Path = DEFAULT_OCCLUDER_MODEL_PATH, use_gpu: bool = True
    ) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self.actual_providers: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise FaceMaskerError(
                f"Occlusion model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-occluder"
            )

        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()

        import onnxruntime

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        from shared.utils.onnx_errors import describe_load_failure

        try:
            session = onnxruntime.InferenceSession(
                str(self.model_path), providers=providers
            )
        except Exception as e:  # noqa: BLE001 - re-raised with an actionable message
            raise FaceMaskerError(describe_load_failure("the occlusion model", e)) from e
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self.actual_providers = session.get_providers()

    def warm_up(self) -> None:
        if self._session is None:
            raise FaceMaskerError("OcclusionMasker.warm_up() called before load().")
        dummy = np.zeros((1, OCCLUDER_SIZE, OCCLUDER_SIZE, 3), dtype=np.float32)
        self._session.run([self._output_name], {self._input_name: dummy})

    def mask_for(self, aligned_face_bgr: np.ndarray, size: int) -> np.ndarray | None:
        """1.0 where the face is genuinely visible, 0.0 where something covers
        it. `aligned_face_bgr` is the aligned crop; the result is resized to
        `size` so it can be multiplied with the contour mask.

        NOTE the input layout: this model is NHWC (channels last), unlike
        every other ONNX model in this project — feeding it NCHW silently
        produces a garbage mask rather than an error.
        """
        if self._session is None:
            raise FaceMaskerError("OcclusionMasker.mask_for() called before load().")
        if aligned_face_bgr.shape[0] != OCCLUDER_SIZE:
            aligned_face_bgr = cv2.resize(
                aligned_face_bgr, (OCCLUDER_SIZE, OCCLUDER_SIZE), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            out = self._session.run(
                [self._output_name], {self._input_name: rgb[np.newaxis, ...]}
            )[0]
        except Exception:  # noqa: BLE001 - a mask failure must not drop the frame
            return None
        mask = np.clip(out[0, :, :, 0], 0.0, 1.0)
        if mask.shape[0] != size:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        # Soften the boundary so an occluder edge doesn't cut a hard line
        # through the composited face.
        blur = max(3, int(size * 0.02)) | 1
        return cv2.GaussianBlur(mask, (blur, blur), 0)


def match_color(
    source_face: np.ndarray, reference_face: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Rescale `source_face`'s colour statistics to `reference_face`'s.

    Operates in LAB, which separates luminance from chroma so the correction
    tracks the room's lighting without dragging hue along with it. Statistics
    are taken inside `mask` only — including background pixels would bias the
    correction toward the wall behind the person rather than their skin.
    """
    source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB).astype(np.float32)

    # The correction is six numbers (per-channel mean and std for each image),
    # and those converge long before full resolution — computing them on a
    # thumbnail is visually identical and much cheaper. Measured: this took the
    # colour-transfer stage from ~25 ms to a few ms at 512x512.
    stats_size = 128
    small_source = cv2.resize(
        source_face, (stats_size, stats_size), interpolation=cv2.INTER_AREA
    )
    small_reference = cv2.resize(
        reference_face, (stats_size, stats_size), interpolation=cv2.INTER_AREA
    )
    small_source_lab = cv2.cvtColor(small_source, cv2.COLOR_BGR2LAB).astype(np.float32)
    small_reference_lab = cv2.cvtColor(
        small_reference, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    if mask is not None:
        small_mask = cv2.resize(
            mask, (stats_size, stats_size), interpolation=cv2.INTER_AREA
        )
        weights = small_mask.reshape(-1, 1)
        total = weights.sum()
        if total < 1e-3:
            return source_face
        source_flat = small_source_lab.reshape(-1, 3)
        reference_flat = small_reference_lab.reshape(-1, 3)
        source_mean = (source_flat * weights).sum(axis=0) / total
        reference_mean = (reference_flat * weights).sum(axis=0) / total
        source_std = np.sqrt(
            (((source_flat - source_mean) ** 2) * weights).sum(axis=0) / total
        )
        reference_std = np.sqrt(
            (((reference_flat - reference_mean) ** 2) * weights).sum(axis=0) / total
        )
    else:
        source_mean = small_source_lab.mean(axis=(0, 1))
        source_std = small_source_lab.std(axis=(0, 1))
        reference_mean = small_reference_lab.mean(axis=(0, 1))
        reference_std = small_reference_lab.std(axis=(0, 1))

    source_std = np.maximum(source_std, 1e-3)
    corrected = (source_lab - source_mean) * (reference_std / source_std) + reference_mean
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
