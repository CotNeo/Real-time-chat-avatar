"""
Face enhancement / restoration stage (Section 15's `face.enhancement` setting).

Why this exists: `inswapper_128` produces a 128x128 face. When that face
occupies 400-500px on a 720p frame it is upscaled ~3-4x, which is exactly the
softness the user reported. Every serious face-swap tool (FaceFusion, ReActor,
roop) solves this the same way — run a face *restoration* model over the
swapped face at 512x512 and composite that back. This module is that stage.

Model: GFPGAN v1.4 (ONNX, 512x512 in/out). Chosen over GPEN-BFR / RestoreFormer
as the most widely-deployed option at this size, with the best documented
speed/quality balance for real-time use — see docs/PROGRESS.md for the measured
cost on this RTX 2060.

Provenance note (Section 23): same community mirror as the swap model
(`Gourieff/ReActor` on Hugging Face). GFPGAN itself is Tencent ARC's, released
under Apache-2.0 with a non-commercial clause on some weights — this project
uses it strictly locally for consensual avatar experimentation (Section 21),
never redistributed. Recorded in `models/registry.yaml`.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENHANCER_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "enhancer" / "GFPGANv1.4.onnx"
)

ENHANCER_SIZE = 512


class FaceEnhancerError(RuntimeError):
    """Actionable message (Section 20)."""


class FaceEnhancer:
    """Restores detail in an already-swapped face.

    `blend` (0.0-1.0) controls how much of the enhanced result is mixed back
    over the original swapped face. GFPGAN at full strength can look
    plasticky/over-smoothed — "professional" output usually blends rather than
    replacing outright, which also preserves the target's own skin texture and
    lighting. Maps to Section 15's OFF / LOW / HIGH setting.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_ENHANCER_MODEL_PATH,
        use_gpu: bool = True,
        blend: float = 0.8,
    ) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.blend = blend
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self.actual_providers: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise FaceEnhancerError(
                f"Face enhancer model not found at {self.model_path}.\n"
                "Run: python scripts/models.py install face-enhancer"
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
            raise FaceEnhancerError(describe_load_failure("the face enhancer", e)) from e
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self.actual_providers = session.get_providers()

        if self.use_gpu and "CUDAExecutionProvider" not in self.actual_providers:
            raise FaceEnhancerError(
                f"Requested CUDA for face enhancement but got {self.actual_providers}. "
                "See services/face/detector.py's identical check for the known cause."
            )

    def warm_up(self) -> None:
        if self._session is None:
            raise FaceEnhancerError("FaceEnhancer.warm_up() called before load().")
        dummy = np.zeros((1, 3, ENHANCER_SIZE, ENHANCER_SIZE), dtype=np.float32)
        self._session.run([self._output_name], {self._input_name: dummy})

    def _run_model(self, face_512_bgr: np.ndarray) -> np.ndarray:
        """GFPGAN expects RGB, CHW, normalized to [-1, 1]."""
        rgb = cv2.cvtColor(face_512_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        out = self._session.run([self._output_name], {self._input_name: blob})[0]

        out = np.transpose(out[0], (1, 2, 0))
        out = np.clip((out + 1.0) / 2.0, 0, 1) * 255.0
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def enhance_frame(
        self, frame_bgr: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Enhance the face at `landmarks` in `frame_bgr`, compositing the
        result back into a copy of the full frame. Returns (frame, ms)."""
        if self._session is None:
            raise FaceEnhancerError("FaceEnhancer.enhance_frame() called before load().")

        start = time.perf_counter()

        aligned, affine_matrix = face_align.norm_crop2(
            frame_bgr, landmarks, ENHANCER_SIZE
        )
        restored = self._run_model(aligned)

        if self.blend < 1.0:
            restored = cv2.addWeighted(
                restored, self.blend, aligned, 1.0 - self.blend, 0
            )

        output = _paste_back(frame_bgr, restored, affine_matrix)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return output, elapsed_ms


def _paste_back(
    target_frame: np.ndarray,
    restored_face: np.ndarray,
    affine_matrix: np.ndarray,
    face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Warp the restored 512x512 face back into the full frame using a soft,
    eroded+blurred mask so the boundary doesn't show as a visible rectangle.

    Deliberately mirrors the masking approach in InsightFace's own
    INSwapper.get() paste-back (erode proportionally to face size, then
    Gaussian-blur the mask edge) — matching the two stages' seam behavior
    matters more than inventing a second, different blend here.

    PERFORMANCE: all warping/masking is confined to the destination bounding
    box of the face, not the whole frame. Measured on a 1280x720 frame, the
    naive full-frame version cost 45.3 ms of pure CPU time per frame — more
    than half the GPU inference cost — because every warpAffine/erode/blur ran
    over 921k pixels to touch a ~500x600 region. Restricting to the region is
    mathematically identical (the warp is affine; translating the output
    origin is exact) and is the single cheapest win available in this stage.
    """
    frame_height, frame_width = target_frame.shape[:2]
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)

    # Where does the 512x512 aligned square land in the full frame?
    size = restored_face.shape[0]
    corners = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32
    ).reshape(-1, 1, 2)
    projected = cv2.transform(corners, inverse_matrix).reshape(-1, 2)

    # Pad generously so the blurred mask edge isn't clipped.
    pad = max(16, size // 8)
    x_min = int(np.floor(projected[:, 0].min())) - pad
    y_min = int(np.floor(projected[:, 1].min())) - pad
    x_max = int(np.ceil(projected[:, 0].max())) + pad
    y_max = int(np.ceil(projected[:, 1].max())) + pad

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)
    if x_max <= x_min or y_max <= y_min:
        return target_frame

    region_width = x_max - x_min
    region_height = y_max - y_min

    # Shift the warp so it renders directly into region-local coordinates.
    region_matrix = inverse_matrix.copy()
    region_matrix[0, 2] -= x_min
    region_matrix[1, 2] -= y_min

    warped_face = cv2.warpAffine(
        restored_face, region_matrix, (region_width, region_height), borderValue=0.0
    )

    if face_mask is not None:
        # A real face-shaped mask (see services/face/masking.py). It is already
        # feathered in aligned space, so warping it is all that's needed —
        # eroding/blurring again would eat into the face contour it was built
        # to follow.
        warped_mask = cv2.warpAffine(
            face_mask, region_matrix, (region_width, region_height), borderValue=0.0
        )
        warped_mask = np.clip(warped_mask, 0.0, 1.0)[:, :, np.newaxis]
    else:
        # Fallback: the whole aligned square, eroded and blurred. Covers
        # forehead/hair/background corners, so the boundary is more visible —
        # only used when landmark masking is unavailable.
        mask = np.full((size, size), 255, dtype=np.float32)
        warped_mask = cv2.warpAffine(
            mask, region_matrix, (region_width, region_height), borderValue=0.0
        )
        warped_mask[warped_mask > 20] = 255

        mask_indices = np.where(warped_mask == 255)
        if mask_indices[0].size == 0:
            return target_frame

        mask_h = np.max(mask_indices[0]) - np.min(mask_indices[0])
        mask_w = np.max(mask_indices[1]) - np.min(mask_indices[1])
        mask_size = int(np.sqrt(max(mask_h * mask_w, 1)))

        erode_k = max(mask_size // 12, 6)
        warped_mask = cv2.erode(
            warped_mask, np.ones((erode_k, erode_k), np.uint8), iterations=1
        )
        blur_k = max(mask_size // 16, 5)
        warped_mask = cv2.GaussianBlur(warped_mask, (2 * blur_k + 1, 2 * blur_k + 1), 0)
        warped_mask = (warped_mask / 255.0)[:, :, np.newaxis]

    output = target_frame.copy()
    region = output[y_min:y_max, x_min:x_max].astype(np.float32)
    merged = warped_mask * warped_face.astype(np.float32) + (1 - warped_mask) * region
    output[y_min:y_max, x_min:x_max] = merged.astype(np.uint8)
    return output
