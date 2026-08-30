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
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from services.face.enhancer import FaceEnhancer
    from services.face.masking import LandmarkMasker, OcclusionMasker
from types import SimpleNamespace

import numpy as np

from services.face.detector import FaceDetector, FaceDetectorError
from services.face.engine import FaceEngine, FaceEngineError, FaceFrameResult
from shared.schemas.identity import IdentitySession

REPO_ROOT = Path(__file__).resolve().parents[2]
# fp32, NOT fp16. Measured directly (docs/PROGRESS.md, Milestone 5): on this
# RTX 2060, the fp16 variant of this model produces visibly degraded output —
# blurred, smeared, discolored — on the exact same well-lit input where the
# fp32 variant produces a clean, sharp face. Same code path, same identity
# embedding, same target image, only the weights file differs. The fp16 build
# is NOT simply "slightly lower precision" here; it is broken enough to be
# unusable.
#
# fp32 does cost real speed — measured 78.0 ms/frame vs 62.2 ms for fp16
# (~25% slower), plus ~277 MB more disk. That trade is still clearly correct:
# a fast unusable image is worth nothing. If the pipeline later needs those
# milliseconds back, the answer is a different model or TensorRT (Section 9),
# NOT reverting to this broken fp16 file.
DEFAULT_SWAP_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "inswapper" / "inswapper_128.onnx"
)
# Alternative swap model. Measured against inswapper on a male target with a
# female reference set (docs/PROGRESS.md, Milestone 6c):
#
#   inswapper_128   128px  62.1 ms  identity 0.831
#   hyperswap_1c    256px  27.9 ms  identity 0.760
#
# hyperswap scores lower on similarity to the reference embedding but is
# visibly sharper and more natural — twice the resolution means far less
# upscaling. Which one is "better" depends on the goal: matching a specific
# face favours inswapper, looking photoreal favours hyperswap. It also emits
# its own occlusion mask as a second output, so it can stand in for the
# separate XSeg pass.
HYPERSWAP_MODEL_PATH = (
    REPO_ROOT / "models" / "face" / "models" / "inswapper" / "hyperswap_1c_256.onnx"
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
        enhancer: "FaceEnhancer | None" = None,
    ) -> None:
        self._detector = detector
        self.model_path = model_path
        self.use_gpu = use_gpu
        # TensorRT fp32 for this model: fp16 destroys identity transfer
        # (0.831 -> 0.122). See shared/utils/providers.py.
        self.use_tensorrt = True
        # Optional restoration stage (Section 15's face.enhancement setting).
        # When present, process_frame() takes a fused path that shares one
        # alignment and one paste-back between swap and enhancement instead of
        # doing each twice — measured 168 ms vs 203 ms per frame for the same
        # output quality (docs/PROGRESS.md, Milestone 5c).
        self.enhancer = enhancer
        # Optional realism stages (docs/PROGRESS.md, Milestone 5d). Both are
        # cheap and both target the "obviously pasted on" look rather than
        # sharpness: a face-contour mask instead of the aligned square, and
        # colour transfer so the generated face carries the room's lighting
        # instead of the reference photo's.
        self.masker: "LandmarkMasker | None" = None
        self.occluder: "OcclusionMasker | None" = None
        self.color_match = True
        # How far the contour mask grows past the landmark hull. 1.04 hugged
        # the face oval and stopped at the eyebrows, which meant the forehead
        # was never swapped — so a raised brow or forehead creases stayed on
        # the user's own skin while the rest of the face was someone else's.
        # 1.3 reaches the hairline and temples; measured no quality cost.
        self.mask_expand = 1.3
        self._swapper = None
        # Initialised here, not only in load(), so process_frame() before
        # load() raises the descriptive FaceEngineError it intends rather than
        # a bare AttributeError.
        self._session = None
        self._is_hyperswap = False
        self._swap_size = 128
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
        self._is_hyperswap = False

        from shared.utils.cuda_env import ensure_onnxruntime_cuda_libs

        if self.use_gpu:
            ensure_onnxruntime_cuda_libs()

        import onnxruntime
        from insightface.model_zoo.inswapper import INSwapper

        from shared.utils.providers import providers_for

        requested = (
            providers_for("swap", use_tensorrt=self.use_tensorrt)
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        session = onnxruntime.InferenceSession(str(self.model_path), providers=requested)
        self.actual_providers = session.get_providers()

        # hyperswap takes the raw 512-d embedding and has no `emap`
        # projection; inswapper projects the embedding through one. Detect
        # which family this file is rather than trusting the filename.
        self._is_hyperswap = "hyperswap" in self.model_path.name.lower()
        if self._is_hyperswap:
            self._session = session
            self._input_names = [i.name for i in session.get_inputs()]
            self._output_names = [o.name for o in session.get_outputs()]
            self._swap_size = session.get_inputs()[
                next(i for i, n in enumerate(self._input_names) if "target" in n)
            ].shape[2]
            self._swapper = None
        else:
            self._swapper = INSwapper(model_file=str(self.model_path), session=session)
            self._session = session
            self._swap_size = 128

        if self.use_gpu and not any(
            p in self.actual_providers
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        ):
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
        if self._session is None:
            raise FaceEngineError("FaceSwapEngine.warm_up() called before load().")

        # Use a non-degenerate unit vector, never zeros. Bug found during
        # Milestone 5 testing: an all-zeros embedding projects through
        # inswapper's emap to an all-zero latent whose norm is 0, and the
        # following `latent /= norm` divides 0/0 — "invalid value encountered
        # in divide" plus a NaN-filled result. Harmless here (the output is
        # discarded) but confusing to debug later, and trivially avoided.
        dummy_vector = np.ones(512, dtype=np.float32)
        dummy_vector /= np.linalg.norm(dummy_vector)
        size = self._swap_size
        dummy_blob = np.zeros((1, 3, size, size), dtype=np.float32)

        if self._is_hyperswap:
            latent = dummy_vector.reshape((1, -1))
            feeds = {
                name: (dummy_blob if "target" in name else latent)
                for name in self._input_names
            }
            self._session.run(self._output_names, feeds)
        else:
            latent = dummy_vector.reshape((1, -1)) @ self._swapper.emap
            latent /= np.linalg.norm(latent)
            self._swapper.session.run(
                self._swapper.output_names,
                {
                    self._swapper.input_names[0]: dummy_blob,
                    self._swapper.input_names[1]: latent.astype(np.float32),
                },
            )

    def reset(self) -> None:
        self._last_faces = []
        self._frames_since_detect = 0

    def process_frame(self, frame: np.ndarray) -> FaceFrameResult:
        if self._session is None:
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

        # A face whose bounding box extends past the frame edge cannot be
        # aligned correctly: the 128x128 ArcFace warp samples from outside the
        # source image and fills that region with black, which the swap model
        # then "reconstructs" into a smeared, discolored mess. Verified
        # directly by dumping the intermediate aligned crop (a visible black
        # wedge) and the model's raw 128x128 output (incoherent) for a frame
        # with bbox x1 = -32. Skipping the swap and passing the real frame
        # through is strictly better than emitting a corrupted face, and the
        # caller gets `skip_reason` so the UI can tell the user to move back
        # into frame rather than leaving them guessing.
        if _is_out_of_frame(face.bbox, frame.shape):
            return FaceFrameResult(
                output_image=frame,
                face_detected=True,
                bbox=face.bbox,
                landmarks=face.landmarks,
                detection_ran_this_frame=detection_ran,
                timings_ms=timings,
                skip_reason="face_partially_out_of_frame",
            )

        infer_start = time.perf_counter()
        output_image = self._swap_and_enhance(frame, face.landmarks, face.bbox)
        timings["inference"] = (time.perf_counter() - infer_start) * 1000

        return FaceFrameResult(
            output_image=output_image,
            face_detected=True,
            bbox=face.bbox,
            landmarks=face.landmarks,
            detection_ran_this_frame=detection_ran,
            timings_ms=timings,
        )


    def _swap_and_enhance(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Single fused path for both enhanced and unenhanced output.

        Aligns once at the working size, runs the 128px swap on a downscale of
        that same crop, optionally restores at 512, and composites exactly
        once with a region-limited soft-mask paste-back.

        Two measured reasons this replaced the naive composition:
          * With enhancement on, doing swap-then-enhance separately warps,
            masks and blends the full frame twice for no benefit — 203 ms vs
            168 ms per frame for equal-or-better output.
          * With enhancement off, InsightFace's own `INSwapper.get(
            paste_back=True)` blends over the whole frame; routing through the
            same region-limited paste-back here is cheaper for identical
            output.
        See docs/PROGRESS.md, Milestone 5c, for the numbers.
        """
        from insightface.utils import face_align

        from services.face.enhancer import ENHANCER_SIZE, _paste_back
        from services.face.masking import match_color

        # Work at 512 when restoring (the enhancer's fixed input size), else
        # 256 — enough to keep the 128px swap result from being resampled up
        # and straight back down, without paying for pixels nothing reads.
        work_size = self.enhancer.size if self.enhancer is not None else 256
        aligned, affine_matrix = face_align.norm_crop2(frame, landmarks, work_size)

        size = self._swap_size
        swap_input = cv2.resize(aligned, (size, size), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            swap_input, 1.0 / 255.0, (size, size), (0.0, 0.0, 0.0), swapRB=True
        )

        model_mask = None
        if self._is_hyperswap:
            # Raw embedding, no emap projection.
            latent = self._source_embedding.reshape((1, -1)).astype(np.float32)
            feeds = {
                name: (blob if "target" in name else latent)
                for name in self._input_names
            }
            outputs = self._session.run(self._output_names, feeds)
            prediction = outputs[0]
            if len(outputs) > 1:
                # Second output is the model's own occlusion mask.
                model_mask = np.clip(outputs[1][0, 0], 0.0, 1.0)
        else:
            latent = self._source_embedding.reshape((1, -1)) @ self._swapper.emap
            latent /= np.linalg.norm(latent)
            prediction = self._swapper.session.run(
                self._swapper.output_names,
                {
                    self._swapper.input_names[0]: blob,
                    self._swapper.input_names[1]: latent,
                },
            )[0]

        swapped = np.clip(255 * prediction.transpose((0, 2, 3, 1))[0], 0, 255)
        swapped = swapped.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR

        face_out = cv2.resize(
            swapped, (work_size, work_size), interpolation=cv2.INTER_LANCZOS4
        )
        if self.enhancer is not None:
            restored = self.enhancer._run_model(face_out)
            if self.enhancer.blend < 1.0:
                restored = cv2.addWeighted(
                    restored, self.enhancer.blend, face_out, 1.0 - self.enhancer.blend, 0
                )
            face_out = restored

        # Face-contour mask (falls back to the aligned square if unavailable —
        # a mask failure must degrade the blend, never drop the frame).
        face_mask = None
        if self.masker is not None and bbox is not None:
            dense = self.masker.landmarks_106(frame, bbox, landmarks)
            if dense is not None:
                face_mask = self.masker.build_mask(
                    dense, affine_matrix, work_size, expand=self.mask_expand
                )

        # hyperswap emits its own occlusion mask, so it can cover this without
        # the separate XSeg pass.
        if model_mask is not None:
            resized = cv2.resize(
                model_mask, (work_size, work_size), interpolation=cv2.INTER_LINEAR
            )
            face_mask = resized if face_mask is None else face_mask * resized

        # Occlusion mask: multiply in, so anything held in front of the face
        # (a hand, a mug) keeps its real pixels instead of having a generated
        # face painted over it. Multiplication is the right combination here —
        # a pixel must be BOTH inside the face contour AND actually visible.
        if self.occluder is not None:
            visible = self.occluder.mask_for(aligned, work_size)
            if visible is not None:
                face_mask = visible if face_mask is None else face_mask * visible

        # Colour transfer, inside the mask, so the generated face picks up the
        # room's lighting instead of the reference photo's. Done after
        # restoration because the enhancer also shifts tone slightly.
        if self.color_match:
            face_out = match_color(face_out, aligned, face_mask)

        return _paste_back(frame, face_out, affine_matrix, face_mask)


def _is_out_of_frame(
    bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...], margin: int = 0
) -> bool:
    """True when the detected face box extends past the frame edge — see the
    call site in process_frame() for why this must skip the swap."""
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    return x1 < margin or y1 < margin or x2 > (width - margin) or y2 > (height - margin)


def _dummy_landmarks() -> np.ndarray:
    return np.array(
        [[96, 96], [160, 96], [128, 128], [100, 160], [156, 160]], dtype=np.float32
    )
