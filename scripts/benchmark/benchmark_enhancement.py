#!/usr/bin/env python3
"""
Milestone 5c — measure the real cost of the face-enhancement stage.

Runs the full live pipeline (camera -> detect -> swap [-> enhance]) against the
real webcam at each Section 15 enhancement level and reports honest per-stage
timings, achievable pipeline FPS, and VRAM. Writes
`benchmarks/enhancement-results.json`.

Run:
    source .venv/bin/activate
    python scripts/benchmark/benchmark_enhancement.py --reference <images...>
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from services.face.detector import FaceDetector  # noqa: E402
from services.face.enhancer import FaceEnhancer  # noqa: E402
from services.face.identity import (  # noqa: E402
    IdentityEncoder,
    build_identity_session,
)
from services.face.swapper import FaceSwapEngine  # noqa: E402
from shared.utils.camera import CameraCapture, CameraConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BLEND_FOR_LEVEL = {"low": 0.5, "high": 0.85}


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - benchmark must not die on a metrics hiccup
        return None


def run_level(
    level: str, detector, session, seconds: float, device: int,
    target_image=None,
) -> dict:
    """`target_image` (a still frame containing a face) makes this measure pure
    compute cost reproducibly, without needing a person sitting in front of the
    camera. Without it, the run uses the live camera — and will refuse to
    report numbers if no face was ever actually swapped, because timing frames
    where the pipeline early-returns measures nothing but detection on an empty
    room (this exact trap produced a bogus "212 FPS" reading during
    development)."""
    enhancer = None
    if level != "off":
        enhancer = FaceEnhancer(blend=BLEND_FOR_LEVEL[level])
        enhancer.load()
        enhancer.warm_up()

    engine = FaceSwapEngine(detector=detector, enhancer=enhancer)
    engine.load()
    engine.load_identity(session)
    engine.warm_up()

    process_ms: list[float] = []
    frames = swapped_frames = 0

    if target_image is not None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            start = time.perf_counter()
            result = engine.process_frame(target_image)
            process_ms.append((time.perf_counter() - start) * 1000)
            frames += 1
            if result.face_detected and result.skip_reason is None:
                swapped_frames += 1
    else:
        cam = CameraCapture(CameraConfig(device_index=device))
        cam.open()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            frame = cam.read()
            start = time.perf_counter()
            result = engine.process_frame(frame.image)
            process_ms.append((time.perf_counter() - start) * 1000)
            frames += 1
            if result.face_detected and result.skip_reason is None:
                swapped_frames += 1
        cam.close()

    mean_ms = statistics.mean(process_ms) if process_ms else 0.0
    return {
        "enhancement": level,
        "frames": frames,
        "frames_actually_swapped": swapped_frames,
        "process_ms": {
            "mean": round(mean_ms, 1),
            "p50": round(statistics.median(process_ms), 1) if process_ms else None,
            "p95": round(statistics.quantiles(process_ms, n=20)[18], 1)
            if len(process_ms) >= 20 else None,
        },
        "max_pipeline_fps": round(1000 / mean_ms, 1) if mean_ms else None,
        "vram_used_mb": vram_used_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--target-image",
        default=None,
        help="Measure against this still image instead of the live camera. "
        "Reproducible, and doesn't require a person in front of the webcam.",
    )
    args = parser.parse_args()

    detector = FaceDetector(); detector.load()
    encoder = IdentityEncoder(); encoder.load()
    images = [(Path(p).name, cv2.imread(p)) for p in args.reference]
    session = build_identity_session(images, detector, encoder)
    if not session.is_usable:
        print("[FAIL] no reference image accepted")
        return 1
    print(f"[ OK ] identity: {len(session.accepted_images)} accepted\n")

    target_image = None
    if args.target_image:
        target_image = cv2.imread(args.target_image)
        if target_image is None:
            print(f"[FAIL] could not read --target-image {args.target_image}")
            return 1

    results = [
        run_level(level, detector, session, args.seconds, args.device, target_image)
        for level in ("off", "low", "high")
    ]

    # Refuse to print timings that measured nothing (Section 8: do not fake
    # measurements). A run where the pipeline never actually swapped a face is
    # measuring detection on an empty room, not the thing being benchmarked.
    if any(r["frames_actually_swapped"] == 0 for r in results):
        print(
            "\n[FAIL] At least one level swapped 0 frames — no face was in view.\n"
            "       These timings would measure only detection on an empty scene "
            "and are NOT reported.\n"
            "       Either sit in front of the camera, or pass --target-image "
            "<a photo containing a face> for a reproducible compute-cost measurement."
        )
        return 1

    out = REPO_ROOT / "benchmarks" / "enhancement-results.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"{'level':>6} | {'mean ms':>8} | {'p95 ms':>7} | {'max fps':>7} | {'VRAM MB':>7}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['enhancement']:>6} | {r['process_ms']['mean']:>8} | "
            f"{str(r['process_ms']['p95']):>7} | {str(r['max_pipeline_fps']):>7} | "
            f"{str(r['vram_used_mb']):>7}"
        )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
