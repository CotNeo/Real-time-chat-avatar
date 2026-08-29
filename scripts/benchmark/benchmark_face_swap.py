#!/usr/bin/env python3
"""
Milestone 5 — real-time face swap benchmark.

Loads a reference identity from local image files, opens the real webcam,
runs the full Mode A pipeline (detect -> track -> swap -> blend) for a fixed
duration, saves a before/after sample pair, and reports FPS/latency. Does not
fake a "swap worked" result — if no face was ever found, that's reported as a
failure explicitly, same discipline as benchmark_face_detection.py.

Run:
    source .venv/bin/activate
    python scripts/benchmark/benchmark_face_swap.py --reference photo1.jpg photo2.jpg --seconds 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import cv2  # noqa: E402

from services.face.detector import FaceDetector, FaceDetectorError  # noqa: E402
from services.face.identity import (  # noqa: E402
    IdentityEncoder,
    IdentityEncoderError,
    build_identity_session,
)
from services.face.swapper import FaceSwapEngine  # noqa: E402
from services.face.engine import FaceEngineError  # noqa: E402
from shared.utils.camera import CameraCapture, CameraConfig, CameraError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True, help="1-5 reference image paths")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    print("Loading detector + identity encoder + swap model...")
    try:
        detector = FaceDetector()
        detector.load()
        encoder = IdentityEncoder()
        encoder.load()
        engine = FaceSwapEngine(detector=detector)
        engine.load()
    except (FaceDetectorError, IdentityEncoderError, FaceEngineError) as e:
        print(f"\n[FAIL] {e}")
        return 1
    print(f"[ OK ] Swap model providers: {engine.actual_providers}")

    images = []
    for path_str in args.reference:
        img = cv2.imread(path_str)
        images.append((Path(path_str).name, img))

    session = build_identity_session(images, detector, encoder)
    if not session.is_usable:
        print("\n[FAIL] No reference image was accepted:")
        print(json.dumps([r.__dict__ for r in session.rejected_images], default=str, indent=2))
        return 1
    print(f"[ OK ] Identity session usable: {len(session.accepted_images)} accepted image(s)")

    engine.load_identity(session)
    print("Warming up...")
    engine.warm_up()

    try:
        cam = CameraCapture(CameraConfig(device_index=args.device))
        cam.open()
    except CameraError as e:
        print(f"\n[FAIL] {e}")
        return 1

    detect_times, infer_times = [], []
    frames_with_face = 0
    frames_total = 0
    before_frame = None
    after_frame = None
    start = time.monotonic()

    try:
        while time.monotonic() - start < args.seconds:
            frame = cam.read()
            result = engine.process_frame(frame.image)
            frames_total += 1
            if "detect" in result.timings_ms:
                detect_times.append(result.timings_ms["detect"])
            if "inference" in result.timings_ms:
                infer_times.append(result.timings_ms["inference"])
            if result.face_detected:
                frames_with_face += 1
                before_frame = frame.image.copy()
                after_frame = result.output_image.copy()
    except CameraError as e:
        print(f"\n[FAIL] Capture interrupted: {e}")
        cam.close()
        return 1

    elapsed = time.monotonic() - start
    cam.close()

    achieved_fps = frames_total / elapsed if elapsed > 0 else 0.0
    result_summary = {
        "swap_model_providers": engine.actual_providers,
        "reference_images_accepted": len(session.accepted_images),
        "duration_s": round(elapsed, 2),
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "achieved_fps": round(achieved_fps, 2),
        "detect_ms": {
            "mean": round(statistics.mean(detect_times), 2) if detect_times else None,
            "p95": round(statistics.quantiles(detect_times, n=20)[18], 2)
            if len(detect_times) >= 20 else None,
        },
        "inference_ms": {
            "mean": round(statistics.mean(infer_times), 2) if infer_times else None,
            "p95": round(statistics.quantiles(infer_times, n=20)[18], 2)
            if len(infer_times) >= 20 else None,
        },
    }

    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "face-swap-results.json").write_text(json.dumps(result_summary, indent=2))

    if before_frame is not None:
        cv2.imwrite(str(out_dir / "face-swap-before.jpg"), before_frame)
        cv2.imwrite(str(out_dir / "face-swap-after.jpg"), after_frame)
    else:
        print("\n[WARN] No face was ever detected during this run — no before/after saved.")

    print("\n=== Result ===")
    print(json.dumps(result_summary, indent=2))
    return 0 if frames_with_face > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
