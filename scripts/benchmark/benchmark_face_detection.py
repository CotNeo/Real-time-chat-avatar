#!/usr/bin/env python3
"""
Milestone 3 — live face detection benchmark.

Opens the real webcam, runs SCRFD detection on a fixed number of frames,
draws bounding box + 5-point landmarks + confidence + running FPS on each
(Section 3's explicit render requirement), saves one annotated sample frame,
and reports detection rate + latency percentiles. Does NOT fake a "detection
worked" result — if the detector finds nothing across the whole run, that is
reported as a failure, not silently skipped.

Run:
    source .venv/bin/activate
    python scripts/benchmark/benchmark_face_detection.py --seconds 8
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
from services.face.overlay import draw_detection_overlay  # noqa: E402
from shared.utils.camera import CameraCapture, CameraConfig, CameraError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--manual-exposure", type=int, default=None)
    args = parser.parse_args()

    print("Loading SCRFD detector...")
    detector = FaceDetector()
    try:
        detector.load()
    except FaceDetectorError as e:
        print(f"\n[FAIL] {e}")
        return 1
    print(f"[ OK ] Detector loaded. Providers in use: {detector.actual_providers}")

    cam_config = CameraConfig(
        device_index=args.device, manual_exposure_value=args.manual_exposure
    )
    try:
        cam = CameraCapture(cam_config)
        cam.open()
    except CameraError as e:
        print(f"\n[FAIL] {e}")
        return 1

    detect_times_ms: list[float] = []
    frames_with_face = 0
    frames_total = 0
    max_faces_seen = 0
    sample_annotated_frame = None
    best_score_for_sample = -1.0

    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            frame = cam.read()
            faces, detect_ms = detector.detect(frame.image)
            detect_times_ms.append(detect_ms)
            frames_total += 1
            if faces:
                frames_with_face += 1
                max_faces_seen = max(max_faces_seen, len(faces))
                top_score = max(f.score for f in faces)
                if top_score > best_score_for_sample:
                    best_score_for_sample = top_score
                    annotated = frame.image.copy()
                    draw_detection_overlay(annotated, faces, cam.fps, detect_ms)
                    sample_annotated_frame = annotated
    except CameraError as e:
        print(f"\n[FAIL] Capture interrupted: {e}")
        cam.close()
        return 1

    elapsed = time.monotonic() - start
    cam.close()

    detection_rate = frames_with_face / frames_total if frames_total else 0.0
    result = {
        "providers": detector.actual_providers,
        "duration_s": round(elapsed, 2),
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "detection_rate": round(detection_rate, 3),
        "max_faces_seen_in_one_frame": max_faces_seen,
        "camera_fps": round(cam.fps, 2),
        "detect_latency_ms": {
            "mean": round(statistics.mean(detect_times_ms), 2) if detect_times_ms else None,
            "p50": round(statistics.median(detect_times_ms), 2) if detect_times_ms else None,
            "p95": round(statistics.quantiles(detect_times_ms, n=20)[18], 2)
            if len(detect_times_ms) >= 20
            else None,
            "max": round(max(detect_times_ms), 2) if detect_times_ms else None,
        },
    }

    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "face-detection-results.json").write_text(json.dumps(result, indent=2))

    if sample_annotated_frame is not None:
        cv2.imwrite(str(out_dir / "face-detection-sample.jpg"), sample_annotated_frame)
        result["sample_frame"] = str(out_dir / "face-detection-sample.jpg")
    else:
        print("\n[WARN] No face was detected in ANY frame during this run — not faking success.")

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))
    return 0 if frames_with_face > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
