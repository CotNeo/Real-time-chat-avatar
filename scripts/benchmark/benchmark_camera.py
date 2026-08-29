#!/usr/bin/env python3
"""
Milestone 2 — raw camera capture benchmark.

Opens the real webcam, captures for a fixed duration, measures actual achieved FPS
(not the number the driver claims it supports), saves one sample frame to disk so
the result can be visually confirmed, and writes a JSON report.

Run:
    source .venv/bin/activate
    python scripts/benchmark/benchmark_camera.py --device 0 --seconds 5
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
import numpy as np  # noqa: E402

from shared.utils.camera import CameraCapture, CameraConfig, CameraError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--manual-exposure",
        type=int,
        default=None,
        help="Force a fixed exposure value (e.g. 150) to work around auto-exposure "
        "throttling FPS in low light. Omit to keep the camera's default behavior.",
    )
    args = parser.parse_args()

    config = CameraConfig(
        device_index=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        manual_exposure_value=args.manual_exposure,
    )

    print(f"Opening /dev/video{args.device} at {args.width}x{args.height}@{args.fps}...")
    try:
        cam = CameraCapture(config)
        cam.open()
    except CameraError as e:
        print(f"\n[FAIL] {e}")
        return 1

    fmt = cam.actual_format()
    print(f"Driver negotiated: {fmt}")

    frame_times: list[float] = []
    brightness_samples: list[float] = []
    sample_frame = None
    start = time.monotonic()
    prev = start
    frame_count = 0

    try:
        while time.monotonic() - start < args.seconds:
            frame = cam.read()
            now = frame.timestamp
            frame_times.append(now - prev)
            prev = now
            frame_count += 1
            brightness_samples.append(float(np.mean(frame.image)))
            if frame.frame_index == 1:
                sample_frame = frame.image.copy()
    except CameraError as e:
        print(f"\n[FAIL] Capture interrupted: {e}")
        cam.close()
        return 1

    elapsed = time.monotonic() - start
    cam.close()

    frame_times_ms = [t * 1000 for t in frame_times[1:]]  # drop first (startup) sample
    achieved_fps = frame_count / elapsed if elapsed > 0 else 0.0

    result = {
        "device": f"/dev/video{args.device}",
        "requested": {"width": args.width, "height": args.height, "fps": args.fps},
        "manual_exposure_value": args.manual_exposure,
        "negotiated": fmt,
        "duration_s": round(elapsed, 3),
        "frames_captured": frame_count,
        "achieved_fps": round(achieved_fps, 2),
        "mean_brightness_0_255": round(statistics.mean(brightness_samples), 1)
        if brightness_samples
        else None,
        "frame_interval_ms": {
            "mean": round(statistics.mean(frame_times_ms), 2) if frame_times_ms else None,
            "p50": round(statistics.median(frame_times_ms), 2) if frame_times_ms else None,
            "p95": round(
                statistics.quantiles(frame_times_ms, n=20)[18], 2
            )
            if len(frame_times_ms) >= 20
            else None,
            "max": round(max(frame_times_ms), 2) if frame_times_ms else None,
        },
    }

    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    suffix = "-manual-exposure" if args.manual_exposure is not None else "-auto-exposure"
    (out_dir / f"camera-results{suffix}.json").write_text(json.dumps(result, indent=2))

    if sample_frame is not None:
        sample_path = out_dir / f"camera-sample-frame{suffix}.jpg"
        cv2.imwrite(str(sample_path), sample_frame)
        result["sample_frame"] = str(sample_path)

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
