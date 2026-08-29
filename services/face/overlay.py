"""Shared debug-overlay drawing (Section 3's render requirement: bounding box,
landmarks, confidence, FPS) — used by both the benchmark script and the live
API preview so there's one implementation, not two copies that drift apart."""
from __future__ import annotations

import cv2
import numpy as np

from services.face.detector import DetectedFace


def draw_detection_overlay(
    frame: np.ndarray,
    faces: list[DetectedFace],
    fps: float,
    detect_ms: float,
) -> np.ndarray:
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (lx, ly) in face.landmarks:
            cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)
        cv2.putText(
            frame, f"{face.score:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    cv2.putText(
        frame, f"FPS: {fps:.1f}  detect: {detect_ms:.1f}ms  faces: {len(faces)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
    )
    return frame
