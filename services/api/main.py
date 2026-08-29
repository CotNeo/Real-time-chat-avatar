"""
Milestone 14 (started early) — minimal FastAPI control service.

This is deliberately a thin slice, not the final API: it exposes /health,
/devices/video, and a live MJPEG camera preview built on the already-verified
ThreadedCameraStream (Milestone 2). There is no face swap, no voice conversion,
and no virtual-device output wired in yet — see docs/PROGRESS.md for the real
milestone status. The point of standing this up now is to have something
genuinely running and viewable in a browser, using only components that have
already been tested against real hardware, rather than a stub.
"""
from __future__ import annotations

import glob
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from services.face.detector import FaceDetector, FaceDetectorError
from services.face.overlay import draw_detection_overlay
from shared.logging.logger import configure_logging, get_logger
from shared.schemas.config import load_config
from shared.utils.camera import CameraConfig, CameraError, ThreadedCameraStream

config = load_config()
configure_logging(level=config.runtime.log_level, fmt="console")
log = get_logger(__name__)

app = FastAPI(title="Real-Time AI Avatar — control API", version="0.0.1-milestone14-partial")

_camera_stream: ThreadedCameraStream | None = None
_detector: FaceDetector | None = None
_last_detect_ms: float = 0.0


@app.on_event("startup")
def _startup() -> None:
    global _camera_stream, _detector
    cam_config = CameraConfig(
        device_index=config.video.device_index,
        width=config.video.width,
        height=config.video.height,
        fps=config.video.fps,
        manual_exposure_value=config.video.manual_exposure_value,
    )
    try:
        _camera_stream = ThreadedCameraStream(cam_config)
        _camera_stream.start()
        log.info("camera_started", device_index=cam_config.device_index)
    except CameraError as e:
        # Section 20: the API must still come up and explain itself — a missing
        # camera should not crash the whole service, since /health and
        # /devices/video are still useful without one.
        _camera_stream = None
        log.error("camera_start_failed", error=str(e))

    try:
        detector = FaceDetector()
        detector.load()
        _detector = detector
        log.info("face_detector_loaded", providers=detector.actual_providers)
    except FaceDetectorError as e:
        # Same graceful-degradation pattern as the camera: no model installed
        # yet (Milestone 3 not run) shouldn't take the whole API down.
        _detector = None
        log.error("face_detector_load_failed", error=str(e))


@app.on_event("shutdown")
def _shutdown() -> None:
    if _camera_stream is not None:
        _camera_stream.stop()
        log.info("camera_stopped")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "camera_connected": _camera_stream is not None,
        "camera_fps": round(_camera_stream.fps, 1) if _camera_stream else None,
        "face_detector_loaded": _detector is not None,
        "face_detector_providers": _detector.actual_providers if _detector else None,
        "last_detect_ms": round(_last_detect_ms, 2) if _detector else None,
        "config": {
            "video": config.video.model_dump(),
            "runtime": config.runtime.model_dump(),
        },
        "milestones_implemented": [
            "0: system audit",
            "1: CUDA / ONNX Runtime GPU verification",
            "2: camera capture engine",
            "3: face detection (SCRFD, live overlay in /preview/stream)",
            "12 (partial): PipeWire virtual microphone (device created, not yet fed real converted audio)",
        ],
        "milestones_not_yet_implemented": [
            "4: reference identity",
            "5: real-time face transfer",
            "7-9: microphone / voice conversion",
            "11: virtual camera (needs one manual sudo step — see README section 9)",
        ],
    }


@app.get("/devices/video")
def list_video_devices() -> dict:
    devices = sorted(glob.glob("/dev/video*"))
    return {"devices": devices}


def _mjpeg_generator():
    global _last_detect_ms
    if _camera_stream is None:
        return
    boundary = b"--frame"
    while True:
        try:
            frame = _camera_stream.get_latest()
        except CameraError as e:
            log.warning("preview_frame_unavailable", error=str(e))
            time.sleep(0.5)
            continue

        image = frame.image
        if _detector is not None:
            faces, detect_ms = _detector.detect(image)
            _last_detect_ms = detect_ms
            image = draw_detection_overlay(image.copy(), faces, _camera_stream.fps, detect_ms)

        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        # Pace the stream to the camera's own measured rate rather than
        # spinning as fast as the HTTP writer allows (Section 25: bounded,
        # not unbounded, work per unit time).
        time.sleep(max(0.0, 1.0 / max(config.video.fps, 1) - 0.005))


@app.get("/preview/stream")
def preview_stream() -> StreamingResponse:
    if _camera_stream is None:
        return StreamingResponse(iter(()), status_code=503)
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html>
    <head><title>Real-Time AI Avatar — dev preview</title>
    <style>
      body { font-family: system-ui, sans-serif; background: #111; color: #eee;
             display: flex; flex-direction: column; align-items: center; padding: 2rem; }
      img { max-width: 90vw; border: 2px solid #333; border-radius: 8px; }
      .note { color: #999; max-width: 640px; text-align: center; margin-top: 1rem; }
      code { background: #222; padding: 0.1rem 0.4rem; border-radius: 4px; }
    </style>
    </head>
    <body>
      <h2>Real-Time AI Avatar — face detection preview (Milestone 3/14 slice)</h2>
      <img src="/preview/stream" alt="live camera preview with face detection overlay" />
      <p class="note">
        Live SCRFD face detection (bounding box, 5-point landmarks, confidence,
        FPS) drawn on the raw webcam feed — still no face swap or voice
        conversion yet (see <code>/health</code> for exactly which milestones
        are done). Sit in front of the camera to see the green box track your
        face in real time.
      </p>
    </body>
    </html>
    """
