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
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from services.face.detector import FaceDetector, FaceDetectorError
from services.face.engine import FaceEngineError
from services.face.enhancer import FaceEnhancer, FaceEnhancerError
from services.face.identity import (
    MAX_REFERENCE_IMAGES,
    IdentityEncoder,
    IdentityEncoderError,
    build_identity_session,
)
from services.face.overlay import draw_detection_overlay
from services.face.swapper import FaceSwapEngine
from shared.logging.logger import configure_logging, get_logger
from shared.schemas.config import load_config
from shared.schemas.identity import IdentitySession
from shared.utils.camera import CameraConfig, CameraError, ThreadedCameraStream

# Below this mean-brightness value (0-255), Milestone 5 testing showed the
# swap model's output visibly degrades into incoherent noise — not a code bug,
# reproduced and isolated: the exact same code produces a coherent (if
# soft-edged) swapped face on a well-lit image, and gamma-correcting a dark
# frame does NOT recover quality because the underlying problem is sensor
# noise, not just low brightness (see docs/PROGRESS.md, Milestone 5). This
# threshold is an honest signal to the user, not a hard block.
LOW_LIGHT_WARNING_THRESHOLD = 70.0

config = load_config()
configure_logging(level=config.runtime.log_level, fmt="console")
log = get_logger(__name__)

app = FastAPI(title="Real-Time AI Avatar — control API", version="0.0.1-milestone14-partial")

# Section 15's face.enhancement levels map to how strongly the restored face
# is blended over the raw swap. Full strength can look plasticky/over-smoothed,
# so "high" stops short of 1.0 deliberately.
_ENHANCEMENT_BLEND = {"low": 0.5, "high": 0.85}

_camera_stream: ThreadedCameraStream | None = None
_detector: FaceDetector | None = None
_encoder: IdentityEncoder | None = None
_enhancer: FaceEnhancer | None = None
_swap_engine: FaceSwapEngine | None = None
_last_detect_ms: float = 0.0
_identity_session: IdentitySession | None = None
_session_active: bool = False


@app.on_event("startup")
def _startup() -> None:
    global _camera_stream, _detector, _encoder, _swap_engine
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

    try:
        encoder = IdentityEncoder()
        encoder.load()
        _encoder = encoder
        log.info("identity_encoder_loaded", providers=encoder.actual_providers)
    except IdentityEncoderError as e:
        _encoder = None
        log.error("identity_encoder_load_failed", error=str(e))

    global _enhancer
    if config.face.enhancement != "off":
        try:
            enhancer = FaceEnhancer(blend=_ENHANCEMENT_BLEND[config.face.enhancement])
            enhancer.load()
            enhancer.warm_up()
            _enhancer = enhancer
            log.info(
                "face_enhancer_loaded",
                providers=enhancer.actual_providers,
                level=config.face.enhancement,
            )
        except FaceEnhancerError as e:
            _enhancer = None
            log.error("face_enhancer_load_failed", error=str(e))
    else:
        _enhancer = None
        log.info("face_enhancer_disabled", reason="face.enhancement=off in config")

    if _detector is not None:
        try:
            swap_engine = FaceSwapEngine(detector=_detector, enhancer=_enhancer)
            swap_engine.load()
            swap_engine.warm_up()
            _swap_engine = swap_engine
            log.info("face_swap_engine_loaded", providers=swap_engine.actual_providers)
        except FaceEngineError as e:
            _swap_engine = None
            log.error("face_swap_engine_load_failed", error=str(e))
    else:
        _swap_engine = None
        log.error("face_swap_engine_skipped", reason="detector not loaded")


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
        "identity_encoder_loaded": _encoder is not None,
        "identity_session_active": _identity_session is not None
        and _identity_session.is_usable,
        "face_swap_engine_loaded": _swap_engine is not None,
        "face_swap_engine_providers": _swap_engine.actual_providers if _swap_engine else None,
        "face_enhancement": config.face.enhancement,
        "face_enhancer_loaded": _enhancer is not None,
        "session_active": _session_active,
        "config": {
            "video": config.video.model_dump(),
            "runtime": config.runtime.model_dump(),
        },
        "milestones_implemented": [
            "0: system audit",
            "1: CUDA / ONNX Runtime GPU verification",
            "2: camera capture engine",
            "3: face detection (SCRFD, live overlay in /preview/stream)",
            "4: reference identity (POST/GET/DELETE /identity)",
            "5: real-time face transfer (inswapper_128, POST/POST /session/start|stop)",
            "12 (partial): PipeWire virtual microphone (device created, not yet fed real converted audio)",
        ],
        "milestones_not_yet_implemented": [
            "7-9: microphone / voice conversion",
            "11: virtual camera (needs one manual sudo step — see README section 9)",
        ],
    }


@app.get("/devices/video")
def list_video_devices() -> dict:
    devices = sorted(glob.glob("/dev/video*"))
    return {"devices": devices}


def _reference_result_to_dict(result) -> dict:
    """Never serialize the embedding itself — it has no reason to leave the
    process, and the raw reference image is never stored past this request
    at all (Section 5/21)."""
    return {
        "filename": result.filename,
        "accepted": result.accepted,
        "problems": [p.value for p in result.problems],
        "quality_score": result.quality_score,
    }


def _session_summary(session: IdentitySession | None) -> dict:
    if session is None:
        return {"active": False}
    return {
        "active": True,
        "session_id": session.session_id,
        "usable": session.is_usable,
        "accepted_images": [_reference_result_to_dict(r) for r in session.accepted_images],
        "rejected_images": [_reference_result_to_dict(r) for r in session.rejected_images],
        "created_at": session.created_at,
    }


@app.post("/identity")
async def upload_identity(images: list[UploadFile] = File(...)) -> dict:
    global _identity_session

    if _detector is None:
        raise HTTPException(
            503,
            "Face detector is not loaded — run `python scripts/models.py "
            "install face-detection` and restart the API.",
        )
    if _encoder is None:
        raise HTTPException(
            503,
            "Identity encoder is not loaded — the same face-detection model "
            "install should include it; restart the API and check its logs.",
        )
    if not (1 <= len(images) <= MAX_REFERENCE_IMAGES):
        raise HTTPException(
            400,
            f"Upload between 1 and {MAX_REFERENCE_IMAGES} reference images "
            f"(got {len(images)}).",
        )

    decoded: list[tuple[str, np.ndarray | None]] = []
    for upload in images:
        raw = await upload.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
        decoded.append((upload.filename or "unnamed", image))

    session = build_identity_session(decoded, _detector, _encoder)
    _identity_session = session
    log.info(
        "identity_session_built",
        session_id=session.session_id,
        accepted=len(session.accepted_images),
        rejected=len(session.rejected_images),
        usable=session.is_usable,
    )
    return _session_summary(session)


@app.get("/identity")
def get_identity() -> dict:
    return _session_summary(_identity_session)


@app.delete("/identity")
def delete_identity() -> dict:
    global _identity_session, _session_active
    had_session = _identity_session is not None
    _identity_session = None
    if _session_active:
        _session_active = False
        if _swap_engine is not None:
            _swap_engine.reset()
        log.info("session_auto_stopped", reason="identity_cleared")
    log.info("identity_session_cleared", had_session=had_session)
    return {"cleared": had_session}


@app.post("/session/start")
def start_session() -> dict:
    global _session_active
    if _swap_engine is None:
        raise HTTPException(
            503,
            "Face swap engine is not loaded — run `python scripts/models.py "
            "install face-swap` and restart the API.",
        )
    if _identity_session is None or not _identity_session.is_usable:
        raise HTTPException(
            400,
            "No usable identity session — upload at least one valid reference "
            "photo via POST /identity before starting a session.",
        )
    _swap_engine.load_identity(_identity_session)
    _session_active = True
    log.info("session_started", session_id=_identity_session.session_id)
    return {"session_active": True}


@app.post("/session/stop")
def stop_session() -> dict:
    global _session_active
    _session_active = False
    if _swap_engine is not None:
        _swap_engine.reset()
    log.info("session_stopped")
    return {"session_active": False}


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
        brightness = float(np.mean(image))

        if _session_active and _swap_engine is not None:
            result = _swap_engine.process_frame(image)
            _last_detect_ms = result.timings_ms.get("detect", 0.0)
            image = result.output_image.copy()
            label = f"SWAP ACTIVE  faces: {1 if result.face_detected else 0}  FPS: {_camera_stream.fps:.1f}"
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if result.skip_reason == "face_partially_out_of_frame":
                cv2.putText(
                    image, "MOVE BACK - face is cut off by the frame edge",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2,
                )
        elif _detector is not None:
            faces, detect_ms = _detector.detect(image)
            _last_detect_ms = detect_ms
            image = draw_detection_overlay(image.copy(), faces, _camera_stream.fps, detect_ms)

        if brightness < LOW_LIGHT_WARNING_THRESHOLD:
            cv2.putText(
                image, f"LOW LIGHT (brightness {brightness:.0f}/255) - quality will degrade",
                (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2,
            )

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
             display: flex; flex-direction: column; align-items: center; padding: 2rem;
             gap: 1.5rem; }
      img { max-width: 90vw; border: 2px solid #333; border-radius: 8px; }
      .note { color: #999; max-width: 640px; text-align: center; }
      code { background: #222; padding: 0.1rem 0.4rem; border-radius: 4px; }
      .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
               padding: 1.5rem; width: min(640px, 90vw); }
      button { background: #2d7; border: none; padding: 0.5rem 1.2rem; border-radius: 6px;
               font-weight: 600; cursor: pointer; }
      button:disabled { background: #555; cursor: default; }
      pre { background: #000; padding: 1rem; border-radius: 6px; overflow-x: auto;
            font-size: 0.85rem; white-space: pre-wrap; }
      .accepted { color: #4d8; }
      .rejected { color: #e66; }
    </style>
    </head>
    <body>
      <h2>Real-Time AI Avatar — dev preview (Milestones 2-5/14 slice)</h2>
      <img src="/preview/stream" alt="live camera preview" />
      <p class="note">
        Face detection overlay by default. Upload a reference identity below
        and click "Start session" to switch the preview to live face swap
        instead. Still no voice conversion or virtual devices yet (see
        <code>/health</code> for exactly which milestones are done). A
        low-light warning appears on the preview itself when the frame is too
        dark for reliable results — this is a measured hardware limitation
        (see docs/PROGRESS.md, Milestone 5), not a bug.
      </p>

      <div class="panel">
        <h3>Reference identity (Milestone 4)</h3>
        <p class="note">Upload 1-5 reference photos. Each is validated (face
        found? exactly one? sharp enough? not too small?) and never stored —
        only a derived embedding survives the request.</p>
        <input type="file" id="refFiles" accept="image/*" multiple />
        <button id="refUpload">Upload</button>
        <button id="refClear">Clear session</button>
        <pre id="refResult">No identity session yet.</pre>
      </div>

      <div class="panel">
        <h3>Face swap session (Milestone 5)</h3>
        <button id="sessionStart">Start session</button>
        <button id="sessionStop">Stop session</button>
        <pre id="sessionResult">Session not started.</pre>
      </div>

      <script>
        async function refreshIdentity() {
          const r = await fetch('/identity');
          document.getElementById('refResult').textContent = JSON.stringify(await r.json(), null, 2);
        }
        document.getElementById('refUpload').onclick = async () => {
          const files = document.getElementById('refFiles').files;
          if (!files.length) { alert('Choose 1-5 images first.'); return; }
          const form = new FormData();
          for (const f of files) form.append('images', f);
          const r = await fetch('/identity', { method: 'POST', body: form });
          const data = await r.json();
          document.getElementById('refResult').textContent = JSON.stringify(data, null, 2);
        };
        document.getElementById('refClear').onclick = async () => {
          await fetch('/identity', { method: 'DELETE' });
          await refreshIdentity();
        };
        document.getElementById('sessionStart').onclick = async () => {
          const r = await fetch('/session/start', { method: 'POST' });
          const data = await r.json();
          document.getElementById('sessionResult').textContent =
            r.ok ? JSON.stringify(data, null, 2) : `Error: ${data.detail}`;
        };
        document.getElementById('sessionStop').onclick = async () => {
          const r = await fetch('/session/stop', { method: 'POST' });
          document.getElementById('sessionResult').textContent = JSON.stringify(await r.json(), null, 2);
        };
        refreshIdentity();
      </script>
    </body>
    </html>
    """
