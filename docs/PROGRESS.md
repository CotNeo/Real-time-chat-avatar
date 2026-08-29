# Progress Log

## Milestone 0 — System Audit

Status: **DONE**

Implemented:
- Full inspection of the real target machine (not a sandbox — confirmed RTX 2060,
  real webcam, real PipeWire audio).
- `docs/ENVIRONMENT_REPORT.md` with measured OS/CPU/RAM/GPU/camera/audio/toolchain
  state and the architecture decisions it drives.
- Repository scaffold created (`realtime-ai-avatar/`, Section 16 layout), git
  initialized.

Measured:
- Ubuntu 24.04.4, kernel 6.17, i5-10400F (12 threads), 15 GiB RAM.
- RTX 2060, 6144 MiB VRAM, driver 580.159.03, compute capability 7.5.
- Disk: started at 22 GB free (95% full) — reclaimed 12 GB from a stale pip cache,
  now ~27–35 GB free depending on what's currently installed. This is the tightest
  resource in the whole project, tighter than VRAM.
- Camera: Logitech HD Webcam C510 on USB 2.0 High-Speed, ACL-granted (no sudo
  needed for capture). Audio: PipeWire 1.0.5, webcam mic is the default source.

Problems:
- `sudo` requires an interactive password — this agent cannot supply one. Anything
  needing root (installing `v4l2loopback-dkms`, `v4l-utils`) must be run manually,
  once, by the user.
- Disk headroom is limited; large model downloads later (voice conversion
  checkpoints especially) need to be chosen conservatively and disk re-checked
  before each install.

Next: Milestone 1.

---

## Milestone 1 — CUDA Test

Status: **DONE**

Implemented:
- `scripts/setup/verify_cuda.py`: doesn't just check `torch.cuda.is_available()` /
  the ONNX Runtime provider list — actually runs a matmul on the GPU and actually
  creates and executes an ONNX Runtime session pinned to `CUDAExecutionProvider`,
  then confirms the session really used it (not silently falling back to CPU).
- Python venv at `.venv/` with `torch`, `onnxruntime-gpu`, `numpy`, `opencv-python-headless`.

Measured:
```
[ OK ] PyTorch CUDA — torch 2.13.0 (cu13.0), RTX 2060, compute cap 7.5, 5736 MiB VRAM
[ OK ] ONNX Runtime CUDA — onnxruntime 1.29.0, session actually used CUDAExecutionProvider
RESULT: PyTorch CUDA = working, ONNX Runtime CUDA provider = working
```
Effective usable VRAM is ~5.5–5.7 GB, not the full 6 GB, because the desktop
compositor holds ~400 MB baseline.

No system CUDA Toolkit/cuDNN install was needed — the pip wheels for `torch` and
`onnxruntime-gpu` bundle their own CUDA 13.0 runtime, which matches the driver.

Problems: none.

Next: Milestone 2.

---

## Milestone 2 — Camera Engine

Status: **DONE**

Implemented:
- `shared/utils/camera.py`: `CameraCapture` (synchronous, V4L2 backend, MJPG
  fourcc) and `ThreadedCameraStream` (background-thread "latest frame wins"
  wrapper — Section 25's "latest frame > every frame" rule, implemented correctly
  as a mailbox pattern rather than by starving the driver's own buffer).
- `scripts/benchmark/benchmark_camera.py`: opens the real camera, measures actual
  achieved FPS (not the number the driver claims it negotiated), records mean
  frame brightness, saves a sample frame, writes `benchmarks/camera-results-*.json`.

Measured (real hardware, this room's lighting, reproduced 3× for consistency):

| Configuration | Achieved FPS | Mean brightness (0–255) |
|---|---|---|
| Driver-negotiated (claimed) | "30" | — |
| First attempt, `BUFFERSIZE=1` forced | **7.7** | — (bug, see below) |
| True auto-exposure (V4L2 mode 3), `BUFFERSIZE` untouched | **14.9** | 40.0 (dim, visible) |
| Manual short exposure (150), `BUFFERSIZE` untouched | **28.7** | 3.0 (**unusably black**) |

Two real bugs were found and fixed by investigation, not left as TODOs:

1. **Self-inflicted throttling**: the first version of `CameraCapture` forced
   `CAP_PROP_BUFFERSIZE=1` to try to implement "latest frame wins" at the driver
   level. Measured effect: this *halved* throughput (14.1 → 7.7 FPS) because the
   V4L2/libv4l path serializes each read behind a full blocking USB transfer with
   only one buffer in flight. Fixed by removing it and implementing latest-frame-wins
   correctly one layer up, in `ThreadedCameraStream`, via a background reader thread
   plus a single-slot mailbox — the camera is read at its own full native rate, the
   consumer just never sees a stale frame.

2. **Wrong OpenCV auto-exposure constant**: the commonly quoted OpenCV/V4L2
   convention (`0.25`=manual, `0.75`=auto) does **not** hold on this camera/driver/
   OpenCV build. Setting `0.75` silently failed to re-engage auto-exposure and left
   the device in whatever mode a *previous* process had set (verified by reading
   back actual frame brightness, not just FPS — a naive before/after FPS comparison
   would have missed this since both runs measured similar FPS due to leftover
   device state). The values that actually work, confirmed by brightness readback,
   are V4L2's own enum: `1` = manual, `3` = aperture-priority/auto. Documented
   in-code as empirically determined, not to be trusted blindly on other hardware.

**Real, unresolved finding (not a software bug — a physical limitation)**: in this
room's current ambient light, the C510's auto-exposure algorithm correctly extends
exposure time to maintain a visible image, which caps throughput at ~15 FPS
regardless of the requested resolution/FPS (confirmed identical behavior via an
independent `ffmpeg -f v4l2` capture, ruling out an OpenCV-specific cause). Forcing
a short manual exposure recovers ~29 FPS but the image is unusably dark — this
camera's `CAP_PROP_GAIN` appears hardware-capped around 16 regardless of the value
requested (255 was requested, 16 was what actually took effect), so there's no
gain-based compensation available on this hardware. **Getting both 30 FPS and a
usable image on this camera requires more ambient light** (a lamp/ring light) —
this is an environmental/hardware constraint, not something the pipeline can code
around. The live pipeline design accounts for this: it should measure actual
sustained capture FPS at session start and use that as the real ceiling for
downstream processing/output rate, rather than assuming 30 is always available
(Section 6's "dynamically reduce" requirement — the camera itself may be the
binding constraint, not just GPU inference time).

Not done in this milestone: an on-screen `cv2.imshow()` preview window was
deliberately skipped — the product's actual "display" surface is the browser (via
the Next.js UI / WebRTC preview in later milestones), so a native OpenCV window
would be throwaway work. Raw-frame verification was instead done by saving and
visually inspecting sample JPEGs, which is more representative of the final
architecture anyway.

Next: Milestone 3 (face detection).

---

## Cross-cutting scaffolding (done alongside Milestones 0–2)

Status: **DONE**

Implemented ahead of schedule because later milestones depend on them and they
carry no hardware/model-download risk:
- `configs/default.yaml` + `shared/schemas/config.py` (Pydantic-validated config,
  Section 18 — no scattered magic constants).
- `services/face/engine.py` (`FaceEngine` ABC) and `services/voice/engine.py`
  (`VoiceEngine` ABC) — Section 17 interfaces, so a model can be swapped later
  without touching the pipeline/API code.
- `shared/logging/logger.py` — structured JSON logging (Section 19 groundwork).
- `requirements/` split by concern (base/face/voice/dev) so installing the voice
  stack later doesn't force reinstalling the face stack.
- `scripts/setup/setup_virtual_camera.sh` / `remove_virtual_camera.sh` (Section 11)
  — written and idempotent, but **not yet run**: needs one manual `sudo apt install
  v4l2loopback-dkms v4l-utils` step from the user first (see blocker in Milestone 0).
- `scripts/setup/setup_virtual_audio.sh` / `remove_virtual_audio.sh` (Section 12)
  — **written and actually run successfully**, entirely in user space via PipeWire
  (`pactl load-module module-null-sink` + `module-remap-source`), no root needed.
  Verified three ways: (1) "AI Avatar Microphone" enumerates in `wpctl status` /
  `pactl list sources`; (2) re-running the script is idempotent (detects the
  existing device, does not duplicate it); (3) **real end-to-end audio proof** —
  played a 440 Hz test tone into the `ai_avatar_mic_sink` sink with ffmpeg and
  simultaneously recorded from the `ai_avatar_microphone` source with `parecord`;
  the captured WAV has a real, non-silent signal (mean −41.3 dB, max −20.6 dB —
  a true digital silence would read −∞ dB). The device is not a stub; audio
  written to the sink is genuinely capturable from the microphone. Not yet wired
  to real converted speech (that's Milestone 12 proper, once Milestone 8/9
  produce a voice-conversion engine to feed it).

Next: continue to Milestone 3 (InsightFace face detection).

---

## Milestone 14 (started early) — Minimal running application

Status: **PARTIAL** (by design — this is a thin vertical slice, not the full API)

The user asked to "bring the application up" before face swap/voice conversion
existed. Rather than fake a demo, stood up the smallest genuinely real slice:
a FastAPI service (`services/api/main.py`) built only on already-verified
components (Milestone 2's `ThreadedCameraStream`, the config loader), serving:

- `GET /health` — reports real camera connection state and live FPS, plus an
  explicit list of which milestones are/aren't implemented (so the endpoint
  can't be mistaken for a finished product).
- `GET /devices/video` — real `/dev/video*` enumeration.
- `GET /preview/stream` — live MJPEG passthrough of the raw webcam feed.
- `GET /` — a plain HTML page embedding the preview, viewable in a browser.

Implemented:
- Found port 8000 already bound by an unrelated process on this machine (`ss
  -tln` confirmed); moved this project's default to 8100 and updated
  `.env.example` with a note explaining why, rather than silently colliding.
- Started the server, then verified it end-to-end rather than trusting the
  "Application startup complete" log line: pulled a single JPEG out of the raw
  multipart MJPEG stream and inspected it directly — it was a real, live frame
  of the room/person in front of the camera, at the same ~15 FPS `/health`
  reports (consistent with the Milestone 2 low-light finding, not a
  regression). The verification frame was deleted immediately after — it's a
  real photo of the user's face and has no reason to persist on disk (Section 21).

Measured: `camera_fps: 15.0` via `/health`, matching Milestone 2's benchmark
for this room's current lighting — consistent, not coincidental.

Problems: none blocking. This is intentionally not Milestone 14 in full —
no `/identity`, `/voices`, `/session/start|stop`, or `/metrics` endpoints yet,
since those depend on Milestones 4, 8, and 5/9 respectively.

Next: keep this server running as the base and build Milestone 3 (face
detection) as a new endpoint/overlay on the same preview, rather than a
separate throwaway script.
