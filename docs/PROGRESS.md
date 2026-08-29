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

---

## Milestone 3 — Face Detection

Status: **DONE** (pipeline verified; live "detects an actual face" confirmation
pending — see below)

Implemented:
- `scripts/models.py`: explicit, documented model installer (Section 23) —
  `python scripts/models.py install face-detection` downloads InsightFace's
  official `buffalo_l` pack (~326 MB) from its real GitHub Releases URL, prints
  source + license (**non-commercial research only** — confirmed against the
  InsightFace project's own README, fits this project exactly) before
  fetching, never auto-downloads silently at runtime.
- `services/face/detector.py`: `FaceDetector` wrapping SCRFD (loads only the
  detection ONNX file, not the full `FaceAnalysis` app with recognition/
  landmark/age/gender models it doesn't need yet — Section 3's "smallest
  practical combination"). Explicitly verifies the ONNX session actually ended
  up on `CUDAExecutionProvider` and raises a descriptive error otherwise,
  rather than trusting the request.
- `services/face/overlay.py`: shared bbox/landmark/confidence/FPS drawing,
  used by both the benchmark script and the live API so there's one
  implementation.
- `scripts/benchmark/benchmark_face_detection.py`: real camera + real model,
  writes `benchmarks/face-detection-results.json`.
- Wired live into `services/api/main.py`'s `/preview/stream` — the running
  app now draws the detection overlay on every frame, and `/health` reports
  `face_detector_loaded`, `face_detector_providers`, `last_detect_ms`.

Measured (RTX 2060, real webcam, CUDA confirmed via `actual_providers`):
```
detect_latency_ms: mean 18.0, p50 15.7-15.8, p95 17.4-17.6, max ~220-320 (outlier, likely GC/first-call)
camera_fps: 15.0 (matches Milestone 2's low-light finding — consistent)
```
At ~16 ms/frame, detection alone could sustain >60 FPS — the camera's own
~15 FPS ceiling (Milestone 2, ambient light) is the binding constraint, not
GPU inference. This is exactly the situation Section 6 anticipated: measure
the real ceiling, don't assume 30 is available.

**Two more real bugs found and fixed** (not left as TODOs):

1. **`insightface` silently broke CUDA and the headless OpenCV build.**
   `insightface` declares plain `onnxruntime` and `opencv-python` (GUI build)
   as dependencies. Both share an import path with this project's
   `onnxruntime-gpu` / `opencv-python-headless` — whichever installs *last*
   silently overwrites the other's files on disk, no error, no warning beyond
   an easy-to-miss pip line. Confirmed directly: after a plain
   `pip install insightface`, `onnxruntime.get_available_providers()` had
   dropped CUDA entirely and `cv2.getBuildInformation()` showed a QT5 GUI
   build had replaced the headless one. Fixed with
   `scripts/setup/install_face_deps.sh`, which installs insightface with
   `--no-deps` and adds only its real, non-conflicting dependencies —
   documented in `requirements/face.txt` with an explicit warning not to
   `pip install -r` it directly.

2. **`onnxruntime-gpu`'s CUDA provider needs libraries it doesn't bundle.**
   Running the face-detection benchmark standalone (no `torch` import in that
   process) failed with `libcublasLt.so.13: cannot open shared object file`,
   caught correctly by `FaceDetector.load()`'s own provider check rather than
   silently falling back to CPU — proving that check earns its place. Root
   cause: there's no system-wide CUDA Toolkit on this machine (see
   `docs/ENVIRONMENT_REPORT.md`); `onnxruntime-gpu` dlopen()s cuBLAS/cuDNN
   `.so` files that, here, only exist inside `torch`'s own `nvidia-*` pip
   dependencies. Milestone 1's `verify_cuda.py` never hit this because it
   imports `torch` first, which loads those libraries via its own baked-in
   RPATH as a side effect. **First fix attempt was wrong and is documented as
   such in `shared/utils/cuda_env.py`**: mutating
   `os.environ["LD_LIBRARY_PATH"]` after the process had already started did
   *not* work, verified directly — `get_available_providers()` still listed
   CUDA and the env var was set correctly, but session creation still fell
   back to CPU with the identical error, because glibc's dynamic linker
   doesn't re-read that variable for dlopen() once a process is running. The
   fix that actually works, verified directly: explicitly `import torch`
   before touching onnxruntime's CUDA provider (`ensure_onnxruntime_cuda_libs()`
   in `shared/utils/cuda_env.py`, called from `FaceDetector.load()`).

Not yet confirmed: an actual live face being detected. The room was empty for
every benchmark run in this session (checked directly by capturing and
inspecting a raw frame before concluding "0 faces" was correct behavior, not a
bug) — 0 false positives across ~200 combined frames of an empty room is
itself a meaningful correctness signal, but "stable live detection" of a real
face per Section 3's success criterion needs someone actually in front of the
camera. The live overlay is wired into `/preview/stream` right now specifically
so this can be confirmed by looking in the browser rather than by the agent
repeatedly capturing/inspecting frames of a person's face for its own sake.

Next: Milestone 4 (reference identity — upload, validate, align, embed 1-5
images), which can reuse this same downloaded `buffalo_l` pack's recognition
model.

---

## Milestone 4 — Reference Identity

Status: **DONE**

Implemented, matching Section 5's exact pipeline (Upload -> Face Detection ->
Quality Validation -> Face Alignment -> Identity Encoding -> Embedding
Aggregation -> Avatar Identity Session):

- `services/face/identity.py`: `IdentityEncoder` wraps the ArcFace recognition
  model (`w600k_r50.onnx`) from the *same* `buffalo_l` pack Milestone 3 already
  downloaded — no second model fetch. `align_face()` does standard 5-point
  ArcFace alignment (`insightface.utils.face_align.norm_crop`) before encoding.
  `process_reference_image()` runs the full per-image validation chain;
  `build_identity_session()` aggregates accepted images into one L2-normalized
  mean embedding.
- Quality checks are honest, simple heuristics, not trained classifiers
  (documented as such in the module) — Section 31: don't over-engineer an MVP:
  - `RESOLUTION_TOO_LOW`: smaller image dimension < 200px.
  - `NO_FACE_DETECTED` / `MULTIPLE_FACES`: from the SCRFD detector directly.
  - `EXCESSIVE_OCCLUSION`: eye-to-eye distance < 15% of bbox width — a crude
    proxy (real occlusion/extreme-angle often degenerates landmark spacing),
    explicitly not a trained occlusion classifier.
  - `TOO_BLURRY`: Laplacian variance on the aligned 112×112 crop < 60.
- `POST /identity`, `GET /identity`, `DELETE /identity` in `services/api/main.py`.
  Enforces the 1-5 image count (400 with a clear message outside that range).
  The response never includes the raw embedding or the source image — only
  filename, accept/reject, problem codes, and quality score (Section 5/21:
  reference images are never retained past the request; there's nothing in
  this codebase that writes an uploaded image to disk).
- Added a small upload form to `/`'s dev page so reference photos can be
  tested through the browser directly, without the agent ever needing to
  see/handle them.
- `tests/test_identity_pipeline.py`: 9 unit tests against fake detector/encoder
  objects (fast, no GPU/model needed) covering every rejection path,
  acceptance, embedding normalization, aggregation math, and the
  `MAX_REFERENCE_IMAGES` cap.

**Tested end-to-end against the real running API** (not just unit tests):
using scikit-image's bundled `astronaut.png` (a standard, freely-usable public
domain test image with one clear face) rather than the user's own photos —

| Test | Input | Result |
|---|---|---|
| Real face, single image | astronaut.png | accepted, quality_score 0.836, session usable |
| `GET /identity` persistence | (after above) | same session returned correctly |
| No face | 400×400 flat gray | rejected: `no_face_detected` |
| Too small | 50×50 | rejected: `resolution_too_low` |
| >5 images | 6 files | HTTP 400, "Upload between 1 and 5 reference images (got 6)" |
| 3 identical accepted images | astronaut.png ×3 | all 3 accepted, `usable: true` |
| `DELETE /identity` | — | `{"cleared": true}`, subsequent GET shows `active: false` |

All seven behaved exactly as designed. `/health` now also reports
`identity_encoder_loaded` and `identity_session_active`.

Problems: none blocking. The occlusion heuristic is intentionally crude — it
will under- and over-flag real cases; revisit with real reference photos if it
turns out to be a nuisance in practice, per Section 31's "don't over-engineer
before you know it's needed."

Next: Milestone 5 (real-time face transfer, Mode A) — this is the first
milestone that actually changes what the live preview looks like, using the
`aggregated_embedding` this milestone now produces.
