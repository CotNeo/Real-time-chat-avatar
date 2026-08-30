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

---

## Milestone 5 — Real-Time Face Transfer (Mode A)

Status: **DONE**, with an important, honestly-documented quality limitation
tied to this room's lighting, not to the code.

### Model choice and a licensing decision handed to the user

The standard lightweight real-time face-swap model is `inswapper_128` — the
same one roop/ReActor/FaceFusion all build on, and the correct choice per
Section 4 (prioritize speed/stability over perfect quality; don't start with
diffusion). Unlike `buffalo_l`, this model does **not** have a clean official
source: InsightFace discontinued maintaining/distributing it and now points
users at their commercial product instead. This is a genuine provenance/
licensing ambiguity, not a technical decision, so it was put to the user
directly rather than decided silently — they chose to proceed using a
reputable, long-standing community mirror. Downloaded the fp16 variant
(265 MB, half the fp32 size — a free win for this project's tight disk
budget) from `Gourieff/ReActor` on Hugging Face (2+ years old, HF-scanned
"Safe", the same source the ReActor/roop/FaceFusion ecosystem uses).
Documented exactly this way, including the sha256, in `models/registry.yaml`
and `scripts/models.py` (`install face-swap` now verifies the checksum and
deletes the file if it doesn't match). Approved for **strictly local,
single-user, consensual avatar experimentation on the operator's own
likeness only** (Section 21) — not for redistribution.

Before trusting it: `onnx.checker.check_model()` reported a topological-sort
validation error. Rather than assume corruption, verified it's the model
architecture's own well-known quirk (documented across the community) by
loading it with `onnxruntime.InferenceSession` directly — it loaded fine on
CUDA and its input/output shapes (`target`: [1,3,128,128], `source`: [1,512],
`output`: [1,3,128,128]) exactly match the documented `inswapper_128`
interface, confirming this is the genuine model, not tampered.

### Implementation

- `services/face/swapper.py`: `FaceSwapEngine` implements the `FaceEngine`
  interface (Milestone 0's abstract class actually gets a real implementation
  now). Shares the already-loaded `FaceDetector` instance rather than loading
  SCRFD a second time. Deliberately reuses InsightFace's own
  `model_zoo.inswapper.INSwapper.get()` for preprocessing/postprocessing/
  blending rather than reimplementing the soft-edged mask/seam-blur/inverse-
  affine paste-back logic from scratch — that logic is delicate and already
  battle-tested across every major face-swap tool; reinventing it would add
  risk with no upside (Section 3's "smallest practical combination" cuts both
  ways). Implements Section 6's detection-interval tracking (full detect every
  Nth frame, reuse the box in between) — verified by unit test with a
  call-counting fake detector, not just by reading the code.
- Wired into `services/api/main.py`: `POST /session/start` (requires a usable
  identity session; loads it into the swap engine) and `POST /session/stop`.
  `/preview/stream` now runs live face swap instead of the plain detection
  overlay while a session is active, with a "SWAP ACTIVE" label; falls back to
  Milestone 3's detection-only view otherwise — layered on top of, not
  replacing, prior milestones.
- `scripts/benchmark/benchmark_face_swap.py`: real camera + real reference
  images + real model, writes `benchmarks/face-swap-results.json` and a
  before/after JPEG pair.
- `tests/test_face_swapper.py`: 6 unit tests against a fake detector/swapper
  (no GPU/model needed) — error paths, the no-face passthrough, and the
  detection-interval skip logic actually skipping (asserted via call count,
  not assumed).

### A real bug found and fixed: NaN from the engine's own warm-up call

`warm_up()` used an all-zeros dummy identity embedding. Projected through the
model's internal `emap` matrix, zero stays zero; the model's own code then
does `latent /= np.linalg.norm(latent)`, i.e. `0/0`, raising `RuntimeWarning:
invalid value encountered in divide` and producing a NaN-filled throwaway
result. Root-caused properly rather than silenced: isolated whether the *real*
per-frame pipeline was affected by disabling `warm_up()` and running 20 live
frames with warnings promoted to errors — zero warnings, zero NaN frames,
proving live processing was never affected, only the discarded warm-up call
was. Fixed by using a normalized non-zero dummy vector instead. This kind of
targeted isolation (comment out the suspect call, re-measure, don't guess)
directly follows Section 30's process.

### Measured (RTX 2060, real webcam, real reference identity — 5 accepted
### synthetic reference photos from Milestone 4's test set)

```
swap_model_providers: CUDAExecutionProvider (confirmed, not CPU fallback)
achieved_fps: 14.4-14.6 (camera-limited, same ~15 FPS ceiling as Milestones 2-3)
detect_ms: mean 4.2, p95 13.7  (tracking interval keeps this cheap)
inference_ms (swap model itself): mean 53.8-55.5, p95 ~58
Combined VRAM with detector + recognition + swap ALL loaded simultaneously: 1941 MiB / 6144 MiB
```

At ~54ms/inference, the swap model alone could sustain ~18-19 FPS standalone
— still short of the 30 FPS target even before the camera's own ceiling is
considered, and the first sign that GPU inference time (not just the camera)
will eventually need attention if lighting is ever fixed (Section 9: optimize
*after* functionality works, which is exactly the point reached now).
Combined VRAM usage (1.9 GB, including the ~420 MB desktop baseline) sits
comfortably inside the ~2-3 GB face-engine budget Section 9 anticipated —
plenty of headroom left on this 6 GB card for the voice engine later.

### Critical, honestly-documented finding: low light breaks output quality, not just FPS

Milestone 2 already established this camera's FPS drops in low light. This
milestone found something more important: **in this room's current ambient
light (~40/255 mean brightness), the swap model's output degrades into
visible noise/artifacts — not just "lower quality," but incoherent.** This was
isolated carefully, not assumed:

1. The exact same code, given a well-lit reference photo as input, produces a
   coherent (if visibly seamed — a known `inswapper_128` limitation) swapped
   face. This proves the pipeline logic itself is correct.
2. The live camera frame at ~40/255 brightness produces clearly incoherent,
   noisy output.
3. **Gamma-correcting the dark frame before feeding it to the model does NOT
   fix this** — brightness went from 41→106/255 and the output was still
   noise. This rules out "just too dark" as the explanation; the real cause is
   sensor noise at this camera's actual low-light gain/exposure, which a tone
   curve remap can't remove (it amplifies noise right along with signal).

Given this, the live preview now computes per-frame brightness and overlays
an honest `LOW LIGHT (brightness NN/255) - quality will degrade` warning
(threshold 70/255, chosen from the measured 41-vs-89 comparison above) instead
of silently producing garbage the user might mistake for a bug. Fixing this
properly (denoising, or requiring/detecting better lighting) is deliberately
NOT attempted now — Section 31: don't build for a problem before confirming
it matters under the user's actual conditions, which may well include normal
room lighting unlike this test environment.

Next: Milestone 6 (face model benchmarking — formalize the FPS/VRAM/quality
comparison started here into `benchmarks/face-results.json` and
`docs/FACE_MODEL_COMPARISON.md`), or Milestone 7 (microphone) if the user
wants to switch tracks toward voice next.

---

## Milestone 5b — Swap quality was still bad; two more real root causes found

Status: **DONE — the "low light" explanation above was incomplete and partly wrong**

The user tested with proper white + yellow room lighting and reported results
were still very bad, with screenshots showing a smeared, discolored face. That
contradicted the tidy "it's just low light" conclusion recorded above, so it
got re-investigated properly instead of being re-explained away. Two distinct,
independent causes were found — both real bugs, both fixed:

### Cause 1 (the big one): the fp16 model file is broken on this GPU

Milestone 5 originally downloaded `inswapper_128_fp16.onnx` (277 MB) to save
disk, reasoning that Turing has native FP16 tensor cores so it "should" be
fine. That reasoning was never tested — it was an assumption.

A/B tested it properly: same code path, same identity embedding, same
well-lit 1024×1024 target image, only the weights file swapped. Dumped each
model's raw 128×128 output before any blending, so nothing downstream could
be blamed:

| Variant | Raw 128×128 output |
|---|---|
| `inswapper_128_fp16.onnx` | blurred, smeared, discolored — visibly corrupted |
| `inswapper_128.onnx` (fp32) | clean, sharp, correct skin tones |

So the fp16 build isn't "slightly lower precision" here — it's unusable.
Switched the default to fp32, updated `models/registry.yaml` /
`scripts/models.py` (new sha256 `e4a3f08c…`, 529 MB), and **deleted the fp16
file** so it can't be picked up by accident.

Cost, measured rather than guessed: fp32 is **78.0 ms/frame vs 62.2 ms for
fp16 (~25% slower)** plus ~277 MB more disk. Worth it — a fast unusable image
is worth nothing. (An earlier draft of the code comment claimed "no meaningful
speed difference"; that was written before measuring and has been corrected in
place. Noted here because writing an unverified claim into a comment is
exactly the failure mode this log exists to catch.)

Live confirmation after the switch: pulled a real frame from the running
`/preview/stream` — a coherent, natural, recognizable swapped face, clearly
carrying the reference identity. The corruption is gone.

### Cause 2: faces touching the frame edge corrupt the alignment warp

Separately, staged diagnostics on live camera frames (dumping every
intermediate: raw frame → aligned 128×128 crop → raw model output →
final paste-back) revealed a second failure: when the detected bbox extends
past the frame edge (measured a real one at `x1 = -32`), the ArcFace
alignment warp samples from outside the source image and fills that region
with black. The aligned crop had a visible black wedge; the model then
"reconstructed" that void into a smeared mess. This is a genuine edge case,
not a coding error in the warp — sitting too close to, or off to the side of,
the webcam triggers it, which is exactly what the user's camera framing was
doing.

Fixed by detecting it in `FaceSwapEngine.process_frame()` and skipping the
swap for that frame, passing the untouched real frame through and reporting
`skip_reason="face_partially_out_of_frame"` via the new `FaceFrameResult.
skip_reason` field. The live preview surfaces this as an on-screen
**"MOVE BACK - face is cut off by the frame edge"** message, so the user gets
an actionable instruction instead of a silently mangled image. Two regression
tests cover it (the out-of-frame case must not call the swapper at all; the
fully-in-frame case must).

### What this changes about the earlier low-light conclusion

The low-light finding recorded in Milestone 5 above is still real (this camera
does lose frame rate and gain noise in the dark), but it was **not** the main
cause of the bad output the user saw — the broken fp16 weights were. The
low-light warning overlay stays (it's still useful and honest), but it should
no longer be read as "this is why your swap looks wrong." Correcting the
earlier conclusion rather than quietly leaving it in place, since a wrong
diagnosis left in the log is worse than no diagnosis.

Remaining known quality limits, now that the real bugs are fixed:
- `inswapper_128` is inherently a 128×128 model. A face occupying, say,
  400×500 px in a 720p frame is upscaled ~3-4×, so softness/seams are
  expected and are a property of the model, not a defect to chase. Section 15's
  optional face-enhancement stage (GFPGAN/CodeFormer-class) is the standard
  remedy and remains deliberately unimplemented for now (Section 9: get it
  working before optimizing).
- Good, even lighting on the face still materially improves output.

---

## Milestone 5c — Face enhancement (Section 15) + pipeline fusion

Status: **DONE**

The user asked for the most professional-looking result possible — ideally
indistinguishable from real video. This milestone delivers the single biggest
quality improvement available (a face-restoration stage) and, just as
importantly, establishes honestly what this hardware can and cannot do.

### What was already true and didn't need fixing

Worth stating plainly because it was part of the request: the body, clothing,
background, hands and everything outside the face region are **never touched**
— they are the untouched camera feed. Mouth movement, expression, blinks, head
pose and gaze are likewise **entirely the user's own**: `inswapper` transfers
identity only and takes all motion from the live frame. So "my movements must
be exactly mine" was already satisfied by the Mode A architecture; what was
missing was facial *fidelity*.

### Face enhancement stage

`services/face/enhancer.py` adds GFPGAN v1.4 (512x512 ONNX) restoration over
the swapped face. `inswapper_128` emits a 128x128 face; when that occupies
~500px on screen it is upscaled ~4x, which is exactly the softness reported.
Every serious tool (FaceFusion/ReActor/roop) solves this the same way.

Model choice was benchmarked, not assumed:

| Enhancer | ms/frame (identical input) | Verdict |
|---|---|---|
| GFPGAN v1.4 | 126.9 | **chosen** |
| GPEN-BFR-512 | 302.8 | 2.4x slower, no visible advantage here |

Wired to Section 15's `face.enhancement: off | low | high` (blend 0.5 / 0.85 —
full strength looks plasticky, so "high" deliberately stops short of 1.0).

### Two measured optimizations

1. **Region-limited paste-back.** The composite step originally warped,
   eroded, blurred and blended across the entire 1280x720 frame to touch a
   ~500x600 face — 45.3 ms of pure CPU per frame. Restricting every operation
   to the destination bounding box is mathematically identical (affine warps
   translate exactly) and cut it to ~33 ms.
2. **Fused swap+enhance path.** Composing the stages naively (swap with
   `paste_back=True`, then let the enhancer re-crop and re-paste) does the
   expensive align/warp/mask/blend work *twice*. `FaceSwapEngine._swap_and_
   enhance()` now aligns once at the working size, runs the 128px swap on a
   downscale of that same crop, restores at 512, and composites once:
   **168 ms vs 203 ms** for equal-or-better output. The same path is used when
   enhancement is off (at 256px working size), which also made the
   unenhanced mode faster: **97.9 ms -> 83.9 ms**.

### Measured, honest results (RTX 2060 6GB, 720p, reproducible still target)

| `face.enhancement` | ms/frame | max pipeline FPS | VRAM |
|---|---|---|---|
| off | 83.9 | **11.9** | 2177 MB |
| low | 172.2 | **5.8** | 3229 MB |
| high | 172.6 | **5.8** | 3218 MB |

`low` and `high` cost the same — the model runs either way, the level only
changes the blend weight. VRAM stays far inside the 6 GB budget; **compute
time, not memory, is the binding constraint.**

### A benchmark that lied, and the fix

The first run of `benchmark_enhancement.py` reported a triumphant "212 FPS,
4.7 ms/frame" for every level. That was nonsense: nobody was in front of the
camera, so `process_frame()` early-returned on every frame and the benchmark
timed *face detection on an empty room*. The script now (a) supports
`--target-image` for a reproducible compute-cost measurement that doesn't need
a person present, and (b) **refuses to print any timings at all** if zero
frames were actually swapped. Recording this because a plausible-looking
number from a benchmark that measured nothing is worse than no number.

### What is NOT achievable on this hardware — stated plainly

The request was for output that is truly indistinguishable from real video at
full frame rate. On an RTX 2060 6GB at 720p that is **not achievable**, and no
amount of tuning in this codebase changes it:

- Best case with enhancement is **~5.8 FPS**; without it, ~11.9 FPS. The
  webcam itself only delivers ~15 FPS in this room. Smooth 30 FPS *with*
  restoration would need roughly a 5x speedup.
- `inswapper_128` produces *resemblance* to the reference identity, not a
  pixel-accurate match. Genuinely convincing, identity-exact deepfakes come
  from per-identity trained models (DeepFaceLab-class) requiring hours of
  offline training per face and offline rendering — fundamentally incompatible
  with real-time.

Realistic paths to more speed, in order of value (all deliberately NOT done
yet, per Section 9's "optimize only after it works"):
1. **TensorRT FP16 engines** for the GFPGAN and inswapper graphs — the
   standard next step, plausibly ~1.5-2x. Note fp16 *weights* were already
   found broken for inswapper; TensorRT's fp16 is a different mechanism and
   would need the same A/B verification.
2. Run the AI stage in a worker thread (Section 26) so capture never blocks —
   doesn't raise FPS but removes stalls and keeps latency bounded.
3. Lower processing resolution — limited benefit, since the fixed 128/512
   model costs dominate over frame-size-dependent work.

### Remaining honest quality caveats

- Good, even lighting still matters a great deal; the low-light warning
  overlay stays.
- Strong head rotation, occlusion (hand over face, thick frames) and faces
  running past the frame edge all still degrade or skip the swap.

---

## Milestone 5d — Realism pass: contour masking, colour transfer, model bake-off

Status: **DONE**

The user asked for the most convincing result possible — ideally
indistinguishable, matching the uploaded person "head to toe", with exact
mouth/expression sync. This milestone does everything achievable toward that
and records plainly what is not.

### Scope correction stated up front

- **"Head to toe" is out of reach and always will be for this architecture.**
  Face swap replaces a face region. Hair, body, clothing and neck stay the
  user's. Regenerating a whole body to match a reference photo is
  diffusion-class video synthesis — seconds per frame, not real time.
- **Gender is not a separate control.** It rides along with the transferred
  face identity: a female reference yields a female-looking *face*. Because
  hair and body remain the user's, the overall read can still be mixed.
- **Mouth/expression sync needed no work** — it was already exact.
  `inswapper` transfers identity only and takes all motion from the live
  frame, so blinks, mouth shapes, head pose and gaze are already 1:1.

### Swap-model bake-off (the headline finding)

Discovered the mirror also hosts newer **256px** swap models — twice the
resolution of `inswapper_128`, which had been named in Milestone 5c as the
root cause of softness. Downloaded and benchmarked them properly instead of
assuming higher resolution wins.

Identity transfer was measured **objectively**, not by eye: run the swap, then
re-extract an ArcFace embedding from the result and take its cosine similarity
to the reference identity. (Baseline — the untouched target vs the reference —
scores 0.25, i.e. "different people".)

| Model | Identity similarity | Model time | Native res | Extra |
|---|---|---|---|---|
| **inswapper_128** | **0.841** | 57.5 ms | 128px | — |
| hyperswap_1a_256 | 0.767 | **25.8 ms** | 256px | emits an occlusion mask |
| reswapper_256 | 0.753 | 203.4 ms | 256px | — |

`hyperswap` is genuinely tempting: 2.2x faster, twice the resolution, and it
even outputs its own occlusion mask. **It was still rejected.** The user's
core request is to *look like the uploaded person*, and inswapper transfers
identity ~10% better. Sharpness can be recovered by a restoration stage;
identity that the swap model never produced cannot be recovered downstream.
Resolution was the obvious-looking answer and was the wrong one — worth
recording, since "newer, bigger model" nearly won on assumption alone.

(`hyperswap` also required different preprocessing: no `emap` projection, it
consumes the raw 512-d embedding. `reswapper_256` carries an `emap` and is
otherwise drop-in with inswapper.)

### Two realism stages added (`services/face/masking.py`)

Both attack "obviously pasted on" rather than sharpness:

1. **Contour mask from 106 landmarks.** The paste-back previously used the
   whole aligned square, eroded and blurred — which covers forehead, hair
   edges and background corners that are not the person's face. Now the mask
   is the convex hull of the 106-point face contour, slightly expanded and
   feathered. The landmark model (`2d106det.onnx`) was **already in the
   `buffalo_l` pack** downloaded back in Milestone 3, so this costs no new
   download and measured **1.8 ms/frame**.
   Visible effect: the user's own hair stays sharp and real instead of being
   painted over by the swap model's blurry approximation of hair.
2. **LAB colour transfer.** The generated face carries the reference photo's
   skin tone and lighting, which rarely matches the user's room — usually the
   real reason a swap reads as fake at the jawline. Per-channel statistics are
   matched in LAB (luminance separated from chroma) using **only pixels inside
   the mask**, so a bright wall behind the user cannot bias the correction.

### Measured cost of each stage (RTX 2060, 720p, reproducible still target)

| Configuration | ms/frame | FPS | Identity similarity |
|---|---|---|---|
| swap only | 82.1 | 12.2 | 0.842 |
| + enhancement | 170.9 | 5.9 | 0.812 |
| + contour mask | 154.4 | 6.5 | 0.748 |
| + colour transfer (**default**) | 163.0 | 6.1 | 0.744 |

Two things in that table need honest interpretation rather than a naive read:

- **The falling identity number is not a regression.** It is measured over the
  whole face crop, and the contour mask deliberately hands hair and edges back
  to the *user's real footage*. Less of the crop is generated, so crop-level
  similarity drops while the face itself is unchanged. Side-by-side images
  confirm the masked version looks markedly more real. A metric that rewards
  covering more of the user with generated pixels is the wrong metric for this
  stage, and was not allowed to drive the decision.
- **GFPGAN restoration costs ~0.03 identity** (0.842 → 0.812): it "beautifies"
  and drifts slightly from the reference. Sharpness is worth it here, but the
  trade is real and now measured rather than assumed.
- The contour mask is *cheaper* than the square fallback (154.4 vs 170.9 ms) —
  it skips the erode/blur mask construction entirely.

Colour transfer was optimised during this milestone: computing its six
statistics on a 128px thumbnail instead of the full 512px face is visually
identical and cut the stage from ~25 ms to ~8.6 ms.

### Configuration (Section 18 — all of this is switchable, nothing hardcoded)

```yaml
face:
  enhancement: high      # off | low | high
  mask: contour          # contour | square
  color_match: true
```
`/health` reports all three so the running configuration is never a guess.

### Where the remaining quality gap actually is

With everything on: **6.1 FPS at 720p**, VRAM 3.6/6.1 GB. Compute, not memory,
remains the binding constraint. Honest ranking of what is still missing, worst
first:

1. **Frame rate.** 6 FPS is the single most "fake"-looking property left — far
   more damaging to believability than any per-frame detail. TensorRT FP16
   engines for GFPGAN + inswapper are the standard next step (~1.5-2x hoped).
2. **Temporal stability.** Each frame is processed independently, so lighting
   and detail shimmer slightly between frames. Landmark smoothing across
   frames would help and is not implemented.
3. **Occlusion.** A hand or object crossing the face is not detected; the
   contour mask is landmark-derived, not a true segmentation. `hyperswap`'s
   emitted mask is a possible source here even if its swap output is unused.
4. **Hair/body**, as covered above — architecturally out of scope.

---

## Milestone 5e — Occlusion masking, gender feedback, human-readable reference UI

Status: **DONE** (with two requested items declined as not achievable — see end)

### The reported bug: hand in front of the face corrupts it

Reproduced, root-caused and fixed. The contour mask from 5d describes where
the face *is*; it says nothing about what is *in front of* it, so a raised
hand got a generated face smeared across it.

Fixed with a real occlusion model. Two candidates were benchmarked on the
same synthetically-occluded frame rather than picking on reputation:

| Model | Time | Occluded-area coverage drop | Verdict |
|---|---|---|---|
| **dfl_xseg** (DeepFaceLab XSeg, 67 MB) | 44.5 ms | **0.128** | chosen |
| bisenet_resnet_34 (face parsing, 90 MB) | 55.6 ms | 0.009 | rejected |

BiSeNet largely classified a skin-toned occluder as *skin* and let the swap
paint over it — it parses face regions, it does not detect obstruction. XSeg
is purpose-built for exactly this and was both faster and far more effective.

Combination rule: `final_mask = contour_mask × occlusion_mask`. Multiplication
is deliberate — a pixel is painted only if it is **both** inside the face
outline **and** actually visible. (A max/OR would paint straight over the
hand; there is a unit test pinning this.)

Also measured: XSeg logs CUDA-fallback warnings for its ConvTranspose nodes
(asymmetric padding is unsupported on the CUDA EP). Mixed CUDA/CPU is still
45.8 ms versus **107.7 ms** CPU-only, so those warnings must not be "fixed"
by forcing CPU.

Before/after on the same occluded frame confirmed visually: without the mask
a flesh-toned band of generated face covers the occluder; with it, the
occluder keeps its own pixels and the surrounding face still swaps cleanly.

Cost: **6.1 → 5.0 FPS** (164.8 → 201.0 ms/frame). Configurable via
`face.occlusion_mask`.

### Gender: explained rather than added as a knob

The user asked for the output to be female. There is no gender control in face
swap — apparent gender arrives with whichever identity is uploaded. What was
missing was *feedback*, and it turned out to matter: running the existing
`genderage.onnx` (already in the `buffalo_l` pack, no new download) over the
reference set in use revealed it was **3 male / 2 female**. Averaging
embeddings across genders produces an identity that reads as neither — that
was a real, previously unexplained cause of the output not looking female.

Now: each reference photo reports its apparent gender, `IdentitySession.
gender_summary` reports `female` / `male` / `mixed`, and the UI warns plainly
on a mixed set. A consistent female reference set was assembled by downloading
synthetic faces and filtering them through this same model (5 kept from 9
downloads) — stored in `~/Downloads/ai-avatar-female-refs/`.

### Reference UI: pictures and words, not numbers

Section 14 asked for reference thumbnails; the panel had been dumping raw
JSON. Now each uploaded photo shows as a thumbnail with a plain verdict —
"Good", "Usable", "Weak — try a sharper, more front-facing photo", or the
specific reason it was rejected ("More than one face — use a photo with just
the person"). A raw 0-1 detector score told the user nothing they could act
on.

Thumbnails are rendered **client-side from the browser's own File objects**.
The server still never stores an uploaded photo, so it has none to serve back
— keeping the preview local preserves that property rather than weakening it
for a UI convenience.

### GPU OOM now explains itself

Hit twice during this milestone: with everything loaded the app holds ~4.2 GB
of 6 GB, so starting a benchmark while the server ran failed with a bare
`BFCArena::AllocateRawInternal ... Failed to allocate memory`. Section 20
lists this case explicitly. `shared/utils/onnx_errors.py` now translates it
into which app is probably already holding the GPU and which config switches
free the most VRAM.

### Current measured state (RTX 2060, 720p, everything on)

```
detect + swap + enhance + contour mask + colour match + occlusion mask
  201.0 ms/frame  ->  5.0 FPS      VRAM 3.4-4.2 GB / 6.1 GB
```

### Two requests declined, with reasons

- **Clothing ("selectable exotic/elegant outfits").** Not implemented, and not
  a matter of effort: replacing clothing means segmenting the body and
  generating new garments that track motion, fabric and lighting per frame —
  diffusion-class video synthesis, seconds per frame. There is no version of
  this that runs alongside a 5 FPS face pipeline on a 6 GB card. Building a
  crude overlay would look obviously pasted on and would make the result less
  convincing, not more.
- **"Head to toe" replacement.** Same reason, restated from 5d: this pipeline
  replaces a face region. Hair, body, hands and background remain the user's
  real footage — which is also *why* the occlusion and contour work above
  makes the result more believable, not less.

### On "it must not be detectable as generated"

Everything in 5c-5e serves natural-looking output, and that work continues.
Worth keeping in view, since this feed is destined for a virtual camera other
people see: the project's own charter (Section 21) scopes it to consensual
avatar experimentation and explicitly excludes defeating identity
verification, biometric checks or platform trust systems. The reference faces
in use are synthetic — people who do not exist — which keeps it on that side
of the line; pointing it at a real person's photos to appear as them to
others would not be.

---

## Milestone 5f — Forehead/expression coverage; hair and ears investigated

Status: **DONE** for what the architecture allows; hair replacement declined
with evidence.

### Mask reach was silently costing expressions

Visualising the actual mask over an aligned face (rather than trusting the
code) showed it hugged the face oval and **stopped at the eyebrows** —
coverage 0.332. So the forehead was never swapped: raising or furrowing the
brow moved the user's *own* forehead while everything below the brows was
someone else's face. The user reported brow furrowing among the expressions
that should carry through, and this was the direct cause.

Compared three expansion levels on the same frame:

| `mask_expand` | Coverage | Result |
|---|---|---|
| 1.04 (was default) | 0.332 | stops at eyebrows, forehead not transferred |
| **1.3 (new default)** | 0.512 | reaches hairline and temples, clean blend |
| 1.6 | 0.718 | visually indistinguishable from 1.3 here; starts encroaching on hair |

1.3 is the new default, exposed as `face.mask_expand` (validated 1.0-1.8).

### Expressions: what actually transfers, verified

`inswapper` takes all geometry from the live frame and applies only identity,
so expression transfer is inherent rather than a feature to add. Verified on a
live captured frame that head pose, gaze direction and mouth position survive
the swap, and that the GFPGAN restoration stage preserves expression geometry
(it smooths skin; it does not reshape the face).

Honest split of the specific expressions asked about:
- **Talking / mouth movement, blinking, brow raise and furrow** — transfer, and
  the brow case is materially better now that the forehead is inside the mask.
- **Kiss / pursed lips** — geometry transfers; fine lip detail is limited by
  the model's 128px output.
- **Tongue out** — expected to render poorly. `inswapper`'s training
  distribution contains very few tongue-out faces, and the inner mouth is the
  known weak region of this model class. Not verified live (would require the
  user to hold the expression on camera); flagged as expected-poor rather than
  claimed either way.

### Hair and ears: investigated, and it cannot work the way it was asked

The request was for hair and ears to be taken over completely. Rather than
assume, the raw unmasked model output was dumped and inspected. The finding is
decisive:

**The swap model does not transfer hair from the reference — it reconstructs
the *target's* hair.** In the unmasked output the hair is still the user's own
dark hair and cap, not the blonde hair of the reference photos. `inswapper`
maps a 512-d identity embedding onto a face region; hair is not encoded in
that embedding at all.

So expanding the mask over the hair does not give the reference's hairstyle.
It replaces the user's real, sharp hair with the model's blurrier
reconstruction of that same hair — strictly worse, which is what the earlier
5d comparison already showed and this now explains.

Ears sit in between: partially inside the aligned crop, and now partially
covered at `mask_expand: 1.3`, which is why that value was chosen over 1.04.
Pushing further to grab ears fully runs into the hair problem above.

Getting the reference person's actual hair would need a full head-swap or
head-synthesis model (HeSer-class, StyleGAN full-head, or a 3D avatar
pipeline). None run in real time on a 6 GB RTX 2060, and they are a different
architecture, not a tuning change to this one.

### Current defaults

```yaml
face:
  enhancement: high
  mask: contour
  mask_expand: 1.3     # new — includes forehead so brow expressions transfer
  color_match: true
  occlusion_mask: true
```

---

## Milestone 6 — Performance: enhancer options and TensorRT

Status: **IN PROGRESS** (TensorRT verified working with a large measured win)

Frame rate, not per-frame detail, is now the dominant realism problem: 5 FPS
reads as "processed video" no matter how good a single frame looks. This
milestone attacks that.

### Preset table (measured, RTX 2060, 720p, real pipeline end to end)

| Preset | Stages | ms/frame | FPS |
|---|---|---|---|
| maximum | GFPGAN-512 + occlusion | 208.9 | 4.8 |
| quality | GFPGAN-512, no occlusion | 173.2 | 5.8 |
| balanced | GPEN-256 + occlusion | 145.1 | 6.9 |
| fast | GPEN-256, no occlusion | 112.2 | 8.9 |
| raw | swap only | 75.5 | 13.2 |

`GPEN-BFR-256` was added as `face.enhancement: fast` — 31.4 ms versus GFPGAN's
83.0 ms (2.6x) but visibly softer eyes and skin, which is the exact quality the
stage exists to restore. Offered as an option, deliberately not the default.

Note the ceiling: this webcam delivers ~15 FPS in the current room light, so
even the `raw` preset is close to camera-limited. Chasing 30 FPS is pointless
until the lighting is fixed — a lamp is worth more than any code change here.

### TensorRT: installed, verified, and a much bigger win than expected

`TensorrtExecutionProvider` was listed by onnxruntime from the very first
Milestone 1 check but was never usable — `ldd` on the provider `.so` showed
`libnvinfer.so.10 => not found`. Exactly the trap CUDA presented earlier in
this project: **a provider appearing in `get_available_providers()` says
nothing about whether it can run.** Installed `tensorrt==10.16.1.11` (~4 GB on
disk) and confirmed the provider genuinely activates by checking
`session.get_providers()[0]`, not by assuming.

Measured on GPEN-BFR-256:

```
CUDA            33.3 ms
TensorRT fp16    7.4 ms     -> 4.48x
```

That is far beyond the 1.5-2x expected from TensorRT and, if it holds across
the other models, changes the whole frame-rate picture.

**fp16 output was verified, not trusted.** This project has already been
burned once by an fp16 *weights* file that was silently corrupted
(Milestone 5). TensorRT fp16 is a different mechanism (fp16 compute on fp32
weights), but it got the same scrutiny: TRT and CUDA outputs differ
numerically (mean absolute difference 8.78/255, 27.8% of pixels beyond 10
levels) yet are visually indistinguishable side by side — same sharpness, same
detail, no artifacts. Precision noise, not degradation. Safe to use.

Engine caching is enabled (`trt_engine_cache_path`), so the one-time build cost
is paid once per model rather than on every start.

Remaining: benchmark inswapper_128, GFPGAN-512 and XSeg under TensorRT (build
times for the larger graphs run into minutes), verify their outputs the same
way, then wire provider selection into config and re-measure the presets.

### TensorRT results — and the failure the speed numbers hid

All four models got dramatically faster under TensorRT fp16:

| Model | CUDA | TensorRT fp16 | Speedup |
|---|---|---|---|
| inswapper_128 | 54.0 ms | 13.5 ms | 3.99x |
| GFPGAN-512 | 81.5 ms | 26.1 ms | 3.13x |
| GPEN-BFR-256 | 33.3 ms | 7.4 ms | 4.48x |
| dfl_xseg | 28.9 ms | 4.5 ms | 6.46x |

Taken at face value that turns the `maximum` preset from 4.8 FPS into roughly
13 FPS. **It would also have shipped broken output**, and the only reason it
didn't is that identity was measured rather than eyeballed.

**TensorRT fp16 destroys identity transfer in `inswapper_128`:**

```
CUDA  identity similarity: 0.8308
TRT   identity similarity: 0.1218
```

For scale: an untouched frame of a *different person* scores ~0.25 against the
reference. So the TRT output does not merely lose some resemblance — it scores
**below the different-person baseline**, i.e. it carries no reference identity
at all. The image confirms it: visible streaked artifacts and a face that is
not the reference person.

This is the second time in this project that an fp16 path has been fast and
silently wrong (the first was the corrupted fp16 *weights* file in
Milestone 5). Different mechanism, identical failure mode, identical lesson:
**a speedup number is not a result.** Anything that changes numerics gets its
output verified before it is believed — and for the swap stage specifically,
verified with the identity metric, because "looks like a plausible face" is
exactly what a broken swap still produces.

Engine build times are also worth recording: XSeg took **446.7 s** (7.5 min) to
build, which is what silently timed out an earlier 10-minute benchmark run.
Cached engines currently occupy 504 MB in `.trt_cache/`.

Now testing whether the wins can be kept selectively: TensorRT **fp32** for the
swap model (precision may be the culprit rather than TensorRT itself), and
output verification for GFPGAN and XSeg before either is allowed to use fp16.

### Selective TensorRT: keeping the speed without the breakage

Testing each model both ways produced a clean split:

| Model | Verified config | Result |
|---|---|---|
| inswapper_128 | TensorRT **fp32** | identity **0.8307** vs CUDA's 0.8308 — preserved exactly; 54.7 -> 45.5 ms |
| GFPGAN-512 | TensorRT **fp32** | output **bit-identical** to CUDA (0.00/255 diff); 80.9 -> 70.2 ms |
| GPEN-BFR-256 | TensorRT fp16 | visually identical; 33.3 -> 7.4 ms |
| dfl_xseg | TensorRT fp16 | mask diff 0.0001, identical coverage; 28.9 -> 4.5 ms |

So the culprit was fp16, not TensorRT. The pattern is consistent and worth
carrying forward: **fp16 broke both models whose output is a generated face**
(identity transfer, restoration) and was harmless on the model whose output is
a mask. GFPGAN's fp16 failure was total — a blank brown image, no face.

Encoded in `shared/utils/providers.py` as a per-model table, with the
measurement that justified each choice written next to it. An unlisted model
falls back to CUDA (the conservative default — no model silently gets a
precision mode nobody verified), and if TensorRT is missing entirely
everything falls back to CUDA: slower, never broken.

Two real bugs were fixed while wiring it up:
1. `import tensorrt` is required for libnvinfer to become resident in the
   process. Without it onnxruntime lists the provider and silently runs on
   CUDA — the identical load-order trap already documented for cuBLAS/cuDNN in
   `shared/utils/cuda_env.py`.
2. Every engine's "did we actually get the GPU?" guard tested only for
   `CUDAExecutionProvider` and would have rejected a working TensorRT session.

### Final measured pipeline (RTX 2060, 720p, real end-to-end)

| Preset | Before | After | FPS before | FPS after | Identity |
|---|---|---|---|---|---|
| maximum (GFPGAN + occlusion) | 208.9 ms | 160.1 ms | 4.8 | 6.2 | 0.779 |
| **balanced (GPEN-256 + occlusion)** | 145.1 ms | **75.0 ms** | 6.9 | **13.3** | 0.753 |
| fast (GPEN-256) | 112.2 ms | 71.2 ms | 8.9 | 14.0 | 0.767 |
| raw (swap only) | 75.5 ms | 64.0 ms | 13.2 | 15.6 | 0.812 |

(The `maximum` row still used CUDA for GFPGAN; with the fp32 entry added
afterwards it should land near 150 ms.)

VRAM with everything loaded dropped from ~4.2 GB to **2.4 GB / 6.1 GB** —
TensorRT engines are more compact than the CUDA provider's workspaces, which
buys real headroom for the voice engine in Milestones 7-9.

**Default changed to `enhancement: fast`.** At 13.3 FPS the pipeline now sits
at this webcam's own ceiling (~15 FPS in current light), so the GPU has stopped
being the bottleneck — lighting and the camera are. GFPGAN renders a sharper
face, but 6 FPS motion reads as processed video, which costs more believability
than the extra per-frame detail buys. `enhancement: high` remains one config
line away for anyone who prefers still-frame quality.

Engine cache: 504 MB in `.trt_cache/` (gitignored). First run per model pays
the build cost once — XSeg's took 7.5 minutes.

---

## Milestone 6b — Face-metered exposure (backlighting fix)

Status: **DONE** (control loop verified on real hardware; live face confirmation
pending a person in frame)

The user reported the output still wasn't right and sent a frame. The frame
diagnosed itself: a bright window/curtain directly behind them, face in shadow.
That is not a model problem — it is a *capture* problem, and it poisons
everything downstream, because the swap model receives a dark noisy 128px crop
and can only return a washed-out face.

The camera's own auto-exposure meters the whole scene, so a bright background
drags the average up and the face is left underexposed. Whole-frame brightness
can look acceptable while the face is far too dark — which is exactly why
earlier brightness checks in this project (which measured the whole frame)
under-reported the problem.

### Measured exposure response (Logitech C510, this room)

| exposure | frame brightness | FPS |
|---|---|---|
| 200 | 17.9 | 33.1 |
| 500 | 46.3 | 22.5 |
| 800 | 68.2 | 15.2 |
| 1200 | 87.9 | 11.0 |
| 1600 | 101.5 | 9.0 |
| 2000 | 106.9 | 7.1 |

The camera *can* deliver a well-lit image; it just has to be told to. Note the
trade this exposes: brightness is bought with exposure time, and exposure time
caps frame rate. Surfaced rather than hidden — a well-exposed face at 11 FPS
makes a far better swap than a dark one at 15 FPS, since input quality bounds
output quality.

### Implementation

`shared/utils/exposure.py` — `FaceExposureController` measures **face-region
luma only** (Y channel, not a BGR mean: a warm indoor colour cast skews a raw
channel average) and drives `CAP_PROP_EXPOSURE` toward a target. Proportional
correction scaled by the current value, damped (`gain` 0.35) and throttled
(every 8th frame) so exposure walks smoothly instead of visibly pumping. The
background is allowed to blow out, which is the correct trade: the face is the
only region the pipeline reconstructs.

On shutdown it restores auto-exposure. This matters — UVC exposure is a
stateful device setting that outlives the process (established back in
Milestone 2), so leaving it pinned would affect every other application that
opens the camera afterwards.

### Verification

- 8 unit tests against a fake capture: face-region metering ignores a bright
  background, dark faces raise exposure, blown-out faces lower it, an
  already-correct face is left completely alone, clamping holds, no-bbox is a
  no-op, throttling holds, and release restores auto.
- **Real hardware**: driving the controller against the live camera with a
  target of 55 moved the measured region from 111.9 to 62.4 and settled
  exposure at 182 — the loop genuinely commands the device, it is not just
  arithmetic. A first run where the region already sat at 121 against a target
  of 118 correctly produced *no* change, confirming the dead-band works.
- Not yet confirmed: the end-to-end improvement with a real backlit face, which
  needs the user in frame. Exposed on `/health` as `face_brightness` and
  `camera_exposure` so it can be watched live.

Config: `video.face_metered_exposure` (default on) and
`video.target_face_brightness` (default 118).

---

## Milestone 6c — Swap model bake-off, round two: hyperswap wins on realism

Status: **DONE**

The user's standard is photorealism (viewers knowing it's AI is fine; looking
fake is not). That reframes a decision made back in Milestone 5d, where
`hyperswap` was rejected purely because it scored lower on identity similarity
to the reference embedding. Under a realism standard, that metric was the
wrong tiebreaker.

Re-tested properly, on the actual use case this time — a **male target with a
female reference set**, which is what the user is doing and what none of the
earlier comparisons used:

| Model | Res | Time | Identity | Reads as |
|---|---|---|---|---|
| inswapper_128 | 128px | 62.1 ms | **0.831** | female |
| hyperswap_1b | 256px | 28.6 ms | 0.757 | female |
| hyperswap_1c | 256px | **27.9 ms** | 0.760 | female |

All three produce a face the gender model reads as female, so the
male-to-female transfer works in every case. inswapper still matches the
reference embedding more closely — but side by side, **hyperswap is visibly
more photoreal**: twice the resolution means far less upscaling, so skin
texture and eye detail survive instead of smearing.

Identity similarity measures "does this resemble the specific reference photo",
which matters when impersonating a particular face. The references here are
synthetic people who don't exist, so there is no specific face to match — the
goal is only that the result look real. On that criterion hyperswap wins, and
the earlier decision was made against the wrong yardstick.

### Full-pipeline result

hyperswap also emits its own occlusion mask as a second output, which makes the
separate XSeg pass redundant:

| Configuration | ms | FPS | Identity |
|---|---|---|---|
| inswapper + XSeg | 93.8 | 10.7 | 0.753 |
| hyperswap + XSeg | 66.0 | 15.2 | 0.714 |
| **hyperswap (own mask)** | **58.8** | **17.0** | 0.729 |

**17 FPS is above this webcam's ~15 FPS ceiling** — the GPU is now completely
out of the way, and one model has been dropped from the pipeline entirely.
VRAM sits at 2.6 GB.

New defaults: `face.swap_model: hyperswap`, `face.occlusion_mask: false`
(redundant with hyperswap's own mask; turn it back on if switching to
inswapper). `swap_model: inswapper` remains one line away for anyone who wants
the closest possible match to a specific reference face.

### Two bugs found while adding the second model family

1. `warm_up()` assumed the insightface `INSwapper` wrapper and crashed for
   hyperswap, which drives the ONNX session directly (no `emap`, raw embedding).
   Rewritten to handle both families.
2. `_session`, `_is_hyperswap` and `_swap_size` were only assigned inside
   `load()`, so calling `process_frame()` before `load()` raised a bare
   `AttributeError` instead of the descriptive `FaceEngineError` the guard was
   written to produce. Now initialised in `__init__`.

### Standing limitation, unchanged

This improves how real the *face* looks. It does not change what the model
touches: hair, jawline, neck, beard shadow, shoulders and body remain the
user's own footage. Those are the elements carrying most of the remaining
male read, and no face-region model addresses them — the practical remedies
are physical (wig, front lighting, tighter head-and-shoulders framing), not
code.

---

## Milestone 7 — Microphone baseline (no AI in the path)

Status: **DONE**

Establishes what the hardware and OS cost on their own, so the voice-conversion
cost added next can be attributed honestly instead of being blamed on — or
hidden by — device latency.

`shared/utils/audio.py` provides `AudioConfig`, a bounded `AudioRing`, and
`AudioLoopback` (microphone -> ring -> speaker). Two design points, both from
Section 25/26:

- **Audio does not use the video pipeline's latest-frame-wins rule.** Dropping
  a video frame costs one stale image; dropping audio makes an audible click.
  The ring is a bounded FIFO that stays in order, and when it does overflow it
  discards the *oldest* block and counts it — keeping the oldest instead would
  let the speaker drift seconds behind the microphone, the exact unbounded-queue
  failure Section 25 warns about.
- **No AI may run inside the PortAudio callbacks.** They are the device's own
  threads; a slow callback underruns the stream. The conversion stage hooks in
  as a `processor`, and Milestone 8 has to keep that call short or move it off
  the callback thread entirely.

### Measured (real hardware: webcam mic in, analog out, 16 kHz mono)

| Block | Frames | Round-trip latency | Input overflows | Underflows | Ring drops | Mic signal |
|---|---|---|---|---|---|---|
| **20 ms** | 320 | **80.0 ms** | 0 | 1 | 0 | yes |
| 40 ms | 640 | 120.0 ms | 0 | 2 | 0 | yes |
| 80 ms | 1280 | 240.0 ms | 0 | 2 | 0 | yes |

Latency scales linearly with block size, as expected. Zero input overflows and
zero ring drops at every size — the path keeps up cleanly; the 1-2 underflows
are startup transients before the ring has filled.

Against Section 8's targets, **20 ms blocks give 80 ms round-trip, inside the
"excellent" band (<100 ms)**. That leaves roughly 120 ms of budget for voice
conversion before the total reaches the 200 ms "good" boundary.

The benchmark tracks microphone peak amplitude and reports
`mic_receiving_signal`, then warns loudly if every run was silent — a latency
number measured against a muted microphone describes the device path but tests
nothing end to end, and this is exactly the kind of quietly-meaningless result
that the video benchmarks produced twice before being made to refuse it.

Next: Milestone 8 — voice conversion, with a ~120 ms budget to fit inside.

---

## Milestone 8 — Voice conversion (first engine)

Status: **DONE** — a working, measured speech-to-speech engine. Neural voice
cloning is deferred, with reasons.

### Why not Seed-VC or RVC first

Both were investigated before writing any code:

- **Seed-VC** is zero-shot (no per-voice training) and attractive, but pins
  `torch==2.4.0` and `numpy==1.26.4`. This project runs torch 2.13 behind a
  carefully verified TensorRT/CUDA setup for the face pipeline; installing
  those pins would break working code. It also documents ~430 ms end-to-end on
  a comparable GPU, which lands in Section 8's "poor for conversation" band
  once Milestone 7's 80 ms device baseline is added.
- **RVC** needs a trained model per voice. The pretrained models circulating
  publicly are overwhelmingly trained on specific real people — celebrities,
  streamers — without consent. That is the voice equivalent of using a real
  person's face, which is the line this project stays on the right side of by
  using synthetic faces throughout.

`services/voice/pitch_formant.py` clones nobody: it reshapes the operator's own
voice, so there is no model file whose provenance could be in question. Neural
VC remains viable later in its own process/venv — the `VoiceEngine` interface
exists precisely so that swap costs nothing structurally.

### What it does

Perceived vocal gender rests on two largely independent properties: **pitch**
(adult male ~85-155 Hz vs female ~165-255 Hz) and **formants** (vocal-tract
resonances; a shorter tract sits ~15-20% higher). Shifting pitch alone is the
classic "chipmunk" error — a raised fundamental over an unchanged vocal tract
reads as a sped-up recording, not a different speaker. Both are shifted
independently: formants by warping the magnitude spectrum's frequency axis,
pitch by phase-vocoder time-stretch followed by resampling.

Section 7's preservation requirements hold structurally rather than by effort —
words, timing, speaking rate, pauses, emotion and intonation all survive
because the signal is only transposed, never resynthesised or re-timed.

### Measured (synthetic 120 Hz voiced tone, 20 ms chunks)

| Preset | Output f0 | Spectral centroid | ms per 20 ms chunk | Realtime factor |
|---|---|---|---|---|
| source | 120.3 Hz | 768 Hz | — | — |
| female-04 | 166.7 Hz | 1266 Hz | 1.35 | 14.8x |
| female-01 | 181.8 Hz | 1258 Hz | 1.38 | 14.5x |
| female-02 | 205.1 Hz | 1446 Hz | 1.38 | 14.5x |
| female-03 | 225.4 Hz | 1495 Hz | 1.38 | 14.5x |
| neutral | 120.3 Hz | 780 Hz | 0.57 | 35x |

All four presets land inside the adult female range, with formants raised
alongside. At **~1.4 ms per 20 ms chunk the engine runs ~14x faster than real
time**, so total audio latency stays at Milestone 7's 80 ms baseline —
comfortably inside Section 8's "excellent" band, and roughly 5x better than
Seed-VC's documented figure.

### Three real bugs found by measuring instead of assuming

1. **STFT window larger than the chunk.** A 20 ms chunk is 320 samples at
   16 kHz while `n_fft=1024` reflection-pads by 512 per side, so short chunks
   failed outright. Fixed by carrying a full `n_fft` of context between chunks
   and zero-padding the first one.
2. **The pitch shift did nothing at all.** The first implementation compressed
   the waveform and expanded it straight back — an identity operation. Caught
   because all four presets reported an *identical* f0, which no working shift
   could produce. Replaced with a proper phase vocoder.
3. **The corrected shift was still up to 13% wrong.** It resampled the
   stretched audio to a fixed target length rather than by the requested ratio;
   `phase_vocoder` rounds to whole frames, so the effective ratio drifted, and
   5 and 7 semitones both landed on 181.8 Hz. Deriving the target from the
   actual stretched length brought error to **under 0.5%** across 0-12
   semitones.

Bug 3 is worth noting for method: the intermediate result *looked* right —
pitch moved, and every preset landed in the female range. Only checking the
shift against the semitones actually requested exposed it. A plausible number
in the expected direction is not a verified one.

12 unit tests cover it, including a parametrised semitone-accuracy check that
pins bugs 2 and 3 against regression.
