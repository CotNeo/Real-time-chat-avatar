# Real-Time AI Avatar

A local-first proof-of-concept: apply an identity from 1–5 reference photos onto
your live webcam feed in real time, convert your live speech into a chosen
target voice in real time, and expose the result to Linux as two standard
devices — **AI Avatar Camera** and **AI Avatar Microphone** — so any application
that can pick a camera/microphone can use them.

This is a personal research/learning project for one user on one Ubuntu
machine. No billing, no multi-tenancy, no Kubernetes — see `ARCHITECTURE.md`
for why.

**Current status: system audit, CUDA verification, camera capture, and live
face detection are done and measured on real hardware, plus a minimal running
FastAPI app with a browser preview. Face identity/swap, voice conversion, the
virtual camera, and the web UI are not built yet.** See `docs/PROGRESS.md`
for the exact, honest state of every milestone — don't trust marketing copy
over that file.

Right now, with the server running (`python -m uvicorn services.api.main:app
--port 8100`), opening `http://localhost:8100` in a browser shows your live
webcam feed with a real-time SCRFD face detection overlay (bounding box,
5-point landmarks, confidence, FPS) — no face swap yet, just detection.

## 1. Overview

```
Reference images ──┐
                    ├─► Identity ──► Face pipeline ──► v4l2loopback ──► AI Avatar Camera
Webcam ─────────────┘

Voice profile ──────┐
                     ├─► Voice conversion ──► PipeWire ──► AI Avatar Microphone
Microphone ──────────┘
```

Full diagrams (system context, video/audio/WebRTC pipelines, sequence
diagrams) are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 2. Architecture diagram

See [`ARCHITECTURE.md`](ARCHITECTURE.md) — Mermaid diagrams render directly on
GitHub.

## 3. Hardware requirements

Developed and measured against:

| Component | This machine | Notes |
|---|---|---|
| GPU | NVIDIA RTX 2060, 6 GB VRAM | Effective budget ~5.5 GB once the desktop compositor's own usage is subtracted — see `docs/ENVIRONMENT_REPORT.md` |
| CPU | 6-core/12-thread mid-range desktop CPU | i5-10400F here |
| RAM | 16 GB | |
| Disk | **Watch this closely.** ML deps + models can total 10–15 GB; this project needs real headroom, not just "some" |
| Webcam | Any UVC webcam | See the low-light caveat in section 12 (Troubleshooting) — cheap sensors throttle FPS in dim rooms |
| Mic | Any PipeWire/ALSA input | |

A different GPU/VRAM size works too, but the config defaults (`configs/default.yaml`)
and the "don't over-engineer for 12–24 GB VRAM" framing in `ARCHITECTURE.md`
assume something in the 6 GB class.

## 4. Ubuntu requirements

- Ubuntu 22.04 or 24.04 LTS (built against 24.04.4)
- A recent NVIDIA driver with `nvidia-smi` working (built against driver 580.x,
  CUDA 13.0 reported)
- PipeWire (with the `pipewire-pulse` compatibility layer) for audio — the
  default on modern Ubuntu
- `sudo` access for one one-time step (installing `v4l2loopback-dkms` — see
  section 8)

## 5. Installation

```bash
git clone <this-repo-url> realtime-ai-avatar
cd realtime-ai-avatar

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Base + face pipeline (Milestones 1-6)
pip install -r requirements/face.txt

# Add these once you reach the corresponding milestone — kept separate so
# iterating on one doesn't force reinstalling the other:
# pip install -r requirements/voice.txt
# pip install -r requirements/webrtc.txt

# Dev/test tooling
pip install -r requirements/dev.txt

cp .env.example .env
```

Frontend (once `apps/web` exists — Milestone 16, not yet built):
```bash
cd apps/web && npm install
```

## 6. CUDA validation

Don't take CUDA availability on faith — run the actual verification script,
which exercises the GPU (a real matmul, a real ONNX Runtime CUDA session) not
just checks a boolean flag:

```bash
source .venv/bin/activate
python scripts/setup/verify_cuda.py
```

Expected output on a working setup:
```
[ OK ] PyTorch CUDA
       torch 2.13.0 (cu13.0)
       GPU: NVIDIA GeForce RTX 2060  (compute capability 7.5)
       VRAM: 5736 MiB
[ OK ] ONNX Runtime CUDA
       onnxruntime 1.29.0
       providers available: [...]
       session actually used: ['CUDAExecutionProvider', 'CPUExecutionProvider']
RESULT: PyTorch CUDA = working, ONNX Runtime CUDA provider = working
```

If either check fails, the script prints numbered remediation steps naming the
actual likely cause (Section 20's "descriptive errors, not Error 500" rule
applies throughout this project) — follow those before opening an issue.

## 7. Camera validation

```bash
source .venv/bin/activate
python scripts/benchmark/benchmark_camera.py --device 0 --seconds 5
```

Writes `benchmarks/camera-results-auto-exposure.json` (or `-manual-exposure`
if you pass `--manual-exposure <value>`) and a sample JPEG so you can visually
confirm the camera is actually capturing your room, not silently failing.
**Read the achieved FPS, not just the "requested" FPS** — see the Troubleshooting
section below if it's well under 30.

## 8. Model installation

Not yet applicable — no face/voice models are integrated yet (Milestones 3–9).
Once they are, models install via:
```bash
python scripts/models.py install face-default
python scripts/models.py install voice-female-01
```
against `models/registry.yaml`, never by auto-downloading from an arbitrary
URL at runtime (Section 23). This section will be filled in with real commands
once that script exists — check `docs/PROGRESS.md` for the current milestone.

## 9. Virtual camera setup

Needs root **once**, for the `v4l2loopback` kernel module. This machine's
`sudo` requires an interactive password that an automated agent can't supply,
so run this yourself in a real terminal:

```bash
./scripts/setup/setup_virtual_camera.sh
```

It installs `v4l2loopback-dkms` + `v4l-utils` if missing, loads the module as
`/dev/video10` labeled "AI Avatar Camera", and is idempotent — safe to re-run.
To remove it:
```bash
./scripts/setup/remove_virtual_camera.sh
```

## 10. Virtual microphone setup

Entirely user-space via PipeWire — **no root needed**, already verified
end-to-end on this machine (see `docs/PROGRESS.md`, Milestone 12):

```bash
./scripts/setup/setup_virtual_audio.sh
```

Creates a `ai_avatar_mic_sink` sink (where the AI voice engine will write) and
remaps its monitor into a source named "AI Avatar Microphone" that any app's
microphone dropdown can select. Verify:
```bash
wpctl status   # look for "AI Avatar Microphone" under Audio > Sources
```
Remove with:
```bash
./scripts/setup/remove_virtual_audio.sh
```
This only ever touches the two PipeWire modules it created — your real
microphone and default devices are never modified.

## 11. Running the project

Not yet applicable end-to-end — the FastAPI service (Milestone 14) and the web
UI (Milestone 16) don't exist yet. Today, run the individual verification
scripts above. This section will list the real `uvicorn`/`npm run dev`
commands once those milestones land — check `docs/PROGRESS.md` first.

## 12. Web UI

Not built yet (Milestone 16). Planned layout is sketched in the original
project brief; it will be a plain developer-oriented Next.js + Tailwind page,
not a marketing site.

## 13. Troubleshooting

**Camera FPS is way below what I requested.** Cheap UVC webcams often extend
exposure time in low light, which caps deliverable FPS independent of the
requested resolution/framerate — measured directly on this project's own
Logitech C510: ~15 FPS in a dim room vs. ~30 FPS with more light, and forcing
a short manual exposure without more light just trades FPS for an unusably
dark image (see the full writeup in `docs/PROGRESS.md`, Milestone 2). Fix:
add a lamp or face a window before assuming the software is at fault. You can
also pass `--manual-exposure <value>` to the camera benchmark script to see
this trade-off yourself.

**`CUDA initialization failed`.** Run `python scripts/setup/verify_cuda.py` —
it distinguishes "no GPU visible" from "GPU visible but the CUDA execution
provider silently fell back to CPU" and prints which shared library or driver
layer to check for each case, rather than a bare exception.

**`sudo: a password is required` when running a setup script.** Expected on
machines without a NOPASSWD rule — run the script yourself in an interactive
terminal rather than through an automated tool; it will prompt normally.

**PipeWire virtual mic doesn't show up in an app.** Some apps cache the device
list at startup — restart the app after running `setup_virtual_audio.sh`.

## 14. Performance

Real, measured numbers only — see `docs/PROGRESS.md` for the running log and
`benchmarks/*.json` for raw data. As of the latest milestone:

- PyTorch CUDA and ONNX Runtime CUDA both confirmed working via real inference
  (not just provider-list checks) on the RTX 2060.
- Raw camera capture: ~15 FPS in this room's ambient light with auto-exposure
  (dim but visible), ~29 FPS achievable with a forced short exposure (currently
  unusably dark without more light). See the full investigation in
  `docs/PROGRESS.md`, Milestone 2 — two real bugs were found and fixed
  (a self-inflicted buffer-size throttle, and a wrong OpenCV exposure constant).
- Face/voice inference latency: not yet measured — no model integrated.

Face and voice model benchmarks will land in `benchmarks/face-results.json` /
`benchmarks/voice-results.json` and `docs/FACE_MODEL_COMPARISON.md` /
`docs/VOICE_MODEL_COMPARISON.md` once Milestones 6 and 9 are reached.

## 15. Known limitations

- Face swap/reenactment and voice conversion are not implemented yet — this is
  scaffolding plus two verified infrastructure milestones (CUDA, camera) plus
  one verified but unfed milestone (virtual microphone), not a finished product.
- The virtual camera (`v4l2loopback`) requires a one-time manual root step this
  agent cannot perform non-interactively.
- Disk space on the development machine is tight (~27–35 GB free); large voice
  model downloads later may require freeing more space first.
- This webcam's real sustained frame rate depends heavily on ambient lighting;
  the 30 FPS target assumes reasonable lighting, not a dark room.

## Privacy

`LOCAL_ONLY=true` by default (`.env.example`) — no reference image, audio, or
video is sent to any external service unless you deliberately change this and
understand the trade-off (Section 21 of the original project brief). Reference
images are processed into an embedding and are not retained beyond a session
by design (`shared/schemas/identity.py`'s `IdentitySession` never stores the
source images). This project does not implement, and will not implement, any
mechanism to bypass biometric authentication, liveness checks, KYC, or platform
trust/access controls — it exists for consensual avatar/creative
experimentation on your own likeness or likenesses you have permission to use.
