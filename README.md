# Real-Time AI Avatar

A local-first, single-user research prototype: apply a face identity from
reference photos to your live webcam feed, and convert your live speech toward
a different vocal character — both in real time, on one Ubuntu machine with a
6 GB GPU.

Everything runs locally. No data leaves the machine.

---

## Honest summary of what this does and does not do

Read this before anything else, because the gap between the two lists is the
single most important thing about this project.

**It does:**

- Replace the **face region** — brow line to chin, plus forehead and temples —
  with an identity derived from 1-5 reference photos, at **17 FPS**.
- Preserve your own motion completely: mouth shapes, blinks, gaze, head pose
  and expression are all yours. Only identity is transferred.
- Handle occlusion: a hand in front of your face keeps its real pixels instead
  of having a generated face smeared over it.
- Match the generated face to your room's lighting, and meter camera exposure
  on your face rather than the whole scene.
- Shift your voice's pitch and formants into a different vocal range at ~14x
  real time, preserving your words, timing, pauses and intonation.

**It does not, and cannot with this approach:**

- **Change your hair.** The swap model reconstructs *your* hair, not the
  reference's. It has no hair information in its identity encoding at all.
- **Change your body, neck, jawline, shoulders or clothing.** Those are
  untouched camera footage.
- **Make you indistinguishable from a different person on a live stream.** The
  face alone is convincing; everything framing it is still you, and that is
  what a viewer reads.

If your goal is to appear convincingly as a different person on camera, this
architecture cannot get you there — see [Known limitations](#known-limitations)
for what actually would, and why it isn't a tuning problem or a GPU problem.

---

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 / 24.04 (built and measured on 24.04.4) |
| GPU | NVIDIA, CUDA-capable. Developed on an RTX 2060 6 GB |
| Driver | Recent NVIDIA driver with working `nvidia-smi` (built against 580.x) |
| Disk | **~16 GB**: venv ~9.7 GB, models ~2.1 GB, TensorRT engine cache ~3 GB |
| Audio | PipeWire (default on modern Ubuntu) |
| Camera | Any UVC webcam. Lighting matters more than the camera — see below |

---

## Install

```bash
git clone <this-repo> realtime-ai-avatar
cd realtime-ai-avatar

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Face stack. Use the script, NOT `pip install -r requirements/face.txt` —
# insightface pulls plain onnxruntime and GUI opencv, which silently overwrite
# the GPU/headless builds this project needs. The script installs around that.
./scripts/setup/install_face_deps.sh

# Audio + optional TensorRT acceleration (3-6x on the GPU models)
pip install sounddevice soundfile torchaudio
pip install tensorrt==10.16.1.11
```

Verify CUDA actually works — this exercises the GPU rather than checking a flag:

```bash
python scripts/setup/verify_cuda.py
```

## Models

Downloaded explicitly, never automatically at runtime. Each entry records its
source, licence and checksum in `models/registry.yaml`.

```bash
python scripts/models.py list
python scripts/models.py install face-detection   # SCRFD + ArcFace + landmarks
python scripts/models.py install face-swap        # inswapper_128
python scripts/models.py install face-enhancer    # GFPGAN v1.4
python scripts/models.py install face-occluder    # DeepFaceLab XSeg
```

The default swap model is `hyperswap_1c_256`, which is not in the installer —
download it manually if you want the default configuration:

```bash
curl -L -o models/face/models/inswapper/hyperswap_1c_256.onnx \
  https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/hyperswap_1c_256.onnx
```

**Licence note:** several of these models come from community mirrors with
unclear or non-commercial licences. `models/registry.yaml` states the position
for each one. This project uses them locally and does not redistribute them.

## Run

```bash
source .venv/bin/activate
python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8100
```

Open **http://localhost:8100**, upload 1-5 reference photos, then click *Start
session*. The first run builds TensorRT engines and takes several minutes
(XSeg alone takes ~7.5 minutes); they are cached in `.trt_cache/` afterwards.

Port 8100 rather than 8000 because 8000 was already occupied on the development
machine.

---

## Configuration

Everything lives in `configs/default.yaml`, with the measurement behind each
default written next to it. The settings that matter most:

```yaml
face:
  swap_model: hyperswap    # hyperswap = sharper (256px, 17 FPS)
                           # inswapper = closer match to the reference (128px)
  enhancement: fast        # off | fast | low | high  (high = sharpest, ~6 FPS)
  mask: contour            # contour follows the real face outline
  mask_expand: 1.3         # 1.3 includes the forehead, so brow expressions transfer
  color_match: true        # match the generated face to the room's light
  occlusion_mask: false    # redundant with hyperswap's own mask

video:
  face_metered_exposure: true   # expose for your face, not the bright window
  target_face_brightness: 118
```

Measured presets on the RTX 2060 at 720p:

| Configuration | ms/frame | FPS |
|---|---|---|
| hyperswap, own mask, GPEN-256 (**default**) | 58.8 | **17.0** |
| hyperswap + XSeg + GPEN-256 | 66.0 | 15.2 |
| inswapper + XSeg + GPEN-256 | 93.8 | 10.7 |
| inswapper + XSeg + GFPGAN-512 | 160.1 | 6.2 |

The webcam itself delivers ~15 FPS in typical indoor light, so the default is
no longer GPU-limited.

---

## Getting the best results

Lighting and framing matter more than any setting in this repo, and that is a
measured claim, not advice:

1. **Light your face from the front.** Sitting with a window behind you makes
   the camera expose for the window. Face-metered exposure compensates, but it
   buys brightness with exposure time, which costs frame rate.
2. **Frame head-and-shoulders.** The further out the shot goes, the more
   untouched body is visible.
3. **Use frontal, well-lit reference photos** of a single person, consistent
   with each other. The UI reports the apparent gender of your set and warns
   if it is mixed — averaging faces across genders produces an identity that
   resembles neither.
4. **Sit fully inside the frame.** A face crossing the frame edge cannot be
   aligned and the swap is skipped for that frame (the preview says so).

---

## Known limitations

**Structural — no amount of tuning or GPU changes these:**

- Hair, body, neck, jawline, shoulders and clothing are never modified. The
  swap model has no hair information in its identity encoding; feeding it a
  blonde reference and reading back the output shows your own hair, verified
  directly.
- Consequently, presenting convincingly as a different person on camera is out
  of reach. What a viewer reads is the whole frame, not the face oval.
- Reaching that goal needs full head/body video synthesis — diffusion-class
  models running at seconds per frame. **This is not a VRAM problem.** Even
  high-end datacentre GPUs do not do photoreal full-body generation at video
  frame rates today; it is a research frontier, not a purchase decision.

**Practical:**

- Poor or backlit lighting degrades output substantially. Input quality bounds
  output quality.
- The 128/256px swap models mean a large on-screen face is upscaled, so some
  softness is inherent.
- Each frame is processed independently, so detail shimmers slightly between
  frames. Temporal smoothing is not implemented.
- Tongue-out and other extreme expressions render poorly — such faces are
  barely represented in the model's training distribution.
- The virtual camera (`v4l2loopback`) was never completed; it needs a one-time
  `sudo` step. The virtual microphone works and is verified.

---

## Project status

Milestones 0-8 are complete and measured. `docs/PROGRESS.md` is the honest
record — every measurement, every wrong turn, and every bug found by testing
rather than assuming. It is the most useful file here for anyone continuing
the work.

| Milestone | Status |
|---|---|
| 0-2: environment, CUDA, camera | done |
| 3-5: detection, identity, face swap | done |
| 6: performance (TensorRT, model selection, exposure) | done |
| 7-8: audio capture, voice conversion | done |
| 9: voice benchmarking | not started |
| 11: virtual camera | scripts written, needs one manual `sudo` step |
| 12: virtual microphone | working, verified end to end |
| 14-16: full API, WebRTC, Next.js UI | partial — a dev UI exists |

**Not built:** virtual camera output, WebRTC, the Next.js frontend, and
wiring the voice engine into the live audio path.

## Privacy

- `LOCAL_ONLY=true` by default. Nothing is sent anywhere.
- **Reference photos are never written to disk.** They are decoded in memory,
  reduced to an embedding, and discarded. Thumbnails in the UI are rendered by
  your browser from your own local files.
- Camera exposure is restored to automatic on shutdown, since UVC exposure is a
  device setting that outlives the process.

This project does not implement, and must not be used for, bypassing identity
verification, biometric authentication, liveness checks, KYC or platform trust
systems. It is for consensual avatar experimentation. The reference faces used
throughout development were synthetic — generated people who do not exist —
which is the intended pattern: using a real person's likeness to represent
yourself to others without their consent is out of scope and not supported.
