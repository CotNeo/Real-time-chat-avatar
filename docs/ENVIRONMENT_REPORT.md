# Environment Report

Generated: 2026-08-30, via Milestone 0 system audit. This machine is the real target
device described in the project brief (RTX 2060 6GB) — not a sandbox — so figures here
are measured, not assumed.

## System

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble), kernel 6.17.0-35-generic |
| Host | cotneo-debian-linux |
| CPU | Intel Core i5-10400F @ 2.90GHz, 12 threads (6 cores + HT) |
| RAM | 15 GiB total (~748 MiB free / ~10 GiB available incl. reclaimable cache at audit time) |
| Swap | 4 GiB |
| Disk (`/`) | 457 GB total, **only ~35 GB free** after reclaiming a stale 12 GB pip cache (started at 22 GB free / 95% full) |

**Disk space is the binding constraint for this project**, more so than VRAM. A CUDA
PyTorch wheel + onnxruntime-gpu + InsightFace models + a voice-conversion checkpoint set
+ `node_modules` for the Next.js UI can plausibly total 10–15 GB. Large pre-existing
Docker images (`infra-api-gateway` 30.6 GB, an Oracle DB image 14 GB) and an Ollama model
store (19 GB under `/usr/share/ollama/.ollama`) are consuming most of the disk and are
unrelated to this project — not touched here, but flagged since they are the largest
reclaimable chunks if more headroom is needed later. Recommendation: periodically run
`pip cache purge` and avoid pulling unrelated Docker images while this project is active.

## GPU / CUDA

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB VRAM |
| Compute capability | 7.5 (Turing — has FP16 tensor cores, good target for `fp16` inference) |
| Driver | 580.159.03 (reports CUDA 13.0 as the max supported runtime) |
| `nvcc` | 12.0.140 present (partial toolkit install; not required — PyTorch/ONNX Runtime wheels bundle their own CUDA/cuDNN runtime) |
| System-wide `/usr/local/cuda` | not present |
| System `libcudnn` | not found |
| `nvidia-container-toolkit` | installed (Docker `--gpus` passthrough available if ever needed, not used for the MVP — see ARCHITECTURE.md) |
| Baseline GPU usage at audit time | ~423 MiB VRAM / 24% util already in use by the desktop compositor (Xorg + gnome-shell + Chrome), so the effective budget for this project is closer to **~5.5 GB**, not the full 6 GB |

Plan: use pip-distributed CUDA runtime wheels (`torch` cu12x build bundles its own
cuDNN/cuBLAS; `onnxruntime-gpu` will reuse those `.so` files via `LD_LIBRARY_PATH` rather
than requiring a separate system CUDA Toolkit/cuDNN install). This avoids installing a
system-wide CUDA toolkit that could conflict with the driver's reported 13.0 runtime.

## Camera

| Item | Value |
|---|---|
| Device | Logitech HD Webcam C510 (USB, vendor `046d:081d`) |
| Nodes | `/dev/video0`, `/dev/video1` (typical UVC split: capture + metadata node) |
| PipeWire | exposes it as V4L2 source node `081d` |
| Permissions | user `cotneo` is **not** in the `video` group, but `/dev/video0` carries an explicit ACL (`user:cotneo:rw-`) — direct OpenCV/V4L2 access works without sudo |
| `v4l2-ctl` (v4l-utils) | **not installed** — needed for device introspection and required for `v4l2loopback` setup tooling |
| `v4l2loopback` kernel module | **not installed / not loaded** — required for the "AI Avatar Camera" virtual device (Milestone 11) |

## Audio

| Item | Value |
|---|---|
| Audio server | PipeWire 1.0.5 (via `pipewire-pulse` compatibility, `pactl`/`wpctl` both work) |
| Capture device | Logitech HD Webcam C510 mic (USB), currently the default source |
| Playback | Built-in Audio (ALC897) analog stereo, plus HDMI via TU106 |
| Virtual devices | none yet — PipeWire `module-null-sink` / `module-remap-source` can create "AI Avatar Microphone" **entirely in user space, no root required** (unlike the virtual camera) |

## Toolchain

| Item | Value |
|---|---|
| Python | 3.12.3 (system), no project venv yet |
| pip | 24.0 |
| Node.js | v22.22.1 |
| npm | 10.9.4 |
| Docker | 29.1.5 |
| git | 2.43.0 |
| ffmpeg | 6.1.1 (full build: libx264/libx265/libopus/libsrt/etc. — sufficient for muxing/streaming needs) |
| ML packages (torch/onnxruntime/opencv/insightface/fastapi/aiortc) | **none installed** — clean slate |

## Privilege constraints

`sudo` on this account **requires an interactive password** (no NOPASSWD rule). This
agent runs non-interactively and cannot supply that password. Practical consequence:

- Anything installable and runnable as the `cotneo` user (Python venv, pip packages,
  PipeWire virtual devices, Node/Next.js) can be set up and verified directly.
- Anything requiring root (`apt install v4l2loopback-dkms v4l-utils`, `modprobe
  v4l2loopback`) **must be run manually by the user, once**, using the provided
  `scripts/setup/setup_virtual_camera.sh`. This is a one-time step; the script is
  idempotent and safe to re-run.

## Conclusions for architecture decisions

1. **No system-wide CUDA/cuDNN install** — rely on pip wheels (`torch`, `nvidia-cudnn-cu12`,
   `onnxruntime-gpu`) to avoid version drift against the 13.0-capable driver.
2. **Effective VRAM budget ≈ 5.5 GB**, not 6 GB, because the desktop compositor already
   holds ~420 MB. Face + voice engines must fit inside that with headroom for spikes.
3. **Native venv, not Docker, runs the AI engine.** Docker would add real friction for
   camera/PipeWire device passthrough with no offsetting benefit for a single-user local
   app (Section 31: don't over-engineer). Docker is left optional for isolated
   sub-tools only, not the main runtime.
4. **Virtual microphone (PipeWire) is fully user-space** and can be built and verified by
   this agent directly. **Virtual camera (v4l2loopback) needs one manual `sudo` step from
   the user** before Milestone 11 can be completed end-to-end.
