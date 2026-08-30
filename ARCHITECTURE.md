# Architecture

Real-Time AI Avatar is a single-user, local-first system: one Ubuntu machine, one
GPU, one person in front of the webcam. Every diagram below reflects that — no
distributed services, no multi-tenancy, no message brokers (Section 31).

## System Context

```mermaid
flowchart TD
    User(("You"))
    WebUI["Next.js Web UI\n(apps/web)"]
    API["FastAPI control API\n(services/api)"]
    Runtime["Real-Time Processing Engine\n(services/face + services/voice)"]
    VCam["v4l2loopback\n/dev/videoN"]
    VMic["PipeWire null-sink +\nremap-source"]
    OSCam[["AI Avatar Camera"]]
    OSMic[["AI Avatar Microphone"]]
    Apps["Any Linux app that can pick\na camera/microphone device"]

    User -- "reference images, voice pick,\nstart/stop" --> WebUI
    User -- "live webcam + mic" --> WebUI
    WebUI <-- "WebRTC media + control calls" --> API
    API --> Runtime
    Runtime -- "AI video frames" --> VCam --> OSCam
    Runtime -- "AI audio" --> VMic --> OSMic
    OSCam --> Apps
    OSMic --> Apps
```

The web UI is a control surface and preview, not the only consumer of the
output — the whole point of exposing standard OS virtual devices is that any
Linux app that lets you pick a camera/microphone can use them, independent of
the browser.

## Component Diagram

```mermaid
flowchart LR
    subgraph apps/web
        UI["dev UI (HTML)\nNext.js not built"]
    end
    subgraph apps/bridge
        Bridge["Local device bridge\n(v4l2loopback + PipeWire writers)"]
    end
    subgraph services/api
        API[FastAPI endpoints]
    end
    subgraph services/face
        FaceEngine["FaceEngine\n(interface)"]
        FaceImpl["Mode A: swap\n(InsightFace + ONNX)"]
        FaceEngine -.implemented by.-> FaceImpl
    end
    subgraph services/voice
        VoiceEngine["VoiceEngine\n(interface)"]
        VoiceImpl["PitchFormantVoiceEngine\n(neural VC deferred)"]
        VoiceEngine -.implemented by.-> VoiceImpl
    end
    subgraph services/webrtc
        RTC[aiortc session]
    end
    subgraph shared
        Config[schemas/config.py]
        Schemas[schemas/identity.py]
        Logging[logging/logger.py]
        CamUtil[utils/camera.py]
    end

    UI <--> API
    UI <--> RTC
    API --> FaceEngine
    API --> VoiceEngine
    RTC --> FaceEngine
    RTC --> VoiceEngine
    FaceEngine --> Bridge
    VoiceEngine --> Bridge
    FaceEngine --> CamUtil
    API --> Config
    API --> Schemas
    FaceEngine --> Logging
    VoiceEngine --> Logging
```

`FaceEngine` and `VoiceEngine` are abstract interfaces (`services/face/engine.py`,
`services/voice/engine.py`) precisely so Mode A can be swapped for Mode B, or one
voice backend for another, without touching the API or the bridge.

## Video Pipeline (as built)

```mermaid
flowchart TD
    Cam["Webcam /dev/video0"] --> Capture["ThreadedCameraStream\n(latest-frame-wins mailbox)"]
    Capture --> Expose["FaceExposureController\n(meters the FACE, not the scene)"]
    Expose --> Detect{"Detect this frame?\n(every Nth)"}
    Detect -- yes --> SCRFD["SCRFD detection\n+ 5-point landmarks"]
    Detect -- no --> Track["Reuse tracked bbox"]
    SCRFD --> Edge{"Face inside\nthe frame?"}
    Track --> Edge
    Edge -- no --> Passthrough["Skip swap, pass frame through\n(alignment would sample outside)"]
    Edge -- yes --> Align["Align once at working size"]
    Align --> Swap["Swap model\nhyperswap 256 / inswapper 128"]
    Swap --> Enh["Restoration\nGPEN-256 / GFPGAN-512"]
    Enh --> Mask["Mask = 106-pt contour hull\n× occlusion mask"]
    Mask --> Colour["LAB colour transfer\n(inside mask only)"]
    Colour --> Paste["Region-limited paste-back"]
    Paste --> Out["Output frame"]
```

Two things this diagram encodes that cost real measurement to learn:

* **One alignment, one paste-back.** Swap and restoration share them. Doing
  each stage's own align/warp/mask/blend measured 203 ms vs 168 ms for
  identical output.
* **The mask is a product, not a union.** A pixel is painted only if it is
  *both* inside the face contour *and* not occluded. A max/OR would paint over
  a raised hand.

Detection does not run on every frame (Section 6): a full detection pass runs
every `face.detection_interval` frames (config), and the tracked box/landmarks
carry over in between. If the camera itself can't sustain the target FPS — see
the Milestone 2 finding in `docs/PROGRESS.md`, where this exact webcam capped at
~15 FPS in low light — the pipeline's real ceiling is measured at session start
and downstream stages target that measured rate, not an assumed 30.

## Audio Pipeline (as built)

```mermaid
flowchart TD
    Mic["Microphone"] --> CB["PortAudio input callback"]
    CB --> Ring["Bounded ring buffer\n(FIFO, drops OLDEST when full)"]
    Ring --> Gate{"Peak above\nsilence gate?"}
    Gate -- no --> Pass["Pass through unchanged"]
    Gate -- yes --> STFT["STFT with carried context"]
    STFT --> Formant["Warp magnitude spectrum\n(formant / vocal-tract size)"]
    Formant --> Pitch["Phase vocoder stretch\n+ resample (pitch)"]
    Pitch --> OutCB["PortAudio output callback"]
    Pass --> OutCB
    OutCB --> Sink["ai_avatar_mic_sink"]
    Sink -. monitor .-> Remap["module-remap-source"]
    Remap --> OSMic[["AI Avatar Microphone"]]
```

Audio deliberately does **not** use the video pipeline's latest-frame-wins
rule. A dropped video frame costs one stale image; dropped audio is an audible
click. The ring stays ordered and bounded, and when it does overflow it
discards the oldest block and counts it — keeping the oldest instead would let
the speaker drift seconds behind the microphone.

No AI runs inside the PortAudio callbacks: those are the device's own threads,
and a slow callback underruns the stream.

Verified independently of any voice model (`docs/PROGRESS.md`, Milestone 12):
audio written to `ai_avatar_mic_sink` is genuinely recordable from the
`AI Avatar Microphone` source — a 440 Hz test tone round-tripped end to end.
The conversion stage exists and is measured (Milestone 8), but is **not yet
wired into this path** — connecting the engine to the live capture loop and on
into the sink is the remaining work.

## WebRTC Pipeline — DESIGN ONLY, NOT BUILT

The browser preview is served as MJPEG over HTTP today. The WebRTC path below
was designed but never implemented (Milestone 15).

```mermaid
flowchart LR
    BrowserCam["Browser camera"] -- WebRTC --> AIORTC["aiortc\n(services/webrtc)"]
    BrowserMic["Browser mic"] -- WebRTC --> AIORTC
    AIORTC --> FaceEngine
    AIORTC --> VoiceEngine
    FaceEngine --> AIORTC2["aiortc\n(output track)"]
    VoiceEngine --> AIORTC2
    AIORTC2 -- WebRTC --> BrowserPreview["Browser preview\n(Live Preview panel)"]
```

Media is decoded once on ingest and encoded once on egress — no intermediate
re-encode round-trip (Section 13). This is the browser's live-preview path
only; the actual OS-level virtual devices (the point of the project) are fed
directly by the processing engine via `apps/bridge`, not by looping back
through WebRTC.

## Sequence — Starting a Session

Partly aspirational: the identity and session steps are real, the voice
selection and virtual-device writer steps are not yet wired.

```mermaid
sequenceDiagram
    participant U as User (Web UI)
    participant API as FastAPI
    participant ID as Identity pipeline
    participant FE as FaceEngine
    participant VE as VoiceEngine
    participant Bridge as apps/bridge

    U->>API: POST /identity (1-5 reference images)
    API->>ID: validate + align + embed each image
    ID-->>API: IdentitySession (or per-image problems)
    API-->>U: accepted/rejected images, quality report
    U->>API: POST /voices/{id}/select
    API->>VE: load_voice(profile)
    U->>API: POST /session/start
    API->>FE: load_identity(session) + warm_up()
    API->>VE: warm_up()
    API->>Bridge: open virtual camera + microphone writers
    loop every camera frame
        Bridge->>FE: process_frame(frame)
        FE-->>Bridge: output frame
        Bridge->>Bridge: write to v4l2loopback
    end
    loop every audio chunk
        Bridge->>VE: process_audio(chunk)
        VE-->>Bridge: converted chunk
        Bridge->>Bridge: write to ai_avatar_mic_sink
    end
    U->>API: POST /session/stop
    API->>FE: reset()
    API->>VE: reset()
    API->>Bridge: close virtual device writers
```

## Why native processes, not Docker, for the runtime

`nvidia-container-toolkit` is installed on this machine (confirmed in
`docs/ENVIRONMENT_REPORT.md`), so containerizing the GPU workload is possible —
but camera (`/dev/video0`) and PipeWire socket passthrough into a container add
real friction (device cgroup rules, `--group-add`, mounting the user's runtime
dir) for a single-user local app with no deployment/isolation requirement to
justify it (Section 31). The AI runtime runs as native Python processes in a
venv; Docker stays available for genuinely isolated sub-tools if one ever
earns its place, not as the default execution path.

## Concurrency (Section 26)

| Worker | Responsibility | Queue discipline |
|---|---|---|
| Camera capture thread | Read camera at native rate | 1-slot mailbox, latest-frame-wins (`ThreadedCameraStream`) |
| Face processing worker | Detection/tracking + inference | Consumes latest frame only; never blocks capture |
| Video output worker | Write to v4l2loopback | Drops a frame rather than blocking on a slow writer |
| Audio capture worker | Read mic PCM | Bounded ring buffer; tracks overruns |
| Voice processing worker | VAD + conversion | Bounded queue; underrun tracked and logged, not silently stretched |
| Audio output worker | Write to PipeWire sink | Bounded queue |
| Metrics worker | Aggregate FPS/latency/VRAM | Pull-based (`/metrics`), not push, to avoid another moving queue |

A slow AI inference call must never stall camera capture or mic capture — each
has its own thread and a bounded, latest-wins (video) or bounded-FIFO-with-
overrun-tracking (audio) queue between them, per Section 25/26.
