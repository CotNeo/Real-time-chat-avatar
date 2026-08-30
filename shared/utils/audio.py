"""
Real-time audio capture and playback (Milestone 7).

The video side taught a lesson that applies here too: measure the real device,
don't trust what it claims. So this module is built to be benchmarked — every
buffer is timestamped, and under/overruns are counted rather than silently
swallowed.

Design follows Section 25/26:
  - Audio uses a bounded ring buffer, NOT the video pipeline's latest-frame-wins
    rule. Dropping a video frame costs one stale image; dropping audio makes a
    click. Audio must stay continuous and in order, so back-pressure is handled
    by counting drops and reporting them, never by growing an unbounded queue.
  - Capture and playback run on PortAudio's own callback threads, so no AI work
    may ever run inside them — a slow callback underruns the device.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class AudioError(RuntimeError):
    """Raised with an actionable message (Section 20), never a bare failure."""


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    # Section 7 asks for 20/40/80 ms to be compared. Frames per buffer is
    # derived from this so the two can never disagree.
    block_ms: int = 40
    input_device: int | str | None = None
    output_device: int | str | None = None
    # Ring capacity as a multiple of the block size. Big enough to ride out a
    # scheduling hiccup, small enough that it cannot hide a systematic backlog
    # behind seconds of latency.
    ring_blocks: int = 16

    @property
    def block_frames(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class AudioStats:
    blocks_captured: int = 0
    blocks_played: int = 0
    input_overflows: int = 0  # device had data we failed to collect in time
    output_underflows: int = 0  # device wanted audio we failed to supply
    ring_drops: int = 0  # buffer was full; oldest block discarded
    capture_latency_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = self.capture_latency_ms
        return {
            "blocks_captured": self.blocks_captured,
            "blocks_played": self.blocks_played,
            "input_overflows": self.input_overflows,
            "output_underflows": self.output_underflows,
            "ring_drops": self.ring_drops,
            "capture_block_interval_ms": {
                "mean": round(float(np.mean(lat)), 2) if lat else None,
                "p95": round(float(np.percentile(lat, 95)), 2) if len(lat) > 5 else None,
                "max": round(float(np.max(lat)), 2) if lat else None,
            },
        }


class AudioRing:
    """Bounded FIFO of fixed-size blocks.

    When full it drops the *oldest* block and counts it. Dropping the oldest
    rather than refusing the newest keeps latency bounded when the consumer
    falls behind — the alternative is a queue that grows until the speaker is
    seconds behind the microphone.
    """

    def __init__(self, capacity_blocks: int) -> None:
        self._deque: deque[np.ndarray] = deque(maxlen=capacity_blocks)
        self._lock = threading.Lock()
        self.drops = 0

    def push(self, block: np.ndarray) -> None:
        with self._lock:
            if len(self._deque) == self._deque.maxlen:
                self.drops += 1
            self._deque.append(block)

    def pop(self) -> np.ndarray | None:
        with self._lock:
            return self._deque.popleft() if self._deque else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class AudioLoopback:
    """Microphone -> ring buffer -> speaker, with no processing in between.

    This is Milestone 7's baseline: it establishes how much latency the
    hardware and OS impose before any AI is added, so the voice-conversion
    cost later can be attributed honestly instead of being confused with
    device latency.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        self.stats = AudioStats()
        self._ring = AudioRing(self.config.ring_blocks)
        self._in_stream = None
        self._out_stream = None
        self._last_capture_time: float | None = None
        # Set to a callable to process each block before playback (Milestone 8).
        self.processor = None

    def _on_input(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        if status and getattr(status, "input_overflow", False):
            self.stats.input_overflows += 1
        if self._last_capture_time is not None:
            self.stats.capture_latency_ms.append((now - self._last_capture_time) * 1000)
        self._last_capture_time = now
        self.stats.blocks_captured += 1
        # Copy: PortAudio reuses this buffer after the callback returns.
        self._ring.push(indata.copy().reshape(-1))

    def _on_output(self, outdata, frames, time_info, status) -> None:
        if status and getattr(status, "output_underflow", False):
            self.stats.output_underflows += 1
        block = self._ring.pop()
        if block is None:
            # Nothing buffered yet — emit silence rather than stalling the
            # device. Counted as an underflow so it shows up in the numbers.
            outdata.fill(0)
            self.stats.output_underflows += 1
            return
        if self.processor is not None:
            block = self.processor(block)
        needed = outdata.shape[0]
        if block.shape[0] < needed:
            block = np.pad(block, (0, needed - block.shape[0]))
        outdata[:] = block[:needed].reshape(outdata.shape)
        self.stats.blocks_played += 1

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioError(
                "sounddevice is not installed.\nRun: pip install sounddevice soundfile"
            ) from e

        common = {
            "samplerate": self.config.sample_rate,
            "channels": self.config.channels,
            "blocksize": self.config.block_frames,
            "dtype": "float32",
        }
        try:
            self._in_stream = sd.InputStream(
                device=self.config.input_device, callback=self._on_input, **common
            )
            self._out_stream = sd.OutputStream(
                device=self.config.output_device, callback=self._on_output, **common
            )
            self._in_stream.start()
            self._out_stream.start()
        except Exception as e:  # noqa: BLE001 - re-raised with guidance
            raise AudioError(
                f"Could not open the audio devices: {e}\n"
                "Check:\n"
                "  1. `python -c \"import sounddevice; print(sounddevice.query_devices())\"` "
                "lists your microphone and speaker.\n"
                "  2. The requested sample rate is supported — try 48000 if 16000 fails.\n"
                "  3. PipeWire/PulseAudio is running (`pactl info`)."
            ) from e

    @property
    def measured_latency_ms(self) -> float | None:
        """Round-trip latency the devices report, plus the ring's own depth.

        Reported rather than assumed: PortAudio knows the device buffer sizes,
        and the ring adds one block per queued item on top.
        """
        if self._in_stream is None or self._out_stream is None:
            return None
        device_latency = (
            self._in_stream.latency + self._out_stream.latency
        ) * 1000
        ring_latency = len(self._ring) * self.config.block_ms
        return device_latency + ring_latency

    def stop(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                stream.stop()
                stream.close()
        self._in_stream = None
        self._out_stream = None

    def __enter__(self) -> "AudioLoopback":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
