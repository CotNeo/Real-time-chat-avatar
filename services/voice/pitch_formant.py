"""
Milestone 8 — first VoiceEngine: pitch + formant conversion.

Why this rather than a neural voice-cloning model first:

* **Dependency safety.** Seed-VC pins `torch==2.4.0` and `numpy==1.26.4`; this
  project runs torch 2.13 with a carefully verified TensorRT/CUDA setup behind
  the face pipeline. Installing those pins would break working code. Neural VC
  belongs in its own process/venv, which is a larger change than getting the
  audio path working end to end.
* **Provenance.** Public RVC voice models are overwhelmingly trained on
  specific real people — celebrities, streamers — without consent. That is the
  voice equivalent of using a real person's face, the exact line this project
  stays on the right side of by using synthetic faces. This engine clones
  nobody: it reshapes the operator's own voice.
* **Latency.** Seed-VC documents ~430 ms end to end on a comparable GPU.
  Milestone 7 measured an 80 ms device baseline, leaving ~120 ms to stay inside
  Section 8's "good" band. This approach fits in single-digit milliseconds.

What it does, and why both halves matter:

Perceived vocal gender is carried by two largely independent things — **pitch**
(fundamental frequency: adult male ~85-155 Hz, female ~165-255 Hz) and
**formants** (resonances set by vocal-tract length; a shorter tract puts them
~15-20% higher). Shifting pitch alone is the classic "chipmunk" mistake: it
raises the fundamental while leaving the vocal tract sounding the same size,
which reads as a sped-up recording rather than a different speaker. Both are
shifted here, independently.

Section 7's requirements are met structurally rather than by effort: words,
timing, speaking rate, pauses, emotion and intonation all survive because the
excitation signal is only transposed, never resynthesised or re-timed.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torchaudio

from services.voice.engine import VoiceChunkResult, VoiceEngine, VoiceEngineError
from shared.schemas.identity import VoiceProfile


class PitchFormantVoiceEngine(VoiceEngine):
    """Real-time voice conversion by transposing pitch and warping formants.

    Processing happens on the GPU when available, but the work is small enough
    that CPU is also viable — this must never become the pipeline's bottleneck.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str | None = None,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pitch_semitones = 0.0
        self.formant_ratio = 1.0
        self._window = torch.hann_window(n_fft, device=self.device)
        self._profile: VoiceProfile | None = None
        # Overlap tail carried between chunks. Without it every chunk boundary
        # is a discontinuity, which is audible as a click at the block rate.
        self._tail: np.ndarray | None = None

    def load_voice(self, profile: VoiceProfile) -> None:
        """Voice profiles here are parameter presets, not cloned speakers.

        `profile.model_path` is unused: nothing is loaded from disk, so there
        is no model file whose provenance could be in question.
        """
        preset = VOICE_PRESETS.get(profile.id)
        if preset is None:
            raise VoiceEngineError(
                f"Unknown voice profile '{profile.id}'. "
                f"Available: {', '.join(sorted(VOICE_PRESETS))}"
            )
        self.pitch_semitones = preset["pitch_semitones"]
        self.formant_ratio = preset["formant_ratio"]
        self._profile = profile
        self.reset()

    def reset(self) -> None:
        self._tail = None

    def warm_up(self) -> None:
        """Pay CUDA/FFT plan setup once, so the first real chunk isn't an
        outlier in the latency numbers Section 8 asks to be measured honestly."""
        silence = np.zeros(int(self.sample_rate * 0.04), dtype=np.float32)
        self.process_audio(silence, self.sample_rate)
        self.reset()

    def _shift_formants(self, magnitude: torch.Tensor, ratio: float) -> torch.Tensor:
        """Resample the spectral envelope along the frequency axis.

        Stretching magnitudes toward higher bins simulates a shorter vocal
        tract. Doing this on the magnitude spectrum only leaves phase — and so
        the harmonic structure carrying pitch — untouched, which is what keeps
        the two controls independent.
        """
        if abs(ratio - 1.0) < 1e-3:
            return magnitude
        bins = magnitude.shape[0]
        source = torch.arange(bins, device=magnitude.device, dtype=torch.float32) / ratio
        low = source.floor().long().clamp(0, bins - 1)
        high = (low + 1).clamp(0, bins - 1)
        frac = (source - low.float()).unsqueeze(-1)
        return magnitude[low] * (1 - frac) + magnitude[high] * frac

    def process_audio(self, pcm_chunk: np.ndarray, sample_rate: int) -> VoiceChunkResult:
        if sample_rate != self.sample_rate:
            raise VoiceEngineError(
                f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz. "
                "Resample before calling, or construct the engine for this rate."
            )

        started = time.perf_counter()
        timings: dict[str, float] = {}

        audio = np.asarray(pcm_chunk, dtype=np.float32).reshape(-1)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        # Cheap gate: silence needs no conversion, and running the vocoder on
        # room noise only amplifies it.
        voiced = peak > 0.005
        if not voiced:
            timings["total"] = (time.perf_counter() - started) * 1000
            return VoiceChunkResult(output_pcm=audio, voiced=False, timings_ms=timings)

        # Prepend the previous chunk's tail so the transform sees continuous
        # signal across the boundary; the extra samples are trimmed after.
        #
        # The context must be at least `n_fft` samples, not merely "some".
        # torch.stft with center=True reflection-pads by n_fft//2 on each side
        # and refuses when the padding exceeds the input length — a 20 ms chunk
        # is 320 samples at 16 kHz while n_fft=1024 needs 512, so short chunks
        # fail outright without enough carried context. The first chunk has no
        # tail at all, so it is zero-padded to the same length and the padding
        # trimmed off with the offset.
        context = self.n_fft
        if self._tail is not None and self._tail.shape[0] >= context:
            history = self._tail[-context:]
        else:
            existing = self._tail if self._tail is not None else np.zeros(0, dtype=np.float32)
            history = np.concatenate(
                [np.zeros(context - existing.shape[0], dtype=np.float32), existing]
            )
        work = np.concatenate([history, audio])
        offset = history.shape[0]

        # Carry enough of this chunk forward to satisfy the next call.
        combined = np.concatenate([history, audio])
        self._tail = combined[-context:].copy()

        tensor = torch.from_numpy(work).to(self.device)

        stft_started = time.perf_counter()
        spectrum = torch.stft(
            tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            return_complex=True,
            center=True,
        )
        timings["stft"] = (time.perf_counter() - stft_started) * 1000

        shift_started = time.perf_counter()
        magnitude = spectrum.abs()
        phase = torch.angle(spectrum)
        magnitude = self._shift_formants(magnitude, self.formant_ratio)
        spectrum = torch.polar(magnitude, phase)

        if abs(self.pitch_semitones) > 1e-3:
            # Pitch shift = time-stretch, then resample by the same factor.
            # The stretch changes duration at constant pitch; the resample then
            # changes pitch and undoes the duration change, leaving timing and
            # speaking rate exactly as spoken (Section 7).
            #
            # A previous version tried to do this by interpolating to a shorter
            # length and straight back. That is an identity operation — it
            # smooths the signal and shifts nothing, which measured as all four
            # presets producing an identical f0. A phase vocoder is required
            # because the phase must be advanced coherently across frames;
            # plain resampling of the waveform cannot preserve duration.
            ratio = 2.0 ** (self.pitch_semitones / 12.0)
            phase_advance = torch.linspace(
                0, np.pi * self.hop_length, spectrum.shape[0], device=self.device
            ).unsqueeze(-1)
            stretched = torchaudio.functional.phase_vocoder(
                spectrum, rate=1.0 / ratio, phase_advance=phase_advance
            )
            stretched_audio = torch.istft(
                stretched,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self._window,
                center=True,
            )
            # Resample by exactly `ratio`, derived from the length the vocoder
            # actually produced — not from the length it was expected to
            # produce. phase_vocoder rounds to whole frames, so its output can
            # differ from input*ratio by a frame or two; interpolating to a
            # fixed target length instead made the effective resampling ratio
            # (actual_length / target) rather than `ratio`, which measured as
            # up to 13% pitch error and two different semitone settings landing
            # on the same output frequency.
            target_length = max(1, int(round(stretched_audio.shape[0] / ratio)))
            output = torch.nn.functional.interpolate(
                stretched_audio.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).view(-1)
            # Duration is restored by the resample; pad or trim only to absorb
            # the sub-frame rounding above.
            if output.shape[0] < tensor.shape[0]:
                output = torch.nn.functional.pad(
                    output, (0, tensor.shape[0] - output.shape[0])
                )
            else:
                output = output[: tensor.shape[0]]
        else:
            output = torch.istft(
                spectrum,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self._window,
                center=True,
                length=tensor.shape[0],
            )
        timings["convert"] = (time.perf_counter() - shift_started) * 1000

        result = output[offset : offset + audio.shape[0]].detach().cpu().numpy()
        if result.shape[0] < audio.shape[0]:
            result = np.pad(result, (0, audio.shape[0] - result.shape[0]))

        # Keep output level close to input so the conversion doesn't double as
        # an unexpected volume change.
        out_peak = float(np.abs(result).max())
        if out_peak > 1e-6:
            result = result * min(peak / out_peak, 4.0)
        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        timings["total"] = (time.perf_counter() - started) * 1000
        return VoiceChunkResult(output_pcm=result, voiced=True, timings_ms=timings)


# Presets rather than cloned identities. Ranges follow typical adult voice
# measurements: male fundamental ~85-155 Hz against female ~165-255 Hz is
# roughly 5-8 semitones, and a shorter vocal tract raises formants ~12-20%.
VOICE_PRESETS: dict[str, dict[str, float]] = {
    "female-01": {"pitch_semitones": 5.0, "formant_ratio": 1.14},
    "female-02": {"pitch_semitones": 6.5, "formant_ratio": 1.18},
    "female-03": {"pitch_semitones": 7.5, "formant_ratio": 1.22},
    "female-04": {"pitch_semitones": 4.0, "formant_ratio": 1.10},
    "neutral": {"pitch_semitones": 0.0, "formant_ratio": 1.0},
}


def available_voices() -> list[VoiceProfile]:
    labels = {
        "female-01": "Female Voice 01 (soft)",
        "female-02": "Female Voice 02 (mid)",
        "female-03": "Female Voice 03 (bright)",
        "female-04": "Female Voice 04 (low)",
        "neutral": "Unchanged (bypass)",
    }
    return [
        VoiceProfile(
            id=key,
            display_name=labels[key],
            engine="pitch-formant",
            model_path="",
            sample_rate=16000,
        )
        for key in VOICE_PRESETS
    ]
