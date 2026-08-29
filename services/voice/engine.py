"""
VoiceEngine interface (Section 17).

Any voice-conversion backend (RVC, Seed-VC, ... — chosen by benchmarking in
Milestone 9, see docs/VOICE_MODEL_COMPARISON.md once it exists) implements
this. The realtime audio loop and the API depend only on this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from shared.schemas.identity import VoiceProfile


@dataclass
class VoiceChunkResult:
    output_pcm: np.ndarray  # converted audio, same duration as input
    voiced: bool  # whether VAD judged this chunk as speech (vs. silence/noise)
    timings_ms: dict[str, float]  # e.g. {"vad": 0.4, "feature_extract": 3.1, "infer": 22.0}


class VoiceEngineError(RuntimeError):
    """Raised with an actionable message (Section 20) — e.g. voice model
    missing, GPU OOM, unsupported sample rate."""


class VoiceEngine(ABC):
    @abstractmethod
    def load_voice(self, profile: VoiceProfile) -> None:
        """Load/activate a target voice profile. Must raise VoiceEngineError
        naming the missing file/model if `profile.model_path` doesn't exist —
        never fail with a bare FileNotFoundError three layers down."""

    @abstractmethod
    def process_audio(self, pcm_chunk: np.ndarray, sample_rate: int) -> VoiceChunkResult:
        """Convert one fixed-size PCM chunk (mono, float32, `sample_rate` Hz —
        chunk size is a pipeline concern, set by configs/default.yaml
        voice.buffer_ms, not by the engine). Must preserve chunk duration:
        real-time playback depends on 1 chunk in == 1 chunk out, same length."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any streaming state (e.g. pitch tracker history, RNN hidden
        state) — call between sessions or after a long silence gap."""

    @abstractmethod
    def warm_up(self) -> None:
        """Run one dummy inference at load time so the first real chunk isn't
        skewed by one-time CUDA/model warm-up cost (see FaceEngine.warm_up for
        the same rationale — Section 8 latency numbers must be steady-state)."""
