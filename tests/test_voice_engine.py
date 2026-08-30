"""Section 24 tests for the pitch/formant voice engine.

These run on CPU with synthetic tones — no microphone, no GPU required. The
measurements against real audio hardware live in
scripts/benchmark/benchmark_audio.py and docs/PROGRESS.md, Milestone 8.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.voice.engine import VoiceEngineError
from services.voice.pitch_formant import (
    VOICE_PRESETS,
    PitchFormantVoiceEngine,
    available_voices,
)
from shared.schemas.identity import VoiceProfile

SR = 16000


def _harmonic_tone(f0: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A voiced-sounding tone: a fundamental plus harmonics, like speech."""
    t = np.arange(int(sr * seconds)) / sr
    signal = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 20))
    return (signal / np.abs(signal).max() * 0.5).astype(np.float32)


def _estimate_f0(x: np.ndarray, sr: int = SR) -> float:
    """Cepstral pitch estimate — enough resolution to check semitone accuracy,
    which autocorrelation does not have at the top of the vocal range."""
    x = x - x.mean()
    spectrum = np.log(np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-10)
    cepstrum = np.fft.irfft(spectrum)
    low, high = int(sr / 400), int(sr / 60)
    return sr / (low + int(np.argmax(cepstrum[low:high])))


def _process_all(engine, audio, block_ms=20) -> np.ndarray:
    block = int(SR * block_ms / 1000)
    out = [
        engine.process_audio(audio[i : i + block], SR).output_pcm
        for i in range(0, len(audio) - block, block)
    ]
    return np.concatenate(out)


def test_neutral_preset_leaves_audio_essentially_unchanged():
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.load_voice(VoiceProfile("neutral", "n", "pitch-formant", "", SR))
    source = _harmonic_tone(120.0)
    assert abs(_estimate_f0(_process_all(engine, source)) - 120.0) < 4


@pytest.mark.parametrize("semitones", [2.0, 5.0, 7.0, 12.0])
def test_pitch_shift_matches_the_requested_semitones(semitones):
    """Regression test for a real bug: the first implementation compressed and
    re-expanded the waveform, which is an identity operation and shifted
    nothing. The second attempt resampled to a fixed length rather than by the
    requested ratio, producing up to 13% pitch error and making two different
    semitone settings land on the same frequency."""
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.formant_ratio = 1.0
    engine.pitch_semitones = semitones
    source = _harmonic_tone(120.0, seconds=2.0)
    measured = _estimate_f0(_process_all(engine, source))
    expected = 120.0 * (2 ** (semitones / 12))
    assert abs(measured - expected) / expected < 0.04


def test_formant_shift_raises_the_spectral_centroid():
    """Formants are what actually sell a vocal-tract change; pitch alone reads
    as a sped-up recording."""
    def centroid(x):
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        return float((mag * freqs).sum() / max(mag.sum(), 1e-9))

    source = _harmonic_tone(120.0)
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.pitch_semitones = 0.0
    engine.formant_ratio = 1.0
    baseline = centroid(_process_all(engine, source))
    engine.formant_ratio = 1.25
    engine.reset()
    shifted = centroid(_process_all(engine, source))
    assert shifted > baseline * 1.1


def test_output_length_always_matches_input_length():
    """Real-time playback depends on one block in producing one block out;
    a length mismatch desynchronises the output stream."""
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.load_voice(VoiceProfile("female-02", "f", "pitch-formant", "", SR))
    for block_ms in (20, 40, 80):
        block = int(SR * block_ms / 1000)
        chunk = _harmonic_tone(130.0, seconds=block_ms / 1000)[:block]
        assert engine.process_audio(chunk, SR).output_pcm.shape[0] == block


def test_silence_is_passed_through_and_flagged_unvoiced():
    """Running the vocoder on room noise only amplifies it."""
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.load_voice(VoiceProfile("female-01", "f", "pitch-formant", "", SR))
    silence = np.zeros(320, dtype=np.float32)
    result = engine.process_audio(silence, SR)
    assert result.voiced is False
    assert np.array_equal(result.output_pcm, silence)


def test_unknown_voice_profile_is_rejected_by_name():
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    with pytest.raises(VoiceEngineError, match="Unknown voice profile"):
        engine.load_voice(VoiceProfile("does-not-exist", "x", "pitch-formant", "", SR))


def test_sample_rate_mismatch_is_rejected_rather_than_silently_wrong():
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    engine.load_voice(VoiceProfile("female-01", "f", "pitch-formant", "", SR))
    with pytest.raises(VoiceEngineError, match="48000 Hz"):
        engine.process_audio(np.zeros(960, dtype=np.float32), 48000)


def test_every_advertised_voice_is_loadable():
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    voices = available_voices()
    assert {v.id for v in voices} == set(VOICE_PRESETS)
    for voice in voices:
        engine.load_voice(voice)


def test_female_presets_land_in_the_adult_female_pitch_range():
    """The point of the presets: a male fundamental (~120 Hz) should come out
    inside the typical adult female range of roughly 165-255 Hz."""
    source = _harmonic_tone(120.0, seconds=2.0)
    engine = PitchFormantVoiceEngine(sample_rate=SR, device="cpu")
    for voice in available_voices():
        if not voice.id.startswith("female"):
            continue
        engine.load_voice(voice)
        f0 = _estimate_f0(_process_all(engine, source))
        assert 160 <= f0 <= 260, f"{voice.id} produced {f0:.1f} Hz"
