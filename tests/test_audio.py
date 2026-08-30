"""Section 24 unit tests for the audio ring buffer and config arithmetic. The
device path is exercised by scripts/benchmark/benchmark_audio.py against real
hardware — see docs/PROGRESS.md, Milestone 7, for those measurements."""
from __future__ import annotations

import numpy as np

from shared.utils.audio import AudioConfig, AudioRing, AudioStats


def test_block_frames_follows_sample_rate_and_block_ms():
    assert AudioConfig(sample_rate=16000, block_ms=20).block_frames == 320
    assert AudioConfig(sample_rate=16000, block_ms=40).block_frames == 640
    assert AudioConfig(sample_rate=48000, block_ms=20).block_frames == 960


def test_ring_is_fifo():
    ring = AudioRing(capacity_blocks=4)
    for i in range(3):
        ring.push(np.full(4, i, dtype=np.float32))
    assert ring.pop()[0] == 0
    assert ring.pop()[0] == 1
    assert ring.pop()[0] == 2


def test_ring_pop_on_empty_returns_none_rather_than_blocking():
    """The output callback runs on PortAudio's thread — it must never block, so
    an empty ring has to return immediately and let the caller emit silence."""
    assert AudioRing(capacity_blocks=2).pop() is None


def test_full_ring_drops_oldest_and_counts_it():
    """Dropping the oldest keeps latency bounded when the consumer falls
    behind. Keeping the oldest instead would let the speaker drift seconds
    behind the microphone — the failure Section 25 warns about."""
    ring = AudioRing(capacity_blocks=3)
    for i in range(5):
        ring.push(np.full(2, i, dtype=np.float32))
    assert ring.drops == 2
    assert len(ring) == 3
    # The two oldest (0, 1) were discarded; the newest survive in order.
    assert ring.pop()[0] == 2
    assert ring.pop()[0] == 3
    assert ring.pop()[0] == 4


def test_ring_reports_its_depth():
    ring = AudioRing(capacity_blocks=8)
    assert len(ring) == 0
    ring.push(np.zeros(4, dtype=np.float32))
    ring.push(np.zeros(4, dtype=np.float32))
    assert len(ring) == 2


def test_stats_summary_handles_a_run_with_no_data():
    """Summary is called even on a failed/empty run; it must not divide by
    zero or raise on empty latency samples."""
    summary = AudioStats().summary()
    assert summary["blocks_captured"] == 0
    assert summary["capture_block_interval_ms"]["mean"] is None


def test_stats_summary_computes_interval_statistics():
    stats = AudioStats()
    stats.capture_latency_ms = [20.0] * 10 + [60.0]
    summary = stats.summary()
    assert summary["capture_block_interval_ms"]["max"] == 60.0
    assert 20.0 <= summary["capture_block_interval_ms"]["mean"] <= 25.0
