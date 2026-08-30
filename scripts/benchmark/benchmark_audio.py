#!/usr/bin/env python3
"""
Milestone 7 — baseline audio latency, with no AI in the path.

Section 7 asks for 20/40/80 ms buffers to be compared, and Section 8 asks that
latency be measured rather than assumed. This establishes what the hardware and
OS cost on their own, so the voice-conversion cost added in Milestone 8 can be
attributed honestly instead of being blamed on (or hidden by) device latency.

Runs silently: it captures from the microphone and plays back, so use
headphones or expect feedback. Writes benchmarks/audio-results.json.

    source .venv/bin/activate
    python scripts/benchmark/benchmark_audio.py --seconds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from shared.utils.audio import AudioConfig, AudioError, AudioLoopback  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_one(block_ms: int, seconds: float, playback: bool) -> dict:
    config = AudioConfig(block_ms=block_ms)
    loop = AudioLoopback(config)

    signal_peak = {"value": 0.0}

    def measure_only(block: np.ndarray) -> np.ndarray:
        # Track whether the microphone is actually producing signal, so a run
        # against a muted or absent mic is visible in the results instead of
        # looking like a clean pass.
        peak = float(np.abs(block).max())
        signal_peak["value"] = max(signal_peak["value"], peak)
        return block if playback else np.zeros_like(block)

    loop.processor = measure_only

    try:
        loop.start()
    except AudioError as e:
        return {"block_ms": block_ms, "error": str(e)}

    time.sleep(seconds)
    latency = loop.measured_latency_ms
    stats = loop.stats.summary()
    loop.stop()

    stats.update(
        {
            "block_ms": block_ms,
            "block_frames": config.block_frames,
            "sample_rate": config.sample_rate,
            "reported_roundtrip_latency_ms": round(latency, 1) if latency else None,
            "mic_peak_amplitude": round(signal_peak["value"], 4),
            "mic_receiving_signal": signal_peak["value"] > 0.001,
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Actually play captured audio back (feedback risk without headphones). "
        "Off by default: the latency figures don't depend on it.",
    )
    args = parser.parse_args()

    results = [run_one(ms, args.seconds, args.playback) for ms in (20, 40, 80)]

    out = REPO_ROOT / "benchmarks" / "audio-results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"{'block':>7} {'frames':>7} {'roundtrip':>11} {'overflow':>9} {'underflow':>10} {'drops':>6} {'mic':>5}")
    print("-" * 62)
    for r in results:
        if "error" in r:
            print(f"{r['block_ms']:>5}ms  FAILED: {r['error'].splitlines()[0]}")
            continue
        print(
            f"{r['block_ms']:>5}ms {r['block_frames']:>7} "
            f"{str(r['reported_roundtrip_latency_ms']):>11} "
            f"{r['input_overflows']:>9} {r['output_underflows']:>10} "
            f"{r['ring_drops']:>6} {'yes' if r['mic_receiving_signal'] else 'NO':>5}"
        )

    if not any(r.get("mic_receiving_signal") for r in results):
        print(
            "\n[WARN] The microphone produced no signal in any run. These latency "
            "numbers describe the device path but nothing was actually spoken — "
            "not a failure of the code, but don't read them as an end-to-end test."
        )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
