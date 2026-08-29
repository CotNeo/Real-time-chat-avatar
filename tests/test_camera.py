"""Section 24 unit tests: pure logic in shared/utils/camera.py that doesn't
need real hardware (the hardware-dependent path is exercised by
scripts/benchmark/benchmark_camera.py against the real device instead — see
docs/PROGRESS.md, Milestone 2, for those measured results)."""
from __future__ import annotations

from shared.utils.camera import CameraConfig, FpsCounter


def test_fps_counter_empty_is_zero():
    assert FpsCounter().fps == 0.0


def test_fps_counter_single_sample_is_zero():
    c = FpsCounter()
    c.tick(now=0.0)
    assert c.fps == 0.0  # can't compute a rate from one timestamp


def test_fps_counter_computes_rate_from_even_spacing():
    c = FpsCounter(window_seconds=10.0)
    for i in range(31):  # 31 samples at 30fps spacing = 1.0s span, 30 intervals
        c.tick(now=i * (1 / 30))
    assert abs(c.fps - 30.0) < 0.5


def test_fps_counter_drops_stale_samples_outside_window():
    c = FpsCounter(window_seconds=1.0)
    c.tick(now=0.0)
    c.tick(now=0.1)
    c.tick(now=5.0)  # far beyond the window — should evict the earlier two
    c.tick(now=5.05)
    # Only the last two samples (5.0, 5.05) should remain in the window.
    assert len(c._timestamps) == 2


def test_camera_config_defaults_match_product_spec():
    cfg = CameraConfig()
    assert (cfg.width, cfg.height, cfg.fps) == (1280, 720, 30)  # Section 6 target
    assert cfg.fourcc == "MJPG"
    assert cfg.manual_exposure_value is None
