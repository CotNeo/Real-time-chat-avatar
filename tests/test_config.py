"""Section 24 unit tests: config loading must validate and fail descriptively."""
from __future__ import annotations

import pytest

from shared.schemas.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads_and_validates():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.video.width == 1280
    assert cfg.video.height == 720
    assert cfg.video.fps == 30
    assert cfg.runtime.local_only is True
    assert cfg.virtual_devices.camera_name == "AI Avatar Camera"
    assert cfg.virtual_devices.microphone_name == "AI Avatar Microphone"


def test_missing_config_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_invalid_yaml_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("video: [this is not: valid: yaml")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(bad)


def test_invalid_field_value_raises_clear_error(tmp_path):
    bad = tmp_path / "bad_value.yaml"
    bad.write_text("face:\n  provider: quantum_gpu\n")  # not a valid literal
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(bad)


def test_unknown_field_is_rejected(tmp_path):
    """extra='forbid' — a typo'd config key should fail loudly, not be silently
    ignored (a scattered-magic-constants failure mode Section 18 wants to avoid)."""
    bad = tmp_path / "typo.yaml"
    bad.write_text("vidoe:\n  width: 100\n")
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(bad)
