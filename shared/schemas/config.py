"""
Structured configuration (Section 18).

Loads configs/default.yaml, validates it with Pydantic (fails fast and
descriptively on a bad value instead of surfacing a confusing error three layers
deep in a model), and allows environment-variable overrides via the AVATAR__
prefix with "__" as the nesting separator — e.g. AVATAR__VIDEO__FPS=15.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class VideoConfig(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 30
    device_index: int = 0
    manual_exposure_value: int | None = None


class FaceConfig(BaseModel):
    provider: Literal["cuda", "cpu"] = "cuda"
    fp16: bool = True
    detection_interval: int = Field(default=3, ge=1)
    mode: Literal["swap", "reenact"] = "swap"
    enhancement: Literal["off", "low", "high"] = "off"
    # "contour" masks to the real face outline (106-point hull), leaving your
    # own hair/ears/background untouched; "square" pastes the whole aligned
    # crop, which also covers hair with the model's blurry approximation of it.
    mask: Literal["contour", "square"] = "contour"
    # Rescale the generated face's colour statistics to the live frame's, so
    # it carries the room's lighting instead of the reference photo's.
    color_match: bool = True
    # Detect things in front of the face (a hand, a mug) and leave their real
    # pixels alone instead of painting the swap over them. Costs ~46 ms/frame.
    occlusion_mask: bool = True


class VoiceConfig(BaseModel):
    engine: str | None = None
    buffer_ms: int = 40
    profile_latency_mode: Literal["low_latency", "balanced", "stable"] = "balanced"
    pitch_shift_semitones: int = 0


class RuntimeConfig(BaseModel):
    device: Literal["cuda", "cpu"] = "cuda"
    local_only: bool = True
    log_level: str = "info"
    log_format: Literal["json", "console"] = "json"


class PathsConfig(BaseModel):
    models_dir: str = "models"
    benchmarks_dir: str = "benchmarks"


class VirtualDevicesConfig(BaseModel):
    camera_name: str = "AI Avatar Camera"
    microphone_name: str = "AI Avatar Microphone"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AVATAR__", env_nested_delimiter="__", extra="forbid"
    )

    video: VideoConfig = VideoConfig()
    face: FaceConfig = FaceConfig()
    voice: VoiceConfig = VoiceConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    paths: PathsConfig = PathsConfig()
    virtual_devices: VirtualDevicesConfig = VirtualDevicesConfig()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load + validate config. Raises a clear error naming the file and field
    on bad YAML or a bad value — never a bare stack trace three layers deep."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected the default config at {DEFAULT_CONFIG_PATH}. "
            "If you moved it, pass the new path explicitly."
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    try:
        return AppConfig(**raw)
    except Exception as e:  # noqa: BLE001 - re-raise with the offending file named
        raise ValueError(f"Invalid configuration in {path}: {e}") from e
