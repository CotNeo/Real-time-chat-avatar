"""Turns onnxruntime's raw allocation failures into something actionable.

Hit twice while building the face pipeline: with every model loaded the app
holds ~4.2 GB of this card's 6 GB, so starting a second process (a benchmark
script while the server is up) fails with a bare
`BFCArena::AllocateRawInternal ... Failed to allocate memory`, which says
nothing about the actual cause or the fix.
"""
from __future__ import annotations


def is_out_of_memory(error: Exception) -> bool:
    text = str(error).lower()
    return "failed to allocate memory" in text or "out of memory" in text or "cuda_error_out_of_memory" in text


def describe_load_failure(model_name: str, error: Exception) -> str:
    if not is_out_of_memory(error):
        return f"Could not load {model_name}: {type(error).__name__}: {error}"
    return (
        f"Ran out of GPU memory loading {model_name}.\n"
        "This GPU has 6 GB and the full face pipeline already uses roughly 4.2 GB.\n"
        "Check:\n"
        "  1. Is another copy of this app (or a benchmark script) already "
        "running? Only one can hold the models at a time — check with "
        "`nvidia-smi` and stop the other one.\n"
        "  2. Free VRAM by turning off a stage in configs/default.yaml: "
        "`face.enhancement: off` (~1.0 GB) or `face.occlusion_mask: false` "
        "(~0.2 GB).\n"
        "  3. Close other GPU applications (browsers with hardware "
        "acceleration are common culprits)."
    )
