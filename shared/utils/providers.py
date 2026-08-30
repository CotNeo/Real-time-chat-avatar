"""
Per-model execution-provider selection.

TensorRT is dramatically faster than the CUDA provider on this hardware — but
**only some models survive it**, and the failures are silent. Every entry below
was decided by running the model both ways and comparing output, never by the
speed number alone. See docs/PROGRESS.md, Milestone 6, for the measurements.

What was found:

| Model | Safe config | Why |
|---|---|---|
| inswapper_128 | TensorRT **fp32** | fp16 collapsed identity similarity from 0.831 to 0.122 — below the score an unrelated person gets (~0.25), i.e. no reference identity at all. fp32 preserves it exactly (0.8307 vs 0.8308) and is still faster. |
| GFPGAN-512 | TensorRT **fp32** | fp16 produced a blank brown image — not degraded, no face at all. fp32 is bit-identical to CUDA (0.00/255) and 1.15x faster. |
| GPEN-BFR-256 | TensorRT fp16 | Verified visually identical to CUDA; 4.48x. |
| dfl_xseg | TensorRT fp16 | Mask differs by 0.0001 mean, identical coverage; 6.46x. |

The pattern worth remembering: fp16 breaks the models whose output *is* a
generated face (identity and restoration), and is harmless on the model whose
output is a mask. Speed alone would have shipped a fast, broken avatar twice.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_CACHE_ROOT = REPO_ROOT / ".trt_cache"

# TensorRT builds an optimized engine per model on first use. That is slow —
# XSeg took 446 seconds — so engines are cached on disk and reused. Each model
# gets its own subdirectory to keep fp16 and fp32 engines from colliding.
_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _tensorrt(cache_name: str, fp16: bool) -> list:
    # Importing tensorrt is what makes libnvinfer resident in the process.
    # Without it onnxruntime lists TensorrtExecutionProvider and then silently
    # falls back to CUDA — the same load-time side-effect trap documented for
    # cuBLAS/cuDNN in shared/utils/cuda_env.py.
    import tensorrt  # noqa: F401

    cache_dir = ENGINE_CACHE_ROOT / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_fp16_enable": fp16,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


# Verified-safe configuration per model. Anything not listed falls back to CUDA
# — the conservative default, since an unverified model must not silently get a
# precision mode nobody checked.
_VERIFIED = {
    "swap": lambda: _tensorrt("swap_fp32", fp16=False),
    "occluder": lambda: _tensorrt("xseg_fp16", fp16=True),
    "enhancer_gpen256": lambda: _tensorrt("gpen256_fp16", fp16=True),
    # GFPGAN gets fp32, never fp16: fp16 returns a blank brown image. fp32
    # output is bit-identical to CUDA (0.00/255 mean difference) for a modest
    # but free 80.9 -> 70.2 ms.
    "enhancer_gfpgan": lambda: _tensorrt("gfpgan_fp32", fp16=False),
}


def providers_for(model_role: str, use_tensorrt: bool = True) -> list:
    """Execution providers for one model role.

    `use_tensorrt=False` forces the CUDA path everywhere — the escape hatch if
    a driver/TensorRT upgrade ever changes behavior, so the app can be brought
    back to a known-good state without editing code.
    """
    if not use_tensorrt:
        return list(_CUDA)
    factory = _VERIFIED.get(model_role)
    if factory is None:
        return list(_CUDA)
    try:
        return factory()
    except Exception:  # noqa: BLE001 - TensorRT missing is a normal state
        # No TensorRT installed (or it failed to load) — fall back rather than
        # refusing to start. Slower, not broken.
        return list(_CUDA)


def tensorrt_available() -> bool:
    """True when the TensorRT libraries can actually be loaded.

    Importing `tensorrt` is what makes libnvinfer resident in the process, the
    same load-time side effect trick documented in shared/utils/cuda_env.py for
    cuBLAS/cuDNN. Without it, onnxruntime lists TensorrtExecutionProvider and
    then silently falls back to CUDA.
    """
    try:
        import tensorrt  # noqa: F401
    except Exception:  # noqa: BLE001 - absence is a normal, handled state
        return False
    return True


def engine_cache_size_mb() -> float:
    if not ENGINE_CACHE_ROOT.exists():
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _dirs, files in os.walk(ENGINE_CACHE_ROOT)
        for f in files
    )
    return total / (1024 * 1024)
