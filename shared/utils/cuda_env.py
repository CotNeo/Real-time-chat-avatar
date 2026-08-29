"""
Fixes a real, reproduced failure: `onnxruntime-gpu`'s CUDAExecutionProvider
does not bundle its own cuBLAS/cuDNN — it dlopen()s shared libraries that, on
this machine, only exist inside the `nvidia-*` pip packages that `torch`
depends on (there is no system-wide CUDA Toolkit install here — see
docs/ENVIRONMENT_REPORT.md). A process that imports `onnxruntime` (directly,
or transitively through `insightface`) WITHOUT ever importing `torch` hits
this for real: confirmed running
scripts/benchmark/benchmark_face_detection.py, which failed with
`libcublasLt.so.13: cannot open shared object file`, correctly caught by
FaceDetector.load()'s own provider check rather than silently falling back to
CPU (that check is the whole reason this was caught at all).

IMPORTANT — the first fix attempted here was WRONG and is worth recording so
it isn't tried again: mutating `os.environ["LD_LIBRARY_PATH"]` from inside an
already-running Python process does NOT work. Verified directly: the env var
was set correctly and readable via `os.environ`, `get_available_providers()`
even still listed CUDAExecutionProvider as compiled in, but actual
`InferenceSession` creation still silently fell back to CPU with the exact
same "file not found" error. Reason: glibc's dynamic linker resolves a
dlopen() by soname against a search-path table that is effectively fixed once
established early in process life; mutating the environment variable
afterward doesn't retroactively change it.

What actually works, verified directly (see docs/PROGRESS.md, Milestone 3):
`import torch` before anything touches onnxruntime's CUDA provider. torch's
own native extension loads its bundled cuBLAS/cuDNN `.so` files via an
absolute RPATH baked into the wheel at build time (not a search path lookup),
which makes them resident in the process. onnxruntime's later dlopen() for
the same soname (e.g. `libcublasLt.so.13`) then succeeds because the dynamic
linker reuses the already-loaded library instead of searching for it again —
this has nothing to do with environment variables at all.

Call `ensure_onnxruntime_cuda_libs()` before importing `onnxruntime` (directly
or via insightface) in any entrypoint that might not otherwise import `torch`
first. It's a no-op if torch is already imported/unavailable — this project
already depends on torch (requirements/base.txt), so the "cost" of this import
is not optional overhead, it's a dependency this codebase pays elsewhere
anyway.
"""
from __future__ import annotations


def ensure_onnxruntime_cuda_libs() -> None:
    import torch  # noqa: F401 — imported for its load-time side effect only,
    # see module docstring. Do not remove this import even though `torch`
    # itself is unused: onnxruntime's CUDAExecutionProvider silently falls
    # back to CPU without it (FaceDetector.load() would then raise, correctly
    # — see its explicit provider check).
