#!/usr/bin/env python3
"""
Milestone 1 — CUDA / ONNX Runtime GPU verification.

Verifies (not just checks availability flags, but actually exercises the GPU path):
  1. PyTorch can see and compute on the CUDA device.
  2. ONNX Runtime can create an inference session on CUDAExecutionProvider and it is
     actually used (not silently falling back to CPU).

Run:
    source .venv/bin/activate
    python scripts/setup/verify_cuda.py

Exit code 0 = both checks pass. Non-zero = see the printed remediation steps.
"""
from __future__ import annotations

import sys


def fail(title: str, detail: str, remediation: list[str]) -> None:
    print(f"\n[FAIL] {title}")
    print(f"       {detail}")
    print("       Check:")
    for i, step in enumerate(remediation, 1):
        print(f"       {i}. {step}")


def check_torch() -> bool:
    try:
        import torch
    except ImportError as e:
        fail(
            "PyTorch not installed",
            str(e),
            ["pip install torch", "Verify with: python -c 'import torch'"],
        )
        return False

    if not torch.cuda.is_available():
        fail(
            "PyTorch cannot see a CUDA device",
            "torch.cuda.is_available() returned False",
            [
                "Run `nvidia-smi` — does it show your GPU?",
                "If nvidia-smi fails: reinstall/reload the NVIDIA driver.",
                "If nvidia-smi works but torch doesn't see CUDA: your torch build "
                "may be CPU-only. Reinstall with a CUDA-enabled wheel from "
                "https://pytorch.org/get-started/locally/",
            ],
        )
        return False

    name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    cap = torch.cuda.get_device_capability(0)

    # Exercise the device, don't just ask it if it exists.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        c = a @ b
        torch.cuda.synchronize()
        assert c.shape == (2048, 2048)
    except Exception as e:  # noqa: BLE001 - report whatever CUDA raised, verbatim
        fail(
            "CUDA device found but a compute op failed",
            f"{type(e).__name__}: {e}",
            [
                "Check `nvidia-smi` for Xid errors or thermal throttling.",
                "Check for a driver/CUDA-runtime version mismatch "
                f"(driver reports up to CUDA {torch.version.cuda}, wheel built for "
                f"{torch.version.cuda}).",
            ],
        )
        return False

    print("[ OK ] PyTorch CUDA")
    print(f"       torch {torch.__version__} (cu{torch.version.cuda})")
    print(f"       GPU: {name}  (compute capability {cap[0]}.{cap[1]})")
    print(f"       VRAM: {total_vram:.0f} MiB")
    return True


def check_onnxruntime() -> bool:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        fail(
            "ONNX Runtime or NumPy not installed",
            str(e),
            ["pip install onnxruntime-gpu numpy"],
        )
        return False

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        fail(
            "ONNX Runtime built without CUDAExecutionProvider",
            f"Available providers: {providers}",
            [
                "Ensure the `onnxruntime-gpu` package is installed, not plain "
                "`onnxruntime` (the two conflict — pip uninstall the other first).",
                "Check `pip show onnxruntime-gpu` matches the CUDA major version your "
                "driver supports (`nvidia-smi` top-right).",
            ],
        )
        return False

    # Build a tiny real graph in-memory and force it onto CUDA, then confirm ORT
    # actually placed it there rather than silently falling back to CPU.
    try:
        from onnx import TensorProto, helper

        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "tiny_relu",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        actual = sess.get_providers()
        if actual[0] != "CUDAExecutionProvider":
            fail(
                "ONNX Runtime session fell back to CPU",
                f"Requested CUDAExecutionProvider first, session is actually using: "
                f"{actual}",
                [
                    "This usually means a cuDNN/cuBLAS shared library ONNX Runtime "
                    "needs could not be found at runtime.",
                    "Check LD_LIBRARY_PATH includes the nvidia-* pip package lib "
                    "dirs, e.g.: "
                    "$(python -c \"import nvidia.cudnn, os; "
                    "print(os.path.dirname(nvidia.cudnn.__file__))\")/lib",
                ],
            )
            return False

        x = np.array([[-1.0, 0.5, 2.0, -3.0]], dtype=np.float32)
        (y,) = sess.run(None, {"x": x})
        assert np.allclose(y, [[0.0, 0.5, 2.0, 0.0]])
    except Exception as e:  # noqa: BLE001
        fail(
            "ONNX Runtime CUDA session creation/inference failed",
            f"{type(e).__name__}: {e}",
            [
                "Check the onnxruntime-gpu CUDA/cuDNN version requirements for your "
                "installed onnxruntime-gpu version match what's importable from the "
                "`nvidia-*` pip packages (installed automatically as torch deps).",
                "Try: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12",
            ],
        )
        return False

    print("[ OK ] ONNX Runtime CUDA")
    print(f"       onnxruntime {ort.__version__}")
    print(f"       providers available: {providers}")
    print(f"       session actually used: {actual}")
    return True


def main() -> int:
    print("=== Milestone 1: CUDA / ONNX Runtime GPU verification ===\n")
    ok_torch = check_torch()
    print()
    ok_ort = check_onnxruntime()
    print()
    if ok_torch and ok_ort:
        print("RESULT: PyTorch CUDA = working, ONNX Runtime CUDA provider = working")
        return 0
    print("RESULT: one or more checks failed. See remediation steps above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
