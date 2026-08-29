#!/usr/bin/env bash
# Installs the face pipeline stack WITHOUT the two dependency conflicts a plain
# `pip install -r requirements/face.txt` would hit, verified by actually
# triggering both while building Milestone 3:
#
#   1. `insightface` depends on plain `onnxruntime` (CPU-only). Both
#      `onnxruntime` and `onnxruntime-gpu` install files into the same
#      `onnxruntime/` import path — whichever installs *last* silently
#      overwrites the other's shared libraries with no error. Installing
#      insightface normally after Milestone 1's `onnxruntime-gpu` setup
#      silently regressed CUDA support (confirmed: `get_available_providers()`
#      dropped CUDAExecutionProvider entirely, with only a buried pip warning).
#
#   2. `insightface` depends on `opencv-python` (GUI build). It conflicts with
#      `opencv-python-headless` the exact same way — confirmed: after a plain
#      insightface install, `cv2.getBuildInformation()` reported a QT5 GUI
#      build had silently replaced the headless one this project deliberately
#      uses (no cv2.imshow() anywhere in this codebase — see ARCHITECTURE.md).
#
# Fix: install insightface with --no-deps, then install its *other* real
# dependencies explicitly, deliberately never letting plain `onnxruntime` or
# `opencv-python` touch the venv.

set -euo pipefail

echo "=== Installing face pipeline dependencies (conflict-safe order) ==="

pip install -r requirements/base.txt
pip install --no-deps insightface
pip install numpy onnx opencv-python-headless tqdm requests scipy scikit-image

echo ""
echo "=== Verifying no regression ==="
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
assert 'CUDAExecutionProvider' in providers, f'CUDA provider missing: {providers}'
print('[ OK ] onnxruntime providers:', providers)

import cv2
gui_line = [l for l in cv2.getBuildInformation().splitlines() if 'GUI' in l][0]
assert 'NONE' in gui_line, f'opencv is not the headless build: {gui_line}'
print('[ OK ]', gui_line.strip())

import insightface
print('[ OK ] insightface', insightface.__version__)
"
echo "[ OK ] Face pipeline dependencies installed without conflicts."
