#!/usr/bin/env bash
# Reverses setup_virtual_camera.sh: unloads v4l2loopback and removes the
# persistence files it may have created. Does NOT uninstall v4l2loopback-dkms
# itself (harmless to leave installed; uninstalling a dkms package while other
# software might depend on it is out of scope for an undo script).

set -euo pipefail

echo "=== Removing AI Avatar virtual camera ==="

if lsmod | grep -q '^v4l2loopback'; then
    sudo modprobe -r v4l2loopback && echo "[ OK ] v4l2loopback unloaded." || {
        echo "[FAIL] Could not unload v4l2loopback — something is still holding a loopback device open."
        echo "       Check: sudo lsof /dev/video*"
        exit 1
    }
else
    echo "[ OK ] v4l2loopback was not loaded."
fi

for f in /etc/modules-load.d/v4l2loopback.conf /etc/modprobe.d/v4l2loopback.conf; do
    if [ -f "$f" ]; then
        sudo rm -f "$f"
        echo "[ OK ] Removed $f"
    fi
done

echo "[ OK ] Virtual camera removed. Your real webcam(s) are untouched."
