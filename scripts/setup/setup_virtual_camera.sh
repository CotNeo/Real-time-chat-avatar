#!/usr/bin/env bash
# Milestone 11 — create the "AI Avatar Camera" virtual video device via v4l2loopback.
#
# Needs root once (kernel module install + load). This machine's sudo requires an
# interactive password, so run this script yourself in a real terminal — it is
# idempotent and safe to re-run.
#
# Usage: ./scripts/setup/setup_virtual_camera.sh [device_number]
#   device_number: which /dev/videoN to use for the virtual device (default: 10,
#   chosen high to avoid colliding with real webcams at /dev/video0, /dev/video1, ...)

set -euo pipefail

DEVICE_NUMBER="${1:-10}"
DEVICE_LABEL="AI Avatar Camera"

echo "=== AI Avatar virtual camera setup ==="

# 1. Dependency check
if ! command -v modprobe >/dev/null 2>&1; then
    echo "[FAIL] 'modprobe' not found. This script requires a standard Linux kernel module toolchain."
    exit 1
fi

if ! dpkg -s v4l2loopback-dkms >/dev/null 2>&1; then
    echo "[INFO] v4l2loopback-dkms is not installed. Installing (requires sudo password)..."
    sudo apt-get update
    sudo apt-get install -y v4l2loopback-dkms v4l-utils
else
    echo "[ OK ] v4l2loopback-dkms already installed."
fi

# 2. Avoid duplicate devices: if the module is already loaded, check whether our
#    target device number is already in use by a v4l2loopback instance before
#    reloading (reloading with different params requires an unload first, which
#    would disrupt anything currently reading the existing loopback device).
if lsmod | grep -q '^v4l2loopback'; then
    if [ -e "/dev/video${DEVICE_NUMBER}" ] && command -v v4l2-ctl >/dev/null 2>&1 \
        && v4l2-ctl --device "/dev/video${DEVICE_NUMBER}" --info 2>/dev/null | grep -q "${DEVICE_LABEL}"; then
        echo "[ OK ] '${DEVICE_LABEL}' already exists at /dev/video${DEVICE_NUMBER}. Nothing to do."
        exit 0
    fi
    echo "[INFO] v4l2loopback is loaded but not configured as '${DEVICE_LABEL}' on /dev/video${DEVICE_NUMBER}."
    echo "       Unloading to reconfigure (this will disrupt any current reader of the existing loopback device)."
    sudo modprobe -r v4l2loopback || {
        echo "[FAIL] Could not unload v4l2loopback — is something still holding it open?"
        echo "       Check: sudo lsof /dev/video${DEVICE_NUMBER}"
        exit 1
    }
fi

# 3. Load the module with our chosen device number and label.
echo "[INFO] Loading v4l2loopback as /dev/video${DEVICE_NUMBER} labeled '${DEVICE_LABEL}'..."
sudo modprobe v4l2loopback \
    video_nr="${DEVICE_NUMBER}" \
    card_label="${DEVICE_LABEL}" \
    exclusive_caps=1

if [ ! -e "/dev/video${DEVICE_NUMBER}" ]; then
    echo "[FAIL] Module loaded but /dev/video${DEVICE_NUMBER} did not appear."
    echo "       Check: dmesg | tail -30"
    exit 1
fi

# 4. Persist across reboots (optional, only if the user wants it — ask, don't
#    silently modify system boot config).
MODULES_LOAD_FILE="/etc/modules-load.d/v4l2loopback.conf"
MODPROBE_OPTS_FILE="/etc/modprobe.d/v4l2loopback.conf"
if [ ! -f "$MODULES_LOAD_FILE" ]; then
    echo ""
    echo "[INFO] The virtual camera will NOT persist across reboots yet."
    echo "       To make it permanent, run:"
    echo "         echo 'v4l2loopback' | sudo tee ${MODULES_LOAD_FILE}"
    echo "         echo 'options v4l2loopback video_nr=${DEVICE_NUMBER} card_label=\"${DEVICE_LABEL}\" exclusive_caps=1' | sudo tee ${MODPROBE_OPTS_FILE}"
fi

echo ""
echo "[ OK ] '${DEVICE_LABEL}' is live at /dev/video${DEVICE_NUMBER}."
echo "       Verify: v4l2-ctl --device /dev/video${DEVICE_NUMBER} --info"
echo "       The AI runtime should write frames to /dev/video${DEVICE_NUMBER}."
