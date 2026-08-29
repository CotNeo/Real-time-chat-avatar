#!/usr/bin/env bash
# Milestone 12 — create the "AI Avatar Microphone" virtual audio device via
# PipeWire (through its pipewire-pulse pactl compatibility layer). Entirely
# user-space: no root, no system-wide audio config changes, fully reversible
# with remove_virtual_audio.sh.
#
# Architecture: a null-sink is the AI voice engine's write target; a
# remap-source exposes that sink's monitor as a clean, separately-named
# microphone that any app can select. This is two PipeWire modules:
#   ai_avatar_mic_sink          (sink)   <- AI voice engine writes converted audio here
#   ai_avatar_mic_sink.monitor  (source, implicit) -> remapped into:
#   ai_avatar_microphone        (source) <- apps (browsers, calling apps) select this
#
# Usage: ./scripts/setup/setup_virtual_audio.sh

set -euo pipefail

SINK_NAME="ai_avatar_mic_sink"
SOURCE_NAME="ai_avatar_microphone"
SOURCE_DISPLAY_NAME="AI Avatar Microphone"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/realtime-ai-avatar"
STATE_FILE="${STATE_DIR}/virtual_audio_modules.txt"

echo "=== AI Avatar virtual microphone setup (PipeWire) ==="

if ! command -v pactl >/dev/null 2>&1; then
    echo "[FAIL] 'pactl' not found."
    echo "       Check: is PipeWire (with pipewire-pulse) or PulseAudio installed and running?"
    echo "       Try: systemctl --user status pipewire pipewire-pulse"
    exit 1
fi

if ! pactl info >/dev/null 2>&1; then
    echo "[FAIL] Could not talk to the audio server."
    echo "       Check:"
    echo "       1. Is PipeWire running? systemctl --user status pipewire"
    echo "       2. Are you in a graphical/user session with a working \$XDG_RUNTIME_DIR?"
    exit 1
fi

if ! pactl info | grep -qi pipewire; then
    echo "[WARN] Audio server does not report itself as PipeWire (found: $(pactl info | grep 'Server Name'))."
    echo "       Continuing anyway — the pactl calls below also work against plain PulseAudio,"
    echo "       per Section 12's fallback allowance, but device naming/behavior may differ slightly."
fi

mkdir -p "$STATE_DIR"

# Idempotency: if our named source already exists, don't create a second one.
if pactl list short sources | awk '{print $2}' | grep -qx "$SOURCE_NAME"; then
    echo "[ OK ] '${SOURCE_DISPLAY_NAME}' already exists. Nothing to do."
    echo "       (If you want to recreate it, run remove_virtual_audio.sh first.)"
    exit 0
fi

echo "[INFO] Creating sink '${SINK_NAME}' (the AI voice engine's output target)..."
SINK_MODULE_ID=$(pactl load-module module-null-sink \
    sink_name="${SINK_NAME}" \
    sink_properties=device.description="AI_Avatar_Mic_Sink")

echo "[INFO] Creating source '${SOURCE_NAME}' remapped from the sink's monitor..."
SOURCE_MODULE_ID=$(pactl load-module module-remap-source \
    master="${SINK_NAME}.monitor" \
    source_name="${SOURCE_NAME}" \
    source_properties=device.description="${SOURCE_DISPLAY_NAME// /_}")

# Record module IDs so remove_virtual_audio.sh can unload precisely, even
# across a shell restart.
{
    echo "sink_module_id=${SINK_MODULE_ID}"
    echo "source_module_id=${SOURCE_MODULE_ID}"
} > "$STATE_FILE"

echo ""
echo "[ OK ] '${SOURCE_DISPLAY_NAME}' is live."
echo "       Verify: pactl list short sources | grep ${SOURCE_NAME}"
echo "       Or:     wpctl status"
echo ""
echo "       The AI runtime should write converted audio to sink '${SINK_NAME}'."
echo "       Any application's microphone dropdown can now select '${SOURCE_DISPLAY_NAME}'."
echo "       This does not touch your real microphone or change any default device."
