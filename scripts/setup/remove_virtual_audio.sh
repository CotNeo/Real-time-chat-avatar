#!/usr/bin/env bash
# Reverses setup_virtual_audio.sh. Unloads exactly the two PipeWire modules it
# created (by ID, from the state file) — never a blanket "unload everything",
# so other software's PipeWire modules are untouched.

set -euo pipefail

SOURCE_NAME="ai_avatar_microphone"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/realtime-ai-avatar"
STATE_FILE="${STATE_DIR}/virtual_audio_modules.txt"

echo "=== Removing AI Avatar virtual microphone ==="

if [ ! -f "$STATE_FILE" ]; then
    echo "[ OK ] No record of a virtual microphone set up by this project. Nothing to do."
    exit 0
fi

# shellcheck disable=SC1090
source "$STATE_FILE"

for id in "${source_module_id:-}" "${sink_module_id:-}"; do
    if [ -n "$id" ]; then
        if pactl unload-module "$id" 2>/dev/null; then
            echo "[ OK ] Unloaded module $id"
        else
            echo "[WARN] Module $id was already gone (unloaded manually, or PipeWire restarted since)."
        fi
    fi
done

rm -f "$STATE_FILE"

if pactl list short sources 2>/dev/null | awk '{print $2}' | grep -qx "$SOURCE_NAME"; then
    echo "[WARN] A source named '${SOURCE_NAME}' still exists — it may have been created outside this script."
else
    echo "[ OK ] 'AI Avatar Microphone' removed. Your real audio devices are untouched."
fi
