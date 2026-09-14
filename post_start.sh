#!/bin/bash
# RunPod post-start hook: bring the DeepSeek-OCR API up at pod boot.
#
# The pod entrypoint (/start.sh) runs this as `bash /post_start.sh` and has
# `set -e`, so this script must:
#   1. always exit 0  — a non-zero exit aborts the entrypoint before it
#      reaches `sleep infinity`, and the pod dies on boot;
#   2. return promptly — the entrypoint blocks here, so the supervisor is
#      dispatched into the background rather than run inline.
#
# Install with:
#   ln -sf /workspace/DeepSeek-OCR-1/post_start.sh /post_start.sh

BOOT=/workspace/DeepSeek-OCR-1/pod_boot.sh
LOG=/workspace/logs/boot.log

mkdir -p "$(dirname "$LOG")" 2>/dev/null

if [ -f "$BOOT" ]; then
    setsid nohup bash "$BOOT" >> "$LOG" 2>&1 < /dev/null &
    echo "DeepSeek-OCR: supervisor dispatched (log: $LOG)"
else
    echo "DeepSeek-OCR: $BOOT not found — skipping"
fi

exit 0
