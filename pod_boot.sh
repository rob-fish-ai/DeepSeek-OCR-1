#!/bin/bash
# Launch the DeepSeek-OCR supervisor unless it is already running.
#
# Called at pod boot from post_start.sh, and safe to run by hand at any time:
# a second invocation exits 0 without starting a duplicate supervisor.
#
# Note: /workspace is a FUSE mount that ignores chmod, so new files land
# without an exec bit. Everything here is invoked via `bash <script>` rather
# than relying on ./script working.
set -u

cd "$(dirname "$0")" || exit 0

PORT="${PORT:-8000}"
LOCK_FILE="${LOCK_FILE:-/tmp/deepseek-ocr-supervisor.lock}"

log() { echo "[pod_boot $(date -u +%FT%TZ)] $*"; }

# Is something already serving? /health answers 503 while the model loads,
# and curl treats that as success, so a still-loading service counts as up.
# This catches a supervisor (or a bare api_service.py) started by hand, which
# the lock below cannot see.
if curl -sS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    log "something is already serving on :${PORT} — nothing to do"
    exit 0
fi

# Guard against two boots racing. The exec'd supervisor inherits fd 9 and
# holds the lock for its lifetime.
exec 9>"$LOCK_FILE" || exit 0
if ! flock -n 9; then
    log "another boot holds the lock — nothing to do"
    exit 0
fi

log "starting supervisor from $(pwd)"
exec bash ./supervise.sh
