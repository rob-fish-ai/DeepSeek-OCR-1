"""Per-request correlation IDs and structured request logging.

This lives in its own module on purpose. When the service is started as a
script (``python api_service.py``), api_service.py is executed twice: once as
``__main__`` and again when uvicorn imports ``api_service:app``. Anything defined
at module level there exists twice, so a context variable set by the running
app would not be the one read by a log filter installed by the other copy, and
every log line would show no request ID. Objects defined here exist once.

Nothing logged here contains OCR text or document images. Uploads are recorded
by size, extension and a content hash, so a caller can match a request to a
file without the file itself appearing in the logs.
"""

import contextvars
import hashlib
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
request_ctx_var: contextvars.ContextVar[Optional["RequestContext"]] = contextvars.ContextVar(
    "request_ctx", default=None
)

# Caller-supplied IDs are echoed into logs and response headers, so only a
# conservative character set is accepted; anything else gets a fresh ID.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

_request_logger = logging.getLogger("deepseek-ocr.requests")
_installed = False


def new_request_id(incoming: Optional[str]) -> str:
    """Use the caller's X-Request-ID when it is well-formed, else mint one."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestIdFilter(logging.Filter):
    """Stamp every record with the current request ID ('-' outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class RequestContext:
    """Everything recorded about one HTTP request, emitted as one JSON line.

    Shared by reference with the tasks a request spawns (asyncio copies the
    context, not the object), so per-page and per-inference records appended
    from concurrent page tasks all land here.
    """

    def __init__(self, request_id: str, method: str, path: str,
                 client: Optional[str], forwarded_for: Optional[str]):
        self.request_id = request_id
        self.method = method
        self.path = path
        self.client = client
        self.forwarded_for = forwarded_for
        self.started = time.monotonic()
        self.params: dict = {}
        self.uploads: list = []
        self.pages: list = []
        self.inferences: list = []
        self.error: Optional[str] = None

    def summary(self, status: int) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "status": status,
            "duration_ms": round((time.monotonic() - self.started) * 1000),
            "client": self.client,
        }
        if self.forwarded_for:
            record["forwarded_for"] = self.forwarded_for
        if self.params:
            record["params"] = self.params
        if self.uploads:
            record["uploads"] = self.uploads
        if self.pages:
            record["pages"] = self.pages
        if self.inferences:
            record["inference"] = {
                "calls": len(self.inferences),
                "tokens": sum(i["tokens"] for i in self.inferences),
                "hit_length_limit": sum(1 for i in self.inferences if i["hit_length_limit"]),
                "ms": sum(i["ms"] for i in self.inferences),
                "detail": self.inferences,
            }
        if self.error:
            record["error"] = self.error
        return record


def describe_upload(filename: Optional[str], data: bytes, include_name: bool) -> dict:
    """Identify an upload without logging its content or (by default) its name.

    Filenames often carry personal data ("J_Smith_SSN.pdf"), so only the
    extension is kept unless LOG_FILENAMES is enabled. The content hash lets a
    caller confirm which file a request carried by hashing their own copy.
    """
    ext = os.path.splitext(filename or "")[1].lower() or None
    info = {
        "bytes": len(data),
        "sha256_16": hashlib.sha256(data).hexdigest()[:16],
        "ext": ext,
    }
    if include_name:
        info["filename"] = filename
    return info


def install(log_file: str, max_mb: int, backups: int) -> None:
    """Attach the request-ID filter and open the JSON request log. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    id_filter = RequestIdFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(id_filter)

    _request_logger.setLevel(logging.INFO)
    _request_logger.propagate = False  # JSON lines go to their own file only
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _request_logger.addHandler(handler)
    except OSError as e:
        logging.getLogger("deepseek-ocr").warning(
            "Request log disabled — cannot open %s: %s", log_file, e
        )


def emit(record: dict) -> None:
    if _request_logger.handlers:
        _request_logger.info(json.dumps(record, separators=(",", ":"), default=str))
