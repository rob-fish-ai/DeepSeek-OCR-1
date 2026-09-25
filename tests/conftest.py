"""Test setup: the real FastAPI app with the GPU model replaced by a stub.

Tests run against api_service exactly as deployed -- routing, cleanup, scoring,
retries, logging -- except that engine.generate() returns scripted text. No GPU
or model weights are needed, and the live service is never touched.
"""

import asyncio
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TMP = tempfile.mkdtemp(prefix="ocr-tests-")
# Must be set before api_service is imported: it reads them at import time.
os.environ.setdefault("FEEDBACK_ENABLED", "false")
os.environ.setdefault("FEEDBACK_DIR", os.path.join(_TMP, "feedback"))
os.environ.setdefault("REQUEST_LOG_FILE", os.path.join(_TMP, "requests.jsonl"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm"))

import httpx  # noqa: E402

import api_service  # noqa: E402

PROMPT_BY_TEXT = {v: k for k, v in api_service.PROMPTS.items()}


class Completion:
    def __init__(self, text, tokens, finish_reason):
        self.text = text
        self.token_ids = [0] * tokens
        self.finish_reason = finish_reason


class Output:
    def __init__(self, text, tokens, finish_reason):
        self.outputs = [Completion(text, tokens, finish_reason)]
        self.prompt_token_ids = [0] * 913


class ScriptedEngine:
    """Stands in for vLLM. `script` maps a prompt key ("document", "free_ocr",
    ...) to (text, num_tokens, finish_reason), or to a callable returning that
    for successive calls. Every call is recorded in `calls`."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def generate(self, prompt, sampling_params, request_id):
        key = PROMPT_BY_TEXT.get(prompt["prompt"], "?")
        self.calls.append(key)
        await asyncio.sleep(0.01)
        spec = self.script[key]
        text, tokens, finish = spec(len(self.calls)) if callable(spec) else spec
        yield Output(text, tokens, finish)


@pytest.fixture
def service(monkeypatch):
    """Returns (make_client, install_engine). Nothing is sent to the GPU."""
    monkeypatch.setattr(api_service, "processor",
                        type("P", (), {"tokenize_with_images": lambda self, **k: [[None] * 7]})())
    monkeypatch.setattr(api_service, "sampling_params", None)

    def install(script):
        engine = ScriptedEngine(script)
        monkeypatch.setattr(api_service, "engine", engine)
        return engine

    def client():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=api_service.app),
                                 base_url="http://test", timeout=120)

    return client, install
