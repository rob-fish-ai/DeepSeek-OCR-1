"""
DeepSeek-OCR API Service
Production-ready FastAPI service for RunPod deployment.

Uses vLLM AsyncLLMEngine for concurrent inference — multiple requests
are batched on the GPU automatically without semaphore serialization.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional


import numpy as np
import torch

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add the vllm source directory to path so imports work
VLLM_SRC = os.path.join(
    os.path.dirname(__file__), "DeepSeek-OCR-master", "DeepSeek-OCR-vllm"
)
sys.path.insert(0, VLLM_SRC)

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image, ImageOps

import request_log as rl

from config import CROP_MODE, MAX_CONCURRENCY, NUM_WORKERS
from deepseek_ocr import DeepseekOCRForCausalLM
from process import (
    clean_output,
    CleanStats,
    enhance_scan,
    enhance_scan_with_preset,
    ENHANCEMENT_PRESETS,
    OCRResult,
    score_result,
    select_best_result,
    needs_retry,
    compute_flags,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_RETRIES,
)
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] — %(message)s",
)
logger = logging.getLogger("deepseek-ocr")

# One JSON line per HTTP request (see request_log.py). Contains no OCR text or
# images; filenames are omitted unless LOG_FILENAMES=true, since they often
# carry personal data.
REQUEST_LOG_FILE = os.environ.get(
    "REQUEST_LOG_FILE", os.path.join(os.environ.get("LOG_DIR", "/workspace/logs"), "requests.jsonl")
)
REQUEST_LOG_MAX_MB = int(os.environ.get("REQUEST_LOG_MAX_MB", "50"))
REQUEST_LOG_BACKUPS = int(os.environ.get("REQUEST_LOG_BACKUPS", "5"))
LOG_FILENAMES = os.environ.get("LOG_FILENAMES", "false").lower() == "true"
rl.install(REQUEST_LOG_FILE, REQUEST_LOG_MAX_MB, REQUEST_LOG_BACKUPS)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/DeepSeek-OCR")
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.80"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Safety limits
MAX_IMAGE_SIZE_MB = int(os.environ.get("MAX_IMAGE_SIZE_MB", "20"))
MAX_PDF_SIZE_MB = int(os.environ.get("MAX_PDF_SIZE_MB", "100"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "50"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "120"))

# Scoring / retry
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", str(DEFAULT_THRESHOLD)))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))



# Feedback storage
FEEDBACK_DIR = os.environ.get("FEEDBACK_DIR", os.path.join(os.path.dirname(__file__), "feedback"))
FEEDBACK_ENABLED = os.environ.get("FEEDBACK_ENABLED", "true").lower() == "true"
FEEDBACK_SCORE_THRESHOLD = float(os.environ.get("FEEDBACK_SCORE_THRESHOLD", "0.70"))
# Disk budget for feedback storage. Oldest pending entries are pruned once the
# directory exceeds this. The default sits above current usage (~12 GB) so
# enabling it does not immediately delete archived pages; lower it to reclaim
# space. Roughly 111 MB/day of growth at peak traffic.
FEEDBACK_MAX_GB = float(os.environ.get("FEEDBACK_MAX_GB", "20"))
FEEDBACK_PRUNE_INTERVAL_S = int(os.environ.get("FEEDBACK_PRUNE_INTERVAL_S", str(6 * 3600)))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\n<|grounding|>OCR this image.",
    "free_ocr": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}
DEFAULT_PROMPT = "document"

# ---------------------------------------------------------------------------
# Global instances (initialized in lifespan)
# ---------------------------------------------------------------------------
engine: Optional[AsyncLLMEngine] = None
sampling_params: Optional[SamplingParams] = None
processor: Optional[DeepseekOCRProcessor] = None
worker_pool: Optional[ThreadPoolExecutor] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note_params(**params) -> None:
    """Record request parameters for this request's log line."""
    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        ctx.params.update(params)


def _note_upload(filename: Optional[str], data: bytes) -> None:
    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        ctx.uploads.append(rl.describe_upload(filename, data, LOG_FILENAMES))


def _note_page(result: dict, **where) -> None:
    """Record how one page came out. Metadata only — never the text."""
    ctx = rl.request_ctx_var.get()
    if ctx is None:
        return
    score = result.get("score") or {}
    ctx.pages.append({
        **where,
        "flag": result.get("flag"),
        "score": score.get("composite"),
        "codes": [d.get("code") for d in result.get("flag_details") or []],
        "chars": len(result.get("text") or ""),
        "tokens": result.get("num_tokens"),
        "attempts": result.get("attempts"),
        "preset": result.get("preset"),
        "engine": result.get("ocr_engine"),
        "needs_external_ocr": result.get("needs_external_ocr"),
    })


def _respond(result: dict) -> JSONResponse:
    """Return a single-page result, recording it for the request log."""
    _note_page(result)
    return JSONResponse(result)


def is_blank_page(image: Image.Image, std_threshold: float = 5.0, dark_threshold: float = 0.02) -> bool:
    """Fast pixel-based blank page detection. Returns True if the image is blank."""
    gray = np.array(image.convert("L"))
    if gray.std() >= std_threshold:
        return False
    dark_ratio = (gray < 240).sum() / gray.size
    return dark_ratio < dark_threshold


def is_low_quality_scan(image: Image.Image, content_area_threshold: float = 0.12) -> bool:
    """Detect scans where content is shrunk to a tiny area, making text unreadable.

    Checks the ratio of the content bounding box to the full page area.
    Returns True if content occupies less than ``content_area_threshold`` of the page.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    # Find rows/cols with meaningful dark pixels
    row_dark = (gray < 200).sum(axis=1)
    col_dark = (gray < 200).sum(axis=0)
    row_thresh = w * 0.01
    col_thresh = h * 0.01
    content_rows = np.where(row_dark > row_thresh)[0]
    content_cols = np.where(col_dark > col_thresh)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return True  # no content at all

    content_h = content_rows[-1] - content_rows[0]
    content_w = content_cols[-1] - content_cols[0]
    content_area_ratio = (content_h * content_w) / (h * w)

    return content_area_ratio < content_area_threshold


_BOILERPLATE_PATTERNS = re.compile(
    r"^("
    r"page\s*\d+|p\.?\s*\d+|\d+\s*/\s*\d+"          # page numbers
    r"|\-\s*\d+\s*\-|\d+"                              # bare numbers, dash-wrapped
    r"|[-=*_~]{3,}"                                     # dividers
    r"|\.{3,}"                                          # dot leaders
    r"|\s+"                                             # whitespace-only lines
    r")$",
    re.IGNORECASE | re.MULTILINE,
)

POST_OCR_BLANK_CHAR_LIMIT = int(os.environ.get("POST_OCR_BLANK_CHAR_LIMIT", "20"))


def _is_boilerplate_only(text: str) -> bool:
    """Return True if text consists only of boilerplate (page numbers, dividers, etc.)."""
    stripped = text.strip()
    if not stripped:
        return True
    # Remove all boilerplate patterns and see if anything remains
    cleaned = _BOILERPLATE_PATTERNS.sub("", stripped).strip()
    return len(cleaned) == 0


def _is_post_ocr_blank(clean_text: str) -> bool:
    """Return True if OCR output indicates a blank/near-blank page.

    Checks: text shorter than threshold OR only boilerplate content.
    """
    stripped = clean_text.strip()
    if len(stripped) < POST_OCR_BLANK_CHAR_LIMIT:
        return True
    return _is_boilerplate_only(stripped)


def _skip_page_result(reason: str, flag_detail: str) -> dict:
    """Return a pre-built result dict for a skipped page (no OCR needed)."""
    return {
        "text": "",
        "raw_text": "",
        "num_tokens": 0,
        "score": {
            "composite": 0.0,
            "variables": {
                "self_consistency": 0.0,
                "hallucination_ratio": 0.0,
                "token_efficiency": 0.0,
                "structural_integrity": 0.0,
                "repetition_density": 0.0,
                "content_density": 0.0,
            },
        },
        "flag": "red",
        "flag_message": reason,
        "flag_details": [{"code": flag_detail, "severity": "info", "message": reason}],
        "attempts": 0,
        "preset": None,
        "needs_external_ocr": False,
        "ocr_engine": "skipped",
    }



def _save_feedback(image: Image.Image, result: dict, filename: str = None):
    """Save low-scoring OCR results for future fine-tuning.

    Runs off the request path via asyncio.to_thread. All I/O failures
    are swallowed and logged — feedback storage must never break OCR.
    """
    if not FEEDBACK_ENABLED:
        return

    score = result.get("score", {}).get("composite", 1.0)
    if score >= FEEDBACK_SCORE_THRESHOLD:
        return

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        entry_id = f"{timestamp}_{uuid.uuid4().hex[:12]}"

        pending_dir = os.path.join(FEEDBACK_DIR, "pending")
        os.makedirs(pending_dir, exist_ok=True)

        img_path = os.path.join(pending_dir, f"{entry_id}.png")
        image.save(img_path, format="PNG")

        meta = {
            "id": entry_id,
            "timestamp": timestamp,
            "filename": filename,
            "ocr_engine": result.get("ocr_engine"),
            "score": result.get("score", {}).get("composite"),
            # Recorded so stored entries can be re-scored offline. num_tokens
            # was previously omitted, which left every archived entry at 0 and
            # made the corpus useless for calibrating token-related metrics.
            "num_tokens": result.get("num_tokens"),
            "score_variables": result.get("score", {}).get("variables"),
            "flag": result.get("flag"),
            "text": result.get("text", ""),
            "raw_text": result.get("raw_text", ""),
            "attempts": result.get("attempts", 0),
            "corrected_text": None,
            "status": "pending",
        }
        # Write via a temp file + atomic rename.  A plain open("w") leaves a
        # zero-byte file behind if the process is killed mid-write, which is
        # how the unreadable entries in feedback/pending got there.
        meta_path = os.path.join(pending_dir, f"{entry_id}.json")
        tmp_path = f"{meta_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)

        logger.info("Feedback saved: %s (score=%.3f, engine=%s)",
                    entry_id, score, result.get("ocr_engine"))
    except OSError as e:
        logger.warning("Feedback save failed (I/O error): %s", e)
    except Exception as e:
        logger.warning("Feedback save failed: %s", e)


def _schedule_feedback(image: Image.Image, result: dict, filename: str = None):
    """Fire-and-forget feedback save; runs in a worker thread."""
    if not FEEDBACK_ENABLED:
        return
    score = result.get("score", {}).get("composite", 1.0)
    if score >= FEEDBACK_SCORE_THRESHOLD:
        return
    asyncio.create_task(asyncio.to_thread(_save_feedback, image, result, filename))


def _mark_needs_external_ocr(result: dict) -> None:
    """Set external OCR flag."""
    result["needs_external_ocr"] = True


def _validate_prompt(prompt: str) -> str:
    if prompt not in PROMPTS:
        raise HTTPException(
            400,
            f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}",
        )
    return prompt


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from bytes with EXIF correction and scan enhancement."""
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = enhance_scan(image)
    return image.convert("RGB")


def preprocess_image(image: Image.Image, prompt_key: str = DEFAULT_PROMPT) -> dict:
    """Preprocess a single image into vLLM input format."""
    prompt = PROMPTS.get(prompt_key, PROMPTS[DEFAULT_PROMPT])
    # The prompt must be passed to tokenize_with_images: vLLM takes its prompt
    # token ids from the features below and ignores the "prompt" string here,
    # so tokenizing with the wrong prompt silently ignores prompt_key.
    features = processor.tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE, conversation=prompt
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": features},
    }


def pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> list[Image.Image]:
    """Convert PDF bytes to a list of enhanced PIL Images (sequential; kept for compatibility)."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(doc) > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(
            400,
            f"PDF has {len(doc)} pages, maximum allowed is {MAX_PDF_PAGES}",
        )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img = enhance_scan(img).convert("RGB")
        images.append(img)
    doc.close()
    return images


def _render_page_chunk(pdf_bytes: bytes, dpi: int, page_indices: list[int]):
    """Render, pre-check, and enhance a chunk of pages. One fitz.Document per worker call.

    Returns list of (page_idx, original_img, enhanced_img_or_none, skip_flag_or_none).
    Skipped pages (blank / low-quality) have enhanced=None so we don't waste enhancement work.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        results = []
        for idx in page_indices:
            pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
            img_data = pix.tobytes("png")
            original = Image.open(io.BytesIO(img_data))
            original.load()  # decode now while we're off the event loop

            if is_blank_page(original):
                results.append((idx, original, None, "blank_page"))
            elif is_low_quality_scan(original):
                results.append((idx, original, None, "low_quality_scan"))
            else:
                enhanced = enhance_scan(original).convert("RGB")
                results.append((idx, original, enhanced, None))
        return results
    finally:
        doc.close()


async def render_pdf_parallel(pdf_bytes: bytes, dpi: int, num_pages: int):
    """Render + enhance + pre-check all pages across the worker pool.

    Returns (originals, enhanced_or_none, skip_flags) indexed by page number.
    ``enhanced_or_none[i]`` is None for blank / low-quality pages.
    """
    loop = asyncio.get_event_loop()

    # Round-robin partition so each worker gets pages spread across the document
    # (avoids one worker getting all the heavy pages if they cluster).
    n_workers = max(1, min(NUM_WORKERS, num_pages))
    chunks: list[list[int]] = [[] for _ in range(n_workers)]
    for i in range(num_pages):
        chunks[i % n_workers].append(i)

    tasks = [
        loop.run_in_executor(None, _render_page_chunk, pdf_bytes, dpi, chunk)
        for chunk in chunks
        if chunk
    ]
    chunk_results = await asyncio.gather(*tasks)

    originals: list[Optional[Image.Image]] = [None] * num_pages
    enhanced: list[Optional[Image.Image]] = [None] * num_pages
    skip_flags: list[Optional[str]] = [None] * num_pages
    for chunk in chunk_results:
        for idx, orig, enh, flag in chunk:
            originals[idx] = orig
            enhanced[idx] = enh
            skip_flags[idx] = flag
    return originals, enhanced, skip_flags


async def _run_inference(image: Image.Image, prompt_key: str = DEFAULT_PROMPT) -> dict:
    """Run inference via AsyncLLMEngine. Supports true concurrent requests."""
    # Preprocess image synchronously (CPU work)
    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(None, preprocess_image, image, prompt_key)

    # Prefix the engine request ID with the HTTP request's ID, so vLLM's own
    # "Added/Finished request ..." lines can be traced back to the caller.
    request_id = f"{rl.request_id_var.get()}-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()

    # Submit to async engine — this returns an async generator
    results_generator = engine.generate(
        prompt=vllm_input,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    # Collect the final output
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except Exception as e:
        logger.error("Inference failed for request %s: %s", request_id, e, exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")

    if final_output is None:
        raise HTTPException(500, "Inference returned no output")

    completion = final_output.outputs[0]
    text = completion.text
    num_tokens = len(completion.token_ids)
    prompt_tokens = len(final_output.prompt_token_ids or [])
    hit_length_limit = getattr(completion, "finish_reason", None) == "length"

    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        ctx.inferences.append({
            "engine_request_id": request_id,
            "prompt": prompt_key,
            "prompt_tokens": prompt_tokens,
            "tokens": num_tokens,
            "hit_length_limit": hit_length_limit,
            "ms": round((time.monotonic() - started) * 1000),
        })

    # finish_reason == "length" means generation ran out of room rather than
    # emitting a stop token. This cannot be inferred from num_tokens, because
    # the usable budget is MAX_MODEL_LEN minus the prompt (~7,280 tokens for a
    # 144-DPI A4 page) and MAX_TOKENS is never reachable.
    return {
        "text": text,
        "num_tokens": num_tokens,
        "prompt_tokens": prompt_tokens,
        "hit_length_limit": hit_length_limit,
    }


async def _format_result(inference_output: dict, raw: bool, image: Image.Image = None) -> dict:
    """Build a consistent result dict from inference output."""
    text = inference_output["text"]
    num_tokens = inference_output["num_tokens"]
    stats = CleanStats()
    cleaned = clean_output(text, stats=stats)

    # Score the result
    ocr_result = OCRResult(
        raw_text=text,
        clean_text=cleaned,
        num_tokens=num_tokens,
        max_tokens=MAX_TOKENS,
        clean_stats=stats,
        hit_length_limit=inference_output.get("hit_length_limit", False),
    )
    score = score_result(ocr_result)
    flag_info = compute_flags(ocr_result, SCORE_THRESHOLD)

    result = {
        "text": text if raw else cleaned,
        "raw_text": text,
        "num_tokens": num_tokens,
        "score": score.to_dict(),
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
        "needs_external_ocr": False,
        "ocr_engine": "deepseek",
    }

    # Post-OCR blank page detection: only flag as blank if the image itself
    # also looks blank/low-quality. If the image has content but OCR returned
    # nothing, that's an OCR failure — not a blank page.
    clean_len = len(cleaned.strip())
    image_looks_blank = image is not None and (is_blank_page(image) or is_low_quality_scan(image))
    if _is_post_ocr_blank(cleaned) and image_looks_blank:
        logger.info("Post-OCR blank page detected (%d chars): %r", clean_len, cleaned.strip()[:50])
        result["flag"] = "red"
        result["flag_message"] = "Blank page detected after OCR — no meaningful content."
        result["flag_details"] = [{
            "code": "blank_page",
            "severity": "info",
            "message": f"Page produced only {clean_len} chars of boilerplate/empty content after OCR.",
        }]
        result["needs_external_ocr"] = False
        result["ocr_engine"] = "deepseek"
        return result

    # Flag OCR extraction failure: page has content but model couldn't read it
    raw_len = len(text.strip())
    needs_fallback = False

    if clean_len <= 10 and score.composite < SCORE_THRESHOLD:
        _mark_needs_external_ocr(result)
        needs_fallback = True
        if not any(d.get("code") == "ocr_failed" for d in result["flag_details"]):
            result["flag_details"].append({
                "code": "ocr_failed",
                "severity": "critical",
                "message": "OCR extraction failed — page has content but model could not read it.",
            })

    # Flag incomplete extraction: model hit max tokens and most output was hallucinated
    if (
        not result["needs_external_ocr"]
        and inference_output.get("hit_length_limit", False)
        and raw_len > 0
        and clean_len / raw_len < 0.10
        and score.composite < SCORE_THRESHOLD
    ):
        _mark_needs_external_ocr(result)
        needs_fallback = True
        result["flag_details"].append({
            "code": "incomplete_extraction",
            "severity": "critical",
            "message": f"Model hit token limit with {clean_len}/{raw_len} chars retained ({clean_len/raw_len*100:.0f}%). Most output was hallucinated.",
        })

    return result


async def _run_inference_with_retry(
    image: Image.Image,
    prompt_key: str,
) -> dict:
    """Run OCR with scoring and retry on low-quality results.

    Tries different enhancement presets and returns the best-scoring result.
    """
    results: list[OCRResult] = []

    # Score below which retrying is pointless — the page needs external OCR
    HOPELESS_THRESHOLD = 0.20

    for attempt, preset in enumerate(ENHANCEMENT_PRESETS):
        if attempt > 0 and results:
            last_score = results[-1].score.composite if results[-1].score else 0
            if not needs_retry(results[-1], SCORE_THRESHOLD):
                break  # previous result was good enough
            if last_score < HOPELESS_THRESHOLD:
                logger.info("Score %.3f < %.2f — skipping remaining retries", last_score, HOPELESS_THRESHOLD)
                break  # clearly hopeless, don't waste time
        if attempt >= MAX_RETRIES:
            break

        # Apply enhancement preset
        if preset["contrast"] is None:
            enhanced = enhance_scan(image)
        else:
            enhanced = enhance_scan_with_preset(
                image, preset["contrast"], preset["sharpness"]
            )
        enhanced = enhanced.convert("RGB")

        output = await _run_inference(enhanced, prompt_key)

        text = output["text"]
        num_tokens = output["num_tokens"]
        retry_stats = CleanStats()
        cleaned = clean_output(text, stats=retry_stats)

        ocr_result = OCRResult(
            raw_text=text,
            clean_text=cleaned,
            num_tokens=num_tokens,
            max_tokens=MAX_TOKENS,
            preset_name=preset["name"],
            clean_stats=retry_stats,
            hit_length_limit=output.get("hit_length_limit", False),
        )
        # No image dimensions here: _score_content_density switches to a
        # pixel-ratio scale when given them, which scores a normal page ~0.26
        # instead of ~1.0.  _format_result scores without dimensions, and the
        # two composites are compared against each other by the PDF and batch
        # endpoints, so both paths must score on the same scale.
        score_result(ocr_result, other_results=results)
        results.append(ocr_result)

        logger.info(
            "Attempt %d/%d (preset=%s): %d tokens, score=%.3f",
            attempt + 1,
            MAX_RETRIES,
            preset["name"],
            num_tokens,
            ocr_result.score.composite,
        )

        if not needs_retry(ocr_result, SCORE_THRESHOLD):
            break

    best = select_best_result(results)

    # select_best_result ranks candidates using cross-run self-consistency,
    # which a single first-pass result cannot have (it scores a flat 1.0).
    # Re-score the winner on a single-run basis so the composite we report —
    # and that the PDF/batch endpoints compare against the first pass — is
    # computed exactly the way _format_result computes it.
    score_result(best)

    flag_info = compute_flags(best, SCORE_THRESHOLD)

    result = {
        "text": best.clean_text,
        "raw_text": best.raw_text,
        "num_tokens": best.num_tokens,
        "score": best.score.to_dict() if best.score else None,
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
        "attempts": len(results),
        "preset": best.preset_name,
        "needs_external_ocr": False,
        "ocr_engine": "deepseek",
    }

    # Post-OCR blank page detection: only flag as blank if the image itself
    # also looks blank/low-quality. If the image has content but OCR returned
    # nothing, that's an OCR failure — not a blank page.
    clean_len = len(best.clean_text.strip())
    image_looks_blank = is_blank_page(image) or is_low_quality_scan(image)
    if _is_post_ocr_blank(best.clean_text) and image_looks_blank:
        logger.info("Post-OCR blank page detected (%d chars): %r", clean_len, best.clean_text.strip()[:50])
        result["flag"] = "red"
        result["flag_message"] = "Blank page detected after OCR — no meaningful content."
        result["flag_details"] = [{
            "code": "blank_page",
            "severity": "info",
            "message": f"Page produced only {clean_len} chars of boilerplate/empty content after OCR.",
        }]
        result["needs_external_ocr"] = False
        return result

    # Flag OCR extraction failure: page has content but model couldn't read it
    raw_len = len(best.raw_text.strip())
    composite = best.score.composite if best.score else 0
    needs_fallback = False

    if clean_len <= 10 and composite < SCORE_THRESHOLD:
        _mark_needs_external_ocr(result)
        needs_fallback = True
        if not any(d.get("code") == "ocr_failed" for d in result["flag_details"]):
            result["flag_details"].append({
                "code": "ocr_failed",
                "severity": "critical",
                "message": "OCR extraction failed — page has content but model could not read it.",
            })

    # Flag incomplete extraction: model hit max tokens and most output was hallucinated
    if (
        not result["needs_external_ocr"]
        and best.hit_length_limit
        and raw_len > 0
        and clean_len / raw_len < 0.10
        and composite < SCORE_THRESHOLD
    ):
        _mark_needs_external_ocr(result)
        needs_fallback = True
        result["flag_details"].append({
            "code": "incomplete_extraction",
            "severity": "critical",
            "message": f"Model hit token limit with {clean_len}/{raw_len} chars retained ({clean_len/raw_len*100:.0f}%). Most output was hallucinated.",
        })

    return result


def _check_file_size(data: bytes, max_mb: int, label: str = "File"):
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            413, f"{label} is {size_mb:.1f} MB, maximum allowed is {max_mb} MB"
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load AsyncLLMEngine.  Shutdown: release resources."""
    global engine, sampling_params, processor, worker_pool

    # Wire NUM_WORKERS to the event loop's default executor so run_in_executor(None, ...)
    # calls use our configured pool instead of Python's default (min(32, cpu+4)).
    worker_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix="ocr")
    asyncio.get_event_loop().set_default_executor(worker_pool)

    # ---- Startup ----
    logger.info("Loading model from %s …", MODEL_PATH)
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        task="generate",
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=128,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        disable_mm_preprocessor_cache=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822},
            max_consecutive_empty_cells=30,
        )
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    processor = DeepseekOCRProcessor()

    # Enforce the feedback storage budget in the background. Runs off the
    # event loop; the directory holds tens of thousands of files on a network
    # mount, so scanning it inline would stall the vLLM engine.
    prune_task = asyncio.create_task(_feedback_prune_loop())

    logger.info("Model loaded and ready (async engine).")

    yield  # ---- App runs here ----

    # ---- Shutdown ----
    logger.info("Shutting down …")
    prune_task.cancel()
    try:
        await prune_task
    except asyncio.CancelledError:
        pass
    if engine:
        engine.shutdown_background_loop()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if worker_pool is not None:
        worker_pool.shutdown(wait=True)
    logger.info("Cleanup complete.")


app = FastAPI(
    title="DeepSeek-OCR API",
    description="Production OCR API powered by DeepSeek-OCR",
    version="4.0.0",
    lifespan=lifespan,
)

# Paths polled by the supervisor or browsed by humans: they still get an
# X-Request-ID header, but no request-log line, so the log stays signal.
_UNLOGGED_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = rl.new_request_id(request.headers.get("x-request-id"))
    ctx = rl.RequestContext(
        rid, request.method, request.url.path,
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
    )
    rid_token = rl.request_id_var.set(rid)
    ctx_token = rl.request_ctx_var.set(ctx)
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception as e:
        ctx.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if request.url.path not in _UNLOGGED_PATHS:
            record = ctx.summary(status)
            rl.emit(record)
            flags = [p["flag"] for p in ctx.pages if p.get("flag")]
            logger.info(
                "%s %s -> %d in %d ms (%d page(s)%s%s)",
                request.method, request.url.path, status, record["duration_ms"], len(ctx.pages),
                f", flags {','.join(flags)}" if flags else "",
                f", error: {ctx.error}" if ctx.error else "",
            )
        rl.request_ctx_var.reset(ctx_token)
        rl.request_id_var.reset(rid_token)


@app.exception_handler(StarletteHTTPException)
async def _log_http_exception(request: Request, exc: StarletteHTTPException):
    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        ctx.error = f"HTTP {exc.status_code}: {exc.detail}"
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _log_validation_error(request: Request, exc: RequestValidationError):
    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        # Field locations only: FastAPI's error detail echoes the submitted
        # values, which here can be an entire base64-encoded document.
        ctx.error = "validation: " + ", ".join(
            ".".join(str(part) for part in err.get("loc", ())) for err in exc.errors()
        )
    return await request_validation_exception_handler(request, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR API", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    if engine is None:
        status = "loading"
        engine_error = None
    elif getattr(engine, "errored", False):
        status = "dead"
        dead = getattr(engine, "dead_error", None)
        engine_error = repr(dead) if dead else "background loop errored"
    else:
        status = "healthy"
        engine_error = None

    body = {
        "status": status,
        "model": MODEL_PATH,
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        ),
        "scoring": {
            "threshold": SCORE_THRESHOLD,
            "max_retries": MAX_RETRIES,
        },
    }
    if engine_error:
        body["engine_error"] = engine_error
    return JSONResponse(body, status_code=200 if status == "healthy" else 503)


@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR a single image with quality scoring and optional retry.

    - **file**: Image file (JPEG, PNG, etc.)
    - **prompt**: Prompt type — one of: document, ocr, free_ocr, figure, describe
    - **raw**: If true, return raw output with grounding annotations
    - **retry**: If true, retry with different enhancements on low scores
    """
    _note_params(prompt=prompt, raw=raw, retry=retry)
    _validate_prompt(prompt)

    data = await file.read()
    _note_upload(file.filename, data)
    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")

    # Load original image (without enhancement — retry system handles it)
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Skip blank pages entirely
    if is_blank_page(image):
        logger.info("Blank page detected — skipping OCR")
        return _respond(_skip_page_result("Blank page detected — skipped OCR", "blank_page"))

    # Skip low-quality scans where content is too small to read
    if is_low_quality_scan(image):
        logger.info("Low-quality scan detected — skipping OCR")
        return _respond(_skip_page_result("Low-quality scan — content too small to read", "low_quality_scan"))

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        _schedule_feedback(image, result, filename=file.filename)
        return _respond(result)
    else:
        enhanced = enhance_scan(image).convert("RGB")
        output = await _run_inference(enhanced, prompt)
        result = await _format_result(output, raw, image=image)
        _schedule_feedback(image, result, filename=file.filename)
        return _respond(result)


@app.post("/ocr/image/base64")
async def ocr_image_base64(
    image_base64: str = Form(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """OCR a single image from base64-encoded data."""
    _note_params(prompt=prompt, raw=raw, retry=retry)
    _validate_prompt(prompt)

    try:
        data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")
    _note_upload(None, data)

    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")

    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Skip blank pages entirely
    if is_blank_page(image):
        logger.info("Blank page detected — skipping OCR")
        return _respond(_skip_page_result("Blank page detected — skipped OCR", "blank_page"))

    # Skip low-quality scans where content is too small to read
    if is_low_quality_scan(image):
        logger.info("Low-quality scan detected — skipping OCR")
        return _respond(_skip_page_result("Low-quality scan — content too small to read", "low_quality_scan"))

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        _schedule_feedback(image, result)
        return _respond(result)
    else:
        enhanced = enhance_scan(image).convert("RGB")
        output = await _run_inference(enhanced, prompt)
        result = await _format_result(output, raw, image=image)
        _schedule_feedback(image, result)
        return _respond(result)


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    dpi: int = Form(144),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR a PDF document (all pages) with per-page scoring and retry.

    - **file**: PDF file
    - **prompt**: Prompt type
    - **dpi**: Resolution for PDF rendering (default 144)
    - **raw**: If true, return raw output with grounding annotations
    - **retry**: If true, retry low-scoring pages
    """
    _note_params(prompt=prompt, dpi=dpi, raw=raw, retry=retry)
    _validate_prompt(prompt)

    pdf_bytes = await file.read()
    _note_upload(file.filename, pdf_bytes)
    _check_file_size(pdf_bytes, MAX_PDF_SIZE_MB, "PDF")

    # Read page count once, before dispatching workers. A malformed or
    # non-PDF upload raises here; that is a client error, not a server one.
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        doc.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")

    if num_pages == 0:
        raise HTTPException(400, "Could not extract any pages from the PDF")
    if num_pages > MAX_PDF_PAGES:
        raise HTTPException(
            400, f"PDF has {num_pages} pages, maximum allowed is {MAX_PDF_PAGES}"
        )

    # Parallel render + enhance + blank/low-quality checks.
    # ``originals`` are cached so the retry pass doesn't re-open the PDF.
    originals, enhanced_images, skip_flags = await render_pdf_parallel(
        pdf_bytes, dpi, num_pages
    )

    skip_count = sum(1 for s in skip_flags if s is not None)
    if skip_count:
        logger.info("Skipping %d page(s) (blank or low-quality) — no OCR needed", skip_count)

    pages = [None] * num_pages
    skip_messages = {
        "blank_page": "Blank page detected — skipped OCR",
        "low_quality_scan": "Low-quality scan — content too small to read",
    }
    for i, skip in enumerate(skip_flags):
        if skip:
            result = _skip_page_result(skip_messages[skip], skip)
            result["page"] = i + 1
            pages[i] = result

    # First-pass OCR on processable pages, concurrently.
    processable_indices = [i for i, skip in enumerate(skip_flags) if skip is None]
    if processable_indices:
        async def _ocr_page(page_idx: int) -> tuple[int, dict]:
            output = await _run_inference(enhanced_images[page_idx], prompt)
            result = await _format_result(output, raw, image=enhanced_images[page_idx])
            result["page"] = page_idx + 1
            return page_idx, result

        ocr_tasks = [_ocr_page(i) for i in processable_indices]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        for item in ocr_results:
            if isinstance(item, Exception):
                logger.error("Page OCR failed: %s", item)
                continue
            page_idx, result = item
            pages[page_idx] = result

    retry_indices = []
    for i, result in enumerate(pages):
        if result is not None and skip_flags[i] is None and retry and result["score"]["composite"] < SCORE_THRESHOLD:
            retry_indices.append(i)

    # Retry low-scoring pages in parallel, reusing the cached originals.
    if retry_indices:
        logger.info("Retrying %d page(s) with low scores", len(retry_indices))

        async def _retry_page(page_idx: int) -> tuple[int, dict]:
            logger.info(
                "Retrying page %d (score=%.3f < %.3f)",
                page_idx + 1,
                pages[page_idx]["score"]["composite"],
                SCORE_THRESHOLD,
            )
            result = await _run_inference_with_retry(originals[page_idx], prompt)
            result["page"] = page_idx + 1
            if raw:
                result["text"] = result["raw_text"]
            return page_idx, result

        retry_tasks = [_retry_page(idx) for idx in retry_indices]
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        for item in retry_results:
            if isinstance(item, Exception):
                logger.error("Retry OCR failed: %s", item)
                continue
            page_idx, retry_result = item
            # Use retry result only if it scored better
            if retry_result.get("score", {}).get("composite", 0) > pages[page_idx]["score"]["composite"]:
                pages[page_idx] = retry_result

    # Fill any None pages (from failed OCR)
    for i, p in enumerate(pages):
        if p is None:
            pages[i] = _skip_page_result("OCR failed", "ocr_failed")
            pages[i]["page"] = i + 1

    for p in pages:
        _note_page(p, page=p["page"])

    full_text = "\n\n---\n\n".join(p["text"] for p in pages)

    # Build summary by flag color
    summary = {"green": 0, "yellow": 0, "red": 0}
    flagged_pages = []
    for p in pages:
        color = p.get("flag", "yellow")
        summary[color] = summary.get(color, 0) + 1
        if color in ("yellow", "red"):
            flagged_pages.append({
                "page": p["page"],
                "flag": color,
                "flag_message": p.get("flag_message"),
                "score": p["score"]["composite"] if p.get("score") else None,
            })

    return JSONResponse(
        {
            "num_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
            "total_tokens": sum(p["num_tokens"] for p in pages),
            "summary": summary,
            "flagged_pages": flagged_pages,
        }
    )


@app.post("/ocr/batch")
async def ocr_batch(
    files: list[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR multiple images in a single batch with scoring.

    - **files**: Multiple image files (max MAX_BATCH_SIZE)
    - **prompt**: Prompt type
    - **raw**: If true, return raw output
    - **retry**: If true, retry low-scoring images
    """
    _note_params(prompt=prompt, raw=raw, retry=retry, files=len(files))
    _validate_prompt(prompt)

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            400,
            f"Too many files ({len(files)}). Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    # Load all images
    raw_images: list[Image.Image] = []
    enhanced_images: list[Image.Image] = []
    valid_indices: list[int] = []
    errors: list[dict] = []

    for i, f in enumerate(files):
        try:
            data = await f.read()
            _note_upload(f.filename, data)
            _check_file_size(data, MAX_IMAGE_SIZE_MB, f"File '{f.filename}'")
            try:
                img = Image.open(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(400, f"Could not decode image: {e}")
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            raw_images.append(img)
            enhanced_images.append(enhance_scan(img).convert("RGB"))
            valid_indices.append(i)
        except HTTPException as e:
            errors.append({"index": i, "filename": f.filename, "error": e.detail})
        except Exception as e:
            errors.append({"index": i, "filename": f.filename, "error": str(e)})

    results: list[dict] = []

    # Separate skippable pages (blank or low-quality) before OCR
    skip_messages = {
        "blank_page": "Blank page detected — skipped OCR",
        "low_quality_scan": "Low-quality scan — content too small to read",
    }
    processable_raw = []
    processable_enhanced = []
    processable_valid = []
    for j, (img_raw, img_enh) in enumerate(zip(raw_images, enhanced_images)):
        original_idx = valid_indices[j]
        if is_blank_page(img_raw):
            skip_type = "blank_page"
        elif is_low_quality_scan(img_raw):
            skip_type = "low_quality_scan"
        else:
            skip_type = None

        if skip_type:
            logger.info("%s — skipping OCR for %s", skip_messages[skip_type], files[original_idx].filename)
            result = _skip_page_result(skip_messages[skip_type], skip_type)
            result["index"] = original_idx
            result["filename"] = files[original_idx].filename
            results.append(result)
        else:
            processable_raw.append(img_raw)
            processable_enhanced.append(img_enh)
            processable_valid.append((j, original_idx))

    if processable_enhanced:
        async def _ocr_batch_item(k: int) -> tuple[int, dict]:
            j, original_idx = processable_valid[k]
            output = await _run_inference(processable_enhanced[k], prompt)
            result = await _format_result(output, raw, image=raw_images[j])
            result["index"] = original_idx
            result["filename"] = files[original_idx].filename
            return k, result

        ocr_tasks = [_ocr_batch_item(k) for k in range(len(processable_enhanced))]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        retry_queue = []  # (result_list_index, raw_image_index, original_file_index)
        for item in ocr_results:
            if isinstance(item, Exception):
                logger.error("Batch item OCR failed: %s", item)
                continue
            k, result = item
            j, original_idx = processable_valid[k]
            result_pos = len(results)
            results.append(result)

            if retry and result["score"]["composite"] < SCORE_THRESHOLD:
                retry_queue.append((result_pos, j, original_idx))

        # Retry low-scoring images
        for result_pos, j, original_idx in retry_queue:
            logger.info(
                "Retrying %s (score=%.3f)",
                files[original_idx].filename,
                results[result_pos]["score"]["composite"],
            )
            retry_result = await _run_inference_with_retry(
                raw_images[j], prompt
            )
            if raw:
                retry_result["text"] = retry_result["raw_text"]
            retry_result["index"] = original_idx
            retry_result["filename"] = files[original_idx].filename

            if retry_result.get("score", {}).get("composite", 0) > results[result_pos]["score"]["composite"]:
                results[result_pos] = retry_result

    for r in results:
        _note_page(r, index=r.get("index"))
    ctx = rl.request_ctx_var.get()
    if ctx is not None:
        for e in errors:
            ctx.pages.append({"index": e["index"], "error": str(e["error"])})

    # Build summary by flag color
    summary = {"green": 0, "yellow": 0, "red": 0}
    flagged_results = []
    for r in results:
        color = r.get("flag", "yellow")
        summary[color] = summary.get(color, 0) + 1
        if color in ("yellow", "red"):
            flagged_results.append({
                "index": r.get("index"),
                "filename": r.get("filename"),
                "flag": color,
                "flag_message": r.get("flag_message"),
                "score": r["score"]["composite"] if r.get("score") else None,
            })

    return JSONResponse(
        {
            "results": results,
            "errors": errors if errors else None,
            "total": len(files),
            "succeeded": len(results),
            "failed": len(errors),
            "summary": summary,
            "flagged_results": flagged_results,
        }
    )


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------


# Entry ids are generated as "<YYYYmmdd>_<HHMMSS>_<12 hex chars>" by
# _save_feedback. Anything else is rejected: entry_id is interpolated into a
# filesystem path below, and without this an id like "../../etc/cron.d/evil"
# would let an unauthenticated caller write, move and delete files anywhere
# the service can reach.
_ENTRY_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{12}$")


def _validate_entry_id(entry_id: str) -> str:
    if not _ENTRY_ID_RE.match(entry_id):
        raise HTTPException(400, "Invalid entry_id format")
    return entry_id


def _verify_feedback_entry(entry_id: str, corrected_text: str) -> dict:
    """Move a pending entry into verified/ with the corrected text attached."""
    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    verified_dir = os.path.join(FEEDBACK_DIR, "verified")

    meta_path = os.path.join(pending_dir, f"{entry_id}.json")
    img_path = os.path.join(pending_dir, f"{entry_id}.png")

    meta = _load_meta(pending_dir, f"{entry_id}.json")
    if meta is None:
        if os.path.exists(meta_path):
            raise HTTPException(422, f"Feedback entry '{entry_id}' is unreadable")
        raise HTTPException(404, f"Feedback entry '{entry_id}' not found")

    meta["corrected_text"] = corrected_text
    meta["status"] = "verified"

    os.makedirs(verified_dir, exist_ok=True)
    verified_meta = os.path.join(verified_dir, f"{entry_id}.json")
    verified_img = os.path.join(verified_dir, f"{entry_id}.png")

    tmp_path = f"{verified_meta}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, verified_meta)

    if os.path.exists(img_path):
        os.replace(img_path, verified_img)
    # Only drop the pending copy once the verified one is safely in place.
    try:
        os.remove(meta_path)
    except FileNotFoundError:
        pass

    return {
        "status": "verified",
        "entry_id": entry_id,
        "corrected_length": len(corrected_text),
    }


@app.post("/feedback/correct")
async def feedback_correct(
    entry_id: str = Form(...),
    corrected_text: str = Form(...),
):
    """Submit corrected text for a previously saved feedback entry.

    The downstream AI or human reviewer sends the correct text for a page
    that scored below threshold. These verified pairs are used for fine-tuning.
    """
    _validate_entry_id(entry_id)
    body = await asyncio.to_thread(_verify_feedback_entry, entry_id, corrected_text)
    logger.info("Feedback verified: %s (%d chars corrected text)", entry_id, len(corrected_text))
    return JSONResponse(body)


# Cap on how many entries a single /feedback/pending call will read.
FEEDBACK_PAGE_MAX = int(os.environ.get("FEEDBACK_PAGE_MAX", "500"))


def _scan_feedback_dir(directory: str, with_size: bool = False) -> tuple[list[str], int]:
    """Return (sorted .json names, total bytes) for a feedback directory.

    Uses scandir so the optional size sweep reuses the directory entry's stat
    instead of issuing a second stat() per file — this directory holds tens of
    thousands of files on a network mount, where per-file stats dominate.
    Pass ``with_size=False`` (the default) to skip them entirely.
    """
    if not os.path.isdir(directory):
        return [], 0
    names: list[str] = []
    total_bytes = 0
    with os.scandir(directory) as it:
        for entry in it:
            if with_size:
                try:
                    total_bytes += entry.stat().st_size
                except OSError:
                    continue
            if entry.name.endswith(".json"):
                names.append(entry.name)
    names.sort()
    return names, total_bytes


def _load_meta(directory: str, name: str) -> Optional[dict]:
    """Read one metadata file, returning None if it is missing or unreadable."""
    try:
        with open(os.path.join(directory, name)) as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


def _collect_feedback_stats() -> dict:
    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    verified_dir = os.path.join(FEEDBACK_DIR, "verified")

    pending_names, pending_bytes = _scan_feedback_dir(pending_dir, with_size=True)
    verified_names, verified_bytes = _scan_feedback_dir(verified_dir, with_size=True)

    engines: dict = {}
    unreadable = 0
    for d, names in [(pending_dir, pending_names), (verified_dir, verified_names)]:
        for name in names:
            meta = _load_meta(d, name)
            if meta is None:
                unreadable += 1
                continue
            eng = meta.get("ocr_engine", "unknown")
            engines[eng] = engines.get(eng, 0) + 1

    pending = len(pending_names)
    verified = len(verified_names)
    used = pending_bytes + verified_bytes
    return {
        "pending": pending,
        "verified": verified,
        "total": pending + verified,
        "ready_for_training": verified >= 50,
        "engines": engines,
        "unreadable": unreadable,
        "disk_usage_mb": round(used / (1024 * 1024), 2),
        "disk_budget_gb": FEEDBACK_MAX_GB,
        "disk_used_pct": round(used / (FEEDBACK_MAX_GB * 1024 ** 3) * 100, 1),
    }


def _collect_pending(limit: int, offset: int) -> dict:
    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    names, _ = _scan_feedback_dir(pending_dir)

    entries = []
    unreadable = 0
    # Only the requested slice is opened. Reading every entry took ~77 s
    # against the current directory, which is far too long to hold a worker.
    for name in names[offset:offset + limit]:
        meta = _load_meta(pending_dir, name)
        if meta is None:
            unreadable += 1
            continue
        entries.append({
            "entry_id": meta.get("id", name[:-len(".json")]),
            "timestamp": meta.get("timestamp"),
            "filename": meta.get("filename"),
            "score": meta.get("score"),
            "flag": meta.get("flag"),
            "ocr_engine": meta.get("ocr_engine"),
            "text_length": len(meta.get("text", "")),
        })

    return {
        "entries": entries,
        "total": len(names),
        "returned": len(entries),
        "limit": limit,
        "offset": offset,
        "unreadable": unreadable,
    }


def _prune_feedback_storage() -> dict:
    """Delete the oldest pending entries until storage fits FEEDBACK_MAX_GB.

    Entry ids are timestamp-prefixed, so sorting by filename is chronological.

    verified/ is never pruned -- those entries carry human-corrected text and
    are the only ones with training value -- but its size counts against the
    budget, so a verified set larger than the budget stops pruning rather than
    deleting every pending entry in a futile attempt to get under it.
    """
    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    verified_dir = os.path.join(FEEDBACK_DIR, "verified")
    budget = FEEDBACK_MAX_GB * (1024 ** 3)

    # Group files by entry id so an entry's .json and .png are removed together.
    entries: dict = {}
    pending_bytes = 0
    if os.path.isdir(pending_dir):
        with os.scandir(pending_dir) as it:
            for e in it:
                try:
                    size = e.stat().st_size
                except OSError:
                    continue
                pending_bytes += size
                entry_id, _, ext = e.name.rpartition(".")
                if not entry_id or ext not in ("json", "png"):
                    continue
                slot = entries.setdefault(entry_id, [0, []])
                slot[0] += size
                slot[1].append(e.path)

    verified_bytes = _scan_feedback_dir(verified_dir, with_size=True)[1]
    total = pending_bytes + verified_bytes

    if total <= budget:
        return {"pruned": 0, "freed_bytes": 0, "total_bytes": total,
                "budget_bytes": int(budget), "over_budget": False}

    freed = 0
    pruned = 0
    # Oldest first.
    for entry_id in sorted(entries):
        if total - freed <= budget:
            break
        size, paths = entries[entry_id]
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                continue
        freed += size
        pruned += 1

    return {"pruned": pruned, "freed_bytes": freed, "total_bytes": total - freed,
            "budget_bytes": int(budget), "over_budget": (total - freed) > budget}


async def _feedback_prune_loop():
    """Enforce the storage budget at startup and periodically thereafter."""
    while True:
        try:
            stats = await asyncio.to_thread(_prune_feedback_storage)
            if stats["pruned"]:
                logger.info(
                    "Feedback prune: removed %d entries, freed %.2f GB, now %.2f GB / %.2f GB budget",
                    stats["pruned"], stats["freed_bytes"] / 1024 ** 3,
                    stats["total_bytes"] / 1024 ** 3, FEEDBACK_MAX_GB,
                )
            if stats["over_budget"]:
                logger.warning(
                    "Feedback storage still over budget (%.2f GB > %.2f GB) — "
                    "verified/ alone exceeds it and is never pruned",
                    stats["total_bytes"] / 1024 ** 3, FEEDBACK_MAX_GB,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Feedback prune failed: %s", e)
        await asyncio.sleep(FEEDBACK_PRUNE_INTERVAL_S)


@app.get("/feedback/stats")
async def feedback_stats():
    """Show feedback storage statistics."""
    # Off the event loop: this walks every metadata file, and the loop it
    # would otherwise block is the one driving the vLLM engine.
    return JSONResponse(await asyncio.to_thread(_collect_feedback_stats))


@app.get("/feedback/pending")
async def feedback_pending(limit: int = FEEDBACK_PAGE_MAX, offset: int = 0):
    """List pending feedback entries awaiting correction (oldest first).

    Entries that cannot be read are skipped and counted in ``unreadable``
    rather than failing the request.
    """
    limit = max(1, min(limit, FEEDBACK_PAGE_MAX))
    offset = max(0, offset)
    return JSONResponse(await asyncio.to_thread(_collect_pending, limit, offset))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_service:app",
        host=HOST,
        port=PORT,
        workers=1,
    )
