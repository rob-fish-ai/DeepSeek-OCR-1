"""Regression tests: each one pins a bug that reached production or was found
while measuring accuracy. The model is stubbed (see conftest.py)."""

import asyncio
import io
import json
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

import api_service
from process import OCRResult, CleanStats, clean_output, compute_flags, score_result
from process.score import _token_similarity

GROUNDED_TEXT = ("<|ref|>title<|/ref|><|det|>[[80, 40, 600, 70]]<|/det|>\n# Tenant Income Certification\n\n"
                 "<|ref|>text<|/ref|><|det|>[[80, 90, 900, 400]]<|/det|>\n"
                 + "The household reports annual income from wages and benefits as listed below. " * 12)
PICTURE_ONLY = "<|ref|>image<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|><｜end▁of▁sentence｜>"
PLAIN_TEXT = "Tenant Income Certification. " + "Annual income from wages and benefits as listed. " * 12
LOOP = ("<table>" + "<tr><td>1.</td><td></td><td></td></tr>" * 900, 7280, "length")


def page_png(lines=45, sparse=False):
    """A full text page, or a sparse title page: one heading, the rest blank."""
    im = Image.new("RGB", (1280, 1920), "white")
    d = ImageDraw.Draw(im)
    if sparse:
        d.text((260, 820), "Annual Report 2025", fill="black", font=ImageFont.load_default(size=72))
    for i in range(0 if sparse else lines):
        d.text((90, 120 + i * 38), "Annual income from wages and benefits for each member", fill="black")
    b = io.BytesIO(); im.save(b, "PNG")
    return b.getvalue()


async def post(client, png, **form):
    return await client.post("/ocr/image", files={"file": ("page_1.png", png, "image/png")},
                             data={k: str(v).lower() if isinstance(v, bool) else v for k, v in form.items()})


# --- fix A: whole page read as a picture ------------------------------------------

def test_picture_only_page_is_reread_with_free_ocr(service):
    client, install = service
    engine = install({"document": (PICTURE_ONLY, 18, "stop"), "free_ocr": (PLAIN_TEXT, 240, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=False)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "free_ocr"]
    assert r["source"] == "free_ocr_fallback"
    assert "Tenant Income Certification" in r["text"]
    assert r["flag"] != "red"


def test_fallback_can_be_disabled(service, monkeypatch):
    client, install = service
    monkeypatch.setattr(api_service, "FALLBACK_FREE_OCR", False)
    engine = install({"document": (PICTURE_ONLY, 18, "stop"), "free_ocr": (PLAIN_TEXT, 240, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=False)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document"]
    assert r["text"] == ""


def test_normal_page_does_not_trigger_fallback(service):
    client, install = service
    engine = install({"document": (GROUNDED_TEXT, 300, "stop"), "free_ocr": (PLAIN_TEXT, 240, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document"]
    assert r["source"] == "document"


# --- fix C: sparse pages ----------------------------------------------------------

def test_sparse_page_is_read_not_skipped(service):
    client, install = service
    install({"document": ("<|ref|>title<|/ref|><|det|>[[1, 1, 2, 2]]<|/det|>\n# Annual Report 2025", 12, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(sparse=True), retry=True)
    r = asyncio.run(go()).json()
    assert r["ocr_engine"] != "skipped"
    assert "Annual Report 2025" in r["text"]
    assert r["flag"] != "red", "a correct title page must not be flagged red"


def test_sparse_short_text_not_capped_but_empty_still_red():
    short = OCRResult(raw_text="Sincerely, Jane Smith", clean_text="Sincerely, Jane Smith",
                      num_tokens=8, max_tokens=8192, clean_stats=CleanStats(), sparse_page=True)
    score_result(short)
    assert short.score.composite > 0.5
    assert compute_flags(short)["flag"] != "red"

    empty = OCRResult(raw_text="", clean_text="", num_tokens=1, max_tokens=8192,
                      clean_stats=CleanStats(), sparse_page=True)
    score_result(empty)
    assert compute_flags(empty)["flag"] == "red"


def test_dense_page_short_output_still_capped():
    """Only sparse pages are exempt: a full page returning 20 chars is a failure."""
    r = OCRResult(raw_text="Form 1040 page 2", clean_text="Form 1040 page 2", num_tokens=8,
                  max_tokens=8192, clean_stats=CleanStats(), sparse_page=False)
    score_result(r)
    assert r.score.composite <= 0.30


# --- fix D: prompt=ocr ------------------------------------------------------------

def test_ocr_prompt_keeps_text_inside_ref_tags():
    raw = ("<|ref|>Contents<|/ref|><|det|>[[116, 100, 208, 118]]<|/det|>\n"
           "<|ref|>1 Introduction<|/ref|><|det|>[[115, 143, 252, 158]]<|/det|>\n")
    assert clean_output(raw) == ""                          # document-mode semantics unchanged
    kept = clean_output(raw, ref_is_content=True)
    assert "Contents" in kept and "1 Introduction" in kept
    assert "[[" not in kept and "<|" not in kept


def test_ocr_prompt_end_to_end(service):
    client, install = service
    raw = "".join(f"<|ref|>Line {i} of the certification form<|/ref|><|det|>[[1, {i}, 2, {i}]]<|/det|>\n"
                  for i in range(40))
    install({"ocr": (raw, 600, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), prompt="ocr", retry=False)
    r = asyncio.run(go()).json()
    assert "Line 7 of the certification form" in r["text"]
    assert r["flag"] != "red"


# --- the 2026-09-14..24 freezes -----------------------------------------------------

def test_similarity_is_linear_time_on_looping_output():
    a = "Nombre del beneficiario | " * 40 + "1 2 3 4 5 6 7 8 9 0 " * 900
    b = a.replace("beneficiario", "beneficiaria", 5)
    t = time.perf_counter()
    for _ in range(9):
        _token_similarity(a, b)
    assert time.perf_counter() - t < 1.0      # was ~110 s with SequenceMatcher(autojunk=False)


def test_looping_page_does_not_block_health(service):
    client, install = service
    install({"document": LOOP, "free_ocr": LOOP})

    async def go():
        async with client() as c:
            worst, done = [0.0], asyncio.Event()

            async def poll():
                while not done.is_set():
                    t = time.perf_counter(); await c.get("/health")
                    worst[0] = max(worst[0], time.perf_counter() - t)
                    await asyncio.sleep(0.05)
            p = asyncio.create_task(poll())
            await post(c, page_png(), retry=True)
            done.set(); await p
            return worst[0]
    assert asyncio.run(go()) < 2.0


# --- request logging ----------------------------------------------------------------

def test_request_id_echoed_and_no_text_logged(service):
    client, install = service
    install({"document": (GROUNDED_TEXT, 300, "stop")})

    async def go():
        async with client() as c:
            return await c.post("/ocr/image", files={"file": ("J_Smith_SSN.png", page_png(), "image/png")},
                                data={"retry": "false"}, headers={"X-Request-ID": "test_abc-1"})
    r = asyncio.run(go())
    assert r.headers["x-request-id"] == "test_abc-1"
    log = open(api_service.REQUEST_LOG_FILE).read()
    record = [json.loads(l) for l in log.splitlines() if '"test_abc-1"' in l][-1]
    assert record["pages"][0]["flag"] and record["inference"]["calls"] == 1
    assert "Tenant Income" not in log and "J_Smith" not in log


# --- fix B: rescuing pages whose generation runs away --------------------------------

def sideways_png():
    """Text lines running vertically, as on a landscape page scanned sideways."""
    im = Image.new("RGB", (1920, 1280), "white")
    d = ImageDraw.Draw(im)
    for i in range(40):
        d.text((90, 60 + i * 30), "Annual income from wages and benefits for each household member", fill="black")
    im = im.rotate(90, expand=True)
    b = io.BytesIO(); im.save(b, "PNG")
    return b.getvalue()


def first_then(first, rest):
    """Engine script: `first` on the first call, `rest` afterwards."""
    return lambda n: first if n == 1 else rest


def test_looping_page_first_retried_without_enhancement(service):
    """The default enhancement itself sends some pages into a loop. Eval: two
    pages rescued by the no-enhancement retry regressed when it was dropped."""
    client, install = service
    engine = install({"document": first_then(LOOP, (GROUNDED_TEXT, 300, "stop"))})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "document"]
    assert r["preset"] == "none" and r["hit_length_limit"] is False


def test_looping_page_rescued_by_free_ocr(service):
    client, install = service
    engine = install({"document": LOOP, "free_ocr": (PLAIN_TEXT, 240, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "document", "free_ocr"]
    assert r["source"] == "free_ocr" and r["hit_length_limit"] is False
    assert r["flag"] != "red"


def test_looping_page_rescued_by_split_when_free_ocr_also_loops(service):
    client, install = service
    # calls: 1 adaptive, 2 no-enhancement, 3 free_ocr, 4-5 the two halves
    engine = install({"document": lambda n: LOOP if n <= 2 else (GROUNDED_TEXT, 300, "stop"), "free_ocr": LOOP})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "document", "free_ocr", "document", "document"]
    assert r["source"] == "document+split" and r["hit_length_limit"] is False


def test_sideways_page_tries_rotation_first(service):
    client, install = service
    engine = install({"document": first_then(LOOP, (GROUNDED_TEXT, 300, "stop")), "free_ocr": LOOP})

    async def go():
        async with client() as c:
            return await post(c, sideways_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "document"]
    assert r["rotation"] == 270 and r["flag"] != "red"


def test_non_looping_failure_keeps_contrast_presets(service):
    client, install = service
    weak = ("<|ref|>text<|/ref|><|det|>[[1, 1, 2, 2]]<|/det|>\nIncome " + "wages " * 30, 60, "stop")
    engine = install({"document": weak, "free_ocr": (PLAIN_TEXT, 240, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    r = asyncio.run(go()).json()
    assert "free_ocr" not in engine.calls
    assert r["attempts"] == len(engine.calls)


def test_max_retries_zero_does_not_crash(service, monkeypatch):
    """select_best_result([]) used to raise when MAX_RETRIES=0."""
    client, install = service
    monkeypatch.setattr(api_service, "MAX_RETRIES", 0)
    install({"document": (GROUNDED_TEXT, 300, "stop")})

    async def go():
        async with client() as c:
            return await post(c, page_png(), retry=True)
    assert asyncio.run(go()).status_code == 200


# --- loops that stop on their own ------------------------------------------------------

DEGENERATE = ("8/8/8/8/8/8/8/8/8/8/8 \n\n" * 30, 900, "stop")    # ended with a stop token


def test_loop_that_stops_on_its_own_is_not_green():
    text = DEGENERATE[0]
    r = OCRResult(raw_text=text, clean_text=text, num_tokens=900, max_tokens=8192, clean_stats=CleanStats())
    score_result(r)
    assert r.score.composite <= 0.35
    assert "degenerate_output" in [d["code"] for d in compute_flags(r)["details"]]


def test_repetitive_but_real_form_is_not_degenerate():
    """Forms repeat labels. Lowest real output measured: 0.254 distinct-word ratio."""
    from process.score import is_degenerate_output
    rows = "".join(f"Name: ____ Date of birth: ____ Relationship: ____ Income source {i}: ____\n" for i in range(60))
    assert not is_degenerate_output(rows)


def test_degenerate_sideways_page_gets_rotated(service):
    client, install = service
    engine = install({"document": first_then(DEGENERATE, (GROUNDED_TEXT, 300, "stop"))})

    async def go():
        async with client() as c:
            return await post(c, sideways_png(), retry=True)
    r = asyncio.run(go()).json()
    assert engine.calls == ["document", "document"]
    assert r["rotation"] == 270 and "8/8/8" not in r["text"]
