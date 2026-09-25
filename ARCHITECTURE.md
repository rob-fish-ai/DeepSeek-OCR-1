# DeepSeek-OCR API Service — Full Architecture

## Table of Contents

- [System Overview](#system-overview)
- [Request Lifecycle](#request-lifecycle)
- [Concurrency Model](#concurrency-model)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Inference Engine](#inference-engine)
- [Post-processing Pipeline](#post-processing-pipeline)
- [Quality Scoring System](#quality-scoring-system)
- [Retry Logic](#retry-logic)
- [Pre-flight Detection](#pre-flight-detection)
- [External OCR Routing](#external-ocr-routing)
- [Feedback System](#feedback-system)
- [API Endpoints](#api-endpoints)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Key Dependencies](#key-dependencies)
- [Performance Characteristics](#performance-characteristics)

---

## System Overview

DeepSeek-OCR API is a production-ready FastAPI service that extracts text from document images using the DeepSeek-OCR vision-language model. It runs on a single GPU via vLLM's `AsyncLLMEngine` for concurrent request handling.

```
                          External Services (1..N)
                                  │
                          POST /ocr/image
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     FastAPI (uvicorn)     │  Port 8000
                    │     1 worker process      │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
        Pre-flight         Pre-flight          Pre-flight
        Detection          Detection           Detection
        (~1ms)             (~1ms)              (~1ms)
              │                  │                   │
              ▼                  ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │              vLLM AsyncLLMEngine                     │
    │         Continuous Batching Scheduler                │
    │    ┌─────────┐ ┌─────────┐ ┌─────────┐              │
    │    │ Req A   │ │ Req B   │ │ Req C   │  GPU         │
    │    │ prefill │ │ decode  │ │ prefill │  Memory      │
    │    └─────────┘ └─────────┘ └─────────┘  80%         │
    └─────────────────────────────────────────────────────┘
              │                  │                   │
              ▼                  ▼                   ▼
        Post-process       Post-process        Post-process
        + Score            + Score              + Score
              │                  │                   │
              ▼                  ▼                   ▼
        JSON Response      JSON Response       JSON Response
```

---

## Request Lifecycle

A single `/ocr/image` request with `retry=true` follows this path:

```
1. Upload received
   ├── Validate file size (≤ 20 MB)
   └── Decode image (PIL) + EXIF correction

2. Pre-flight check (~1ms)
   └── Blank page? → RED flag, skip OCR, return immediately
       (sparse pages are NOT skipped -- see Pre-flight Detection)

3. Attempt 1: "adaptive" preset
   ├── Enhance image (adaptive contrast/sharpness for grayscale scans)
   ├── Tile image into 640×640 crops (2-9 tiles)
   ├── Tokenize tiles → vLLM multi-modal input
   ├── AsyncLLMEngine.generate() → raw text + token count
   │   └── Whole page labelled one picture? → re-read with free_ocr
   ├── Post-process: strip grounding tags, clean tables, collapse
   │   repetition, deduplicate sections, normalize whitespace
   ├── Score (weighted metrics → composite 0.0-1.0)
   └── Score ≥ 0.60? → STOP, use this result

4. Retries, chosen by how attempt 1 failed (see Retry Logic)
   ├── Ran out of room, or page looks sideways → rescue ladder
   └── Otherwise → "none" and "strong" contrast presets

5. Best result selected (highest composite score)

6. Flag assignment
   ├── GREEN (≥ 0.70): Good quality
   ├── YELLOW (0.50-0.69): Spot-check recommended
   └── RED (< 0.50): Manual review required

6. External OCR routing check
   ├── Clean text ≤ 10 chars → needs_external_ocr: true
   └── Token limit hit (≥85% max) + >90% hallucinated + low score
       → needs_external_ocr: true (incomplete_extraction)

7. Feedback storage
   └── Score < 0.70 → save image + metadata to feedback/pending/

8. JSON response returned
```

---

## Concurrency Model

The service uses vLLM's `AsyncLLMEngine` (v0.8.4) for true concurrent GPU inference.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    AsyncLLMEngine                         │
│                                                           │
│  ┌─────────────┐    ┌──────────────────┐                 │
│  │ Request      │    │ Continuous        │                │
│  │ Queue        │───▶│ Batching          │                │
│  │              │    │ Scheduler         │                │
│  │ req_A ──────▶│    │                   │                │
│  │ req_B ──────▶│    │ Interleaves       │   ┌─────────┐ │
│  │ req_C ──────▶│    │ prefill/decode    │──▶│  GPU    │ │
│  │ req_D ──────▶│    │ across requests   │   │ (single)│ │
│  └─────────────┘    └──────────────────┘   └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

- **Single GPU, multiple requests**: The scheduler interleaves prefill (processing a new image) and decode (generating tokens for in-flight requests) within the same GPU.
- **No semaphore needed**: Unlike the sync `LLM` class which crashes on concurrent calls, `AsyncLLMEngine` handles batching internally.
- **`max_num_seqs`**: Controls maximum concurrent sequences on GPU (set to `MAX_CONCURRENCY` from config, default 100).
- **1 uvicorn worker**: Always 1 process to prevent model duplication in GPU memory.
- **CPU parallelism**: Image preprocessing (tiling, tokenization) runs in thread pool via `run_in_executor`.

### Why Not Sync LLM?

vLLM's sync `llm.generate()` triggers `AssertionError` in the scheduler when called concurrently — the prefill logic asserts single-request-at-a-time. This was the original architecture with `Semaphore(1)`, which serialized all requests to one at a time.

### Why Not Separate vLLM Server?

The model's HuggingFace code (`modeling_deepseekv2.py`) imports `LlamaFlashAttention2`, which doesn't exist in the installed transformers version. The in-process `AsyncLLMEngine` bypasses this by using the custom registered `DeepseekOCRForCausalLM` class directly. The separate server (`vllm.entrypoints.openai.api_server`) spawns a subprocess that loads HF code first, hitting the import error.

---

## Image Processing Pipeline

### Tiling System

Images are split into 640×640 tiles before model inference. This is how the model sees high-resolution documents.

```
Config (config.py):
  BASE_SIZE  = 1024    # Base dimension for aspect ratio calculation
  IMAGE_SIZE = 640     # Tile size
  CROP_MODE  = True    # Enable tiling
  MIN_CROPS  = 2       # Minimum tiles
  MAX_CROPS  = 9       # Maximum tiles (3×3 grid)
```

```
Input Image                     Tile Grid
┌─────────────────┐            ┌──────┬──────┬──────┐
│                  │            │ Tile │ Tile │ Tile │
│  1700 × 2200    │   resize   │  1   │  2   │  3   │
│  (original)     │ ────────▶  ├──────┼──────┼──────┤
│                  │            │ Tile │ Tile │ Tile │
│                  │            │  4   │  5   │  6   │
│                  │            └──────┴──────┴──────┘
└─────────────────┘              1280 × 1280
                                 (2×3 grid = 6 tiles)
```

**Optimal resolution: 1280×1920** — maps perfectly to a 2×3 tile grid (6 tiles at 640×640) with no resizing artifacts. Higher resolutions are downscaled to fit the tile grid anyway.

### Tile Calculation Flow

```
dynamic_preprocess() in image_process.py:
  1. Calculate target aspect ratio from available tile configs
  2. Find closest match: (cols × 640, rows × 640)
  3. Resize image to fit the grid
  4. Split into 640×640 tiles
  5. Each tile becomes a separate visual token sequence
```

### Image Enhancement

Three preset strategies applied before tiling:

| Preset | Contrast | Sharpness | When Used |
|--------|----------|-----------|-----------|
| `adaptive` | Auto (target RMS 0.186) | 1.5× | Attempt 1 — adjusts grayscale scans to match known-good reference |
| `none` | 1.0× | 1.0× | Attempt 2 — raw image, no enhancement |
| `strong` | 1.5× | 2.0× | Attempt 3 — aggressive enhancement for difficult scans |

Adaptive enhancement only activates for grayscale images (detected by channel difference < 10). Color images pass through unchanged.

---

## Inference Engine

### Model Details

```
Model:          DeepSeek-OCR (custom architecture)
Class:          DeepseekOCRForCausalLM (registered with vLLM ModelRegistry)
Precision:      bfloat16
VRAM:           ~6.2 GB (at GPU_MEM_UTIL=0.80)
Max context:    8,192 tokens
Temperature:    0.0 (deterministic)
```

### Generation Parameters

```python
SamplingParams(
    temperature=0.0,           # Deterministic output
    max_tokens=8192,           # Max output tokens
    skip_special_tokens=False, # Keep special tokens for post-processing
    include_stop_str_in_output=True,
    logits_processors=[
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,                    # 20-token n-gram window
            window_size=50,                   # Look-back window
            whitelist_token_ids={128821, 128822},  # Allow table tags
            max_consecutive_empty_cells=30,   # Cap empty <td></td> runs
        )
    ],
)
```

### N-gram No-Repeat Processor

Custom logits processor that prevents generation loops by blocking repeated 20-token sequences. Table structure tokens (`<td>`, `</td>`) are whitelisted to allow natural table generation, but empty cell runs are capped at 30 consecutive.

### Prompt Templates

| Key | Prompt | Use Case |
|-----|--------|----------|
| `document` | `<image>\n<\|grounding\|>Convert the document to markdown.` | Default — structured document extraction |
| `ocr` | `<image>\n<\|grounding\|>OCR this image.` | General OCR |
| `free_ocr` | `<image>\nFree OCR.` | Without layout/grounding |
| `figure` | `<image>\nParse the figure.` | Charts, diagrams |
| `describe` | `<image>\nDescribe this image in detail.` | Image description |

---

## Post-processing Pipeline

Raw model output goes through 5 cleanup stages in `postprocess.py`:

```
Raw Model Output
    │
    ▼
1. Strip grounding tags
   Remove <|ref|>...<|/ref|>, <|det|>...<|/det|>, coordinate arrays [[x,y,w,h]]
   Remove end-of-sentence marker
    │
    ▼
2. Table cleanup (_collapse_empty_table_cells)
   ├── Trim empty <td></td> runs (max 15 per row)
   ├── Remove entirely empty rows
   ├── Remove hallucinated numbered empty-row sequences
   ├── Trim bloated tables (>100 empty cells, cap 60 rows)
   ├── Handle unclosed <table> tags
   ├── Collapse repetitive table rows (>80% duplicate)
   ├── Remove diagonal repetition (single value >40% of cells)
   └── Remove empty tables
    │
    ▼
3. Collapse repeating patterns (_collapse_repeating_patterns)
   ├── Incrementing number + digit filler
   ├── Long digit-space runs (20+ or 8+)
   ├── Dot-separated digits (6+)
   ├── Single char repeated with spaces (12+)
   └── Numbered sequences (15+)
    │
    ▼
4. Section deduplication (_deduplicate_sections)
   ├── Split by markdown headers (# ## ###)
   ├── Detect duplicate headers
   ├── Keep longer version (or more columns for expanded table variants)
   └── Track dedup chars separately from hallucination for scoring
    │
    ▼
5. Whitespace normalization
   ├── Collapse 3+ newlines → 2
   └── Collapse multiple spaces → 1
    │
    ▼
Clean Text Output
```

### CleanStats Tracking

Post-processing tracks removal categories separately:
- `dedup_chars_removed` — Characters removed by section deduplication (not hallucination)
- `hallucination_chars_removed` — Characters removed as fabricated content

This distinction is critical for accurate hallucination scoring — dedup removal should not penalize the score.

---

## Quality Scoring System

Six independent metrics, each normalized to 0.0-1.0, combined with fixed weights:

```
Composite = 0.30 × hallucination_ratio
          + 0.30 × token_efficiency
          + 0.15 × content_density
          + 0.15 × structural_integrity
          + 0.10 × repetition_density
          + 0.00 × self_consistency
```

**The composite measures whether generation behaved normally, not whether the
text is correct.** Every metric is computed from the output string; none
compares against the image, so fluent-but-wrong OCR is invisible by
construction. Treat a green flag as "the model did not visibly malfunction".

Two hard caps override the weighted sum:

| Condition | Cap |
|-----------|-----|
| Clean text ≤ 10 chars | 0.10 |
| Clean text ≤ 30 chars | 0.30 |
| Ran out of room **and** output collapsed (see repetition_density) | 0.35 |

The caps do most of the work at the failure end: in a sample of 1,499 stored
results, 40.8% were exactly 0.100 and 4.7% exactly 0.300.

### Metric Details

#### hallucination_ratio (weight: 0.30)

Measures how much raw output survived post-processing.

```
effective_raw = raw_length - grounding_tag_chars - dedup_chars
ratio = clean_length / effective_raw
```

- Grounding tags (`<|ref|>`, `<|det|>`, coordinates) are excluded from the denominator — they are expected format, not fabrication.
- Dedup-removed content is also excluded.
- If `effective_raw ≤ clean_length`: returns 1.0 (everything was tags).

#### self_consistency (weight: 0.00)

Pairwise text similarity between multiple OCR runs of the same image, via
`SequenceMatcher(..., autojunk=False)`.

**Carries no weight in the composite.** It can only be computed when several
attempts exist, so at report time it is always 1.0 — a constant offset with no
discriminative power. It is still computed, reported in the breakdown, and
used by `select_best_result` to rank retry candidates and break ties.

`autojunk=False` is required, not cosmetic: with the default, any character
appearing in >1% of a string longer than 200 chars is treated as junk, which
is nearly every letter in real text. Two near-identical tables score 0.009
instead of 0.949, which used to crush table-heavy pages to ~0.12 composite.

#### token_efficiency (weight: 0.30)

Whether generation was cut off, and how much of what it emitted was real.

```
if finish_reason != "length" → 1.0 (the model stopped on its own)
otherwise:
  effective_raw = raw_length - grounding_tag_chars
  survival      = clean_length / effective_raw
  score         = clamp(survival, 0.1, 0.8)      # capped: the page is incomplete
```

Whether the model ran out of room is taken from the engine's `finish_reason`,
**not** inferred from `num_tokens / MAX_TOKENS`. The usable budget is
`max_model_len` minus the prompt — about 7,280 tokens for a 144-DPI A4 page,
where the prompt costs ~913 — so the configured `MAX_TOKENS` of 8,192 is never
reachable and any threshold expressed as a fraction of it is unreliable.

#### content_density (weight: 0.15)

Absolute volume of extracted text.

```
≥500 chars → 1.0, ≥100 → 0.5-1.0, ≥20 → 0.2-0.5, else 0.1
```

An earlier pixel-ratio variant scaled this against image area. It was removed:
it scored a normal A4 page ~0.26 where the char-count path scored the same
text 1.0, and because the two paths were compared against each other by the
PDF and batch endpoints, retries were systematically discarded.

#### structural_integrity (weight: 0.15)

Presence of recognizable patterns (does NOT require any specific structure):

| Signal | Credit |
|--------|--------|
| Markdown headers (`# ##`) | +1.0 |
| Tables with content cells | +1.0 |
| Meaningful text (>30 chars non-markup) | +1.0 |
| Data patterns (dates, amounts, names, emails, phones) | +1.0 |

Scoring: 1 signal = 0.75, 2 = 0.875, 3 = 0.95, 4 = 1.0. Substantial text with no signals = 0.4.

#### repetition_density (weight: 0.10)

Detects generation loops by comparing how compressible the **tail** of the raw
output is against its **head**. A loop decays over time, so its last quarter
collapses while its first quarter still looks like text; uniform repetition —
a form with repeated labels, an invoice with similar line items — compresses
the same at both ends and is not penalised.

```
score = 1.0 - (zlib(head) - zlib(tail)) / zlib(head)
```

- Measured on **raw** output: post-processing strips runaway rows and repeated
  patterns, so by the time text is cleaned the evidence is often gone.
- Grounding tags and markup are stripped before measuring.
- Below 2,000 chars: returns 1.0 (zlib overhead makes short buffers noise).

This replaced an n-gram counting version that saturated to 0.0 on any
structured document — an invoice with twelve line items and a true generation
loop both scored 0.000 — making it a flat penalty on exactly the forms and
invoices this service handles rather than a signal.

*Blind spot:* a loop made purely of empty table cells is stripped along with
the markup and reads as 0.0. Those are caught by `token_efficiency` instead,
since post-processing removes the bloat and survival collapses. Measuring with
markup included was rejected — a legitimate dense table compresses just as
hard and would be capped as a loop.

### Hard Caps

Applied after the weighted sum, in `_apply_composite`:

```
clean_text ≤ 10 chars                     → composite capped at 0.10
clean_text ≤ 30 chars                     → composite capped at 0.30
finish_reason == "length" AND (           → composite capped at 0.35
    tail-vs-head degeneration > 0.45
    OR whole-output compression < 0.025
)
```

The loop cap is gated on `finish_reason` because a legitimate form with forty
repeated label rows compresses to ~0.028 — close to a true loop's 0.006-0.016 —
and must not be capped. Pages that stop on their own are never affected,
however repetitive they are. These thresholds are calibrated on constructed
examples, not labelled production data; see *Calibration debt* below.

### Flag Assignment

| Flag | Composite | Meaning |
|------|-----------|---------|
| GREEN | ≥ 0.70 | Good quality, use as-is |
| YELLOW | 0.50-0.69 | Spot-check recommended |
| RED | < 0.50 | Manual review required |

**Warning downgrades** (green → yellow):
- `hallucination_ratio < 0.25` — severe hallucination
- `token_efficiency < 0.2` — stuck generation loop

### Request logging

Every request gets an ID. A caller may supply one in `X-Request-ID` (letters,
digits and `_.:-`, up to 128 chars — anything else is replaced); otherwise one is
generated. It is returned in the `X-Request-ID` response header, stamped on every
application log line as `[<id>]`, and used as the prefix of vLLM's own engine
request IDs, so `Added/Finished request <id>-xxxxxxxx` lines trace back too.

Each request also writes one JSON line to `REQUEST_LOG_FILE`:

```json
{"ts": "...", "request_id": "...", "method": "POST", "path": "/ocr/pdf",
 "status": 200, "duration_ms": 98158, "client": "154.54.102.19",
 "params": {"prompt": "document", "dpi": 144, "raw": false, "retry": true},
 "uploads": [{"bytes": 712345, "sha256_16": "…", "ext": ".pdf"}],
 "pages": [{"page": 1, "flag": "green", "score": 0.966, "codes": [], "chars": 5153,
            "tokens": 1780, "attempts": null, "preset": null, "engine": "deepseek",
            "needs_external_ocr": false}],
 "inference": {"calls": 3, "tokens": 10480, "hit_length_limit": 1, "ms": 95000,
               "detail": [{"engine_request_id": "…", "prompt": "document",
                           "prompt_tokens": 913, "tokens": 1780,
                           "hit_length_limit": false, "ms": 23000}]},
 "error": "HTTP 400: …"}
```

**No OCR text or images are logged.** Uploads are identified by size, extension
and a 16-hex SHA-256 prefix, so a caller can confirm which file a request carried
by hashing their own copy. Filenames are withheld unless `LOG_FILENAMES=true`.
Validation errors record field names only, because FastAPI's error detail echoes
submitted values, which can be an entire base64-encoded document.

`/health`, `/`, and the docs pages get the header but no log line — the
supervisor polls `/health` every 30 seconds.

The request-tracking state lives in `request_log.py`, not `api_service.py`.
Started as `python api_service.py`, that file executes twice (as `__main__`, then
again when uvicorn imports `api_service:app`), so module-level state there
exists twice and a context variable set by one copy would not be read by the
other's log filter.

### Batching vs reproducibility

`MAX_CONCURRENCY` (vLLM `max_num_seqs`, env-overridable) trades throughput
against determinism. Batching changes decode numerics, so at temperature 0 a
page that is perfectly stable alone can diverge — sometimes into a generation
loop that runs to the length limit.

Measured on this A100, 12 mixed pages, 8 concurrent clients, retry enabled:

| `MAX_CONCURRENCY` | Throughput | Corrupted | Reproducible |
|---|---|---|---|
| 1 | 21.0 s/page | 0/12 | yes — byte-identical token counts across repeats |
| 24 | 9.8 s/page | 2/12 | no — same page gave 1214/1305/1305/1412 tokens |

Intermediate values do not help predictably. A single page swept across
concurrency 2/4/6/8/12/16 looped 6%/68%/6%/56%/31%/37% of the time — the rate
tracks exact batch composition at each scheduler step, not the cap itself.

Corrupted pages are **flagged, not silent**: a loop trips the 0.35 cap and
reports `truncated_output`, so a caller routing red pages to review loses
correctness only if it ignores the flag. Retry recovers a large share —
at concurrency 8 the same page went from 87% corrupted with `retry=false`
to 31% with `retry=true`.

Choose 1 when reproducibility matters (the same document must OCR identically
twice) and the higher cost is acceptable; keep the default when throughput
dominates and the flags are acted on.

### Calibration debt

There is no ground truth for any of this. The feedback corpus stores only
results scoring below 0.70, so it contains no known-good pages to validate
against, and none of the ~39,000 stored entries has verified corrected text.
Thresholds here were chosen from constructed examples plus the observed
behaviour of real pages, not measured against labelled data.

To close this, collect corrected text via `POST /feedback/correct` and re-score
offline. Stored entries now record `num_tokens` and the full `score_variables`
breakdown for exactly this purpose — entries archived before that change have
`num_tokens: 0` and cannot be used to calibrate token-related metrics.

---

## Retry Logic

When `retry=true` (default) and attempt 1 scores < 0.60, the retries depend on
*how* it failed (`_rescue_ladder`). Each stops as soon as an attempt scores ≥ 0.60.

| Attempt 1 failed because… | Then tries, in order |
|---|---|
| It ran out of room (`finish_reason == "length"`) on an upright page | `free_ocr` prompt → the page read as two halves |
| The page looks sideways (it ran out of room or not) | rotated 270° → rotated 90° → `free_ocr` |
| Anything else | `"none"` preset → `"strong"` preset (contrast 1.5×, sharpness 2×) |

Why: on pages that looped under every contrast preset, `free_ocr` rescued 3 of 4
real table pages, reading the page in halves (each half needs about half the
output budget) rescued the 4th, and the correct rotation fixed 4 of 4 sideways
pages. The contrast presets rescued none of them. At most 4 model calls per page
— the same order of cost as the old three presets.

"Looks sideways" is a cheap ink-profile test, consulted only for pages that
already failed, and only to order the attempts: on labelled pages it was right
8/8; across the failure corpus about half its calls were really sparse pages or
ID cards, which costs extra attempts, not correctness. Both rotations are tried
because both occur in production. Disable the ladder with `LOOP_RESCUE=false`.

Separately, whenever the model labels an entire page as one picture region and
emits nothing else — which cleanup then discards, leaving empty text — the page
is immediately re-read with `free_ocr` (`FALLBACK_FREE_OCR`, default on). It
recovered 12 of 12 such pages. `free_ocr` is a fallback only: as the default
prompt it scored lower on forms and tables (F1 0.702 vs 0.731 on the simulated
set, 66 pages worse against 31 better).

After all attempts, `select_best_result()` re-scores `self_consistency` using the full result set and picks the highest composite.

---

## Pre-flight Detection

One instant check (~1ms) runs before any OCR processing:

### Blank Page Detection

```python
def is_blank_page(image):
    gray = to_grayscale(image)
    if gray.std() >= 5.0:        # Has variation → not blank
        return False
    dark_ratio = pixels_below_240 / total_pixels
    return dark_ratio < 0.02     # Almost no dark pixels → blank
```

Result: RED flag, `blank_page` detail, zero text, `ocr_engine: "skipped"`.

### Low-Quality Scan Detection

```python
def is_low_quality_scan(image):
    # Find bounding box of content (rows/cols with >1% dark pixels)
    content_area_ratio = (content_h × content_w) / (page_h × page_w)
    return content_area_ratio < 0.12  # Content too small
```

**No longer skips pages by default** (`SKIP_LOW_QUALITY_SCANS=false`). The
12% content-area test was meant for shrunken or thumbnail scans, but it also
matched every cover page, chapter divider and short closing page, and small ID
cards photographed on a full page — all returned as empty text. Such pages are
now read normally. The same test now marks them `sparse_page` for scoring, so a
short correct answer ("Annual Report 2025") is neither capped at 0.30 nor
flagged red. Empty output from a sparse page is still red.

A post-OCR "blank page" verdict now needs empty or boilerplate-only output (a
page number, a divider). It used to fire on any output under 20 characters,
which marked correct short text as blank.

---

## External OCR Routing

The `needs_external_ocr` flag tells callers to route the page to an external OCR service. Two conditions trigger it:

### Condition 1: OCR Failed (empty extraction)

```
clean_text ≤ 10 chars AND composite < 0.60
→ needs_external_ocr: true
→ flag_detail: "ocr_failed"
```

### Condition 2: Incomplete Extraction

For dense forms where the model extracts headers but hallucinates table content:

```
num_tokens ≥ MAX_TOKENS × 0.85      (hit token limit)
AND clean_len / raw_len < 0.10      (>90% removed as hallucination)
AND composite < 0.60                 (low quality score)
→ needs_external_ocr: true
→ flag_detail: "incomplete_extraction"
```

Example: A dense HUD compliance form — model generates 7,280 tokens but only 287 chars survive post-processing (1% retained). Headers extracted correctly, but all table content was hallucinated filler.

---

## Feedback System

Automatically stores low-scoring results for future LoRA fine-tuning.

### Storage Flow

```
OCR Result
    │
    ├── Score ≥ 0.70 → NOT saved (no disk wasted)
    │
    └── Score < 0.70
        │
        ▼
  feedback/pending/
  ├── {timestamp}_{uuid}.png    ← Original image
  └── {timestamp}_{uuid}.json   ← OCR result + metadata
        │
        │  POST /feedback/correct
        │  (human or AI sends corrected text)
        ▼
  feedback/verified/
  ├── {timestamp}_{uuid}.png    ← Same image
  └── {timestamp}_{uuid}.json   ← Metadata + corrected_text
        │
        │  50+ verified pairs accumulated
        ▼
  Ready for LoRA fine-tuning
```

### What Gets Saved

| Condition | Saved? |
|-----------|--------|
| GREEN (≥ 0.70) | No |
| YELLOW (0.50-0.69) | Yes |
| RED (< 0.50) | Yes |

Estimated ~5% of pages saved. Storage: ~210 KB per entry (image + JSON metadata).

Disable with: `FEEDBACK_ENABLED=false`

#### Retention

Feedback storage was previously unbounded and reached 38,963 entries / 12.2 GB
at roughly 111 MB/day. A background task now enforces a disk budget
(`FEEDBACK_MAX_GB`, default 20) at startup and every `FEEDBACK_PRUNE_INTERVAL_S`,
deleting the oldest pending entries until storage fits. Entry ids are
timestamp-prefixed, so ordering by filename is chronological.

`verified/` is never pruned — those entries carry human-corrected text and are
the only ones with training value — but its size counts against the budget. If
`verified/` alone exceeds the budget, pruning stops and logs a warning rather
than deleting every pending entry trying to get under it.

The default budget sits above current usage, so enabling it does not delete
anything; lower it to reclaim space. `GET /feedback/stats` reports
`disk_budget_gb` and `disk_used_pct`.

Metadata is written via a temp file and `os.replace`. A plain `open("w")` left
a zero-byte file behind whenever the process was killed mid-write, which is how
41 unreadable entries and 104 zero-byte images accumulated before this was
fixed.

---

## API Endpoints

### OCR Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ocr/image` | POST | OCR a single image with scoring + retry |
| `/ocr/image/base64` | POST | Same but accepts base64-encoded image |
| `/ocr/pdf` | POST | OCR all pages of a PDF (concurrent page processing) |
| `/ocr/batch` | POST | OCR multiple images in one request |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model info, GPU info |
| `/` | GET | Service info and links |

### Feedback Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/feedback/stats` | GET | Counts, disk usage, training readiness |
| `/feedback/pending` | GET | List entries awaiting correction |
| `/feedback/correct` | POST | Submit corrected text for an entry |

### Response Format

All OCR endpoints return:

```json
{
  "text": "cleaned markdown output",
  "raw_text": "raw model output with grounding tags",
  "num_tokens": 1024,
  "score": {
    "composite": 0.95,
    "variables": {
      "self_consistency": 1.0,
      "hallucination_ratio": 0.97,
      "token_efficiency": 1.0,
      "structural_integrity": 0.88,
      "repetition_density": 0.95,
      "content_density": 0.85
    },
    "weights": { ... }
  },
  "flag": "green",
  "flag_message": "Good quality (0.95).",
  "flag_details": [],
  "attempts": 1,
  "preset": "adaptive",
  "ocr_engine": "deepseek",
  "needs_external_ocr": false,
  "source": "document",
  "rotation": 0,
  "hit_length_limit": false
}
```

`source` is how the returned text was produced: the prompt used (`document`,
`free_ocr`, …), `free_ocr_fallback` for a page first read as a single picture,
a `+split` suffix for a page read in two halves, or `skipped`. `rotation` is the
counter-clockwise turn applied before reading. `hit_length_limit` means the
returned text was cut off by the output budget, so the end of the page is missing.

### Flag Detail Codes

| Code | Severity | Trigger |
|------|----------|---------|
| `no_content` | critical | No meaningful text (≤10 chars) |
| `ocr_failed` | critical | Page has content but model couldn't read it |
| `incomplete_extraction` | critical | Model hit token limit, >90% hallucinated |
| `blank_page` | critical | Blank page skipped |
| `low_quality_scan` | critical | Content too small, skipped (only with `SKIP_LOW_QUALITY_SCANS=true`) |
| `possible_hallucination` | warning | >75% of output removed |
| `max_tokens_hit` | warning | Stuck generation loop |
| `repetitive_content` | info | Repetitive patterns detected |
| `sparse_content` | info | Very little text vs image size |
| `low_content` | info | Less than 30 chars extracted |

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/DeepSeek-OCR` | Path to model weights |
| `PORT` | `8000` | API port |
| `HOST` | `0.0.0.0` | API bind address |
| `GPU_MEM_UTIL` | `0.80` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Max model context length |
| `MAX_TOKENS` | `8192` | Max output tokens per inference |
| `MAX_IMAGE_SIZE_MB` | `20` | Max upload size per image |
| `MAX_PDF_SIZE_MB` | `100` | Max upload size per PDF |
| `MAX_PDF_PAGES` | `50` | Max pages per PDF |
| `MAX_BATCH_SIZE` | `16` | Max images per batch request |
| `REQUEST_TIMEOUT_S` | `120` | Request timeout (seconds) |
| `SCORE_THRESHOLD` | `0.60` | Score below this triggers retry |
| `MAX_RETRIES` | `3` | Attempts per page with contrast presets (the rescue ladder has its own fixed steps) |
| `FALLBACK_FREE_OCR` | `true` | Re-read a page labelled as one picture with `free_ocr` |
| `LOOP_RESCUE` | `true` | Rescue ladder for runaway or sideways pages |
| `SKIP_LOW_QUALITY_SCANS` | `false` | Restore the old skip of pages with little content |
| `FEEDBACK_DIR` | `./feedback` | Feedback storage path |
| `FEEDBACK_ENABLED` | `true` | Enable feedback storage |
| `FEEDBACK_SCORE_THRESHOLD` | `0.70` | Save results below this score |
| `FEEDBACK_MAX_GB` | `20` | Disk budget; oldest pending entries pruned above it |
| `REQUEST_LOG_FILE` | `/workspace/logs/requests.jsonl` | One JSON line per request (see *Request logging*) |
| `REQUEST_LOG_MAX_MB` / `REQUEST_LOG_BACKUPS` | `50` / `5` | Rotation for the request log |
| `LOG_FILENAMES` | `false` | Include upload filenames in the request log (they often contain PII) |
| `LOG_MAX_MB` / `LOG_BACKUPS` | `100` / `5` | Rotation for `api.log`, done by `supervise.sh` |
| `FEEDBACK_PRUNE_INTERVAL_S` | `21600` | How often the budget is enforced |
| `FEEDBACK_PAGE_MAX` | `500` | Max entries returned by `/feedback/pending` |

### Forced Environment

```bash
VLLM_USE_V1=0           # Use legacy vLLM engine (required for custom model)
CUDA_VISIBLE_DEVICES=0   # Single GPU
```

---

## Project Structure

```
DeepSeek-OCR-1/
├── api_service.py                          ← Main FastAPI service (v4.0.0)
│   ├── Lifespan: AsyncLLMEngine init
│   ├── Pre-flight: blank page + low-quality scan detection
│   ├── Inference: async generate via vLLM
│   ├── Retry: up to 3 enhancement presets
│   ├── Feedback: auto-save low-scoring results
│   └── Endpoints: /ocr/image, /ocr/pdf, /ocr/batch, /feedback/*
│
├── start.sh                                ← Entrypoint script
├── requirements.txt                        ← Python dependencies (pinned)
├── README.md                               ← User-facing documentation
├── ARCHITECTURE.md                         ← This file
│
├── images/                                 ← Input images directory
├── feedback/                               ← Feedback storage (auto-created)
│   ├── pending/                            ← Awaiting correction
│   └── verified/                           ← Ready for fine-tuning
│
└── DeepSeek-OCR-master/
    └── DeepSeek-OCR-vllm/
        ├── config.py                       ← Image tiling config (sizes, crops)
        ├── deepseek_ocr.py                 ← Custom vLLM model class
        │   ├── DeepseekOCRForCausalLM      ← Registered with ModelRegistry
        │   ├── DeepseekOCRProcessingInfo
        │   ├── DeepseekOCRMultiModalProcessor
        │   └── DeepseekOCRDummyInputsBuilder
        │
        └── process/
            ├── __init__.py                 ← Public API exports
            ├── image_process.py            ← Image tokenization (DO NOT MODIFY)
            ├── ngram_norepeat.py           ← Logits processor (DO NOT MODIFY)
            ├── postprocess.py              ← Output cleanup + hallucination removal
            ├── enhance.py                  ← Adaptive image enhancement
            └── score.py                    ← Quality scoring system
```

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `vllm` | 0.8.4 | GPU inference engine (AsyncLLMEngine) |
| `torch` | 2.6.0 | PyTorch backend |
| `transformers` | 4.57.6 | Tokenizer + model loading |
| `flash_attn` | 2.8.3 | Flash Attention 2 for fast inference |
| `fastapi` | 0.135.2 | HTTP API framework |
| `uvicorn` | 0.42.0 | ASGI server |
| `pillow` | 12.1.1 | Image loading and manipulation |
| `PyMuPDF` | 1.27.2 | PDF to image conversion |
| `scipy` | 1.17.1 | Numerical utilities |
| `numpy` | 1.26.4 | Image array operations |

---

## Performance Characteristics

### Benchmarks (54-page housing compliance document set)

| Metric | Value |
|--------|-------|
| Average processing time | ~4.6s per page |
| GREEN (good quality) | 94.4% of pages |
| YELLOW (spot-check) | 3.7% of pages |
| RED (manual review) | 1.9% of pages |
| Average composite score | 0.876 |
| Blank/low-quality skip time | ~1ms |
| Total chars extracted | 131,610 |

### Concurrent Request Throughput

| Scenario | Time | Speedup |
|----------|------|---------|
| 4 sequential requests | ~32.5s | 1× |
| 4 concurrent requests | ~25.3s | 1.28× |

Concurrent requests share GPU compute via continuous batching. Speedup is modest because the GPU is the bottleneck — more requests don't add GPU capacity, they just reduce idle time between requests.

### Memory Usage

| Component | VRAM |
|-----------|------|
| Model weights (bfloat16) | ~6.2 GB |
| KV cache (at 80% util) | Remaining allocation |
| Per-request overhead | Managed by vLLM scheduler |

### Latency Breakdown (single page)

| Stage | Time |
|-------|------|
| Pre-flight detection | ~1ms |
| Image enhancement | ~10ms |
| Tiling + tokenization | ~50ms |
| vLLM inference | ~2-6s (varies with content density) |
| Post-processing | ~5ms |
| Scoring | ~2ms |
| **Total (single attempt)** | **~2-6s** |
| **With retry (3 attempts)** | **~6-18s** |
