"""Weighted multi-variable scoring system for OCR output quality.

General-purpose scoring designed for diverse document types: forms,
letters, reports, invoices, receipts, legal documents, handwritten
notes, certificates, spreadsheets, etc.

Evaluates OCR results using multiple independent metrics, each with
its own weight. The composite score determines whether a result is
acceptable or needs to be retried with different preprocessing.

Design principles:
- No assumption about document structure (headers, tables, etc.)
- Coordinate/grounding tags stripped during cleaning are NOT hallucination
- Natural text repetition (legal boilerplate, form labels) is expected
- Single-run deterministic inference should not be penalized
"""

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .postprocess import CleanStats


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------

# self_consistency is deliberately weighted 0.0. It compares a result against
# other runs of the same page, so it is only meaningful while ranking retry
# candidates -- at report time nothing passes other_results and it is a flat
# 1.0, i.e. a constant offset with no discriminative power. It stays in the
# breakdown (and still drives retry ranking in select_best_result) but no
# longer inflates every composite by 0.20.
DEFAULT_WEIGHTS = {
    "self_consistency": 0.00,
    "hallucination_ratio": 0.30,
    "token_efficiency": 0.30,
    "structural_integrity": 0.15,
    "repetition_density": 0.10,
    "content_density": 0.15,
}

# Quality threshold — results below this are candidates for retry
DEFAULT_THRESHOLD = 0.6

# Max number of retry attempts per page
DEFAULT_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """Detailed breakdown of an OCR quality score."""
    self_consistency: float = 0.0
    hallucination_ratio: float = 0.0
    token_efficiency: float = 0.0
    structural_integrity: float = 0.0
    repetition_density: float = 0.0
    content_density: float = 0.0
    composite: float = 0.0
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def to_dict(self) -> dict:
        return {
            "composite": round(self.composite, 4),
            "variables": {
                "self_consistency": round(self.self_consistency, 4),
                "hallucination_ratio": round(self.hallucination_ratio, 4),
                "token_efficiency": round(self.token_efficiency, 4),
                "structural_integrity": round(self.structural_integrity, 4),
                "repetition_density": round(self.repetition_density, 4),
                "content_density": round(self.content_density, 4),
            },
            "weights": self.weights,
        }


@dataclass
class OCRResult:
    """A single OCR inference result with metadata for scoring."""
    raw_text: str
    clean_text: str
    num_tokens: int
    max_tokens: int
    preset_name: str = "adaptive"
    score: Optional[ScoreBreakdown] = None
    clean_stats: Optional["CleanStats"] = None
    # Whether generation stopped because it ran out of room rather than
    # because the model emitted a stop token. vLLM reports this directly as
    # finish_reason == "length"; inferring it from num_tokens/max_tokens does
    # not work, because the usable budget is max_model_len minus the prompt
    # (~7,280 tokens for a 144-DPI A4 page, never the configured 8,192).
    hit_length_limit: bool = False
    # True for the "ocr" prompt, where <|ref|>...<|/ref|> holds the recognized
    # text rather than a layout label, so it is content, not markup.
    ref_is_content: bool = False
    # The page has little ink (a cover, divider or signature page). Short
    # output is then correct, not a failure, and must not be capped or flagged.
    sparse_page: bool = False
    # How the text was produced: the prompt used, or e.g. "free_ocr_fallback".
    source: str = "document"
    # Degrees the page was turned (counter-clockwise) before reading; 0 = as received.
    rotation: int = 0


# ---------------------------------------------------------------------------
# Tag measurement — separates grounding tags from real content
# ---------------------------------------------------------------------------

# Patterns that are expected model output format, not hallucination
_GROUNDING_TAG_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>"
    r"|<\|det\|>.*?<\|/det\|>"
    r"|<\uff5cend\u2581of\u2581sentence\uff5c>"
    r"|\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]"
)


# Markup only -- for the "ocr" prompt, whose <|ref|> contents are real text.
_MARKUP_ONLY_PATTERN = re.compile(
    r"<\|/?ref\|>"
    r"|<\|det\|>.*?<\|/det\|>"
    r"|<\uff5cend\u2581of\u2581sentence\uff5c>"
)


def _tag_pattern(ref_is_content: bool) -> re.Pattern:
    return _MARKUP_ONLY_PATTERN if ref_is_content else _GROUNDING_TAG_PATTERN


def _measure_grounding_tags(raw_text: str, ref_is_content: bool = False) -> int:
    """Count characters in the raw text that are grounding/coordinate tags.

    These tags are expected output format and their removal during
    cleaning should not be counted as hallucination. In the "ocr" prompt mode
    the <|ref|> contents are the recognized text, so only the markup counts.
    """
    return sum(len(m.group()) for m in _tag_pattern(ref_is_content).finditer(raw_text))


# ---------------------------------------------------------------------------
# Individual scoring variables (each returns 0.0 - 1.0)
# ---------------------------------------------------------------------------

def _score_hallucination_ratio(result: OCRResult) -> float:
    """How much of the raw output survived post-processing.

    Excludes grounding tags and dedup-removed content from the
    denominator, since neither represents fabricated content.

    For general documents: the ratio measures clean_chars / effective_raw,
    where effective_raw = raw - tags - dedup.
    """
    raw_len = len(result.raw_text.strip())
    clean_len = len(result.clean_text.strip())
    if raw_len == 0:
        return 0.0

    # Subtract characters that are expected format, not hallucination
    tag_chars = _measure_grounding_tags(result.raw_text, result.ref_is_content)
    dedup_removed = 0
    if result.clean_stats is not None:
        dedup_removed = result.clean_stats.dedup_chars_removed

    effective_raw = raw_len - tag_chars - dedup_removed
    # If after removing tags the effective raw is smaller than clean,
    # that means almost everything was tags — score is perfect
    if effective_raw <= clean_len:
        return 1.0

    ratio = clean_len / effective_raw
    return min(ratio, 1.0)


def _score_token_efficiency(result: OCRResult) -> float:
    """Penalize runs that were cut off mid-generation.

    Whether the model ran out of room is reported by the engine as
    finish_reason == "length"; it cannot be inferred from
    num_tokens / max_tokens, because the usable budget is max_model_len
    minus the prompt and the configured max_tokens is never reachable.

    A run that stopped on its own is complete by definition. A run that was
    cut off lost content, and the more of its output failed to survive
    cleaning, the more of it was degenerate rather than real text.
    """
    if not result.hit_length_limit:
        return 1.0

    clean_len = len(result.clean_text.strip())
    raw_len = len(result.raw_text.strip())
    if raw_len == 0:
        return 0.1

    # Grounding tags are expected output format, not content, and they are a
    # large fraction of the raw text. Excluding them here matches what
    # _score_hallucination_ratio does; leaving them in depressed survival for
    # every grounded page and pushed truncated-but-good pages into red.
    effective_raw = max(1, raw_len - _measure_grounding_tags(result.raw_text, result.ref_is_content))

    # Of everything the model emitted before running out of room, how much
    # was real content? A truncated dense page keeps nearly all of it; a
    # generation loop is almost entirely stripped by post-processing.
    survival = clean_len / effective_raw
    # Cap at 0.8: the page was cut off, so it is incomplete regardless.
    return max(0.1, min(0.8, survival))


def _score_structural_integrity(result: OCRResult) -> float:
    """Check for recognizable content patterns in the output.

    General-purpose: does NOT require any specific structure type.
    Awards credit for ANY recognizable pattern — a plain text letter
    scores just as well as a complex form with tables.
    """
    text = result.clean_text
    if not text.strip():
        return 0.0

    signals = 0.0

    # Has markdown headers?
    if re.search(r"#{1,3}\s+\S", text):
        signals += 1.0

    # Has table structure with real content?
    tables = re.findall(r"<table>.*?</table>", text, re.DOTALL)
    if tables:
        for table in tables:
            content_cells = re.findall(r"<td[^>]*>([^<]+)</td>", table)
            if len(content_cells) >= 2:
                signals += 1.0
                break
        else:
            signals += 0.3

    # Has meaningful text (>30 chars of non-markup text)?
    non_markup = re.sub(r"<[^>]+>", "", text)
    non_markup = re.sub(r"#{1,3}\s+", "", non_markup).strip()
    if len(non_markup) > 30:
        signals += 1.0

    # Has recognizable data patterns (dates, amounts, names, emails, phones)?
    data_patterns = (
        re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text)      # dates
        or re.search(r"\$[\d,]+\.?\d*", text)               # dollar amounts
        or re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text)     # proper names
        or re.search(r"\S+@\S+\.\S+", text)                 # emails
        or re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", text)  # phone numbers
    )
    if data_patterns:
        signals += 1.0

    # Normalize: any 1 signal is enough for a decent score.
    # 1 signal = 0.75, 2 = 0.875, 3 = 0.95, 4 = 1.0
    max_signals = 4.0
    if signals >= max_signals:
        return 1.0
    if signals >= 1.0:
        return 0.5 + 0.5 * (signals / max_signals)
    # No signals at all — but if there's substantial text, still give partial credit
    if len(non_markup) > 100:
        return 0.4
    return 0.25


def _score_repetition_density(result: OCRResult) -> float:
    """Detect generation loops via tail-vs-head compressibility.

    Measured on the RAW output: post-processing strips runaway table rows and
    repeated patterns, so by the time the text is cleaned the evidence of a
    loop is frequently gone.

    Replaces an n-gram counting version that saturated to 0.0 on any
    structured document -- an invoice with twelve line items and a true
    generation loop both scored 0.000 -- making it a flat penalty on exactly
    the forms and invoices this service handles rather than a signal.
    """
    return 1.0 - _measure_degeneration(result.raw_text, result.ref_is_content)


def _score_content_density(result: OCRResult) -> float:
    """How much text was extracted, in absolute terms.

    A pixel-ratio variant used to live here, scaled against the image area.
    It was removed: it scored a normal A4 page at ~0.26 while the char-count
    path scored the same text 1.0, and since the two paths were compared
    against each other by the PDF and batch endpoints, retries were
    systematically discarded. Nothing passed image dimensions any more, so
    the branch was also dead.
    """
    clean_len = len(result.clean_text.strip())

    if clean_len == 0:
        return 0.0
    if clean_len >= 500:
        return 1.0
    if clean_len >= 100:
        return 0.5 + 0.5 * ((clean_len - 100) / 400)
    if clean_len >= 20:
        return 0.2 + 0.3 * ((clean_len - 20) / 80)
    return 0.1


# Minimum characters before the tail/head comparison is meaningful — zlib's
# fixed overhead dominates short buffers and makes the ratio noise.
_DEGENERATION_MIN_CHARS = 2000


def _measure_degeneration(text: str, ref_is_content: bool = False) -> float:
    """How much more compressible the tail of the output is than its head.

    A generation loop decays over time, so its last quarter collapses to
    near-nothing while its first quarter still looks like real text. Uniform
    repetition -- a form with repeated labels, an invoice with similar line
    items -- compresses the same at both ends and scores ~0.

    Returns 0.0 (uniform) to ~1.0 (tail fully collapsed).

    Blind spot: markup is stripped before measuring, so a loop made purely of
    empty table cells leaves nothing to compare and this returns 0.0. Those
    are caught by _score_token_efficiency instead -- post-processing strips
    the bloat, so almost nothing survives and survival collapses. Measuring
    with markup included was rejected: a legitimate dense table compresses
    just as hard and would be capped as a loop.
    """
    stripped = _tag_pattern(ref_is_content).sub(" ", text)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip().encode("utf-8", "ignore")
    if len(stripped) < _DEGENERATION_MIN_CHARS:
        return 0.0

    quarter = len(stripped) // 4
    head = len(zlib.compress(stripped[:quarter], 6)) / quarter
    tail = len(zlib.compress(stripped[-quarter:], 6)) / quarter
    if head <= 0:
        return 0.0
    return max(0.0, (head - tail) / head)


_WORD = re.compile(r"[^\W_]+")


# A loop that ends on its own ("8/8/8/8/..." then a stop token) never hits the
# length limit, so the length-gated loop cap misses it and it used to score
# green. Measured distinct-word ratios on outputs of 200+ words: degenerate
# outputs 0.004-0.072; lowest correct output (F1 >= 0.8) 0.115; lowest of 220
# real scanned-page outputs 0.254. 0.05 sits well below anything real.
_DEGENERATE_MIN_WORDS = 200
_DEGENERATE_DISTINCT_RATIO = 0.05


def is_degenerate_output(text: str) -> bool:
    """Long output made of almost no distinct words: a generation loop."""
    words = _WORD.findall(text.lower())
    return len(words) >= _DEGENERATE_MIN_WORDS and len(set(words)) / len(words) < _DEGENERATE_DISTINCT_RATIO


def _token_similarity(a: str, b: str) -> float:
    """Word-overlap similarity (F1 over word multisets), linear time.

    This replaced difflib.SequenceMatcher, which failed both ways. With its
    default autojunk it treats any character in >1% of a 200+ char string as
    junk -- nearly every letter -- so two near-identical tables scored 0.009.
    With autojunk=False it is correct but quadratic: one comparison of two
    looping outputs took 4-12 seconds, the retry path makes about nine, and
    the event loop stalled long enough for the supervisor to kill the
    service. That caused 12 production restarts between 2026-09-14 and
    2026-09-24. Word order is not needed here: this only ranks retry
    attempts of the same page against each other.
    """
    ta = _WORD.findall(a.lower())
    tb = _WORD.findall(b.lower())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    return 2 * common / (len(ta) + len(tb))


def _score_self_consistency(
    current: OCRResult,
    others: list[OCRResult],
) -> float:
    """Pairwise similarity between multiple OCR runs.

    If the model produces similar text across different preprocessing
    runs, the result is likely correct. Wildly different outputs
    indicate unreliable generation.

    Single-run results return 1.0 because a deterministic model at
    temperature 0 will always produce the same output for the same
    input — the inability to measure consistency should not penalize
    the score at all.
    """
    if not others:
        return 1.0

    similarities = [_token_similarity(current.clean_text, other.clean_text) for other in others]

    if not similarities:
        return 1.0

    return sum(similarities) / len(similarities)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _compression_ratio(text: str, ref_is_content: bool = False) -> float:
    """Compressed size over raw size for the whole output. Lower = more
    repetitive. Normal prose sits around 0.25-0.40; a generation loop
    collapses below 0.05.

    Only safe to act on when the run hit the length limit: a legitimate form
    with forty repeated label rows also compresses to ~0.03, and must not be
    treated as a loop.
    """
    stripped = _tag_pattern(ref_is_content).sub(" ", text)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip().encode("utf-8", "ignore")
    if len(stripped) < _DEGENERATION_MIN_CHARS:
        return 1.0
    return len(zlib.compress(stripped, 6)) / len(stripped)


# A run that was cut off AND either collapsed toward the end or is uniformly
# near-incompressible is a generation loop.
# Gated on hit_length_limit so it can only fire on runs that actually ran out
# of room, which is the only situation where loops occur -- a page that
# stopped on its own is never penalised by this, however repetitive it is.
_LOOP_DEGENERATION_THRESHOLD = 0.45
# Measured: true loops land at 0.006-0.016, while a legitimate form with
# forty repeated label rows sits at 0.028 and an invoice with repeated line
# items at 0.048. 0.025 sits in that gap. This is calibrated on constructed
# examples, not labelled production data -- see the note in compute_flags.
_LOOP_COMPRESSION_THRESHOLD = 0.025
_LOOP_COMPOSITE_CAP = 0.35


def _apply_composite(
    breakdown: ScoreBreakdown,
    result: OCRResult,
    w: dict,
) -> float:
    """Weighted sum plus the hard caps, in one place.

    score_result and select_best_result both need this; they used to carry
    separate copies of the formula, which had already drifted apart.
    """
    composite = (
        w["self_consistency"] * breakdown.self_consistency
        + w["hallucination_ratio"] * breakdown.hallucination_ratio
        + w["token_efficiency"] * breakdown.token_efficiency
        + w["structural_integrity"] * breakdown.structural_integrity
        + w["repetition_density"] * breakdown.repetition_density
        + w["content_density"] * breakdown.content_density
    )

    # Blank / near-blank pages: several metrics return perfect scores on empty
    # output, so cap rather than let the composite float up.
    clean_len = len(result.clean_text.strip())
    if result.sparse_page and clean_len > 0:
        pass    # little ink on the page: little text is the right answer
    elif clean_len <= 10:
        composite = min(composite, 0.10)
    elif clean_len <= 30:
        composite = min(composite, 0.30)

    # Generation loop: ran out of room with a collapsed tail. Without this a
    # verbose loop scores green, because it produces plenty of characters and
    # only trips repetition_density, which carries 0.10 of the weight.
    # Two shapes of loop: one that decays partway through (tail collapses
    # relative to head), and one that starts early enough that head and tail
    # look alike and only absolute compressibility gives it away.
    if result.hit_length_limit and (
        _measure_degeneration(result.raw_text, result.ref_is_content) > _LOOP_DEGENERATION_THRESHOLD
        or _compression_ratio(result.raw_text, result.ref_is_content) < _LOOP_COMPRESSION_THRESHOLD
    ):
        composite = min(composite, _LOOP_COMPOSITE_CAP)
    elif is_degenerate_output(result.clean_text):
        composite = min(composite, _LOOP_COMPOSITE_CAP)

    return composite


def score_result(
    result: OCRResult,
    other_results: Optional[list[OCRResult]] = None,
    weights: Optional[dict] = None,
) -> ScoreBreakdown:
    """Compute the weighted composite quality score for an OCR result.

    Note this measures whether generation behaved normally -- it is computed
    entirely from the output text and never compares against the image, so it
    cannot detect fluent-but-wrong OCR.

    Args:
        result: The OCR result to score.
        other_results: Other runs of the same image for self-consistency.
            Only meaningful when ranking retry candidates; self_consistency
            carries no weight in the composite.
        weights: Override default scoring weights.

    Returns:
        ScoreBreakdown with per-variable and composite scores.
    """
    w = weights or DEFAULT_WEIGHTS

    breakdown = ScoreBreakdown(weights=w)

    breakdown.self_consistency = _score_self_consistency(
        result, other_results or []
    )
    breakdown.hallucination_ratio = _score_hallucination_ratio(result)
    breakdown.token_efficiency = _score_token_efficiency(result)
    breakdown.structural_integrity = _score_structural_integrity(result)
    breakdown.repetition_density = _score_repetition_density(result)
    breakdown.content_density = _score_content_density(result)

    breakdown.composite = _apply_composite(breakdown, result, w)

    result.score = breakdown
    return breakdown


def select_best_result(results: list[OCRResult]) -> OCRResult:
    """From multiple scored results, return the one with the highest composite score.

    Also re-scores self_consistency using the full set of results.
    """
    if len(results) == 1:
        return results[0]

    # Re-score self_consistency with the full result set. It carries no
    # weight in the composite, but agreement between attempts is still a
    # useful tie-breaker, so it is applied explicitly below rather than
    # through the weights.
    for i, result in enumerate(results):
        others = [r for j, r in enumerate(results) if j != i]
        if result.score is not None:
            result.score.self_consistency = _score_self_consistency(
                result, others
            )
            result.score.composite = _apply_composite(
                result.score, result, result.score.weights
            )

    def _rank(r: OCRResult) -> tuple:
        if r.score is None:
            return (0.0, 0.0)
        return (r.score.composite, r.score.self_consistency)

    return max(results, key=_rank)


def needs_retry(
    result: OCRResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Check if a result's score is below the retry threshold."""
    if result.score is None:
        return True
    return result.score.composite < threshold


# ---------------------------------------------------------------------------
# Flagging — Green / Yellow / Red quality flags
# ---------------------------------------------------------------------------

# Composite score boundaries for color flags
FLAG_GREEN_THRESHOLD = 0.70   # >= 0.70 → green
FLAG_YELLOW_THRESHOLD = 0.50  # >= 0.50 → yellow, below → red


def compute_flags(
    result: OCRResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute a Green/Yellow/Red quality flag for an OCR result.

    Returns a dict with:
        - flag: "green", "yellow", or "red"
        - message: short summary for the flag color
        - details: list of individual issue dicts (code + message + severity)

    Flag logic (applied in order):
        red    — no content OR composite < 0.50
        yellow — composite between 0.50 and 0.70, OR green demoted by warning
        green  — composite >= 0.70

    Warnings (downgrade by one level: green→yellow, yellow stays yellow):
        - hallucination_ratio below 0.25 (severe — most content may be fabricated)
        - token_efficiency below 0.2 (model stuck in generation loop)

    Informational (included in details but don't change color):
        - repetition_density below threshold
        - low content length

    Args:
        result: A scored OCRResult.
        threshold: Composite score threshold (unused, reserved for future use).

    Returns:
        Flag dict with color, message, and details.
    """
    details: list[dict] = []
    clean_len = len(result.clean_text.strip())
    score = result.score

    # --- No content → always red (a sparse page's few words do count) ---
    if clean_len == 0 or (clean_len <= 10 and not result.sparse_page):
        return {
            "flag": "red",
            "message": "No meaningful text extracted. Manual review required.",
            "details": [{
                "code": "no_content",
                "severity": "critical",
                "message": "No meaningful text was extracted from this page.",
            }],
        }

    # --- Very little content (informational) ---
    if clean_len < 30:
        details.append({
            "code": "low_content",
            "severity": "info",
            "message": f"Very little text extracted ({clean_len} chars). Page may be mostly blank or handwritten.",
        })

    # --- Unscored → yellow ---
    if score is None:
        details.append({
            "code": "unscored",
            "severity": "warning",
            "message": "Page was not scored — quality is unknown.",
        })
        return {
            "flag": "yellow",
            "message": "Quality could not be determined. Spot-check recommended.",
            "details": details,
        }

    # --- Check for warnings (downgrade one level, not force red) ---
    has_warning = False

    # Flag hallucination if severe (>75% removed after excluding tags)
    if score.hallucination_ratio < 0.25:
        pct = (1 - score.hallucination_ratio) * 100
        details.append({
            "code": "possible_hallucination",
            "severity": "warning",
            "message": f"~{pct:.0f}% of output was removed as hallucinated content.",
        })
        has_warning = True

    # Generation ran out of room. The text we did get may be fine, but the
    # bottom of the page is missing, so this must not read green.
    if result.hit_length_limit:
        details.append({
            "code": "truncated_output",
            "severity": "warning",
            "message": "Generation hit the length limit — the end of the page is missing.",
        })
        has_warning = True

    # A loop that stopped on its own: almost no distinct words.
    if is_degenerate_output(result.clean_text):
        details.append({
            "code": "degenerate_output",
            "severity": "warning",
            "message": "Output repeats a handful of words over and over -- a generation loop, not page text.",
        })
        has_warning = True

    # Cut off with almost nothing surviving post-processing: a loop, not a
    # dense page that ran long.
    if score.token_efficiency < 0.2:
        details.append({
            "code": "max_tokens_hit",
            "severity": "warning",
            "message": "Model hit the length limit with very little clean output — likely stuck in a generation loop.",
        })
        has_warning = True

    # --- Informational (don't change flag color) ---
    if score.repetition_density < 0.4:
        details.append({
            "code": "repetitive_content",
            "severity": "info",
            "message": "Output contains repetitive patterns that may indicate hallucination.",
        })

    if score.content_density < 0.15:
        details.append({
            "code": "sparse_content",
            "severity": "info",
            "message": "Extracted text is very short relative to image size.",
        })

    # --- Determine color from composite score ---
    composite = score.composite

    if composite < FLAG_YELLOW_THRESHOLD:
        flag = "red"
        message = f"Low quality score ({composite:.2f}). Manual review required."
    elif composite < FLAG_GREEN_THRESHOLD:
        flag = "yellow"
        message = f"Borderline quality score ({composite:.2f}). Spot-check recommended."
    else:
        flag = "green"
        message = f"Good quality ({composite:.2f})."

    # --- Warnings downgrade by one level (green→yellow, yellow stays) ---
    if has_warning:
        if flag == "green":
            flag = "yellow"
            message = f"Score OK ({composite:.2f}) but has warnings. Spot-check recommended."

    return {
        "flag": flag,
        "message": message,
        "details": details,
    }
