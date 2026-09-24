"""Accuracy metrics for comparing OCR output against ground-truth text.

The headline metric is word-level F1 over a multiset of tokens, which ignores
reading order. That matters here: on forms and tables the model's layout (and a
PDF's text layer) can order the same words differently, and an order-sensitive
metric would count a correct page as badly wrong. WER is reported alongside it
for pages where order is meaningful (prose).
"""

import re
import unicodedata
from collections import Counter

# Coordinates are layout metadata, never content. <|ref|> tags are stripped but
# their contents kept: in the "ocr" prompt mode the recognized text lives
# inside them.
_DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_REF_TAGS = re.compile(r"<\|/?ref\|>")
_EOS = "<｜end▁of▁sentence｜>"


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens with markup removed.

    NFKC folds ligatures that PDF text layers are full of ("ﬁ" -> "fi"), and
    the word pattern keeps accented letters, so Spanish words stay whole.
    """
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace(_EOS, " ")
    t = _DET.sub(" ", t)
    t = _REF_TAGS.sub(" ", t)
    t = re.sub(r"<[^>]+>", " ", t)          # html (tables, <br>, <center>)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)      # latex commands
    return re.findall(r"[^\W_]+", t.lower())


def bag_scores(ref: list[str], hyp: list[str]) -> tuple[float, float, float]:
    """Order-insensitive precision, recall, F1 over token multisets."""
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    hit = sum((Counter(ref) & Counter(hyp)).values())
    precision = hit / len(hyp) if hyp else 0.0
    recall = hit / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate: word-level Levenshtein distance / reference length.

    A runaway generation can emit thousands of junk words; the hypothesis is
    capped at 3x the reference so one such page cannot take minutes to score.
    Every word past the cap would be an insertion anyway, so the cap only
    bounds the value, and it is clamped to 1.0 below.
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    hyp = hyp[: 3 * len(ref)]
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return min(1.0, prev[-1] / len(ref))


def score_page(gt_text: str, ocr_text: str) -> dict:
    ref, hyp = tokens(gt_text), tokens(ocr_text)
    p, r, f1 = bag_scores(ref, hyp)
    return {
        "gt_words": len(ref),
        "out_words": len(hyp),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "wer": round(wer(ref, hyp), 4),
        "empty": len(hyp) == 0 and len(ref) > 0,
    }
