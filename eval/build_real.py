"""Build the real-scan eval set: production pages with human-verified text.

Step 1 -- prepare drafts (needs the service running):
    python eval/build_real.py --prepare [--n 80]
    python eval/build_real.py --prepare --pdf-set scanned_pdfs --per-doc 6 --prefix s

  The first samples the feedback corpus; the second samples pages at random
  from a set built by build_pdf_set.py (representative real documents). Both
  add to the same review folder.

  Either way, each sampled page gets a draft transcription in
  <data-dir>/real/review/:
      rNNN.png   the page exactly as the service received it
      rNNN.txt   draft text, first line a DRAFT marker
  A person corrects each .txt against its image and deletes the marker line.

Step 2 -- freeze verified pages into the eval set:
    python eval/build_real.py --freeze

  Only pages whose marker line has been removed are included, so a partly
  reviewed folder can be frozen at any time.

The feedback corpus only holds pages that scored below 0.70, so this set is
biased toward pages the engine already struggles with. That is where the
fixes are aimed, but it cannot catch a regression on pages that currently
work -- the simulated set covers those, and representative real pages should
be added when available (drop PNG + verified TXT pairs into review/).

Everything under <data-dir>/real is customer data. It must stay out of the
repository.
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from metrics import tokens  # noqa: E402

MARKER = "#DRAFT - correct the text below against the image, then delete this line"
LENGTH_LIMIT = 7200   # num_tokens at or above this means generation was cut off

# The external service names pages page_N.png. Anything else in the corpus came
# from ad-hoc testing (p3.png, g1.png, x.png, ...) and is excluded.
EXTERNAL_NAME = re.compile(r"^page_\d+\.png$")
TEST_REQUEST_PREFIXES = ("eval-", "verify-", "test-")


def candidates(feedback_dir):
    """Real production pages, filtered on metadata only (cheap).

    Duplicates are removed later, and only among the pages actually sampled:
    hashing every image up front would read ~12 GB over the network mount.
    """
    out = []
    for fn in sorted(os.listdir(feedback_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            meta = json.load(open(f"{feedback_dir}/{fn}"))
        except (OSError, ValueError):
            continue
        if not EXTERNAL_NAME.match(meta.get("filename") or ""):
            continue
        if (meta.get("request_id") or "").startswith(TEST_REQUEST_PREFIXES):
            continue
        out.append({"entry": fn[:-5], "png": f"{feedback_dir}/{fn[:-5]}.png",
                    "score": meta.get("score") or 0.0})
    return out


def stratified(pages, n, rng):
    """Spread the sample across total failures, bad and borderline pages."""
    buckets = {
        "total_failure": [p for p in pages if p["score"] <= 0.10],
        "poor": [p for p in pages if 0.10 < p["score"] <= 0.45],
        "borderline": [p for p in pages if p["score"] > 0.45],
    }
    for b in buckets.values():
        rng.shuffle(b)
    picked, seen, i = [], set(), 0
    while len(picked) < n and any(buckets.values()):
        b = list(buckets.values())[i % 3]
        i += 1
        if not b:
            continue
        p = b.pop()
        try:
            digest = hashlib.sha256(open(p["png"], "rb").read()).hexdigest()
        except OSError:
            continue
        if digest in seen:          # the same page resent, or copied by a test
            continue
        seen.add(digest)
        picked.append(p)
    return picked


def ocr(url, png_bytes, rid):
    r = requests.post(f"{url}/ocr/image", files={"file": ("page_1.png", png_bytes, "image/png")},
                      data={"prompt": "free_ocr", "retry": "false"},
                      headers={"X-Request-ID": rid}, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d.get("text", ""), d.get("num_tokens") or 0


def draft(url, img, rid):
    """Best-effort draft: free_ocr, retried sideways if it comes back empty or runs away.

    free_ocr is used because it was the most accurate mode measured and it does not
    discard pages it takes for a picture. Returns (text, orientation note).
    """
    def png(im):
        b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()
    text, toks = ocr(url, png(img), f"{rid}-r0")
    if tokens(text) and toks < LENGTH_LIMIT:
        return text, "upright"
    best = (len(tokens(text)) if toks < LENGTH_LIMIT else -1, text, "upright")
    for deg in (90, 270):
        t, k = ocr(url, png(img.rotate(deg, expand=True)), f"{rid}-r{deg}")
        cand = (len(tokens(t)) if k < LENGTH_LIMIT else -1, t, f"rotated {deg} degrees counter-clockwise")
        best = max(best, cand, key=lambda c: c[0])
    return best[1], best[2]


def from_pdf_set(args, rng):
    """Random pages per document from a build_pdf_set.py set -- representative,
    unlike the feedback corpus, which only holds pages that already failed."""
    set_dir = f"{args.data_dir}/{args.pdf_set}"
    items = json.load(open(f"{set_dir}/manifest.json"))["items"]
    # The same scan can arrive in more than one PDF (two of the first seven
    # were identical). Keep each distinct page once, under the first document
    # it appears in, so nobody verifies the same page twice.
    seen, distinct = set(), {}
    for i in sorted(items, key=lambda i: (int(i["doc"][1:]) if i["doc"][1:].isdigit() else 0, i["page"])):
        digest = hashlib.sha256(open(f"{set_dir}/{i['image']}", "rb").read()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            distinct.setdefault(i["doc"], []).append(i)
    picked = []
    for doc, pages in distinct.items():
        for i in rng.sample(pages, min(args.per_doc, len(pages))):
            picked.append({"entry": f"{args.pdf_set}/{i['id']}", "png": f"{set_dir}/{i['image']}", "score": None})
    dropped = sorted({i["doc"] for i in items} - set(distinct))
    print(f"sampled {len(picked)} pages ({args.per_doc} per document) from {len(distinct)} distinct documents"
          + (f"; skipped {dropped} (duplicates of earlier documents)" if dropped else ""))
    return picked


def prepare(args):
    rng = random.Random(args.seed)
    review = f"{args.data_dir}/real/review"
    os.makedirs(review, exist_ok=True)
    if args.pdf_set:
        picked = from_pdf_set(args, rng)
    else:
        pool = candidates(args.feedback_dir)
        picked = stratified(pool, args.n, rng)
        print(f"{len(pool)} candidate production pages in the feedback corpus "
              f"(duplicates removed only among the sample); sampled {len(picked)} distinct")
    index_path = f"{args.data_dir}/real/index.json"
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    for k, p in enumerate(picked, 1):
        pid = f"{args.prefix}{k:03d}"
        if os.path.exists(f"{review}/{pid}.txt"):
            print(f"  {pid} already exists -- skipped (never overwrite a page someone may be reviewing)")
            continue
        img = Image.open(p["png"]).convert("RGB")
        shutil.copy(p["png"], f"{review}/{pid}.png")
        text, orient = draft(args.url, img, f"eval-draft-{pid}")
        with open(f"{review}/{pid}.txt", "w", encoding="utf-8") as f:
            f.write(f"{MARKER}\n# draft read the page {orient}; correct it in normal reading order\n{text}\n")
        index.append({"id": pid, "entry": p["entry"], "stored_score": p["score"], "draft_orientation": orient})
        stored = f"stored score {p['score']:.3f}" if p["score"] is not None else "representative"
        print(f"  {pid}  {stored}  draft {len(tokens(text)):4d} words  ({orient})", flush=True)
        with open(index_path, "w") as f:           # saved as we go, so an interrupted run loses nothing
            json.dump(index, f, indent=1)
    with open(f"{review}/README.txt", "w") as f:
        f.write(REVIEW_GUIDE)
    print(f"\nreview folder ready: {review}")


def freeze(args):
    real = f"{args.data_dir}/real"
    os.makedirs(f"{real}/gt", exist_ok=True); os.makedirs(f"{real}/images", exist_ok=True)
    items, pending = [], 0
    for fn in sorted(os.listdir(f"{real}/review")):
        if not fn.endswith(".txt") or fn == "README.txt":
            continue
        pid = fn[:-4]
        lines = open(f"{real}/review/{fn}", encoding="utf-8").read().splitlines()
        if lines and lines[0].startswith("#DRAFT"):
            pending += 1
            continue
        text = "\n".join(l for l in lines if not l.startswith("# draft read the page"))
        with open(f"{real}/gt/{pid}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        shutil.copy(f"{real}/review/{pid}.png", f"{real}/images/{pid}.png")
        items.append({"id": pid, "doc": pid, "category": "real", "page": 1, "variant": "as_received",
                      "image": f"images/{pid}.png", "gt": f"gt/{pid}.txt", "gt_words": len(tokens(text))})
    with open(f"{real}/manifest.json", "w") as f:
        json.dump({"set": "real", "items": items}, f, indent=1)
    print(f"froze {len(items)} verified page(s); {pending} still awaiting review")


REVIEW_GUIDE = """HOW TO VERIFY THESE PAGES
=========================

Each rNNN.png is a real page as the OCR service received it. rNNN.txt is a
machine draft of its text. Your job: make each .txt match the page exactly.

1. Open the image and its .txt side by side.
2. Fix every wrong, missing or extra word. Read the image, not the draft --
   the draft was written by the same kind of model being tested, and it is
   easy to wave its mistakes through.
3. Include ALL legible text: headers, footers, fine print, table cells,
   form labels, handwriting, stamps.
4. Order: normal reading order, top to bottom. Tables row by row. Exact order
   matters less than getting every word -- accuracy is mostly scored per word.
5. Formatting does not matter: no need for markdown, tables or line breaks
   that match the page.
6. A word you genuinely cannot read: write [?].
7. If the page has no text at all, leave the file empty.
8. When a page is done, DELETE THE FIRST LINE (#DRAFT ...). That line is how
   finished pages are told apart from unfinished ones.

These pages are customer documents. Do not copy them anywhere else.
"""


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true"); g.add_argument("--freeze", action="store_true")
    ap.add_argument("--data-dir", default=os.environ.get("EVAL_DATA_DIR", "/workspace/eval_data"))
    ap.add_argument("--feedback-dir", default=os.path.join(os.path.dirname(__file__), "..", "feedback", "pending"))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--pdf-set", default="", help="sample from this build_pdf_set.py set instead of feedback")
    ap.add_argument("--per-doc", type=int, default=6)
    ap.add_argument("--prefix", default="r", help="page id prefix: r = feedback sample, s = scanned PDFs")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    prepare(args) if args.prepare else freeze(args)


if __name__ == "__main__":
    main()
