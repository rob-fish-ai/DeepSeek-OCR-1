"""Build the simulated-scan eval set.

Born-digital PDFs have exact ground truth in their text layer. Each selected
page is rendered, then turned into several variants that imitate how scanned
pages actually reach the service:

    clean_external  rendered page, resized to 1280x1920 -- the fixed size the
                    external service sends every page at, regardless of the
                    page's real shape
    clean_native    same page at its real aspect ratio (control for the above)
    scan_light      light scanner damage: slight skew, noise, JPEG
    scan_medium     moderate: more skew, blur, grey paper, lower resolution
    scan_heavy      severe: uneven lighting, heavy noise and blur, low resolution
    rotated         medium damage, content turned 90 or 270 degrees inside a
                    portrait frame -- a landscape page scanned sideways

Simulated damage is not a real scanner. Treat these numbers as fast,
repeatable signal; the human-verified real set decides whether changes ship.

Usage:
    python eval/build_simulated.py [--data-dir /workspace/eval_data]
"""

import argparse
import io
import json
import os
import random
import sys

import fitz
import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from metrics import tokens  # noqa: E402
import synthetic  # noqa: E402

# Public born-digital sources. Downloaded by hand into <data-dir>/sources; see
# README.md for URLs. category drives per-slice reporting.
PUBLIC = {
    "irs_w9": "form", "irs_w9_es": "form_es", "irs_w4": "form", "irs_1040": "form",
    "irs_4506c": "form", "irs_ss4": "form", "irs_8821": "form",
    "irs_p15t": "publication", "arxiv_attention": "paper", "arxiv_resnet": "paper",
}
MAX_PAGES_PER_DOC = 2
MIN_WORDS = 40            # skip near-empty pages of public docs (synthetic sparse pages are kept on purpose)
RENDER_DPI = 200          # a common scanner resolution
EXTERNAL_FRAME = (1280, 1920)

VARIANTS = ["clean_external", "clean_native", "scan_light", "scan_medium", "scan_heavy", "rotated"]

SCAN_LEVELS = {
    #          skew deg, noise sigma, blur, paper tone, contrast, jpeg q, resolution scale, uneven light
    "light":  dict(skew=0.5, noise=5,  blur=0.0, paper=246, contrast=0.95, jpeg=85, res=1.0,  shade=0.0),
    "medium": dict(skew=1.5, noise=11, blur=0.7, paper=232, contrast=0.85, jpeg=60, res=0.75, shade=0.10),
    "heavy":  dict(skew=3.0, noise=18, blur=1.2, paper=215, contrast=0.72, jpeg=40, res=0.55, shade=0.25),
}


def render(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def to_external_frame(img: Image.Image) -> Image.Image:
    """What the external service does: force every page to 1280x1920."""
    return img.resize(EXTERNAL_FRAME, Image.LANCZOS)


def to_native(img: Image.Image) -> Image.Image:
    """Same pixel budget as the external frame, real aspect ratio kept."""
    scale = min(EXTERNAL_FRAME[0] / img.width, EXTERNAL_FRAME[1] / img.height)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def scan(img: Image.Image, level: str, rng: random.Random) -> Image.Image:
    p = SCAN_LEVELS[level]
    g = img.convert("L")
    if p["res"] < 1.0:                                   # lower scan resolution
        small = g.resize((int(g.width * p["res"]), int(g.height * p["res"])), Image.BILINEAR)
        g = small.resize(g.size, Image.BILINEAR)
    g = g.rotate(rng.uniform(-p["skew"], p["skew"]), resample=Image.BICUBIC, fillcolor=255)
    if p["blur"]:
        g = g.filter(ImageFilter.GaussianBlur(p["blur"]))
    a = np.asarray(g, dtype=np.float32)
    a = 255 - (255 - a) * p["contrast"]                  # faded ink
    a = a * (p["paper"] / 255.0)                         # grey / off-white paper
    if p["shade"]:                                       # uneven lighting across the page
        ramp = np.linspace(1.0 - p["shade"], 1.0, a.shape[1], dtype=np.float32)
        a = a * (ramp if rng.random() < 0.5 else ramp[::-1])[None, :]
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    a = a + nrng.normal(0, p["noise"], a.shape)
    g = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); g.save(buf, "JPEG", quality=p["jpeg"])
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def rotated(img: Image.Image, idx: int, rng: random.Random) -> Image.Image:
    """Content turned sideways inside a portrait frame, as sideways scans arrive."""
    turned = scan(img, "medium", rng).rotate(90 if idx % 2 == 0 else 270, expand=True)
    frame = Image.new("RGB", EXTERNAL_FRAME, (255, 255, 255))
    fit = ImageOps.contain(turned, EXTERNAL_FRAME, Image.LANCZOS)
    frame.paste(fit, ((EXTERNAL_FRAME[0] - fit.width) // 2, (EXTERNAL_FRAME[1] - fit.height) // 2))
    return frame


def select_pages(doc: fitz.Document, keep_sparse: bool) -> list[int]:
    ok = [i for i in range(len(doc)) if keep_sparse or len(tokens(doc[i].get_text())) >= MIN_WORDS]
    if len(ok) <= MAX_PAGES_PER_DOC:
        return ok
    step = (len(ok) - 1) / (MAX_PAGES_PER_DOC - 1)       # spread across the document
    return [ok[round(k * step)] for k in range(MAX_PAGES_PER_DOC)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("EVAL_DATA_DIR", "/workspace/eval_data"))
    args = ap.parse_args()
    src = f"{args.data_dir}/sources"
    out = f"{args.data_dir}/simulated"
    os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/gt", exist_ok=True)

    docs = {name: cat for name, cat in PUBLIC.items() if os.path.exists(f"{src}/{name}.pdf")}
    missing = sorted(set(PUBLIC) - set(docs))
    if missing:
        print(f"warning: missing public sources (see README): {missing}")
    docs.update(synthetic.build_all(src))

    items, rng = [], random.Random(7)
    for name, category in docs.items():
        doc = fitz.open(f"{src}/{name}.pdf")
        for pno in select_pages(doc, keep_sparse=category == "sparse"):
            base = f"{name}_p{pno + 1}"
            gt = doc[pno].get_text()
            with open(f"{out}/gt/{base}.txt", "w", encoding="utf-8") as f:
                f.write(gt)
            img = render(doc[pno])
            variants = {
                "clean_external": to_external_frame(img),
                "clean_native": to_native(img),
                "scan_light": to_external_frame(scan(img, "light", rng)),
                "scan_medium": to_external_frame(scan(img, "medium", rng)),
                "scan_heavy": to_external_frame(scan(img, "heavy", rng)),
                "rotated": rotated(img, len(items), rng),
            }
            for variant, vimg in variants.items():
                path = f"images/{base}__{variant}.png"
                vimg.save(f"{out}/{path}")
                items.append({"id": f"{base}__{variant}", "doc": name, "category": category,
                              "page": pno + 1, "variant": variant, "image": path,
                              "gt": f"gt/{base}.txt", "gt_words": len(tokens(gt))})
    with open(f"{out}/manifest.json", "w") as f:
        json.dump({"set": "simulated", "items": items}, f, indent=1)
    pages = len({i["gt"] for i in items})
    print(f"built {len(items)} items = {pages} pages x {len(VARIANTS)} variants -> {out}")
    for cat in sorted({i["category"] for i in items}):
        n = len({i["gt"] for i in items if i["category"] == cat})
        print(f"  {cat:12s} {n} page(s)")


if __name__ == "__main__":
    main()
