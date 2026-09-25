"""Turn a folder of real scanned PDFs into an unlabeled page set.

Every page is rendered the way the external service's traffic arrives: with
the PDF's /Rotate flag honoured (scanners commonly store a landscape image
plus a rotation flag), then resized to the fixed 1280x1920 frame. There is no
ground truth, so run_eval.py reports failure rates only -- empty output,
picture-only pages, length-limit hits, red flags -- which is enough to measure
how often each failure occurs on representative documents.

Source documents are identified by alias (d1, d2, ...) only; their filenames
are personal names and never leave <data-dir>.

Usage:
    python eval/build_pdf_set.py --pdf-dir /workspace/eval_data/drive --name scanned_pdfs
"""

import argparse
import io
import json
import os

import fitz
from PIL import Image

RENDER_DPI = 200
EXTERNAL_FRAME = (1280, 1920)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--name", default="scanned_pdfs", help="set name (directory under data-dir)")
    ap.add_argument("--data-dir", default=os.environ.get("EVAL_DATA_DIR", "/workspace/eval_data"))
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf"))
    aliases = {f: f"d{i + 1}" for i, f in enumerate(files)}
    out = f"{args.data_dir}/{args.name}"
    os.makedirs(f"{out}/images", exist_ok=True)
    with open(f"{out}/aliases.json", "w") as f:        # name -> alias; stays beside the data
        json.dump(aliases, f, indent=1)

    items = []
    for fname, alias in aliases.items():
        doc = fitz.open(f"{args.pdf_dir}/{fname}")
        for pno, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").resize(EXTERNAL_FRAME, Image.LANCZOS)
            pid = f"{alias}_p{pno + 1:02d}"
            img.save(f"{out}/images/{pid}.png")
            items.append({"id": pid, "doc": alias, "category": alias, "page": pno + 1,
                          "variant": "as_rendered", "image": f"images/{pid}.png", "gt": None})
    with open(f"{out}/manifest.json", "w") as f:
        json.dump({"set": args.name, "labeled": False, "items": items}, f, indent=1)
    print(f"{len(items)} pages from {len(files)} PDFs -> {out}")


if __name__ == "__main__":
    main()
