"""Run an eval set against the live service and score it.

Each page image is sent to POST /ocr/image the way the external service sends
production traffic (a PNG named page_1.png, default prompt, retry on), tagged
with X-Request-ID "eval-<label>-<item>" so eval traffic is identifiable in the
request log and excluded from the feedback corpus.

Writes, under <data-dir>/runs/:
    <label>.json   per-item metrics and slice summaries -- no OCR text, safe to
                   commit (copy into eval/baselines/ to freeze a baseline)
    <label>/       the OCR text for each item, for debugging. Stays outside the
                   repo: for the real set it is customer data.

Usage:
    python eval/run_eval.py --set simulated --label baseline_a
    python eval/run_eval.py --set simulated --label quick --variants clean_external,scan_medium
"""

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(__file__))
from metrics import score_page  # noqa: E402

FAIL_F1 = 0.5     # a page below this F1 counts as a failed page
LENGTH_LIMIT = 7200   # num_tokens at or above this: generation was cut off (the ceiling is ~7,280)
_REF_LABEL = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.S)
_GROUNDED = re.compile(r"<\|ref\|>.*?<\|/det\|>|<｜end▁of▁sentence｜>", re.S)


def picture_only(raw_text: str) -> bool:
    """The model labelled the whole page a picture and emitted nothing else --
    the most common total failure in production, since cleanup then deletes it."""
    labels = _REF_LABEL.findall(raw_text or "")
    return bool(labels) and set(labels) == {"image"} and not _GROUNDED.sub("", raw_text).strip()


def run_item(item, args, data_dir, text_dir):
    with open(f"{data_dir}/{item['image']}", "rb") as f:
        png = f.read()
    gt = None
    if item.get("gt"):                      # unlabeled sets profile failures without accuracy
        with open(f"{data_dir}/{item['gt']}", encoding="utf-8") as f:
            gt = f.read()
    rid = f"eval-{args.label}-{item['id']}"[:128]
    started = time.monotonic()
    try:
        r = requests.post(f"{args.url}/ocr/image",
                          files={"file": ("page_1.png", png, "image/png")},
                          data={"prompt": args.prompt, "retry": str(args.retry).lower()},
                          headers={"X-Request-ID": rid}, timeout=1800)
        latency = time.monotonic() - started
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except requests.RequestException as e:
        return {**_ident(item), "status": None, "error": str(e), "latency_s": round(time.monotonic() - started, 2)}
    text = body.get("text", "") if r.status_code == 200 else ""
    with open(f"{text_dir}/{item['id']}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    score = body.get("score") or {}
    row = {
        **_ident(item),
        "status": r.status_code,
        "latency_s": round(latency, 2),
        "flag": body.get("flag"),
        "score": score.get("composite"),
        "tokens": body.get("num_tokens"),
        "attempts": body.get("attempts"),
        "engine": body.get("ocr_engine"),
        "codes": [d.get("code") for d in body.get("flag_details") or []],
        "picture_only": picture_only(body.get("raw_text", "")),
        "hit_limit": (body.get("num_tokens") or 0) >= LENGTH_LIMIT,
        "out_chars": len(text),
    }
    if gt is not None:
        row.update(score_page(gt, text))
    else:
        row["empty"] = not text.strip()
    return row


def _ident(item):
    return {k: item[k] for k in ("id", "doc", "category", "page", "variant")}


def summarize(rows):
    ok = [r for r in rows if r.get("status") == 200]
    if not ok:
        return {"n": len(rows), "errors": len(rows)}
    pct = lambda cond: round(100 * sum(1 for r in ok if cond(r)) / len(ok), 1)
    out = {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "empty_pct": pct(lambda r: r["empty"]),
        "picture_only_pct": pct(lambda r: r.get("picture_only")),
        "hit_limit_pct": pct(lambda r: r.get("hit_limit")),
        "red_pct": pct(lambda r: r["flag"] == "red"),
        "latency_s": round(st.mean(r["latency_s"] for r in ok), 2),
        "tokens": sum(r["tokens"] or 0 for r in ok),
    }
    labeled = [r for r in ok if "f1" in r]
    if labeled:
        out.update({
            "f1": round(st.mean(r["f1"] for r in labeled), 4),
            "recall": round(st.mean(r["recall"] for r in labeled), 4),
            "precision": round(st.mean(r["precision"] for r in labeled), 4),
            "wer": round(st.mean(r["wer"] for r in labeled), 4),
            "failed_pct": pct(lambda r: "f1" in r and r["f1"] < FAIL_F1),
        })
    return out


def slices(rows):
    out = {"overall": summarize(rows)}
    for key in ("variant", "category"):
        for v in sorted({r[key] for r in rows}):
            out[f"{key}={v}"] = summarize([r for r in rows if r[key] == v])
    return out


def print_table(summary):
    print(f"\n{'slice':28s} {'n':>4} {'F1':>6} {'WER':>6} {'fail%':>6} "
          f"{'empty%':>7} {'pict%':>6} {'limit%':>7} {'red%':>6} {'sec':>6}")
    for name, s in summary.items():
        if "empty_pct" not in s:
            print(f"{name:28s} {s['n']:4d}  all errored"); continue
        f1 = f"{s['f1']:6.3f} {s['wer']:6.3f} {s['failed_pct']:6.1f}" if "f1" in s else f"{'-':>6} {'-':>6} {'-':>6}"
        print(f"{name:28s} {s['n']:4d} {f1} {s['empty_pct']:7.1f} {s['picture_only_pct']:6.1f} "
              f"{s['hit_limit_pct']:7.1f} {s['red_pct']:6.1f} {s['latency_s']:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="simulated", help="simulated, real, or any set dir with a manifest.json")
    ap.add_argument("--label", required=True, help="name for this run, e.g. baseline_a")
    ap.add_argument("--data-dir", default=os.environ.get("EVAL_DATA_DIR", "/workspace/eval_data"))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--prompt", default="document")
    ap.add_argument("--retry", default=True, type=lambda s: s.lower() == "true")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="1 gives the model's deterministic best case; production batches")
    ap.add_argument("--variants", default="", help="comma-separated subset of variants")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    set_dir = f"{args.data_dir}/{args.set}"
    items = json.load(open(f"{set_dir}/manifest.json"))["items"]
    if args.variants:
        wanted = set(args.variants.split(","))
        items = [i for i in items if i["variant"] in wanted]
    if args.limit:
        items = items[: args.limit]
    runs = f"{args.data_dir}/runs"
    text_dir = f"{runs}/{args.label}"
    os.makedirs(text_dir, exist_ok=True)

    health = requests.get(f"{args.url}/health", timeout=10).json()
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    print(f"{len(items)} items, set={args.set}, concurrency={args.concurrency}, commit={commit}{'+dirty' if dirty else ''}")

    rows, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, row in enumerate(ex.map(lambda it: run_item(it, args, set_dir, text_dir), items), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(items):
                el = time.monotonic() - started
                print(f"  {i}/{len(items)} done, {el / 60:.1f} min elapsed, ~{el / i * (len(items) - i) / 60:.0f} min left", flush=True)

    summary = slices(rows)
    result = {
        "label": args.label, "set": args.set,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit + ("+dirty" if dirty else ""),
        "service": health, "params": {"prompt": args.prompt, "retry": args.retry, "concurrency": args.concurrency},
        "summary": summary, "items": rows,
    }
    with open(f"{runs}/{args.label}.json", "w") as f:
        json.dump(result, f, indent=1)
    print_table(summary)
    print(f"\nwrote {runs}/{args.label}.json")


if __name__ == "__main__":
    main()
