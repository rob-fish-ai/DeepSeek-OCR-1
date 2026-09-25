# OCR accuracy evaluation

Measures OCR accuracy against known-correct text, so that changes to the engine
are judged by numbers rather than by the quality score (which does not track
accuracy: correlation with real word error was +0.05).

Two sets, reported separately:

| Set | What | Ground truth | Role |
|---|---|---|---|
| `simulated` | Born-digital pages rendered and damaged to imitate scans | Exact (the PDF's own text) | Fast, repeatable signal |
| `real` | Real production pages | Verified by a person | Decides whether a change ships |

Data lives in `$EVAL_DATA_DIR` (default `/workspace/eval_data`), **outside the
repository**. The real set is customer data and must never be committed. Only
code and metric-only results belong in the repo.

## Metrics

Per page, against ground truth: word-level **F1** (headline), recall,
precision, and WER. F1 is order-insensitive on purpose — on forms and tables the
same words can legitimately come out in a different order. Slices also report
`empty%` (no text returned), `fail%` (F1 < 0.5) and `red%` (flagged red).

## Simulated set

```
# public sources (downloaded once)
mkdir -p /workspace/eval_data/sources && cd /workspace/eval_data/sources
curl -Lo irs_w9.pdf     https://www.irs.gov/pub/irs-pdf/fw9.pdf
curl -Lo irs_w9_es.pdf  https://www.irs.gov/pub/irs-pdf/fw9sp.pdf
curl -Lo irs_w4.pdf     https://www.irs.gov/pub/irs-pdf/fw4.pdf
curl -Lo irs_1040.pdf   https://www.irs.gov/pub/irs-pdf/f1040.pdf
curl -Lo irs_4506c.pdf  https://www.irs.gov/pub/irs-pdf/f4506c.pdf
curl -Lo irs_ss4.pdf    https://www.irs.gov/pub/irs-pdf/fss4.pdf
curl -Lo irs_8821.pdf   https://www.irs.gov/pub/irs-pdf/f8821.pdf
curl -Lo irs_p15t.pdf   https://www.irs.gov/pub/irs-pdf/p15t.pdf
curl -Lo arxiv_attention.pdf https://arxiv.org/pdf/1706.03762
curl -Lo arxiv_resnet.pdf    https://arxiv.org/pdf/1512.03385

python eval/build_simulated.py
```

Public forms and papers are supplemented by synthetic documents
(`synthetic.py`: bank statement, pay stub, letter, invoice, dense table, sparse
cover and signature pages) built from fixed word lists — no real personal data.

Each page becomes six variants: `clean_external` (resized to 1280×1920, the
fixed size the external service sends every page at), `clean_native` (real
aspect ratio — a control for that resize), `scan_light`, `scan_medium`,
`scan_heavy`, and `rotated` (sideways content in a portrait frame).

Known limitation: `rotated` fits a sideways portrait page into a portrait frame,
so its text is also ~⅔ the size — it mixes "sideways" with "smaller".

## Real set

```
python eval/build_real.py --prepare --n 80     # drafts for review
# a person corrects <data-dir>/real/review/rNNN.txt against rNNN.png
# and deletes the #DRAFT line of each finished page (see review/README.txt)
python eval/build_real.py --freeze             # verified pages -> eval set
```

Sampled from the feedback corpus, which only holds pages that scored below
0.70 — so it is biased toward pages the engine struggles with. Add
representative real pages (PNG + verified TXT) to `review/` when available.

### Representative real PDFs

```
python eval/build_pdf_set.py --pdf-dir /workspace/eval_data/drive --name scanned_pdfs
python eval/run_eval.py --set scanned_pdfs --label profile        # failure rates, no labels needed
python eval/build_real.py --prepare --pdf-set scanned_pdfs --per-doc 6 --prefix s
```

`build_pdf_set.py` renders every page as production traffic arrives (PDF
`/Rotate` honoured, 1280×1920 frame). Documents are referred to by alias
(`d1`, `d2`, …): source filenames are personal names and stay beside the data.
Without ground truth, `run_eval.py` still reports the failures it can see in
each response — empty output, picture-only pages (the whole page labelled an
image and discarded), length-limit hits, red flags. Pages sampled with
`--pdf-set` are de-duplicated by content, since the same scan can arrive in
more than one PDF.

## Running and comparing

```
python eval/run_eval.py --set simulated --label baseline_a
python eval/run_eval.py --set simulated --label baseline_b     # same code again
python eval/run_eval.py --set simulated --label my_change

python eval/compare.py runs/baseline_a.json runs/my_change.json \
       --noise runs/baseline_a.json runs/baseline_b.json
```

The service is not fully deterministic — batching with other traffic changes
decode numerics — so a change counts only if it beats the spread between two
runs of identical code. Frozen baselines are kept in `eval/baselines/`.

Eval requests carry `X-Request-ID: eval-...`. Feedback entries record the
request ID, so eval traffic can be excluded from the training corpus.

Runs default to concurrency 1: the model's deterministic best case. Production
traffic is batched, which is measurably worse (see ARCHITECTURE.md, *Batching
vs reproducibility*); use `--concurrency` to measure that.
