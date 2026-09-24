"""Compare two eval runs, judged against the run-to-run noise floor.

The service is not perfectly deterministic -- batching with other traffic
changes decode numerics -- so two runs of identical code differ. Pass two runs
of the SAME code as --noise to measure that spread; a change only counts as an
improvement or regression on a page if it moves F1 by more than the spread seen
between identical runs (with a 0.02 minimum).

Usage:
    python eval/compare.py BASE.json NEW.json [--noise BASE_A.json BASE_B.json]
"""

import argparse
import json


def load(path):
    return json.load(open(path))


def page_f1(run):
    return {r["id"]: r["f1"] for r in run["items"] if r.get("status") == 200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("new")
    ap.add_argument("--noise", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="two runs of identical code, used to set the per-page threshold")
    args = ap.parse_args()
    base, new = load(args.base), load(args.new)

    threshold = 0.02
    if args.noise:
        a, b = page_f1(load(args.noise[0])), page_f1(load(args.noise[1]))
        diffs = sorted(abs(a[k] - b[k]) for k in a.keys() & b.keys())
        if diffs:
            p95 = diffs[int(len(diffs) * 0.95) - 1] if len(diffs) > 1 else diffs[0]
            threshold = max(threshold, p95)
            moved = sum(d > 0.02 for d in diffs)
            print(f"noise floor: identical runs moved {moved}/{len(diffs)} pages by >0.02 F1; "
                  f"95th percentile |dF1| = {p95:.3f} -> per-page threshold {threshold:.3f}\n")

    print(f"base: {base['label']} ({base['code_commit']})   new: {new['label']} ({new['code_commit']})\n")
    print(f"{'slice':28s} {'F1 base':>8} {'F1 new':>7} {'dF1':>7} {'fail% base':>11} {'new':>6} {'empty% base':>12} {'new':>6}")
    for name, s in base["summary"].items():
        n = new["summary"].get(name)
        if not n or "f1" not in s or "f1" not in n:
            continue
        d = n["f1"] - s["f1"]
        mark = "  +" if d > 0.01 else ("  -" if d < -0.01 else "")
        print(f"{name:28s} {s['f1']:8.3f} {n['f1']:7.3f} {d:+7.3f} {s['failed_pct']:11.1f} {n['failed_pct']:6.1f} "
              f"{s['empty_pct']:12.1f} {n['empty_pct']:6.1f}{mark}")

    b, n = page_f1(base), page_f1(new)
    common = b.keys() & n.keys()
    better = sorted((n[k] - b[k], k) for k in common if n[k] - b[k] > threshold)
    worse = sorted((n[k] - b[k], k) for k in common if b[k] - n[k] > threshold)
    print(f"\npages compared: {len(common)}   improved: {len(better)}   regressed: {len(worse)}   "
          f"(threshold |dF1| > {threshold:.3f})")
    for d, k in worse[:15]:
        print(f"  REGRESSED {d:+.3f}  {k}")
    for d, k in better[-10:][::-1]:
        print(f"  improved  {d:+.3f}  {k}")


if __name__ == "__main__":
    main()
