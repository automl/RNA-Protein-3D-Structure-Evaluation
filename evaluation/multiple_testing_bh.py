"""
Benjamini-Hochberg correction over every P value reported in the paper.

Collects the P values written in main.tex (Results onwards, including the Supplementary
Information; commented-out lines skipped), treats them as ONE family -- more conservative than
correcting within each analysis -- and reports which ones lose significance at FDR 0.05.
"P < x" is taken as x (conservative). Values that are only bounds below 0.05 and stay below the
BH threshold are significant either way.

Usage:  python evaluation/multiple_testing_bh.py [path/to/main.tex]
Writes results/multiple_testing_bh.tsv (line, context, P, BH q).
"""

import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(_REPO_ROOT, "2025_NAR_AF3_Evaluation", "af3_final.tex")
OUT = os.path.join(_REPO_ROOT, "results", "multiple_testing_bh.tsv")

# $P=1.5\times10^{-72}$, P=0.17, P<0.001, P\sim10^{-102}, P = 4\times10^{-4}, P=3.4e-39
_NUM = r"(\d*\.?\d+)\s*(?:\\times\s*10\^\{?(-?\d+)\}?|e(-?\d+))?|10\^\{?(-?\d+)\}?"
_P = re.compile(r"(?<![A-Za-z\\])[Pp]\s*(?:\\,)?\s*(=|<|\\sim|\\approx|\\le|\\leq)\s*\$?\s*(?:" + _NUM + ")")


def _value(m):
    mant, exp1, exp2, exp_only = m.group(2), m.group(3), m.group(4), m.group(5)
    if exp_only is not None:
        return 10.0 ** int(exp_only)
    v = float(mant)
    exp = exp1 or exp2
    return v * 10.0 ** int(exp) if exp else v


def collect(tex):
    out, started = [], False
    for i, line in enumerate(open(tex, encoding="utf-8"), 1):
        if line.lstrip().startswith("%"):
            continue
        if "\\section{Results" in line:          # "Results" or "Results and Discussion"
            started = True
        if not started:
            continue
        line = line.split("%")[0] if "\\%" not in line else line
        for m in _P.finditer(line):
            try:
                p = _value(m)
            except (TypeError, ValueError):
                continue
            if 0 <= p <= 1:
                ctx = line[max(0, m.start() - 70):m.end() + 10].replace("\t", " ").strip()
                out.append(dict(line=i, rel=m.group(1), p=p, ctx=ctx))
    return out


def bh(ps):
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * n / rank)
        q[i] = prev
    return q


def main():
    tex = sys.argv[1] if len(sys.argv) > 1 else TEX
    rows = collect(tex)
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = qq
    sig = [r for r in rows if r["p"] < 0.05]
    lost = [r for r in sig if r["q"] >= 0.05]
    print(f"{len(rows)} P values reported; {len(sig)} nominally < 0.05; "
          f"{len(sig) - len(lost)} remain significant after BH (FDR 0.05, one family)")
    for r in lost:
        print(f"  loses significance: line {r['line']}: P={r['p']:.3g} -> q={r['q']:.3g} | {r['ctx']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("line\trelation\tP\tBH_q\tcontext\n")
        for r in rows:
            fh.write(f"{r['line']}\t{r['rel']}\t{r['p']:.3g}\t{r['q']:.3g}\t{r['ctx']}\n")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
