"""
Failures on pre-cutoff protein-RNA complexes next to the post-cutoff ones (paper table tab:pre_failures).

Per complex (protein_rna.db of each period, bundle databases):
  poorly folded RNA   RNA_LDDT <= 0.85, the backbone (C3') LDDT_r of the longest RNA chain (fold classes iii and iv)
  failed interface    at least one RNA chain outside correct_interface, the RNA chain taking the best class over
                      its protein-RNA interfaces (RNA_interface_placement); --any-interface counts every failed
                      protein-RNA interface instead
  properties          from the deposited structure (exp_protein_rna): cryo-EM, >= 3 protein chains, >= 2 RNA
                      chains, short RNA (longest RNA chain < SHORT_NT nt), RNA duplex, ions, Rfam family; and the
                      predicted RNA poorly folded (interface column only)
For each property: share of complexes that fail with / without it, n = complexes with it, Fisher's exact test.
Lower rows: C2'-endo and syn share of the deposited RNA for poorly vs well-folded predictions (median, Mann-Whitney).
Also prints the composition of the failed interfaces (main-text sentence before the table).

Run from the repo root (RNABench env):  python evaluation/pre_failures_table.py [--any-interface]
Outputs: results/plots/all_chains/pre_failures/{pre_failures.tsv, pre_failures_rows.tex}
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sqlite3
import sys
from collections import Counter

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.bundle_paths import PROTEIN_RNA_DB
from evaluation.placement_labels import best_per_rna_chain, interface_placement

OUT = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "pre_failures")
GOOD_LDDT = 0.85
SHORT_NT = 40          # longest RNA chain < 40 nt; reproduces the earlier table's n (703 pre-cutoff) and fold shares
SET_LABEL = {"pre": "pre-cutoff", "post": "post-cutoff"}

PROPERTIES = [          # (column, LaTeX label)
    ("cryo_em", r"cryo-EM structure"),
    ("many_proteins", r"$\geq3$ protein chains"),
    ("multi_rna", r"$\geq2$ RNA chains"),
    ("short_rna", rf"RNA $<{SHORT_NT}$~nt"),
    ("duplex", r"RNA duplex"),
    ("ions", r"Ions in the structure"),
    ("rfam", r"Rfam family annotated"),
    ("poorly_folded", r"RNA poorly folded"),
]
# failed-interface classes as grouped in the main text
FAIL_GROUPS = {"right_protein_site_wrong_rna_pose": "correct site, wrong pose",
               "same_place_duplex_slid": "correct site, wrong pose",
               "different_region": "distal site",
               "protein_binding_region_mispredicted": "mispredicted protein binding region",
               "shifted_nearby": "shifted or rotated", "same_place_rna_turned": "shifted or rotated",
               "same_place_same_orientation": "shifted or rotated"}


def _list(v):
    if v is None:
        return []
    if isinstance(v, (int, float)):
        return [v]
    for parse in (json.loads, ast.literal_eval):
        try:
            x = parse(v)
            return x if isinstance(x, list) else [x]
        except (ValueError, SyntaxError, TypeError):
            continue
    return [v]


def _nums(v):
    out = []
    for x in _list(v):
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            pass
    return out


def _dict(v):
    if not v:
        return {}
    for parse in (json.loads, ast.literal_eval):
        try:
            x = parse(v)
            return x if isinstance(x, dict) else {}
        except (ValueError, SyntaxError, TypeError):
            continue
    return {}


def _share(d, key):
    d = _dict(d)
    tot = sum(d.values())
    return d.get(key, 0) / tot if tot else None


def load(period, per_chain=True):
    con = sqlite3.connect(PROTEIN_RNA_DB[period])
    q = ("SELECT e.PDBId, e.Experiment, e.NumberProteins, e.NumberRNAs, e.RNALength, e.RNA_isDuplex, e.Ions, "
         "e.RNAFamily, e.RNA_SugarPucker, e.RNA_GlycosidicBond, p.RNA_LDDT, p.RNA_interface_placement "
         "FROM exp_protein_rna e JOIN pred_protein_rna p ON p.exp_db_id = e.PDBId")
    rows, fails = [], Counter()
    for (pid, exp, npro, nrna, rlen, dup, ions, fam, pucker, gly, lddt, plc) in con.execute(q):
        lddt = _nums(lddt)
        if not lddt:
            continue
        classes = best_per_rna_chain(plc) if per_chain else interface_placement(plc)
        if not classes:
            continue
        failed = [c for c in classes.values() if c != "correct_interface"]
        fails.update(FAIL_GROUPS.get(c, c) for c in failed)
        fails["_total"] += len(classes)
        lens = _nums(rlen)
        rows.append(dict(
            pdb=pid, cryo_em=exp == "ELECTRON MICROSCOPY", many_proteins=(npro or 0) >= 3,
            multi_rna=(nrna or 0) >= 2, short_rna=bool(lens) and max(lens) < SHORT_NT,
            duplex=str(dup) == "1", ions=len(_list(ions)) > 0,
            rfam=any(f not in (None, "None", "") for f in _list(fam)),
            poorly_folded=lddt[0] <= GOOD_LDDT, failed_interface=bool(failed),
            c2endo=_share(pucker, "C2'-endo"), syn=_share(gly, "syn")))
    return pd.DataFrame(rows), fails


def _p_tex(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def _with_without(df, prop, outcome):
    w, wo = df[df[prop]], df[~df[prop]]
    a, b = int(w[outcome].sum()), int(len(w) - w[outcome].sum())
    c, d = int(wo[outcome].sum()), int(len(wo) - wo[outcome].sum())
    p = fisher_exact([[a, b], [c, d]])[1] if len(w) and len(wo) else float("nan")
    pct = lambda k, n: round(100 * k / n) if n else float("nan")
    return pct(a, len(w)), pct(c, len(wo)), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--any-interface", action="store_true", help="failed interface = any failed protein-RNA interface")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = {p: load(p, not args.any_interface) for p in ("pre", "post")}

    tsv, tex = [], []
    for prop, label in PROPERTIES:
        for i, period in enumerate(("pre", "post")):
            df = data[period][0]
            n = int(df[prop].sum())
            if prop == "poorly_folded":
                fw = fwo = fp = None
            else:
                fw, fwo, fp = _with_without(df, prop, "poorly_folded")
            iw, iwo, ip = _with_without(df, prop, "failed_interface")
            tsv.append(dict(property=prop, set=period, n=n, n_set=len(df), poor_with=fw, poor_without=fwo, poor_p=fp,
                            fail_with=iw, fail_without=iwo, fail_p=ip))
            fold = "-- & --" if fw is None else f"{fw} / {fwo} & {_p_tex(fp)}"
            tex.append(f"{label if i == 0 else ''} & {SET_LABEL[period]} & {n} & {fold} & {iw} / {iwo} & {_p_tex(ip)} \\\\")
        tex.append(r"\addlinespace[2pt]")
    tex[-1] = r"\midrule"
    tex.append(r" & & & poorly / well folded & $P$ & & \\")
    for key, label in (("c2endo", r"C2$'$-\textit{endo} share"), ("syn", r"\textit{syn} share")):
        for i, period in enumerate(("pre", "post")):
            df = data[period][0].dropna(subset=[key])
            poor, well = df[df.poorly_folded][key], df[~df.poorly_folded][key]
            p = mannwhitneyu(poor, well).pvalue
            tsv.append(dict(property=key, set=period, n=len(df), poor_with=round(poor.median(), 3),
                            poor_without=round(well.median(), 3), poor_p=p))
            tex.append(f"{label if i == 0 else ''} & {SET_LABEL[period]} & & {poor.median():.2f} / {well.median():.2f} "
                       f"& {_p_tex(p)} & & \\\\")

    pd.DataFrame(tsv).to_csv(os.path.join(OUT, "pre_failures.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, "pre_failures_rows.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    unit = "protein-RNA interfaces" if args.any_interface else "RNA chains"
    for period in ("pre", "post"):
        df, fails = data[period]
        total = fails.pop("_total")
        print(f"{SET_LABEL[period]}: {len(df)} complexes, {int(df.poorly_folded.sum())} poorly folded, "
              f"{int(df.failed_interface.sum())} with a failed interface | failed {unit}: {sum(fails.values())} of "
              f"{total} ({', '.join(f'{k} {v}' for k, v in fails.most_common())})")
    print("\n".join(tex))
    print(f"\nwritten {OUT}/pre_failures.tsv and pre_failures_rows.tex")


if __name__ == "__main__":
    main()
