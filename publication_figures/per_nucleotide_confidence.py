"""SI figure: does AF3's per-nucleotide confidence know which nucleotide-level features it got wrong?

Protein-RNA complexes before (b2021) and after (a2021) the training cutoff. The DSSR annotations of the experimental
structure and of the AF3 model are compared nucleotide by nucleotide (nucleotides matched by aligning the two DSSR
sequences, as in si_interaction_swaps.py). Per-nucleotide pLDDT_r = mean B-factor of the nucleotide's atoms in the AF3
model; an element's pLDDT_r = mean over its AF3 nucleotides. Elements and when AF3 gets them wrong:

  syn base            experimental syn glycosidic bond (|chi| < 90 deg), wrong if AF3 models it anti
  C2'-endo sugar      experimental south pucker (delta >= 115 deg), wrong if AF3 models it north (C3'-endo)
  non-g+ gamma        experimental gamma in trans or g- (outside 0-120 deg), wrong if AF3 models another gamma class
  anti / C3'-endo / g+   the generic states, wrong if AF3 models another state (reference)
  Watson-Crick pair   wrong if AF3 does not pair the two nucleotides (reference)
  non-WC pair         experimental non-canonical pair, wrong if AF3 does not pair the two nucleotides
  base stack          stacking contact (consecutive nucleotides of a DSSR stack), wrong if absent in AF3
  pseudoknot pair     pair in pseudoknot brackets of the DSSR dot-bracket, wrong if unpaired in AF3
  duplex strand       RNA strand of a duplex, wrong if out of place (occ_duplex < 0.5, symmetry check)

Read-outs:
  A  pLDDT_r of reproduced and missed elements; AUC = P(missed element ranked below a reproduced one)
  B  the same AUC over all elements and within structures (only pairs of elements of the same structure compared),
     separating 'pLDDT_r flags a poor structure' from 'pLDDT_r flags the wrong nucleotide'
  C  how often a non-generic state that AF3 models is also in the experiment, by AF3 pLDDT_r band
  Eye levels as in the graphical abstract: AUC >= 0.8 clear, 0.7-0.8 partial, < 0.7 weak separation.

Run from the repo root in the RNABench env:  python publication_figures/per_nucleotide_confidence.py
Output: results/plots/all_chains/publication_figures/main_final/per_nucleotide_confidence.{png,pdf,tsv}
"""
import csv
import json
import os
import sqlite3
import sys
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _REPO)
import figstyle
from si_interaction_swaps import entries, glyco, EXCLUDE
from evaluation.interface_rna_pose import _locate_model

ZIP = "data/all_data_zip/all_databases"
COHORTS = {"post": (f"{ZIP}/a2021_post_cutoff/protein_rna.db",
                    ["data/all_data/a2021_post_cutoff/protein_rna/dssr"]),
           "pre": (f"{ZIP}/b2021_pre_cutoff/protein_rna.db",
                   ["data/all_data/b2021_pre_cutoff/protein_rna/dssr"])}
SYMMETRY = "results/plots/all_chains/interface_symmetry_check/interface_symmetry_check.tsv"
OUT = "results/plots/all_chains/publication_figures/main_final"
OPEN, CLOSE = "[{<", "]}>"
FEATURES = ["syn base", "C2′-endo sugar", "non-g+ γ", "non-WC pair", "base stack", "pseudoknot pair", "duplex strand"]
REFERENCE = ["anti base", "C3′-endo sugar", "g+ γ", "WC pair"]
STATE = {"glyco": ("syn base", "anti base"), "pucker": ("C2′-endo sugar", "C3′-endo sugar"), "gamma": ("non-g+ γ", "g+ γ")}
BANDS = [(0, 0.7, "< 70"), (0.7, 0.9, "70–90"), (0.9, 1.01, "≥ 90")]


def pucker(n):
    d = n.get("delta")
    return None if d is None else ("south" if d >= 115 else "north")


def gamma(n):
    g = n.get("gamma")
    if g is None:
        return None
    g %= 360
    return "g+" if g < 120 else ("t" if g < 240 else "g-")


def states(n):
    gl = glyco(n)
    return dict(glyco=gl if gl in ("syn", "anti") else None, pucker=pucker(n), gamma=gamma(n))


GENERIC = dict(glyco="anti", pucker="north", gamma="g+")


def annot(x):
    idx = {n["nt_id"]: i for i, n in enumerate(x["nts"])}
    pairs, nc = set(), set()
    for p in x.get("pairs", []):
        if p["nt1"] in idx and p["nt2"] in idx:
            q = tuple(sorted((idx[p["nt1"]], idx[p["nt2"]]))); pairs.add(q)
            if p["name"] not in ("WC", "Wobble"):
                nc.add(q)
    st = set()
    for s in x.get("stacks", []):
        ids = [idx[i] for i in s["nts_long"].split(",") if i in idx]
        st |= {tuple(sorted(p)) for p in zip(ids, ids[1:])}
    s = (x.get("dbn") or {}).get("all_chains", {}).get("sstr", "").replace("&", "")
    pk = {i for i, ch in enumerate(s) if ch in OPEN + CLOSE} if len(s) == len(x["nts"]) else None
    return dict(pairs=pairs, nc=nc, stacks=st if "stacks" in x else None, pk=pk)


_PL = {}


def res_plddt(cif):
    if cif not in _PL:
        m = MMCIF2Dict(cif); acc = {}
        for c, s, b in zip(m["_atom_site.auth_asym_id"], m["_atom_site.auth_seq_id"], m["_atom_site.B_iso_or_equiv"]):
            acc.setdefault((c, s), []).append(float(b))
        _PL[cif] = {k: np.mean(v) / 100 for k, v in acc.items()}
    return _PL[cif]


def collect():
    """one row per experimental element (and per AF3 nucleotide in a non-generic state, kind 'call')"""
    rows = []
    for cohort, (db, dirs) in COHORTS.items():
        con = sqlite3.connect(db)
        fn = {r[0].strip().upper(): r[1] for r in con.execute("select exp_db_id, FileName from pred_protein_rna")}
        lddt = {r[0].strip().upper(): r[1] for r in con.execute("select exp_db_id, RNA_LDDT_allAtom from pred_protein_rna")}
        n_struct = 0
        for d in dirs:
            for pid, (f, g) in entries(d).items():
                pid = pid.upper()
                if pid in EXCLUDE or pid not in fn:
                    continue
                cif = _locate_model(fn[pid])
                if not cif:
                    continue
                T, A = json.load(open(f)), json.load(open(g))
                t = "".join(n["nt_code"].upper() for n in T["nts"]); a = "".join(n["nt_code"].upper() for n in A["nts"])
                m = {}
                for blk in SequenceMatcher(None, t, a, autojunk=False).get_matching_blocks():
                    m.update({blk.a + k: blk.b + k for k in range(blk.size)})
                if len(m) < 0.9 * len(t):
                    continue
                pl = res_plddt(cif)

                def p(j):
                    n = A["nts"][j]
                    return pl.get((n["chain_name"], str(n["nt_resnum"])))

                def el(ids):
                    v = [p(m[i]) for i in ids if i in m]
                    return float(np.mean(v)) if len(v) == len(ids) and v and None not in v else None
                n_struct += 1
                base = dict(cohort=cohort, pdb=pid, lddt_r=lddt.get(pid))
                # nucleotide conformers, experimental side
                for i, nt in enumerate(T["nts"]):
                    if i not in m or el([i]) is None:
                        continue
                    se, sa = states(nt), states(A["nts"][m[i]])
                    for k, (rare, generic) in STATE.items():
                        if se[k] is None or sa[k] is None:
                            continue
                        kind = generic if se[k] == GENERIC[k] else rare
                        same = se[k] == sa[k] if k != "gamma" else (se[k] == sa[k])
                        rows.append(dict(base, kind=kind, wrong=not same, plddt=el([i])))
                    # AF3 calls: a non-generic state in the model, confirmed if the experiment has the same state
                    for k, (rare, _) in STATE.items():
                        if sa[k] is not None and se[k] is not None and sa[k] != GENERIC[k]:
                            rows.append(dict(base, kind="call " + rare, wrong=se[k] != sa[k], plddt=el([i])))
                et, ea = annot(T), annot(A)
                mp = lambda q: tuple(sorted((m[q[0]], m[q[1]]))) if q[0] in m and q[1] in m else None
                for q in et["pairs"]:
                    if mp(q) and el(q) is not None:
                        rows.append(dict(base, kind="non-WC pair" if q in et["nc"] else "WC pair",
                                         wrong=mp(q) not in ea["pairs"], plddt=el(q)))
                        if et["pk"] and q[0] in et["pk"] and q[1] in et["pk"]:
                            rows.append(dict(base, kind="pseudoknot pair", wrong=mp(q) not in ea["pairs"], plddt=el(q)))
                if et["stacks"] is not None and ea["stacks"] is not None:
                    for q in et["stacks"]:
                        if mp(q) and el(q) is not None:
                            rows.append(dict(base, kind="base stack", wrong=mp(q) not in ea["stacks"], plddt=el(q)))
        print(f"{cohort}: {n_struct} structures")
        # duplex strands: per RNA chain of a duplex, pLDDT_r of that chain in the AF3 model
        strand = {}
        for r in csv.DictReader(open(SYMMETRY), delimiter="\t"):
            if r["period"] == cohort and r["duplex"] == "True":
                try:
                    occ = float(r["occ_duplex"])
                except ValueError:
                    continue
                k = (r["pdb"].upper(), r["model_pair"].split(":")[1])
                strand[k] = min(strand.get(k, 1.0), occ)
        for (pid, ch), occ in strand.items():
            cif = _locate_model(fn[pid]) if pid in fn else None
            if not cif:
                continue
            v = [x for (c, _), x in res_plddt(cif).items() if c == ch]
            if v:
                rows.append(dict(cohort=cohort, pdb=pid, lddt_r=lddt.get(pid), kind="duplex strand", wrong=occ < 0.5,
                                 plddt=float(np.mean(v))))
    return pd.DataFrame(rows)


def auc(lo, hi):
    """P(an element of `lo` scores below one of `hi`), ties count half"""
    lo, hi = np.asarray(lo, float)[:, None], np.asarray(hi, float)[None, :]
    return float((lo < hi).mean() + 0.5 * (lo == hi).mean()) if lo.size and hi.size else np.nan


def auc_within(d):
    """AUC over pairs (missed, reproduced) of the same structure only"""
    num = den = 0.0
    for _, s in d.groupby("pdb"):
        w, c = s.plddt[s.wrong].to_numpy(float)[:, None], s.plddt[~s.wrong].to_numpy(float)[None, :]
        if w.size and c.size:
            num += (w < c).sum() + 0.5 * (w == c).sum(); den += w.size * c.size
    return num / den if den else np.nan


def summary(df):
    out = []
    for cohort in ("post", "pre"):
        for k in FEATURES + REFERENCE:
            d = df[(df.cohort == cohort) & (df.kind == k)]
            if d.empty:
                continue
            w, c = d[d.wrong], d[~d.wrong]
            both = set(w.pdb) & set(c.pdb)
            out.append(dict(cohort=cohort, kind=k, n=len(d), structures=d.pdb.nunique(), missed=len(w),
                            missed_pct=100 * len(w) / len(d), med_missed=w.plddt.median(), med_kept=c.plddt.median(),
                            auc=auc(w.plddt, c.plddt), auc_within=auc_within(d[d.pdb.isin(both)]),
                            n_both=len(both)))
    return pd.DataFrame(out)


def calls(df):
    out = []
    for cohort in ("post", "pre"):
        for rare in ("syn base", "C2′-endo sugar", "non-g+ γ"):
            d = df[(df.cohort == cohort) & (df.kind == "call " + rare)]
            for lo, hi, lab in BANDS:
                s = d[(d.plddt >= lo) & (d.plddt < hi)]
                out.append(dict(cohort=cohort, kind=rare, band=lab, n=len(s),
                                confirmed=100 * (~s.wrong).mean() if len(s) else np.nan))
    return pd.DataFrame(out)


MISSED, KEPT = "#B5481A", "#2E6B27"
PRE_C, POST_C = "#1f77b4", "#E8743B"
ROWS = FEATURES + ["WC pair"]


def build(df):
    figstyle.apply("plain")
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    S = summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.95]))
    y = np.arange(len(ROWS))[::-1]

    # A: pLDDT_r of missed and reproduced elements (post-cutoff), median and interquartile range
    ax = axes[0]
    post = df[df.cohort == "post"]
    for yy, k in zip(y, ROWS):
        d = post[post.kind == k]
        for w, col, dy in ((True, MISSED, 0.14), (False, KEPT, -0.14)):
            v = 100 * d.plddt[d.wrong == w]
            if len(v):
                q1, md, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([q1, q3], [yy + dy] * 2, color=col, lw=1.2)
                ax.plot(md, yy + dy, "o", color=col, ms=3.5)
        s = S[(S.cohort == "post") & (S.kind == k)].iloc[0]
        ax.text(101, yy, f"{s.missed_pct:.0f}% of {int(s.n):,}", va="center", fontsize=5.8, color="0.3")
    ax.axhline(0.5, color="0.75", lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(ROWS)
    ax.set_xlim(0, 100); ax.set_xlabel("Per-nucleotide pLDDT$_r$ (post-cutoff)")
    ax.plot([], [], "o-", color=MISSED, ms=3.5, lw=1.2, label="missed by AF3")
    ax.plot([], [], "o-", color=KEPT, ms=3.5, lw=1.2, label="reproduced")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.2)
    ax.text(101, len(ROWS) - 0.35, "missed", fontsize=5.8, color="0.3")

    # B: AUC over all elements (filled) and within structures (open), post- and pre-cutoff
    ax = axes[1]
    for lo, hi, shade in ((0.4, 0.7, "0.94"), (0.7, 0.8, "0.88"), (0.8, 1.0, "0.82")):
        ax.axvspan(lo, hi, color=shade, lw=0, zorder=0)
    for xx, lab in ((0.55, "weak"), (0.75, "partial"), (0.9, "clear")):
        ax.text(xx, len(ROWS) - 0.3, lab, ha="center", fontsize=6, color="0.35")
    for cohort, col, dy in (("post", POST_C, 0.14), ("pre", PRE_C, -0.14)):
        for yy, k in zip(y, ROWS):
            s = S[(S.cohort == cohort) & (S.kind == k)]
            if s.empty:
                continue
            s = s.iloc[0]
            ax.plot([s.auc_within, s.auc], [yy + dy] * 2, color=col, lw=0.8)
            ax.plot(s.auc, yy + dy, "o", color=col, ms=3.5)
            ax.plot(s.auc_within, yy + dy, "o", mfc="white", mec=col, ms=3.5, mew=0.9)
    ax.axhline(0.5, color="0.75", lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.6, len(ROWS) - 0.1)
    ax.set_xlabel("AUC, missed ranked below reproduced")
    h = [plt.Line2D([], [], marker="o", ls="", color=POST_C, ms=3.5, label="post-cutoff"),
         plt.Line2D([], [], marker="o", ls="", color=PRE_C, ms=3.5, label="pre-cutoff"),
         plt.Line2D([], [], marker="o", ls="", color="0.3", ms=3.5, label="all elements"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=3.5, label="within structures")]
    ax.legend(handles=h, frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handletextpad=0.1, columnspacing=0.8)
    axes[0].set_ylim(ax.get_ylim())

    # C: share of AF3's non-generic states confirmed by the experiment, by AF3 pLDDT_r band
    ax = axes[2]
    C = calls(df)
    x = np.arange(len(BANDS))
    for (rare, mk) in (("syn base", "o"), ("C2′-endo sugar", "s"), ("non-g+ γ", "^")):
        for cohort, col, ls in (("post", POST_C, "-"), ("pre", PRE_C, "--")):
            c = C[(C.cohort == cohort) & (C.kind == rare)]
            ax.plot(x, c.confirmed, ls=ls, marker=mk, color=col, ms=3.5, lw=1.0,
                    label=rare if cohort == "post" else None, mfc=col if cohort == "post" else "white")
    ax.set_xticks(x); ax.set_xticklabels([b[2] for b in BANDS])
    ax.set_xlabel("pLDDT$_r$ of the AF3 nucleotide"); ax.set_ylabel("AF3 state confirmed (%)")
    ax.set_ylim(0, 100); ax.set_xlim(-0.3, len(BANDS) - 0.7)
    h = [plt.Line2D([], [], marker=mk, ls="", color="0.3", ms=3.5, label=lab)
         for lab, mk in (("syn base", "o"), ("C2′-endo sugar", "s"), ("non-g+ γ", "^"))]
    h += [plt.Line2D([], [], ls="-", color=POST_C, label="post-cutoff"),
          plt.Line2D([], [], ls="--", color=PRE_C, label="pre-cutoff")]
    ax.legend(handles=h, frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handlelength=1.6, columnspacing=0.8)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a, lab in zip(axes, "ABC"):
        a.text(-0.1 if a is not axes[0] else -0.42, 1.04, lab, transform=a.transAxes, fontsize=9, fontweight="bold",
               va="bottom")
    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/per_nucleotide_confidence.{ext}", dpi=400, bbox_inches="tight")
    print("saved", f"{OUT}/per_nucleotide_confidence.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/per_nucleotide_confidence.tsv"
    if os.path.exists(path) and "--recollect" not in sys.argv:
        df = pd.read_csv(path, sep="\t")
    else:
        df = collect(); df.to_csv(path, sep="\t", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(summary(df).round(3).to_string(index=False))
    print(calls(df).round(1).to_string(index=False))
    build(df)
