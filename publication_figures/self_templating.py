"""
Self-templating and family-representative collapse in AF3 template selection.

Even though every pre-cutoff target's own PDB entry predates the training cutoff
(2021-09-30) and is therefore available as a template, AF3 rarely templates onto
it. Instead it routes each protein/RBP family onto a small set of canonical
representative structures, reused across many targets. This script quantifies:

  A  self-templating rate by family size (pre-cutoff), with the post-cutoff
     control (own entry unavailable -> 0%);
  B  concentration of template usage (Lorenz curve + Gini), split by channel
     (protein-only vs RBP/complex templates);
  C  the largest families, each dominated by one representative that also
     captures the post-cutoff members.

Families are operational template-sharing clusters (targets connected if they
share >=1 selected template), a proxy used because the deposited ProteinFamily
annotation is a messy per-chain/per-domain list.

Sources: data/b2021_merged/b2021.db and data/a2021_merged/merged.db
(pred_protein_rna template columns joined to exp_protein_rna Deposited_date).

Usage:  conda run -n RNABench python publication_figures/self_templating.py
"""

import os
import sys
import ast
import sqlite3
from collections import Counter, defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, style_axis
from evaluation.excluded_targets import POST_EXCLUDED, PRE_EXCLUDED
from plots.plotCreator import PH_RANGE_COLORS

CUTOFF = "2021-09-30"
# the pre- and post-cutoff pools of the published bundle (other_analyses/self_templating/)
_POOLS = os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases", "other_analyses", "self_templating")
DBS = [os.path.join(_POOLS, "protein_rna_pre_cutoff_pool.db"),
       os.path.join(_POOLS, "protein_rna_post_cutoff_pool.db")]

# Human-readable family labels for the recurrent representatives (verified via RCSB).
# Only entries confirmed by hand are labelled; others show the bare PDB id.
REP_FAMILY = {
    "6QNW": "Influenza A polymerase", "6R5K": "Pan2-Pan3 deadenylase",
    "3GVO": "Pumilio PUF domain", "6P80": "CdnC (cGAS/DncV-like)",
    "1MST": "MS2 capsid", "6IFN": "CRISPR Csm (III-A)",
    "6NUR": "SARS-CoV nsp12 RdRp", "7JWB": "SARS-CoV-2 spike",
}

COL_PRE = PH_RANGE_COLORS["5-6"]    # blue
COL_POST = "0.65"                    # grey (post-cutoff control)
COL_PROT = PH_RANGE_COLORS["6-7"]   # orange - protein-only templates
COL_RBP = PH_RANGE_COLORS["7-8"]    # green  - RBP/complex templates


def _parse(s):
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [str(x).upper() for x in v] if isinstance(v, list) else []


def load():
    """PDBId -> dict(P, R, T, era). Deduped across both dbs, dated by exp table."""
    rec = {}
    for path in DBS:
        con = sqlite3.connect(path)
        q = ("SELECT p.exp_db_id, p.Protein_PDB_Templates, p.RBP_PDB_Templates, "
             "e.Deposited_date FROM pred_protein_rna p "
             "JOIN exp_protein_rna e ON e.PDBId = p.exp_db_id")
        for eid, pt, rbp, dd in con.execute(q):
            eid = eid.upper()
            if eid in rec:
                continue
            if pt is None and rbp is None:        # no template metadata
                continue
            P, R = _parse(pt), _parse(rbp)
            era = ("pre" if (dd and dd != "None" and dd <= CUTOFF)
                   else "post" if (dd and dd != "None") else "unk")
            # by AF3 training membership (release date): 7PMQ was deposited before the cutoff but
            # released in 2022; 6FQL (released 2018) is pre-cutoff (evaluation/excluded_targets.py)
            if eid in PRE_EXCLUDED:
                era = "post"
            elif eid in POST_EXCLUDED:
                era = "pre"
            rec[eid] = dict(P=P, R=R, T=set(P) | set(R), era=era)
        con.close()
    return rec


def families(rec):
    """Union-find over all targets sharing >=1 template -> {root: [members]}."""
    parent = {e: e for e in rec}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    t2g = defaultdict(list)
    for e, r in rec.items():
        for t in r["T"]:
            t2g[t].append(e)
    for members in t2g.values():
        for other in members[1:]:
            ra, rb = find(members[0]), find(other)
            if ra != rb:
                parent[ra] = rb
    comp = defaultdict(list)
    for e in rec:
        comp[find(e)].append(e)
    return list(comp.values())


# --------------------------------------------------------------------------- #
def compute():
    rec = load()
    comp = families(rec)

    # --- A: self-rate by family-size bin, pre vs post ---
    BINS = [("singleton", lambda n: n == 1),
            ("2-5", lambda n: 2 <= n <= 5),
            ("6-20", lambda n: 6 <= n <= 20),
            ("21+", lambda n: n >= 21)]
    binmembers = {lab: [] for lab, _ in BINS}
    for members in comp:
        n = len(members)
        for lab, f in BINS:
            if f(n):
                binmembers[lab] += members
                break
    A = {}
    for lab, mem in binmembers.items():
        for era in ("pre", "post"):
            sub = [e for e in mem if rec[e]["era"] == era]
            s = sum(1 for e in sub if e in rec[e]["T"])
            A[(lab, era)] = (s, len(sub))

    # --- B: usage concentration per channel ---
    useP, useR = Counter(), Counter()
    for r in rec.values():
        for t in set(r["P"]):
            useP[t] += 1
        for t in set(r["R"]):
            useR[t] += 1

    def lorenz(counter):
        v = np.sort(np.array(list(counter.values()), float))
        cx = np.insert(np.cumsum(v) / v.sum(), 0, 0)
        px = np.linspace(0, 1, len(cx))
        gini = (2 * np.sum((np.arange(1, len(v) + 1)) * v)
                / (len(v) * v.sum())) - (len(v) + 1) / len(v)
        return px, cx, gini

    B = {"protein": lorenz(useP), "rbp": lorenz(useR)}

    # --- C: largest families, representative dominance ---
    C = []
    for members in sorted(comp, key=len, reverse=True)[:8]:
        cnt = Counter()
        for e in members:
            for t in rec[e]["T"]:
                cnt[t] += 1
        rep, rn = cnt.most_common(1)[0]
        pre = [e for e in members if rec[e]["era"] == "pre"]
        post = [e for e in members if rec[e]["era"] == "post"]
        channel = "rbp" if useR.get(rep, 0) >= useP.get(rep, 0) else "protein"
        C.append(dict(rep=rep, size=len(members), cover=rn,
                      npre=len(pre), npost=len(post),
                      self_pre=sum(1 for e in pre if e in rec[e]["T"]),
                      post_rep=sum(1 for e in post if rep in rec[e]["T"]),
                      channel=channel))

    # overall pre/post self-rate
    pre_all = [e for e, r in rec.items() if r["era"] == "pre"]
    post_all = [e for e, r in rec.items() if r["era"] == "post"]
    overall = dict(
        pre=(sum(1 for e in pre_all if e in rec[e]["T"]), len(pre_all)),
        post=(sum(1 for e in post_all if e in rec[e]["T"]), len(post_all)))
    return dict(A=A, B=B, C=C, BINS=[b[0] for b in BINS],
                overall=overall, n_fam=len(comp), n_tg=len(rec),
                useP=useP, useR=useR)


# --------------------------------------------------------------------------- #
def build(res):
    A, B, C, BIN_LABELS = res["A"], res["B"], res["C"], res["BINS"]

    def drawA(ax):
        x = np.arange(len(BIN_LABELS))
        w = 0.38
        pre = [100 * A[(b, "pre")][0] / max(A[(b, "pre")][1], 1) for b in BIN_LABELS]
        post = [100 * A[(b, "post")][0] / max(A[(b, "post")][1], 1) for b in BIN_LABELS]
        ax.bar(x - w / 2, pre, w, color=COL_PRE, edgecolor="black", linewidth=0.3,
               label="pre-cutoff (own entry available)")
        ax.bar(x + w / 2, post, w, color=COL_POST, edgecolor="black", linewidth=0.3,
               label="post-cutoff (unavailable)")
        for xi, b in zip(x, BIN_LABELS):
            s, n = A[(b, "pre")]
            ax.text(xi - w / 2, 100 * s / max(n, 1) + 0.6, f"{s}/{n}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_xlabel("family size (targets sharing templates)")
        ax.set_ylabel("self-templating rate (%)")
        ax.set_ylim(0, max(pre) * 1.25)
        ax.legend(loc="upper right")
        style_axis(ax, grid_axis="y")

    def drawB(ax):
        for key, col, lab in (("protein", COL_PROT, "protein-only templates"),
                              ("rbp", COL_RBP, "RBP (complex) templates")):
            px, cx, gini = B[key]
            ax.plot(px, cx, color=col, lw=1.6,
                    label=f"{lab} (Gini {gini:.2f})")
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", label="equal use")
        ax.set_xlabel("cumulative fraction of template PDB entries")
        ax.set_ylabel("cumulative fraction of template uses")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        style_axis(ax, grid_axis="both")

    def drawC(ax):
        rows = C[::-1]                      # largest at top
        y = np.arange(len(rows))
        cover = [100 * r["cover"] / r["size"] for r in rows]
        cols = [COL_RBP if r["channel"] == "rbp" else COL_PROT for r in rows]
        ax.barh(y, cover, color=cols, edgecolor="black", linewidth=0.3)
        labels = []
        for r in rows:
            fam = REP_FAMILY.get(r["rep"], "")
            labels.append(f"{r['rep']}" + (f"\n{fam}" if fam else ""))
        for yi, r in zip(y, rows):
            ax.text(2, yi, f"n={r['size']}  self={r['self_pre']}",
                    va="center", ha="left", fontsize=6.2, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.8)
        ax.set_xlabel("family members using the representative (%)")
        ax.set_xlim(0, 100)
        ax.axvline(100, color="0.7", lw=0.6, ls=":")
        style_axis(ax, grid_axis="x")
        # legend proxies
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=COL_PROT, edgecolor="black",
                                 label="protein-tmpl rep."),
                           Patch(facecolor=COL_RBP, edgecolor="black",
                                 label="RBP-tmpl rep.")],
                  loc="lower right", fontsize=6.5)

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    fig = PublicationFigure(ncols=3, panel_size=(4.6, 4.2), outdir=outdir)
    fig.add_panel(drawA, title=None)
    fig.add_panel(drawB, title=None)
    fig.add_panel(drawC, title=None)
    return fig.save("self_templating")


def _report(res):
    print("\n=== numbers for SI table ===")
    o = res["overall"]
    print(f"targets (dated, w/ template metadata): {res['n_tg']}  families: {res['n_fam']}")
    print(f"overall self: pre {o['pre'][0]}/{o['pre'][1]} "
          f"({100*o['pre'][0]/o['pre'][1]:.1f}%)  "
          f"post {o['post'][0]}/{o['post'][1]} ({100*o['post'][0]/max(o['post'][1],1):.1f}%)")
    print("self-rate by family size (pre / post):")
    for b in res["BINS"]:
        sp, np_ = res["A"][(b, "pre")]
        so, no = res["A"][(b, "post")]
        print(f"  {b:10s}: pre {sp}/{np_} ({100*sp/max(np_,1):.1f}%)  "
              f"post {so}/{no} ({100*so/max(no,1):.1f}%)")
    print("largest families (rep, size, coverage, pre-self, post-using-rep, channel):")
    for r in res["C"]:
        print(f"  {r['rep']} {REP_FAMILY.get(r['rep'],''):24s} size {r['size']:3d}  "
              f"cover {r['cover']}/{r['size']} ({100*r['cover']/r['size']:.0f}%)  "
              f"pre-self {r['self_pre']}  post-rep {r['post_rep']}/{r['npost']}  {r['channel']}")


if __name__ == "__main__":
    res = compute()
    _report(res)
    build(res)
