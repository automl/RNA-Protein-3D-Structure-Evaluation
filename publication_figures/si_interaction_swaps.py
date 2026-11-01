"""SI figure: which nucleotide-level interactions and conformers AF3 loses and which it adds (post-cutoff sets).

For every post-cutoff protein-RNA complex (a2021x + a2021nonX) and single-chain RNA (a2021_rnaOnly) the DSSR
annotations of the experimental structure and of the AF3 model are compared nucleotide by nucleotide.
Nucleotides are matched by aligning the two DSSR nucleotide sequences, so chain breaks and extra modified
nucleotides in the experimental file do not drop an entry. JSON built from DSSR web-server text
(evaluation/dssrTxt2json.py, e.g. 7VFT) has no stacks; such entries count for pairs and conformers only.

  lost   = present in the experiment, absent in AF3   (% of the experimental count)
  gained = present in AF3, absent in the experiment   (% of the experimental count)

Interactions: Watson-Crick/wobble pairs, non-canonical pairs, stacking contacts (consecutive nucleotides of each
DSSR stack).  Conformers: syn glycosidic bond, C2'-endo sugar pucker.

  A  protein-RNA complexes, pooled lost (down) and gained (up)
  B  single-chain RNAs, the same
  C  stacking contacts lost against gained per structure (the swap), examples of Fig. si_ex_pairing marked

Run from the repo root in the RNABench env:  python publication_figures/si_interaction_swaps.py
Output: results/plots/all_chains/publication_figures/main_final/si_interaction_swaps.{png,pdf} (+ per-structure TSV)
"""
import glob
import json
import os
import re
import sys
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figstyle.py is in the repo root
import figstyle

SETS = [("protein_rna", "data/all_data/a2021_post_cutoff/protein_rna/dssr"),       # published layout (fetch_data.sh)
        ("rna_only", "data/all_data/a2021_post_cutoff/rna_singlechain/dssr")]
OUT = "results/plots/all_chains/publication_figures/main_final"
KINDS = [("wc", "WC\npairs"), ("noncanonical", "Non-WC\npairs"), ("stack", "Stacks"),
         ("syn", "syn"), ("c2endo", "C2′-endo")]
LOST, GAINED = "#B5481A", "#2E6B27"          # one-side-only shades of the arc diagrams (experiment, AF3)
EXAMPLES = {"8S9V", "8FUK"}
EXCLUDE = {"7VFT", "8FEO", "8FEP"}   # two RNA strands in the PDB entry, removed from the single-chain sets (2026-10-21)


def entries(d):
    """pid -> (experimental json, AF3 json); aligned-RNA files preferred"""
    got = {}
    for f in sorted(glob.glob(f"{d}/*_pdb*_dssr.json")):
        m = re.match(r"(\w{4})_pdb(_aligned_rna)?_dssr\.json$", os.path.basename(f))
        g = f.replace("_pdb", "_af")
        if m and os.path.exists(g) and (m.group(1) not in got or m.group(2)):
            got[m.group(1)] = (f, g)
    return got


def features(x, keep):
    """interaction and conformer sets in nucleotide-index space; `keep` maps own index -> common index"""
    nts = x["nts"]; idx = {n["nt_id"]: keep[i] for i, n in enumerate(nts) if i in keep}
    st = set()
    for s in x.get("stacks", []):
        ids = [idx[i] for i in s["nts_long"].split(",") if i in idx]
        st |= {tuple(sorted(p)) for p in zip(ids, ids[1:])}
    wc, nc = set(), set()
    for p in x.get("pairs", []):
        if p["nt1"] in idx and p["nt2"] in idx:
            (wc if p["name"] in ("WC", "Wobble") else nc).add(tuple(sorted((idx[p["nt1"]], idx[p["nt2"]]))))
    syn = {keep[i] for i, n in enumerate(nts) if i in keep and glyco(n) == "syn"}
    c2 = {keep[i] for i, n in enumerate(nts) if i in keep and n.get("puckering") == "C2'-endo"}
    # JSON built from the DSSR web server's text (evaluation/dssrTxt2json.py) carries no stacks
    return dict(wc=wc, noncanonical=nc, stack=st if "stacks" in x else None, syn=syn, c2endo=c2)


def glyco(n):
    """DSSR glycosidic class; derived from chi (syn when |chi| < 90 deg) for converter-built JSON without the label"""
    if n.get("glyco_bond"):
        return n["glyco_bond"]
    chi = n.get("chi")
    return None if chi is None else ("syn" if abs(chi) < 90 else "anti")


def compare(f, g):
    T, A = json.load(open(f)), json.load(open(g))
    t = "".join(n["nt_code"].upper() for n in T["nts"]); a = "".join(n["nt_code"].upper() for n in A["nts"])
    kt, ka = {}, {}
    for blk in SequenceMatcher(None, t, a, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            kt[blk.a + k] = blk.a + k; ka[blk.b + k] = blk.a + k   # both into experimental index space
    ft, fa = features(T, kt), features(A, ka)
    row = dict(n_exp=len(t), n_matched=len(kt))
    for k in ft:
        if ft[k] is None or fa[k] is None:                      # stacks unavailable for this entry
            row[k + "_exp"] = row[k + "_lost"] = row[k + "_gained"] = np.nan
            continue
        row[k + "_exp"] = len(ft[k]); row[k + "_lost"] = len(ft[k] - fa[k]); row[k + "_gained"] = len(fa[k] - ft[k])
    return row


def collect():
    rows = []
    for label, d in SETS:
        for pid, (f, g) in entries(d).items():
            try:
                rows.append(dict(set=label, pid=pid.upper(), **compare(f, g)))
            except (KeyError, json.JSONDecodeError):
                continue
    df = pd.DataFrame(rows).drop_duplicates(["set", "pid"])
    df = df[~df.pid.isin(EXCLUDE)]
    return df[df.n_matched >= 0.9 * df.n_exp]                 # sequence match covers the RNA


def bars(ax, d):
    x = np.arange(len(KINDS))
    lost = [100 * d[k + "_lost"].sum() / max(d[k + "_exp"].sum(), 1) for k, _ in KINDS]
    gain = [100 * d[k + "_gained"].sum() / max(d[k + "_exp"].sum(), 1) for k, _ in KINDS]
    ax.bar(x, [-v for v in lost], 0.62, color=LOST, label="Lost by AF3")
    ax.bar(x, gain, 0.62, color=GAINED, label="Added by AF3")
    for xi, l, g in zip(x, lost, gain):
        ax.text(xi, -l - 3, f"{l:.0f}", ha="center", va="top", fontsize=6)
        ax.text(xi, g + 3, f"{g:.0f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in KINDS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-110, 75); ax.set_yticks([-100, -50, 0, 50]); ax.set_yticklabels(["100", "50", "0", "50"])
    ax.set_ylabel("% of experimental count")
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def build():
    figstyle.apply("plain")
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/si_interaction_swaps.tsv"
    if os.path.exists(path) and "--recollect" not in sys.argv:     # table shipped in analysis_inputs.zip
        df = pd.read_csv(path, sep="\t")
    else:
        df = collect(); df.to_csv(path, sep="\t", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for ax, (s, title) in zip(axes[:2], (("protein_rna", "Protein–RNA complexes"), ("rna_only", "Single-chain RNAs"))):
        d = df[df.set == s]; bars(ax, d)
        ax.set_title(f"{title} (n = {len(d)})", fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.36, 1.08))
    axes[1].set_ylabel("")
    ax = axes[2]
    for s, mk, col in (("protein_rna", "o", "0.35"), ("rna_only", "s", "0.7")):
        d = df[(df.set == s) & (df.stack_exp > 0)]
        ax.scatter(100 * d.stack_lost / d.stack_exp, (100 * d.stack_gained / d.stack_exp).clip(upper=150), s=7, marker=mk,
                   color=col, lw=0, alpha=0.8, label="Protein–RNA" if s == "protein_rna" else "Single-chain")
    ex = df[df.pid.isin(EXAMPLES) & (df.set == "protein_rna")]
    ex_x, ex_y = 100 * ex.stack_lost / ex.stack_exp, 100 * ex.stack_gained / ex.stack_exp
    ax.scatter(ex_x, ex_y, s=24, facecolor="none", edgecolor="black", lw=0.9)
    for (_, r), xx, yy, dy in zip(ex.iterrows(), ex_x, ex_y, (8, -12)):
        ax.annotate(r.pid, (xx, yy), xytext=(6, dy), textcoords="offset points", fontsize=6)
    ax.plot([0, 100], [0, 100], color="0.5", lw=0.7, ls="--")
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 153); ax.set_yticks([0, 50, 100, 150]); ax.set_yticklabels(["0", "50", "100", "≥150"])
    ax.set_xlabel("Stacks lost (% of experimental)"); ax.set_ylabel("Stacks added (% of experimental)")
    ax.legend(frameon=False, fontsize=6, loc="upper right", handletextpad=0.2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for ax, lab in zip(axes, "ABC"):
        ax.text(-0.18, 1.06, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/si_interaction_swaps.{ext}", dpi=400, bbox_inches="tight")
    print("saved", f"{OUT}/si_interaction_swaps.png")

    # numbers quoted in the text
    for s, d in df.groupby("set"):
        print(f"\n{s}: n = {len(d)}")
        for k, _ in KINDS:
            e, l, g = (int(d[k + c].sum()) for c in ("_exp", "_lost", "_gained"))
            print(f"  {k:13s} exp {e:6d} lost {l:6d} ({100 * l / max(e, 1):.0f}%) added {g:6d} ({100 * g / max(e, 1):.0f}%)")
        st = d[d.stack_exp > 0]
        print(f"  structures adding more stacks than they lose: {(st.stack_gained > st.stack_lost).sum()}/{len(st)}"
              f"; with both a loss and an addition: {((st.stack_lost > 0) & (st.stack_gained > 0)).sum()}")
    print(df[df.pid.isin(EXAMPLES)][["set", "pid", "stack_exp", "stack_lost", "stack_gained"]].to_string(index=False))


if __name__ == "__main__":
    build()
