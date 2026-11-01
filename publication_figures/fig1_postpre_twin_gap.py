"""
Main figure 1 of main_final.tex: the pre- versus post-cutoff accuracy gap is large and specific to
the RNA, and it follows the novelty of the RNA fold.

  A  split violins, pre- (b2021) vs post-cutoff (a2021), for the protein–RNA complexes, the
     single-chain RNAs and the multi-chain RNA complexes (same metrics, loaders and colours as
     panel A of post_pre_comparison_figure.py build_combined(include_multimeric=True))
  B  paired difference in LDDT_r between every post-cutoff target and its closest pre-cutoff twin of
     the same system (median, bootstrap 95% CI, n per bar), by structural similarity of the longest
     RNA chain (US-align TM-score to the pre-cutoff RNAs: >= 0.7 redundant, 0.5–0.7 similar fold,
     < 0.5 novel fold; evaluation/structural_similarity_to_pre.py) and, for RNAs shorter than 20 nt
     whose TM-score is not interpretable, against the sequence twin
     (evaluation/postpre_matched_similarity.py)

Writes results/plots/all_chains/publication_figures/main_final/fig1_postpre_twin_gap.{pdf,png} and
copies them to 2025_NAR_AF3_Evaluation/figures2/.
Usage (RNABench env):  python publication_figures/fig1_postpre_twin_gap.py
"""

import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import post_pre_comparison_figure as ppc
from plots.publication_figure import PUB_RC, onpage_to_gen, style_axis
from evaluation.postpre_matched_similarity import SYSTEMS, boot_median_ci, load
from evaluation.structural_similarity_to_pre import MIN_NT_TM, TM_BINS

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures", "main_final")
PAPER_FIGS = os.path.join(_REPO_ROOT, "2025_NAR_AF3_Evaluation", "figures2")
STRUCT_DIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "structural_similarity")
SEQ_DIR = next(d for d in (os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "postpre_matched"),
                           os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "postpre_matched"))
               if os.path.isdir(d))
GEN_W = 17.0
SYSTEM_STYLE = [("protein_rna", "Protein–RNA", ppc.PR_COL),
                ("rna_only", "Single-chain RNA", ppc.MONO_COL),
                ("rna_multichain", "Multi-chain RNA", ppc.MULTI_COL)]
BIN_LABELS = {"TM>=0.7": "TM ≥ 0.7\nredundant fold", "TM 0.5-0.7": "TM 0.5–0.7\nsimilar fold",
              "TM<0.5": "TM < 0.5\nnovel fold", "short": f"< {MIN_NT_TM} nt\nsequence twin"}
BINS = [name for _lo, _hi, name in TM_BINS] + ["short"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def paired_differences():
    """{system: {bin: array of post - twin LDDT_r}}"""
    out = {}
    for system, cfg in SYSTEMS.items():
        pre = {r["id"]: r for r in load(cfg["pre"], cfg["table"], cfg["protein"])}
        d = {b: [] for b in BINS}
        struct = list(csv.DictReader(open(os.path.join(STRUCT_DIR, f"best_structural_twin_{system}.tsv")),
                                     delimiter="\t"))
        short_ids = set()
        for r in struct:
            n, tm = _f(r["rna_len"]), _f(r["best_tm_same_system"])
            if np.isfinite(n) and n < MIN_NT_TM:
                short_ids.add(r["post_id"])
                continue
            twin = pre.get(r["structural_twin_same_system"])
            post_l = _f(r["post_RNA_LDDT"])
            if not np.isfinite(tm) or twin is None or twin["lddt"] is None or not np.isfinite(post_l):
                continue
            for lo, hi, name in TM_BINS:
                if lo <= tm < hi:
                    d[name].append(post_l - twin["lddt"])
        for r in csv.DictReader(open(os.path.join(SEQ_DIR, f"matched_pairs_{system}.tsv")), delimiter="\t"):
            if r["post_id"] in short_ids:
                a, b = _f(r["post_RNA_LDDT"]), _f(r["pre_RNA_LDDT"])
                if np.isfinite(a) and np.isfinite(b):
                    d["short"].append(a - b)
        out[system] = {b: np.array(v) for b, v in d.items()}
    return out


def panel_violins(axA):
    """Panel A of post_pre_comparison_figure.build_combined(include_multimeric=True)."""
    pr = pd.concat([ppc._load_pr(os.path.join(ppc.PR_DIR, "b2021.db"), ppc.PRE),
                    ppc._load_pr(os.path.join(ppc.PR_DIR, "a2021.db"), ppc.POST)], ignore_index=True)
    rna = pd.concat([ppc._load_rna(os.path.join(ppc.RNA_DIR, "b2021_rna.db"), ppc.PRE),
                     ppc._load_rna(os.path.join(ppc.RNA_DIR, "a2021_rna.db"), ppc.POST)], ignore_index=True)
    multi = pd.concat([ppc._load_rna(os.path.join(ppc.MULTI_DIR, "b2021_rna_rna.db"), ppc.PRE),
                       ppc._load_rna(os.path.join(ppc.MULTI_DIR, "a2021_rna_rna.db"), ppc.POST)], ignore_index=True)

    def _both(src, c):
        return (src.loc[src.database == ppc.PRE, c].notna().any()
                and src.loc[src.database == ppc.POST, c].notna().any())
    pr_cols = ["TM_prot", "pTM_prot", "LDDT_prot", "LDDT_c", "DockQ", "DockQ_Nr", "ipTM_rna", "LDDT_r",
               "pLDDT_rna", "TM_r", "pTM_rna"]
    rna_cols = ["LDDT_r", "pLDDT_rna", "TM_r", "pTM_rna"]
    items = [(f"PR::{c}", pr, c) for c in pr_cols if _both(pr, c)]
    b1 = len(items)
    items += [(f"RNA::{c}", rna, c) for c in rna_cols if _both(rna, c)]
    b2 = len(items)
    items += [(f"MULTI::{c}", multi, c) for c in rna_cols if _both(multi, c)]
    n = len(items)
    long = pd.concat([src[["database", c]].dropna(subset=[c]).rename(columns={c: "value"}).assign(xkey=k)
                      for k, src, c in items], ignore_index=True)
    sns.violinplot(data=long, x="xkey", y="value", hue="database", order=[k for k, _, _ in items],
                   hue_order=ppc.HUE, palette=ppc.PAL, split=True, inner="box", cut=0,
                   density_norm="width", linewidth=0.8, ax=axA)
    axA.set_xlabel("")
    axA.set_ylabel("Score")
    axA.set_ylim(0, 1.16)
    blocks = [(0, b1, "Protein–RNA", ppc.PR_COL), (b1, b2, "Single-chain RNA", ppc.MONO_COL),
              (b2, n, "Multi-chain RNA", ppc.MULTI_COL)]
    for _lo, hi, _lab, _c in blocks[:-1]:
        axA.axvline(hi - 0.5, color="0.4", ls="--", lw=1.6)
    for lo, hi, lab, c in blocks:
        axA.text((lo + hi - 1) / 2, 1.05, lab, ha="center", va="bottom", fontsize=onpage_to_gen(7.5, GEN_W),
                 color=c, fontweight="bold")
    axA.set_xticks(range(n))
    axA.set_xticklabels([ppc.DISP[c] for _, _, c in items], rotation=40, ha="right", rotation_mode="anchor")
    h, l = axA.get_legend_handles_labels()
    axA.legend(h, ["Pre-cutoff (b2021)", "Post-cutoff (a2021)"], title="", loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
    style_axis(axA, grid_axis="y")


def panel_twin_gap(axB, diffs):
    x = np.arange(len(BINS))
    w = 0.26
    ebk = dict(ecolor="0.25", elinewidth=1.1, capsize=3)
    fs = onpage_to_gen(6.5, GEN_W)
    for j, (system, lab, col) in enumerate(SYSTEM_STYLE):
        xs = x + (j - 1) * w
        for xi, b in zip(xs, BINS):
            v = diffs[system][b]
            if len(v) == 0:
                continue
            med = float(np.median(v))
            lo, hi = boot_median_ci(v)
            err = [[med - lo], [hi - med]] if np.isfinite(lo) else None
            axB.bar(xi, med, w * 0.92, color=col, edgecolor="black", linewidth=0.4,
                    yerr=err, error_kw=ebk, label=lab if b == BINS[0] else None)
            axB.text(xi, 0.008, f"n={len(v)}", ha="center", va="bottom", fontsize=fs, color="0.25")
    axB.axhline(0, color="0.4", lw=0.8)
    axB.set_ylim(top=0.06)
    axB.axvline(2.5, color="0.4", ls="--", lw=1.2)
    axB.set_xticks(x)
    axB.set_xticklabels([BIN_LABELS[b] for b in BINS])
    axB.set_ylabel("Paired $\\Delta$LDDT$_r$\n(post-cutoff $-$ closest pre-cutoff twin)")
    axB.legend(loc="lower left", frameon=False, ncol=1)
    style_axis(axB, grid_axis="y")


def build():
    diffs = paired_differences()
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(GEN_W, 13.0))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1.0], hspace=0.42)
        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1])
        panel_violins(axA)
        panel_twin_gap(axB, diffs)
        ppc._letters([(axA, "A"), (axB, "B")], GEN_W)
        os.makedirs(OUTDIR, exist_ok=True)
        base = os.path.join(OUTDIR, "fig1_postpre_twin_gap")
        for ext in ("pdf", "png"):
            fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)
    for ext in ("pdf", "png") if os.path.isdir(PAPER_FIGS) else ():   # copy for the manuscript repo, when checked out here
        shutil.copy2(f"{base}.{ext}", os.path.join(PAPER_FIGS, f"fig1_postpre_twin_gap.{ext}"))
    for system, _lab, _c in SYSTEM_STYLE:
        print(system, {b: (len(v), round(float(np.median(v)), 3) if len(v) else None) for b, v in diffs[system].items()})
    print("Saved:", base + ".png")


if __name__ == "__main__":
    build()
