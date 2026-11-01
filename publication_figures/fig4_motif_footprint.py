"""
Main figure 4 of main_final.tex: AF3 places the motif-bearing footprint of the native RNA where
experiments see it. Panels of af3_clip_atlas_motif_figure.py are imported, not modified.

  A  motif-specificity z, AF3 vs deposited structure, by RBP recognition class  (old A)
  B  positional test (clip/results/positional.tsv, clip/scripts/positional.py): per entry, the
     share of the experimentally motif-bearing positions that AF3 contacts against the share of
     all RNA positions it contacts (chance level); points above the diagonal are recovered more
     often than chance. The box gives, per cohort, how often AF3's own best-scoring footprint
     lies on the native site
  C  specificity by recognition class and cohort                                 (old B)
  SI motif provenance, in vitro vs in vivo (old C) and per-gene motif table (old D)

Writes results/plots/all_chains/publication_figures/main_final/{fig4_motif_footprint,
si_motif_provenance}.{pdf,png} and copies them to 2025_NAR_AF3_Evaluation/figures2/.
Usage (RNABench env):  python publication_figures/fig4_motif_footprint.py
"""

import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import figstyle
import af3_clip_atlas_motif_figure as cam
from plots.publication_figure import style_axis

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures", "main_final")
PAPER_FIGS = os.path.join(_REPO_ROOT, "2025_NAR_AF3_Evaluation", "figures2")
POSITIONAL = os.path.join(_REPO_ROOT, "clip", "results", "positional.tsv")


def _positional():
    rows = list(csv.DictReader(open(POSITIONAL), delimiter="\t"))
    for r in rows:
        for k in ("recall_any", "baseline", "recall_pair", "same_site"):
            r[k] = cam._fl(r[k])
    return rows


def panel_positional(ax):
    rows = _positional()
    ok = [r for r in rows if np.isfinite(r["recall_any"]) and np.isfinite(r["baseline"])]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.9, zorder=1)
    rng = np.random.default_rng(0)
    for c, (col, _lab) in cam.CLASS_STYLE.items():
        for coh, mk, sz in (("pre", "o", 14), ("post", "^", 20)):
            s = [r for r in ok if r["rbp_class"] == c and r["cohort"] == coh]
            if not s:
                continue
            jit = lambda n: (rng.random(n) - 0.5) * 0.02          # many entries sit at 1.0
            ax.scatter(np.array([r["baseline"] for r in s]) + jit(len(s)),
                       np.array([r["recall_any"] for r in s]) + jit(len(s)),
                       s=sz, marker=mk, color=col, edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3)
    w = stats.wilcoxon([r["recall_any"] for r in ok], [r["baseline"] for r in ok])
    ax.text(0.03, 0.03, f"$n$ = {len(ok)}\nWilcoxon $P$ = {cam._pfmt(w.pvalue)}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=cam.FL, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5, alpha=0.95))
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("All RNA positions contacted by AF3")
    ax.set_ylabel("Motif-bearing positions\ncontacted by AF3")
    style_axis(ax, grid_axis="both")


def panel_same_site(ax):
    """Share of entries whose AF3 best-scoring footprint lies on the native site, per cohort."""
    rows = _positional()
    xs = np.arange(2)
    for xi, (coh, lab) in zip(xs, (("pre", "Pre-\ncutoff"), ("post", "Post-\ncutoff"))):
        ss = [r["same_site"] for r in rows if r["cohort"] == coh and np.isfinite(r["same_site"])]
        ax.bar(xi, 100 * np.mean(ss), 0.6, color=cam.PRED_C, edgecolor="black", linewidth=0.4,
               hatch="//" if coh == "post" else "")
        ax.text(xi, 100 * np.mean(ss) + 2, f"{int(sum(ss))}/{len(ss)}", ha="center", va="bottom", fontsize=cam.FL)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Pre-\ncutoff", "Post-\ncutoff"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Best AF3 RNA motif on\nnative site (% entries)")
    style_axis(ax, grid_axis="y")


def _copy(base):
    for ext in ("pdf", "png") if os.path.isdir(PAPER_FIGS) else ():   # copy for the manuscript repo, when checked out here
        shutil.copy2(f"{base}.{ext}", os.path.join(PAPER_FIGS, os.path.basename(base) + f".{ext}"))


def build():
    figstyle.apply("plain")
    os.makedirs(OUTDIR, exist_ok=True)
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.55, width_ratios=[1.0, 1.0, 0.45],
                          height_ratios=[1.0, 0.8])
    axA, axB, axB2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    axC = fig.add_subplot(gs[1, :2])
    cam._panel_scatter(axA)
    panel_positional(axB)
    panel_same_site(axB2)
    cam._panel_by_class(axC)
    leg = axC.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1.0))
        leg._loc = 2                                   # upper left, outside the axes on the right
    for ax, ltr, x in ((axA, "A", -0.24), (axB, "B", -0.30), (axC, "C", -0.11)):
        figstyle.panel_letter(ax, ltr, x=x, y=1.04)
    base = os.path.join(OUTDIR, "fig4_motif_footprint")
    figstyle.save(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    _copy(base)
    print("Saved:", base + ".png")

    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.62, width_ratios=[0.8, 1.2])
    axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cam._panel_provenance(axC)
    cam._panel_alignment(axD)
    for ax, ltr, x in ((axC, "A", -0.16), (axD, "B", -0.36)):
        figstyle.panel_letter(ax, ltr, x=x, y=1.06)
    base = os.path.join(OUTDIR, "si_motif_provenance")
    figstyle.save(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    _copy(base)
    print("Saved:", base + ".png")


if __name__ == "__main__":
    build()
