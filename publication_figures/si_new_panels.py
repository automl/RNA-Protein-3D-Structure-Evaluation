"""
Supplementary figures for main_final.tex built from the panels designed for the main figures and
not used there (the main text keeps the original Figs 1, 3 and 4):

  si_twin_gap            paired LDDT_r difference against the closest pre-cutoff twin, by structural
                         similarity (and sequence twin for RNAs < 20 nt)    (fig1_postpre_twin_gap.py)
  si_sampling_recovery   failed default-model interfaces recovered by any of the 15 sampled models,
                         by placement class                        (fig3_interface_failure_recovery.py)
  si_positional          positional test of the motif-bearing footprint and the share of entries
                         whose best AF3 footprint lies on the native site, per cohort
                                                                    (fig4_motif_footprint.py)

Writes results/plots/all_chains/publication_figures/main_final/<name>.{pdf,png} and copies them to
2025_NAR_AF3_Evaluation/figures2/.
Usage (RNABench env):  python publication_figures/si_new_panels.py
"""

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import figstyle
import fig1_postpre_twin_gap as f1
import fig3_interface_failure_recovery as f3
import fig4_motif_footprint as f4
from plots.publication_figure import PUB_RC

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures", "main_final")
PAPER_FIGS = os.path.join(_REPO_ROOT, "2025_NAR_AF3_Evaluation", "figures2")


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    for ext in ("pdf", "png") if os.path.isdir(PAPER_FIGS) else ():   # copy for the manuscript repo, when checked out here
        shutil.copy2(f"{base}.{ext}", os.path.join(PAPER_FIGS, f"{name}.{ext}"))
    print("Saved:", base + ".png")


def twin_gap():
    diffs = f1.paired_differences()
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(f1.GEN_W, 6.0))
        f1.panel_twin_gap(ax, diffs)
        _save(fig, "si_twin_gap")


def sampling_recovery():
    figstyle.apply("plain")
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_2COL * 0.75, 2.2))
    f3.panel_recovery(ax, f3._recovery())
    _save(fig, "si_sampling_recovery")


def positional():
    figstyle.apply("plain")
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL * 0.75, 2.8))
    gs = fig.add_gridspec(1, 2, wspace=0.55, width_ratios=[1.0, 0.45])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    f4.panel_positional(axA)
    f4.panel_same_site(axB)
    for ax, ltr in ((axA, "A"), (axB, "B")):
        figstyle.panel_letter(ax, ltr, x=-0.30 if ax is axA else -0.55, y=1.04)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=5, mfc=col, mec="black", mew=0.4, label=lab)
               for col, lab in f4.cam.CLASS_STYLE.values()]
    handles += [Line2D([0], [0], marker="o", ls="", ms=5, mfc="#999999", mec="black", mew=0.4, label="pre-cutoff"),
                Line2D([0], [0], marker="^", ls="", ms=5.5, mfc="#999999", mec="black", mew=0.4, label="post-cutoff")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               fontsize=figstyle.FONT_LEGEND, handletextpad=0.3, columnspacing=1.0)
    _save(fig, "si_positional")


if __name__ == "__main__":
    twin_gap()
    sampling_recovery()
    positional()
