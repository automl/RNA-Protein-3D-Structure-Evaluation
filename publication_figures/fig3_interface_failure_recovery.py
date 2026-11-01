"""
Main figure 3 of main_final.tex: how AF3 protein–RNA interfaces fail, and the SI figure on whether sampling
recovers them.

Main figure (fig3_interface_failure): the figure of interface_failure_modes_figure.py, panels
  A  interface quality (DockQ_Nr good / intermediate / poor) within each fold class i–iv
  B  placement class of each RNA chain with its main protein partner (the protein with the largest native
     interface), pre- vs post-cutoff, and the protein assembly and RNA fold of the RNA chains whose interface is
     incorrect (placement_per_rna_chain.py)
  C  whole-RNA translation vs rotation for the failed main-partner interfaces without native protein contacts
  D  Spearman heatmap of the placement-class shares of a complex's RNA chains

SI figure (si_interface_sampling_recovery): share of the interfaces that fail in the default model (seed 1,
model 0) for which any of the 15 sampled models (3 seeds x 5 models) reaches DockQ >= 0.23, by placement class
(evaluation/seed_sample_interfaces.py -> per_interface.tsv; per interface, as sampled).

Writes results/plots/all_chains/publication_figures/main_final/{fig3_interface_failure,
si_interface_sampling_recovery}.{pdf,png} and copies them to 2025_NAR_AF3_Evaluation/figures2/.
Usage (RNABench env):  python publication_figures/fig3_interface_failure_recovery.py [--no-si]
"""

import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import figstyle
import interface_failure_modes_figure as ifm

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures", "main_final")
PAPER_FIGS = os.path.join(_REPO_ROOT, "2025_NAR_AF3_Evaluation", "figures2")
SEED_TSV = next(p for p in (
    os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "seed_sample_interfaces", "per_interface.tsv"),
    os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "seed_sample_interfaces", "per_interface.tsv"))
    if os.path.isfile(p))
ACCEPT = 0.23
FS, FSS, FSL = figstyle.FONT_BASE, figstyle.FONT_SMALL, figstyle.FONT_LEGEND


def _recovery():
    """Per placement class: (failed in default model, recovered by any of 15)."""
    out = {}
    for r in csv.DictReader(open(SEED_TSV), delimiter="\t"):
        top, best = float(r["top"]), float(r["best15"])
        if not np.isfinite(top) or top >= ACCEPT or not r["placement"]:
            continue
        key = ifm.MERGE.get(r["placement"], r["placement"])
        n, k = out.get(key, (0, 0))
        out[key] = (n + 1, k + (best >= ACCEPT))
    return out


def panel_recovery(ax, rec):
    keys = [k for k, _, _ in ifm.CLASSES if k in rec]
    y = np.arange(len(keys))[::-1]
    for yi, key in zip(y, keys):
        n, k = rec[key]
        ax.barh(yi, 100.0, 0.62, color="0.92", edgecolor="none")
        ax.barh(yi, 100.0 * k / n, 0.62, color=ifm.COLOR[key], edgecolor="white", lw=0.4)
        ax.text(101.5, yi, f"{k}/{n}", va="center", ha="left", fontsize=FSS)
    ax.set_yticks(y)
    ax.set_yticklabels([ifm.LABEL[k] for k in keys], fontsize=FSS)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Failed interfaces with an acceptable\npose in any of 15 models (%)")
    ax.grid(axis="x", color="0.9", lw=figstyle.GRIDWIDTH)
    ax.set_axisbelow(True)


def _copy(base):
    for ext in ("pdf", "png") if os.path.isdir(PAPER_FIGS) else ():   # copy for the manuscript repo, when checked out here
        shutil.copy2(f"{base}.{ext}", os.path.join(PAPER_FIGS, os.path.basename(base) + f".{ext}"))


def build_main():
    os.makedirs(OUTDIR, exist_ok=True)
    base = ifm.build(os.path.join(OUTDIR, "fig3_interface_failure"))
    _copy(base)


def build_si_recovery():
    figstyle.apply("plain")
    rec = _recovery()
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_1COL, 2.4))
    panel_recovery(ax, rec)
    base = os.path.join(OUTDIR, "si_interface_sampling_recovery")
    figstyle.save(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    _copy(base)
    print("Saved:", base + ".png", rec)


if __name__ == "__main__":
    build_main()
    if "--no-si" not in sys.argv:
        build_si_recovery()
