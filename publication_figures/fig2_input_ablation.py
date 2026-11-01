"""
Main figure 2 of main_final.tex (controlled input ablation) and its SI companion, rebuilt from the
panels of inputs_and_confidence_figure.py (imported, not modified):

  main  A  pTM_r against TM_r per predicted RNA chain, with and without an RNA alignment  (old a)
        B  ipTM_r by fold class i–iv and interface subclass                                (old c)
        C  single-chain RNAs with and without the RNA alignment, by depth and length        (old f)
        D  protein–RNA complexes, change in TM_r on removing the RNA alignment, by length   (old g)
        E  effect of each input on TM_r, paired within target, 95% CI                      (old h)
        F  paired standardised effect of each input on each accuracy metric                (old i)
  SI       mean pTM_r and TM_r by MSA depth (old b), global PAE vs RMSD_c (old d),
           confidence-vs-accuracy Spearman map (old e)

Writes results/plots/all_chains/publication_figures/main_final/{fig2_input_ablation,
si_confidence_panels}.{pdf,png} and copies them to 2025_NAR_AF3_Evaluation/figures2/.
Usage (RNABench env):  python publication_figures/fig2_input_ablation.py
"""

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import inputs_and_confidence_figure as icf       # sets its own config before the repo imports
from plots.publication_figure import PublicationFigure

OUTDIR = os.path.join(_REPO, "results", "plots", "all_chains", "publication_figures", "main_final")
PAPER_FIGS = os.path.join(_REPO, "2025_NAR_AF3_Evaluation", "figures2")
GEN_W = 16.5


def _en_dash(fig):
    """Typography: protein–RNA with an en dash in every text the imported panels draw."""
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]:
            s = t.get_text()
            if "Protein-RNA" in s or "protein-RNA" in s:
                t.set_text(s.replace("Protein-RNA", "Protein–RNA").replace("protein-RNA", "protein–RNA"))


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    for ext in ("pdf", "png") if os.path.isdir(PAPER_FIGS) else ():   # copy for the manuscript repo, when checked out here
        shutil.copy2(f"{base}.{ext}", os.path.join(PAPER_FIGS, f"{name}.{ext}"))
    print("Saved:", base + ".png")


def build():
    fac = icf.load_factorial(verbose=False)
    av = icf.availability()
    av = av[av.exp_db_id.isin(set(fac.exp_db_id))].set_index("exp_db_id")
    p_len, p_inputs, _interaction, p_heat = icf.make_ablation_panels(fac, av, icf.rna_lengths())
    pc, cat, heat, pae = icf.confidence_data()
    p_ptm, p_depth, p_iptm, p_conf_heat, p_pae = icf.make_confidence_panels(pc, cat, heat, pae)

    main = PublicationFigure(ncols=3, panel_size=(5.5, 4.3), outdir=OUTDIR)
    for draw in (p_ptm, p_iptm, icf.panel_single_chain_rna, p_len, p_inputs, p_heat):
        main.add_panel(draw, title=None)
    fig = main.build()
    _en_dash(fig)
    # A and B are protein–RNA chains too (confidence_data reads pred_protein_rna); label them as D–F.
    # A carries its legend above the axes, so its label sits inside, as the single-chain label of C.
    axA, axB = fig.axes[0], fig.axes[1]
    axA.annotate("Protein–RNA complexes", (0.02, 0.98), xycoords="axes fraction", va="top",
                 fontsize=11, color="0.35")
    axB.annotate("Protein–RNA complexes", (0.0, 1.02), xycoords="axes fraction", va="bottom",
                 fontsize=11, color="0.35")
    _save(fig, "fig2_input_ablation")

    si = PublicationFigure(ncols=3, panel_size=(5.5, 4.3), outdir=OUTDIR)
    for draw in (p_depth, p_pae, p_conf_heat):
        si.add_panel(draw, title=None)
    fig = si.build()
    _en_dash(fig)
    _save(fig, "si_confidence_panels")


if __name__ == "__main__":
    build()
