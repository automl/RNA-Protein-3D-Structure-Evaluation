"""
Supplementary figure: similarity-matched pre/post-cutoff comparison (RNA matching).

Every post-cutoff target is paired with the pre-cutoff target whose RNA is most similar
(evaluation/postpre_matched_similarity.py). Protein-RNA pairs are split by the RNA sequence
similarity of the twin (>= 80% / 50-80%) and by whether the twin also carries the same protein
(>= 80% identity: "same", else "other"); RNA-only pairs by RNA similarity.

  A  Paired difference in RNA accuracy (post - matched pre LDDT_r; median, bootstrap 95% CI).
     Accuracy is retained when the RNA has a near-identical pre-cutoff twin and drops when the
     RNA is novel -- even when the protein is the same one AF3 has already seen.
  B  Share of well-folded RNAs (LDDT_r > 0.85, classes i+ii), pre vs post, for the same groups.

Same style as the neighbouring SI figure postpre_family_comparison_figure.py (large canvas,
same fonts, colours, panel letters; saved as PNG + PDF).
Reads results/plots/all_chains/postpre_matched/matched_summary.tsv (runs the analysis first if
missing). Usage:  python publication_figures/postpre_matched_figure.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import style_axis

SUMMARY = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "postpre_matched",
                       "matched_summary.tsv")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")

# Same palette / fonts as postpre_family_comparison_figure.py
PRE_C, POST_C = "#1f77b4", "#E8743B"                      # blue / orange
CTX_C = {"protein_rna": "#5B8C5A", "rna_only": "#9B5DE5"}  # green / purple
PUB_RC = {
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.size": 15,
}
LBL = 12                                                   # in-plot annotation size

# (system, analysis, subset, tick label): RNA similarity of the twin / twin's protein
GROUPS = [
    ("protein_rna", "rna_bin_x_protein", ">=80% RNA, same protein", "≥80%\nsame\nprotein"),
    ("protein_rna", "rna_bin_x_protein", ">=80% RNA, other protein", "≥80%\nother\nprotein"),
    ("protein_rna", "rna_bin_x_protein", "50-80% RNA, same protein", "50–80%\nsame\nprotein"),
    ("protein_rna", "rna_bin_x_protein", "50-80% RNA, other protein", "50–80%\nother\nprotein"),
    ("rna_only", "rna_identity_bin", ">=80%", "≥80%"),
    ("rna_only", "rna_identity_bin", "50-80%", "50–80%"),
]


def _load():
    if not os.path.exists(SUMMARY):
        from evaluation.postpre_matched_similarity import main as run_analysis
        run_analysis()
    rows = {}
    with open(SUMMARY) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[(r["system"], r["analysis"], r["subset"], r["metric"])] = r
    return [rows[(s, a, g, "LDDT_r")] for s, a, g, _ in GROUPS]


def _xaxis(ax, rows):
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{g[3]}\nn={r['n']}" for g, r in zip(GROUPS, rows)])
    ax.set_xlabel("RNA sequence similarity")
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.axvline(3.5, color="0.6", lw=1.0, ls=":")
    lab = ax.get_xaxis_transform()
    ax.text(3.35, 1.02, "Protein–RNA", transform=lab, ha="right", va="bottom",
            fontsize=LBL + 1, color=CTX_C["protein_rna"], fontweight="bold")
    ax.text(3.65, 1.02, "Single-chain RNA", transform=lab, ha="left", va="bottom",
            fontsize=LBL + 1, color=CTX_C["rna_only"], fontweight="bold")


def _panel_paired_diff(ax, rows):
    for xi, (g, r) in enumerate(zip(GROUPS, rows)):
        m, lo, hi = float(r["diff_median"]), float(r["ci_lo"]), float(r["ci_hi"])
        ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt="o", color=CTX_C[g[0]], ms=11,
                    capsize=5, elinewidth=2.0, capthick=1.6,
                    markeredgecolor="black", markeredgewidth=0.8)
    ax.axhline(0, color="0.35", lw=1.2)
    ax.set_ylabel(r"$\Delta$ LDDT$_r$ (post $-$ matched pre)")
    ax.set_ylim(-0.52, 0.06)
    _xaxis(ax, rows)
    style_axis(ax, grid_axis="y")


def _panel_well_folded(ax, rows):
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w / 2 - 0.01, [float(r["pct_good_pre"]) for r in rows], w, color=PRE_C,
           edgecolor="black", linewidth=0.5, label="pre-cutoff twin")
    ax.bar(x + w / 2 + 0.01, [float(r["pct_good_post"]) for r in rows], w, color=POST_C,
           edgecolor="black", linewidth=0.5, label="post-cutoff target")
    ax.set_ylabel(r"Well-folded RNAs (%, LDDT$^{\mathrm{C3}'}_r>0.85$)")     # backbone C3' LDDT, as in the fold classes
    ax.set_ylim(0, 100)
    _xaxis(ax, rows)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2,
              framealpha=0.9)
    style_axis(ax, grid_axis="y")


def _letters(pairs):
    for ax, ltr in pairs:
        ax.text(-0.06, 1.06, ltr, transform=ax.transAxes, fontsize=22,
                fontweight="bold", va="bottom", ha="right")


def build():
    rows = _load()
    with plt.rc_context(PUB_RC):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(16.0, 6.8),
                                       gridspec_kw=dict(wspace=0.26))
        _panel_paired_diff(axA, rows)
        _panel_well_folded(axB, rows)
        _letters([(axA, "A"), (axB, "B")])
        # what "same" / "other" means, level with the legend under panel B
        axA.text(0.5, -0.36, "same / other: the pre-cutoff RNA twin carries the same protein\n"
                 "(≥80% identity) / a different protein",
                 transform=axA.transAxes, ha="center", va="top", fontsize=LBL + 1,
                 color="0.25")
        os.makedirs(OUTDIR, exist_ok=True)
        paths = []
        for ext in ("png", "pdf"):
            p = os.path.join(OUTDIR, f"postpre_matched.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
