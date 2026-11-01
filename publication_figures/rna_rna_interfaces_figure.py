"""
Supplementary figure: AF3 RNA-RNA complexes before and after the training cutoff.

  A  global accuracy per complex: RNA all-atom LDDT (OpenStructure), RNA TM-score and complex
     TM-score, pre- vs post-cutoff
  B  contact match per RNA-RNA interface (Jaccard overlap of predicted and experimental contacting
     nucleotides after exact residue mapping, both sides pooled)
  C  pose of the partner RNA in the frame of the superposed receptor RNA: translation vs rotation
  D  AF3 chain-pair ipTM against the contact match

Reads results/plots/all_chains/rna_rna_interfaces/rna_rna_interfaces.tsv
(evaluation/rna_rna_interfaces.py). Same style as the neighbouring SI figures.
Usage (RNABench env):  python publication_figures/rna_rna_interfaces_figure.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import style_axis

TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_rna_interfaces", "rna_rna_interfaces.tsv")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
PUB_RC = {"axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
          "legend.fontsize": 13, "font.size": 15}
LBL = 13
PRE_C, POST_C = "#1f77b4", "#E8743B"
PERIODS = [("pre", "pre-cutoff", PRE_C), ("post", "post-cutoff", POST_C)]


def _f(v):
    return float(v) if v not in ("", "nan", None) else np.nan


def _load():
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    for r in rows:
        for k in ("contact_match", "translation", "rotation", "lrmsd", "pair_iptm", "RNA_LDDT", "RNA_TM",
                  "Complex_TM", "Complex_LDDT"):
            r[k] = _f(r[k])
    return rows


def _pstr(p):
    return f"P = {p:.1e}" if p < 0.001 else f"P = {p:.3f}"


def panel_global(ax, rows):
    metrics = [("RNA_LDDT", "RNA LDDT"), ("RNA_TM", "RNA TM"), ("Complex_TM", "Complex TM")]
    for i, (k, lab) in enumerate(metrics):
        vals = {}
        for j, (per, _, col) in enumerate(PERIODS):
            cx = {r["pdb"]: r[k] for r in rows if r["period"] == per}
            v = [x for x in cx.values() if np.isfinite(x)]
            vals[per] = v
            bp = ax.boxplot([v], positions=[i + (j - 0.5) * 0.36], widths=0.32, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.6))
            bp["boxes"][0].set(facecolor=col, alpha=0.75, edgecolor="0.2")
            rng = np.random.RandomState(1)
            ax.scatter(i + (j - 0.5) * 0.36 + rng.uniform(-0.09, 0.09, len(v)), v, s=12, c="0.2",
                       alpha=0.5, zorder=3, edgecolor="none")
        p = mannwhitneyu(vals["pre"], vals["post"]).pvalue
        ax.text(i, 1.06, _pstr(p), ha="center", va="bottom", fontsize=LBL - 1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylim(-0.03, 1.16)
    ax.set_ylabel("Score per complex")
    n = {per: len({r["pdb"] for r in rows if r["period"] == per}) for per, _, _ in PERIODS}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for _, _, c in PERIODS],
              [f"{lab} (n={n[per]} complexes)" for per, lab, _ in PERIODS], loc="lower left", frameon=False)
    style_axis(ax, grid_axis="y")


def panel_contact(ax, rows):
    vals = {}
    for j, (per, lab, col) in enumerate(PERIODS):
        v = [r["contact_match"] for r in rows if r["period"] == per and np.isfinite(r["contact_match"])]
        vals[per] = v
        bp = ax.boxplot([v], positions=[j], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.6))
        bp["boxes"][0].set(facecolor=col, alpha=0.75, edgecolor="0.2")
        rng = np.random.RandomState(2)
        ax.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=14, c="0.2", alpha=0.5, zorder=3, edgecolor="none")
    p = mannwhitneyu(vals["pre"], vals["post"]).pvalue
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{lab}\n(n={len(vals[per])} interfaces)" for per, lab, _ in PERIODS])
    ax.set_ylim(-0.03, 1.12)
    ax.text(0.5, 1.05, _pstr(p), ha="center", va="bottom", fontsize=LBL)
    ax.set_ylabel("RNA-RNA contact match")
    style_axis(ax, grid_axis="y")


def panel_pose(ax, rows):
    for per, lab, col in PERIODS:
        s = [r for r in rows if r["period"] == per and np.isfinite(r["translation"]) and np.isfinite(r["rotation"])]
        ax.scatter(np.clip([r["translation"] for r in s], 0.1, 200), [r["rotation"] for r in s], s=36, c=col,
                   alpha=0.75, edgecolor="black", linewidth=0.3, label=f"{lab} (n={len(s)})",
                   marker="o" if per == "pre" else "^")
    ax.axvline(10, color="0.3", ls="--", lw=1.2)
    ax.axhline(45, color="0.3", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 200)
    ax.set_ylim(-3, 183)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Partner-RNA translation (Å)")
    ax.set_ylabel("Partner-RNA rotation (°)")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax, grid_axis="both")


def panel_iptm(ax, rows):
    for per, lab, col in PERIODS:
        s = [r for r in rows if r["period"] == per and np.isfinite(r["pair_iptm"]) and np.isfinite(r["contact_match"])]
        x, y = [r["pair_iptm"] for r in s], [r["contact_match"] for r in s]
        rho = spearmanr(x, y)[0]
        ax.scatter(x, y, s=36, c=col, alpha=0.75, edgecolor="black", linewidth=0.3,
                   marker="o" if per == "pre" else "^", label=f"{lab}: $\\rho$ = {rho:+.2f} (n={len(s)})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("AF3 chain-pair ipTM")
    ax.set_ylabel("RNA-RNA contact match")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax, grid_axis="both")


def _letters(pairs):
    for ax, ltr in pairs:
        ax.text(-0.08, 1.06, ltr, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom", ha="right")


def build():
    rows = _load()
    with plt.rc_context(PUB_RC):
        fig, axs = plt.subplots(2, 2, figsize=(16.0, 13.5), gridspec_kw=dict(hspace=0.34, wspace=0.24))
        panel_global(axs[0, 0], rows)
        panel_contact(axs[0, 1], rows)
        panel_pose(axs[1, 0], rows)
        panel_iptm(axs[1, 1], rows)
        _letters([(axs[0, 0], "A"), (axs[0, 1], "B"), (axs[1, 0], "C"), (axs[1, 1], "D")])
        os.makedirs(OUTDIR, exist_ok=True)
        paths = []
        for ext in ("png", "pdf"):
            p = os.path.join(OUTDIR, f"rna_rna_interfaces.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
