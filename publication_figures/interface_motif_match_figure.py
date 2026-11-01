"""
Supplementary figure: per-interface protein and RNA contact match of AF3 protein-RNA interfaces,
and how it relates to where AF3 places the RNA.

Each point is one native protein-RNA interface (one protein chain, one RNA chain;
evaluation/interface_motif_match.py): RNA sequence match (x) against protein sequence match (y) --
the Jaccard overlap of experimental and predicted contact residues (H-bond + vdW) after exact
residue mapping between the experimental and predicted structure (no motif sliding) -- coloured by
that interface's own DockQ (RBP_DockQ; grey: no interface DockQ). Native interfaces AF3 did not
form score 0. Quadrants at 0.5 separate interfaces with the right contacts on both sides (top
right), right protein contacts / wrong RNA contacts (top left), right RNA contacts / wrong protein
contacts (bottom right) and wrong on both (bottom left); counts are printed in the corners.
  A  pre-cutoff (b2021)     B  post-cutoff (a2021)
  C  protein contact match per placement class (evaluation/interface_rna_pose.py; classes, labels
     and colours as in the main figure interface_failure_modes_figure.py), pre- vs post-cutoff
  D  RNA contact match per placement class (not used by the placement rule)

Same style as the neighbouring SI figures (postpre_family_comparison_figure.py).
Reads results/plots/all_chains/interface_motif_match/interface_motif_match.tsv (runs the analysis first if
missing) and results/plots/all_chains/interface_rna_pose/interface_rna_pose.tsv. Usage:  python publication_figures/interface_motif_match_figure.py
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
sys.path.insert(0, os.path.join(_REPO_ROOT, "publication_figures"))

TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_motif_match",
                   "interface_motif_match.tsv")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
PUB_RC = {"axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
          "legend.fontsize": 13, "font.size": 15}
LBL = 13
SPLIT = 0.5
PERIODS = [("pre", "pre-cutoff (b2021)"), ("post", "post-cutoff (a2021)")]


def _load():
    if not os.path.exists(TSV):
        from evaluation.interface_motif_match import run
        run()
    data = {p: [] for p, _ in PERIODS}
    with open(TSV) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["native"] != "True":
                continue
            try:
                fp_, fr_ = float(r["aa_match"]), float(r["rna_match"])
            except ValueError:
                continue
            if not (np.isfinite(fp_) and np.isfinite(fr_)):
                continue
            dq = float(r["dockq"]) if r["dockq"] not in ("", "nan") else np.nan
            data[r["period"]].append((fr_, fp_, dq))
    return {p: np.array(v, float) for p, v in data.items()}


def _panel(ax, arr, title):
    x, y, dq = arr[:, 0], arr[:, 1], arr[:, 2]
    has = np.isfinite(dq)
    ax.scatter(x[~has], y[~has], s=26, c="0.75", edgecolor="none", alpha=0.6)
    order = np.argsort(-dq[has])                     # low-DockQ points drawn on top
    x, y = x.copy(), y.copy()
    xs_, ys_, dq_ = x[has][order], y[has][order], dq[has][order]
    sc = ax.scatter(xs_, ys_, c=dq_, cmap="coolwarm_r", vmin=0, vmax=1, s=34,
                    edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.axvline(SPLIT, color="0.4", ls="--", lw=1.2)
    ax.axhline(SPLIT, color="0.4", ls="--", lw=1.2)
    n = len(x)
    for (xs, ys, ha, va, m) in ((0.02, 0.98, "left", "top", (x < SPLIT) & (y >= SPLIT)),
                                (0.98, 0.98, "right", "top", (x >= SPLIT) & (y >= SPLIT)),
                                (0.02, 0.02, "left", "bottom", (x < SPLIT) & (y < SPLIT)),
                                (0.98, 0.02, "right", "bottom", (x >= SPLIT) & (y < SPLIT))):
        ax.text(xs, ys, f"n = {m.sum()} ({100 * m.sum() / n:.0f}%)", transform=ax.transAxes,
                ha=ha, va=va, fontsize=LBL, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("RNA contact match")
    ax.set_ylabel("Protein contact match")
    ax.set_title(f"{title}, n = {n} interfaces", fontsize=LBL + 2)
    style_axis(ax, grid_axis="both")
    return sc


def _load_pose():
    """Per-interface placement and contact match (interfaces with a DockQ)."""
    from interface_failure_modes_figure import MERGE, POSE_TSV
    out = []
    with open(POSE_TSV) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["dockq"] in ("", "nan"):
                continue
            out.append((r["period"], MERGE.get(r["placement"], r["placement"]),
                        float(r["protein_match"]), float(r["rna_match"])))
    return out


def _box_panel(ax, pose, col_idx, ylabel):
    """Contact match per placement class, pre (open) and post (filled) side by side."""
    from interface_failure_modes_figure import CLASSES
    for i, (key, _, colour) in enumerate(CLASSES):
        for per, dx, filled in (("pre", -0.2, False), ("post", 0.2, True)):
            v = [r[col_idx] for r in pose if r[0] == per and r[1] == key]
            if not v:
                continue
            bp = ax.boxplot([v], positions=[i + dx], widths=0.34, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.6),
                            whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
            bp["boxes"][0].set(facecolor=colour if filled else "white", edgecolor=colour if not filled
                               else "0.2", linewidth=2.0 if not filled else 1.0)
            ax.text(i + dx, -0.09, f"{len(v)}", ha="center", va="top", fontsize=LBL - 3, color="0.35")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([lab.replace(", ", ",\n").replace(" binding", "\nbinding") for _, lab, _ in CLASSES],
                       fontsize=LBL - 1, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax.set_ylim(-0.17, 1.05)
    ax.set_ylabel(ylabel)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="white", edgecolor="0.3", lw=1.5), Patch(facecolor="0.55", edgecolor="0.2")],
              ["pre-cutoff", "post-cutoff"], loc="upper right", frameon=False)
    style_axis(ax, grid_axis="y")


def _letters(pairs):
    for ax, ltr in pairs:
        ax.text(-0.06, 1.06, ltr, transform=ax.transAxes, fontsize=22,
                fontweight="bold", va="bottom", ha="right")


def build():
    data = _load()
    pose = _load_pose()
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(16.0, 15.6))
        gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)
        axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        sc = None
        for ax, (p, title) in zip((axA, axB), PERIODS):
            sc = _panel(ax, data[p], title)
        cb = fig.colorbar(sc, ax=[axA, axB], fraction=0.025, pad=0.02)
        cb.set_label("DockQ$_{pr}$")
        _box_panel(axC, pose, 2, "Protein contact match")
        _box_panel(axD, pose, 3, "RNA contact match")
        _letters([(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")])
        os.makedirs(OUTDIR, exist_ok=True)
        paths = []
        for ext in ("png", "pdf"):
            p = os.path.join(OUTDIR, f"interface_motif_match.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
