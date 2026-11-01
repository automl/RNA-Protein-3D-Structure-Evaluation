"""
RNA sequence- and structure-descriptors vs AlphaFold3 prediction quality
(Nature-style, re-drawn natively).

Relates the per-RNA descriptor columns of pred_protein_rna
  RNA_GC_Content, RNA_AU_Content, RNA_GC_Skew, RNA_AU_Skew,
  RNA_SequenceComplexity, RNA_Shannon_Entropy_K1/K2,
  RNA_MFE_Value (+ derived MFE-per-nt and base-paired fraction from
  RNA_MFE_Structure)
to RNA prediction quality (RNA TM-score, all-atom lDDT) and to the predicted
secondary structure.

  A  Spearman correlation of each descriptor with RNA TM-score and with lDDT:
     composition terms are weak; the thermodynamic/structural terms (MFE,
     base-paired fraction) carry most of the (modest) signal, and
     local lDDT is essentially descriptor-independent.
  B  RNA TM-score vs MFE base-paired fraction (more-paired RNAs fold better).
  C  RNA TM-score vs MFE per nucleotide (more thermodynamically stable RNAs
     fold better).
  D  Spearman-correlation heatmap of every descriptor against prediction
     quality (TM, lDDT, INF) and secondary-structure content (paired fraction,
     stem count).

Usage:  python -m evaluation.rna_descriptors_figure
"""

import ast
import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PUB_RC, style_axis, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS

RC = dict(PUB_RC)
RC.update({
    "axes.labelsize": onpage_to_gen(7.5, 13.0),
    "axes.titlesize": onpage_to_gen(7.5, 13.0),
    "xtick.labelsize": onpage_to_gen(7, 13.0),
    "ytick.labelsize": onpage_to_gen(7, 13.0),
    "legend.fontsize": onpage_to_gen(7, 13.0),
    "legend.frameon": True,
})

DB = os.path.join(_REPO_ROOT, "database", "rbpDatabase.db")
COL_TM = PH_RANGE_COLORS["5-6"]    # blue
COL_LD = PH_RANGE_COLORS["6-7"]    # orange
DIV = LinearSegmentedColormap.from_list(
    "bwr", [PH_RANGE_COLORS["5-6"], "white", PH_RANGE_COLORS["<5"]])

# descriptor column -> display label (kept short so they fit as panel rows/ticks)
DESC = [
    ("RNA_GC_Content", "GC content"),
    ("RNA_GC_Skew", "GC skew"),
    ("RNA_AU_Skew", "AU skew"),
    ("RNA_SequenceComplexity", "Complexity"),
    ("RNA_Shannon_Entropy_K1", "Entropy k1"),
    ("RNA_Shannon_Entropy_K2", "Entropy k2"),
    ("RNA_MFE_Value", "MFE"),
    # MFE/nt omitted from panels A/D (= E/F of rna_secondary_structure): it is
    # near-collinear with MFE (rho ~ 0.95 with paired fraction) and adds nothing;
    # the supplementary scatter (build(), panel C) still plots it.
    ("MFE_paired_frac", "Paired frac."),
]


def _paired_frac(s):
    s = str(s)
    if not s or s in ("None", "nan"):
        return np.nan
    n = len(s)
    return (s.count("(") + s.count(")")) / n if n else np.nan


def _struct_len(s):
    s = str(s)
    return len(s) if s and s not in ("None", "nan") else np.nan


def _stem_count(c):
    try:
        v = ast.literal_eval(str(c))
        return len(v) if isinstance(v, (list, tuple)) else 0
    except (ValueError, SyntaxError):
        return 0


def _inf0(c):
    try:
        v = ast.literal_eval(str(c))
        if isinstance(v, list) and v:
            return v[0]
    except (ValueError, SyntaxError):
        pass
    return np.nan


def _load():
    con = sqlite3.connect(DB)
    pr = pd.read_sql_query("SELECT * FROM pred_protein_rna", con)
    con.close()
    pr["TM"] = pd.to_numeric(pr["RNA_TM"], errors="coerce")
    pr["LDDT"] = pd.to_numeric(pr["RNA_LDDT_allAtom"], errors="coerce")
    pr["INF"] = pr["RNA_INF"].apply(_inf0).replace(-1, np.nan)
    for col in ("RNA_GC_Content", "RNA_AU_Content", "RNA_GC_Skew", "RNA_AU_Skew",
                "RNA_SequenceComplexity", "RNA_Shannon_Entropy_K1",
                "RNA_Shannon_Entropy_K2", "RNA_MFE_Value"):
        pr[col] = pd.to_numeric(pr[col], errors="coerce")
    pr["MFE_paired_frac"] = pr["RNA_MFE_Structure"].apply(_paired_frac)
    pr["_slen"] = pr["RNA_MFE_Structure"].apply(_struct_len)
    pr["MFE_per_nt"] = pr["RNA_MFE_Value"] / pr["_slen"]
    pr["n_stems"] = pr["RNA_Stems"].apply(_stem_count)
    return pr


def _rho(pr, a, b):
    d = pr.dropna(subset=[a, b])
    if len(d) < 3:
        return np.nan, 0
    return spearmanr(d[a], d[b]).correlation, len(d)


# --------------------------------------------------------------------------- #
def _draw_A(ax, pr):
    rows = []
    for col, lab in DESC:
        rtm, _ = _rho(pr, col, "TM")
        rld, _ = _rho(pr, col, "LDDT")
        rows.append((lab, rtm, rld))
    rows.sort(key=lambda r: r[1])              # ascending by TM correlation
    labs = [r[0] for r in rows]
    y = np.arange(len(labs)); h = 0.38
    ax.barh(y + h / 2, [r[1] for r in rows], h, color=COL_TM, label=r"TM$_r$",
            edgecolor="black", linewidth=0.3)
    ax.barh(y - h / 2, [r[2] for r in rows], h, color=COL_LD, label=r"LDDT$_r$",
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlabel(r"Spearman $\rho$ with accuracy")
    ax.set_xlim(-0.45, 0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False)
    style_axis(ax, grid_axis="x")


def _scatter(ax, pr, xcol, xlabel):
    d = pr.dropna(subset=[xcol, "TM"])
    ax.scatter(d[xcol], d["TM"], s=22, color=COL_TM, alpha=0.5, edgecolor="none")
    rho = spearmanr(d[xcol], d["TM"]).correlation
    b, a = np.polyfit(d[xcol], d["TM"], 1)
    xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
    ax.plot(xs, b * xs + a, color="0.25", lw=1.6, ls="--")
    ax.text(0.04, 0.95, fr"$\rho = {rho:.2f}$" + f"\nn = {len(d)}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=onpage_to_gen(6.5, 13.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RNA TM-score")
    ax.set_ylim(0, 1.02)
    style_axis(ax, grid_axis="both")


def _draw_D(ax, pr):
    out_cols = [("TM", r"TM$_r$"), ("LDDT", r"LDDT$_r$"), ("INF", "RNA INF"),
                ("MFE_paired_frac", "Paired frac."), ("n_stems", "# Stems")]
    mat = np.full((len(DESC), len(out_cols)), np.nan)
    for i, (col, _) in enumerate(DESC):
        for j, (oc, _) in enumerate(out_cols):
            if col == oc:
                continue
            mat[i, j] = _rho(pr, col, oc)[0]
    df = pd.DataFrame(mat, index=[l for _, l in DESC],
                      columns=[l for _, l in out_cols])
    sns.heatmap(df, annot=True, fmt="+.2f", cmap=DIV, center=0, vmin=-0.6, vmax=0.6,
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": onpage_to_gen(6.5, 13.0)},
                cbar_kws={"label": r"Spearman $\rho$", "fraction": 0.046, "pad": 0.04},
                ax=ax)
    # Steeper angle than the other panels: these column names ("Paired frac.",
    # "# Stems", ...) are long and merged into each other at 20 deg.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", rotation=0)


def build():
    """Supplementary figure: the two descriptor-vs-TM scatters (B and C).

    Panels A and D (correlation bars and heatmap) are reused as panels E and F
    of the main rna_secondary_structure figure, so only the scatters remain here.
    """
    pr = _load()
    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    os.makedirs(outdir, exist_ok=True)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.0, 5.6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.26)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        _scatter(axA, pr, "MFE_paired_frac", "MFE base-paired fraction")
        _scatter(axB, pr, "MFE_per_nt", "MFE per nucleotide (kcal/mol)")
        for ax, letter in ((axA, "A"), (axB, "B")):
            ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                    fontsize=onpage_to_gen(8, 13.0),
                    fontweight="bold", va="bottom", ha="right")
        paths = []
        for ext in ("png", "pdf", "svg", "eps"):
            out = os.path.join(outdir, f"rna_descriptors_supp.{ext}")
            fig.savefig(out, dpi=600, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
