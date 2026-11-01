"""
How AlphaFold3 RNA accuracy depends on RNA secondary structure (Nature-style).

Re-drawn natively (no embedded PNGs) so the whole figure shares one publication
style: large axis/tick/legend fonts, no panel titles, bold panel letters. Uses
the styling primitives of plots.publication_figure (PUB_RC, style_axis) with a
custom GridSpec so panel D can be given more room.

  A  RNA base-pair interaction-network fidelity by type (INF / WC / non-WC /
     stacking): canonical Watson-Crick pairs are reproduced almost perfectly,
     non-canonical pairs moderately and base stacking poorly.
  B  RNA accuracy (all-atom lDDT) versus base-pair fidelity (RNA INF): the more
     faithfully the base-pairing network is reproduced, the better the RNA fold
     (Spearman rho annotated).
  C  RNA accuracy (LDDT_r) per structural class of the target RNA (one class per
     target, from the experimental structure; evaluation/rna_structure_classes.py),
     pre- vs post-cutoff: every class is predicted well before the cutoff, while
     after it accuracy collapses for pseudoknots and internal loops.
  D  Where the protein-contacting RNA residues fall across the secondary
     -structure features, experimental vs predicted.

Usage:  python -m evaluation.rna_secondary_structure_figure
"""

import ast
import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PUB_RC, style_axis, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS
from evaluation.secondaryFeatPlots import (extract_motif_indices,
                                           is_single_stranded_dangling)
from rna_descriptors_figure import (_load as _load_desc,
                                    _draw_A as _draw_descA,
                                    _draw_D as _draw_descD)
from evaluation.rna_structure_classes import GOOD_RNA_LDDT, load_classified

PRE_C, POST_C = "#1f77b4", "#E8743B"   # pre/post-cutoff colours of post_pre_comparison_figure
C_CLASSES = ["L_shaped", "pseudoknot", "multiway_junction", "internal_loop",
             "hairpin", "single_stranded"]          # duplex (n=2 per period) omitted
C_SHORT = {"L_shaped": "L-shaped", "pseudoknot": "Pseudokn.",
           "multiway_junction": "Multiway", "internal_loop": "Internal",
           "hairpin": "Hairpin", "single_stranded": "Single-str."}

# Large, Nature-style fonts (cf. plotCreator.py); no titles anywhere.
RC = dict(PUB_RC)
RC.update({
    "axes.labelsize": onpage_to_gen(7.5, 17.5),
    "axes.titlesize": onpage_to_gen(7.5, 17.5),
    "xtick.labelsize": onpage_to_gen(7, 17.5),
    "ytick.labelsize": onpage_to_gen(7, 17.5),
    "legend.fontsize": onpage_to_gen(7, 17.5),
    "legend.frameon": True,
})

DB = os.path.join(_REPO_ROOT, "database", "rbpDatabase.db")

COL_A = ["#1f77b4", "#9467bd", PH_RANGE_COLORS["7-8"], PH_RANGE_COLORS["<5"]]
COL_HI = PH_RANGE_COLORS["5-6"]   # blue   — present / experimental
COL_LO = PH_RANGE_COLORS["6-7"]   # orange — absent / predicted

LDDTr, TMr = r"LDDT$_r$", r"TM$_r$"
SS_FEATS = ["Stems", "HairpinLoops", "InternalLoops", "MultibranchLoops",
            "Pseudoknots"]
SS_SHORT = {"Stems": "Stems", "HairpinLoops": "Hairpin", "InternalLoops": "Internal",
            "MultibranchLoops": "Multiway", "Pseudoknots": "Pseudokn."}
MOTIF_CATS = ["Stems", "Hairpin Loops", "Internal Loops", "Multibranch Loops",
              "Pseudoknots", "Dangling Ends", "Single Stranded"]
MOTIF_SHORT = ["Stems", "Hairpin", "Internal", "Multiway", "Pseudokn.",
               "Dangling", "Single-str."]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _cnt(cell):
    try:
        v = ast.literal_eval(str(cell))
        return len(v) if isinstance(v, (list, tuple)) else 0
    except (ValueError, SyntaxError):
        return 0


def _inf_component(cell, i):
    try:
        v = ast.literal_eval(str(cell))
        if isinstance(v, list) and len(v) > i:
            return v[i]
    except (ValueError, SyntaxError):
        pass
    return np.nan


def _load():
    con = sqlite3.connect(DB)
    ex = pd.read_sql_query("SELECT * FROM exp_protein_rna", con)
    pr = pd.read_sql_query("SELECT * FROM pred_protein_rna", con)
    con.close()
    pr["TM"] = pd.to_numeric(pr["RNA_TM"], errors="coerce")
    pr["LDDT"] = pd.to_numeric(pr["RNA_LDDT_allAtom"], errors="coerce")
    for name, i in (("INF", 0), ("INF_WC", 1), ("INF_nonWC", 2), ("INF_STACK", 3)):
        pr[name] = pr["RNA_INF"].apply(lambda c, i=i: _inf_component(c, i)).replace(-1, np.nan)
    for f in SS_FEATS:
        pr["e_" + f] = ex.set_index("PDBId").reindex(pr["exp_db_id"])["RNA_" + f].values
        pr["en_" + f] = pd.Series(pr["e_" + f]).apply(_cnt).values
    return ex, pr


def _motif_locations(ex, pr):
    """Percentage of protein-contacting RNA residues falling in each SS feature."""
    m = pd.merge(ex, pr, left_on="PDBId", right_on="exp_db_id", suffixes=("_e", "_p"))
    fc = {c: {"e": 0, "p": 0} for c in MOTIF_CATS}
    tot = {"e": 0, "p": 0}
    for _, row in m.iterrows():
        for sfx, k in (("_e", "e"), ("_p", "p")):
            motif = row["RNAMotif" + sfx]
            if not motif:
                continue
            idx = extract_motif_indices(motif, row["ChainIDpairList_proteinRNA_e"],
                                        row["RNAChainIDs_e"])
            if not idx:
                continue
            res = set(idx)
            tot[k] += 1
            col = lambda n: row[n + sfx]

            def _hit(colname, flatten):
                v = col(colname)
                return bool(v) and bool(res & set(flatten(ast.literal_eval(v))))
            if _hit("RNA_Stems", lambda L: (r for s in L for r in s)):
                fc["Stems"][k] += 1
            if _hit("RNA_HairpinLoops", lambda L: (r for s in L for r in s)):
                fc["Hairpin Loops"][k] += 1
            if _hit("RNA_InternalLoops", lambda L: (r for s in L for r in s)):
                fc["Internal Loops"][k] += 1
            if _hit("RNA_MultibranchLoops", lambda L: (r for s in L for r in s)):
                fc["Multibranch Loops"][k] += 1
            if _hit("RNA_Pseudoknots", lambda L: (r for p in L for r in p)):
                fc["Pseudoknots"][k] += 1
            if col("RNA_DanglingEnds"):
                dl = ast.literal_eval(col("RNA_DanglingEnds"))
                rt = is_single_stranded_dangling(
                    dl, stems=col("RNA_Stems"), hairpins=col("RNA_HairpinLoops"),
                    internal=col("RNA_InternalLoops"),
                    multibranch=col("RNA_MultibranchLoops"),
                    pseudoknots=col("RNA_Pseudoknots"))
                if rt == "single_stranded":
                    fc["Single Stranded"][k] += 1
                elif rt == "dangling_ends":
                    for er in dl:
                        for se in er:
                            if se and res & set(range(se[0], se[1] + 1)):
                                fc["Dangling Ends"][k] += 1
                                break
    exp_pct = [fc[c]["e"] / tot["e"] * 100 for c in MOTIF_CATS]
    pred_pct = [fc[c]["p"] / tot["p"] * 100 for c in MOTIF_CATS]
    return exp_pct, pred_pct


def _sem(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _draw_A(ax, pr):
    labels = ["INF", "INF-WC", "INF-nonWC", "INF-STACK"]
    cols = ["INF", "INF_WC", "INF_nonWC", "INF_STACK"]
    data = [pr[c].dropna().values for c in cols]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none",
                                    markeredgecolor="0.4"))
    for patch, c in zip(bp["boxes"], COL_A):
        patch.set_facecolor(c); patch.set_alpha(0.9); patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Interaction-network fidelity")
    ax.set_ylim(0, 1.02)
    style_axis(ax, grid_axis="y")


def _draw_B(ax, pr):
    d = pr.dropna(subset=["INF", "LDDT"])
    ax.scatter(d["INF"], d["LDDT"], s=22, color=COL_HI, alpha=0.55,
               edgecolor="none")
    rho = spearmanr(d["INF"], d["LDDT"]).correlation
    # least-squares guide line
    b, a = np.polyfit(d["INF"], d["LDDT"], 1)
    xs = np.linspace(d["INF"].min(), d["INF"].max(), 50)
    ax.plot(xs, b * xs + a, color="0.25", lw=1.6, ls="--")
    ax.text(0.05, 0.93, fr"$\rho = {rho:.2f}$" + f"\nn = {len(d)}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=onpage_to_gen(6.5, 17.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("RNA INF")
    ax.set_ylabel(LDDTr)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    style_axis(ax, grid_axis="both")


def _class_lddt():
    """{period: {class: [RNA_LDDT, ...]}} for the protein-RNA pre/post-cutoff sets."""
    out = {}
    for period in ("pre", "post"):
        recs, _ = load_classified("protein_rna", period)
        out[period] = {c: [r["lddt"] for r in recs if r["cls"] == c and r["lddt"] is not None]
                       for c in C_CLASSES}
    return out


def _draw_C(ax, cls_lddt):
    # LDDT_r per structural class, pre- vs post-cutoff; dashed line = good-RNA
    # threshold (classes i+ii). Bottom rows: n, pre (blue) above post (orange).
    x = np.arange(len(C_CLASSES)); w = 0.36
    small = onpage_to_gen(5.5, 17.5)
    rows = {"n": (0.15, 0.09)}                             # (pre y, post y)
    for off, period, col in ((-w / 2 - 0.02, "pre", PRE_C), (w / 2 + 0.02, "post", POST_C)):
        data = [cls_lddt[period][c] for c in C_CLASSES]
        bp = ax.boxplot(data, positions=x + off, widths=w, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.4),
                        whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9))
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.9); patch.set_edgecolor("black")
        ax.plot([], [], color=col, lw=8, alpha=0.9,
                label="pre-cutoff" if period == "pre" else "post-cutoff")
    for xi, c in zip(x, C_CLASSES):
        for i, (period, col) in enumerate((("pre", PRE_C), ("post", POST_C))):
            ax.text(xi, rows["n"][i], f"{len(cls_lddt[period][c])}", ha="center",
                    va="center", fontsize=small, color=col)
    lab = ax.get_yaxis_transform()            # x in axes coords, y in data coords
    ax.text(-0.03, sum(rows["n"]) / 2, "n", transform=lab, ha="right",
            va="center", fontsize=small, color="0.25")
    ax.axhline(GOOD_RNA_LDDT, color="0.35", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([C_SHORT[c] for c in C_CLASSES], rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(LDDTr)
    ax.set_ylim(0.04, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.16), frameon=False,
              handlelength=1.0, borderaxespad=0.2)
    style_axis(ax, grid_axis="y")


def _draw_D(ax, exp_pct, pred_pct):
    x = np.arange(len(MOTIF_CATS)); w = 0.4
    ax.bar(x - w / 2, exp_pct, w, color=COL_HI, label="Experimental",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, pred_pct, w, color=COL_LO, label="Predicted",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(MOTIF_SHORT, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Interface residues (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False)
    style_axis(ax, grid_axis="y")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build():
    ex, pr = _load()
    exp_pct, pred_pct = _motif_locations(ex, pr)
    pr_desc = _load_desc()                    # for panels E, F (from descriptors)
    cls_lddt = _class_lddt()                  # panel C (pre/post structural classes)

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    os.makedirs(outdir, exist_ok=True)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(17.5, 11.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.50)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[0, 2])
        axD = fig.add_subplot(gs[1, 0])
        axE = fig.add_subplot(gs[1, 1])
        axF = fig.add_subplot(gs[1, 2])

        _draw_A(axA, pr)
        _draw_B(axB, pr)
        _draw_C(axC, cls_lddt)
        _draw_D(axD, exp_pct, pred_pct)
        _draw_descA(axE, pr_desc)             # descriptor correlations (bars)
        _draw_descD(axF, pr_desc)             # descriptor correlation heatmap

        for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"),
                           (axE, "E"), (axF, "F")):
            ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                    fontsize=onpage_to_gen(8, 17.5),
                    fontweight="bold", va="bottom", ha="right")

        paths = []
        for ext in ("png", "pdf", "svg", "eps"):
            out = os.path.join(outdir, f"rna_secondary_structure.{ext}")
            fig.savefig(out, dpi=600, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
