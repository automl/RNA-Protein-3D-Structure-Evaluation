"""
RNA bond-geometry metrics for AlphaFold3 protein-RNA predictions (Nature-style).

Merges the "Bond Metrics" section of plotResults.py into one big-font figure,
re-drawn natively from database/rbpDatabase.db (no embedded PNGs). The emphasised
panel is the per-category count accuracy split by RNA-MSA status
(plotResults' ``bond_metrics_accuracy_by_MSA``).

  A  Glycosidic-bond conformer distribution (anti / syn), experimental vs pred.
  B  Gamma-angle conformer distribution (g+, g-, t), experimental vs predicted.
  C  Sugar-pucker conformer distribution, experimental vs predicted.
  D  Count accuracy (within tau = 15 %) per bond category, +rMSA vs -rMSA.
  E  Spearman heatmap relating per-sample bond-conformer descriptors (rows) to
     RNA/complex accuracy (TM, lDDT, iLDDT, DockQ) and to base-pair fidelity
     (INF, WC, non-WC, stacking). Bond fidelity tracks local lDDT most; over-
     predicting the generic conformer (anti, C3'-endo) hurts, recovering the
     specific one (syn, C2'-endo) helps; both couple to non-WC/stacking, not WC.

Usage:  python -m evaluation.bond_metrics_figure
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
    "axes.labelsize": onpage_to_gen(7.5, 18.5),
    "axes.titlesize": onpage_to_gen(7.5, 18.5),
    "xtick.labelsize": onpage_to_gen(7, 18.5),
    "ytick.labelsize": onpage_to_gen(7, 18.5),
    "legend.fontsize": onpage_to_gen(7, 18.5),
    "legend.frameon": False,
})

DB = os.path.join(_REPO_ROOT, "database", "rbpDatabase.db")
COL_EXP = PH_RANGE_COLORS["5-6"]    # blue   — experimental (panels A-C)
COL_PRED = PH_RANGE_COLORS["6-7"]   # orange — predicted   (panels A-C)
COL_MSA = "#2E8B57"                 # sea green  — +rMSA (panel D)
COL_NOMSA = "#C0392B"               # brick red  — -rMSA (panel D)
DIV = LinearSegmentedColormap.from_list(
    "bwr", [PH_RANGE_COLORS["5-6"], "white", PH_RANGE_COLORS["<5"]])

GLYCO = ["anti", "syn"]
GAMMA = ["g+", "g-", "t"]
PUCKER = ["C3'-endo", "C2'-endo", "C3'-exo", "C4'-exo", "C2'-exo",
          "O4'-endo", "O4'-exo"]
TOL = 0.15
# (display label, metric, category) for the accuracy-by-MSA panel
ACC_CATS = [("Glyco-anti", "glyco", "anti"), ("Glyco-syn", "glyco", "syn"),
            ("Pucker C3'-endo", "pucker", "C3'-endo"),
            ("Pucker C2'-endo", "pucker", "C2'-endo")]

# Per-sample bond descriptors (E, F): conformer-distribution fidelity and the
# signed over/under-prediction of generic vs specific conformers.
DESC_ROWS = [
    ("gly_fid", "Glyco fidelity"),
    ("puc_fid", "Pucker fidelity"),
    ("gam_fid", "Gamma fidelity"),
    ("d_anti", r"$\Delta$anti (generic)"),
    ("d_syn", r"$\Delta$syn (specific)"),
    ("d_c3", r"$\Delta$C3$'$-endo (gen.)"),
    ("d_c2", r"$\Delta$C2$'$-endo (spec.)"),
]
QUAL_COLS = [("RNA_TM", r"TM$_r$"), ("RNA_LDDT_allAtom", r"LDDT$_r$"),
             ("Complex_TM", r"TM$_c$"), ("Complex_LDDT_allAtom", r"LDDT$_c$"),
             ("Dockq", "DockQ")]
INF_COLS = [("INF", "INF"), ("WC", "INF-WC"), ("nonWC", "INF-nonWC"),
            ("STACK", "INF-STACK")]


# --------------------------------------------------------------------------- #
def _parse_dict(s):
    """Replicate plotResults.parse_dict_string ({'anti': 26} -> {'anti':26})."""
    try:
        body = str(s).strip("{}")
        out = {}
        for pair in body.split(","):
            if ":" in pair:
                k, v = pair.split(":")
                out[k.strip().strip("'").strip('"')] = int(v.strip())
        return out
    except (ValueError, AttributeError):
        return {}


def _safe(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _classify_msa(row):
    """+rMSA if the (longest) RNA chain has an MSA (msa size > 1), else -rMSA."""
    msa = _safe(row.get("RNA_msa_size"))
    if not isinstance(msa, dict) or not msa:
        return np.nan
    if len(msa) == 1:
        msa_val = list(msa.values())[0]
    else:
        chains = _safe(row.get("RNAChainIDs"))
        lengths = _safe(row.get("RNALength"))
        chains = chains if isinstance(chains, (list, tuple)) else [chains]
        lengths = lengths if isinstance(lengths, (list, tuple)) else [lengths]
        try:
            idx = list(lengths).index(max(lengths))
            msa_val = msa.get(str(chains[idx]), 1)
        except (ValueError, IndexError):
            return np.nan
    return "+rMSA" if msa_val > 1 else "-rMSA"


def _load():
    con = sqlite3.connect(DB)
    ex = pd.read_sql_query(
        "SELECT PDBId, RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle "
        "FROM exp_protein_rna", con)
    pr = pd.read_sql_query(
        "SELECT exp_db_id, RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle, "
        "RNA_msa_size, RNAChainIDs, RNALength, RNA_TM, RNA_LDDT_allAtom, "
        "Complex_TM, Complex_LDDT_allAtom, iLDDT, Dockq, RNA_INF "
        "FROM pred_protein_rna", con)
    con.close()
    m = pd.merge(ex, pr, left_on="PDBId", right_on="exp_db_id",
                 suffixes=("_exp", "_pred"))
    m["msa_status"] = m.apply(_classify_msa, axis=1)
    _add_descriptors(m)
    return m


def _frac(s, k):
    d = _parse_dict(s)
    t = sum(d.values()) or 1
    return d.get(k, 0) / t


def _fid(es, ps):
    """1 - total-variation distance between exp and pred conformer fractions."""
    e, p = _parse_dict(es), _parse_dict(ps)
    te, tp = sum(e.values()) or 1, sum(p.values()) or 1
    tvd = 0.5 * sum(abs(e.get(k, 0) / te - p.get(k, 0) / tp)
                    for k in set(e) | set(p))
    return 1 - tvd


def _inf_component(cell, i):
    try:
        v = ast.literal_eval(str(cell))
        if isinstance(v, list) and len(v) > i:
            return v[i]
    except (ValueError, SyntaxError):
        pass
    return np.nan


def _add_descriptors(m):
    ge, gp = "RNA_GlycosidicBond_exp", "RNA_GlycosidicBond_pred"
    pe, pp = "RNA_SugarPucker_exp", "RNA_SugarPucker_pred"
    ae, ap = "RNA_GammaAngle_exp", "RNA_GammaAngle_pred"
    m["gly_fid"] = [_fid(a, b) for a, b in zip(m[ge], m[gp])]
    m["puc_fid"] = [_fid(a, b) for a, b in zip(m[pe], m[pp])]
    m["gam_fid"] = [_fid(a, b) for a, b in zip(m[ae], m[ap])]
    m["d_anti"] = [_frac(b, "anti") - _frac(a, "anti") for a, b in zip(m[ge], m[gp])]
    m["d_syn"] = [_frac(b, "syn") - _frac(a, "syn") for a, b in zip(m[ge], m[gp])]
    m["d_c3"] = [_frac(b, "C3'-endo") - _frac(a, "C3'-endo") for a, b in zip(m[pe], m[pp])]
    m["d_c2"] = [_frac(b, "C2'-endo") - _frac(a, "C2'-endo") for a, b in zip(m[pe], m[pp])]
    for col, _ in QUAL_COLS:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    for name, i in (("INF", 0), ("WC", 1), ("nonWC", 2), ("STACK", 3)):
        m[name] = m["RNA_INF"].apply(lambda c, i=i: _inf_component(c, i)).replace(-1, np.nan)


def _corr_frame(m, col_specs):
    mat = np.full((len(DESC_ROWS), len(col_specs)), np.nan)
    for i, (rcol, _) in enumerate(DESC_ROWS):
        for j, (ccol, _) in enumerate(col_specs):
            d = m.dropna(subset=[rcol, ccol])
            if len(d) > 5:
                mat[i, j] = spearmanr(d[rcol], d[ccol]).correlation
    return pd.DataFrame(mat, index=[l for _, l in DESC_ROWS],
                        columns=[l for _, l in col_specs])


def _distribution(m, col, cats):
    """Percentage of residues in each conformer, experimental and predicted."""
    out = {}
    for who, suff in (("exp", "_exp"), ("pred", "_pred")):
        tot = {c: 0 for c in cats}
        for s in m[col + suff].dropna():
            d = _parse_dict(s)
            for c in cats:
                tot[c] += d.get(c, 0)
        n = sum(tot.values()) or 1
        out[who] = [tot[c] / n * 100 for c in cats]
    return out


def _accuracy_by_msa(m):
    cols = {"glyco": "RNA_GlycosidicBond", "pucker": "RNA_SugarPucker"}
    acc = {lab: {"+rMSA": [0, 0], "-rMSA": [0, 0]} for lab, _, _ in ACC_CATS}
    for _, row in m.iterrows():
        status = row.get("msa_status")
        if status not in ("+rMSA", "-rMSA"):
            continue
        for lab, metric, cat in ACC_CATS:
            base = cols[metric]
            pe, ee = row.get(base + "_pred"), row.get(base + "_exp")
            if pd.isna(pe) or pd.isna(ee):
                continue
            ed = _parse_dict(ee).get(cat, 0)
            pdc = _parse_dict(pe).get(cat, 0)
            if ed > 0:
                acc[lab][status][1] += 1
                if abs(pdc - ed) / float(ed) <= TOL:
                    acc[lab][status][0] += 1

    def rate(c, n):
        if n == 0:
            return 0.0, 0.0
        p = c / n
        return p * 100, np.sqrt(p * (1 - p) / n) * 100
    msa_m, msa_e, no_m, no_e = [], [], [], []
    for lab, _, _ in ACC_CATS:
        a, ea = rate(*acc[lab]["+rMSA"])
        b, eb = rate(*acc[lab]["-rMSA"])
        msa_m.append(a); msa_e.append(ea); no_m.append(b); no_e.append(eb)
    return msa_m, msa_e, no_m, no_e


# --------------------------------------------------------------------------- #
def _grouped(ax, cats, exp, pred, ylabel, rotation=0, leg_bbox=None):
    x = np.arange(len(cats)); w = 0.4
    ax.bar(x - w / 2, exp, w, color=COL_EXP, label="Experimental",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, pred, w, color=COL_PRED, label="Predicted",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=rotation, ha="right" if rotation else "center",
                       rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)   # shared 0-100% axis across A/B/C for direct comparability
    # legend is drawn once, shared, below the A-B-C row (see build())
    style_axis(ax, grid_axis="y")


def _draw_D(ax, msa_m, msa_e, no_m, no_e):
    cats = [lab for lab, _, _ in ACC_CATS]
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, msa_m, w, yerr=msa_e, color=COL_MSA, label="+rMSA",
           edgecolor="black", linewidth=0.4,
           error_kw=dict(ecolor="0.3", elinewidth=1.3, capsize=4))
    ax.bar(x + w / 2, no_m, w, yerr=no_e, color=COL_NOMSA, label="$-$rMSA",
           edgecolor="black", linewidth=0.4,
           error_kw=dict(ecolor="0.3", elinewidth=1.3, capsize=4))
    ax.axvline(1.5, color="0.6", ls=":", lw=1.2)      # glyco | pucker separator
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"Count accuracy (%, $\tau$=15%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", ncol=1)
    style_axis(ax, grid_axis="y")


def _heatmap(ax, frame, cbar_label, sep=None):
    sns.heatmap(frame, annot=True, fmt="+.2f", cmap=DIV, center=0,
                vmin=-0.6, vmax=0.6, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": onpage_to_gen(6.5, 18.5)},
                cbar_kws={"label": cbar_label, "fraction": 0.04, "pad": 0.03},
                ax=ax)
    if sep is not None:
        ax.axvline(sep, color="0.2", lw=2.5)   # quality | INF column separator
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="y", rotation=0)


def build():
    m = _load()
    gly = _distribution(m, "RNA_GlycosidicBond", GLYCO)
    gam = _distribution(m, "RNA_GammaAngle", GAMMA)
    puc = _distribution(m, "RNA_SugarPucker", PUCKER)
    msa_m, msa_e, no_m, no_e = _accuracy_by_msa(m)

    corr = _corr_frame(m, QUAL_COLS + INF_COLS)   # combined E+F heatmap

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    os.makedirs(outdir, exist_ok=True)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(18.5, 11.0))
        gs = GridSpec(2, 7, figure=fig, height_ratios=[1.0, 1.08],
                      hspace=0.55, wspace=1.1,
                      width_ratios=[1, 1, 0.35, 1, 1, 1, 1])
        axA = fig.add_subplot(gs[0, 0:2])
        axB = fig.add_subplot(gs[0, 3:5])
        axC = fig.add_subplot(gs[0, 5:7])
        axD = fig.add_subplot(gs[1, 0:2])
        axE = fig.add_subplot(gs[1, 3:7])   # spacer column (2) separates D and E

        _grouped(axA, GLYCO, gly["exp"], gly["pred"], "Residues (%)",
                 leg_bbox=(1.12, 1.0))
        _grouped(axB, GAMMA, gam["exp"], gam["pred"], "Residues (%)")
        _grouped(axC, PUCKER, puc["exp"], puc["pred"], "Residues (%)", rotation=40)
        # one shared Experimental/Predicted legend centered BELOW the A-B-C row
        _h, _l = axA.get_legend_handles_labels()
        fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.551),
                   ncol=2, frameon=False, fontsize=onpage_to_gen(7.5, 18.5),
                   columnspacing=2.6, handlelength=1.4)
        _draw_D(axD, msa_m, msa_e, no_m, no_e)
        _heatmap(axE, corr, r"Spearman $\rho$", sep=len(QUAL_COLS))

        for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"),
                           (axE, "E")):
            ax.text(-0.13, 1.05, letter, transform=ax.transAxes,
                    fontsize=onpage_to_gen(8, 18.5),
                    fontweight="bold", va="bottom", ha="right")
        paths = []
        for ext in ("png", "pdf", "svg", "eps"):
            out = os.path.join(outdir, f"bond_metrics.{ext}")
            fig.savefig(out, dpi=600, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
