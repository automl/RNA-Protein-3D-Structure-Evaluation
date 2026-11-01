#!/usr/bin/env python3
"""
motif_vs_interface_figure -- do AlphaFold3 get the right interface *residues* even when the
interface *geometry* is wrong?

Per protein-RNA complex we take the sequence/motif recovery scores (RNAmatch_score,
AAmatch_score, RNAmotif_score) versus the protein-RNA interface quality DockQ_Nr (mean RBP_DockQ
over the protein-RNA interfaces in ChainIDpairList_proteinRNA), coloured by the dominant
per-chain interface class.

  A  RNAmatch_score  vs DockQ_Nr        C  RNAmotif_score vs DockQ_Nr
  B  AAmatch_score   vs DockQ_Nr        D  match/motif scores by interface class (violins)

Points in the bottom-right of A-C (high match, low DockQ) are "right residues, wrong geometry":
the model recovers the interface residues/motif but places the interface incorrectly. The dashed
guides mark match = 0.7 and the CAPRI-acceptable DockQ = 0.23.

Run from the repo root:
  python publication_figures/motif_vs_interface_figure.py
"""
import ast
import json
import os
import sqlite3
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import spearmanr

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC, onpage_to_gen

# Larger, publication (Nature-style) fonts.
PUB_RC.update({
    "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
})

TABLE = "pred_protein_rna"
DOCKQ_ACCEPTABLE = 0.23
MATCH_HIGH = 0.7

# dominant interface class -> (colour, quality rank; higher = better); "missing" folds into "unknown".
CLASS_STYLE = {
    "correct_interface":   ("#1a9850", 3, "correct"),
    "partial_interface":   ("#fdae61", 2, "partial"),
    "incorrect_interface": ("#d73027", 1, "incorrect"),
    "unknown":             ("#c7c7c7", 0, "unknown"),
}
CLASS_ALIAS = {"missing": "unknown"}
CLASS_ORDER = ["correct", "partial", "incorrect", "unknown"]

SCORES = [
    ("RNAmatch_score", "RNA sequence match"),
    ("AAmatch_score",  "protein sequence match"),
    ("RNAmotif_score", "RNA motif recovery"),
]


def _db_path():
    with open(os.path.join(_REPO_ROOT, "database", os.environ.get("EVAL_CONFIG_NAME", "config.json"))) as fh:
        return json.load(fh)["db_path"]


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _dockq_nr(rbp_cell, pair_cell):
    """Mean RBP_DockQ over the protein-RNA interfaces in ChainIDpairList_proteinRNA."""
    allowed = set()
    try:
        pl = json.loads(str(pair_cell)) if pair_cell is not None else None
    except (ValueError, TypeError):
        pl = None
    if isinstance(pl, list):
        for pr in pl:
            if isinstance(pr, list) and len(pr) == 2:
                allowed.add(frozenset(str(x) for x in pr))
    s = "" if rbp_cell is None else str(rbp_cell).strip()
    if s in ("", "[]", "None", "null"):
        return np.nan
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return np.nan
    vals = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                for k, v in el.items():
                    parts = str(k).split(":")
                    if len(parts) == 2 and (not allowed or frozenset(parts) in allowed):
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
    return float(np.mean(vals)) if vals else np.nan


def _dominant_class(cell):
    if not cell:
        return None
    try:
        d = json.loads(str(cell))
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or not d:
        return None
    best, best_rank = None, -1
    for raw in d.values():
        c = CLASS_ALIAS.get(raw, raw)
        rank = CLASS_STYLE.get(c, ("", -1, ""))[1]
        if rank > best_rank:
            best, best_rank = c, rank
    return CLASS_STYLE.get(best, ("", 0, None))[2]


# extra metrics correlated against the three scores in panel D (df column, display label).
HEAT_COLS = [
    ("RNA_LDDT_allAtom", r"LDDT$_r$"), ("af3_rna_pLDDT_avg", r"pLDDT$_r$"),
    ("RNA_TM", r"TM$_r$"), ("af3_rna_pTM_avg", r"pTM$_r$"),
    ("af3_rna_ipTM_avg", r"ipTM$_r$"),
    ("Complex_TM", r"TM$_c$"), ("Complex_LDDT_allAtom", r"LDDT$_c$"),
    ("af3_protein_ipTM_avg", r"ipTM$_p$"),
    ("Protein_TM", r"TM$_p$"), ("af3_protein_pTM_avg", r"pTM$_p$"),
    ("Protein_LDDT", r"LDDT$_p$"), ("af3_protein_pLDDT_avg", r"pLDDT$_p$"),
    ("iLDDT", "iLDDT"), ("DockQ_Nr", r"DockQ$_{Nr}$"),
    ("msa_depth", r"MSA$_r$ depth"), ("has_rbp", "RBP template"),
    ("rna_len", "RNA length"),
]


def _first_num(x):
    """Leading number from an int, float, or list-valued cell (e.g. '[36, 19]' -> 36)."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return _num(s)
    if isinstance(v, (list, tuple)):
        nums = [float(z) for z in v if isinstance(z, (int, float))]
        return max(nums) if nums else np.nan
    return _num(v)


def _mean_list(x):
    """Mean of a list-valued cell like '[0.17, 0.01]' or '[0.45]' (per-chain scores)."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return _num(s)
    if isinstance(v, (list, tuple)):
        nums = [float(z) for z in v if isinstance(z, (int, float)) and np.isfinite(z)]
        return float(np.mean(nums)) if nums else np.nan
    return _num(v)


def _msa_depth(x):
    """RNA_msa_size is {chain: depth} (or a scalar); use the deepest chain."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return _num(s)
    if isinstance(v, dict):
        nums = [float(z) for z in v.values() if isinstance(z, (int, float))]
        return max(nums) if nums else np.nan
    if isinstance(v, (list, tuple)):
        nums = [float(z) for z in v if isinstance(z, (int, float))]
        return max(nums) if nums else np.nan
    return _num(v)


def _has_rbp(x):
    s = "" if x is None else str(x).strip()
    return 0.0 if s in ("", "[]", "None", "null") else 1.0


def _load():
    cols = ["exp_db_id", "RBP_DockQ", "ChainIDpairList_proteinRNA",
            "RNAmatch_score", "AAmatch_score", "RNAmotif_score", "RNA_interface_class",
            "RNA_TM", "RNA_LDDT_allAtom", "af3_rna_pLDDT_avg", "RNAComplex_TM", "Complex_TM",
            "Complex_LDDT_allAtom", "Protein_TM", "Protein_LDDT", "af3_protein_pLDDT_avg", "iLDDT",
            "RNA_msa_size", "RBP_PDB_Templates", "RNALength",
            "af3_rna_pTM", "af3_rna_ipTM", "af3_protein_pTM", "af3_protein_ipTM"]
    con = sqlite3.connect(_db_path())
    df = pd.read_sql_query("SELECT " + ", ".join(cols) + " FROM " + TABLE, con)
    con.close()
    df["af3_rna_pTM_avg"] = df["af3_rna_pTM"].apply(_mean_list)
    df["af3_rna_ipTM_avg"] = df["af3_rna_ipTM"].apply(_mean_list)
    df["af3_protein_ipTM_avg"] = df["af3_protein_ipTM"].apply(_mean_list)
    df["af3_protein_pTM_avg"] = df["af3_protein_pTM"].apply(_mean_list)
    df["DockQ_Nr"] = [_dockq_nr(a, b) for a, b in
                      zip(df["RBP_DockQ"], df["ChainIDpairList_proteinRNA"])]
    for s, _ in SCORES:
        df[s] = df[s].apply(_num)
    for c in ("RNA_TM", "RNA_LDDT_allAtom", "af3_rna_pLDDT_avg", "RNAComplex_TM", "Complex_TM",
              "Complex_LDDT_allAtom", "Protein_TM", "Protein_LDDT", "af3_protein_pLDDT_avg", "iLDDT"):
        df[c] = df[c].apply(_num)
    df["msa_depth"] = df["RNA_msa_size"].apply(_msa_depth)
    df["has_rbp"] = df["RBP_PDB_Templates"].apply(_has_rbp)
    df["rna_len"] = df["RNALength"].apply(_first_num)
    df["cls"] = df["RNA_interface_class"].apply(_dominant_class)
    return df


def _scatter(df, score_col, score_lab):
    def draw(ax):
        sub = df[np.isfinite(df[score_col]) & np.isfinite(df["DockQ_Nr"])]
        for cname in CLASS_ORDER:
            d = sub[sub["cls"] == cname]
            if len(d):
                ax.scatter(d[score_col], d["DockQ_Nr"], s=28,
                           c=CLASS_STYLE_BY_NAME[cname][0], edgecolor="black",
                           linewidth=0.3, alpha=0.85, zorder=3)
        ax.axvline(MATCH_HIGH, color="0.5", ls="--", lw=1.0, zorder=1)
        ax.axhline(DOCKQ_ACCEPTABLE, color="0.5", ls="--", lw=1.0, zorder=1)
        # "right residues, wrong geometry" quadrant: high match, low DockQ
        n_rr = int(((sub[score_col] > MATCH_HIGH) & (sub["DockQ_Nr"] < DOCKQ_ACCEPTABLE)).sum())
        ax.axvspan(MATCH_HIGH, 1.03, ymin=0, ymax=DOCKQ_ACCEPTABLE / 1.03,
                   color="#d73027", alpha=0.06, zorder=0)
        rho, p = spearmanr(sub[score_col], sub["DockQ_Nr"])
        ptxt = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
        # Stats box (top-left). The high-match/low-DockQ count is folded in as a legend entry
        # whose swatch matches the shaded lower-right quadrant, so it no longer needs a label
        # sitting on top of the dense cluster of points it describes.
        from matplotlib.patches import Patch
        _fs = onpage_to_gen(6.5, 11.2)
        quad_swatch = Patch(facecolor=(0.843, 0.188, 0.153, 0.06),
                            edgecolor="#d73027", linewidth=0.8)
        leg = ax.legend([quad_swatch], [f"n = {n_rr}"],
                        title=f"$\\rho$ = {rho:+.2f}  ({ptxt})\nn = {len(sub)}",
                        loc="upper left", bbox_to_anchor=(0.02, 0.98),
                        fontsize=_fs, title_fontsize=_fs, handlelength=1.1,
                        handleheight=1.1, borderpad=0.5, labelspacing=0.5,
                        handletextpad=0.5, fancybox=True, framealpha=0.9,
                        frameon=True)
        leg._legend_box.align = "left"
        leg.set_zorder(6)
        leg.get_frame().set_edgecolor("0.6")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_linewidth(1.0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(score_lab, fontsize=onpage_to_gen(7.5, 11.2))
        ax.set_ylabel(r"DockQ$_{Nr}$", fontsize=onpage_to_gen(7.5, 11.2))
        ax.tick_params(labelsize=12)
        style_axis(ax)
    return draw


def _heatmap(df, rows=None):
    """Spearman rho of each match/motif score (rows) against other metrics (columns).

    `rows`: [(df column, row label)]; default the three scores of this figure.
    """
    if rows is None:
        rows = [("RNAmatch_score", "RNA \nseq match"), ("AAmatch_score", "Protein \nseq match"),
                ("RNAmotif_score", "RNA motif")]
    for c, _ in HEAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mat = np.full((len(rows), len(HEAT_COLS)), np.nan)
    for i, (rc, _) in enumerate(rows):
        for j, (cc, _) in enumerate(HEAT_COLS):
            m = np.isfinite(df[rc]) & np.isfinite(df[cc])
            if m.sum() >= 8 and df.loc[m, rc].nunique() > 1 and df.loc[m, cc].nunique() > 1:
                mat[i, j], _p = spearmanr(df.loc[m, rc], df.loc[m, cc])

    def draw(ax):
        dm = pd.DataFrame(mat, index=[lab for _, lab in rows], columns=[c[1] for c in HEAT_COLS])
        sns.heatmap(dm, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": onpage_to_gen(6.5, 11.2)},
                    cbar_kws={"label": r"Spearman $\rho$", "fraction": 0.046, "pad": 0.04}, ax=ax)
        for t in ax.texts:            # rotate the rho values 90 deg (fit narrow cells)
            t.set_rotation(90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                           rotation_mode="anchor", fontsize=11)
        # nudge the rotated column labels a touch right so each reads centered
        # under its column (anchored-right rotation lands slightly left of centre).
        from matplotlib.transforms import ScaledTranslation
        _dx = ScaledTranslation(0.045, 0, ax.figure.dpi_scale_trans)
        for _lbl in ax.get_xticklabels():
            _lbl.set_transform(_lbl.get_transform() + _dx)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=onpage_to_gen(7, 11.2))
        ax.set_title("")
    return draw


CLASS_STYLE_BY_NAME = {v[2]: v for v in CLASS_STYLE.values()}


def main():
    df = _load()
    finite = int(np.isfinite(df["DockQ_Nr"]).sum())
    print(f"complexes: {len(df)}  DockQ_Nr finite: {finite}")
    for s, _ in SCORES:
        m = np.isfinite(df[s]) & np.isfinite(df["DockQ_Nr"])
        rho, p = spearmanr(df.loc[m, s], df.loc[m, "DockQ_Nr"])
        print(f"  {s:15s} rho={rho:+.3f} p={p:.4f} (n={int(m.sum())})")

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
    pf = PublicationFigure(ncols=2, panel_size=(5.6, 4.8), letters=True, outdir=outdir)
    pf.add_panel(_scatter(df, "RNAmatch_score", "RNA sequence match"), title=None)   # A
    pf.add_panel(_scatter(df, "AAmatch_score", "Protein sequence match"), title=None)  # B
    pf.add_panel(_scatter(df, "RNAmotif_score", "RNA motif recovery"), title=None)   # C
    pf.add_panel(_heatmap(df), title=None)                                           # D

    fig = pf.build()
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=9,
                      markerfacecolor=CLASS_STYLE_BY_NAME[c][0], markeredgecolor="black",
                      label=c) for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=onpage_to_gen(7, 11.2), bbox_to_anchor=(0.5, -0.02),
               title="interface class")
    import os as _os
    _os.makedirs(pf.results_dir, exist_ok=True)
    for ext in ("png", "pdf", "svg", "eps"):
        out = _os.path.join(pf.results_dir, "motif_vs_interface." + ext)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
