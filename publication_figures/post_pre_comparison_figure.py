"""
Pre- vs post-AlphaFold3-training-cutoff comparison (Nature-style, big fonts).

Compares structures deposited *before* (b2021, pre-cutoff, training-era) and
*after* (a2021, post-cutoff, novel) the AF3 ~2021 training cutoff, for both
protein--RNA complexes and RNA-only folds, across every vital metric
(TM, lDDT, RMSD, DockQ/QS/ICS, INF, and the AF3 pTM/ipTM/PAE confidences).

Databases (data/compareDatabases/post_pre):
  a2021_b2021_protein_rna/{b2021.db, a2021.db}   (pred_protein_rna)
  a2021_b2021_rna_only/{b2021_rna.db, a2021_rna.db}   (pred_rna_rna)

Emits, under results/plots/all_chains/publication_figures:
  postpre_protein_rna  : protein--RNA metric distributions, pre vs post.
  postpre_rna_only     : RNA-only metric distributions (incl. INF), pre vs post.
  postpre_gap          : synthesis -- the generalization gap is RNA-specific.

Usage:  python -m evaluation_db.post_pre_comparison_figure
"""

import json
import os
import sqlite3
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import (
    PublicationFigure, style_axis, PUB_RC, onpage_to_gen)

# This module-level RC is shared (via plt.rc_context(PUB_RC)) across the three
# figures built here at different generation widths (protein_rna 16.5,
# rna_only 15.0, combined 17.0). Since one RC serves all three, scale it at the
# median width so on-page sizes land near the cross-figure targets
# (axis 7.5 pt, tick/legend 7 pt). Per-figure hardcoded sizes are additionally
# scaled by each figure's own width below.
_RC_W = 16.5
PUB_RC.update({
    "axes.labelsize": onpage_to_gen(7.5, _RC_W),
    "axes.titlesize": onpage_to_gen(7.5, _RC_W),
    "xtick.labelsize": onpage_to_gen(7.0, _RC_W),
    "ytick.labelsize": onpage_to_gen(7.0, _RC_W),
    "legend.fontsize": onpage_to_gen(7.0, _RC_W),
    "legend.title_fontsize": onpage_to_gen(7.0, _RC_W),
})

_BASE = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre")
PR_DIR = os.path.join(_BASE, "a2021_b2021_protein_rna")
RNA_DIR = os.path.join(_BASE, "a2021_b2021_rna_only")
MULTI_DIR = os.path.join(_BASE, "a2021_b2021_rna_rna")     # multi-chain RNA (>= 2 RNA chains, run_eval.py)
PR_COL, MONO_COL, MULTI_COL = "#5B8C5A", "#9B5DE5", "#2A9D8F"

PRE = "pre-cutoff (b2021)"
POST = "post-cutoff (a2021)"
PRE_C, POST_C = "#1f77b4", "#E8743B"           # blue / orange
PAL = {PRE: PRE_C, POST: POST_C}
HUE = [PRE, POST]

DISP = {
    "TM_r": r"TM$_r$", "LDDT_r": r"LDDT$_r$", "TM_c": r"TM$_c$",
    "LDDT_c": r"LDDT$_c$", "iLDDT": "iLDDT", "DockQ": "DockQ",
    "DockQ_Nr": r"DockQ$_{Nr}$",
    "QS": r"QS$_{global}$", "ICS": "ICS", "pTM_prot": r"pTM$_p$",
    "ipTM_rna": r"ipTM$_r$", "pTM_rna": r"pTM$_r$", "PAE": "PAE",
    "PAE_min": "PAE min", "TM_prot": r"TM$_p$", "LDDT_prot": r"LDDT$_p$",
    "RMSD_r": r"RMSD$_r$", "INF": "INF", "WC": "INF-WC", "nonWC": "INF-nonWC",
    "STACK": "INF-STACK", "pLDDT_rna": r"pLDDT$_r$",
}
LOWER_BETTER = {"PAE", "PAE_min", "RMSD_r", "RMSD_c"}


# --------------------------------------------------------------------------- #
def _g(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    try:
        v = eval(s) if isinstance(s, str) else s
        return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
    except Exception:
        return np.nan


def _ptm(s, lenstr):
    try:
        L = eval(lenstr) if isinstance(lenstr, str) else lenstr
        if not isinstance(L, list):
            L = [L]
        idx = L.index(max(L))
        v = eval(s) if isinstance(s, str) else s
        if not isinstance(v, list):
            v = [v]
        return float(v[idx]) if idx < len(v) else float(v[0])
    except Exception:
        return np.nan


def _inf(s, i):
    try:
        x = float(eval(s)[i])
        return np.nan if x == -1 else x
    except Exception:
        return np.nan


def _dockq_nr(rbp_cell, pair_cell):
    """Mean DockQ_Nr: average of RBP_DockQ over the protein-RNA interfaces only.

    RBP_DockQ is a JSON keyed dict-list [{"A:B": 0.5}, ...];
    ChainIDpairList_proteinRNA is [["A","B"], ...]. Only pairs listed there
    (order-insensitive) contribute; if that list is empty/missing, all RBP_DockQ
    pairs are used. NaN if no scorable DockQ. Matches motif_vs_interface /
    compare_ion_input / domain_publication figures."""
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


def _load_pr(path, label):
    con = sqlite3.connect(path)
    d = pd.read_sql_query("SELECT * FROM pred_protein_rna", con)
    con.close()
    o = pd.DataFrame(index=range(len(d)))
    o["database"] = label
    o["TM_r"] = d["RNA_TM"].apply(_g)
    o["LDDT_r"] = d["RNA_LDDT_allAtom"].apply(_g)
    o["TM_c"] = d["Complex_TM"].apply(_g)
    o["LDDT_c"] = d["Complex_LDDT_allAtom"].apply(_g)
    o["iLDDT"] = d["iLDDT"].apply(_g)
    o["DockQ"] = d["Dockq"].apply(_g)
    if "RBP_DockQ" in d.columns and "ChainIDpairList_proteinRNA" in d.columns:
        o["DockQ_Nr"] = [_dockq_nr(rq, prp) for rq, prp
                         in zip(d["RBP_DockQ"], d["ChainIDpairList_proteinRNA"])]
    else:
        o["DockQ_Nr"] = np.nan
    o["QS"] = d["QS_global"].apply(_g)
    o["ICS"] = d["ICS"].apply(_g)
    o["pTM_prot"] = d["af3_protein_pTM"].apply(_g)
    o["ipTM_rna"] = [_ptm(s, l) for s, l in zip(d["af3_rna_ipTM"], d["RNALength"])]
    o["pTM_rna"] = [_ptm(s, l) for s, l in zip(d["af3_rna_pTM"], d["RNALength"])]
    o["PAE"] = d["af3_global_pae_avg"].apply(_g)
    o["PAE_min"] = d["af3_chain_pair_pae_min"].apply(_g)
    o["pLDDT_rna"] = d["af3_rna_pLDDT_avg"].apply(_g)
    o["TM_prot"] = d["Protein_TM"].apply(_g)
    o["LDDT_prot"] = d["Protein_LDDT"].apply(_g)
    o["RMSD_c"] = d["Complex_RMSD"].apply(_g)
    for n, i in (("INF", 0), ("WC", 1), ("nonWC", 2), ("STACK", 3)):
        o[n] = d["RNA_INF"].apply(lambda s, i=i: _inf(s, i))
    return o


def _load_rna(path, label):
    con = sqlite3.connect(path)
    d = pd.read_sql_query("SELECT * FROM pred_rna_rna", con)
    con.close()
    o = pd.DataFrame(index=range(len(d)))
    o["database"] = label
    o["TM_r"] = d["RNA_TM"].apply(_g)
    o["LDDT_r"] = d["RNA_LDDT_allAtom"].apply(_g)
    o["RMSD_r"] = d["RNA_RMSD"].apply(_g)
    o["pTM_rna"] = [_ptm(s, l) for s, l in zip(d["af3_rna_pTM"], d["RNALength"])]
    o["pLDDT_rna"] = d["af3_rna_pLDDT_avg"].apply(_g)
    for n, i in (("INF", 0), ("WC", 1), ("nonWC", 2), ("STACK", 3)):
        o[n] = d["RNA_INF"].apply(lambda s, i=i: _inf(s, i))
    return o


def _cohend(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / sp if sp else np.nan


def _d_ci(a, b):
    """Cohen's d and the half-width of its 95% CI (analytic SE)."""
    a = a.dropna(); b = b.dropna()
    n1, n2 = len(a), len(b)
    d = _cohend(a, b)
    if np.isnan(d) or n1 < 2 or n2 < 2:
        return d, 0.0
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d, 1.96 * se


def _median_ci(x, n_boot=2000, seed=0):
    """Median and its bootstrap 95% CI as (low_err, high_err) about the median."""
    x = np.asarray(x.dropna(), float)
    if len(x) < 2:
        m = np.median(x) if len(x) else np.nan
        return m, 0.0, 0.0
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    m = np.median(x)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return m, m - lo, hi - m


# --------------------------------------------------------------------------- #
def _violin(ax, df, cols, ylabel, ylim, legend=False, rotate=20,
            leg_loc="best", leg_bbox=None):
    drop = [c for c in cols if c not in df.columns or not df[c].notna().any()]
    if drop:
        print(f"  [skip] no data for: {', '.join(drop)}")
    cols = [c for c in cols if c not in drop]
    long = df.melt(id_vars=["database"], value_vars=cols, var_name="m",
                   value_name="value").dropna(subset=["value"])
    long["metric"] = long["m"].map(DISP)
    order = [DISP[c] for c in cols]
    sns.violinplot(data=long, x="metric", y="value", hue="database", order=order,
                   hue_order=HUE, palette=PAL, split=True, inner="box", cut=0,
                   density_norm="width", linewidth=0.8, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate,
                       ha="right" if rotate else "center",
                       rotation_mode="anchor")
    leg = ax.get_legend()
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, title="", loc=leg_loc, bbox_to_anchor=leg_bbox,
                  framealpha=0.9, ncol=2 if leg_bbox else 1)
    elif leg:
        leg.remove()
    style_axis(ax, grid_axis="y")


def _effects(ax, df, cols, xlabel=r"Cohen's $d$ (pre $-$ post)"):
    rows = []
    for c in cols:
        a = df.loc[df.database == PRE, c]
        b = df.loc[df.database == POST, c]
        d = _cohend(a, b)
        if c in LOWER_BETTER:          # flip so + = pre-cutoff better
            d = -d
        rows.append((DISP[c], d))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=[PRE_C if v >= 0 else POST_C for v in vals],
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(xlabel)
    style_axis(ax, grid_axis="x")


# --------------------------------------------------------------------------- #
def _save_custom(fig, name):
    out = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                       "publication_figures")
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg", "eps"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    print("Saved:", paths[0])
    return paths


def _letters(axes, fig_width):
    for ax, ltr in axes:
        ax.text(-0.07, 1.04, ltr, transform=ax.transAxes,
                fontsize=onpage_to_gen(8, fig_width),
                fontweight="bold", va="bottom", ha="right")


def build_protein_rna():
    df = pd.concat([_load_pr(os.path.join(PR_DIR, "b2021.db"), PRE),
                    _load_pr(os.path.join(PR_DIR, "a2021.db"), POST)],
                   ignore_index=True)
    acc = ["TM_r", "LDDT_r", "TM_c", "LDDT_c", "DockQ",
           "pTM_prot", "ipTM_rna", "pTM_rna"]
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(16.5, 11.0))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0],
                      hspace=0.5, wspace=0.28)
        axA = fig.add_subplot(gs[0, :])         # combined accuracy + confidence
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[1, 1])
        _violin(axA, df, acc, "Score", (0, 1.05), legend=True,
                leg_loc="lower center", leg_bbox=(0.5, 1.0))
        _violin(axB, df, ["PAE", "PAE_min"], r"PAE (Å)", (0, 32), rotate=0)
        _effects(axC, df, ["TM_r", "LDDT_r", "TM_c", "LDDT_c", "DockQ",
                           "pTM_prot", "ipTM_rna"])
        _letters([(axA, "A"), (axB, "B"), (axC, "C")], 16.5)
    return _save_custom(fig, "postpre_protein_rna")


def build_rna_only():
    df = pd.concat([_load_rna(os.path.join(RNA_DIR, "b2021_rna.db"), PRE),
                    _load_rna(os.path.join(RNA_DIR, "a2021_rna.db"), POST)],
                   ignore_index=True)
    acc = ["TM_r", "LDDT_r", "INF", "WC", "nonWC", "STACK"]
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15.0, 11.0))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0],
                      hspace=0.5, wspace=0.28)
        axA = fig.add_subplot(gs[0, :])         # combined RNA accuracy + INF
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[1, 1])
        _violin(axA, df, acc, "Score", (0, 1.05), legend=True,
                leg_loc="lower center", leg_bbox=(0.5, 1.0))
        _violin(axB, df, ["RMSD_r"], r"RMSD$_r$ (Å)", (0, 30), rotate=0)
        _effects(axC, df, ["TM_r", "LDDT_r", "INF", "WC", "nonWC", "STACK",
                           "RMSD_r"])
        _letters([(axA, "A"), (axB, "B"), (axC, "C")], 15.0)
    return _save_custom(fig, "postpre_rna_only")


def _draw_gap_effect(ax, pr, rna, cols=("TM_r", "LDDT_r")):
    """Cohen's d (pre - post, 95% CI) for protein-RNA vs RNA-only datasets."""
    cols = list(cols); x = np.arange(len(cols)); w = 0.38
    pr_d = [_d_ci(pr.loc[pr.database == PRE, c], pr.loc[pr.database == POST, c])
            for c in cols]
    rna_d = [_d_ci(rna.loc[rna.database == PRE, c], rna.loc[rna.database == POST, c])
             for c in cols]
    ebk = dict(ecolor="0.3", elinewidth=1.1, capsize=3)
    ax.bar(x - w / 2, [d[0] for d in pr_d], w, yerr=[d[1] for d in pr_d],
           color="#5B8C5A", label="protein–RNA", edgecolor="black",
           linewidth=0.4, error_kw=ebk)
    ax.bar(x + w / 2, [d[0] for d in rna_d], w, yerr=[d[1] for d in rna_d],
           color="#9B5DE5", label="single-chain RNA", edgecolor="black",
           linewidth=0.4, error_kw=ebk)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DISP[c] for c in cols])
    ax.set_ylabel(r"Cohen's $d$ (pre $-$ post)")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)
    style_axis(ax, grid_axis="y")


def build_combined(include_multimeric=False):
    """postpre_combined (protein-RNA | monomeric RNA; main text) or, with include_multimeric,
    postpre_combined_multichain (+ multi-chain RNA block; SI)."""
    pr = pd.concat([_load_pr(os.path.join(PR_DIR, "b2021.db"), PRE),
                    _load_pr(os.path.join(PR_DIR, "a2021.db"), POST)],
                   ignore_index=True)
    rna = pd.concat([_load_rna(os.path.join(RNA_DIR, "b2021_rna.db"), PRE),
                     _load_rna(os.path.join(RNA_DIR, "a2021_rna.db"), POST)],
                    ignore_index=True)
    multi = (pd.concat([_load_rna(os.path.join(MULTI_DIR, "b2021_rna_rna.db"), PRE),
                        _load_rna(os.path.join(MULTI_DIR, "a2021_rna_rna.db"), POST)],
                       ignore_index=True) if include_multimeric else None)

    # One shared x-axis: protein-RNA | single-chain RNA | multi-chain RNA blocks. Only keep
    # metrics that have data in BOTH cutoff groups (drops one-sided metrics).
    def _both(src, c):
        return (src.loc[src.database == PRE, c].notna().any()
                and src.loc[src.database == POST, c].notna().any())

    pr_cols = ["TM_prot", "pTM_prot", "LDDT_prot", "LDDT_c", "DockQ", "DockQ_Nr", "ipTM_rna", "LDDT_r", "pLDDT_rna", "TM_r", "pTM_rna"]
    rna_cols = ["LDDT_r", "pLDDT_rna", "TM_r", "pTM_rna"]
    items = [(f"PR::{c}", pr, c) for c in pr_cols if _both(pr, c)]
    boundary = len(items)
    items += [(f"RNA::{c}", rna, c) for c in rna_cols if _both(rna, c)]
    boundary2 = len(items)
    if include_multimeric:
        items += [(f"MULTI::{c}", multi, c) for c in rna_cols if _both(multi, c)]
    order = [k for k, _, _ in items]
    labels = [DISP[c] for _, _, c in items]
    n = len(items)

    long = pd.concat([
        src[["database", c]].dropna(subset=[c]).rename(columns={c: "value"}).assign(xkey=k)
        for k, src, c in items], ignore_index=True)

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(17.0, 11.0))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1.0], hspace=0.18)
        axA = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1], sharex=axA)

        # --- A: distributions ---
        sns.violinplot(data=long, x="xkey", y="value", hue="database", order=order,
                       hue_order=HUE, palette=PAL, split=True, inner="box", cut=0,
                       density_norm="width", linewidth=0.8, ax=axA)
        axA.set_xlabel(""); axA.set_ylabel("Score"); axA.set_ylim(0, 1.16)
        blocks = [(0, boundary, "Protein–RNA"), (boundary, boundary2, "Single-chain RNA")]
        if include_multimeric:
            blocks.append((boundary2, n, "Multi-chain RNA"))
        for b in [hi for _, hi, _ in blocks[:-1]]:
            axA.axvline(b - 0.5, color="0.4", ls="--", lw=1.6)
        for lo, hi, lab in blocks:
            axA.text((lo + hi - 1) / 2, 1.05, lab, ha="center", va="bottom",
                     fontsize=onpage_to_gen(7.5, 17.0), color="0.25")
        h, l = axA.get_legend_handles_labels()
        axA.legend(h, l, title="", loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=2, framealpha=0.9)
        style_axis(axA, grid_axis="y")
        axA.tick_params(axis="x", labelbottom=False)   # metric names are shown under panel B

        # --- B: effect size aligned under each violin (+ = pre-cutoff better) ---
        ds = [_cohend(src.loc[src.database == PRE, c], src.loc[src.database == POST, c])
              for _, src, c in items]
        bar_c = [PR_COL] * boundary + [MONO_COL] * (boundary2 - boundary) + [MULTI_COL] * (n - boundary2)
        axB.bar(range(n), ds, 0.7, color=bar_c, edgecolor="black", linewidth=0.4)
        for i, d in enumerate(ds):
            axB.text(i, d + (0.03 if d >= 0 else -0.03), f"{d:.2f}", ha="center",
                     va="bottom" if d >= 0 else "top",
                     fontsize=onpage_to_gen(6.5, 17.0), color="0.25")
        axB.axhline(0, color="0.4", lw=0.8)
        for b in [hi for _, hi, _ in blocks[:-1]]:
            axB.axvline(b - 0.5, color="0.4", ls="--", lw=1.6)
        axB.set_xticks(range(n))
        axB.set_xticklabels(labels, rotation=20, ha="right",
                            rotation_mode="anchor")
        axB.set_ylabel(r"Cohen's $d$ (pre $-$ post)")
        axB.set_ylim(min(0, min(ds) - 0.3), max(ds) + 0.8)   # headroom for the compartment labels
        axB.set_xlim(-0.6, n - 0.4)
        # one label per compartment (instead of a legend), in the colour of its bars
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(axB.transData, axB.transAxes)
        block_col = {"Protein–RNA": PR_COL, "Single-chain RNA": MONO_COL, "Multi-chain RNA": MULTI_COL}
        for lo, hi, lab in blocks:
            axB.text((lo + hi - 1) / 2, 0.96, lab, transform=tr, ha="center", va="top",
                     fontsize=onpage_to_gen(7.0, 17.0), color=block_col[lab], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        style_axis(axB, grid_axis="y")
        _letters([(axA, "A"), (axB, "B")], 17.0)
    return _save_custom(fig, "postpre_combined_multichain" if include_multimeric else "postpre_combined")


def build_gap():
    pr = pd.concat([_load_pr(os.path.join(PR_DIR, "b2021.db"), PRE),
                    _load_pr(os.path.join(PR_DIR, "a2021.db"), POST)],
                   ignore_index=True)
    rna = pd.concat([_load_rna(os.path.join(RNA_DIR, "b2021_rna.db"), PRE),
                     _load_rna(os.path.join(RNA_DIR, "a2021_rna.db"), POST)],
                    ignore_index=True)

    def draw_dissoc(ax):
        groups = ["pTM_prot", "TM_prot", "LDDT_prot", "ipTM_rna", "TM_r", "LDDT_r"]
        x = np.arange(len(groups)); w = 0.38
        pre = [_median_ci(pr.loc[pr.database == PRE, c]) for c in groups]
        post = [_median_ci(pr.loc[pr.database == POST, c]) for c in groups]
        pre_m = [p[0] for p in pre]; post_m = [p[0] for p in post]
        pre_err = np.array([[p[1] for p in pre], [p[2] for p in pre]])
        post_err = np.array([[p[1] for p in post], [p[2] for p in post]])
        ebk = dict(ecolor="0.3", elinewidth=1.1, capsize=3)
        ax.bar(x - w / 2, pre_m, w, yerr=pre_err, color=PRE_C, label=PRE,
               edgecolor="black", linewidth=0.4, error_kw=ebk)
        ax.bar(x + w / 2, post_m, w, yerr=post_err, color=POST_C, label=POST,
               edgecolor="black", linewidth=0.4, error_kw=ebk)
        ax.axvline(2.5, color="0.6", ls=":", lw=1.4)   # protein | RNA divider
        ax.text(1.0, 1.02, "protein", ha="center",
                fontsize=onpage_to_gen(7, 18.6), color="0.3")
        ax.text(4.0, 1.02, "RNA", ha="center",
                fontsize=onpage_to_gen(7, 18.6), color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels([DISP[c] for c in groups], rotation=25, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("Median score"); ax.set_ylim(0, 1.1)
        # Legend as one line below the panel (aligned with panels B and C).
        ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, frameon=False)
        style_axis(ax, grid_axis="y")

    def draw_combined_d(ax):
        _draw_gap_effect(ax, pr, rna)

    def draw_calib(ax):
        d = pr.dropna(subset=["ipTM_rna", "TM_r"])
        for lab, c in ((PRE, PRE_C), (POST, POST_C)):
            s = d[d.database == lab]
            ax.scatter(s["ipTM_rna"], s["TM_r"], s=20, color=c, alpha=0.5,
                       edgecolor="none", label=lab)
        ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1.2)
        ax.set_xlabel(r"ipTM$_r$ (confidence)")
        ax.set_ylabel(r"TM$_r$ (actual)")
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
        ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, frameon=False, markerscale=2.8)
        style_axis(ax, grid_axis="both")

    out = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                       "publication_figures")
    fig = PublicationFigure(ncols=3, panel_size=(6.2, 5.2), outdir=out)
    fig.add_panel(draw_dissoc)
    fig.add_panel(draw_combined_d)
    fig.add_panel(draw_calib)
    return fig.save("postpre_gap")


if __name__ == "__main__":
    build_protein_rna()
    build_rna_only()
    build_gap()
    build_combined()
    build_combined(include_multimeric=True)
