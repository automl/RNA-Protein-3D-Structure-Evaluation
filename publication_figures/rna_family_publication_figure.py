"""
RNA-family dependence of AlphaFold3 RNA LDDT — Nature-style two-panel figures.

For protein-RNA complexes (database/rbpDatabase.db) and RNA-only complexes
(data/compare_ions/a2021_rna_ONLY.db), reusing the parsing from
analyzeRNAFamilyMSA_singleDB.py. Reads the tables directly, so it does not depend
on database/config.json. Colours follow plots.plotCreator.PH_RANGE_COLORS.

Emits two figures (each panel A = protein-RNA, B = RNA-only):
  rnafamily_msa_heatmap   -- RNAFamily x MSA-size bin, % change from overall mean
                             (the requested heatmap; sparse because most samples
                             are unannotated and each family sits in one MSA bin)
  rnafamily_lddt          -- per-family mean RNA LDDT +/- s.e.m. vs overall mean
                             (a more direct view of family-level quality)

Usage:  python -m evaluation.rna_family_publication_figure
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, style_axis, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS
from evaluation.analyzeRNAFamilyMSA_singleDB import (
    process_rna_family, process_rna_msa_size, process_metric_value)

METRIC = "RNA_LDDT_allAtom"
MIN_N = 1                       # families with at least this many samples (all)
MSA_BINS = [0, 1, 20, 100, 1000, float("inf")]
MSA_LABELS = ["0-1", "1-20", "20-100", "100-1000", "+1000"]

BAR_C = PH_RANGE_COLORS["5-6"]   # blue bars
NONE_C = "0.7"                    # grey for the unannotated reference
# Diverging map from the shared palette: blue (below) -> white -> red (above), i.e.
# red = positive -- matching every other heatmap in the paper (Figure2 D, rna_secondary F,
# pub_fig4 F, bond_metrics E) for one consistent colormap polarity.
DIV_CMAP = LinearSegmentedColormap.from_list(
    "bwr", [PH_RANGE_COLORS["5-6"], "white", PH_RANGE_COLORS["<5"]])

_SOURCES = [
    ("protein-RNA", os.path.join(_REPO_ROOT, "database", "rbpDatabase.db"),
     "exp_protein_rna", "pred_protein_rna"),
    ("RNA-only", os.path.join(_REPO_ROOT, "data", "compare_ions", "a2021_rna_ONLY.db"),
     "exp_rna_rna", "pred_rna_rna"),
]


def _load(db, exp, pred):
    con = sqlite3.connect(db)
    df = pd.read_sql_query(
        f"SELECT e.PDBId, e.RNAFamily, p.RNA_msa_size, p.{METRIC} "
        f"FROM {exp} e JOIN {pred} p ON e.PDBId = p.exp_db_id", con)
    con.close()
    df["fam"] = df["RNAFamily"].apply(process_rna_family)
    df["msa"] = df["RNA_msa_size"].apply(process_rna_msa_size)
    df["val"] = df[METRIC].apply(process_metric_value)
    return df[df["val"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure 1: RNAFamily x MSA-size heatmap (% change from overall mean)
# --------------------------------------------------------------------------- #
def _heatmap_panel(df, ax):
    overall = df["val"].mean()
    # keep every family with enough samples, including the unannotated "None"
    # group (as in the original RNAFamily_MSA_size_heatmap)
    keep = df["fam"].value_counts()
    named = df[df["fam"].isin(keep[keep >= MIN_N].index)].copy()
    named["MSA"] = pd.cut(named["msa"], bins=MSA_BINS, labels=MSA_LABELS)

    piv = named.pivot_table(values="val", index="fam", columns="MSA",
                            aggfunc="mean", observed=False)
    cnt = named.pivot_table(values="val", index="fam", columns="MSA",
                            aggfunc="count", observed=False)
    pct = (piv - overall) / overall * 100.0
    # drop all-empty MSA columns so the rendered x-labels line up with the cells,
    # and use plain string labels to avoid categorical-index pitfalls in seaborn
    keep_cols = [c for c in pct.columns if pct[c].notna().any()]
    pct = pct[keep_cols].copy()
    cnt = cnt[keep_cols].copy()
    pct.columns = [str(c) for c in keep_cols]
    cnt.columns = [str(c) for c in keep_cols]
    order = pct.mean(axis=1).sort_values(ascending=False).index
    pct, cnt = pct.loc[order], cnt.loc[order]

    annot = pct.astype(object).copy()
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v, c = pct.iloc[i, j], cnt.iloc[i, j]
            annot.iloc[i, j] = (f"{int(c)}|{'+' if v >= 0 else ''}{v:.0f}%"
                                if pd.notna(v) and pd.notna(c) else "")
    vmax = np.nanmax(np.abs(pct.values)) if np.isfinite(pct.values).any() else 1.0
    sns.heatmap(pct, annot=annot, fmt="", cmap=DIV_CMAP, center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
                # cell strings run to 7 glyphs ("312|+1%"); shrink only the cell
                # annotation just enough that the widest fits inside one cell
                annot_kws={"fontsize": onpage_to_gen(5.5, 12.8)}, ax=ax,
                cbar_kws={"label": r"LDDT$_r$ % change vs overall mean",
                          "fraction": 0.046, "pad": 0.04})
    ax.set_xlabel("RNA MSA size")
    ax.set_ylabel("RNA family")
    ax.tick_params(axis="y", labelsize=onpage_to_gen(7, 12.8), rotation=0)
    ax.tick_params(axis="x", labelsize=onpage_to_gen(7, 12.8))


def build_heatmap():
    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    fig = PublicationFigure(ncols=2, panel_size=(6.4, 5.0), outdir=outdir)
    for name, db, exp, pred in _SOURCES:
        df = _load(db, exp, pred)
        print(f"[heatmap] {name}: n={len(df)}, named families "
              f">={MIN_N}: {(df[df.fam!='None'].fam.value_counts() >= MIN_N).sum()}")
        fig.add_panel(lambda ax, d=df: _heatmap_panel(d, ax), title=None)
    return fig.save("rnafamily_msa_heatmap")


# --------------------------------------------------------------------------- #
# Figure 2: per-family mean RNA LDDT (more direct family-dependence view)
# --------------------------------------------------------------------------- #
def _family_quality_panel(df, ax):
    overall = df["val"].mean()
    g = (df[df["fam"] != "None"].groupby("fam")["val"]
         .agg(["mean", "std", "count"]))
    g = g[g["count"] >= MIN_N].sort_values("mean")
    means = g["mean"].values
    sems = (g["std"] / np.sqrt(g["count"])).fillna(0.0).values
    counts = g["count"].values.astype(int)
    labels = list(g.index)

    # unannotated reference at the bottom
    none_vals = df[df["fam"] == "None"]["val"]
    y = np.arange(len(labels) + 1)
    all_means = np.append(none_vals.mean(), means)
    all_sems = np.append(none_vals.std() / np.sqrt(max(len(none_vals), 1)), sems)
    all_counts = np.append(len(none_vals), counts)
    colors = [NONE_C] + [BAR_C] * len(labels)
    ylabels = ["unannotated"] + labels

    ax.barh(y, all_means, xerr=all_sems, color=colors, alpha=0.9,
            edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="0.25", elinewidth=0.9, capsize=2))
    ax.axvline(overall, color="0.35", ls="--", lw=1.0,
               label=f"overall mean = {overall:.2f}")
    for yi, c in zip(y, all_counts):
        ax.text(0.01, yi, f"n={c}", va="center", ha="left",
                fontsize=onpage_to_gen(6.5, 12.0),
                color="white" if yi == 0 else "0.15")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=onpage_to_gen(7, 12.0))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"LDDT$_r$ (all-atom)")
    ax.legend(fontsize=onpage_to_gen(7, 12.0), loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False)
    style_axis(ax, grid_axis="x")


def build_family_quality():
    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    fig = PublicationFigure(ncols=2, panel_size=(6.0, 5.0), outdir=outdir)
    for name, db, exp, pred in _SOURCES:
        df = _load(db, exp, pred)
        print(f"\n[family rLDDT] {name}: overall={df['val'].mean():.3f}")
        g = (df[df.fam != "None"].groupby("fam")["val"]
             .agg(["mean", "count"]))
        for fam, row in g[g["count"] >= MIN_N].sort_values("mean").iterrows():
            print(f"  {fam:<22} n={int(row['count']):<3} mean={row['mean']:.3f}")
        fig.add_panel(lambda ax, d=df: _family_quality_panel(d, ax), title=None)
    return fig.save("rnafamily_lddt")


if __name__ == "__main__":
    build_heatmap()
    build_family_quality()
