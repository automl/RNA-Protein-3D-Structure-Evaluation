"""
When and where are MSA_r, MSA_p, Tmpl_p and Tmpl_c necessary?  (Nature-style)

One figure, four panels (all data under
data/compareDatabases/noMSA/...):
  A  RNA-only TM_r by MSA_r depth | RNA length, with vs without the RNA MSA
     (RNAonly_noMSA: af3_rna.db, noMSA_rna.db).
  B  Protein-RNA TM_r by MSA_r depth | RNA length, with vs without the RNA MSA,
     holding protein MSA and both templates on (factorial
     +/-rMSA_+pMSA_+pTemp_+cTemp).
  C  Standardised effect (Cohen's d, + = improvement) of each of the four
     ablations on each metric, from the 16-way factorial.
  D  Protein-RNA TM_c by RNA length class for four input regimes
     (neither / +MSA_p / +templates / both), showing how MSA and templates
     combine differently with RNA length.

Usage:  python -m evaluation_db.factor_necessity_figure
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
from matplotlib.transforms import ScaledTranslation

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS

# Larger, Nature-style fonts for a multi-panel figure (cf. plotCreator.py).
PUB_RC.update({
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})
from utils.plotting_utils import (process_tm, process_rmsd, process_rna_length,
                                  process_msa_sizes)

_BASE = os.path.join(_REPO_ROOT, "data", "compareDatabases", "noMSA")
A_DIR = os.path.join(_BASE, "RNAonly_noMSA")
F_DIR = os.path.join(_BASE, "single_protANDRNA_template_remove_experiment")

COL_HI = PH_RANGE_COLORS["5-6"]    # blue   — MSA present
COL_LO = PH_RANGE_COLORS["6-7"]    # orange — MSA removed
HEAT = LinearSegmentedColormap.from_list("wb", ["white", PH_RANGE_COLORS["5-6"]])
# Panel D regimes: single-factor additions to the all-off baseline (each is one
# database, identified by its rMSA/pMSA/pTemp/cTemp sign string).
_R_NEITHER = "neither"
_R_MSAR = r"$+$MSA$_r$"
_R_MSAP = r"$+$MSA$_p$"
_R_TEMP = r"$+$Tmpl$_p$$+$Tmpl$_c$"
_R_BOTH = r"$+$MSA$_p$$+$Tmpl$_p$$+$Tmpl$_c$"
D_ORDER = [_R_NEITHER, _R_MSAP, _R_TEMP, _R_BOTH]   # +MSA_r dropped: only ~20/75 have an RNA MSA
D_SIGNS = {"----": _R_NEITHER, "-+--": _R_MSAP,
           "--++": _R_TEMP, "-+++": _R_BOTH}
D_COLORS = ["0.6", PH_RANGE_COLORS["5-6"], PH_RANGE_COLORS["6-7"], PH_RANGE_COLORS["7-8"]]
E_COLORS = ["0.6", PH_RANGE_COLORS[">8"], PH_RANGE_COLORS["5-6"], PH_RANGE_COLORS["6-7"]]
# Panel E template conditions (pTemp, cTemp): explicit, baseline both-off
E_CONDS = [(0, 0, r"$-$Tmpl$_p$$-$Tmpl$_c$"), (1, 0, r"$+$Tmpl$_p$"),
           (0, 1, r"$+$Tmpl$_c$"), (1, 1, r"$+$Tmpl$_p$$+$Tmpl$_c$")]

LEN_BINS = [0, 20, 200, float("inf")]
LEN_LABELS = ["0-20", "20-200", "200+"]
MSA_BINS = [-float("inf"), 1, 1000, float("inf")]
MSA_LABELS = ["0", "1-1000", "+1000"]

TMr, TMc = r"TM$_r$", r"TM$_c$"


def _add_classes(df):
    df["RNALength"] = df["RNALength"].apply(process_rna_length)
    df["length_class"] = pd.cut(df["RNALength"], bins=LEN_BINS, labels=LEN_LABELS)
    df["MSAsize"] = df.apply(
        lambda r: process_msa_sizes(r, "RNA_msa_size", "RNAChainIDs", "RNALength"),
        axis=1)
    df["msa_class"] = pd.cut(df["MSAsize"], bins=MSA_BINS, labels=MSA_LABELS)
    df["combined_class"] = (df["msa_class"].astype(str) + "\n"
                            + df["length_class"].astype(str))
    return df


def _rna_msa_depth(cell):
    """Max RNA MSA depth from a RNA_msa_size dict cell, e.g. {"A": 4} -> 4, {"A": 1} -> 1."""
    try:
        d = ast.literal_eval(str(cell))
        if isinstance(d, dict) and d:
            return max(float(v) for v in d.values())
    except (ValueError, SyntaxError):
        pass
    return 0.0


def _load_pair(pairs, table, msa_gt1_from=None):
    """pairs: list of (path, label). Combined box-ready frame with TM_r.

    If ``msa_gt1_from`` is a label, both boxes are restricted to the SAME paired set of
    structures: present in every database of the pair, with RNA_msa_size depth > 1 in
    that (with-MSA) database, and a finite TM_r in every database. The MSA-depth | RNA-
    length class is taken from the with-MSA database and shared by both boxes, so paired
    structures land in the same x-group and none fall in the MSA-depth "0" bin."""
    raw = {}
    for path, label in pairs:
        con = sqlite3.connect(path)
        raw[label] = pd.read_sql_query(f"SELECT * FROM {table}", con)
        con.close()

    if msa_gt1_from is None or msa_gt1_from not in raw:
        frames = []
        for path, label in pairs:
            df = _add_classes(raw[label].copy())
            df[TMr] = df["RNA_TM"].apply(process_tm)
            df["database"] = label
            frames.append(df[[TMr, "combined_class", "length_class", "database"]])
        return pd.concat(frames, ignore_index=True)

    # --- paired mode: classes from the with-MSA db, shared by both boxes ---
    w = raw[msa_gt1_from].copy()
    w["_id"] = w["exp_db_id"].astype(str)
    w["_depth"] = w["RNA_msa_size"].apply(_rna_msa_depth)
    w["RNALength"] = w["RNALength"].apply(process_rna_length)
    w["length_class"] = pd.cut(w["RNALength"], bins=LEN_BINS, labels=LEN_LABELS)
    w["msa_class"] = pd.cut(w["_depth"], bins=MSA_BINS, labels=MSA_LABELS)
    w["combined_class"] = w["msa_class"].astype(str) + "\n" + w["length_class"].astype(str)
    w = w.drop_duplicates("_id", keep="first")
    combined_map = dict(zip(w["_id"], w["combined_class"]))
    length_map = dict(zip(w["_id"], w["length_class"].astype(str)))

    keep = set(w.loc[w["_depth"] > 1, "_id"])                 # MSA present (depth > 1) in with-MSA db
    for label, df in raw.items():                             # present + valid TM_r in every db
        d = df.copy()
        d["_id"] = d["exp_db_id"].astype(str)
        d[TMr] = d["RNA_TM"].apply(process_tm)
        keep &= set(d.loc[d[TMr].notna(), "_id"])

    frames = []
    for path, label in pairs:
        df = raw[label].copy()
        df["_id"] = df["exp_db_id"].astype(str)
        df = df[df["_id"].isin(keep)].copy()
        df[TMr] = df["RNA_TM"].apply(process_tm)
        df["combined_class"] = df["_id"].map(combined_map)
        df["length_class"] = df["_id"].map(length_map)
        df["database"] = label
        frames.append(df[[TMr, "combined_class", "length_class", "database"]])
    return pd.concat(frames, ignore_index=True)


def _combined_order(df):
    order = [f"{m}\n{ln}" for m in MSA_LABELS for ln in LEN_LABELS]
    return [g for g in order if g in set(df["combined_class"])]


def _tmpl_count(cell):
    """Number of PDB templates in a list-valued column (e.g. ['6F5O','6QWL']->2)."""
    try:
        v = ast.literal_eval(str(cell))
        return len(v) if isinstance(v, (list, tuple)) else 0
    except (ValueError, SyntaxError):
        return 0


def _load_factorial():
    frames = []
    for f in sorted(x for x in os.listdir(F_DIR) if x.endswith(".db")):
        con = sqlite3.connect(os.path.join(F_DIR, f))
        df = pd.read_sql_query("SELECT * FROM pred_protein_rna", con)
        con.close()
        lab = f[:-3]
        for fac in ("rMSA", "pMSA", "pTemp", "cTemp"):
            df[fac] = 1 if ("+" + fac) in lab else 0
        df["RNALength"] = df["RNALength"].apply(process_rna_length)
        df["length_class"] = pd.cut(df["RNALength"], bins=LEN_BINS, labels=LEN_LABELS)
        df[TMc] = df["Complex_TM"].apply(process_tm)
        df[TMr] = df["RNA_TM"].apply(process_tm)
        df["Complex_RMSD"] = df["Complex_RMSD"].apply(process_rmsd)
        for m in ("Complex_LDDT_allAtom", "RNA_LDDT_allAtom", "Dockq"):
            df[m] = pd.to_numeric(df[m], errors="coerce")
        df["pT_n"] = (df["Protein_PDB_Templates"].apply(_tmpl_count)
                      if "Protein_PDB_Templates" in df.columns else 0)
        df["cT_n"] = (df["RBP_PDB_Templates"].apply(_tmpl_count)
                      if "RBP_PDB_Templates" in df.columns else 0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _effects(b):
    metrics = [("Complex_TM", r"TM$_c$"), ("Complex_LDDT_allAtom", r"LDDT$_c$"),
               ("Dockq", "DockQ"), ("Complex_RMSD", r"RMSD$_c$"),
               ("RNA_TM", r"TM$_r$"), ("RNA_LDDT_allAtom", r"LDDT$_r$")]
    facs = [("rMSA", r"MSA$_r$"), ("pMSA", r"MSA$_p$"),
            ("pTemp", r"Tmpl$_p$"), ("cTemp", r"Tmpl$_c$")]

    def cohend(p, m):
        p = pd.to_numeric(pd.Series(p), errors="coerce").to_numpy(float)
        m = pd.to_numeric(pd.Series(m), errors="coerce").to_numpy(float)
        p, m = p[np.isfinite(p)], m[np.isfinite(m)]
        nx, ny = len(p), len(m)
        sp = np.sqrt(((nx - 1) * p.var(ddof=1) + (ny - 1) * m.var(ddof=1)) / (nx + ny - 2))
        return (p.mean() - m.mean()) / sp if sp > 0 else 0.0

    mat = np.zeros((len(metrics), len(facs)))
    for i, (col, _) in enumerate(metrics):
        for j, (fac, _) in enumerate(facs):
            d = cohend(b.loc[b[fac] == 1, col].values, b.loc[b[fac] == 0, col].values)
            mat[i, j] = -d if col == "Complex_RMSD" else d
    return pd.DataFrame(mat, index=[m[1] for m in metrics],
                        columns=[f[1] for f in facs])


def _signs_str(r):
    """Sign string for one row, order: MSA_r MSA_p Tmpl_p Tmpl_c (e.g. '+-+-')."""
    return "".join("+" if r[c] == 1 else "-"
                   for c in ("rMSA", "pMSA", "pTemp", "cTemp"))


def _full_label(signs):
    """'++++' -> '+MSA_r+MSA_p+Tmpl_p+Tmpl_c' (with proper +/- per factor)."""
    names = [r"MSA$_r$", r"MSA$_p$", r"Tmpl$_p$", r"Tmpl$_c$"]
    return "".join(("+" if s == "+" else "−") + n for s, n in zip(signs, names))


def build():
    a = _load_pair([(os.path.join(A_DIR, "af3_rna.db"), "with MSA"),
                    (os.path.join(A_DIR, "noMSA_rna.db"), "no MSA")], "pred_rna_rna",
                   msa_gt1_from="with MSA")
    bdf = _load_pair([(os.path.join(F_DIR, "+rMSA_+pMSA_+pTemp_+cTemp.db"), "with MSA"),
                      (os.path.join(F_DIR, "-rMSA_+pMSA_+pTemp_+cTemp.db"), "no MSA")],
                     "pred_protein_rna", msa_gt1_from="with MSA")
    fac = _load_factorial()
    fac["signs"] = fac.apply(_signs_str, axis=1)
    eff = _effects(fac)
    print("Cohen's d (factorial, + = improvement):\n", eff.round(2))

    # Panel D: single-factor regimes (each = one database) added to all-off baseline
    fac["regime"] = fac["signs"].map(D_SIGNS).fillna("mixed")
    d_df = fac[fac.regime != "mixed"].dropna(subset=[TMc])

    def _box(ax, data, order, hue_order, palette, x, y, leg_loc, leg_ncol=1,
             leg_bbox=None):
        sns.boxplot(data=data, x=x, y=y, hue="database" if "database" in data else "regime",
                    order=order, hue_order=hue_order, palette=palette,
                    fliersize=1.5, linewidth=0.7, ax=ax)
        ax.set_ylim(0, 1.0)
        ax.legend(title="", fontsize=onpage_to_gen(7, 18.0), loc=leg_loc, ncol=leg_ncol,
                  framealpha=0.9, bbox_to_anchor=leg_bbox)
        style_axis(ax, grid_axis="y")

    def drawA(ax):
        d = a.dropna(subset=[TMr])
        _box(ax, d, _combined_order(d), ["with MSA", "no MSA"], [COL_HI, COL_LO],
             "combined_class", TMr, "upper center", leg_ncol=2, leg_bbox=(0.5, -0.33))
        ax.set_xlabel(r"MSA$_r$ depth | RNA length")
        ax.set_ylabel(TMr + "\n(single-chain RNA)")
        ax.tick_params(axis="x", labelsize=onpage_to_gen(7, 18.0))

    def drawB(ax):
        d = bdf.dropna(subset=[TMr])
        _box(ax, d, _combined_order(d), ["with MSA", "no MSA"], [COL_HI, COL_LO],
             "combined_class", TMr, "upper center", leg_ncol=2, leg_bbox=(0.5, -0.33))
        ax.set_xlabel(r"MSA$_r$ depth | RNA length")
        ax.set_ylabel(TMr + "  (protein-RNA)")
        ax.tick_params(axis="x", labelsize=onpage_to_gen(7, 18.0))

    def drawC(ax):
        sns.heatmap(eff, annot=True, fmt="+.2f", cmap=HEAT, vmin=0, vmax=1.25,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": onpage_to_gen(6.5, 18.0)},
                    cbar_kws={"label": "Effect size (Cohen's d)", "fraction": 0.046,
                              "pad": 0.04}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=onpage_to_gen(7, 18.0), rotation=0)
        ax.tick_params(axis="x", labelsize=onpage_to_gen(7, 18.0))

    def drawD(ax):
        # the box groups fill the interior, so the legend sits below the axis
        _box(ax, d_df, LEN_LABELS, D_ORDER, D_COLORS, "length_class", TMc,
             "upper center", leg_ncol=2, leg_bbox=(0.5, -0.24))
        ax.set_xlabel("RNA length")
        ax.set_ylabel(TMc + "  (protein-RNA)")

    # Panels E/F: templates (Complex RMSD)
    def _sem(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    fac["tmpl"] = np.where((fac.pTemp == 1) | (fac.cTemp == 1),
                           "with template", "no template")

    def drawE(ax):
        for i, (pt, ct, lab) in enumerate(E_CONDS):
            sub = fac.loc[(fac.pTemp == pt) & (fac.cTemp == ct), "Complex_RMSD"]
            ax.bar(i, sub.mean(), 0.6, yerr=_sem(sub), color=E_COLORS[i],
                   edgecolor="black", linewidth=0.3,
                   error_kw=dict(ecolor="0.3", elinewidth=0.8, capsize=3))
        ax.set_xticks(range(len(E_CONDS)))
        ax.set_xticklabels([])   # drawn manually below (as text) so the rightward
                                 # shift survives tight_layout; E_DX is in data units
        E_DX = 0.18
        for _i, _c in enumerate(E_CONDS):
            ax.text(_i + E_DX, -0.02, _c[2], transform=ax.get_xaxis_transform(),
                    rotation=25, ha="right", va="top", rotation_mode="anchor",
                    fontsize=onpage_to_gen(7, 18.0))
        ax.set_ylabel(r"RMSD$_c$ (Å)")
        ax.set_xlabel("")
        style_axis(ax, grid_axis="y")

    # Panel F: effect of template count (0-4) on TM_c, per template type.
    # Grouped bars; the per-count sample size n is printed above each column.
    def drawF(ax):
        counts = [0, 1, 2, 3, 4]
        w = 0.4
        series = [("pT_n", "pTemp", PH_RANGE_COLORS["5-6"], r"Tmpl$_p$"),
                  ("cT_n", "cTemp", PH_RANGE_COLORS["6-7"], r"Tmpl$_c$")]
        top = 0.0
        for k, (ncol, oncol, color, lab) in enumerate(series):
            sub = fac[fac[oncol] == 1]
            xs = np.array(counts) + (k - 0.5) * w
            for c, xpos in zip(counts, xs):
                vals = pd.to_numeric(sub.loc[sub[ncol] == c, TMc], errors="coerce")
                vals = vals[np.isfinite(vals)]
                n = int(len(vals))
                if n == 0:
                    continue
                m = vals.mean()
                e = _sem(vals)
                ax.bar(xpos, m, w, yerr=e, color=color,
                       label=lab if c == counts[0] else None,
                       edgecolor="black", linewidth=0.3,
                       error_kw=dict(ecolor="0.3", elinewidth=0.8, capsize=2))
                ax.text(xpos, m + e + 0.012, f"{n}", ha="center", va="bottom",
                        fontsize=onpage_to_gen(6.5, 18.0), rotation=90, color="0.25")
                top = max(top, m + e)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts])
        # headroom above the tallest bar so the vertical n-labels are not clipped
        ax.set_ylim(0.7, min(1.05, top + 0.1))
        ax.set_xlabel("Templates (count)")
        ax.set_ylabel(TMc + "  (protein-RNA)")
        # legend below the plot (matches the other panels)
        ax.legend(title="", fontsize=onpage_to_gen(7, 18.0), loc="upper center", ncol=2,
                  framealpha=0.9, bbox_to_anchor=(0.5, -0.24))
        style_axis(ax, grid_axis="y")

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    pf = PublicationFigure(ncols=3, panel_size=(6.0, 5.0), outdir=outdir)
    for draw in (drawA, drawB, drawC, drawD, drawE, drawF):
        pf.add_panel(draw, title=None)
    figobj = pf.build()
    # visual test: centre panel C (heatmap) on the A/B block midline, where the
    # block spans from the plot top down to below the A/B legends (~ -0.40 axes).
    axes = figobj.axes
    if len(axes) >= 7:
        pA, pC = axes[0].get_position(), axes[2].get_position()
        mid = ((pA.y0 + pA.height) + (pA.y0 - 0.28 * pA.height)) / 2
        dy = mid - (pC.y0 + pC.height / 2)
        for k in (2, 6):   # panel C axes + its colorbar
            p = axes[k].get_position()
            axes[k].set_position([p.x0, p.y0 + dy, p.width, p.height])
        # keep panel C's letter at its original top-row height (aligned with A/B),
        # not dragged down with the heatmap
        for _t in axes[2].texts:
            if _t.get_text() == "C":
                _x, _y = _t.get_position()
                _t.set_position((_x, _y - dy / pC.height))
    paths = []
    for ext in ("png", "pdf", "svg", "eps"):
        out = os.path.join(pf.results_dir, f"factor_necessity.{ext}")
        figobj.savefig(out, dpi=600, bbox_inches="tight")
        paths.append(out)
    plt.close(figobj)
    print("Publication figure saved:", paths[0])
    return paths


def build_full_conditions():
    """Standalone supplementary: TM_c by RNA length for all 16 input conditions,
    labelled by the MSA_r / MSA_p / Tmpl_p / Tmpl_c sign string."""
    fac = _load_factorial()
    fac["signs"] = fac.apply(_signs_str, axis=1)
    fac["label"] = fac["signs"].apply(_full_label)
    sign_order = sorted(fac["signs"].unique())          # '++++' first, '----' last
    label_order = [_full_label(s) for s in sign_order]
    palette = sns.color_palette("husl", len(label_order))

    def draw(ax):
        d = fac.dropna(subset=[TMc])
        sns.boxplot(data=d, x="length_class", y=TMc, hue="label",
                    order=LEN_LABELS, hue_order=label_order, palette=palette,
                    fliersize=1.0, linewidth=0.4, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("RNA length")
        ax.set_ylabel(TMc + "  (protein-RNA)")
        ax.legend(title="", fontsize=onpage_to_gen(7, 8.5), ncol=1, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), handlelength=1.2)
        style_axis(ax, grid_axis="y")

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    fig = PublicationFigure(ncols=1, panel_size=(8.5, 5.0), outdir=outdir,
                            letters=False)
    fig.add_panel(draw, title=None)
    return fig.save("factor_necessity_full_conditions_supp")


if __name__ == "__main__":
    build()
    build_full_conditions()
