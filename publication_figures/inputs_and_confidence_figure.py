"""Inputs and confidence: one 3x3 publication figure.

Merges three earlier figures so the paper carries one figure instead of three. Nothing is
recomputed differently -- the data helpers are imported from the modules that built the
originals, which are left untouched.

  A    RNA-only TM_r with and without the RNA MSA, by MSA depth and RNA length
       (msa_temp_influence.py panel A)
  B-E  the 16-way input ablation (data/nar_msa_template_ablation_20261007):
       B  removing the RNA MSA, by RNA length
       C  every input, effect on the RNA (dTM_r, 95% CI)
       D  MSA_r x MSA_p interaction -- the RNA MSA matters more when the protein MSA is gone
       E  input x metric, paired standardised effect
  F-I  AF3 confidence (pub_fig4 panels B, C, D, F):
       F  pTM_r vs TM_r per RNA chain, split by RNA MSA
       G  mean pTM_r and TM_r by RNA MSA depth
       H  ipTM_r by audit category
       I  confidence metric vs ground truth, Spearman rho

Effects in B-E are paired within target: for one input the other three give 8 background
combinations, so the run with the input present is compared with the otherwise identical run
without it, and only targets that carry the input are included.
"""
import os
import sys

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The confidence panels read the a2021 audit set through RNAMetrics, which takes its database
# from database/config.json. That file belongs to whichever processing run touched it last, so
# this figure writes its own config and selects it with EVAL_CONFIG_NAME (startConfig.py honours
# the variable). Done before any repo import, because StartConfig resolves the path at import.
CONF_DB = os.path.join(_REPO, "data", "nar_msa_template_ablation_20261007", "a2021.db")
_CONF_NAME = "config_inputs_and_confidence.json"
if os.path.exists(CONF_DB):
    import json as _json
    with open(os.path.join(_REPO, "database", _CONF_NAME), "w") as _fh:
        _json.dump({"classification": "protein_rna",
                    "base_data_path": os.path.dirname(CONF_DB),
                    "db_path": CONF_DB}, _fh, indent=4)
    os.environ["EVAL_CONFIG_NAME"] = _CONF_NAME

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS, PlotCreator
from publication_figures.ablation_data import AB, FACTORS, availability, load_factorial
from publication_figures.ablation_effects import paired_differences, summarise
from publication_figures.msa_temp_influence import _load_pair
from evaluation.bundle_paths import ALL_DB, PERIOD_SET

# single-chain RNAs predicted with and without the RNA MSA (post-cutoff, all lengths)
SC_DIR = os.path.join(ALL_DB, PERIOD_SET["post"])

PUB_RC.update({"axes.labelsize": 12, "axes.titlesize": 13,
               "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 12})

C_1V1, C_MULTI = PH_RANGE_COLORS["5-6"], PH_RANGE_COLORS["6-7"]
MSA_COLORS = {"with MSA": "#1f77b4", "no MSA": "#ff7f0e",
              "+MSA": "#1f77b4", "-MSA": "#ff7f0e"}
HEAT = LinearSegmentedColormap.from_list("eff", [PH_RANGE_COLORS["<5"], "white",
                                                 PH_RANGE_COLORS["5-6"]])
FACTOR_LABEL = {"rMSA": r"MSA$_r$", "pMSA": r"MSA$_p$",
                "pTemp": r"Tmpl$_p$", "cTemp": r"Tmpl$_c$"}
METRICS = [("Complex_TM", r"TM$_c$"), ("RNA_TM", r"TM$_r$"),
           ("Complex_LDDT_allAtom", r"LDDT$_c$"), ("RNA_LDDT_allAtom", r"LDDT$_r$"),
           ("DockQ_Nr", r"DockQ$_{Nr}$"), ("Complex_RMSD", r"RMSD$_c$")]
LEN_BINS, LEN_LABELS = [20, 200], ["<20", "20-200", ">200"]


def style_heatmap(ax, mat, row_lbls, col_lbls, cbar_label, fig_w=16.5,
                  vmin=-1.0, vmax=1.0, cmap="RdBu_r", bold_dominant=True):
    """Heatmap drawn in the style of pub_fig4 panel F: RdBu_r on a fixed scale, white
    minor grid, onpage_to_gen font sizes, and each row's dominant |value| set in bold
    inside a black box."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_lbls)))
    xt = ax.set_xticklabels(col_lbls, rotation=55, ha="right", rotation_mode="anchor",
                            fontsize=onpage_to_gen(6.5, fig_w))
    for t in xt:
        t.set_multialignment("center")
    ax.set_yticks(range(len(row_lbls)))
    ax.set_yticklabels(row_lbls, fontsize=onpage_to_gen(7.0, fig_w))
    ax.set_xticks(np.arange(-0.5, len(col_lbls), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_lbls), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    scale = max(abs(vmin), abs(vmax))
    for i in range(mat.shape[0]):
        finite = np.where(np.isfinite(mat[i]))[0]
        jmax = int(finite[np.argmax(np.abs(mat[i][finite]))]) if finite.size else -1
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                    fontsize=onpage_to_gen(5.5, fig_w),
                    fontweight="bold" if (bold_dominant and j == jmax) else "normal",
                    color="white" if abs(mat[i, j]) > 0.55 * scale else "black")
        if bold_dominant and jmax >= 0:
            ax.add_patch(plt.Rectangle((jmax - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=2.2))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, fontsize=onpage_to_gen(7.5, fig_w))
    cb.ax.tick_params(labelsize=onpage_to_gen(6.5, fig_w))


def _msa_label(key):
    """+/- MSA_r, the RNA alignment, however the source frame spelled it."""
    plus = str(key).strip().lower() in ("+msa", "with msa")
    return r"$+$MSA$_r$" if plus else r"$-$MSA$_r$"


def rna_lengths():
    con = sqlite3.connect(os.path.join(AB, "a2021.db"))
    out = {}
    for t, v in con.execute("select exp_db_id, RNALength from pred_protein_rna"):
        try:
            out[t] = float(str(v).strip("[]").split(",")[0])
        except (TypeError, ValueError):
            pass
    con.close()
    return out


# --------------------------------------------------------------------------- #
# A: RNA-only, with vs without the RNA MSA
# --------------------------------------------------------------------------- #
def panel_single_chain_rna(ax):
    """Single-chain RNA targets, with vs without the RNA MSA, grouped by MSA depth and RNA length.

    _load_pair returns the paired set (same structures in both databases) with the group in
    `combined_class` ("<MSA depth>\n<length>") and the database label in `database`.
    """
    from publication_figures.msa_temp_influence import TMr, _combined_order
    a = _load_pair([(os.path.join(SC_DIR, "rna_singlechain_with_msa.db"), "with MSA"),
                    (os.path.join(SC_DIR, "rna_singlechain_no_msa.db"), "no MSA")],
                   "pred_rna_rna", msa_gt1_from="with MSA")
    groups = _combined_order(a)
    xs = np.arange(len(groups))
    for k, lab in enumerate(("with MSA", "no MSA")):
        col = MSA_COLORS[lab]
        vals = [a.loc[(a["combined_class"] == g) & (a["database"] == lab), TMr]
                 .dropna().to_numpy(float) for g in groups]
        bp = ax.boxplot(vals, positions=xs + (k - 0.5) * 0.34, widths=0.3,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="0.15", lw=1.2))
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.75, lw=0.8, edgecolor="0.3")
        for key in ("whiskers", "caps"):
            for ln in bp[key]:
                ln.set(color="0.45", lw=0.9)
        ax.plot([], [], "s", color=col,
                label=f"{_msa_label(lab)}\n(n={sum(len(v) for v in vals)})")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=onpage_to_gen(7, 16.5))
    ax.set_xlabel(r"RNA MSA depth | RNA length (nt)")
    ax.set_ylabel(r"TM$_r$")
    ax.annotate("Single-chain RNA", (0.02, 0.98), xycoords="axes fraction", va="top",
                fontsize=11, color="0.35")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.01), ncol=2,
              fontsize=12, columnspacing=1.4, handletextpad=0.5)
    style_axis(ax, grid_axis="y")


# --------------------------------------------------------------------------- #
# B-E: the input ablation
# --------------------------------------------------------------------------- #
def make_ablation_panels(fac, av, lengths):
    cls = av["cls"].to_dict()
    carriers = {f: set(av.index[av[f]]) for f in FACTORS}
    ids = {g: {t for t in av.index if cls.get(t) == g} for g in ("1v1", "multi")}

    def eff(factor, metric, subset=None):
        d = paired_differences(fac, factor, metric)
        d = d[d.index.isin(carriers[factor])]
        return d if subset is None else d[d.index.isin(subset)]

    def panel_b(ax):
        d = eff("rMSA", "RNA_TM")
        df = pd.DataFrame({"d": d})
        df["L"] = [lengths.get(t, np.nan) for t in df.index]
        df = df.dropna(subset=["L"])
        df["bin"] = [LEN_LABELS[int(np.digitize(L, LEN_BINS))] for L in df.L]
        present = [b for b in LEN_LABELS if (df.bin == b).any()]
        rng = np.random.default_rng(0)
        for i, b in enumerate(present):
            v = df.loc[df.bin == b, "d"].to_numpy(float)
            ax.scatter(np.full(len(v), i) + (rng.random(len(v)) - 0.5) * 0.22, v,
                       s=22, color="0.55", alpha=0.75, linewidths=0, zorder=3)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color=C_1V1, lw=2.5, zorder=4)
            ax.text(i, v.mean() + 0.05, f"{v.mean():+.2f}\nn={len(v)}", ha="center",
                    fontsize=9, color=C_1V1)
        ax.axhline(0, color="0.7", lw=1, zorder=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_xlabel("RNA length (nt)")
        ax.set_ylabel(r"$\Delta$TM$_r$ on removing MSA$_r$")
        ax.yaxis.label.set_y(0.42)          # the long label would otherwise run into the panel letter
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.annotate("Protein-RNA complexes", (0.0, 1.02), xycoords="axes fraction",
                    va="bottom", fontsize=11, color="0.35")
        style_axis(ax, grid_axis="y")

    def panel_c(ax):
        ypos = np.arange(len(FACTORS))[::-1]
        for k, (g, col) in enumerate((("1v1", C_1V1), ("multi", C_MULTI))):
            off = (k - 0.5) * 0.3
            for y, f in zip(ypos, FACTORS):
                s = summarise(eff(f, "RNA_TM", ids[g]))
                if not np.isfinite(s["mean"]):
                    continue
                ax.plot([s["lo"], s["hi"]], [y + off] * 2, color=col, lw=2, zorder=3)
                ax.plot(s["mean"], y + off, "o", ms=6, color=col, zorder=4,
                        label=f"{g.capitalize()} (n={len(ids[g])})" if f == FACTORS[0] else None)
        ax.axvline(0, color="0.7", lw=1, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([FACTOR_LABEL[f] for f in FACTORS])
        ax.set_xlabel(r"$\Delta$TM$_r$ (input present $-$ removed)")
        ax.annotate("Protein-RNA complexes", (0.0, 1.02), xycoords="axes fraction",
                    va="bottom", fontsize=11, color="0.35")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax, grid_axis="x")

    def panel_d(ax):
        """MSA_r effect with the protein MSA present vs removed (same targets, same backgrounds)."""
        have = carriers["rMSA"]
        out = {}
        for metric in ("RNA_TM", "Complex_TM"):
            vals = []
            for pm in (1, 0):
                sub = fac[fac.pMSA == pm]
                w = sub.pivot_table(index=["exp_db_id", "pTemp", "cTemp"],
                                    columns="rMSA", values=metric)
                if 0 not in w.columns or 1 not in w.columns:
                    vals.append(None)
                    continue
                d = (w[1] - w[0]).dropna().groupby("exp_db_id").mean()
                vals.append(d[d.index.isin(have)])
            out[metric] = vals
        x = np.arange(2)
        for k, (metric, lab, col) in enumerate((("RNA_TM", r"TM$_r$", C_1V1),
                                                ("Complex_TM", r"TM$_c$", C_MULTI))):
            means, los, his = [], [], []
            for d in out[metric]:
                s = summarise(d) if d is not None else dict(mean=np.nan, lo=np.nan, hi=np.nan)
                means.append(s["mean"]); los.append(s["mean"] - s["lo"]); his.append(s["hi"] - s["mean"])
            ax.errorbar(x + (k - 0.5) * 0.16, means, yerr=[los, his], fmt="o-", ms=7,
                        lw=1.8, capsize=4, color=col, label=lab, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([r"MSA$_p$ present", r"MSA$_p$ removed"])
        ax.set_ylabel(r"effect of MSA$_r$ ($\Delta$)")
        n = len(carriers["rMSA"] & set(fac.exp_db_id))
        ax.annotate(f"n={n} carrying MSA$_r$", (0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, -12), textcoords="offset points", ha="center",
                    fontsize=9, color="0.4")
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax, grid_axis="y")

    def panel_e(ax):
        mat = np.full((len(FACTORS), len(METRICS)), np.nan)
        for i, f in enumerate(FACTORS):
            for j, (m, _) in enumerate(METRICS):
                d = eff(f, m).to_numpy(float)
                d = d[np.isfinite(d)]
                if d.size > 2 and d.std(ddof=1) > 0:
                    mat[i, j] = d.mean() / d.std(ddof=1)
        lim = float(np.nanmax(np.abs(mat)))
        style_heatmap(ax, mat, [FACTOR_LABEL[f] for f in FACTORS],
                      [l for _, l in METRICS], "Paired $d$",
                      vmin=-lim, vmax=lim)
        ax.annotate("Protein-RNA complexes", (0.0, 1.02), xycoords="axes fraction",
                    va="bottom", fontsize=11, color="0.35")

    return panel_b, panel_c, panel_d, panel_e


# --------------------------------------------------------------------------- #
# F-I: AF3 confidence (pub_fig4 panels B, C, D, F)
# --------------------------------------------------------------------------- #
_QUAD_COLORS = ["#4E79A7", "#59A14F", "#E15759", "#B07AA1"]
_QUAD_LABELS = ["(i)", "(ii)", "(iii)", "(iv)"]
_QUAD_KEYS = ["quad_i", "quad_ii", "quad_iii", "quad_iv"]
_CAT_SUFFIXES = [("_a", "a"), ("_b", "b"), ("_c_ca", "ca"), ("_c_cb", "cb")]
MSA_DEPTH_BINS = [(-np.inf, 0.5, "0"), (0.5, 10, "1-9"), (10, 100, "10-99"),
                  (100, 1000, "100-999"), (1000, np.inf, r"$\geq$1000")]


def _keyed_list(raw):
    """[{'B': 0.53}, {'C': 0.6}] -> [0.53, 0.6] (the per-chain keyed columns)."""
    import ast as _ast
    import json as _json
    if raw in (None, "", "null", "[]"):
        return []
    v = None
    for loader in (_json.loads, _ast.literal_eval):
        try:
            v = loader(str(raw))
            break
        except (ValueError, SyntaxError, TypeError):
            continue
    out = []
    if isinstance(v, list):
        for el in v:
            if isinstance(el, dict):
                for val in el.values():
                    try:
                        out.append(float(val))
                    except (TypeError, ValueError):
                        out.append(np.nan)
    return out


def confidence_data(table="pred_protein_rna"):
    """Per-RNA-chain records and audit-category series, exactly as pub_fig4 derives them."""
    from evaluation.evaluationMetrics import (RNAMetrics, _as_chain_list, _as_float_list,
                                              _audit_extract_motif_bin_metrics,
                                              _audit_load_prediction_by_eid,
                                              _effective_msa_depth_for_chain)
    # PlotCreator is only initialised when these kwargs are given, and get_filtered_values
    # needs it; same construction as pub_fig1_classification_framework.main() (full set, no filter).
    rm = RNAMetrics(msa_option=None, single_chain_only=False, rbp_template_option=None)
    audit = rm.get_sample_audit_global_fold_interface_lists(table)
    if not audit or "quad_i_all" not in audit:
        raise RuntimeError(f"no audit data from {table}; check the database in "
                           f"database/{_CONF_NAME}")
    by_eid = _audit_load_prediction_by_eid(rm, table)
    quad_all = [audit.get(f"{p}_all") or [] for p in _QUAD_KEYS]

    ptm, tm, dep, length = [], [], [], []
    for recs in quad_all:
        for rec in recs:
            eid = str(rec.get("exp_db_id", "")).strip()
            if not by_eid or eid not in by_eid:
                continue
            rd = by_eid[eid]
            chains = _as_chain_list(rd.get("RNAChainIDs"))
            ptms = _as_float_list(rd.get("af3_rna_pTM"))
            tms = _keyed_list(rd.get("RNA_TM_AllChains"))
            lens = _as_float_list(rd.get("RNALength"))
            for i, ch in enumerate(chains):
                P = ptms[i] if i < len(ptms) else np.nan
                T = tms[i] if i < len(tms) else np.nan
                L = lens[i] if i < len(lens) else np.nan
                d = _effective_msa_depth_for_chain(rd.get("RNA_msa_size"), ch)
                if np.isfinite(P) and np.isfinite(T) and np.isfinite(d):
                    ptm.append(float(P)); tm.append(float(T)); dep.append(float(d))
                    length.append(float(L) if np.isfinite(L) else np.nan)
    pc = dict(ptm=np.array(ptm), tm=np.array(tm), depth=np.array(dep),
              length=np.array(length))
    pc["has_msa"] = pc["depth"] > 0

    labels, series, colors = [], [], []
    for qi, (qk, roman) in enumerate(zip(_QUAD_KEYS, _QUAD_LABELS)):
        for sfx, slbl in _CAT_SUFFIXES:
            vals = []
            for rec in audit.get(f"{qk}{sfx}") or []:
                eid = str(rec.get("exp_db_id", "")).strip()
                if by_eid and eid in by_eid:
                    v = _audit_extract_motif_bin_metrics(by_eid[eid]).get("af3_rna_ipTM")
                    if v is not None and np.isfinite(v):
                        vals.append(float(v))
            labels.append(f"{roman} {slbl}"); series.append(vals); colors.append(_QUAD_COLORS[qi])

    # Global PAE vs RMSD_c per complex, split by whether the RNA has an MSA
    pae = {"+MSA": ([], []), "-MSA": ([], [])}
    for rd in (by_eid or {}).values():
        fv = _audit_extract_motif_bin_metrics(rd)
        has = fv.get("has_RNA_MSA")
        if has is None or not np.isfinite(has):
            continue
        key = "+MSA" if float(has) >= 0.5 else "-MSA"
        x = _first_float(rd.get("Complex_RMSD"))
        y = _first_float(rd.get("af3_global_pae_avg"))
        if np.isfinite(x) and np.isfinite(y):
            pae[key][0].append(x); pae[key][1].append(y)

    heat_rows, heat_cols, heat = _confidence_heatmap(by_eid)
    rm.close_connection()
    return pc, (labels, series, colors), (heat_rows, heat_cols, heat), pae


def _confidence_heatmap(by_eid):
    """Spearman rho between each AF3 confidence metric and each ground-truth axis."""
    from scipy.stats import spearmanr
    conf = [("af3_protein_pTM", r"pTM$_p$"), ("af3_rna_pTM", r"pTM$_r$"),
            ("af3_rna_ipTM", r"ipTM$_r$"), ("af3_protein_pLDDT_avg", r"pLDDT$_p$"),
            ("af3_rna_pLDDT_avg", r"pLDDT$_r$"), ("af3_global_pae_avg", "PAE")]
    # Column names as _audit_load_prediction_by_eid provides them: it carries Protein_LDDT
    # (not the all-atom variant) and only the per-interface RBP_DockQ, whose mean over the
    # protein-RNA interfaces is the paper's DockQ_Nr.
    truth = [("Complex_TM", r"TM$_c$"), ("RNA_TM", r"TM$_r$"),
             ("Protein_LDDT", r"LDDT$_p$"), ("RNA_LDDT_allAtom", r"LDDT$_r$"),
             ("RBP_DockQ", r"DockQ$_{Nr}$"), ("Complex_RMSD", r"RMSD$_c$")]
    rows = {k: [] for k, _ in conf}
    cols = {k: [] for k, _ in truth}
    for rd in (by_eid or {}).values():
        for k, _ in conf:
            rows[k].append(_first_float(rd.get(k)))
        for k, _ in truth:
            cols[k].append(_mean_keyed(rd.get(k)) if k == "RBP_DockQ"
                           else _first_float(rd.get(k)))
    mat = np.full((len(conf), len(truth)), np.nan)
    for i, (ck, _) in enumerate(conf):
        for j, (tk, _) in enumerate(truth):
            a, b = np.array(rows[ck], float), np.array(cols[tk], float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 10:
                mat[i, j] = spearmanr(a[m], b[m]).correlation
    return [l for _, l in conf], [l for _, l in truth], mat


def _mean_keyed(raw):
    """Mean over a keyed per-interface column, e.g. RBP_DockQ -> DockQ_Nr."""
    vals = [v for v in _keyed_list(raw) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _first_float(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.startswith("["):
        vals = _keyed_list(s)
        if vals:
            return vals[0]
        try:
            parts = [float(x) for x in s.strip("[] ").split(",") if x.strip()]
        except ValueError:
            return np.nan
        if not parts:
            return np.nan
        return parts[1] if len(parts) > 1 else parts[0]
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def make_confidence_panels(pc, cat, heat, pae):
    from scipy.stats import spearmanr
    labels, series, colors = cat
    rows_lab, cols_lab, mat = heat

    def panel_f(ax):
        for s, m in (("+MSA", pc["has_msa"]), ("-MSA", ~pc["has_msa"])):
            if not m.sum():
                continue
            x, y = pc["ptm"][m], pc["tm"][m]
            r = float(np.corrcoef(x, y)[0, 1]) if m.sum() >= 3 else np.nan
            ax.scatter(x, y, c=MSA_COLORS[s], alpha=0.55, s=16, edgecolors="none",
                       label=f"{_msa_label(s)}: $r$={r:+.2f}\n(n={int(m.sum())})")
        ax.set_xlabel(r"pTM$_r$"); ax.set_ylabel(r"TM$_r$")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.01),
                  ncol=2, markerscale=1.8, fontsize=12, columnspacing=1.4,
                  handletextpad=0.5)
        style_axis(ax)

    def panel_g(ax):
        xs, mp, mt, ns = [], [], [], []
        for i, (lo, hi, lab) in enumerate(MSA_DEPTH_BINS):
            m = (pc["depth"] > lo) & (pc["depth"] <= hi)
            if m.sum() < 3:
                continue
            xs.append(lab); mp.append(pc["ptm"][m].mean()); mt.append(pc["tm"][m].mean())
            ns.append(int(m.sum()))
        x = np.arange(len(xs))
        ax.plot(x, mp, "o-", color=MSA_COLORS["+MSA"], lw=2, ms=7, label=r"pTM$_r$")
        ax.plot(x, mt, "s--", color=MSA_COLORS["-MSA"], lw=2, ms=7, label=r"TM$_r$")
        for xi, n in zip(x, ns):
            ax.annotate(f"n={n}", (xi, 0), xytext=(0, -26), textcoords="offset points",
                        ha="center", fontsize=8, color="0.45")
        ax.set_xticks(x); ax.set_xticklabels(xs)
        ax.set_xlabel(r"RNA MSA depth"); ax.set_ylabel("Mean score")
        ax.legend(frameon=False, loc="upper left")
        style_axis(ax, grid_axis="y")

    def panel_h(ax):
        keep = [(l, s, c) for l, s, c in zip(labels, series, colors) if len(s)]
        bp = ax.boxplot([s for _, s, _ in keep], positions=np.arange(len(keep)),
                        widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="0.15", lw=1.2))
        for box, (_, _, c) in zip(bp["boxes"], keep):
            box.set(facecolor=c, alpha=0.75, lw=0.7, edgecolor="0.3")
        for key in ("whiskers", "caps"):
            for ln in bp[key]:
                ln.set(color="0.45", lw=0.8)
        ax.set_xticks(np.arange(len(keep)))
        ax.set_xticklabels([l for l, _, _ in keep], rotation=90,
                           fontsize=onpage_to_gen(7, 16.5))
        ax.set_ylabel(r"ipTM$_r$"); ax.set_ylim(-0.05, 1.05)
        style_axis(ax, grid_axis="y")

    def panel_i(ax):
        style_heatmap(ax, mat, rows_lab, cols_lab, r"Spearman $\rho$",
                      vmin=-1.0, vmax=1.0)

    def panel_pae(ax):
        """Global PAE vs RMSD_c per complex, split by RNA MSA (pub_fig4 panel E)."""
        for s_ in ("+MSA", "-MSA"):
            xs, ys = pae[s_]
            if not len(xs):
                continue
            rho = spearmanr(xs, ys).correlation if len(xs) >= 3 else np.nan
            ax.scatter(xs, ys, c=MSA_COLORS[s_], alpha=0.55, s=18, edgecolors="none",
                       label=f"{_msa_label(s_)}: $\\rho$={rho:+.2f}\n(n={len(xs)})")
        ax.set_xlabel("RMSD$_c$ (Å)")
        ax.set_ylabel("Global PAE")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.01),
                  ncol=2, markerscale=1.8, fontsize=12, columnspacing=1.4,
                  handletextpad=0.5)
        style_axis(ax)

    return panel_f, panel_g, panel_h, panel_i, panel_pae


def main():
    fac = load_factorial(verbose=False)
    av = availability()
    av = av[av.exp_db_id.isin(set(fac.exp_db_id))].set_index("exp_db_id")
    print(f"ablation: {len(av)} targets (1v1 {(av.cls=='1v1').sum()}, multi {(av.cls=='multi').sum()})")
    b, c, _unused_interaction, e = make_ablation_panels(fac, av, rna_lengths())
    pc, cat, heat, pae = confidence_data()
    print(f"confidence: {len(pc['ptm'])} RNA chains")
    f, g, h, i, panel_pae = make_confidence_panels(pc, cat, heat, pae)

    # AF3 confidence (A-E), then what the inputs contribute (F-I). No panel titles: the
    # axis labels carry the meaning and the caption names each panel.
    fig = PublicationFigure(ncols=3, panel_size=(5.5, 4.3))
    for draw in (f, g, h, panel_pae, i,
                 panel_single_chain_rna, b, c, e):
        fig.add_panel(draw, title=None)
    return fig.save("inputs_and_confidence", formats=("png", "pdf"))


if __name__ == "__main__":
    main()
