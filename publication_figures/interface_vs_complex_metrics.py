#!/usr/bin/env python3
"""
Nature-style protein-RNA interface figures (two related figures from one script).

Both are built with plots.publication_figure.PublicationFigure (which drives
plots.plotCreator.PlotCreator for output-path handling) so panels get bold letters (A, B, C, ...)
and the shared Nature styling. The DB is loaded once and both figures are written to the standard
PlotCreator location <repo>/results/plots/all_chains/publication_figures (the repo is inferred from
this script's location; run it from the AF3InterfaceEval tree so plots/ and the DB resolve there),
exactly like the other AF3InterfaceEval/publication_figures scripts.

Figure 1 -- interface_vs_complex_metrics.{pdf,png}
    Per-RNA-chain / per-interface metrics vs the single whole-complex metric. Each panel: one point
    per predicted complex, x = whole-complex scalar, y = mean of the separate per-chain / per-interface
    values (grey bar = min..max spread), y = x diagonal, Pearson r and n, coloured by dominant
    RNA_interface_class. Interface panels (C-F) are restricted to ChainIDpairList_proteinRNA pairs.

Figure 2 -- interface_gap_vs_composition.{pdf,png}
    Why the whole-complex interface metrics over-state protein-RNA quality, and whether the extra
    interfaces also degrade the protein-RNA interfaces themselves.
      A  iLDDT gap Delta = iLDDT_whole - iLDDT_Nr vs the number of protein chains (boxplot); the gap
         grows with the protein content.
      B  Spearman rho of every metric gap against six composition drivers (heatmap); the gaps track
         the protein-side drivers -- above all the number of protein-protein interfaces -- and only
         weakly the RNA-side drivers.
      C  The protein-RNA-only metric itself (DockQ_Nr by default, see NR_METRIC) vs the number of
         protein-protein (blue) and RNA-RNA (orange) interfaces; both trends are flat, so the extra
         interfaces inflate the whole-complex value without changing protein-RNA interface quality.
    Protein-protein and RNA-RNA interface counts are derived from the RBP_* interface pair keys,
    classified with ProteinChainIDs / RNAChainIDs.

Run (env with matplotlib/pandas/scipy/seaborn, e.g. RNABench), from the repo root:
  python publication_figures/interface_vs_complex_metrics.py
"""

import ast
import json
import os
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plots.publication_figure import PublicationFigure, style_axis, onpage_to_gen

TABLE = "pred_protein_rna"

# --- Figure 1 config ------------------------------------------------------------------------------
# per-chain/interface column [JSON keyed], whole-complex scalar column, is_interface, y-label, x-label
# y-axis shows the MEAN of the separate per-chain/interface values, hence the "Mean" suffix.
PANELS = [
    ("RNA_LDDT_AllChains",  "RNA_LDDT_allAtom", False, r"LDDT$_{Nr}$ Mean",  r"LDDT$_{r}$"),
    ("RNA_TM_AllChains",    "RNA_TM",           False, r"TM$_{Nr}$ Mean",    r"TM$_{r}$"),
    ("RNA_iLDDT_AllChains", "iLDDT",            True,  r"iLDDT$_{Nr}$ Mean", r"iLDDT"),
    ("RNA_ICS_AllChains",   "ICS",              True,  r"ICS$_{Nr}$ Mean",   r"ICS"),
    ("RBP_DockQ",           "Dockq",            True,  r"DockQ$_{Nr}$ Mean", r"DockQ"),
    ("RBP_QS",              "QS_global",        True,  r"QS$_{Nr}$ Mean",    r"QS"),
]

# RNA_interface_class -> (colour, quality rank; higher = better) for the per-complex dominant class.
# DockQ-based classes (classify_rna_interface_quality.py, complying with evaluationMetrics a/b/c):
# correct = DockQ>=0.8, partial = [0.23,0.8), incorrect = <0.23, missing = no scorable interface.
CLASS_STYLE = {
    "correct_interface":   ("#1a9850", 3),  # (a) very good
    "partial_interface":   ("#fdae61", 2),  # (b) medium
    "incorrect_interface": ("#d73027", 1),  # (c) bad
    "unknown":             ("#c7c7c7", 0),  # no scorable protein-RNA interface
}

# No classes are merged for display any more.
CLASS_ALIAS = {}

# --- Figure 2 config ------------------------------------------------------------------------------
# gap label, per-chain/interface column [JSON keyed], whole-complex scalar column
METRICS = [
    (r"$\Delta$iLDDT", "RNA_iLDDT_AllChains", "iLDDT"),
    (r"$\Delta$ICS",   "RNA_ICS_AllChains",   "ICS"),
    (r"$\Delta$DockQ", "RBP_DockQ",           "Dockq"),
    (r"$\Delta$QS",    "RBP_QS",              "QS_global"),
]

# composition driver column -> axis label (heatmap D). "No. protein-protein interfaces" and
# "No. RNA-RNA interfaces" are derived counts; the RNA-RNA column is dropped if none are present.
DRIVERS = [
    ("NumberProteins", "Protein chains"),
    ("PP_iface",       "P–P interfaces"),
    ("ProteinLen_tot", "Protein length"),
    ("NumberRNAs",     "RNA chains"),
    ("RR_iface",       "R–R interfaces"),
    ("RNALen_tot",     "RNA length"),
]

ACCENT = "#2166ac"   # blue (protein-protein)
ACCENT2 = "#cb181d"  # red (RNA-RNA)

# The protein-RNA-only "Nr" metric plotted in panels C/D of the gap figure (column, y-label).
# Swap to ("RNA_iLDDT_AllChains", r"iLDDT$_{Nr}$ mean"), ("RNA_ICS_AllChains", r"ICS$_{Nr}$ mean")
# or ("RBP_QS", r"QS$_{Nr}$ mean") to show a different interface metric.
NR_METRIC = ("RBP_DockQ", r"DockQ$_{Nr}$ mean")


# --- shared helpers -------------------------------------------------------------------------------
def db_path():
    try:
        with open(os.path.join(REPO, "database", os.environ.get("EVAL_CONFIG_NAME", "config.json"))) as fh:
            p = json.load(fh).get("db_path")
        if p:
            return p
    except (OSError, ValueError):
        pass
    return os.path.join(REPO, "database", "rbpDatabase.db")


def parse_values(cell, allowed_pairs=None):
    """Numeric per-chain/interface values from a JSON keyed cell. If allowed_pairs is given (a set of
    frozenset chain-pairs), keep only interface entries whose 'X:Y' key is that pair (order-insensitive)."""
    if cell is None:
        return []
    s = str(cell).strip()
    if s in ("", "[]"):
        return []
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return []
    out = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                for k, v in el.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if allowed_pairs is not None:
                        parts = str(k).split(":")
                        if len(parts) != 2 or frozenset(parts) not in allowed_pairs:
                            continue
                    out.append(v)
            elif isinstance(el, list):  # legacy matrix row
                out.extend(v for v in el if isinstance(v, (int, float)) and v != 0)
            elif isinstance(el, (int, float)):
                out.append(el)
    return [float(v) for v in out]


def parse_protein_rna_pairs(cell):
    """ChainIDpairList_proteinRNA [["A","D"], ...] -> set of frozenset({protein, RNA}) pairs."""
    try:
        lst = json.loads(str(cell))
    except (ValueError, TypeError):
        return set()
    pairs = set()
    if isinstance(lst, list):
        for pr in lst:
            if isinstance(pr, list) and len(pr) == 2:
                pairs.add(frozenset(str(x) for x in pr))
    return pairs


def dominant_class(cell):
    """Best-quality class among a complex's per-chain RNA_interface_class dict."""
    if not cell:
        return None
    try:
        d = json.loads(str(cell))
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or not d:
        return None
    best, best_rank = None, -1
    for cls in d.values():
        cls = CLASS_ALIAS.get(cls, cls)  # (no aliases currently)
        rank = CLASS_STYLE.get(cls, ("", -1))[1]
        if rank > best_rank:
            best, best_rank = cls, rank
    return best


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_chain_ids(raw):
    """ProteinChainIDs / RNAChainIDs -> set of chain ids (JSON list, py literal, or single id)."""
    if raw is None:
        return set()
    s = str(raw).split("//", 1)[0].strip()
    if not s:
        return set()
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        try:
            d = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            d = s
    if isinstance(d, (list, tuple)):
        return {str(x).strip() for x in d if str(x).strip()}
    return {str(d).strip()}


def _pair_keys(cell):
    """The 'X:Y' interface keys from a JSON keyed-dict-list cell (RBP_* columns)."""
    if not cell:
        return []
    try:
        obj = json.loads(str(cell))
    except (ValueError, TypeError):
        return []
    keys = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                keys += [str(k) for k in el.keys()]
    return keys


def count_interface_types(dq_cell, qs_cell, prot_set, rna_set):
    """(protein-protein, protein-RNA, RNA-RNA) interface counts from the RBP_* pair keys."""
    keys = _pair_keys(dq_cell) or _pair_keys(qs_cell)
    pp = pr = rr = 0
    for k in keys:
        if ":" not in k:
            continue
        a, b = (x.strip() for x in k.split(":", 1))
        if a in prot_set and b in prot_set:
            pp += 1
        elif a in rna_set and b in rna_set:
            rr += 1
        elif (a in prot_set and b in rna_set) or (a in rna_set and b in prot_set):
            pr += 1
    return pp, pr, rr


def sum_length(raw):
    """ProteinLength / RNALength is an int or a JSON/py list of per-chain lengths -> total."""
    if raw is None:
        return np.nan
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return np.nan
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return np.nan
    if isinstance(v, (list, tuple)):
        nums = [float(x) for x in v if isinstance(x, (int, float))]
        return float(sum(nums)) if nums else np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _save(fig, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    for ext in ("pdf", "png", "svg", "eps"):
        out = os.path.join(results_dir, name + "." + ext)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


# --- Figure 1: per-chain/interface vs whole-complex -----------------------------------------------
def make_panel(df, per_col, whole_col, is_iface, ylabel, xlabel):
    """Return a draw(ax) callback rendering one comparison panel."""
    def draw(ax):
        sub = df[[per_col, whole_col, "dom_class", "pair_set"]].copy()
        if is_iface:
            sub["per_vals"] = sub.apply(lambda r: parse_values(r[per_col], r["pair_set"]), axis=1)
        else:
            sub["per_vals"] = sub[per_col].apply(parse_values)
        sub["whole"] = sub[whole_col].apply(to_float)
        sub = sub[(sub["per_vals"].apply(len) > 0) & sub["whole"].notna()]

        y_mean = sub["per_vals"].apply(np.mean).to_numpy()
        y_lo = sub["per_vals"].apply(min).to_numpy()
        y_hi = sub["per_vals"].apply(max).to_numpy()
        x = sub["whole"].to_numpy()
        colours = [CLASS_STYLE.get(c, ("#000000", 0))[0] for c in sub["dom_class"]]

        ax.vlines(x, y_lo, y_hi, color="#b0b0b0", lw=0.8, alpha=0.6, zorder=1)
        ax.scatter(x, y_mean, c=colours, s=26, edgecolor="black", linewidth=0.35, zorder=3)
        lim = [-0.03, 1.03]
        ax.plot(lim, lim, ls="--", color="black", lw=1.0, alpha=0.7, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=onpage_to_gen(7.5, 13.8))
        ax.set_ylabel(ylabel, fontsize=onpage_to_gen(7.5, 13.8))
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])

        if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y_mean) > 0:
            r, _ = pearsonr(x, y_mean)
            txt = f"r = {r:.2f}\nn = {len(x)}"
        else:
            txt = f"n = {len(x)}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=onpage_to_gen(6.5, 13.8), fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    return draw


def build_comparison_figure(df):
    pf = PublicationFigure(ncols=3, panel_size=(4.6, 4.4), letters=True)
    for per_col, whole_col, is_iface, ylab, xlab in PANELS:
        pf.add_panel(make_panel(df, per_col, whole_col, is_iface, ylab, xlab), title=None)
    fig = pf.build()

    # Shared legend (interface classes) beneath the panels.
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                      markerfacecolor=CLASS_STYLE[c][0], markeredgecolor="black",
                      label=c.replace("_", " "))
               for c in sorted(CLASS_STYLE, key=lambda k: -CLASS_STYLE[k][1])]
    # Points whose dominant class is missing fall back to black (make_panel's
    # CLASS_STYLE.get default); give them an explicit legend entry.
    handles += [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                       markerfacecolor="#000000", markeredgecolor="black",
                       label="unclassified")]
    handles += [Line2D([0], [0], color="#b0b0b0", lw=2.5, label="min–max range"),
                Line2D([0], [0], ls="--", color="black", lw=1.2, label="y = x")]
    # Drop the legend clear of the bottom-row x-axis labels (DockQ / QS / ...);
    # at -0.04 the first legend row collided with those labels.
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=onpage_to_gen(7, 13.8), bbox_to_anchor=(0.5, -0.02))
    _save(fig, pf.results_dir, "interface_vs_complex_metrics")


# --- Figure 2: gap vs composition -----------------------------------------------------------------
def nr_mean_series(df, per_col):
    """Mean of the protein-RNA-only (Nr) values for an interface metric; NaN when none available."""
    def _per_mean(r):
        vals = parse_values(r[per_col], r["pair_set"])
        return np.mean(vals) if vals else np.nan
    return df.apply(_per_mean, axis=1)


def gap_series(df, per_col, whole_col):
    """Delta = whole - mean(protein-RNA-only Nr values); NaN when either side is unavailable."""
    return df[whole_col].apply(to_float) - nr_mean_series(df, per_col)


def _bin_counts(vals, edges, labels):
    """Assign each integer count to a labelled bin (edges are inclusive upper bounds; last = +inf)."""
    out = []
    for v in vals:
        if np.isnan(v):
            out.append(None)
            continue
        lab = labels[-1]
        for e, l in zip(edges, labels):
            if v <= e:
                lab = l
                break
        out.append(lab)
    return out


def spearman_txt(x, y):
    m = (~np.isnan(x)) & (~np.isnan(y))
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return None, int(m.sum())
    rho, _ = spearmanr(x[m], y[m])
    return rho, int(m.sum())


def _annot(ax, rho, n):
    txt = (f"Spearman $\\rho$ = {rho:.2f}\nn = {n}" if rho is not None else f"n = {n}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=onpage_to_gen(6.5, 16.2),
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))


def _boxplot_by_bin(ax, yvals, counts, edges, labels, xlabel, ylabel, hline=None):
    """Boxplot of ``yvals`` grouped into count bins, with the raw-count Spearman annotated."""
    binned = _bin_counts(counts, edges, labels)
    groups, keep_labels = [], []
    for lab in labels:
        vals = np.array([g for g, b in zip(yvals, binned) if b == lab and not np.isnan(g)])
        if len(vals):
            groups.append(vals)
            keep_labels.append(f"{lab}\n(n={len(vals)})")
    bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.3)
    ax.set_xticks(range(1, len(keep_labels) + 1))
    # The stacked "(n=..)" second line is wider than the bin slot; drop this tick
    # font one step so adjacent count labels no longer touch.
    ax.set_xticklabels(keep_labels, fontsize=onpage_to_gen(6.0, 16.2))
    if hline is not None:
        ax.axhline(hline, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel(xlabel, fontsize=onpage_to_gen(7.5, 16.2))
    ax.set_ylabel(ylabel, fontsize=onpage_to_gen(7.5, 16.2))
    rho, n = spearman_txt(counts.astype(float), np.asarray(yvals, dtype=float))
    _annot(ax, rho, n)
    style_axis(ax, grid_axis="y")


def _quality_vs_interfaces(ax, pp, rr, nr, ylabel):
    """Protein-RNA-only metric (nr) vs the count of the *other* interfaces, split protein-protein
    (blue) vs RNA-RNA (orange). Both trends are near-flat: the extra interfaces do not change the
    protein-RNA interface quality, even though they inflate the whole-complex value (panels A, B)."""
    pp = np.asarray(pp, dtype=float)
    rr = np.asarray(rr, dtype=float)
    nr = np.asarray(nr, dtype=float)
    for xv, col, lab in ((pp, ACCENT, "protein–protein"), (rr, ACCENT2, "RNA–RNA")):
        m = (~np.isnan(xv)) & (~np.isnan(nr))
        ax.scatter(xv[m], nr[m], s=22, c=col, alpha=0.55, edgecolor="black", linewidth=0.3, label=lab)
    rho_pp, n_pp = spearman_txt(pp, nr)
    rho_rr, n_rr = spearman_txt(rr, nr)
    f = lambda r: (f"{r:.2f}" if r is not None else "n/a")
    txt = f"P–P: $\\rho$ = {f(rho_pp)}\nR–R: $\\rho$ = {f(rho_rr)}\nn = {n_pp}"
    # Abbreviated to one line (P–P / R–R match the panel-B interface labels); the full
    # "protein–protein / RNA–RNA" wording spelled out the two-line version that got clipped.
    ax.set_xlabel("No. P–P / R–R interfaces", fontsize=onpage_to_gen(7.5, 16.2))
    ax.set_ylabel(ylabel, fontsize=onpage_to_gen(7.5, 16.2))
    # Dot legend as a single row below the panel; the Spearman box takes the freed top-right corner.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
              fontsize=onpage_to_gen(7, 16.2), columnspacing=1.6, handletextpad=0.4)
    ax.text(0.97, 0.96, txt, transform=ax.transAxes, va="top", ha="right",
            fontsize=onpage_to_gen(6.5, 16.2),
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    style_axis(ax)


def _heatmap(ax, corr, metric_labels, driver_labels):
    sns.heatmap(corr, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": onpage_to_gen(6.5, 16.2)},
                cbar_kws={"label": r"Spearman $\rho$", "fraction": 0.046, "pad": 0.04}, ax=ax)
    ax.set_xticklabels(driver_labels, rotation=30, ha="right", rotation_mode="anchor",
                       fontsize=onpage_to_gen(7, 16.2))
    ax.set_yticklabels(metric_labels, rotation=0, fontsize=onpage_to_gen(7, 16.2))
    ax.set_title("")


def build_gap_figure(df):
    df = df.copy()
    df["ProteinLen_tot"] = df["ProteinLength"].apply(sum_length)
    df["RNALen_tot"] = df["RNALength"].apply(sum_length)
    df["NumberProteins"] = df["NumberProteins"].apply(to_float)
    df["NumberRNAs"] = df["NumberRNAs"].apply(to_float)

    # Protein-protein / RNA-RNA interface counts from the RBP_* pair keys.
    prot_sets = df["ProteinChainIDs"].apply(parse_chain_ids)
    rna_sets = df["RNAChainIDs"].apply(parse_chain_ids)
    counts = [count_interface_types(dq, qs, P, R)
              for dq, qs, P, R in zip(df["RBP_DockQ"], df["RBP_QS"], prot_sets, rna_sets)]
    df["PP_iface"] = [c[0] for c in counts]
    df["RR_iface"] = [c[2] for c in counts]

    gaps = {label: gap_series(df, per, whole).to_numpy() for label, per, whole in METRICS}
    ilddt_gap = gaps[r"$\Delta$iLDDT"]

    # Drop the RNA-RNA driver from the heatmap if the dataset has no RNA-RNA interfaces.
    drivers = [d for d in DRIVERS if not (d[0] == "RR_iface" and df["RR_iface"].sum() == 0)]

    corr = np.full((len(METRICS), len(drivers)), np.nan)
    for i, (label, _, _) in enumerate(METRICS):
        g = gaps[label]
        for j, (dcol, _) in enumerate(drivers):
            rho, _n = spearman_txt(df[dcol].to_numpy(dtype=float), g)
            corr[i, j] = rho if rho is not None else np.nan
    metric_labels = [m[0] for m in METRICS]
    driver_labels = [d[1] for d in drivers]

    n_prot = df["NumberProteins"].to_numpy()
    pp_iface = df["PP_iface"].to_numpy(dtype=float)
    rr_iface = df["RR_iface"].to_numpy(dtype=float)

    # Panel C: protein-protein / RNA-RNA interface counts vs the protein-RNA-only Nr metric.
    nr_col, nr_label = NR_METRIC
    nr_mean = nr_mean_series(df, nr_col).to_numpy()

    # Just "ΔiLDDT" (matches panel B's row label for the same quantity); the whole − protein–RNA
    # definition lives in the caption. Short label sits mid-axis, clear of the panel letter.
    ILDDT_GAP_YLABEL = r"$\Delta$iLDDT"

    pf = PublicationFigure(ncols=3, panel_size=(5.4, 4.8), letters=True)
    # A: unchanged -- iLDDT gap vs number of protein chains.
    pf.add_panel(lambda ax: _boxplot_by_bin(
        ax, ilddt_gap, n_prot, [1, 2, 3, 4, 6], ["1", "2", "3", "4", "5-6", "7+"],
        "No. protein chains", ILDDT_GAP_YLABEL, hline=0), title=None)
    # B: the gap-vs-driver correlation heatmap.
    pf.add_panel(lambda ax: _heatmap(ax, corr, metric_labels, driver_labels), title=None)
    # C: protein-protein AND RNA-RNA interface counts vs the protein-RNA-only Nr metric (both flat).
    pf.add_panel(lambda ax: _quality_vs_interfaces(ax, pp_iface, rr_iface, nr_mean, nr_label),
                 title=None)

    fig = pf.build()
    _save(fig, pf.results_dir, "interface_gap_vs_composition")


# --- driver ---------------------------------------------------------------------------------------
def load_df():
    cols = ["RNA_interface_class", "ChainIDpairList_proteinRNA",
            "ProteinChainIDs", "RNAChainIDs",
            "NumberProteins", "NumberRNAs", "ProteinLength", "RNALength"]
    cols += sorted({c for per, whole, *_ in PANELS for c in (per, whole)})
    cols += sorted({c for _, per, whole in METRICS for c in (per, whole)})
    cols = sorted(set(cols))
    col_sql = ", ".join('"' + c + '"' for c in cols)
    conn = sqlite3.connect(db_path())
    df = pd.read_sql_query("SELECT " + col_sql + " FROM " + TABLE, conn)
    conn.close()
    df["dom_class"] = df["RNA_interface_class"].apply(dominant_class)
    df["pair_set"] = df["ChainIDpairList_proteinRNA"].apply(parse_protein_rna_pairs)
    return df


def main():
    df = load_df()
    build_comparison_figure(df)
    build_gap_figure(df)


if __name__ == "__main__":
    main()
