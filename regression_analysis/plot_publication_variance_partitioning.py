#!/usr/bin/env python3
"""
plot_publication_variance_partitioning.py

Publication-ready figures for the Shapley-R^2 variance partition of the
multivariate regression analysis. Reads the aggregated summary written by
plot_variance_partitioning_summary.py and emits vector (PDF) + raster (PNG)
figures with consistent, journal-style formatting.

Outputs (to --outdir, default publication_figures/):
  fig_shapley_complex.{pdf,png}      stacked Shapley-R^2, complex level (cTM, DockQ)
  fig_shapley_perchain.{pdf,png}     stacked Shapley-R^2, per-chain (stable subsets)
  fig_shapley_residual.{pdf,png}     stacked residual partition (what AF3 misses)
  fig_shapley_share_heatmap.{pdf,png} share-% heatmap (direct, stable runs)
  fig_delta_r2_complex_vs_perchain.{pdf,png}
                                     per-group Delta R^2 (multichain complex minus
                                     per-chain): principled replacement for Fig 2E
  delta_r2_complex_vs_perchain.csv   the underlying deltas (transparency)

Run: conda run -n af3_eval python plot_publication_variance_partitioning.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from figsave import save_all_formats  # png600/pdf/svg/eps

# =====================================================================
# Shared publication style -- IDENTICAL to the existing figure scripts
# (plot_final_regression_analysis_results.py / plot_final_d-f_system_comparison.py)
# so these panels combine seamlessly into multipanel figures.
# =====================================================================
def set_pub_style(dpi: int = 300) -> None:
    sns.set_style("whitegrid", {
        "axes.edgecolor": "0.15",
        "grid.color": "0.88",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    sns.set_context("paper")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "savefig.dpi": dpi,
    })
    if __import__("os").environ.get("MATPLOTLIBRC"):
        # central figure style (figstyle.py) wins -> look consistent with all figures;
        # standalone runs (no MATPLOTLIBRC) keep the settings above.
        mpl.rcParams.update(mpl.rc_params_from_file(
            __import__("os").environ["MATPLOTLIBRC"], use_default_template=False))

# Feature-group palette -- EXACTLY matches GROUP_COLORS in
# plot_final_d-f_system_comparison.py (Tableau-10), so colours are consistent
# across every panel of the paper.
GROUP_ORDER = ["G0_af3", "G1_evolution", "G2_complexity", "G3_rna_intrinsic", "G4_interface", "G5_experimental"]
GROUP_LABEL = {
    "G0_af3": "AF3", "G1_evolution": "Evolution", "G2_complexity": "Complexity",
    "G3_rna_intrinsic": "RNA intrinsic", "G4_interface": "Interface",
    "G5_experimental": "Experimental",
}
GROUP_COLOR = {
    "G0_af3": "#4C78A8", "G1_evolution": "#59A14F", "G2_complexity": "#9C755F",
    "G3_rna_intrinsic": "#B279A2", "G4_interface": "#E15759", "G5_experimental": "#F28E2B",
}
ZERO_LINE_COLOR = "0.2"
DS_LABEL = {
    "all_complex": "All complexes", "multi_complex": "Multichain", "rp_1v1_complex": "RNA–protein 1:1",
    "multi_per_chain": "Multichain", "rnaOnly_per_chain": "Single-chain RNA", "rna_rna_per_chain": "Multi-chain RNA", "rp_1v1_per_chain": "RNA–protein 1:1",
}
TGT = {"Complex_TM": "cTM", "Dockq": "DockQ", "DockQ_Nr": "DockQ$_{Nr}$", "assigned_rna_tm": "RNA TM"}


def _row_label(ds, tg):
    return f"{DS_LABEL.get(ds, ds)}\n[{TGT.get(tg, tg)}]"


# figure widths (inches) -- aligned with the existing scripts (~6.8 in full width)
W_FULL = 6.8
W_HEAT = 6.6
W_DELTA = 5.6
VALUE_FS = 8         # value-label size (just below the 9 pt tick size)
SHOW_TITLES = True   # set False (via --no-titles) for clean multipanel assembly
# Panel E of the Figure-2 montage IS fig_delta; on the ~5.6 in canvas the central-rc
# fonts render much smaller than the neighbouring bar-ladder panels (ytick 18 / axis 20).
# Set explicit sizes so panel E matches the rest of Figure 2 after montage placement.
# Panel E is a shorter native canvas than panel F (6 rows ~4.1in vs 15 rows ~5.1in), so the
# montage scales E down LESS. To match panel F on-page (F: ytick 18, xlabel 20, xtick 16),
# panel E fonts are set to ~0.8x those pt values.
DELTA_TICK_FS = 15     # group labels (AF3, Complexity, ...) -> matches panel F y-labels
DELTA_XTICK_FS = 13    # numeric x-axis ticks -> matches panel F x-ticks
DELTA_XLABEL_FS = 13   # x-axis label -> smaller so the long label fits the 5.6in canvas
DELTA_VALUE_FS = 12    # +/-0.00 value annotations


def _save(fig, outdir, name):
    save_all_formats(fig, outdir / name, dpi=600)
    plt.close(fig)


def _maybe_title(ax, title):
    if SHOW_TITLES and title:
        ax.set_title(title, pad=6)


def _stacked(rows, groups, title, xlabel, value_prefix, outdir, name, full_lookup):
    """rows: list of (label, {group: shapley_r2}); groups: which groups to stack."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W_FULL, 0.46 * n + 1.5))
    y = np.arange(n)
    left = np.zeros(n)
    for g in groups:
        vals = np.array([max(0.0, rows[i][1].get(g, 0.0)) for i in range(n)])
        ax.barh(y, vals, left=left, height=0.66, color=GROUP_COLOR[g],
                label=GROUP_LABEL[g], edgecolor="white", linewidth=0.6, zorder=3)
        left += vals
    for i, (lab, _) in enumerate(rows):
        ax.text(left[i] + max(left) * 0.01, y[i], f"{value_prefix}{full_lookup[i]:.2f}",
                va="center", ha="left", fontsize=VALUE_FS)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(left) * 1.14)
    ax.grid(axis="y", visible=False)
    _maybe_title(ax, title)
    # legend below the axes so it never overlaps short bars / value labels
    ax.legend(ncol=len(groups), loc="upper center",
              bbox_to_anchor=(0.5, -0.16 - 0.02 * max(0, 6 - n)),
              handlelength=1.1, columnspacing=1.1, borderaxespad=0.0, frameon=False)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    _save(fig, outdir, name)


def load(summary_csv: Path) -> pd.DataFrame:
    return pd.read_csv(summary_csv)


def fig_complex(df, outdir):
    sub = df[(df["mode"] == "direct") & (df["dataset"].str.contains("complex"))].copy()
    sub = sub[sub["full_r2"] >= sub["af3_only_r2"]]          # stable only
    sub["is_dockq"] = sub["target"].str.lower().str.contains("ockq")
    sub = sub.sort_values(["is_dockq", "dataset"])
    rows, full = [], []
    for r in sub.itertuples():
        seg = {g: getattr(r, f"shap_{g}") for g in GROUP_ORDER}
        rows.append((_row_label(r.dataset, r.target), seg)); full.append(r.full_r2)
    _stacked(rows, GROUP_ORDER,
             "Variance in complex-level AF3 accuracy explained per dataset",
             "Out-of-fold $R^2$ (Shapley contribution of each feature group)",
             "$R^2=$", outdir, "fig_shapley_complex", full)


def fig_perchain(df, outdir):
    sub = df[(df["mode"] == "direct") & (df["dataset"].str.contains("per_chain"))].copy()
    sub = sub[sub["full_r2"] >= sub["af3_only_r2"]]          # drops unstable 1v1 per-chain
    sub = sub.sort_values("dataset")
    rows, full = [], []
    for r in sub.itertuples():
        seg = {g: getattr(r, f"shap_{g}") for g in GROUP_ORDER}
        rows.append((_row_label(r.dataset, r.target), seg)); full.append(r.full_r2)
    _stacked(rows, GROUP_ORDER,
             "Variance in per-chain RNA accuracy explained per system class",
             "Out-of-fold $R^2$ (Shapley contribution of each feature group)",
             "$R^2=$", outdir, "fig_shapley_perchain", full)


def fig_residual(df, outdir):
    sub = df[df["mode"] == "af3_error"].copy()
    if sub.empty:
        return
    sub["is_dockq"] = sub["target"].str.lower().str.contains("ockq")
    sub = sub.sort_values(["is_dockq", "dataset"])
    groups = [g for g in GROUP_ORDER if g != "G0_af3"]
    rows, full = [], []
    for r in sub.itertuples():
        seg = {g: getattr(r, f"shap_{g}") for g in groups}
        rows.append((_row_label(r.dataset, r.target), seg)); full.append(r.full_r2)
    _stacked(rows, groups,
             "What AF3 confidence misses: residual variance partition",
             "Out-of-fold $R^2$ on the AF3 residual (Shapley contribution)",
             "resid $R^2=$", outdir, "fig_shapley_residual", full)


def fig_heatmap(df, outdir):
    sub = df[(df["mode"] == "direct")].copy()
    sub = sub[sub["full_r2"] >= sub["af3_only_r2"]]
    sub["is_dockq"] = sub["target"].str.lower().str.contains("ockq")
    sub["is_pc"] = sub["dataset"].str.contains("per_chain")
    sub = sub.sort_values(["is_pc", "is_dockq", "dataset"])
    mat = np.array([[getattr(r, f"share_{g}") for g in GROUP_ORDER] for r in sub.itertuples()], dtype=float)
    ylabels = [_row_label(r.dataset, r.target).replace("\n", " ") for r in sub.itertuples()]
    fig, ax = plt.subplots(figsize=(W_HEAT, 0.45 * len(ylabels) + 1.4))
    vmax = max(60.0, np.nanmax(mat))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], rotation=28, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=VALUE_FS,
                        color="white" if v > 0.55 * vmax else "0.15")
    _maybe_title(ax, "Share of explained variance per feature group (%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("% of explained variance", fontsize=9)
    fig.tight_layout()
    _save(fig, outdir, "fig_shapley_share_heatmap")


def fig_delta(df, outdir):
    """Per-group Delta R^2 = (multichain complex) - (multichain per-chain), TM.
    Principled replacement for Fig 2E (coefficient delta on mismatched scales)."""
    d = df[df["mode"] == "direct"]
    cx = d[(d["dataset"] == "multi_complex") & (d["target"] == "Complex_TM")]
    pc = d[(d["dataset"] == "multi_per_chain") & (d["target"] == "assigned_rna_tm")]
    if cx.empty or pc.empty:
        return
    cx, pc = cx.iloc[0], pc.iloc[0]
    deltas = {g: float(cx[f"shap_{g}"]) - float(pc[f"shap_{g}"]) for g in GROUP_ORDER}
    order = sorted(GROUP_ORDER, key=lambda g: deltas[g])
    pd.DataFrame({
        "group": [GROUP_LABEL[g] for g in GROUP_ORDER],
        "shapley_complex_r2": [float(cx[f"shap_{g}"]) for g in GROUP_ORDER],
        "shapley_perchain_r2": [float(pc[f"shap_{g}"]) for g in GROUP_ORDER],
        "delta_r2": [deltas[g] for g in GROUP_ORDER],
    }).to_csv(outdir / "delta_r2_complex_vs_perchain.csv", index=False)

    fig, ax = plt.subplots(figsize=(W_DELTA, 0.5 * len(order) + 1.1))
    y = np.arange(len(order))
    vals = [deltas[g] for g in order]
    colors = [GROUP_COLOR[g] for g in order]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="none", zorder=3)
    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([GROUP_LABEL[g] for g in order], fontsize=DELTA_TICK_FS)
    ax.set_xlabel(r"$\Delta R^2$ (Shapley): multichain complex $-$ per-chain", fontsize=DELTA_XLABEL_FS)
    ax.tick_params(axis="x", labelsize=DELTA_XTICK_FS)
    ax.grid(axis="y", visible=False)
    _maybe_title(ax, "Feature-group contribution shift: global vs per-chain accuracy")
    mx = max(abs(min(vals)), abs(max(vals))) * 1.28
    ax.set_xlim(-mx, mx)
    for yi, v in zip(y, vals):
        ax.text(v + (0.012 * mx if v >= 0 else -0.012 * mx), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=DELTA_VALUE_FS)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    _save(fig, outdir, "fig_delta_r2_complex_vs_perchain")


def main():
    global SHOW_TITLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="variance_partitioning_final/_summary/summary_overview.csv")
    ap.add_argument("--outdir", default="publication_figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-titles", action="store_true",
                    help="omit panel titles (for clean multipanel assembly)")
    args = ap.parse_args()
    SHOW_TITLES = not args.no_titles
    set_pub_style(dpi=args.dpi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.summary))
    fig_complex(df, outdir)
    fig_perchain(df, outdir)
    fig_residual(df, outdir)
    fig_heatmap(df, outdir)
    fig_delta(df, outdir)
    print("wrote publication figures to", outdir.resolve())


if __name__ == "__main__":
    main()
