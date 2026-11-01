#!/usr/bin/env python3
"""
plot_variance_partitioning_summary.py

Aggregate all per-run Shapley-R^2 outputs (from variance_partitioning_shapley.py)
into:
  1. summary_overview.csv  -- one row per dataset x target x mode:
       n, af3_only_r2, full_r2, gap(full-af3), top non-AF3 group, and each
       group's Shapley-R^2 and share%.
  2. shapley_stacked_direct.png  -- stacked bars (length=full OOF R^2,
       segments=group Shapley-R^2) for direct targets. Bar = how much of the
       quality metric is explained, and by which feature group.
  3. shapley_heatmap_direct.png  -- datasets x groups, colour = Shapley share%.
  4. shapley_stacked_residual.png -- residual (af3_error) partitioning: what
       AF3 confidence MISSES, and which groups explain it (non-AF3 groups only).

Runs whose full model is UNSTABLE (full_r2 < af3_only_r2, i.e. adding groups
hurts out-of-fold) are excluded from the stacked/heatmap figures and flagged in
the table -- a stacked bar is not meaningful when contributions sum to < the
baseline.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figsave import save_all_formats  # png600/pdf/svg/eps

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUP_ORDER = ["G0_af3", "G1_evolution", "G2_complexity", "G3_rna_intrinsic", "G4_interface", "G5_experimental"]
GROUP_LABEL = {
    "G0_af3": "AF3 confidence",
    "G1_evolution": "Evolution",
    "G2_complexity": "Complexity",
    "G3_rna_intrinsic": "RNA intrinsic",
    "G4_interface": "Interface (pred.)",
    "G5_experimental": "Experimental/chem",
}
GROUP_COLOR = {
    "G0_af3": "#4C72B0",
    "G1_evolution": "#55A868",
    "G2_complexity": "#C44E52",
    "G3_rna_intrinsic": "#8172B3",
    "G4_interface": "#CCB974",
    "G5_experimental": "#64B5CD",
}

# nicer dataset labels
DS_LABEL = {
    "all_complex": "All complexes",
    "multi_complex": "Multichain complexes",
    "rp_1v1_complex": "RNA–protein 1:1 complexes",
    "multi_per_chain": "Multichain (per-chain)",
    "rp_1v1_per_chain": "RNA–protein 1:1 (per-chain)",
    "rnaOnly_per_chain": "Single-chain RNA (per-chain)",
    "rna_rna_per_chain": "Multi-chain RNA (per-chain)",
}


def load_all(root: Path) -> pd.DataFrame:
    rows = []
    for cfg_path in root.rglob("run_config.json"):
        d = cfg_path.parent
        sh = d / "shapley_r2.csv"
        if not sh.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        sdf = pd.read_csv(sh)
        full_r2 = float(cfg.get("full_model_oof_r2", np.nan))
        mode = cfg.get("target_mode")
        ycol = cfg.get("ycol")
        dataset = cfg.get("dataset")
        n = int(cfg.get("n_rows", 0))
        af3_only = np.nan
        g0row = sdf[sdf["group"] == "G0_af3"]
        if len(g0row):
            af3_only = float(g0row["increment_over_af3_r2"].iloc[0])
        for _, r in sdf.iterrows():
            rows.append({
                "dataset": dataset, "ycol": ycol, "mode": mode, "n": n,
                "full_r2": full_r2, "af3_only_r2": af3_only,
                "group": r["group"], "shapley_r2": float(r["shapley_r2"]),
                "share_pct": float(r["shapley_share_pct"]),
                "n_features": int(r["n_features_used"]),
            })
    return pd.DataFrame(rows)


def build_overview(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (dataset, ycol, mode), g in df.groupby(["dataset", "ycol", "mode"]):
        full_r2 = g["full_r2"].iloc[0]
        af3 = g["af3_only_r2"].iloc[0]
        n = g["n"].iloc[0]
        nonaf3 = g[g["group"] != "G0_af3"]
        top = nonaf3.sort_values("shapley_r2", ascending=False).iloc[0]
        unstable = (not np.isnan(af3)) and (full_r2 < af3)
        rec = {
            "dataset": dataset, "target": ycol, "mode": mode, "n": n,
            "af3_only_r2": round(af3, 3) if not np.isnan(af3) else np.nan,
            "full_r2": round(full_r2, 3),
            "gap_full_minus_af3": round(full_r2 - af3, 3) if not np.isnan(af3) else np.nan,
            "top_non_af3_group": GROUP_LABEL[top["group"]],
            "top_non_af3_shapley_r2": round(top["shapley_r2"], 3),
            "unstable_full_lt_af3": unstable,
        }
        for gk in GROUP_ORDER:
            gr = g[g["group"] == gk]
            rec[f"shap_{gk}"] = round(float(gr["shapley_r2"].iloc[0]), 3) if len(gr) else np.nan
            rec[f"share_{gk}"] = round(float(gr["share_pct"].iloc[0]), 1) if len(gr) else np.nan
        recs.append(rec)
    out = pd.DataFrame(recs)
    order = {"direct": 0, "af3_error": 1}
    out = out.sort_values(["mode", "target", "dataset"], key=lambda s: s.map(order).fillna(s) if s.name == "mode" else s)
    return out.reset_index(drop=True)


def _row_label(dataset, ycol):
    return f"{DS_LABEL.get(dataset, dataset)}\n[{ycol}]"


def stacked_direct(df: pd.DataFrame, out_png: Path):
    sub = df[df["mode"] == "direct"].copy()
    # build per (dataset,ycol)
    combos = sub[["dataset", "ycol", "full_r2", "af3_only_r2"]].drop_duplicates()
    combos["unstable"] = combos["full_r2"] < combos["af3_only_r2"]
    combos = combos[~combos["unstable"]]
    # order: TM first then DockQ; complexes before per-chain
    combos["is_dockq"] = combos["ycol"].str.lower().str.contains("ockq")
    combos = combos.sort_values(["is_dockq", "dataset"]).reset_index(drop=True)

    labels, bars = [], []
    for _, c in combos.iterrows():
        g = sub[(sub["dataset"] == c["dataset"]) & (sub["ycol"] == c["ycol"])]
        seg = {row["group"]: max(0.0, row["shapley_r2"]) for _, row in g.iterrows()}
        bars.append(seg)
        labels.append(_row_label(c["dataset"], c["ycol"]))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(labels) + 2.2), constrained_layout=True)
    left = np.zeros(len(labels))
    for gk in GROUP_ORDER:
        vals = np.array([bars[i].get(gk, 0.0) for i in range(len(labels))])
        ax.barh(y, vals, left=left, color=GROUP_COLOR[gk], label=GROUP_LABEL[gk], edgecolor="white", height=0.7)
        left += vals
    for i, c in enumerate(combos.itertuples()):
        ax.text(left[i] + 0.005, y[i], f"R²={c.full_r2:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-fold R²  (Shapley contribution of each feature group)")
    ax.set_title("Variance in AF3 quality explained per dataset — Shapley-R² partition\n(bar length = full-model OOF R²; segments = group contributions)")
    ax.legend(ncol=3, fontsize=8, loc="lower right", frameon=True)
    ax.set_xlim(0, max(left) * 1.12)
    save_all_formats(fig, out_png, dpi=600)
    plt.close(fig)


def heatmap_direct(df: pd.DataFrame, out_png: Path):
    sub = df[df["mode"] == "direct"].copy()
    combos = sub[["dataset", "ycol", "full_r2", "af3_only_r2"]].drop_duplicates()
    combos = combos[combos["full_r2"] >= combos["af3_only_r2"]]
    combos["is_dockq"] = combos["ycol"].str.lower().str.contains("ockq")
    combos = combos.sort_values(["is_dockq", "dataset"]).reset_index(drop=True)

    mat = np.full((len(combos), len(GROUP_ORDER)), np.nan)
    ylabels = []
    for i, c in enumerate(combos.itertuples()):
        g = sub[(sub["dataset"] == c.dataset) & (sub["ycol"] == c.ycol)]
        for j, gk in enumerate(GROUP_ORDER):
            gr = g[g["group"] == gk]
            if len(gr):
                mat[i, j] = float(gr["share_pct"].iloc[0])
        ylabels.append(_row_label(c.dataset, c.ycol))

    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(combos) + 2.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="rocket_r" if "rocket_r" in plt.colormaps() else "viridis", vmin=0, vmax=max(60, np.nanmax(mat)))
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        color="white" if mat[i, j] > 35 else "black", fontsize=8)
    ax.set_title("Share of explained variance per feature group (%)\nShapley-R² / full-model R²")
    fig.colorbar(im, ax=ax, label="% of explained variance")
    save_all_formats(fig, out_png, dpi=600)
    plt.close(fig)


def stacked_residual(df: pd.DataFrame, out_png: Path):
    sub = df[df["mode"] == "af3_error"].copy()
    if sub.empty:
        return
    combos = sub[["dataset", "ycol", "full_r2"]].drop_duplicates()
    combos["is_dockq"] = combos["ycol"].str.lower().str.contains("ockq")
    combos = combos.sort_values(["is_dockq", "dataset"]).reset_index(drop=True)
    groups_resid = [g for g in GROUP_ORDER if g != "G0_af3"]

    labels, bars = [], []
    for _, c in combos.iterrows():
        g = sub[(sub["dataset"] == c["dataset"]) & (sub["ycol"] == c["ycol"])]
        seg = {row["group"]: max(0.0, row["shapley_r2"]) for _, row in g.iterrows()}
        bars.append(seg)
        labels.append(_row_label(c["dataset"], c["ycol"]))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(labels) + 2.2), constrained_layout=True)
    left = np.zeros(len(labels))
    for gk in groups_resid:
        vals = np.array([bars[i].get(gk, 0.0) for i in range(len(labels))])
        ax.barh(y, vals, left=left, color=GROUP_COLOR[gk], label=GROUP_LABEL[gk], edgecolor="white", height=0.7)
        left += vals
    for i, c in enumerate(combos.itertuples()):
        ax.text(left[i] + 0.003, y[i], f"resid R²={c.full_r2:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-fold R² on the AF3-confidence residual (Shapley contribution)")
    ax.set_title("What AF3 confidence MISSES — residual variance partition\n(target = quality − AF3-only OOF prediction; AF3 group excluded by construction)")
    ax.legend(ncol=3, fontsize=8, loc="lower right", frameon=True)
    ax.set_xlim(0, max(left) * 1.12 if max(left) > 0 else 1)
    save_all_formats(fig, out_png, dpi=600)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="variance_partitioning_final")
    ap.add_argument("--outdir", default="variance_partitioning_final/_summary")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(root)
    if df.empty:
        raise SystemExit(f"No shapley_r2.csv found under {root}")
    overview = build_overview(df)
    overview.to_csv(out / "summary_overview.csv", index=False)

    stacked_direct(df, out / "shapley_stacked_direct.png")
    heatmap_direct(df, out / "shapley_heatmap_direct.png")
    stacked_residual(df, out / "shapley_stacked_residual.png")

    # console: compact overview
    cols = ["dataset", "target", "mode", "n", "af3_only_r2", "full_r2",
            "gap_full_minus_af3", "top_non_af3_group", "unstable_full_lt_af3"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(overview[cols].to_string(index=False))
    print(f"\nWrote: {out/'summary_overview.csv'} and 3 figures under {out}/")


if __name__ == "__main__":
    main()
