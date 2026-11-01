#!/usr/bin/env python3
"""
plot_final_d-f_system_comparison.py

Create one coefficient-comparison plot at a time for:
  D = cross-system heatmap
  E = multichain per-chain vs multichain complex delta plot
  F = stepwise system delta plot

Expected input run structure
----------------------------
<RUN_DIR>/
  run_config.json
  path_name_map.json
  f/
    c/
      coef__a31.csv
      feat__a31.csv
      metrics_summary.csv
      model_name_map.csv
      ...

The script works directly on resolved RUN_DIR paths, i.e. directories
that contain run_config.json.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figsave import save_all_formats  # multi-format (png600/pdf/svg/eps) saver

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Labels and grouping
# =============================================================================

FEATURE_PRETTY = {
    "af3_global_pae_avg": "AF3 PAE",
    "af3_rna_pLDDT_avg": "AF3 RNA pLDDT",
    "af3_chain_pair_pae_min": "Min chain-pair PAE",
    "af3_rna_pTM_list_value": "AF3 RNA pTM",
    "af3_protein_pLDDT_avg": "AF3 protein pLDDT",
    "af3_protein_pTM_list_value": "AF3 protein pTM",
    "af3_protein_pTM_mean": "AF3 protein pTM",
    "af3_protein_ipTM_list_value": "AF3 protein ipTM",
    "af3_protein_ipTM_mean": "AF3 protein ipTM",
    "af3_rna_ipTM_list_value": "AF3 RNA ipTM",
    "af3_rna_ipTM_mean": "AF3 RNA ipTM",

    "log10_rna_msa_size": "RNA MSA size",
    "log10_protein_msa_size": "Protein MSA size",
    "n_unique_rna_families": "RNA families",
    "n_rbp_templates": "RBP templates",
    "n_protein_templates": "Protein templates",

    "NumberRNAs": "RNA chains",
    "NumberProteins": "Protein chains",
    "rna_total_length": "RNA total length",
    "pred_rna_chain_length": "RNA chain length",
    "protein_total_length": "Protein total length",
    "total_complex_length": "Total complex length",
    "protein_to_rna_length_ratio": "Protein/RNA length ratio",
    "max_rna_fraction_of_total_rna": "Largest RNA fraction",

    "RNA_SequenceComplexity": "RNA sequence complexity",
    "RNA_Shannon_Entropy_K1": "RNA entropy (k=1)",
    "RNA_MFE_Value": "RNA MFE",
    "RNA_Ensemble_Diversity": "RNA ensemble diversity",
    "RNA_Centroid_Energy": "RNA centroid energy",
    "agg_rna_sequence_complexity": "Agg. RNA complexity",
    "agg_rna_shannon_entropy_k2": "Agg. RNA entropy (k=2)",
    "num_stems": "Stems",
    "num_hairpin_loops": "Hairpin loops",
    "num_internal_loops": "Internal loops",
    "num_multibranch_loops": "Multibranch loops",
    "num_pseudoknots": "Pseudoknots",
    "log10_kd": "log10 Kd",

    "pred_ProteinRNAInterfaceArea": "Pred. interface area",
    "pred_ProteinRNAInterfaceRatio": "Pred. interface ratio",
    "n_unique_rna_interface_residues_global": "RNA interface residues",
    "n_unique_protein_interface_residues_global": "Protein interface residues",
    "rna_interface_coverage_global": "RNA interface coverage",
    "interface_contact_distance_mean": "Mean contact distance",
    "interface_contact_distance_median": "Median contact distance",
    "frac_contacts_lt_3_5A": "Frac. contacts < 3.5Å",
    "interface_rna_motif_length_sum": "RNA motif length sum",
    "interface_rna_motif_coverage_of_total_rna": "RNA motif coverage",
    "interface_frac_contacts_to_C": "Frac. contacts to C",
    "chain_n_unique_rna_interface_residues": "Chain RNA interface residues",
    "chain_interface_contact_distance_mean": "Chain mean contact distance",
    "chain_frac_contacts_lt_3_5A": "Chain frac. contacts < 3.5Å",
    "chain_interface_rna_motif_length_sum": "Chain RNA motif length sum",

    "n_ions_listed": "Listed ions",
    "n_ligands_listed": "Listed ligands",
    "ionic_strength": "Ionic strength",
    "peg_percent": "PEG %",
    "buffer_m": "Buffer molarity",
    "has_peg": "PEG present",
    "exp_pH": "pH",
    "exp_Temp": "Temperature",
    "XRayResolution": "X-ray resolution",
    "is_xray": "X-ray structure",
}

GROUP_LABELS = {
    "g0": "AF3",
    "g1": "Evolution",
    "g2": "Complexity",
    "g3": "RNA intrinsic",
    "g4": "Interface",
    "g5": "Experimental",
    "other": "Other",
}

GROUP_COLORS = {
    "g0": "#4C78A8",
    "g1": "#59A14F",
    "g2": "#9C755F",
    "g3": "#B279A2",
    "g4": "#E15759",
    "g5": "#F28E2B",
    "other": "#7F7F7F",
}

GROUP_ORDER = ["g0", "g1", "g2", "g3", "g4", "g5", "other"]

GROUP_REGISTRY = {
    "g0": {
        "af3_rna_pLDDT_avg",
        "af3_global_pae_avg",
        "af3_chain_pair_pae_min",
        "af3_rna_pTM_list_value",
        # protein-side + RNA ipTM confidence (added to G0); complex uses *_mean,
        # per-chain uses *_list_value -- list both so neither falls to "other".
        "af3_protein_pLDDT_avg",
        "af3_protein_pTM_mean",
        "af3_protein_ipTM_mean",
        "af3_rna_ipTM_mean",
        "af3_protein_pTM_list_value",
        "af3_protein_ipTM_list_value",
        "af3_rna_ipTM_list_value",
    },
    "g1": {
        "log10_rna_msa_size",
        "log10_protein_msa_size",
        "n_unique_rna_families",
        "n_rbp_templates",
        "n_protein_templates",
        "rna_family",
    },
    "g2": {
        "NumberRNAs",
        "NumberProteins",
        "rna_total_length",
        "pred_rna_chain_length",
        "protein_total_length",
        "total_complex_length",
        "protein_to_rna_length_ratio",
        "max_rna_fraction_of_total_rna",
    },
    "g3": {
        "RNA_SequenceComplexity",
        "RNA_Shannon_Entropy_K1",
        "RNA_MFE_Value",
        "RNA_Ensemble_Diversity",
        "RNA_Centroid_Energy",
        "agg_rna_sequence_complexity",
        "agg_rna_shannon_entropy_k2",
        "num_stems",
        "num_hairpin_loops",
        "num_internal_loops",
        "num_multibranch_loops",
        "num_pseudoknots",
        "log10_kd",
    },
    "g4": {
        "pred_ProteinRNAInterfaceArea",
        "pred_ProteinRNAInterfaceRatio",
        "n_unique_rna_interface_residues_global",
        "n_unique_protein_interface_residues_global",
        "rna_interface_coverage_global",
        "interface_contact_distance_mean",
        "interface_contact_distance_median",
        "frac_contacts_lt_3_5A",
        "interface_rna_motif_length_sum",
        "interface_rna_motif_coverage_of_total_rna",
        "interface_frac_contacts_to_C",
        "chain_n_unique_rna_interface_residues",
        "chain_interface_contact_distance_mean",
        "chain_frac_contacts_lt_3_5A",
        "chain_interface_rna_motif_length_sum",
    },
    "g5": {
        "n_ions_listed",
        "n_ligands_listed",
        "ionic_strength",
        "peg_percent",
        "buffer_m",
        "has_peg",
        "peg_concentration_category",
        "buffer_category",
        "exp_pH",
        "exp_Temp",
        "XRayResolution",
        "is_xray",
    },
}


def pretty_feature_name(name: str) -> str:
    name = str(name)

    if name.startswith("rna_family_"):
        return f"RNA family: {name.replace('rna_family_', '').upper()}"

    if name.startswith("peg_concentration_category_"):
        val = name.replace("peg_concentration_category_", "")
        return f"PEG: {val.replace('_', ' ')}"

    if name.startswith("buffer_category_"):
        val = name.replace("buffer_category_", "")
        return f"Buffer: {val.replace('_', ' ')}"

    return FEATURE_PRETTY.get(name, name.replace("_", " "))


def infer_group(feature: str) -> str:
    for group, feats in GROUP_REGISTRY.items():
        if feature in feats:
            return group

    if feature.startswith("rna_family_"):
        return "g1"
    if feature.startswith("peg_concentration_category_"):
        return "g5"
    if feature.startswith("buffer_category_"):
        return "g5"

    return "other"


# =============================================================================
# IO helpers
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_model_name_map(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["model_full"].astype(str), df["model_tag"].astype(str)))


def resolve_coef_tag(coef_model: str, model_map: Dict[str, str]) -> str:
    s = str(coef_model).strip()

    tag_lut = {str(v).lower(): str(v).lower() for v in model_map.values()}
    if s.lower() in tag_lut:
        return tag_lut[s.lower()]

    if s in model_map:
        return str(model_map[s]).lower()

    full_lut = {str(k).lower(): str(v).lower() for k, v in model_map.items()}
    if s.lower() in full_lut:
        return full_lut[s.lower()]

    raise ValueError(
        f"Could not resolve coef model '{coef_model}'. "
        f"Known model names: {list(model_map.keys())}"
    )


def locate_coef_csv(run_dir: Path, coef_model: str) -> Path:
    cdir = run_dir / "f" / "c"
    mm = cdir / "model_name_map.csv"
    if not mm.exists():
        raise FileNotFoundError(f"Missing model_name_map.csv: {mm}")
    model_map = load_model_name_map(mm)
    coef_tag = resolve_coef_tag(coef_model, model_map)
    coef_csv = cdir / f"coef__{coef_tag}.csv"
    if not coef_csv.exists():
        raise FileNotFoundError(f"Missing coefficient file: {coef_csv}")
    return coef_csv


def load_coef_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)

    if "feature" in df.columns and "coef" in df.columns:
        s = pd.Series(df["coef"].to_numpy(dtype=float), index=df["feature"].astype(str))
        return s.dropna()

    if "coef" in df.columns:
        first_col = df.columns[0]
        s = pd.Series(df["coef"].to_numpy(dtype=float), index=df[first_col].astype(str))
        return s.dropna()

    s = pd.read_csv(path, index_col=0).squeeze("columns")
    if isinstance(s, pd.Series):
        s.index = s.index.astype(str)
        return s.astype(float).dropna()

    raise ValueError(f"Unexpected coefficient file format: {path}")


def parse_labelled_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    run_dir = Path(path).expanduser()
    if not (run_dir / "run_config.json").exists():
        raise FileNotFoundError(f"Not a valid run directory: {run_dir}")
    return label.strip(), run_dir


# =============================================================================
# Plot style
# =============================================================================

def set_pub_style(args: argparse.Namespace) -> None:
    sns.set_style(
        "whitegrid",
        {
            "axes.edgecolor": args.axes_edgecolor,
            "grid.color": args.grid_color,
            "grid.linestyle": args.grid_linestyle,
            "grid.linewidth": args.grid_linewidth,
            "axes.facecolor": args.axes_facecolor,
            "figure.facecolor": args.figure_facecolor,
        },
    )
    sns.set_context(args.context)

    mpl.rcParams.update(
        {
            "font.family": args.font_family,
            "font.sans-serif": [x.strip() for x in args.font_sans.split(",") if x.strip()],
            "pdf.fonttype": args.pdf_fonttype,
            "ps.fonttype": args.ps_fonttype,
            "savefig.dpi": args.dpi,
            "axes.linewidth": args.axes_linewidth,
            "axes.labelsize": args.axes_labelsize,
            "axes.titlesize": args.axes_titlesize,
            "xtick.labelsize": args.xtick_labelsize,
            "ytick.labelsize": args.ytick_labelsize,
            "legend.fontsize": args.legend_fontsize,
            "figure.titlesize": args.figure_titlesize,
        }
    )
    if __import__("os").environ.get("MATPLOTLIBRC"):
        # central figure style (figstyle.py) sets font family + colors so we look
        # consistent with all figures; standalone runs keep the settings above.
        mpl.rcParams.update(mpl.rc_params_from_file(
            __import__("os").environ["MATPLOTLIBRC"], use_default_template=False))
        # ...but re-assert the explicit CLI font SIZES: these panels are rendered on
        # a ~5-7 in canvas then downscaled ~2x into the 6.9 in montage, so they need
        # larger-than-central sizes to land at the common on-page target. Without this
        # the central 8 pt would win and the panel labels collapse (Fig2 D/E/F bug).
        mpl.rcParams.update({
            "axes.labelsize": args.axes_labelsize,
            "axes.titlesize": args.axes_titlesize,
            "xtick.labelsize": args.xtick_labelsize,
            "ytick.labelsize": args.ytick_labelsize,
            "legend.fontsize": args.legend_fontsize,
            "figure.titlesize": args.figure_titlesize,
        })


# =============================================================================
# Data prep
# =============================================================================

def build_coef_table(run_items: List[Tuple[str, Path]], coef_model: str) -> pd.DataFrame:
    series = []
    for label, run_dir in run_items:
        coef_csv = locate_coef_csv(run_dir, coef_model)
        s = load_coef_series(coef_csv)
        s.name = label
        series.append(s)
    return pd.concat(series, axis=1)


def finalize_feature_table(
    df: pd.DataFrame,
    feature_set: str,
    drop_all_zero_rows: bool,
    sort_features: str,
    top_n: int,
    top_k_per_group: int,
) -> pd.DataFrame:
    out = df.copy()

    if feature_set == "intersection":
        out = out.dropna(how="any")
    else:
        out = out.fillna(0.0)

    if drop_all_zero_rows:
        out = out.loc[(out.abs().sum(axis=1) > 0)]

    if out.empty:
        return out

    if sort_features == "max_abs":
        scores = out.abs().max(axis=1)
    elif sort_features == "mean_abs":
        scores = out.abs().mean(axis=1)
    elif sort_features == "variance":
        scores = out.var(axis=1)
    elif sort_features == "alphabetical":
        scores = pd.Series(np.arange(len(out)), index=sorted(out.index))
        out = out.loc[sorted(out.index)]
        scores = pd.Series(np.arange(len(out)), index=out.index)
    else:
        raise ValueError(f"Unknown sort_features: {sort_features}")

    work = pd.DataFrame({"feature": out.index, "score": scores.loc[out.index].values})
    work["group"] = work["feature"].map(infer_group)
    work["group_order"] = work["group"].map(lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)

    if top_k_per_group > 0:
        work = (
            work.sort_values(["group_order", "score", "feature"], ascending=[True, False, True])
            .groupby("group", sort=False, group_keys=False)
            .head(top_k_per_group)
        )

    if top_n and top_n > 0:
        work = work.sort_values(["score", "feature"], ascending=[False, True]).head(top_n)

    if sort_features == "alphabetical":
        work = work.sort_values(["group_order", "feature"], ascending=[True, True])
    else:
        work = work.sort_values(["group_order", "score", "feature"], ascending=[True, False, True])

    return out.loc[work["feature"].tolist()]


def prepare_group_boundaries(features: List[str]) -> Tuple[List[Tuple[str, int, int]], List[float]]:
    groups = [infer_group(f) for f in features]
    group_blocks: List[Tuple[str, int, int]] = []
    separator_positions: List[float] = []

    if not groups:
        return group_blocks, separator_positions

    start = 0
    current = groups[0]
    for i in range(1, len(groups)):
        if groups[i] != current:
            group_blocks.append((current, start, i - 1))
            separator_positions.append(i)
            start = i
            current = groups[i]
    group_blocks.append((current, start, len(groups) - 1))
    return group_blocks, separator_positions


# =============================================================================
# Plotters
# =============================================================================

def plot_heatmap(df: pd.DataFrame, outfile: Path, args: argparse.Namespace) -> None:
    set_pub_style(args)

    plot_df = df.copy()
    feature_names = plot_df.index.tolist()
    pretty_names = [pretty_feature_name(x) for x in feature_names]
    group_blocks, separator_positions = prepare_group_boundaries(feature_names)

    n_rows = len(plot_df)
    fig_h = max(args.heatmap_min_height, args.heatmap_row_height * n_rows)

    # 3 axes:
    #   ax_labels : feature labels only
    #   ax_strip  : colored group stripe with vertical group names
    #   ax        : heatmap
    fig, (ax_labels, ax_strip, ax) = plt.subplots(
        ncols=3,
        figsize=(args.heatmap_width, fig_h),
        gridspec_kw={
            "width_ratios": [
                args.heatmap_label_col_width,
                args.heatmap_group_strip_width,
                1.0,
            ],
            "wspace": args.heatmap_strip_pad,
        },
    )

    # -----------------------------------------------------------------
    # Heatmap
    # -----------------------------------------------------------------
    hm = sns.heatmap(
        plot_df,
        cmap=args.heatmap_cmap,
        center=0.0,
        vmin=args.heatmap_vmin,
        vmax=args.heatmap_vmax,
        linewidths=args.heatmap_linewidth,
        linecolor=args.heatmap_linecolor,
        cbar_kws={
            "label": args.heatmap_cbar_label,
            "shrink": args.heatmap_cbar_shrink,
        },
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("" if args.hide_ylabel else args.ylabel)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, length=0)
    ax.tick_params(axis="x", rotation=args.heatmap_xtick_rotation)

    if args.title:
        ax.set_title(args.title, pad=args.title_pad)

    # match custom side axes to heatmap row coordinates
    # seaborn heatmap uses data coordinates [0, n_rows] with inverted y-axis
    ax_labels.set_xlim(0.0, 1.0)
    ax_labels.set_ylim(ax.get_ylim())
    ax_labels.axis("off")

    ax_strip.set_xlim(0.0, 1.0)
    ax_strip.set_ylim(ax.get_ylim())
    ax_strip.axis("off")

    # -----------------------------------------------------------------
    # Feature label column
    # -----------------------------------------------------------------
    for i, label in enumerate(pretty_names):
        ax_labels.text(
            args.heatmap_feature_label_x,
            i + 0.5,
            label,
            ha="right",
            va="center",
            fontsize=args.heatmap_feature_label_fontsize,
            color=args.heatmap_feature_label_color,
        )

    # -----------------------------------------------------------------
    # Group stripe with vertical labels
    # -----------------------------------------------------------------
    strip_x0 = args.heatmap_group_strip_x0
    strip_x1 = args.heatmap_group_strip_x1
    strip_w = strip_x1 - strip_x0

    for group, start, end in group_blocks:
        y0 = start
        h = end - start + 1
        color = GROUP_COLORS.get(group, GROUP_COLORS["other"])

        ax_strip.add_patch(
            Rectangle(
                (strip_x0, y0),
                strip_w,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=args.heatmap_group_strip_alpha,
            )
        )

        if args.heatmap_show_group_names:
            ax_strip.text(
                strip_x0 + 0.5 * strip_w,
                y0 + 0.5 * h,
                GROUP_LABELS.get(group, group),
                ha="center",
                va="center",
                rotation=90,
                fontsize=args.heatmap_group_name_fontsize,
                color=args.heatmap_group_name_color,
                fontweight=args.heatmap_group_name_fontweight,
            )

    # -----------------------------------------------------------------
    # Separators
    # -----------------------------------------------------------------
    if args.heatmap_show_group_separators:
        x0_hm, x1_hm = ax.get_xlim()
        for pos in separator_positions:
            ax.hlines(
                pos,
                x0_hm,
                x1_hm,
                colors=args.heatmap_group_linecolor,
                linewidth=args.heatmap_group_linewidth,
            )
            ax_strip.hlines(
                pos,
                0.0,
                1.0,
                colors=args.heatmap_group_linecolor,
                linewidth=args.heatmap_group_linewidth,
            )
            ax_labels.hlines(
                pos,
                args.heatmap_label_separator_x0,
                args.heatmap_label_separator_x1,
                colors=args.heatmap_group_linecolor,
                linewidth=args.heatmap_group_linewidth,
            )

    fig.subplots_adjust(
        left=args.heatmap_left,
        right=args.heatmap_right,
        top=args.heatmap_top,
        bottom=args.heatmap_bottom,
    )

    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)

def plot_multichain_shift(
    s_a: pd.Series,
    s_b: pd.Series,
    label_a: str,
    label_b: str,
    outfile: Path,
    args: argparse.Namespace,
) -> None:
    set_pub_style(args)

    df = pd.concat([s_a.rename("a"), s_b.rename("b")], axis=1).fillna(0.0)
    df["delta"] = df["b"] - df["a"]
    df["abs_delta"] = df["delta"].abs()
    df["group"] = [infer_group(f) for f in df.index]

    if args.drop_all_zero_rows:
        df = df.loc[df["abs_delta"] > 0]

    if args.top_n and args.top_n > 0:
        df = df.sort_values("abs_delta", ascending=False).head(args.top_n)

    df = df.sort_values("delta", ascending=True)

    labels = [pretty_feature_name(f) for f in df.index]
    colors = [GROUP_COLORS.get(g, GROUP_COLORS["other"]) for g in df["group"]]

    fig_h = max(args.bar_min_height, args.bar_row_height * len(df))
    fig, ax = plt.subplots(figsize=(args.bar_width, fig_h))

    ax.barh(labels, df["delta"].values, color=colors, edgecolor="none", height=args.single_bar_height, zorder=3)
    ax.axvline(0, color=args.zero_line_color, linewidth=args.zero_linewidth, zorder=4)

    ax.set_xlabel(args.delta_xlabel or f"Coefficient change ({label_b} − {label_a})")
    ax.set_ylabel("" if args.hide_ylabel else args.ylabel)

    if args.show_group_legend:
        handles = []
        used = []
        present = set(df["group"])
        for g in GROUP_ORDER:
            if g in present:
                handles.append(Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]))
                used.append(GROUP_LABELS[g])
        ax.legend(
            handles,
            used,
            loc=args.legend_loc,
            frameon=args.legend_frameon,
            ncol=args.legend_ncol,
            borderaxespad=0.4,
        )

    if args.title:
        ax.set_title(args.title, pad=args.title_pad)

    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.subplots_adjust(left=args.bar_left, right=args.bar_right, top=args.bar_top, bottom=args.bar_bottom)

    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)


def plot_system_delta(
    s_rna: pd.Series,
    s_1v1: pd.Series,
    s_multi: pd.Series,
    outfile: Path,
    args: argparse.Namespace,
) -> None:
    set_pub_style(args)

    df = pd.concat(
        [
            s_rna.rename("rna_only"),
            s_1v1.rename("one_v_one"),
            s_multi.rename("multichain"),
        ],
        axis=1,
    ).fillna(0.0)

    df["delta_1"] = df["one_v_one"] - df["rna_only"]
    df["delta_2"] = df["multichain"] - df["one_v_one"]
    df["score"] = np.maximum(df["delta_1"].abs(), df["delta_2"].abs())

    if args.drop_all_zero_rows:
        df = df.loc[df["score"] > 0]

    if args.top_n and args.top_n > 0:
        df = df.sort_values("score", ascending=False).head(args.top_n)

    if args.system_delta_sort == "transition_1":
        df = df.sort_values("delta_1", ascending=True)
    elif args.system_delta_sort == "transition_2":
        df = df.sort_values("delta_2", ascending=True)
    else:
        df = df.sort_values("score", ascending=True)

    labels = [pretty_feature_name(f) for f in df.index]
    y = np.arange(len(df))
    h = args.system_delta_bar_height

    fig_h = max(args.bar_min_height, args.bar_row_height * len(df))
    fig, ax = plt.subplots(figsize=(args.bar_width, fig_h))

    ax.barh(
        y - h / 2.0,
        df["delta_1"].values,
        height=h,
        color=args.transition1_color,
        edgecolor="none",
        label=args.transition1_label,
        zorder=3,
    )
    ax.barh(
        y + h / 2.0,
        df["delta_2"].values,
        height=h,
        color=args.transition2_color,
        edgecolor="none",
        label=args.transition2_label,
        zorder=3,
    )
    ax.axvline(0, color=args.zero_line_color, linewidth=args.zero_linewidth, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(args.delta_xlabel or "Coefficient change")
    ax.set_ylabel("" if args.hide_ylabel else args.ylabel)

    _legkw = dict(loc=args.legend_loc, frameon=args.legend_frameon,
                  ncol=args.legend_ncol, borderaxespad=0.4)
    if args.legend_bbox is not None:
        _legkw["bbox_to_anchor"] = tuple(args.legend_bbox)
    ax.legend(**_legkw)

    if args.title:
        ax.set_title(args.title, pad=args.title_pad)

    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, top=True, right=True)

    fig.subplots_adjust(left=args.bar_left, right=args.bar_right, top=args.bar_top, bottom=args.bar_bottom)

    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AF3 coefficient comparison panels one plot at a time.")

    p.add_argument("--mode", required=True, choices=["heatmap", "multichain_shift", "system_delta"])

    p.add_argument("--run", action="append", default=[], help="For heatmap mode: LABEL=RUN_DIR")
    p.add_argument("--run-a", default=None, help="For multichain_shift: LABEL=RUN_DIR")
    p.add_argument("--run-b", default=None, help="For multichain_shift: LABEL=RUN_DIR")
    p.add_argument("--run-rnaonly", default=None, help="For system_delta: LABEL=RUN_DIR")
    p.add_argument("--run-1v1", default=None, help="For system_delta: LABEL=RUN_DIR")
    p.add_argument("--run-multi", default=None, help="For system_delta: LABEL=RUN_DIR")

    p.add_argument(
        "--coef-model",
        default="A31_af3_evolution_complexity_rna_intrinsic_interface_experimental",
        help="Model full name or tag, e.g. A31... or a31",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default=None)
    p.add_argument("--ylabel", default="Feature")
    p.add_argument("--hide-ylabel", action="store_true")

    p.add_argument("--feature-set", choices=["union", "intersection"], default="union")
    p.add_argument("--sort-features", choices=["max_abs", "mean_abs", "variance", "alphabetical"], default="max_abs")
    p.add_argument("--top-n", type=int, default=0)
    p.add_argument("--top-k-per-group", type=int, default=3)
    p.add_argument("--drop-all-zero-rows", action="store_true")

    p.add_argument("--system-delta-sort", choices=["score", "transition_1", "transition_2"], default="score")
    p.add_argument("--delta-xlabel", default=None)

    # Style
    p.add_argument("--context", default="paper", choices=["paper", "notebook", "talk", "poster"])
    p.add_argument("--font-family", default="sans-serif")
    p.add_argument("--font-sans", default="Arial,DejaVu Sans,Liberation Sans,Helvetica")
    p.add_argument("--pdf-fonttype", type=int, default=42)
    p.add_argument("--ps-fonttype", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--figure-facecolor", default="white")
    p.add_argument("--axes-facecolor", default="white")
    p.add_argument("--axes-edgecolor", default="0.15")
    p.add_argument("--grid-color", default="0.88")
    p.add_argument("--grid-linestyle", default="-")
    p.add_argument("--grid-linewidth", type=float, default=0.7)
    p.add_argument("--axes-linewidth", type=float, default=0.8)

    p.add_argument("--axes-labelsize", type=float, default=11)
    p.add_argument("--axes-titlesize", type=float, default=12)
    p.add_argument("--xtick-labelsize", type=float, default=9)
    p.add_argument("--ytick-labelsize", type=float, default=9)
    p.add_argument("--legend-fontsize", type=float, default=9)
    p.add_argument("--figure-titlesize", type=float, default=14)
    p.add_argument("--title-pad", type=float, default=8.0)

    # Heatmap layout
    p.add_argument("--heatmap-width", type=float, default=4.9)
    p.add_argument("--heatmap-row-height", type=float, default=0.34)
    p.add_argument("--heatmap-min-height", type=float, default=4.8)
    p.add_argument("--heatmap-cmap", default="vlag")
    p.add_argument("--heatmap-vmin", type=float, default=None)
    p.add_argument("--heatmap-vmax", type=float, default=None)
    p.add_argument("--heatmap-linewidth", type=float, default=0.45)
    p.add_argument("--heatmap-linecolor", default="white")
    p.add_argument("--heatmap-cbar-label", default="Coefficient")
    p.add_argument("--heatmap-cbar-shrink", type=float, default=0.86)
    p.add_argument("--heatmap-xtick-rotation", type=float, default=0.0)

    p.add_argument("--heatmap-show-group-separators", action="store_true")
    p.add_argument("--heatmap-group-linecolor", default="0.72")
    p.add_argument("--heatmap-group-linewidth", type=float, default=0.8)

    # p.add_argument("--heatmap-group-strip-width", type=float, default=0.18)
    # p.add_argument("--heatmap-strip-pad", type=float, default=0.02)
    # p.add_argument("--heatmap-group-strip-alpha", type=float, default=1.0)
    # p.add_argument("--heatmap-show-group-names", action="store_true")
    # p.add_argument("--heatmap-group-name-fontsize", type=float, default=7.8)
    # p.add_argument("--heatmap-group-name-color", default="0.35")

    p.add_argument("--heatmap-left", type=float, default=0.10)
    p.add_argument("--heatmap-right", type=float, default=0.96)
    p.add_argument("--heatmap-top", type=float, default=0.97)
    p.add_argument("--heatmap-bottom", type=float, default=0.08)

    # p.add_argument("--heatmap-label-col-width", type=float, default=1.05)
    # p.add_argument("--heatmap-feature-label-x", type=float, default=0.98)
    p.add_argument("--heatmap-feature-label-fontsize", type=float, default=8.8)
    p.add_argument("--heatmap-feature-label-color", default="0.15")

    p.add_argument("--heatmap-group-strip-width", type=float, default=0.20)
    p.add_argument("--heatmap-strip-pad", type=float, default=0.015)
    p.add_argument("--heatmap-group-strip-alpha", type=float, default=1.0)

    p.add_argument("--heatmap-group-strip-x0", type=float, default=0.10)
    p.add_argument("--heatmap-group-strip-x1", type=float, default=0.90)

    p.add_argument("--heatmap-show-group-names", action="store_true")
    p.add_argument("--heatmap-group-name-fontsize", type=float, default=7.5)
    p.add_argument("--heatmap-group-name-color", default="white")
    p.add_argument("--heatmap-group-name-fontweight", default="bold")

    p.add_argument("--heatmap-label-col-width", type=float, default=0.92)
    p.add_argument("--heatmap-feature-label-x", type=float, default=0.985)
    p.add_argument("--heatmap-label-separator-x0", type=float, default=0.56)
    p.add_argument("--heatmap-label-separator-x1", type=float, default=0.995)
    
    # Bar plots
    p.add_argument("--bar-width", type=float, default=5.3)
    p.add_argument("--bar-row-height", type=float, default=0.34)
    p.add_argument("--bar-min-height", type=float, default=4.8)
    p.add_argument("--single-bar-height", type=float, default=0.78)
    p.add_argument("--zero-line-color", default="0.2")
    p.add_argument("--zero-linewidth", type=float, default=1.0)

    p.add_argument("--show-group-legend", action="store_true")
    p.add_argument("--legend-loc", default="best")
    p.add_argument("--legend-frameon", action="store_true")
    p.add_argument("--legend-ncol", type=int, default=1)
    p.add_argument("--legend-bbox", type=float, nargs=2, default=None,
                   help="bbox_to_anchor (x y) for the legend; e.g. '0.5 -0.18' to place it "
                        "one row below the axes. When unset, plain loc-based placement is used.")

    p.add_argument("--bar-left", type=float, default=0.30)
    p.add_argument("--bar-right", type=float, default=0.98)
    p.add_argument("--bar-top", type=float, default=0.97)
    p.add_argument("--bar-bottom", type=float, default=0.08)

    p.add_argument("--transition1-color", default="#4C78A8")
    p.add_argument("--transition2-color", default="#F28E2B")
    p.add_argument("--transition1-label", default="1v1 − RNA$_1$")
    p.add_argument("--transition2-label", default="Multichain − 1v1")
    p.add_argument("--system-delta-bar-height", type=float, default=0.28)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.mode == "heatmap":
        if len(args.run) < 2:
            raise ValueError("--mode heatmap requires at least two --run LABEL=PATH entries")

        run_items = [parse_labelled_run(x) for x in args.run]
        df = build_coef_table(run_items, args.coef_model)
        df = finalize_feature_table(
            df=df,
            feature_set=args.feature_set,
            drop_all_zero_rows=args.drop_all_zero_rows,
            sort_features=args.sort_features,
            top_n=args.top_n,
            top_k_per_group=args.top_k_per_group,
        )
        plot_heatmap(df, args.out, args)

    elif args.mode == "multichain_shift":
        if args.run_a is None or args.run_b is None:
            raise ValueError("--mode multichain_shift requires --run-a and --run-b")

        label_a, run_a = parse_labelled_run(args.run_a)
        label_b, run_b = parse_labelled_run(args.run_b)

        s_a = load_coef_series(locate_coef_csv(run_a, args.coef_model))
        s_b = load_coef_series(locate_coef_csv(run_b, args.coef_model))

        plot_multichain_shift(
            s_a=s_a,
            s_b=s_b,
            label_a=label_a,
            label_b=label_b,
            outfile=args.out,
            args=args,
        )

    elif args.mode == "system_delta":
        if args.run_rnaonly is None or args.run_1v1 is None or args.run_multi is None:
            raise ValueError("--mode system_delta requires --run-rnaonly, --run-1v1, and --run-multi")

        _, run_rna = parse_labelled_run(args.run_rnaonly)
        _, run_1v1 = parse_labelled_run(args.run_1v1)
        _, run_multi = parse_labelled_run(args.run_multi)

        s_rna = load_coef_series(locate_coef_csv(run_rna, args.coef_model))
        s_1v1 = load_coef_series(locate_coef_csv(run_1v1, args.coef_model))
        s_multi = load_coef_series(locate_coef_csv(run_multi, args.coef_model))

        plot_system_delta(
            s_rna=s_rna,
            s_1v1=s_1v1,
            s_multi=s_multi,
            outfile=args.out,
            args=args,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()