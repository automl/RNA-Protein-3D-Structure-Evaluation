#!/usr/bin/env python3
"""
make_af3_publication_figures_v3.py

Create publication-ready figures from the NEW regression output structure,
with each plot written as an individual standalone PNG.

Expected run structure
----------------------
<RUN_DIR>/
  run_config.json
  path_name_map.json
  f/
    scenario_name_map.json
    c/
      metrics_summary.csv
      metrics_per_fold.csv
      model_name_map.csv
      candidate_features.csv
      missingness.csv
      coef__g0.csv
      coef__a7.csv
      coef__gx.csv
      coef_pos__gx.csv
      coef_neg__gx.csv
      feat__gx.csv
    p/
      coef__gx.png
      pvt__gx.png
      resid__gx.png

Notes
-----
- coef-model can be either the full model name (e.g. GX_all, A7_af3_evolution_interface_rnaintr_complexity)
  or the short model tag (e.g. gx, a7).
- The script resolves model tags from f/c/model_name_map.csv.
- Scatter plots are generated individually, one PNG per requested variable spec.
- Performance is generated as two separate standalone barplots.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figsave import save_all_formats  # multi-format (png600/pdf/svg/eps) saver

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ----------------------------- general utils -------------------------------- #

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def panel_letter(
    ax: plt.Axes,
    letter: str,
    x: float = -0.13,
    y: float = 1.08,
    fontsize: float = 13,
) -> None:
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top"
    )


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def load_model_name_map(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    required = {"model_full", "model_tag"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected model_name_map format: {path}")
    return dict(zip(df["model_full"].astype(str), df["model_tag"].astype(str)))


def resolve_model_identifier(name_or_tag: str, model_map: Dict[str, str]) -> str:
    s = str(name_or_tag).strip()

    if s in model_map:
        return s

    tag_to_full = {str(v).lower(): str(k) for k, v in model_map.items()}
    if s.lower() in tag_to_full:
        return tag_to_full[s.lower()]

    # allow case-insensitive full-name matching
    full_lower = {str(k).lower(): str(k) for k in model_map.keys()}
    if s.lower() in full_lower:
        return full_lower[s.lower()]

    raise ValueError(
        f"Could not resolve model '{name_or_tag}'. "
        f"Known full names: {', '.join(model_map.keys())}"
    )


def resolve_coef_tag(coef_model: str, model_map: Dict[str, str]) -> str:
    coef_model = str(coef_model)

    valid_tags = {v.lower() for v in model_map.values()}
    if coef_model.lower() in valid_tags:
        return coef_model.lower()

    if coef_model in model_map:
        return str(model_map[coef_model]).lower()

    # allow case-insensitive full-name match
    full_lower = {str(k).lower(): str(v).lower() for k, v in model_map.items()}
    if coef_model.lower() in full_lower:
        return full_lower[coef_model.lower()]

    raise ValueError(
        f"Could not resolve coef model '{coef_model}'. "
        f"Valid full names include: {', '.join(model_map.keys())}"
    )


def discover_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    run_dirs = [
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "run_config.json").exists()
    ]
    return sorted(run_dirs)


def pick_run_dir(
    runs_root: Path,
    target_label: Optional[str] = None,
    analysis_name_contains: Optional[str] = None,
    run_name_contains: Optional[str] = None,
) -> Path:
    candidates = discover_run_dirs(runs_root)

    if target_label is not None:
        kept = []
        for p in candidates:
            try:
                cfg = load_json(p / "run_config.json")
            except Exception:
                continue
            if str(cfg.get("target_label", "")) == str(target_label):
                kept.append(p)
        candidates = kept

    if analysis_name_contains is not None:
        kept = []
        for p in candidates:
            try:
                cfg = load_json(p / "run_config.json")
            except Exception:
                continue
            if analysis_name_contains in str(cfg.get("analysis_name", "")):
                kept.append(p)
        candidates = kept

    if run_name_contains is not None:
        candidates = [p for p in candidates if run_name_contains in p.name]

    if len(candidates) == 0:
        raise FileNotFoundError("No matching run directory found.")
    if len(candidates) > 1:
        msg = "Multiple matching run directories found:\n" + "\n".join(str(p) for p in candidates)
        raise ValueError(msg)

    return candidates[0]


def locate_analysis_files(run_dir: Path, coef_model: Optional[str] = None) -> Dict[str, Path]:
    cdir = run_dir / "f" / "c"
    run_cfg = run_dir / "run_config.json"
    model_map_csv = cdir / "model_name_map.csv"

    if not run_cfg.exists():
        raise FileNotFoundError(f"Missing run_config.json: {run_cfg}")
    if not cdir.exists():
        raise FileNotFoundError(f"Missing f/c directory: {cdir}")
    if not model_map_csv.exists():
        raise FileNotFoundError(f"Missing model_name_map.csv: {model_map_csv}")

    files = {
        "run_config": run_cfg,
        "metrics_summary": cdir / "metrics_summary.csv",
        "metrics_per_fold": cdir / "metrics_per_fold.csv",
        "model_name_map": model_map_csv,
        "candidate_features": cdir / "candidate_features.csv",
        "missingness": cdir / "missingness.csv",
        "coef_csv": None,
        "coef_pos_csv": None,
        "coef_neg_csv": None,
        "feat_csv": None,
    }

    if coef_model is not None:
        model_map = load_model_name_map(model_map_csv)
        coef_tag = resolve_coef_tag(coef_model, model_map)
        files.update({
            "coef_csv": cdir / f"coef__{coef_tag}.csv",
            "coef_pos_csv": cdir / f"coef_pos__{coef_tag}.csv",
            "coef_neg_csv": cdir / f"coef_neg__{coef_tag}.csv",
            "feat_csv": cdir / f"feat__{coef_tag}.csv",
        })

    return files


# ----------------------------- style ---------------------------------------- #

def set_pub_style(args: argparse.Namespace) -> None:
    sns.set_style("whitegrid", {
        "axes.edgecolor": args.axes_edgecolor,
        "grid.color": args.grid_color,
        "grid.linestyle": args.grid_linestyle,
        "grid.linewidth": args.grid_linewidth,
        "axes.facecolor": args.axes_facecolor,
        "figure.facecolor": args.figure_facecolor,
    })

    sns.set_context(args.context)

    mpl.rcParams.update({
        "font.family": args.font_family,
        "font.sans-serif": [x.strip() for x in args.font_sans.split(",") if x.strip()],
        "pdf.fonttype": args.pdf_fonttype,
        "ps.fonttype": args.ps_fonttype,
        "axes.linewidth": args.axes_linewidth,
        "axes.labelsize": args.axes_labelsize,
        "axes.titlesize": args.axes_titlesize,
        "xtick.labelsize": args.xtick_labelsize,
        "ytick.labelsize": args.ytick_labelsize,
        "legend.fontsize": args.legend_fontsize,
        "figure.titlesize": args.figure_titlesize,
        "savefig.dpi": args.dpi,
    })
    if __import__("os").environ.get("MATPLOTLIBRC"):
        # central figure style (figstyle.py) sets font family + colors so we look
        # consistent with all figures; standalone runs keep the settings above.
        mpl.rcParams.update(mpl.rc_params_from_file(
            __import__("os").environ["MATPLOTLIBRC"], use_default_template=False))
        # ...but re-assert the explicit CLI font SIZES: these panels are downscaled
        # into the montage, so they need larger-than-central sizes to land at the
        # common on-page target (else the central 8 pt wins and labels collapse).
        mpl.rcParams.update({
            "axes.labelsize": args.axes_labelsize,
            "axes.titlesize": args.axes_titlesize,
            "xtick.labelsize": args.xtick_labelsize,
            "ytick.labelsize": args.ytick_labelsize,
            "legend.fontsize": args.legend_fontsize,
            "figure.titlesize": args.figure_titlesize,
        })

# ----------------------------- labels --------------------------------------- #

def infer_task_metric(df_metrics: pd.DataFrame, prefer_secondary: bool = False) -> Tuple[str, str]:
    task = (
        str(df_metrics["task"].dropna().iloc[0])
        if "task" in df_metrics.columns and not df_metrics["task"].dropna().empty
        else "regress"
    )
    if task == "classify":
        return ("oof_mcc", "Matthews corr. coeff.") if prefer_secondary else ("oof_auroc", "AUROC")
    return ("oof_spearman", "Spearman ρ") if prefer_secondary else ("oof_r2", "R²")


def _short_group_name(token: str) -> str:
    mapping = {
        "evolution": "Evolution",
        "complexity": "Complexity",
        "rna_intrinsic": "RNA intrinsic",
        "rnaintr": "RNA Intrinsic",
        "interface": "Interface",
        "experimental": "Experimental",
        "af3": "AF3",
    }
    return mapping.get(token, token.replace("_", " ").title())

def get_model_extras(m: str) -> List[str]:
    m = str(m)

    # fix the split issue
    m = m.replace("rna_intrinsic", "rna-intrinsic")

    parts = m.split("_")
    if len(parts) >= 2 and parts[1] == "af3":
        return parts[2:]
    return []

def clean_model_label(m: str) -> str:
    m = str(m)

    singles = {
        "G0_af3": "AF3",
        "G1_evolution": "Evolution",
        "G2_complexity": "Complexity",
        "G3_rna_intrinsic": "RNA Intrinsic",
        "G4_interface": "Interface",
        "G5_experimental": "Experimental",
        "GX_all": "All features",
    }
    if m in singles:
        return singles[m]

    if m.startswith("A"):
        extras = get_model_extras(m)

        # special cases (clean labels)
        if set(extras) == {"evolution", "complexity", "rna-intrinsic", "interface"}:
            return "AF3 + Core Determinants"

        if set(extras) == {"evolution", "complexity", "rna-intrinsic", "interface", "experimental"}:
            return "AF3 + All Features"

        label_map = {
            "evolution": "Evolution",
            "complexity": "Complexity",
            "rna-intrinsic": "RNA Intrinsic",
            "interface": "Interface",
            "experimental": "Experimental",
        }

        extras = [label_map.get(e, e) for e in extras]

        if len(extras) == 0:
            return "AF3"

        return "AF3 + " + " + ".join(extras)

    return m

def model_order_info(model_name: str) -> Tuple[int, int, str]:
    m = str(model_name)

    single_rank = {
        "G0_af3": 0,
        "G1_evolution": 1,
        "G2_complexity": 2,
        "G3_rna_intrinsic": 3,
        "G4_interface": 4,
        "G5_experimental": 5,
    }

    if m in single_rank:
        return 0, single_rank[m], "Single groups"

    if m == "GX_all":
        return 6, 0, "Full model"

    if m.startswith("A"):
        extras = get_model_extras(m)
        n_extra = len(extras)

        if n_extra == 0:
            return 1, 0, "AF3 + 0 groups"

        return min(n_extra, 5), 0, f"AF3 + {n_extra} group" + ("" if n_extra == 1 else "s")

    return 99, 0, "Other"

def resolve_requested_model_list(
    requested: Optional[List[str]],
    excluded: Optional[List[str]],
    model_map: Dict[str, str],
    available_models: List[str],
) -> List[str]:
    available_set = set(map(str, available_models))

    if requested:
        resolved = []
        for item in requested:
            full = resolve_model_identifier(item, model_map)
            if full in available_set:
                resolved.append(full)
        selected = list(dict.fromkeys(resolved))
    else:
        selected = list(available_models)

    if excluded:
        exclude_resolved = set()
        for item in excluded:
            full = resolve_model_identifier(item, model_map)
            exclude_resolved.add(full)
        selected = [m for m in selected if m not in exclude_resolved]

    return selected


def pretty_feature_name(name: str) -> str:
    name = str(name)

    # ------------------ OHE categorical handling ------------------ #
    if name.startswith("peg_concentration_category_"):
        val = name.replace("peg_concentration_category_", "")
        val_map = {
            "no_peg": "PEG: None",
            "low": "PEG: Low",
            "medium": "PEG: Med",
            "high": "PEG: High",
        }
        return val_map.get(val, f"PEG: {val.replace('_', ' ').title()}")

    if name.startswith("buffer_category_"):
        val = name.replace("buffer_category_", "")
        val_map = {
            "unknown": "Buffer: Unknown",
            "phosphate": "Buffer: Phosphate",
            "tris": "Buffer: Tris",
        }
        return val_map.get(val, f"Buffer: {val.replace('_', ' ').title()}")

    if name.startswith("rna_family_"):
        val = name.replace("rna_family_", "")
        return f"RNA Family: {val.upper()}"

    # ------------------ main mapping ------------------ #
    repl = {
        # AF3 confidence
        "af3_global_pae_avg": "AF3 PAE",
        "af3_protein_pLDDT_avg": "AF3 Protein pLDDT",
        "af3_rna_pLDDT_avg": "AF3 RNA pLDDT",
        "af3_chain_pair_pae_min": "Min Chain PAE",
        "af3_ranking_score": "AF3 Score",
        "af3_rna_pTM_list_value": "AF3 RNA pTM",

        # Composition / size
        "max_rna_fraction_of_total_rna": "RNA Fraction",
        "NumberRNAs": "RNA Chains",
        "NumberProteins": "Protein Chains",
        "protein_total_length": "Protein Length",
        "rna_total_length": "RNA Length",
        "total_complex_length": "Complex Length",
        "protein_chain_length_cv": "Protein Length CV",
        "protein_to_rna_length_ratio": "Protein/RNA Ratio",

        # RNA intrinsic
        "agg_rna_gc_content": "RNA GC Content",
        "agg_rna_sequence_complexity": "RNA Complexity",
        "agg_rna_shannon_entropy_k1": "RNA Entropy (k=1)",
        "agg_rna_shannon_entropy_k2": "RNA Entropy (k=2)",

        # Interface
        "pred_ProteinRNAInterfaceArea": "Interface Area",
        "pred_ProteinRNAInterfaceRatio": "Interface Ratio",
        "n_unique_rna_interface_residues_global": "RNA Interface Res.",
        "n_unique_protein_interface_residues_global": "Protein Interface Res.",
        "rna_interface_coverage_global": "RNA Interface",
        "interface_contact_distance_mean": "Contact Dist. Mean",
        "interface_contact_distance_median": "Contact Dist. Median",
        "frac_contacts_lt_3_5A": "Contacts < 3.5Å",
        "interface_rna_motif_length_sum": "Motif Length",
        "interface_rna_motif_coverage_of_total_rna": "Motif Coverage",
        "interface_frac_contacts_to_C": "Contacts To C",

        # Evolution / templates
        "n_rbp_templates": "RBP Templates",
        "n_protein_templates": "Protein Templates",
        "log10_rna_msa_size": "RNA MSA Size",
        "n_unique_rna_families": "RNA Families",

        # Experimental
        "is_post_cutoff": "Post-Cutoff",
        "XRayResolution": "X-Ray Resolution",
        "exp_pH": "pH",
        "exp_Temp": "Temperature",
        "n_ions_listed": "Ions",
        "n_ligands_listed": "Ligands",
        "ionic_strength": "Ionic Strength",
        "peg_percent": "PEG %",
        "buffer_m": "Buffer M",
        "has_peg": "PEG Present",
        "log10_kd": "Log Kd",

        # Targets
        "Complex_TM": "Complex TM",
        "Dockq": "DockQ",
        "DockQ_Nr": "DockQ$_{Nr}$",
        "AF3_ERROR__DockQ_Nr": "AF3 error (DockQ$_{Nr}$)",
        "DockQ_minus_complex_TM": "DockQ - TM",
    }

    if name in repl:
        return repl[name]

    # fallback (clean but controlled)
    return name.replace("_", " ").title()

# ----------------------------- scatter plots -------------------------------- #

DEFAULT_SCATTER_VARS_COMPLEX = [
    (
        "max_rna_fraction_of_total_rna",
        None,
        "Largest RNA fraction",
        None,
        "Accuracy vs RNA dominance",
    ),
    (
        "NumberRNAs",
        None,
        "RNA chain number",
        None,
        "Accuracy vs RNA chain number",
    ),
    (
        "protein_total_length",
        None,
        "Protein total length",
        None,
        "Accuracy vs protein scaffold size",
    ),
    (
        "af3_global_pae_avg",
        None,
        "AF3 global PAE",
        None,
        "Accuracy vs AF3 global PAE",
    ),
]


def add_discrete_jitter(x: pd.Series, frac: float = 0.06, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xvals = x.to_numpy(dtype=float)
    uniq = pd.Series(xvals).dropna().nunique()
    if uniq <= 20:
        return xvals + rng.normal(0, frac, size=len(xvals))
    return xvals


def parse_scatter_vars(
    var_specs: Optional[List[str]],
    default_y: str,
    default_y_label: str,
) -> List[Tuple[str, str, str, str, str]]:
    """
    Each spec:
      x|y|xlabel|ylabel|title

    If omitted, defaults are used with y/default_y and ylabel/default_y_label.
    """
    if not var_specs:
        out = []
        for x, y, xlabel, ylabel, title in DEFAULT_SCATTER_VARS_COMPLEX:
            out.append((
                x,
                y if y is not None else default_y,
                xlabel,
                ylabel if ylabel is not None else default_y_label,
                title,
            ))
        return out

    parsed: List[Tuple[str, str, str, str, str]] = []
    for spec in var_specs:
        parts = [p.strip() for p in spec.split("|")]
        if len(parts) != 5:
            raise ValueError(
                "Each --scatter-vars entry must have exactly 5 fields:\n"
                "x|y|xlabel|ylabel|title"
            )
        parsed.append((parts[0], parts[1], parts[2], parts[3], parts[4]))
    return parsed


def draw_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    args: argparse.Namespace,
    letter: Optional[str] = None,
) -> None:
    sub = df[[x, y]].dropna().copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No data for\n{x} vs {y}", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel, fontsize=args.axes_labelsize)
        ax.set_ylabel(ylabel, fontsize=args.axes_labelsize)
        ax.tick_params(axis="x", labelsize=args.xtick_labelsize)
        ax.tick_params(axis="y", labelsize=args.ytick_labelsize)
        if letter:
            panel_letter(ax, letter, x=args.panel_letter_x, y=args.panel_letter_y, fontsize=args.panel_letter_fontsize)
        return

    xplot = (
        add_discrete_jitter(sub[x], frac=args.scatter_jitter_frac, seed=args.scatter_jitter_seed)
        if args.scatter_jitter_if_discrete else sub[x].to_numpy(dtype=float)
    )

    ax.scatter(
        xplot, sub[y],
        s=args.scatter_point_size,
        alpha=args.scatter_point_alpha,
        color=args.scatter_color,
        edgecolors="none",
        rasterized=True,
    )

    ci = None if args.scatter_ci < 0 else args.scatter_ci
    sns.regplot(
        x=sub[x],
        y=sub[y],
        scatter=False,
        ci=ci,
        line_kws={"linewidth": args.scatter_reg_linewidth},
        color=args.scatter_reg_color,
        ax=ax,
    )

    text_parts = []
    x_unique = sub[x].nunique()
    y_unique = sub[y].nunique()

    if not args.hide_pearson and x_unique > 1 and y_unique > 1:
        r = pearsonr(sub[x], sub[y])[0]
        text_parts.append(f"Pearson r = {r:.2f}")

    if not args.hide_spearman and x_unique > 1 and y_unique > 1:
        rho = spearmanr(sub[x], sub[y])[0]
        text_parts.append(f"Spearman ρ = {rho:.2f}")

    if not args.hide_n:
        text_parts.append(f"n = {len(sub)}")

    if text_parts:
        loc_map = {
            "upper_left": (0.03, 0.97, "left", "top"),
            "upper_right": (0.97, 0.97, "right", "top"),
            "lower_left": (0.03, 0.03, "left", "bottom"),
            "lower_right": (0.97, 0.03, "right", "bottom"),
        }
        tx, ty, ha, va = loc_map[args.scatter_annot_loc]
        ax.text(
            tx, ty, "\n".join(text_parts),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=args.scatter_annot_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.95,
            ),
        )

    ax.set_xlabel(xlabel, fontsize=args.axes_labelsize)
    ax.set_ylabel(ylabel, fontsize=args.axes_labelsize)
    ax.tick_params(axis="x", labelsize=args.xtick_labelsize)
    ax.tick_params(axis="y", labelsize=args.ytick_labelsize)

    if letter:
        panel_letter(ax, letter, x=args.panel_letter_x, y=args.panel_letter_y, fontsize=args.panel_letter_fontsize)

    sns.despine(ax=ax, top=True, right=True)


def make_single_scatter_figure(
    raw_csv: Path,
    outfile: Path,
    args: argparse.Namespace,
    spec: Tuple[str, str, str, str, str],
    figure_title: Optional[str] = None,
    letter: Optional[str] = None,
) -> None:
    set_pub_style(args)
    df = pd.read_csv(raw_csv)

    x, y, xlabel, ylabel, title = spec

    fig, ax = plt.subplots(figsize=(args.scatter_fig_width, args.scatter_fig_height))
    draw_scatter(
        ax=ax,
        df=df,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        args=args,
        letter=letter,
    )

    if figure_title:
        fig.suptitle(figure_title, y=args.suptitle_y)

    plt.tight_layout(rect=[0, 0, 1, args.tight_layout_top] if figure_title else None)
    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)


# ----------------------------- coefficient figure --------------------------- #

def load_coef_file(path: Path) -> pd.Series:
    df = pd.read_csv(path)

    if "feature" in df.columns and "coef" in df.columns:
        return pd.Series(df["coef"].to_numpy(dtype=float), index=df["feature"].astype(str))

    if "coef" in df.columns:
        first_col = df.columns[0]
        if str(first_col).startswith("Unnamed") or first_col == "":
            return pd.Series(df["coef"].to_numpy(dtype=float), index=df.iloc[:, 0].astype(str))

    try:
        s = pd.read_csv(path, index_col=0).squeeze("columns")
        if isinstance(s, pd.Series):
            s.index = s.index.astype(str)
            return s.astype(float)
    except Exception:
        pass

    raise ValueError(f"Unexpected coefficient file format: {path}")


def make_coef_figure(
    coef_csv: Path,
    outfile: Path,
    args: argparse.Namespace,
    title: str,
    subtitle: Optional[str],
) -> None:
    set_pub_style(args)

    s = load_coef_file(coef_csv).dropna()
    if s.empty:
        raise ValueError(f"No coefficients found in {coef_csv}")

    s = s.sort_values()
    neg = s.head(args.coef_top_n)
    pos = s.tail(args.coef_top_n)
    shown = pd.concat([neg, pos]).sort_values()

    labels = [pretty_feature_name(i) for i in shown.index]
    colors = [args.coef_neg_color if v < 0 else args.coef_pos_color for v in shown.values]

    if args.coef_horizontal:
        fig_w = max(args.coef_fig_width, args.coef_horizontal_fig_width)
        fig_h = max(args.coef_fig_height_min, args.coef_horizontal_fig_height)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(len(shown))
        ax.bar(
            x,
            shown.values,
            color=colors,
            edgecolor="none",
            width=args.coef_horizontal_bar_width,
        )
        ax.axhline(0, color=args.coef_zero_line_color, linewidth=args.coef_zero_linewidth)

        ax.set_xticks(x)
        _dx = getattr(args, "coef_horizontal_label_dx", 0.0)
        if _dx:
            # draw the rotated x labels as manual text shifted right by _dx data
            # units; done as text (not tick labels) so tight_layout cannot reset it
            ax.set_xticklabels([])
            for _i, _lab in zip(x, labels):
                ax.text(_i + _dx, -0.02, _lab, transform=ax.get_xaxis_transform(),
                        rotation=args.coef_horizontal_label_rotation,
                        ha=args.coef_horizontal_label_ha, va="top",
                        rotation_mode="anchor", fontsize=args.xtick_labelsize)
        else:
            ax.set_xticklabels(
                labels,
                rotation=args.coef_horizontal_label_rotation,
                ha=args.coef_horizontal_label_ha,
                fontsize=args.xtick_labelsize,
            )
        ax.tick_params(axis="y", labelsize=args.ytick_labelsize)
        ax.set_ylabel(args.coef_xlabel, fontsize=args.axes_labelsize)
        ax.set_xlabel("")

    else:
        fig_h = max(args.coef_fig_height_min, args.coef_row_height * len(shown) + args.coef_fig_height_pad)
        fig, ax = plt.subplots(figsize=(args.coef_fig_width, fig_h))

        ax.barh(labels, shown.values, color=colors, edgecolor="none")
        ax.axvline(0, color=args.coef_zero_line_color, linewidth=args.coef_zero_linewidth)
        ax.set_xlabel(args.coef_xlabel, fontsize=args.axes_labelsize)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=args.xtick_labelsize)
        ax.tick_params(axis="y", labelsize=args.ytick_labelsize)

    if subtitle:
        ax.text(
            0.0, 1.01, subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=args.coef_subtitle_fontsize,
            color=args.coef_subtitle_color,
        )

    if not args.hide_coef_direction_note:
        ax.text(
            0.98, 0.02,
            args.coef_direction_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=args.coef_note_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9,
            ),
        )

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)

# ----------------------------- performance figures -------------------------- #

def compute_fold_stats(metrics_per_fold_csv: Path, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_per_fold_csv)

    fold_metric_map = {
        "oof_r2": "r2",
        "oof_spearman": "spearman",
        "oof_rmse": "rmse",
        "oof_mae": "mae",
        "oof_auroc": "auroc",
        "oof_ap": "ap",
        "oof_mcc": "mcc",
        "oof_brier": "brier",
    }

    fold_metric_col = fold_metric_map.get(metric_col, metric_col)

    if "model" not in df.columns or fold_metric_col not in df.columns:
        raise ValueError(
            f"metrics_per_fold.csv missing required columns: model, {fold_metric_col} "
            f"(requested summary metric was {metric_col})"
        )

    agg = (
        df.groupby("model")[fold_metric_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{metric_col}_mean", "std": f"{metric_col}_std"})
    )

    return agg


def prepare_performance_dataframe(
    metrics_csv: Path,
    metrics_per_fold_csv: Optional[Path],
    model_name_map_csv: Path,
    metric_col: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise ValueError(f"No successful models in {metrics_csv}")

    if metric_col not in df.columns or "model" not in df.columns:
        raise ValueError(f"Required columns missing from {metrics_csv}")

    model_map = load_model_name_map(model_name_map_csv)

    available_models = list(df["model"].astype(str).unique())
    selected_models = resolve_requested_model_list(
        requested=args.performance_models,
        excluded=args.performance_exclude_models,
        model_map=model_map,
        available_models=available_models,
    )

    d = df[df["model"].astype(str).isin(selected_models)][["model", metric_col]].copy()
    if d.empty:
        raise ValueError("No models left after applying performance model filters.")

    if args.performance_error_bars and metrics_per_fold_csv and metrics_per_fold_csv.exists():
        df_fold = compute_fold_stats(metrics_per_fold_csv, metric_col)
        d = d.merge(df_fold, on="model", how="left")
        d["value"] = d[f"{metric_col}_mean"]
        d["error"] = d[f"{metric_col}_std"]
    else:
        d["value"] = d[metric_col]
        d["error"] = np.nan

    d["model_label"] = d["model"].map(clean_model_label)

    order_info = d["model"].map(model_order_info)
    d["section_rank"] = [x[0] for x in order_info]
    d["within_rank"] = [x[1] for x in order_info]
    d["section_name"] = [x[2] for x in order_info]

    if args.performance_sort_within == "value_asc":
        d = d.sort_values(
            by=["section_rank", "value", "model_label"],
            ascending=[True, True, True],
            kind="stable",
        )
    elif args.performance_sort_within == "value_desc":
        d = d.sort_values(
            by=["section_rank", "value", "model_label"],
            ascending=[True, False, True],
            kind="stable",
        )
    else:
        d = d.sort_values(
            by=["section_rank", "model_label"],
            ascending=[True, True],
            kind="stable",
        )
    
    d = d.reset_index(drop=True)
    return d


def make_single_performance_figure(
    metrics_csv: Path,
    metrics_per_fold_csv: Optional[Path],
    model_name_map_csv: Path,
    outfile: Path,
    args: argparse.Namespace,
    metric_col: str,
    xlabel: str,
    bar_color: str,
    title: str,
    letter: Optional[str] = None,
) -> None:
    set_pub_style(args)

    d = prepare_performance_dataframe(
        metrics_csv=metrics_csv,
        metrics_per_fold_csv=metrics_per_fold_csv,
        model_name_map_csv=model_name_map_csv,
        metric_col=metric_col,
        args=args,
    )

    y_positions = []
    separator_positions = []
    current_y = 0.0
    last_section = None

    for _, row in d.iterrows():
        section = row["section_name"]
        if last_section is not None and section != last_section:
            current_y += args.performance_section_gap
            separator_positions.append(current_y - args.performance_section_gap / 2.0)
        y_positions.append(current_y)
        current_y += 1.0
        last_section = section

    d["y"] = y_positions

    fig_h = max(
        args.performance_fig_height_min,
        args.performance_row_height * len(d) + args.performance_section_gap * max(d["section_rank"].nunique() - 1, 0),
    )
    fig, ax = plt.subplots(figsize=(args.performance_fig_width, fig_h))

    ax.barh(
        y=d["y"],
        width=d["value"],
        color=bar_color,
        xerr=d["error"].values if args.performance_error_bars else None,
        capsize=args.performance_error_cap_size if args.performance_error_bars else 0,
        error_kw={"linewidth": args.performance_error_linewidth} if args.performance_error_bars else None,
    )

    ax.set_yticks(d["y"])
    ax.set_yticklabels(d["model_label"], fontsize=args.ytick_labelsize)
    ax.tick_params(axis="x", labelsize=args.xtick_labelsize)
    ax.tick_params(axis="y", labelsize=args.ytick_labelsize)
    ax.invert_yaxis()

    for ypos in separator_positions:
        ax.axhline(y=ypos, color=args.performance_separator_color, linewidth=args.performance_separator_linewidth)

    ax.set_xlabel(xlabel, fontsize=args.axes_labelsize)
    ax.set_ylabel("")

    if args.performance_show_section_labels:
        sections = (
            d.groupby("section_name", sort=False)["y"]
            .agg(["min", "max"])
            .reset_index()
        )
        x_left, x_right = ax.get_xlim()
        x_text = x_left + (x_right - x_left) * args.performance_section_label_xfrac
        for _, sec in sections.iterrows():
            y_mid = 0.5 * (sec["min"] + sec["max"])
            ax.text(
                x_text,
                y_mid,
                str(sec["section_name"]),
                fontsize=args.performance_section_label_fontsize,
                color=args.performance_section_label_color,
                va="center",
                ha="left",
                alpha=0.95,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

    if letter:
        panel_letter(
            ax,
            letter,
            x=args.panel_letter_x,
            y=args.panel_letter_y,
            fontsize=args.panel_letter_fontsize,
        )

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    ensure_dir(outfile.parent)
    save_all_formats(fig, outfile, dpi=args.dpi)
    plt.close(fig)

# ----------------------------- title helpers -------------------------------- #

def infer_default_ycol(run_cfg: Dict) -> str:
    target_label = str(run_cfg.get("target_label", ""))
    ycol_input = str(run_cfg.get("ycol_input", ""))

    if target_label.startswith("AF3_ERROR__"):
        if ycol_input:
            return ycol_input
        return target_label.replace("AF3_ERROR__", "", 1)

    if ycol_input:
        return ycol_input

    if target_label:
        return target_label

    return "target"


def infer_default_y_label(run_cfg: Dict) -> str:
    y = infer_default_ycol(run_cfg)
    return pretty_feature_name(y)


def infer_scatter_figure_title(run_cfg: Dict, user_title: Optional[str]) -> Optional[str]:
    return user_title


def infer_coef_title(run_cfg: Dict, user_title: Optional[str]) -> str:
    if user_title:
        return user_title
    if str(run_cfg.get("target_mode", "")) == "af3_error":
        return "Multivariate determinants of AF3 residual error"
    return "Multivariate determinants of prediction accuracy"


def auto_subtitle_from_config(run_cfg: Dict, coef_model: str) -> str:
    parts = []

    target_label = run_cfg.get("target_label")
    group_by = run_cfg.get("group_by")
    dataset = run_cfg.get("dataset")
    subset = run_cfg.get("subset")

    if target_label:
        parts.append(str(target_label))
    if dataset:
        parts.append(str(dataset))
    if subset:
        parts.append(str(subset))
    if group_by:
        parts.append(f"grouped by {group_by}")
    if coef_model:
        parts.append(str(coef_model))

    return " | ".join(parts)


# ----------------------------- CLI ------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create standalone publication-ready figures from the new AF3 regression output structure."
    )

    # input selection
    p.add_argument("--run-dir", default=None, help="Direct path to one run directory containing run_config.json")
    p.add_argument("--runs-root", default=None, help="Parent directory containing multiple run directories")
    p.add_argument("--target-label", default=None, help="Used with --runs-root to pick a run, e.g. Complex_TM or AF3_ERROR__Complex_TM")
    p.add_argument("--analysis-name-contains", default=None, help="Substring filter applied to run_config['analysis_name']")
    p.add_argument("--run-name-contains", default=None, help="Substring filter applied to the run directory name")
    p.add_argument("--raw-csv", default=None, help="Raw CSV used for scatter figure(s)")
    p.add_argument("--mode", default="all", choices=["all", "scatter", "coefs", "performance"])
    p.add_argument("--coef-model", default="A31_af3_evolution_complexity_rna_intrinsic_interface_experimental", help="Full model name")
    p.add_argument("--outdir", default=None, help="Output directory. Defaults to <run-dir>/pubfigs")
    p.add_argument("--base-filename-prefix", default="", help="Optional prefix added to all output figure filenames")

    # global style
    p.add_argument("--context", default="paper", choices=["paper", "notebook", "talk", "poster"])
    p.add_argument("--font-family", default="sans-serif")
    p.add_argument("--font-sans", default="Arial,DejaVu Sans,Liberation Sans,Helvetica")
    p.add_argument("--pdf-fonttype", type=int, default=42)
    p.add_argument("--ps-fonttype", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--axes-linewidth", type=float, default=0.8)
    p.add_argument("--axes-labelsize", type=float, default=11)
    p.add_argument("--axes-titlesize", type=float, default=12)
    p.add_argument("--xtick-labelsize", type=float, default=9)
    p.add_argument("--ytick-labelsize", type=float, default=9)
    p.add_argument("--legend-fontsize", type=float, default=9)
    p.add_argument("--figure-titlesize", type=float, default=15)

    p.add_argument("--axes-edgecolor", default="0.15")
    p.add_argument("--grid-color", default="0.88")
    p.add_argument("--grid-linestyle", default="-")
    p.add_argument("--grid-linewidth", type=float, default=0.7)
    p.add_argument("--axes-facecolor", default="white")
    p.add_argument("--figure-facecolor", default="white")

    p.add_argument("--panel-letter-x", type=float, default=-0.13)
    p.add_argument("--panel-letter-y", type=float, default=1.08)
    p.add_argument("--panel-letter-fontsize", type=float, default=13)

    p.add_argument("--suptitle-y", type=float, default=0.995)
    p.add_argument("--tight-layout-top", type=float, default=0.97)

    # scatter options
    p.add_argument(
        "--scatter-vars",
        nargs="*",
        default=None,
        help="One or more scatter specs. Format: x|y|xlabel|ylabel|title"
    )
    p.add_argument("--scatter-fig-width", type=float, default=5.2)
    p.add_argument("--scatter-fig-height", type=float, default=4.4)
    p.add_argument("--scatter-color", default="#2C7FB8")
    p.add_argument("--scatter-reg-color", default="black")
    p.add_argument("--scatter-point-size", type=float, default=14)
    p.add_argument("--scatter-point-alpha", type=float, default=0.55)
    p.add_argument("--scatter-reg-linewidth", type=float, default=1.9)
    p.add_argument("--scatter-ci", type=int, default=95, help="Use -1 to disable CI band")
    p.add_argument("--scatter-jitter-if-discrete", action="store_true", default=True)
    p.add_argument("--no-scatter-jitter-if-discrete", dest="scatter_jitter_if_discrete", action="store_false")
    p.add_argument("--scatter-jitter-frac", type=float, default=0.06)
    p.add_argument("--scatter-jitter-seed", type=int, default=0)
    p.add_argument("--scatter-annot-fontsize", type=float, default=8.5)
    p.add_argument("--scatter-title-pad", type=float, default=7.0)
    p.add_argument("--scatter-annot-loc", default="upper_left",
                   choices=["upper_left", "upper_right", "lower_left", "lower_right"])
    p.add_argument("--hide-n", action="store_true")
    p.add_argument("--hide-pearson", action="store_true")
    p.add_argument("--hide-spearman", action="store_true")
    p.add_argument("--scatter-figure-title", default=None, help="Optional suptitle for each standalone scatter plot")
    p.add_argument("--scatter-add-panel-letter", action="store_true")
    p.add_argument("--scatter-letter-start", default="A", help="Starting letter if --scatter-add-panel-letter is used")

    # coefficient options
    p.add_argument("--coef-top-n", type=int, default=8)
    p.add_argument("--coef-fig-width", type=float, default=7.6)
    p.add_argument("--coef-fig-height-min", type=float, default=4.8)
    p.add_argument("--coef-row-height", type=float, default=0.34)
    p.add_argument("--coef-fig-height-pad", type=float, default=1.8)
    # Project convention: red = positive influence, blue = negative (matches the
    # heatmaps). Positive coefficients -> red (#B64E4E), negative -> blue (#2F7FBF).
    p.add_argument("--coef-pos-color", default="#B64E4E")
    p.add_argument("--coef-neg-color", default="#2F7FBF")
    p.add_argument("--coef-zero-line-color", default="0.25")
    p.add_argument("--coef-zero-linewidth", type=float, default=0.9)
    p.add_argument("--coef-xlabel", default="Standardized Ridge Coefficient")
    p.add_argument("--coef-title", default=None)
    p.add_argument("--coef-title-pad", type=float, default=8.0)
    p.add_argument("--coef-subtitle", default=None)
    p.add_argument("--coef-subtitle-fontsize", type=float, default=9.0)
    p.add_argument("--coef-subtitle-color", default="0.35")
    p.add_argument("--coef-note-fontsize", type=float, default=8.5)
    p.add_argument("--hide-coef-direction-note", action="store_true")
    p.add_argument(
        "--coef-direction-note",
        default="Blue: positive association\nRed: negative association",
    )

    p.add_argument("--coef-horizontal", action="store_true", help="Plot coefficient bars vertically with feature labels on the x-axis")
    p.add_argument("--coef-horizontal-fig-width", type=float, default=12.0)
    p.add_argument("--coef-horizontal-fig-height", type=float, default=4.8)
    p.add_argument("--coef-horizontal-bar-width", type=float, default=0.82)
    p.add_argument("--coef-horizontal-label-rotation", type=float, default=45.0)
    p.add_argument("--coef-horizontal-label-ha", default="right", choices=["left", "center", "right"])
    p.add_argument("--coef-horizontal-label-dx", type=float, default=0.0,
                   help="shift rotated x labels right by this many data units")

    # performance options
    p.add_argument("--performance-fig-width", type=float, default=6.8)
    p.add_argument("--performance-fig-height-min", type=float, default=5.4)
    p.add_argument("--performance-row-height", type=float, default=0.25)
    p.add_argument("--performance-primary-color", default="#2F7FBF")
    p.add_argument("--performance-secondary-color", default="#8FB6D9")
    p.add_argument("--performance-primary-title", default="Primary performance metric")
    p.add_argument("--performance-secondary-title", default="Secondary performance metric")
    p.add_argument("--performance-title-pad", type=float, default=8.0)
    p.add_argument("--performance-primary-letter", default=None)
    p.add_argument("--performance-secondary-letter", default=None)

    # performance model filtering
    p.add_argument(
        "--performance-models",
        nargs="*",
        default=None,
        help="Optional list of models to include in performance plots. Accepts full model names or short tags."
    )
    p.add_argument(
        "--performance-exclude-models",
        nargs="*",
        default=None,
        help="Optional list of models to exclude from performance plots. Accepts full model names or short tags."
    )

    # performance section formatting
    p.add_argument("--performance-section-gap", type=float, default=0.8)
    p.add_argument("--performance-separator-color", default="0.75")
    p.add_argument("--performance-separator-linewidth", type=float, default=0.8)
    p.add_argument("--performance-show-section-labels", action="store_true")
    p.add_argument("--performance-section-label-fontsize", type=float, default=8.5)
    p.add_argument("--performance-section-label-color", default="0.35")
    p.add_argument("--performance-section-label-xfrac", type=float, default=0.02)

    # performance error bars
    p.add_argument("--performance-error-bars", action="store_true")
    p.add_argument("--performance-error-cap-size", type=float, default=3.0)
    p.add_argument("--performance-error-linewidth", type=float, default=1.2)

    p.add_argument(
        "--performance-sort-within",
        choices=["value_asc", "value_desc", "name"],
        default="value_asc",
        help="How to sort models within each section"
    )

    return p.parse_args()


# ----------------------------- main ----------------------------------------- #

def main() -> None:
    args = parse_args()

    if args.run_dir is None and args.runs_root is None:
        raise ValueError("Provide either --run-dir or --runs-root")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not (run_dir / "run_config.json").exists():
            raise FileNotFoundError(f"Not a valid run directory: {run_dir}")
    else:
        run_dir = pick_run_dir(
            runs_root=Path(args.runs_root),
            target_label=args.target_label,
            analysis_name_contains=args.analysis_name_contains,
            run_name_contains=args.run_name_contains,
        )

    need_coef_files = args.mode in {"all", "coefs"}
    files = locate_analysis_files(run_dir, args.coef_model if need_coef_files else None)
    run_cfg = load_json(files["run_config"])

    if args.mode in {"all", "performance"} and not files["metrics_summary"].exists():
        raise FileNotFoundError(f"Missing metrics_summary.csv: {files['metrics_summary']}")
    if args.mode in {"all", "coefs"} and (files["coef_csv"] is None or not files["coef_csv"].exists()):
        raise FileNotFoundError(f"Missing coefficient CSV: {files['coef_csv']}")

    outdir = Path(args.outdir) if args.outdir else (run_dir / "pubfigs")
    ensure_dir(outdir)

    prefix = args.base_filename_prefix
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    y_default = infer_default_ycol(run_cfg)
    y_default_label = infer_default_y_label(run_cfg)

    print(f"[INFO] Using run directory: {run_dir}")
    print(f"[INFO] Target label: {run_cfg.get('target_label')}")
    if need_coef_files:
        print(f"[INFO] Coefficient model request: {args.coef_model}")
        print(f"[INFO] Resolved coefficient file: {files['coef_csv']}")

    # 1) scatter plots
    if args.mode in {"all", "scatter"}:
        if args.raw_csv is None:
            print("[WARN] --raw-csv not provided; skipping scatter plot generation.")
        else:
            scatter_specs = parse_scatter_vars(
                args.scatter_vars,
                default_y=y_default,
                default_y_label=y_default_label,
            )

            start_letter = ord(args.scatter_letter_start.upper()) if args.scatter_add_panel_letter else None

            for i, spec in enumerate(scatter_specs):
                x, y, xlabel, ylabel, title = spec
                tag = slugify(x)
                scatter_out = outdir / f"{prefix}Figure_scatter__{i+1:02d}__{tag}.png"
                letter = chr(start_letter + i) if start_letter is not None else None

                make_single_scatter_figure(
                    raw_csv=Path(args.raw_csv),
                    outfile=scatter_out,
                    args=args,
                    spec=spec,
                    figure_title=infer_scatter_figure_title(run_cfg, args.scatter_figure_title),
                    letter=letter,
                )
                print(f"Saved: {scatter_out}")

    # 2) coefficient plot
    if args.mode in {"all", "coefs"}:
        coef_out = outdir / f"{prefix}Figure_coefficients__{slugify(args.coef_model)}.png"
        coef_subtitle = args.coef_subtitle
        if coef_subtitle is None:
            coef_subtitle = auto_subtitle_from_config(run_cfg, args.coef_model)

        make_coef_figure(
            coef_csv=files["coef_csv"],
            outfile=coef_out,
            args=args,
            title=infer_coef_title(run_cfg, args.coef_title),
            subtitle=False,  # coef_subtitle,
        )
        print(f"Saved: {coef_out}")

    # 3) performance plots
    if args.mode in {"all", "performance"}:
        df_metrics = pd.read_csv(files["metrics_summary"])
        metric_primary, lab1 = infer_task_metric(df_metrics, prefer_secondary=False)
        metric_secondary, lab2 = infer_task_metric(df_metrics, prefer_secondary=True)

        perf1_out = outdir / f"{prefix}Figure_model_family_performance__primary.png"
        make_single_performance_figure(
            metrics_csv=files["metrics_summary"],
            metrics_per_fold_csv=files.get("metrics_per_fold"),
            model_name_map_csv=files["model_name_map"],
            outfile=perf1_out,
            args=args,
            metric_col=metric_primary,
            xlabel=lab1,
            bar_color=args.performance_primary_color,
            title=args.performance_primary_title,
            letter=args.performance_primary_letter,
        )
        print(f"Saved: {perf1_out}")

        perf2_out = outdir / f"{prefix}Figure_model_family_performance__secondary.png"
        make_single_performance_figure(
            metrics_csv=files["metrics_summary"],
            metrics_per_fold_csv=files.get("metrics_per_fold"),
            model_name_map_csv=files["model_name_map"],
            outfile=perf2_out,
            args=args,
            metric_col=metric_secondary,
            xlabel=lab2,
            bar_color=args.performance_secondary_color,
            title=args.performance_secondary_title,
            letter=args.performance_secondary_letter,
        )
        print(f"Saved: {perf2_out}")

    print(f"\nDone. Publication figures written to: {outdir}")


if __name__ == "__main__":
    main()