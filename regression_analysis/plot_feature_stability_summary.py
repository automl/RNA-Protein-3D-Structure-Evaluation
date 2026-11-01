#!/usr/bin/env python3
"""
Summarize grouped feature stability from coefficient files and generate a reviewer-friendly grouped barplot.

Default behavior:
- uses ONLY full-feature runs
- uses ONLY GX model coefficient files
- ignores subgroup-only coefficient files (g0-g5)
- counts top N positive and top N negative coefficients per run
- assigns each feature to a predefined group (G0-G5 / num-cat)
- handles one-hot expanded categorical features by prefix matching
- writes CSV summaries, LaTeX tables, included-file manifest, metadata, and a grouped barplot
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from figsave import save_all_formats  # multi-format (png600/pdf/svg/eps) saver

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------

AF3_NUM = [
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_ranking_score",
    "af3_rna_pTM_list_value",
    "af3_protein_ipTM_list_value",
    "af3_rna_ipTM_list_value",
    "af3_protein_pTM_list_value",
    "af3_protein_pLDDT_avg",
    "af3_protein_pTM_mean",
    "af3_protein_ipTM_mean",
    "af3_rna_ipTM_mean",
]
AF3_CAT = []

EVO_NUM = [
    "log10_rna_msa_size",
    "log10_protein_msa_size",
    "n_unique_rna_families",
    "n_unique_protein_families",
    "n_rbp_templates",
    "n_protein_templates",
]
EVO_CAT = ["rna_family"]

COMPLEXITY_NUM = [
    "NumberProteins",
    "NumberRNAs",
    "protein_total_length",
    "rna_total_length",
    "total_complex_length",
    "protein_mean_chain_length",
    "protein_max_chain_length",
    "protein_min_chain_length",
    "protein_std_chain_length",
    "protein_chain_length_cv",
    "rna_mean_chain_length",
    "rna_max_chain_length",
    "rna_min_chain_length",
    "rna_std_chain_length",
    "rna_chain_length_cv",
    "protein_residue_fraction",
    "rna_residue_fraction",
    "protein_to_rna_length_ratio",
    "max_rna_fraction_of_total_rna",
    "min_rna_fraction_of_total_rna",
    "pred_rna_chain_length",
    "pred_rna_is_modified",
]
COMPLEXITY_CAT = []

RNA_INTRINSIC_NUM = [
    "RNA_GC_Content",
    "RNA_SequenceComplexity",
    "RNA_Shannon_Entropy_K1",
    "RNA_Shannon_Entropy_K2",
    "RNA_GC_Skew",
    "RNA_AU_Skew",
    "RNA_AU_Content",
    "RNA_MFE_Value",
    "RNA_Ensemble_Diversity",
    "RNA_Centroid_Energy",
    "agg_rna_gc_content",
    "agg_rna_au_content",
    "agg_rna_gc_skew",
    "agg_rna_au_skew",
    "agg_rna_sequence_complexity",
    "agg_rna_shannon_entropy_k1",
    "agg_rna_shannon_entropy_k2",
]
RNA_INTRINSIC_CAT = []

RNA_STRUCTURE_NUM = [
    "num_stems",
    "num_hairpin_loops",
    "num_internal_loops",
    "num_multibranch_loops",
    "num_dangling_ends",
    "num_pseudoknots",
    "stems_per_100nt",
    "hairpin_loops_per_100nt",
    "internal_loops_per_100nt",
    "multibranch_loops_per_100nt",
    "dangling_ends_per_100nt",
    "pseudoknots_per_100nt",
]
RNA_STRUCTURE_CAT = []

BIOPHYS_NUM = [
    "log10_kd",
    "RNA_ElectrostaticPotential",
]
BIOPHYS_CAT = []

INTERFACE_AGREEMENT_NUM = [
    "pred_ProteinRNAInterfaceArea",
    "pred_ProteinRNAInterfaceRatio",
]
INTERFACE_AGREEMENT_CAT = []

GLOBAL_INTERFACE_NUM = [
    "n_pairs_with_contacts",
    "n_unique_rna_interface_residues_global",
    "rna_interface_coverage_global",
    "interface_contact_distance_mean",
    "interface_contact_distance_std",
    "n_protein_rna_interface_pairs",
    "frac_pairs_with_contacts",
    "total_hbond_proteinRNA",
    "total_vdwbond_proteinRNA",
    "total_contact_entries",
    "n_unique_protein_interface_residues_global",
    "protein_interface_coverage_global",
    "interface_contact_distance_median",
    "interface_contact_distance_min",
    "interface_contact_distance_max",
    "frac_contacts_lt_2_5A",
    "frac_contacts_lt_3_0A",
    "frac_contacts_lt_3_5A",
    "frac_contacts_lt_4_0A",
    "interface_hbond_per_contact",
    "interface_vdw_per_contact",
    "interface_hbond_vdw_ratio",
    "interface_rna_motif_length_sum",
    "interface_rna_motif_length_mean",
    "interface_rna_motif_length_max",
    "interface_rna_motif_coverage_of_total_rna",
    "interface_frac_contacts_to_A",
    "interface_frac_contacts_to_C",
    "interface_frac_contacts_to_G",
    "interface_frac_contacts_to_U",
    "interface_frac_contacts_to_modified",
    "interface_frac_positive_contacts",
    "interface_frac_aromatic_contacts",
    "interface_frac_polar_contacts",
    "interface_frac_hydrophobic_contacts",
    "interface_frac_acidic_contacts",
    "interface_frac_other_contacts",
]
GLOBAL_INTERFACE_CAT = []

CHAIN_INTERFACE_NUM = [
    "chain_n_interface_pairs",
    "chain_n_pairs_with_contacts",
    "chain_frac_pairs_with_contacts",
    "chain_total_hbond_proteinRNA",
    "chain_total_vdwbond_proteinRNA",
    "chain_total_contact_entries",
    "chain_n_unique_protein_interface_residues",
    "chain_n_unique_rna_interface_residues",
    "chain_protein_interface_coverage",
    "chain_rna_interface_coverage",
    "chain_interface_contact_distance_mean",
    "chain_interface_contact_distance_median",
    "chain_interface_contact_distance_min",
    "chain_interface_contact_distance_max",
    "chain_interface_contact_distance_std",
    "chain_frac_contacts_lt_2_5A",
    "chain_frac_contacts_lt_3_0A",
    "chain_frac_contacts_lt_3_5A",
    "chain_frac_contacts_lt_4_0A",
    "chain_interface_hbond_per_contact",
    "chain_interface_vdw_per_contact",
    "chain_interface_hbond_vdw_ratio",
    "chain_interface_rna_motif_length_sum",
    "chain_interface_rna_motif_length_mean",
    "chain_interface_rna_motif_length_max",
    "chain_interface_rna_motif_coverage_of_chain",
    "chain_interface_frac_contacts_to_A",
    "chain_interface_frac_contacts_to_C",
    "chain_interface_frac_contacts_to_G",
    "chain_interface_frac_contacts_to_U",
    "chain_interface_frac_contacts_to_modified",
    "chain_interface_frac_positive_contacts",
    "chain_interface_frac_aromatic_contacts",
    "chain_interface_frac_polar_contacts",
    "chain_interface_frac_hydrophobic_contacts",
    "chain_interface_frac_acidic_contacts",
    "chain_interface_frac_other_contacts",
]
CHAIN_INTERFACE_CAT = []

CHEM_NUM = [
    "n_ions_listed",
    "n_supported_ions",
    "n_unsupported_ions",
    "n_unique_ions",
    "n_ligands_listed",
    "n_supported_ligands",
    "n_unsupported_ligands",
    "n_unique_ligands",
    "ionic_strength",
    "peg_percent",
    "peg_mw",
    "buffer_m",
    "has_peg",
]
CHEM_CAT = [
    "ionic_strength_category",
    "peg_concentration_category",
    "peg_mw_category",
    "buffer_category",
    "buffer_name",
]

EXP_NUM = [
    "exp_pH",
    "exp_Temp",
    "XRayResolution",
    "is_xray",
]
EXP_CAT = []

GROUP_DEFS = {
    "g0_num": AF3_NUM,
    "g0_cat": AF3_CAT,
    "g1_num": EVO_NUM,
    "g1_cat": EVO_CAT,
    "g2_num": COMPLEXITY_NUM,
    "g2_cat": COMPLEXITY_CAT,
    "g3_num": RNA_INTRINSIC_NUM + RNA_STRUCTURE_NUM + BIOPHYS_NUM,
    "g3_cat": RNA_INTRINSIC_CAT + RNA_STRUCTURE_CAT + BIOPHYS_CAT,
    "g4_num": INTERFACE_AGREEMENT_NUM + GLOBAL_INTERFACE_NUM + CHAIN_INTERFACE_NUM,
    "g4_cat": INTERFACE_AGREEMENT_CAT + GLOBAL_INTERFACE_CAT + CHAIN_INTERFACE_CAT,
    "g5_num": CHEM_NUM + EXP_NUM,
    "g5_cat": CHEM_CAT + EXP_CAT,
}

GROUP_ORDER = [
    "g0_num", "g0_cat", "g1_num", "g1_cat", "g2_num", "g2_cat",
    "g3_num", "g3_cat", "g4_num", "g4_cat", "g5_num", "g5_cat"
]

MAIN_GROUP_ORDER = ["g0", "g1", "g2", "g3", "g4", "g5"]
MAIN_GROUP_LABELS = {
    "g0": "G0 AF3",
    "g1": "G1 Evolutionary",
    "g2": "G2 Complexity",
    "g3": "G3 RNA / Biophys",
    "g4": "G4 Interface",
    "g5": "G5 Exp / Chem",
}

FEATURE_TO_GROUP_EXACT = {}
GROUP_PREFIXES: list[tuple[str, str]] = []

for group_name, feats in GROUP_DEFS.items():
    for feat in feats:
        FEATURE_TO_GROUP_EXACT[feat] = group_name
        if group_name.endswith("_cat"):
            GROUP_PREFIXES.append((feat + "_", group_name))

GROUP_PREFIXES = sorted(GROUP_PREFIXES, key=lambda x: len(x[0]), reverse=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize recurring positive/negative coefficients across selected runs.")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("feature_stability_outputs"))

    p.add_argument("--coef-name", default="coef__gx.csv")
    p.add_argument("--groups", default="gx")
    p.add_argument("--run-scope", choices=["full", "pruned", "both"], default="full")
    p.add_argument("--require-path-substring", action="append", default=[])
    p.add_argument("--exclude-path-substring", action="append", default=[])

    p.add_argument("--top-pos-k", type=int, default=10)
    p.add_argument("--top-neg-k", type=int, default=10)

    p.add_argument("--display-top-k", type=int, default=0)
    p.add_argument("--display-max-per-group", type=int, default=0,
                   help="Maximum number of displayed features per MAIN group (G0..G5).")
    p.add_argument("--display-max-per-subgroup", type=int, default=0,
                   help="Optional maximum number of displayed features per subgroup (g*_num/g*_cat).")
    p.add_argument("--min-total", type=float, default=0.0)
    p.add_argument("--sort-by", choices=["group_total", "total", "signed_bias", "alphabetical"], default="group_total")

    p.add_argument("--group-separator-linewidth", type=float, default=1.0)
    p.add_argument("--group-gap", type=float, default=0.8)
    p.add_argument("--group-header-mode", choices=["inside", "left", "none"], default="inside")
    p.add_argument("--group-header-x", type=float, default=0.52)
    p.add_argument("--group-header-y-pad", type=float, default=0.55)
    p.add_argument("--group-header-fontsize", type=float, default=10.5)
    p.add_argument("--group-header-weight", default="bold")
    p.add_argument("--group-header-color", default="black")
    p.add_argument("--group-header-box-facecolor", default="white")
    p.add_argument("--group-header-box-edgecolor", default="none")
    p.add_argument("--group-header-box-alpha", type=float, default=0.8)
    p.add_argument("--group-header-box-pad", type=float, default=0.2)

    p.add_argument("--group-label-x", type=float, default=-0.18)
    p.add_argument("--group-label-fontsize", type=float, default=10.0)
    p.add_argument("--group-label-weight", default="bold")
    p.add_argument("--group-label-color", default="black")

    p.add_argument("--barplot-normalize", action="store_true")
    p.add_argument("--barplot-normalize-denominator", choices=["runs", "slots"], default="runs")
    p.add_argument("--barplot-show-values", action="store_true")
    p.add_argument("--barplot-value-fontsize", type=float, default=9.0)
    p.add_argument("--barplot-value-format", default=".2f")

    p.add_argument("--selected-features", type=Path, default=None)
    p.add_argument("--selected-regex", default=r'"([^"]+)"|\'([^\']+)\'')
    p.add_argument("--highlight-selected", action="store_true")
    p.add_argument("--selected-highlight-mode", choices=["ticklabel", "marker", "both"], default="both")
    p.add_argument("--selected-marker", default="*")
    p.add_argument("--selected-marker-size", type=float, default=60.0)
    p.add_argument("--selected-marker-offset", type=float, default=0.06)
    p.add_argument("--selected-color", default="black")
    p.add_argument("--unselected-color", default="#7F7F7F")

    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--barplot-width", type=float, default=14.0)
    p.add_argument("--barplot-row-height", type=float, default=0.34)
    p.add_argument("--barplot-min-height", type=float, default=6.0)

    p.add_argument("--title-fontsize", type=float, default=15.0)
    p.add_argument("--label-fontsize", type=float, default=12.0)
    p.add_argument("--tick-fontsize", type=float, default=9.0)
    p.add_argument("--legend-fontsize", type=float, default=10.0)

    p.add_argument("--figure-facecolor", default="white")
    p.add_argument("--axes-facecolor", default="white")
    p.add_argument("--pos-color", default="#4C78A8")
    p.add_argument("--neg-color", default="#E45756")
    p.add_argument("--group-separator-color", default="#B0B0B0")
    p.add_argument("--xlabel-color", default="black")
    p.add_argument("--ylabel-color", default="black")
    p.add_argument("--title-color", default="black")
    p.add_argument("--xtick-color", default="black")
    p.add_argument("--ytick-color", default="black")
    p.add_argument("--spine-color", default="black")
    p.add_argument("--grid-color", default="#DDDDDD")
    p.add_argument("--grid-alpha", type=float, default=0.6)
    p.add_argument("--grid-linewidth", type=float, default=0.8)
    p.add_argument("--show-x-grid", action="store_true")
    p.add_argument("--show-y-grid", action="store_true")

    p.add_argument("--barplot-title", default="Feature recurrence among top positive / negative coefficients")
    p.add_argument("--barplot-xlabel", default="Count across runs")
    p.add_argument("--barplot-ylabel", default="Feature")
    p.add_argument("--hide-barplot-ylabel", action="store_true")

    p.add_argument("--legend-loc", default="best")
    p.add_argument("--legend-frameon", action="store_true")
    p.add_argument("--legend-ncol", type=int, default=1)
    p.add_argument("--legend-anchor-x", type=float, default=None)
    p.add_argument("--legend-anchor-y", type=float, default=None)
    p.add_argument("--legend-borderaxespad", type=float, default=0.5)

    p.add_argument("--xlim-min", type=float, default=None)
    p.add_argument("--xlim-max", type=float, default=None)
    return p.parse_args()


def infer_feature_column(df: pd.DataFrame) -> str:
    for col in ["feature", "Feature", "Unnamed: 0", "name", "variable"]:
        if col in df.columns:
            return col
    return df.columns[0]


def infer_coef_column(df: pd.DataFrame) -> str:
    for col in ["coef", "coefficient", "beta", "weight"]:
        if col in df.columns:
            return col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError("Could not infer coefficient column.")


def read_coeff_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    feat_col = infer_feature_column(df)
    coef_col = infer_coef_column(df)
    out = df[[feat_col, coef_col]].copy()
    out.columns = ["feature", "coef"]
    out["feature"] = out["feature"].astype(str).str.strip()
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    out = out.dropna(subset=["feature", "coef"])
    out = out[out["feature"] != ""]
    return out


def load_selected_features(path: Optional[Path], regex: str) -> set[str]:
    if path is None:
        return set()
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".txt":
        return {line.strip() for line in text.splitlines() if line.strip()}

    if suffix == ".json":
        obj = json.loads(text)
        features: set[str] = set()

        def walk(x):
            if isinstance(x, str):
                features.add(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(obj)
        return features

    if suffix == ".py":
        features: set[str] = set()
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    features.add(node.value)
        except SyntaxError:
            pass
        for m in re.finditer(regex, text):
            val = m.group(1) or m.group(2)
            if val:
                features.add(val)
        return {f for f in features if re.match(r"^[A-Za-z0-9_\.]+$", f)}

    raise ValueError(f"Unsupported selected-features file type: {path.suffix}")


def normalize_groups(groups_arg: str) -> set[str]:
    return {g.strip() for g in groups_arg.split(",") if g.strip()}


def map_feature_to_group(feature: str) -> str:
    if feature in FEATURE_TO_GROUP_EXACT:
        return FEATURE_TO_GROUP_EXACT[feature]
    for prefix, group in GROUP_PREFIXES:
        if feature.startswith(prefix):
            return group
    raise ValueError(
        f"Feature '{feature}' could not be assigned to a predefined group. "
        "Please add it to the grouping definitions."
    )


def is_selected_feature(feature: str, selected: set[str]) -> bool:
    if feature in selected:
        return True

    for sel in selected:
        if feature.startswith(sel + "_"):
            return True

    return False

def subgroup_to_main_group(subgroup: str) -> str:
    return subgroup.split("_", 1)[0]


def path_matches_filters(path: Path, groups: set[str], run_scope: str,
                         require_substrings: list[str], exclude_substrings: list[str]) -> bool:
    path_str = str(path)
    m = re.search(r"coef__([^.]+)\.csv$", path.name)
    group = m.group(1) if m else None
    if group not in groups:
        return False

    path_lower = path_str.lower()
    has_full = "_full/" in path_lower or "full/" in path_lower or "_full\\" in path_lower or "full\\" in path_lower
    has_pruned = "_pruned/" in path_lower or "pruned/" in path_lower or "_pruned\\" in path_lower or "pruned\\" in path_lower

    if run_scope == "full" and not has_full:
        return False
    if run_scope == "pruned" and not has_pruned:
        return False

    for s in require_substrings:
        if s not in path_str:
            return False
    for s in exclude_substrings:
        if s in path_str:
            return False
    return True


def find_coeff_files(input_dir: Path, coef_name: str, groups: set[str], run_scope: str,
                     require_substrings: list[str], exclude_substrings: list[str]) -> list[Path]:
    files = sorted(input_dir.rglob(coef_name))
    return [p for p in files if p.is_file() and path_matches_filters(p, groups, run_scope, require_substrings, exclude_substrings)]


def summarize_feature_counts(coeff_files: Iterable[Path], top_pos_k: int, top_neg_k: int) -> tuple[pd.DataFrame, int]:
    counts: dict[str, dict[str, int]] = {}
    coeff_files = list(coeff_files)
    n_runs = 0

    for path in coeff_files:
        df = read_coeff_file(path)
        n_runs += 1
        pos = df[df["coef"] > 0].sort_values("coef", ascending=False).head(top_pos_k)
        neg = df[df["coef"] < 0].sort_values("coef", ascending=True).head(top_neg_k)

        for feat in pos["feature"]:
            counts.setdefault(feat, {"pos": 0, "neg": 0})
            counts[feat]["pos"] += 1
        for feat in neg["feature"]:
            counts.setdefault(feat, {"pos": 0, "neg": 0})
            counts[feat]["neg"] += 1

    if not counts:
        raise RuntimeError("No usable coefficient data found after filtering.")

    summary = pd.DataFrame([{"feature": feat, "pos_count": vals["pos"], "neg_count": vals["neg"]} for feat, vals in counts.items()])
    summary["feature_group"] = summary["feature"].apply(map_feature_to_group)
    summary["main_group"] = summary["feature_group"].map(subgroup_to_main_group)
    summary["main_group_label"] = summary["main_group"].map(lambda x: MAIN_GROUP_LABELS[x])
    summary["group_order"] = summary["feature_group"].map(lambda x: GROUP_ORDER.index(x))
    summary["main_group_order"] = summary["main_group"].map(lambda x: MAIN_GROUP_ORDER.index(x))

    summary["total_count"] = summary["pos_count"] + summary["neg_count"]
    summary["signed_bias"] = np.where(summary["total_count"] > 0, (summary["pos_count"] - summary["neg_count"]) / summary["total_count"], 0.0)
    summary["dominant_sign"] = np.select(
        [summary["pos_count"] > summary["neg_count"], summary["neg_count"] > summary["pos_count"]],
        ["+", "-"],
        default="±",
    )
    summary["pos_frac_of_runs"] = summary["pos_count"] / n_runs
    summary["neg_frac_of_runs"] = summary["neg_count"] / n_runs
    summary["total_frac_of_runs"] = summary["total_count"] / n_runs
    summary["pos_frac_of_slots"] = summary["pos_count"] / n_runs
    summary["neg_frac_of_slots"] = summary["neg_count"] / n_runs
    summary["total_frac_of_slots"] = summary["total_count"] / (2.0 * n_runs)
    return summary, n_runs


def sort_summary(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "group_total":
        return df.sort_values(
            ["main_group_order", "group_order", "total_count", "pos_count", "neg_count", "feature"],
            ascending=[True, True, False, False, False, True]
        )
    if sort_by == "total":
        return df.sort_values(["total_count", "pos_count", "neg_count", "feature"], ascending=[False, False, False, True])
    if sort_by == "signed_bias":
        return df.sort_values(["signed_bias", "total_count", "feature"], ascending=[False, False, True])
    return df.sort_values("feature", ascending=True)


def apply_display_filters(df: pd.DataFrame, min_total: float, display_top_k: int,
                          display_max_per_group: int, display_max_per_subgroup: int,
                          normalize_for_threshold: bool) -> pd.DataFrame:
    out = df.copy()
    threshold_col = "total_frac_of_runs" if normalize_for_threshold else "total_count"
    if min_total > 0:
        out = out[out[threshold_col] >= min_total]

    if display_max_per_subgroup and display_max_per_subgroup > 0:
        out = out.groupby("feature_group", sort=False, group_keys=False).head(display_max_per_subgroup)

    if display_max_per_group and display_max_per_group > 0:
        out = out.groupby("main_group", sort=False, group_keys=False).head(display_max_per_group)

    if display_top_k and display_top_k > 0:
        out = out.head(display_top_k)

    # Final display order: within each main group, sort by bar length/frequency descending.
    out = out.sort_values(
        ["main_group_order", "total_count", "pos_count", "neg_count", "feature"],
        ascending=[True, False, False, False, True]
    )

    return out


def escape_latex(s: str) -> str:
    for k, v in {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "{": r"\{", "}": r"\}"}.items():
        s = s.replace(k, v)
    return s


def write_latex_table(df: pd.DataFrame, outpath: Path, include_selected: bool) -> None:
    cols = ["main_group_label", "feature", "pos_count", "neg_count", "total_count", "dominant_sign"]
    if include_selected:
        cols.append("Selected")
    header = ["Group", "Feature", "Pos", "Neg", "Total", "Sign"] + (["Selected"] if include_selected else [])
    colspec = "llcccc" + ("c" if include_selected else "")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Feature stability across selected GX full-feature coefficient runs. For each run, the top positive and top negative coefficients were counted separately. Features are grouped according to the predefined feature families.}",
        fr"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]
    last_group = None
    for _, row in df[cols].iterrows():
        current_group = row["main_group_label"]
        if last_group is not None and current_group != last_group:
            lines.append(r"\hline")
        vals = [escape_latex(str(row[c])) if c in {"feature", "main_group_label"} else str(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")
        last_group = current_group
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def build_group_layout(df: pd.DataFrame, group_gap: float):
    y_positions = []
    group_infos = []
    separator_positions = []
    current_y = 0.0

    group_counts = df.groupby("main_group_label", sort=False).size()
    for i, (group_label, count) in enumerate(group_counts.items()):
        first_y = current_y
        for _ in range(count):
            y_positions.append(current_y)
            current_y += 1.0
        last_y = current_y - 1.0
        group_infos.append({"label": group_label, "first_y": first_y, "last_y": last_y, "mid_y": (first_y + last_y) / 2.0})
        if i < len(group_counts) - 1:
            separator_positions.append(current_y - 0.5 + group_gap / 2.0)
            current_y += group_gap

    return np.array(y_positions), group_infos, separator_positions


def add_group_annotations(ax, group_infos, separator_positions, args):
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for sep in separator_positions:
        ax.axhline(sep, color=args.group_separator_color, linewidth=args.group_separator_linewidth, zorder=1)

    if args.group_header_mode == "none":
        return

    if args.group_header_mode == "left":
        for info in group_infos:
            ax.text(
                args.group_label_x, info["mid_y"], info["label"],
                va="center", ha="right",
                fontsize=args.group_label_fontsize, fontweight=args.group_label_weight,
                color=args.group_label_color, transform=trans, clip_on=False, zorder=5
            )
        return

    for info in group_infos:
        y = info["first_y"] - args.group_header_y_pad
        ax.text(
            args.group_header_x, y, info["label"],
            va="center", ha="center",
            fontsize=args.group_header_fontsize, fontweight=args.group_header_weight,
            color=args.group_header_color, transform=trans, clip_on=False, zorder=5,
            bbox={
                "facecolor": args.group_header_box_facecolor,
                "edgecolor": args.group_header_box_edgecolor,
                "alpha": args.group_header_box_alpha,
                "pad": args.group_header_box_pad,
            },
        )


def make_barplot(df: pd.DataFrame, outpath: Path, args: argparse.Namespace, include_selected: bool) -> None:
    plot_df = df.copy()

    if args.barplot_normalize:
        if args.barplot_normalize_denominator == "runs":
            plot_df["neg_plot"] = plot_df["neg_frac_of_runs"]
            plot_df["pos_plot"] = plot_df["pos_frac_of_runs"]
            xlabel_default = "Fraction of runs"
        else:
            plot_df["neg_plot"] = plot_df["neg_frac_of_slots"]
            plot_df["pos_plot"] = plot_df["pos_frac_of_slots"]
            xlabel_default = "Fraction of sign-specific slots"
    else:
        plot_df["neg_plot"] = plot_df["neg_count"]
        plot_df["pos_plot"] = plot_df["pos_count"]
        xlabel_default = "Count across runs"

    xlabel = xlabel_default if (args.barplot_xlabel == "Count across runs" and args.barplot_normalize) else args.barplot_xlabel

    n = len(plot_df)
    n_groups = plot_df["main_group_label"].nunique()
    fig_h = max(args.barplot_min_height, args.barplot_row_height * n + args.group_gap * max(n_groups - 1, 0))
    fig, ax = plt.subplots(figsize=(args.barplot_width, fig_h), facecolor=args.figure_facecolor)
    ax.set_facecolor(args.axes_facecolor)

    y, group_infos, separator_positions = build_group_layout(plot_df, args.group_gap)

    ax.barh(y, plot_df["neg_plot"], label="Negative", color=args.neg_color, zorder=2)
    ax.barh(y, plot_df["pos_plot"], left=plot_df["neg_plot"], label="Positive", color=args.pos_color, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"], fontsize=args.tick_fontsize, color=args.ytick_color)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=args.label_fontsize, color=args.xlabel_color)
    ax.set_ylabel("" if args.hide_barplot_ylabel else args.barplot_ylabel,
                  fontsize=args.label_fontsize, color=args.ylabel_color)
    ax.set_title(args.barplot_title, fontsize=args.title_fontsize, color=args.title_color)
    ax.tick_params(axis="x", labelsize=args.tick_fontsize, colors=args.xtick_color)
    ax.tick_params(axis="y", labelsize=args.tick_fontsize, colors=args.ytick_color)

    for spine in ax.spines.values():
        spine.set_color(args.spine_color)

    if args.show_x_grid:
        ax.grid(axis="x", color=args.grid_color, alpha=args.grid_alpha, linewidth=args.grid_linewidth, zorder=0)
    if args.show_y_grid:
        ax.grid(axis="y", color=args.grid_color, alpha=args.grid_alpha, linewidth=args.grid_linewidth, zorder=0)

    totals = plot_df["neg_plot"] + plot_df["pos_plot"]
    max_total = float(totals.max()) if len(totals) else 0.0

    if include_selected and args.highlight_selected:
        is_selected = plot_df["Selected"].eq("Yes").values

        if args.selected_highlight_mode in {"ticklabel", "both"}:
            for tick, selected in zip(ax.get_yticklabels(), is_selected):
                tick.set_color(args.selected_color if selected else args.unselected_color)

        if args.selected_highlight_mode in {"marker", "both"}:
            marker_x = max_total + args.selected_marker_offset
            y_selected = y[is_selected]
            if len(y_selected) > 0:
                ax.scatter(
                    np.full(len(y_selected), marker_x, dtype=float),
                    y_selected,
                    marker=args.selected_marker,
                    s=args.selected_marker_size,
                    color=args.selected_color,
                    clip_on=False,
                    label="Selected",
                    zorder=4,
                )
            if args.xlim_max is None:
                ax.set_xlim(right=marker_x + args.selected_marker_offset)

    if args.barplot_show_values:
        pad = 0.01 * max(max_total, 1.0)
        for yi, total_val in zip(y, totals):
            ax.text(
                total_val + pad, yi, format(total_val, args.barplot_value_format),
                va="center", ha="left",
                fontsize=args.barplot_value_fontsize, color=args.title_color, zorder=4
            )

    add_group_annotations(ax, group_infos, separator_positions, args)

    if args.xlim_min is not None or args.xlim_max is not None:
        current_left, current_right = ax.get_xlim()
        ax.set_xlim(
            left=args.xlim_min if args.xlim_min is not None else current_left,
            right=args.xlim_max if args.xlim_max is not None else current_right,
        )

    legend_kwargs = {
        "fontsize": args.legend_fontsize,
        "loc": args.legend_loc,
        "frameon": args.legend_frameon,
        "ncol": args.legend_ncol,
        "borderaxespad": args.legend_borderaxespad,
    }
    if args.legend_anchor_x is not None and args.legend_anchor_y is not None:
        legend_kwargs["bbox_to_anchor"] = (args.legend_anchor_x, args.legend_anchor_y)
    ax.legend(**legend_kwargs)

    fig.tight_layout()
    save_all_formats(fig, outpath, dpi=max(600, args.dpi))
    plt.close(fig)


def write_manifest(coeff_files: list[Path], outpath: Path) -> None:
    pd.DataFrame({"coef_file": [str(p) for p in coeff_files]}).to_csv(outpath, index=False)


def write_group_summary(df: pd.DataFrame, outpath: Path) -> None:
    grouped = (
        df.groupby(["main_group", "main_group_label"], dropna=False)
        .agg(
            n_features=("feature", "count"),
            pos_count_sum=("pos_count", "sum"),
            neg_count_sum=("neg_count", "sum"),
            total_count_sum=("total_count", "sum"),
            mean_total_count=("total_count", "mean"),
        )
        .reset_index()
        .sort_values(["main_group"])
    )
    grouped.to_csv(outpath, index=False)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = normalize_groups(args.groups)
    coeff_files = find_coeff_files(
        input_dir=args.input_dir,
        coef_name=args.coef_name,
        groups=groups,
        run_scope=args.run_scope,
        require_substrings=args.require_path_substring,
        exclude_substrings=args.exclude_path_substring,
    )
    if not coeff_files:
        raise SystemExit("No coefficient files found after filtering.")

    selected = load_selected_features(args.selected_features, args.selected_regex)
    summary, n_runs = summarize_feature_counts(coeff_files, args.top_pos_k, args.top_neg_k)

    include_selected = bool(selected)

    if include_selected:
        summary["Selected"] = np.where(
            summary["feature"].apply(lambda x: is_selected_feature(str(x), selected)),
            "Yes",
            "No",
        )
    
    summary = sort_summary(summary, args.sort_by)
    summary_all = summary.copy()
    summary_display = apply_display_filters(
        summary,
        min_total=args.min_total,
        display_top_k=args.display_top_k,
        display_max_per_group=args.display_max_per_group,
        display_max_per_subgroup=args.display_max_per_subgroup,
        normalize_for_threshold=args.barplot_normalize,
    )

    summary_all.to_csv(args.outdir / "feature_stability_summary_all.csv", index=False)
    summary_display.to_csv(args.outdir / "feature_stability_summary_display.csv", index=False)
    write_group_summary(summary_all, args.outdir / "feature_stability_group_summary.csv")
    write_manifest(coeff_files, args.outdir / "included_coefficient_files.csv")
    write_latex_table(summary_all, args.outdir / "feature_stability_table_all.tex", include_selected)
    write_latex_table(summary_display, args.outdir / "feature_stability_table_display.tex", include_selected)
    make_barplot(summary_display, args.outdir / "feature_stability_barplot.png", args, include_selected)

    metadata = {
        "n_runs": n_runs,
        "n_coeff_files": len(coeff_files),
        "coef_name": args.coef_name,
        "groups": sorted(groups),
        "run_scope": args.run_scope,
        "top_pos_k": args.top_pos_k,
        "top_neg_k": args.top_neg_k,
        "display_top_k": args.display_top_k,
        "display_max_per_group": args.display_max_per_group,
        "display_max_per_subgroup": args.display_max_per_subgroup,
        "min_total": args.min_total,
        "sort_by": args.sort_by,
        "barplot_normalize": args.barplot_normalize,
        "barplot_normalize_denominator": args.barplot_normalize_denominator,
        "group_gap": args.group_gap,
        "group_header_mode": args.group_header_mode,
        "selected_features_provided": include_selected,
        "n_selected_features_loaded": len(selected),
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Included coefficient files: {len(coeff_files)}")
    print(f"Runs summarized: {n_runs}")
    print(f"Output directory: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
