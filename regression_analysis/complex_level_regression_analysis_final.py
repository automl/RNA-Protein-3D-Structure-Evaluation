#!/usr/bin/env python3
"""
complex_level_regression_analysis_with_interface.py

Grouped CV analysis for complex-level performance.

Supports:
- Standard regression/classification on a target column
- Optional AF3-error regression:
    residual = y_true - y_pred_from_AF3_metrics_OOF
  Then model the residual with the same feature families.

Model families:
  M0   : complexity only
  M1   : templates only
  M2   : af3 metrics only
  M3   : chemistry only
  M4   : experimental setup only
  M5a  : complexity + templates
  M5b  : complexity + af3 metrics
  M5c  : complexity + chemistry
  M5d  : complexity + experimental setup
  M6   : composition only
  M7   : predicted interface only
  M8   : RNA intrinsic only
  M9   : complexity + composition
  M10  : complexity + predicted interface
  M11  : af3 + composition
  M12  : af3 + predicted interface
  M13  : af3 + RNA intrinsic
  M14  : interface features only
  M15  : complexity + interface features
  M16  : af3 + interface features
  M17  : interface predicted + interface features
  M18  : composition + interface features
  MX   : all

Scenarios:
  - full
  - chem_subset

Design goals:
- easy feature inclusion/exclusion via editable lists below
- stable chemistry regression with:
    * sparse-feature filtering
    * configurable Ridge alpha
    * optional RobustScaler
- optional AF3 residual analysis without changing your CSVs

Usage examples
--------------
Standard regression:
python complex_level_regression_analysis.py \
  --csv csvs4analysis/b2021/b2021__rna_protein_1v1.csv \
  --ycol Dockq \
  --task regress \
  --group_by pdb_id \
  --cv 5 \
  --dataset b2021 \
  --subset rna_protein_1v1 \
  --run_chem_subset \
  --chem_required_cols ionic_strength,buffer_m

AF3-error regression:
python complex_level_regression_analysis.py \
  --csv csvs4analysis/b2021/b2021__rna_protein_1v1.csv \
  --ycol Dockq \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --dataset b2021 \
  --subset rna_protein_1v1 \
  --run_chem_subset \
  --chem_required_cols ionic_strength,buffer_m

Outputs:
  results/<dataset>/complex_analysis/<subset>/<analysis_name>/<scenario>/{plots,csvs}/...
"""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(x: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-+") else "_" for ch in str(x))


def short_token(x: str, n: int = 12) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()[:n]


def shorten_name_for_path(x: str, keep: int = 24, n_hash: int = 12) -> str:
    x = safe_slug(x)
    if len(x) <= keep + n_hash + 1:
        return x
    return f"{x[:keep]}__{short_token(x, n_hash)}"


def model_tag(model_name: str) -> str:
    mapping = {
        "G0_af3": "g0",
        "G1_evolution": "g1",
        "G2_complexity": "g2",
        "G3_rna_intrinsic": "g3",
        "G4_interface": "g4",
        "G5_experimental": "g5",
    }
    if model_name in mapping:
        return mapping[model_name]

    if model_name.startswith("A"):
        prefix = model_name.split("_", 1)[0]
        if len(prefix) > 1 and prefix[1:].isdigit():
            return prefix.lower()

    return shorten_name_for_path(model_name, keep=16, n_hash=8)


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_required_cols(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def make_groups(df: pd.DataFrame, group_by: str) -> Tuple[np.ndarray, bool, str]:
    if group_by == "none":
        return np.arange(len(df)), False, "group_by=none"

    if group_by not in df.columns:
        return np.arange(len(df)), False, f"[WARN] group_by='{group_by}' not found -> using non-group CV"

    g = df[group_by].astype(str).fillna("NA").values
    n_groups = len(pd.unique(g))
    if n_groups < 2:
        return np.arange(len(df)), False, f"[WARN] group_by='{group_by}' has only {n_groups} group -> using non-group CV"

    return g, True, f"group_by={group_by} (n_groups={n_groups})"


def choose_splitter(
    y: np.ndarray,
    groups: np.ndarray,
    use_group_cv: bool,
    cv: int,
    seed: int,
    task: str,
) -> Tuple[object, int, str]:
    if use_group_cv:
        n_groups = len(pd.unique(groups))
        if task == "classify":
            uniq, counts = np.unique(y, return_counts=True)
            if len(uniq) < 2:
                return None, 0, "need at least 2 classes for classification"
            min_class = int(np.min(counts)) if len(counts) else 0
            eff = min(cv, n_groups, min_class)
            if eff < 2:
                return None, 0, f"not enough groups/classes for grouped stratified CV (n_groups={n_groups}, min_class={min_class})"
            return StratifiedGroupKFold(n_splits=eff, shuffle=True, random_state=seed), eff, f"StratifiedGroupKFold(n_splits={eff})"
        eff = min(cv, n_groups)
        if eff < 2:
            return None, 0, f"not enough groups (n_groups={n_groups})"
        return GroupKFold(n_splits=eff), eff, f"GroupKFold(n_splits={eff})"

    if task == "classify":
        _, counts = np.unique(y, return_counts=True)
        min_class = int(np.min(counts)) if len(counts) else 0
        eff = min(cv, min_class)
        if eff < 2:
            return None, 0, f"not enough samples per class (min_class={min_class})"
        return StratifiedKFold(n_splits=eff, shuffle=True, random_state=seed), eff, f"StratifiedKFold(n_splits={eff})"

    eff = min(cv, len(y))
    if eff < 2:
        return None, 0, f"not enough samples (n={len(y)})"
    return KFold(n_splits=eff, shuffle=True, random_state=seed), eff, f"KFold(n_splits={eff})"

# =============================================================================
# Feature registry
# Comment features in/out here
# =============================================================================

# -------------------------------------------------------------------
# G0 AF3 internal confidence
# -------------------------------------------------------------------
AF3_NUM = [
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    # protein-side + RNA-interface confidence (mean over chains); the confidence
    # analysis flags these as the strongest complex-accuracy correlates
    "af3_protein_pLDDT_avg",
    "af3_protein_pTM_mean",
    "af3_protein_ipTM_mean",
    "af3_rna_ipTM_mean",
    # "af3_rna_pTM_list_value",
]
AF3_CAT: List[str] = []

# -------------------------------------------------------------------
# G1 Evolutionary signal
# -------------------------------------------------------------------
EVO_NUM = [
    "log10_rna_msa_size",
    "log10_protein_msa_size",
    "n_unique_rna_families",
    "n_rbp_templates",
    "n_protein_templates",
]
EVO_CAT: List[str] = [
    "rna_family",
]

# -------------------------------------------------------------------
# G2 Complexity / composition
# -------------------------------------------------------------------
COMPLEXITY_NUM = [
    "NumberRNAs",
    "NumberProteins",
    "rna_total_length",
    "protein_total_length",
    "total_complex_length",
    "protein_to_rna_length_ratio",
    "max_rna_fraction_of_total_rna",
]
COMPLEXITY_CAT: List[str] = []

# -------------------------------------------------------------------
# G3 RNA intrinsic properties
# -------------------------------------------------------------------
RNA_INTRINSIC_NUM = [
    # "RNA_SequenceComplexity",
    # "RNA_Shannon_Entropy_K1",
    # "RNA_MFE_Value",
    # "RNA_Ensemble_Diversity",
    # "RNA_Centroid_Energy",
    "agg_rna_sequence_complexity",
    "agg_rna_shannon_entropy_k2",
]
RNA_INTRINSIC_CAT: List[str] = []

RNA_STRUCTURE_NUM = [
    # "num_stems",
    # "num_hairpin_loops",
    # "num_internal_loops",
    # "num_multibranch_loops",
    # "num_pseudoknots",
]
RNA_STRUCTURE_CAT: List[str] = []

BIOPHYS_NUM = [
    "log10_kd",
]
BIOPHYS_CAT: List[str] = []

# -------------------------------------------------------------------
# G4 Interface features
# -------------------------------------------------------------------
INTERFACE_AGREEMENT_NUM = [
    "pred_ProteinRNAInterfaceArea",
    "pred_ProteinRNAInterfaceRatio",
]
INTERFACE_AGREEMENT_CAT: List[str] = []

GLOBAL_INTERFACE_NUM = [
    "n_unique_rna_interface_residues_global",
    "n_unique_protein_interface_residues_global",
    "rna_interface_coverage_global",
    "interface_contact_distance_mean",
    "interface_contact_distance_median",
    "frac_contacts_lt_3_5A",
    "interface_rna_motif_length_sum",
    "interface_rna_motif_coverage_of_total_rna",
    "interface_frac_contacts_to_C",
]
GLOBAL_INTERFACE_CAT: List[str] = []

CHAIN_INTERFACE_NUM = [
    # "chain_n_unique_rna_interface_residues",
    # "chain_interface_contact_distance_mean",
    # "chain_frac_contacts_lt_3_5A",
    # "chain_interface_rna_motif_length_sum",
]
CHAIN_INTERFACE_CAT: List[str] = []

# -------------------------------------------------------------------
# G5 Experimental setup + chemistry / environment
# -------------------------------------------------------------------
CHEM_NUM = [
    "n_ions_listed",
    "n_ligands_listed",
    "ionic_strength",
    "peg_percent",
    "buffer_m",
    "has_peg",
]
CHEM_CAT = [
    "peg_concentration_category",
    "buffer_category",
]

EXP_NUM = [
    "exp_pH",
    "exp_Temp",
    # "is_post_cutoff",
]
# X-ray-only metadata (structure-determination method + resolution) is experimental
# "oracle" info: known only once the experimental structure exists, NOT available from
# a prediction, and a non-mechanistic composition proxy (X-ray vs cryo-EM) that is
# treated elsewhere as a confound to control for. It is therefore EXCLUDED from the
# default reproduction / regression. Set INCLUDE_XRAY_FEATURES=1 to switch it back on
# (e.g. for the dedicated X-ray-vs-EM importance analysis).
if __import__("os").environ.get("INCLUDE_XRAY_FEATURES", "0") == "1":
    EXP_NUM += ["XRayResolution", "is_xray"]
EXP_CAT: List[str] = []

# =============================================================================
# Dynamic feature discovery
# =============================================================================
def get_dynamic_feature_groups(
    df: pd.DataFrame,
    include_solution_features: bool,
    include_unsupported_code_features: bool,
) -> Dict[str, List[str]]:
    unsupported_ion_dynamic = sorted(
        c for c in df.columns if c.startswith("has_unsupported_ion_")
    )
    unsupported_ligand_dynamic = sorted(
        c for c in df.columns if c.startswith("has_unsupported_ligand_")
    )

    solution_num_dynamic = sorted(
        c for c in df.columns if c.startswith("sol_") or c.startswith("sample_")
    )

    chem_num = CHEM_NUM.copy()
    chem_cat = CHEM_CAT.copy()

    if include_unsupported_code_features:
        chem_num += unsupported_ion_dynamic + unsupported_ligand_dynamic

    if include_solution_features:
        chem_num += solution_num_dynamic

    return {
        "g0_num": [c for c in AF3_NUM if c in df.columns],
        "g0_cat": [c for c in AF3_CAT if c in df.columns],

        "g1_num": [c for c in EVO_NUM if c in df.columns],
        "g1_cat": [c for c in EVO_CAT if c in df.columns],

        "g2_num": [c for c in COMPLEXITY_NUM if c in df.columns],
        "g2_cat": [c for c in COMPLEXITY_CAT if c in df.columns],

        "g3_num": [
            c for c in (RNA_INTRINSIC_NUM + RNA_STRUCTURE_NUM + BIOPHYS_NUM)
            if c in df.columns
        ],
        "g3_cat": [
            c for c in (RNA_INTRINSIC_CAT + RNA_STRUCTURE_CAT + BIOPHYS_CAT)
            if c in df.columns
        ],

        "g4_num": [
            c for c in (INTERFACE_AGREEMENT_NUM + GLOBAL_INTERFACE_NUM + CHAIN_INTERFACE_NUM)
            if c in df.columns
        ],
        "g4_cat": [
            c for c in (INTERFACE_AGREEMENT_CAT + GLOBAL_INTERFACE_CAT + CHAIN_INTERFACE_CAT)
            if c in df.columns
        ],

        "g5_num": [c for c in (chem_num + EXP_NUM) if c in df.columns],
        "g5_cat": [c for c in (chem_cat + EXP_CAT) if c in df.columns],

        "dynamic_solution_num": [c for c in solution_num_dynamic if c in df.columns],
        "dynamic_unsupported_num": [
            c for c in (unsupported_ion_dynamic + unsupported_ligand_dynamic)
            if c in df.columns
        ],
    }


def _dedup_keep_order(cols: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _spec(num_cols: List[str], cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    return _dedup_keep_order(num_cols), _dedup_keep_order(cat_cols)


def merge_specs(*specs: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for num, cat in specs:
        num_cols.extend(num)
        cat_cols.extend(cat)
    return _spec(num_cols, cat_cols)


def generate_af3_anchored_model_specs(
    S0: Tuple[List[str], List[str]],
    addon_specs: List[Tuple[str, Tuple[List[str], List[str]]]],
) -> Dict[str, Tuple[List[str], List[str]]]:
    out: Dict[str, Tuple[List[str], List[str]]] = {
        "A0_af3": S0,
    }

    idx = 1
    for k in range(1, len(addon_specs) + 1):
        for combo in combinations(addon_specs, k):
            combo_names = [name for name, _ in combo]
            combo_specs = [spec for _, spec in combo]
            model_name = f"A{idx}_af3_" + "_".join(combo_names)
            out[model_name] = merge_specs(S0, *combo_specs)
            idx += 1

    return out


def get_model_order(model_specs: Dict[str, Tuple[List[str], List[str]]]) -> List[str]:
    single_order = [
        "G0_af3",
        "G1_evolution",
        "G2_complexity",
        "G3_rna_intrinsic",
        "G4_interface",
        "G5_experimental",
    ]

    out = [m for m in single_order if m in model_specs]

    a_models = []
    for m in model_specs:
        if m.startswith("A"):
            prefix = m.split("_", 1)[0]
            if len(prefix) > 1 and prefix[1:].isdigit():
                a_models.append((int(prefix[1:]), m))
            else:
                a_models.append((10**9, m))

    a_models = [m for _, m in sorted(a_models, key=lambda x: (x[0], x[1]))]
    out.extend(a_models)
    return out


def build_model_specs(
    df: pd.DataFrame,
    include_solution_features: bool,
    include_unsupported_code_features: bool,
    target_mode: str,
    include_rna_family_ohe: bool = False,
    full_model_universe: bool = False,
) -> Dict[str, Tuple[List[str], List[str]]]:
    groups = get_dynamic_feature_groups(
        df,
        include_solution_features=include_solution_features,
        include_unsupported_code_features=include_unsupported_code_features,
    )

    g0_num = groups["g0_num"]
    g0_cat = groups["g0_cat"]

    g1_num = groups["g1_num"]
    g1_cat = groups["g1_cat"].copy()
    if not include_rna_family_ohe:
        g1_cat = [c for c in g1_cat if c != "rna_family"]

    g2_num = groups["g2_num"]
    g2_cat = groups["g2_cat"]

    g3_num = groups["g3_num"]
    g3_cat = groups["g3_cat"]

    g4_num = groups["g4_num"]
    g4_cat = groups["g4_cat"]

    g5_num = groups["g5_num"]
    g5_cat = groups["g5_cat"]

    S0 = _spec(g0_num, g0_cat)
    S1 = _spec(g1_num, g1_cat)
    S2 = _spec(g2_num, g2_cat)
    S3 = _spec(g3_num, g3_cat)
    S4 = _spec(g4_num, g4_cat)
    S5 = _spec(g5_num, g5_cat)

    single_models = {
        "G0_af3": S0,
        "G1_evolution": S1,
        "G2_complexity": S2,
        "G3_rna_intrinsic": S3,
        "G4_interface": S4,
        "G5_experimental": S5,
    }

    if not full_model_universe:
        additive_models = {
            "A0_af3": S0,
            "A1_af3_evolution": merge_specs(S0, S1),
            "A2_af3_interface": merge_specs(S0, S4),
            "A3_af3_experimental": merge_specs(S0, S5),
            "A4_af3_evolution_experimental": merge_specs(S0, S1, S5),
            "A5_af3_evolution_interface": merge_specs(S0, S1, S4),
            "A6_af3_evolution_interface_rnaintr": merge_specs(S0, S1, S4, S3),
            "A7_af3_evolution_interface_rnaintr_complexity": merge_specs(S0, S1, S4, S3, S2),
            "A8_af3_evolution_interface_rnaintr_complexity_experimental": merge_specs(S0, S1, S4, S3, S2, S5),
        }
        return {**single_models, **additive_models}

    addon_specs = [
        ("evolution", S1),
        ("complexity", S2),
        ("rna_intrinsic", S3),
        ("interface", S4),
        ("experimental", S5),
    ]
    af3_combo_models = generate_af3_anchored_model_specs(S0, addon_specs)

    return {**single_models, **af3_combo_models}

# =============================================================================
# Pipelines
# =============================================================================
def make_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    use_robust_scaler: bool,
    min_category_count: int = 10,
) -> ColumnTransformer:
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scaler),
        ]
    )

    if cat_cols:
        if min_category_count and min_category_count > 1:
            oh = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=min_category_count,
                sparse_output=False,
            )
        else:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh", oh),
            ]
        )
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def make_model(task: str, seed: int, ridge_alpha: float):
    if task == "classify":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=4000,
            class_weight="balanced",
        )
    return Ridge(alpha=ridge_alpha, random_state=seed)


def make_pipeline(
    task: str,
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    ridge_alpha: float,
    use_robust_scaler: bool,
    variance_threshold: float = 1e-8,
    min_category_count: int = 10,
) -> Pipeline:
    pre = make_preprocessor(
        num_cols,
        cat_cols,
        use_robust_scaler=use_robust_scaler,
        min_category_count=min_category_count,
    )
    model = make_model(task, seed, ridge_alpha=ridge_alpha)
    return Pipeline(steps=[("pre", pre), ("var", VarianceThreshold(threshold=variance_threshold)), ("model", model)])

def get_feature_names_after_fit(pipe: Pipeline, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names: List[str] = []
    names.extend(num_cols)

    if cat_cols:
        pre: ColumnTransformer = pipe.named_steps["pre"]
        if "cat" in pre.named_transformers_:
            cat_trans = pre.named_transformers_["cat"]
            oh: OneHotEncoder = cat_trans.named_steps["oh"]
            try:
                names.extend(list(oh.get_feature_names_out(cat_cols)))
            except Exception:
                names.extend(cat_cols)

    if "var" in pipe.named_steps:
        selector: VarianceThreshold = pipe.named_steps["var"]
        support = selector.get_support()
        if len(support) == len(names):
            names = [n for n, keep in zip(names, support) if keep]
    return names

# =============================================================================
# Metrics
# =============================================================================
def get_valid_prediction_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return pd.notna(y_true_arr) & np.isfinite(y_pred_arr)


def compute_metrics_classify(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    valid = get_valid_prediction_mask(y_true_arr, y_prob_arr)

    if valid.sum() == 0:
        return {"auroc": np.nan, "ap": np.nan, "mcc": np.nan, "brier": np.nan}

    y_true_v = y_true_arr[valid].astype(int)
    y_prob_v = np.clip(y_prob_arr[valid], 0.0, 1.0)
    y_pred_v = (y_prob_v >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true_v)) > 1

    return {
        "auroc": roc_auc_score(y_true_v, y_prob_v) if has_both_classes else np.nan,
        "ap": average_precision_score(y_true_v, y_prob_v) if has_both_classes else np.nan,
        "mcc": matthews_corrcoef(y_true_v, y_pred_v) if has_both_classes else np.nan,
        "brier": brier_score_loss(y_true_v, y_prob_v),
    }


def compute_metrics_regress(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid = get_valid_prediction_mask(y_true_arr, y_pred_arr)

    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}

    y_true_v = y_true_arr[valid]
    y_pred_v = y_pred_arr[valid]
    rmse = float(np.sqrt(mean_squared_error(y_true_v, y_pred_v)))
    mae = float(mean_absolute_error(y_true_v, y_pred_v))
    r2 = float(r2_score(y_true_v, y_pred_v)) if len(y_true_v) >= 2 else np.nan

    if len(np.unique(y_true_v)) < 2 or len(np.unique(y_pred_v)) < 2:
        spearman = np.nan
    else:
        spearman = float(pd.Series(y_true_v).corr(pd.Series(y_pred_v), method="spearman"))

    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}

@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]


def summarize_folds(fr: List[FoldResult]) -> Dict[str, str]:
    if not fr:
        return {}

    keys = sorted(fr[0].metrics.keys())
    out: Dict[str, str] = {}
    for k in keys:
        vals = np.array([r.metrics.get(k, np.nan) for r in fr], dtype=float)
        if np.all(np.isnan(vals)):
            out[k] = "nan±nan"
        else:
            out[k] = f"{np.nanmean(vals):.3f}±{np.nanstd(vals):.3f}"
    return out

# =============================================================================
# Plotting
# =============================================================================
def plot_pred_vs_true(y_true: np.ndarray, y_hat: np.ndarray, title: str, out_png: Path) -> None:
    valid = get_valid_prediction_mask(y_true, y_hat)
    if valid.sum() == 0:
        return
    y_true_v = np.asarray(y_true, dtype=float)[valid]
    y_hat_v = np.asarray(y_hat, dtype=float)[valid]

    plt.figure(figsize=(6.5, 6.0), constrained_layout=True)
    plt.scatter(y_true_v, y_hat_v, alpha=0.7)
    mn = float(np.nanmin([np.nanmin(y_true_v), np.nanmin(y_hat_v)]))
    mx = float(np.nanmax([np.nanmax(y_true_v), np.nanmax(y_hat_v)]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_hat: np.ndarray, title: str, out_png: Path) -> None:
    valid = get_valid_prediction_mask(y_true, y_hat)
    if valid.sum() == 0:
        return
    y_true_v = np.asarray(y_true, dtype=float)[valid]
    y_hat_v = np.asarray(y_hat, dtype=float)[valid]
    resid = y_true_v - y_hat_v
    plt.figure(figsize=(6.5, 4.8), constrained_layout=True)
    plt.scatter(y_hat_v, resid, alpha=0.7)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (true - pred)")
    plt.title(title)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_coefs(coefs: pd.Series, title: str, out_png: Path, top_k: int = 20) -> None:
    coefs = coefs.dropna()
    if len(coefs) == 0:
        return
    top_pos = coefs.sort_values(ascending=False).head(top_k)
    top_neg = coefs.sort_values(ascending=True).head(top_k)
    shown = pd.concat([top_neg, top_pos], axis=0)

    plt.figure(figsize=(10, max(6, 0.25 * len(shown))), constrained_layout=True)
    plt.barh(shown.index, shown.values)
    plt.axvline(0)
    plt.title(title)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roc_pr_if_needed(task: str, y_true: np.ndarray, y_prob: np.ndarray, title: str, out_dir: Path, prefix: str) -> None:
    if task != "classify":
        return

    valid = get_valid_prediction_mask(y_true, y_prob)
    if valid.sum() == 0:
        return

    y_true_v = np.asarray(y_true)[valid].astype(int)
    y_prob_v = np.clip(np.asarray(y_prob, dtype=float)[valid], 0.0, 1.0)
    if len(np.unique(y_true_v)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true_v, y_prob_v)
    auc = roc_auc_score(y_true_v, y_prob_v)
    plt.figure(figsize=(6.5, 5.6), constrained_layout=True)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title}\nAUROC={auc:.3f}")
    plt.savefig(out_dir / f"{prefix}__roc.png", dpi=200)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true_v, y_prob_v)
    ap = average_precision_score(y_true_v, y_prob_v)
    plt.figure(figsize=(6.5, 5.6), constrained_layout=True)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nAP={ap:.3f}")
    plt.savefig(out_dir / f"{prefix}__pr.png", dpi=200)
    plt.close()


# =============================================================================
# Feature filtering
# =============================================================================
def filter_feature_cols(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    min_non_na_feature: int,
    max_cat_levels: int,
    max_missing_frac_feature: float,
) -> Tuple[List[str], List[str]]:
    num = [c for c in num_cols if c in df.columns]
    cat = [c for c in cat_cols if c in df.columns]

    def drop_all_nan(cols: List[str]) -> Tuple[List[str], List[str]]:
        kept, dropped = [], []
        for c in cols:
            if df[c].notna().any():
                kept.append(c)
            else:
                dropped.append(c)
        return kept, dropped

    def drop_too_sparse_numeric(cols: List[str], min_non_na: int) -> Tuple[List[str], List[str]]:
        kept, dropped = [], []
        for c in cols:
            n_obs = int(df[c].notna().sum())
            if n_obs >= min_non_na:
                kept.append(c)
            else:
                dropped.append(c)
        return kept, dropped

    def drop_high_missing(cols: List[str], max_missing_frac: float) -> Tuple[List[str], List[str]]:
        kept, dropped = [], []
        for c in cols:
            frac = float(df[c].isna().mean())
            if frac <= max_missing_frac:
                kept.append(c)
            else:
                dropped.append(c)
        return kept, dropped

    def drop_constant(cols: List[str]) -> Tuple[List[str], List[str]]:
        kept, dropped = [], []
        for c in cols:
            nun = df[c].nunique(dropna=True)
            if nun is None or nun <= 1:
                dropped.append(c)
            else:
                kept.append(c)
        return kept, dropped

    def drop_high_cardinality_cat(cols: List[str], max_levels: int) -> Tuple[List[str], List[str]]:
        kept, dropped = [], []
        for c in cols:
            nun = df[c].nunique(dropna=True)
            if nun > max_levels:
                dropped.append(c)
            else:
                kept.append(c)
        return kept, dropped

    num, dropped_num_nan = drop_all_nan(num)
    cat, dropped_cat_nan = drop_all_nan(cat)
    num, dropped_num_sparse = drop_too_sparse_numeric(num, min_non_na=min_non_na_feature)
    num, dropped_num_missing = drop_high_missing(num, max_missing_frac_feature)
    cat, dropped_cat_missing = drop_high_missing(cat, max_missing_frac_feature)
    num, dropped_num_const = drop_constant(num)
    cat, dropped_cat_const = drop_constant(cat)
    cat, dropped_cat_high = drop_high_cardinality_cat(cat, max_levels=max_cat_levels)

    if dropped_num_nan:
        print(f"  - drop all-NaN numeric: {dropped_num_nan}")
    if dropped_cat_nan:
        print(f"  - drop all-NaN categorical: {dropped_cat_nan}")
    if dropped_num_sparse:
        print(f"  - drop sparse numeric (<{min_non_na_feature} observed): {dropped_num_sparse}")
    if dropped_num_missing:
        print(f"  - drop high-missing numeric (>{max_missing_frac_feature:.2f} missing): {dropped_num_missing}")
    if dropped_cat_missing:
        print(f"  - drop high-missing categorical (>{max_missing_frac_feature:.2f} missing): {dropped_cat_missing}")
    if dropped_num_const:
        print(f"  - drop constant numeric: {dropped_num_const}")
    if dropped_cat_const:
        print(f"  - drop constant categorical: {dropped_cat_const}")
    if dropped_cat_high:
        print(f"  - drop high-cardinality categorical (>{max_cat_levels} levels): {dropped_cat_high}")

    return num, cat

# =============================================================================
# Core runner
# =============================================================================
def run_cv_one_model(
    df: pd.DataFrame,
    y_raw: pd.Series,
    task: str,
    groups: np.ndarray,
    use_group_cv: bool,
    cv: int,
    seed: int,
    num_cols: List[str],
    cat_cols: List[str],
    min_rows_model: int,
    ridge_alpha: float,
    use_robust_scaler: bool,
    min_non_na_feature: int,
    max_cat_levels: int,
    max_missing_frac_feature: float,
    variance_threshold: float,
    min_category_count: int,
) -> Tuple[List[FoldResult], np.ndarray, List[str], List[str], str]:
    if len(df) < min_rows_model:
        return [], np.array([]), [], [], f"SKIP: too few rows (n={len(df)} < {min_rows_model})"

    num_eff, cat_eff = filter_feature_cols(
        df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        min_non_na_feature=min_non_na_feature,
        max_cat_levels=max_cat_levels,
        max_missing_frac_feature=max_missing_frac_feature,
    )
    if not num_eff and not cat_eff:
        return [], np.array([]), [], [], "SKIP: no usable features"

    num_eff = list(dict.fromkeys(num_eff))
    cat_eff = list(dict.fromkeys(cat_eff))

    X = df[num_eff + cat_eff].copy()
    y = y_raw.astype(int).values if task == "classify" else y_raw.astype(float).values

    splitter, eff_cv, cv_note = choose_splitter(
        y=y,
        groups=groups,
        use_group_cv=use_group_cv,
        cv=cv,
        seed=seed,
        task=task,
    )
    if splitter is None:
        return [], np.array([]), num_eff, cat_eff, f"SKIP: {cv_note}"

    splits = splitter.split(X, y_raw, groups=groups) if use_group_cv else splitter.split(X, y_raw)

    oof = np.full(len(df), np.nan, dtype=float)
    fold_results: List[FoldResult] = []

    for fold, (tr, te) in enumerate(splits, start=1):
        pipe = make_pipeline(
            task=task,
            num_cols=num_eff,
            cat_cols=cat_eff,
            seed=seed,
            ridge_alpha=ridge_alpha,
            use_robust_scaler=use_robust_scaler,
            variance_threshold=variance_threshold,
            min_category_count=min_category_count,
        )
        if task == "classify":
            y_tr = y_raw.iloc[tr].astype(int).values
            y_te = y_raw.iloc[te].astype(int).values
            if len(np.unique(y_tr)) < 2:
                prob = np.full(len(te), float(np.mean(y_tr)) if len(y_tr) else np.nan, dtype=float)
            else:
                pipe.fit(X.iloc[tr], y_raw.iloc[tr])
                prob = pipe.predict_proba(X.iloc[te])[:, 1]
            prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
            oof[te] = prob
            m = compute_metrics_classify(y_te, prob)
        else:
            pipe.fit(X.iloc[tr], y_raw.iloc[tr])
            pred = pipe.predict(X.iloc[te]).astype(float)
            oof[te] = pred
            m = compute_metrics_regress(y_raw.iloc[te].astype(float).values, pred)

        fold_results.append(FoldResult(fold=fold, metrics=m))

    return fold_results, oof, num_eff, cat_eff, cv_note

def scenario_df(df: pd.DataFrame, scenario: str, chem_required_cols: List[str]) -> Tuple[pd.DataFrame, str]:
    if scenario == "full":
        return df, "full"

    if scenario == "chem_subset":
        if not chem_required_cols:
            return df, "chem_subset (no required cols specified -> same as full)"

        missing_cols = [c for c in chem_required_cols if c not in df.columns]
        if missing_cols:
            return df.iloc[0:0].copy(), f"chem_subset SKIP: required cols missing from CSV: {missing_cols}"

        mask = np.ones(len(df), dtype=bool)
        for c in chem_required_cols:
            mask &= df[c].notna().values

        out = df.loc[mask].copy()
        return out, f"chem_subset requiring notna: {chem_required_cols} (kept n={len(out)})"

    raise ValueError(f"Unknown scenario: {scenario}")


# =============================================================================
# AF3 residual target
# =============================================================================
def compute_af3_error_target_oof(
    df: pd.DataFrame,
    y_raw: pd.Series,
    task: str,
    group_by: str,
    cv: int,
    seed: int,
    ridge_alpha: float,
    use_robust_scaler: bool,
    min_rows_model: int,
    min_non_na_feature: int,
    max_cat_levels: int,
    max_missing_frac_feature: float,
    variance_threshold: float,
    min_category_count: int,
) -> Tuple[pd.Series, pd.Series, str]:
    """
    Build OOF AF3-only predictions and residual target:
      residual = y_true - y_pred_af3_oof

    Returns:
      residual_series, af3_pred_series, note
    """
    if task != "regress":
        raise ValueError("AF3 error target is only implemented for regress task.")

    af3_num = [c for c in AF3_NUM if c in df.columns]
    af3_cat = [c for c in AF3_CAT if c in df.columns]

    groups, use_group_cv, group_msg = make_groups(df, group_by)

    fr, oof_pred, num_eff, cat_eff, cv_note = run_cv_one_model(
        df=df,
        y_raw=y_raw,
        task="regress",
        groups=groups,
        use_group_cv=use_group_cv,
        cv=cv,
        seed=seed,
        num_cols=af3_num,
        cat_cols=af3_cat,
        min_rows_model=min_rows_model,
        ridge_alpha=ridge_alpha,
        use_robust_scaler=use_robust_scaler,
        min_non_na_feature=min_non_na_feature,
        max_cat_levels=max_cat_levels,
        max_missing_frac_feature=max_missing_frac_feature,
        variance_threshold=variance_threshold,
        min_category_count=min_category_count,
    )

    if not fr:
        raise ValueError(f"Could not build AF3 baseline residual target: {cv_note}")

    residual = y_raw.astype(float) - pd.Series(oof_pred, index=df.index, dtype=float)
    pred_series = pd.Series(oof_pred, index=df.index, dtype=float)
    note = f"AF3 residual target built with {cv_note}; features_used={num_eff + cat_eff}"
    return residual, pred_series, note

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True, help="Input CSV")
    ap.add_argument("--ycol", required=True, help="Target column for standard mode, or base target for af3_error mode")
    ap.add_argument("--task", choices=["regress", "classify"], default="regress")
    ap.add_argument("--ythresh", type=float, default=None, help="Only for classification")

    ap.add_argument(
        "--target_mode",
        choices=["direct", "af3_error", "dockq_minus_tm"],
        default="direct",
        help=(
            "direct: use ycol directly; "
            "af3_error: use OOF residual from AF3-only baseline; "
            "dockq_minus_tm: use DockQ - complex_TM as derived target"
        ),
    )
    ap.add_argument("--group_by", default="pdb_id", help="Grouping column or 'none' (default pdb_id: grouped CV prevents same-PDB chains splitting across folds)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--analysis_name", default=None)
    ap.add_argument("--outdir", default="results")

    ap.add_argument("--min_rows", type=int, default=50)
    ap.add_argument("--min_rows_model", type=int, default=40)

    ap.add_argument("--run_chem_subset", action="store_true")
    ap.add_argument(
        "--chem_required_cols",
        default="ionic_strength,buffer_m",
        help="Comma-separated columns required to be non-missing for chem_subset",
    )
    ap.add_argument(
        "--include_solution_features",
        action="store_true",
        help="Include parsed sol_* and sample_* columns in chemistry models",
    )
    ap.add_argument(
        "--include_unsupported_code_features",
        action="store_true",
        help="Include dynamic has_unsupported_ion_* and has_unsupported_ligand_* columns in chemistry models",
    )

    ap.add_argument(
        "--ridge_alpha",
        type=float,
        default=10.0,
        help="Alpha for Ridge regression",
    )
    ap.add_argument(
        "--use_robust_scaler",
        action="store_true",
        help="Use RobustScaler instead of StandardScaler",
    )
    ap.add_argument(
        "--min_non_na_feature",
        type=int,
        default=10,
        help="Drop numeric features with fewer than this many observed values",
    )
    ap.add_argument(
        "--max_cat_levels",
        type=int,
        default=25,
        help="Drop categorical columns with more than this many unique levels",
    )

    ap.add_argument(
        "--min_category_count",
        type=int,
        default=10,
        help="Minimum category count before OneHotEncoder keeps a level separate; rarer levels are grouped as infrequent",
    )
    ap.add_argument(
        "--max_missing_frac_feature",
        type=float,
        default=0.80,
        help="Drop features with missing fraction greater than this threshold",
    )
    ap.add_argument(
        "--variance_threshold",
        type=float,
        default=1e-8,
        help="Post-preprocessing variance threshold applied before model fitting",
    )
    ap.add_argument(
        "--include_rna_family_ohe",
        action="store_true",
        help="Accepted for runner compatibility; complex-level models currently ignore this flag",
    )
    ap.add_argument(
        "--full_model_universe",
        action="store_true",
        help="Run exhaustive AF3-anchored model universe over all non-AF3 feature groups",
    )
    args = ap.parse_args()

    df0 = pd.read_csv(args.csv)

    if args.ycol not in df0.columns:
        raise ValueError(f"ycol '{args.ycol}' not found in CSV columns.")

    if args.target_mode == "af3_error" and args.task != "regress":
        raise ValueError("--target_mode af3_error currently requires --task regress")

    # -------------------------------------------------------------------------
    # Target preparation
    # -------------------------------------------------------------------------
    if args.task == "classify":
        if args.ythresh is None:
            raise ValueError("--ythresh is required for --task classify")
        y_raw0 = to_float_series(df0[args.ycol])
        keep = y_raw0.notna()
        df0 = df0.loc[keep].copy()
        y_raw0 = y_raw0.loc[keep]
        df0["_y"] = (y_raw0 >= float(args.ythresh)).astype(int)
        final_target = df0["_y"]
        target_label = args.ycol
        af3_baseline_note = None
        af3_pred_oof = None
    else:
        if args.target_mode == "direct":
            y_raw0 = to_float_series(df0[args.ycol])
            keep = y_raw0.notna()
            df0 = df0.loc[keep].copy()
            y_raw0 = y_raw0.loc[keep]

            final_target = y_raw0.astype(float)
            target_label = args.ycol
            af3_baseline_note = None
            af3_pred_oof = None

        elif args.target_mode == "dockq_minus_tm":
            required_cols = ["Dockq", "Complex_TM"]
            missing_cols = [c for c in required_cols if c not in df0.columns]
            if missing_cols:
                raise ValueError(
                    f"--target_mode dockq_minus_tm requires columns {required_cols}; "
                    f"missing: {missing_cols}"
                )

            dockq = to_float_series(df0["Dockq"])
            ctm = to_float_series(df0["Complex_TM"])
            derived = dockq - ctm

            keep = derived.notna()
            df0 = df0.loc[keep].copy()
            derived = derived.loc[keep]

            df0["_dockq_minus_tm"] = derived.astype(float)
            final_target = df0["_dockq_minus_tm"]
            target_label = "DockQ_minus_complex_TM"
            af3_baseline_note = None
            af3_pred_oof = None

        elif args.target_mode == "af3_error":
            y_raw0 = to_float_series(df0[args.ycol])
            keep = y_raw0.notna()
            df0 = df0.loc[keep].copy()
            y_raw0 = y_raw0.loc[keep]

            residual, af3_pred_oof, af3_baseline_note = compute_af3_error_target_oof(
                df=df0,
                y_raw=y_raw0.astype(float),
                task="regress",
                group_by=args.group_by,
                cv=args.cv,
                seed=args.seed,
                ridge_alpha=args.ridge_alpha,
                use_robust_scaler=args.use_robust_scaler,
                min_rows_model=args.min_rows_model,
                min_non_na_feature=args.min_non_na_feature,
                max_cat_levels=args.max_cat_levels,
                max_missing_frac_feature=args.max_missing_frac_feature,
                variance_threshold=args.variance_threshold,
                min_category_count=args.min_category_count,
            )
            df0["_af3_pred_oof"] = af3_pred_oof
            df0["_af3_error"] = residual
            final_target = df0["_af3_error"].astype(float)
            target_label = f"AF3_ERROR__{args.ycol}"

        else:
            raise ValueError(f"Unsupported target_mode: {args.target_mode}")
    if len(df0) < args.min_rows:
        raise ValueError(f"Too few rows after target filtering: n={len(df0)} < min_rows={args.min_rows}")

    model_specs = build_model_specs(
        df0,
        include_solution_features=args.include_solution_features,
        include_unsupported_code_features=args.include_unsupported_code_features,
        target_mode=args.target_mode,
        include_rna_family_ohe=args.include_rna_family_ohe,
        full_model_universe=args.full_model_universe,
    )

    groups, use_group_cv, group_msg = make_groups(df0, args.group_by)

    analysis_name = args.analysis_name
    if analysis_name is None:
        base = safe_slug(target_label)
        if args.task == "classify":
            base += f"_thr{str(args.ythresh).replace('.', 'p')}"
        if args.target_mode == "af3_error":
            base += "__af3error"
        if args.include_solution_features:
            base += "__with_solution"
        if args.include_unsupported_code_features:
            base += "__with_unsupported_codes"
        if args.use_robust_scaler:
            base += "__robust"
        base += f"__a{str(args.ridge_alpha).replace('.', 'p')}"
        base += f"__grp{safe_slug(args.group_by)}__cv{args.cv}"
        analysis_name = base

    analysis_dir_name = shorten_name_for_path(analysis_name, keep=24, n_hash=12)
    out_root = Path(args.outdir) / args.dataset / "cx" / shorten_name_for_path(args.subset, keep=16, n_hash=8) / analysis_dir_name
    if (out_root / ".run_complete").exists() and not __import__("os").environ.get("FORCE_RERUN"):
        print(f"[resume] skip (already complete): {out_root}  (set FORCE_RERUN=1 to redo)")
        return
    ensure_dir(out_root)

    cfg = {
        "csv": args.csv,
        "ycol_input": args.ycol,
        "target_label": target_label,
        "target_mode": args.target_mode,
        "task": args.task,
        "ythresh": args.ythresh,
        "group_by": args.group_by,
        "cv": args.cv,
        "seed": args.seed,
        "dataset": args.dataset,
        "subset": args.subset,
        "analysis_name": analysis_name,
        "min_rows": args.min_rows,
        "min_rows_model": args.min_rows_model,
        "run_chem_subset": bool(args.run_chem_subset),
        "chem_required_cols": args.chem_required_cols,
        "include_solution_features": bool(args.include_solution_features),
        "include_unsupported_code_features": bool(args.include_unsupported_code_features),
        "ridge_alpha": args.ridge_alpha,
        "use_robust_scaler": bool(args.use_robust_scaler),
        "min_non_na_feature": args.min_non_na_feature,
        "max_cat_levels": args.max_cat_levels,
        "min_category_count": args.min_category_count,
        "max_missing_frac_feature": args.max_missing_frac_feature,
        "variance_threshold": args.variance_threshold,
        "include_rna_family_ohe": bool(args.include_rna_family_ohe),
        "full_model_universe": bool(args.full_model_universe),
        "af3_baseline_note": af3_baseline_note,
        "model_specs": {k: {"num": v[0], "cat": v[1]} for k, v in model_specs.items()},
        "feature_registry": {
            "AF3_NUM": AF3_NUM,
            "AF3_CAT": AF3_CAT,
            "EVO_NUM": EVO_NUM,
            "EVO_CAT": EVO_CAT,
            "COMPLEXITY_NUM": COMPLEXITY_NUM,
            "COMPLEXITY_CAT": COMPLEXITY_CAT,
            "RNA_INTRINSIC_NUM": RNA_INTRINSIC_NUM,
            "RNA_INTRINSIC_CAT": RNA_INTRINSIC_CAT,
            "RNA_STRUCTURE_NUM": RNA_STRUCTURE_NUM,
            "RNA_STRUCTURE_CAT": RNA_STRUCTURE_CAT,
            "BIOPHYS_NUM": BIOPHYS_NUM,
            "BIOPHYS_CAT": BIOPHYS_CAT,
            "INTERFACE_AGREEMENT_NUM": INTERFACE_AGREEMENT_NUM,
            "INTERFACE_AGREEMENT_CAT": INTERFACE_AGREEMENT_CAT,
            "GLOBAL_INTERFACE_NUM": GLOBAL_INTERFACE_NUM,
            "GLOBAL_INTERFACE_CAT": GLOBAL_INTERFACE_CAT,
            "CHAIN_INTERFACE_NUM": CHAIN_INTERFACE_NUM,
            "CHAIN_INTERFACE_CAT": CHAIN_INTERFACE_CAT,
            "CHEM_NUM": CHEM_NUM,
            "CHEM_CAT": CHEM_CAT,
            "EXP_NUM": EXP_NUM,
            "EXP_CAT": EXP_CAT,
        },
    }
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with open(out_root / "path_name_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis_name_full": analysis_name,
                "analysis_dir_name": analysis_dir_name,
                "subset_full": args.subset,
                "subset_dir_name": shorten_name_for_path(args.subset, keep=16, n_hash=8),
            },
            f,
            indent=2,
        )

    scenarios = ["full"]
    if args.run_chem_subset:
        scenarios.append("chem_subset")

    chem_required_cols = parse_required_cols(args.chem_required_cols)

    print("=" * 90)
    print(f"CSV: {args.csv}")
    print(f"Task: {args.task}")
    print(f"Target label: {target_label}")
    print(f"Target mode: {args.target_mode}")
    print(group_msg)
    print(f"Include solution features: {args.include_solution_features}")
    print(f"Include unsupported code features: {args.include_unsupported_code_features}")
    print(f"Ridge alpha: {args.ridge_alpha}")
    print(f"Use RobustScaler: {args.use_robust_scaler}")
    print(f"Min non-NA feature count: {args.min_non_na_feature}")
    print(f"Max missing fraction per feature: {args.max_missing_frac_feature}")
    print(f"Variance threshold: {args.variance_threshold}")
    print(f"Min category count: {args.min_category_count}")
    print(f"Full model universe: {args.full_model_universe}")
    if af3_baseline_note:
        print(f"AF3 baseline note: {af3_baseline_note}")

    print("\nFeature registry actually requested:")
    print(f"  AF3_NUM ({len(AF3_NUM)}): {AF3_NUM}")
    print(f"  EVO_NUM ({len(EVO_NUM)}): {EVO_NUM}")
    print(f"  EVO_CAT ({len(EVO_CAT)}): {EVO_CAT}")
    print(f"  COMPLEXITY_NUM ({len(COMPLEXITY_NUM)}): {COMPLEXITY_NUM}")
    print(f"  RNA_INTRINSIC_NUM ({len(RNA_INTRINSIC_NUM)}): {RNA_INTRINSIC_NUM}")
    print(f"  RNA_STRUCTURE_NUM ({len(RNA_STRUCTURE_NUM)}): {RNA_STRUCTURE_NUM}")
    print(f"  BIOPHYS_NUM ({len(BIOPHYS_NUM)}): {BIOPHYS_NUM}")
    print(f"  INTERFACE_AGREEMENT_NUM ({len(INTERFACE_AGREEMENT_NUM)}): {INTERFACE_AGREEMENT_NUM}")
    print(f"  GLOBAL_INTERFACE_NUM ({len(GLOBAL_INTERFACE_NUM)}): {GLOBAL_INTERFACE_NUM}")
    print(f"  CHAIN_INTERFACE_NUM ({len(CHAIN_INTERFACE_NUM)}): {CHAIN_INTERFACE_NUM}")
    print(f"  CHEM_NUM ({len(CHEM_NUM)}): {CHEM_NUM}")
    print(f"  CHEM_CAT ({len(CHEM_CAT)}): {CHEM_CAT}")
    print(f"  EXP_NUM ({len(EXP_NUM)}): {EXP_NUM}")
    print(f"  EXP_CAT ({len(EXP_CAT)}): {EXP_CAT}")
    dyn = get_dynamic_feature_groups(
        df0,
        include_solution_features=args.include_solution_features,
        include_unsupported_code_features=args.include_unsupported_code_features,
    )
    print(f"  dynamic solution numeric ({len(dyn['dynamic_solution_num'])})")
    print(f"  dynamic unsupported numeric ({len(dyn['dynamic_unsupported_num'])})")
    print(f"Number of requested models: {len(get_model_order(model_specs))}")

    for scenario in scenarios:
        df, scen_msg = scenario_df(df0, scenario=scenario, chem_required_cols=chem_required_cols)
        print("\n" + "-" * 90)
        print(f"[SCENARIO] {scenario}: {scen_msg}")

        if len(df) < args.min_rows_model:
            print(f"SKIP scenario '{scenario}': n={len(df)} < min_rows_model={args.min_rows_model}")
            continue

        model_specs_s = build_model_specs(
            df,
            include_solution_features=args.include_solution_features,
            include_unsupported_code_features=args.include_unsupported_code_features,
            target_mode=args.target_mode,
            include_rna_family_ohe=args.include_rna_family_ohe,
            full_model_universe=args.full_model_universe,
        )

        scenario_dir = {"full": "f", "chem_subset": "c"}.get(scenario, shorten_name_for_path(scenario, keep=8, n_hash=6))
        scen_root = out_root / scenario_dir
        plots_dir = scen_root / "p"
        csvs_dir = scen_root / "c"
        ensure_dir(plots_dir)
        ensure_dir(csvs_dir)

        with open(scen_root / "scenario_name_map.json", "w", encoding="utf-8") as f:
            json.dump({"scenario_full": scenario, "scenario_dir": scenario_dir}, f, indent=2)

        groups_s, use_group_cv_s, group_msg_s = make_groups(df, args.group_by)

        all_candidate_cols = sorted({c for spec in model_specs_s.values() for c in (spec[0] + spec[1])})
        present_candidate_cols = [c for c in all_candidate_cols if c in df.columns]

        if present_candidate_cols:
            miss = df[present_candidate_cols].isna().mean().sort_values(ascending=False)
            miss.to_csv(csvs_dir / "missingness.csv")
        else:
            pd.DataFrame({"message": ["no candidate feature columns present"]}).to_csv(
                csvs_dir / "missingness.csv", index=False
            )

        pd.DataFrame(
            {
                "feature": all_candidate_cols,
                "present_in_csv": [c in df.columns for c in all_candidate_cols],
            }
        ).to_csv(csvs_dir / "candidate_features.csv", index=False)

        # Save AF3 baseline residual table if applicable
        if args.target_mode == "af3_error" and "_af3_error" in df.columns:
            pd.DataFrame(
                {
                    "index": df.index,
                    "y_true_base": to_float_series(df[args.ycol]).astype(float).values,
                    "af3_pred_oof": df["_af3_pred_oof"].astype(float).values,
                    "af3_error": df["_af3_error"].astype(float).values,
                }
            ).to_csv(csvs_dir / "af3_error_target.csv", index=False)

        metrics_rows = []
        per_fold_rows = []
        model_name_rows = []

        model_order = get_model_order(model_specs_s)

        for model_name in model_order:
            if model_name not in model_specs_s:
                continue
            num_cols, cat_cols = model_specs_s[model_name]
            print(f"\n[{scenario} | {model_name}]")
            model_name_rows.append({"model_full": model_name, "model_tag": model_tag(model_name)})

            y_for_model = df["_y"] if args.task == "classify" else final_target.loc[df.index]

            fr, oof, num_eff, cat_eff, cv_note = run_cv_one_model(
                df=df,
                y_raw=y_for_model,
                task=args.task,
                groups=groups_s,
                use_group_cv=use_group_cv_s,
                cv=args.cv,
                seed=args.seed,
                num_cols=num_cols,
                cat_cols=cat_cols,
                min_rows_model=args.min_rows_model,
                ridge_alpha=args.ridge_alpha,
                use_robust_scaler=args.use_robust_scaler,
                min_non_na_feature=args.min_non_na_feature,
                max_cat_levels=args.max_cat_levels,
                max_missing_frac_feature=args.max_missing_frac_feature,
                variance_threshold=args.variance_threshold,
                min_category_count=args.min_category_count,
            )

            if not fr:
                print(f"  {cv_note}")
                metrics_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "status": cv_note,
                        "n_rows": len(df),
                        "cv_effective": 0,
                    }
                )
                continue

            summ = summarize_folds(fr)
            print(f"  CV: {cv_note}")
            print("  " + "  ".join([f"{k.upper()}={v}" for k, v in summ.items()]))

            if args.task == "classify":
                y_true = df["_y"].astype(int).values
                oof_metrics = compute_metrics_classify(y_true, oof)
            else:
                y_true = y_for_model.astype(float).values
                oof_metrics = compute_metrics_regress(y_true, oof)

            valid_oof_mask = get_valid_prediction_mask(y_true, oof)
            n_valid_oof = int(valid_oof_mask.sum())
            n_missing_oof = int(len(valid_oof_mask) - n_valid_oof)
            if n_missing_oof > 0:
                print(f"  [WARN] {n_missing_oof} rows had non-finite OOF predictions and were excluded from OOF summary metrics")

            metrics_rows.append(
                {
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "analysis_name": analysis_name,
                    "scenario": scenario,
                    "model": model_name,
                    "task": args.task,
                    "ycol_input": args.ycol,
                    "target_label": target_label,
                    "target_mode": args.target_mode,
                    "ythresh": args.ythresh,
                    "group_by": args.group_by,
                    "cv_requested": args.cv,
                    "cv_effective": len(fr),
                    "n_rows": len(df),
                    "use_group_cv": bool(use_group_cv_s),
                    "cv_note": cv_note,
                    "num_features_requested": len(num_cols),
                    "cat_features_requested": len(cat_cols),
                    "num_features_used": len(num_eff),
                    "cat_features_used": len(cat_eff),
                    "status": "ok",
                    "n_valid_oof": n_valid_oof,
                    "n_missing_oof": n_missing_oof,
                    **{f"oof_{k}": float(v) for k, v in oof_metrics.items()},
                    **{f"fold_{k}": v for k, v in summ.items()},
                }
            )

            for r in fr:
                per_fold_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "fold": r.fold,
                        **{k: float(v) for k, v in r.metrics.items()},
                    }
                )

            X_full = df[num_eff + cat_eff].copy()
            y_full = y_for_model

            pipe_full = make_pipeline(
                task=args.task,
                num_cols=num_eff,
                cat_cols=cat_eff,
                seed=args.seed,
                ridge_alpha=args.ridge_alpha,
                use_robust_scaler=args.use_robust_scaler,
                variance_threshold=args.variance_threshold,
                min_category_count=args.min_category_count,
            )
            pipe_full.fit(X_full, y_full)

            feat_names = get_feature_names_after_fit(pipe_full, num_eff, cat_eff)

            pd.DataFrame(
                {
                    "feature": feat_names,
                    "feature_type": ["numeric"] * len(num_eff) + ["categorical_expanded"] * (len(feat_names) - len(num_eff)),
                }
            ).to_csv(csvs_dir / f"feat__{model_tag(model_name)}.csv", index=False)

            model = pipe_full.named_steps["model"]
            if hasattr(model, "coef_"):
                coef = model.coef_[0] if args.task == "classify" else model.coef_
                coef_s = pd.Series(np.asarray(coef).ravel(), index=feat_names).sort_values()
                coef_s.to_csv(csvs_dir / f"coef__{model_tag(model_name)}.csv", header=["coef"])
                top_neg = coef_s.head(10)
                top_pos = coef_s.tail(10).sort_values(ascending=False)
                pd.DataFrame({"feature": top_neg.index, "coef": top_neg.values}).to_csv(csvs_dir / f"coef_neg__{model_tag(model_name)}.csv", index=False)
                pd.DataFrame({"feature": top_pos.index, "coef": top_pos.values}).to_csv(csvs_dir / f"coef_pos__{model_tag(model_name)}.csv", index=False)

                print("  Top 10 negative coefficients:")
                print(top_neg.to_string())
                print("")
                print("  Top 10 positive coefficients:")
                print(top_pos.to_string())


                title = f"{args.dataset} | {args.subset} | {scenario} | {model_name}\n{target_label} | {group_msg_s} | cv={len(fr)}"
                plot_coefs(coef_s, title, plots_dir / f"coef__{model_tag(model_name)}.png", top_k=20)

            title = f"{args.dataset} | {args.subset} | {scenario} | {model_name}\n{target_label} | {group_msg_s} | cv={len(fr)}"
            if args.task == "regress":
                y_true_plot = y_for_model.astype(float).values
                plot_pred_vs_true(y_true_plot, oof, title, plots_dir / f"pvt__{model_tag(model_name)}.png")
                plot_residuals(y_true_plot, oof, title, plots_dir / f"resid__{model_tag(model_name)}.png")
            else:
                y_true_plot = df["_y"].astype(int).values
                plot_roc_pr_if_needed("classify", y_true_plot, oof, title, plots_dir, prefix=model_tag(model_name))

        pd.DataFrame(metrics_rows).to_csv(csvs_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(per_fold_rows).to_csv(csvs_dir / "metrics_per_fold.csv", index=False)
        pd.DataFrame(model_name_rows).drop_duplicates().to_csv(csvs_dir / "model_name_map.csv", index=False)

    print("\nDone.")
    try:
        (out_root / ".run_complete").write_text("done", encoding="utf-8")
    except Exception as _e:
        print(f"[resume] marker write failed: {_e}")
    print("Outputs in:", out_root.resolve())


if __name__ == "__main__":
    main()