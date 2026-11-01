#!/usr/bin/env python3
"""
variance_partitioning_shapley.py

Order-independent attribution of cross-validated explained variance (R^2) to the
six feature GROUPS used throughout the paper:

    G0_af3          AF3 internal confidence
    G1_evolution    evolutionary signal (MSA / templates / family)
    G2_complexity   size / composition
    G3_rna_intrinsic RNA intrinsic + biophysics
    G4_interface    predicted-interface features
    G5_experimental experimental setup + chemistry

Motivation
----------
The additive ladder (A0 -> A8 ...) reports the *incremental* R^2 of each group in
ONE fixed entry order. With correlated groups that ordering is arbitrary and the
"credit" each group receives depends on it. To make a defensible *ranking* claim
("group X is the largest non-AF3 contributor"), we use the Shapley decomposition
of R^2 (a.k.a. LMG / Lindeman-Merenda-Gold): each group's attribution is its
marginal contribution to out-of-fold R^2 averaged over ALL orderings of the
groups. Shapley values are order-independent and sum exactly to the full-model
explained variance, v(all) - v(empty).

Consistency with the rest of the pipeline
-----------------------------------------
This script does NOT redefine features or the model. It imports the *exact* group
definitions from the analysis module you point it at (--registry_module), and
rebuilds the harmonized pipeline used everywhere else:
    median impute -> StandardScaler  (numeric)
    most_frequent impute -> OneHotEncoder(min_frequency=min_category_count) (categorical)
    VarianceThreshold(variance_threshold)
    Ridge(alpha=ridge_alpha)
with GroupKFold by pdb_id (cv folds, seed). Coalition R^2 is the *pooled
out-of-fold* R^2 -- identical in spirit to the oof_r2 reported by the analysis.

target_mode:
  direct    : partition v(direct target) over all 6 groups (2^6 = 64 coalitions)
  af3_error : first build the AF3-only OOF residual (y - OOF(G0)), then partition
              the residual variance over the 5 NON-AF3 groups G1..G5 (2^5 = 32
              coalitions). This answers "what explains the part AF3 confidence
              misses", which is exactly the residual story in the paper.

Outputs (under <outdir>/<dataset>/<subset>/<analysis_name>/):
  shapley_r2.csv      per-group attribution + share + increment-over-AF3
  coalition_r2.csv    every coalition and its pooled OOF R^2
  shapley_r2.png      barplot of the attribution
  run_config.json     full configuration for reproducibility
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


GROUP_LABELS = {
    "G0_af3": "AF3 confidence",
    "G1_evolution": "Evolution",
    "G2_complexity": "Complexity",
    "G3_rna_intrinsic": "RNA intrinsic",
    "G4_interface": "Interface (pred.)",
    "G5_experimental": "Experimental/chem",
}
GROUP_ORDER = ["G0_af3", "G1_evolution", "G2_complexity", "G3_rna_intrinsic", "G4_interface", "G5_experimental"]


# ---------------------------------------------------------------------------
# Import the analysis module so we reuse its EXACT fixed feature registry
# ---------------------------------------------------------------------------
def load_registry_module(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"registry_module not found: {path}")
    spec = importlib.util.spec_from_file_location("af3_registry_mod", str(p))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod  # needed so @dataclass in the module resolves
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "get_dynamic_feature_groups"):
        raise AttributeError(f"{path} has no get_dynamic_feature_groups()")
    return mod


def build_group_specs(
    mod,
    df: pd.DataFrame,
    include_rna_family_ohe: bool,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Reproduce the single-group specs (S0..S5) exactly as build_model_specs does."""
    g = mod.get_dynamic_feature_groups(
        df,
        include_solution_features=False,
        include_unsupported_code_features=False,
    )
    g1_cat = list(g["g1_cat"])
    if not include_rna_family_ohe:
        g1_cat = [c for c in g1_cat if c != "rna_family"]

    return {
        "G0_af3": (list(g["g0_num"]), list(g["g0_cat"])),
        "G1_evolution": (list(g["g1_num"]), g1_cat),
        "G2_complexity": (list(g["g2_num"]), list(g["g2_cat"])),
        "G3_rna_intrinsic": (list(g["g3_num"]), list(g["g3_cat"])),
        "G4_interface": (list(g["g4_num"]), list(g["g4_cat"])),
        "G5_experimental": (list(g["g5_num"]), list(g["g5_cat"])),
    }


# ---------------------------------------------------------------------------
# Harmonized pipeline + filtering (mirrors the analysis exactly)
# ---------------------------------------------------------------------------
def filter_cols(df: pd.DataFrame, num: List[str], cat: List[str], args) -> Tuple[List[str], List[str]]:
    num = [c for c in num if c in df.columns]
    cat = [c for c in cat if c in df.columns]

    def keep_num(c):
        s = df[c]
        n_obs = int(s.notna().sum())
        if n_obs < args.min_non_na_feature:
            return False
        if float(s.isna().mean()) > args.max_missing_frac_feature:
            return False
        if s.nunique(dropna=True) <= 1:
            return False
        return True

    def keep_cat(c):
        s = df[c]
        if not s.notna().any():
            return False
        if float(s.isna().mean()) > args.max_missing_frac_feature:
            return False
        nun = s.nunique(dropna=True)
        if nun <= 1 or nun > args.max_cat_levels:
            return False
        return True

    return [c for c in num if keep_num(c)], [c for c in cat if keep_cat(c)]


def make_pipeline(num: List[str], cat: List[str], args) -> Pipeline:
    transformers = []
    if num:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            num,
        ))
    if cat:
        oh = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=args.min_category_count,
            sparse_output=False,
        )
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", oh)]),
            cat,
        ))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("var", VarianceThreshold(threshold=args.variance_threshold)),
        ("model", Ridge(alpha=args.ridge_alpha, random_state=args.seed)),
    ])


def make_splits(df: pd.DataFrame, groups: np.ndarray, use_group_cv: bool, cv: int, seed: int):
    if use_group_cv:
        n_groups = len(pd.unique(groups))
        eff = min(cv, n_groups)
        if eff < 2:
            return None, 0
        return list(GroupKFold(n_splits=eff).split(df, groups=groups)), eff
    eff = min(cv, len(df))
    if eff < 2:
        return None, 0
    return list(KFold(n_splits=eff, shuffle=True, random_state=seed).split(df)), eff


def oof_predictions(df: pd.DataFrame, y: pd.Series, num: List[str], cat: List[str],
                    splits, args) -> np.ndarray:
    """Pooled out-of-fold predictions. Empty feature set -> per-fold train mean."""
    oof = np.full(len(df), np.nan, dtype=float)
    yv = y.astype(float).values
    use_features = bool(num or cat)
    X = df[num + cat].copy() if use_features else None
    for tr, te in splits:
        if not use_features:
            oof[te] = float(np.nanmean(yv[tr]))
            continue
        pipe = make_pipeline(num, cat, args)
        pipe.fit(X.iloc[tr], yv[tr])
        oof[te] = pipe.predict(X.iloc[te]).astype(float)
    return oof


def coalition_r2(df, y, group_specs, members: Tuple[str, ...], splits, args) -> float:
    num, cat = [], []
    for m in members:
        n, c = group_specs[m]
        num += n
        cat += c
    num = list(dict.fromkeys(num))
    cat = list(dict.fromkeys(cat))
    num, cat = filter_cols(df, num, cat, args)
    oof = oof_predictions(df, y, num, cat, splits, args)
    yv = y.astype(float).values
    mask = np.isfinite(oof) & pd.notna(yv)
    if mask.sum() < 2 or len(np.unique(yv[mask])) < 2:
        return float("nan")
    return float(r2_score(yv[mask], oof[mask]))


# ---------------------------------------------------------------------------
# Shapley decomposition of R^2
# ---------------------------------------------------------------------------
def shapley_r2(df, y, group_specs, group_keys: List[str], splits, args
               ) -> Tuple[Dict[str, float], Dict[frozenset, float]]:
    n = len(group_keys)
    # value of every coalition
    v: Dict[frozenset, float] = {}
    for k in range(0, n + 1):
        for combo in combinations(group_keys, k):
            key = frozenset(combo)
            r2 = coalition_r2(df, y, group_specs, tuple(combo), splits, args)
            v[key] = 0.0 if (k == 0 or math.isnan(r2)) else r2
    # Shapley value per group
    phi: Dict[str, float] = {g: 0.0 for g in group_keys}
    full_set = list(group_keys)
    for g in group_keys:
        others = [x for x in full_set if x != g]
        total = 0.0
        for k in range(0, len(others) + 1):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for combo in combinations(others, k):
                S = frozenset(combo)
                Sg = frozenset(combo + (g,))
                total += w * (v[Sg] - v[S])
        phi[g] = total
    return phi, v


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------
def prepare(df0: pd.DataFrame, args, group_specs, splits_factory) -> Tuple[pd.DataFrame, pd.Series, List[str], np.ndarray, bool]:
    y = pd.to_numeric(df0[args.ycol], errors="coerce")
    keep = y.notna()
    df = df0.loc[keep].reset_index(drop=True).copy()
    y = y.loc[keep].reset_index(drop=True).astype(float)

    groups, use_group_cv = splits_factory(df)
    splits, eff = make_splits(df, groups, use_group_cv, args.cv, args.seed)
    if splits is None:
        raise ValueError("Could not build CV splits (too few groups/rows).")

    if args.target_mode == "direct":
        return df, y, GROUP_ORDER, groups, use_group_cv

    if args.target_mode == "af3_error":
        # AF3-only OOF residual, then partition residual over G1..G5
        num0, cat0 = group_specs["G0_af3"]
        num0, cat0 = filter_cols(df, num0, cat0, args)
        af3_oof = oof_predictions(df, y, num0, cat0, splits, args)
        resid = y.values - af3_oof
        resid = pd.Series(resid, index=df.index, dtype=float)
        # rows where AF3 OOF is undefined cannot contribute
        ok = np.isfinite(resid.values)
        df = df.loc[ok].reset_index(drop=True).copy()
        resid = resid.loc[ok].reset_index(drop=True)
        groups2, use_group_cv2 = splits_factory(df)
        return df, resid, [g for g in GROUP_ORDER if g != "G0_af3"], groups2, use_group_cv2

    raise ValueError(f"unknown target_mode {args.target_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ycol", required=True)
    ap.add_argument("--registry_module", required=True,
                    help="Path to the analysis .py whose fixed feature registry to reuse")
    ap.add_argument("--group_by", default="pdb_id")
    ap.add_argument("--target_mode", choices=["direct", "af3_error"], default="direct")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--variance_threshold", type=float, default=1e-8)
    ap.add_argument("--min_category_count", type=int, default=10)
    ap.add_argument("--min_non_na_feature", type=int, default=10)
    ap.add_argument("--max_cat_levels", type=int, default=25)
    ap.add_argument("--max_missing_frac_feature", type=float, default=0.80)
    ap.add_argument("--min_rows", type=int, default=50)
    ap.add_argument("--include_rna_family_ohe", action="store_true")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--outdir", default="variance_partitioning")
    args = ap.parse_args()

    mod = load_registry_module(args.registry_module)
    df0 = pd.read_csv(args.csv)
    if args.ycol not in df0.columns:
        raise ValueError(f"ycol '{args.ycol}' not in CSV")

    group_specs = build_group_specs(mod, df0, args.include_rna_family_ohe)

    def splits_factory(df: pd.DataFrame):
        if args.group_by in df.columns:
            g = df[args.group_by].astype(str).fillna("NA").values
            if len(pd.unique(g)) >= 2:
                return g, True
        return np.arange(len(df)), False

    df, y, group_keys, groups, use_group_cv = prepare(df0, args, group_specs, splits_factory)
    if len(df) < args.min_rows:
        raise ValueError(f"too few rows after target filtering: n={len(df)} < {args.min_rows}")

    splits, eff = make_splits(df, groups, use_group_cv, args.cv, args.seed)

    # report usable feature counts per group (after filtering on the analysis df)
    group_nfeat = {}
    for gk in group_keys:
        n, c = group_specs[gk]
        nf, cf = filter_cols(df, n, c, args)
        group_nfeat[gk] = len(nf) + len(cf)

    phi, v = shapley_r2(df, y, group_specs, group_keys, splits, args)

    full_r2 = v[frozenset(group_keys)]
    total_attrib = sum(phi.values())  # == full_r2 - v(empty) == full_r2 (v(empty)=0)

    # increment over AF3 baseline (only meaningful in direct mode where G0 present)
    inc_over_af3 = {}
    if "G0_af3" in group_keys:
        af3_only = v[frozenset(["G0_af3"])]
        for gk in group_keys:
            if gk == "G0_af3":
                inc_over_af3[gk] = af3_only
            else:
                inc_over_af3[gk] = v[frozenset(["G0_af3", gk])] - af3_only
    else:
        for gk in group_keys:
            inc_over_af3[gk] = float("nan")

    rows = []
    for gk in group_keys:
        rows.append({
            "group": gk,
            "label": GROUP_LABELS.get(gk, gk),
            "n_features_used": group_nfeat[gk],
            "shapley_r2": phi[gk],
            "shapley_share_pct": 100.0 * phi[gk] / total_attrib if total_attrib else float("nan"),
            "increment_over_af3_r2": inc_over_af3[gk],
        })
    res = pd.DataFrame(rows).sort_values("shapley_r2", ascending=False).reset_index(drop=True)

    out_root = Path(args.outdir) / args.dataset / args.subset / f"{args.ycol}__{args.target_mode}"
    out_root.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_root / "shapley_r2.csv", index=False)

    coalition_rows = [{"members": "+".join(sorted(s)) if s else "(empty)", "size": len(s), "oof_r2": v[s]}
                      for s in sorted(v.keys(), key=lambda x: (len(x), sorted(x)))]
    pd.DataFrame(coalition_rows).to_csv(out_root / "coalition_r2.csv", index=False)

    cfg = vars(args).copy()
    cfg.update({
        "n_rows": int(len(df)),
        "n_groups_cv": int(len(pd.unique(groups))) if use_group_cv else None,
        "cv_effective": int(eff),
        "use_group_cv": bool(use_group_cv),
        "group_keys": group_keys,
        "full_model_oof_r2": full_r2,
        "empty_value": 0.0,
        "group_feature_lists": {gk: {"num": group_specs[gk][0], "cat": group_specs[gk][1]} for gk in group_keys},
    })
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # barplot
    plt.figure(figsize=(8, 4.6), constrained_layout=True)
    order = res.iloc[::-1]
    plt.barh(order["label"], order["shapley_r2"])
    plt.axvline(0, color="k", lw=0.8)
    ttl = f"{args.dataset} | {args.subset} | {args.ycol} ({args.target_mode})\nShapley-R2 attribution | full OOF R2={full_r2:.3f} | n={len(df)}"
    plt.title(ttl)
    plt.xlabel("Shapley contribution to out-of-fold R²")
    plt.savefig(out_root / "shapley_r2.png", dpi=200)
    plt.close()

    # console summary
    print("=" * 78)
    print(f"{args.dataset} | {args.subset} | ycol={args.ycol} | mode={args.target_mode}")
    print(f"n_rows={len(df)}  groups={cfg['n_groups_cv']}  cv={eff}  full_OOF_R2={full_r2:.3f}")
    if "G0_af3" in group_keys:
        print(f"AF3-only OOF R2 = {v[frozenset(['G0_af3'])]:.3f}")
    print("-" * 78)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)


if __name__ == "__main__":
    main()
