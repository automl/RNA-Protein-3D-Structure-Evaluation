#!/usr/bin/env python
"""Automated, data-driven feature selection for the AF3 regression, replacing the
human-curated pruned lists.

Method (locked with the user):
  * Candidate pool = the union of the six feature groups G0..G5 defined in the
    analysis module (already leakage-free: the groups never contain accuracy-derived
    columns such as INF/DI/match scores). We additionally exclude, defensively, any
    known accuracy-adjacent column that might slip in.
  * Stability selection: over B grouped resamples (subsample 50% of the pdb_id
    GROUPS each time, so chains from one structure stay together), fit an ElasticNet
    at a CV-chosen (alpha, l1_ratio) and record which features receive a non-zero
    coefficient. selection_frequency[f] = fraction of resamples selecting f.
    A categorical feature counts as selected if ANY of its one-hot levels is non-zero.
  * Selected set = features with selection_frequency >= pi_thr (default 0.70). Each
    selected feature keeps its G0..G5 label, so the downstream grouped Shapley is
    unchanged (a group may end empty -> 0 share).
  * Reporting (HYBRID rigor): the pooled selection defines the fixed feature set
    (SI table + used for the ridge fit + Shapley). We ADDITIONALLY report a
    nested-CV R^2 (selection redone inside each training fold, scored on the held-out
    fold) as the honest, leakage-free full-model performance number, alongside the
    in-sample-selected pooled R^2.

Framing: in-distribution / associational, matching the rest of the analysis.

Usage:
  python stability_selection.py --module single_chain_level_regression_analysis_final.py \
    --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
    --ycol assigned_rna_tm --group_by pdb_id --outdir feature_selection_stability \
    --dataset multi_per_chain --subset multi
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# accuracy-derived columns that must never enter selection (belt-and-suspenders;
# the groups already exclude them)
LEAKY = {
    "RNA_INF", "RNA_DI", "RNA_f1_score", "RNA_precision", "RNA_recall", "RNA_mcc",
    "RNA_wl", "RNAmatch_score", "AAmatch_score", "RNAmotif_score", "RNAMotif_inExp",
    "RNAMotif_isDuplex",
}


def load_registry_module(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("af3_registry_mod", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_group_specs(mod, df):
    g = mod.get_dynamic_feature_groups(
        df, include_solution_features=False, include_unsupported_code_features=False)
    return {
        "G0_af3": (list(g["g0_num"]), list(g["g0_cat"])),
        "G1_evolution": (list(g["g1_num"]), list(g["g1_cat"])),
        "G2_complexity": (list(g["g2_num"]), list(g["g2_cat"])),
        "G3_rna_intrinsic": (list(g["g3_num"]), list(g["g3_cat"])),
        "G4_interface": (list(g["g4_num"]), list(g["g4_cat"])),
        "G5_experimental": (list(g["g5_num"]), list(g["g5_cat"])),
    }


def filter_cols(df, num, cat, a):
    num = [c for c in num if c in df.columns and c not in LEAKY]
    cat = [c for c in cat if c in df.columns and c not in LEAKY]
    def keep_num(c):
        s = df[c]
        return (int(s.notna().sum()) >= a.min_non_na_feature
                and float(s.isna().mean()) <= a.max_missing_frac_feature
                and s.nunique(dropna=True) > 1)
    def keep_cat(c):
        s = df[c]
        nun = s.nunique(dropna=True)
        return (s.notna().any() and float(s.isna().mean()) <= a.max_missing_frac_feature
                and 1 < nun <= a.max_cat_levels)
    return [c for c in num if keep_num(c)], [c for c in cat if keep_cat(c)]


def make_preprocessor(num, cat, a):
    t = []
    if num:
        t.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                   ("s", StandardScaler())]), num))
    if cat:
        t.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                   ("o", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                       min_frequency=a.min_category_count,
                                                       sparse_output=False))]), cat))
    return ColumnTransformer(t, remainder="drop")


def transformed_to_original(names, num, cat):
    """Map a fitted ColumnTransformer's output column names back to the original
    feature name (numeric 1:1; one-hot levels collapse to their parent feature)."""
    out = []
    catset = sorted(cat, key=len, reverse=True)
    for nm in names:
        if nm.startswith("num__"):
            out.append(nm[len("num__"):])
        elif nm.startswith("cat__"):
            rest = nm[len("cat__"):]
            parent = next((c for c in catset if rest.startswith(c)), rest)
            out.append(parent)
        else:
            out.append(nm)
    return out


def make_groups(df, group_by):
    if group_by and group_by != "none" and group_by in df.columns:
        return df[group_by].astype(str).values, True
    return np.arange(len(df)), False


def group_kfold_splits(df, groups, use_group, cv, seed):
    if use_group:
        eff = min(cv, len(pd.unique(groups)))
        if eff < 2:
            return None
        return list(GroupKFold(n_splits=eff).split(df, groups=groups))
    eff = min(cv, len(df))
    return list(KFold(n_splits=eff, shuffle=True, random_state=seed).split(df)) if eff >= 2 else None


def choose_enet(Xt, y, groups, use_group, cv, seed):
    """CV-choose (alpha, l1_ratio) once on the full transformed matrix."""
    splits = group_kfold_splits(pd.DataFrame(Xt), groups, use_group, cv, seed)
    cvsplits = splits if splits else 5
    m = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95], n_alphas=50, cv=cvsplits,
                     random_state=seed, max_iter=5000)
    m.fit(Xt, y)
    return float(m.alpha_), float(m.l1_ratio_)


def stability_select(Xt, y, groups, use_group, feat_of_col, alpha, l1, n_boot, frac, seed):
    uniq = pd.unique(groups)
    rng = np.random.RandomState(seed)
    feats = sorted(set(feat_of_col))
    counts = {f: 0 for f in feats}
    n_take = max(2, int(round(frac * len(uniq))))
    for b in range(n_boot):
        take = set(rng.choice(uniq, size=n_take, replace=False))
        mask = np.array([g in take for g in groups])
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=seed + b)
        en.fit(Xt[mask], y[mask])
        nz = np.abs(en.coef_) > 1e-8
        sel = set(feat_of_col[i] for i in np.where(nz)[0])
        for f in sel:
            counts[f] += 1
    return {f: counts[f] / n_boot for f in feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ycol", required=True)
    ap.add_argument("--group_by", default="pdb_id")
    ap.add_argument("--outdir", default="feature_selection_stability")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--subset", default="subset")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--n_boot", type=int, default=100)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--pi_thr", type=float, default=0.70)
    ap.add_argument("--min_non_na_feature", type=int, default=10)
    ap.add_argument("--max_missing_frac_feature", type=float, default=0.8)
    ap.add_argument("--max_cat_levels", type=int, default=25)
    ap.add_argument("--min_category_count", type=int, default=10)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    y_all = pd.to_numeric(df[a.ycol], errors="coerce")
    keep = y_all.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y_all.loc[keep].astype(float).values
    if len(df) < 40 or len(np.unique(y)) < 2:
        print(f"[skip] {a.dataset}/{a.subset}: too few rows/targets ({len(df)})")
        return

    mod = load_registry_module(a.module)
    specs = build_group_specs(mod, df)
    num, cat, group_of = [], [], {}
    for gname, (gn, gc) in specs.items():
        fn, fc = filter_cols(df, gn, gc, a)
        for c in fn + fc:
            group_of[c] = gname
        num += fn; cat += fc
    num = list(dict.fromkeys(num)); cat = list(dict.fromkeys(cat))
    if not num and not cat:
        print(f"[skip] {a.dataset}/{a.subset}: no candidate features"); return

    groups, use_group = make_groups(df, a.group_by)

    # pooled preprocessing (unsupervised -> fit on full data, matches the paper)
    pre = make_preprocessor(num, cat, a)
    Xt = pre.fit_transform(df[num + cat])
    tnames = list(pre.get_feature_names_out())
    feat_of_col = transformed_to_original(tnames, num, cat)

    alpha, l1 = choose_enet(Xt, y, groups, use_group, a.cv, a.seed)
    freq = stability_select(Xt, y, groups, use_group, feat_of_col, alpha, l1,
                            a.n_boot, a.frac, a.seed)
    selected = sorted([f for f, v in freq.items() if v >= a.pi_thr])

    # ---- performance: pooled-selected ridge R^2 vs nested-CV R^2 vs full R^2 ----
    def ridge_oof(numf, catf):
        numf = [c for c in numf if c in df.columns]; catf = [c for c in catf if c in df.columns]
        if not numf and not catf:
            return float("nan")
        splits = group_kfold_splits(df, groups, use_group, a.cv, a.seed)
        if not splits:
            return float("nan")
        oof = np.full(len(df), np.nan)
        for tr, te in splits:
            p = Pipeline([("pre", make_preprocessor(numf, catf, a)),
                          ("m", Ridge(alpha=a.ridge_alpha, random_state=a.seed))])
            p.fit(df[numf + catf].iloc[tr], y[tr])
            oof[te] = p.predict(df[numf + catf].iloc[te])
        m = np.isfinite(oof)
        return float(r2_score(y[m], oof[m])) if m.sum() > 2 else float("nan")

    sel_num = [c for c in selected if c in num]; sel_cat = [c for c in selected if c in cat]
    r2_full = ridge_oof(num, cat)
    r2_selected_pooled = ridge_oof(sel_num, sel_cat)

    # nested: reselect inside each training fold, ridge on selected, score test fold
    splits = group_kfold_splits(df, groups, use_group, a.cv, a.seed)
    oof = np.full(len(df), np.nan)
    if splits:
        for tr, te in splits:
            dtr = df.iloc[tr].reset_index(drop=True); ytr = y[tr]
            gtr, ug = make_groups(dtr, a.group_by)
            pre_tr = make_preprocessor(num, cat, a)
            Xtr = pre_tr.fit_transform(dtr[num + cat])
            fo = transformed_to_original(list(pre_tr.get_feature_names_out()), num, cat)
            try:
                al, ll = choose_enet(Xtr, ytr, gtr, ug, min(a.cv, 4), a.seed)
                fq = stability_select(Xtr, ytr, gtr, ug, fo, al, ll,
                                      max(30, a.n_boot // 2), a.frac, a.seed)
                sel_f = [f for f, v in fq.items() if v >= a.pi_thr] or [f for f, v in fq.items() if v >= 0.5]
            except Exception:
                sel_f = num + cat
            sn = [c for c in sel_f if c in num]; sc = [c for c in sel_f if c in cat]
            if not sn and not sc:
                sn, sc = num, cat
            p = Pipeline([("pre", make_preprocessor(sn, sc, a)),
                          ("m", Ridge(alpha=a.ridge_alpha, random_state=a.seed))])
            p.fit(dtr[sn + sc], ytr)
            oof[te] = p.predict(df[sn + sc].iloc[te])
    m = np.isfinite(oof)
    r2_nested = float(r2_score(y[m], oof[m])) if m.sum() > 2 else float("nan")

    # ---- write outputs ----
    outd = Path(a.outdir) / f"{a.dataset}" / f"{a.subset}"
    outd.mkdir(parents=True, exist_ok=True)
    rows = [{"feature": f, "group": group_of.get(f, "?"),
             "selection_frequency": round(freq[f], 3),
             "selected": int(freq[f] >= a.pi_thr)} for f in sorted(freq, key=lambda x: -freq[x])]
    pd.DataFrame(rows).to_csv(outd / "selection_frequency.csv", index=False)
    summary = {
        "dataset": a.dataset, "subset": a.subset, "ycol": a.ycol, "n_rows": int(len(df)),
        "n_candidates": len(num) + len(cat), "n_selected": len(selected),
        "enet_alpha": round(alpha, 5), "enet_l1_ratio": l1, "pi_thr": a.pi_thr, "n_boot": a.n_boot,
        "r2_full_universe": round(r2_full, 4),
        "r2_selected_pooled": round(r2_selected_pooled, 4),
        "r2_selected_nested": round(r2_nested, 4),
        "selected_by_group": {g: sorted([f for f in selected if group_of.get(f) == g])
                              for g in specs},
    }
    (outd / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{a.dataset}/{a.subset}] cand={len(num)+len(cat)} selected={len(selected)} "
          f"| R2 full={r2_full:.3f} sel_pooled={r2_selected_pooled:.3f} sel_nested={r2_nested:.3f} "
          f"| enet(a={alpha:.4f},l1={l1})")
    print("  selected:", ", ".join(selected))


if __name__ == "__main__":
    main()
