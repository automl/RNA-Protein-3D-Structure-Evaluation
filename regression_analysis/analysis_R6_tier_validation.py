#!/usr/bin/env python
"""R6 Layer-2 validation: do the reliability tiers separate held-out accuracy?

The decision tree's per-structure reliability tiers are built from FIXED AF3
confidence cutpoints (pre-specified, not fitted), so evaluating them against actual
accuracy is inherently out-of-sample. We report:
  - RNA-fold tier from RNA pLDDT (good >=0.775, bad <0.55, else mid) vs per-chain
    RNA TM-score;
  - Interface tier from RNA ipTM + chain-pair PAE (good: ipTM_r>=0.5 & PAE<=8;
    bad: ipTM_r<0.25 | PAE>15; else mid) vs DockQ.
For each: mean accuracy per tier (monotone separation = the tier "earns its keep"),
and the out-of-fold discrimination (ROC-AUC of the underlying confidence predicting
an 'accurate' outcome) under GroupKFold by pdb_id.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper_new_analyses/R6_tier_validation"
os.makedirs(OUT, exist_ok=True)
CX = ["csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv",
      "csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv"]
RO = "csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv"


def ci(x):
    x = pd.Series(x).dropna().astype(float); n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (round(x.mean() - h, 3), round(x.mean() + h, 3))


def grouped_auc(score, y, groups, k=5):
    m = score.notna() & y.notna() & groups.notna()
    score, y, groups = score[m].values, y[m].astype(int).values, groups[m].values
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    aucs = []
    for _, te in GroupKFold(n_splits=k).split(score, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], score[te]))
    return (round(np.mean(aucs), 3), round(np.std(aucs), 3)) if aucs else (np.nan, np.nan)


def tier_report(df, tier, acc, score, acc_thresh, name):
    print(f"\n=== {name}: tier vs accuracy (n={len(df)}) ===")
    for t in ["bad", "mid", "good"]:
        s = df[df[tier] == t][acc]
        lo, hi = ci(s)
        print(f"  {t:4s}: n={s.notna().sum():4d}  mean {acc}={s.mean():.3f}  CI[{lo},{hi}]")
    rho, p = stats.spearmanr(df[tier].map({"bad": 0, "mid": 1, "good": 2}), df[acc], nan_policy="omit")
    y = (df[acc] > acc_thresh).astype(float)
    auc_m, auc_s = grouped_auc(df[score], y, df["pdb_id"])
    print(f"  Spearman(tier, {acc})={rho:.3f} (p={p:.1e});  grouped-CV ROC-AUC of {score} "
          f"predicting {acc}>{acc_thresh}: {auc_m} +/- {auc_s}")
    return rho, auc_m


# ---------- RNA-fold tier ----------
parts = []
for p in CX + [RO]:
    d = pd.read_csv(p, low_memory=False)
    d = d[["pdb_id", "af3_rna_pLDDT_avg", "assigned_rna_tm"]].dropna(subset=["af3_rna_pLDDT_avg", "assigned_rna_tm"])
    parts.append(d)
fold = pd.concat(parts, ignore_index=True)
fold["fold_tier"] = np.where(fold.af3_rna_pLDDT_avg >= 0.775, "good",
                     np.where(fold.af3_rna_pLDDT_avg < 0.55, "bad", "mid"))
tier_report(fold, "fold_tier", "assigned_rna_tm", "af3_rna_pLDDT_avg", 0.45, "RNA-fold tier (pLDDT)")

# ---------- Interface tier ----------
iparts = []
for p in CX:
    d = pd.read_csv(p, low_memory=False)
    keep = ["pdb_id", "af3_rna_ipTM_list_value", "af3_chain_pair_pae_min", "Dockq"]
    d = d[[c for c in keep if c in d.columns]].dropna(subset=["af3_rna_ipTM_list_value", "af3_chain_pair_pae_min", "Dockq"])
    iparts.append(d)
itf = pd.concat(iparts, ignore_index=True)
ip, pae = itf.af3_rna_ipTM_list_value, itf.af3_chain_pair_pae_min
itf["itf_tier"] = np.where((ip >= 0.5) & (pae <= 8), "good",
                   np.where((ip < 0.25) | (pae > 15), "bad", "mid"))
itf["itf_score"] = ip - pae / 30.0  # monotone combined score for AUC
tier_report(itf, "itf_tier", "Dockq", "itf_score", 0.23, "Interface tier (ipTM_r + PAE)")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
order = ["bad", "mid", "good"]; col = {"bad": "#E15759", "mid": "#F0C000", "good": "#59A14F"}
for a, (df, tier, acc, thr, ttl) in zip(ax, [
        (fold, "fold_tier", "assigned_rna_tm", 0.45, "A  RNA-fold tier vs RNA TM"),
        (itf, "itf_tier", "Dockq", 0.23, "B  Interface tier vs DockQ")]):
    data = [df[df[tier] == t][acc].dropna().values for t in order]
    bp = a.boxplot(data, labels=[f"{t}\nn={len(d)}" for t, d in zip(order, data)],
                   patch_artist=True, showfliers=False, widths=0.6)
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(col[t]); patch.set_alpha(0.65)
    for i, d in enumerate(data, 1):
        if len(d):
            a.plot(i, np.mean(d), "D", color="black", ms=4)
    a.axhline(thr, ls="--", color="0.4", lw=1)
    a.set_ylabel(acc); a.set_title(ttl); a.set_ylim(0, 1)
fig.suptitle("Reliability tiers separate held-out accuracy (dashed = acceptability threshold)", y=1.02, fontsize=10.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/R6_tier_validation.{ext}", dpi=300, bbox_inches="tight")
print(f"\nSaved figure + summary to {OUT}/")
