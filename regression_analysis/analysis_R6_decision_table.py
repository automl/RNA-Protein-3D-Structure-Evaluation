#!/usr/bin/env python
"""R6 Layer-1 decision-accuracy table (regenerated on the new data).

Conditional means (no modelling) of per-chain RNA TM-score at the branches of the
decision framework: by system class, and by RNA chain length within class. Reports
combined + pre/post-cutoff columns with n, SD and 95% CI, and flags thin bins.
A post-cutoff length view is added so the length gradient is not read off
training-familiar long RNAs.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "paper_new_analyses/R6_decision_table"
os.makedirs(OUT, exist_ok=True)
TM = "assigned_rna_tm"
L = "pred_rna_chain_length"
LEN_BINS = [0, 40, 100, 200, np.inf]
LEN_LAB = ["<=40", "41-100", "101-200", ">200"]
THIN = 15

SRC = {
    "RNA-only": "csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv",
    "RNA-protein 1v1": "csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv",
    "RNA-protein multichain": "csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv",
}


def summ(x):
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    else:
        h = np.nan
    return dict(n=n, mean=round(m, 3), sd=round(sd, 3),
                lo=round(m - h, 3) if np.isfinite(h) else np.nan,
                hi=round(m + h, 3) if np.isfinite(h) else np.nan)


def load(path):
    d = pd.read_csv(path, low_memory=False).dropna(subset=[TM]).copy()
    d["era"] = d["is_post_cutoff"].map({0: "pre", 1: "post"})
    d["lenbin"] = pd.cut(d[L], LEN_BINS, labels=LEN_LAB)
    return d

frames = {k: load(v) for k, v in SRC.items()}

# ---- by system class ----
class_rows = []
for cls, d in frames.items():
    row = {"branch": cls}
    for era, sub in [("combined", d), ("pre", d[d.era == "pre"]), ("post", d[d.era == "post"])]:
        s = summ(sub[TM])
        row[f"{era}_mean"] = s["mean"]; row[f"{era}_n"] = s["n"]
        row[f"{era}_ci"] = f"[{s['lo']},{s['hi']}]" if s["n"] > 1 else "n/a"
    class_rows.append(row)
class_tab = pd.DataFrame(class_rows)

# ---- by length within class (combined AND post-cutoff view) ----
len_rows = []
for cls, d in frames.items():
    for lb in LEN_LAB:
        sub = d[d.lenbin == lb]
        if len(sub) == 0:
            continue
        c = summ(sub[TM]); p = summ(sub[sub.era == "post"][TM])
        len_rows.append(dict(branch=cls, length=lb,
                             combined_mean=c["mean"], combined_n=c["n"],
                             combined_ci=f"[{c['lo']},{c['hi']}]" if c["n"] > 1 else "n/a",
                             post_mean=p["mean"], post_n=p["n"],
                             thin=("*" if (c["n"] < THIN or p["n"] < THIN) else "")))
len_tab = pd.DataFrame(len_rows)

pd.set_option("display.width", 200)
print("=== Decision-accuracy by SYSTEM CLASS (per-chain RNA TM) ===")
print(class_tab.to_string(index=False))
print(f"\n=== By RNA LENGTH within class (thin '*' means combined or post n<{THIN}) ===")
print(len_tab.to_string(index=False))

# internal consistency check vs family-based estimate (~0.59 in the paper)
post_ro = frames["RNA-only"]
print("\nInternal check: post-cutoff RNA-only mean RNA TM =",
      round(post_ro[post_ro.era == "post"][TM].mean(), 3),
      f"(n={int((post_ro.era=='post').sum())}) -- paper family-based estimate ~0.59")

class_tab.to_csv(f"{OUT}/decision_accuracy_by_class.csv", index=False)
len_tab.to_csv(f"{OUT}/decision_accuracy_by_length.csv", index=False)
print(f"\nSaved to {OUT}/")
