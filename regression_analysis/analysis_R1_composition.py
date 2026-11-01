#!/usr/bin/env python
"""R1 composition-confound control (SI figure).

Question: is the post-cutoff RNA-accuracy drop just a composition artifact
(post-cutoff has more cryo-EM and bigger/floppier RNAs, which are intrinsically
harder), or does it survive when method and RNA length are held fixed?

We stratify the per-chain RNA TM-score by deposition era (pre = b2021 training-era,
post = a2021 novel) WITHIN experimental method (X-ray vs cryo-EM) and WITHIN RNA
length bins. If the pre->post drop persists in every stratum, the gap is not merely
compositional. Primary set: RNA chains of protein-RNA complexes; RNA-only reported
as a secondary check.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from figsave import save_all_formats

# Repo root (AF3InterfaceEval) on sys.path so `plots.publication_figure` imports
# when this script is run from regression_analysis/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from plots.publication_figure import onpage_to_gen

OUT = "paper_new_analyses/R1_composition"
os.makedirs(OUT, exist_ok=True)
TM = "assigned_rna_tm"
L = "pred_rna_chain_length"
LEN_BINS = [0, 40, 100, 200, np.inf]
LEN_LAB = ["<=40", "41-100", "101-200", ">200"]


def cohend(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan  # pre - post


def mwu_p(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        return np.nan


def load(path, label):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[TM]).copy()
    df["method"] = np.where(df.get("is_xray", 0) == 1, "X-ray",
                     np.where(df.get("is_em", 0) == 1, "cryo-EM", "other"))
    df["era"] = df["is_post_cutoff"].map({0: "pre", 1: "post"})
    df["dataset"] = label
    return df


def gap_table(df, strat_col, strat_vals):
    rows = []
    for v in strat_vals:
        sub = df if v == "overall" else df[df[strat_col] == v]
        pre = sub[sub.era == "pre"][TM]
        post = sub[sub.era == "post"][TM]
        rows.append(dict(stratum=v, n_pre=int(pre.notna().sum()), n_post=int(post.notna().sum()),
                         mean_pre=round(pre.mean(), 3), mean_post=round(post.mean(), 3),
                         drop=round(pre.mean() - post.mean(), 3),
                         cohen_d=round(cohend(pre, post), 3), mwu_p=mwu_p(pre, post)))
    return pd.DataFrame(rows)


cx = load("csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv", "protein-RNA")
ro = load("csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv", "RNA-only")
for d in (cx, ro):
    d["lenbin"] = pd.cut(d[L], LEN_BINS, labels=LEN_LAB)

print("=== protein-RNA per-chain RNA TM: pre vs post, by METHOD ===")
by_method = gap_table(cx, "method", ["overall", "X-ray", "cryo-EM"])
print(by_method.to_string(index=False))
print("\n=== protein-RNA per-chain RNA TM: pre vs post, by LENGTH bin ===")
by_len = gap_table(cx, "lenbin", LEN_LAB)
print(by_len.to_string(index=False))
print("\n=== RNA-only per-chain RNA TM: pre vs post, by METHOD (secondary) ===")
ro_method = gap_table(ro, "method", ["overall", "X-ray", "cryo-EM"])
print(ro_method.to_string(index=False))

by_method.to_csv(f"{OUT}/rna_tm_pre_post_by_method_complexes.csv", index=False)
by_len.to_csv(f"{OUT}/rna_tm_pre_post_by_length_complexes.csv", index=False)
ro_method.to_csv(f"{OUT}/rna_tm_pre_post_by_method_rnaonly.csv", index=False)

# ---- figure: A) by method  B) by length bin (protein-RNA complexes) ----
FIGW = 9.5
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": onpage_to_gen(7, FIGW),
    "axes.labelsize": onpage_to_gen(7.5, FIGW),
    "axes.titlesize": onpage_to_gen(7.5, FIGW),
    "xtick.labelsize": onpage_to_gen(7, FIGW),
    "ytick.labelsize": onpage_to_gen(7, FIGW),
    "legend.fontsize": onpage_to_gen(7, FIGW),
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
C = {"pre": "#4C78A8", "post": "#F28E2B"}
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 3.8))


def grouped(axi, table, xlab, title, letter):
    labels = table["stratum"].tolist()
    x = np.arange(len(labels)); w = 0.38
    for i, era in enumerate(["pre", "post"]):
        m = table[f"mean_{era}"].values
        n = table[f"n_{era}"].values
        axi.bar(x + (i - 0.5) * w, m, w, color=C[era], label=f"{era}-cutoff", edgecolor="white")
        for xi, mi, ni in zip(x + (i - 0.5) * w, m, n):
            if np.isfinite(mi):
                axi.text(xi, mi + 0.01, f"n={ni}", ha="center", va="bottom",
                         fontsize=onpage_to_gen(6.5, FIGW), rotation=0)
    for xi, (_, r) in zip(x, table.iterrows()):
        if np.isfinite(r["cohen_d"]):
            axi.text(xi, 0.02, f"d={r['cohen_d']:.2f}", ha="center", va="bottom",
                     fontsize=onpage_to_gen(8, FIGW), color="black", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))
    axi.set_xticks(x); axi.set_xticklabels(labels)
    axi.set_ylabel("per-chain RNA TM-score"); axi.set_xlabel(xlab)
    axi.set_ylim(0, max(0.8, table[["mean_pre", "mean_post"]].max().max() + 0.12))
    axi.legend(frameon=False, fontsize=onpage_to_gen(7, FIGW))
    axi.text(-0.12, 1.04, letter, transform=axi.transAxes, fontsize=onpage_to_gen(8, FIGW),
             fontweight="bold", va="bottom", ha="right")


grouped(ax[0], by_method, "experimental method", "RNA drop within method", "A")
grouped(ax[1], by_len, "RNA chain length (nt)", "RNA drop within length", "B")
fig.tight_layout()
save_all_formats(fig, f"{OUT}/R1_composition_control", dpi=600)
print(f"\nSaved figure + CSVs to {OUT}/")
