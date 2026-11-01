#!/usr/bin/env python
"""postpre_family_comparison: pre/post-cutoff composition + confound-control figure (SI).

Six-panel companion to the generalization-gap result, matching Fig.~\\ref{fig:postpre_composition}:
whether the post-cutoff RNA-accuracy drop is a composition artefact (harder/bigger cryo-EM RNAs)
or survives when composition, family and experimental method are held fixed.

  A  RNA length distributions (log10 nt), pre vs post, per dataset.
  B  Cohen's d (pre - post) of RNA complexity descriptors, protein-RNA vs RNA-only.
  C  RNA family composition per dataset x era (families turn over between eras).
  D  Mean RNA accuracy pre vs post, matched by shared family and by fixed length band.
  E  X-ray vs cryo-EM share per dataset x era.
  F  RNA accuracy pre vs post WITHIN X-ray and WITHIN cryo-EM (protein-RNA).

Reproducible: read-only on the per-chain analysis CSVs (produced by preprocessing), deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from plots.publication_figure import onpage_to_gen           # noqa: E402
from figsave import save_all_formats                          # noqa: E402

OUT = "paper_new_analyses/postpre_composition"
os.makedirs(OUT, exist_ok=True)

TM, LDDT, L, FAM = "assigned_rna_tm", "assigned_rna_lddt", "pred_rna_chain_length", "rna_family"
ERA_C = {"pre": "#4C78A8", "post": "#F28E2B"}       # blue / orange
DS_C = {"protein-RNA": "#59A14F", "RNA-only": "#B279A2"}  # green / purple
FIGW = 15.5
fs = lambda pt: onpage_to_gen(pt, FIGW)


def cohend(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan  # pre - post


def msem(v):
    v = pd.Series(v, dtype=float).dropna()
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0, len(v))


def load(path, label):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=[TM]).copy()
    df["method"] = np.where(df.get("is_xray", 0) == 1, "X-ray",
                            np.where(df.get("is_em", 0) == 1, "cryo-EM", "other"))
    df["era"] = df["is_post_cutoff"].map({0: "pre", 1: "post"})
    df["dataset"] = label
    df["logL"] = np.log10(pd.to_numeric(df[L], errors="coerce").replace(0, np.nan))
    df["_fam"] = df.get(FAM, pd.Series(index=df.index, dtype=object)).fillna("unannotated").replace("", "unannotated")
    return df


cx = load("csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv", "protein-RNA")
ro = load("csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv", "RNA-only")
df = pd.concat([cx, ro], ignore_index=True)
for c in (TM, LDDT, "RNA_Shannon_Entropy_K2", "RNA_MFE_Value", "RNA_GC_Content",
          "RNA_GC_Skew", "RNA_AU_Skew", "RNA_SequenceComplexity"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": fs(7), "axes.labelsize": fs(7.5), "axes.titlesize": fs(7.5),
    "xtick.labelsize": fs(6.5), "ytick.labelsize": fs(7), "legend.fontsize": fs(6.8),
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

fig, axes = plt.subplots(2, 3, figsize=(FIGW, 8.8))
(axA, axB, axC), (axD, axE, axF) = axes
ERAS = ["pre", "post"]
DGRID = [("protein-RNA", "pre"), ("protein-RNA", "post"), ("RNA-only", "pre"), ("RNA-only", "post")]


# ---------- A: RNA length distributions ----------
posB, dataB, colB = [], [], []
for i, (ds, era) in enumerate(DGRID):
    x = df[(df.dataset == ds) & (df.era == era)]["logL"].dropna().values
    p = i + (0.6 if i >= 2 else 0.0)
    posB.append(p); dataB.append(x); colB.append(ERA_C[era])
parts = axA.violinplot(dataB, positions=posB, widths=0.8, showmedians=True, showextrema=False)
for pc, c in zip(parts["bodies"], colB):
    pc.set_facecolor(c); pc.set_alpha(0.75); pc.set_edgecolor("0.3")
parts["cmedians"].set_color("0.2")
for p, x in zip(posB, dataB):
    if len(x):
        axA.text(p, np.median(x), f" {10 ** np.median(x):.0f}", ha="left", va="center",
                 fontsize=fs(6), color="0.15")
axA.set_ylabel("RNA length")
axA.set_xticks([0.5, 3.1]); axA.set_xticklabels(["protein–RNA", "RNA-only"])
axA.legend(handles=[Patch(fc=ERA_C[e], label=f"{e}-cutoff") for e in ERAS], frameon=False,
           loc="center", bbox_to_anchor=(0.5, -0.45), ncol=2)

# ---------- B: Cohen's d of RNA descriptors ----------
DESCR = [("logL", "RNA length (log$_{10}$)"), ("RNA_Shannon_Entropy_K2", "Shannon entropy"),
         ("RNA_MFE_Value", "MFE"), ("RNA_GC_Content", "GC content"),
         ("RNA_GC_Skew", "GC skew"), ("RNA_AU_Skew", "AU skew"),
         ("RNA_SequenceComplexity", "complexity")]
y = np.arange(len(DESCR)); h = 0.38
axB.axvspan(-0.2, 0.2, color="0.85", alpha=0.6, zorder=0)
for k, ds in enumerate(["protein-RNA", "RNA-only"]):
    d = df[df.dataset == ds]
    vals = [cohend(d[d.era == "pre"][c], d[d.era == "post"][c]) if c in d.columns else np.nan
            for c, _ in DESCR]
    axB.barh(y + (0.5 - k) * h, vals, h, color=DS_C[ds], label=ds, edgecolor="white", zorder=3)
axB.axvline(0, color="0.3", lw=0.8)
axB.set_yticks(y); axB.set_yticklabels([l for _, l in DESCR]); axB.invert_yaxis()
axB.set_xlabel("Cohen's $d$ (pre $-$ post)")
axB.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, -0.45), ncol=2)

# ---------- C: family composition ----------
top_fam = df.loc[df._fam != "unannotated", "_fam"].value_counts().head(5).index.tolist()
fam_labs = top_fam + ["other/unannot."]
cmap = plt.get_cmap("tab20")
fcol = [cmap(i / max(1, len(fam_labs) - 1)) for i in range(len(fam_labs))]
yC = np.arange(len(DGRID)); leftC = np.zeros(len(DGRID))
comp = np.array([[(df[(df.dataset == ds) & (df.era == era)]["_fam"] == f).mean() for f in top_fam]
                 for ds, era in DGRID])
comp = np.column_stack([comp, 1 - comp.sum(axis=1)])
for j, lab in enumerate(fam_labs):
    axC.barh(yC, comp[:, j], left=leftC, color=fcol[j], label=lab, edgecolor="white", height=0.72)
    leftC += comp[:, j]
axC.set_yticks(yC); axC.set_yticklabels([f"{ds}\n{era}" for ds, era in DGRID]); axC.invert_yaxis()
axC.set_xlabel("RNA family composition (fraction)"); axC.set_xlim(0, 1)
axC.legend(frameon=False, ncol=2, fontsize=fs(5.6), loc="center", bbox_to_anchor=(0.5, -0.45),
           columnspacing=1.2, handletextpad=0.4)

# ---------- D: accuracy matched by family & length ----------
def famsub(ds, pat):
    return df[(df.dataset == ds) & df["_fam"].astype(str).str.contains(pat, case=False, na=False)]

STRATA = [
    ("tRNA\n(PR)", famsub("protein-RNA", "tRNA")),
    ("gpII intron\n(RO)", famsub("RNA-only", "intron|group.?ii|gpii|gpi")),
    ("<25 nt\n(PR)", df[(df.dataset == "protein-RNA") & (pd.to_numeric(df[L], errors="coerce") < 25)]),
    ("50–100 nt\n(RO)", df[(df.dataset == "RNA-only") & (pd.to_numeric(df[L], errors="coerce").between(50, 100))]),
]
xD = np.arange(len(STRATA)); wD = 0.38
for i, era in enumerate(ERAS):
    ms, es, ns = [], [], []
    for _, sub in STRATA:
        m, s, n = msem(sub[sub.era == era][LDDT]); ms.append(m); es.append(s); ns.append(n)
    axD.bar(xD + (i - 0.5) * wD, ms, wD, yerr=es, capsize=3, color=ERA_C[era],
            label=f"{era}-cutoff", edgecolor="white")
    for xi, m, e, n in zip(xD + (i - 0.5) * wD, ms, es, ns):
        if np.isfinite(m):
            axD.text(xi, m + e + 0.035, f"n={n}", ha="center", va="bottom", fontsize=fs(5.6), rotation=90)
axD.set_xticks(xD); axD.set_xticklabels([s[0] for s in STRATA], fontsize=fs(6.2))
axD.set_ylabel("RNA LDDT$_r$"); axD.set_ylim(0, 1.15)
axD.legend(frameon=False, loc="center", ncol=2, bbox_to_anchor=(0.5, -0.34))

# ---------- E: method share ----------
METH = ["X-ray", "cryo-EM", "other"]; MC = {"X-ray": "#4C78A8", "cryo-EM": "#E15759", "other": "0.75"}
yE = np.arange(len(DGRID)); leftE = np.zeros(len(DGRID))
shr = np.array([[(df[(df.dataset == ds) & (df.era == era)]["method"] == m).mean() for m in METH]
                for ds, era in DGRID])
for j, m in enumerate(METH):
    axE.barh(yE, shr[:, j], left=leftE, color=MC[m], label=m, edgecolor="white", height=0.72)
    leftE += shr[:, j]
axE.set_yticks(yE); axE.set_yticklabels([f"{ds}\n{era}" for ds, era in DGRID]); axE.invert_yaxis()
axE.set_xlabel("experimental-method share"); axE.set_xlim(0, 1)
axE.legend(frameon=False, ncol=3, loc="center", bbox_to_anchor=(0.5, -0.34))

# ---------- F: accuracy within method (protein-RNA); two-level x-axis (method / metric) ----------
# colour = era (pre/post); left pair per method = LDDT_r (solid), right pair = TM_r (light hatch).
# The metric is read off the x-axis sub-labels, so the legend only needs the era colours.
d = df[df.dataset == "protein-RNA"]; methods = ["X-ray", "cryo-EM"]
xF = np.arange(len(methods)); wF = 0.20
for mi, (col, hatch) in enumerate([(LDDT, ""), (TM, "//")]):
    for ei, era in enumerate(ERAS):
        ms, es = [], []
        for meth in methods:
            m, s, _ = msem(d[(d.method == meth) & (d.era == era)][col]); ms.append(m); es.append(s)
        axF.bar(xF + (mi * 2 + ei - 1.5) * wF, ms, wF, yerr=es, capsize=2, color=ERA_C[era],
                hatch=hatch, edgecolor="white")
# TM_r pre-vs-post Cohen's d, annotated under each method's TM_r pair
for j, meth in enumerate(methods):
    dd = cohend(d[(d.method == meth) & (d.era == "pre")][TM], d[(d.method == meth) & (d.era == "post")][TM])
    if np.isfinite(dd):
        axF.text(xF[j] + wF, 0.04, f"$d$={dd:.2f}", ha="center", va="bottom", fontsize=fs(6.2),
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))
# two-level x-axis: metric sub-labels as ticks; method names on a lower level
sub = np.concatenate([[xF[j] - wF, xF[j] + wF] for j in range(len(methods))])
axF.set_xticks(sub); axF.set_xticklabels(["LDDT$_r$", "TM$_r$"] * len(methods), fontsize=fs(6))
axF.tick_params(axis="x", length=0)
for j, meth in enumerate(methods):
    axF.text(xF[j], -0.14, meth, transform=axF.get_xaxis_transform(), ha="center", va="top", fontsize=fs(7.5))
axF.set_ylabel("RNA accuracy"); axF.set_ylim(0, 1.05)
axF.legend(handles=[Patch(fc=ERA_C["pre"], label="pre-cutoff"), Patch(fc=ERA_C["post"], label="post-cutoff")],
           frameon=False, ncol=2, loc="center", bbox_to_anchor=(0.5, -0.34))

for ax, ltr in zip([axA, axB, axC, axD, axE, axF], "ABCDEF"):
    ax.text(-0.13, 1.05, ltr, transform=ax.transAxes, fontsize=fs(8.5), fontweight="bold",
            va="bottom", ha="right")

fig.tight_layout(w_pad=2.2, h_pad=1.0)
fig.subplots_adjust(hspace=0.80)   # trim slack between rows (just clears the deep upper legends)
save_all_formats(fig, f"{OUT}/postpre_family_comparison", dpi=600)
print(f"[postpre_composition] n protein-RNA={len(cx)}, RNA-only={len(ro)}; saved 4 formats to {OUT}/")
