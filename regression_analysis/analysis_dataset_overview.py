#!/usr/bin/env python
"""Dataset descriptive overview (two SI figures + summary CSVs).

Characterises the benchmark used throughout the paper "in general" -- how many
systems, of which kind, how big, what is bound, how deep the alignments are, at
what temperature -- so a reader can see the composition of the data before the
accuracy results. Read-only on the preprocessing per-chain CSVs (deterministic).

Two figures (both SI):

  Figure S-comp  dataset_overview_composition (whole dataset, 6 panels)
    A  counts per system class (complexes vs RNA chains)
    B  experimental method (X-ray / cryo-EM / other)
    C  RNA chain-length distribution
    D  protein size distribution
    E  experimental-solution ion / additive prevalence
    F  RNA family composition

  Figure S-cov   dataset_overview_covariates (inputs + one before/after panel)
    A  MSA depth (protein & RNA), with an explicit "no MSA" bucket
    B  template availability (protein / RBP templates)
    C  experimental temperature
    D  before/after (pre vs post AF3 cutoff): the contrasts NOT already shown by
       postpre_family_comparison -- MSA depth, template & solution-ion prevalence

Deliberately does NOT re-plot length / family / method split by era: that lives
in postpre_family_comparison (Fig.~\\ref{fig:postpre_composition}). Here the
before/after axis is confined to panel S-cov-D.

Ion / ligand note: the structural ion & ligand occupancy columns are not carried
in the per-chain protein-RNA table, so panel E uses the recorded experimental
*solution* composition (`sol_conc_*_m`) as the ionic-environment descriptor. The
RNA-only subset does carry structural `Ions`/`Ligands`; that prevalence is written
to the summary CSV.
"""
import os
import sys
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# figstyle lives in the repo root (AF3InterfaceEval/); figsave is local.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import figstyle                                            # noqa: E402
from figsave import save_all_formats                       # noqa: E402

# Plain (non-usetex) sans-serif: portable, matches the pipeline matplotlibrc and
# the other analysis_*.py SI scripts; avoids the fragile LaTeX text path (which
# breaks on '&', '%', unicode) if a latex toolchain happens to be on PATH.
figstyle.apply("plain")
C = figstyle.COLORS

OUT = "paper_new_analyses/dataset_overview"
os.makedirs(OUT, exist_ok=True)

CX_PATH = "csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv"
RO_PATH = "csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv"

L = "pred_rna_chain_length"
FAM = "rna_family"

# Curated ion / additive groups -> substrings matched against sol_conc_*_m columns.
ION_GROUPS = {
    "Na":  ["nacl", "sodium", "_na_"],
    "Mg":  ["mgcl2", "magnesium", "_mg_", "mg_"],
    "K":   ["kcl", "potassium", "_k_"],
    "Li":  ["licl", "lithium"],
    "Mn":  ["mncl2", "manganese", "_mn_"],
    "Ca":  ["cacl2", "calcium", "_ca_"],
    "Zn":  ["zn", "zinc"],
    # cobalt hexammine ([Co(NH3)6]3+, 18 complexes, incl. the "hexamine" spelling) is kept
    # apart from Co2+ salts (cobaltous chloride / plain "cobalt", 4 complexes)
    "Co(NH3)6": ["hexam"],
    "Co":  ["cobalt"],
    "Ni":  ["nicl", "nickel"],
    "Cd":  ["cadmium", "cdcl"],
    "spermidine/spermine": ["spermidine", "spermine"],
}
# Substrings that EXCLUDE a column from a group (hexammine columns also contain "cobalt").
ION_EXCLUDE = {"Co": ["hexam"]}
# Display labels for panel E: ions in their ionic form (charge as superscript),
# spermidine/spermine shown as their class. Keys above stay unchanged.
ION_LABELS = {   # "{+}" keeps mathtext from spacing "+" as a binary operator
    "Na": r"Na$^{+}$", "K": r"K$^{+}$", "Li": r"Li$^{+}$",
    "Mg": r"Mg$^{2{+}}$", "Mn": r"Mn$^{2{+}}$", "Ca": r"Ca$^{2{+}}$", "Zn": r"Zn$^{2{+}}$",
    "Ni": r"Ni$^{2{+}}$", "Cd": r"Cd$^{2{+}}$", "Co": r"Co$^{2{+}}$",
    "Co(NH3)6": r"[Co(NH$_3$)$_6$]$^{3{+}}$",
    "spermidine/spermine": "Polyamine",
}


def load(path, label):
    df = pd.read_csv(path, low_memory=False)
    df["method"] = np.where(df.get("is_xray", 0) == 1, "X-ray",
                     np.where(df.get("is_em", 0) == 1, "cryo-EM", "other"))
    df["era"] = df.get("is_post_cutoff", 0).map({0: "pre", 1: "post"})
    df["dataset"] = label
    df["_fam"] = (df.get(FAM, pd.Series(index=df.index, dtype=object))
                    .fillna("unannotated").replace("", "unannotated"))
    return df


def per_complex(df, cols):
    """One row per complex (first non-null within a pdb_id) for complex-level cols."""
    keep = [c for c in cols if c in df.columns]
    return df.groupby("pdb_id")[keep].first().reset_index()


def ion_presence_per_complex(df):
    """Boolean (pdb_id x ion-group): does the recorded solution contain the group?"""
    ioncols = [c for c in df.columns if c.startswith("sol_conc_")]
    if not ioncols:
        return pd.DataFrame(index=df["pdb_id"].unique())
    conc = df[ioncols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0
    conc["pdb_id"] = df["pdb_id"].values
    per_cx = conc.groupby("pdb_id").any()
    out = {}
    for grp, subs in ION_GROUPS.items():
        cols = [c for c in ioncols if any(s in c for s in subs)
                and not any(x in c for x in ION_EXCLUDE.get(grp, ()))]
        out[grp] = per_cx[cols].any(axis=1) if cols else pd.Series(False, index=per_cx.index)
    return pd.DataFrame(out)


def parse_struct_list(cell, key):
    """RNA-only 'Ions'/'Ligands' cells look like "[{'ions':'MG','n':5,...}, ...]"."""
    if not isinstance(cell, str) or not cell.strip() or cell.strip() in ("[]", "nan", "None"):
        return []
    try:
        items = ast.literal_eval(cell)
        return [str(d.get(key, "")).upper() for d in items if isinstance(d, dict)]
    except (ValueError, SyntaxError):
        return []


cx = load(CX_PATH, "protein-RNA")
ro = load(RO_PATH, "RNA-only")

# Authoritative benchmark counts (match the paper's "1309 protein-RNA complexes /
# 2356 RNA chains; 163 RNA-only"). The complex count comes from the complex-level
# table (one row per complex); the per-chain grouping under-counts by a few
# structures. The analysed RNA-only set is the chains with a valid RNA metric
# (168 raw rows -> 163 scored).
WI_SUMMARY = "csvs4analysis_with_interface/combined/combined__subset_summary.csv"
if os.path.exists(WI_SUMMARY):
    _wi = dict(pd.read_csv(WI_SUMMARY).itertuples(index=False, name=None))
    PR_1V1_CX = int(_wi.get("rna_protein_1v1", cx.iloc[:0].shape[0]))
    PR_MULTI_CX = int(_wi.get("multichain_complex", 0))
    PR_COMPLEXES = int(_wi.get("all_merged", PR_1V1_CX + PR_MULTI_CX))
else:  # fall back to per-chain grouping if the complex-level summary is absent
    PR_1V1_CX, PR_MULTI_CX, PR_COMPLEXES = 0, 0, int(cx["pdb_id"].nunique())
if "assigned_rna_tm" in ro.columns:
    ro = ro[ro["assigned_rna_tm"].notna()].copy()   # analysed RNA-only set: 163

# ---------------------------------------------------------------------------
#  Summary CSVs
# ---------------------------------------------------------------------------
def class_row(name, df, n_complexes=None):
    cxx = per_complex(df, ["protein_total_length", "method", "log10_protein_msa_size",
                           "log10_rna_msa_size", "n_protein_templates", "n_rbp_templates",
                           "exp_Temp", "sol_temp_k", "rna_family_is_known"])
    temp = pd.to_numeric(cxx.get("exp_Temp"), errors="coerce")
    temp = temp.fillna(pd.to_numeric(cxx.get("sol_temp_k"), errors="coerce"))
    ions = ion_presence_per_complex(df)
    return dict(
        system_class=name,
        n_complexes=int(df["pdb_id"].nunique()) if n_complexes is None else int(n_complexes),
        n_rna_chains=int(len(df)),
        median_rna_len=round(float(pd.to_numeric(df[L], errors="coerce").median()), 1),
        median_protein_len=round(float(pd.to_numeric(cxx.get("protein_total_length"), errors="coerce").median()), 1),
        pct_xray=round(100 * (cxx["method"] == "X-ray").mean(), 1),
        pct_cryoem=round(100 * (cxx["method"] == "cryo-EM").mean(), 1),
        pct_other=round(100 * (cxx["method"] == "other").mean(), 1),
        pct_with_solution_ion=round(100 * ions.any(axis=1).mean(), 1) if len(ions) else np.nan,
        median_protein_msa_log10=round(float(pd.to_numeric(cxx.get("log10_protein_msa_size"), errors="coerce").median()), 2)
            if "log10_protein_msa_size" in cxx else np.nan,
        median_rna_msa_log10=round(float(pd.to_numeric(cxx.get("log10_rna_msa_size"), errors="coerce").median()), 2),
        pct_with_protein_template=round(100 * (pd.to_numeric(cxx.get("n_protein_templates"), errors="coerce") > 0).mean(), 1)
            if "n_protein_templates" in cxx else np.nan,
        pct_with_rbp_template=round(100 * (pd.to_numeric(cxx.get("n_rbp_templates"), errors="coerce") > 0).mean(), 1)
            if "n_rbp_templates" in cxx else np.nan,
        pct_rna_family_known=round(100 * (pd.to_numeric(cxx.get("rna_family_is_known"), errors="coerce") == 1).mean(), 1)
            if "rna_family_is_known" in cxx else np.nan,
        median_temp_k=round(float(temp.median()), 1) if temp.notna().any() else np.nan,
    )


summary_rows = [
    class_row("all_protein_RNA", cx, n_complexes=PR_COMPLEXES),
    class_row("RNA_only", ro),
    class_row("overall", pd.concat([cx, ro], ignore_index=True),
              n_complexes=PR_COMPLEXES + int(ro["pdb_id"].nunique())),
]
summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{OUT}/dataset_overview_summary.csv", index=False)

# pre/post contrast table (panel S-cov-D)
def era_stats(df, era):
    d = df[df.era == era]
    cxx = per_complex(d, ["log10_protein_msa_size", "log10_rna_msa_size",
                          "n_protein_templates", "n_rbp_templates"])
    ions = ion_presence_per_complex(d)
    return dict(
        era=era, n_complexes=int(d["pdb_id"].nunique()), n_rna_chains=int(len(d)),
        median_protein_msa_log10=round(float(pd.to_numeric(cxx.get("log10_protein_msa_size"), errors="coerce").median()), 2)
            if "log10_protein_msa_size" in cxx else np.nan,
        median_rna_msa_log10=round(float(pd.to_numeric(cxx.get("log10_rna_msa_size"), errors="coerce").median()), 2),
        pct_with_template=round(100 * ((pd.to_numeric(cxx.get("n_protein_templates"), errors="coerce").fillna(0) > 0) |
                                       (pd.to_numeric(cxx.get("n_rbp_templates"), errors="coerce").fillna(0) > 0)).mean(), 1),
        pct_with_solution_ion=round(100 * ions.any(axis=1).mean(), 1) if len(ions) else np.nan,
    )


prepost = pd.DataFrame([era_stats(cx, "pre"), era_stats(cx, "post")])
prepost.to_csv(f"{OUT}/dataset_overview_pre_post.csv", index=False)
print("[dataset_overview] summary:\n", summary.to_string(index=False))
print("[dataset_overview] pre/post:\n", prepost.to_string(index=False))

def align_letters(fig, rows, dx=0.004, dy=0.006):
    """Draw bold panel letters at ONE common height per row (figure coords), so
    they line up across panels regardless of differing title heights or panel
    types. `rows` is a list of (axes_list, letters) pairs."""
    fig.canvas.draw()
    for axrow, letters in rows:
        top = max(a.get_position().y1 for a in axrow)
        for a, lab in zip(axrow, letters):
            fig.text(a.get_position().x0 - dx, top + dy, lab,
                     ha="right", va="bottom",
                     fontsize=figstyle.PANEL_LETTER_SIZE,
                     fontweight=figstyle.PANEL_LETTER_WEIGHT)


# ===========================================================================
#  FIGURE S-comp : dataset_overview_composition (2 x 3)
# ===========================================================================
GRID = ["blue", "orange", "green", "red", "purple", "brown"]
fig, axes = plt.subplots(2, 3, figsize=(figstyle.WIDTH_2COL, 5.6))
(axA, axB, axC), (axD, axE, axF) = axes
for ax in axes.ravel():
    ax.grid(True, alpha=0.3, axis="both"); ax.set_axisbelow(True)

# ---- A: counts per system class ----
# Chain counts from the per-chain subset tables; complex counts from the
# authoritative complex-level summary (see PR_* above). RNA-only = 163 (scored).
_ic = "csvs4analysis_per_chain_complex/combined"
n_ch_1v1 = len(pd.read_csv(f"{_ic}/combined__per_chain_rna_protein_1v1.csv", usecols=["pdb_id"])) \
    if os.path.exists(f"{_ic}/combined__per_chain_rna_protein_1v1.csv") else 0
n_ch_multi = len(pd.read_csv(f"{_ic}/combined__per_chain_multichain_complex.csv", usecols=["pdb_id"])) \
    if os.path.exists(f"{_ic}/combined__per_chain_multichain_complex.csv") else 0
names = ["1v1", "multichain", "RNA-only"]
n_cx = [PR_1V1_CX, PR_MULTI_CX, int(ro["pdb_id"].nunique())]
n_ch = [n_ch_1v1, n_ch_multi, len(ro)]
x = np.arange(len(names)); w = 0.38
b1 = axA.bar(x - w / 2, n_cx, w, color=C["rna"], label="complexes", edgecolor="white")
b2 = axA.bar(x + w / 2, n_ch, w, color=C["protein"], label="RNA chains", edgecolor="white")
for bars in (b1, b2):
    axA.bar_label(bars, fontsize=figstyle.FONT_SMALL, padding=1)
axA.set_xticks(x); axA.set_xticklabels([{"multichain": "multi-" + chr(10) + "chain", "RNA-only": "single-chain" + chr(10) + "RNA"}.get(n, n) for n in names],  # two-line labels: no overlap
                                        fontsize=figstyle.FONT_SMALL)
axA.set_ylabel("count"); axA.set_ylim(0, max(n_ch) * 1.22)
axA.legend(frameon=False, loc="upper left", fontsize=figstyle.FONT_SMALL,
           handlelength=1.0, handletextpad=0.5, borderaxespad=0.3)
axA.set_title("system classes", fontsize=figstyle.FONT_BASE, pad=8)

# ---- B: experimental method (per complex, whole dataset) ----
allc = per_complex(pd.concat([cx, ro], ignore_index=True), ["method"])
mc = {"X-ray": C["blue"], "cryo-EM": C["post"], "other": C["grey"]}
_mo = [(m, int((allc["method"] == m).sum())) for m in ["X-ray", "cryo-EM", "other"]]
meth_order = [m for m, c in _mo if c > 0]
counts = [c for _, c in _mo if c > 0]
wed, txt, aut = axB.pie(counts, labels=None, colors=[mc[m] for m in meth_order],
                        autopct=lambda p: f"{p:.0f}%", pctdistance=0.72,
                        wedgeprops=dict(width=0.42, edgecolor="white"),
                        textprops=dict(fontsize=figstyle.FONT_SMALL))
axB.legend(wed, [f"{m} (n={c})" for m, c in zip(meth_order, counts)],
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02),
           fontsize=figstyle.FONT_SMALL, handlelength=1.2, handletextpad=0.5)
axB.set_title("experimental method", fontsize=figstyle.FONT_BASE, pad=8)

# ---- C: RNA chain-length distribution ----
bins = np.logspace(np.log10(4), np.log10(1600), 26)
axC.hist(pd.to_numeric(cx[L], errors="coerce").dropna(), bins=bins, color=C["rna"],
         alpha=0.75, label="protein-RNA", edgecolor="white", linewidth=0.3)
axC.hist(pd.to_numeric(ro[L], errors="coerce").dropna(), bins=bins, color=C["purple"],
         alpha=0.7, label="single-chain RNA", edgecolor="white", linewidth=0.3)
axC.set_xscale("log")
med = pd.to_numeric(cx[L], errors="coerce").median()
axC.axvline(med, color="0.25", ls="--", lw=0.9)
axC.text(med, axC.get_ylim()[1] * 0.55, f" median\n {med:.0f} nt", fontsize=figstyle.FONT_SMALL,
         color="0.25", va="center")
axC.set_xlabel("RNA chain length (nt)"); axC.set_ylabel("RNA chains")
axC.legend(frameon=False, loc="upper right")

# ---- D: protein size distribution (per complex) ----
pl = pd.to_numeric(per_complex(cx, ["protein_total_length"])["protein_total_length"], errors="coerce")
pl = pl[pl > 0]
pbins = np.logspace(np.log10(max(20, pl.min())), np.log10(pl.max()), 26)
axD.hist(pl, bins=pbins, color=C["protein"], alpha=0.8, edgecolor="white", linewidth=0.3)
axD.set_xscale("log")
axD.axvline(pl.median(), color="0.25", ls="--", lw=0.9)
axD.text(pl.median(), axD.get_ylim()[1] * 0.92, f" median {pl.median():.0f} aa",
         fontsize=figstyle.FONT_SMALL, color="0.25")
axD.set_xlabel("protein length (residues)"); axD.set_ylabel("complexes")

# ---- E: experimental-solution ion / additive prevalence ----
ions_all = pd.concat([ion_presence_per_complex(cx), ion_presence_per_complex(ro)], axis=0)
prev = (100 * ions_all.mean(axis=0)).sort_values(ascending=True)
prev = prev[prev > 0]
axE.barh(np.arange(len(prev)), prev.values, color=C["teal"], edgecolor="white")
axE.set_yticks(np.arange(len(prev)))
axE.set_yticklabels([ION_LABELS.get(k, k) for k in prev.index], fontsize=figstyle.FONT_SMALL)
for i, v in enumerate(prev.values):
    axE.text(v + 0.4, i, f"{v:.0f}%" if v >= 1 else "<1%", va="center",
             fontsize=figstyle.FONT_SMALL)
axE.set_xlabel("% of complexes (solution)")
axE.set_xlim(0, prev.values.max() * 1.25)
axE.set_title("ions & additives in solution", fontsize=figstyle.FONT_BASE)

# ---- F: RNA family composition (annotated chains only; the 87% unannotated
#         would otherwise dwarf every family bar -- its share is stated instead) ----
famall = pd.concat([cx, ro], ignore_index=True)
known = famall[famall["_fam"] != "unannotated"]
top = known["_fam"].value_counts().head(8)
comp = top.copy()
comp["other annotated"] = len(known) - top.sum()
comp = comp[::-1]
cols_f = [C["grey"] if k == "other annotated" else C["green"] for k in comp.index]
axF.barh(np.arange(len(comp)), comp.values, color=cols_f, edgecolor="white")
axF.set_yticks(np.arange(len(comp))); axF.set_yticklabels(comp.index, fontsize=figstyle.FONT_SMALL)
axF.set_xlabel("RNA chains (annotated)")
axF.set_xlim(0, comp.values.max() * 1.15)
axF.set_title("RNA families", fontsize=figstyle.FONT_BASE, pad=8)
axF.text(0.97, 0.06, f"{100*len(known)/len(famall):.0f}% of chains annotated\n(n={len(known)})",
         transform=axF.transAxes, ha="right", va="bottom", fontsize=figstyle.FONT_SMALL, color="0.35")

fig.tight_layout(w_pad=1.6, h_pad=1.8)
align_letters(fig, [([axA, axB, axC], "ABC"), ([axD, axE, axF], "DEF")])
save_all_formats(fig, f"{OUT}/dataset_overview_composition", dpi=600)
plt.close(fig)

# ===========================================================================
#  FIGURE S-cov : dataset_overview_covariates (2 x 2)
# ===========================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(figstyle.WIDTH_2COL, 5.2))
(gA, gB), (gC, gD) = axes2
for ax in axes2.ravel():
    ax.grid(True, alpha=0.3, axis="both"); ax.set_axisbelow(True)

# ---- A: MSA depth (protein & RNA), with a "no MSA" note ----
pmsa = pd.to_numeric(per_complex(cx, ["log10_protein_msa_size"])["log10_protein_msa_size"], errors="coerce").dropna()
rmsa = pd.to_numeric(cx["log10_rna_msa_size"], errors="coerce").dropna()
mbins = np.linspace(0, 4.5, 24)
gA.hist(pmsa, bins=mbins, color=C["protein"], alpha=0.75, label="protein MSA", edgecolor="white", linewidth=0.3)
gA.hist(rmsa, bins=mbins, color=C["rna"], alpha=0.7, label="RNA MSA", edgecolor="white", linewidth=0.3)
frac_no_rna = float((rmsa <= 0).mean())
gA.axvline(0, color="0.4", lw=0.8)
gA.text(0.05, gA.get_ylim()[1] * 0.9, f"{100*frac_no_rna:.0f}% RNA\nchains: no MSA",
        fontsize=figstyle.FONT_SMALL, color="0.25", va="top")
gA.set_xlabel("MSA depth (log$_{10}$ # sequences)"); gA.set_ylabel("count")
gA.legend(frameon=False, loc="upper right")

# ---- B: template availability ----
cc = per_complex(cx, ["n_protein_templates", "n_rbp_templates"])
npt = pd.to_numeric(cc.get("n_protein_templates"), errors="coerce").fillna(0)
nrt = pd.to_numeric(cc.get("n_rbp_templates"), errors="coerce").fillna(0)
labels_t = ["≥1 protein\ntemplate", "≥1 RBP\ntemplate", "no\ntemplate"]
vals_t = [100 * (npt > 0).mean(), 100 * (nrt > 0).mean(),
          100 * ((npt == 0) & (nrt == 0)).mean()]
bt = gB.bar(np.arange(3), vals_t, color=[C["protein"], C["green"], C["grey"]], edgecolor="white", width=0.6)
gB.bar_label(bt, fmt="%.0f%%", fontsize=figstyle.FONT_SMALL, padding=1)
gB.set_xticks(np.arange(3)); gB.set_xticklabels(labels_t, fontsize=figstyle.FONT_SMALL)
gB.set_ylabel("% of complexes"); gB.set_ylim(0, 105)
gB.set_title("structural templates", fontsize=figstyle.FONT_BASE)

# ---- C: experimental temperature ----
tcx = per_complex(cx, ["exp_Temp", "sol_temp_k"])
temp = pd.to_numeric(tcx["exp_Temp"], errors="coerce")
temp = temp.fillna(pd.to_numeric(tcx["sol_temp_k"], errors="coerce"))
temp = temp[(temp > 250) & (temp < 360)]  # physiological/crystallisation range
gC.hist(temp, bins=np.arange(270, 320, 2.5), color=C["teal"], alpha=0.85, edgecolor="white", linewidth=0.3)
for tk, lab in [(277, "cold room"), (293, "RT"), (310, "37°C")]:
    gC.axvline(tk, color="0.5", ls=":", lw=0.8)
    gC.text(tk, gC.get_ylim()[1] * 0.96, f" {lab}", rotation=90, va="top",
            fontsize=figstyle.FONT_SMALL, color="0.4")
gC.set_xlabel("experimental temperature (K)"); gC.set_ylabel("complexes")
gC.set_title(f"n={int(temp.notna().sum())} with recorded T", fontsize=figstyle.FONT_SMALL)

# ---- D: before/after (pre vs post cutoff) -- the genuinely-new contrasts ----
metrics = [("RNA MSA\n(median log$_{10}$)", "median_rna_msa_log10", 1.0),
           ("protein MSA\n(median log$_{10}$)", "median_protein_msa_log10", 1.0),
           ("% with\ntemplate", "pct_with_template", 0.01),
           ("% with\nsolution ion", "pct_with_solution_ion", 0.01)]
xm = np.arange(len(metrics)); wm = 0.38
pp = prepost.set_index("era")
for i, era in enumerate(["pre", "post"]):
    vals = [pp.loc[era, key] * scl for _, key, scl in metrics]
    bars = gD.bar(xm + (i - 0.5) * wm, vals, wm, color=C[era], label=f"{era}-cutoff", edgecolor="white")
    gD.bar_label(bars, fmt="%.2f", fontsize=figstyle.FONT_SMALL, padding=1)
gD.set_xticks(xm); gD.set_xticklabels([m[0] for m in metrics], fontsize=figstyle.FONT_SMALL)
gD.set_ylim(0, 4.2)
gD.set_ylabel("value (MSA log$_{10}$ | fraction)")
gD.set_title(f"before / after AF3 cutoff  (pre n={int(pp.loc['pre','n_complexes'])}, "
             f"post n={int(pp.loc['post','n_complexes'])})", fontsize=figstyle.FONT_SMALL)
gD.legend(frameon=False, loc="upper right")

fig2.tight_layout(w_pad=1.8, h_pad=2.0)
align_letters(fig2, [([gA, gB], "AB"), ([gC, gD], "CD")])
save_all_formats(fig2, f"{OUT}/dataset_overview_covariates", dpi=600)
plt.close(fig2)

print(f"[dataset_overview] saved 2 figures (4 formats each) + 2 summary CSVs to {OUT}/")
