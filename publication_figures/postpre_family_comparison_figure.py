"""
Pre- vs post-cutoff COMPOSITION check: are the a2021 (post) and b2021 (pre) sets
comparable in RNA complexity and family -- for BOTH protein-RNA and RNA-only -- and
how are families predicted pre vs post? Companion to post_pre_comparison_figure.py;
family parsing is reused from rna_family_publication_figure.py
(evaluation.analyzeRNAFamilyMSA_singleDB.process_rna_family).

The point is to check that the pre/post accuracy gap in Fig.~postpre_combined is not an
apples-to-pears artefact of one set holding intrinsically harder RNAs, and to look at
family-level generalisation.

Emits results/plots/all_chains/publication_figures/postpre_family_comparison.png(/pdf):
  A  Cohen's d (pre - post) of RNA complexity descriptors, per context; the shaded
     |d|<0.2 band = negligible. Protein-RNA is well matched; the RNA-only post set
     skews to longer / more stable RNAs.
  B  RNA length distribution (log10 nt), pre vs post split-violins, both contexts.
  C  RNA family composition (stacked family fractions) for the four dataset x cutoff
     groups; most annotated families do not overlap pre vs post.
  D  Per-family mean RNA accuracy (LDDT_r and TM_r), pre vs post, for families present
     on both sides (>= MIN_BOTH each) -- direct within-family generalisation.

Data: data/compareDatabases/post_pre/{a2021_b2021_protein_rna,a2021_b2021_rna_only}.

Usage:  python -m publication_figures.postpre_family_comparison_figure
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.analyzeRNAFamilyMSA_singleDB import process_rna_family

PRE, POST = "pre-cutoff (b2021)", "post-cutoff (a2021)"
PRE_C, POST_C = "#1f77b4", "#E8743B"          # blue / orange (matches post_pre figure)
PAL = {PRE: PRE_C, POST: POST_C}
CTX_C = {"protein-RNA": "#5B8C5A", "RNA-only": "#9B5DE5"}   # green / purple
CTX_SHORT = {"protein-RNA": "PR", "RNA-only": "RNA"}
CTX_DISPLAY = {"protein-RNA": "Protein-RNA", "RNA-only": "Single-chain RNA"}
CTX_ORDER = ["protein-RNA", "RNA-only"]
MIN_BOTH = 3                                   # min samples each side for panel D

# Nature-style fonts (large enough for a full-page figure).
PUB_RC = {
    "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13, "font.size": 15,
}
LBL = 12                                       # in-plot annotation size

_BASE = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre")
CONTEXTS = [
    ("protein-RNA", "exp_protein_rna", "pred_protein_rna",
     os.path.join(_BASE, "a2021_b2021_protein_rna", "b2021.db"),
     os.path.join(_BASE, "a2021_b2021_protein_rna", "a2021.db")),
    ("RNA-only", "exp_rna_rna", "pred_rna_rna",
     os.path.join(_BASE, "a2021_b2021_rna_only", "b2021_rna.db"),
     os.path.join(_BASE, "a2021_b2021_rna_only", "a2021_rna.db")),
]

# descriptor column -> display label (all computed for every RNA, annotated or not)
DESCR = [
    ("RNALength", "RNA length (log)"),
    ("RNA_Shannon_Entropy_K2", "Shannon entropy"),
    ("RNA_MFE_Value", "MFE"),
    ("RNA_GC_Content", "GC content"),
    ("RNA_GC_Skew", "GC skew"),
    ("RNA_AU_Skew", "AU skew"),
]

# accuracy metrics shown in panel D
METRICS = [("LDDT_r", "RNA_LDDT_allAtom", r"LDDT$_r$"),
           ("TM_r", "RNA_TM", r"TM$_r$")]

# human-readable RNA family names for the legend / family labels
FAM_DISPLAY = {
    "SSU_rRNA_bacteria": "Bacterial SSU rRNA",
    "Purine": "Purine riboswitch",
    "group-II-D1D4-3": "Group II intron",
    "SAM": "SAM riboswitch",
    "Intron_gpI": "Group I intron",
    "nadA": "nadA motif",
}


METHOD_ORDER = ["X-ray", "cryo-EM"]
METHOD_C = {"X-ray": "#8C510A", "cryo-EM": "#01665E"}      # brown / teal


def _method(x):
    x = str(x).upper()
    if "DIFFRACTION" in x or "X-RAY" in x:
        return "X-ray"
    if "MICROSCOPY" in x or "CRYO" in x or x == "EM":
        return "cryo-EM"
    return "other"


# --------------------------------------------------------------------------- #
def _g(s):
    """Scalar-or-first-of-list -> float; NaN on failure."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    try:
        v = eval(s) if isinstance(s, str) else s
        return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
    except Exception:
        return np.nan


def _cohend(a, b):
    a = pd.Series(a, dtype=float).dropna()
    b = pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / sp if sp else np.nan


def _load(db, exp, pred):
    con = sqlite3.connect(db)
    have = [r[1] for r in con.execute(f"PRAGMA table_info({pred})")]
    dcols = [k for k, _ in DESCR if k in have]
    mcols = [(key, col) for key, col, _ in METRICS if col in have]
    sel = "e.RNAFamily, e.Experiment" + "".join(f", p.{c}" for c in dcols) \
        + "".join(f", p.{col}" for _, col in mcols)
    df = pd.read_sql_query(
        f"SELECT {sel} FROM {exp} e JOIN {pred} p ON e.PDBId = p.exp_db_id", con)
    con.close()
    df["fam"] = df["RNAFamily"].apply(process_rna_family)
    df["method"] = df["Experiment"].apply(_method)
    for c in dcols:
        df[c] = df[c].apply(_g)
    for key, col in mcols:
        df[key] = df[col].apply(_g)
    return df


def _load_all():
    data = {}
    for ctx, exp, pred, pre_db, post_db in CONTEXTS:
        data[ctx] = {PRE: _load(pre_db, exp, pred), POST: _load(post_db, exp, pred)}
    return data


# --------------------------------------------------------------------------- #
def _descr_series(df, key):
    """Descriptor series, log10 for the heavily right-skewed RNA length."""
    s = df.get(key)
    if s is None:
        return None
    if key == "RNALength":
        return np.log10(s.where(s > 0))
    return s


def _panel_descriptor_d(ax, data):
    """A: Cohen's d (pre-post) per descriptor, grouped by context."""
    keys = [k for k, _ in DESCR]
    labels = [l for _, l in DESCR]
    x = np.arange(len(keys))
    w = 0.38
    ax.axhspan(-0.2, 0.2, color="0.88", zorder=0)     # |d|<0.2 negligible band
    for ctx, off in zip(CTX_ORDER, (-w / 2, +w / 2)):
        ds = [_cohend(_descr_series(data[ctx][PRE], k),
                      _descr_series(data[ctx][POST], k)) for k in keys]
        ax.bar(x + off, ds, w, color=CTX_C[ctx], edgecolor="black", linewidth=0.4,
               label=CTX_DISPLAY[ctx])
        for xi, dv in zip(x + off, ds):
            if np.isfinite(dv):
                ax.text(xi, dv + (0.03 if dv >= 0 else -0.03), f"{dv:+.2f}",
                        ha="center", va="bottom" if dv >= 0 else "top", fontsize=LBL)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"Cohen's $d$ (pre $-$ post)")
    ax.set_ylim(-0.7, 0.9)
    ax.legend(title="", loc="upper right", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5)


def _panel_length(ax, data):
    """B: RNA length (log10) pre vs post split-violins, both contexts."""
    rows = []
    for ctx in CTX_ORDER:
        for cut in (PRE, POST):
            for v in data[ctx][cut]["RNALength"].dropna():
                if v > 0:
                    rows.append((ctx, cut, np.log10(v)))
    long = pd.DataFrame(rows, columns=["ctx", "cut", "loglen"])
    sns.violinplot(data=long, x="ctx", y="loglen", hue="cut", order=CTX_ORDER,
                   hue_order=[PRE, POST], palette=PAL, split=True, inner="box",
                   cut=0, density_norm="width", linewidth=0.8, ax=ax)
    # median-length annotations (nt) above each half
    for xi, ctx in enumerate(CTX_ORDER):
        for cut, dx in ((PRE, -0.22), (POST, 0.22)):
            med = data[ctx][cut]["RNALength"].median()
            if np.isfinite(med):
                ax.text(xi + dx, np.log10(med), f"{med:.0f}", ha="center",
                        va="center", fontsize=LBL, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                  alpha=0.75))
    ax.set_xlabel("")
    ax.set_xticks(range(len(CTX_ORDER)))
    ax.set_xticklabels([CTX_DISPLAY[c] for c in CTX_ORDER])
    ax.set_ylabel("RNA length (log$_{10}$ nt)")
    ax.legend(title="", loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5)


def _panel_family_composition(ax, data):
    """C: stacked RNA-family fractions for the four dataset x cutoff groups."""
    groups = [(ctx, cut) for ctx in CTX_ORDER for cut in (PRE, POST)]
    glabels = ["PR\npre", "PR\npost", "RNA\npre", "RNA\npost"]
    totals = {}
    for ctx, cut in groups:
        for fam, c in data[ctx][cut]["fam"].value_counts().items():
            if fam != "None":
                totals[fam] = totals.get(fam, 0) + c
    top = [f for f, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:7]]
    cats = top + ["other", "unannotated"]
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(top)}
    colors["other"] = "0.6"
    colors["unannotated"] = "0.85"

    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for cat in cats:
        fracs = []
        for ctx, cut in groups:
            fam = data[ctx][cut]["fam"]
            n = len(fam)
            if cat == "unannotated":
                c = (fam == "None").sum()
            elif cat == "other":
                c = (~fam.isin(top) & (fam != "None")).sum()
            else:
                c = (fam == cat).sum()
            fracs.append(c / n * 100.0 if n else 0.0)
        ax.bar(x, fracs, 0.72, bottom=bottom, color=colors[cat], edgecolor="white",
               linewidth=0.5, label=FAM_DISPLAY.get(cat, cat))
        bottom += np.array(fracs)
    ax.set_xticks(x)
    ax.set_xticklabels(glabels)
    ax.set_ylabel("Share of samples (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="", loc="upper left", fontsize=11, framealpha=0.92,
              handlelength=1.2, labelspacing=0.3, borderpad=0.4)


LEN_BINS = [(0, 25, "<25 nt"), (25, 50, "25-50 nt"), (50, 100, "50-100 nt"),
            (100, 200, "100-200 nt"), (200, 10 ** 9, ">200 nt")]


def _sem(vals):
    v = pd.Series(vals, dtype=float).dropna()
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) >= 2 else 0.0


def _panel_family_quality(ax, data):
    """D: RNA accuracy (LDDT_r, TM_r) pre vs post, matched by family AND by length.

    Length matching uses the modal length bin (most samples) that has >=MIN_BOTH on both
    sides, so the pre/post comparison is made within the same size range -- controlling
    for the RNA-only length confound seen in panel A.
    """
    groups = []   # (name, ctx, pre_df, post_df, n_pre, n_post)
    # family-matched: families present on both sides
    for ctx in CTX_ORDER:
        pre, post = data[ctx][PRE], data[ctx][POST]
        for fam in sorted((set(pre["fam"]) & set(post["fam"])) - {"None"}):
            vp, vq = pre[pre["fam"] == fam], post[post["fam"] == fam]
            if (vp["LDDT_r"].notna().sum() >= MIN_BOTH
                    and vq["LDDT_r"].notna().sum() >= MIN_BOTH):
                groups.append((FAM_DISPLAY.get(fam, fam), ctx, vp, vq, len(vp), len(vq)))
    n_family = len(groups)
    # length-matched: modal length bin per context
    for ctx in CTX_ORDER:
        pre, post = data[ctx][PRE], data[ctx][POST]
        best = None
        for lo, hi, blab in LEN_BINS:
            vp = pre[(pre["RNALength"] >= lo) & (pre["RNALength"] < hi)]
            vq = post[(post["RNALength"] >= lo) & (post["RNALength"] < hi)]
            if (vp["LDDT_r"].notna().sum() >= MIN_BOTH
                    and vq["LDDT_r"].notna().sum() >= MIN_BOTH):
                tot = len(vp) + len(vq)
                if best is None or tot > best[0]:
                    best = (tot, blab, vp, vq)
        if best:
            _, blab, vp, vq = best
            groups.append((blab, ctx, vp, vq, len(vp), len(vq)))

    if not groups:
        ax.text(0.5, 0.5, "insufficient matched samples", transform=ax.transAxes,
                ha="center", va="center", color="0.4")
        ax.set_axis_off()
        return

    w = 0.36
    gap = 1.3
    xt, xtl, headers = [], [], []
    pos = 0.0
    divider_x = None
    for gi, (name, ctx, vp, vq, npre, npost) in enumerate(groups):
        if gi == n_family and 0 < n_family < len(groups):
            divider_x = pos - gap / 2
        start = pos
        for mkey, _col, mlbl in METRICS:
            ap, aq = vp[mkey].dropna(), vq[mkey].dropna()
            ax.bar(pos - w / 2, ap.mean(), w, yerr=_sem(ap), capsize=3, color=PRE_C,
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="0.25", elinewidth=1.0),
                   label=PRE if pos == 0 else None)
            ax.bar(pos + w / 2, aq.mean(), w, yerr=_sem(aq), capsize=3, color=POST_C,
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="0.25", elinewidth=1.0),
                   label=POST if pos == 0 else None)
            xt.append(pos)
            xtl.append(mlbl)
            pos += 1.0
        headers.append(((start + pos - 1.0) / 2,
                        f"{name}\n{CTX_SHORT[ctx]}, n={npre}/{npost}"))
        pos += gap

    if divider_x is not None:
        ax.axvline(divider_x, color="0.6", ls="--", lw=1.2)
        fam_c = np.mean([h[0] for h in headers[:n_family]])
        len_c = np.mean([h[0] for h in headers[n_family:]])
        for cx, txt in ((fam_c, "family-matched"), (len_c, "length-matched")):
            ax.text(cx, 1.02, txt, ha="center", va="bottom", fontsize=13,
                    fontstyle="italic", color="0.3",
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, fontsize=14, rotation=45)
    for cx, hdr in headers:
        ax.text(cx, -0.185, hdr, ha="center", va="top", fontsize=10.5,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.8, pos - 0.6)
    ax.legend(title="", loc="lower right", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5)


def _panel_method_composition(ax, data):
    """E: experimental-method (X-ray / cryo-EM) share per dataset x cutoff."""
    groups = [(ctx, cut) for ctx in CTX_ORDER for cut in (PRE, POST)]
    glabels = ["PR\npre", "PR\npost", "RNA\npre", "RNA\npost"]
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for cat in METHOD_ORDER:
        fr = []
        for ctx, cut in groups:
            m = data[ctx][cut]["method"]
            fr.append((m == cat).sum() / len(m) * 100.0 if len(m) else 0.0)
        ax.bar(x, fr, 0.72, bottom=bottom, color=METHOD_C[cat], edgecolor="white",
               linewidth=0.5, label=cat)
        bottom += np.array(fr)
    ax.set_xticks(x)
    ax.set_xticklabels(glabels)
    ax.set_ylabel("Share of samples (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              framealpha=0.92)


def _panel_method_quality(ax, data):
    """F: RNA accuracy (LDDT_r, TM_r) pre vs post within X-ray and within cryo-EM.

    Protein-RNA only: the RNA-only set has just 2 pre-cutoff cryo-EM structures, too few
    to stratify. Cohen's d (pre-post) is annotated above each metric.
    """
    pre, post = data["protein-RNA"][PRE], data["protein-RNA"][POST]
    w, pos = 0.36, 0.0                              # uniform spacing across all bars
    xt, xtl, headers = [], [], []
    for meth in METHOD_ORDER:
        p, q = pre[pre["method"] == meth], post[post["method"] == meth]
        start = pos
        for mkey, _col, mlbl in METRICS:
            ap, aq = p[mkey].dropna(), q[mkey].dropna()
            ax.bar(pos - w / 2, ap.mean(), w, yerr=_sem(ap), capsize=3, color=PRE_C,
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="0.25", elinewidth=1.0),
                   label=PRE if pos == 0 else None)
            ax.bar(pos + w / 2, aq.mean(), w, yerr=_sem(aq), capsize=3, color=POST_C,
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="0.25", elinewidth=1.0),
                   label=POST if pos == 0 else None)
            d = _cohend(ap, aq)
            if np.isfinite(d):
                top = max(ap.mean() + _sem(ap), aq.mean() + _sem(aq))
                ax.text(pos, top + 0.03, f"$d$={d:.2f}", ha="center", va="bottom",
                        fontsize=10.5, color="0.2")
            xt.append(pos)
            xtl.append(mlbl)
            pos += 1.0
        headers.append(((start + pos - 1.0) / 2,
                        f"{meth}\nPR, n={p['LDDT_r'].notna().sum()}"
                        f"/{q['LDDT_r'].notna().sum()}"))
    ax.axvline(len(METRICS) - 0.5, color="0.6", ls="--", lw=1.2)   # method divider
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, fontsize=14, rotation=45, ha="right")
    for cx, hdr in headers:
        ax.text(cx, -0.185, hdr, ha="center", va="top", fontsize=10.5,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.7, pos - 0.3)
    ax.legend(title="", loc="lower right", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5)


# --------------------------------------------------------------------------- #
def _letters(pairs):
    for ax, ltr in pairs:
        ax.text(-0.06, 1.06, ltr, transform=ax.transAxes, fontsize=22,
                fontweight="bold", va="bottom", ha="right")


def build():
    data = _load_all()
    for ctx in CTX_ORDER:
        for cut in (PRE, POST):
            d = data[ctx][cut]
            print(f"[{ctx:11s} {cut}] n={len(d):4d}  "
                  f"annotated={int((d['fam'] != 'None').sum()):3d}  "
                  f"med_len={d['RNALength'].median():.0f}")

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(24.0, 13.0))
        gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.26)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[0, 2])
        axD = fig.add_subplot(gs[1, 0])
        axE = fig.add_subplot(gs[1, 1])
        axF = fig.add_subplot(gs[1, 2])
        _panel_descriptor_d(axA, data)
        _panel_length(axB, data)
        _panel_family_composition(axC, data)
        _panel_family_quality(axD, data)
        _panel_method_composition(axE, data)
        _panel_method_quality(axF, data)
        _letters([(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"),
                  (axE, "E"), (axF, "F")])

        out = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                           "publication_figures")
        os.makedirs(out, exist_ok=True)
        paths = []
        for ext in ("png", "pdf"):
            p = os.path.join(out, f"postpre_family_comparison.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    print("Saved:", paths[0])
    return paths


if __name__ == "__main__":
    build()
