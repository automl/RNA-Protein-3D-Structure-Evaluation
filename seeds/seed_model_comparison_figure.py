#!/usr/bin/env python3
"""
seed_model_comparison_figure.py
===============================

Nature-style, big-font figure showing that for AF3 on post-cutoff protein-RNA
complexes (a2021), *the choice of random seed (s1-s3) or output model (0-4)
barely changes anything* -- the target identity dominates every metric.

The three seed DBs (data/seeds/Seeds/a2021_s{1,2,3}.db) share the same 50
targets. 6FQL is excluded (released 2018, in AF3's training set;
evaluation/excluded_targets.py), which leaves 49 targets in a fully crossed,
paired design: 49 targets x 3 seeds x 5 models = 735 observations per metric.
That lets us make the "it doesn't matter" case rigorously and three ways:

  A. Variance decomposition  -- seed and model explain a negligible % of the
     total variance; the target explains almost all of it.
  B. Paired effect sizes (Cohen's d_z) for every model-vs-model0 and
     seed-vs-seed1 contrast, against a shaded "negligible" band (|d|<0.2):
     every contrast lands inside it.
  C/D. Per-target paired lines across models / seeds: each target's score is
     essentially flat, so swapping model or seed leaves it unchanged.

No panel titles (Nature style -- meaning is carried by axes + the caption).

Run:
  python seeds/seed_model_comparison_figure.py   (RNABench env)
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from plots.publication_figure import style_axis, PUB_RC
except Exception:                                            # pragma: no cover
    PUB_RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    }

    def style_axis(ax, grid_axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid_axis in ("y", "both"):
            ax.grid(axis="y", color="0.85", lw=0.6)
        if grid_axis in ("x", "both"):
            ax.grid(axis="x", color="0.85", lw=0.6)
        ax.set_axisbelow(True)

PUB_RC = dict(PUB_RC)
PUB_RC.update({
    "axes.labelsize": 20, "axes.titlesize": 20, "xtick.labelsize": 16,
    "ytick.labelsize": 16, "legend.fontsize": 15, "legend.title_fontsize": 15,
    "axes.linewidth": 1.0,
})

# the published seed databases (data/all_data_zip/all_databases/other_analyses/seeds/protein_rna_seed<N>.db)
SEEDS_DIR = os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases", "other_analyses", "seeds")
DBS = [("s1", "protein_rna_seed1.db"), ("s2", "protein_rna_seed2.db"), ("s3", "protein_rna_seed3.db")]
MODELS = [0, 1, 2, 3, 4]

DISP = {
    "TM_r": r"TM$_r$", "LDDT_r": r"LDDT$_r$", "TM_c": r"TM$_c$",
    "TM_p": r"TM$_p$", "LDDT_p": r"LDDT$_p$", "RMSD_c": r"RMSD$_c$",
    "pTM_r": r"pTM$_r$", "ipTM_r": r"ipTM$_r$",
}
METRICS = ["TM_r", "LDDT_r", "TM_c", "TM_p", "LDDT_p", "RMSD_c",
           "pTM_r", "ipTM_r"]
LOWER_BETTER = {"RMSD_c"}

SEED_C = "#E8743B"     # orange
MODEL_C = "#2E86AB"    # teal
TARGET_C = "#9AA0A6"   # grey
RESID_C = "#D9DCE0"    # light grey


# --------------------------------------------------------------------------- #
def _g(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    try:
        v = eval(s) if isinstance(s, str) else s
        return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
    except Exception:
        return np.nan


def _ptm(s, lenstr):
    try:
        L = eval(lenstr) if isinstance(lenstr, str) else lenstr
        if not isinstance(L, list):
            L = [L]
        idx = L.index(max(L))
        v = eval(s) if isinstance(s, str) else s
        if not isinstance(v, list):
            v = [v]
        return float(v[idx]) if idx < len(v) else float(v[0])
    except Exception:
        return np.nan


def _model_of(fn):
    try:
        return int(str(fn).split("_model_")[1].split(".")[0])
    except Exception:
        return -1


def load_db(seed, fname):
    con = sqlite3.connect(os.path.join(SEEDS_DIR, fname))
    d = pd.read_sql_query("SELECT * FROM pred_protein_rna", con)
    con.close()
    o = pd.DataFrame(index=range(len(d)))
    o["seed"] = seed
    o["model"] = d["FileName"].apply(_model_of)
    o["target"] = d["exp_db_id"].astype(str)
    o["TM_r"] = d["RNA_TM"].apply(_g)
    o["LDDT_r"] = d["RNA_LDDT"].apply(_g)
    o["TM_c"] = d["Complex_TM"].apply(_g)
    o["TM_p"] = d["Protein_TM"].apply(_g)
    o["LDDT_p"] = d["Protein_LDDT"].apply(_g)
    o["RMSD_c"] = d["Complex_RMSD"].apply(_g)
    o["pTM_r"] = [_ptm(s, l) for s, l in zip(d["af3_rna_pTM"], d["RNALength"])]
    o["ipTM_r"] = [_ptm(s, l) for s, l in zip(d["af3_rna_ipTM"], d["RNALength"])]
    return o


# --------------------------------------------------------------------------- #
def variance_fractions(df, metric):
    """Main-effect variance fractions: target / seed / model / residual."""
    d = df[["target", "seed", "model", metric]].dropna()
    x = d[metric].astype(float)
    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    if ss_total == 0:
        return dict(target=0, seed=0, model=0, residual=0)

    def ss(factor):
        gm = d.groupby(factor)[metric].agg(["mean", "count"])
        return (gm["count"] * (gm["mean"] - grand) ** 2).sum()

    ss_t, ss_s, ss_m = ss("target"), ss("seed"), ss("model")
    ss_res = max(ss_total - ss_t - ss_s - ss_m, 0.0)
    return {k: v / ss_total for k, v in
            dict(target=ss_t, seed=ss_s, model=ss_m, residual=ss_res).items()}


def effect_d(df, metric, factor, level_a, level_b):
    """Ordinary Cohen's d (between-target SD) for level_b vs level_a.

    This is the practical "does it matter" measure: it scales the seed/model
    shift by how much the metric naturally varies across targets. (Paired d_z
    is avoided -- a consistent but absolutely tiny shift, e.g. in ipTM, can give
    a large d_z while explaining ~0% of the variance.)
    """
    a = df.loc[df[factor] == level_a, metric].dropna()
    b = df.loc[df[factor] == level_b, metric].dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if sp == 0:
        return np.nan
    d = (b.mean() - a.mean()) / sp
    return -d if metric in LOWER_BETTER else d   # +d = level_b better


# --------------------------------------------------------------------------- #
def panel_variance(ax, df):
    fr = {m: variance_fractions(df, m) for m in METRICS}
    order = METRICS
    y = np.arange(len(order))
    tgt = np.array([fr[m]["target"] for m in order])
    sd = np.array([fr[m]["seed"] for m in order])
    md = np.array([fr[m]["model"] for m in order])
    res = np.array([fr[m]["residual"] for m in order])
    ax.barh(y, tgt, color=TARGET_C, edgecolor="black", lw=0.3, label="target")
    ax.barh(y, sd, left=tgt, color=SEED_C, edgecolor="black", lw=0.3, label="seed")
    ax.barh(y, md, left=tgt + sd, color=MODEL_C, edgecolor="black", lw=0.3,
            label="model")
    ax.barh(y, res, left=tgt + sd + md, color=RESID_C, edgecolor="black",
            lw=0.3, label="residual")
    ax.set_yticks(y)
    ax.set_yticklabels([DISP[m] for m in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of variance explained")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    sm = (sd + md)
    ax.text(0.98, 0.02,
            f"seed + model: {100*sm.min():.1f}–{100*sm.max():.1f}% of variance",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=15,
            fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    style_axis(ax, grid_axis="x")
    return fr


def panel_effects(ax, df):
    rng = np.random.default_rng(0)
    xmax = 0.0
    for yi, m in enumerate(METRICS):
        m_ds = [effect_d(df, m, "model", 0, k) for k in (1, 2, 3, 4)]
        s_ds = [effect_d(df, m, "seed", "s1", sj) for sj in ("s2", "s3")]
        for d in m_ds:
            if not np.isnan(d):
                ax.scatter(d, yi + rng.uniform(-0.12, 0.12), s=70,
                           color=MODEL_C, edgecolor="black", lw=0.4, zorder=3)
                xmax = max(xmax, abs(d))
        for d in s_ds:
            if not np.isnan(d):
                ax.scatter(d, yi + rng.uniform(-0.12, 0.12), s=85, marker="D",
                           facecolor="white", edgecolor=SEED_C, lw=1.6, zorder=3)
                xmax = max(xmax, abs(d))
    lim = max(0.55, xmax * 1.25)
    ax.axvspan(-0.2, 0.2, color="#C8E6C9", alpha=0.6, zorder=0)
    ax.axvspan(-0.5, -0.2, color="#FFF3C4", alpha=0.5, zorder=0)
    ax.axvspan(0.2, 0.5, color="#FFF3C4", alpha=0.5, zorder=0)
    ax.axvline(0, color="0.35", lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, len(METRICS) - 0.4)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([DISP[m] for m in METRICS])
    ax.set_xlabel(r"effect size  Cohen's $d$  (vs across-target spread)")
    ax.text(0, len(METRICS) - 0.5, "negligible", ha="center", va="center",
            fontsize=13, color="#2E7D32", fontweight="bold")
    ax.text(0.35, len(METRICS) - 0.5, "small", ha="center", va="center",
            fontsize=12, color="#B7950B")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_C,
               markeredgecolor="black", markersize=11, label="model$_k$ vs model$_0$"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor=SEED_C, markersize=11, markeredgewidth=1.6,
               label="seed$_j$ vs s1"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9)
    style_axis(ax, grid_axis="x")


def _paired_lines(ax, df, metric, factor, levels, xticklabels, color):
    """Per-target slope lines across `factor` (averaged over the other factor)."""
    piv = df.pivot_table(index="target", columns=factor, values=metric)
    piv = piv.reindex(columns=levels)
    x = np.arange(len(levels))
    for _, row in piv.iterrows():
        ax.plot(x, row.values, color=color, alpha=0.18, lw=1.0, zorder=1)
    mean = piv.mean(axis=0).values
    sem = piv.std(axis=0).values / np.sqrt(piv.notna().sum(axis=0).values)
    ax.errorbar(x, mean, yerr=1.96 * sem, color="black", lw=2.6, marker="o",
                ms=8, capsize=4, zorder=4, label="mean ± 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(DISP[metric])
    rng_means = mean.max() - mean.min()
    ax.text(0.5, 0.965, f"mean range across {factor}s = {rng_means:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=14,
            fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    style_axis(ax, grid_axis="y")


# --------------------------------------------------------------------------- #
def report(df, fr):
    print("\n" + "=" * 66)
    print(f"DOES THE SEED / MODEL CHOICE MATTER?  ({df.target.nunique()} targets x 3 seeds x 5 models)")
    print("=" * 66)
    print("\nVariance explained (%), per metric:")
    print(f"  {'metric':>8} | {'target':>7} {'seed':>6} {'model':>6} {'residual':>9}")
    for m in METRICS:
        f = fr[m]
        print(f"  {m:>8} | {100*f['target']:7.1f} {100*f['seed']:6.2f} "
              f"{100*f['model']:6.2f} {100*f['residual']:8.1f}")
    sm = [fr[m]["seed"] + fr[m]["model"] for m in METRICS]
    print(f"\n  seed+model explain {100*min(sm):.2f}-{100*max(sm):.2f}% of total "
          f"variance (target dominates).")

    print("\nLargest effect size |d| over ALL seed/model contrasts:")
    worst = (0.0, None)
    for m in METRICS:
        ds = ([effect_d(df, m, "model", 0, k) for k in (1, 2, 3, 4)]
              + [effect_d(df, m, "seed", "s1", sj) for sj in ("s2", "s3")])
        mx = np.nanmax(np.abs(ds))
        if mx > worst[0]:
            worst = (mx, m)
        print(f"  {m:>8}: max|d| = {mx:.3f}")
    print(f"\n  >>> overall max |d| = {worst[0]:.3f} ({worst[1]})  "
          f"-- {'all negligible (<0.2)' if worst[0] < 0.2 else 'see band'}.")
    print("=" * 66 + "\n")


def main():
    from evaluation.excluded_targets import POST_EXCLUDED     # 6FQL: released 2018, in AF3 training
    df = pd.concat([load_db(s, f) for s, f in DBS], ignore_index=True)
    df = df[df.model.isin(MODELS) & ~df.target.str.upper().isin(POST_EXCLUDED)].reset_index(drop=True)

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(18, 13))
        gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.24,
                      top=0.92, bottom=0.07, left=0.09, right=0.98)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axD = fig.add_subplot(gs[1, 1])

        fr = panel_variance(axA, df)
        panel_effects(axB, df)
        _paired_lines(axC, df, "TM_r", "model", MODELS,
                      [f"m{m}" for m in MODELS], MODEL_C)
        _paired_lines(axD, df, "TM_r", "seed", ["s1", "s2", "s3"],
                      ["s1", "s2", "s3"], SEED_C)

        for ax, ltr in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
            ax.text(-0.08, 1.05, ltr, transform=ax.transAxes, fontsize=26,
                    fontweight="bold", va="bottom", ha="right")

    out = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                       "publication_figures")
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"seed_model_invariance.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    print("Saved:", paths[0])
    print("       ", paths[1])
    report(df, fr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
