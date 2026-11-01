#!/usr/bin/env python3
"""
ph_peg_composite -- combined publication figure on solution conditions:
  A  charged-residue occurrences by pH (from evaluation/phCorrelation.py, composite_panels)
  B  buffer type x concentration interaction on crTM (grouped bars)   [replaces DockQ-by-pH]
  C  crTM (RNAComplex_TM) by PEG concentration
  D  crTM (RNAComplex_TM) by PEG molecular weight
PEG/buffer data come from evaluation/peg_analysis_plots.py (PEGAnalysisPlotter).

Run from the repo root:
  python publication_figures/ph_peg_composite.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC, onpage_to_gen
from evaluation.phCorrelation import composite_panels
from evaluation.peg_analysis_plots import PEGAnalysisPlotter

# Larger, publication (Nature-style) fonts across all panels.
PUB_RC.update({
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

CRTM_COL = "RNAComplex_TM"
CRTM_LAB = r"TM$_{cr}$"
CONC_ORDER = ["No PEG", "PEG (0-10%)", "PEG (10-20%)", "PEG (20-30%)", "PEG (30-50%)"]
MW_ORDER = ["No PEG", "PEG 0-1k", "PEG 1k-2k", "PEG 2k-10k", "PEG 10k-20k"]
# buffer type display names; anything else -> "other"
BUF_NAME = {"hepes": "HEPES", "tris": "Tris", "phosphate": "phosphate",
            "mes": "MES", "citrate": "citrate"}
# buffer concentration bins to compare (raw label, display label)
CONC_BINS = [("0-0.1 M", "≤0.1 M"), ("0.1-0.5 M", "0.1–0.5 M")]
CONC_COL = {"0-0.1 M": "#4E79A7", "0.1-0.5 M": "#E15759"}


def _sem(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def _stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _short_peg(c):
    return c.replace("PEG ", "").replace("(", "").replace(")", "")


def _peg_by_cat(df, catcol, order, xlabel):
    """Mean crTM per category (bars, sem, n above), for one PEG factor."""
    d = df.copy()
    d[CRTM_COL] = pd.to_numeric(d[CRTM_COL], errors="coerce")
    d = d.dropna(subset=[CRTM_COL, catcol])
    cats = [c for c in order if c in set(d[catcol])]

    def draw(ax):
        x = np.arange(len(cats))
        means, sems, ns = [], [], []
        for c in cats:
            v = d.loc[d[catcol] == c, CRTM_COL].values
            means.append(v.mean() if len(v) else np.nan)
            sems.append(_sem(v)); ns.append(len(v))
        ax.bar(x, means, 0.62, yerr=sems, color="#59A14F", alpha=0.9,
               error_kw=dict(capsize=3, elinewidth=1.0, ecolor="0.3"))
        for xi, m, s, nn in zip(x, means, sems, ns):
            if nn > 0 and np.isfinite(m):
                ax.text(xi, m + s + 0.03, f"n={nn}", ha="center", va="bottom",
                        fontsize=onpage_to_gen(6.5, 11.6), color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_peg(c) for c in cats], rotation=20, ha="right",
                           rotation_mode="anchor", fontsize=onpage_to_gen(7, 11.6))
        ax.set_ylabel(CRTM_LAB, fontsize=onpage_to_gen(7.5, 11.6))
        ax.set_xlabel(xlabel, fontsize=onpage_to_gen(7.5, 11.6))
        ax.set_ylim(0, 1.14)
        ax.tick_params(axis="y", labelsize=onpage_to_gen(7, 11.6))
        style_axis(ax, grid_axis="y")

    return draw


def _buffer_panel(df):
    """crTM by buffer type (x) x concentration bin (hue): the buffer-type x concentration
    interaction, with per-bar n and within-buffer low-vs-high significance stars."""
    d = df.copy()
    d[CRTM_COL] = pd.to_numeric(d[CRTM_COL], errors="coerce")
    d = d.dropna(subset=[CRTM_COL])
    if "buffer_matches" not in d.columns or "buffer_category" not in d.columns:
        return lambda ax: ax.text(0.5, 0.5, "buffer data unavailable",
                                  transform=ax.transAxes, ha="center", va="center", color="grey")
    d["btype"] = d["buffer_matches"].apply(
        lambda x: BUF_NAME.get(str(x).strip().lower(), "other"))
    d = d[d["buffer_category"].isin([b[0] for b in CONC_BINS])]
    counts = d["btype"].value_counts()
    order = [b for b in counts.index if b != "other"]
    if "other" in counts.index:
        order.append("other")

    def draw(ax):
        x = np.arange(len(order))
        w = 0.38
        # collect per-category n-label slots so the two grouped bars' labels can
        # be vertically staggered when they would otherwise collide (e.g. two
        # short adjacent bars whose "n=" labels sit at the same height).
        label_slots = {i: [] for i in range(len(order))}
        for k, (cb, clab) in enumerate(CONC_BINS):
            means, sems, ns, xs = [], [], [], []
            for i, bt in enumerate(order):
                v = d.loc[(d["btype"] == bt) & (d["buffer_category"] == cb), CRTM_COL].values
                xs.append(i - w / 2 + k * w)
                means.append(v.mean() if len(v) else np.nan)
                sems.append(_sem(v)); ns.append(len(v))
            ax.bar(xs, means, w, yerr=sems, color=CONC_COL[cb], alpha=0.9, label=clab,
                   error_kw=dict(capsize=3, elinewidth=1.0, ecolor="0.3"))
            for i, (xi, m, s, nn) in enumerate(zip(xs, means, sems, ns)):
                if nn > 0 and np.isfinite(m):
                    label_slots[i].append([xi, m + s + 0.03, f"n={nn}"])
        for slots in label_slots.values():
            if len(slots) == 2 and abs(slots[0][1] - slots[1][1]) < 0.09:
                # lift the right-hand label onto a second tier for clearance
                tier = min(slots[0][1], slots[1][1]) + 0.11
                (slots[0] if slots[0][0] > slots[1][0] else slots[1])[1] = tier
            for xi, yi, txt in slots:
                ax.text(xi, yi, txt, ha="center", va="bottom",
                        fontsize=onpage_to_gen(6.5, 11.6), color="0.3")
        # within-buffer low-vs-high concentration significance
        for i, bt in enumerate(order):
            lo = d.loc[(d["btype"] == bt) & (d["buffer_category"] == "0-0.1 M"), CRTM_COL].values
            hi = d.loc[(d["btype"] == bt) & (d["buffer_category"] == "0.1-0.5 M"), CRTM_COL].values
            if len(lo) >= 3 and len(hi) >= 3:
                try:
                    _u, p = mannwhitneyu(lo, hi)
                except ValueError:
                    p = np.nan
                s = _stars(p)
                if s:
                    top = max(np.nanmean(lo) + _sem(lo), np.nanmean(hi) + _sem(hi))
                    ax.text(i, top + 0.10, s, ha="center", va="bottom",
                            fontsize=onpage_to_gen(6.5, 11.6))
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha="right", rotation_mode="anchor",
                           fontsize=onpage_to_gen(7, 11.6))
        ax.set_ylabel(CRTM_LAB, fontsize=onpage_to_gen(7.5, 11.6))
        ax.set_xlabel("Buffer type", fontsize=onpage_to_gen(7.5, 11.6))
        ax.set_ylim(0, 1.20)
        ax.tick_params(axis="y", labelsize=onpage_to_gen(7, 11.6))
        ax.legend(fontsize=onpage_to_gen(7, 11.6), title="Buffer conc.",
                  title_fontsize=onpage_to_gen(7, 11.6),
                  frameon=False, loc="upper right")
        style_axis(ax, grid_axis="y")

    return draw


def main():
    panels = composite_panels(["charged_residue_occurrences_by_ph"])
    df = PEGAnalysisPlotter().df

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
    pf = PublicationFigure(ncols=2, panel_size=(5.8, 4.8), letters=True, outdir=outdir)
    pf.add_panel(panels["charged_residue_occurrences_by_ph"][0], title=None)          # A
    pf.add_panel(_buffer_panel(df), title=None)                                       # B
    pf.add_panel(_peg_by_cat(df, "PEG_concentration", CONC_ORDER, "PEG concentration"), title=None)  # C
    pf.add_panel(_peg_by_cat(df, "PEG_MW_category", MW_ORDER, "PEG MW"), title=None)   # D
    return pf.save("ph_peg_composite")


if __name__ == "__main__":
    main()
