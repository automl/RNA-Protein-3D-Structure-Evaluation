"""
SI figure: held-out validation of the decision-framework reliability tiers.

  A  interface tier vs placement class of every protein-RNA RNA chain (pre/post cutoff)
  B  interface tier vs DockQ of all 15 AF3 samples of the 50 seed targets
  C  fold tier vs RNA all-atom LDDT in the multi-chain RNA set
  D  scored framework v2, fold levels: % well-folded RNAs (95% CI), pre- and post-cutoff
  E  scored framework v2, interface levels: % correct interfaces (95% CI), pre- and post-cutoff
  F  CLIP motif atlas: positional recovery of the motif-bearing footprint by v2 interface level

Reads the outputs of decision_tree.validate, decision_tree.develop and decision_tree.applications.
Run from the repo root (RNABench env):  python -m decision_tree.figure
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import figstyle
from decision_tree.rules import TIERS

OUT = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree")
TIER_C = {"good": "#59A14F", "mid": "#EDC948", "bad": "#E15759"}
PLACE = [("correct_interface", "Correct interface", "#59A14F"),
         ("right_protein_site_wrong_rna_pose", "Correct site, incorrect RNA", "#EDC948"),
         ("same_place_rna_turned", "Correct site, rotated RNA", "#76B7B2"),
         ("shifted_nearby", "Shifted site", "#FF9DA7"),
         ("different_region", "Distal site", "#9C755F"),
         ("protein_binding_region_mispredicted", "Incorrect binding site", "#B07AA1")]
MERGE = {"same_place_same_orientation": "right_protein_site_wrong_rna_pose"}


def _tsv(name):
    return list(csv.DictReader(open(os.path.join(OUT, name)), delimiter="\t"))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def panel_placement(ax):
    rows = _tsv("validation_placement.tsv")
    x, labels = [], []
    pos = 0
    for period, lab in (("pre", "pre"), ("post", "post")):
        for t in TIERS:
            d = [MERGE.get(r["placement"], r["placement"]) for r in rows if r["period"] == period and r["tier"] == t]
            bottom = 0.0
            for key, _, col in PLACE:
                frac = d.count(key) / len(d) if d else 0
                ax.bar(pos, frac, 0.8, bottom=bottom, color=col, edgecolor="white", linewidth=0.3)
                bottom += frac
            ax.text(pos, 1.01, f"{len(d)}", ha="center", va="bottom", fontsize=figstyle.FONT_SMALL - 0.5)
            x.append(pos); labels.append(t)
            pos += 1
        pos += 0.6
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Fraction of RNA chains")
    for i, lab in enumerate(("pre-cutoff", "post-cutoff")):
        ax.text(1 + i * 3.6, -0.11, lab, ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=figstyle.FONT_BASE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PLACE]
    ax.set_xlabel("interface tier", labelpad=16)
    ax.legend(handles, [l for _, l, _ in PLACE], loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False,
              ncol=3, fontsize=figstyle.FONT_SMALL, handlelength=0.9, handletextpad=0.3, columnspacing=0.8)


def _box(ax, groups, ylabel, thr=None):
    data = [g for _, g in groups]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for b, (t, _) in zip(bp["boxes"], groups):
        b.set(facecolor=TIER_C[t], alpha=0.75, linewidth=0.5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{t}\n{len(g)}" for t, g in groups])
    if thr is not None:
        ax.axhline(thr, color="0.4", ls="--", lw=0.6)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1)


def panel_seeds(ax):
    rows = _tsv("validation_seed_samples.tsv")
    _box(ax, [(t, [_f(r["dockq"]) for r in rows if r["tier"] == t]) for t in TIERS], "DockQ (sampled interfaces)", 0.23)


def panel_multichain(ax):
    rows = _tsv("validation_multichain_fold.tsv")
    _box(ax, [(t, [_f(r["rna_lddt"]) for r in rows if r["tier"] == t and np.isfinite(_f(r["rna_lddt"]))])
              for t in TIERS], "RNA LDDT (multi-chain RNA)", 0.85)
    ax.set_xlabel("fold tier")


def _cal_bars(ax, cal, names, ylabel):
    xs = np.arange(len(names))
    for j, (period, col, hatch) in enumerate((("pre", "#4E79A7", ""), ("post", "#E15759", ""))):
        rates = [cal[period][n]["rate"] if cal[period][n]["n"] else np.nan for n in names]
        lo = [cal[period][n]["ci_lo"] if cal[period][n]["n"] else np.nan for n in names]
        hi = [cal[period][n]["ci_hi"] if cal[period][n]["n"] else np.nan for n in names]
        err = np.array([np.subtract(rates, lo), np.subtract(hi, rates)])
        ax.bar(xs + (j - 0.5) * 0.38, rates, 0.36, color=col, alpha=0.8, label=f"{period}-cutoff",
               yerr=err, capsize=1.5, error_kw=dict(elinewidth=0.6), edgecolor="black", linewidth=0.3)
        for x, n in zip(xs + (j - 0.5) * 0.38, [cal[period][n]["n"] for n in names]):
            ax.text(x, 0.02, str(n), ha="center", va="bottom", fontsize=figstyle.FONT_SMALL - 1.5, rotation=90,
                    color="white")
    ax.set_xticks(xs); ax.set_xticklabels([n.title() for n in names], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=figstyle.FONT_SMALL, handlelength=0.9, ncol=2,
              columnspacing=0.8)


def panel_fold_levels(ax):
    import json
    cal = json.load(open(os.path.join(OUT, "calibration.json")))["fold"]["all"]
    _cal_bars(ax, cal, ["WEAK", "MODERATE", "STRONG"], "Well-folded RNAs\n(LDDT$_r>0.85$)")
    ax.set_xlabel("fold level (v2)")


def panel_interface_levels(ax):
    import json
    cal = json.load(open(os.path.join(OUT, "calibration.json")))["interface"]
    _cal_bars(ax, cal, ["UNRELIABLE", "CAUTION", "RELIABLE"], "Correct interfaces")
    ax.set_xlabel("interface level (v2)")


def panel_clip_levels(ax):
    rows = _tsv("applications_v2_clip.tsv")
    names = ["UNRELIABLE", "CAUTION", "RELIABLE"]
    xs = np.arange(3)
    for j, (k, lab, col, test) in enumerate((("recall", "footprint recall $\\geq$0.5", "#4E79A7", lambda v: v >= 0.5),
                                             ("same_site", "AF3 footprint on native site", "#F28E2B", lambda v: v > 0))):
        vals = []
        for n in names:
            v = np.array([_f(r[k]) for r in rows if r["level"] == n], float)
            v = v[np.isfinite(v)]
            vals.append(np.mean(test(v)) if len(v) else np.nan)
        ax.bar(xs + (j - 0.5) * 0.36, vals, 0.34, color=col, label=lab, edgecolor="black", linewidth=0.3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{n.title()} ({sum(r['level'] == n for r in rows)})" for n in names],
                                          rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("CLIP motif-atlas entries")
    ax.set_xlabel("interface level (v2)")
    ax.legend(loc="upper left", frameon=False, fontsize=figstyle.FONT_SMALL, handlelength=0.9)


def main():
    figstyle.apply("plain")
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 6.6))
    gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.75, width_ratios=[1.9, 1.0, 1.0])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axD, axE, axF = (fig.add_subplot(gs[1, i]) for i in range(3))
    panel_placement(axA); panel_seeds(axB); panel_multichain(axC)
    panel_fold_levels(axD); panel_interface_levels(axE); panel_clip_levels(axF)
    axB.set_xlabel("interface tier")
    for ax, l in zip((axA, axB, axC, axD, axE, axF), "ABCDEF"):
        figstyle.panel_letter(ax, l, x=-0.14 if l in "AD" else -0.30, y=1.04)
    base = figstyle.save(fig, os.path.join(OUT, "decision_tree_validation"), formats=("png", "pdf"))
    plt.close(fig)
    print("Saved:", base + ".png (+ .pdf)")


if __name__ == "__main__":
    main()
