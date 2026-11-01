#!/usr/bin/env python3
"""
af3_clip_atlas_motif_figure -- atlas RBP motifs vs experimental (PDB) and AF3 interface
footprints.

An interface footprint is a maximal run of consecutive protein-contacted RNA
positions, rendered back as sequence: the structural analogue of a CLIP k-mer.
Each footprint is scored against its RBP's position weight matrices with
partial-overlap log-odds, and against a null in which the same footprints are
scored against *other* RBPs' motifs, so z measures motif specificity.

  A  z(experimental) vs z(AF3), coloured by RBP recognition class
  B  motif specificity by recognition class, experimental vs AF3
  C  motif provenance: in-vitro (assay-matched to crystallography) vs CLIP (in vivo)
  D  atlas motif consensus aligned to the experimental and AF3 footprints

Run from the repo root with the RNABench env:
  ~/miniconda3/envs/RNABench/bin/python publication_figures/af3_clip_atlas_motif_figure.py
"""
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "clip", "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from plots.publication_figure import style_axis, sns_violin
import figstyle

# Drawn at the NAR double-column width with the central figstyle sizes, so every text
# element keeps its point size in the paper (included at scale 1).
FS, FL = figstyle.FONT_BASE, figstyle.FONT_LEGEND     # labels/ticks, legends/annotations
FSEQ = 7                                              # monospace motif text in panel D

CLIP_DIR = os.path.join(_REPO_ROOT, "clip")
# Colour convention, matched to the PyMOL structure figures in the draft
# (deposited structure wheat/orange, AlphaFold3 prediction green):
#   PDB / experimental -> orange      AF3 / predicted -> green
# Okabe-Ito pair, colour-blind safe. RBP recognition class is therefore encoded on a
# separate blue/purple/red ramp so it never competes with the PDB-vs-AF3 distinction.
EXP_C, PRED_C = "#D55E00", "#009E73"
CLASS_STYLE = {
    "sequence":  ("#4E79A7", "sequence-specific"),
    "structure": ("#9B5DE5", "structure-driven"),
    "guide":     ("#E15759", "guide-directed"),
}
BASE_C = {"A": "#3b8a3b", "C": "#2c6fbb", "G": "#e8a33d", "U": "#c0392b", "T": "#c0392b"}


def _fl(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _pfmt(p):
    """Format a p-value for display.

    Small p-values are shown in scientific notation rather than rounded to three
    decimals: a fixed-point format turns 6.4e-05 into "0.000", which reads as a
    p-value of exactly zero. A p-value is never zero, only bounded above.
    """
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        return f"{p / 10**exp:.1f}" + r"$\times 10^{" + str(exp) + r"}$"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _rows():
    with open(os.path.join(CLIP_DIR, "results", "entry_scores.tsv")) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    cls = {}
    with open(os.path.join(CLIP_DIR, "scripts", "rbp_class.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            cls[r["gene"]] = r["class"]
    for r in rows:
        r["class"] = cls.get(r["gene"], "sequence")
    return rows


# ----------------------------------------------------------------- panel A -- #
def _panel_scatter(ax):
    rows = [r for r in _rows()
            if not np.isnan(_fl(r["z_all_exp"])) and not np.isnan(_fl(r["z_all_pred"]))]
    x = np.array([_fl(r["z_all_exp"]) for r in rows])
    y = np.array([_fl(r["z_all_pred"]) for r in rows])
    lim = (min(x.min(), y.min()) - 0.35, max(x.max(), y.max()) + 0.35)
    ax.plot(lim, lim, ls="--", color="grey", lw=0.9, zorder=1)
    for c, (col, lab) in CLASS_STYLE.items():
        for coh, mk, sz in (("pre", "o", 52), ("post", "^", 78)):
            s = [r for r in rows if r["class"] == c and r["cohort"] == coh]
            if not s:
                continue
            ax.scatter([_fl(r["z_all_exp"]) for r in s], [_fl(r["z_all_pred"]) for r in s],
                       s=sz / 3.5, marker=mk, color=col, edgecolor="black", linewidth=0.4,
                       alpha=0.85, label=lab if coh == "pre" else None, zorder=3)
    rp = stats.pearsonr(x, y)
    ax.text(0.04, 0.95, f"Pearson $r$ = {rp.statistic:.3f}\n$P$ = {rp.pvalue:.1e}\n$n$ = {len(rows)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=FL)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Motif-specificity $z$, experimental (PDB)", labelpad=6)
    ax.set_ylabel("Motif-specificity $z$, AF3")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], marker="o", linestyle="", markersize=4.5, markerfacecolor="#999999",
                 markeredgecolor="black", label="pre-cutoff (trained on)"),
          Line2D([0], [0], marker="^", linestyle="", markersize=5, markerfacecolor="#999999",
                 markeredgecolor="black", label="post-cutoff (novel)")]
    l += ["pre-cutoff (trained on)", "post-cutoff (novel)"]
    ax.legend(h, l, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    style_axis(ax, grid_axis="both")


# ----------------------------------------------------------------- panel B -- #
def _panel_by_class(ax):
    rows = _rows()
    order = ["sequence", "structure", "guide"]
    xs, width = np.arange(len(order)), 0.20
    combos = ((-1.5, "pre",  "z_all_exp",  EXP_C,  "PDB, pre-cutoff",  ""),
              (-0.5, "pre",  "z_all_pred", PRED_C, "AF3, pre-cutoff",  ""),
              (+0.5, "post", "z_all_exp",  EXP_C,  "PDB, post-cutoff", "//"),
              (+1.5, "post", "z_all_pred", PRED_C, "AF3, post-cutoff", "//"))
    for mult, coh, key, col, lab, hatch in combos:
        means, errs = [], []
        for c in order:
            v = [_fl(r[key]) for r in rows
                 if r["class"] == c and r["cohort"] == coh and not np.isnan(_fl(r[key]))]
            means.append(np.mean(v) if v else np.nan)
            errs.append(np.std(v, ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(xs + mult*width, means, width, yerr=errs, capsize=1.5, color=col,
               edgecolor="black", linewidth=0.4, alpha=0.9, label=lab, hatch=hatch,
               error_kw=dict(elinewidth=0.6),
               zorder=2)
    sq = [_fl(r["z_all_exp"]) for r in rows
          if r["class"] == "sequence" and not np.isnan(_fl(r["z_all_exp"]))]
    ns = [_fl(r["z_all_exp"]) for r in rows
          if r["class"] in ("structure", "guide") and not np.isnan(_fl(r["z_all_exp"]))]
    p = stats.mannwhitneyu(sq, ns).pvalue
    ax.axhline(0, color="grey", lw=0.9, ls="--")
    ax.text(0.5, 1.02,
            f"sequence vs non-sequence: Mann-Whitney $P$ = {_pfmt(p)}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=FL)
    ns_lab = []
    for c in order:
        n = sum(1 for r in rows if r["class"] == c and not np.isnan(_fl(r["z_all_exp"])))
        wrapped = CLASS_STYLE[c][1].replace("-", "-\n")
        ns_lab.append(wrapped + "\n($n$=" + str(n) + ")")
    ax.set_xticks(xs); ax.set_xticklabels(ns_lab)
    ax.set_ylabel("Motif-specificity $z$")
    ax.set_ylim(top=max(2.6, ax.get_ylim()[1]))
    ax.legend(loc="upper right", ncol=1, frameon=False, handlelength=1.6, borderaxespad=0.2)
    style_axis(ax, grid_axis="y")


# ----------------------------------------------------------------- panel C -- #
def _panel_provenance(ax):
    rows = _rows()
    labels = ["in vitro\n(SELEX, RNAcompete)", "in vivo\n(CLIP)"]
    data, colors = [], []
    for arm in ("invitro", "clip"):
        for key, col in ((f"z_{arm}_exp", EXP_C), (f"z_{arm}_pred", PRED_C)):
            data.append([_fl(r[key]) for r in rows if not np.isnan(_fl(r[key]))])
            colors.append(col)
    sns_violin(ax, data, ["PDB", "AF3", "PDB", "AF3"], colors=colors,
               ylabel="Motif-specificity $z$")
    ax.axhline(0, color="grey", lw=0.9, ls="--")
    # group labels sit just under the PDB/AF3 tick labels, not at the panel foot
    for i, lab in enumerate(labels):
        ax.text(0.5 + 2*i, -0.10, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS)
    iv = [_fl(r["z_invitro_exp"]) for r in rows if not np.isnan(_fl(r["z_invitro_exp"]))]
    cl = [_fl(r["z_clip_exp"]) for r in rows if not np.isnan(_fl(r["z_clip_exp"]))]
    u = stats.mannwhitneyu(iv, cl)
    ax.text(0.5, 1.02,
            f"in vitro $n$={len(iv)} vs in vivo $n$={len(cl)}\n"
            f"Mann-Whitney $P$ = {_pfmt(u.pvalue)}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=FL)
    ax.legend(handles=[
        Line2D([0], [0], marker="s", linestyle="", markersize=6, markerfacecolor=EXP_C,
               markeredgecolor="black", label="experimental (PDB)"),
        Line2D([0], [0], marker="s", linestyle="", markersize=6, markerfacecolor=PRED_C,
               markeredgecolor="black", label="AF3")],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    style_axis(ax, grid_axis="y")


# ----------------------------------------------------------------- panel D -- #
def _panel_alignment(ax):
    """Atlas motif vs the PDB and AF3 footprints, 10 CLIP-derived and 10 in-vitro genes.

    Where AF3 (or the structure) has no run of at least 4 nt, the longest sub-threshold
    contact run is shown instead, dimmed and marked -- those are real contacts that are
    simply too short to align to a PPM, so reporting them as absent would mislead.
    """
    rows = _rows()
    blocks = []
    for arm, label in (("clip", "CLIP-derived motifs"), ("invitro", "In-vitro motifs")):
        best = {}
        for r in rows:
            if int(r[f"n_motif_{arm}"]) == 0 or np.isnan(_fl(r[f"z_{arm}_exp"])):
                continue
            g = r["gene"]
            if g not in best or _fl(r[f"z_{arm}_exp"]) > _fl(best[g][f"z_{arm}_exp"]):
                best[g] = r
        sel = sorted(best.values(), key=lambda r: -_fl(r[f"z_{arm}_exp"]))[:10]
        blocks.append((label, arm, sel))

    CW, GAP = 1.0, 2.0
    def cell(r, arm, t):
        fp = r[f"fp_{arm}_{t}"]
        return (fp, False) if fp else (r[f"anyrun_{t}"] or "-", True)

    cols = [[r[f"motif_{arm}_exp"] for _, arm, sel in blocks for r in sel],
            [cell(r, arm, "exp")[0] for _, arm, sel in blocks for r in sel],
            [cell(r, arm, "pred")[0] for _, arm, sel in blocks for r in sel]]
    widths = [max(len(v) for v in c) for c in cols]
    x0 = [0.0]
    for w in widths[:-1]:
        x0.append(x0[-1] + w * CW + GAP)
    x_coh = x0[-1] + widths[-1] * CW + GAP

    def draw(seq, x, y, dim=False):
        for i, ch in enumerate(seq):
            ax.text(x + i * CW, y, ch, fontsize=FSEQ, family="DejaVu Sans Mono",
                    ha="left", va="center", fontweight="bold",
                    color="#b9b9b9" if dim else BASE_C.get(ch, "#444444"))

    y = 0.0
    for label, arm, sel in blocks:
        ax.text(-GAP, y, label, fontsize=FS, ha="right", va="center",
                fontweight="bold", style="italic",
                color="#1a9850" if arm == "clip" else "#7a7a7a")
        y -= 1.0
        for r in sel:
            ax.text(-GAP, y, f"{r['gene']} {r['pdb']}", fontsize=FSEQ, ha="right", va="center")
            draw(r[f"motif_{arm}_exp"], x0[0], y)
            for k, t in ((1, "exp"), (2, "pred")):
                txt, dim = cell(r, arm, t)
                draw(txt, x0[k], y, dim=dim)
                if dim:
                    ax.text(x0[k] + len(txt) * CW + 0.3, y, "$^{<4}$", fontsize=FSEQ,
                            ha="left", va="center", color="#b9b9b9")
            ax.text(x_coh, y, r["cohort"], fontsize=FSEQ, ha="left", va="center",
                    color="#111111",
                    fontweight="bold" if r["cohort"] == "post" else "normal")
            y -= 1.0
        y -= 0.5

    for x, lab in ((x0[0], "Atlas\nmotif"), (x0[1], "PDB\nmotifs"),
                   (x0[2], "AF3\nmotifs"), (x_coh, "Data-\nset")):
        ax.text(x, 1.6, lab, fontsize=FSEQ, ha="left", va="center", fontweight="bold",
                linespacing=1.0)
    # gene/PDB labels sit left of the axes (not clipped), so the full axes width goes to
    # the sequence columns and one data unit stays wider than a monospace character
    ax.set_xlim(-0.5, x_coh + 4.5)
    ax.set_ylim(y + 0.6, 2.6)
    ax.axis("off")


def main():
    figstyle.apply("plain")
    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 7.4))
    gs = fig.add_gridspec(2, 1, hspace=0.36, height_ratios=[1, 1.15])
    top = gs[0].subgridspec(1, 2, wspace=0.34)
    # bottom row: panel D gets the wider column; its row labels are drawn left of its axes
    bot = gs[1].subgridspec(1, 2, wspace=0.62, width_ratios=[0.8, 1.2])
    axs = np.array([[fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])],
                    [fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])]])
    _panel_scatter(axs[0, 0])
    _panel_by_class(axs[0, 1])
    _panel_provenance(axs[1, 0])
    _panel_alignment(axs[1, 1])
    for ax, letter in zip(axs.flat, "ABCD"):
        figstyle.panel_letter(ax, letter, x=-0.36 if letter == "D" else -0.16, y=1.06)
    base = figstyle.save(fig, os.path.join(outdir, "af3_clip_atlas_motifs"), formats=("png", "pdf"))
    plt.close(fig)
    print("Saved:", base + ".png (+ .pdf)")
    return base


if __name__ == "__main__":
    main()
