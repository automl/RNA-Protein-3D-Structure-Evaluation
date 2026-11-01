"""SI figure: one example per input of the controlled ablation, with the 4 x 4 factorial tile of its score.
Run from the repo root (RNABench env). Output: 2025_NAR_AF3_Evaluation/figures2/si_ex_ablation.{png,pdf}"""
import os, sys
sys.path[:0] = [".", "publication_figures"]
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import ablation_data as A
import figstyle

SC = "results/plots/all_chains/publication_figures/site_chosen"
OUT = "2025_NAR_AF3_Evaluation/figures2/si_ex_ablation"
M = r"$_{r}$"
# (letter, target, side, metric, metric label, [(image, label, condition (rMSA, pMSA, pTemp, cTemp))] * 2)
ROWS = [
    ("A", "7UO5", "protein_side", "RNA_TM", "TM$_\mathregular{r}$",
     [("interface/msa_7UO5_with", "+MSA$_\mathregular{r}$", (1, 1, 1, 1)), ("interface/msa_7UO5_without", "−MSA$_\mathregular{r}$", (0, 1, 1, 1))]),
    ("B", "7YFQ", "protein_side", "Complex_TM", "TM$_\mathregular{c}$",
     [("ablation/7YFQ_all", "+MSA$_\mathregular{p}$", (1, 1, 1, 1)), ("ablation/7YFQ_noPMSA", "−MSA$_\mathregular{p}$", (1, 0, 1, 1))]),
    ("C", "8SNX", "rna_side", "Complex_TM", "TM$_\mathregular{c}$",
     [("ablation/8SNX_noPMSA", "−MSA$_\mathregular{p}$, +Tmpl$_\mathregular{p}$", (1, 0, 1, 1)),
      ("ablation/8SNX_noPMSA_noPTemp", "−MSA$_\mathregular{p}$, −Tmpl$_\mathregular{p}$", (1, 0, 0, 1))]),
    ("D", "8PDS", "rna_side", "DockQ_Nr", "DockQ$_\mathregular{Nr}$",
     [("ablation/8PDS_all", "+Tmpl$_\mathregular{c}$", (1, 1, 1, 1)), ("ablation/8PDS_noCTemp", "−Tmpl$_\mathregular{c}$", (1, 1, 1, 0))]),
]
RLAB = ["+MSA$_\mathregular{r}$ +MSA$_\mathregular{p}$", "+MSA$_\mathregular{r}$ −MSA$_\mathregular{p}$", "−MSA$_\mathregular{r}$ +MSA$_\mathregular{p}$", "−MSA$_\mathregular{r}$ −MSA$_\mathregular{p}$"]
CLAB = ["+Tmpl$_\mathregular{p}$\n+Tmpl$_\mathregular{c}$", "+Tmpl$_\mathregular{p}$\n−Tmpl$_\mathregular{c}$", "−Tmpl$_\mathregular{p}$\n+Tmpl$_\mathregular{c}$", "−Tmpl$_\mathregular{p}$\n−Tmpl$_\mathregular{c}$"]


def img(path):
    im = Image.open(f"{SC}/{path}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg.convert("RGB")
    box = Image.eval(im.convert("L"), lambda v: 255 - v).point(lambda v: 255 if v > 8 else 0).getbbox()
    return np.asarray(im.crop(box))


def grid(fac, target, side, metric):
    g = fac[(fac.exp_db_id == target) & (fac.side == side)]
    out = np.full((4, 4), np.nan)
    for r in g.itertuples():
        i = (1 - r.rMSA) * 2 + (1 - r.pMSA); j = (1 - r.pTemp) * 2 + (1 - r.cTemp)
        out[i, j] = getattr(r, metric)
    return out


def cell(cond):
    r, p, t, c = cond
    return (1 - r) * 2 + (1 - p), (1 - t) * 2 + (1 - c)


def main():
    figstyle.apply("plain")
    fac = A.load_factorial(verbose=False)
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 9.6))
    gs = fig.add_gridspec(len(ROWS), 3, width_ratios=[1, 1, 0.95], hspace=0.35, wspace=0.12)
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)        # cell edges in figure coordinates
    for k, (letter, t, side, metric, mlab, pans) in enumerate(ROWS):
        # all labels of a row at one height: the top of the row's grid cells (the images keep their own
        # aspect ratio, so axes titles would sit at different heights)
        ytop = tops[k] + 0.006                        # just above the tallest panel of the row
        for j, (path, lab, _cond) in enumerate(pans):
            ax = fig.add_subplot(gs[k, j]); ax.imshow(img(path)); ax.set_axis_off()
            ax.set_anchor("S")
            fig.text((lefts[j] + rights[j]) / 2, ytop, lab, ha="center", va="baseline", fontsize=figstyle.FONT_BASE)
            if j == 0:
                fig.text(lefts[j] - 0.01, ytop, letter, ha="left", va="baseline", fontsize=figstyle.FONT_BASE + 2,
                         fontweight="bold")
        ax = fig.add_subplot(gs[k, 2])
        v = grid(fac, t, side, metric)
        ax.imshow(v, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        for i in range(4):
            for j in range(4):
                if np.isfinite(v[i, j]):
                    ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=figstyle.FONT_SMALL - 1,
                            color="white" if v[i, j] < 0.55 else "black")
        for _p, _l, cond in pans:
            i, j = cell(cond)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#D7301F", lw=1.6))
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLAB if k == len(ROWS) - 1 else [""] * 4, fontsize=figstyle.FONT_SMALL - 1.5)
        ax.set_yticklabels(RLAB, fontsize=figstyle.FONT_SMALL - 1.5)
        ax.yaxis.tick_right(); ax.tick_params(length=0)
        ax.set_anchor("S")
        fig.text((lefts[2] + rights[2]) / 2, ytop, mlab, ha="center", va="baseline", fontsize=figstyle.FONT_BASE)  # PDB ids go in the caption
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=400, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
