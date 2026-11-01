"""Panel D of interface_failure_modes for the dry-run classifications b and b.1 (evaluation/placement_all_chains.py).

Rows: share of a post-cutoff complex's interfaces in each class (correct interface and every class with >= MIN_N
post-cutoff interfaces); columns: the descriptors of interface_failure_modes panel D; cells: Spearman rho.
b2 / b2.1 = b / b.1 without the protein-protein axis; rows ordered correct interface, correct, intermediate, wrong RNA.
Output: results/plots/all_chains/publication_figures/placement_b_heatmap_{b,b1,b2}.png
Run in the RNABench env:  python publication_figures/placement_b_heatmap.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interface_failure_modes_figure as F
import figstyle

DRY = os.path.join(F.POSE_DIR, "placement_all_chains.tsv")
MIN_N = 10
SHORT = {"same place, duplex slid": "slid duplex", "RNA rotated at the pocket": "rotated at pocket",
         "wrong RNA segment at the pocket": "wrong segment at pocket",
         "correct RNA placement, incorrect contacts": "in place, wrong contacts", "distal site": "distal",
         "protein binding region mispredicted": "incorrect binding site", "RNA shifted": "shifted"}


def label(c):
    if c == "correct interface":
        return "Correct interface"
    *grp, pose = c.split(" / ")
    return " · ".join(grp + [SHORT.get(pose, pose)])


def build(col, tag):
    figstyle.apply("plain")
    d = pd.read_csv(DRY, sep="\t")
    d = d[d.dockq.notna()]
    # b2: classification b without the protein-protein axis (RNA fold x pose)
    d["placement_b2"] = d.placement_bf.where(d.placement_bf == "correct interface",
                                             d.placement_bf.str.split(" / ", n=1).str[1])
    # b2.1: b.1 without the protein-protein axis (RNA correct / intermediate / wrong x pose)
    d["placement_b21"] = d.placement_b1.where(d.placement_b1 == "correct interface",
                                              d.placement_b1.str.split(" / ", n=1).str[1])
    # b2.2: b2.1 with correct and intermediate RNA combined (RNA good: LDDT_r > 0.75); pose rules unchanged
    d["placement_b22"] = d.placement_b21.str.replace("RNA correct", "RNA good").str.replace("RNA intermediate", "RNA good")
    # b2.3: b2.2 with the wrong-RNA wrong segment, shifted, in place and incorrect binding site classes merged
    merged = "RNA wrong / wrong segment, shifted, in place or incorrect site"
    d["placement_b23"] = d.placement_b22.replace(
        {f"RNA wrong / {p}": merged for p in ("wrong RNA segment at the pocket", "RNA shifted",
                                               "correct RNA placement, incorrect contacts",
                                               "protein binding region mispredicted")})
    # b2.4: b2.3 with every good-RNA class except distal and shifted merged
    keep_good = ("RNA good / distal site", "RNA good / RNA shifted")
    d["placement_b24"] = d.placement_b23.where(~d.placement_b23.str.startswith("RNA good / ") |
                                               d.placement_b23.isin(keep_good), "RNA good / everything else")
    # b2.5: b2.4 with good-RNA shifted merged too; merged classes named "at or near the site"
    near = {"RNA good / RNA shifted": "RNA good / at or near the site",
            "RNA good / everything else": "RNA good / at or near the site", merged: "RNA wrong / at or near the site"}
    d["placement_b25"] = d.placement_b24.replace(near)
    # b2.6: b2.5 with slid duplex merged too -> per RNA level only distal vs at or near the site
    # b2.7: as b2.6 but intermediate RNA counted as wrong (RNA correct = paper's well-folded definition)
    two = lambda s: s.where(s == "correct interface",
                            s.str.split(" / ").str[0] + s.str.endswith("/ distal site").map(
                                {True: " / distal site", False: " / at or near the site"}))
    d["placement_b26"] = two(d.placement_b22)
    d["placement_b27"] = two(d.placement_b2)
    # b3: pose only, without the RNA-fold axis
    d["placement_b3"] = d.placement_b22.str.split(" / ").str[-1]
    # b3.1: b3 with every class except distal merged into at or near the site
    d["placement_b31"] = d.placement_b3.where(d.placement_b3.isin(["correct interface", "distal site"]),
                                              "at or near the site")
    rows = [dict(period=r.period, pdb=r.pdb, placement=r[col]) for _, r in d.iterrows()]
    post = d[d.period == "post"][col].value_counts()
    keep = ["correct interface"] + [c for c in post.index if c != "correct interface" and post[c] >= MIN_N]
    # rows: correct interface, then correct-RNA, intermediate-RNA and wrong-RNA classes (each by size)
    rank = lambda c: (0 if c == "correct interface" else 1 if ("RNA correct" in c or "RNA good" in c) else 2 if "RNA intermediate" in c else 3,
                      -post[c])
    keep = sorted(keep, key=rank)
    F.SHARE_ROWS = [(f"share_{c}", f"{label(c)} (n={post[c]})") for c in keep]
    hdf = F._heatmap_df(rows)
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_2COL, 0.32 * len(keep) + 1.6))
    F.panel_heatmap(ax, hdf)
    ax.set_title(f"Classification {tag}: Spearman ρ of each class's share of a complex's interfaces "
                 f"(post-cutoff; classes with ≥ {MIN_N} interfaces)", fontsize=figstyle.FONT_BASE)
    out = os.path.join(F.OUTDIR, f"placement_b_heatmap_{tag.replace('.', '')}")
    figstyle.save(fig, out, formats=("png", "pdf")); plt.close(fig)
    print("Saved:", out + ".png", "| rows:", len(keep))


if __name__ == "__main__":
    build("placement_bf", "b")
    build("placement_b1", "b.1")
    build("placement_b2", "b2")
    build("placement_b21", "b2.1")
    build("placement_b22", "b2.2")
    build("placement_b23", "b2.3")
    build("placement_b24", "b2.4")
    build("placement_b25", "b2.5")
    build("placement_b26", "b2.6")
    build("placement_b27", "b2.7")
    build("placement_b3", "b3")
    build("placement_b31", "b3.1")
