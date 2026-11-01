"""
Graphical abstract: the protein-RNA interface failure AF3's confidence does not see.

Two variants are written:
  graphical_abstract    3 structures  - correct, the dominant failure, and a distal site
  graphical_abstract2   5 structures  - every placement class that has a representative,
                                        ordered by how far the RNA ends up from its site

One message, three bands:
  (top)    after the training cutoff, 48% of interfaces are correct (81% before);
  (middle) the dominant failure is a correctly sited RNA in the wrong pose, and it is the
           failure AF3's confidence registers least (ipTM_r rho -0.25) while the rarer
           distal placement is registered (-0.41);
  (bottom) one representative structure per class, native RNA vs AF3 RNA in the native
           protein frame (the frame the translation/rotation numbers are measured in).

Representatives are post-cutoff and chosen near their class median, preferring few chains
and an RNA large enough to read at print size. Only the one protein chain of the interface
is drawn, so multi-protein assemblies stay legible.

Shares are recomputed from interface_rna_pose.tsv, not hard-coded.

Usage (RNABench env):
    python publication_figures/graphical_abstract.py            # render + assemble both
    python publication_figures/graphical_abstract.py --no-pymol # reuse cached PNGs
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import figstyle

# The pose table currently lives under plots_odl (interface_rna_pose.OUTDIR points at
# results/plots/..., which this checkout does not have); prefer whichever exists.
_CANDIDATE_TSV = [
    os.path.join(_REPO, "results", "plots", "all_chains", "interface_rna_pose", "interface_rna_pose.tsv"),
    os.path.join(_REPO, "results", "plots_odl", "all_chains", "interface_rna_pose", "interface_rna_pose.tsv"),
]
CARTOON_DIR = os.path.join(_REPO, "results", "plots", "all_chains", "publication_figures", "ga_cartoons")
FIG_DIR = os.path.join(_REPO, "results", "plots", "all_chains", "publication_figures")

# An RNA in place and unrotated that still fails is a correct-site failure (as in the
# main failure-mode figure), so it is merged into that class.
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}

CORRECT = "correct interface"
WRONGPOSE = "right protein site, wrong RNA pose"
ROTATED = "same place, RNA turned"
SHIFTED = "shifted nearby"
DISTAL = "different region"
BADSITE = "protein binding region mispredicted"

# label, placement key, colour, the ipTM_r Spearman rho reported in the paper
FAILS = [
    ("Correct site,\nincorrect RNA", WRONGPOSE, "#E15759", -0.25),
    ("Distal site", DISTAL, "#9C755F", -0.41),
    ("Incorrect\nbinding site", BADSITE, "#B07AA1", None),
    ("Shifted site", SHIFTED, "#FF9DA7", None),
    ("Correct site,\nrotated RNA", ROTATED, "#76B7B2", None),
]
CLS_COLOR = dict([(k, c) for _l, k, c, _r in FAILS] + [(CORRECT, "#59A14F")])

# pdb, model pair (AF3 chains), native pair (auth chains), class, caption
PANELS3 = [
    ("8YE6", "B:C", "A:C", CORRECT, "Correct interface"),
    ("8E29", "A:B", "A:B", WRONGPOSE, "Correct site, incorrect RNA"),
    ("8HKE", "B:C", "A:C", DISTAL, "Distal site"),
]
# ordered by how far the RNA ends up from where it belongs
PANELS5 = [
    ("8YE6", "B:C", "A:C", CORRECT, "Correct interface"),
    ("8E29", "A:B", "A:B", WRONGPOSE, "Correct site, incorrect RNA"),
    ("7VG2", "A:C", "A:D", ROTATED, "Correct site, rotated RNA"),
    ("7DLZ", "A:F", "A:X", SHIFTED, "Shifted site"),
    ("8HKE", "B:C", "A:C", DISTAL, "Distal site"),
]

NATIVE_RNA_COLOR = "#2B5D8A"   # experimental RNA, same in every panel
PROTEIN_COLOR = "grey70"


def pose_table():
    for p in _CANDIDATE_TSV:
        if os.path.exists(p):
            d = pd.read_csv(p, sep="\t")
            d = d[d.dockq.notna()].copy()
            d["placement"] = d.placement.replace(MERGE)
            d["pdb"] = d.pdb.str.upper()
            return d
    raise SystemExit("interface_rna_pose.tsv not found; run evaluation/interface_rna_pose.py")


def shares(d):
    """{period: (n, n_correct, frac_correct, {class: frac_of_failures})}"""
    out = {}
    for per in ("pre", "post"):
        s = d[d.period == per]
        f = s[s.placement != CORRECT]
        out[per] = (len(s), (s.placement == CORRECT).sum(),
                    (s.placement == CORRECT).mean(),
                    {k: v / len(f) for k, v in f.placement.value_counts().items()})
    return out


# --------------------------------------------------------------------------- #
# PyMOL cartoons
# --------------------------------------------------------------------------- #
def _structures(pdb):
    """Native cif and AF3 model cif, resolved exactly as the pose analysis does."""
    import sqlite3
    from evaluation.interface_rna_pose import PERIODS, INPUT_ROOTS, _locate_model
    from evaluation.exact_residue_mapping import locate_inputs
    con = sqlite3.connect(PERIODS["post"])
    con.row_factory = sqlite3.Row
    r = con.execute("SELECT exp_db_id AS id, FileName AS fn FROM pred_protein_rna "
                    "WHERE UPPER(TRIM(exp_db_id))=?", (pdb,)).fetchone()
    con.close()
    if r is None:
        raise SystemExit(f"{pdb} not in the post-cutoff database")
    cif, _js = locate_inputs(r["id"], r["fn"], INPUT_ROOTS)
    return cif, _locate_model(r["fn"])


def render_cartoons(panels, size=1100, force=False):
    """One PNG per panel: native protein surface, native RNA, AF3 RNA, in the native frame."""
    import pymol2
    os.makedirs(CARTOON_DIR, exist_ok=True)
    done = set()
    for pdb, mpair, npair, key, _cap in panels:
        tag = f"{pdb}_{mpair.replace(':', '')}"
        out = os.path.join(CARTOON_DIR, f"{tag}.png")
        if tag in done or (os.path.exists(out) and not force):
            continue
        done.add(tag)
        native, model = _structures(pdb)
        mP, mR = mpair.split(":")
        nP, nR = npair.split(":")
        with pymol2.PyMOL() as p:
            cmd = p.cmd
            cmd.load(native, "nat")
            cmd.load(model, "mdl")
            cmd.remove("not polymer")
            # put the model in the native protein frame
            cmd.align(f"mdl and chain {mP} and polymer.protein",
                      f"nat and chain {nP} and polymer.protein")
            cmd.create("prot", f"nat and chain {nP} and polymer.protein")
            cmd.create("rnaN", f"nat and chain {nR} and polymer.nucleic")
            cmd.create("rnaM", f"mdl and chain {mR} and polymer.nucleic")
            cmd.delete("nat")
            cmd.delete("mdl")
            for obj in ("prot", "rnaN", "rnaM"):
                if cmd.count_atoms(obj) == 0:
                    raise SystemExit(f"{pdb} {mpair}: object {obj} is empty - check the chain ids")

            cmd.hide("everything")
            cmd.show("surface", "prot")
            cmd.color(PROTEIN_COLOR, "prot")
            cmd.set("transparency", 0.45, "prot")
            cmd.set("surface_quality", 1)
            for obj, col in (("rnaN", NATIVE_RNA_COLOR), ("rnaM", CLS_COLOR[key])):
                cmd.show("cartoon", obj)
                cmd.set_color(f"c_{obj}", [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)])
                cmd.color(f"c_{obj}", obj)
            cmd.set("cartoon_ring_mode", 3)
            cmd.set("cartoon_ring_finder", 1)
            cmd.set("cartoon_nucleic_acid_mode", 4)
            cmd.set("cartoon_tube_radius", 0.6)
            cmd.set("ray_opaque_background", 0)
            cmd.set("antialias", 2)
            cmd.set("ambient", 0.35)
            cmd.set("specular", 0.2)
            cmd.orient("rnaN or rnaM or prot")
            cmd.zoom("rnaN or rnaM or prot", buffer=2.0)
            cmd.ray(size, size)
            cmd.png(out, dpi=600)
        print("  rendered", out)


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def _cropped(png):
    """Trim the transparent border PyMOL leaves, so the structures fill their panel."""
    from PIL import Image
    out = png.replace(".png", "_crop.png")
    im = Image.open(png).convert("RGBA")
    bbox = im.split()[-1].getbbox()
    if bbox:
        im = im.crop(bbox)
    im.save(out)
    return out


def build(panels, out_name, height=4.6):
    figstyle.apply()
    d = pose_table()
    sh = shares(d)
    n_post, _n_ok, frac_post, fails_post = sh["post"]
    n_pre, _k, frac_pre, _fp = sh["pre"]

    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, height))
    # three independently positioned bands: the xlabel of the middle band and the
    # titles of the structure band must not meet, so they do not share a gridspec
    L, R = 0.150, 0.985
    gs_top = fig.add_gridspec(1, 1, left=L, right=R, top=0.905, bottom=0.845)
    gs_mid = fig.add_gridspec(1, 1, left=L, right=R, top=0.790, bottom=0.555)
    gs_bot = fig.add_gridspec(1, len(panels), left=0.012, right=0.990, top=0.430,
                              bottom=0.105, wspace=0.03)

    fig.text(0.5, 0.978, "AlphaFold 3's most frequent protein–RNA failure is the one "
                         "its confidence misses",
             ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(L, 0.925, "Protein–RNA interfaces predicted correctly (DockQ ≥ 0.23)",
             ha="left", va="bottom", fontsize=8, color="#4A4A4A")

    # ---- correct interfaces, before vs after the cutoff --------------------
    ax = fig.add_subplot(gs_top[0, 0])
    rows = [("Before cutoff", frac_pre * 100, n_pre, "#BAB0AC", "#4A4A4A"),
            ("After cutoff", frac_post * 100, n_post, "#4E79A7", "white")]
    for i, (lab, v, n, col, tcol) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh([y], [v], color=col, height=0.66)
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=8.5)
        ax.text(v - 1.2, y, f"{v:.0f}% correct", ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=tcol)
        ax.text(v + 1.5, y, f"n = {n:,}", ha="left", va="center", fontsize=7, color="#4A4A4A")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 1.65)
    ax.axis("off")

    # ---- how the remainder fails ------------------------------------------
    ax = fig.add_subplot(gs_mid[0, 0])
    labels = [l for l, k, _c, _r in FAILS]
    vals = [fails_post.get(k, 0) * 100 for _l, k, _c, _r in FAILS]
    cols = [c for _l, _k, c, _r in FAILS]
    rhos = [r for _l, _k, _c, r in FAILS]
    y = list(range(len(labels)))[::-1]
    ax.barh(y, vals, color=cols, height=0.70)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, linespacing=0.95)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the failed interfaces after the training cutoff (%)",
                  fontsize=8, labelpad=2)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for yy, v, r in zip(y, vals, rhos):
        ax.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=8.5, fontweight="bold")
        if r is not None:
            missed = r > -0.33
            tag = "missed by AF3 confidence" if missed else "flagged by AF3 confidence"
            ax.text(v + 9.5, yy, f"ipTM$_r$ $\\rho$ = −{abs(r):.2f}   {tag}",
                    va="center", fontsize=7.5,
                    color="#B3272D" if missed else "#4A4A4A",
                    fontweight="bold" if missed else "normal")

    # ---- one structure per class ------------------------------------------
    cap_fs = 8.5 if len(panels) <= 3 else 7.0
    sub_fs = 7.5 if len(panels) <= 3 else 6.5
    for i, (pdb, mpair, _npair, key, cap) in enumerate(panels):
        ax = fig.add_subplot(gs_bot[0, i])
        ax.axis("off")
        png = os.path.join(CARTOON_DIR, f"{pdb}_{mpair.replace(':', '')}.png")
        if os.path.exists(png):
            ax.imshow(mpimg.imread(_cropped(png)))
        row = d[(d.pdb == pdb) & (d.model_pair == mpair)].iloc[0]
        ax.set_title(cap, fontsize=cap_fs, fontweight="bold", color=CLS_COLOR[key], pad=3)
        ax.text(0.5, -0.04, f"{pdb}    DockQ {row.dockq:.2f}", ha="center", va="top",
                fontsize=sub_fs, color="#4A4A4A", transform=ax.transAxes)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=NATIVE_RNA_COLOR, lw=3, label="Native RNA"),
               Line2D([], [], color="#9A9A9A", lw=3,
                      label="AlphaFold 3 RNA (coloured by class above)"),
               Line2D([], [], color="#D0D0D0", lw=3, label="Protein (native)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, handlelength=1.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.004))

    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, out_name)
    figstyle.save(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    print("wrote", base + ".pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pymol", action="store_true", help="reuse cached cartoon PNGs")
    ap.add_argument("--force", action="store_true", help="re-render cartoons that exist")
    a = ap.parse_args()
    allp = PANELS3 + [p for p in PANELS5 if p not in PANELS3]
    if not a.no_pymol:
        render_cartoons(allp, force=a.force)
    build(PANELS3, "graphical_abstract")
    build(PANELS5, "graphical_abstract2", height=4.7)
