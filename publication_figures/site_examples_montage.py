"""Example renders and overview montages of the placement classes (results/.../publication_figures/site_examples).

For each class of the paper's placement classification (interface_rna_pose.tsv, post-cutoff), the 20 most
illustrative failed (or, for `correct`, successful) interfaces are rendered with PyMOL: the native protein chain as a
grey surface, the native RNA in blue and the AF3 RNA in the class colour, after superposing the AF3 protein chain on
the native one. One interface per complex; RNAs shorter than the class's minimum length are left out unless that
leaves fewer than 20 complexes. Ranking per class:
  correct     highest DockQ
  incorrect   turned by >= 45 deg first, then highest protein contact match (at the right site, wrongly posed)
  rotated     largest rotation (closest to a full flip)
  shifted     largest translation (10-20 A band)
  distal      largest translation
Existing renders are reused: an image is only rendered when its file is missing. The current PyMOL build of the
RNABench env (Schrodinger 3.1.3.1 without a licence) watermarks new renders.

Output: site_examples/<class>/NN_<PDB>_<model chains>_<n>nt_shift..A_rot..deg_pm.._dockq...png, README.txt and
_index_<class>.png.
Run in the RNABench env:  python publication_figures/site_examples_montage.py [--dry-run | --montage-only]
  --dry-run        list the selected interfaces and whether their render exists; write nothing
  --montage-only   rebuild README.txt and _index_<class>.png from the existing renders; render nothing
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import interface_rna_pose as IRP
from placement_overview_montage import file_names, render as _render_base

OUT = os.path.join(_REPO, "results", "plots", "all_chains", "publication_figures", "site_examples")
N, MIN_COMPLEXES = 20, 20
NAT_RNA = "0x2B5B84"
# folder: (placement class, AF3 RNA colour, minimum RNA length, ranking key, description of the ranking)
CLASSES = {
    "correct": ("correct interface", "0x59A14F", 15, lambda d: -d.dockq, "highest DockQ (clearest successes)"),
    "incorrect": ("right protein site, wrong RNA pose", "0xE15759", 15,
                  lambda d: (d.rna_rotation < 45).astype(int) * 10 - d.protein_match,
                  "turned by >=45 deg, then highest protein contact match - at the right site, wrongly posed"),
    "rotated": ("same place, RNA turned", "0x76B7B2", 8, lambda d: -d.rna_rotation,
                "largest rotation about the correct site (closest to a full flip)"),
    "shifted": ("shifted nearby", "0xFF9DA7", 8, lambda d: -d.rna_translation,
                "largest in-band displacement (10-20 A)"),
    "distal": ("different region", "0x9C755F", 15, lambda d: -d.rna_translation,
               "largest displacement from the native site"),
}


def load():
    d = pd.read_csv(os.path.join(IRP.OUTDIR, "interface_rna_pose.tsv"), sep="\t")
    return d[(d.period == "post") & d.dockq.notna()].copy()


def pick(d, cls, min_nt, key):
    s = d[d.placement == cls]
    if s[s.rna_n_nt >= min_nt].pdb.nunique() >= MIN_COMPLEXES:
        s = s[s.rna_n_nt >= min_nt]
    s = s.assign(_k=key(s)).sort_values(["_k", "pdb", "native_pair"])
    return s.drop_duplicates("pdb").head(N), len(d[d.placement == cls])


def existing(outdir):
    """(PDB, model chains) -> render already in the folder (the rank and values in the name may have changed)."""
    out = {}
    for f in sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []:
        p = f.split("_")
        if f.endswith(".png") and len(p) > 3 and p[0].isdigit():
            out.setdefault((p[1], p[2]), os.path.join(outdir, f))
    return out


def png_name(k, r):
    return (f"{k:02d}_{r.pdb}_{str(r.model_pair).replace(':', '')}_{int(r.rna_n_nt)}nt_shift{r.rna_translation:.0f}A"
            f"_rot{r.rna_rotation:.0f}deg_pm{r.protein_match:.2f}_dockq{r.dockq:.2f}.png")


def render(r, fn, png, colour):
    """The native protein as an opaque grey surface, native RNA blue, AF3 RNA in the class colour."""
    from pymol import cmd
    if not _render_base(r, fn, png):
        return False
    P, R = str(r.native_pair).split(":")
    _, rr = str(r.model_pair).split(":")
    cmd.color(colour, f"mdl and chain {rr}"); cmd.color(NAT_RNA, f"nat and chain {R}")
    cmd.set("transparency", 0.0)
    cmd.png(png, width=700, height=700, dpi=200, ray=1)
    return os.path.exists(png)


def montage(items, title, colour, path):
    rows = int(np.ceil(len(items) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.6 * rows), squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (png, lab) in zip(np.ravel(axes), items):
        ax.imshow(plt.imread(png)); ax.set_title(lab, fontsize=8, color="#" + colour[2:], fontweight="bold")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    dry, montage_only = "--dry-run" in sys.argv, "--montage-only" in sys.argv
    d = load(); fns = {} if dry else file_names()
    for folder, (cls, colour, min_nt, key, how) in CLASSES.items():
        sel, n_cls = pick(d, cls, min_nt, key)
        outdir = os.path.join(OUT, folder)
        old = existing(outdir)
        items, missing = [], 0
        for k, (_, r) in enumerate(sel.iterrows(), 1):
            png = old.get((r.pdb, str(r.model_pair).replace(":", "")), os.path.join(outdir, png_name(k, r)))
            have = os.path.exists(png)
            if dry:
                print(f"  {folder:9s} {'ok ' if have else 'NEW'} {os.path.basename(png)}")
                missing += not have
                continue
            if not have and (montage_only or not render(r, fns.get(r.pdb, ""), png, colour)):
                missing += 1
                continue
            items.append((png, f"{r.pdb} {r.model_pair}  DockQ {r.dockq:.2f}\n{int(r.rna_n_nt)} nt  shift "
                               f"{r.rna_translation:.0f}Å  rot {r.rna_rotation:.0f}°  pm {r.protein_match:.2f}"))
        print(f"{folder}: {len(sel)} selected, {missing} without a render (class size {n_cls})", flush=True)
        if dry or not items:
            continue
        with open(os.path.join(outdir, "README.txt"), "w") as fh:
            fh.write(f"{folder}: {N} most illustrative post-cutoff examples\nclass            : {cls}\n"
                     f"ranked by        : {how}\nclass size (post): {n_cls}\n"
                     f"min RNA length   : {min_nt} nt (dropped if it leaves fewer than {MIN_COMPLEXES} complexes)\n"
                     "selection        : one interface per complex\n\n"
                     "blue = native RNA, coloured = AlphaFold 3 RNA, grey = native protein chain\n"
                     "filename: rank_PDB_modelchains_length_shift_rotation_proteinmatch_dockq\n")
        montage(items, f"{folder.upper()} - {len(items)} most illustrative post-cutoff examples ({how})", colour,
                os.path.join(outdir, f"_index_{folder}.png"))


if __name__ == "__main__":
    main()
