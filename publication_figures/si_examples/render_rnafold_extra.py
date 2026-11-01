"""Extra RNA-fold schematics (unlabelled; target orange, AF3 green):
  ions_8K7W   single-chain RNA without vs with ions (AF3 given the target's K+/Na+)
  msa_8OLS    single-chain RNA with vs without RNA MSA
  nonWC_8OMR_arcs  base-pair arc diagram, non-WC pairs emphasised
Run from the repo root in the RNABench env."""
import os, json, glob, subprocess, tempfile
import numpy as np
import gemmi
import pymol2
from PIL import Image

OUT = "results/plots/all_chains/publication_figures/site_chosen/rnafold"
TGT, TGT2, TGTD, AF, AF2, AFD = "#E8743B", "#F5B48A", "#B5481A", "#59A14F", "#A6D49A", "#2E6B27"
USALIGN = "evaluation/USalign/USalign"
IONS = "K+NA+MG+CA+MN+ZN+CL"
TMP = tempfile.mkdtemp()


def hexrgb(h): return [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def rna_only(path, tag):
    """polymer RNA chain(s) only, for US-align"""
    s = gemmi.read_structure(path); s.setup_entities(); s.remove_ligands_and_waters()
    for m in s:
        for c in list(m):
            if len(c) < 2: m.remove_chain(c.name)
    out = os.path.join(TMP, tag + ".pdb"); s.write_pdb(out); return out


def usalign_matrix(model, target):
    """4x4 (row-major) superposing model on target, from US-align -m"""
    mf = os.path.join(TMP, "m.txt")
    subprocess.run([USALIGN, rna_only(model, "m"), rna_only(target, "t"), "-mol", "RNA", "-m", mf],
                   check=True, capture_output=True)
    rows = [l.split() for l in open(mf) if l.strip()[:1] in "012" and len(l.split()) == 5]
    M = np.eye(4)
    for r in rows[:3]:
        k = int(r[0]); M[k, 3] = float(r[1]); M[k, :3] = [float(x) for x in r[2:]]
    return M.ravel().tolist()


def style(cmd):
    for n, h in (("cT", TGT), ("cA", AF)): cmd.set_color(n, hexrgb(h))
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1); cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_tube_radius", 0.6); cmd.set("ray_opaque_background", 0); cmd.set("antialias", 2)
    cmd.set("ambient", 0.35); cmd.set("specular", 0.2); cmd.set("sphere_scale", 0.9)


def pair_panels(target, models, names, ions=False, size=1100):
    """one image per model, same target view in every panel"""
    view = None
    for model, name in zip(models, names):
        with pymol2.PyMOL() as p:
            cmd = p.cmd; style(cmd)
            cmd.load(target, "nat"); cmd.load(model, "mdl")
            cmd.transform_selection("mdl", usalign_matrix(model, target), homogenous=1)
            cmd.hide("everything")
            cmd.show("cartoon", "polymer.nucleic"); cmd.color("cT", "nat"); cmd.color("cA", "mdl")
            if ions:
                cmd.show("spheres", f"resn {IONS}"); cmd.color("cT", f"nat and resn {IONS}"); cmd.color("cA", f"mdl and resn {IONS}")
            if view is None:
                cmd.orient("nat and polymer.nucleic"); cmd.zoom("polymer.nucleic", 3, complete=1); view = cmd.get_view()
            else:
                cmd.set_view(view)
            cmd.ray(size, size); path = os.path.join(OUT, name + ".png"); cmd.png(path, dpi=600); print("saved", path)


def stack(names, out, gap=20):
    ims = [Image.open(os.path.join(OUT, n + ".png")) for n in names]
    ims = [im.crop(im.getbbox()) for im in ims]
    w = max(im.width for im in ims); h = sum(im.height for im in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0)); y = 0
    for im in ims:
        canvas.paste(im, ((w - im.width) // 2, y)); y += im.height + gap
    canvas.save(os.path.join(OUT, out + ".png"), dpi=(600, 600)); print("saved", out)


def arcs(pid, out):
    import sys
    sys.path.insert(0, "publication_figures")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    import figstyle
    figstyle.apply("plain")
    d = {k: json.load(open((glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_aligned_rna_dssr.json") or
                            glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_dssr.json"))[0]))
         for k in ("pdb", "af")}  # full-structure DSSR when no aligned file exists
    assert d["pdb"]["dbn"]["all_chains"]["bseq"] == d["af"]["dbn"]["all_chains"]["bseq"]
    seq = d["pdb"]["dbn"]["all_chains"]["bseq"]

    def pairs(dd):
        out = {}; idx = {n["nt_id"]: i for i, n in enumerate(dd["nts"])}
        for p in dd["pairs"]:
            if p["LW"] == "--": continue
            i, j = sorted(idx[p[k]] for k in ("nt1", "nt2"))  # position in the DSSR nucleotide list
            out[(i, j)] = p["LW"] == "cWW" and p["name"] in ("WC", "Wobble")
        return out
    T, A = pairs(d["pdb"]), pairs(d["af"])
    span = max(j - i for i, j in list(T) + list(A))
    fig, ax = plt.subplots(figsize=(max(3.4, 0.049 * len(seq)), 2.4))  # constant letter spacing as for 8OMR
    # target above, AF3 below: canonical pale; non-canonical in the main colour; pairs only one side has, dark and thick
    for P, Q, (c_wc, c_nc, c_only), t1, t2 in ((T, A, (TGT2, TGT, TGTD), 0, 180), (A, T, (AF2, AF, AFD), 180, 360)):
        for (i, j), wc in sorted(P.items(), key=lambda kv: (not kv[1], (kv[0] not in Q))):
            only = (i, j) not in Q
            col = c_only if only else (c_wc if wc else c_nc)
            lw = 2.0 if only else (0.5 if wc else 0.8)
            ax.add_patch(Arc(((i + j) / 2, 0), j - i, (j - i) * 0.9, theta1=t1, theta2=t2, color=col, lw=lw))
    for i, c in enumerate(seq):
        ax.text(i, 0, c, ha="center", va="center", fontsize=3.6, family="monospace", color="0.25",
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_xlim(-1, len(seq)); ax.set_ylim(-span * 0.45 - 1, span * 0.45 + 1); ax.set_aspect("equal"); ax.axis("off")
    base = os.path.join(OUT, out)
    fig.savefig(base + ".pdf", bbox_inches="tight", transparent=True)
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", transparent=True); print("saved", base)
    print("target only:", sorted(set(T) - set(A)), " AF3 only:", sorted(set(A) - set(T)))


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["ions", "msa", "arcs"]
    os.makedirs(OUT, exist_ok=True)
    R = "data/RNAonly"
    if "ions" in which:
        pair_panels(f"{R}/a2021_rnaOnly/rna_rna/pdb/8k7w.cif",
                    [f"{R}/a2021_rnaOnly/rna_rna/af/fold_8k7w_s1_model_0.cif",
                     f"{R}/a2021_rnaOnly_ions/rna_rna/af/fold_8k7w_ion_s1_model_0.cif"],
                    ["ions_8K7W_without", "ions_8K7W_with"], ions=True)
        stack(["ions_8K7W_without", "ions_8K7W_with"], "ions_8K7W")
    if "msa" in which:
        pair_panels(f"{R}/a2021_rnaOnly/rna_rna/pdb/8ols.cif",
                    [f"{R}/a2021_rnaOnly/rna_rna/af/fold_8ols_s1_model_0.cif",
                     f"{R}/a2021_rnaOnly_add/a2021_x_noMSA/rna_rna/af/fold_8ols_s1_model_0.cif"],
                    ["msa_8OLS_with", "msa_8OLS_without"])
        stack(["msa_8OLS_with", "msa_8OLS_without"], "msa_8OLS")
    if "arcs" in which:
        arcs("8OMR", "nonWC_8OMR_arcs")
