"""Incorrect binding site, sub-molecular view: the protein's RNA-binding segment itself misfolded.
AF3 superposed on the protein around the site (CA within 20 A of the target RNA, the contact segment
excluded); context protein faint grey; binding segment target orange vs AF3 green; RNAs orange / green.
usage: python render_badsite.py PDB natP natR mdlP mdlR seg_from seg_to [out_name]"""
import glob, os, sys
import numpy as np
import pymol2

OUT = "results/plots/all_chains/publication_figures/site_chosen/interface"
TGT, AF = "#E8743B", "#59A14F"
def hexrgb(h): return [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]

pid, nP, nR, mP, mR, s0, s1 = sys.argv[1:8]
name = sys.argv[8] if len(sys.argv) > 8 else f"incorrect_site_{pid.upper()}"
s0, s1 = int(s0), int(s1)
with pymol2.PyMOL() as p:
    cmd = p.cmd
    cmd.set_color("cT", hexrgb(TGT)); cmd.set_color("cA", hexrgb(AF))
    cmd.load(glob.glob(f"data/a2021*/protein_rna/pdb/{pid.lower()}.cif")[0], "nat")
    cmd.load(glob.glob(f"data/a2021*/protein_rna/af/fold_{pid.lower()}_s1_model_0.cif")[0], "mdl")
    cmd.remove("not polymer")
    cmd.align(f"mdl and chain {mP} and name CA", f"nat and chain {nP} and name CA", cycles=0, transform=0, object="aln")
    pairs = [(a[1], b[1]) if a[0] == "nat" else (b[1], a[1]) for a, b in cmd.get_raw_alignment("aln")]
    rna = cmd.get_coords(f"nat and chain {nR}")
    resv = {}
    cmd.iterate(f"nat and chain {nP} and name CA", "resv_[index]=int(resv)", space={"resv_": resv, "int": int})
    mres = {}
    cmd.iterate(f"mdl and chain {mP} and name CA", "m_[index]=int(resv)", space={"m_": mres, "int": int})
    env_n, env_m, seg_m = [], [], []
    for n, m in pairs:
        r = resv[n]
        if s0 <= r <= s1:
            seg_m.append(mres[m]); continue
        d = np.min(np.linalg.norm(rna - cmd.get_coords(f"nat and index {n}")[0], axis=1))
        if d < 20:
            env_n.append(n); env_m.append(m)
    rms = cmd.pair_fit("mdl and index " + "+".join(map(str, env_m)), "nat and index " + "+".join(map(str, env_n)))
    print("env fit RMSD", round(rms, 2), "on", len(env_n), "CA; AF3 segment", min(seg_m), "-", max(seg_m))
    cmd.create("ctx", f"nat and chain {nP} and not resi {s0}-{s1}")
    cmd.create("segT", f"nat and chain {nP} and resi {s0}-{s1}")
    cmd.create("segA", f"mdl and chain {mP} and resi {min(seg_m)}-{max(seg_m)}")
    cmd.create("rnaT", f"nat and chain {nR}"); cmd.create("rnaA", f"mdl and chain {mR}")
    cmd.delete("nat"); cmd.delete("mdl")
    cmd.hide("everything")
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_nucleic_acid_mode", 4); cmd.set("cartoon_tube_radius", 0.6)
    cmd.set("cartoon_transparency", 0.75, "ctx"); cmd.set("ray_opaque_background", 0); cmd.set("antialias", 2)
    cmd.set("ambient", 0.35); cmd.set("specular", 0.2); cmd.set("cartoon_fancy_helices", 1); cmd.dss("segT or segA or ctx")
    cmd.show("cartoon"); cmd.color("grey70", "ctx")
    for o, c in (("segT", "cT"), ("rnaT", "cT"), ("segA", "cA"), ("rnaA", "cA")): cmd.color(c, o)
    # keep the context to the region around the site
    cmd.hide("cartoon", "ctx and not byres (ctx within 22 of (segT or rnaT))")
    cmd.orient("segT or segA or rnaT"); cmd.zoom("segT or segA or rnaT", 4, complete=1)
    cmd.ray(1100, 1100); path = os.path.join(OUT, name + ".png"); cmd.png(path, dpi=600); print("saved", path)
