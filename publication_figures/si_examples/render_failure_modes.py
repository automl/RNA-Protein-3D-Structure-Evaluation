import os, glob
import pymol2
OUT="results/plots/all_chains/publication_figures/site_chosen"
TGT, TGT2, AF, AF2 = "#E8743B", "#F5B48A", "#59A14F", "#A6D49A"     # target orange, AF3 green (lighter: partner chains)
def hexrgb(h): return [int(h[i:i+2],16)/255 for i in (1,3,5)]
def aligned(pid, kind): return glob.glob(f"data/a2021*/protein_rna/aligned/{pid.lower()}_{kind}_aligned_rna.pdb")[0]
def full(pid):
    for d in ("data/a2021x/protein_rna","data/a2021nonX/protein_rna"):
        c=f"{d}/pdb/{pid.lower()}.cif"; a=f"{d}/af/fold_{pid.lower()}_s1_model_0.cif"
        if os.path.exists(c) and os.path.exists(a): return c,a
def style(cmd):
    for n,h in (("cT",TGT),("cT2",TGT2),("cA",AF),("cA2",AF2)): cmd.set_color(n,hexrgb(h))
    cmd.set("cartoon_ring_mode",3); cmd.set("cartoon_ring_finder",1); cmd.set("cartoon_nucleic_acid_mode",4)
    cmd.set("cartoon_tube_radius",0.6); cmd.set("ray_opaque_background",0); cmd.set("antialias",2)
    cmd.set("ambient",0.35); cmd.set("specular",0.2); cmd.set("stick_radius",0.22); cmd.set("surface_quality",1)
def save(cmd, name, size=1100):
    cmd.ray(size,size); p=os.path.join(OUT,name+".png"); cmd.png(p,dpi=600); print("saved",p)

def rna_panel(pid, name, highlight=None, sticks=True, zoom_on_highlight=True):
    with pymol2.PyMOL() as p:
        cmd=p.cmd; style(cmd)
        cmd.load(aligned(pid,"pdb"),"nat"); cmd.load(aligned(pid,"af"),"mdl"); cmd.remove("not polymer.nucleic")
        cmd.align("mdl","nat"); cmd.hide("everything"); cmd.show("cartoon"); cmd.color("cT","nat"); cmd.color("cA","mdl")
        focus="all"
        if highlight:
            sel=" or ".join(f"resi {r}" for r in highlight)
            if sticks:
                cmd.set("cartoon_transparency",0.65); cmd.set("cartoon_ring_mode",0)
                cmd.show("sticks",f"({sel}) and not name P+OP1+OP2")
            else:
                cmd.color("cT2","nat"); cmd.color("cA2","mdl")
            cmd.color("cT",f"nat and ({sel})"); cmd.color("cA",f"mdl and ({sel})")
            if zoom_on_highlight: focus=f"({sel})"
        cmd.orient(focus); cmd.zoom(focus, 4 if focus!="all" else 2)
        save(cmd,name)

os.makedirs(OUT,exist_ok=True)
rna_panel("6XN5","torsion_6XN5",[20,22,24,25])
rna_panel("8S9V","stacking_8S9V",[9,10,11,12,13])
rna_panel("8OMR","nonWC_8OMR",[8,9,11,12,21,23])
rna_panel("7QR3","pseudoknot_7QR3",list(range(10,17))+list(range(63,70)),sticks=False,zoom_on_highlight=False)
rna_panel("8FZR","over_compact_8FZR")

# duplex not formed (7W0D): AF3 superposed on the interface protein (target A = AF3 C); all four RNA chains
nat,mdl=full("7W0D")
with pymol2.PyMOL() as p:
    cmd=p.cmd; style(cmd)
    cmd.load(nat,"nat"); cmd.load(mdl,"mdl"); cmd.remove("not polymer")
    cmd.align("mdl and chain C and polymer.protein","nat and chain A and polymer.protein")
    cmd.create("prot","nat and chain A and polymer.protein")
    cmd.hide("everything"); cmd.show("surface","prot"); cmd.color("grey70","prot"); cmd.set("transparency",0.45,"prot")
    for obj,ch,col in (("nat","D","cT"),("nat","C","cT2"),("mdl","E","cA"),("mdl","F","cA2")):
        cmd.show("cartoon",f"{obj} and chain {ch}"); cmd.color(col,f"{obj} and chain {ch}")
    rna="(nat and chain C+D) or (mdl and chain E+F)"
    cmd.orient(rna); cmd.zoom(f"{rna} or prot", 6)
    save(cmd,"missing_duplex_7W0D")

# protein-protein arrangement (8VMA): proteins only; AF3 superposed on protein A; target H orange, AF3 copy of H (chain C) green
nat,mdl=full("8VMA")
with pymol2.PyMOL() as p:
    cmd=p.cmd; style(cmd)
    cmd.load(nat,"nat"); cmd.load(mdl,"mdl"); cmd.remove("not polymer.protein")
    cmd.align("mdl and chain A","nat and chain A")
    for obj,sel,col in (("pA","nat and chain A","grey70"),("tH","nat and chain H","cT"),("aC","mdl and chain C","cA")):
        cmd.create(obj,sel)
    cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
    for obj,col,tr in (("pA","grey70",0.45),("tH","cT",0.25),("aC","cA",0.25)):
        cmd.show("surface",obj); cmd.color(col,obj); cmd.set("transparency",tr,obj)
    cmd.orient("pA or tH or aC"); cmd.zoom("pA or tH or aC",2)
    save(cmd,"protein_protein_8VMA")

