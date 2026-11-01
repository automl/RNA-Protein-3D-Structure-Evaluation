"""Structure panels for the controlled input ablation (target orange, AF3 green; RNA in the darker shade).
Run from the repo root in the RNABench env."""
import os
import pymol2

OUT = "results/plots/all_chains/publication_figures/site_chosen/ablation"
BC = "data/nar_msa_template_ablation_20261007/by_condition"
TGT, TGT2, AF, AF2 = "#E8743B", "#B5481A", "#59A14F", "#2E6B27"
hexrgb = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
COND = lambda r, p, t, c: f"{'+-'[1-r]}rMSA_{'+-'[1-p]}pMSA_{'+-'[1-t]}pTemp_{'+-'[1-c]}cTemp"


def style(cmd):
    for n, h in (("cT", TGT), ("cT2", TGT2), ("cA", AF), ("cA2", AF2)): cmd.set_color(n, hexrgb(h))
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1); cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_tube_radius", 0.6); cmd.set("ray_opaque_background", 0); cmd.set("antialias", 2)
    cmd.set("ambient", 0.35); cmd.set("specular", 0.2); cmd.set("surface_quality", 1)


def files(pid, side, cond):
    d = os.path.join(BC, side, cond, "protein_rna")
    return os.path.join(d, "pdb", f"{pid}.cif"), os.path.join(d, "af", f"fold_{pid}_s1_model_0.cif")


def complex_panel(pid, side, cond, name, view=None):
    """whole complex, AF3 superposed on the deposited complex (protein light, RNA dark)"""
    nat, mdl = files(pid, side, cond)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        cmd.super("mdl", "nat")
        cmd.hide("everything"); cmd.show("cartoon")
        cmd.color("cT", "nat and polymer.protein"); cmd.color("cT2", "nat and polymer.nucleic")
        cmd.color("cA", "mdl and polymer.protein"); cmd.color("cA2", "mdl and polymer.nucleic")
        if view: cmd.set_view(view)
        else: cmd.orient("nat"); cmd.zoom("nat or mdl", 2, complete=1)
        cmd.ray(1100, 1100); cmd.png(os.path.join(OUT, f"{name}.png"), dpi=600)
        return cmd.get_view()


def interface_panel(pid, side, cond, name, view=None, chains=None):
    """superposed on the protein: protein grey surface (deposited), target RNA orange, AF3 RNA green.
    chains: (model chain, deposited chain) of one protein copy to fit on; aligning several identical copies in one
    selection (or structure-based super) can pair one copy with the other"""
    nat, mdl = files(pid, side, cond)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        if chains:
            cmd.align(f"mdl and polymer.protein and chain {chains[0]}", f"nat and polymer.protein and chain {chains[1]}")
        else:
            cmd.super("mdl and polymer.protein", "nat and polymer.protein")
        cmd.create("prot", "nat and polymer.protein"); cmd.create("rnaN", "nat and polymer.nucleic")
        cmd.create("rnaM", "mdl and polymer.nucleic"); cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
        cmd.show("surface", "prot"); cmd.color("grey70", "prot"); cmd.set("transparency", 0.45, "prot")
        for o, c in (("rnaN", "cT"), ("rnaM", "cA")): cmd.show("cartoon", o); cmd.color(c, o)
        if view: cmd.set_view(view)
        else: cmd.orient("prot or rnaN"); cmd.zoom("prot or rnaN or rnaM", 4, complete=1)
        cmd.ray(1100, 1100); cmd.png(os.path.join(OUT, f"{name}.png"), dpi=600)
        return cmd.get_view()


def rna_fit_panel(pid, side, cond, name, view=None):
    """fitted on the RNA phosphates: deposited protein grey surface, AF3 protein light-green cartoon,
    target RNA orange, AF3 RNA green (where the protein sits relative to the RNA shows the interface)"""
    nat, mdl = files(pid, side, cond)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        cmd.align("polymer and name P and (mdl)", "polymer and name P and (nat)", quiet=0, reset=1)
        cmd.create("prot", "nat and polymer.protein"); cmd.create("protM", "mdl and polymer.protein")
        cmd.create("rnaN", "nat and polymer.nucleic"); cmd.create("rnaM", "mdl and polymer.nucleic")
        cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
        cmd.show("surface", "prot"); cmd.color("grey70", "prot"); cmd.set("transparency", 0.45, "prot")
        cmd.set_color("cA3", hexrgb("#A6D49A")); cmd.show("cartoon", "protM"); cmd.color("cA3", "protM")
        cmd.set("cartoon_transparency", 0.35, "protM")
        for o, c in (("rnaN", "cT"), ("rnaM", "cA")): cmd.show("cartoon", o); cmd.color(c, o)
        if view: cmd.set_view(view)
        else: cmd.orient("prot or rnaN"); cmd.zoom("prot or rnaN", 4, complete=1)
        cmd.ray(1100, 1100); cmd.png(os.path.join(OUT, f"{name}.png"), dpi=600)
        return cmd.get_view()


if __name__ == "__main__":
    if "8pds_rna" in __import__("sys").argv:
        v = rna_fit_panel("8pds", "rna_side", COND(1, 1, 1, 1), "8PDS_all")
        rna_fit_panel("8pds", "rna_side", COND(1, 1, 1, 0), "8PDS_noCTemp", v)
        raise SystemExit
    os.makedirs(OUT, exist_ok=True)
    # protein MSA: all inputs vs protein MSA removed (same view)
    if "only8pds" in __import__("sys").argv:
        v = interface_panel("8pds", "rna_side", COND(1, 1, 1, 1), "8PDS_all", chains=("A", "A"))
        interface_panel("8pds", "rna_side", COND(1, 1, 1, 0), "8PDS_noCTemp", v, chains=("A", "A"))
        raise SystemExit
    v = complex_panel("7yfq", "protein_side", COND(1, 1, 1, 1), "7YFQ_all")
    complex_panel("7yfq", "protein_side", COND(1, 0, 1, 1), "7YFQ_noPMSA", v)
    # protein templates as backup: protein MSA removed, with vs without protein templates
    v = complex_panel("8snx", "rna_side", COND(1, 0, 1, 1), "8SNX_noPMSA")
    complex_panel("8snx", "rna_side", COND(1, 0, 0, 1), "8SNX_noPMSA_noPTemp", v)
    # complex template: all inputs vs RBP template removed
    v = rna_fit_panel("8pds", "rna_side", COND(1, 1, 1, 1), "8PDS_all")
    rna_fit_panel("8pds", "rna_side", COND(1, 1, 1, 0), "8PDS_noCTemp", v)
    print("done")
