"""SI interface examples after the slid-duplex class: a new rotated example (7TJ2) and two slid duplexes
(7VG2, 8S9X), drawn like the other interface panels (protein surface grey, target RNA orange, AF3 RNA green,
superposed on the protein); for slid duplexes both strands, the native strand in the darker shade and the
predicted chain mapped to it in the darker green. Run from the repo root in the RNABench env."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymol2
from render_si_examples import OUT, full, interface_panel, save, style


def slid_panel(pid, natP, mdlP, natRs, mdlRs, name):
    nat, mdl = full(pid)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        cmd.align(f"mdl and chain {mdlP} and polymer.protein", f"nat and chain {natP} and polymer.protein")
        cmd.create("prot", f"nat and chain {natP} and polymer.protein")
        objs = []
        for src, chs, cols, tag in (("nat", natRs, ("cT", "cT2"), "N"), ("mdl", mdlRs, ("cA", "cA2"), "M")):
            for i, (ch, col) in enumerate(zip(chs, cols)):
                o = f"r{tag}{i}"; cmd.create(o, f"{src} and chain {ch} and polymer.nucleic"); cmd.color(col, o); objs.append(o)
        cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
        cmd.show("surface", "prot"); cmd.color("grey70", "prot"); cmd.set("transparency", 0.45, "prot")
        for o in objs: cmd.show("cartoon", o)
        sel = " or ".join(["prot"] + objs)
        cmd.orient(sel); cmd.zoom(sel, 8, complete=1)
        save(cmd, f"{OUT}/interface/{name}.png")


if __name__ == "__main__":
    interface_panel("7TJ2", "A", "G", "E", "G", "rotated_7TJ2_AG")
    slid_panel("7VG2", "A", "A", ("D", "C"), ("C", "B"), "slid_7VG2")     # native D (interface strand) / C; model C, B mapped to them
    slid_panel("8S9X", "A", "A", ("G", "F"), ("G", "F"), "slid_8S9X")
