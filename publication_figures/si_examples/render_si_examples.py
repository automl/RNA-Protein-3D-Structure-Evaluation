"""Second structural example per failure category of the graphical abstract (target orange, AF3 green).
Run from the repo root in the RNABench env:  python render_si_examples.py"""
import glob, json, os, subprocess, sys
import pymol2

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S); sys.path.insert(0, "publication_figures"); sys.path.insert(0, ".")
OUT = "results/plots/all_chains/publication_figures/site_chosen"
TGT, TGT2, TGTD, AF, AF2, AFD = "#E8743B", "#F5B48A", "#B5481A", "#59A14F", "#A6D49A", "#2E6B27"
hexrgb = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def style(cmd):
    for n, h in (("cT", TGT), ("cT2", TGT2), ("cA", AF), ("cA2", AF2)): cmd.set_color(n, hexrgb(h))
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1); cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_tube_radius", 0.6); cmd.set("ray_opaque_background", 0); cmd.set("antialias", 2)
    cmd.set("ambient", 0.35); cmd.set("specular", 0.2); cmd.set("stick_radius", 0.22); cmd.set("surface_quality", 1)


def save(cmd, path, size=1100):
    cmd.ray(size, size); cmd.png(path, dpi=600); print("saved", path)


def full(pid):
    for d in ("data/a2021x/protein_rna", "data/a2021nonX/protein_rna"):
        c = f"{d}/pdb/{pid.lower()}.cif"; a = f"{d}/af/fold_{pid.lower()}_s1_model_0.cif"
        if os.path.exists(c) and os.path.exists(a): return c, a
    raise SystemExit("no structures for " + pid)


def aligned(pid, kind):
    return glob.glob(f"data/a2021*/protein_rna/aligned/{pid.lower()}_{kind}_aligned_rna.pdb")[0]


def rna_panel(pid, name, highlight=None):
    """as render_failure_modes.rna_panel: aligned RNA chains, highlighted nucleotides as sticks"""
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd)
        cmd.load(aligned(pid, "pdb"), "nat"); cmd.load(aligned(pid, "af"), "mdl"); cmd.remove("not polymer.nucleic")
        cmd.align("mdl", "nat"); cmd.hide("everything"); cmd.show("cartoon"); cmd.color("cT", "nat"); cmd.color("cA", "mdl")
        focus = "all"
        if highlight:
            sel = " or ".join(f"resi {r}" for r in highlight)
            cmd.set("cartoon_transparency", 0.65); cmd.set("cartoon_ring_mode", 0)
            cmd.show("sticks", f"({sel}) and not name P+OP1+OP2")
            cmd.color("cT", f"nat and ({sel})"); cmd.color("cA", f"mdl and ({sel})"); focus = f"({sel})"
        cmd.orient(focus); cmd.zoom(focus, 4 if focus != "all" else 2)
        save(cmd, f"{OUT}/rnafold/{name}.png")


def interface_panel(pid, natP, natR, mdlP, mdlR, name):
    """as the 8H1B panel: protein surface (grey) + target RNA (orange) and AF3 RNA (green), superposed on the protein"""
    nat, mdl = full(pid)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        cmd.align(f"mdl and chain {mdlP} and polymer.protein", f"nat and chain {natP} and polymer.protein")
        cmd.create("prot", f"nat and chain {natP} and polymer.protein")
        cmd.create("rnaN", f"nat and chain {natR} and polymer.nucleic"); cmd.create("rnaM", f"mdl and chain {mdlR} and polymer.nucleic")
        cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
        cmd.show("surface", "prot"); cmd.color("grey70", "prot"); cmd.set("transparency", 0.45, "prot")
        for obj, col in (("rnaN", "cT"), ("rnaM", "cA")): cmd.show("cartoon", obj); cmd.color(col, obj)
        cmd.orient("prot or rnaN or rnaM"); cmd.zoom("prot or rnaN or rnaM", 8, complete=1)
        save(cmd, f"{OUT}/interface/{name}.png")


def duplex_panel(pid, natP, mdlP, natRs, mdlRs, name):
    """as missing_duplex_7W0D: RNA strands only, superposed on the protein (two shades per structure)"""
    nat, mdl = full(pid)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer")
        cmd.align(f"mdl and chain {mdlP} and polymer.protein", f"nat and chain {natP} and polymer.protein")
        cmd.hide("everything")
        for obj, chs, cols in (("nat", natRs, ("cT", "cT2")), ("mdl", mdlRs, ("cA", "cA2"))):
            for ch, col in zip(chs, cols):
                cmd.show("cartoon", f"{obj} and chain {ch} and polymer.nucleic"); cmd.color(col, f"{obj} and chain {ch}")
        rna = " or ".join(f"(nat and chain {c})" for c in natRs) + " or " + " or ".join(f"(mdl and chain {c})" for c in mdlRs)
        cmd.orient(rna); cmd.zoom(rna, 4, complete=1)
        save(cmd, f"{OUT}/rnafold/{name}.png")


def protein_panel(pid, natA, natB, mdlA, mdlB, name):
    """as protein_protein_8VMA: proteins only, AF3 superposed on protein A"""
    nat, mdl = full(pid)
    with pymol2.PyMOL() as p:
        cmd = p.cmd; style(cmd); cmd.load(nat, "nat"); cmd.load(mdl, "mdl"); cmd.remove("not polymer.protein")
        cmd.align(f"mdl and chain {mdlA}", f"nat and chain {natA}")
        cmd.create("tgt", f"nat and chain {natA}+{natB}"); cmd.create("af3", f"mdl and chain {mdlA}+{mdlB}")
        cmd.delete("nat"); cmd.delete("mdl"); cmd.hide("everything")
        for obj, col in (("tgt", "cT"), ("af3", "cA")):
            cmd.show("surface", obj); cmd.color(col, obj); cmd.set("transparency", 0.35, obj)
        cmd.set("surface_quality", -1); cmd.set("solvent_radius", 2.0)  # 2.8k-residue complexes, coarse surface keeps ray-tracing tractable
        cmd.orient("tgt or af3"); cmd.zoom("tgt or af3", 6, complete=1)
        save(cmd, f"{OUT}/interface/{name}.png")


def pk_arcs(pid, name):
    """pseudoknot arc diagram as for 7QR3: target above (pseudoknot arcs dark and thick), AF3 below"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    import figstyle
    figstyle.apply("plain")
    def dssr(k):  # aligned RNA file if present, else the full-structure one (8T2B, identical 90-nt bseq)
        f = glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_aligned_rna_dssr.json") or \
            glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_dssr.json")
        return json.load(open(f[0]))
    d = {k: dssr(k) for k in ("pdb", "af")}
    assert d["pdb"]["dbn"]["all_chains"]["bseq"] == d["af"]["dbn"]["all_chains"]["bseq"]
    seq = d["pdb"]["dbn"]["all_chains"]["bseq"]
    def pairs(s):
        out = []; st = {o: [] for o in "([{<"}; cl = {")": "(", "]": "[", "}": "{", ">": "<"}
        for i, c in enumerate(s):
            if c in st: st[c].append(i)
            elif c in cl and st[cl[c]]: out.append((st[cl[c]].pop(), i, cl[c] != "("))
        return out
    P = pairs(d["pdb"]["dbn"]["all_chains"]["sstr"]); A = pairs(d["af"]["dbn"]["all_chains"]["sstr"])
    span = max([j - i for i, j, _ in P + A] or [1])
    W = max(3.4, 0.049 * len(seq))  # constant letter spacing (as 7QR3/8OMR)
    fig, ax = plt.subplots(figsize=(W, W * (span * 0.9 + 2) / (len(seq) + 1)))
    for i, j, pk in P:
        ax.add_patch(Arc(((i + j) / 2, 0), j - i, (j - i) * 0.9, theta1=0, theta2=180, color=TGTD if pk else TGT, lw=1.6 if pk else 0.7))
    for i, j, pk in A:
        ax.add_patch(Arc(((i + j) / 2, 0), j - i, (j - i) * 0.9, theta1=180, theta2=360, color=AF, lw=1.6 if pk else 0.7))
    for i, c in enumerate(seq):
        ax.text(i, 0, c, ha="center", va="center", fontsize=3.6, family="monospace", color="0.25",
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.set_xlim(-1, len(seq)); ax.set_ylim(-span * 0.45 - 1, span * 0.45 + 1); ax.set_aspect("equal"); ax.axis("off")
    base = f"{OUT}/rnafold/{name}"
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", transparent=True); fig.savefig(base + ".pdf", bbox_inches="tight", transparent=True)
    plt.close(fig); print("saved", base)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rna", "arcs", "interface", "pocket"]
    os.makedirs(f"{OUT}/rnafold", exist_ok=True); os.makedirs(f"{OUT}/interface", exist_ok=True)
    if "rna" in which:
        rna_panel("8BMW", "torsion_8BMW", [2, 4, 5, 6, 7, 15, 18, 28, 34])        # syn->anti and C2'->C3' (DSSR index + 1)
        rna_panel("8FUK", "stacking_8FUK", [15, 16, 17, 21, 22, 23, 33, 34, 35])  # native stacks absent in AF3
        rna_panel("8D97", "over_compact_8D97")
        duplex_panel("8SSW", "A", "A", ("C", "D"), ("C", "D"), "missing_duplex_8SSW")
    if "arcs" in which:
        import render_rnafold_extra as rre
        rre.arcs("8UBE", "nonWC_8UBE_arcs")
        pk_arcs("8T2B", "pseudoknot_8T2B_arcs")
    if "interface" in which:
        interface_panel("8E29", "A", "B", "A", "B", "incorrect_8E29_AB")
        interface_panel("7VG2", "A", "D", "A", "C", "rotated_7VG2_AC")
        interface_panel("7DLZ", "A", "X", "A", "F", "shifted_7DLZ_AF")
        interface_panel("8JY0", "A", "F", "A", "F", "distal_8JY0_AF")
        protein_panel("7OS0", "A", "C", "A", "B", "protein_protein_7OS0")
    if "pocket" in which:
        subprocess.run([sys.executable, f"{S}/render_badsite.py", "8J62", "H", "Y", "C", "K", "48", "76"], check=True)


# --- nucleotide-level diagrams in the arc-diagram style (sequence on the midline, target above, AF3 below) ---
def _dssr_pair(pid):
    def one(k):
        f = glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_aligned_rna_dssr.json") or \
            glob.glob(f"data/a2021*/protein_rna/dssr/{pid.lower()}_{k}_dssr.json")
        return json.load(open(f[0]))
    T, A = one("pdb"), one("af")
    assert T["dbn"]["all_chains"]["bseq"] == A["dbn"]["all_chains"]["bseq"]
    return T, A


def _seq_axis(ax, seq):
    for i, c in enumerate(seq):
        ax.text(i, 0, c, ha="center", va="center", fontsize=3.6, family="monospace", color="0.25")


def stack_arcs(pid, name):
    """stacking contacts (consecutive nucleotides of each DSSR stack): shared pale, one-side-only dark and thick"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    import figstyle; figstyle.apply("plain")
    T, A = _dssr_pair(pid); seq = T["dbn"]["all_chains"]["bseq"]

    def contacts(x):
        idx = {n["nt_id"]: i for i, n in enumerate(x["nts"])}; out = set()
        for st in x.get("stacks", []):
            ids = [idx[i] for i in st["nts_long"].split(",")]
            out |= {tuple(sorted(p)) for p in zip(ids, ids[1:])}
        return out
    t, a = contacts(T), contacts(A)
    span = max([j - i for i, j in t | a] + [3])
    W = max(3.4, 0.049 * len(seq)); H = span * 0.45 + 2.2
    fig, ax = plt.subplots(figsize=(W, W * 2 * H / (len(seq) + 1)))
    for P, Q, (pale, dark), t1, t2 in ((t, a, (TGT2, TGTD), 0, 180), (a, t, (AF2, AFD), 180, 360)):
        for i, j in sorted(P, key=lambda p: p not in Q):
            only = (i, j) not in Q
            ax.add_patch(Arc(((i + j) / 2, 0.75 if t1 == 0 else -0.75), j - i, max(j - i, 1.6) * 0.9, theta1=t1, theta2=t2,  # clear of the letters
                             color=dark if only else pale, lw=2.0 if only else 0.8))
    _seq_axis(ax, seq)
    ax.set_xlim(-1, len(seq)); ax.set_ylim(-H, H); ax.set_aspect("equal"); ax.axis("off")
    base = f"{OUT}/rnafold/{name}"
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", transparent=True); fig.savefig(base + ".pdf", bbox_inches="tight", transparent=True)
    plt.close(fig); print("saved", base, "T-only", len(t - a), "AF3-only", len(a - t), "shared", len(t & a))


def conformer_track(pid, name):
    """per-nucleotide tiles: glycosidic bond (inner row) and sugar pucker (outer row), target above and AF3 below.
    Generic state (anti, C3'-endo) pale, syn and C2'-endo dark, other puckers mid-tone; tiles that differ
    between target and AF3 are outlined in black."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import figstyle; figstyle.apply("plain")
    T, A = _dssr_pair(pid); seq = T["dbn"]["all_chains"]["bseq"]; n = len(seq)
    W = max(3.4, 0.049 * n)
    fig, ax = plt.subplots(figsize=(W, W * 6.4 / (n + 1)))
    def col(prop, v, pal):
        pale, mid, dark = pal
        if prop == "glyco_bond": return {"anti": pale, "syn": dark}.get(v)
        return {"C3'-endo": pale, "C2'-endo": dark}.get(v, mid)
    for i in range(n):
        for prop, row in (("glyco_bond", 0), ("puckering", 1)):
            vt, va = T["nts"][i][prop], A["nts"][i][prop]
            diff = vt != va and "--" not in (vt, va)
            for v, pal, sign in ((vt, (TGT2, TGT, TGTD), 1), (va, (AF2, AF, AFD), -1)):
                c = col(prop, v, pal)
                y = 0.55 + row * 1.05 if sign > 0 else -(0.55 + row * 1.05) - 0.95
                ax.add_patch(Rectangle((i - 0.45, y), 0.9, 0.95, facecolor=c or "white",
                                       edgecolor="black" if diff else (c or "0.6"), lw=0.9 if diff else 0.3))
    _seq_axis(ax, seq)
    ax.set_xlim(-1, n); ax.set_ylim(-2.9, 2.9); ax.set_aspect("equal"); ax.axis("off")
    base = f"{OUT}/rnafold/{name}"
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", transparent=True); fig.savefig(base + ".pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)
    d = lambda p: sum(a[p] != b[p] and "--" not in (a[p], b[p]) for a, b in zip(T["nts"], A["nts"]))
    print("saved", base, "differ glyco", d("glyco_bond"), "pucker", d("puckering"))
