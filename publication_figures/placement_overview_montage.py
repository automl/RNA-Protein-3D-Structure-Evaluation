"""Overview montages of the RNA-pose classes of the two dry-run placement classifications (a and b).

For each pose class of classification a (placement_v2) and b (b_pose; b split into correct- and wrong-RNA sets, the
wrong-RNA set ordered by LDDT_r to judge the well-folded threshold), the
20 most illustrative post-cutoff failed interfaces are rendered with PyMOL: the native protein chain as a grey surface,
the native RNA chain in dark blue and the AF3 RNA chain in teal, after superposing the AF3 protein chain on the native
one. At most two interfaces per PDB entry. Most illustrative = largest rotation (rotated), largest shift (shifted,
distal), all slid duplexes.

Input: results/plots/all_chains/interface_rna_pose/placement_all_chains.tsv (evaluation/placement_all_chains.py)
Output: results/plots/all_chains/publication_figures/site_examples_v2/{a,b}/<class>/_index_<class>.png
Run in the RNABench env:  python publication_figures/placement_overview_montage.py
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
from evaluation import interface_rna_pose as IRP
from evaluation.exact_residue_mapping import locate_inputs

DRY = os.path.join(IRP.OUTDIR, "placement_all_chains.tsv")
OUT = os.path.join(_REPO, "results", "plots", "all_chains", "publication_figures", "site_examples_v2")
NAT_RNA, MDL_RNA = "0x2B5B84", "0x76B7B2"
CLASSES = {
    "a": {"rotated": ("placement_v2", "RNA rotated at the pocket"), "shifted": ("placement_v2", "RNA shifted"),
          "distal": ("placement_v2", "distal site"), "slid": ("placement_v2", "same place, duplex slid")},
    "b_correct_rna": {k: ("b_pose", k) for k in ("rotated", "shifted", "distal", "slid", "in place")},
    "b_wrong_rna": {k: ("b_pose", k) for k in ("rotated", "shifted", "distal", "slid", "in place")},
}
# classification 3 (interface_failure_modes panels B-D): the pose classes, combined into at or near the site / distal
POSE3 = {"slid duplex": "same place, duplex slid", "correct RNA segment": "correct RNA placement, incorrect contacts",
         "rotated RNA": "RNA rotated at the pocket", "wrong RNA segment": "wrong RNA segment at the pocket",
         "shifted site": "RNA shifted", "incorrect binding site": "protein binding region mispredicted",
         "distal site": "distal site"}
COMBINED3 = {"at or near the site": [k for k in POSE3 if k != "distal site"], "distal site": ["distal site"]}
N3 = 5
SORT = {"rotated": ("rot", False), "shifted": ("shift", False), "distal": ("shift", False), "slid": ("shift", True),
        "in place": ("shift", True), "slid duplex": ("shift", True), "correct RNA segment": ("shift", True),
        "rotated RNA": ("rot", False), "wrong RNA segment": ("shift", True), "shifted site": ("shift", False),
        "incorrect binding site": ("shift", False), "distal site": ("shift", False)}


def load(failed_only=True):
    d = pd.read_csv(DRY, sep="\t")
    pose = pd.read_csv(os.path.join(IRP.OUTDIR, "interface_rna_pose.tsv"), sep="\t")[
        ["period", "pdb", "native_pair", "model_pair", "rna_n_nt"]]
    d = d.merge(pose, on=["period", "pdb", "native_pair"], how="left")
    d = d[(d.period == "post") & ((d.dockq < 0.23) | (not failed_only))].copy()
    d["shift"] = d.t_loc.where(d.t_loc.notna(), d.rna_translation)
    d["rot"] = np.where(d.t_loc.notna() & (d.t_loc <= IRP.SAME_PLACE_A), d.a_loc, d.rna_rotation)
    d["pose3"] = d.placement_b1.str.split(" / ").str[-1]
    return d


def main3():
    """Classification 3: N3 examples per pose class, one montage per combined class (at or near the site, distal)."""
    d = load(); fns = file_names()
    for comb, kinds in COMBINED3.items():
        items = []
        for kind in kinds:
            outdir = os.path.join(OUT, "3", comb.replace(" ", "_"), kind.replace(" ", "_"))
            os.makedirs(outdir, exist_ok=True)
            n = 0
            for k, r in enumerate(pick(d, "pose3", POSE3[kind], kind, n=N3, scheme="3"), 1):
                png = os.path.join(outdir, f"{k:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}_{int(r.rna_n_nt or 0)}nt"
                                           f"_shift{r['shift']:.0f}A_rot{r['rot']:.0f}deg_dockq{r.dockq:.2f}.png")
                if not os.path.exists(png) and not render(r, fns.get(r.pdb, ""), png):
                    continue
                n += 1
                items.append((png, f"{kind.upper()}\n{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  {int(r.rna_n_nt or 0)} nt\n"
                                   f"shift {r['shift']:.0f} Å  rot {r['rot']:.0f}°  pm {r.protein_match:.2f}\n"
                                   f"LDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}"))
            items += [None] * ((-n) % N3)                # one row per pose class
            print(f"3/{comb}/{kind}: {n} rendered", flush=True)
        if items:
            montage(items, f"Classification 3: {comb.upper()} - {N3} post-cutoff examples per class",
                    os.path.join(OUT, "3", comb.replace(" ", "_"), f"_index_{comb.replace(' ', '_')}.png"), cols=N3)


def pick(d, col, label, kind, n=20, scheme="a"):
    s = d[d[col] == label]
    if scheme == "b_correct_rna":
        s = s[s.rna_ok]
    if scheme == "b_wrong_rna":                # closest to the well-folded threshold first
        s = s[~s.rna_ok].sort_values("rna_lddt", ascending=False)
    else:
        s = s.sort_values(SORT[kind][0], ascending=SORT[kind][1])
    out, per = [], {}
    for _, r in s.iterrows():
        if per.get(r.pdb, 0) >= 2:
            continue
        per[r.pdb] = per.get(r.pdb, 0) + 1
        out.append(r)
        if len(out) == n:
            break
    return out


def file_names():
    import sqlite3
    return {str(a).strip().upper(): b for a, b in sqlite3.connect(IRP.PERIODS["post"]).execute(
        "SELECT exp_db_id, FileName FROM pred_protein_rna")}


def render(r, fn, png):
    from pymol import cmd
    cif, _ = locate_inputs(r.pdb, fn, IRP.INPUT_ROOTS)
    af = IRP._locate_model(fn)
    if not (cif and af):
        return False
    P, R = str(r.native_pair).split(":")
    p, rr = str(r.model_pair).split(":")
    cmd.reinitialize()
    cmd.set("cif_use_auth", 1)
    cmd.load(cif, "nat"); cmd.load(af, "mdl")
    cmd.remove("hydro or solvent")
    try:
        cmd.align(f"mdl and chain {p} and name CA", f"nat and chain {P} and name CA", cycles=5)
    except Exception:                                    # noqa: BLE001
        return False
    cmd.hide("everything")
    cmd.show("surface", f"nat and chain {P}"); cmd.color("grey80", f"nat and chain {P}")
    cmd.show("cartoon", f"nat and chain {R}"); cmd.color(NAT_RNA, f"nat and chain {R}")
    cmd.show("cartoon", f"mdl and chain {rr}"); cmd.color(MDL_RNA, f"mdl and chain {rr}")
    cmd.set("cartoon_ring_mode", 3); cmd.set("transparency", 0.45)       # RNA stays visible behind the protein
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("depth_cue", 0)
    cmd.orient(f"(nat and chain {R}) or (mdl and chain {rr})")            # view the two RNAs face on
    cmd.zoom(f"(nat and chain {P}+{R}) or (mdl and chain {rr})", buffer=4, complete=1)
    cmd.clip("slab", 2000)                                                # no clipping planes through the surface
    cmd.png(png, width=520, height=520, dpi=150, ray=1)
    return os.path.exists(png)


def render_moves(r, fn, png):
    """As render, plus the AF3 protein chain as a salmon cartoon (to judge the pocket)."""
    from pymol import cmd
    if not render(r, fn, png):
        return False
    p, _ = str(r.model_pair).split(":")
    cmd.show("cartoon", f"mdl and chain {p}"); cmd.color("salmon", f"mdl and chain {p}")
    cmd.png(png, width=520, height=520, dpi=150, ray=1)
    return os.path.exists(png)


# paper class (placement) -> new pose class: the largest moves, N_MOVES examples each, most dramatic first
MOVES = [("incorrect_site_to_distal", "protein binding region mispredicted", "distal site", "pocket_rmsd", False),
         ("shifted_to_distal", "shifted nearby", "distal site", "shift", False),
         ("incorrect_rna_to_rotated", "right protein site, wrong RNA pose", "RNA rotated at the pocket", "rot", False)]
N_MOVES = 10
# the paper's rotated RNAs that leave the class
MOVES_ROTATED = [("rotated_to_incorrect_rna", "same place, RNA turned", "wrong RNA segment at the pocket", "rot", False),
                 ("rotated_to_shifted", "same place, RNA turned", "RNA shifted", "shift", False)]


# one example of each change from the paper's classes: (title, paper class, new class, preferred PDB or None)
CHANGES = [("incorrect RNA -> rotated RNA", "right protein site, wrong RNA pose", "RNA rotated at the pocket", "8VU0"),
           ("incorrect RNA -> slid duplex", "right protein site, wrong RNA pose", "same place, duplex slid", None),
           ("rotated RNA -> incorrect RNA contacts", "same place, RNA turned", "wrong RNA segment at the pocket", "7EIU"),
           ("incorrect binding site -> slid duplex", "protein binding region mispredicted", "same place, duplex slid", None)]


def main_changes():
    d = load(); fns = file_names()
    outdir = os.path.join(OUT, "3", "changes"); os.makedirs(outdir, exist_ok=True)
    items = []
    for k, (title, old, new, pref) in enumerate(CHANGES, 1):
        s = d[(d.placement == old) & (d.pose3 == new)]
        s = pd.concat([s[s.pdb == pref], s[s.pdb != pref]]) if pref else s
        for _, r in s.iterrows():
            png = os.path.join(outdir, f"{k:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}.png")
            if os.path.exists(png) or render_moves(r, fns.get(r.pdb, ""), png):
                items.append((png, f"{title.upper()} (n={len(s)})\n{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  "
                                   f"{int(r.rna_n_nt or 0)} nt\nwhole RNA {r.rna_translation:.0f} Å  {r.rna_rotation:.0f}°  "
                                   f"pm {r.protein_match:.2f}\nLDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}  "
                                   f"pocket {r.pocket_rmsd:.1f} Å"))
                print(title, r.pdb, r.native_pair, flush=True)
                break
    montage(items, "Paper class -> new class: one post-cutoff example per change; grey = native protein, salmon = AF3 "
                   "protein, dark blue = native RNA, teal = AF3 RNA", os.path.join(outdir, "_index_changes.png"), cols=4)


# incorrect interfaces by protein side (protein contact match) and RNA side (RNA contact match)
SIDES = [("both_sides_right", "both sides right (protein and RNA match >= 0.5)",
          lambda d: (d.protein_match >= 0.5) & (d.rna_match >= 0.5), lambda d: -(d.protein_match + d.rna_match)),
         ("right_site_wrong_segment", "right protein site, wrong RNA segment (protein >= 0.1, RNA < 0.1)",
          lambda d: (d.protein_match >= 0.1) & (d.rna_match < 0.1), lambda d: -d.protein_match),
         ("right_segment_wrong_surface", "right RNA segment, wrong protein surface (protein < 0.1, RNA >= 0.1)",
          lambda d: (d.protein_match < 0.1) & (d.rna_match >= 0.1), lambda d: -d.rna_match),
         ("neither_side", "neither side (protein and RNA match < 0.1)",
          lambda d: (d.protein_match < 0.1) & (d.rna_match < 0.1), lambda d: d.dockq)]


def main_sides(n=5):
    d = load(); fns = file_names()
    for tag, title, sel, key in SIDES:
        s = d[sel(d)].assign(_k=lambda x: key(x)).sort_values("_k")
        outdir = os.path.join(OUT, "3", "sides", tag); os.makedirs(outdir, exist_ok=True)
        items, seen = [], set()
        for _, r in s.iterrows():
            if r.pdb in seen:
                continue
            png = os.path.join(outdir, f"{len(items) + 1:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}.png")
            if not os.path.exists(png) and not render_moves(r, fns.get(r.pdb, ""), png):
                continue
            seen.add(r.pdb)
            items.append((png, f"{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  {int(r.rna_n_nt or 0)} nt\n"
                               f"protein match {r.protein_match:.2f}  RNA match {r.rna_match:.2f}\n"
                               f"{r.placement}\nwhole RNA {r.rna_translation:.0f} Å  {r.rna_rotation:.0f}°  "
                               f"LDDT_r {r.rna_lddt:.2f}"))
            if len(items) == n:
                break
        montage(items, f"{title}: {len(items)} of {len(s)} post-cutoff; grey = native protein, salmon = AF3 protein, "
                       f"dark blue = native RNA, teal = AF3 RNA", os.path.join(outdir, f"_index_{tag}.png"), cols=n)
        print(f"sides/{tag}: {len(items)} of {len(s)}", flush=True)


def main_correct_wrong_folds(n=5):
    """RNA chains placed correctly on their main partner (DockQ >= 0.23) although the RNA is misfolded and the main
    partner protein inaccurate (placement_per_rna_chain.py); lowest RNA LDDT first, one per PDB entry."""
    d = load(failed_only=False); fns = file_names()
    ic = pd.read_csv(os.path.join(_REPO, "results", "plots", "all_chains", "interface_class_composition",
                                  "interface_class_composition.tsv"), sep="\t")
    d = d.merge(ic[["period", "pdb", "native_pair", "protein_chain_tm", "protein_chain_lddt"]],
                on=["period", "pdb", "native_pair"], how="left")
    c = pd.read_csv(os.path.join(IRP.OUTDIR, "placement_per_rna_chain.tsv"), sep="\t")
    c = c[(c.period == "post") & (c["class"] == "Correct interface") & (c.rna_fold == "wrong") & ~c.partner_accurate]
    keys = set(zip(c.pdb, c.main_partner + ":" + c.rna_chain))
    s = d[[(p, q) in keys for p, q in zip(d.pdb, d.native_pair)]].sort_values("rna_lddt")
    pl = pd.read_csv(os.path.join(IRP.OUTDIR, "placement_per_rna_chain.tsv"), sep="\t")
    outdir = os.path.join(OUT, "3", "correct_interface_wrong_folds"); os.makedirs(outdir, exist_ok=True)
    items, seen = [], set()
    for _, r in s.iterrows():
        if r.pdb in seen:
            continue
        png = os.path.join(outdir, f"{len(items) + 1:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}.png")
        if not os.path.exists(png) and not render_moves(r, fns.get(r.pdb, ""), png):
            continue
        seen.add(r.pdb)
        m = pl[(pl.period == "post") & (pl.pdb == r.pdb) & (pl.main_partner + ":" + pl.rna_chain == r.native_pair)].iloc[0]
        items.append((png, f"{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  {int(r.rna_n_nt or 0)} nt\n"
                           f"LDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}\n"
                           f"main partner LDDT_p {r.protein_chain_lddt:.2f}  TM_p {r.protein_chain_tm:.2f}  "
                           f"assembly {m.assembly}\n"
                           f"protein side {r.protein_match:.2f}  RNA side {r.rna_match:.2f}"))
        if len(items) == n:
            break
    montage(items, f"Correct interface with misfolded RNA and inaccurate main partner: {len(items)} of {len(c)} "
                   f"post-cutoff RNA chains (lowest LDDT_r first); grey = native protein, salmon = AF3 protein, "
                   f"dark blue = native RNA, teal = AF3 RNA",
            os.path.join(outdir, "_index_correct_interface_wrong_folds.png"), cols=n)
    print(f"correct_interface_wrong_folds: {len(items)} of {len(c)}", flush=True)


def main_moves(moves=MOVES):
    d = load(); fns = file_names()
    for tag, old, new, key, asc in moves:
        s = d[(d.placement == old) & (d.pose3 == new)].sort_values(key, ascending=asc)
        outdir = os.path.join(OUT, "3", "moves", tag); os.makedirs(outdir, exist_ok=True)
        items, per = [], {}
        for _, r in s.iterrows():
            if per.get(r.pdb, 0) >= 2:
                continue
            png = os.path.join(outdir, f"{len(items) + 1:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}.png")
            if not os.path.exists(png) and not render_moves(r, fns.get(r.pdb, ""), png):
                continue
            per[r.pdb] = per.get(r.pdb, 0) + 1
            items.append((png, f"{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  {int(r.rna_n_nt or 0)} nt\n"
                               f"local shift {r['shift']:.0f} Å  rot {r['rot']:.0f}°  whole RNA {r.rna_translation:.0f} Å\n"
                               f"pocket RMSD {r.pocket_rmsd:.1f} Å  pm {r.protein_match:.2f}  LDDT_r {r.rna_lddt:.2f}"))
            if len(items) == N_MOVES:
                break
        montage(items, f"{old}  ->  {new}: {len(items)} of {len(s)} post-cutoff (largest {key} first); "
                       f"grey = native protein, salmon = AF3 protein, dark blue = native RNA, teal = AF3 RNA",
                os.path.join(outdir, f"_index_{tag}.png"), cols=5)
        print(f"moves/{tag}: {len(items)} of {len(s)} rendered", flush=True)


def render_pp(r, fn, png):
    """Incorrect protein assembly: also the other protein chains, native (wheat) and AF3 (salmon), after superposing
    the AF3 interface protein chain on the native one."""
    from pymol import cmd
    if not render(r, fn, png):
        return False
    P, R = str(r.native_pair).split(":")
    p, _ = str(r.model_pair).split(":")
    cmd.show("cartoon", f"nat and polymer.protein and not chain {P}"); cmd.color("wheat", f"nat and polymer.protein and not chain {P}")
    cmd.show("cartoon", f"mdl and polymer.protein and not chain {p}"); cmd.color("salmon", f"mdl and polymer.protein and not chain {p}")
    cmd.orient("nat or mdl")
    cmd.zoom("nat or mdl", buffer=2, complete=1)
    cmd.clip("slab", 2000)
    cmd.png(png, width=520, height=520, dpi=150, ray=1)
    return os.path.exists(png)


def main_pp(n=N3):
    """N3 post-cutoff incorrect interfaces with an incorrect protein assembly (lowest protein-protein DockQ first)."""
    d = load(); fns = file_names()
    s = d[d.pp_partner_min < 0.23].sort_values("pp_partner_min")
    outdir = os.path.join(OUT, "3", "incorrect_protein_assembly"); os.makedirs(outdir, exist_ok=True)
    items, seen = [], set()
    for _, r in s.iterrows():
        if r.pdb in seen:
            continue
        png = os.path.join(outdir, f"{len(items) + 1:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}"
                                   f"_ppdockq{r.pp_partner_min:.2f}_dockq{r.dockq:.2f}.png")
        if not os.path.exists(png) and not render_pp(r, fns.get(r.pdb, ""), png):
            continue
        seen.add(r.pdb)
        items.append((png, f"{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}  DockQ_pp {r.pp_partner_min:.2f}\n"
                           f"{r.pose3}\nLDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}"))
        if len(items) == n:
            break
    montage(items, f"Classification 3: INCORRECT PROTEIN ASSEMBLY - {len(items)} post-cutoff examples (lowest DockQ_pp)",
            os.path.join(outdir, "_index_incorrect_protein_assembly.png"), cols=n)
    print(f"3/incorrect_protein_assembly: {len(items)} rendered", flush=True)


def montage(items, title, path, cols=4):
    n = len(items)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.6 * rows), squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, item in zip(np.ravel(axes), items):
        if item is None:
            continue
        png, lab = item
        ax.imshow(plt.imread(png)); ax.set_title(lab, fontsize=8, color="#4E9A96", fontweight="bold")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    d = load(); fns = file_names()
    for scheme, classes in CLASSES.items():
        for kind, (col, label) in classes.items():
            outdir = os.path.join(OUT, scheme, kind.replace(" ", "_")); os.makedirs(outdir, exist_ok=True)
            items = []
            for k, r in enumerate(pick(d, col, label, kind, scheme=scheme), 1):
                png = os.path.join(outdir, f"{k:02d}_{r.pdb}_{str(r.native_pair).replace(':', '')}_{int(r.rna_n_nt or 0)}nt"
                                           f"_shift{r['shift']:.0f}A_rot{r['rot']:.0f}deg_dockq{r.dockq:.2f}.png")
                if not os.path.exists(png) and not render(r, fns.get(r.pdb, ""), png):
                    continue
                grp = (f"\n{r.b_protein.replace(' protein-protein', ' pp')}  LDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}"
                       if scheme.startswith("b") else f"\nLDDT_r {r.rna_lddt:.2f}  TM_r {r.rna_tm:.2f}")
                items.append((png, f"{r.pdb} {r.native_pair}  DockQ {r.dockq:.2f}\n{int(r.rna_n_nt or 0)} nt  shift "
                                   f"{r['shift']:.0f} Å  rot {r['rot']:.0f}°  pm {r.protein_match:.2f}{grp}"))
            if items:
                how = ("highest LDDT_r first (closest to the well-folded threshold)" if scheme == "b_wrong_rna" else
                       "largest rotation" if kind == "rotated" else "all slid duplexes" if kind == "slid" else "largest shift")
                montage(items, f"{kind.upper()} ({scheme}) - {len(items)} post-cutoff examples ({how})",
                        os.path.join(outdir, f"_index_{kind.replace(' ', '_')}.png"))
            print(f"{scheme}/{kind}: {len(items)} rendered", flush=True)


if __name__ == "__main__":
    scheme = sys.argv[sys.argv.index("--scheme") + 1] if "--scheme" in sys.argv else ""
    {"3": main3, "3pp": main_pp, "moves": main_moves, "changes": main_changes, "sides": main_sides, "correct_wrong_folds": main_correct_wrong_folds,
     "moves_rotated": lambda: main_moves(MOVES_ROTATED)}.get(scheme, main)()
