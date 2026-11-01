"""Per-interface context of the placement classes, for all protein-RNA chain pairs.

Adds to every interface of interface_rna_pose.tsv (whose `placement` column is the paper's class, as stored in the
databases) the chain-level context used by interface_failure_modes_figure.py, placement_per_rna_chain.py and
placement_attributes_table.py:
  rna_lddt, rna_tm, rna_ok    per-chain fold of the interface's RNA chain (well folded: LDDT_r > 0.85 and, from
                              60 nt, TM_r > 0.5), from RNA_LDDT_AllChains / RNA_TM_AllChains or, where missing, the
                              exact-mapping fallback (rna_chain_scores_fallback.py)
  pp_partner_min              lowest protein-protein DockQ of the interface's protein chain against its protein partners
                              (protein assembly correct when >= 0.23)
  t_loc, a_loc                translation / rotation of the native interface nucleotides in the protein frame
  pose_t, pose_a              the translation / rotation the pose rules below used
  placement_b1                an alternative pose class (last field), explored on 2026-10-24/25 and not adopted for the
                              paper: slid duplex (with an intact pocket); with protein contact (match >= 0.1) rotated RNA
                              (well folded, > 45 deg), correct RNA placement / wrong RNA segment; without contact,
                              incorrect binding site (pocket > 4 A), distal (whole RNA > 20 A), rotated (well folded,
                              whole RNA <= 10 A, > 45 deg) or shifted
The remaining placement_* / b_* columns are earlier variants of that exploration.

Run in the RNABench env:  python evaluation/placement_all_chains.py [--reuse]
(--reuse reads the local poses of placement_all_chains_local_pose.tsv instead of recomputing them.)
Writes results/plots/all_chains/interface_rna_pose/placement_all_chains.tsv and prints the class shares.
"""
import csv
import os
import sqlite3
import sys
from collections import Counter

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from evaluation import interface_rna_pose as IRP
from evaluation.exact_residue_mapping import ComplexMapping, MIN_NAME_AGREEMENT, interfaces_of, locate_inputs

OUT = os.path.join(IRP.OUTDIR, "placement_all_chains.tsv")
POSE_TSV = os.path.join(IRP.OUTDIR, "interface_rna_pose.tsv")
SYM_TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_symmetry_check", "interface_symmetry_check.tsv")
RNA_MATCH_OK = 0.5


def local_pose():
    """(period, PDB, native pair) -> local translation, rotation and n of the native interface nucleotides."""
    out = {}
    for period, db in IRP.PERIODS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
                   p.ChainIDpairList_proteinRNA AS p_pairs, e.AAMotif AS e_aa, e.RNAMotif AS e_rna,
                   e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for k, row in enumerate(rows):
            cif, js = locate_inputs(row["id"], row["fn"], IRP.INPUT_ROOTS)
            af = IRP._locate_model(row["fn"])
            if not (cif and af):
                continue
            try:
                mdl, mseqs = IRP.read_model(af)
                mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
                IRP.drop_inconsistent_ost_chains(mapping, mseqs)
                IRP.interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                                     dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                                     mapping)
                nat = IRP.native_coords(cif)
            except Exception as e:                       # noqa: BLE001 -- dry run: skip and count
                print(f"  skip {period} {row['id']}: {type(e).__name__}", flush=True)
                continue
            for P, R, aa, rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
                laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
                p, r = mapping.pred_chain(P), mapping.pred_chain(R)
                if None in (laP, laR, p, r):
                    continue
                Xp, Yp = IRP._pair_atoms(mapping, laP, P, p, range(len(mapping.label_residues[laP])), nat, mdl, (IRP.PROT_ATOM,))
                if len(Xp) < 3:
                    continue
                Rm, t, _, _ = IRP.fit_with_rejection(Xp, Yp)
                Er, _, fr = mapping._motif_indices(laR, rna, "author")
                Xr, Yr = IRP._pair_atoms(mapping, laR, R, r, Er if fr >= MIN_NAME_AGREEMENT else [], nat, mdl, IRP.RNA_ATOMS)
                if len(Xr) >= 3:
                    pose = IRP.rna_pose((Rm @ Xr.T).T + t, Yr)
                    out[(period, str(row["id"]).strip().upper(), f"{P}:{R}")] = (pose["translation"], pose["rotation"], len(Xr))
            if k % 100 == 0:
                print(f"  {period}: {k}/{len(rows)} complexes", flush=True)
    return out


def pp_dockq():
    """(period, PDB, frozenset(predicted protein chains)) -> protein-protein DockQ from RBP_DockQ."""
    import json, ast
    out = {}
    for period, db in IRP.PERIODS.items():
        for pid, cell, pch in sqlite3.connect(db).execute("SELECT exp_db_id, RBP_DockQ, ProteinChainIDs FROM pred_protein_rna"):
            prot = set(map(str, IRP._list(pch) or []))
            for item in IRP._list(cell) or []:
                for k, v in (item.items() if isinstance(item, dict) else []):
                    a, b = str(k).split(":")
                    if a in prot and b in prot and v is not None:
                        out[(period, str(pid).strip().upper(), frozenset((a, b)))] = float(v)
    return out


def rna_fold():
    """(period, PDB, predicted RNA chain) -> (per-chain all-atom LDDT_r, TM_r). a2021 keyed cells; b2021 matrix rows
    in the order of RNAChainIDs, best experimental chain per row."""
    import json, ast
    def parse(v):
        for f in (json.loads, ast.literal_eval):
            try:
                return f(v)
            except Exception:                            # noqa: BLE001
                pass
    out = {}
    for period, db in IRP.PERIODS.items():
        for pid, ch, l, t in sqlite3.connect(db).execute(
                "SELECT exp_db_id, RNAChainIDs, RNA_LDDT_AllChains, RNA_TM_AllChains FROM pred_protein_rna"):
            pid = str(pid).strip().upper(); ch, l, t = parse(ch) or ch, parse(l), parse(t)
            if isinstance(ch, str):                       # a single RNA chain is stored as a plain letter ('B')
                ch = [c.strip() for c in ch.split(',') if c.strip()]
            if isinstance(l, list) and l and isinstance(l[0], list):        # matrix layout
                for k, c in enumerate(ch if isinstance(ch, list) else []):
                    if k < len(l) and k < len(t or []):
                        out[(period, pid, str(c))] = (max(l[k]), max(t[k]))
                continue
            kv = lambda x: {kk: vv for e in (x if isinstance(x, list) else [x]) if isinstance(e, dict) for kk, vv in e.items()}
            L, T = kv(l), kv(t)
            for k, v in L.items():
                c = str(k).split(":")[-1]
                tv = next((vv for kk, vv in T.items() if str(kk).split(":")[-1] == c), np.nan)
                out[(period, pid, c)] = (_f(v), _f(tv))
    return out


def rna_fold_fallback():
    """(period, PDB, model RNA chain) -> (LDDT_r, TM_r) from evaluation/rna_chain_scores_fallback.py for chains
    without a stored per-chain value; LDDT_r = 0 on <= 5 mapped nucleotides is OpenStructure skipping the chain."""
    path = os.path.join(IRP.OUTDIR, "rna_chain_fallback.tsv")
    out = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path), delimiter="\t"):
            l = _f(r["lddt"])
            if l == 0 and int(r["n_mapped"]) <= 5:
                l = np.nan
            out[(r["period"], r["pdb"], r["model_chain"])] = (l, _f(r["tm"]))
    return out


def well_folded(lddt, tm, n_nt):
    """paper definition: LDDT_r > 0.85 and, from 60 nt, TM_r > 0.5"""
    return lddt == lddt and lddt > 0.85 and (n_nt < 60 or (tm == tm and tm > 0.5))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def classify(r, partner_ok, sym):
    if r["dockq"] >= 0.23:
        return "correct interface"
    if partner_ok:
        return "protein partner misplaced"
    s = sym.get((r["period"], r["pdb"], r["native_pair"]))
    duplex = s is not None and s.get("duplex") == "True" and _f(s["occ_duplex"]) >= 0.5 and _f(s["occ_self"]) < 0.5
    shift, tilt = _f(r.get("duplex_shift")), _f(r.get("duplex_axis_tilt"))
    # slid duplex only with an intact pocket: with a mispredicted protein the duplex pose in the protein frame is not
    # meaningful
    pocket_ok = not (r["pocket_rmsd"] == r["pocket_rmsd"] and r["pocket_rmsd"] > IRP.POCKET_A)
    if duplex and pocket_ok and shift == shift and shift <= IRP.SAME_PLACE_A and tilt <= IRP.TURNED_DEG:
        return "same place, duplex slid"
    t, a = (shift, tilt) if duplex and shift == shift else (r["t_loc"], r["a_loc"])
    if t != t:
        t, a = r["rna_translation"], r["rna_rotation"]
    # the protein site is contacted: what is wrong is the RNA at that site
    if r["protein_match"] == r["protein_match"] and r["protein_match"] >= IRP.PROTEIN_SITE_MATCH:
        if t == t and t <= IRP.SAME_PLACE_A:            # native interface nucleotides at the site
            if a == a and a > IRP.TURNED_DEG:
                return "RNA rotated at the pocket" if r["rna_ok"] else "wrong RNA segment at the pocket"
            return ("correct RNA placement, incorrect contacts" if r["rna_match"] >= RNA_MATCH_OK
                    else "wrong RNA segment at the pocket")
        rot = r["rna_rotation"]                           # native nucleotides moved away: is the RNA turned?
        return ("RNA rotated at the pocket" if rot == rot and rot > IRP.TURNED_DEG and r["rna_ok"]
                else "wrong RNA segment at the pocket")
    # no contact with the native site: location from the whole RNA (or duplex), orientation from its rotation. The
    # native interface nucleotides are not used here: an RNA flipped in place moves them far away without moving.
    t, a = whole_pose(r, duplex, shift, tilt)
    if t != t:
        return ""
    # a mispredicted pocket first: with the AF3 protein misfolded or its domains misplaced, the distances in the
    # protein frame are not meaningful
    if r["pocket_rmsd"] == r["pocket_rmsd"] and r["pocket_rmsd"] > IRP.POCKET_A:
        return "protein binding region mispredicted"
    if t > IRP.NEARBY_A:
        return "distal site"
    # without protein contact there is no correct site: a well-folded RNA flipped in place is rotated, anything else
    # within 20 A is shifted
    if t <= IRP.SAME_PLACE_A and a == a and a > IRP.TURNED_DEG and r["rna_ok"]:
        return "RNA rotated at the pocket"
    return "RNA shifted"


def whole_pose(r, duplex, shift, tilt):
    """Translation / rotation of the whole RNA in the protein frame (of the whole duplex for a strand exchange)."""
    if duplex and shift == shift:
        return shift, tilt
    return r["rna_translation"], r["rna_rotation"]


def classify_b(r, sym):
    """Classification b: correct interface, else protein assembly x RNA fold x pose.
    protein-protein correct: the interface's protein chain has DockQ >= 0.23 against every protein partner it
    contacts in the deposited structure (a single protein chain counts as correct); RNA correct: well_folded().
    pose: slid duplex (strand exchange, duplex moved <= 10 A, axis <= 45 deg) / at the site (protein contact
    match >= 0.1 or native interface nucleotides within 10 A): rotated (> 45 deg) or in place / shifted (10-20 A) /
    distal (> 20 A)."""
    if r["dockq"] >= 0.23:
        return "correct interface", "", "", ""
    ppc = "correct protein-protein" if not (r["pp_partner_min"] == r["pp_partner_min"] and r["pp_partner_min"] < 0.23) \
        else "incorrect protein-protein"
    rna = "correct RNA" if r["rna_ok"] else "wrong RNA"
    s = sym.get((r["period"], r["pdb"], r["native_pair"]))
    duplex = s is not None and s.get("duplex") == "True" and _f(s["occ_duplex"]) >= 0.5 and _f(s["occ_self"]) < 0.5
    shift, tilt = _f(r.get("duplex_shift")), _f(r.get("duplex_axis_tilt"))
    if duplex and shift == shift and shift <= IRP.SAME_PLACE_A and tilt <= IRP.TURNED_DEG:
        pose = "slid"
    else:
        t, a = (shift, tilt) if duplex and shift == shift else (r["t_loc"], r["a_loc"])
        if t != t:
            t, a = r["rna_translation"], r["rna_rotation"]
        at_site = (r["protein_match"] == r["protein_match"] and r["protein_match"] >= IRP.PROTEIN_SITE_MATCH) \
            or (t == t and t <= IRP.SAME_PLACE_A)
        if at_site:
            rot = a if (t == t and t <= IRP.SAME_PLACE_A) else r["rna_rotation"]
            pose = "rotated" if rot == rot and rot > IRP.TURNED_DEG else "in place"
        elif t != t:
            pose = ""
        else:
            pose = "shifted" if t <= IRP.NEARBY_A else "distal"
    return f"wrong interface / {ppc} / {rna} / {pose}", ppc, rna, pose


def main():
    loc_path = OUT.replace(".tsv", "_local_pose.tsv")
    if "--reuse" in sys.argv and os.path.isfile(loc_path):
        loc = {(r["period"], r["pdb"], r["native_pair"]): (_f(r["t"]), _f(r["a"]), int(r["n"]))
               for r in csv.DictReader(open(loc_path), delimiter="\t")}
    else:
        loc = local_pose()
        with open(loc_path, "w") as fh:
            w = csv.writer(fh, delimiter="\t"); w.writerow(["period", "pdb", "native_pair", "t", "a", "n"])
            w.writerows([[*k, *v] for k, v in loc.items()])
    rows = list(csv.DictReader(open(POSE_TSV), delimiter="\t"))
    for r in rows:
        for c in ("dockq", "pocket_rmsd", "rna_match", "protein_match", "rna_translation", "rna_rotation"):
            r[c] = _f(r[c])
        r["t_loc"], r["a_loc"], r["n_loc"] = loc.get((r["period"], r["pdb"], r["native_pair"]), (np.nan, np.nan, 0))
    sym = {(r["period"], r["pdb"].upper(), r["native_pair"]): r for r in csv.DictReader(open(SYM_TSV), delimiter="\t")}
    good = {}                                     # (period, pdb, native RNA chain) -> native protein chains with DockQ >= 0.23
    pred = {}                                     # (period, pdb, native chain) -> predicted chain (DB naming of RBP_DockQ)
    for r in rows:
        P, R = r["native_pair"].split(":"); p, rr = r["pred_pair"].split(":")
        pred[(r["period"], r["pdb"], P)], pred[(r["period"], r["pdb"], R)] = p, rr
        if r["dockq"] >= 0.23:
            good.setdefault((r["period"], r["pdb"], R), set()).add(P)
    pp = pp_dockq()
    fold = rna_fold()
    fb = rna_fold_fallback()
    for r in rows:
        P, R = r["native_pair"].split(":"); key = (r["period"], r["pdb"])
        r["rna_lddt"], r["rna_tm"] = fold.get(key + (r["pred_pair"].split(":")[1],), (np.nan, np.nan))
        if r["rna_lddt"] != r["rna_lddt"]:              # no stored per-chain value: exact-mapping fallback
            r["rna_lddt"], r["rna_tm"] = fb.get(key + (r["pred_pair"].split(":")[1],), (np.nan, np.nan))
        r["rna_ok"] = well_folded(r["rna_lddt"], r["rna_tm"], _f(r.get("rna_n_nt")))
        partners = good.get(key + (R,), set()) - {P}
        # the RNA is placed correctly on another protein B, and this protein P is misplaced against B
        r["pp_dockq_min"] = min((pp.get(key + (frozenset((pred.get(key + (P,)), pred.get(key + (B,)))),), np.nan)
                                 for B in partners), default=np.nan, key=lambda v: v if v == v else 9)
        r["placement_v2"] = classify(r, bool(partners) and r["pp_dockq_min"] == r["pp_dockq_min"]
                                     and r["pp_dockq_min"] < 0.23, sym)
        pc = pred.get(key + (P,))                 # all protein partners of this protein chain (b)
        vals = [v for (pe, pi, pair), v in pp.items() if pe == r["period"] and pi == r["pdb"] and pc in pair]
        r["pp_partner_min"] = min(vals) if vals else np.nan
        r["placement_b"], r["b_protein"], r["b_rna"], r["b_pose"] = classify_b(r, sym)
        # b (final) and b.1: protein assembly x RNA fold x the pose rules of a (protein contacts as site evidence,
        # duplex before pocket, distal only without protein contact, pocket after distal, local pose; rotated only
        # for a correct RNA). The partner rule of a is replaced by the protein-protein axis.
        if r["dockq"] >= 0.23:
            r["placement_bf"] = r["placement_b1"] = "correct interface"
        else:
            pose = classify(r, False, sym)
            # the translation / rotation the pose rules used: whole duplex for a strand exchange; with contact at the
            # native site the native interface nucleotides, without it the whole RNA
            s = sym.get((r["period"], r["pdb"], r["native_pair"]))
            dup = s is not None and s.get("duplex") == "True" and _f(s["occ_duplex"]) >= 0.5 and _f(s["occ_self"]) < 0.5
            ds, dt = _f(r.get("duplex_shift")), _f(r.get("duplex_axis_tilt"))
            contact = r["protein_match"] == r["protein_match"] and r["protein_match"] >= IRP.PROTEIN_SITE_MATCH
            r["pose_t"], r["pose_a"] = ((ds, dt) if dup and ds == ds else
                                        (r["t_loc"], r["a_loc"]) if contact and r["t_loc"] == r["t_loc"] else
                                        whole_pose(r, dup, ds, dt))
            pp_ok = not (r["pp_partner_min"] == r["pp_partner_min"] and r["pp_partner_min"] < 0.23)
            ppc = "pp correct" if pp_ok else "pp incorrect"
            rna2 = "RNA correct" if r["rna_ok"] else "RNA wrong"
            rna3 = "RNA correct" if r["rna_ok"] else ("RNA intermediate" if r["rna_lddt"] == r["rna_lddt"]
                                                       and r["rna_lddt"] > 0.75 else "RNA wrong")
            r["placement_bf"] = f"{ppc} / {rna2} / {pose}"
            r["placement_b1"] = f"{ppc} / {rna3} / {pose}"
    keys = ["period", "pdb", "pred_pair", "native_pair", "dockq", "pp_dockq_min", "protein_match", "rna_match", "pocket_rmsd", "rna_lddt", "rna_tm", "rna_ok",
            "rna_translation", "rna_rotation", "t_loc", "a_loc", "n_loc", "placement", "placement_v2",
            "pp_partner_min", "b_protein", "b_rna", "b_pose", "placement_b", "placement_bf", "placement_b1",
            "pose_t", "pose_a"]
    with open(OUT, "w") as fh:
        w = csv.DictWriter(fh, keys, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    for per in ("post", "pre"):
        rr = [r for r in rows if r["period"] == per and r["dockq"] == r["dockq"] and r["dockq"] < 0.23]
        print(f"\n{per}-cutoff failed interfaces: {len(rr)}")
        for c, n in Counter(r["placement_v2"] for r in rr).most_common():
            print(f"  {c:45s} {n:4d}  {100 * n / len(rr):5.1f}%")
        print("  old class -> new class (top moves):")
        for (o, nw), n in Counter((r["placement"], r["placement_v2"]) for r in rr).most_common(12):
            print(f"    {o:38s} -> {nw:42s} {n}")
    for pid in ("8YE6", "8VU0"):
        for r in rows:
            if r["period"] == "post" and r["pdb"] == pid:
                print(f"{pid} {r['native_pair']}: old {r['placement']} | new {r['placement_v2']} | local t {r['t_loc']:.1f} A, a {r['a_loc']:.0f} deg (n {r['n_loc']})")


if __name__ == "__main__":
    main()
