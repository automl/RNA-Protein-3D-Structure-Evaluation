"""
Placement of duplex RNAs from the pose of the whole duplex.

A failed interface whose native RNA is a duplex strand can show the duplex at the native site with the wrong strand
in the native position (interface_symmetry_check.py: occ_duplex >= 0.5 while occ_self < 0.5). AF3 then slides the
duplex along its own helix axis and exchanges the strands. Measured on that strand alone, the RNA looks turned by
about 180 degrees (the duplex dyad) or moved by more than 10 A, although the duplex barely moves. For these interfaces
the pose is therefore measured on the whole duplex (C4' atoms of R and of the paired part of its partner strand,
nucleotides present in both model and native, in the same protein frame as interface_rna_pose.py): centroid shift (split along and across the native helix axis) and tilt of the helix axis.
Where the pose decides the class (not 'right protein site, wrong RNA pose' or 'protein binding region
mispredicted'), the class is read from the duplex with the thresholds of interface_rna_pose.py:
  same place, duplex slid    duplex moved <= SAME_PLACE_A and its axis tilted <= TURNED_DEG (figure label
                             'Correct site, slid duplex'): the duplex stays on its native helical track, slid a few
                             base pairs with the strands exchanged
  same place, RNA turned     duplex moved <= SAME_PLACE_A, axis tilted more
  shifted nearby             duplex moved <= NEARBY_A
  different region           beyond
The strand-based class is kept in placement_strand.

Rewrites interface_rna_pose.tsv in place (adds placement_strand, duplex_shift, duplex_along_axis, duplex_across_axis,
duplex_axis_tilt) and RNA_interface_placement in the databases (interface_rna_pose.write_db), including the copies
in data/all_data_zip/all_databases. Idempotent: the strand-based class is always taken from placement_strand.
Run after interface_rna_pose.py and interface_symmetry_check.py:  python evaluation/apply_duplex_pose.py
"""
import csv
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))

from evaluation.exact_residue_mapping import ComplexMapping, interface_scores, locate_inputs
from evaluation.interface_rna_pose import (INPUT_ROOTS, NEARBY_A, PERIODS, PROT_ATOM, SAME_PLACE_A, TURNED_DEG, _locate_model,
                                           _pair_atoms, drop_inconsistent_ost_chains, fit_with_rejection,
                                           native_coords, read_model)

POSE_TSV = next(p for p in (os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
                            for d in ("plots", "plots_odl")) if os.path.exists(p))
SYM_TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_symmetry_check", "interface_symmetry_check.tsv")
OCC_MIN = 0.5
PAIRED_A = 20.0                     # partner nucleotides within this C4'-C4' distance of R form the duplex
POSE_INDEPENDENT = ("correct interface", "right protein site, wrong RNA pose", "protein binding region mispredicted")
ZIP_DBS = {"post_zip": os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases", "a2021_post_cutoff", "protein_rna.db"),
           "pre_zip": os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases", "b2021_pre_cutoff", "protein_rna.db")}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _key(r):
    return r["period"], r["pdb"].upper(), r["native_pair"]


def flagged(sym_rows):
    """{key: duplex partner (author chain)} of interfaces with the duplex at the native site and the strand elsewhere"""
    return {_key(r): r["partner_native"] for r in sym_rows
            if r.get("duplex") == "True" and _f(r["occ_duplex"]) >= OCC_MIN and _f(r["occ_self"]) < OCC_MIN}


def _c4(coords, chain):
    return np.array([a["C4'"] for k, a in sorted(coords.items()) if k[0] == chain and "C4'" in a])


def _axis(X):
    return np.linalg.svd(X - X.mean(0))[2][0]


def duplex_pose(targets):
    """{key: dict(shift, along, across, tilt)} for targets {key: partner chain}"""
    out = {}
    for period, db in PERIODS.items():
        ids = {k[1] for k in targets if k[0] == period}
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.ChainIDpairList_proteinRNA AS p_pairs, p.AAMotif AS p_aa,
                   p.RNAMotif AS p_rna, e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e
              ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for row in rows:
            pid = str(row["id"]).strip().upper()
            if pid not in ids:
                continue
            cif, js = locate_inputs(row["id"], row["fn"], INPUT_ROOTS)
            mdl, mseqs = read_model(_locate_model(row["fn"]))
            mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
            drop_inconsistent_ost_chains(mapping, mseqs)
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                             mapping)
            nat = native_coords(cif)
            for k in [k for k in targets if k[:2] == (period, pid)]:
                P, R = k[2].split(":")
                R2 = targets[k]
                laP, laR, laR2 = (mapping.auth_to_label.get(c) for c in (P, R, R2))
                p, r, r2 = (mapping.pred_chain(c) for c in (P, R, R2))
                if None in (laP, laR, laR2, p, r, r2):
                    continue
                X, Y = _pair_atoms(mapping, laP, P, p, range(len(mapping.label_residues[laP])), nat, mdl, (PROT_ATOM,))
                Rm, t, _n, _rms = fit_with_rejection(X, Y)
                # nucleotides present in both model and native (exact residue mapping); of the partner strand only
                # the part paired with R (C4' within PAIRED_A of R), so a long partner (e.g. an rRNA) does not
                # dominate the duplex
                mR, nR = _pair_atoms(mapping, laR, R, r, range(len(mapping.label_residues[laR])), nat, mdl, ("C4'",))
                mR2, nR2 = _pair_atoms(mapping, laR2, R2, r2, range(len(mapping.label_residues[laR2])), nat, mdl, ("C4'",))
                if len(nR) < 3 or len(nR2) < 3:
                    continue
                paired = np.linalg.norm(nR2[:, None] - nR[None], axis=2).min(1) <= PAIRED_A
                if paired.sum() < 3:
                    continue
                nD = np.vstack([nR, nR2[paired]])
                mD = (Rm @ np.vstack([mR, mR2[paired]]).T).T + t
                ax = _axis(nD)
                d = mD.mean(0) - nD.mean(0)
                along = float(d @ ax)
                out[k] = dict(shift=float(np.linalg.norm(d)), along=abs(along),
                              across=float(np.linalg.norm(d - along * ax)),
                              tilt=float(np.degrees(np.arccos(min(1.0, abs(float(_axis(mD) @ ax)))))))
    return out


SLID = "same place, duplex slid"


def pose_class(shift, tilt):
    """class of a duplex at the native site with the wrong strand in place"""
    if shift <= SAME_PLACE_A:
        return SLID if tilt <= TURNED_DEG else "same place, RNA turned"
    return "shifted nearby" if shift <= NEARBY_A else "different region"


def apply(rows, poses):
    n = 0
    for r in rows:
        strand = r.get("placement_strand") or r["placement"]
        r["placement_strand"] = strand
        r["placement"] = strand
        for c in ("duplex_shift", "duplex_along_axis", "duplex_across_axis", "duplex_axis_tilt"):
            r[c] = ""
        r.pop("shifted_subclass", None)
        p = poses.get(_key(r))
        if p is None:
            continue
        r.update(duplex_shift=f"{p['shift']:.2f}", duplex_along_axis=f"{p['along']:.2f}",
                 duplex_across_axis=f"{p['across']:.2f}", duplex_axis_tilt=f"{p['tilt']:.1f}")
        if strand not in POSE_INDEPENDENT and strand:
            r["placement"] = pose_class(p["shift"], p["tilt"])
            n += r["placement"] != strand
    return n


def main():
    rows = list(csv.DictReader(open(POSE_TSV), delimiter="\t"))
    targets = flagged(list(csv.DictReader(open(SYM_TSV), delimiter="\t")))
    poses = duplex_pose(targets)
    n = apply(rows, poses)
    keys = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "shifted_subclass"]
    with open(POSE_TSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(poses)} duplex interfaces measured, {n} reclassified -> {POSE_TSV}")
    import interface_rna_pose as pose
    pose.write_db(rows, dict(PERIODS))
    for period, db in (("post", ZIP_DBS["post_zip"]), ("pre", ZIP_DBS["pre_zip"])):   # write_db keys are periods
        if os.path.exists(db):
            pose.write_db(rows, {period: db})
    return rows


if __name__ == "__main__":
    main()
