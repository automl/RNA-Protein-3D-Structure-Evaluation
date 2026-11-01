"""
RNA-RNA interfaces of AF3 predictions: contact recovery and pose of the partner RNA.

Input: the run_eval.py databases of the RNA-RNA sets (results/rna_rna_eval/{a2021,b2021}/run.db;
pred_rna_rna / exp_rna_rna with ChainIDpairList_rnaRNA and RNAMotif) and their structures
(data/RNA_RNA_ab/...). Contacts come from ContactList, one entry per ordered chain pair of
ChainIDpairList_rnaRNA, each contact a residue pair "G(8)-C(7): 2.89" (residue of the first chain -
residue of the second chain: distance); RNAMotif is per chain, not per pair, and is not used.
For every native RNA-RNA interface (an unordered pair of RNA chains that
contact each other in the deposited structure):

  * chains and residues are put in correspondence with evaluation/exact_residue_mapping.py; no
    OpenStructure output is used (its chain mapping drops short or repeated RNA chains here), so
    every chain is paired by sequence identity (identical copies: largest contact overlap) and its
    residues are mapped by exact alignment to the AF3 sequence;
  * contact match: Jaccard overlap of the predicted and the mapped experimental contacting
    nucleotides, per side and pooled over both sides (sum |intersection| / sum |union|);
  * pose: the AF3 receptor RNA (the longer chain of the pair) is superposed on the native one (C4',
    else P; PyMOL-style outlier rejection) and the partner RNA is compared in that frame:
    translation of its centroid, rotation still needed to superpose it, and its RMSD without
    further fitting (ligand RMSD, LRMSD);
  * RNA_LDDT / Complex_LDDT are OpenStructure all-atom LDDT (RNA_LDDT_allAtom,
    Complex_LDDT_allAtom), as elsewhere in the paper;
  * AF3 confidence of the chain pair: chain_pair_iptm and chain_pair_pae_min (minimum of both
    directions) from *_summary_confidences_0.json.

Writes results/plots/all_chains/rna_rna_interfaces/rna_rna_interfaces.tsv.
Run in the RNABench env:  python evaluation/rna_rna_interfaces.py
"""

import csv
import json
import os
import re
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.exact_residue_mapping import ComplexMapping, _list, jaccard
from evaluation.interface_rna_pose import (RNA_ATOMS, _pair_atoms, fit_with_rejection, native_coords,
                                           read_model, rna_pose)

from evaluation.bundle_paths import ALL_DATA, ALL_DB
SETS = {  # period: (database, ground-truth folder, AF3 folder) of the published multi-chain RNA sets
    p: (os.path.join(ALL_DB, s, "rna_multichain.db"), os.path.join(ALL_DATA, s, "rna_multichain", "pdb"),
        os.path.join(ALL_DATA, s, "rna_multichain", "af"))
    for p, s in (("post", "a2021_post_cutoff"), ("pre", "b2021_pre_cutoff"))
}
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_rna_interfaces")
_CONTACT = re.compile(r"([A-Za-z0-9]+)\((-?\d+)\)\s*-\s*([A-Za-z0-9]+)\((-?\d+)\)")   # "G(8)-C(7): 2.89"


def parse_rr_contacts(pairs_raw, contacts_raw):
    """{(X, Y): [(name, number)]} -- the residues of chain X that contact chain Y, from ContactList
    (both orientations of a pair are merged)."""
    pairs, contacts = _list(pairs_raw), _list(contacts_raw)
    out = {}
    for pr, cl in zip(pairs, contacts):
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            continue
        x, y = str(pr[0]), str(pr[1])
        items = cl if isinstance(cl, list) else _list(cl)
        for c in items:
            m = _CONTACT.search(str(c))
            if not m:
                continue
            n1, k1, n2, k2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            out.setdefault((x, y), set()).add((n1, k1))
            out.setdefault((y, x), set()).add((n2, k2))
    return {k: sorted(v, key=lambda t: t[1]) for k, v in out.items()}


def _first(v):
    """Scalar from a number or a JSON list cell (first element)."""
    try:
        return float(json.loads(v)[0]) if str(v).startswith("[") else float(v)
    except (TypeError, ValueError, IndexError):
        return np.nan


def af3_pair_confidence(af_dir, file_name, model_chain_order):
    """{frozenset(chain pair): (chain_pair_iptm, chain_pair_pae_min)} from summary_confidences_0."""
    stem = os.path.basename(str(file_name)).replace("_model_0.cif", "")
    p = os.path.join(af_dir, f"{stem}_summary_confidences_0.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    ip, pae = d.get("chain_pair_iptm"), d.get("chain_pair_pae_min")
    out = {}
    for i, a in enumerate(model_chain_order):
        for j, b in enumerate(model_chain_order):
            if i < j:
                v_ip = ip[i][j] if ip and i < len(ip) and j < len(ip[i]) else np.nan
                v_pae = (min(pae[i][j], pae[j][i]) if pae and max(i, j) < len(pae) else np.nan)
                out[frozenset((a, b))] = (v_ip, v_pae)
    return out


def _model_chain_order(af_path):
    order = []
    with open(af_path) as fh:
        cols, idx = [], None
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1])
            elif line.startswith(("ATOM", "HETATM")):
                if idx is None:
                    idx = cols.index("auth_asym_id")
                ch = line.split()[idx]
                if ch not in order:
                    order.append(ch)
    return order


def run():
    os.makedirs(OUTDIR, exist_ok=True)
    rows_out, skipped = [], []
    for period, (db, gt_dir, af_dir) in SETS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        exp = {r["PDBId"].strip().upper(): r for r in con.execute(
            "SELECT PDBId, ChainIDpairList_rnaRNA AS pairs, ContactList AS contacts FROM exp_rna_rna")}
        preds = con.execute("""SELECT exp_db_id, FileName, ChainIDpairList_rnaRNA AS pairs, ContactList AS contacts,
                                      RNA_LDDT_allAtom, RNA_TM, Complex_TM, Complex_LDDT_allAtom,
                                      af3_rna_ipTM, af3_ranking_score
                               FROM pred_rna_rna""").fetchall()
        con.close()
        for p in preds:
            pid = p["exp_db_id"].strip().upper()
            e = exp.get(pid)
            cif = os.path.join(gt_dir, f"{pid.lower()}.cif")
            af = os.path.join(af_dir, os.path.basename(str(p["FileName"])))
            if not os.path.exists(af):           # model files are lower case, some database entries upper case
                af = os.path.join(af_dir, os.path.basename(str(p["FileName"])).lower())
            if e is None or not (os.path.exists(cif) and os.path.exists(af)):
                skipped.append((period, pid, "missing ground truth or model"))
                continue
            mdl, mseqs = read_model(af)
            mapping = ComplexMapping(cif, None, pred_seqs=mseqs)        # no OST: sequence pairing
            exp_m = parse_rr_contacts(e["pairs"], e["contacts"])
            pred_m = parse_rr_contacts(p["pairs"], p["contacts"])
            exp_contacts = {}
            for (x, _), res in exp_m.items():
                prev = exp_contacts.get(x, ([], "author"))
                exp_contacts[x] = (prev[0] + res, "author")
            pred_contacts = {}
            for (x, _), res in pred_m.items():
                pred_contacts[x] = pred_contacts.get(x, set()) | {k for _, k in res}
            mapping.assign_fallback(exp_contacts, pred_contacts)
            nat = native_coords(cif)
            conf = af3_pair_confidence(af_dir, os.path.basename(af), _model_chain_order(af))
            done = set()
            for (X, Y) in exp_m:
                key = frozenset((X, Y))
                if key in done or X == Y:
                    continue
                done.add(key)
                x, y = mapping.pred_chain(X), mapping.pred_chain(Y)
                laX, laY = mapping.auth_to_label.get(X), mapping.auth_to_label.get(Y)
                if None in (x, y, laX, laY):
                    skipped.append((period, pid, f"{X}:{Y} chain not mapped"))
                    continue
                rec = dict(period=period, pdb=pid, native_pair=f"{X}:{Y}", pred_pair=f"{x}:{y}",
                           len_X=len(mapping.label_residues[laX]), len_Y=len(mapping.label_residues[laY]),
                           RNA_LDDT=_first(p["RNA_LDDT_allAtom"]), RNA_TM=_first(p["RNA_TM"]),
                           Complex_TM=_first(p["Complex_TM"]), Complex_LDDT=_first(p["Complex_LDDT_allAtom"]))
                inter = union = 0
                unmapped = False
                for side, (A, B, a, b) in (("X", (X, Y, x, y)), ("Y", (Y, X, y, x))):
                    ev, _ = mapping.map_motif(A, exp_m.get((A, B), []), prefer="author")
                    if ev is None:          # residue names disagree with the structure: not scored
                        rec[f"contact_match_{side}"] = np.nan
                        unmapped = True
                        continue
                    pv = {k for _, k in pred_m.get((a, b), [])}
                    rec[f"contact_match_{side}"] = jaccard(pv, ev)
                    inter += len(pv & ev); union += len(pv | ev)
                rec["contact_match"] = inter / union if union and not unmapped else np.nan
                rec["formed"] = bool(pred_m.get((x, y)) or pred_m.get((y, x)))
                # pose of the partner in the frame of the receptor (longer native chain)
                (R_, la_R, r_), (L_, la_L, l_) = (((X, laX, x), (Y, laY, y)) if rec["len_X"] >= rec["len_Y"]
                                                    else ((Y, laY, y), (X, laX, x)))
                Xr, Yr = _pair_atoms(mapping, la_R, R_, r_, range(len(mapping.label_residues[la_R])), nat, mdl, RNA_ATOMS)
                Xl, Yl = _pair_atoms(mapping, la_L, L_, l_, range(len(mapping.label_residues[la_L])), nat, mdl, RNA_ATOMS)
                rec.update(receptor=f"{R_}", receptor_fit_rmsd=np.nan, translation=np.nan, rotation=np.nan, lrmsd=np.nan)
                if len(Xr) >= 3 and len(Xl):
                    Rm, t, _, frms = fit_with_rejection(Xr, Yr)
                    pose = rna_pose((Rm @ Xl.T).T + t, Yl)
                    rec.update(receptor_fit_rmsd=frms, translation=pose["translation"], rotation=pose["rotation"],
                               lrmsd=pose["lrmsd"])
                ip, pae = conf.get(frozenset((x, y)), (np.nan, np.nan))
                rec.update(pair_iptm=ip, pair_pae_min=pae)
                rows_out.append(rec)

    keys = ["period", "pdb", "native_pair", "pred_pair", "len_X", "len_Y", "receptor", "formed",
            "contact_match", "contact_match_X", "contact_match_Y", "receptor_fit_rmsd", "translation",
            "rotation", "lrmsd", "pair_iptm", "pair_pae_min", "RNA_LDDT", "RNA_TM", "Complex_TM", "Complex_LDDT"]
    path = os.path.join(OUTDIR, "rna_rna_interfaces.tsv")
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else v)
    with open(path, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in rows_out:
            w.writerow([fmt(r[k]) for k in keys])
    print(f"wrote {path}: {len(rows_out)} RNA-RNA interfaces "
          f"(post {sum(r['period'] == 'post' for r in rows_out)}, pre {sum(r['period'] == 'pre' for r in rows_out)})")
    for s in skipped:
        print("  skipped:", s)
    return rows_out


if __name__ == "__main__":
    run()
