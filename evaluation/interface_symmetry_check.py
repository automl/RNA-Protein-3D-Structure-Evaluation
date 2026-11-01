"""
Duplex and strand-/copy-swap check of the protein-RNA interface placement classes.

A failed interface may be a mapping artefact rather than a modelling error: if the native RNA is a
strand of a duplex, or the protein / RNA has identical copies, AF3 can place the native arrangement
on the other strand or on another copy, and the interface then looks rotated, shifted or distal.
For every native protein-RNA interface (P, R) of the pre-/post-cutoff sets (same chain and residue
mapping as evaluation/interface_rna_pose.py) this records:

  duplex           the native RNA chain R is base paired with another RNA chain R2 (>= MIN_PAIRS
                   purine N1 - pyrimidine N3 distances <= PAIR_A, a Watson-Crick proxy);
  model_duplex     the predicted chains of R and R2 are paired in the model in the same way;
  occupancy        with the predicted protein p superposed on the native protein P (C-alpha, as in
                   the placement analysis), the share of native C4' atoms of R that have a predicted
                   C4' atom within OCC_A:
                     self          the predicted chain of R itself,
                     partner       the predicted chain of R2 (strand swap),
                     rna_copy      another predicted RNA chain with the sequence of R (RNA copy swap),
                     other_rna     any other predicted RNA chain,
                     protein_copy  the predicted chain of R with an identical copy p' of the protein
                                   superposed instead of p (protein copy swap);
  occ_duplex       for a duplex, the same share for both native strands (R and R2) occupied by the two
                   predicted strands: a duplex at the native site but turned or with its strands
                   exchanged has a high occ_duplex and a low occ_self;
  swap             the best alternative when it occupies the native site (>= OCC_MIN) and the
                   predicted chain of R does not.

Writes results/plots/all_chains/interface_symmetry_check/{interface_symmetry_check.tsv, _summary.tsv}.
Run in the RNABench env:  python evaluation/interface_symmetry_check.py  (--summary-only: re-merge the placement
classes into the existing table and rewrite the summary, without recomputing the occupancies)
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.exact_residue_mapping import ComplexMapping, interface_scores, interfaces_of, locate_inputs
from evaluation.interface_rna_pose import (INPUT_ROOTS, PERIODS, PROT_ATOM, _locate_model, _pair_atoms,
                                           drop_inconsistent_ost_chains, fit_with_rejection, native_coords,
                                           read_model)

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_symmetry_check")
POSE_TSV = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
            for d in ("plots", "plots_odl")]
PAIR_A, MIN_PAIRS = 3.5, 4          # base pair proxy and duplex threshold
OCC_A, OCC_MIN = 4.0, 0.5           # occupancy distance and "site occupied" threshold
PURINES, PYRIMIDINES = {"A", "G"}, {"C", "U"}
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}
CLASS_LABEL = {"correct interface": "Correct interface", "same place, duplex slid": "Correct site, slid duplex",
               "right protein site, wrong RNA pose": "Correct site, incorrect RNA",
               "same place, RNA turned": "Correct site, rotated RNA",
               "shifted nearby": "Shifted site", "different region": "Distal site",
               "protein binding region mispredicted": "Incorrect binding site"}


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def _base_atoms(cif_path, native):
    """{chain: [(resname, N1 or None, N3 or None)]} of the nucleotides, first model.
    native: chains are label_asym_id (as ComplexMapping); model: auth_asym_id (as read_model)."""
    m = MMCIF2Dict(cif_path)
    first = m["_atom_site.pdbx_PDB_model_num"][0]
    chain_col = "_atom_site.label_asym_id" if native else "_atom_site.auth_asym_id"
    res = {}
    for ch, sid, comp, atom, mdl, x, y, z in zip(
            m[chain_col], m["_atom_site.label_seq_id"], m["_atom_site.label_comp_id"],
            m["_atom_site.label_atom_id"], m["_atom_site.pdbx_PDB_model_num"],
            m["_atom_site.Cartn_x"], m["_atom_site.Cartn_y"], m["_atom_site.Cartn_z"]):
        if mdl != first or comp not in PURINES | PYRIMIDINES or atom not in ("N1", "N3"):
            continue
        res.setdefault((ch, sid), [comp, {}])[1][atom] = np.array((float(x), float(y), float(z)))
    out = {}
    for (ch, _sid), (comp, at) in res.items():
        out.setdefault(ch, []).append((comp, at.get("N1"), at.get("N3")))
    return out


def base_pairs(bases, a, b):
    """Purine N1 - pyrimidine N3 distances <= PAIR_A between chains a and b (Watson-Crick proxy)."""
    def split(ch):
        pu = np.array([n1 for c, n1, _n3 in bases.get(ch, []) if c in PURINES and n1 is not None])
        py = np.array([n3 for c, _n1, n3 in bases.get(ch, []) if c in PYRIMIDINES and n3 is not None])
        return pu, py
    (pu_a, py_a), (pu_b, py_b) = split(a), split(b)
    n = 0
    for pu, py in ((pu_a, py_b), (pu_b, py_a)):
        if len(pu) and len(py):
            d = np.linalg.norm(pu[:, None, :] - py[None, :, :], axis=2)
            n += int((d <= PAIR_A).sum())
    return n


def _c4(coords, chain):
    """C4' coordinates of one chain (keys of native_coords / read_model start with the chain id)."""
    xyz = [a["C4'"] for key, a in coords.items() if key[0] == chain and "C4'" in a]
    return np.array(xyz) if xyz else np.empty((0, 3))


def occupancy(native_xyz, model_xyz):
    """Share of native atoms with a model atom within OCC_A."""
    if not len(native_xyz) or not len(model_xyz):
        return np.nan
    d = np.linalg.norm(native_xyz[:, None, :] - model_xyz[None, :, :], axis=2).min(1)
    return float((d <= OCC_A).mean())


def _rna_chains(coords):
    """Chains whose residues are mostly nucleotides (C4' present)."""
    tot, nuc = {}, {}
    for key, a in coords.items():
        tot[key[0]] = tot.get(key[0], 0) + 1
        nuc[key[0]] = nuc.get(key[0], 0) + ("C4'" in a)
    return {c for c in tot if nuc[c] >= 0.5 * tot[c]}


# --------------------------------------------------------------------------- #
def run():
    rows_out = []
    for period, db in PERIODS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.ChainIDpairList_proteinRNA AS p_pairs,
                   p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
                   e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e
              ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for row in rows:
            cif, js = locate_inputs(row["id"], row["fn"], INPUT_ROOTS)
            af = _locate_model(row["fn"])
            if not (cif and af):
                continue
            try:
                mdl, mseqs = read_model(af)
                mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
                drop_inconsistent_ost_chains(mapping, mseqs)
                interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                                 dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                                 mapping)
                nat = native_coords(cif)
                nat_bases, mdl_bases = _base_atoms(cif, True), _base_atoms(af, False)
            except Exception as e:
                print(f"{period} {row['id']}: skipped ({e})")
                continue
            nat_rna, mdl_rna = _rna_chains(nat), _rna_chains(mdl)
            mdl_c4 = {c: _c4(mdl, c) for c in mdl_rna}
            pid = str(row["id"]).strip().upper()
            seen = set()
            for P, R, _aa, _rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
                laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
                p, r = mapping.pred_chain(P), mapping.pred_chain(R)
                if None in (laP, laR, p, r) or (p, r) in seen or laR not in nat_rna:
                    continue
                seen.add((p, r))
                idx = range(len(mapping.label_residues[laP]))
                Xp, Yp = _pair_atoms(mapping, laP, P, p, idx, nat, mdl, (PROT_ATOM,))
                if len(Xp) < 3:
                    continue
                y_r = _c4(nat, laR)
                # duplex partner of R in the native structure (label chains), and in the model
                partners = [(base_pairs(nat_bases, laR, c), c) for c in nat_rna if c != laR]
                n_pairs, laR2 = max(partners) if partners else (0, None)
                duplex = n_pairs >= MIN_PAIRS
                auR2 = next((au for au, la in mapping.auth_to_label.items() if la == laR2), None) if duplex else None
                r2 = mapping.pred_chain(auR2) if auR2 else None
                model_pairs = base_pairs(mdl_bases, r, r2) if r2 else 0

                def frame(model_protein):
                    X, Y = _pair_atoms(mapping, laP, P, model_protein, idx, nat, mdl, (PROT_ATOM,))
                    if len(X) < 3:
                        return None
                    Rm, t, _n, _rms = fit_with_rejection(X, Y)
                    return lambda A: (Rm @ A.T).T + t if len(A) else A

                f_p = frame(p)
                occ = {"self": occupancy(y_r, f_p(mdl_c4.get(r, np.empty((0, 3)))))}
                occ_duplex = np.nan
                if duplex and r2 in mdl_c4:
                    occ_duplex = occupancy(np.vstack([y_r, _c4(nat, laR2)]),
                                           f_p(np.vstack([mdl_c4.get(r, np.empty((0, 3))), mdl_c4[r2]])))
                alt = {}
                for c in mdl_rna - {r}:
                    v = occupancy(y_r, f_p(mdl_c4[c]))
                    kind = ("partner" if c == r2 else
                            "rna_copy" if mseqs.get(c) and mseqs.get(c) == mseqs.get(r) else "other_rna")
                    if not np.isnan(v) and v > alt.get(kind, (-1, None))[0]:
                        alt[kind] = (v, c)
                for c in mseqs:
                    if c != p and c not in mdl_rna and mseqs.get(c) == mseqs.get(p):
                        f_c = frame(c)
                        if f_c is None:
                            continue
                        v = occupancy(y_r, f_c(mdl_c4.get(r, np.empty((0, 3)))))
                        if not np.isnan(v) and v > alt.get("protein_copy", (-1, None))[0]:
                            alt["protein_copy"] = (v, c)
                best_kind, (best_v, best_c) = max(alt.items(), key=lambda kv: kv[1][0]) if alt else (None, (np.nan, None))
                swap = (best_kind if best_kind and best_v >= OCC_MIN and
                        not (occ["self"] >= OCC_MIN) else "")
                rows_out.append(dict(
                    period=period, pdb=pid, native_pair=f"{P}:{R}", model_pair=f"{p}:{r}",
                    duplex=duplex, native_pairs=n_pairs, partner_native=auR2 or "", partner_model=r2 or "",
                    model_duplex=bool(r2) and model_pairs >= MIN_PAIRS, model_pairs=model_pairs,
                    occ_self=occ["self"], occ_duplex=occ_duplex,
                    **{f"occ_{k}": alt.get(k, (np.nan, None))[0] for k in ("partner", "rna_copy", "other_rna", "protein_copy")},
                    best_alternative=best_kind or "", best_alternative_chain=best_c or "",
                    best_alternative_occ=best_v, swap=swap,
                    has_rna_copy=any(mseqs.get(c) == mseqs.get(r) for c in mdl_rna - {r}),
                    has_protein_copy=any(c != p and c not in mdl_rna and mseqs.get(c) == mseqs.get(p) for c in mseqs),
                ))
    return pd.DataFrame(rows_out)


def summarise(df):
    out = []
    for period in ("post", "pre"):
        for key, label in CLASS_LABEL.items():
            g = df[(df.period == period) & (df.placement == key)]
            if g.empty:
                continue
            row = dict(period=period, placement=label, n=len(g),
                       duplex_pct=100 * g.duplex.mean(),
                       duplex_formed_in_model_pct=100 * g[g.duplex].model_duplex.mean() if g.duplex.any() else np.nan,
                       has_rna_copy_pct=100 * g.has_rna_copy.mean(),
                       has_protein_copy_pct=100 * g.has_protein_copy.mean(),
                       occ_self_median=g.occ_self.median(),
                       duplex_in_place_pct=100 * ((g.occ_duplex >= OCC_MIN) & (g.occ_self < OCC_MIN)).sum()
                       / max(g.duplex.sum(), 1) if g.duplex.any() else np.nan,
                       swap_any_pct=100 * (g.swap != "").mean())
            for k in ("partner", "rna_copy", "protein_copy", "other_rna"):
                row[f"swap_{k}_pct"] = 100 * (g.swap == k).mean()
            out.append(row)
    return pd.DataFrame(out)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if "--summary-only" in sys.argv:       # re-merge the placement classes (e.g. after apply_duplex_pose.py) only
        df = pd.read_csv(os.path.join(OUTDIR, "interface_symmetry_check.tsv"), sep="\t").drop(columns=["placement", "dockq"],
                                                                                            errors="ignore")
        df["swap"] = df["swap"].fillna("")
    else:
        df = run()
    pose = next((p for p in POSE_TSV if os.path.isfile(p)), None)
    if pose:
        pl = pd.read_csv(pose, sep="\t")[["period", "pdb", "native_pair", "model_pair", "placement", "dockq"]]
        pl = pl[pl.dockq.notna()]
        pl["placement"] = pl["placement"].replace(MERGE)
        df = df.merge(pl, on=["period", "pdb", "native_pair", "model_pair"], how="left")
    df.to_csv(os.path.join(OUTDIR, "interface_symmetry_check.tsv"), sep="\t", index=False)
    if "placement" in df:
        s = summarise(df[df.placement.notna()])
        s.to_csv(os.path.join(OUTDIR, "interface_symmetry_check_summary.tsv"), sep="\t", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(s.round(1).to_string(index=False))
    print(f"\n{len(df)} interfaces -> {OUTDIR}")


if __name__ == "__main__":
    main()
