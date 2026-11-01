"""
Translation / rotation split of the RNA pose at each protein-RNA interface.

DockQ's ligand RMSD (LRMSD) superposes the protein and takes the RMSD of the RNA: a short RNA
turned round in the correct pocket and an RNA sitting in a different region both give a large
LRMSD. Here the RNA error is split into where the RNA is and how it is turned.

For every native protein-RNA interface (protein chain P, RNA chain R; exact residue mapping of
evaluation/exact_residue_mapping.py, i.e. OpenStructure chain mapping + alignment, with the
sequence fallback for U-repeat chains):

  0. chain assignments come from OST, but are checked against the AF3 model file used here; where
     OST's model differs from that file (a few post-cutoff entries, e.g. 8DFO) the affected
     chains are re-mapped by sequence to the file and the OST LRMSD is not used;
  1. the AF3 protein chain is superposed on the native protein chain (mapped CA atoms; iterative
     outlier rejection as PyMOL `align`: 5 cycles, atoms deviating > 2 x RMSD dropped). The
     rejection hides mispredicted parts, so in the same frame are also reported
       protein_rmsd_all  all mapped CA atoms
       pocket_rmsd       CA atoms of the native protein contact residues of this interface
                         (AAMotif) -- is the binding region of the protein itself right?
  2. in that frame the mapped atoms of the whole RNA chain (C4', else P) are compared with the
     native RNA:
       rna_translation  centroid displacement (A)  -- is the RNA in the same place?
       rna_rotation     angle of the optimal superposition onto the native RNA (deg)
                        -- is it turned?
  3. lrmsd: DockQ's ligand RMSD of the interface, taken from the same OpenStructure run
     (full-complex compare-structures JSON, `dockq_interfaces` / `lrmsd`). Where OST has none
     (U-repeat RNAs on which it fails, chains it leaves out) our own LRMSD is used: RMSD of the
     whole mapped RNA chain in the protein frame of step 1, without further fitting
     (lrmsd_source = "ost" / "own"). Where both exist they agree (Spearman 0.95, median
     difference 0.25 A), which also validates the protein superposition used for the split.
     DockQ likewise comes from that OpenStructure run (RBP_DockQ); where it has none, the DockQ of
     interface_own_dockq.py on the exact residue mapping is used (dockq_source = "ost" / "own").
  4. placement:
       correct interface              DockQ >= 0.23
     Among incorrect interfaces, where AF3 still contacts native protein residues (exact-mapped
     per-interface protein match, AAmatch_per_interface, >= 0.1) the RNA is at the right protein
     site; a large whole-RNA translation there reflects the rest of the RNA, not a different
     binding region:
       right protein site, wrong RNA pose      protein match >= 0.1
     Otherwise (protein match < 0.1, incl. interfaces AF3 does not form) the pose is only read
     where the protein's binding region is itself predicted correctly:
       protein binding region mispredicted     pocket_rmsd > POCKET_A (95th percentile of correct
                                               interfaces: 3.8 A post-, 4.4 A pre-cutoff)
     and else the whole-RNA pose decides (rotation is only needed where the RNA is in place):
       same place, same orientation   translation <= 10 A and rotation <= 45 deg
       same place, RNA turned         translation <= 10 A and rotation >  45 deg
       shifted nearby                 10 A < translation <= 20 A
       different region               translation > 20 A
     For a duplex at the native site with the wrong strand in the native position, the pose is re-read on the
     whole duplex by evaluation/apply_duplex_pose.py, which adds the class
       same place, duplex slid        duplex moved <= 10 A and its axis tilted <= 45 deg
     10 A is about the 95th percentile of the translation in correct interfaces (DockQ >= 0.23;
     10.8 A post-, 10.9 A pre-cutoff).

Writes results/plots/all_chains/interface_rna_pose/interface_rna_pose.tsv (all native interfaces)
and the incorrect ones (DockQ < 0.23) with their per-interface protein / RNA match to
results/plots/all_chains/interface_motif_match/incorrect_interfaces_protein_match_crmsd.tsv, and the
placement class of every interface to pred_protein_rna.RNA_interface_placement (a2021.db, b2021.db and
database/rbpDatabase.db). `--relabel` re-applies the rule to the existing TSV; `--db` also writes
the DB column. Run in the RNABench env:  python evaluation/interface_rna_pose.py
"""

import csv
import os
import sqlite3
import sys
from collections import Counter

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.exact_residue_mapping import (ComplexMapping, MIN_NAME_AGREEMENT, _list, interface_scores,
                                              interfaces_of, locate_inputs, one_letter)

from evaluation.bundle_paths import ALL_DATA, PROTEIN_RNA_DB, PROTEIN_RNA_DIR

PERIODS = dict(PROTEIN_RNA_DB)
# the published structures only (Zenodo bundle, data/all_data/<set>/protein_rna: pdb/, af/, lddt/); older local copies
# of some post-cutoff models elsewhere in data/ (January 2025) share the file names and are never searched
INPUT_ROOTS = [PROTEIN_RNA_DIR["post"], PROTEIN_RNA_DIR["pre"]]
MODEL_ROOTS = [os.path.join(r, "af") for r in INPUT_ROOTS]
DATA = ALL_DATA
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_rna_pose")
INCORRECT_TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_motif_match",
                             "incorrect_interfaces_protein_match_crmsd.tsv")
# DockQ where OpenStructure has none (poly-U RNAs, failed comparisons), on the exact residue mapping
OWN_DOCKQ_TSV = os.path.join(OUTDIR, "own_dockq.tsv")
POSE_TSVS = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
             for d in ("plots", "plots_odl")]
PROT_ATOM = "CA"
RNA_ATOMS = ("C4'", "P")
CYCLES, CUTOFF = 5, 2.0          # PyMOL align defaults
SAME_PLACE_A, NEARBY_A, TURNED_DEG = 10.0, 20.0, 45.0
PROTEIN_SITE_MATCH = 0.1
POCKET_A = 4.0


PLACEMENTS = ["correct interface", "same place, duplex slid", "right protein site, wrong RNA pose",
              "protein binding region mispredicted", "same place, same orientation", "same place, RNA turned",
              "shifted nearby", "different region"]


def placement(dockq, translation, rotation, protein_match, pocket_rmsd):
    """Placement class of the RNA relative to its protein partner (see module docstring, 4.)."""
    t, a = translation, rotation
    if dockq == dockq and dockq >= 0.23:
        return "correct interface"
    if protein_match == protein_match and protein_match >= PROTEIN_SITE_MATCH:
        return "right protein site, wrong RNA pose"
    if pocket_rmsd == pocket_rmsd and pocket_rmsd > POCKET_A:
        return "protein binding region mispredicted"
    if t is None or t != t:
        return ""
    if t <= SAME_PLACE_A:
        if a != a:
            return "same place"
        return "same place, RNA turned" if a > TURNED_DEG else "same place, same orientation"
    return "shifted nearby" if t <= NEARBY_A else "different region"


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def _atoms(m, chain_key):
    """{(chain, residue key): {atom name: xyz}} for the first model, polymer residues."""
    first = m["_atom_site.pdbx_PDB_model_num"][0]
    ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
    out = {}
    for la, au, sid, ic, lsid, atom, mdl, x, y, z in zip(
            m["_atom_site.label_asym_id"], m["_atom_site.auth_asym_id"], m["_atom_site.auth_seq_id"], ins,
            m["_atom_site.label_seq_id"], m["_atom_site.label_atom_id"], m["_atom_site.pdbx_PDB_model_num"],
            m["_atom_site.Cartn_x"], m["_atom_site.Cartn_y"], m["_atom_site.Cartn_z"]):
        if mdl != first or lsid in (".", "?"):
            continue
        atom = atom.strip('"')
        if atom != PROT_ATOM and atom not in RNA_ATOMS:
            continue
        key = chain_key(la, au, sid, "" if ic in ("?", ".") else ic, lsid)
        out.setdefault(key, {}).setdefault(atom, np.array((float(x), float(y), float(z))))
    return out


def native_coords(cif_path):
    """(label chain, auth number, insertion code) -> {atom: xyz}; matches ComplexMapping.label_residues."""
    return _atoms(MMCIF2Dict(cif_path), lambda la, au, sid, ic, lsid: (la, sid, ic))


def read_model(af_path):
    """AF3 model -> ((chain, sequence position) -> {atom: xyz}, {chain: one-letter sequence})."""
    m = MMCIF2Dict(af_path)
    seqs = {}
    for au, lsid, comp in zip(m["_atom_site.auth_asym_id"], m["_atom_site.label_seq_id"],
                              m["_atom_site.label_comp_id"]):
        if lsid not in (".", "?"):
            seqs.setdefault(au, {})[int(lsid)] = one_letter(comp)
    seqs = {c: "".join(v[k] for k in sorted(v)) for c, v in seqs.items()}
    return _atoms(m, lambda la, au, sid, ic, lsid: (au, int(lsid))), seqs


def drop_inconsistent_ost_chains(mapping, model_seqs):
    """Remove OST chain assignments that do not fit the AF3 model file actually used.

    OST was run on model files that, for a few post-cutoff entries, differ from the AF3 files in
    data/ (e.g. 8DFO: OST's model has 13 chains, the file 12). An OST assignment is kept only if
    the model chain exists in the file and OST's aligned model sequence matches the file's chain
    (same length, >= 90% identical); the dropped chains are re-mapped by the sequence fallback.
    """
    bad = []
    for la, mc in list(mapping.chain_mapping.items()):
        ok = mc in model_seqs
        if ok and la in mapping._aln:
            ost_mdl = mapping._aln[la][1].replace("-", "")
            fs = model_seqs[mc]
            ok = len(ost_mdl) == len(fs) and sum(a == b for a, b in zip(ost_mdl, fs)) >= 0.9 * len(fs)
        if not ok:
            bad.append(la)
    for la in bad:
        del mapping.chain_mapping[la]
        mapping.source.pop(la, None)
        mapping._index.pop(la, None)
    return bad


def _locate_model(file_name):
    key = os.path.basename(str(file_name)).lower()
    for path in _file_index_models():
        if os.path.basename(path).lower() == key:
            return path
    return None


_MODELS = None


def _file_index_models():
    global _MODELS
    if _MODELS is None:
        import glob
        _MODELS = [p for r in MODEL_ROOTS for p in sorted(glob.glob(os.path.join(r, "*.cif")))]
    return _MODELS


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def kabsch(X, Y):
    """R, t minimising |R x + t - y|."""
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cy - R @ cx


def fit_with_rejection(X, Y):
    """PyMOL-style iterative fit: returns R, t, n atoms kept, final RMSD."""
    keep = np.ones(len(X), bool)
    for _ in range(CYCLES + 1):
        R, t = kabsch(X[keep], Y[keep])
        d = np.linalg.norm((R @ X.T).T + t - Y, axis=1)
        rms = np.sqrt((d[keep] ** 2).mean())
        new = keep & (d <= CUTOFF * rms)
        if new.sum() == keep.sum() or new.sum() < 3:
            break
        keep = new
    return R, t, int(keep.sum()), float(rms)


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rna_pose(Xm, Yn):
    """Translation, rotation and own LRMSD of model RNA atoms Xm (already in the native frame)."""
    out = dict(translation=float(np.linalg.norm(Xm.mean(0) - Yn.mean(0))),
               lrmsd=float(np.sqrt(((Xm - Yn) ** 2).sum(1).mean())), rotation=np.nan)
    if len(Xm) >= 3:
        out["rotation"] = angle(kabsch(Xm, Yn)[0])
    return out


def _pair_atoms(mapping, la, auth_chain, model_chain, idx, nat, mdl, atoms):
    """Matched (model, native) coordinates for native polymer indices `idx` of a label chain."""
    fwd = mapping._model_index(la)
    res = mapping.label_residues[la]
    X, Y = [], []
    for k in idx:
        if k not in fwd:
            continue
        a_n = nat.get((la, res[k][0], res[k][1]), {})
        a_m = mdl.get((model_chain, fwd[k] + 1), {})
        for a in atoms:
            if a in a_n and a in a_m:
                X.append(a_m[a]); Y.append(a_n[a])
                break
    return np.array(X), np.array(Y)


def ost_lrmsd(json_path):
    """{frozenset(model chain pair): LRMSD} from the full-complex OST JSON."""
    if not json_path:
        return {}
    import json
    d = json.load(open(json_path))
    return {frozenset(i[2:4]): v for i, v in zip(d.get("dockq_interfaces") or [], d.get("lrmsd") or [])}


def own_dockq():
    """(period, PDB, native pair) -> DockQ of evaluation/interface_own_dockq.py ({} before that step has run)."""
    if not os.path.isfile(OWN_DOCKQ_TSV):
        return {}
    return {(r["period"], r["pdb"].upper(), r["native_pair"]): float(r["dockq"])
            for r in csv.DictReader(open(OWN_DOCKQ_TSV), delimiter="\t") if r["dockq"] not in ("", "nan")}


def _fill_dockq(rec, own):
    """OpenStructure's DockQ where it exists, else the exact-mapping DockQ (dockq_source 'ost' / 'own' / '').
    An interface listed in own_dockq.tsv has no OpenStructure DockQ: its RBP_DockQ entry is the fallback written by
    interface_own_dockq.write_db, so the current fallback value is used and the interface stays 'own'."""
    v = own.get((rec["period"], str(rec["pdb"]).upper(), rec["native_pair"]))
    if v is not None:
        rec["dockq"], rec["dockq_source"] = v, "own"
    elif rec["dockq"] == rec["dockq"]:
        rec["dockq_source"] = "ost"
    else:
        rec["dockq"], rec["dockq_source"] = np.nan, ""


def _value_of(raw, pair):
    p, r = pair.split(":")
    for item in _list(raw):
        if isinstance(item, dict):
            for k in (f"{p}:{r}", f"{r}:{p}"):
                if k in item and item[k] is not None:
                    return float(item[k])
    return np.nan


def _match_pair(row, db_pair):
    """Pair naming of AAmatch_per_interface / RNAmatch_per_interface (the pred table's ChainIDpairList_proteinRNA).
    Where OpenStructure renamed the RNA chain, the RBP_DockQ naming of db_pair has an RNA name that is no RNA chain
    of the pred table (e.g. 8SNY A:D for A:F): the pair is then the protein's only pair in the pred table."""
    p, r = db_pair.split(":")
    if str(r) in {str(x) for x in _list(row["p_rch"])}:
        return db_pair
    pairs = [(str(a), str(b)) for a, b in (_list(row["p_pairs"]) or []) if str(a) == str(p)]
    return f"{pairs[0][0]}:{pairs[0][1]}" if len(pairs) == 1 else db_pair


# --------------------------------------------------------------------------- #
def run():
    os.makedirs(OUTDIR, exist_ok=True)
    own = own_dockq()
    rows_out, st = [], Counter()
    for period, db in PERIODS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.RBP_DockQ AS dockq,
                   p.AAmatch_per_interface AS aa_iface, p.RNAmatch_per_interface AS rna_iface,
                   p.AAMotif AS p_aa, p.RNAMotif AS p_rna, p.ChainIDpairList_proteinRNA AS p_pairs,
                   p.ProteinChainIDs AS p_pch, p.ProteinSequence AS p_pseq,
                   p.RNAChainIDs AS p_rch, p.RNASequence AS p_rseq,
                   e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e
              ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for row in rows:
            cif, js = locate_inputs(row["id"], row["fn"], INPUT_ROOTS)
            af = _locate_model(row["fn"])
            if not (cif and af):
                st[f"{period}: no structure"] += 1
                continue
            mdl, mseqs = read_model(af)
            mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
            db_map = dict(mapping.chain_mapping)        # OST naming = chain naming of the DB rows
            dropped = drop_inconsistent_ost_chains(mapping, mseqs)
            if dropped:
                st[f"{period}: complexes with OST chains re-mapped to the model file"] += 1
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                             mapping)                                   # assigns fallback chains
            nat = native_coords(cif)
            lr = {} if dropped else ost_lrmsd(js)       # OST's LRMSD refers to its own model file
            seen = set()
            for P, R, aa, rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
                laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
                p, r = mapping.pred_chain(P), mapping.pred_chain(R)
                if None in (laP, laR, p, r) or (p, r) in seen:
                    st[f"{period}: interface unmapped"] += (p is None or r is None)
                    continue
                seen.add((p, r))
                # DB naming of this pair (RBP_DockQ, AAMotif, the other result tables)
                db_pair = f"{db_map.get(laP, p)}:{db_map.get(laR, r)}"
                Xp, Yp = _pair_atoms(mapping, laP, P, p, range(len(mapping.label_residues[laP])), nat, mdl, (PROT_ATOM,))
                if len(Xp) < 3:
                    st[f"{period}: <3 protein atoms"] += 1
                    continue
                Rm, t, nkeep, prms = fit_with_rejection(Xp, Yp)
                in_frame = lambda X: (Rm @ X.T).T + t
                rmsd = lambda X, Y: float(np.sqrt(((in_frame(X) - Y) ** 2).sum(1).mean())) if len(X) else np.nan
                Ep, _, fp = mapping._motif_indices(laP, aa, "author")
                Xk, Yk = _pair_atoms(mapping, laP, P, p, Ep if fp >= MIN_NAME_AGREEMENT else [], nat, mdl, (PROT_ATOM,))
                rec = dict(period=period, pdb=str(row["id"]).strip().upper(), pred_pair=db_pair,
                           model_pair=f"{p}:{r}", ost_chains_remapped=",".join(dropped),
                           native_pair=f"{P}:{R}", dockq=_value_of(row["dockq"], db_pair),
                           lrmsd=float(lr.get(frozenset((p, r)), np.nan)), lrmsd_source="ost",
                           protein_fit_rmsd=prms, protein_fit_atoms=f"{nkeep}/{len(Xp)}",
                           protein_rmsd_all=rmsd(Xp, Yp), pocket_rmsd=rmsd(Xk, Yk))
                Xr, Yr = _pair_atoms(mapping, laR, R, r, range(len(mapping.label_residues[laR])), nat, mdl, RNA_ATOMS)
                pose = (rna_pose(in_frame(Xr), Yr) if len(Xr)
                        else dict(translation=np.nan, rotation=np.nan, lrmsd=np.nan))
                rec["rna_n_nt"] = len(Xr)
                rec["rna_translation"], rec["rna_rotation"] = pose["translation"], pose["rotation"]
                if not np.isfinite(rec["lrmsd"]) and np.isfinite(pose["lrmsd"]):
                    rec["lrmsd"], rec["lrmsd_source"] = pose["lrmsd"], "own"
                if not np.isfinite(rec["lrmsd"]):
                    rec["lrmsd_source"] = ""
                mp = _match_pair(row, db_pair)
                pm, rm_ = _value_of(row["aa_iface"], mp), _value_of(row["rna_iface"], mp)
                rec["protein_match"] = 0.0 if pm != pm else pm   # pair AF3 did not form: no native contact
                rec["rna_match"] = 0.0 if rm_ != rm_ else rm_
                _fill_dockq(rec, own)
                rec["placement"] = placement(rec["dockq"], rec["rna_translation"], rec["rna_rotation"],
                                             rec["protein_match"], rec["pocket_rmsd"])
                rows_out.append(rec)

    write_tables(rows_out)
    print(f"{len(rows_out)} interfaces  {dict(st)}")
    return rows_out


POSE_KEYS = ["period", "pdb", "pred_pair", "model_pair", "native_pair", "ost_chains_remapped", "dockq",
             "dockq_source", "protein_match", "rna_match", "lrmsd", "lrmsd_source", "protein_fit_rmsd",
             "protein_fit_atoms", "protein_rmsd_all", "pocket_rmsd", "rna_n_nt", "rna_translation", "rna_rotation",
             "placement"]


def _match_bin(v):
    return "P=0" if v == 0 else ("0<P<=0.1" if v <= 0.1 else ("0.1<P<0.5" if v < 0.5 else "P>=0.5"))


def write_tables(rows):
    """All interfaces -> interface_rna_pose.tsv; incorrect ones (DockQ < 0.23) -> INCORRECT_TSV."""
    os.makedirs(OUTDIR, exist_ok=True)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else v
    with open(os.path.join(OUTDIR, "interface_rna_pose.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(POSE_KEYS)
        for r in rows:
            w.writerow([fmt(r[k]) for k in POSE_KEYS])
    inc = [r for r in rows if r["dockq"] == r["dockq"] and r["dockq"] < 0.23]
    order = {"P=0": 0, "0<P<=0.1": 1, "0.1<P<0.5": 2, "P>=0.5": 3}
    inc.sort(key=lambda r: (r["period"] != "post", order[_match_bin(r["protein_match"])], r["pdb"], r["pred_pair"]))
    keys = POSE_KEYS[:8] + ["protein_match_bin"] + POSE_KEYS[8:]
    os.makedirs(os.path.dirname(INCORRECT_TSV), exist_ok=True)
    with open(INCORRECT_TSV, "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(keys)
        for r in inc:
            w.writerow([_match_bin(r["protein_match"]) if k == "protein_match_bin" else fmt(r[k]) for k in keys])
    print(f"wrote {OUTDIR}/interface_rna_pose.tsv ({len(rows)}) and {INCORRECT_TSV} ({len(inc)} incorrect)")


def relabel():
    """Re-apply `placement` to an existing interface_rna_pose.tsv (with the fallback DockQ of own_dockq.tsv where
    OpenStructure has none) and rewrite both tables."""
    pose = next((p for p in POSE_TSVS if os.path.isfile(p)), None)
    rows = list(csv.DictReader(open(pose), delimiter="\t"))
    own = own_dockq()
    num = {"dockq", "protein_match", "rna_match", "lrmsd", "protein_fit_rmsd", "protein_rmsd_all",
           "pocket_rmsd", "rna_translation", "rna_rotation"}
    for r in rows:
        for k in num:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
        if r.get("dockq_source") == "own":     # re-read the fallback, it may have been recomputed
            r["dockq"], r["dockq_source"] = np.nan, ""
        _fill_dockq(r, own)
        r["placement"] = placement(r["dockq"], r["rna_translation"], r["rna_rotation"],
                                   r["protein_match"], r["pocket_rmsd"])
    write_tables(rows)
    return rows


def db_label(placement_class):
    """'right protein site, wrong RNA pose' -> 'right_protein_site_wrong_rna_pose' (RNA_interface_class style)."""
    return placement_class.replace(",", "").replace(" ", "_").lower()


def write_db(rows, db_paths=None):
    """RNA_interface_placement in pred_protein_rna: the placement class of every protein-RNA interface.

    Pair notation of RBP_DockQ with the DB's chain ids (pred_pair) and snake_case labels, e.g.
    [{"A:C": "correct_interface"}, {"B:C": "shifted_nearby"}]; RNA_interface_class is left untouched. One class per
    RNA chain is derived by the analyses (evaluation/placement_labels.py: best class over its interfaces;
    publication_figures/placement_per_rna_chain.py: class with the main protein partner).
    """
    import json
    targets = db_paths or {**PERIODS, "post_working_copy": os.path.join(_REPO_ROOT, "database", "rbpDatabase.db")}
    for name, db in targets.items():
        period = next((p for p in PERIODS if name.startswith(p)), name)
        per_pdb = {}
        for r in rows:
            if r["period"] == period and r["placement"] in PLACEMENTS:
                per_pdb.setdefault(r["pdb"], {})[r["pred_pair"]] = r["placement"]
        con = sqlite3.connect(db)
        cols = {c[1] for c in con.execute("PRAGMA table_info(pred_protein_rna)")}
        if "RNA_interface_placement" not in cols:
            con.execute("ALTER TABLE pred_protein_rna ADD COLUMN RNA_interface_placement TEXT")
        con.execute("UPDATE pred_protein_rna SET RNA_interface_placement = NULL")
        n = 0
        for (eid,) in con.execute("SELECT exp_db_id FROM pred_protein_rna").fetchall():
            d = per_pdb.get(str(eid).strip().upper())
            if d:
                con.execute("UPDATE pred_protein_rna SET RNA_interface_placement = ? WHERE exp_db_id = ?",
                            (json.dumps([{k: db_label(v)} for k, v in sorted(d.items())]), eid))
                n += 1
        con.commit()
        con.close()
        print(f"{db}: RNA_interface_placement written for {n} complexes")


if __name__ == "__main__":
    arg = sys.argv[1:2]
    if arg == ["--relabel"]:
        relabel()
    elif arg == ["--db"]:
        write_db(relabel())
    else:
        write_db(run())
