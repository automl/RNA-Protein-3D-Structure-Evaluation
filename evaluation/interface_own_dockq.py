"""
DockQ for the protein-RNA interfaces that OpenStructure's full-complex comparison leaves without one.

OpenStructure's compare-structures run (the source of RBP_DockQ) returns no protein-RNA DockQ for some complexes:
either the comparison fails (status FAILURE) or its chain mapping cannot place the RNA chains, which happens mostly
for short poly-U RNAs, and only the protein-protein interfaces are scored. interface_rna_pose.py then has no DockQ for
these interfaces and the downstream analyses drop them (post-cutoff 174 interfaces in 56 complexes, 44 of them without
any; pre-cutoff 567 interfaces in 156 complexes).

Here the interface is scored with OpenStructure's own DockQ implementation (ost.mol.alg.dockq.DockQ), but with the
residue correspondence fixed by the exact residue mapping (exact_residue_mapping.ComplexMapping, as in
interface_rna_pose.py) instead of OpenStructure's chain mapping. Both chains of the interface are cut to the mapped
residues and renumbered 1..n in model and native, so DockQ pairs residues by number. OpenStructure names mmCIF chains
by label_asym_id and numbers residues by label_seq_id, so the native side is exported in label numbering.

Two steps, two environments:
  python evaluation/interface_own_dockq.py export [--validate]   (RNABench env)  residue correspondences -> JSON
  python evaluation/interface_own_dockq.py score  [--validate]   (ost env)       DockQ -> own_dockq.tsv
`--validate` scores a quarter of the interfaces that do have an OpenStructure DockQ (own_dockq_validation.tsv) and
`python evaluation/interface_own_dockq.py compare` reports the agreement (both periods: 1,099 interfaces, Spearman
0.995, median difference 0.000, 99.7% on the same side of DockQ 0.23).
Result: 737 of 741 interfaces scored; the other 4 (pre-cutoff) have no native contact among the mapped residues.
interface_rna_pose.py reads own_dockq.tsv as fallback (dockq_source = "ost" / "own").
"""
import csv
import json
import os
import sys
import zlib

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_rna_pose")
POSE_TSV = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
            for d in ("plots", "plots_odl")]
OWN_TSV = os.path.join(OUTDIR, "own_dockq.tsv")
VALIDATION_TSV = os.path.join(OUTDIR, "own_dockq_validation.tsv")
FIELDS = ["period", "pdb", "file", "native_pair", "model_pair", "dockq", "fnat", "irmsd", "lrmsd", "nnat", "ilddt", "ics",
          "qs", "n_protein_res", "n_rna_res", "error"]
COMPLEX_TSV = os.path.join(OUTDIR, "own_complex.tsv")
COMPLEX_VALIDATION_TSV = os.path.join(OUTDIR, "own_complex_validation.tsv")
COMPLEX_FIELDS = ["period", "pdb", "file", "n_native_chains", "n_mapped_chains", "lddt", "ilddt", "ics", "qs_global",
                  "dockq_wave", "error"]


def _scorer(mod, ref, mapping):
    """OpenStructure's Scorer on cut, renumbered entities with the chain mapping fixed (no automatic chain mapping)."""
    from ost.mol.alg.scoring import Scorer
    return Scorer(mod, ref, resnum_alignments=True, min_nuc_length=3, custom_mapping=mapping)


def _scores(sc, names):
    """({name: rounded value} of the Scorer metrics OpenStructure computes, error text of the others). The LDDT-based
    metrics fail on their own when OpenStructure discards a chain for them (e.g. an RNA of four nucleotides), while
    DockQ, QS and ICS of the same complex are still defined."""
    out, errors = {}, []
    for n in names:
        try:
            out[n] = _r(getattr(sc, n))
        except Exception as e:
            errors.append(f"{n}: {type(e).__name__}: {e}")
    return out, "; ".join(errors)[:200]


def _r(v, nd=4):
    return "" if v is None else round(float(v), nd)


REST_TSV = os.path.join(OUTDIR, "own_dockq_rest.tsv")
# model chains OpenStructure's full-complex comparison could not map (the former OST_fail column: "1" = the whole
# comparison failed, else the unmapped model chain ids), kept here so the fallback knows which values are placeholders
UNMAPPED_TSV = os.path.join(OUTDIR, "ost_unmapped_chains.tsv")
if not os.path.isfile(UNMAPPED_TSV):          # the published list (analysis_inputs.zip, fetch_data.sh)
    UNMAPPED_TSV = os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "interface_rna_pose",
                                "ost_unmapped_chains.tsv")


def ost_unmapped():
    """{(period, PDB): set of unmapped model chains ('*' = all)}."""
    out = {}
    if os.path.isfile(UNMAPPED_TSV):
        for r in csv.DictReader(open(UNMAPPED_TSV), delimiter="\t"):
            v = r["ost_unmapped"].strip()
            if v == "1":
                ids = {"*"}
            else:
                try:
                    x = json.loads(v)
                    ids = set(map(str, x)) if isinstance(x, list) else {v}
                except ValueError:
                    ids = {v}
            out[(r["period"], r["pdb"].upper())] = ids
    return out


def _mode(validate):
    """'missing' (OpenStructure has no DockQ), 'validate' (a quarter of the scored ones) or 'rest' (all others)."""
    return validate if isinstance(validate, str) else ("validate" if validate else "missing")


def _maps_path(validate):
    return os.path.join(OUTDIR, {"missing": "own_dockq_maps.json", "validate": "own_dockq_maps_validation.json",
                                 "rest": "own_dockq_maps_rest.json"}[_mode(validate)])


def _out_path(validate):
    return {"missing": OWN_TSV, "validate": VALIDATION_TSV, "rest": REST_TSV}[_mode(validate)]


def all_interface_metrics():
    """{(period, PDB, native pair): row} over the three runs (missing, validation, rest), i.e. every native interface."""
    out = {}
    for p in (REST_TSV, VALIDATION_TSV, OWN_TSV):
        if os.path.isfile(p):
            for r in csv.DictReader(open(p), delimiter="\t"):
                out[(r["period"], r["pdb"].upper(), r["native_pair"])] = r
    return out


def _missing(v):
    return v in ("", "nan", "NaN", None)


def _pose_rows():
    pose = next((p for p in POSE_TSV if os.path.isfile(p)), None)
    if pose is None:
        raise SystemExit("interface_rna_pose.tsv not found; run evaluation/interface_rna_pose.py first")
    return list(csv.DictReader(open(pose), delimiter="\t"))


# --------------------------------------------------------------------------- #
# Step 1 (RNABench): residue correspondences from the exact residue mapping
# --------------------------------------------------------------------------- #
def export(validate=False):
    import sqlite3
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from evaluation.interface_rna_pose import (PERIODS, INPUT_ROOTS, ComplexMapping, locate_inputs, _locate_model, read_model,
                                               drop_inconsistent_ost_chains, interface_scores, interfaces_of)
    mode = _mode(validate)
    done = set()
    if mode == "rest":                        # every native interface the other two runs did not score
        for p in (OWN_TSV, VALIDATION_TSV):
            done |= {(r["period"], r["pdb"].upper(), r["native_pair"]) for r in csv.DictReader(open(p), delimiter="\t")}
    need = {}
    for r in _pose_rows():
        ost_missing = r.get("dockq_source") == "own" or _missing(r["dockq"])
        # validation: a fixed quarter of the OST-scored interfaces
        if mode == "rest":
            pick = (r["period"], r["pdb"].upper(), r["native_pair"]) not in done
        else:
            pick = (not ost_missing and zlib.crc32((r["pdb"] + r["native_pair"]).encode()) % 4 == 0) if validate else ost_missing
        if pick:
            need.setdefault((r["period"], r["pdb"].upper()), set()).add(r["native_pair"])
    out = []
    for period, db in PERIODS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
                   p.ChainIDpairList_proteinRNA AS p_pairs, e.AAMotif AS e_aa, e.RNAMotif AS e_rna,
                   e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for row in rows:
            pid = str(row["id"]).strip().upper()
            if (period, pid) not in need:
                continue
            cif, js = locate_inputs(row["id"], row["fn"], INPUT_ROOTS)
            af = _locate_model(row["fn"])
            mdl, mseqs = read_model(af)
            mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
            drop_inconsistent_ost_chains(mapping, mseqs)
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                             mapping)
            m = MMCIF2Dict(cif)
            ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
            lseq = {(la, au, "" if ic in ("?", ".") else ic): ls for la, au, ic, ls in
                    zip(m["_atom_site.label_asym_id"], m["_atom_site.auth_seq_id"], ins, m["_atom_site.label_seq_id"])}
            seen = set()
            for P, R, _aa, _rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
                if f"{P}:{R}" not in need[(period, pid)]:
                    continue
                laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
                p, r = mapping.pred_chain(P), mapping.pred_chain(R)
                if None in (laP, laR, p, r) or (p, r) in seen:
                    continue
                seen.add((p, r))
                rec = dict(period=period, pdb=pid, native_pair=f"{P}:{R}", model_pair=f"{p}:{r}", cif=cif, model=af)
                for side, la, mc in (("protein", laP, p), ("rna", laR, r)):
                    fwd, res = mapping._model_index(la), mapping.label_residues[la]
                    pairs = []
                    for k in range(len(res)):
                        ls = lseq.get((la, str(res[k][0]), str(res[k][1] or "").strip()))
                        if k in fwd and ls not in (None, ".", "?"):
                            pairs.append([int(ls), int(fwd[k]) + 1])
                    rec[side] = dict(native_chain=la, model_chain=mc, pairs=pairs)
                out.append(rec)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(_maps_path(validate), "w"))
    print(f"{len(out)} interfaces from {len({(o['period'], o['pdb']) for o in out})} complexes "
          f"({sum(len(v) for v in need.values())} requested) -> {_maps_path(validate)}")


# --------------------------------------------------------------------------- #
# Step 2 (ost env): DockQ on the fixed correspondence
# --------------------------------------------------------------------------- #
def _load(path, cache):
    from ost import io
    if path not in cache:
        try:
            cache[path] = io.LoadMMCIF(path)
        except Exception:                     # e.g. duplicate atoms in the deposited file (5WT3)
            cache[path] = io.LoadMMCIF(path, fault_tolerant=True)
    return cache[path]


def _cut(ent, specs):
    """Entity with only the mapped residues of the given chains, renumbered 1..n per chain."""
    from ost import mol
    view, renum = ent.CreateEmptyView(), {}
    for chain, numbers in specs:
        ch = ent.FindChain(chain)
        if not ch.IsValid():
            raise ValueError(f"chain {chain} not found")
        renum[chain] = {}
        for i, n in enumerate(numbers):
            r = ch.FindResidue(mol.ResNum(int(n)))
            if r.IsValid():
                view.AddResidue(r, mol.INCLUDE_ATOMS)
                renum[chain][r.number] = i + 1
    new = mol.CreateEntityFromView(view, True)
    ed = new.EditXCS()
    for ch in new.chains:                     # two passes so that no intermediate number collides
        for r in list(ch.residues):
            ed.SetResidueNumber(r, mol.ResNum(renum[ch.name][r.number] + 100000))
        for r in list(ch.residues):
            ed.SetResidueNumber(r, mol.ResNum(r.number.num - 100000))
    return new


def score(validate=False):
    from ost.mol.alg import dockq
    cache, out = {}, []
    for it in json.load(open(_maps_path(validate))):
        rec = {k: it.get(k, "") for k in ("period", "pdb", "file", "native_pair", "model_pair")}
        P, R = it["protein"], it["rna"]
        try:
            ref = _cut(_load(it["cif"], cache), [(P["native_chain"], [a for a, _ in P["pairs"]]),
                                                 (R["native_chain"], [a for a, _ in R["pairs"]])])
            mod = _cut(_load(it["model"], cache), [(P["model_chain"], [b for _, b in P["pairs"]]),
                                                   (R["model_chain"], [b for _, b in R["pairs"]])])
            d = dockq.DockQ(mod, ref, P["model_chain"], R["model_chain"], P["native_chain"], R["native_chain"])
            rec.update(dockq=round(d["DockQ"], 4), fnat=round(d["fnat"], 4), irmsd=round(d["irmsd"], 3),
                       lrmsd=round(d["lrmsd"], 3), nnat=d["nnat"], n_protein_res=len(P["pairs"]), n_rna_res=len(R["pairs"]))
            if d["nnat"] == 0:                # no native contact among the mapped residues: DockQ undefined
                rec.update(dockq="", error="no native contact among mapped residues")
            else:                             # interface lDDT, contact similarity and QS of the same two chains
                sc = _scorer(mod, ref, {P["native_chain"]: P["model_chain"], R["native_chain"]: R["model_chain"]})
                vals, err = _scores(sc, ("ilddt", "ics", "qs_global"))
                rec.update(ilddt=vals.get("ilddt", ""), ics=vals.get("ics", ""), qs=vals.get("qs_global", ""), error=err)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        out.append(rec)
    path = _out_path(validate)
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    ok = sum(1 for r in out if r.get("dockq") not in (None, ""))
    print(f"{ok}/{len(out)} interfaces scored -> {path}")
    for r in out:
        if r.get("error"):
            print(f"  {r['period']} {r['pdb']} {r['native_pair']}: {r['error']}")


def compare():
    import numpy as np
    from scipy.stats import spearmanr
    ost = {(r["period"], r["pdb"].upper(), r["native_pair"]): float(r["dockq"])
           for r in _pose_rows() if not _missing(r["dockq"]) and r.get("dockq_source", "ost") == "ost"}
    own = [(ost[k], float(r["dockq"])) for r in csv.DictReader(open(VALIDATION_TSV), delimiter="\t")
           if not _missing(r["dockq"]) and (k := (r["period"], r["pdb"].upper(), r["native_pair"])) in ost]
    a, b = np.array(own).T
    d = np.abs(a - b)
    print(f"{len(a)} interfaces: Spearman {spearmanr(a, b)[0]:.3f}, median |diff| {np.median(d):.3f}, "
          f"90th percentile {np.quantile(d, 0.9):.3f}, same side of 0.23 {100 * np.mean((a >= 0.23) == (b >= 0.23)):.1f}%")


# --------------------------------------------------------------------------- #
# Whole complex: all mapped chains scored together (lDDT, iLDDT, ICS, QS, DockQ averaged as OpenStructure does)
# --------------------------------------------------------------------------- #
def _complex_maps_path(validate):
    return os.path.join(OUTDIR, "own_complex_maps_validation.json" if validate else "own_complex_maps.json")


def export_complex(validate=False):
    """Every polymer chain of the complexes with a fallback interface or an unmapped chain in OpenStructure's comparison
    (with --validate: an eighth of the complexes that OpenStructure scored completely), with the residue correspondence
    of the exact mapping."""
    import sqlite3
    from evaluation.interface_rna_pose import (PERIODS, INPUT_ROOTS, ComplexMapping, locate_inputs, _locate_model, read_model,
                                               drop_inconsistent_ost_chains, interface_scores)
    own = {(r["period"], r["pdb"].upper()) for r in csv.DictReader(open(OWN_TSV), delimiter="\t")}
    poses = _pose_rows()
    incomplete = {(r["period"], r["pdb"].upper()) for r in poses if r.get("dockq_source") != "ost"} | set(ost_unmapped())
    if validate:
        need = {(r["period"], r["pdb"].upper()) for r in poses} - incomplete
        need = {k for k in need if zlib.crc32(k[1].encode()) % 8 == 0}
    else:
        need = own | set(ost_unmapped())        # fallback interfaces or chains OpenStructure left unmapped
    out = []
    for period, db in PERIODS.items():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
                   p.ChainIDpairList_proteinRNA AS p_pairs, e.AAMotif AS e_aa, e.RNAMotif AS e_rna,
                   e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        for row in rows:
            pid = str(row["id"]).strip().upper()
            if (period, pid) not in need:
                continue
            cif, js = locate_inputs(row["id"], row["fn"], INPUT_ROOTS)
            af = _locate_model(row["fn"])
            mdl, mseqs = read_model(af)
            mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
            drop_inconsistent_ost_chains(mapping, mseqs)
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                             mapping)
            out.append(dict(period=period, pdb=pid, file=os.path.basename(af), cif=cif, model=af,
                            chains=_complex_chains(cif, mapping), n_native_chains=len(mapping.label_residues)))
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(_complex_maps_path(validate), "w"))
    print(f"{len(out)} complexes ({len(need)} requested) -> {_complex_maps_path(validate)}")


def _complex_chains(cif, mapping):
    """Every mapped polymer chain of the complex with its residue correspondence ([label_seq_id, model index]) from
    the exact mapping."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    m = MMCIF2Dict(cif)
    ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
    lseq = {(la, au, "" if ic in ("?", ".") else ic): ls for la, au, ic, ls in
            zip(m["_atom_site.label_asym_id"], m["_atom_site.auth_seq_id"], ins, m["_atom_site.label_seq_id"])}
    auth_of = {la: au for au, la in mapping.auth_to_label.items()}
    chains, used = [], set()
    for la, res in mapping.label_residues.items():
        au = auth_of.get(la)
        mc = mapping.pred_chain(au) if au is not None else None
        if mc is None or mc in used:
            continue
        fwd = mapping._model_index(la)
        pairs = []
        for k in range(len(res)):
            ls = lseq.get((la, str(res[k][0]), str(res[k][1] or "").strip()))
            if k in fwd and ls not in (None, ".", "?"):
                pairs.append([int(ls), int(fwd[k]) + 1])
        if pairs:
            used.add(mc)
            chains.append(dict(native_chain=la, model_chain=mc, pairs=pairs))
    return chains


def score_complex(validate=False, maps=None, path=None):
    cache, out = {}, []
    for it in json.load(open(maps or _complex_maps_path(validate))):
        rec = {k: it.get(k, "") for k in ("period", "pdb", "file", "n_native_chains")}
        rec["n_mapped_chains"] = len(it["chains"])
        try:
            ref = _cut(_load(it["cif"], cache), [(c["native_chain"], [a for a, _ in c["pairs"]]) for c in it["chains"]])
            mod = _cut(_load(it["model"], cache), [(c["model_chain"], [b for _, b in c["pairs"]]) for c in it["chains"]])
            sc = _scorer(mod, ref, {c["native_chain"]: c["model_chain"] for c in it["chains"]})
            vals, err = _scores(sc, ("lddt", "ilddt", "ics", "qs_global", "dockq_wave"))
            rec.update(vals, error=err)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        out.append(rec)
        for k in [k for k in cache if k not in (it["cif"], it["model"])]:
            del cache[k]                       # keep memory bounded on large assemblies
    path = path or (COMPLEX_VALIDATION_TSV if validate else COMPLEX_TSV)
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, COMPLEX_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{sum(1 for r in out if not r.get('error'))}/{len(out)} complexes scored -> {path}")
    for r in out:
        if r.get("error"):
            print(f"  {r['period']} {r['pdb']}: {r['error']}")


def _db_values(period):
    """{PDB: row} of the reference database of a period (per-interface and complex-level OpenStructure metrics)."""
    import sqlite3
    con = sqlite3.connect(f"file:{DB_FAMILIES[period][0]}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return {str(r["exp_db_id"]).strip().upper(): r for r in con.execute(
        "SELECT exp_db_id, RBP_QS, RNA_iLDDT_AllChains, RNA_ICS_AllChains, Complex_LDDT_allAtom, iLDDT, ICS, QS_global, "
        "Dockq FROM pred_protein_rna")}


def compare_all():
    """Fallback against OpenStructure where both exist: per interface (iLDDT, ICS, QS) and per complex."""
    import numpy as np
    from scipy.stats import spearmanr
    db = {p: _db_values(p) for p in DB_FAMILIES}
    def keyed(cell):
        return {frozenset(k.split(":")): float(v) for d in _rbp_list(cell) for k, v in d.items() if v is not None}
    def show(name, pairs):
        if not pairs:
            print(f"  {name}: no pairs"); return
        a, b = np.array(pairs, float).T
        print(f"  {name:22s} n={len(a):4d}  Spearman {spearmanr(a, b)[0]:.3f}  median |diff| {np.median(np.abs(a - b)):.3f}  "
              f"90th pct {np.quantile(np.abs(a - b), 0.9):.3f}")
    rows = list(csv.DictReader(open(VALIDATION_TSV), delimiter="\t"))
    for col, own_key in (("RNA_iLDDT_AllChains", "ilddt"), ("RNA_ICS_AllChains", "ics"), ("RBP_QS", "qs")):
        pairs = []
        for r in rows:
            row = db.get(r["period"], {}).get(r["pdb"].upper())
            if row is None or _missing(r.get(own_key)):
                continue
            v = keyed(row[col]).get(frozenset(r["model_pair"].split(":")))
            if v is not None:
                pairs.append((v, float(r[own_key])))
        show(f"interface {own_key}", pairs)
    if os.path.isfile(COMPLEX_VALIDATION_TSV):
        rows = list(csv.DictReader(open(COMPLEX_VALIDATION_TSV), delimiter="\t"))
        for col, own_key in (("Complex_LDDT_allAtom", "lddt"), ("iLDDT", "ilddt"), ("ICS", "ics"), ("QS_global", "qs_global"),
                             ("Dockq", "dockq_wave")):
            pairs = [(float(db[r["period"]][r["pdb"]][col]), float(r[own_key])) for r in rows
                     if not _missing(r.get(own_key)) and r["pdb"] in db.get(r["period"], {})
                     and not _missing(db[r["period"]][r["pdb"]][col])]
            show(f"complex {own_key}", pairs)


# --------------------------------------------------------------------------- #
# Step 3: write the fallback DockQ into RBP_DockQ
# --------------------------------------------------------------------------- #
from evaluation.bundle_paths import ALL_DB as _ZIP, PROTEIN_RNA_DB
# databases holding the default AF3 predictions of each period (the first is the reference the copies must match)
DB_FAMILIES = {
    "post": [PROTEIN_RNA_DB["post"],
             os.path.join(_ZIP, "a2021_post_cutoff", "protein_rna_ion_baseline.db"),
             os.path.join(_ZIP, "other_analyses", "self_templating", "protein_rna_post_cutoff_pool.db")],
    "pre": [PROTEIN_RNA_DB["pre"],
            os.path.join(_ZIP, "other_analyses", "self_templating", "protein_rna_pre_cutoff_pool.db")],
}


def _rbp_list(cell):
    """RBP_DockQ cell -> list of {pair: value}; empty, null and scalar cells give []."""
    if cell is None or isinstance(cell, (int, float)) or str(cell).strip() in ("", "[]", "None", "null"):
        return []
    try:
        v = json.loads(cell)
    except ValueError:
        return []
    return [d for d in v if isinstance(d, dict)] if isinstance(v, list) else []


def _replace_entries(items, add):
    """RBP_DockQ items with the fallback values `add` ({pair: DockQ}) set: an earlier fallback value of the same pair (in
    either order) is replaced, as these interfaces have no OpenStructure value. -> (new items, number changed)"""
    keys = {frozenset(k.split(":")): k for k in add}
    old = {frozenset(k.split(":")): v for d in items for k, v in d.items()}
    changed = sum(old.get(f) != add[k] for f, k in keys.items())
    return [d for d in items if frozenset(next(iter(d)).split(":")) not in keys] + [{k: v} for k, v in add.items()], changed


def write_db(dry_run=False):
    """Write the fallback DockQ of each native interface without an OpenStructure one into RBP_DockQ ({"P:R": value},
    predicted chain names as in ChainIDpairList_proteinRNA), replacing an earlier fallback value of the pair, in every
    database of DB_FAMILIES whose rows match the reference."""
    import sqlite3
    pred_pair = {(r["period"], r["pdb"].upper(), r["native_pair"]): r["pred_pair"] for r in _pose_rows()}
    fill = {}
    for r in csv.DictReader(open(OWN_TSV), delimiter="\t"):
        if _missing(r["dockq"]):
            continue
        pp = pred_pair.get((r["period"], r["pdb"].upper(), r["native_pair"]))
        if pp:
            fill.setdefault((r["period"], r["pdb"].upper()), {})[pp] = round(float(r["dockq"]), 3)
    for period, dbs in DB_FAMILIES.items():
        ref = None
        for db in dbs:
            if not os.path.isfile(db):
                print(f"  skip (missing) {db}")
                continue
            con = sqlite3.connect(db)
            rows = {str(i).strip().upper(): (q, p) for i, q, p in
                    con.execute("SELECT exp_db_id, RBP_DockQ, ChainIDpairList_proteinRNA FROM pred_protein_rna")}
            if ref is None:
                ref = rows
            elif rows != ref:
                print(f"  skip (rows differ from {dbs[0]}) {db}")
                con.close()
                continue
            n_pairs = n_cx = 0
            for pid, (cell, _pairs) in rows.items():
                add = fill.get((period, pid))
                if not add:
                    continue
                items, changed = _replace_entries(_rbp_list(cell), add)
                if not changed:
                    continue
                n_pairs += changed; n_cx += 1
                if not dry_run:
                    con.execute("UPDATE pred_protein_rna SET RBP_DockQ = ? WHERE UPPER(TRIM(exp_db_id)) = ?",
                                (json.dumps(items), pid))
            if not dry_run:
                con.commit()
            con.close()
            print(f"  {period}: {n_pairs} interfaces in {n_cx} complexes {'would be ' if dry_run else ''}written -> {db}")


RBP_COLUMNS = (("RBP_QS", "qs"), ("RBP_fnat", "fnat"), ("RBP_irmsd", "irmsd"))     # SWISS-MODEL per-interface scores
ALLCHAINS_COLUMNS = (("RNA_iLDDT_AllChains", "ilddt"), ("RNA_ICS_AllChains", "ics"))  # local OpenStructure per interface
COMPLEX_COLUMNS = (("iLDDT", "ilddt"), ("ICS", "ics"), ("QS_global", "qs_global"), ("Dockq", "dockq_wave"))
POST_KEYED_COPIES = [os.path.join(_ZIP, "a2021_post_cutoff", "protein_rna_ion_baseline.db")]   # same predictions as a2021


def _keyed_cell(cell):
    """Keyed list cell ([{"P:R": v}, ...]) -> list; the older matrix layout ([[...]]) and empty cells give []."""
    return _rbp_list(cell) if str(cell or "").lstrip().startswith("[{") else []


def write_db_metrics(dry_run=False):
    """Fill the other OpenStructure metrics wherever OpenStructure could not map the chains.

    Per interface (keyed like RBP_DockQ by the predicted protein:RNA pair):
      RBP_QS, RBP_fnat, RBP_irmsd      interfaces whose DockQ is the fallback (SWISS-MODEL returned none)
      RNA_iLDDT_AllChains, RNA_ICS_AllChains
                                       keyed cells: OpenStructure's value is kept where both chains were mapped, and
                                       replaced where the DockQ is the fallback, a chain was unmapped (UNMAPPED_TSV) or
                                       the value is missing; cells in the older matrix layout (pre-cutoff) are rebuilt
                                       as keyed cells from the exact-mapping values of every interface
    Per complex (fallback interface or unmapped chain; the whole complex scored with all mapped chains): iLDDT, ICS,
    QS_global, Dockq (OpenStructure's dockq_wave) and Complex_LDDT_allAtom. Copies of the post-cutoff predictions
    (POST_KEYED_COPIES) receive the final keyed cells of the reference database."""
    import sqlite3
    pred_pair = {(r["period"], r["pdb"].upper(), r["native_pair"]): (r["pred_pair"], r.get("dockq_source"))
                 for r in _pose_rows()}
    unm = ost_unmapped()
    iface = {}
    for k, r in all_interface_metrics().items():
        pp = pred_pair.get(k)
        if pp and not _missing(r.get("dockq")):
            iface.setdefault(k[:2], {})[pp[0]] = (r, pp[1] == "own")
    cx = {(r["period"], r["pdb"].upper()): r for r in csv.DictReader(open(COMPLEX_TSV), delimiter="\t")
          if not r.get("error") and not _missing(r.get("dockq_wave"))}
    for period, dbs in DB_FAMILIES.items():
        for db in dbs:
            if not os.path.isfile(db):
                continue
            con = sqlite3.connect(db)
            cols = {c[1] for c in con.execute("PRAGMA table_info(pred_protein_rna)")}
            sel = ["exp_db_id", "Complex_LDDT_allAtom"] + [c for c, _ in RBP_COLUMNS + ALLCHAINS_COLUMNS if c in cols]
            n = dict(rbp=0, kept=0, filled=0, rebuilt=0, cx=0)
            for row in con.execute(f"SELECT {', '.join(sel)} FROM pred_protein_rna").fetchall():
                row = dict(zip(sel, row))
                key = (period, str(row["exp_db_id"]).strip().upper())
                ifs, bad = iface.get(key, {}), unm.get(key, set())
                sets, vals = [], []
                own_ifs = {p: r for p, (r, own) in ifs.items() if own}
                for c, k in RBP_COLUMNS:
                    if c in row and own_ifs:
                        items = [d for d in _rbp_list(row[c]) if next(iter(d)) not in own_ifs]
                        items += [{p: round(float(r[k]), 3)} for p, r in own_ifs.items() if not _missing(r.get(k))]
                        sets.append(f"{c} = ?"); vals.append(json.dumps(items)); n["rbp"] += len(own_ifs)
                for c, k in ALLCHAINS_COLUMNS:
                    if c not in row or not ifs:
                        continue
                    old = {frozenset(next(iter(d)).split(":")): d for d in _keyed_cell(row[c])}
                    matrix = str(row[c] or "").lstrip().startswith("[[")
                    items = []
                    for p, (r, own) in ifs.items():
                        a, b = p.split(":")
                        keep = (not matrix and not own and frozenset((a, b)) in old and "*" not in bad
                                and a not in bad and b not in bad)
                        if keep:
                            items.append(old[frozenset((a, b))]); n["kept"] += 1
                        elif not _missing(r.get(k)):
                            items.append({p: round(float(r[k]), 3)}); n["rebuilt" if matrix else "filled"] += 1
                    # pairs that are not protein-RNA pose interfaces (e.g. RNA-RNA) have no exact-mapping value: keep them
                    done = {frozenset(p.split(":")) for p in ifs}
                    items += [d for kk, d in old.items() if kk not in done]
                    sets.append(f"{c} = ?"); vals.append(json.dumps(items))
                c_row = cx.get(key)
                if c_row:
                    for c, k in COMPLEX_COLUMNS:
                        if c in cols and not _missing(c_row.get(k)):
                            sets.append(f"{c} = ?"); vals.append(round(float(c_row[k]), 3))
                    if "Complex_LDDT_allAtom" in cols and not _missing(c_row.get("lddt")) and (bad or _missing(row["Complex_LDDT_allAtom"])):
                        sets.append("Complex_LDDT_allAtom = ?"); vals.append(round(float(c_row["lddt"]), 3))
                    n["cx"] += 1
                if sets and not dry_run:
                    con.execute(f"UPDATE pred_protein_rna SET {', '.join(sets)} WHERE exp_db_id = ?", vals + [row["exp_db_id"]])
            if not dry_run:
                con.commit()
            con.close()
            print(f"  {period}: {'would update' if dry_run else 'updated'} {db}  RBP_* {n['rbp']} interfaces | iLDDT/ICS kept "
                  f"{n['kept']}, filled {n['filled']}, rebuilt {n['rebuilt']} | complexes {n['cx']}")
    # copies of the post-cutoff predictions take the reference's final keyed cells and complex values
    if dry_run:
        return
    ref = sqlite3.connect(DB_FAMILIES["post"][0])
    cols_copy = [c for c, _ in RBP_COLUMNS + ALLCHAINS_COLUMNS + COMPLEX_COLUMNS] + ["RBP_DockQ", "Complex_LDDT_allAtom"]
    vals = {str(r[0]).strip().upper(): r[1:] for r in ref.execute(f"SELECT exp_db_id, {', '.join(cols_copy)} FROM pred_protein_rna")}
    ref.close()
    for db in POST_KEYED_COPIES:
        if not os.path.isfile(db):
            continue
        con = sqlite3.connect(db)
        have = {c[1] for c in con.execute("PRAGMA table_info(pred_protein_rna)")}
        use = [i for i, c in enumerate(cols_copy) if c in have]
        k = 0
        for pid, v in vals.items():
            con.execute(f"UPDATE pred_protein_rna SET {', '.join(cols_copy[i] + ' = ?' for i in use)} "
                        "WHERE UPPER(TRIM(exp_db_id)) = ?", [v[i] for i in use] + [pid]); k += 1
        con.commit(); con.close()
        print(f"  post copy: {k} rows take the keyed cells of {DB_FAMILIES['post'][0]} -> {db}")


# --------------------------------------------------------------------------- #
# Other prediction sets (own AF3 models, e.g. no-MSA runs): one database + its folder with pdb/ and af/
# --------------------------------------------------------------------------- #
def _tag_paths(tag):
    return (os.path.join(OUTDIR, f"own_dockq_maps_{tag}.json"), os.path.join(OUTDIR, f"own_dockq_{tag}.tsv"))


def _complex_tag_paths(tag):
    return (os.path.join(OUTDIR, f"own_complex_maps_{tag}.json"), os.path.join(OUTDIR, f"own_complex_{tag}.tsv"))


def _columns(con, table="pred_protein_rna"):
    return {c[1] for c in con.execute(f"PRAGMA table_info({table})")}


def drop_misaligned_ost_chains(mapping, model_seqs, min_ratio=0.8):
    """Remove OST chain assignments whose alignment pairs clearly fewer identical residues than the exact global
    alignment of the two chain sequences (OpenStructure 2.11 aligns some poly-U RNAs to a single residue); the
    removed chains are re-mapped by the sequence fallback. -> removed label chains."""
    bad = []
    for la, mc in list(mapping.chain_mapping.items()):
        if la not in mapping._aln or mc not in model_seqs:
            continue
        n_seq = len(mapping._exact_alignment(mapping._seq(la), model_seqs[mc]))
        if n_seq and len(mapping._model_index(la)) < min_ratio * n_seq:
            bad.append(la)
    for la in bad:
        del mapping.chain_mapping[la]
        mapping.source.pop(la, None)
        mapping._index.pop(la, None)
    return bad


def _structure_file(folder, name):
    """Path of `name` in `folder`, matched without regard to case (PDB ids are staged in either case), or None."""
    exact = os.path.join(folder, name)
    if os.path.isfile(exact):
        return exact
    if os.path.isdir(folder):
        for fn_ in os.listdir(folder):
            if fn_.lower() == name.lower():
                return os.path.join(folder, fn_)
    return None


def export_db(db, root, tag, validate=False, ost_root=None, check_alignment=False, fill_missing=False):
    """Native protein-RNA interfaces of `db` whose predicted pair has no RBP_DockQ value (with --validate: those that
    have one). Models from root/af, references from root/pdb; the chain mapping comes from sequence alone, because the
    OpenStructure JSONs of the default predictions do not describe these models. With `ost_root` (a folder holding the
    OpenStructure JSONs of exactly these models, *_lddt.json) OpenStructure's chain assignment is used where it fits the
    model file, as for the default predictions; with `check_alignment` also only where its alignment is complete
    (drop_misaligned_ost_chains). With `fill_missing` an interface also counts as unscored when it has no
    RNA_iLDDT_AllChains value (OpenStructure's whole-complex LDDT can fail where its DockQ does not). The predicted
    pair is written in the model's chain names, which are the database's (ChainIDpairList_proteinRNA, RBP_DockQ)."""
    import sqlite3
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from evaluation.interface_rna_pose import (ComplexMapping, read_model, interface_scores, interfaces_of,
                                               drop_inconsistent_ost_chains)
    root = os.path.abspath(root)                  # the scoring step runs elsewhere
    ost = {}
    if ost_root:
        for dp, _dn, fns in os.walk(ost_root):
            for fn_ in fns:
                if fn_.endswith("_lddt.json"):
                    ost.setdefault(fn_[:-len("_lddt.json")].lower(), os.path.join(dp, fn_))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = _columns(con)
    rbp = "p.RBP_DockQ" if "RBP_DockQ" in cols else "NULL"   # a run without interface scoring has none
    allc = "p.RNA_iLDDT_AllChains" if fill_missing and "RNA_iLDDT_AllChains" in cols else "NULL"
    rows = con.execute(f"""
        SELECT p.exp_db_id AS id, p.FileName AS fn, {rbp} AS rbp, {allc} AS allc, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
               p.ChainIDpairList_proteinRNA AS p_pairs, e.AAMotif AS e_aa, e.RNAMotif AS e_rna,
               e.ChainIDpairList_proteinRNA AS e_pairs
        FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
    con.close()
    # pairs whose RBP_DockQ entry is a fallback written earlier by write-db (own_dockq_<tag>.tsv) have no OpenStructure value
    _, written_tsv = _tag_paths(tag)
    written = {(r["pdb"].upper(), r["file"], frozenset(r["model_pair"].split(":")))
               for r in csv.DictReader(open(written_tsv), delimiter="\t")} if os.path.isfile(written_tsv) else set()
    out, skipped = [], []
    for row in rows:
        pid = str(row["id"]).strip().upper()
        cif = _structure_file(os.path.join(root, "pdb"), f"{pid}.cif")
        af = _structure_file(os.path.join(root, "af"), os.path.basename(str(row["fn"])))
        if not (cif and af):
            skipped.append(pid)
            continue
        have = {frozenset(k.split(":")) for d in _rbp_list(row["rbp"]) for k, v in d.items() if v is not None
                and (pid, os.path.basename(af), frozenset(k.split(":"))) not in written}
        if fill_missing:
            have &= {frozenset(k.split(":")) for d in _keyed_cell(row["allc"]) for k, v in d.items() if v is not None}
        mdl, mseqs = read_model(af)
        js = ost.get(os.path.splitext(os.path.basename(af))[0].lower())
        mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
        if js:
            drop_inconsistent_ost_chains(mapping, mseqs)
            if check_alignment:
                drop_misaligned_ost_chains(mapping, mseqs)
        interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                         dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                         mapping)
        m = MMCIF2Dict(cif)
        ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
        lseq = {(la, au, "" if ic in ("?", ".") else ic): ls for la, au, ic, ls in
                zip(m["_atom_site.label_asym_id"], m["_atom_site.auth_seq_id"], ins, m["_atom_site.label_seq_id"])}
        seen = set()
        for P, R, _aa, _rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
            laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
            p, r = mapping.pred_chain(P), mapping.pred_chain(R)
            if None in (laP, laR, p, r) or (p, r) in seen:
                continue
            seen.add((p, r))
            if (frozenset((p, r)) in have) != validate:
                continue
            rec = dict(period=tag, pdb=pid, file=os.path.basename(af), native_pair=f"{P}:{R}", model_pair=f"{p}:{r}",
                       cif=cif, model=af)
            for side, la, mc in (("protein", laP, p), ("rna", laR, r)):
                fwd, res = mapping._model_index(la), mapping.label_residues[la]
                pairs = []
                for k in range(len(res)):
                    ls = lseq.get((la, str(res[k][0]), str(res[k][1] or "").strip()))
                    if k in fwd and ls not in (None, ".", "?"):
                        pairs.append([int(ls), int(fwd[k]) + 1])
                rec[side] = dict(native_chain=la, model_chain=mc, pairs=pairs)
            out.append(rec)
    maps, _ = _tag_paths(tag + ("_validation" if validate else ""))
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(maps, "w"))
    print(f"{tag}: {len(out)} interfaces from {len({o['pdb'] for o in out})} complexes -> {maps}"
          + (f"; no model or reference file for {len(skipped)} complexes ({', '.join(skipped[:8])})" if skipped else ""))


def compare_db(db, tag):
    """Fallback DockQ (sequence mapping) against the RBP_DockQ values the database already has."""
    import sqlite3
    import numpy as np
    from scipy.stats import spearmanr
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {}
    for pid, fn, cell in con.execute("SELECT exp_db_id, FileName, RBP_DockQ FROM pred_protein_rna"):
        for d in _rbp_list(cell):
            for k, v in d.items():
                if v is not None:
                    have[(str(pid).strip().upper(), os.path.basename(str(fn)), frozenset(k.split(":")))] = float(v)
    _, tsv = _tag_paths(tag + "_validation")
    pairs = [(have[k], float(r["dockq"])) for r in csv.DictReader(open(tsv), delimiter="\t") if not _missing(r["dockq"])
             and (k := (r["pdb"].upper(), r["file"], frozenset(r["model_pair"].split(":")))) in have]
    a, b = np.array(pairs).T
    d = np.abs(a - b)
    print(f"{tag}: {len(a)} interfaces, Spearman {spearmanr(a, b)[0]:.3f}, median |diff| {np.median(d):.3f}, "
          f"same side of 0.23 {100 * np.mean((a >= 0.23) == (b >= 0.23)):.1f}%")


def write_db_tag(db, tag, dry_run=False):
    """Write own_dockq_<tag>.tsv into RBP_DockQ of `db` (replacing an earlier fallback value of the pair), keyed by the
    model pair (the database's chain names); rows are matched by PDB ID and model file, as some databases hold several
    models of a target (seeds)."""
    import sqlite3
    _, tsv = _tag_paths(tag)
    fill = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        if not _missing(r["dockq"]):
            fill.setdefault((r["pdb"].upper(), r["file"]), {})[r["model_pair"]] = round(float(r["dockq"]), 3)
    con = sqlite3.connect(db)
    n_pairs = n_cx = 0
    for pid, fn, cell in con.execute("SELECT exp_db_id, FileName, RBP_DockQ FROM pred_protein_rna").fetchall():
        add = fill.get((str(pid).strip().upper(), os.path.basename(str(fn))))
        if not add:
            continue
        items, changed = _replace_entries(_rbp_list(cell), add)
        if changed:
            n_pairs += changed; n_cx += 1
            if not dry_run:
                con.execute("UPDATE pred_protein_rna SET RBP_DockQ = ? WHERE exp_db_id = ? AND FileName = ?",
                            (json.dumps(items), pid, fn))
    if not dry_run:
        con.commit()
    con.close()
    print(f"  {tag}: {n_pairs} interfaces in {n_cx} complexes {'would be ' if dry_run else ''}written -> {db}")


def _unmapped(cell):
    """OST_unmapped_chains cell -> True when OpenStructure left model chains unmapped or the comparison failed."""
    return str(cell or "").strip() not in ("", "0", "None", "[]")


def export_complex_db(db, root, tag, ost_root=None, check_alignment=False):
    """Whole-complex residue correspondences (exact mapping) for the complexes of `db` whose OpenStructure comparison
    is incomplete: model chains left unmapped or a failed comparison (OST_unmapped_chains), or a protein-RNA interface
    scored by the fallback (own_dockq_<tag>.tsv). Files as in export_db."""
    import sqlite3
    from evaluation.interface_rna_pose import ComplexMapping, read_model, interface_scores, drop_inconsistent_ost_chains
    root = os.path.abspath(root)
    ost = {}
    if ost_root:
        for dp, _dn, fns in os.walk(ost_root):
            for fn_ in fns:
                if fn_.endswith("_lddt.json"):
                    ost.setdefault(fn_[:-len("_lddt.json")].lower(), os.path.join(dp, fn_))
    _, own_tsv = _tag_paths(tag)
    own = {(r["pdb"].upper(), r["file"]) for r in csv.DictReader(open(own_tsv), delimiter="\t")
           if not _missing(r["dockq"])} if os.path.isfile(own_tsv) else set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    unm = "p.OST_unmapped_chains" if "OST_unmapped_chains" in _columns(con) else "NULL"
    rows = con.execute(f"""
        SELECT p.exp_db_id AS id, p.FileName AS fn, {unm} AS unmapped, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
               p.ChainIDpairList_proteinRNA AS p_pairs, e.AAMotif AS e_aa, e.RNAMotif AS e_rna,
               e.ChainIDpairList_proteinRNA AS e_pairs
        FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
    con.close()
    out = []
    for row in rows:
        pid = str(row["id"]).strip().upper()
        if not (_unmapped(row["unmapped"]) or (pid, os.path.basename(str(row["fn"]))) in own):
            continue
        cif = _structure_file(os.path.join(root, "pdb"), f"{pid}.cif")
        af = _structure_file(os.path.join(root, "af"), os.path.basename(str(row["fn"])))
        if not (cif and af):
            continue
        mdl, mseqs = read_model(af)
        js = ost.get(os.path.splitext(os.path.basename(af))[0].lower())
        mapping = ComplexMapping(cif, js, pred_seqs=mseqs)
        if js:
            drop_inconsistent_ost_chains(mapping, mseqs)
            if check_alignment:
                drop_misaligned_ost_chains(mapping, mseqs)
        interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                         dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                         mapping)
        out.append(dict(period=tag, pdb=pid, file=os.path.basename(af), cif=cif, model=af,
                        chains=_complex_chains(cif, mapping), n_native_chains=len(mapping.label_residues)))
    maps, _ = _complex_tag_paths(tag)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(maps, "w"))
    print(f"{tag}: {len(out)} complexes -> {maps}")


# per-interface columns of the fallback (own_dockq_<tag>.tsv field) and whole-complex columns (own_complex_<tag>.tsv)
RUN_INTERFACE_COLUMNS = (("RBP_DockQ", "dockq"), ("RBP_QS", "qs"), ("RBP_fnat", "fnat"), ("RBP_irmsd", "irmsd"),
                         ("RNA_iLDDT_AllChains", "ilddt"), ("RNA_ICS_AllChains", "ics"))


def write_run_db(db, tag, dry_run=False):
    """Write the fallback values of one run into `db`: per interface (own_dockq_<tag>.tsv, keyed by the model pair) the
    RUN_INTERFACE_COLUMNS OpenStructure gave no value for; per complex (own_complex_<tag>.tsv) iLDDT, ICS, QS_global,
    Dockq and Complex_LDDT_allAtom where OpenStructure left chains unmapped or missed an interface (a fallback
    DockQ; its values then cover only part of the complex, as in the paper) or gave no value. Every row gets Interface_score_source ("ost", or "exact_mapping" when complex values
    are the fallback's) and Fallback_interfaces (the model pairs whose DockQ is the fallback's)."""
    import sqlite3
    _, own_tsv = _tag_paths(tag)
    _, cx_tsv = _complex_tag_paths(tag)
    iface = {}
    if os.path.isfile(own_tsv):
        for r in csv.DictReader(open(own_tsv), delimiter="\t"):
            if not _missing(r["dockq"]):
                iface.setdefault((r["pdb"].upper(), r["file"]), {})[r["model_pair"]] = r
    cx = {(r["pdb"].upper(), r["file"]): r for r in csv.DictReader(open(cx_tsv), delimiter="\t")
          if any(not _missing(r.get(k)) for _, k in COMPLEX_COLUMNS)} if os.path.isfile(cx_tsv) else {}
    con = sqlite3.connect(db)
    for c in [c for c, _ in RUN_INTERFACE_COLUMNS] + ["Interface_score_source", "Fallback_interfaces"]:
        if c not in _columns(con):
            con.execute(f"ALTER TABLE pred_protein_rna ADD COLUMN {c} TEXT")
    cols = _columns(con)
    cx_cols = [c for c, _ in COMPLEX_COLUMNS] + ["Complex_LDDT_allAtom"]
    names = ["exp_db_id", "FileName", "OST_unmapped_chains"] + [c for c, _ in RUN_INTERFACE_COLUMNS] + cx_cols
    sel = [c if c in cols else "NULL" for c in names]
    n_if = n_cx = 0
    for row in con.execute(f"SELECT {', '.join(sel)} FROM pred_protein_rna").fetchall():
        row = dict(zip(["id", "fn", "unmapped"] + names[3:], row))
        key = (str(row["id"]).strip().upper(), os.path.basename(str(row["fn"])))
        own_ifs, c_row = iface.get(key, {}), cx.get(key)
        sets, vals, fallback_pairs = [], [], []
        for c, k in RUN_INTERFACE_COLUMNS:
            items = _keyed_cell(row[c])
            have = {frozenset(next(iter(d)).split(":")) for d in items if next(iter(d.values())) is not None}
            add = {p: round(float(r[k]), 3) for p, r in own_ifs.items()
                   if not _missing(r.get(k)) and frozenset(p.split(":")) not in have}
            if add:
                sets.append(f"{c} = ?"); vals.append(json.dumps(items + [{p: v} for p, v in add.items()]))
                if c == "RBP_DockQ":
                    fallback_pairs = sorted(add)
        n_if += len(own_ifs)
        rescored = False
        if c_row:
            for c, k in list(COMPLEX_COLUMNS) + [("Complex_LDDT_allAtom", "lddt")]:
                if not _missing(c_row.get(k)) and (_unmapped(row["unmapped"]) or fallback_pairs or _missing(row[c])):
                    sets.append(f"{c} = ?"); vals.append(round(float(c_row[k]), 3)); rescored = True
            n_cx += rescored
        sets += ["Interface_score_source = ?", "Fallback_interfaces = ?"]
        vals += ["exact_mapping" if rescored else "ost", json.dumps(fallback_pairs)]
        if not dry_run:
            con.execute(f"UPDATE pred_protein_rna SET {', '.join(sets)} WHERE exp_db_id = ? AND FileName = ?",
                        vals + [row["id"], row["fn"]])
    if not dry_run:
        con.commit()
    con.close()
    print(f"  {tag}: {n_if} interfaces and {n_cx} whole complexes {'would be ' if dry_run else ''}written -> {db}")


# --------------------------------------------------------------------------- #
# Seed/model samples (evaluation/seed_sample_interfaces.py): 15 samples per target, each with its own OST JSON
# --------------------------------------------------------------------------- #
SEEDS_TAG = "seed_samples"


def export_seeds():
    """Native protein-RNA interfaces of every seed/model sample that OpenStructure did not score. The chain mapping
    is the sample's own OpenStructure assignment where it fits the model file, the sequence fallback otherwise."""
    import sqlite3
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from evaluation import seed_sample_interfaces as S
    from evaluation.interface_rna_pose import (PERIODS, ComplexMapping, read_model, interface_scores, interfaces_of,
                                               drop_inconsistent_ost_chains)
    con = sqlite3.connect(f"file:{PERIODS['post']}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    db = {str(r["id"]).strip().lower(): r for r in con.execute("""
        SELECT p.exp_db_id AS id, p.AAMotif AS p_aa, p.RNAMotif AS p_rna, p.ChainIDpairList_proteinRNA AS p_pairs,
               e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
        FROM pred_protein_rna p JOIN exp_protein_rna e ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""")}
    out = []
    for pid in S.targets():
        row, ref = db.get(pid), os.path.join(S.REF_DIR, f"{pid}.cif")
        if row is None or not os.path.isfile(ref):
            continue
        m = MMCIF2Dict(ref)
        ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
        lseq = {(la, au, "" if ic in ("?", ".") else ic): ls for la, au, ic, ls in
                zip(m["_atom_site.label_asym_id"], m["_atom_site.auth_seq_id"], ins, m["_atom_site.label_seq_id"])}
        for seed, model, cif, _summ in S.sample_files(pid):
            jpath = S.ost_json_path(pid, seed, model)
            js = json.load(open(jpath)) if jpath and os.path.isfile(jpath) else {}
            scored = {frozenset(it[:2]) for it in (js.get("dockq_interfaces") or []) if len(it) == 4}
            mdl, mseqs = read_model(cif)
            mapping = ComplexMapping(ref, jpath if js.get("status") == "SUCCESS" else None, pred_seqs=mseqs)
            drop_inconsistent_ost_chains(mapping, mseqs)
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                             mapping)
            seen = set()
            for P, R, _aa, _rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
                laP, laR = mapping.auth_to_label.get(str(P)), mapping.auth_to_label.get(str(R))
                p, r = mapping.pred_chain(P), mapping.pred_chain(R)
                if None in (laP, laR, p, r) or (p, r) in seen or frozenset((laP, laR)) in scored:
                    continue
                seen.add((p, r))
                rec = dict(period=f"s{seed}m{model}", pdb=pid.upper(), file=os.path.basename(cif),
                           native_pair=":".join(sorted((laP, laR))), model_pair=f"{p}:{r}", cif=ref, model=cif)
                for side, la, mc in (("protein", laP, p), ("rna", laR, r)):
                    fwd, res = mapping._model_index(la), mapping.label_residues[la]
                    pairs = []
                    for k in range(len(res)):
                        ls = lseq.get((la, str(res[k][0]), str(res[k][1] or "").strip()))
                        if k in fwd and ls not in (None, ".", "?"):
                            pairs.append([int(ls), int(fwd[k]) + 1])
                    rec[side] = dict(native_chain=la, model_chain=mc, pairs=pairs)
                out.append(rec)
    maps, _ = _tag_paths(SEEDS_TAG)
    json.dump(out, open(maps, "w"))
    print(f"{SEEDS_TAG}: {len(out)} sample interfaces of {len({o['pdb'] for o in out})} targets -> {maps}")


def seed_fallback():
    """{(PDB, seed, model): [(native label pair 'A:B' sorted, model protein:RNA pair, DockQ)]} for
    seed_sample_interfaces.py ({} before export_seeds/score have run)."""
    _, tsv = _tag_paths(SEEDS_TAG)
    out = {}
    if os.path.isfile(tsv):
        for r in csv.DictReader(open(tsv), delimiter="\t"):
            if not _missing(r["dockq"]):
                s, m = r["period"][1:].split("m")
                out.setdefault((r["pdb"].upper(), int(s), int(m)), []).append((r["native_pair"], r["model_pair"],
                                                                              float(r["dockq"])))
    return out


def _opt(flags, name):
    return flags[flags.index(name) + 1] if name in flags else None


if __name__ == "__main__":
    cmd, flags = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    db, root, tag, ost_root = _opt(flags, "--db"), _opt(flags, "--root"), _opt(flags, "--tag"), _opt(flags, "--ost-root")
    if "--outdir" in flags:                       # keep the maps and tables of one run (run_eval.py) with that run
        OUTDIR = os.path.abspath(_opt(flags, "--outdir"))
    if cmd == "export-complex" and db:
        export_complex_db(db, root, tag, ost_root, "--check-alignment" in flags)
        raise SystemExit
    if cmd == "score-complex" and tag:
        score_complex(maps=_complex_tag_paths(tag)[0], path=_complex_tag_paths(tag)[1])
        raise SystemExit
    if cmd == "write-run" and db:
        write_run_db(db, tag, "--dry-run" in flags)
        raise SystemExit
    if cmd == "write-metrics":
        write_db_metrics("--dry-run" in flags)
        raise SystemExit
    if cmd == "write-db" and db:
        write_db_tag(db, tag, "--dry-run" in flags)
        raise SystemExit
    if cmd == "write-db":
        write_db("--dry-run" in flags)
        raise SystemExit
    if cmd == "export" and db:
        export_db(db, root, tag, "--validate" in flags, ost_root, "--check-alignment" in flags, "--fill-missing" in flags)
        raise SystemExit
    if cmd == "score" and tag:
        _MAPS_OVERRIDE = _tag_paths(tag + ("_validation" if "--validate" in flags else ""))
        _maps_path = lambda validate: _MAPS_OVERRIDE[0]
        OWN_TSV = VALIDATION_TSV = _MAPS_OVERRIDE[1]
        score("--validate" in flags)
        raise SystemExit
    if cmd == "export-complex":
        export_complex("--validate" in flags)
        raise SystemExit
    if cmd == "score-complex":
        score_complex("--validate" in flags)
        raise SystemExit
    if cmd == "compare-all":
        compare_all()
        raise SystemExit
    if cmd == "export-seeds":
        export_seeds()
        raise SystemExit
    if cmd == "compare" and db:
        compare_db(db, tag)
        raise SystemExit
    if cmd == "export":
        export("rest" if "--rest" in flags else ("--validate" in flags))
    elif cmd == "score":
        score("rest" if "--rest" in flags else ("--validate" in flags))
    elif cmd == "compare":
        compare()
    else:
        raise SystemExit(__doc__)
