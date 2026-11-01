"""
Well-folded labels of the decision trees (LDDT_r > 0.85). Two definitions are available:

  backbone (default, the paper)  C3' LDDT_r of evaluation.extractEvalMetrics.calculate_complex_lddt, RNA_LDDT of
                                 the databases; per RNA chain computed from the structures (decision_tree.combined)
  allatom                        all-atom LDDT_r of OpenStructure, RNA_LDDT_allAtom; per RNA chain
                                 RNA_LDDT_AllChains (the predicted chain against its best-matching experimental chain)

Choose with the environment variable DT_LDDT=backbone|allatom before running the design and test scripts
(develop_v3, develop_rna, develop_rna_multi, combined, develop_probability, validate_2026). The shipped calibrations
are those of the paper (backbone); with DT_LDDT=allatom, rerun the design steps first (evaluation/make_analysis_tables.sh
lists them in order).

Some published databases have no RNA_LDDT: the 2026 test sets and the multi-chain RNA development sets added after the
benchmark (rna_multichain_added_*). Their backbone values come from the re-run databases of the Zenodo record
(rerun_2026-10.zip, computed with the paper's function; identical to the published values wherever both exist) and are
shipped as results/plots_odl/all_chains/decision_tree/backbone_lddt_supplement.tsv in analysis_inputs.zip.
"""
import csv
import json
import os
import sqlite3

LABEL = os.environ.get("DT_LDDT", "backbone").lower()
if LABEL not in ("backbone", "allatom"):
    raise ValueError(f"DT_LDDT must be backbone or allatom, not {LABEL!r}")
COLUMN = "RNA_LDDT" if LABEL == "backbone" else "RNA_LDDT_allAtom"
NAME = "backbone C3' LDDT_r" if LABEL == "backbone" else "all-atom LDDT_r"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPLEMENT = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "decision_tree", "backbone_lddt_supplement.tsv")
              for d in ("plots", "plots_odl")]


def _num(v):
    try:
        x = json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v
    except ValueError:
        return None
    if isinstance(x, list):
        x = x[0] if x else None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_SUPP = None


def _supplement():
    """{(database path below all_databases/, PDB): backbone LDDT_r} for databases without RNA_LDDT."""
    global _SUPP
    if _SUPP is None:
        _SUPP = {}
        path = next((p for p in SUPPLEMENT if os.path.exists(p)), None)
        if path:
            for r in csv.DictReader(open(path), delimiter="\t"):
                _SUPP[(r["database"], r["exp_db_id"].upper())] = float(r["RNA_LDDT"])
    return _SUPP


def _rel(db):
    db = os.path.abspath(db)
    key = os.sep + "all_databases" + os.sep
    return db.split(key, 1)[1] if key in db else os.path.basename(db)


def complex_lddt(db, table="pred_rna_rna"):
    """{PDB: LDDT_r of the aligned (longest) RNA chain} under the chosen label."""
    out = {str(e).strip().upper(): _num(v)
           for e, v in sqlite3.connect(db).execute(f"SELECT exp_db_id, {COLUMN} FROM {table}")}
    if LABEL == "backbone":
        rel, supp = _rel(db), _supplement()
        for k, v in out.items():
            if v is None and (rel, k) in supp:
                out[k] = supp[(rel, k)]
    return out


def chain_lddt(db, table="pred_protein_rna"):
    """{PDB: {predicted RNA chain: all-atom LDDT_r}} (RNA_LDDT_AllChains). The backbone per-chain value has no
    database column; decision_tree.combined computes it from the structures."""
    out = {}
    for e, raw in sqlite3.connect(db).execute(f"SELECT exp_db_id, RNA_LDDT_AllChains FROM {table}"):
        try:
            cells = json.loads(raw) if raw else []
        except ValueError:
            cells = []
        d = {}
        for c in cells if isinstance(cells, list) else [cells]:
            if isinstance(c, dict):
                d.update({str(k): _num(v) for k, v in c.items() if _num(v) is not None})
        out[str(e).strip().upper()] = d
    return out
