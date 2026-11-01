"""Load the 2 x 16 MSA/template ablation runs into one long table, with per-target
component availability.

Every condition folder holds one AlphaFold 3 run of the same targets with a different
subset of inputs removed, so the factorial is read as 32 databases keyed by
(target, rMSA, pMSA, pTemp, cTemp), where 1 = the component was PRESENT.

Component availability comes from a2021.db (the natural run): the ablation databases
cannot carry it, because the AlphaFold 3 input JSONs with the MSAs and templates are not
part of the released set. Removing a component a target never had is a no-op, so effect
estimates are meaningful only on the targets that carry it -- `carriers()` gives those.
"""
import ast
import json
import os
import sqlite3

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(REPO, "data", "nar_msa_template_ablation_20261007")
BC = os.path.join(AB, "by_condition")
NATURAL_DB = os.path.join(AB, "a2021.db")

SIDES = ("protein_side", "rna_side")
FACTORS = ("rMSA", "pMSA", "pTemp", "cTemp")
EXCLUDE = {"6FQL"}          # in the AF3 training set (released 2018)

METRICS = ["Complex_TM", "RNA_TM", "Complex_LDDT_allAtom", "RNA_LDDT_allAtom",
           "Complex_RMSD", "Dockq", "DockQ_Nr", "iLDDT", "ICS"]


def _conditions():
    out = []
    for r in "+-":
        for p in "+-":
            for t in "+-":
                for c in "+-":
                    out.append((f"{r}rMSA_{p}pMSA_{t}pTemp_{c}cTemp",
                                int(r == "+"), int(p == "+"), int(t == "+"), int(c == "+")))
    return out


def _num(v):
    """Complex_RMSD / RNA_RMSD are stored as a list, e.g. "[1.26, 2.35]" = [PyMOL, US-align].
    PyMOL's whole-complex align can mis-pair chain copies, so take the US-align value."""
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            parts = [float(x) for x in v.strip("[] ").split(",") if x.strip()]
        except ValueError:
            return None
        if not parts:
            return None
        return parts[1] if len(parts) > 1 else parts[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_factorial(verbose=True):
    """-> DataFrame with one row per (target, condition)."""
    frames = []
    for side in SIDES:
        for cond, rM, pM, pT, cT in _conditions():
            db = os.path.join(BC, side, cond, "protein_rna", "rbpDatabase.db")
            if not os.path.exists(db):
                if verbose:
                    print(f"  missing: {side}/{cond}")
                continue
            con = sqlite3.connect(db)
            df = pd.read_sql_query(
                "SELECT exp_db_id, " + ", ".join(f'"{m}"' for m in METRICS)
                + " FROM pred_protein_rna WHERE exp_db_id IS NOT NULL", con)
            con.close()
            for m in METRICS:
                df[m] = df[m].map(_num)
            df["side"] = side
            df["condition"] = cond
            df["rMSA"], df["pMSA"], df["pTemp"], df["cTemp"] = rM, pM, pT, cT
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[~out.exp_db_id.isin(EXCLUDE)].copy()
    return out


def _parse(v):
    if v in (None, "", "null"):
        return None
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(v)
        except Exception:
            return None


def _all_chains_have_msa(v):
    d = _parse(v)
    if not isinstance(d, dict) or not d:
        return False
    try:
        return all(float(x) > 1 for x in d.values())
    except (TypeError, ValueError):
        return False


def _has_templates(v):
    l = _parse(v)
    return bool(l) if isinstance(l, list) else False


def _availability_from_runs():
    """Fallback for targets with no a2021.db row: whether a component was present is read from
    the runs themselves -- AlphaFold 3 is deterministic, so if removing a component leaves the
    atom coordinates byte-identical across all 8 backgrounds, that target carried nothing to
    remove. Validated against a2021.db on the 223 targets that have both (pMSA 99.6%, pTemp
    97.8%, cTemp 91.9%, rMSA 93.3%)."""
    path = os.path.join(AB, "availability_from_runs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh).get("inferred", {})


def availability():
    """-> DataFrame: one row per target, with chain counts, class and the four
    'component present' flags, read from the natural run."""
    con = sqlite3.connect(NATURAL_DB)
    exp = pd.read_sql_query(
        "SELECT PDBId AS exp_db_id, NumberProteins, NumberRNAs FROM exp_protein_rna", con)
    pred = pd.read_sql_query(
        "SELECT exp_db_id, RNA_msa_size, Protein_msa_size, "
        "Protein_PDB_Templates, RBP_PDB_Templates FROM pred_protein_rna", con)
    con.close()
    av = exp.merge(pred, on="exp_db_id", how="left")
    av["rMSA"] = av.RNA_msa_size.map(_all_chains_have_msa)
    av["pMSA"] = av.Protein_msa_size.map(_all_chains_have_msa)
    av["pTemp"] = av.Protein_PDB_Templates.map(_has_templates)
    av["cTemp"] = av.RBP_PDB_Templates.map(_has_templates)
    av["cls"] = ["1v1" if (p == 1 and r == 1) else "multi"
                 for p, r in zip(av.NumberProteins, av.NumberRNAs)]
    av["all_four"] = av[list(FACTORS)].all(axis=1)
    av = av[["exp_db_id", "NumberProteins", "NumberRNAs", "cls", "all_four"] + list(FACTORS)]

    # Targets that are in the ablation set but not in a2021.db (8FN6): take the chain counts from
    # the folder database the run itself used, and the component flags from the runs.
    inferred = _availability_from_runs()
    extra = []
    for side in SIDES:
        db = os.path.join(BC, side, "+rMSA_+pMSA_+pTemp_+cTemp", "protein_rna", "rbpDatabase.db")
        if not os.path.exists(db):
            continue
        con = sqlite3.connect(db)
        rows = con.execute("select PDBId, NumberProteins, NumberRNAs from exp_protein_rna").fetchall()
        con.close()
        for pdb_id, npro, nrna in rows:
            if pdb_id in set(av.exp_db_id) or pdb_id not in inferred or pdb_id in EXCLUDE:
                continue
            flags = inferred[pdb_id]
            extra.append(dict(exp_db_id=pdb_id, NumberProteins=npro, NumberRNAs=nrna,
                              cls="1v1" if (npro == 1 and nrna == 1) else "multi",
                              all_four=all(flags.get(f, False) for f in FACTORS),
                              **{f: bool(flags.get(f, False)) for f in FACTORS}))
    if extra:
        av = pd.concat([av, pd.DataFrame(extra)], ignore_index=True)
    return av


def carriers(av, factor):
    """Targets that actually carry `factor` (so removing it can do something)."""
    return set(av.loc[av[factor], "exp_db_id"])


if __name__ == "__main__":
    fac = load_factorial()
    av = availability()
    known = av.set_index("exp_db_id")
    targets = sorted(set(fac.exp_db_id))
    missing = [t for t in targets if t not in known.index]
    print(f"factorial rows: {len(fac)}   targets: {len(targets)}   conditions: {fac.condition.nunique()}")
    print(f"targets with no row in a2021.db (no availability): {missing}")
    sub = av[av.exp_db_id.isin(targets)]
    print("\n                                   1v1   multi")
    for f, label in zip(FACTORS, ["RNA MSA (all RNA chains)", "Protein MSA (all protein chains)",
                                  "Protein templates", "RBP templates"]):
        a = ((sub.cls == "1v1") & sub[f]).sum()
        b = ((sub.cls == "multi") & sub[f]).sum()
        print(f"{label:<34}{a:>6}{b:>8}")
    print(f"{'All four':<34}{((sub.cls=='1v1')&sub.all_four).sum():>6}"
          f"{((sub.cls=='multi')&sub.all_four).sum():>8}")
    print(f"{'n':<34}{(sub.cls=='1v1').sum():>6}{(sub.cls=='multi').sum():>8}")
