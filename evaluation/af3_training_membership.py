"""
Which of our targets are in the AlphaFold3 PDB training set?

AF3 does not publish its training PDB IDs. The supplementary information of Abramson et al.
(Nature 2024), section 2.5.4 "Filtering of targets", defines the set instead:
  1. released to the PDB before the cutoff date of 2021-09-30;
  2. reported resolution of 9 A or less;
  3. at most 300 polymer chains (training);
  4. polymer chains with fewer than 4 resolved residues are removed (chains, not entries).
This script applies rules 1-3 to every target from its deposited mmCIF (initial release date =
first _pdbx_audit_revision_history entry; resolution from _refine / _em_3d_reconstruction /
_reflns; polymer chains = struct_asym ids of polymer entities). NMR entries report no resolution;
whether AF3 kept them is not stated, so they are reported separately.

Writes results/af3_training_membership.tsv and prints a summary per set.
Run in the RNABench env:  python evaluation/af3_training_membership.py
"""

import csv
import glob
import os
import sqlite3
import sys

import gemmi

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PP = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre")
SETS = [
    ("protein-RNA", "pre", os.path.join(PP, "a2021_b2021_protein_rna", "b2021.db"), "exp_protein_rna"),
    ("protein-RNA", "post", os.path.join(PP, "a2021_b2021_protein_rna", "a2021.db"), "exp_protein_rna"),
    ("single-chain RNA", "pre", os.path.join(PP, "a2021_b2021_rna_only", "b2021_rna.db"), "exp_rna_rna"),
    ("single-chain RNA", "post", os.path.join(PP, "a2021_b2021_rna_only", "a2021_rna.db"), "exp_rna_rna"),
    ("multi-chain RNA", "pre", os.path.join(PP, "a2021_b2021_rna_rna", "b2021_rna_rna.db"), "exp_rna_rna"),
    ("multi-chain RNA", "post", os.path.join(PP, "a2021_b2021_rna_rna", "a2021_rna_rna.db"), "exp_rna_rna"),
]
CUTOFF = "2021-09-30"
OUT = os.path.join(_REPO_ROOT, "results", "af3_training_membership.tsv")


def cif_index():
    idx = {}
    for d in glob.glob(os.path.join(_REPO_ROOT, "data", "**", "pdb"), recursive=True):
        for p in glob.glob(os.path.join(d, "*.cif")):
            idx.setdefault(os.path.basename(p)[:-4].lower(), p)
    return idx


def _first(block, *tags):
    for t in tags:
        v = block.find_value(t)
        if v not in (None, "?", "."):
            return gemmi.cif.as_string(v)
    return None


def metadata(cif):
    block = gemmi.cif.read(cif).sole_block()
    dates = [d for d in block.find_values("_pdbx_audit_revision_history.revision_date") if d not in ("?", ".")]
    release = min(gemmi.cif.as_string(d) for d in dates) if dates else None
    res = _first(block, "_refine.ls_d_res_high", "_em_3d_reconstruction.resolution", "_reflns.d_resolution_high")
    method = _first(block, "_exptl.method")
    poly_entities = {gemmi.cif.as_string(e) for e in block.find_values("_entity_poly.entity_id")}
    asym = block.find("_struct_asym.", ["id", "entity_id"])
    n_poly = sum(1 for row in asym if gemmi.cif.as_string(row[1]) in poly_entities)
    try:
        res = float(res) if res is not None else None
    except ValueError:
        res = None
    return release, res, method, n_poly


def main():
    idx = cif_index()
    rows = []
    for system, period, db, table in SETS:
        if not os.path.exists(db):
            print(f"  (missing {db})")
            continue
        ids = [r[0] for r in sqlite3.connect(db).execute(f"SELECT DISTINCT PDBId FROM {table}")]
        for pid in ids:
            p = idx.get(str(pid).strip().lower()[:4])
            if p is None:
                rows.append(dict(system=system, period=period, pdb=pid, release=None, resolution=None,
                                 method=None, polymer_chains=None, in_training=None, reason="no mmCIF"))
                continue
            rel, res, meth, n = metadata(p)
            reasons = []
            if rel is None or rel >= CUTOFF:
                reasons.append(f"released {rel}")
            if res is None:
                reasons.append(f"no resolution ({meth})")
            elif res > 9.0:
                reasons.append(f"resolution {res}")
            if n > 300:
                reasons.append(f"{n} polymer chains")
            rows.append(dict(system=system, period=period, pdb=pid, release=rel, resolution=res, method=meth,
                             polymer_chains=n, in_training=not reasons, reason="; ".join(reasons)))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for system, period, _, _ in SETS:
        d = [r for r in rows if r["system"] == system and r["period"] == period]
        if not d:
            continue
        n_in = sum(r["in_training"] is True for r in d)
        print(f"{system:17s} {period:4s}: {len(d):4d} entries, meet the AF3 training rules: {n_in} "
              f"({100 * n_in / len(d):.1f}%)")
        from collections import Counter
        why = Counter(r["reason"].split(" ")[0] + (" " + r["reason"].split(" ")[1] if r["reason"].startswith("no") else "")
                      for r in d if r["in_training"] is not True)
        for k, v in why.most_common():
            print(f"      not: {k:24s} {v}")
        late = [r for r in d if r["release"] and r["release"] >= CUTOFF]
        if period == "pre" and late:
            print("      released on/after the cutoff: " + ", ".join(f"{r['pdb']} ({r['release']})" for r in late[:12]))
        if period == "post":
            early = [r for r in d if r["release"] and r["release"] < CUTOFF]
            print(f"      released before the cutoff: {len(early)}" + (": " + ", ".join(
                f"{r['pdb']} ({r['release']})" for r in early[:12]) if early else ""))
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    sys.exit(main())
