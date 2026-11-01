"""
Single-character chain ids for AF3 models predicted with many ions (the ion input sets).

AF3 names the supplied ions A, B, ... and the RNA last; above 26 ions the ids become two letters (AA, BA, ...).
extractEvalMetrics.py writes the superposed model in PDB format, which allows one character per chain id, so these
models get no US-align TM/RMSD, no aligned file and therefore no all-atom LDDT. This script gives every chain a
single-character id (auth and label): the polymer chains first (A, B, ...), then the ions in their original order
(the remaining upper-case letters, lower-case letters, digits). Atoms, coordinates and B-factors are unchanged.
Polymers come first because PyMOL selects chains case-insensitively (an RNA 'b' would also select an ion 'B').

Run it on the af/ folder before the -dssr pass (ost env, which has gemmi):

    conda run -n ost python dataPrep/rename_ion_chains.py <af folder> [--db <rbpDatabase.db> --table pred_rna_rna]

Only models with a multi-letter chain id are rewritten (in place); <af folder>/chain_mapping.json records the old and
new ids. If the models were already parsed into a database, pass --db: the chain-id list stored at parse time
(RNAChainIDs) is updated with the same mapping, so the evaluation steps find the renamed chains.
"""
import argparse
import glob
import json
import os
import sqlite3
import string
import sys

import gemmi

FREE = string.ascii_uppercase + string.ascii_lowercase + string.digits


def rename(path):
    """Rewrite one model in place; returns {old id: new id} (empty if nothing had to change)."""
    st = gemmi.make_structure_from_block(gemmi.cif.read(path).sole_block())
    chains = list(st[0])
    if all(len(ch.name) == 1 for ch in chains):
        return {}
    if len(chains) > len(FREE):
        sys.exit(f"{path}: {len(chains)} chains, more than {len(FREE)} single-character ids")
    poly = [ch.name for ch in chains if len(ch) > 1]
    rest = [ch.name for ch in chains if len(ch) <= 1]
    mapping = {old: new for old, new in zip(poly + rest, FREE) if old != new}
    for model in st:
        for ch in model:
            if ch.name in mapping:
                ch.name = "~" + mapping[ch.name]          # two passes, so a new id never clashes with an old one
        for ch in model:
            if ch.name.startswith("~"):
                ch.name = ch.name[1:]
                for res in ch:
                    res.subchain = ch.name
    st.setup_entities()
    st.make_mmcif_document().write_file(path)
    return mapping


def update_db(db, table, maps):
    con = sqlite3.connect(db)
    for name, mapping in maps.items():
        row = con.execute(f"SELECT RNAChainIDs FROM {table} WHERE FileName = ?", (name,)).fetchone()
        if not row or row[0] is None:
            continue
        new = ",".join(mapping.get(x.strip(), x.strip()) for x in str(row[0]).split(","))
        con.execute(f"UPDATE {table} SET RNAChainIDs = ? WHERE FileName = ?", (new, name))
    con.commit()
    con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("af_folder")
    ap.add_argument("--db", help="database whose RNAChainIDs should follow the renaming")
    ap.add_argument("--table", default="pred_rna_rna")
    a = ap.parse_args()
    maps = {}
    for path in sorted(glob.glob(os.path.join(a.af_folder, "*.cif"))):
        mapping = rename(path)
        if mapping:
            maps[os.path.basename(path)] = mapping
            print(os.path.basename(path), mapping)
    out = os.path.join(a.af_folder, "chain_mapping.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    json.dump({**old, **maps}, open(out, "w"), indent=1)
    if a.db and maps:
        update_db(a.db, a.table, maps)
    print(f"{len(maps)} models renamed -> {out}")


if __name__ == "__main__":
    main()
