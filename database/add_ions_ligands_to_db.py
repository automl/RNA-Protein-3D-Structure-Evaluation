"""
Write Ions / Ligands columns on the exp_* table from mmCIF (StartConfig pdb folder).

Each entry is a dict:
  - Ions column:     ions='CL',     n=…, af3_supported=…
  - Ligands column: ligands='ATP', n=…, af3_supported=…
  n = residue instance count in the deposited model.
"""
import os
import sqlite3
from collections import Counter

from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from database.add_ion_location_to_db import ION_ELEMENTS
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig

allowed_ligands = {
    "ADP",
    "ATP",
    "AMP",
    "GTP",
    "GDP",
    "FAD",
    "NAD",
    "NAP",
    "NDP",
    "HEM",
    "HEC",
    "PLM",
    "OLA",
    "MYR",
    "CIT",
    "CLA",
    "CHL",
    "BCL",
    "BCB",
}

allowed_ions = set(ION_ELEMENTS)

config = StartConfig()
db = DatabaseMethods()
ref_folder = config.ref_folder
exp_table = config.get_reference_table_name()


def add_column_if_not_exists(conn, table, column, coltype):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        conn.commit()


def count_residues_by_resname(cif_path):
    """Count Biopython residue objects per uppercased _atom_site auth label_comp_id equivalent."""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("m", cif_path)
    except Exception:
        return Counter()
    ctr = Counter()
    for model in structure:
        for chain in model:
            for res in chain:
                ctr[res.get_resname().strip().upper()] += 1
    return ctr


def nonpoly_comp_ids(pdb_info):
    non_poly_ids = pdb_info.get("_pdbx_entity_nonpoly.comp_id", [])
    if isinstance(non_poly_ids, str):
        return [non_poly_ids]
    return list(non_poly_ids)


def build_ion_ligand_rows(non_poly_ids, resname_counts):
    """
    Returns (ions_list, ligands_list) of per-row dicts with keys ions/ligands, n, af3_supported.
    """
    ion_af3 = {}
    ligand_af3 = {}

    for raw_id in non_poly_ids:
        comp_id = raw_id.strip()
        if not comp_id or comp_id == "HOH":
            continue
        cu = comp_id.upper()

        if cu == "SO4":
            ion_af3["SO4"] = False
            continue

        if cu in allowed_ions:
            ion_af3[cu] = True
        elif cu in allowed_ligands:
            ligand_af3[cu] = True
        elif len(comp_id) <= 2:
            ion_af3[cu] = False
        elif len(comp_id) == 3:
            ligand_af3[cu] = False

    def rows_from(af3_map, species_key):
        out = []
        for code in sorted(af3_map.keys()):
            n = int(resname_counts.get(code, 0))
            out.append(
                {
                    species_key: code,
                    "n": n,
                    "af3_supported": af3_map[code],
                }
            )
        return out

    return rows_from(ion_af3, "ions"), rows_from(ligand_af3, "ligands")


add_column_if_not_exists(db.connection, exp_table, "Ions", "TEXT")
add_column_if_not_exists(db.connection, exp_table, "Ligands", "TEXT")

query = f"SELECT PDBId FROM {exp_table}"
structures = db.execute_query(query)

for (pdb_id,) in structures:
    cif_file = os.path.join(ref_folder, f"{pdb_id}.cif")
    if not os.path.isfile(cif_file):  # case-sensitive file systems: pdb/ files are usually lower case
        cif_file = os.path.join(ref_folder, f"{pdb_id.lower()}.cif")
    if not os.path.isfile(cif_file):
        print(f"CIF file not found for {pdb_id}")
        continue

    try:
        pdb_info = MMCIF2Dict(cif_file)
        non_poly_ids = nonpoly_comp_ids(pdb_info)
    except Exception as e:
        print(f"Error parsing mmCIF dict {cif_file}: {e}")
        continue

    resname_counts = count_residues_by_resname(cif_file)
    ions, ligands = build_ion_ligand_rows(non_poly_ids, resname_counts)

    ions_str = str(ions) if ions else "[]"
    ligands_str = str(ligands) if ligands else "[]"

    try:
        db.connection.execute(
            f"UPDATE {exp_table} SET Ions = ?, Ligands = ? WHERE PDBId = ?",
            (ions_str, ligands_str, pdb_id),
        )
        db.connection.commit()
    except Exception as e:
        print(f"Error updating {pdb_id}: {e}")

db.close_connection()
print("Done updating Ions and Ligands columns.")
