"""
Populate Ion_Location and Ligand_Location on the experimental reference table using mmCIF
from StartConfig ref (pdb) folder.

Ions: RNA coordination (phosphate / ribose / base); chelating vs non-chelating uses an 8 Å
RNA proximity rule, then protein vs nucleotide nearest heavy-atom distances when farther.

Ligands: same *routing* as ions (RNA ``min_distance`` vs a threshold, then nearest protein vs nucleotide
heavy atoms), but JSON keys are ``bound`` / ``non_binding`` and the RNA-distance threshold defaults to
**5 Å** (ions remain **8 Å**; override with ``--ligand-rna-site-threshold``). Interaction modes come from
atom pairs within ``--ligand-cutoff``. Populate the Ligands column via add_ions_ligands_to_db.

Residue typing order: RNA (A,U,G,C) before ions; ligand resnames from DB must not duplicate RNA codes.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sqlite3
from collections import defaultdict

from Bio.PDB import MMCIFParser

from database.startConfig import StartConfig
from database.databaseMethods import DatabaseMethods

# --- Ions: shared with add_ions_ligands_to_db (allowed_ions). Ligand geometry uses exp table Ligands column. ---
ION_ELEMENTS = {
    "NA",
    "MG",
    "K",
    "CA",
    "ZN",
    "MN",
    "FE",
    "CL",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "AG",
    "SR",
    "BA",
    "RB",
    "CS",
    "LI",
    "BR",
    "I",
    "AL",
    "CR",
    "MO",
    "W",
    "V",
    "PB",
    "SO4",
}

RNA_RESNAMES = {"A", "U", "G", "C"}

DNA_RESNAMES = {"DA", "DT", "DC", "DG", "DI", "DN"}

NUCLEOTIDE_RESNAMES = RNA_RESNAMES | DNA_RESNAMES

STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
    "SEC",
    "PYL",
    "ASX",
    "GLX",
    "UNK",
}

PHOSPHATE_ATOMS = {"OP1", "OP2", "O1P", "O2P"}
RIBOSE_ATOMS = {"O2'", "O3'", "O4'"}
BASE_ATOMS = {"N1", "N3", "N7", "O2", "O4", "O6"}

RELEVANT_ATOMS = PHOSPHATE_ATOMS | RIBOSE_ATOMS | BASE_ATOMS


def rna_coordination_atom_region(atom_name: str) -> str:
    """Coarse type for a ``RELEVANT_ATOMS`` RNA atom name (phosphate / ribose / base)."""
    an = atom_name.strip()
    if an in PHOSPHATE_ATOMS:
        return "phosphate"
    if an in RIBOSE_ATOMS:
        return "ribose"
    return "base"


# Ligand atoms that usually carry negative charge / backbone-like contacts
LIGAND_PHOSPHATE_HINT = {"OP1", "OP2", "O1P", "O2P", "O3A", "O3B", "O5'", "O3'", "PA", "PB", "PG", "P", "S1G", "O1G", "O2G", "O3G"}

DEFAULT_ION_CUTOFF = 3.0
# If closest RNA coordination-site distance is below this (Å), ion is always "chelating".
# If larger, compare ion→nearest protein heavy atom vs ion→nearest nucleotide heavy atom.
DEFAULT_ION_CHELATING_DIST_THRESHOLD = 8.0
# Ligand_Location only: RNA coordination distance below this → ``bound``; else protein vs nucleotide.
DEFAULT_LIGAND_RNA_SITE_THRESHOLD = 5.0
DEFAULT_LIGAND_CUTOFF = 4.0
DEFAULT_STACKING_CUTOFF = 5.0

def ligand_resnames_from_db_column(ligands_cell) -> set[str]:
    """
    Parse Ligands column from add_ions_ligands_to_db.

    Supports the current format (list of dicts with ``ligands`` key) and legacy list/tuple rows
    ``[..., comp_id, ...]`` (same idea as ``plotIonsLigands._ion_ligand_entry_code``).
    """
    if ligands_cell is None:
        return set()
    s = ligands_cell if isinstance(ligands_cell, str) else str(ligands_cell)
    if not s.strip():
        return set()
    try:
        parsed = ast.literal_eval(s.strip())
    except (ValueError, SyntaxError, TypeError):
        return set()
    if not isinstance(parsed, list):
        return set()
    out = set()
    for item in parsed:
        if isinstance(item, dict):
            name = item.get("ligands") or item.get("Ligands")
            if isinstance(name, str) and name.strip():
                out.add(name.strip().upper())
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name = item[1]
            if isinstance(name, str) and name.strip():
                out.add(name.strip().upper())
    return out


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def min_ion_heavy_atom_distance_to_residue(ion_coord, residue) -> float:
    """Minimum distance from ``ion_coord`` to any non-hydrogen atom in ``residue``."""
    best = float("inf")
    for atom in residue:
        el = (atom.element or "").strip().upper()
        if el == "H" or el == "D":
            continue
        d = distance(atom.get_coord(), ion_coord)
        if d < best:
            best = d
    return best


def min_ion_heavy_distance_to_residues(ion_coord, residues) -> float:
    best = float("inf")
    for res in residues:
        d = min_ion_heavy_atom_distance_to_residue(ion_coord, res)
        if d < best:
            best = d
    return best


def min_heavy_atom_pair_distance_between_residues(res_a, res_b) -> float:
    """Minimum heavy-atom distance between two Bio.PDB residues."""
    best = float("inf")
    for a in res_a:
        el = (a.element or "").strip().upper()
        if el in ("H", "D"):
            continue
        for b in res_b:
            elb = (b.element or "").strip().upper()
            if elb in ("H", "D"):
                continue
            d = distance(a.get_coord(), b.get_coord())
            if d < best:
                best = d
    return best


def min_ligand_to_residue_list_distance(lig_residue, other_residues) -> float:
    """Min heavy-atom distance from ``lig_residue`` to any residue in ``other_residues``."""
    best = float("inf")
    for other in other_residues:
        d = min_heavy_atom_pair_distance_between_residues(lig_residue, other)
        if d < best:
            best = d
    return best


def get_residue_id(residue):
    chain_id = residue.get_parent().id
    _, res_seq, icode = residue.get_id()
    icode = (icode or "").strip()
    if icode:
        res_id = f"{res_seq}{icode}"
    else:
        res_id = int(res_seq) if isinstance(res_seq, int) else res_seq
    return chain_id, res_id


def is_ligand_stacking_candidate(atom):
    """Rough π-stacking heuristic: C/N on ligand, excluding obvious phosphate/P centers."""
    el = (atom.element or "").strip().upper()
    if el not in {"C", "N"}:
        return False
    name = atom.get_name().strip().upper()
    if name in {"P", "PA", "PB", "PG"} or name.startswith("OP"):
        return False
    if name.startswith("O") and len(name) <= 4:
        return False
    return True


def classify_ions(
    structure,
    cutoff=DEFAULT_ION_CUTOFF,
    chelating_dist_threshold=DEFAULT_ION_CHELATING_DIST_THRESHOLD,
):
    """
    Classify ion–RNA proximity per ion instance in the structure.

    * If the closest RNA coordination-site distance (``min_distance``) is below
      ``chelating_dist_threshold`` (default 8 Å), the ion is **chelating** (RNA-centric).
    * If that distance is **≥** the threshold, compare the ion's distance to the nearest
      protein heavy atom vs the nearest nucleotide (RNA/DNA) heavy atom: if the protein
      is closer, **non_chelating**; otherwise **chelating**.

    Each entry's ``atom_types`` always includes the region (phosphate / ribose / base) of the
    **nearest** RELEVANT RNA atom (the one that sets ``min_distance``), plus any additional
    regions from atoms within ``cutoff`` (default 3 Å).

    Returns:
        dict: ion_name -> {"chelating": [...], "non_chelating": [...]}
    """
    rna_residues = []
    nucleotide_residues = []
    protein_residues = []
    ions = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()

                if resname in RNA_RESNAMES:
                    rna_residues.append(residue)
                if resname in NUCLEOTIDE_RESNAMES:
                    nucleotide_residues.append(residue)
                if resname in STANDARD_AMINO_ACIDS:
                    protein_residues.append(residue)
                if resname in ION_ELEMENTS:
                    ions.append(residue)

    result = defaultdict(lambda: {"chelating": [], "non_chelating": []})

    for ion in ions:
        ion_name = ion.get_resname().strip().upper()
        ion_atoms = list(ion.get_atoms())
        if not ion_atoms:
            continue
        ion_atom = ion_atoms[0]
        ion_coord = ion_atom.get_coord()

        best_dist = float("inf")
        best_res = None
        best_atom_name: str | None = None
        coordinating_atoms = []

        for rna_res in rna_residues:
            for atom in rna_res:
                atom_name = atom.get_name().strip()

                if atom_name not in RELEVANT_ATOMS:
                    continue

                d = distance(atom.get_coord(), ion_coord)
                if d < best_dist:
                    best_dist = d
                    best_res = rna_res
                    best_atom_name = atom_name

                if d <= cutoff:
                    coordinating_atoms.append((rna_res, atom_name, d))

        if best_res is None or not best_atom_name:
            continue

        chain_id, res_id = get_residue_id(best_res)
        n_contacts = len(coordinating_atoms)

        atom_types = set()
        for _, aname, _ in coordinating_atoms:
            atom_types.add(rna_coordination_atom_region(aname))
        # Nearest RELEVANT atom (defines min_distance) always contributes a type, even when
        # no atoms fall within ``cutoff`` (e.g. 3 Å) — avoids empty atom_types for 3–8 Å shells.
        atom_types.add(rna_coordination_atom_region(best_atom_name))

        entry = {
            "rna_chain": chain_id,
            "rna_res": res_id,
            "min_distance": round(best_dist, 2),
            "n_contacts": n_contacts,
            "atom_types": sorted(atom_types),
            "multidentate": n_contacts >= 2,
        }

        if best_dist < chelating_dist_threshold:
            result[ion_name]["chelating"].append(entry)
        else:
            d_protein = min_ion_heavy_distance_to_residues(ion_coord, protein_residues)
            d_nucleotide = min_ion_heavy_distance_to_residues(
                ion_coord, nucleotide_residues
            )
            if d_protein < d_nucleotide:
                result[ion_name]["non_chelating"].append(entry)
            else:
                result[ion_name]["chelating"].append(entry)

    return dict(result)


def classify_ligands(
    structure,
    ligand_resnames: set[str],
    ligand_cutoff=DEFAULT_LIGAND_CUTOFF,
    stacking_cutoff=DEFAULT_STACKING_CUTOFF,
    rna_site_threshold=DEFAULT_LIGAND_RNA_SITE_THRESHOLD,
):
    """
    Per ligand instance: RNA coordination geometry, then the same distance routing as ions, but buckets
    are named ``bound`` / ``non_binding`` and ``rna_site_threshold`` defaults to 5 Å (not 8 Å).

    ligand_resnames: component ids from exp table Ligands column (mmCIF resnames).
    """
    if not ligand_resnames:
        return {}
    rna_residues = []
    nucleotide_residues = []
    protein_residues = []
    ligands_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()

                if resname in RNA_RESNAMES:
                    rna_residues.append(residue)
                if resname in NUCLEOTIDE_RESNAMES:
                    nucleotide_residues.append(residue)
                if resname in STANDARD_AMINO_ACIDS:
                    protein_residues.append(residue)
                if resname in ION_ELEMENTS:
                    continue
                if resname in ligand_resnames:
                    ligands_list.append(residue)

    result = defaultdict(lambda: {"bound": [], "non_binding": []})

    for lig in ligands_list:
        lig_name = lig.get_resname().strip().upper()
        lig_atoms = list(lig.get_atoms())
        if not lig_atoms or not rna_residues:
            continue

        pairs = []
        best_dist = float("inf")
        best_res = None

        for rna_res in rna_residues:
            for rna_atom in rna_res:
                rna_aname = rna_atom.get_name().strip()
                if rna_aname not in RELEVANT_ATOMS:
                    continue
                rna_coord = rna_atom.get_coord()
                for lig_atom in lig_atoms:
                    d = distance(rna_coord, lig_atom.get_coord())
                    if d < best_dist:
                        best_dist = d
                        best_res = rna_res
                    if d <= ligand_cutoff:
                        pairs.append((rna_res, rna_aname, lig_atom, d))

        if best_res is None:
            continue

        primary_chain, primary_res = get_residue_id(best_res)
        n_atom_pairs = len(pairs)
        rna_res_keys = {get_residue_id(r[0]) for r in pairs}
        n_rna_residues = len(rna_res_keys)

        modes = set()
        for _rna_res, rna_aname, lig_atom, _d in pairs:
            if rna_aname in PHOSPHATE_ATOMS:
                modes.add("phosphate")
            elif rna_aname in RIBOSE_ATOMS:
                modes.add("ribose")
            elif rna_aname in BASE_ATOMS:
                modes.add("base")
            ln = lig_atom.get_name().strip().upper()
            if ln in LIGAND_PHOSPHATE_HINT:
                modes.add("phosphate")

        stacking = False
        for rna_res in rna_residues:
            for rna_atom in rna_res:
                if rna_atom.get_name().strip() not in BASE_ATOMS:
                    continue
                rc = rna_atom.get_coord()
                for lig_atom in lig_atoms:
                    if not is_ligand_stacking_candidate(lig_atom):
                        continue
                    if distance(rc, lig_atom.get_coord()) <= stacking_cutoff:
                        stacking = True
                        break
                if stacking:
                    break
            if stacking:
                break
        if stacking:
            modes.add("stacking")

        full_entry = {
            "rna_chain": primary_chain,
            "rna_res": primary_res,
            "min_distance": round(best_dist, 2),
            "n_atom_pairs": n_atom_pairs,
            "n_rna_residues": n_rna_residues,
            "interaction_modes": sorted(modes),
        }

        slim_entry = {
            "rna_chain": primary_chain,
            "rna_res": primary_res,
            "min_distance": round(best_dist, 2),
        }

        if best_dist < rna_site_threshold:
            result[lig_name]["bound"].append(full_entry)
        else:
            d_protein = min_ligand_to_residue_list_distance(lig, protein_residues)
            d_nucleotide = min_ligand_to_residue_list_distance(lig, nucleotide_residues)
            if d_protein < d_nucleotide:
                result[lig_name]["non_binding"].append(slim_entry)
            else:
                result[lig_name]["bound"].append(full_entry)

    return dict(result)


def compact_ion_location_for_db(raw):
    """
    Shrink Ion_Location JSON: omit empty chelating/non_chelating keys; non_chelating
    entries only keep chain_id, min_distance, res_id.
    """
    out = {}
    for ion_name, buckets in raw.items():
        compact = {}
        chel = buckets.get("chelating") or []
        nonchel = buckets.get("non_chelating") or []
        if chel:
            compact["chelating"] = chel
        if nonchel:
            compact["non_chelating"] = []
            for e in nonchel:
                row = {
                    "rna_chain": e["rna_chain"],
                    "min_distance": e["min_distance"],
                    "rna_res": e["rna_res"],
                }
                at = e.get("atom_types")
                if at:
                    row["atom_types"] = at
                compact["non_chelating"].append(row)
        if compact:
            out[ion_name] = compact
    return out


def compact_ligand_location_for_db(raw):
    """
    Shrink Ligand_Location JSON: ``bound`` keeps full entries; ``non_binding`` slim fields.
    Legacy ligand keys ``chelating`` / ``non_chelating`` (and ``weakly_bound``) are folded in or
    passed through when re-saving old rows.
    """
    out = {}
    for lig_name, buckets in raw.items():
        if not isinstance(buckets, dict):
            continue
        compact = {}
        bnd = buckets.get("bound") or buckets.get("chelating") or []
        nb = buckets.get("non_binding") or buckets.get("non_chelating") or []
        if bnd:
            compact["bound"] = bnd
        if nb:
            compact["non_binding"] = [
                {
                    "rna_chain": e["rna_chain"],
                    "min_distance": e["min_distance"],
                    "rna_res": e["rna_res"],
                }
                for e in nb
            ]
        for key in ("weakly_bound",):
            items = buckets.get(key) or []
            if items:
                compact[key] = items
        if compact:
            out[lig_name] = compact
    return out


def add_column_if_not_exists(conn, table, column, coltype):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        conn.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ion-cutoff",
        type=float,
        default=DEFAULT_ION_CUTOFF,
        help=f"Ion–RNA coordination cutoff (Å), default {DEFAULT_ION_CUTOFF}",
    )
    ap.add_argument(
        "--chelating-dist-threshold",
        type=float,
        default=DEFAULT_ION_CHELATING_DIST_THRESHOLD,
        help=(
            "Ions (Ion_Location): if closest RNA coordination distance is below this (Å), ion is "
            "chelating; else compare nearest protein vs nucleotide heavy-atom distance "
            f"(default {DEFAULT_ION_CHELATING_DIST_THRESHOLD})"
        ),
    )
    ap.add_argument(
        "--ligand-rna-site-threshold",
        type=float,
        default=DEFAULT_LIGAND_RNA_SITE_THRESHOLD,
        help=(
            "Ligands (Ligand_Location) only: same routing as ions but JSON uses bound/non_binding; "
            f"RNA distance threshold in Å (default {DEFAULT_LIGAND_RNA_SITE_THRESHOLD})"
        ),
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=f"Deprecated alias for --ion-cutoff (default {DEFAULT_ION_CUTOFF})",
    )
    ap.add_argument(
        "--ligand-cutoff",
        type=float,
        default=DEFAULT_LIGAND_CUTOFF,
        help=f"Ligand–RNA atom pair cutoff (Å), default {DEFAULT_LIGAND_CUTOFF}",
    )
    ap.add_argument(
        "--stacking-cutoff",
        type=float,
        default=DEFAULT_STACKING_CUTOFF,
        help=f"Base–ligand stacking heuristic cutoff (Å), default {DEFAULT_STACKING_CUTOFF}",
    )
    args = ap.parse_args()
    ion_cutoff = args.ion_cutoff if args.cutoff is None else args.cutoff

    config = StartConfig()
    db = DatabaseMethods()
    ref_folder = config.ref_folder
    exp_table = config.get_reference_table_name()

    add_column_if_not_exists(db.connection, exp_table, "Ion_Location", "TEXT")
    add_column_if_not_exists(db.connection, exp_table, "Ligand_Location", "TEXT")
    add_column_if_not_exists(db.connection, exp_table, "Ligands", "TEXT")

    query = f"SELECT PDBId, Ligands FROM {exp_table}"
    structures = db.execute_query(query)

    for row in structures:
        if len(row) >= 2:
            pdb_id, ligands_cell = row[0], row[1]
        else:
            pdb_id, ligands_cell = row[0], None
        ligand_ids = ligand_resnames_from_db_column(ligands_cell)
        cif_file = os.path.join(ref_folder, f"{pdb_id}.cif")
        if not os.path.isfile(cif_file):  # case-sensitive file systems: pdb/ files are usually lower case
            cif_file = os.path.join(ref_folder, f"{pdb_id.lower()}.cif")
        if not os.path.isfile(cif_file):
            print(f"CIF file not found for {pdb_id}")
            continue
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("model", cif_file)
            ion_dict = compact_ion_location_for_db(
                classify_ions(
                    structure,
                    cutoff=ion_cutoff,
                    chelating_dist_threshold=args.chelating_dist_threshold,
                )
            )
            lig_dict = compact_ligand_location_for_db(
                classify_ligands(
                    structure,
                    ligand_ids,
                    ligand_cutoff=args.ligand_cutoff,
                    stacking_cutoff=args.stacking_cutoff,
                    rna_site_threshold=args.ligand_rna_site_threshold,
                )
            )
            ion_json = json.dumps(ion_dict, sort_keys=True)
            lig_json = json.dumps(lig_dict, sort_keys=True)
        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue

        try:
            db.connection.execute(
                f"UPDATE {exp_table} SET Ion_Location = ?, Ligand_Location = ? WHERE PDBId = ?",
                (ion_json, lig_json, pdb_id),
            )
            db.connection.commit()
        except sqlite3.Error as e:
            print(f"Error updating {pdb_id}: {e}")

    db.close_connection()
    print("Done updating Ion_Location and Ligand_Location columns.")


if __name__ == "__main__":
    main()
