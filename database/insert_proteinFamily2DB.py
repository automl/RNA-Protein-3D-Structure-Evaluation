import argparse
import os
import sqlite3
import json
import ast
from database.protein_family_request import get_protein_family_from_sequence
from database.protein_family_request import get_protein_families_per_sequences
from database.protein_family_request import extract_interpro_entries
from structures.protein import Protein
from database.startConfig import StartConfig

# Honor the configured DB (config.json db_path) so per-run isolation works; StartConfig falls back to
# <repo>/database/rbpDatabase.db when config sets no db_path.
db_path = StartConfig().get_database_path()

def get_pdb_ids(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT PDBId FROM {table_name}"
    cursor.execute(query)
    pdb_ids = cursor.fetchall()
    conn.close()
    return [pdb_id[0] for pdb_id in pdb_ids]

def get_uniprot_ids(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT UniprotId FROM {table_name}"
    cursor.execute(query)
    pdb_ids = cursor.fetchall()
    conn.close()
    return [pdb_id[0] for pdb_id in pdb_ids]

# def insert_family_into_db(pdb_ids):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#
#     def handle_list_or_value(item):
#         if isinstance(item, list):
#             if len(set(item)) == 1:
#                 return str(item[0])
#             # Return JSON-formatted string if elements are not identical
#             return json.dumps(item)
#         return str(item)
#
#     for pdb_id in pdb_ids:
#         protein = Protein.get_protein_from_db(pdb_id)
#         sequence = protein.get_protein_sequence()
#         if sequence.startswith("["):
#             sequence_list = ast.literal_eval(sequence)
#             sequence = max(sequence_list, key=len)
#         print(pdb_id, sequence)
#         protein_families = get_protein_family_from_sequence(sequence, pdb_id)
#         print("protein_families", protein_families)
#         if protein_families:
#             protein_familyH = [handle_list_or_value(protein_families[i]) for i in range(len(protein_families))]
#             protein_family = json.dumps(protein_familyH)
#         else:
#             protein_family = 'None'
#         if table_name == 'exp_protein_rna':
#             cursor.execute('''UPDATE exp_protein_rna
#                                   SET ProteinFamily = ?
#                                   WHERE PDBId = ?''', (protein_family, pdb_id))
#             print("success", protein_family, pdb_id)
#             conn.commit()
#         if table_name == 'exp_rna_rna':
#             cursor.execute('''UPDATE exp_rna_rna
#                                   SET ProteinFamily = ?
#                                   WHERE PDBId = ?''', (protein_family, pdb_id))
#             conn.commit()
#         if table_name == 'exp_protein_dna':
#             cursor.execute('''UPDATE exp_protein_dna
#                                   SET ProteinFamily = ?
#                                   WHERE PDBId = ?''', (protein_family, pdb_id))
#             conn.commit()
#         if table_name == 'exp_protein_rna_dna':
#             cursor.execute('''UPDATE exp_protein_rna_dna
#                                   SET ProteinFamily = ?
#                                   WHERE PDBId = ?''', (protein_family, pdb_id))
#             conn.commit()
#
#     conn.close()

def insert_family_into_db(pdb_ids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def handle_list_or_value(item):
        if isinstance(item, list):
            if len(set(item)) == 1:
                return str(item[0])
            # Return JSON-formatted string if elements are not identical
            return json.dumps(item)
        return str(item)

    for pdb_id in pdb_ids:
        protein = Protein.get_protein_from_db(pdb_id)
        sequence = protein.get_protein_sequence()
        if sequence.startswith("["):
            sequence_list = ast.literal_eval(sequence)
            sequence = max(sequence_list, key=len)
        print(pdb_id, sequence)
        protein_families = extract_interpro_entries(pdb_id)
        print("protein_families", protein_families)
        if protein_families:
            protein_familyH = [handle_list_or_value(protein_families[i]) for i in range(len(protein_families))]
            protein_family = json.dumps(protein_familyH)
        else:
            protein_family = 'None'
        if table_name == 'exp_protein_rna':
            cursor.execute('''UPDATE exp_protein_rna
                                  SET ProteinFamily = ?
                                  WHERE PDBId = ?''', (protein_family, pdb_id))
            print("success", protein_family, pdb_id)
            conn.commit()
        if table_name == 'exp_rna_rna':
            cursor.execute('''UPDATE exp_rna_rna
                                  SET ProteinFamily = ?
                                  WHERE PDBId = ?''', (protein_family, pdb_id))
            conn.commit()
        if table_name == 'exp_protein_dna':
            cursor.execute('''UPDATE exp_protein_dna
                                  SET ProteinFamily = ?
                                  WHERE PDBId = ?''', (protein_family, pdb_id))
            conn.commit()
        if table_name == 'exp_protein_rna_dna':
            cursor.execute('''UPDATE exp_protein_rna_dna
                                  SET ProteinFamily = ?
                                  WHERE PDBId = ?''', (protein_family, pdb_id))
            conn.commit()

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert or update ProteinFamily in exp_* tables.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite ProteinFamily even when it already has a value")
    args = parser.parse_args()

    config = StartConfig()
    table_name = config.ref_table
    if not table_name:
        print("Error: ref_table not set in StartConfig (check config.json).")
        raise SystemExit(1)

    pdb_ids = get_pdb_ids(table_name)
    uniprot_ids = get_uniprot_ids(table_name)
    print("pdb_ids", pdb_ids)
    print("uniprot_ids", uniprot_ids)
    # input_string = "'7OFW', '7VSJ', '7PMQ', '7VKL', '7S02', '7SOV', '7D8O', '7UJ1', '7TUV', '8H0S', '7F36', '7M50', '7PMM', '7ENI', '7OZQ', '7E8O', '7M2V', '7PDV', '7KI3', '7W9S', '7M3T', '7DWH', '7WNU', '7RZZ', '7ENR', '8IDF', '8T2B'"
    # pdb_ids = [x.strip().strip("'") for x in input_string.split(",")]
    print("uniprot_ids", uniprot_ids)
    # insert_family_into_db(uniprot_ids)
    def handle_list_or_value(value):
        """Handles JSON lists or single values."""
        if isinstance(value, list):
            return value
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value

    def parse_sequences(value):
        """Parse ProteinSequence from DB into list of sequences (one per chain)."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return []
        s = value.strip()
        if s.startswith("["):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return [s]
        return [s]

    def _strip_id(x):
        """Strip and remove surrounding single/double quotes from an id."""
        if x is None:
            return ""
        s = (x.strip() if isinstance(x, str) else str(x)).strip()
        for q in ("'", '"'):
            if len(s) >= 2 and s[0] == q and s[-1] == q:
                s = s[1:-1].strip()
        return s

    def uniprot_ids_list(value):
        """Parse UniprotId into a list of ids. Handles: already a list, JSON array '["P27194","O41353"]', comma-separated 'a,b,c', or single id 'Q8CDG1'."""
        if value is None:
            return []
        # DB driver may return a list already (e.g. from JSON column)
        if isinstance(value, list):
            out = [_strip_id(x) for x in value if x is not None and _strip_id(x)]
            return out
        s = value.strip() if isinstance(value, str) else str(value)
        if not s:
            return []
        # Try JSON array first (double-quoted: ["P27194","O41353"])
        if s.startswith("["):
            try:
                out = json.loads(s)
                if isinstance(out, list):
                    return [_strip_id(x) for x in out if x is not None and _strip_id(x)]
                return [s]
            except (ValueError, TypeError):
                pass
            # Fallback: strip brackets and split by comma (handles single-quoted Python repr)
            inner = s.strip("[]").strip()
            if inner:
                return [_strip_id(x) for x in inner.split(",") if _strip_id(x)]
            return []
        # Single value or comma-separated without brackets
        if "," in s:
            return [_strip_id(x) for x in s.split(",") if _strip_id(x)]
        return [s] if s else []

    def families_for_uniprot_id(uid, pdb_id, protein_sequence_raw):
        """Get protein family list for one Uniprot id. Returns a list (e.g. ["Fam1"] or [None])."""
        uid = (uid or "").strip()
        if not uid:
            return [None]
        try:
            entries = extract_interpro_entries(uid)
            if entries:
                return [e["FamilyName"] for e in entries]
        except Exception:
            pass
        if uid == pdb_id and protein_sequence_raw:
            try:
                sequences = parse_sequences(protein_sequence_raw)
                if sequences:
                    seq = max(sequences, key=len)
                    if seq:
                        fam = get_protein_family_from_sequence(seq, pdb_id)
                        if fam:
                            return fam
            except Exception:
                pass
        return [None]

    # Tables that have ProteinChainIDs and ProteinSequence: use nested list of families per chain
    PROTEIN_TABLES = ("exp_protein_rna", "exp_protein_dna", "exp_protein_rna_dna")

    db_file = db_path
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    if table_name in PROTEIN_TABLES:
        cursor.execute(
            f"SELECT PDBId, UniprotId, ProteinFamily, ProteinChainIDs, ProteinSequence FROM {table_name}"
        )
    else:
        cursor.execute(f"SELECT PDBId, UniprotId, ProteinFamily FROM {table_name}")
    rows = cursor.fetchall()

    for row in rows:
        if table_name in PROTEIN_TABLES:
            pdb_id, uniprot_id_raw, protein_family_value, protein_chain_ids_raw, protein_sequence_raw = row[0], row[1], row[2], row[3], row[4]
        else:
            pdb_id, uniprot_id_raw, protein_family_value = row[0], row[1], row[2]

        # Skip if ProteinFamily already has a value, unless --overwrite
        if not args.overwrite and protein_family_value and protein_family_value.strip():
            print(f"Skipping PDBId {pdb_id} because ProteinFamily already has a value {protein_family_value}.")
            continue

        # Nested list of families per chain: use only UniprotId format to get number of chains and fetch families per id
        if table_name in PROTEIN_TABLES:
            uniprot_ids = uniprot_ids_list(uniprot_id_raw)
            n_chains = len(uniprot_ids)
            if n_chains == 0:
                protein_family_json = json.dumps([])
            else:
                families_per_chain = []
                for uid in uniprot_ids:
                    families_per_chain.append(families_for_uniprot_id(uid, pdb_id, protein_sequence_raw))
                # Guarantee each entry is a list so we always get [[...],[...],...] never null or flat list
                families_per_chain = [x if isinstance(x, list) else [x] if x is not None else [None] for x in families_per_chain]
                if len(families_per_chain) != n_chains:
                    print(f"WARNING {pdb_id}: expected {n_chains} family lists, got {len(families_per_chain)}; uniprot_id_raw type={type(uniprot_id_raw)!r} len={len(repr(uniprot_id_raw))}")
                protein_family_json = json.dumps(families_per_chain)
            cursor.execute(
                f"UPDATE {table_name} SET ProteinFamily = ? WHERE PDBId = ?",
                (protein_family_json, pdb_id),
            )
            conn.commit()
            print("success (nested per chain)", pdb_id, "n_chains", n_chains, "uniprot_ids", uniprot_ids)
            continue

        # Parse UniprotId (could be JSON array or a single string)
        uniprot_ids = handle_list_or_value(uniprot_id_raw)
        print("uniprot_ids", uniprot_ids)

        if isinstance(uniprot_ids, list): # CASE 1: more than 1 uniprot_ids
            # Check if any of the Uniprot IDs match the PDBId
            if pdb_id in uniprot_ids:
                # Case: At least one Uniprot ID equals PDBId
                if all(uid == pdb_id for uid in uniprot_ids): # CASE 1a: more than 1 uniprot_ids & all are pdb_id
                    protein = Protein.get_protein_from_db(pdb_id)
                    sequence = protein.get_protein_sequence()
                    if sequence.startswith("["):
                        sequence_list = ast.literal_eval(sequence)
                        sequence = max(sequence_list, key=len)
                    print(pdb_id, sequence)
                    protein_families = get_protein_family_from_sequence(sequence, pdb_id)
                    protein_familyH = [handle_list_or_value(protein_families[i]) for i in range(len(protein_families))]
                    protein_family = json.dumps(protein_familyH)
                    # Update the database
                    cursor.execute(f'''UPDATE {table_name}
                                    SET ProteinFamily = ?
                                    WHERE PDBId = ?''', (protein_family, pdb_id))
                    conn.commit()
                else: # CASE 1a: more than 1 uniprot_ids & not all are pdb_id
                    # Case: Not all Uniprot IDs match PDBId
                    # interpro_entries = extract_interpro_entries(uniprot_ids)
                    all_protein_families = []
                    # Store all protein_families_per_entry for comparison
                    protein_families_collection = []
                    for uid in uniprot_ids:
                        if uid != pdb_id:
                            # Extract InterPro entries for each Uniprot ID
                            protein_families_per_entry = extract_interpro_entries(uid)
                            protein_families_collection.append(protein_families_per_entry)
                    seen_families = []
                    unique_protein_families = []

                    # Iterate over all entries and keep only unique ones
                    for entry in all_protein_families:
                        if entry["FamilyName"] not in seen_families:
                            unique_protein_families.append(entry)
                            seen_families.append(entry["FamilyName"])  # Add to seen_families list

                    if not unique_protein_families:
                        protein = Protein.get_protein_from_db(pdb_id)
                        sequence = protein.get_protein_sequence()
                        if sequence.startswith("["):
                            sequence_list = ast.literal_eval(sequence)
                            sequence = max(sequence_list, key=len)
                        protein_family = get_protein_family_from_sequence(sequence, pdb_id)
                        if protein_family:
                            unique_protein_families = [handle_list_or_value(protein_family[i]) for i in
                                                range(len(protein_family))]
                        else:
                            unique_protein_families = None
                    # Convert unique_protein_families to JSON format
                    protein_family_json = json.dumps(unique_protein_families)
                    cursor.execute(f'''UPDATE {table_name}
                                       SET ProteinFamily = ?
                                       WHERE PDBId = ?''', (protein_family_json, pdb_id))
                    conn.commit()
            else:
                # Case: None of the Uniprot IDs match PDBId
                # Initialize an empty list to collect unique protein families
                all_protein_families = []
                # Store all protein_families_per_entry for comparison
                protein_families_collection = []
                for uid in uniprot_ids:
                    # Extract InterPro entries for each Uniprot ID
                    protein_families_per_entry = extract_interpro_entries(uid)
                    protein_families_collection.append(protein_families_per_entry)
                seen_families = []
                unique_protein_families = []

                # Iterate over all entries and keep only unique ones
                for entry in all_protein_families:
                    if entry["FamilyName"] not in seen_families:
                        unique_protein_families.append(entry)
                        seen_families.append(entry["FamilyName"])  # Add to seen_families list

                if not unique_protein_families:
                    protein = Protein.get_protein_from_db(pdb_id)
                    sequence = protein.get_protein_sequence()
                    if sequence.startswith("["):
                        sequence_list = ast.literal_eval(sequence)
                        sequence = max(sequence_list, key=len)
                    protein_family = get_protein_family_from_sequence(sequence, pdb_id)
                    if protein_family:
                        unique_protein_families = [handle_list_or_value(protein_family[i]) for i in
                                            range(len(protein_family))]
                    else:
                        unique_protein_families = None

                # Convert unique_protein_families to JSON format
                protein_family_json = json.dumps(unique_protein_families)
                # Update the database
                cursor.execute(f'''UPDATE {table_name}
                                   SET ProteinFamily = ?
                                   WHERE PDBId = ?''', (protein_family_json, pdb_id))
                conn.commit()

        else:
            # Case: Single Uniprot ID
            if pdb_id == uniprot_ids:
                protein = Protein.get_protein_from_db(pdb_id)
                sequence = protein.get_protein_sequence()
                if sequence.startswith("["):
                    sequence_list = ast.literal_eval(sequence)
                    sequence = max(sequence_list, key=len)
                print(pdb_id, sequence)
                protein_families = get_protein_family_from_sequence(sequence, pdb_id)
                if protein_families:
                    protein_familyH = [handle_list_or_value(protein_families[i]) for i in range(len(protein_families))]
                    protein_family = json.dumps(protein_familyH)
                else:
                    protein_family = "None"
                # Update the database
                cursor.execute(f'''UPDATE {table_name}
                                                      SET ProteinFamily = ?
                                                      WHERE PDBId = ?''', (protein_family, pdb_id))
                conn.commit()

            else:
                interpro_entries = extract_interpro_entries(uniprot_ids)
                if not interpro_entries:
                    protein = Protein.get_protein_from_db(pdb_id)
                    sequence = protein.get_protein_sequence()
                    if sequence.startswith("["):
                        sequence_list = ast.literal_eval(sequence)
                        sequence = max(sequence_list, key=len)
                    protein_family = get_protein_family_from_sequence(sequence, pdb_id)
                    if protein_family:
                        protein_families = [handle_list_or_value(protein_family[i]) for i in range(len(protein_family))]
                    else:
                        protein_families = None
                else:
                    protein_families = [entry['FamilyName'] for entry in interpro_entries]
                protein_family_json = json.dumps(protein_families)
                cursor.execute(f'''UPDATE {table_name}
                                   SET ProteinFamily = ?
                                   WHERE PDBId = ?''', (protein_family_json, pdb_id))
                conn.commit()

    rna_binding_domains = ["CSD", "Cold_shock", "RRM", "La", "PAZ", "Piwi", "KH", "RBD", "dsRBD", "DEAD", "DEAH", "PUM",
                           "SAM", "S1", "R3H", "PUA", "PUM", "THUMP", "YTH", "Zn_finger", "Znf", "GRP", "RNA_binding",
                           "disordered", "pentatricopeptide"]


    conn.close()
