import sqlite3
import pandas as pd
from database.startConfig import StartConfig
import json
import re

# Define modified residue mappings
modified_residues_nucleotide = {
    '6MZ': 'A', 'PSU': 'U', '5MC': 'C', 'OMC': 'C', '4OC': 'C', '5MU': 'U',
    'OMU': 'U', 'UR3': 'U', 'A2M': 'A', 'MA6': 'A', '2MG': 'G', 'OMG': 'G',
    '7MG': 'G', 'RSQ': 'G', '5CM': 'C', 'C34': 'C', '5HC': 'C', '6OG': 'G',
    '6MA': 'A', '1CC': 'C', '8OG': 'G', '5FC': 'C', '3DR': 'T'
}

modified_residues_protein = {
    'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'NEP': 'H', 'HIP': 'H', 'ALY': 'K', 
    'MLY': 'K', 'M3L': 'K', 'MLZ': 'K', '2MR': 'R', 'AGM': 'R', 'MCS': 'C', 
    'HYP': 'P', 'HY3': 'H', 'LYZ': 'K', 'AHB': 'A', 'P1L': 'P', 'SNN': 'S', 
    'SNC': 'C', 'TRF': 'W', 'KCR': 'K', 'CIR': 'R', 'YHA': 'Y'
}

def replace_modified_residues(sequence, is_protein=False):
    """Replace modified residues with their standard equivalents."""
    if is_protein:
        for mod, std in modified_residues_protein.items():
            sequence = sequence.replace(f"({mod})", std)
    else:
        for mod, std in modified_residues_nucleotide.items():
            sequence = sequence.replace(f"({mod})", std)
    return sequence

def normalize_sequence(sequence):
    """Remove any non-standard characters and convert to uppercase."""
    # First replace modified residues
    for mod, std in modified_residues_nucleotide.items():
        sequence = sequence.replace(f"({mod})", std)
    for mod, std in modified_residues_protein.items():
        sequence = sequence.replace(f"({mod})", std)
    
    # Remove any whitespace and convert to uppercase
    sequence = ''.join(sequence.split()).upper()
    # Remove any non-alphabetic characters except parentheses
    sequence = re.sub(r'[^A-Z()]', '', sequence)
    return sequence

def find_sequence_match(seq_to_match, sequences, chain_ids):
    """Find the best matching sequence and return its chain ID."""
    # Normalize the sequence to match
    seq_to_match = normalize_sequence(seq_to_match)
    
    # Handle both string and list inputs for sequences and chain_ids
    if isinstance(sequences, str):
        sequences = [sequences]
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    
    # Normalize all sequences
    normalized_seqs = [normalize_sequence(s) for s in sequences]
    
    # Find exact matches
    for seq, chain_id in zip(normalized_seqs, chain_ids):
        if seq == seq_to_match:
            return chain_id
    
    # If no exact match, try partial matches
    for seq, chain_id in zip(normalized_seqs, chain_ids):
        if seq_to_match in seq or seq in seq_to_match:
            return chain_id
    
    return None

def find_all_sequence_matches(seq_to_match, sequences, chain_ids):
    """Find all matching sequences and return their chain IDs."""
    # Normalize the sequence to match
    seq_to_match = normalize_sequence(seq_to_match)
    
    # Handle both string and list inputs for sequences and chain_ids
    if isinstance(sequences, str):
        sequences = [sequences]
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]
    
    # Normalize all sequences
    normalized_seqs = [normalize_sequence(s) for s in sequences]
    
    matching_chains = []
    
    # Find all exact matches
    for seq, chain_id in zip(normalized_seqs, chain_ids):
        if seq == seq_to_match:
            matching_chains.append(chain_id)
    
    # If we found exact matches, return them
    if matching_chains:
        return matching_chains
    
    # If no exact match, try partial matches
    for seq, chain_id in zip(normalized_seqs, chain_ids):
        if seq_to_match in seq or seq in seq_to_match:
            matching_chains.append(chain_id)
    
    return matching_chains

def parse_sequence_string(seq_str):
    """Parse sequence string from database into a list of sequences."""
    if not seq_str:
        return []
    # Remove quotes and split by commas if it's a list
    seq_str = seq_str.strip('[]')
    if ',' in seq_str:
        return [s.strip().strip("'\"") for s in seq_str.split(',')]
    return [seq_str.strip().strip("'\"")]

# Load the CSV file
import os
csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "msa_info_all_data.csv")
# csv_file = "data/2May2025_synthetic/combined_databases/msa_info_nonx_data_af3_preds_for_iris.csv"
df = pd.read_csv(csv_file, dtype={"has_msa": str}, encoding="utf-8", delimiter=",")
# print("df.columns", df.columns) IF TROUBLE WITH DELIMITER

# Ensure 'mol_type' and 'pdb_id' are strings and handle case issues
df["mol_type"] = df["mol_type"].astype(str).str.lower()
df["pdb_id"] = df["pdb_id"].astype(str).str.lower()

# print(df["has_msa"].unique())

# Connect to the SQLite database
config = StartConfig()
db_path = config.get_database_path()
pred_table = config.pred_table
# db_path = "data/0707/compare_databases/spotrna_config23.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if this is an RNA-only table
is_rna_only = (pred_table == "pred_rna_rna")

# Add new columns if they don't exist
try:
    cursor.execute(f"ALTER TABLE {pred_table} ADD COLUMN RNA_msa_size TEXT")
    if not is_rna_only:
        cursor.execute(f"ALTER TABLE {pred_table} ADD COLUMN Protein_msa_size TEXT")
    print("Added new MSA size columns")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("MSA size columns already exist")
    else:
        raise e

# Get all database entries first
if is_rna_only:
    cursor.execute(f"""
        SELECT exp_db_id, RNASequence, RNAChainIDs 
        FROM {pred_table}
    """)
else:
    cursor.execute(f"""
        SELECT exp_db_id, ProteinSequence, RNASequence, ProteinChainIDs, RNAChainIDs 
        FROM {pred_table}
    """)
db_entries = cursor.fetchall()

# Process each database entry
for db_entry in db_entries:
    if is_rna_only:
        exp_db_id, rna_sequences, rna_chain_ids = db_entry
    else:
        exp_db_id, protein_sequences, rna_sequences, protein_chain_ids, rna_chain_ids = db_entry
    
    pdb_id = exp_db_id.lower()
    
    # Parse sequences and chain IDs
    if not is_rna_only:
        protein_sequences = parse_sequence_string(protein_sequences)
        protein_chain_ids = parse_sequence_string(protein_chain_ids)
    
    rna_sequences = parse_sequence_string(rna_sequences)
    rna_chain_ids = parse_sequence_string(rna_chain_ids)
    
    # Initialize MSA size dictionaries
    protein_msa_sizes = {}
    rna_msa_sizes = {}
    
    # Get rows for this PDB ID from CSV
    pdb_rows = df[df["pdb_id"] == pdb_id]
    
    # Process each row from CSV
    for _, row in pdb_rows.iterrows():
        sequence = row["sequence"]
        msa_length = row["msa_length"]
        
        # Skip invalid entries
        if pd.isna(sequence) or pd.isna(msa_length):
            continue
        
        matched = False
        
        if not is_rna_only:
            # Try to match with protein sequences - find ALL matching chains
            protein_chains = find_all_sequence_matches(sequence, protein_sequences, protein_chain_ids)
            if protein_chains:
                # Add all matching chains with the same MSA size
                for protein_chain in protein_chains:
                    protein_msa_sizes[protein_chain] = int(msa_length)
                matched = True
        
        if not matched:
            # Try to match with RNA sequences - find ALL matching chains
            rna_chains = find_all_sequence_matches(sequence, rna_sequences, rna_chain_ids)
            if rna_chains:
                # Add all matching chains with the same MSA size
                for rna_chain in rna_chains:
                    rna_msa_sizes[rna_chain] = int(msa_length)
                matched = True
            
        if not matched:
            print(f"Warning: Could not match sequence for {pdb_id}: {sequence[:50]}...")

    # Convert MSA sizes to JSON strings
    rna_msa_size_str = json.dumps(rna_msa_sizes)

    # Create binary MSA lists
    rna_msa = [1 if size > 1 else 0 for size in rna_msa_sizes.values()]

    # Convert binary lists to string format
    rna_msa_str = str(rna_msa)

    # Update the database
    if is_rna_only:
        cursor.execute(f"""
            UPDATE {pred_table}
            SET af3_rna_MSA = ?,
                RNA_msa_size = ?
            WHERE exp_db_id = ?
        """, (rna_msa_str, rna_msa_size_str, exp_db_id))
        
        print(f"\nPDB ID: {pdb_id} (EXP_DB_ID: {exp_db_id})")
        print(f"RNA MSA sizes: {rna_msa_sizes}")
        print(f"RNA MSA: {rna_msa}")
    else:
        protein_msa_size_str = json.dumps(protein_msa_sizes)
        protein_msa = [1 if size > 1 else 0 for size in protein_msa_sizes.values()]
        protein_msa_str = str(protein_msa)
        
        cursor.execute(f"""
            UPDATE {pred_table}
            SET af3_protein_MSA = ?, 
                af3_rna_MSA = ?,
                Protein_msa_size = ?,
                RNA_msa_size = ?
            WHERE exp_db_id = ?
        """, (protein_msa_str, rna_msa_str, protein_msa_size_str, rna_msa_size_str, exp_db_id))

        print(f"\nPDB ID: {pdb_id} (EXP_DB_ID: {exp_db_id})")
        print(f"Protein MSA sizes: {protein_msa_sizes}")
        print(f"Protein MSA: {protein_msa}")
        print(f"RNA MSA sizes: {rna_msa_sizes}")
        print(f"RNA MSA: {rna_msa}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Database updated successfully!")
