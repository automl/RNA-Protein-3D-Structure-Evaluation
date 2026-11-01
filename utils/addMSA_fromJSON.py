#!/usr/bin/env python3
"""
Extract MSA size information from AlphaFold3 data JSON files and update database directly.

This script combines the functionality of get_msa_size_info.py and csv_addMSA.py:
- Reads *_data.json files from a folder
- Extracts MSA information directly from JSON files
- Matches sequences with database entries
- Updates the database directly without creating a CSV file

Usage:
    python addMSA_fromJSON.py /path/to/folder/with/json/files
"""

import sqlite3
import pandas as pd
from pathlib import Path
import argparse
import json
import re
import os
from tqdm import tqdm
from database.startConfig import StartConfig

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

def parse_sequence_string(seq_str):
    """Parse sequence string from database into a list of sequences."""
    if not seq_str:
        return []
    # Remove quotes and split by commas if it's a list
    seq_str = seq_str.strip('[]')
    if ',' in seq_str:
        return [s.strip().strip("'\"") for s in seq_str.split(',')]
    return [seq_str.strip().strip("'\"")]

def extract_pdb_id_from_filename(filename):
    """Extract PDB ID from filename (first 4 characters, converted to uppercase)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    if len(base) >= 4:
        pdb_id = base[:4].upper()
        if re.match(r'^[A-Z0-9]{4}$', pdb_id):
            return pdb_id
    return None

def get_msa_info_from_json(json_path):
    """
    Extract MSA information from a single JSON file.
    
    Returns:
        List of dictionaries with sequence, chain_id, mol_type, msa_length
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        name = data.get('name', '')
        pdb_id = name[:4].upper() if len(name) >= 4 else None
        
        samples = []
        sequences = data.get('sequences', [])
        
        for s in sequences:
            for mol_type, mol_data in s.items():
                if not isinstance(mol_data, dict):
                    continue
                
                seq = mol_data.get('sequence', '')
                chain_id = mol_data.get('id', '')
                unpaired_msa = mol_data.get('unpairedMsa', '')
                
                # Calculate MSA length
                if unpaired_msa:
                    msa_length = int((len(str(unpaired_msa).split('\n')) - 1) / 2)
                else:
                    msa_length = 1  # No MSA
                
                sample = {
                    'sequence': seq,
                    'chain_id': chain_id,
                    'mol_type': mol_type,
                    'pdb_id': pdb_id,
                    'msa_length': msa_length,
                }
                samples.append(sample)
        
        return samples
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Extract MSA size information from AlphaFold3 JSON files and update database'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing *_data.json files'
    )
    
    args = parser.parse_args()
    
    json_dir = Path(args.folder)
    
    if not json_dir.exists():
        print(f"Error: Folder does not exist: {json_dir}")
        return
    
    if not json_dir.is_dir():
        print(f"Error: Path is not a directory: {json_dir}")
        return
    
    # Connect to the SQLite database
    config = StartConfig()
    db_path = config.get_database_path()
    pred_table = config.pred_table
    
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"Processing JSON files from: {json_dir.resolve()}")
    print()
    
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
    
    # Get all database entries
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
    
    # Create a dictionary mapping pdb_id (lowercase) to database entries
    db_dict = {}
    for db_entry in db_entries:
        exp_db_id = db_entry[0]
        pdb_id = exp_db_id.lower()
        db_dict[pdb_id] = db_entry
    
    print(f"Found {len(db_dict)} entries in database")
    print()
    
    # Find all JSON files
    json_files = list(json_dir.rglob('*_data.json'))
    print(f"Found {len(json_files)} JSON files to process")
    print()
    
    # Process each JSON file
    successful = 0
    skipped = 0
    failed = 0
    not_found = 0
    
    pbar = tqdm(total=len(json_files), desc="Processing JSON files")
    
    for json_file in json_files:
        pbar.update(1)
        
        # Extract PDB ID from filename
        base = json_file.stem  # filename without extension
        if len(base) >= 4:
            pdb_id = base[:4].upper()
        else:
            skipped += 1
            continue
        
        pdb_id_lower = pdb_id.lower()
        
        # Check if this PDB ID exists in database
        if pdb_id_lower not in db_dict:
            not_found += 1
            continue
        
        # Get database entry
        db_entry = db_dict[pdb_id_lower]
        
        if is_rna_only:
            exp_db_id, rna_sequences, rna_chain_ids = db_entry
        else:
            exp_db_id, protein_sequences, rna_sequences, protein_chain_ids, rna_chain_ids = db_entry
        
        # Parse sequences and chain IDs from database
        if not is_rna_only:
            protein_sequences = parse_sequence_string(protein_sequences)
            protein_chain_ids = parse_sequence_string(protein_chain_ids)
        
        rna_sequences = parse_sequence_string(rna_sequences)
        rna_chain_ids = parse_sequence_string(rna_chain_ids)
        
        # Extract MSA info from JSON file
        msa_samples = get_msa_info_from_json(json_file)
        
        if not msa_samples:
            failed += 1
            continue
        
        # Initialize MSA size dictionaries
        protein_msa_sizes = {}
        rna_msa_sizes = {}
        
        # Process each MSA sample from JSON
        for sample in msa_samples:
            sequence = sample['sequence']
            chain_id = sample['chain_id']
            mol_type = sample['mol_type']
            msa_length = sample['msa_length']
            
            if not sequence or msa_length < 1:
                continue
            
            # Match with database sequences
            if not is_rna_only and mol_type.lower() in ['protein', 'p']:
                # Try to match with protein sequences
                protein_chain = find_sequence_match(sequence, protein_sequences, protein_chain_ids)
                if protein_chain:
                    protein_msa_sizes[protein_chain] = msa_length
                    continue
            
            # Try to match with RNA sequences
            if mol_type.lower() in ['rna', 'r']:
                rna_chain = find_sequence_match(sequence, rna_sequences, rna_chain_ids)
                if rna_chain:
                    rna_msa_sizes[rna_chain] = msa_length
        
        # Convert MSA sizes to JSON strings
        rna_msa_size_str = json.dumps(rna_msa_sizes)
        
        # Create binary MSA lists
        rna_msa = [1 if size > 1 else 0 for size in rna_msa_sizes.values()]
        rna_msa_str = str(rna_msa)
        
        # Update the database
        try:
            if is_rna_only:
                cursor.execute(f"""
                    UPDATE {pred_table}
                    SET af3_rna_MSA = ?,
                        RNA_msa_size = ?
                    WHERE exp_db_id = ?
                """, (rna_msa_str, rna_msa_size_str, exp_db_id))
                
                successful += 1
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
                
                successful += 1
        except Exception as e:
            print(f"\nError updating {pdb_id}: {e}")
            failed += 1
    
    pbar.close()
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Not found in DB: {not_found}")
    print(f"{'='*60}")
    print("Database updated successfully!")


if __name__ == '__main__':
    main()

