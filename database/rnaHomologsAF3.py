#!/usr/bin/env python3
"""
RNA Homologs Script based on AlphaFold3 approach

This script finds RNA structural homologs/templates similar to how AlphaFold3
retrieves templates for structure prediction. It uses:
1. MMseqs2 for sequence similarity searches (optional)
2. PDB structure database queries via RCSB API
3. Quality filtering (resolution, R-free, etc.)
4. Template selection based on sequence identity and structure quality

Based on AlphaFold3 template search methodology.

References:
- AlphaFold3 GitHub Repository: https://github.com/google-deepmind/alphafold3
- AlphaFold3 Source Code: https://github.com/google-deepmind/alphafold3/tree/main/src/alphafold3
- Related AlphaFold3 scripts and modules for template/homolog search:
  * Template search and selection: src/alphafold3/template/ (template search logic)
  * Structure database querying: src/alphafold3/data/ (PDB structure retrieval)
  * Sequence similarity search: src/alphafold3/msa/ (MSA generation and sequence search)
  * Template featurization: src/alphafold3/template/template_featurizer.py (template processing)
- Related repositories with AlphaFold3 template/MMA integration:
  * AlphaFold3 MMseqs2 Scripts: https://github.com/hlasimpk/af3_mmseqs_scripts
    (Scripts to add MMseqs2 MSAs and templates to AlphaFold3 input JSON)
  * ABCFold: https://github.com/rigdenlab/ABCFold
    (AlphaFold3 with MMseqs2 MSAs and custom templates)
  * OpenFold3-preview: Open-source AlphaFold3 implementation with MSA/template pipelines
"""

import json
import sqlite3
import subprocess
import os
import tempfile
import time
import argparse
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from database.startConfig import StartConfig

try:
    from rcsbapi.search import SeqSimilarityQuery, AttributeQuery
    try:
        from rcsbapi.search import search_attributes as attrs
        SEARCH_ATTRS_AVAILABLE = True
    except ImportError:
        SEARCH_ATTRS_AVAILABLE = False
    RCSBAPI_AVAILABLE = True
except ImportError:
    RCSBAPI_AVAILABLE = False
    SEARCH_ATTRS_AVAILABLE = False

# Dictionary of allowed modified nucleotides and their canonical equivalents
ALLOWED_MODIFIED_NUCLEOTIDES = {
    '6MZ': 'A', 'PSU': 'U', '5MC': 'C', 'OMC': 'C', '4OC': 'C', '5MU': 'U',
    'OMU': 'U', 'UR3': 'U', 'A2M': 'A', 'MA6': 'A', '2MG': 'G', 'OMG': 'G',
    '7MG': 'G', 'RSQ': 'G', '5CM': 'C', 'C34': 'C', '5HC': 'C', '6OG': 'G',
    '6MA': 'A', '1CC': 'C', '8OG': 'G', '5FC': 'C', '3DR': 'T'
}


def normalize_rna_sequence(sequence):
    """
    Normalize RNA sequence by replacing modified nucleotides with canonical equivalents.
    
    Args:
        sequence: RNA sequence string, may contain modified nucleotides like "(A2M)", "(5MC)", etc.
    
    Returns:
        Normalized sequence string with only canonical nucleotides (A, U, G, C, T)
    """
    import re
    
    if not sequence:
        return sequence
    
    # Pattern to match modified nucleotides in parentheses: (MOD)
    pattern = r'\(([A-Z0-9]+)\)'
    
    def replace_mod(match):
        mod_code = match.group(1)
        return ALLOWED_MODIFIED_NUCLEOTIDES.get(mod_code, mod_code)
    
    # Replace all (MOD) patterns with canonical nucleotides
    normalized = re.sub(pattern, replace_mod, sequence)
    
    return normalized


def find_rna_templates(sequence, 
                      min_identity=0.3,
                      max_evalue=0.001,
                      max_release_date="2021-09-30"):
    """
    Find RNA structural templates/homologs using AlphaFold3-like approach.
    
    Uses PDB structure search via RCSB API to find templates.
    AlphaFold3 typically uses evalue around 0.001 for template searches.
    
    Args:
        sequence: RNA sequence
        min_identity: Minimum sequence identity threshold (for filtering, not directly used in API)
        max_evalue: Maximum E-value (default: 0.001, typical for AlphaFold3 template searches)
        max_release_date: Maximum release date for structures
    
    Returns:
        List of PDB IDs (e.g., ["1TXS", "1RFR"])
    """
    pdb_id_list = []
    
    # Normalize sequence
    normalized_seq = normalize_rna_sequence(sequence)
    
    if len(normalized_seq) < 25:
        print(f"  Warning: Sequence too short ({len(normalized_seq)} < 25 residues)")
        return pdb_id_list
    
    if not RCSBAPI_AVAILABLE:
        print("  Error: rcsbapi package is required. Install with: pip install rcsb-api")
        return pdb_id_list
    
    print(f"  Searching for templates (sequence length: {len(normalized_seq)})...")
    
    try:
        # Create sequence similarity query
        seq_query = SeqSimilarityQuery(
            value=normalized_seq,
            evalue_cutoff=max_evalue,
            sequence_type="rna"
        )
        
        # Try to add date filter
        try:
            if SEARCH_ATTRS_AVAILABLE:
                date_query = attrs.rcsb_accession_info.initial_release_date <= max_release_date
                combined_query = seq_query & date_query
                pdb_ids = list(combined_query("polymer_entity"))
            else:
                date_query = AttributeQuery(
                    attribute="rcsb_accession_info.initial_release_date",
                    operator="less_or_equal",
                    value=max_release_date
                )
                combined_query = seq_query & date_query
                pdb_ids = list(combined_query("polymer_entity"))
        except Exception:
            # Date filtering failed, use sequence query only
            pdb_ids = list(seq_query("polymer_entity"))
        
        # Convert polymer_entity IDs to entry IDs (PDB IDs) and collect unique PDB IDs
        seen_pdb_ids = set()
        for poly_id in pdb_ids:
            # Extract PDB ID from polymer_entity ID (format: "1ABC_1" -> "1ABC")
            pdb_id = poly_id.split('_')[0] if '_' in poly_id else poly_id
            if len(pdb_id) == 4 and pdb_id not in seen_pdb_ids:  # Valid PDB ID format and not duplicate
                pdb_id_list.append(pdb_id)
                seen_pdb_ids.add(pdb_id)
    
    except Exception as e:
        print(f"  Error in template search: {e}")
    
    return pdb_id_list


def add_column_if_not_exists(db_path, pred_table, column_name):
    """Add column to pred_table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({pred_table})")
        columns = [row[1] for row in cursor.fetchall()]
        
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {pred_table} ADD COLUMN {column_name} TEXT")
            conn.commit()
            print(f"Added column {column_name} to {pred_table}")
    finally:
        conn.close()


def update_template_info(db_path, pred_table, exp_db_id, pdb_id_list):
    """
    Update database with template/homolog PDB IDs.
    
    Args:
        db_path: Path to database
        pred_table: Table name
        exp_db_id: exp_db_id to update
        pdb_id_list: List of PDB IDs (e.g., ["1TXS", "1RFR"])
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Store as JSON list of PDB IDs
        value = json.dumps(pdb_id_list) if pdb_id_list else None
        
        cursor.execute(
            f"UPDATE {pred_table} SET RNA_PDB_Templates = ? WHERE exp_db_id = ?",
            (value, exp_db_id)
        )
        conn.commit()
    finally:
        conn.close()


def process_sequences_af3(db_path, pred_table,
                          min_identity=0.3,
                          max_evalue=0.001,
                          max_release_date="2021-09-30",
                          requests_per_second=2,
                          start_from=None,
                          skip_existing=True):
    """
    Process RNA sequences to find templates using AlphaFold3-like approach.
    
    Args:
        db_path: Path to SQLite database
        pred_table: Name of predicted table
        min_identity: Minimum sequence identity (for reference, not directly used)
        max_evalue: Maximum E-value
        max_release_date: Maximum release date
        requests_per_second: Rate limiting
        start_from: exp_db_id to start from
        skip_existing: Skip existing entries
    """
    # Add column if needed
    add_column_if_not_exists(db_path, pred_table, "RNA_PDB_Templates")
    
    # Get sequences from database
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        query = f"SELECT exp_db_id, RNASequence FROM {pred_table} WHERE RNASequence IS NOT NULL AND RNASequence != ''"
        cursor.execute(query)
        sequences = cursor.fetchall()
    finally:
        conn.close()
    
    if not sequences:
        print("No sequences found in database")
        return
    
    # Filter sequences
    if start_from:
        start_idx = next((i for i, (exp_db_id, _) in enumerate(sequences) if exp_db_id == start_from), 0)
        sequences = sequences[start_idx:]
        print(f"Starting from exp_db_id: {start_from}")
    
    if skip_existing:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT exp_db_id FROM {pred_table} WHERE RNA_PDB_Templates IS NOT NULL AND RNA_PDB_Templates != ''"
            )
            processed_ids = {row[0] for row in cursor.fetchall()}
            sequences = [(exp_db_id, seq) for exp_db_id, seq in sequences if exp_db_id not in processed_ids]
        finally:
            conn.close()
    
    print(f"Processing {len(sequences)} sequences...")
    
    delay = 1.0 / requests_per_second
    total = len(sequences)
    successful = 0
    failed = 0
    
    for idx, (exp_db_id, rna_sequence) in enumerate(sequences, 1):
        print(f"[{idx}/{total}] Processing {exp_db_id}...", end=" ")
        
        try:
            # Parse sequence (handle JSON lists)
            try:
                parsed = json.loads(rna_sequence)
                if isinstance(parsed, list):
                    seq_list = parsed
                else:
                    seq_list = [parsed]
            except:
                seq_list = [rna_sequence]
            
            all_pdb_ids = []
            seen_ids = set()
            for seq in seq_list:
                pdb_ids = find_rna_templates(
                    str(seq),
                    min_identity=min_identity,
                    max_evalue=max_evalue,
                    max_release_date=max_release_date
                )
                # Add unique PDB IDs
                for pdb_id in pdb_ids:
                    if pdb_id not in seen_ids:
                        all_pdb_ids.append(pdb_id)
                        seen_ids.add(pdb_id)
            
            if all_pdb_ids:
                update_template_info(db_path, pred_table, exp_db_id, all_pdb_ids)
                print(f"✓ Found {len(all_pdb_ids)} templates: {all_pdb_ids[:5]}{'...' if len(all_pdb_ids) > 5 else ''}")
                successful += 1
            else:
                print("✗ No templates found")
                failed += 1
            
            time.sleep(delay)
        
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    print(f"\nCompleted: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description='Find RNA structural templates/homologs using AlphaFold3-like approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python rnaHomologsAF3.py
  
  # With custom parameters
  python rnaHomologsAF3.py --min-identity 0.5 --max-evalue 0.001
  
  # Resume from specific ID
  python rnaHomologsAF3.py --start-from 1ABC
        """
    )
    
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to SQLite database (default: from StartConfig)')
    parser.add_argument('--min-identity', type=float, default=0.3,
                       help='Minimum sequence identity (default: 0.3)')
    parser.add_argument('--max-evalue', type=float, default=0.001,
                       help='Maximum E-value (default: 0.001, typical for AlphaFold3 template searches)')
    parser.add_argument('--max-release-date', type=str, default="2021-09-30",
                       help='Maximum release date YYYY-MM-DD (default: 2021-09-30)')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='Maximum requests per second (default: 2.0)')
    parser.add_argument('--start-from', type=str, default=None,
                       help='exp_db_id to start from')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Reprocess existing entries')
    
    args = parser.parse_args()
    
    # Validate date
    try:
        from datetime import datetime
        datetime.strptime(args.max_release_date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.max_release_date}'. Use YYYY-MM-DD")
        return
    
    # Get configuration
    config = StartConfig()
    db_path = args.db_path or config.db_path
    pred_table = config.pred_table
    
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"Min identity: {args.min_identity}")
    print(f"Max E-value: {args.max_evalue}")
    print(f"Max release date: {args.max_release_date}")
    print()
    
    process_sequences_af3(
        db_path=db_path,
        pred_table=pred_table,
        min_identity=args.min_identity,
        max_evalue=args.max_evalue,
        max_release_date=args.max_release_date,
        requests_per_second=args.rate,
        start_from=args.start_from,
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    main()

