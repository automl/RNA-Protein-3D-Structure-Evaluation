#!/usr/bin/env python3
"""
Load template PDB IDs from CSV file into database.

This script reads a CSV file with columns: pdb_id, RBP_PDB_Templates, Protein_PDB_Templates, RNA_PDB_Templates
and updates the database pred_table with this information.

Handles CSV conversion issues:
- "3 BWP" -> "3BWP" (removes spaces)
- "5E+02" -> "5E02" (removes + sign)

Usage:
    python loadTemplatesFromCSV.py /path/to/templates.csv
    python loadTemplatesFromCSV.py /path/to/templates.csv --overwrite
"""

import json
import sqlite3
import os
import argparse
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional
from database.startConfig import StartConfig


def normalize_pdb_id(pdb_id: str) -> str:
    """
    Normalize PDB ID by removing spaces and + signs.
    
    Handles conversion issues:
    - "3 BWP" -> "3BWP"
    - "5E+02" -> "5E02"
    
    Args:
        pdb_id: PDB ID string (may contain spaces or + signs)
    
    Returns:
        Normalized PDB ID (uppercase, no spaces, no + signs)
    """
    if not pdb_id:
        return ""
    
    # Convert to string and uppercase
    pdb_id = str(pdb_id).upper()
    
    # Remove spaces
    pdb_id = pdb_id.replace(" ", "")
    
    # Remove + signs (e.g., "5E+02" -> "5E02")
    pdb_id = pdb_id.replace("+", "")
    
    # Validate format (4 alphanumeric characters)
    if re.match(r'^[A-Z0-9]{4}$', pdb_id):
        return pdb_id
    
    return pdb_id


def parse_template_column(value: Optional[str]) -> Optional[List[str]]:
    """
    Parse template column value from CSV.
    
    Args:
        value: CSV value (can be None, empty string, "[]", or JSON list string)
    
    Returns:
        List of normalized PDB IDs or None if empty/invalid
    """
    if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
        return None
    
    if isinstance(value, str):
        value = value.strip()
        if value == "[]":
            return []
    
    try:
        # Try to parse as JSON
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, list):
            # Normalize all PDB IDs in the list
            normalized = [normalize_pdb_id(pdb_id) for pdb_id in parsed if normalize_pdb_id(pdb_id)]
            return normalized if normalized else []
        return None
    except (json.JSONDecodeError, TypeError):
        # If not JSON, try to parse as comma-separated string
        if isinstance(value, str) and ',' in value:
            pdb_ids = [normalize_pdb_id(p.strip()) for p in value.split(',')]
            return [p for p in pdb_ids if p] if pdb_ids else []
        # Try single value
        normalized = normalize_pdb_id(value)
        return [normalized] if normalized else None


def add_column_if_not_exists(db_path: str, pred_table: str, column_name: str):
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


def update_templates_in_db(db_path: str, pred_table: str, exp_db_id: str, 
                          rbp_pdb_ids: Optional[List[str]], 
                          protein_pdb_ids: Optional[List[str]]):
    """
    Update template columns in database.
    
    Args:
        db_path: Path to database
        pred_table: Table name
        exp_db_id: exp_db_id to update
        rbp_pdb_ids: List of RBP template PDB IDs or None
        protein_pdb_ids: List of Protein template PDB IDs or None
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Store as JSON lists (empty list becomes "[]")
        rbp_value = json.dumps(sorted(rbp_pdb_ids)) if rbp_pdb_ids else "[]"
        protein_value = json.dumps(sorted(protein_pdb_ids)) if protein_pdb_ids else "[]"
        
        # Build UPDATE query based on pred_table
        if pred_table == "pred_protein_rna":
            # Always update RBP and Protein columns
            columns_to_update = ["RBP_PDB_Templates", "Protein_PDB_Templates"]
            values = [rbp_value, protein_value]
            
            placeholders = ", ".join([f"{col} = ?" for col in columns_to_update])
            values.append(exp_db_id)
            cursor.execute(
                f"UPDATE {pred_table} SET {placeholders} WHERE exp_db_id = ?",
                values
            )
        else:
            # For pred_rna_rna and others, no template columns to update
            print(f"Warning: {pred_table} is not pred_protein_rna, skipping template update")
        
        conn.commit()
    finally:
        conn.close()


def process_csv_file(csv_path: Path, db_path: str, pred_table: str, overwrite: bool = False):
    """
    Process CSV file and update database with template information.
    
    Args:
        csv_path: Path to CSV file
        db_path: Path to database
        pred_table: Name of prediction table
        overwrite: If True, overwrite existing values
    """
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        return
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path, dtype=str)  # Read all as strings to preserve format
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check required columns
    required_cols = ['pdb_id']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain 'pdb_id' column. Found columns: {df.columns.tolist()}")
        return
    
    # Check which template columns are present
    has_rbp = 'RBP_PDB_Templates' in df.columns
    has_protein = 'Protein_PDB_Templates' in df.columns
    
    if not (has_rbp or has_protein):
        print("Warning: No template columns found in CSV (RBP_PDB_Templates, Protein_PDB_Templates)")
    
    # Add columns to database if they don't exist
    if has_rbp:
        add_column_if_not_exists(db_path, pred_table, "RBP_PDB_Templates")
    if has_protein:
        add_column_if_not_exists(db_path, pred_table, "Protein_PDB_Templates")
    
    # Check which exp_db_ids already have values (if not overwriting)
    existing_ids = set()
    if not overwrite:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            # Build query based on pred_table and available columns
            conditions = []
            if has_rbp:
                conditions.append("(RBP_PDB_Templates IS NOT NULL AND RBP_PDB_Templates != '' AND RBP_PDB_Templates != '[]')")
            if has_protein:
                conditions.append("(Protein_PDB_Templates IS NOT NULL AND Protein_PDB_Templates != '' AND Protein_PDB_Templates != '[]')")
            
            if conditions:
                query = f"SELECT exp_db_id FROM {pred_table} WHERE {' OR '.join(conditions)}"
                cursor.execute(query)
                existing_ids = {row[0] for row in cursor.fetchall()}
                print(f"Found {len(existing_ids)} existing entries (will be skipped)")
        finally:
            conn.close()
    
    # Process each row
    successful = 0
    skipped = 0
    failed = 0
    not_found = 0
    
    for idx, row in df.iterrows():
        # Get and normalize PDB ID
        pdb_id_raw = row.get('pdb_id', '')
        pdb_id = normalize_pdb_id(pdb_id_raw)
        
        if not pdb_id:
            print(f"Row {idx + 1}: Invalid PDB ID '{pdb_id_raw}', skipping")
            skipped += 1
            continue
        
        # Check if already exists
        if not overwrite and pdb_id in existing_ids:
            print(f"Skipping {pdb_id}: Already in database")
            skipped += 1
            continue
        
        # Parse template columns
        rbp_pdb_ids = parse_template_column(row.get('RBP_PDB_Templates')) if has_rbp else None
        protein_pdb_ids = parse_template_column(row.get('Protein_PDB_Templates')) if has_protein else None
        
        # Check if exp_db_id exists in database
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {pred_table} WHERE exp_db_id = ?",
                (pdb_id,)
            )
            count = cursor.fetchone()[0]
            
            if count == 0:
                print(f"✗ exp_db_id '{pdb_id}' not found in {pred_table}")
                not_found += 1
            else:
                # Update database
                update_templates_in_db(db_path, pred_table, pdb_id, rbp_pdb_ids, protein_pdb_ids)
                
                msg_parts = []
                if rbp_pdb_ids:
                    msg_parts.append(f"{len(rbp_pdb_ids)} RBP")
                if protein_pdb_ids:
                    msg_parts.append(f"{len(protein_pdb_ids)} Protein")
                
                if msg_parts:
                    print(f"✓ {pdb_id}: Updated with {', '.join(msg_parts)} templates")
                else:
                    print(f"✓ {pdb_id}: Updated with empty templates")
                successful += 1
        except Exception as e:
            print(f"✗ Error updating {pdb_id}: {e}")
            failed += 1
        finally:
            conn.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Not found in DB: {not_found}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Load template PDB IDs from CSV file into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load templates from CSV
  python loadTemplatesFromCSV.py templates.csv
  
  # Overwrite existing entries
  python loadTemplatesFromCSV.py templates.csv --overwrite
        """
    )
    
    parser.add_argument('csv_file', type=str,
                       help='Path to CSV file with template data')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing template values')
    
    args = parser.parse_args()
    
    # Get configuration
    config = StartConfig()
    db_path = config.get_database_path()
    pred_table = config.pred_table
    
    print(f"CSV file: {args.csv_file}")
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"Overwrite existing: {args.overwrite}")
    print()
    
    process_csv_file(
        csv_path=Path(args.csv_file),
        db_path=db_path,
        pred_table=pred_table,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()

