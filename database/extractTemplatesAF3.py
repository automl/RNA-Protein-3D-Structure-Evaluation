#!/usr/bin/env python3
"""
Extract RNA template PDB IDs from AlphaFold3 data JSON files

This script processes AlphaFold3 output JSON files (e.g., 6xn3_s1_data.json) and extracts
template PDB IDs from the "templates" section. The PDB IDs are extracted from the "mmcif"
field (e.g., "data_6IFN" -> "6IFN") and stored in the database.

Usage:
    python extractTemplatesFromAF3Data.py /path/to/data/folder
    python extractTemplatesFromAF3Data.py /path/to/data/folder --db-path /path/to/db.db
"""

import json
import sqlite3
import os
import argparse
import re
from pathlib import Path
from typing import List, Set, Dict
from database.startConfig import StartConfig


def extract_pdb_id_from_filename(filename):
    """
    Extract PDB ID from filename (first 4 characters, converted to uppercase).
    
    Args:
        filename: Filename like "6xn3_s1_data.json" or "6XN3_s1_data.json"
    
    Returns:
        PDB ID in uppercase (e.g., "6XN3") or None if invalid
    """
    # Get base filename without extension
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # Extract first 4 characters and convert to uppercase
    if len(base) >= 4:
        pdb_id = base[:4].upper()
        # Validate PDB ID format (4 alphanumeric characters)
        if re.match(r'^[A-Z0-9]{4}$', pdb_id):
            return pdb_id
    return None


def extract_pdb_id_from_mmcif(mmcif_value):
    """
    Extract PDB ID from mmcif field value.
    
    Args:
        mmcif_value: Value like "data_6IFN\n#\n_entry.id" or "data_6IFN"
    
    Returns:
        PDB ID (e.g., "6IFN") or None if not found
    """
    if not mmcif_value:
        return None
    
    # Convert to string and extract PDB ID
    mmcif_str = str(mmcif_value)
    
    # Pattern: "data_" followed by 4 alphanumeric characters (PDB ID)
    # Examples: "data_6IFN", "data_6IFN\n#\n_entry.id"
    pattern = r'data_([A-Z0-9]{4})'
    match = re.search(pattern, mmcif_str, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    return None


def classify_template_from_mmcif(mmcif_text):
    """
    Classify template type based on mmcif content.
    
    Args:
        mmcif_text: mmcif field value (string)
    
    Returns:
        Classification string: "RNA", "protein-RNA", "protein", or "unknown"
    """
    if not mmcif_text:
        return "unknown"
    
    mmcif_str = str(mmcif_text)
    
    rna_keywords = {"RNA linking", "RNA MONOMER", "RNA monomer"}
    protein_keywords = {"L-peptide linking", "D-peptide linking", "peptide linking"}
    dna_keywords = {"DNA linking"}
    
    found_rna = any(k in mmcif_str for k in rna_keywords)
    found_protein = any(k in mmcif_str for k in protein_keywords)
    found_dna = any(k in mmcif_str for k in dna_keywords)
    
    if found_rna and found_protein:
        return "protein-RNA"
    elif found_rna:
        return "RNA"
    elif found_protein:
        return "protein"
    elif found_dna:
        return "DNA"
    else:
        return "unknown"


FIRST_ENTITY_ONLY = False   # True reproduces the earlier extraction (first protein entity only)


def extract_templates_from_json(json_file_path):
    """
    Extract template PDB IDs from an AlphaFold3 data JSON file, separated by type.

    Templates are collected from EVERY entity of the job (each protein chain has its own template
    search); with FIRST_ENTITY_ONLY (--first-entity-only) only the first "templates" list is read,
    as in the extraction used up to 2026-10-07.

    Args:
        json_file_path: Path to JSON file (e.g., "6xn3_s1_data.json")
    
    Returns:
        Tuple of (rna_template_pdb_ids, rbp_template_pdb_ids, protein_template_pdb_ids) as sets,
        or (None, None, None) if file is invalid
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        rna_template_pdb_ids = set()
        rbp_template_pdb_ids = set()  # protein-RNA complexes
        protein_template_pdb_ids = set()
        
        # Helper function to find templates recursively
        def find_templates(obj):
            """Recursively search for 'templates' key in JSON structure."""
            if isinstance(obj, dict):
                if 'templates' in obj:
                    return obj['templates']
                # Search in nested dicts
                for value in obj.values():
                    result = find_templates(value)
                    if result is not None:
                        return result
            elif isinstance(obj, list):
                # Search in list items
                for item in obj:
                    result = find_templates(item)
                    if result is not None:
                        return result
            return None
        
        def find_all_templates(obj, out):
            """Every 'templates' list in the JSON structure (one per protein entity)."""
            if isinstance(obj, dict):
                if isinstance(obj.get('templates'), list):
                    out.append(obj['templates'])
                for key, value in obj.items():
                    if key != 'templates':
                        find_all_templates(value, out)
            elif isinstance(obj, list):
                for item in obj:
                    find_all_templates(item, out)
            return out

        # Find templates in the JSON structure
        if FIRST_ENTITY_ONLY:
            templates = find_templates(data)
        else:
            lists = find_all_templates(data, [])
            templates = [tp for lst in lists for tp in lst] if lists else None
        
        if templates is None:
            return None, None, None, False  # Return empty_found flag
        
        if not isinstance(templates, list):
            return None, None, None, False
        
        # If templates array is empty, return empty lists
        if len(templates) == 0:
            return [], [], [], True  # Return empty lists and empty_found=True
        
        # Process each template
        for template in templates:
            if not isinstance(template, dict):
                continue
            
            if 'mmcif' not in template:
                continue
            
            mmcif_value = template['mmcif']
            pdb_id = extract_pdb_id_from_mmcif(mmcif_value)
            
            if not pdb_id:
                continue
            
            classification = classify_template_from_mmcif(mmcif_value)
            
            if classification == "RNA":
                rna_template_pdb_ids.add(pdb_id)
            elif classification == "protein-RNA":
                rbp_template_pdb_ids.add(pdb_id)
            elif classification == "protein":
                protein_template_pdb_ids.add(pdb_id)
            # Skip "DNA" and "unknown" classifications
        
        # Return empty sets instead of None if we processed templates but found none of the relevant types
        # This distinguishes between "no templates" and "templates found but all DNA/unknown"
        # Return empty lists if sets are empty (to distinguish from None which means templates not found)
        return (list(rna_template_pdb_ids) if rna_template_pdb_ids else [],
                list(rbp_template_pdb_ids) if rbp_template_pdb_ids else [],
                list(protein_template_pdb_ids) if protein_template_pdb_ids else [],
                True)  # empty_found=True means templates array was found (even if empty)
    
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
        return None, None, None, False
    except IOError as e:
        print(f"  IO error: {e}")
        return None, None, None, False
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None, None, None, False


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


def update_templates_in_db(db_path, pred_table, exp_db_id, rna_pdb_ids, rbp_pdb_ids, protein_pdb_ids):
    """
    Update template columns based on pred_table type.
    
    Args:
        db_path: Path to database
        pred_table: Table name
        exp_db_id: exp_db_id to update
        rna_pdb_ids: List of PDB IDs with only RNA (e.g., ["6IFN", "1TXS"])
        rbp_pdb_ids: List of PDB IDs with protein-RNA complexes (e.g., ["1ABC"])
        protein_pdb_ids: List of PDB IDs with only protein (e.g., ["2DEF"])
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Store as JSON lists (empty list becomes "[]")
        rna_value = json.dumps(sorted(rna_pdb_ids)) if rna_pdb_ids else "[]"
        rbp_value = json.dumps(sorted(rbp_pdb_ids)) if rbp_pdb_ids else "[]"
        protein_value = json.dumps(sorted(protein_pdb_ids)) if protein_pdb_ids else "[]"
        
        # Build UPDATE query based on pred_table
        if pred_table == "pred_protein_rna":
            # Always update RBP and Protein columns
            # Only update RNA column if it exists (will be added if RNA templates found)
            columns_to_update = ["RBP_PDB_Templates", "Protein_PDB_Templates"]
            values = [rbp_value, protein_value]
            
            # Check if RNA_PDB_Templates column exists
            cursor.execute(f"PRAGMA table_info({pred_table})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            if "RNA_PDB_Templates" in existing_columns:
                columns_to_update.append("RNA_PDB_Templates")
                values.append(rna_value)
            
            placeholders = ", ".join([f"{col} = ?" for col in columns_to_update])
            values.append(exp_db_id)
            cursor.execute(
                f"UPDATE {pred_table} SET {placeholders} WHERE exp_db_id = ?",
                values
            )
        else:
            # For pred_rna_rna and others, only update RNA_PDB_Templates
            cursor.execute(
                f"UPDATE {pred_table} SET RNA_PDB_Templates = ? WHERE exp_db_id = ?",
                (rna_value, exp_db_id)
            )
        
        conn.commit()
    finally:
        conn.close()




def process_data_folder(data_folder, db_path, pred_table, overwrite=False):
    """
    Process all _data.json files in a folder and update the database.
    
    Args:
        data_folder: Path to folder containing *_data.json files
        db_path: Path to SQLite database
        pred_table: Name of predicted table
        overwrite: If True, overwrite existing values. If False, skip existing entries.
    """
    data_folder = Path(data_folder)
    
    if not data_folder.exists():
        print(f"Error: Folder does not exist: {data_folder}")
        return

    "1 folder"
    # Find all *_data.json files
    # json_files = list(data_folder.glob("*_data.json"))

    "SUBFOLDERS"
    # Find all *_data.json files in the main folder and one level of subfolders
    json_files = []
    # Search in main folder
    json_files.extend(list(data_folder.glob("*_data.json")))
    # Search in immediate subfolders (one level deep)
    for subfolder in data_folder.iterdir():
        if subfolder.is_dir():
            json_files.extend(list(subfolder.glob("*_data.json")))

    if not json_files:
        print(f"No *_data.json files found in {data_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Track if we find any RNA templates (for conditional column creation)
    found_rna_templates = False
    
    # Add columns conditionally based on pred_table
    if pred_table == "pred_protein_rna":
        # Always add RBP and Protein columns for pred_protein_rna
        add_column_if_not_exists(db_path, pred_table, "RBP_PDB_Templates")
        add_column_if_not_exists(db_path, pred_table, "Protein_PDB_Templates")
        # RNA column will be added later if RNA templates are found
    else:
        # For pred_rna_rna and others, only add RNA column
        add_column_if_not_exists(db_path, pred_table, "RNA_PDB_Templates")
    
    # Check which exp_db_ids already have values (if not overwriting)
    existing_ids = set()
    if not overwrite:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            # Build query based on pred_table
            if pred_table == "pred_protein_rna":
                cursor.execute(
                    f"SELECT exp_db_id FROM {pred_table} WHERE (RBP_PDB_Templates IS NOT NULL AND RBP_PDB_Templates != '' AND RBP_PDB_Templates != '[]') OR (Protein_PDB_Templates IS NOT NULL AND Protein_PDB_Templates != '' AND Protein_PDB_Templates != '[]')"
                )
            else:
                cursor.execute(
                    f"SELECT exp_db_id FROM {pred_table} WHERE RNA_PDB_Templates IS NOT NULL AND RNA_PDB_Templates != '' AND RNA_PDB_Templates != '[]'"
                )
            existing_ids = {row[0] for row in cursor.fetchall()}
            print(f"Found {len(existing_ids)} existing entries (will be skipped)")
        finally:
            conn.close()
    
    # Process each JSON file
    successful = 0
    skipped = 0
    failed = 0
    not_found = 0
    
    for json_file in sorted(json_files):
        # Extract PDB ID from filename
        pdb_id = extract_pdb_id_from_filename(json_file.name)
        
        if not pdb_id:
            print(f"Skipping {json_file.name}: Invalid PDB ID format")
            skipped += 1
            continue
        
        # Check if already exists
        if not overwrite and pdb_id in existing_ids:
            print(f"Skipping {json_file.name} ({pdb_id}): Already in database")
            skipped += 1
            continue
        
        print(f"Processing {json_file.name} (PDB ID: {pdb_id})...", end=" ")
        
        # Extract template PDB IDs from JSON (separated by type: RNA, protein-RNA, protein)
        result = extract_templates_from_json(json_file)
        if len(result) == 4:
            rna_template_pdb_ids, rbp_template_pdb_ids, protein_template_pdb_ids, templates_found = result
        else:
            # Fallback for old return format
            rna_template_pdb_ids, rbp_template_pdb_ids, protein_template_pdb_ids = result
            templates_found = False
        
        # Check if templates were found (None means not found, [] means empty array was found)
        if rna_template_pdb_ids is None and rbp_template_pdb_ids is None and protein_template_pdb_ids is None:
            # Try to get more info about why templates weren't found
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check structure
                if isinstance(data, list):
                    # Check if any item in the list has templates
                    found_templates = False
                    for item in data:
                        if isinstance(item, dict) and 'templates' in item:
                            found_templates = True
                            templates = item['templates']
                            if isinstance(templates, list) and len(templates) > 0:
                                print(f"✗ Found {len(templates)} templates but none matched RNA/protein-RNA/protein criteria")
                            else:
                                print(f"✗ 'templates' array is empty or not a list")
                            break
                    if not found_templates:
                        print(f"✗ No 'templates' key in JSON (top-level keys: {list(data[0].keys()) if data and isinstance(data[0], dict) else 'N/A'})")
                elif isinstance(data, dict):
                    if 'templates' not in data:
                        print(f"✗ No 'templates' key in JSON (top-level keys: {list(data.keys())})")
                    elif not isinstance(data['templates'], list):
                        print(f"✗ 'templates' is not a list (type: {type(data['templates'])})")
                    elif len(data['templates']) == 0:
                        # Empty templates array - we'll update with []
                        print("Found empty templates array, will update with []")
                        rna_template_pdb_ids = []
                        rbp_template_pdb_ids = []
                        protein_template_pdb_ids = []
                        templates_found = True
                    else:
                        print(f"✗ Found {len(data['templates'])} templates but none matched RNA/protein-RNA/protein criteria")
                        failed += 1
                        continue
                else:
                    print(f"✗ Unexpected JSON structure (type: {type(data)})")
                    failed += 1
                    continue
            except Exception as e:
                print(f"✗ Error reading file: {e}")
                failed += 1
                continue
        
        # Convert to lists (already lists if from extract_templates_from_json, or empty lists if templates was empty)
        if isinstance(rna_template_pdb_ids, list):
            rna_list = sorted(rna_template_pdb_ids) if rna_template_pdb_ids else []
        else:
            rna_list = sorted(list(rna_template_pdb_ids)) if rna_template_pdb_ids else []
        
        if isinstance(rbp_template_pdb_ids, list):
            rbp_list = sorted(rbp_template_pdb_ids) if rbp_template_pdb_ids else []
        else:
            rbp_list = sorted(list(rbp_template_pdb_ids)) if rbp_template_pdb_ids else []
        
        if isinstance(protein_template_pdb_ids, list):
            protein_list = sorted(protein_template_pdb_ids) if protein_template_pdb_ids else []
        else:
            protein_list = sorted(list(protein_template_pdb_ids)) if protein_template_pdb_ids else []
        
        # Track if we found RNA templates (for conditional column creation)
        if rna_list:
            found_rna_templates = True
            # Add RNA_PDB_Templates column if it doesn't exist and we're in pred_protein_rna
            if pred_table == "pred_protein_rna":
                add_column_if_not_exists(db_path, pred_table, "RNA_PDB_Templates")
        
        # If templates array was found but empty (templates_found=True and all lists empty), update with []
        # If templates were not found at all (templates_found=False) and no valid templates, skip
        if not templates_found and not rna_list and not rbp_list and not protein_list:
            print("✗ No valid template PDB IDs found")
            failed += 1
            continue
        
        # If templates array was empty (templates_found=True but all lists empty), we still update with []
        # This case is handled by continuing to the update step below
        
        # Update database
        try:
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
                    update_templates_in_db(db_path, pred_table, pdb_id, rna_list, rbp_list, protein_list)
                    msg_parts = []
                    if rna_list:
                        msg_parts.append(f"{len(rna_list)} RNA")
                    if rbp_list:
                        msg_parts.append(f"{len(rbp_list)} protein-RNA")
                    if protein_list:
                        msg_parts.append(f"{len(protein_list)} protein")
                    print(f"✓ Found {', '.join(msg_parts)} templates")
                    if rna_list:
                        print(f"  RNA: {rna_list[:5]}{'...' if len(rna_list) > 5 else ''}")
                    if rbp_list:
                        print(f"  Protein-RNA: {rbp_list[:5]}{'...' if len(rbp_list) > 5 else ''}")
                    if protein_list:
                        print(f"  Protein: {protein_list[:5]}{'...' if len(protein_list) > 5 else ''}")
                    successful += 1
            finally:
                conn.close()
        
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
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
        description='Extract RNA template PDB IDs from AlphaFold3 data JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all *_data.json files in a folder
  python extractTemplatesFromAF3Data.py /path/to/data/folder
  
  # Overwrite existing entries
  python extractTemplatesFromAF3Data.py /path/to/data/folder --overwrite
  
  # Use custom database path
  python extractTemplatesFromAF3Data.py /path/to/data/folder --db-path /path/to/db.db
        """
    )
    
    parser.add_argument('data_folder', type=str,
                       help='Path to folder containing *_data.json files')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to SQLite database (default: from StartConfig)')
    parser.add_argument('--first-entity-only', action='store_true',
                        help='Read only the first protein entity\'s templates (the extraction used up to '
                             '2026-10-07); default: templates of all protein entities.')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing RNA_PDB_Templates values')
    
    args = parser.parse_args()
    global FIRST_ENTITY_ONLY
    FIRST_ENTITY_ONLY = args.first_entity_only

    # Get configuration
    config = StartConfig()
    db_path = args.db_path or config.db_path
    pred_table = config.pred_table
    
    print(f"Data folder: {args.data_folder}")
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"Overwrite existing: {args.overwrite}")
    print()
    
    process_data_folder(
        data_folder=args.data_folder,
        db_path=db_path,
        pred_table=pred_table,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()

