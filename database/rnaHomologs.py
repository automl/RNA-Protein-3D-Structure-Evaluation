import json
import sqlite3
import time
import argparse
from typing import Optional, Dict, Tuple
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
    # Don't print warning here - will be handled in the function if needed

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
    
    Handles sequences with modified nucleotides in parentheses, e.g., "GGUUGAAACACGACCUGAGA(A2M)AG"
    Replaces them with canonical nucleotides based on ALLOWED_MODIFIED_NUCLEOTIDES dictionary.
    
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
        # Return canonical equivalent if in dictionary, otherwise keep original
        return ALLOWED_MODIFIED_NUCLEOTIDES.get(mod_code, mod_code)
    
    # Replace all (MOD) patterns with canonical nucleotides
    normalized = re.sub(pattern, replace_mod, sequence)
    
    return normalized


def count_pdb_rna_hits(rna_sequence, 
                       evalue_cutoff=1.0, 
                       timeout=30,
                       max_retries=3,
                       retry_delay=1,
                       min_sequence_length=25,
                       max_release_date="2021-09-30"):
    """
    Returns the number of PDB RNA homologs for a given RNA sequence.
    
    Uses rcsbapi package (SeqSimilarityQuery) to query RCSB PDB Search API.
    See: https://rcsbapi.readthedocs.io/en/latest/search_api/additional_examples.html
    and: https://search.rcsb.org/#search-services
    
    Args:
        rna_sequence: RNA sequence string
        evalue_cutoff: E-value cutoff for BLAST search
        timeout: Request timeout in seconds (not used with rcsbapi, kept for compatibility)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        min_sequence_length: Minimum sequence length (sequences shorter than this will return None to skip)
                          Note: RCSB API requires at least 25 residues, so default is 25
        max_release_date: Maximum release date (YYYY-MM-DD format). Only structures released on or before
                         this date will be included. Default is "2021-09-30".
    
    Returns:
        Number of PDB RNA structural homologs, or None if query failed or sequence too short
    """
    # Validate sequence
    if not rna_sequence or len(rna_sequence.strip()) == 0:
        return None
    
    # Clean sequence (remove whitespace, convert to uppercase)
    rna_sequence = rna_sequence.strip().upper()
    
    # Normalize modified nucleotides to canonical equivalents
    # e.g., "GGUUGAAACACGACCUGAGA(A2M)AG" -> "GGUUGAAACACGACCUGAGAAG"
    rna_sequence = normalize_rna_sequence(rna_sequence)
    
    # Check minimum length (RCSB requires at least 25 residues)
    # Check length after normalization
    if len(rna_sequence) < min_sequence_length:
        return None  # Return None to indicate skip
    
    if not RCSBAPI_AVAILABLE:
        print("  Error: rcsbapi package is required. Install with: pip install rcsb-api")
        return None
    
    # Use rcsbapi SeqSimilarityQuery for sequence similarity search
    # According to RCSB API docs: sequence service supports "rna" for RNA sequences
    # See: https://search.rcsb.org/#search-services
    # Documentation: https://rcsbapi.readthedocs.io/en/latest/search_api/additional_examples.html
    for attempt in range(max_retries):
        try:
            # Create sequence similarity query using rcsbapi
            # sequence_type="rna" for RNA sequences
            seq_query = SeqSimilarityQuery(
                value=rna_sequence,
                evalue_cutoff=evalue_cutoff,
                sequence_type="rna"  # Use "rna" for RNA sequences per RCSB API docs
            )
            
            # Try to combine with date filter
            # Note: The RCSB API may not support combining SeqSimilarityQuery with date filters
            # If it fails, we'll use sequence query only (without date filtering)
            try:
                # Try using search_attributes if available
                if SEARCH_ATTRS_AVAILABLE:
                    date_query = attrs.rcsb_accession_info.initial_release_date <= max_release_date
                    combined_query = seq_query & date_query
                    results = list(combined_query("polymer_entity"))
                else:
                    # Use AttributeQuery as fallback
                    date_query = AttributeQuery(
                        attribute="rcsb_accession_info.initial_release_date",
                        operator="less_or_equal",
                        value=max_release_date
                    )
                    combined_query = seq_query & date_query
                    results = list(combined_query("polymer_entity"))
            except Exception:
                # Date filtering failed - this is a known limitation
                # The RCSB API may not support combining sequence similarity with date filters
                # Fall back to sequence query only (no date filtering)
                results = list(seq_query("polymer_entity"))
            
            return len(results)
            
        except Exception as e:
            error_msg = str(e)
            # Try to extract more detailed error information
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    import json
                    error_data = json.loads(e.response.text)
                    if 'message' in error_data:
                        error_msg += f" - {error_data['message']}"
                except:
                    error_msg += f" - Response: {e.response.text[:200]}"
            
            if attempt < max_retries - 1:
                print(f"  Error on attempt {attempt + 1}/{max_retries}: {error_msg}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"  Error after {max_retries} attempts: {error_msg}")
                return None
    
    return None


def parse_rna_sequence(sequence_value, min_length=25):
    """
    Parse RNASequence from database - handles both string and JSON list formats.
    Filters out sequences shorter than minimum length.
    
    Args:
        sequence_value: RNASequence value from database (can be string or JSON list)
        min_length: Minimum sequence length in nucleotides (default: 25, since RCSB requires at least 25 residues)
    
    Returns:
        List of RNA sequence strings (≥ min_length), or empty list if invalid
    """
    if not sequence_value:
        return []
    
    sequences = []
    
    # Try to parse as JSON (in case it's stored as a list)
    try:
        parsed = json.loads(sequence_value)
        if isinstance(parsed, list):
            # Return list of sequences (filter out empty ones and short ones)
            sequences = [str(seq).strip().upper() for seq in parsed if seq and str(seq).strip()]
        elif isinstance(parsed, str):
            sequences = [parsed.strip().upper()] if parsed.strip() else []
    except (json.JSONDecodeError, TypeError):
        # Not JSON, treat as plain string
        seq_str = str(sequence_value).strip().upper()
        sequences = [seq_str] if seq_str else []
    
    # Normalize modified nucleotides and filter sequences by length
    filtered_sequences = []
    for seq in sequences:
        # Normalize modified nucleotides (e.g., "GGUUGAAACACGACCUGAGA(A2M)AG" -> "GGUUGAAACACGACCUGAGAAG")
        normalized_seq = normalize_rna_sequence(seq)
        # Check length after normalization
        if len(normalized_seq) >= min_length:
            filtered_sequences.append(normalized_seq)
    
    return filtered_sequences


def get_rna_sequences_from_db(db_path, pred_table):
    """
    Get all exp_db_id and RNASequence pairs from pred_table.
    Handles JSON list format for RNASequence.
    
    Args:
        db_path: Path to SQLite database
        pred_table: Name of predicted table (contains exp_db_id and RNASequence)
    
    Returns:
        List of tuples: [(exp_db_id, [sequence1, sequence2, ...]), ...]
        Each tuple contains exp_db_id and a list of sequences (even if single sequence)
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        # Get RNASequence directly from pred_table
        query = f"SELECT exp_db_id, RNASequence FROM {pred_table} WHERE RNASequence IS NOT NULL AND RNASequence != ''"
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Parse sequences (handle JSON lists, filter short sequences)
        parsed_results = []
        skipped_short = 0
        for exp_db_id, rna_sequence in results:
            parsed_seqs = parse_rna_sequence(rna_sequence, min_length=25)
            if parsed_seqs:  # Only add if we have at least one sequence >= 25 residues
                parsed_results.append((exp_db_id, parsed_seqs))
            else:
                skipped_short += 1
        
        if skipped_short > 0:
            print(f"Skipped {skipped_short} exp_db_ids with no sequences >= 25 residues")
        
        return parsed_results
    finally:
        conn.close()


def add_column_if_not_exists(db_path, pred_table, column_name):
    """Add column to pred_table if it doesn't exist. Uses TEXT to store JSON lists."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        # Check if column exists
        cursor.execute(f"PRAGMA table_info({pred_table})")
        columns = [row[1] for row in cursor.fetchall()]
        
        if column_name not in columns:
            # Use TEXT to store both integers and JSON lists
            cursor.execute(f"ALTER TABLE {pred_table} ADD COLUMN {column_name} TEXT")
            conn.commit()
            print(f"Added column {column_name} to {pred_table}")
        else:
            print(f"Column {column_name} already exists in {pred_table}")
    finally:
        conn.close()


def update_homolog_count(db_path, pred_table, exp_db_id, counts):
    """
    Update the homolog count(s) for a specific exp_db_id in pred_table.
    
    Args:
        db_path: Path to database
        pred_table: Table name
        exp_db_id: exp_db_id identifier
        counts: Integer (single count) or list of integers (multiple counts)
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        # Store as JSON string if it's a list, otherwise as integer
        if isinstance(counts, list):
            value = json.dumps(counts)
        else:
            value = counts
        cursor.execute(
            f"UPDATE {pred_table} SET RNA_PDB_StructuralHomologs = ? WHERE exp_db_id = ?",
            (value, exp_db_id)
        )
        conn.commit()
    finally:
        conn.close()


def process_sequences(db_path, pred_table,
                     evalue_cutoff=1.0,
                     requests_per_second=2,
                     start_from=None,
                     skip_existing=True,
                     max_release_date="2021-09-30"):
    """
    Process all RNA sequences and update the database.
    
    Args:
        db_path: Path to SQLite database
        pred_table: Name of predicted table (contains exp_db_id and RNASequence)
        evalue_cutoff: E-value cutoff for BLAST search
        requests_per_second: Maximum requests per second (rate limiting)
        start_from: exp_db_id to start from (for resuming)
        skip_existing: Skip exp_db_ids that already have a value
        max_release_date: Maximum release date (YYYY-MM-DD format). Only structures released
                         on or before this date will be included. Default is "2021-09-30".
    """
    # Add column if it doesn't exist
    add_column_if_not_exists(db_path, pred_table, "RNA_PDB_StructuralHomologs")
    
    # Get all sequences from pred_table
    sequences = get_rna_sequences_from_db(db_path, pred_table)
    print(f"Found {len(sequences)} sequences to process")
    
    if not sequences:
        print("No sequences found in database")
        return
    
    # Check which exp_db_ids already have values (if skip_existing)
    processed_ids = set()
    if skip_existing:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            # Check for both integer and text (JSON) values
            cursor.execute(
                f"SELECT exp_db_id FROM {pred_table} WHERE RNA_PDB_StructuralHomologs IS NOT NULL AND RNA_PDB_StructuralHomologs != ''"
            )
            processed_ids = {row[0] for row in cursor.fetchall()}
            print(f"Found {len(processed_ids)} exp_db_ids that already have values (will be skipped)")
        finally:
            conn.close()
    
    # Filter sequences
    if start_from:
        start_idx = next((i for i, (exp_db_id, _) in enumerate(sequences) if exp_db_id == start_from), 0)
        sequences = sequences[start_idx:]
        print(f"Starting from exp_db_id: {start_from} (index {start_idx})")
    
    if skip_existing:
        sequences = [(exp_db_id, seq) for exp_db_id, seq in sequences if exp_db_id not in processed_ids]
        print(f"After filtering, {len(sequences)} sequences remain to process")
    
    if not sequences:
        print("No sequences to process")
        return
    
    # Rate limiting: delay between requests
    delay = 1.0 / requests_per_second
    
    # Statistics
    total = len(sequences)
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\nProcessing {total} sequences...")
    print(f"Rate limit: {requests_per_second} requests/second (delay: {delay:.2f}s)")
    print(f"Parameters: evalue_cutoff={evalue_cutoff}")
    print(f"          : max_release_date={max_release_date} (only structures released on or before this date)")
    print()
    
    start_time = time.time()
    
    for idx, (exp_db_id, rna_sequences) in enumerate(sequences, 1):
        # Skip if already processed
        if skip_existing and exp_db_id in processed_ids:
            skipped += 1
            continue
        
        print(f"[{idx}/{total}] Processing {exp_db_id}...", end=" ")
        
        # Query RCSB API for each sequence separately
        counts = []
        for seq_idx, rna_sequence in enumerate(rna_sequences, 1):
            if len(rna_sequences) > 1:
                print(f"\n  Sequence {seq_idx}/{len(rna_sequences)}: ", end="")
            
            # Check length before querying (should already be filtered, but double-check)
            # RCSB requires > 8 nt, so skip sequences ≤ 8
            if len(rna_sequence) <= 8:
                if len(rna_sequences) > 1:
                    print(f"skipped (≤ 8 nt)", end="")
                else:
                    print(f"skipped (≤ 8 nt)")
                continue  # Skip this sequence
            
            count = count_pdb_rna_hits(
                rna_sequence,
                evalue_cutoff=evalue_cutoff,
                min_sequence_length=25,
                max_release_date=max_release_date
            )
            
            if count is not None:
                counts.append(count)
                if len(rna_sequences) > 1:
                    print(f"{count} homologs", end="")
            else:
                if len(rna_sequences) > 1:
                    print(f"failed/skipped", end="")
                # If query failed or sequence too short, skip it (don't add to list)
        
        # Update database with list of counts (or single count if only one sequence)
        if not counts:
            # All sequences were skipped (too short)
            print(f"✗ All sequences < 8 nt, skipped")
            skipped += 1
        elif any(c is not None for c in counts):
            # At least one query succeeded
            # Filter out None values (failed queries) from the list
            valid_counts = [c for c in counts if c is not None]
            if len(valid_counts) == 1:
                update_homolog_count(db_path, pred_table, exp_db_id, valid_counts[0])
                print(f"✓ Found {valid_counts[0]} homologs")
            else:
                update_homolog_count(db_path, pred_table, exp_db_id, valid_counts)
                print(f"✓ Found {valid_counts} homologs")
            successful += 1
        else:
            # All queries failed
            print(f"✗ All queries failed")
            failed += 1
        
        # Rate limiting (except for last item)
        if idx < total:
            time.sleep(delay)
        
        # Progress update every 10 sequences
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total - idx) / rate if rate > 0 else 0
            print(f"\n  Progress: {idx}/{total} ({idx/total*100:.1f}%) | "
                  f"Success: {successful}, Failed: {failed}, Skipped: {skipped} | "
                  f"Rate: {rate:.2f} seq/s | ETA: {remaining/60:.1f} min\n")
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Average rate: {total/elapsed:.2f} sequences/second")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Query RCSB API for RNA structural homologs and update database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sequences with default parameters
  python queryRCSBRNAHomologs.py
  
  # Custom parameters
  python queryRCSBRNAHomologs.py --evalue 0.1 --identity 0.3 --rate 1
  
  # Resume from a specific PDBId
  python queryRCSBRNAHomologs.py --start-from 1ABC
  
  # Reprocess all sequences (don't skip existing)
  python queryRCSBRNAHomologs.py --no-skip-existing
        """
    )
    
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to SQLite database (default: from StartConfig)')
    parser.add_argument('--evalue', type=float, default=1.0,
                       help='E-value cutoff for BLAST search (default: 1.0)')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='Maximum requests per second (default: 2.0)')
    parser.add_argument('--start-from', type=str, default=None,
                       help='exp_db_id to start from (for resuming)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Reprocess sequences that already have values')
    parser.add_argument('--max-release-date', type=str, default="2021-09-30",
                       help='Maximum release date (YYYY-MM-DD format). Only structures released on or before this date will be included. Default: 2021-09-30')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        # Validate date format
        from datetime import datetime
        datetime.strptime(args.max_release_date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.max_release_date}'. Please use YYYY-MM-DD format (e.g., 2021-09-30)")
        return
    
    # Get configuration
    config = StartConfig()
    db_path = args.db_path or config.db_path
    pred_table = config.pred_table
    
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"E-value cutoff: {args.evalue}")
    print(f"Rate limit: {args.rate} requests/second")
    print(f"Max release date: {args.max_release_date} (only structures released on or before this date)")
    print(f"Note: Using sequence_type='rna' (RCSB API requirement for RNA sequences)")
    print()
    
    # Process sequences
    process_sequences(
        db_path=db_path,
        pred_table=pred_table,
        evalue_cutoff=args.evalue,
        requests_per_second=args.rate,
        start_from=args.start_from,
        skip_existing=not args.no_skip_existing,
        max_release_date=args.max_release_date
    )


if __name__ == "__main__":
    main()

