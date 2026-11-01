#!/usr/bin/env python3
"""
Load RBP_PDB_Templates and Protein_PDB_Templates into pred_protein_rna from a CSV.

CSV must include:
  - full_path  (used to derive exp_db_id)
  - RBP_PDB_Templates and/or Protein_PDB_Templates

exp_db_id mapping (rnacompete):
  full_path: .../rncmpt00001maxgauaauurbd_v3_002261_data.json
  → basename without _data.json → rncmpt00001maxgauaauurbd_v3_002261
  → remove all '_' → uppercase → RNCMPT00001MAXGAAUUURBDV3002261  (= pred_protein_rna.exp_db_id)

Usage:
  python rnacompete/update_pred_templates_from_csv.py /path/to/templates.csv
  python rnacompete/update_pred_templates_from_csv.py /path/to/templates.csv --overwrite
  python rnacompete/update_pred_templates_from_csv.py /path/to/templates.csv --db /path/to/rbpDatabase.db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.loadTemplatesFromCSV import (
    add_column_if_not_exists,
    parse_template_column,
    update_templates_in_db,
)
from database.startConfig import StartConfig


def exp_db_id_from_full_path(full_path: str) -> str:
    """
    Map rnacompete full_path to pred_protein_rna.exp_db_id.

    Example:
      .../rncmpt00001maxgauaauurbd_v3_002261_data.json
      → RNCMPT00001MAXGAAUUURBDV3002261
    """
    if full_path is None or (isinstance(full_path, float) and pd.isna(full_path)):
        return ""
    path = str(full_path).strip()
    if not path:
        return ""
    base = os.path.basename(path.replace("\\", "/"))
    if base.endswith("_data.json"):
        stem = base[: -len("_data.json")]
    elif base.endswith(".json"):
        stem = base[: -len(".json")]
    else:
        stem = os.path.splitext(base)[0]
    return stem.replace("_", "").upper()


def process_csv(
    csv_path: Path,
    db_path: str,
    pred_table: str,
    *,
    overwrite: bool = False,
) -> int:
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return 1

    if "full_path" not in df.columns:
        print(f"Error: CSV must contain 'full_path'. Columns: {df.columns.tolist()}")
        return 1

    has_rbp = "RBP_PDB_Templates" in df.columns
    has_protein = "Protein_PDB_Templates" in df.columns
    if not (has_rbp or has_protein):
        print("Error: CSV needs RBP_PDB_Templates and/or Protein_PDB_Templates")
        return 1

    if has_rbp:
        add_column_if_not_exists(db_path, pred_table, "RBP_PDB_Templates")
    if has_protein:
        add_column_if_not_exists(db_path, pred_table, "Protein_PDB_Templates")

    import sqlite3

    existing_ids = set()
    if not overwrite:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            conditions = []
            if has_rbp:
                conditions.append(
                    "(RBP_PDB_Templates IS NOT NULL AND RBP_PDB_Templates != '' "
                    "AND RBP_PDB_Templates != '[]')"
                )
            if has_protein:
                conditions.append(
                    "(Protein_PDB_Templates IS NOT NULL AND Protein_PDB_Templates != '' "
                    "AND Protein_PDB_Templates != '[]')"
                )
            if conditions:
                query = (
                    f"SELECT exp_db_id FROM {pred_table} WHERE {' OR '.join(conditions)}"
                )
                cursor.execute(query)
                existing_ids = {row[0] for row in cursor.fetchall()}
                print(f"Skipping {len(existing_ids)} rows with existing templates (use --overwrite)")
        finally:
            conn.close()

    successful = skipped = failed = not_found = invalid = 0

    for idx, row in df.iterrows():
        eid = exp_db_id_from_full_path(row.get("full_path"))
        if not eid:
            print(f"Row {idx + 1}: empty/invalid full_path, skipping")
            invalid += 1
            continue

        if not overwrite and eid in existing_ids:
            skipped += 1
            continue

        rbp_ids = parse_template_column(row.get("RBP_PDB_Templates")) if has_rbp else None
        protein_ids = (
            parse_template_column(row.get("Protein_PDB_Templates")) if has_protein else None
        )

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {pred_table} WHERE exp_db_id = ?",
                (eid,),
            )
            if cursor.fetchone()[0] == 0:
                print(f"✗ exp_db_id '{eid}' not in {pred_table} (from {row.get('full_path')})")
                not_found += 1
                continue

            update_templates_in_db(db_path, pred_table, eid, rbp_ids, protein_ids)
            parts = []
            if rbp_ids is not None:
                parts.append(f"{len(rbp_ids)} RBP")
            if protein_ids is not None:
                parts.append(f"{len(protein_ids)} Protein")
            print(f"✓ {eid}: {', '.join(parts) or 'empty templates'}")
            successful += 1
        except Exception as exc:
            print(f"✗ {eid}: {exc}")
            failed += 1
        finally:
            conn.close()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Updated:    {successful}")
    print(f"  Skipped:    {skipped}")
    print(f"  Not in DB:  {not_found}")
    print(f"  Invalid:    {invalid}")
    print(f"  Failed:     {failed}")
    return 0 if failed == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Update pred_protein_rna template columns from rnacompete CSV (full_path → exp_db_id)."
    )
    parser.add_argument("csv_path", help="Path to CSV with full_path and template columns")
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument("--table", default="pred_protein_rna", help="Prediction table (default: pred_protein_rna)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite rows that already have template values",
    )
    args = parser.parse_args(argv)

    cfg = StartConfig()
    db_path = os.path.abspath(args.db or cfg.get_database_path())
    csv_path = Path(args.csv_path).expanduser().resolve()

    print(f"CSV:   {csv_path}")
    print(f"DB:    {db_path}")
    print(f"Table: {args.table}")
    return process_csv(csv_path, db_path, args.table, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
