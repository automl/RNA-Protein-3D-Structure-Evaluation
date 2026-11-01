"""
Add protein-domain annotations (domain_ids / domain_names / domain_counts / domain_pos) to the
exp_<classification> table from a user-supplied TSV.

TSV format (tab-separated, header required):
    <id-col>\tdomain_ids\tdomain_names\tdomain_counts\tdomain_pos
where <id-col> defaults to "PDBId" (use --id-col ProteinID for RNAcompete data). The four domain
cells are stored verbatim as TEXT (any internal multi-domain delimiter, e.g. ";", is the producer's
choice and is not parsed here).

Usage:
    python database/addDomain.py --tsv <domains.tsv> [--db <path>] [--table exp_protein_rna] [--id-col PDBId]

The DB defaults to the configured db_path (config.json via StartConfig); the previous hardcoded
macOS DB_PATH/TSV_PATH have been removed so this runs on any machine and honors per-run isolation.
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.startConfig import StartConfig


def add_domains(tsv_path, db_path=None, table="exp_protein_rna", id_col="PDBId"):
    db_path = db_path or StartConfig().get_database_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        for col in ("domain_ids", "domain_names", "domain_counts", "domain_pos"):
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)  # e.g. PDB 8E10 stays text
        n = 0
        for _, row in df.iterrows():
            cursor.execute(
                f"""UPDATE {table}
                    SET domain_ids = ?, domain_names = ?, domain_counts = ?, domain_pos = ?
                    WHERE PDBId = ?""",
                (row["domain_ids"], row["domain_names"], row["domain_counts"], row["domain_pos"],
                 row[id_col]),
            )
            n += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        conn.commit()
        print(f"addDomain: updated {n} rows in {table} from {tsv_path} (db={db_path})")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Add domain annotations to exp_<classification> from a TSV.")
    ap.add_argument("--tsv", required=True, help="Tab-separated domain file (see module docstring).")
    ap.add_argument("--db", default=None, help="SQLite DB path (default: config.json db_path).")
    ap.add_argument("--table", default="exp_protein_rna", help="Target exp table.")
    ap.add_argument("--id-col", default="PDBId", help="TSV column used as the PDBId join key.")
    args = ap.parse_args()
    add_domains(args.tsv, db_path=args.db, table=args.table, id_col=args.id_col)


if __name__ == "__main__":
    main()
