#!/usr/bin/env python3

import sqlite3
import pandas as pd
import sys

def insert_domains(csv_file, db_file):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    updated = 0

    # Insert rows
    for _, row in df.iterrows():
        cur.execute(
            """
            UPDATE exp_protein_rna
            SET
                domain_ids = ?,
                domain_names = ?,
                domain_counts = ?,
                domain_pos = ?
            WHERE PDBId = ?
            """,
            (
                row["interpro_ids"],
                row["domain_names"],
                row["domain_counts"],
                row["hit_pos"],
                row["PDBId"],
            ),
        )

        updated += cur.rowcount

    conn.commit()
    conn.close()

    print(f"Updated {updated} rows in exp")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rnaCompete/addDomains.py added_domain_infos.csv database.db")
        sys.exit(1)
    insert_domains(sys.argv[1], sys.argv[2])
