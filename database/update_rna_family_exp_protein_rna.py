#!/usr/bin/env python3
"""
Update only the RNAFamily column in exp_protein_rna for all rows that have RNASequence.

Uses the same Rfam annotation as parsePDBCIF2DB (utils.annotate_rna_family.get_rna_family_list).
Database path is taken from database.startConfig.StartConfig.

Usage:
  python database/update_rna_family_exp_protein_rna.py
"""

import json
import logging
import os
import sqlite3

from database.startConfig import StartConfig
from utils.annotate_rna_family import get_rna_family_list

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _sequences_from_db_value(rnasequence_value):
    """Turn RNASequence TEXT from DB into a list of sequences (one per RNA chain)."""
    if rnasequence_value is None or (isinstance(rnasequence_value, str) and not rnasequence_value.strip()):
        return []
    s = rnasequence_value.strip()
    if s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return [s]
    return [s]


def _family_value_for_db(family_list):
    """Format RNA family list for DB: single string or JSON list, matching parsePDBCIF2DB."""
    if not family_list:
        return None
    if len(family_list) == 1:
        return family_list[0]
    return json.dumps(family_list)


def update_rna_family_column(db_path: str, rfam_dir: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT PDBId, RNASequence FROM exp_protein_rna WHERE RNASequence IS NOT NULL AND RNASequence != ''"
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.info("No rows with RNASequence in exp_protein_rna")
        return

    logger.info("Updating RNAFamily for %d rows in exp_protein_rna", len(rows))

    updated = 0
    errors = 0
    for row in rows:
        pdb_id = row["PDBId"]
        sequences = _sequences_from_db_value(row["RNASequence"])
        if not sequences:
            continue
        try:
            family_list = get_rna_family_list(sequences, rfam_dir)
            family_db_value = _family_value_for_db(family_list)
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE exp_protein_rna SET RNAFamily = ? WHERE PDBId = ?", (family_db_value, pdb_id))
            conn.commit()
            conn.close()
            updated += 1
        except Exception as e:
            errors += 1
            logger.warning("Failed %s: %s", pdb_id, e)

    logger.info("Done. Updated: %d, Errors: %d", updated, errors)


def main():
    config = StartConfig()
    db_path = config.get_database_path()
    if not db_path:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "rbpDatabase.db")
    if not os.path.isfile(db_path):
        logger.error("Database not found: %s", db_path)
        return 1

    project_root = os.path.dirname(os.path.dirname(__file__))
    rfam_dir = os.path.join(project_root, "evaluation", "data", "rfam")
    if not os.path.isdir(rfam_dir):
        logger.error("Rfam directory not found: %s", rfam_dir)
        return 1

    update_rna_family_column(db_path, rfam_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
