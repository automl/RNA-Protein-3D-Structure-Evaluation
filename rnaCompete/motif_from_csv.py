#!/usr/bin/env python3
"""motif_from_csv.py -- RNA-motif scores on arbitrary predictions, from a CSV.

The generalized drop-in path for the RNAcompete motif metrics: instead of the paper's
RNAcompete SQLite/TSV probe pipeline, take a plain CSV describing each prediction's RNA
sequence and reference (experimental) motif, reuse the DB-free scoring in
``rnaCompete.rnaCompete``, and write the results into the run DB's prediction table so the
downstream plotting sees them like any other metric.

CSV columns (header required):
    pred_id, rna_sequence, exp_motif [, pred_motif, interface_area, protein_length]
  - pred_id       : shared id (matches the DB join key exp_db_id, case-insensitive).
  - rna_sequence  : the predicted RNA sequence.
  - exp_motif     : the reference motif; MUST be a literal substring of rna_sequence.
  - pred_motif    : predicted motif in Base(pos) format, e.g. "U(11), G(12), C(13)".
                    If omitted, the RNAMotif already stored for this id (from the eval's
                    extractInteractions) is used.
  - interface_area, protein_length : for InterfaceArea_norm; if omitted, the values
                    already in the DB (ProteinRNAInterfaceArea / ProteinLength) are used.

Writes, on pred_<classification>, keyed by exp_db_id = UPPER(pred_id):
    RNAmotif_score, RNAMotif_inExp, InterfaceArea_norm
(the latter two columns are added if the table does not have them yet).

Run inside the RNABench env with PYTHONPATH=<repo> and a config.json pointing at the run DB:
    python rnaCompete/motif_from_csv.py --csv motifs.csv
"""

import argparse
import csv
import importlib.util
import os
import sqlite3
import sys

# The rnaCompete/ directory is not an importable package (no __init__.py) and it contains a
# module with the same name (rnaCompete.py), so `from rnaCompete.rnaCompete import ...` is
# ambiguous when this file is run as a script. Put the repo root on sys.path (so the scoring
# module's own `from database... / from plots...` imports resolve) and load rnaCompete.py by
# file path under a private module name.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_scoring_path = os.path.join(_REPO, "rnaCompete", "rnaCompete.py")
_spec = importlib.util.spec_from_file_location("_rnacompete_scoring", _scoring_path)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)
calculate_motif_score = _scoring.calculate_motif_score
calculate_motif_in_exp_score = _scoring.calculate_motif_in_exp_score
calculate_similarity_score = _scoring.calculate_similarity_score
get_motif_positions = _scoring.get_motif_positions

from database.startConfig import StartConfig

OUT_COLUMNS = ("RNAmotif_score", "RNAMotif_inExp", "InterfaceArea_norm")


def _table_columns(cur, table):
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _ensure_columns(cur, table, existing):
    """Add the score columns the schema may not have yet (RNAmotif_score is already there)."""
    for col in OUT_COLUMNS:
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} FLOAT")
            existing.add(col)


def _score_row(rna_sequence, exp_motif, pred_motif, interface_area, protein_length):
    """Return (RNAmotif_score, RNAMotif_inExp, InterfaceArea_norm) for one prediction."""
    motif_score = 0.0
    motif_in_exp = 0.0
    if rna_sequence and exp_motif and pred_motif:
        motif_score = calculate_motif_score(exp_motif, pred_motif, rna_sequence)
        start = rna_sequence.find(exp_motif)
        if start != -1:
            exp_positions = [(start + i + 1, base) for i, base in enumerate(exp_motif)]
            pred_positions = get_motif_positions(pred_motif, is_predicted=True)
            if pred_positions:
                motif_in_exp = calculate_motif_in_exp_score(exp_positions, pred_positions)

    def _f(v):
        try:
            return float(v) if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0

    norm = calculate_similarity_score(_f(interface_area), _f(protein_length))
    return motif_score, motif_in_exp, norm


def main():
    ap = argparse.ArgumentParser(description="Compute RNA-motif scores from a CSV into the run DB.")
    ap.add_argument("--csv", required=True, help="CSV with pred_id, rna_sequence, exp_motif[, ...].")
    ap.add_argument("--table", default=None,
                    help="Prediction table (default: pred_<classification> from config.json).")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"motif_from_csv: CSV not found: {args.csv}")

    config = StartConfig()
    db_path = config.get_database_path()
    table = args.table or config.get_predicted_table_name()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cols = _table_columns(cur, table)
    if not cols:
        sys.exit(f"motif_from_csv: table {table} not found in {db_path} (run parse first).")
    _ensure_columns(cur, table, cols)
    has_rnamotif = "RNAMotif" in cols
    has_area = "ProteinRNAInterfaceArea" in cols
    has_plen = "ProteinLength" in cols

    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        for req in ("pred_id", "rna_sequence", "exp_motif"):
            if req not in header:
                sys.exit(f"motif_from_csv: CSV missing required column '{req}'.")

        def cell(row, name):
            col = header.get(name)
            return (row[col].strip() if col and row[col] is not None else "")

        updated = 0
        for row in reader:
            pred_id = cell(row, "pred_id")
            if not pred_id:
                continue
            join_id = pred_id.upper()
            rna_sequence = cell(row, "rna_sequence")
            exp_motif = cell(row, "exp_motif")
            pred_motif = cell(row, "pred_motif")
            interface_area = cell(row, "interface_area")
            protein_length = cell(row, "protein_length")

            # Fall back to values already in the DB when the CSV omits them.
            if (not pred_motif and has_rnamotif) or (not interface_area and has_area) \
                    or (not protein_length and has_plen):
                sel = ["RNAMotif" if has_rnamotif else "NULL",
                       "ProteinRNAInterfaceArea" if has_area else "NULL",
                       "ProteinLength" if has_plen else "NULL"]
                cur.execute(f"SELECT {', '.join(sel)} FROM {table} "
                            f"WHERE UPPER(exp_db_id) = ?", (join_id,))
                got = cur.fetchone()
                if got:
                    pred_motif = pred_motif or (got[0] or "")
                    interface_area = interface_area or got[1]
                    protein_length = protein_length or got[2]

            score, in_exp, norm = _score_row(
                rna_sequence, exp_motif, pred_motif, interface_area, protein_length)
            cur.execute(
                f"UPDATE {table} SET RNAmotif_score = ?, RNAMotif_inExp = ?, "
                f"InterfaceArea_norm = ? WHERE UPPER(exp_db_id) = ?",
                (score, in_exp, norm, join_id))
            updated += cur.rowcount

    conn.commit()
    conn.close()
    print(f"motif_from_csv: updated {updated} row(s) in {table} ({db_path}).")


if __name__ == "__main__":
    main()
