#!/usr/bin/env python
"""Prepare the RNAcompete DBs for figure generation (idempotent).

The two RNAcompete SQLite DBs (rnaCompete_v4.db = per-probe ranking set;
rnaCompete_v5_MINMAX.db = maxprobe-vs-minprobe competition set) ship without the
two derived artefacts the figure scripts expect:

  1. ``exp_protein_rna.csv`` -- sequence -> Class lookup used by
     ``exp_probeMax_combined.py`` (exported here from the ``exp_protein_rna`` table).
  2. the ``af3_chain_pair_pae_min_perchain`` column on ``pred_protein_rna`` -- the
     per-RNA-chain min chain-pair PAE. It is normally populated by
     ``evaluation/repopulate_chain_pair_pae_min.py`` from the raw AF3
     ``summary_confidences`` JSONs; when those JSONs are not present the figure only
     needs the column to EXIST (values stay NULL and the per-chain PAE metric is
     skipped gracefully). We add it as NULL so the competition figure still renders.

Run:  python rnaCompete/prep_rnacompete_dbs.py [--dir data/rnacompete]
Read-mostly: only ALTER TABLE ADD COLUMN (NULL) + a CSV export; no row edits.
"""
import argparse
import csv
import os
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

CSV_COLS = ["PDBId", "RNASequence", "Class", "BindingIntensity",
            "RNAMotif", "RNAMotif_z_score", "RNAMotifLength", "RNALength",
            "domain_ids", "domain_names"]
PERCHAIN_COL = "af3_chain_pair_pae_min_perchain"


def export_exp_csv(db_path, out_csv):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    have = [c[1] for c in cur.execute("PRAGMA table_info(exp_protein_rna)").fetchall()]
    cols = [c for c in CSV_COLS if c in have]
    rows = cur.execute(f"SELECT {','.join(cols)} FROM exp_protein_rna").fetchall()
    con.close()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"[prep] wrote {out_csv} ({len(rows)} rows, cols={cols})")


def ensure_perchain_column(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cols = [c[1] for c in cur.execute("PRAGMA table_info(pred_protein_rna)").fetchall()]
    if PERCHAIN_COL not in cols:
        cur.execute(f"ALTER TABLE pred_protein_rna ADD COLUMN {PERCHAIN_COL} TEXT")
        con.commit()
        print(f"[prep] added NULL column {PERCHAIN_COL} to {os.path.basename(db_path)} "
              f"(populate via evaluation/repopulate_chain_pair_pae_min.py if AF3 JSONs are available)")
    else:
        print(f"[prep] {os.path.basename(db_path)}: {PERCHAIN_COL} already present")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(_REPO, "data", "rnacompete"),
                    help="Directory holding rnaCompete_v4.db + rnaCompete_v5_MINMAX.db")
    args = ap.parse_args()
    v4 = os.path.join(args.dir, "rnaCompete_v4.db")
    v5 = os.path.join(args.dir, "rnaCompete_v5_MINMAX.db")
    if not (os.path.exists(v4) and os.path.exists(v5)):
        raise SystemExit(f"[prep] RNAcompete DBs not found in {args.dir} "
                         f"(need rnaCompete_v4.db + rnaCompete_v5_MINMAX.db)")
    # CSV export from the competition DB (both DBs carry the same exp_protein_rna table)
    export_exp_csv(v5, os.path.join(args.dir, "exp_protein_rna.csv"))
    # per-chain PAE column on the competition DB (needed by exp_probeMax_combined.py)
    ensure_perchain_column(v5)


if __name__ == "__main__":
    main()
