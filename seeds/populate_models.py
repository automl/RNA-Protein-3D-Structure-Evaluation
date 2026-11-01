#!/usr/bin/env python3
"""
populate_models.py
==================

Add rows for AF3 models 1-4 to each per-seed database, cloned from the
existing model_0 rows.

Background
----------
Each seed DB (a2021_s1.db, a2021_s2.db, a2021_s3.db) currently holds only the
top-ranked prediction (FileName = fold_<pdb>_s<seed>_model_0.cif). AF3 also
produces models 1-4 in the *same* fold folder. Both evaluation scripts we run
afterwards key their writes by **FileName** (not exp_db_id):

  * af3Metrics_new.py     -> update_or_insert(..., condition="FileName = ?")
  * extractEvalMetrics.py -> UPDATE pred_protein_rna ... WHERE FileName = ?

`update_or_insert` only UPDATEs an existing row (and a bare INSERT would leave
the identity columns NULL), so the model_1-4 rows must already exist, with the
identity columns filled in, before those scripts run. This script creates them.

Because FileName is globally unique (it embeds pdb id + seed + model), keeping
all five models in one per-seed DB is unambiguous even though exp_db_id repeats.
No change to af3Metrics_new.py / extractEvalMetrics.py is needed.

What it copies
--------------
For every model_0 row it inserts one row per target model (default 1-4) with:
  * COPY_COLUMNS copied verbatim from the model_0 row,
  * FileName rewritten   _model_0.cif -> _model_<N>.cif,
  * Model set to <N>,
  * every per-prediction metric column left NULL (filled later by the two
    evaluation scripts).

Idempotent: a target FileName that already exists is skipped.
Dry run by default; pass --apply to write.

Usage
-----
    python populate_models.py                 # dry run, all three seed DBs
    python populate_models.py --apply         # actually insert
    python populate_models.py --db /path/a2021_s1.db --apply
    python populate_models.py --models 1,2,3,4 --apply
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# --- Configuration ----------------------------------------------------------
SEEDS_DIR = Path(__file__).resolve().parents[1] / "data" / "seeds" / "Seeds"
DEFAULT_DBS = [SEEDS_DIR / f"a2021_s{n}.db" for n in (1, 2, 3)]
TABLE = "pred_protein_rna"
SOURCE_MODEL = 0
DEFAULT_TARGET_MODELS = [1, 2, 3, 4]

# Identity / annotation columns copied verbatim from the model_0 row.
# (Exactly the list specified for the task. These are model-independent.)
COPY_COLUMNS = [
    "exp_db_id",
    "FileName",          # rewritten per target model below
    "ProteinSequence",
    "ProteinChainIDs",
    "ProteinLength",
    "NumberProteins",
    "RNASequence",
    "RNAChainIDs",
    "RNALength",
    "NumberRNAs",
]


def model_filename(file_name_model0: str, target_model: int) -> str:
    """fold_7d8o_s1_model_0.cif -> fold_7d8o_s1_model_<N>.cif"""
    needle = f"_model_{SOURCE_MODEL}.cif"
    if not file_name_model0.endswith(needle):
        raise ValueError(
            f"Unexpected FileName (no '{needle}' suffix): {file_name_model0}"
        )
    return file_name_model0[: -len(needle)] + f"_model_{target_model}.cif"


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def populate_db(db_path: Path, target_models: list[int], apply: bool) -> None:
    print(f"\n=== {db_path} ===")
    if not db_path.exists():
        print(f"  SKIP: file not found")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = table_columns(conn, TABLE)
        has_model_col = "Model" in cols
        missing = [c for c in COPY_COLUMNS if c not in cols]
        if missing:
            print(f"  ERROR: table missing expected columns: {missing}")
            return

        # Existing FileNames (to skip already-present rows = idempotent).
        existing = {
            r[0] for r in conn.execute(f"SELECT FileName FROM {TABLE}")
        }

        source_rows = conn.execute(
            f"SELECT * FROM {TABLE} WHERE FileName LIKE '%\\_model\\_{SOURCE_MODEL}.cif' ESCAPE '\\'"
        ).fetchall()
        print(f"  model_{SOURCE_MODEL} source rows: {len(source_rows)}")

        insert_cols = list(COPY_COLUMNS)
        if has_model_col:
            insert_cols.append("Model")
        placeholders = ", ".join("?" for _ in insert_cols)
        col_list = ", ".join(insert_cols)
        insert_sql = f"INSERT INTO {TABLE} ({col_list}) VALUES ({placeholders})"

        to_insert = 0
        skipped = 0
        for row in source_rows:
            for m in target_models:
                new_fn = model_filename(row["FileName"], m)
                if new_fn in existing:
                    skipped += 1
                    continue
                values = []
                for c in COPY_COLUMNS:
                    values.append(new_fn if c == "FileName" else row[c])
                if has_model_col:
                    values.append(m)
                to_insert += 1
                if apply:
                    conn.execute(insert_sql, values)
                    existing.add(new_fn)
                else:
                    print(f"  [dry run] + {new_fn}")

        if apply:
            conn.commit()

        verb = "Inserted" if apply else "Would insert"
        print(
            f"  {verb} {to_insert} row(s) "
            f"(models {target_models}); skipped {skipped} already-present."
        )
        if apply:
            total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            print(f"  Table now holds {total} row(s).")
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, action="append", default=None,
                    help="Seed DB to process (repeatable). "
                         "Default: a2021_s1/2/3.db in the Seeds folder.")
    ap.add_argument("--models", type=str, default="1,2,3,4",
                    help="Comma-separated target model numbers (default 1,2,3,4).")
    ap.add_argument("--apply", action="store_true",
                    help="Actually insert rows (default is a dry run).")
    args = ap.parse_args()

    dbs = args.db if args.db else DEFAULT_DBS
    try:
        target_models = [int(x) for x in args.models.split(",") if x.strip() != ""]
    except ValueError:
        print(f"ERROR: bad --models value: {args.models!r}", file=sys.stderr)
        return 1
    if not target_models:
        print("ERROR: no target models given.", file=sys.stderr)
        return 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"populate_models.py [{mode}] target models {target_models}")
    for db in dbs:
        populate_db(Path(db), target_models, args.apply)

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to insert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
