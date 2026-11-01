#!/usr/bin/env python
"""Split the 4 combined AF3-evaluation databases into the 8 files the rest of the
preprocessing pipeline (run_preprocessing.sh) consumes.

Future data may arrive as combined per-era databases rather than pre-split by
experimental method, so this is the dedicated upstream step that produces the
method-split (X-ray vs non-X-ray) databases on demand.

Inputs (defaults; override via CLI):
  data/post_pre/a2021_b2021_protein_rna/{a2021,b2021}.db   (tables pred/exp_protein_rna)
  data/post_pre/a2021_b2021_rna_only/{a2021_rna,b2021_rna}.db (tables pred/exp_rna_rna)

Outputs into --outdir (default data/):
  a2021.db, b2021.db                         (complex-level, copied verbatim)
  a2021x.db, a2021nonX.db,
  b2021x.db, b2021nonX.db                    (split on exp_protein_rna.Experiment)
  a2021_rnaOnly_with_af3.db,
  b2021_rnaOnly_with_af3.db                  (RNA-only, copied/renamed verbatim)

Method split: X-ray  = Experiment contains 'X-RAY' (case-insensitive);
              non-X-ray = everything else (electron microscopy, NMR, ...).
The split is done by copying the source verbatim (schema, indexes, extra columns
all preserved) and deleting the rows that do not belong, so the split files are
byte-for-schema identical to the source.
"""
import argparse
import os
import shutil
import sqlite3
import sys

XRAY_LIKE = "%X-RAY%"


def _counts(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    out = {}
    for (t,) in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        out[t] = cur.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
    con.close()
    return out


def copy_verbatim(src, dst):
    shutil.copyfile(src, dst)
    print(f"  copied  {os.path.basename(src)} -> {os.path.basename(dst)}  {_counts(dst)}")


def split_by_experiment(src, dst_xray, dst_nonxray):
    """Write dst_xray (X-ray only) and dst_nonxray (everything else) from src."""
    # sanity: pred and exp must reference the same structures. The join key is the
    # PDB entry (pred_protein_rna.exp_db_id <-> exp_protein_rna.PDBId), which is what
    # every downstream preprocessing script merges on. The integer row Id is NOT a
    # join key: the two tables are numbered independently (in the 2026-10-13 dbs
    # pred Ids are consecutive while exp Ids keep gaps from deleted entries).
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    pred_keys = {r[0] for r in con.execute("SELECT exp_db_id FROM pred_protein_rna")}
    exp_keys = {r[0] for r in con.execute("SELECT PDBId FROM exp_protein_rna")}
    con.close()
    if pred_keys != exp_keys:
        sys.exit(
            f"ERROR: pred/exp PDB sets differ in {src} "
            f"(pred-only={sorted(pred_keys - exp_keys)[:5]}, exp-only={sorted(exp_keys - pred_keys)[:5]}); "
            "join key assumption violated, aborting."
        )

    for dst, keep_xray in ((dst_xray, True), (dst_nonxray, False)):
        shutil.copyfile(src, dst)
        con = sqlite3.connect(dst)
        # PDB entries to KEEP in this split
        if keep_xray:
            keep_sql = "SELECT PDBId FROM exp_protein_rna WHERE upper(Experiment) LIKE ?"
        else:
            keep_sql = "SELECT PDBId FROM exp_protein_rna WHERE upper(Experiment) NOT LIKE ? OR Experiment IS NULL"
        # delete pred rows not kept (evaluate against exp BEFORE trimming exp)
        con.execute(
            f"DELETE FROM pred_protein_rna WHERE exp_db_id NOT IN ({keep_sql})", (XRAY_LIKE,)
        )
        # delete exp rows not kept
        if keep_xray:
            con.execute(
                "DELETE FROM exp_protein_rna WHERE upper(Experiment) NOT LIKE ? OR Experiment IS NULL",
                (XRAY_LIKE,),
            )
        else:
            con.execute(
                "DELETE FROM exp_protein_rna WHERE upper(Experiment) LIKE ?", (XRAY_LIKE,)
            )
        con.commit()
        con.execute("VACUUM")
        con.commit()
        con.close()
        print(f"  split   {os.path.basename(src)} -> {os.path.basename(dst)}  {_counts(dst)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="data/post_pre", help="dir holding the 6 combined dbs")
    ap.add_argument("--outdir", default="data", help="dir to write the 10 pipeline dbs")
    ap.add_argument("--a_pr", default="a2021_b2021_protein_rna/a2021.db")
    ap.add_argument("--b_pr", default="a2021_b2021_protein_rna/b2021.db")
    ap.add_argument("--a_rna", default="a2021_b2021_rna_only/a2021_rna.db")
    ap.add_argument("--b_rna", default="a2021_b2021_rna_only/b2021_rna.db")
    ap.add_argument("--a_rr", default="a2021_b2021_rna_rna/a2021_rna_rna.db")
    ap.add_argument("--b_rr", default="a2021_b2021_rna_rna/b2021_rna_rna.db")
    args = ap.parse_args()

    j = lambda *p: os.path.join(*p)
    a_pr = j(args.indir, args.a_pr)
    b_pr = j(args.indir, args.b_pr)
    a_rna = j(args.indir, args.a_rna)
    b_rna = j(args.indir, args.b_rna)
    a_rr = j(args.indir, args.a_rr)
    b_rr = j(args.indir, args.b_rr)
    # Split only the source dbs that are present, so a partial database family (e.g. only
    # protein-RNA, or only the post-cutoff era) still produces its pipeline dbs instead of
    # aborting. Absent sources are skipped-and-logged; all four missing is a hard error.
    absent = [p for p in (a_pr, b_pr, a_rna, b_rna) if not os.path.exists(p)]
    if len(absent) == 4:
        sys.exit(f"ERROR: none of the 4 source dbs found under {args.indir}")
    for p in absent:
        print(f"  note: source db absent -> skipping its outputs: {p}")
    os.makedirs(args.outdir, exist_ok=True)
    o = lambda name: j(args.outdir, name)

    if os.path.exists(a_pr):
        print("Complex-level (verbatim) + method split (X-ray vs non-X-ray): a2021")
        copy_verbatim(a_pr, o("a2021.db"))
        split_by_experiment(a_pr, o("a2021x.db"), o("a2021nonX.db"))
    if os.path.exists(b_pr):
        print("Complex-level (verbatim) + method split (X-ray vs non-X-ray): b2021")
        copy_verbatim(b_pr, o("b2021.db"))
        split_by_experiment(b_pr, o("b2021x.db"), o("b2021nonX.db"))
    if os.path.exists(a_rna):
        print("RNA-only (verbatim rename): a2021")
        copy_verbatim(a_rna, o("a2021_rnaOnly_with_af3.db"))
    if os.path.exists(b_rna):
        print("RNA-only (verbatim rename): b2021")
        copy_verbatim(b_rna, o("b2021_rnaOnly_with_af3.db"))

    if os.path.exists(a_rr):
        print("Multi-chain RNA (verbatim copy): a2021")
        copy_verbatim(a_rr, o("a2021_rna_rna.db"))
    if os.path.exists(b_rr):
        print("Multi-chain RNA (verbatim copy): b2021")
        copy_verbatim(b_rr, o("b2021_rna_rna.db"))

    print("\nDone. pipeline dbs written to", args.outdir)


if __name__ == "__main__":
    main()
