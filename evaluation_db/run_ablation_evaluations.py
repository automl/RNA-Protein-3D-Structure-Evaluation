#!/usr/bin/env python3
"""Evaluate the 2 x 16 MSA/template-ablation condition folders.

Drives evaluation/mainExecution.py once per condition folder.

Per condition folder (<side>/<condition>/protein_rna):
  1. af/ files are renamed to the pipeline's convention fold_<id>_s1_* (parseAFcif2DB reads the
     id as parts[1].upper(), parseAFcif2DB.py:123) and the original names are recorded in
     af/af_renamed_from.tsv, so the rename is reversible.
  2. rbpDatabase.db (exp_protein_rna copied from a2021.db) and pdb/ must already be in place;
     seed_<side>.db is restored if the folder database is missing.
  3. python evaluation/mainExecution.py -dssr -pdbparse -noRBP --folder <folder>
     -pdbparse keeps the seeded database and skips parsePDBCIF2DB (the a2021 pdbs are already
     processed); -noRBP skips the remote SWISS-MODEL step.

The working database database/rbpDatabase.db is removed before each run so checkout_db() copies
the folder's own database in rather than hitting its ownership check. Each run ends with
checkin_db() writing the result back to <folder>/rbpDatabase.db.

Resumable: a folder whose database already holds predictions is skipped unless --force.
"""
import argparse
import concurrent.futures
import json
import itertools
import os
import shutil
import sqlite3
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(REPO, "data/nar_msa_template_ablation_20261007")
BC = os.path.join(AB, "by_condition")
WORK_DB = os.path.join(REPO, "database", "rbpDatabase.db")
MAIN = os.path.join(REPO, "evaluation", "mainExecution.py")
CONDS = ["%srMSA_%spMSA_%spTemp_%scTemp" % s for s in itertools.product("+-", repeat=4)]
SUFFIXES = {"_model_0.cif": "_s1_model_0.cif",
            "_full_data_0.json": "_s1_full_data_0.json",
            "_summary_confidences_0.json": "_s1_summary_confidences_0.json"}


def restore_quarantined(folder):
    """extractInteractions (and parsePDBCIF2DB) move a structure to <folder>/check_pdb when they
    raise on it -- including when a run is interrupted mid-file. Those models would then be
    invisible to the next run, which would silently score fewer structures, so put them back.
    """
    check = os.path.join(folder, "check_pdb")
    af = os.path.join(folder, "af")
    if not os.path.isdir(check):
        return 0
    moved = 0
    for fn in os.listdir(check):
        if not fn.endswith(tuple(SUFFIXES) + tuple(SUFFIXES.values())):
            continue
        dst = os.path.join(af, fn)
        if os.path.exists(dst):
            os.remove(os.path.join(check, fn))
        else:
            shutil.move(os.path.join(check, fn), dst)
        moved += 1
    return moved


def clean_af(folder):
    """Drop artefacts an earlier run wrote into af/ (interaction lists, US-align output), so a
    rerun cannot mix files named before and after the rename. The three AF3 output files and the
    rename log are kept."""
    af = os.path.join(folder, "af")
    removed = 0
    for fn in os.listdir(af):
        if fn == "af_renamed_from.tsv" or any(fn.endswith(s) for s in SUFFIXES.values()):
            continue
        if any(fn.endswith(s) for s in SUFFIXES):
            continue
        os.remove(os.path.join(af, fn))
        removed += 1
    return removed


def rename_af(folder):
    """af/<ID>_<condition><suffix>  ->  af/fold_<id>_s1<suffix>; returns (renamed, already)."""
    af = os.path.join(folder, "af")
    tsv = os.path.join(af, "af_renamed_from.tsv")
    mapping, renamed, already = [], 0, 0
    for fn in sorted(os.listdir(af)):
        if fn.startswith("fold_"):
            already += 1
            continue
        for old_suf, new_suf in SUFFIXES.items():
            if fn.endswith(old_suf):
                pdb_id = fn.split("_")[0]
                new = "fold_%s%s" % (pdb_id.lower(), new_suf)
                os.rename(os.path.join(af, fn), os.path.join(af, new))
                mapping.append((fn, new))
                renamed += 1
                break
    if mapping:
        header = not os.path.exists(tsv)
        with open(tsv, "a") as fh:
            if header:
                fh.write("original\trenamed\n")
            for a, b in mapping:
                fh.write("%s\t%s\n" % (a, b))
    return renamed, already


def n_pred(db):
    if not os.path.exists(db):
        return -1
    try:
        con = sqlite3.connect(db)
        n = con.execute("select count(*) from pred_protein_rna").fetchone()[0]
        con.close()
        return n
    except sqlite3.Error:
        return -1


def run_folder(side, cond, args):
    folder = os.path.join(BC, side, cond, "protein_rna")
    db = os.path.join(folder, "rbpDatabase.db")
    seed = os.path.join(AB, "seed_%s.db" % side)
    label = "%s/%s" % (side, cond)

    if not os.path.isdir(folder):
        return label, "MISSING_FOLDER", 0
    if n_pred(db) > 0 and not args.force:
        return label, "SKIPPED (already has %d predictions)" % n_pred(db), 0
    if not os.path.exists(db) or args.force:
        if not os.path.exists(seed):
            return label, "NO_SEED_DB", 0
        shutil.copy2(seed, db)

    restored = restore_quarantined(folder)
    scrubbed = clean_af(folder)
    renamed, already = rename_af(folder)
    print("  af/: renamed %d, already named %d, removed %d stale artefacts, "
          "restored %d from check_pdb" % (renamed, already, scrubbed, restored), flush=True)

    cmd = ["conda", "run", "--no-capture-output", "-n", args.env,
           "python", MAIN, "-dssr", "-pdbparse", "-noRBP", "--folder", folder]
    # ost_calcRNA_AllChains runs on both sides: its 1v1 shortcut already reuses the
    # whole-structure scores for single-protein/single-RNA complexes, so nothing is wasted,
    # and rna_side keeps its per-interface iLDDT/ICS (2-15 protein chains per complex).
    if args.no_allchains:
        cmd.insert(-2, "-noAllChains")
    # extractEvalMetrics calls US-align as "./USalign/USalign" (a relative path), so the child
    # processes must run from evaluation/ or every TM computation fails and Complex_TM / RNA_TM
    # are left empty. mainExecution resolves everything else from __file__, so this is safe.
    cwd = os.path.join(REPO, "evaluation")

    # Concurrent runs must not share the working database or the config file that every child
    # script reads. EVAL_WORK_DB (mainExecution) and EVAL_CONFIG_NAME (startConfig) give each run
    # its own pair; subprocess.run inherits os.environ, so the ost/conda children see them too.
    slot = "%s_%s" % (side[:4], cond.replace("+", "p").replace("-", "m"))
    env = dict(os.environ)
    env["EVAL_WORK_DB"] = os.path.join(REPO, "database", "work_%s.db" % slot)
    env["EVAL_CONFIG_NAME"] = "config_%s.json" % slot
    for stale in (env["EVAL_WORK_DB"], env["EVAL_WORK_DB"] + ".folder"):
        if os.path.exists(stale):
            os.remove(stale)
    # StartConfig.load_config() only sets parent_folder when the config file already exists, and
    # mainExecution reads get_parent_folder() before writing it, so seed the file up front.
    with open(os.path.join(REPO, "database", env["EVAL_CONFIG_NAME"]), "w") as fh:
        json.dump({"classification": os.path.basename(folder),
                   "base_data_path": os.path.dirname(folder),
                   "db_path": env["EVAL_WORK_DB"]}, fh, indent=4)

    t0 = time.time()
    log = os.path.join(folder, "mainExecution.log")
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    n = n_pred(db)
    status = "OK  %d predictions" % n if rc == 0 and n > 0 else "FAILED rc=%s predictions=%s" % (rc, n)
    return label, status, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", nargs="+", default=["protein_side", "rna_side"])
    ap.add_argument("--conditions", nargs="+", default=CONDS)
    ap.add_argument("--env", default="RNABench", help="conda env for mainExecution (default RNABench)")
    ap.add_argument("--jobs", "-j", type=int, default=8,
                    help="condition folders to run concurrently (default 8; each run is ~1 core)")
    ap.add_argument("--no-allchains", action="store_true",
                    help="skip ost_calcRNA_AllChains entirely (by default it runs, using its 1v1 shortcut)")
    ap.add_argument("--force", action="store_true", help="rerun folders that already have predictions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [(s, c) for s in args.sides for c in args.conditions]
    print("%d condition folders to process\n" % len(jobs))
    if args.dry_run:
        for s, c in jobs:
            f = os.path.join(BC, s, c, "protein_rna")
            print("  %-13s %-28s db=%s af=%d pdb=%d" % (
                s, c, n_pred(os.path.join(f, "rbpDatabase.db")),
                len(os.listdir(os.path.join(f, "af"))) if os.path.isdir(os.path.join(f, "af")) else 0,
                len(os.listdir(os.path.join(f, "pdb"))) if os.path.isdir(os.path.join(f, "pdb")) else 0))
        return 0

    results = []
    print("running %d at a time\n" % args.jobs, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_folder, s_, c_, args): (s_, c_) for s_, c_ in jobs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            side, cond = futures[fut]
            try:
                label, status, dt = fut.result()
            except Exception as exc:                      # keep the batch going
                label, status, dt = "%s/%s" % (side, cond), "EXCEPTION %s" % exc, 0.0
            print("[%d/%d] %-13s %-28s -> %s  (%.1f min)"
                  % (i, len(jobs), side, cond, status, dt / 60), flush=True)
            results.append((label, status))

    print("=" * 72)
    ok = [r for r in results if r[1].startswith("OK")]
    print("done: %d OK, %d skipped, %d failed" % (
        len(ok), len([r for r in results if r[1].startswith("SKIPPED")]),
        len([r for r in results if not r[1].startswith(("OK", "SKIPPED"))])))
    for label, status in results:
        if not status.startswith(("OK", "SKIPPED")):
            print("  FAILED  %s: %s" % (label, status))
    return 0 if len(ok) + len([r for r in results if r[1].startswith("SKIPPED")]) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
