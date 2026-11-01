#!/usr/bin/env python3
"""
distribute_model_files.py
=========================

Flatten the per-fold AF3 output folders under

    data/seeds/all/fold_<pdb>_s<N>/

into one `af` folder per seed:

    data/seeds/all/seed1/af/
    data/seeds/all/seed2/af/
    data/seeds/all/seed3/af/

A fold folder is routed by its `_s1` / `_s2` / `_s3` suffix
(e.g. fold_6fql_s1 -> seed1).

Every file in a fold folder is taken EXCEPT the model_0 outputs:
    * <name>_model_0.cif        (excluded)
    * <name>_*_0.json           (excluded: full_data_0.json, summary_confidences_0.json)
Everything else is included, i.e. the model_1-4 predictions
(`_model_[1-4].cif`, `_full_data_[1-4].json`, `_summary_confidences_[1-4].json`)
plus per-fold side files (`_job_request.json`, `terms_of_use.md`).

model_0 is left in place because it has already been evaluated.

The destination `seed1/seed2/seed3` folders live inside `all/`, so they are
never treated as source fold folders.

Default = COPY, and a DRY RUN. Pass --apply to act, --move to move instead of
copy. Re-running is safe: a destination file that already exists is skipped.

Usage
-----
    python distribute_model_files.py                 # dry run (copy plan)
    python distribute_model_files.py --apply         # copy files
    python distribute_model_files.py --apply --move  # move files
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ALL_DIR = Path(__file__).resolve().parents[1] / "data" / "seeds" / "all"

# fold_<pdb>_s<N>  ->  capture the seed number
FOLD_RE = re.compile(r"^fold_.+_s([0-9]+)$")

# Files to exclude (the model_0 outputs).
EXCLUDE_RE = re.compile(r"(_model_1\.cif|_1\.json)$")


def is_excluded(name: str) -> bool:
    return EXCLUDE_RE.search(name) is not None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all-dir", type=Path, default=ALL_DIR,
                    help=f"Folder containing the fold_* dirs (default: {ALL_DIR}).")
    ap.add_argument("--apply", action="store_true",
                    help="Actually transfer files (default is a dry run).")
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copying them.")
    args = ap.parse_args()

    all_dir: Path = args.all_dir
    if not all_dir.is_dir():
        print(f"ERROR: not a directory: {all_dir}", file=sys.stderr)
        return 1

    action = "move" if args.move else "copy"
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"distribute_model_files.py [{mode}] action={action}")
    print(f"Source: {all_dir}\n")

    transferred = 0
    excluded = 0
    skipped_existing = 0
    folders = 0
    per_seed: dict[str, int] = {}

    for entry in sorted(all_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = FOLD_RE.match(entry.name)
        if not m:
            continue  # skips seed1/seed2/seed3 destinations and anything else
        folders += 1
        seed_dir = f"seed{m.group(1)}"
        dest_af = all_dir / seed_dir / "af"
        if args.apply:
            dest_af.mkdir(parents=True, exist_ok=True)

        for f in sorted(entry.iterdir()):
            if not f.is_file():
                continue
            if is_excluded(f.name):
                excluded += 1
                continue
            dest = dest_af / f.name
            if dest.exists():
                skipped_existing += 1
                continue
            transferred += 1
            per_seed[seed_dir] = per_seed.get(seed_dir, 0) + 1
            if args.apply:
                if args.move:
                    shutil.move(str(f), str(dest))
                else:
                    shutil.copy2(f, dest)
            else:
                print(f"  [{action}] {entry.name}/{f.name} -> {seed_dir}/af/")

    verb = "Transferred" if args.apply else f"Would {action}"
    print()
    print(f"Fold folders processed: {folders}")
    print(f"{verb}: {transferred} file(s)")
    for seed_dir in sorted(per_seed):
        print(f"    {seed_dir}/af : {per_seed[seed_dir]}")
    print(f"Excluded (model_0):     {excluded}")
    print(f"Skipped (already there):{skipped_existing}")
    if not args.apply:
        print("\nDry run complete. Re-run with --apply (add --move to move).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
