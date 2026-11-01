"""
Repopulate per-RNA-chain chain_pair_pae_min from AF3 summary_confidences JSONs.

Background
----------
AF3 emits ``chain_pair_pae_min`` as a square [num_chains, num_chains] matrix: element
(i, j) is the lowest PAE across rows restricted to chain i and columns restricted to
chain j. Per the AlphaFold Server guide it correlates with whether two chains interact
(low PAE => interaction), so it can distinguish interacting from non-interacting
molecules (https://alphafoldserver.com/guides#section-3:-interpreting-results...).

The DB column ``af3_chain_pair_pae_min`` stores only a single scalar per complex
(``evaluation/af3Metrics_new.py`` reduced the matrix to one chain pair via
``get_pae_value_for_chain_pair``). For competition complexes (one protein + several
RNA chains) that scalar cannot separate the maxprobe RNA chain from the competing
minprobe chains. This script recovers the *per-RNA-chain* protein-RNA minimum PAE and
stores it as a JSON list aligned to ``RNAChainIDs`` (same layout as the per-chain
``af3_rna_ipTM`` column), in a new column ``af3_chain_pair_pae_min_perchain``.

Chain indexing matches af3Metrics_new.py: chains are labelled alphabetically A, B, C…
(matrix index = ord(chain) - ord('A')). We validate this per row by reconstructing
``af3_rna_ipTM`` from the JSON ``chain_iptm`` and comparing to the stored value.

Per-chain value
---------------
For each RNA chain r, the protein-RNA minimum PAE is
    min over protein chains p of  min(M[p][r], M[r][p])
i.e. the symmetric lowest PAE between that RNA chain and (the closest of) the protein
chain(s). Lower => stronger evidence of interaction.

Usage
-----
  conda run -n RNABench python evaluation/repopulate_chain_pair_pae_min.py \
      --af-dir data/rnaCompete_v5_MINMAX/protein_rna/af [--af-dir <more dirs>] [--dry-run]

  # or point at a root and let it walk for summary JSONs:
  ... --scan-root data [--dry-run]
"""

import argparse
import glob
import json
import os
import sqlite3
import sys

import numpy as np

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "rbpDatabase.db")
NEW_COLUMN = "af3_chain_pair_pae_min_perchain"
IPTM_TOL = 0.02  # per-chain ipTM rounding tolerance for the validation cross-check


def _parse_chain_ids(raw):
    """ProteinChainIDs/RNAChainIDs cells are JSON lists ('["B","C"]') or a bare id ('A')."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(c).strip() for c in json.loads(s)]
        except json.JSONDecodeError:
            pass
    return [c.strip() for c in s.replace(",", " ").split()]


def _chain_index(chain):
    """Alphabetical matrix index (A=0, B=1, …), matching af3Metrics_new.py."""
    if len(chain) != 1 or not ("A" <= chain.upper() <= "Z"):
        return None
    return ord(chain.upper()) - ord("A")


def _summary_path_for(file_name, index):
    """Map a DB FileName (fold_*_s1_model_0.cif) to its summary_confidences JSON basename."""
    base = os.path.basename(str(file_name))
    return base.replace("_model_0.cif", "_summary_confidences_0.json")


def build_summary_index(af_dirs, scan_root):
    """{summary_json_basename: full_path} over the given af dirs and/or a walked root."""
    idx = {}
    for d in af_dirs:
        for p in glob.glob(os.path.join(d, "*_summary_confidences_*.json")):
            idx.setdefault(os.path.basename(p), p)
    if scan_root:
        for root, _dirs, files in os.walk(scan_root):
            for f in files:
                if f.endswith(".json") and "_summary_confidences_" in f:
                    idx.setdefault(f, os.path.join(root, f))
    return idx


def per_chain_pae(matrix, protein_idx, rna_idx):
    """Symmetric protein-RNA min PAE for one RNA chain: min_p min(M[p][r], M[r][p])."""
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return None
    n = M.shape[0]
    if rna_idx is None or rna_idx >= n:
        return None
    vals = []
    for p in protein_idx:
        if p is None or p >= n:
            continue
        vals.append(min(float(M[p][rna_idx]), float(M[rna_idx][p])))
    return round(min(vals), 2) if vals else None


def ensure_column(conn, table, column):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" TEXT')
        conn.commit()
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    ap.add_argument("--table", default="pred_protein_rna", help="Predicted table")
    ap.add_argument("--column", default=NEW_COLUMN, help="Per-chain PAE column to write")
    ap.add_argument("--af-dir", action="append", default=[],
                    help="Directory with *_summary_confidences_0.json (repeatable)")
    ap.add_argument("--scan-root", default=None,
                    help="Root to os.walk for summary JSONs (in addition to --af-dir)")
    ap.add_argument("--only-competition", action="store_true",
                    help="Restrict to MAX#MIN# competition rows")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N rows")
    ap.add_argument("--dry-run", action="store_true", help="Compute + validate, do not write")
    args = ap.parse_args(argv)

    if not args.af_dir and not args.scan_root:
        ap.error("provide at least one --af-dir or a --scan-root")

    db = os.path.abspath(args.db)
    summary_idx = build_summary_index(args.af_dir, args.scan_root)
    print(f"DB: {db}")
    print(f"summary JSONs indexed: {len(summary_idx)}")

    conn = sqlite3.connect(db)
    where = "WHERE exp_db_id LIKE '%MAX%MIN%'" if args.only_competition else ""
    rows = conn.execute(
        f"SELECT Id, FileName, ProteinChainIDs, RNAChainIDs, af3_rna_ipTM "
        f"FROM {args.table} {where}"
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    if not args.dry_run:
        added = ensure_column(conn, args.table, args.column)
        if added:
            print(f"added column {args.table}.{args.column}")

    n_total = len(rows)
    n_json = n_written = n_iptm_ok = n_iptm_bad = n_no_json = n_no_rna = 0
    examples = []

    for row_id, file_name, pcids, rcids, iptm_raw in rows:
        rna_chains = _parse_chain_ids(rcids)
        prot_chains = _parse_chain_ids(pcids)
        if not rna_chains:
            n_no_rna += 1
            continue

        summ = _summary_path_for(file_name, 0)
        path = summary_idx.get(summ)
        if path is None:
            n_no_json += 1
            continue
        n_json += 1

        with open(path) as fh:
            data = json.load(fh)
        matrix = data.get("chain_pair_pae_min", [])
        chain_iptm = data.get("chain_iptm", [])

        prot_idx = [_chain_index(c) for c in prot_chains]
        per_chain = []
        for c in rna_chains:
            per_chain.append(per_chain_pae(matrix, prot_idx, _chain_index(c)))

        # Validation cross-check: reconstruct af3_rna_ipTM from chain_iptm via the same
        # alphabetical index mapping used for the PAE lookup.
        stored_iptm = None
        if iptm_raw and str(iptm_raw).strip().startswith("["):
            try:
                stored_iptm = json.loads(iptm_raw)
            except json.JSONDecodeError:
                stored_iptm = None
        if stored_iptm is not None and chain_iptm:
            recon = []
            for c in rna_chains:
                ci = _chain_index(c)
                v = chain_iptm[ci] if (ci is not None and ci < len(chain_iptm)) else None
                recon.append(round(v, 2) if v is not None else None)
            ok = len(recon) == len(stored_iptm) and all(
                (a is None and b is None) or
                (a is not None and b is not None and abs(a - b) <= IPTM_TOL)
                for a, b in zip(recon, stored_iptm)
            )
            n_iptm_ok += ok
            n_iptm_bad += (not ok)

        if len(examples) < 6:
            examples.append((file_name.split("_s1")[0], rna_chains, per_chain))

        if not args.dry_run:
            conn.execute(
                f'UPDATE {args.table} SET "{args.column}"=? WHERE Id=?',
                (json.dumps(per_chain), row_id),
            )
            n_written += 1

    if not args.dry_run:
        conn.commit()
    conn.close()

    print(f"\nrows considered:        {n_total}")
    print(f"  matched summary JSON: {n_json}")
    print(f"  no JSON found:        {n_no_json}")
    print(f"  no RNA chains:        {n_no_rna}")
    print(f"ipTM mapping check:     ok={n_iptm_ok}  mismatch={n_iptm_bad}")
    print(f"{'would write' if args.dry_run else 'wrote'} {args.column}: {n_written if not args.dry_run else n_json} rows")
    print("\nexamples (complex | RNA chains | per-chain PAE_min):")
    for name, rc, pc in examples:
        print(f"  {name}: {rc} -> {pc}")
    if n_iptm_bad:
        print(f"\nWARNING: {n_iptm_bad} rows failed the ipTM index cross-check; "
              f"inspect chain ordering before trusting those PAE values.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
