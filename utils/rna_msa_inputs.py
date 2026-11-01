#!/usr/bin/env python3
"""
RNA alignment depth for long RNAs, with AlphaFold 3's own RNA MSA search.

AF3 builds an RNA alignment per chain (src/alphafold3/data/pipeline.py: process_rna_chain -> _get_rna_msa):
nhmmer (alphabet rna, e-value 1e-3, at most 10,000 sequences) against Rfam, RNAcentral and NT-RNA, hits merged
and deduplicated; no templates and no MSA pairing for RNA. So one data-pipeline run per unique RNA sequence,
with only that RNA chain in the input, gives the alignment AF3 would use, without the protein search.

  make     collect every unique RNA sequence >= --min-len nt (default 200) of the given databases whose depth is
           not known yet (a depth measured for an identical sequence in any listed database is reused), and write
             <out>/inputs/<seq_id>.json   AF3 input with that single RNA chain (no unpairedMsa: the pipeline searches)
             <out>/manifest.tsv           seq_id, length, known depth, entries (set:PDB id:chain) carrying the sequence
             <out>/run_data_pipeline.sh   the run_alphafold.py command (data pipeline only) for the AF3 machine
  collect  read <alignment output>/**/<seq_id>_data.json, count the sequences of the RNA unpairedMsa (depth, query
           included, as RNA_msa_size), and write <out>/rna_msa_depth.tsv per entry. No database is changed.

Usage (repo root):
  python utils/rna_msa_inputs.py make --out results/rna_msa_200nt
  python utils/rna_msa_inputs.py collect --out results/rna_msa_200nt --alignments <folder with the AF3 outputs>
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sqlite3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(_REPO_ROOT, *p)
PP, RO, RR = (D("data", "compareDatabases", "post_pre", s) for s in
              ("a2021_b2021_protein_rna", "a2021_b2021_rna_only", "a2021_b2021_rna_rna"))
DATABASES = {   # set name: (database, prediction table)
    "protein_rna_post": (os.path.join(PP, "a2021.db"), "pred_protein_rna"),
    "protein_rna_pre": (os.path.join(PP, "b2021.db"), "pred_protein_rna"),
    "protein_rna_2026": (D("data", "2026", "protein_rna", "2026_a2021.db"), "pred_protein_rna"),
    "rna_only_post": (os.path.join(RO, "a2021_rna.db"), "pred_rna_rna"),
    "rna_only_pre": (os.path.join(RO, "b2021_rna.db"), "pred_rna_rna"),
    "rna_single_2026": (D("data", "2026", "rna_singlechain", "rna_rna", "rbpDatabase.db"), "pred_rna_rna"),
    "rna_single_2026_local": (D("data", "2026", "rna_singlechain", "rna_rna", "af3_local_freiburg", "rbpDatabase.db"), "pred_rna_rna"),
    "rna_multi_post": (os.path.join(RR, "a2021_rna_rna.db"), "pred_rna_rna"),
    "rna_multi_pre": (os.path.join(RR, "b2021_rna_rna.db"), "pred_rna_rna"),
    "rna_multi_2026": (D("data", "2026", "rna_multichain", "rna_rna", "rbpDatabase.db"), "pred_rna_rna"),
}


def _json(v, default):
    try:
        return json.loads(v) if isinstance(v, str) and v.strip()[:1] in "[{\"" else (v if v not in (None, "") else default)
    except json.JSONDecodeError:
        return default


def _chains(seq_field, chain_field):
    """[(chain id, sequence)] from the RNASequence / RNAChainIDs columns (lists, dicts or plain strings)."""
    seqs, ids = _json(seq_field, []), _json(chain_field, [])
    if isinstance(seqs, dict):
        return [(k, v) for k, v in seqs.items() if v]
    seqs = [seqs] if isinstance(seqs, str) else list(seqs or [])
    ids = [ids] if isinstance(ids, str) else list(ids or [])
    if len(seqs) == 1 and "&" in seqs[0]:
        seqs = seqs[0].split("&")
    return [(ids[i] if i < len(ids) else str(i), s) for i, s in enumerate(seqs) if s]


def _depths(field):
    d = _json(field, {})
    return d if isinstance(d, dict) else {}


def seq_id(seq):
    return "rna_" + hashlib.sha1(seq.encode()).hexdigest()[:12]


def make(out, min_len):
    known, entries = {}, {}
    for name, (db, table) in DATABASES.items():
        if not os.path.exists(db):
            print(f"  skipped (missing): {name} {db}")
            continue
        con = sqlite3.connect(db)
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        msa = "RNA_msa_size" if "RNA_msa_size" in cols else "NULL"
        for pid, seqs, chains, depth in con.execute(f"SELECT exp_db_id, RNASequence, RNAChainIDs, {msa} FROM {table}"):
            dep = _depths(depth)
            for ch, s in _chains(seqs, chains):
                s = s.replace("&", "").upper().replace("T", "U")
                if ch in dep and isinstance(dep[ch], (int, float)) and dep[ch] >= 1:
                    known[s] = max(known.get(s, 0), int(dep[ch]))
                if len(s) >= min_len:
                    entries.setdefault(s, []).append(f"{name}:{pid}:{ch}")
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    todo = 0
    with open(os.path.join(out, "manifest.tsv"), "w") as fh:
        fh.write("seq_id\tlength\tknown_depth\tn_entries\tentries\n")
        for s, ent in sorted(entries.items(), key=lambda kv: -len(kv[0])):
            sid = seq_id(s)
            fh.write(f"{sid}\t{len(s)}\t{known.get(s, '')}\t{len(ent)}\t{','.join(sorted(set(ent)))}\n")
            if s in known:
                continue
            todo += 1
            job = {"name": sid, "modelSeeds": [1], "dialect": "alphafold3", "version": 2,
                   "sequences": [{"rna": {"id": "A", "sequence": s}}]}
            json.dump(job, open(os.path.join(out, "inputs", f"{sid}.json"), "w"), indent=1)
    open(os.path.join(out, "run_data_pipeline.sh"), "w").write(
        "#!/bin/bash\n# On the AF3 machine (needs nhmmer and the Rfam, RNAcentral and NT-RNA databases of AF3).\n"
        "# Data pipeline only: writes <output_dir>/<seq_id>/<seq_id>_data.json with the RNA unpairedMsa.\n"
        "python run_alphafold.py --input_dir=inputs --output_dir=alignments --norun_inference \\\n"
        "  --db_dir=\"${DB_DIR}\"\n")
    print(f"unique RNA sequences >= {min_len} nt: {len(entries)} | depth already known: {len(entries) - todo} | "
          f"to search: {todo} -> {out}/inputs")


def collect(out, aln_dir):
    depth = {}
    for f in glob.glob(os.path.join(aln_dir, "**", "*_data.json"), recursive=True):
        j = json.load(open(f))
        for e in j.get("sequences", []):
            r = e.get("rna")
            if r and r.get("unpairedMsa") is not None:
                depth[seq_id(r["sequence"].upper())] = r["unpairedMsa"].count(">")
    rows = list(csv.DictReader(open(os.path.join(out, "manifest.tsv")), delimiter="\t"))
    with open(os.path.join(out, "rna_msa_depth.tsv"), "w") as fh:
        fh.write("set\tpdb\tchain\tlength\tseq_id\tdepth\tsource\n")
        for r in rows:
            d, src = (r["known_depth"], "earlier local run") if r["known_depth"] else (depth.get(r["seq_id"], ""), "AF3 data pipeline")
            for e in r["entries"].split(","):
                s, pid, ch = e.split(":", 2)
                fh.write(f"{s}\t{pid}\t{ch}\t{r['length']}\t{r['seq_id']}\t{d}\t{src if d != '' else 'missing'}\n")
    print(f"depths found for {sum(1 for r in rows if r['known_depth'] or r['seq_id'] in depth)}/{len(rows)} sequences -> "
          f"{out}/rna_msa_depth.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("step", choices=["make", "collect"])
    ap.add_argument("--out", default=D("results", "rna_msa_200nt"))
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--alignments", help="collect: folder with the AF3 data-pipeline outputs")
    a = ap.parse_args()
    make(a.out, a.min_len) if a.step == "make" else collect(a.out, a.alignments)
