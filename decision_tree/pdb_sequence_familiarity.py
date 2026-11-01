"""
Sequence familiarity against every RNA in the PDB released before the AF3 training cutoff (30 September 2021).

Reference: data/pdb_reference/pdb_seqres.txt (wwPDB) restricted to the RNA-containing entries released on or before
2021-09-30 (data/pdb_reference/rna_entries_pre2021.txt, RCSB search API), RNA chains only (nucleic-acid chains
without T), unique sequences -> data/pdb_reference/pdb_rna_pre2021.fasta. The tool reads the same FASTA
(--reference-fasta). Identity = identical aligned columns over the longer sequence (framework.sequence_familiar).

Writes results/plots/all_chains/decision_tree/probability/pdb_seq_identity.json  {set: {pdb: best identity}}
Run (RNABench env):  python -m decision_tree.pdb_sequence_familiarity
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.develop_probability import BASE_OUT

REF = os.path.join(_REPO_ROOT, "data", "pdb_reference")
FASTA = os.path.join(REF, "pdb_rna_pre2021.fasta")
OUTFILE = os.path.join(BASE_OUT, "pdb_seq_identity.json")
from evaluation.bundle_paths import PROTEIN_RNA_DB, Y2026_DB
SETS = {"post": PROTEIN_RNA_DB["post"], "2026": Y2026_DB}


SEQRES_URL = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def download_reference():
    """pdb_seqres.txt (wwPDB) and the RNA-containing entries released on or before the cutoff (RCSB search API),
    into data/pdb_reference/; only the files that are not there yet."""
    import urllib.request
    os.makedirs(REF, exist_ok=True)
    seqres = os.path.join(REF, "pdb_seqres.txt")
    if not os.path.exists(seqres):
        print("downloading", SEQRES_URL, flush=True)
        urllib.request.urlretrieve(SEQRES_URL, seqres + ".part")
        os.replace(seqres + ".part", seqres)
    entries = os.path.join(REF, "rna_entries_pre2021.txt")
    if not os.path.exists(entries):
        query = {"query": {"type": "group", "logical_operator": "and", "nodes": [
                     {"type": "terminal", "service": "text", "parameters": {
                         "attribute": "rcsb_entry_info.polymer_entity_count_RNA", "operator": "greater", "value": 0}},
                     {"type": "terminal", "service": "text", "parameters": {
                         "attribute": "rcsb_accession_info.initial_release_date", "operator": "less_or_equal",
                         "value": "2021-09-30"}}]},
                 "return_type": "entry", "request_options": {"return_all_hits": True}}
        req = urllib.request.Request(RCSB_SEARCH, data=json.dumps(query).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            ids = [h["identifier"] for h in json.load(r)["result_set"]]
        with open(entries, "w") as fh:
            fh.write("\n".join(sorted(ids)) + "\n")
        print(f"{len(ids)} RNA-containing PDB entries released up to 2021-09-30", flush=True)


def build_fasta():
    download_reference()
    entries = {l.strip().lower() for l in open(os.path.join(REF, "rna_entries_pre2021.txt")) if l.strip()}
    seqs, head = {}, None
    for line in open(os.path.join(REF, "pdb_seqres.txt")):
        if line.startswith(">"):
            head = line[1:].split()[0] if " mol:na " in line else None
            continue
        if head and head[:4] in entries:
            s = line.strip().upper()
            if s and "T" not in s and "U" in s:          # RNA, not DNA
                seqs.setdefault(s, head)
    with open(FASTA, "w") as fh:
        for s, h in seqs.items():
            fh.write(f">{h}\n{s}\n")
    return len(entries), len(seqs)


def _aligner():
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score, al.open_gap_score, al.extend_gap_score = 2, -3, -5, -2
    return al


def best_identity(seq, refs, al):
    """highest identity (identical aligned columns / longer length) of seq to any reference"""
    seq = re.sub(r"\([^)]*\)", "N", seq.upper()).replace("T", "U")
    best = 0.0
    for r in refs:
        lo, hi = sorted((len(r), len(seq)))
        if lo / hi <= best:                              # cannot beat the current best
            continue
        a = al.align(seq, r)[0]
        ident = sum(x == y for blk_a, blk_b in zip(*a.aligned) for x, y in
                    zip(seq[blk_a[0]:blk_a[1]], r[blk_b[0]:blk_b[1]])) / hi
        best = max(best, ident)
        if best >= 0.999:
            break
    return best


def run():
    if not os.path.exists(FASTA):
        print("reference: %d pre-cutoff RNA entries, %d unique RNA sequences" % build_fasta())
    refs = [l.strip() for l in open(FASTA) if not l.startswith(">")]
    al = _aligner()
    out = {}
    for name, db in SETS.items():
        d = out.setdefault(name, {})
        for pid, seqs in sqlite3.connect(db).execute("SELECT PDBId, RNASequence FROM exp_protein_rna"):
            try:
                lst = json.loads(seqs) if str(seqs).startswith("[") else [seqs]
            except json.JSONDecodeError:
                lst = [seqs]
            d[str(pid).strip().upper()] = max((best_identity(s, refs, al) for s in lst if s), default=None)
        print(f"  {name}: {len(d)} targets, {sum((v or 0) >= 0.8 for v in d.values())} familiar (>= 80%)", flush=True)
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    json.dump(out, open(OUTFILE, "w"), indent=0)
    return out


if __name__ == "__main__":
    run()
