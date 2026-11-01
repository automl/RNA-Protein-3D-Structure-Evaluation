#!/usr/bin/env python3
"""
RNA-binding domain (RBD) annotation of the protein chains of exp_protein_rna (domain_ids / domain_names /
domain_counts / domain_pos), fully automated: InterProScan through the EMBL-EBI web service, then the RBD rules of
michauhl/rbd-annotation (https://github.com/michauhl/rbd-annotation, commit 1efb655, adapted with the author's
permission).

  1. Protein sequences: every protein chain of the table (ProteinChainIDs / ProteinSequence, the deposited
     sequence); modified residues written as (SEP) / (TPO) / (ALY) ... are given as their parent residue.
  2. InterProScan 5 (EBI Job Dispatcher REST service iprscan5, TSV output), one job per unique sequence. Results are
     cached per sequence MD5 in <cache>/<md5>.tsv (an empty file = no match), so a rerun never queries EBI again and
     the cache can be shipped; <cache>/VERSION.txt records the InterProScan version of the run.
  3. RBD rules (refine_interproscan_results_keep_rbds.py): keep matches whose InterPro entry is in the RBD table (database/rbd_annotation/rbd_parent_infos.tsv: RBP2GO "Selected and new RBDs" plus additions,
     with InterPro parents), map each to its parent entry, drop intervals contained in another, merge intervals
     overlapping by >= 60% of the shorter one. The E-value cutoff (<= 1e-5) applies to the member databases that
     report an E-value; ProSite profiles and patterns, PRINTS and HAMAP report a score in that column and are kept
     (--strict-evalue applies the cutoff to every match, as rbd-annotation itself does).
  4. Per complex: identical chain sequences count once, and copies of a domain in several chains (constructs of one
     protein differing by a few residues) are pooled with the same refinement and merging, so they count once.
     domain_counts: a domain counts per occurrence, a repeat
     (parent type "repeat") once per complex; domain_pos lists every occurrence (positions on the deposited chain
     sequence), entries ordered by their first position.

Output: <out_dir>/domains_rbd.tsv (PDBId, domain_ids, domain_names, domain_counts, domain_pos; "-" when no RBD), the
format of database/addDomain.py; --load writes it into the database. <out_dir>/domains_rbd_per_chain.json keeps every
domain separately per complex and chain: all InterPro entries matched by InterProScan (with each signature match,
E-value and source database) and the RBDs after the rules.

Usage (RNABench env; the EBI service requires an e-mail address per job, passed with --email):
    python database/annotateRBDs.py <out_dir> --db <file.db> [--db ...] --email <address> [--load] [--workers 25]
    python database/annotateRBDs.py --test        # the author's test case: rules reproduce his expected output
"""
import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
HERE = os.path.join(_ROOT, "database", "rbd_annotation")
PARENTS = os.path.join(HERE, "rbd_parent_infos.tsv")
CACHE = os.path.join(_ROOT, "data", "interproscan_cache")
EBI = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
E_THR, OVERLAP = 1e-5, 0.6
# member databases whose InterProScan TSV score column is a score, not an E-value
SCORE_METHODS = {"ProSiteProfiles", "ProSitePatterns", "PRINTS", "HAMAP"}
STRICT = False                                         # --strict-evalue
PARENT_RESIDUE = {"SEP": "S", "TPO": "T", "PTR": "Y", "ALY": "K", "MSE": "M", "MLY": "K", "M3L": "K", "CSO": "C",
                  "HYP": "P", "SEC": "C"}


# --------------------------------------------------------------------------- RBD rules (michauhl/rbd-annotation)
def read_parents(path=PARENTS):
    """InterPro accession -> (parent accession, parent short name, parent type)."""
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        out[r["interpro_id"]] = (r["parent_id"], r["parent_short"], r["parent_type"])
    return out


def refine_intervals(intervals):
    """Drop intervals fully contained in another (rbd_common.refine_intervals)."""
    intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    refined = []
    for iv in intervals:
        if not refined or iv[1] > refined[-1][1]:
            refined.append(list(iv))
    return refined


def merge_intervals(intervals, overlap_threshold=OVERLAP):
    """Merge intervals overlapping by >= overlap_threshold of the shorter one (rbd_common.merge_intervals)."""
    merged = []
    for start, end in sorted(intervals):
        if merged:
            ls, le = merged[-1]
            overlap = max(0, min(end, le) - max(start, ls))
            shorter = min(end - start, le - ls)
            if shorter > 0 and overlap / shorter >= overlap_threshold:
                merged[-1] = [min(ls, start), max(le, end)]
                continue
        merged.append([start, end])
    return merged


def rbd_hits(tsv_lines, parents, e_thr=E_THR, strict=None):
    """InterProScan TSV lines -> {query: {parent accession: [[start, end], ...]}} after refinement and merging.
    strict: the E-value cutoff also on the score-based member databases (rbd-annotation's own behaviour)."""
    strict = STRICT if strict is None else strict
    hits = {}
    for line in tsv_lines:
        f = line.rstrip("\n").split("\t")
        if len(f) < 12 or f[11] not in parents:
            continue
        if strict or f[3] not in SCORE_METHODS:
            if f[8] == "-" or float(f[8]) > e_thr:
                continue
        hits.setdefault(f[0], {}).setdefault(parents[f[11]][0], []).append([int(f[6]), int(f[7])])
    return {q: {p: merge_intervals(refine_intervals(iv)) for p, iv in d.items()} for q, d in hits.items()}


# --------------------------------------------------------------------------- sequences
def clean(seq):
    seq = re.sub(r"\s", "", str(seq or "")).upper()
    return re.sub(r"\(([A-Z0-9]+)\)", lambda m: PARENT_RESIDUE.get(m.group(1), "X"), seq)


def _list(v):
    s = str(v or "").strip()
    return json.loads(s) if s.startswith("[") else ([s] if s else [])


def complexes(db, table="exp_protein_rna"):
    """[(PDBId, [unique chain sequences])] of a database table."""
    out = []
    for pid, seqs in sqlite3.connect(db).execute(f"SELECT PDBId, ProteinSequence FROM {table}"):
        u = []
        for s in map(clean, _list(seqs)):
            if s and s != "-" and s not in u:
                u.append(s)
        out.append((str(pid).strip(), u))
    return out


def chain_sequences(db, table="exp_protein_rna"):
    """{PDBId: {chain id: sequence}} of a database table."""
    out = {}
    for pid, ch, seqs in sqlite3.connect(db).execute(f"SELECT PDBId, ProteinChainIDs, ProteinSequence FROM {table}"):
        out[str(pid).strip()] = {str(c): clean(s) for c, s in zip(_list(ch), _list(seqs)) if clean(s) not in ("", "-")}
    return out


def md5(seq):
    return hashlib.md5(seq.encode()).hexdigest()


# --------------------------------------------------------------------------- EBI InterProScan service
def _http(url, data=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None,
                                         headers={"User-Agent": "AF3InterfaceEval annotateRBDs"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode()
        except (urllib.error.URLError, http.client.HTTPException, OSError):  # incl. socket timeouts
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def interproscan(seq, email, cache=CACHE, poll=15, max_wait=7200):
    """InterProScan TSV lines of one sequence, from the cache or the EBI service."""
    path = os.path.join(cache, md5(seq) + ".tsv")
    if os.path.exists(path):
        return open(path).read().splitlines()
    os.makedirs(cache, exist_ok=True)
    job = _http(f"{EBI}/run", dict(email=email, title=md5(seq), goterms="false", pathways="false",
                                   sequence=f">{md5(seq)}\n{seq}\n"))
    t0 = time.time()
    while True:
        status = _http(f"{EBI}/status/{job}").strip()
        if status == "FINISHED":
            break
        if status in ("ERROR", "FAILURE", "NOT_FOUND") or time.time() - t0 > max_wait:
            raise RuntimeError(f"InterProScan job {job}: {status}")
        time.sleep(poll)
    tsv = _http(f"{EBI}/result/{job}/tsv")
    version = os.path.join(cache, "VERSION.txt")
    if not os.path.exists(version):
        try:
            v = json.loads(_http(f"{EBI}/result/{job}/json")).get("interproscan-version", "")
            open(version, "w").write(f"InterProScan {v} (EBI iprscan5 service), first job {job}\n")
        except (urllib.error.URLError, ValueError):
            pass
    tmp = path + ".part"
    open(tmp, "w").write(tsv)
    os.replace(tmp, path)
    return tsv.splitlines()


def run_all(seqs, email, cache=CACHE, workers=25):
    """{sequence: TSV lines} for all sequences, querying EBI only for those not cached."""
    os.makedirs(cache, exist_ok=True)
    todo = [s for s in seqs if not os.path.exists(os.path.join(cache, md5(s) + ".tsv"))]
    print(f"{len(seqs)} unique sequences, {len(seqs) - len(todo)} cached, {len(todo)} to run on EBI", flush=True)
    if todo and not email:
        sys.exit("--email is required to submit InterProScan jobs to EBI")
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(interproscan, s, email, cache): s for s in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                fu.result()
            except Exception as e:                      # noqa: BLE001
                failed.append((md5(futs[fu]), str(e)[:120]))
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} done, {len(failed)} failed", flush=True)
    for m, e in failed:
        print("  FAILED", m, e)
    return {s: interproscan(s, email, cache) for s in seqs
            if os.path.exists(os.path.join(cache, md5(s) + ".tsv"))}


# --------------------------------------------------------------------------- per complex
def complex_row(pdb, seqs, results, parents):
    """One row of domains_rbd.tsv: hits of the distinct protein sequences of a complex."""
    names = {p: short for p, short, _ in parents.values()}
    types = {p: typ for p, _, typ in parents.values()}
    pooled = {}                                        # parent -> intervals over all distinct chain sequences
    for s in seqs:
        for p, ivs in rbd_hits(results.get(s, []), parents).get(md5(s), {}).items():
            pooled.setdefault(p, []).extend(ivs)
    # copies of a domain in several chains (constructs differing by a few residues) merge into one occurrence
    occ = [(a, b, p) for p, ivs in pooled.items() for a, b in merge_intervals(refine_intervals(ivs))]
    if not occ:
        return dict(PDBId=pdb, domain_ids="-", domain_names="-", domain_counts="-", domain_pos="-")
    occ.sort()
    order = []
    for _, _, p in occ:
        if p not in order:
            order.append(p)
    counts = {p: 1 if types.get(p, "").lower() == "repeat" else sum(1 for *_, q in occ if q == p) for p in order}
    return dict(PDBId=pdb, domain_ids=";".join(order), domain_names=";".join(names[p] for p in order),
                domain_counts=";".join(str(counts[p]) for p in order),
                domain_pos=";".join(f"{p}:" + ",".join(f"{a}-{b}" for a, b, q in occ if q == p) for p in order))


def chain_domains(seq, lines, parents):
    """Every domain of one chain sequence, kept separately: all InterPro entries matched by InterProScan
    (accession -> name and the match intervals of its member-database signatures, E-value and source) and the RBDs
    after the RBD rules (parent accession -> merged intervals)."""
    allent = {}
    for line in lines:
        f = line.rstrip("\n").split("\t")
        if len(f) < 12 or f[11] in ("", "-"):
            continue
        e = allent.setdefault(f[11], dict(name=f[12] if len(f) > 12 else "", matches=[]))
        e["matches"].append(dict(start=int(f[6]), end=int(f[7]), evalue=f[8], analysis=f[3], signature=f[4]))
    rbd = {p: dict(short=next(s for q, s, _ in parents.values() if q == p), intervals=ivs)
           for p, ivs in rbd_hits(lines, parents).get(md5(seq), {}).items()}
    return dict(length=len(seq), md5=md5(seq), interpro=allent, rbd=rbd)


# --------------------------------------------------------------------------- test and main
def self_test():
    """The author's test case: InterProScan output -> RBD hits must equal his expected table."""
    parents = read_parents()
    lines = open(os.path.join(HERE, "test", "test_protein_seqs.interproscan_results.tsv")).read().splitlines()
    got = {(q, p, a, b) for q, d in rbd_hits(lines, parents, strict=True).items() for p, ivs in d.items()
           for a, b in ivs}
    exp = {(r["query_id"], r["interpro_id"], int(r["hit_pos_s"]), int(r["hit_pos_e"]))
           for r in csv.DictReader(open(os.path.join(HERE, "test", "test_protein_seqs.interpro_rbd_domains.expected.tsv")),
                                   delimiter="\t")}
    ok = got == exp
    print(f"self-test: {len(got)} RBD hits, expected {len(exp)}: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  missing", sorted(exp - got)[:10], "\n  extra", sorted(got - exp)[:10])
    return ok


def main():
    ap = argparse.ArgumentParser(description="RBD annotation of exp_protein_rna (InterProScan via EBI + RBD rules)")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--db", action="append", default=[], help="database file(s); default: database/config.json")
    ap.add_argument("--table", default="exp_protein_rna")
    ap.add_argument("--email", help="e-mail address required by the EBI service (not stored)")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--workers", type=int, default=25, help="parallel EBI jobs (EBI allows up to 30)")
    ap.add_argument("--load", action="store_true", help="write the annotation into the database(s)")
    ap.add_argument("--test", action="store_true", help="run the rule self-test only")
    ap.add_argument("--strict-evalue", action="store_true",
                    help="apply the E-value cutoff also to score-based member databases (rbd-annotation's behaviour)")
    a = ap.parse_args()
    global STRICT
    STRICT = a.strict_evalue
    if a.test:
        sys.exit(0 if self_test() else 1)
    if not a.out_dir:
        ap.error("out_dir is required")
    if not a.db:
        from database.startConfig import StartConfig
        a.db = [StartConfig().get_database_path()]
    parents = read_parents()
    per_db = {db: complexes(db, a.table) for db in a.db}
    chains = {db: chain_sequences(db, a.table) for db in a.db}
    seqs = sorted({s for cx in per_db.values() for _, ss in cx for s in ss})
    results = run_all(seqs, a.email, a.cache, a.workers)
    os.makedirs(a.out_dir, exist_ok=True)
    cols = ["PDBId", "domain_ids", "domain_names", "domain_counts", "domain_pos"]
    for db, cx in per_db.items():
        rows = [complex_row(pdb, ss, results, parents) for pdb, ss in cx]
        name = os.path.splitext(os.path.basename(db))[0]
        out = os.path.join(a.out_dir, f"domains_rbd_{name}.tsv" if len(a.db) > 1 else "domains_rbd.tsv")
        with open(out, "w") as fh:
            w = csv.DictWriter(fh, cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
        missing = sum(1 for _, ss in cx for s in ss if s not in results)
        print(f"{db}: {len(rows)} complexes, {sum(r['domain_ids'] != '-' for r in rows)} with an RBD, "
              f"{missing} chain sequences without an InterProScan result -> {out}")
        per_chain = {pdb: {ch: chain_domains(sq, results.get(sq, []), parents) for ch, sq in cs.items() if sq in results}
                     for pdb, cs in chains[db].items()}
        jout = out.replace(".tsv", "_per_chain.json")
        json.dump(per_chain, open(jout, "w"), indent=1)
        print(f"  per-chain domains (all InterPro entries and RBDs) -> {jout}")
        if a.load:
            if missing:
                sys.exit("not loading: some sequences have no InterProScan result")
            from database.addDomain import add_domains
            add_domains(out, db_path=db, table=a.table, id_col="PDBId")


if __name__ == "__main__":
    main()
