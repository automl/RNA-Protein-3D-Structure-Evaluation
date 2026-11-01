#!/usr/bin/env python3
"""
Annotate protein domains (domain_ids / domain_names / domain_counts / domain_pos) of the
exp_<classification> table from InterPro, via the UniProt accession of every protein chain.

For each UniProt accession the InterPro entries of type "domain" or "repeat" are fetched with their
locations (InterPro API), the UniProt sequence is aligned to the deposited chain sequence, and each
domain is kept when >= 50% of its residues are present in the deposited construct.

Vocabulary: the paper's annotation lists RNA-binding (and RNA-processing) domains only, not every
InterPro domain (e.g. no viral polymerase sub-domains). Its producer is not in the repository, so new
annotations are restricted to the InterPro entries that occur in it: the domain_ids / domain_names
of the pre- and post-cutoff protein-RNA databases plus the RNAcompete RNA-binding-domain table
(--all-domains switches the filter off). Positions are on
the deposited chain sequence, the convention of the paper's annotation (e.g. 9IJ3: PAZ_dom 166-298,
Piwi 440-724). Output format as consumed by database/addDomain.py:

    PDBId  domain_ids              domain_names   domain_counts  domain_pos
    9IJ3   IPR003100;IPR003165     PAZ_dom;Piwi   1;1            IPR003100:166-298;IPR003165:440-724

Accessions: an isoform (Q6PKG0-3) is looked up as its canonical entry, a merged UniProt accession is followed
to the entry it was merged into, and chains without a stored accession get theirs from the PDBe SIFTS mapping.
domain_counts give, per InterPro entry, the occurrences of a domain (repeats count once per protein), summed over
the distinct proteins of the complex (identical copies count once); domain_pos lists every occurrence.

Entries without an annotated domain get "-" in every column. Responses are cached in
<out_dir>/interpro_cache.json, so reruns do not query the APIs again.

Entries for which no UniProt accession can be found at all keep their current annotation in the database.

Usage (DB and table from database/config.json, as for every pipeline script, unless --db / --table are given):
    python database/fetchInterProDomains.py <out_dir>          # writes <out_dir>/domains_interpro.tsv and loads it
    python database/fetchInterProDomains.py <out_dir> --db <file.db> --no-load   # only the TSV, for another database
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.addDomain import add_domains
from database.startConfig import StartConfig

UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.fasta"
UNIPROT_JSON = "https://rest.uniprot.org/uniprotkb/{}.json"
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"
ENTRY = "https://www.ebi.ac.uk/interpro/api/entry/interpro/{}"
INTERPRO = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{}?page_size=200&extra_fields=short_name"
TYPES = ("domain", "repeat")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_DBS = [os.path.join(_ROOT, "data", "compareDatabases", "post_pre", "a2021_b2021_protein_rna", f)
             for f in ("a2021.db", "b2021.db")]
VOCAB_TSVS = [os.path.join(_ROOT, "data", "rnaCompete_v4", "RNAcompeteRNAbindingdomainstats-rnacompete.tsv")]


def vocabulary():
    """InterPro accessions and short names of the paper's domain annotation."""
    v = set()
    for db in VOCAB_DBS:
        if os.path.exists(db):
            for ids, names in sqlite3.connect(db).execute("SELECT domain_ids, domain_names FROM exp_protein_rna"):
                for x in (ids, names):
                    v |= {s.strip() for s in str(x or "").split(";") if s.strip() not in ("", "-")}
    for tsv in VOCAB_TSVS:
        if os.path.exists(tsv):
            for line in open(tsv).readlines()[1:]:
                f = line.split("\t")
                v |= {f[0].strip(), f[1].strip()}
    return v
MIN_COVERED = 0.5
ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")


def _get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return ""
            if k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if k == tries - 1:
                raise
        time.sleep(5 * (k + 1))
    return ""


def uniprot_sequence(acc, cache):
    key = f"seq:{acc}"
    if key not in cache:
        txt = _get(UNIPROT.format(acc))
        cache[key] = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
    return cache[key]


def interpro_domains(acc, cache):
    """[(ipr, short_name, start, end)] in UniProt numbering."""
    key = f"ipr:{acc}"
    if key not in cache:
        out, url = [], INTERPRO.format(acc)
        while url:
            txt = _get(url)
            if not txt:
                break
            d = json.loads(txt)
            for r in d.get("results", []):
                m = r["metadata"]
                if m.get("type") not in TYPES:
                    continue
                short = (r.get("extra_fields") or {}).get("short_name") or m["accession"]
                for p in r.get("proteins", []):
                    for loc in p.get("entry_protein_locations") or []:
                        for f in loc.get("fragments", []):
                            out.append((m["accession"], short, int(f["start"]), int(f["end"])))
            url = d.get("next")
        cache[key] = out
    return cache[key]


def entry_type(ipr, cache):
    """InterPro entry type ('domain', 'repeat', ...) of an accession, cached."""
    key = f"type:{ipr}"
    if key not in cache:
        txt = _get(ENTRY.format(ipr))
        cache[key] = (json.loads(txt).get("metadata") or {}).get("type", "") if txt else ""
    return cache[key]


def canonical(acc):
    """InterPro has no isoform entries: Q6PKG0-3 -> Q6PKG0 (domains and positions of the canonical sequence)."""
    return acc.split("-")[0]


def current_accession(acc, cache):
    """Follow a merged (inactive) UniProt accession to the entry it was merged into; else the accession itself."""
    key = f"cur:{acc}"
    if key not in cache:
        cur = acc
        txt = _get(UNIPROT_JSON.format(acc))      # UniProt redirects a merged accession to the current entry
        if txt:
            d = json.loads(txt)
            to = ((d.get("inactiveReason") or {}).get("mergeDemergeTo") or []) if d.get("entryType") == "Inactive" else []
            cur = to[0] if to else d.get("primaryAccession", acc)
        cache[key] = cur
    return cache[key]


def sifts_accessions(pdb, cache):
    """{author chain id: UniProt accession} from the PDBe SIFTS mapping (for entries without a stored accession)."""
    key = f"sifts:{pdb.lower()}"
    if key not in cache:
        out = {}
        txt = _get(SIFTS.format(pdb.lower()))
        if txt:
            for acc, v in (json.loads(txt).get(pdb.lower(), {}).get("UniProt") or {}).items():
                for m in v.get("mappings", []):
                    out.setdefault(str(m.get("chain_id")), acc)
        cache[key] = out
    return cache[key]


def sifts_chain_accessions(pdb, cache):
    """{author chain id: [UniProt accessions]} from SIFTS; a fusion chain maps to several accessions."""
    key = f"sifts_all:{pdb.lower()}"
    if key not in cache:
        out = {}
        txt = _get(SIFTS.format(pdb.lower()))
        if txt:
            for acc, v in (json.loads(txt).get(pdb.lower(), {}).get("UniProt") or {}).items():
                for m in v.get("mappings", []):
                    lst = out.setdefault(str(m.get("chain_id")), [])
                    if acc not in lst:
                        lst.append(acc)
        cache[key] = out
    return cache[key]


FUSION_IDENTITY = 0.8      # below this share of matched chain residues, the chain may be a fusion: add SIFTS accessions


def resolve(acc, pdb, chain, cache):
    """Accession with InterPro domains for one protein chain: the stored one (isoform -> canonical; merged ->
    current entry), else the chain's accession in the PDBe SIFTS mapping (no stored accession, or a deleted one).
    Returns (None, []) when no accession can be found."""
    acc = (acc or "").strip().upper()
    if ACC.match(acc):
        acc = canonical(acc)
        doms = interpro_domains(acc, cache)
        if doms:
            return acc, doms
        cur = current_accession(acc, cache)
        if cur != acc:
            doms = interpro_domains(cur, cache)
            if doms:
                return cur, doms
    sifts = sifts_accessions(pdb, cache).get(str(chain), "")
    if not sifts:
        return (acc if ACC.match(acc) else None), []
    sifts = canonical(sifts)
    return sifts, interpro_domains(sifts, cache)


def _clean(seq):
    return re.sub(r"\([^)]*\)", "X", str(seq)).upper().replace(" ", "")


def _list(v):
    if v is None:
        return []
    s = str(v).strip()
    if s.startswith("["):
        try:
            x = json.loads(s)
            return [str(i) for i in x] if isinstance(x, list) else [str(x)]
        except json.JSONDecodeError:
            pass
    return [s] if s else []


def map_positions(uni_seq, chain_seq):
    """{uniprot position (1-based): chain position (1-based)} from a global alignment, free end gaps."""
    from Bio import Align
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.open_gap_score, a.extend_gap_score = -10, -0.5
    a.target_end_gap_score = a.query_end_gap_score = 0
    aln = a.align(uni_seq, chain_seq)[0]
    m = {}
    for (u0, u1), (s0, s1) in zip(*aln.aligned):
        for k in range(u1 - u0):
            m[u0 + k + 1] = s0 + k + 1
    return m


def _identity(uni_seq, chain_seq):
    """share of the chain residues aligned to an identical UniProt residue"""
    m = map_positions(uni_seq, chain_seq)
    return sum(uni_seq[u - 1] == chain_seq[c - 1] for u, c in m.items()) / max(len(chain_seq), 1)


def assign_accessions(pdb, uniprots, seqs, chains, cache):
    """One accession per chain when the stored list does not have one per chain: each chain gets the candidate
    (stored accessions, resolved, plus the entry's SIFTS accessions) whose UniProt sequence matches it best."""
    cands = set()
    for acc in uniprots:
        acc = (acc or "").strip().upper()
        if ACC.match(acc):
            cands.add(canonical(acc))
            cands.add(current_accession(canonical(acc), cache))
    cands |= {canonical(a) for a in sifts_accessions(pdb, cache).values()}
    out = []
    for cs in seqs:
        best, score = "", 0.0
        for acc in sorted(cands):
            useq = uniprot_sequence(acc, cache)
            if useq and cs:
                sc = _identity(useq, cs)
                if sc > score:
                    best, score = acc, sc
        out.append(best if score >= 0.5 else "")
    return out


def annotate(entry, cache, vocab=None):
    """Domains of a complex. A domain occurrence (accession + UniProt range) found in several chains counts once
    (identical copies, or fragments of one protein in several chains); a chain matching its stored accession for
    less than FUSION_IDENTITY of its residues also gets the domains of its other SIFTS accessions (fusions)."""
    pdb, uniprots, seqs, chains = entry if len(entry) == 4 else (*entry, [])
    seqs = [_clean(s) for s in seqs]
    if len(uniprots) == 1 and len(seqs) > 1:
        uniprots = uniprots * len(seqs)
    elif len(uniprots) != len(seqs):           # not one accession per chain: match chains by sequence
        uniprots = assign_accessions(pdb, uniprots, seqs, chains, cache)
    chains = list(chains) + [None] * (len(seqs) - len(chains))
    hits = []          # (chain start, chain end, ipr, short, accession)
    seen, resolved = set(), False      # seen: (accession, ipr, UniProt start, end), so copies count once
    for acc, cs, ch in zip(uniprots, seqs, chains):
        if not cs:
            continue
        acc, doms = resolve(acc, pdb, ch, cache)
        resolved = resolved or acc is not None
        cands = []
        ident = 0.0
        if doms:
            useq = uniprot_sequence(acc, cache)
            if useq:
                cands.append((acc, doms, useq))
                ident = _identity(useq, cs)
        if ident < FUSION_IDENTITY:     # partial match: a fusion chain, or a construct of another protein
            for a2 in sifts_chain_accessions(pdb, cache).get(str(ch), []):
                a2 = canonical(a2)
                if a2 == acc:
                    continue
                d2, u2 = interpro_domains(a2, cache), uniprot_sequence(a2, cache)
                resolved = True
                if d2 and u2:
                    cands.append((a2, d2, u2))
        for a, ds, useq in cands:
            m = map_positions(useq, cs)
            for ipr, short, st, en in ds:
                if vocab is not None and ipr not in vocab and short not in vocab:
                    continue
                if (a, ipr, st, en) in seen:
                    continue
                cov = [m[i] for i in range(st, en + 1) if i in m]
                if len(cov) >= MIN_COVERED * (en - st + 1):
                    seen.add((a, ipr, st, en))
                    hits.append((min(cov), max(cov), ipr, short, a))
    if not hits:
        return dict(PDBId=pdb, domain_ids="-", domain_names="-", domain_counts="-", domain_pos="-",
                    no_accession=not resolved)
    hits.sort()
    # domain_counts: domains count per occurrence, repeats once per protein (a WD40 propeller is one domain),
    # summed over the distinct proteins of the complex
    order, counts, names, counted = [], {}, {}, set()
    for _, _, ipr, short, acc in hits:
        if ipr not in counts:
            order.append(ipr)
            counts[ipr] = 0
        names[ipr] = short
        if entry_type(ipr, cache) == "repeat":
            if (ipr, acc) in counted:
                continue
            counted.add((ipr, acc))
        counts[ipr] += 1
    return dict(PDBId=pdb, domain_ids=";".join(order), domain_names=";".join(names[i] for i in order),
                domain_counts=";".join(str(counts[i]) for i in order),
                domain_pos=";".join(f"{i}:" + ",".join(f"{a}-{b}" for a, b, ipr, _, _ in hits if ipr == i) for i in order),
                no_accession=False)


def main():
    ap = argparse.ArgumentParser(description="InterPro domain annotation of the exp table")
    ap.add_argument("out_dir", help="folder for domains_interpro.tsv and interpro_cache.json")
    ap.add_argument("--no-load", action="store_true", help="only write the TSV, do not update the database")
    ap.add_argument("--all-domains", action="store_true", help="keep every InterPro domain/repeat (no vocabulary filter)")
    ap.add_argument("--db", help="database to annotate (default: db_path of database/config.json)")
    ap.add_argument("--table", default=None, help="exp table (default: exp_<classification> of database/config.json)")
    a = ap.parse_args()
    cfg = StartConfig()
    table, db = a.table or cfg.get_reference_table_name(), a.db or cfg.get_database_path()
    os.makedirs(a.out_dir, exist_ok=True)
    cache_path = os.path.join(a.out_dir, "interpro_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    con = sqlite3.connect(db)
    rows = con.execute(f"SELECT PDBId, UniprotId, ProteinSequence, ProteinChainIDs FROM {table}").fetchall()
    con.close()
    from tqdm import tqdm
    vocab = None if a.all_domains else vocabulary()
    if vocab is not None:
        print(f"domain vocabulary of the paper's annotation: {len(vocab)} InterPro accessions / short names")
    out = []
    try:
        for k, (pdb, uni, seq, chs) in enumerate(tqdm(rows, desc="InterPro domains", unit="entry"), 1):
            out.append(annotate((pdb, _list(uni), _list(seq), _list(chs)), cache, vocab))
            if k % 25 == 0:                       # keep the answers if the run is interrupted
                json.dump(cache, open(cache_path, "w"))
    finally:
        json.dump(cache, open(cache_path, "w"))

    tsv = os.path.join(a.out_dir, "domains_interpro.tsv")
    cols = ["PDBId", "domain_ids", "domain_names", "domain_counts", "domain_pos"]
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols + ["no_accession"]) + "\n")
        for r in out:
            fh.write("\t".join([r[c] for c in cols] + [str(int(r["no_accession"]))]) + "\n")
    n = sum(r["domain_ids"] != "-" for r in out)
    skip = [r["PDBId"] for r in out if r["no_accession"]]
    print(f"{len(out)} entries, {n} with an InterPro domain/repeat -> {tsv}")
    if skip:
        print(f"{len(skip)} entries without any UniProt accession (stored, merged or SIFTS) keep their current "
              f"annotation: {', '.join(skip)}")
    if not a.no_load:
        load = tsv.replace(".tsv", "_load.tsv")      # entries without an accession are not overwritten
        with open(load, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in out:
                if not r["no_accession"]:
                    fh.write("\t".join(r[c] for c in cols) + "\n")
        add_domains(load, db_path=db, table=table, id_col="PDBId")


if __name__ == "__main__":
    main()
