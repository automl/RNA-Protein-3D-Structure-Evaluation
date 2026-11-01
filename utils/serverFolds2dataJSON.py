#!/usr/bin/env python3
"""
Convert AlphaFold Server downloads (fold folders) into local-AF3 style *_data.json files.

addMSA_fromJSON.py (MSA depth) and database/extractTemplatesAF3.py (template PDB ids) read the
*_data.json written by a local AlphaFold3 run, where every entity carries its MSA and templates
inline. An AlphaFold Server download stores the same information as separate files instead:

    <job>/fold_<job>_job_request.json                          entities (sequence, count)
    <job>/msas/fold_<job>_unpaired_msa_chains_<x>.a3m           MSA per chain group
    <job>/templates/fold_<job>_template_hit_<k>_chains_<x>.cif  template hit (full mmCIF entry)
    <job>/templates/fold_<job>_template_hits_chains_<x>_query_to_hit.json  residue mapping

This script writes <out_dir>/<job>_data.json with, per entity, "unpairedMsa", "pairedMsa" and
"templates" ([{"mmcif", "queryIndices", "templateIndices"}]) so both scripts run unchanged.
The chain-group letters of the server files (chains_a, chains_b_c, ...) number the chains that carry
an MSA (the protein chains), not every entity of the job request. Each a3m is therefore assigned
to its entity by its query sequence, and templates follow the letters of those assignments
(fallback: the protein entities in job-request order, expanded by "count").

The server download contains MSAs for protein chains only (no RNA a3m files), so RNA entities get
no "unpairedMsa"; they are left out of the output so that addMSA_fromJSON.py does not record a
depth of 1 ("no alignment") for an RNA whose alignment is simply not exported.

Usage:
    python utils/serverFolds2dataJSON.py <folds folder> <out_dir> [--overwrite]
The folds folder may contain job folders directly or nested download folders (folds_YYYY_MM_DD_*).
"""

import argparse
import glob
import json
import os
import re
import string


def _entities(job_request):
    j = json.load(open(job_request))
    j = j[0] if isinstance(j, list) else j
    ents, all_ids, prot_letters = [], iter(string.ascii_uppercase), iter(string.ascii_lowercase)
    for s in j.get("sequences", []):
        for kind, v in s.items():
            n = int(v.get("count", 1))
            e = dict(kind=kind, sequence=v.get("sequence", ""), ids=[next(all_ids) for _ in range(n)], letters=[])
            if kind == "proteinChain":
                e["letters"] = [next(prot_letters) for _ in range(n)]
            ents.append(e)
    return j.get("name"), ents


def _query(a3m):
    seq, seen = [], False
    for line in open(a3m):
        if line.startswith(">"):
            if seen:
                break
            seen = True
        elif seen:
            seq.append(line.strip())
    return "".join(seq).upper()


def _group_letters(path, tag):
    m = re.search(rf"{tag}_chains_([A-Za-z_]+?)(?:_query_to_hit)?\.(?:a3m|cif|json)$", os.path.basename(path))
    return m.group(1).split("_") if m else []


def _entity_of(letters, ents, by_letter):
    if letters and letters[0] in by_letter:
        return by_letter[letters[0]]
    for e in ents:
        if letters and letters[0] in e["letters"]:
            return e
    return None


def convert(job_dir, out_dir, overwrite=False):
    reqs = glob.glob(os.path.join(job_dir, "*_job_request.json"))
    if not reqs:
        return None
    name, ents = _entities(reqs[0])
    name = name or os.path.basename(job_dir)
    out = os.path.join(out_dir, f"{name}_data.json")
    if os.path.exists(out) and not overwrite:
        return "exists"

    by_letter = {}
    for kind in ("unpaired", "paired"):
        for a3m in glob.glob(os.path.join(job_dir, "msas", f"*_{kind}_msa_chains_*.a3m")):
            q = _query(a3m)
            letters = _group_letters(a3m, f"{kind}_msa")
            # by query sequence; identical chains listed as separate entities share one a3m
            hits = [x for x in ents if x["sequence"].upper() == q] or [_entity_of(letters, ents, by_letter)]
            text = open(a3m).read()
            for e in hits:
                if e is not None:
                    e[f"{kind}Msa"] = text
            if hits[0] is not None:
                by_letter.update({l: hits[0] for l in letters})

    for qth in glob.glob(os.path.join(job_dir, "templates", "*_query_to_hit.json")):
        e = _entity_of(_group_letters(qth, "template_hits"), ents, by_letter)
        if e is None:
            continue
        hits = []
        for hit in json.load(open(qth)):
            cif = os.path.join(job_dir, "templates", hit["name"])
            if os.path.exists(cif):
                hits.append(dict(mmcif=open(cif).read(), queryIndices=hit.get("queryIndices", []),
                                 templateIndices=hit.get("templateIndices", [])))
        for x in ents:
            if x is e or (x["kind"] == e["kind"] and x["sequence"] == e["sequence"]):
                x.setdefault("templates", []).extend(hits)

    seqs = []
    for e in ents:
        mol = {"proteinChain": "protein", "rnaSequence": "rna", "dnaSequence": "dna"}.get(e["kind"], e["kind"])
        if mol != "protein" and "unpairedMsa" not in e:
            continue                                    # no exported alignment: unknown, not "none"
        entry = dict(id=e["ids"], sequence=e["sequence"])
        for k in ("unpairedMsa", "pairedMsa", "templates"):
            if k in e:
                entry[k] = e[k]
        seqs.append({mol: entry})
    json.dump(dict(name=name, sequences=seqs, dialect="alphafold3", source="alphafoldserver"),
              open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("folds", help="folder with AlphaFold Server job folders (searched recursively)")
    ap.add_argument("out_dir", help="where to write <job>_data.json (e.g. the af/ processing folder)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(a.folds, "**", "*_job_request.json"),
                                                         recursive=True)})
    written = kept = 0
    for job in jobs:
        r = convert(job, a.out_dir, a.overwrite)
        written += r not in (None, "exists")
        kept += r == "exists"
    print(f"{len(jobs)} AlphaFold Server jobs: {written} *_data.json written, {kept} already present -> {a.out_dir}")


if __name__ == "__main__":
    main()
