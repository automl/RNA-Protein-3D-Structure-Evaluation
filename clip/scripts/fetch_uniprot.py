"""Map the structure DB's human/mouse UniProt accessions to gene symbols."""
import json, re, sqlite3, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A

ACC_RE = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")

def main(cohort="post"):
    db = A.DBS[cohort]
    con = sqlite3.connect(db)
    rows = con.execute("select PDBId, UniprotId, ProteinName, ProteinOrganism "
                       "from exp_protein_rna").fetchall()
    con.close()
    acc, pairs = set(), []
    for pdb, ui, pn, po in rows:
        U, N, O = A.jl(ui), A.jl(pn), A.jl(po)
        for i, u in enumerate(U):
            org = O[i] if i < len(O) else (O[0] if O else "")
            nm = N[i] if i < len(N) else ""
            if org in ("Homo sapiens", "Mus musculus") and ACC_RE.fullmatch(str(u)):
                acc.add(u); pairs.append((pdb, u, nm, org))
    (A.ROOT/'results').mkdir(exist_ok=True)
    (A.ROOT/f'results/_acc_{cohort}.txt').write_text("\n".join(sorted(acc)))
    json.dump(pairs, open(A.ROOT/f'results/_pdb_acc_{cohort}.json', 'w'))
    out_tsv = A.ROOT/f'results/uniprot_map_{cohort}.tsv'
    # UniProt truncates very long accession queries -- fetch in chunks and concatenate.
    CHUNK, header, body = 40, None, []
    accs = sorted(acc)
    for k in range(0, len(accs), CHUNK):
        part = accs[k:k+CHUNK]
        tmp = A.ROOT/f'results/_uni_chunk_{cohort}.tsv'
        subprocess.run(["curl", "-sS", "--max-time", "120", "-G",
            "https://rest.uniprot.org/uniprotkb/search",
            "--data-urlencode", "query=accession:(" + " OR ".join(part) + ")",
            "--data-urlencode", "fields=accession,id,gene_primary,protein_name,organism_name",
            "--data-urlencode", "format=tsv", "--data-urlencode", "size=500",
            "-o", str(tmp)], check=True)
        lines = tmp.read_text().splitlines()
        if not lines:
            continue
        header = header or lines[0]
        body += lines[1:]
        tmp.unlink()
    out_tsv.write_text("\n".join([header or ""] + body) + "\n")
    print(f"[{cohort}] {len(acc)} accessions -> {len(body)} UniProt records")


def fetch_domains():
    """Pfam/InterPro records for every gene in the recognition-class annotation."""
    import csv as _csv
    genes = {r["gene"] for r in
             _csv.DictReader(open(A.ROOT / "scripts/rbp_class.tsv"), delimiter="\t")}
    acc = {}
    for cohort in ("post", "pre"):
        f = A.ROOT / f"results/uniprot_map_{cohort}.tsv"
        if not f.exists():
            continue
        for r in _csv.DictReader(open(f), delimiter="\t"):
            g = (r["Gene Names (primary)"] or "").upper()
            if g in genes:
                acc.setdefault(g, r["Entry"])
    accs = sorted(acc.values())
    header, body = None, []
    for k in range(0, len(accs), 40):
        tmp = A.ROOT / "results/_dom_chunk.tsv"
        subprocess.run(["curl", "-sS", "--max-time", "120", "-G",
            "https://rest.uniprot.org/uniprotkb/search",
            "--data-urlencode", "query=accession:(" + " OR ".join(accs[k:k+40]) + ")",
            "--data-urlencode",
            "fields=accession,gene_primary,protein_name,xref_pfam,xref_interpro",
            "--data-urlencode", "format=tsv", "--data-urlencode", "size=500",
            "-o", str(tmp)], check=True)
        lines = tmp.read_text().splitlines()
        if lines:
            header = header or lines[0]
            body += lines[1:]
        tmp.unlink()
    out = A.ROOT / "results/uniprot_domains.tsv"
    out.write_text("\n".join([header or ""] + body) + "\n")
    print(f"[domains] {len(acc)} genes -> {len(body)} domain records")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["post", "pre"]):
        main(c)
    fetch_domains()
