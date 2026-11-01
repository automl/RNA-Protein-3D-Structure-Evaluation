"""Map structure-DB proteins -> gene symbols -> motif catalog; emit the analysis set."""
import sys, csv, json, collections
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import af3clip as A
import atlas

def main():
    # both datasets: a2021 (post-cutoff) and b2021 (pre-cutoff)
    acc2gene, pairs = {}, []
    for cohort in ("post", "pre"):
        for r in csv.DictReader(open(A.ROOT/f'results/uniprot_map_{cohort}.tsv'), delimiter='\t'):
            if r['Gene Names (primary)']:
                acc2gene[r['Entry']] = r['Gene Names (primary)'].upper()   # ortholog-match mouse
        pairs += json.load(open(A.ROOT/f'results/_pdb_acc_{cohort}.json'))
    ppms = atlas.load_all()
    by_gene = collections.defaultdict(list)
    for p in ppms:
        by_gene[p.gene.upper()].append(p)

    gene2pdb = collections.defaultdict(set)
    for pdb, a, nm, org in pairs:
        g = acc2gene.get(a)
        if g:
            gene2pdb[g].add(pdb)

    hit = {g: sorted(v) for g, v in gene2pdb.items() if g in by_gene}
    with open(A.ROOT/'results/overlap.tsv','w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['gene','n_pdb','n_ppm','n_clip_ppm','clip_consensus','invitro_consensus','pdb_ids'])
        for g in sorted(hit, key=lambda x: (-sum(p.is_clip for p in by_gene[x]), -len(hit[x]))):
            ps = by_gene[g]
            w.writerow([g, len(hit[g]), len(ps), sum(getattr(p,'clip',p.is_clip) for p in ps),
                        ';'.join(p.consensus() for p in ps if getattr(p,'clip',p.is_clip)),
                        ';'.join(p.consensus() for p in ps if not getattr(p,'clip',p.is_clip)),
                        ','.join(hit[g])])
    json.dump(hit, open(A.ROOT/'results/gene2pdb.json','w'), indent=1)
    n_all = len(set().union(*hit.values())) if hit else 0
    clip = {g: v for g, v in hit.items() if any(getattr(p,'clip',p.is_clip) for p in by_gene[g])}
    n_clip = len(set().union(*clip.values())) if clip else 0
    print(f"genes in structure DB with a PPM : {len(hit):>3}  ({n_all} PDB entries)")
    print(f"          ... with a CLIP-derived: {len(clip):>3}  ({n_clip} PDB entries)")
    return hit, by_gene

if __name__ == '__main__':
    main()
