"""Do AF3-predicted interfaces sit on the sequence the RBP actually binds?

For each PDB entry whose protein maps to an RBP with known motifs, score the
interface footprints (exp = ground truth crystallography, pred = AF3) against
that RBP's motifs, and against a matched null of motifs from other RBPs.

Three motif arms are scored separately:
  all      -- every motif for the gene
  clip     -- CLIP-derived only (in vivo crosslinking)
  invitro  -- SELEX / RNAcompete / cisBP-RNA / RBPDB (in vitro)
The in-vitro arm is the assay-matched comparator: a PDB crystal structure and an
AF3 prediction of one are both in-vitro-like observations of a binary complex,
whereas CLIP reports binding in a living cell against the whole transcriptome.
"""
from __future__ import annotations
import csv, collections, json, random, sqlite3, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A
import atlas

N_NULL = 300
ARMS = ("all", "clip", "invitro")
# AF3's own confidence outputs (what the model reports about itself)
AF3_CONF = ["af3_rna_ipTM", "af3_rna_pTM", "af3_chain_pair_pae_min", "af3_ranking_score"]
# accuracy / composition metrics computed by us against the deposited structure
CALC_MET = ["RNAmotif_score", "RNAmatch_score", "RNA_LDDT", "iLDDT", "RNA_TM", "DockQ_Nr"]
MET = AF3_CONF + CALC_MET


def best_score(motifs, footprints):
    """Max normalised PPM score over all (motif, footprint) pairs, plus the argmax."""
    best, arg = float("-inf"), (None, None)
    for m in motifs:
        for f in footprints:
            s = A.normalised_score(m, f.seq)
            if s == s and s > best:
                best, arg = s, (m, f)
    return (None if best == float("-inf") else best), arg


def null_dist(rng, all_genes, by_gene, gene, footprints, arm):
    """Same footprints, motifs drawn from other RBPs -> motif-specificity null."""
    out = []
    for _ in range(N_NULL):
        g2 = rng.choice(all_genes)
        if g2 == gene:
            continue
        pool = [m for m in by_gene[g2] if arm == "all"
                or (arm == "clip" and m.clip) or (arm == "invitro" and not m.clip)]
        if not pool:
            continue
        m2 = rng.choice(pool)
        for f in footprints:
            v = A.normalised_score(m2, f.seq)
            if v == v:
                out.append(v)
                break
    return out


def main(bg=None, out=None):
    # bg: mononucleotide background (A, C, G, U) used for the log-odds conversion at
    # scoring time; None keeps af3clip.BACKGROUND (uniform).
    if bg is not None:
        A.BACKGROUND = tuple(bg)
    print(f"  background = {tuple(round(float(x), 4) for x in A.BACKGROUND)}")
    motifs = atlas.load_all()
    by_gene = collections.defaultdict(list)
    for m in motifs:
        by_gene[m.gene].append(m)

    rows = []
    for cohort in ("post", "pre"):
        rows += _cohort_rows(cohort, by_gene, all_genes=list(by_gene))
    return _write(rows, out)


def _cohort_rows(cohort, by_gene, all_genes):
    umap = {r['Entry']: r for r in csv.DictReader(
        open(A.ROOT/f'results/uniprot_map_{cohort}.tsv'), delimiter='\t')}
    acc2gene = {a: r['Gene Names (primary)'].upper() for a, r in umap.items()
                if r['Gene Names (primary)']}
    g2p, g2org = collections.defaultdict(set), {}
    for pdb, a, nm, org in json.load(open(A.ROOT/f'results/_pdb_acc_{cohort}.json')):
        g = acc2gene.get(a)
        if g:
            g2p[g].add(pdb)
            g2org[g] = "mouse" if org == "Mus musculus" else "human"
    targets = {g: sorted(v) for g, v in g2p.items() if g in by_gene}

    db = A.DBS[cohort]
    fps = {t: collections.defaultdict(list) for t in ("exp", "pred")}
    for t in ("exp", "pred"):
        for f in A.load_footprints(t, db=db):
            fps[t][f.entry].append(f)

    con = sqlite3.connect(db)
    # DockQ_Nr is derived per row; the AF3 confidence columns are per-chain JSON lists
    af3 = {}
    q = ("select exp_db_id, RBP_DockQ, ChainIDpairList_proteinRNA, "
         + ", ".join(AF3_CONF + ["RNAmotif_score", "RNAmatch_score",
                                 "RNA_LDDT", "iLDDT", "RNA_TM"])
         + " from pred_protein_rna")
    for row in con.execute(q):
        eid, rbpq, pairs = row[0], row[1], row[2]
        rest = row[3:]
        vals = {}
        for k, v in zip(AF3_CONF, rest[:len(AF3_CONF)]):
            vals[k] = A.af3_scalar(v)
        for k, v in zip(["RNAmotif_score", "RNAmatch_score", "RNA_LDDT", "iLDDT", "RNA_TM"],
                        rest[len(AF3_CONF):]):
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                vals[k] = float("nan")
        vals["DockQ_Nr"] = A.dockq_nr(rbpq, pairs)
        af3[eid] = [vals[m] for m in MET]
    # longest contact run at any length, for entries whose footprint falls under 4 nt
    anyrun = {"exp": {}, "pred": {}}
    for t, table, idcol in (("exp", "exp_protein_rna", "PDBId"),
                            ("pred", "pred_protein_rna", "exp_db_id")):
        for eid, mo in con.execute(f"select {idcol}, RNAMotif from {table}"):
            anyrun[t][eid] = A.longest_run(mo)[1]
    con.close()

    rng = random.Random(0)
    rows = []
    for gene, pdbs in sorted(targets.items()):
        pools = {"all": by_gene[gene],
                 "clip": [m for m in by_gene[gene] if m.clip],
                 "invitro": [m for m in by_gene[gene] if not m.clip]}
        for pdb in pdbs:
            rec = {"gene": gene, "pdb": pdb, "cohort": cohort,
                   "organism": g2org.get(gene, "human")}
            for arm in ARMS:
                rec[f"n_motif_{arm}"] = len(pools[arm])
            for t in ("exp", "pred"):
                F = fps[t].get(pdb, [])
                rec[f"n_fp_{t}"] = len(F)
                rec[f"maxfp_{t}"] = max((f.length for f in F), default=0)
                for arm in ARMS:
                    s, (bm, bf) = best_score(pools[arm], F)
                    rec[f"score_{arm}_{t}"] = s
                    rec[f"motif_{arm}_{t}"] = bm.consensus() if bm else ""
                    rec[f"fp_{arm}_{t}"] = bf.seq if bf else ""
                    if arm == "all":
                        rec[f"motif_{t}"] = bm.consensus() if bm else ""
                        rec[f"src_{t}"] = getattr(bm, "source", "") if bm else ""
                        rec[f"clip_{t}"] = int(bool(bm and bm.clip))
                        rec[f"fp_{t}"] = bf.seq if bf else ""
                    nd = null_dist(rng, all_genes, by_gene, gene, F, arm)
                    if nd and s is not None:
                        mu, sd = float(np.mean(nd)), float(np.std(nd))
                        rec[f"z_{arm}_{t}"] = (s - mu)/sd if sd > 0 else float("nan")
                        rec[f"p_{arm}_{t}"] = (sum(1 for v in nd if v >= s)+1)/(len(nd)+1)
                    else:
                        rec[f"z_{arm}_{t}"] = rec[f"p_{arm}_{t}"] = float("nan")
            for arm in ARMS:
                se, sp = rec.get(f"score_{arm}_exp"), rec.get(f"score_{arm}_pred")
                rec[f"d_score_{arm}"] = (sp - se) if (se is not None and sp is not None) else float("nan")
                ze, zp = rec[f"z_{arm}_exp"], rec[f"z_{arm}_pred"]
                rec[f"d_z_{arm}"] = zp - ze
            for t in ("exp", "pred"):
                rec[f"anyrun_{t}"] = anyrun[t].get(pdb, "")
            for k, v in zip(MET, af3.get(pdb, [None]*len(MET))):
                rec[k] = v
            rows.append(rec)
    print(f"  [{cohort}] {len(targets)} genes, {len(rows)} entries scored")
    return rows


def _write(rows, out=None):
    cols = (["gene", "pdb", "cohort", "organism"]
            + [f"n_motif_{a}" for a in ARMS]
            + [f"{c}_{t}" for c in ("n_fp", "maxfp") for t in ("exp", "pred")]
            + [f"{c}_{a}_{t}" for c in ("score", "z", "p") for a in ARMS for t in ("exp", "pred")]
            + [f"{c}_{t}" for c in ("motif", "src", "clip", "fp", "anyrun") for t in ("exp", "pred")]
            + [f"{c}_{a}_{t}" for c in ("motif", "fp") for a in ARMS for t in ("exp", "pred")]
            + [f"d_score_{a}" for a in ARMS] + [f"d_z_{a}" for a in ARMS] + MET)
    out = A.ROOT/'results/entry_scores.tsv' if out is None else Path(out)
    with open(out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out}  ({len(rows)} entry rows; arms: {', '.join(ARMS)})")
    return rows


def _parse_bg(txt):
    """'uniform' or four comma-separated A,C,G,U probabilities (renormalised)."""
    if txt is None or txt.lower() == "uniform":
        return None
    v = [float(x) for x in txt.split(",")]
    if len(v) != 4 or min(v) <= 0:
        raise SystemExit("--bg needs four positive A,C,G,U values, or 'uniform'")
    t = sum(v)
    return tuple(x / t for x in v)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bg", default="uniform",
                    help="background for the log-odds conversion: 'uniform' (default) "
                         "or four comma-separated A,C,G,U probabilities")
    ap.add_argument("--out", default=None, help="output TSV (default results/entry_scores.tsv)")
    a = ap.parse_args()
    main(bg=_parse_bg(a.bg), out=a.out)
