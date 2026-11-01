"""Does AF3 contact the SAME RNA positions as the motif-bearing experimental footprint?

run_analysis.py scores footprints by sequence (PPM z vs an other-RBP null), so an AF3
footprint can score well while sitting on a different, motif-like stretch of the RNA (repeats,
U-tracts). With the exact residue mapping between experimental and predicted structures
(evaluation/exact_residue_mapping.py) the motif-bearing footprint can be followed position by
position:

  * the experimental footprint that best matches the RBP's motifs (arm "all", as in
    run_analysis.py) is mapped nucleotide by nucleotide onto the AF3 model;
  * recall_pair  = share of these nucleotides AF3 contacts with the mapped protein chain
    recall_any   = share AF3 contacts with any protein chain
    baseline     = share of the whole RNA chain AF3 contacts (chance level for recall_any)
  * same_site    = AF3's own best-scoring footprint overlaps the mapped experimental footprint

Writes results/positional.tsv and prints a summary (pooled and per cohort / recognition class).
Run in the RNABench env:  python positional.py
"""
from __future__ import annotations
import collections, csv, sqlite3, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A
import atlas
from run_analysis import best_score

REPO = A.ROOT.parent
sys.path.insert(0, str(REPO))
from evaluation.exact_residue_mapping import (ComplexMapping, _list, interface_scores, interfaces_of,
                                              locate_inputs)


def _pred_seqs(row):
    seqs = {}
    for chs, ss in ((row["pch"], row["pseq"]), (row["rch"], row["rseq"])):
        chs, ss = [str(c) for c in _list(chs)], [str(s) for s in _list(ss)]
        for i, c in enumerate(chs):
            seqs[c] = ss[i] if len(ss) > 1 else (ss[0] if ss else "")
    return seqs


def main():
    motifs = atlas.load_all()
    by_gene = collections.defaultdict(list)
    for m in motifs:
        by_gene[m.gene].append(m)
    cls = {r["gene"]: r["class"] for r in csv.DictReader(open(A.ROOT / "scripts/rbp_class.tsv"), delimiter="\t")}
    entries = list(csv.DictReader(open(A.ROOT / "results/entry_scores.tsv"), delimiter="\t"))

    out = []
    for cohort, db in A.DBS.items():
        fps = {t: collections.defaultdict(list) for t in ("exp", "pred")}
        for t in ("exp", "pred"):
            for f in A.load_footprints(t, db=db):
                fps[t][f.entry].append(f)
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        dbrows = {r["id"]: r for r in con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.AAMotif AS p_aa, p.RNAMotif AS p_rna,
                   p.ChainIDpairList_proteinRNA AS p_pairs, p.ProteinChainIDs AS pch,
                   p.ProteinSequence AS pseq, p.RNAChainIDs AS rch, p.RNASequence AS rseq,
                   e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e ON p.exp_db_id = e.PDBId""")}
        con.close()
        for ent in (e for e in entries if e["cohort"] == cohort):
            gene, pdb = ent["gene"], ent["pdb"]
            row = dbrows.get(pdb)
            _, (_, fe) = best_score(by_gene[gene], fps["exp"].get(pdb, []))
            if row is None or fe is None or fe.rna_chain is None:
                continue
            cif, js = locate_inputs(pdb, row["fn"], [str(REPO / "data")])
            if not cif:
                continue
            seqs = _pred_seqs(row)
            mapping = ComplexMapping(cif, js, pred_seqs=seqs)
            interface_scores(dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"],
                                  RNAMotif=row["p_rna"]),
                             dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"],
                                  RNAMotif=row["e_rna"]), mapping)
            fp_res = [(b, fe.start + i) for i, b in enumerate(fe.seq)]
            mapped, _ = mapping.map_motif(fe.rna_chain, fp_res, prefer="position")
            r, p = mapping.pred_chain(fe.rna_chain), mapping.pred_chain(fe.protein_chain)
            if not mapped or r is None:
                continue
            contacts = {(pc, rc): set(k for _, k in rna)
                        for pc, rc, _, rna in interfaces_of(row["p_pairs"], row["p_aa"], row["p_rna"])}
            on_pair = contacts.get((p, r), set())
            on_any = set().union(*[s for (pc, rc), s in contacts.items() if rc == r]) if contacts else set()
            rlen = len(seqs.get(r, "")) or np.nan
            _, (_, fpred) = best_score(by_gene[gene], fps["pred"].get(pdb, []))
            same = np.nan
            if fpred is not None:
                pred_pos = set(range(fpred.start, fpred.start + fpred.length))
                same = float(fpred.rna_chain == r and bool(pred_pos & mapped))
            out.append(dict(gene=gene, pdb=pdb, cohort=cohort, rbp_class=cls.get(gene, "sequence"),
                            fp_exp=fe.seq, n_fp=len(mapped),
                            recall_pair=len(mapped & on_pair) / len(mapped),
                            recall_any=len(mapped & on_any) / len(mapped),
                            baseline=len(on_any) / rlen, same_site=same,
                            z_exp=float(ent["z_all_exp"] or "nan"), z_pred=float(ent["z_all_pred"] or "nan"),
                            DockQ_Nr=float(ent["DockQ_Nr"] or "nan")))

    keys = list(out[0])
    with open(A.ROOT / "results/positional.tsv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    from scipy import stats
    def summary(label, rs):
        if not rs:
            return
        a = lambda k: np.array([r[k] for r in rs], float)
        rp, ra, bl, ss = a("recall_pair"), a("recall_any"), a("baseline"), a("same_site")
        ok = np.isfinite(ra) & np.isfinite(bl)
        w = stats.wilcoxon(ra[ok], bl[ok]) if ok.sum() > 5 else None
        print(f"  {label:28s} n={len(rs):3d}  recall pair {np.median(rp):.2f} (>=0.5: {np.mean(rp >= .5)*100:3.0f}%)"
              f"  any {np.median(ra):.2f}  vs baseline {np.nanmedian(bl):.2f}"
              + (f" (Wilcoxon P={w.pvalue:.1e})" if w else "")
              + f"  AF3 best motif hit on native site {np.nanmean(ss)*100:3.0f}%")
    print(f"\nmotif-bearing experimental footprints followed into the AF3 model: {len(out)} entries")
    summary("all", out)
    for c in ("post", "pre"):
        summary(f"cohort {c}", [r for r in out if r["cohort"] == c])
    for c in ("sequence", "structure", "guide"):
        summary(f"class {c}", [r for r in out if r["rbp_class"] == c])
    zp, rp = np.array([r["z_pred"] for r in out]), np.array([r["recall_pair"] for r in out])
    m = np.isfinite(zp)
    print(f"  Spearman(recall_pair, z_pred) = {stats.spearmanr(rp[m], zp[m])[0]:+.2f} (n={m.sum()})")
    hi = [r for r in out if r["z_pred"] == r["z_pred"] and r["z_pred"] >= 1.5]
    print(f"  AF3 footprints scoring as motif (z_pred>=1.5, n={len(hi)}): "
          f"on the native positions (recall_pair>=0.5) {np.mean([r['recall_pair'] >= .5 for r in hi])*100:.0f}%, "
          f"elsewhere (recall_pair<0.2) {np.mean([r['recall_pair'] < .2 for r in hi])*100:.0f}%")
    print(f"\nwrote {A.ROOT / 'results/positional.tsv'}")


if __name__ == "__main__":
    main()
