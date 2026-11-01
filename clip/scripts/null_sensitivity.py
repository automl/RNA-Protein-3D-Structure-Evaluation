"""Sensitivity of the motif z-scores to the null: are look-alike motifs of related RBPs inflating or deflating them?

run_analysis.py scores each interface footprint against the RBP's own motifs and against a null of motifs drawn from
other RBPs, excluding only the RBP itself. Paralogs (PUM1/PUM2, ELAVL1-4, HNRNPA1/A2B1, RBFOX1-3, ...) and unrelated
RBPs with similar motifs can therefore enter the null. Here the same analysis is repeated with nulls that exclude
  paralogs        other genes of the same family (gene symbol without its trailing number, e.g. ELAVL1 -> ELAVL)
  similar motifs  any motif resembling one of the RBP's own motifs: mean Pearson correlation of the position weight
                  matrix columns over the best ungapped overlap of >= 4 positions (Tomtom's column measure) above a
                  threshold (0.75; 0.60 as the stricter null)
  both            paralogs and similar motifs
and the headline statistics of stats.py are recomputed for each null (motif arm 'all').

Run from clip/scripts:  python null_sensitivity.py   -> ../results/null_sensitivity.txt
"""
from __future__ import annotations
import collections, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import af3clip as A
import atlas
import run_analysis as R

MIN_OVERLAP = 4


def family(gene: str) -> str:
    """Gene symbol without its trailing paralog number (ELAVL1 -> ELAVL, HNRNPA2B1 -> HNRNPA, U2AF2 -> U2AF)."""
    stem = re.sub(r"\d+[A-Z]?\d*$", "", gene.upper())
    return stem if len(stem) >= 3 else gene.upper()


def _col_pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def motif_similarity(m1, m2) -> float:
    """Best mean column PCC over ungapped overlaps of >= MIN_OVERLAP positions (RNA motifs: one strand only)."""
    x, y = m1.mat, m2.mat
    best = -1.0
    for off in range(-(len(y) - MIN_OVERLAP), len(x) - MIN_OVERLAP + 1):
        cols = [(i, i - off) for i in range(len(x)) if 0 <= i - off < len(y)]
        if len(cols) < MIN_OVERLAP:
            continue
        best = max(best, float(np.mean([_col_pcc(x[i], y[j]) for i, j in cols])))
    return best


def make_null(by_gene, mode: str, threshold: float):
    """null_dist replacement for run_analysis with the chosen exclusions."""
    sim_cache = {}

    def similar(gene, m2):
        key = (gene, id(m2))
        if key not in sim_cache:
            sim_cache[key] = max(motif_similarity(m, m2) for m in by_gene[gene]) >= threshold
        return sim_cache[key]

    def null_dist(rng, all_genes, by_gene_, gene, footprints, arm):
        out = []
        fam = family(gene)
        for _ in range(R.N_NULL):
            g2 = rng.choice(all_genes)
            if g2 == gene or (mode in ("paralogs", "both") and family(g2) == fam):
                continue
            pool = [m for m in by_gene_[g2] if arm == "all"
                    or (arm == "clip" and m.clip) or (arm == "invitro" and not m.clip)]
            if mode in ("similar", "both"):
                pool = [m for m in pool if not similar(gene, m)]
            if not pool:
                continue
            m2 = rng.choice(pool)
            for f in footprints:
                v = A.normalised_score(m2, f.seq)
                if v == v:
                    out.append(v)
                    break
        return out
    return null_dist


def summary(rows, label):
    ok = lambda r, t: r.get(f"z_all_{t}") == r.get(f"z_all_{t}") and r.get(f"z_all_{t}") is not None
    e = [r for r in rows if ok(r, "exp")]
    ze = np.array([r["z_all_exp"] for r in e]); pe = np.array([r["p_all_exp"] for r in e])
    pr = [r for r in rows if ok(r, "exp") and ok(r, "pred")]
    a = np.array([r["z_all_exp"] for r in pr]); b = np.array([r["z_all_pred"] for r in pr])
    k = int((pe < 0.05).sum())
    return (f"{label:34s} n={len(e):3d}  mean z_exp {ze.mean():+.2f} (median {np.median(ze):+.2f}; t P={stats.ttest_1samp(ze, 0).pvalue:.1e})"
            f"  p<0.05 {k}/{len(pe)}  | paired n={len(pr)} z_pred {b.mean():+.2f} (delta {b.mean() - a.mean():+.2f}; "
            f"paired t P={stats.ttest_rel(a, b).pvalue:.2g}, Wilcoxon P={stats.wilcoxon(b - a).pvalue:.2g}, "
            f"r={stats.pearsonr(a, b).statistic:.3f})")


def main():
    motifs = atlas.load_all()
    by_gene = collections.defaultdict(list)
    for m in motifs:
        by_gene[m.gene].append(m)
    all_genes = list(by_gene)
    lines = [f"{len(motifs)} motifs of {len(all_genes)} RBPs; null of {R.N_NULL} draws per entry (arm 'all')"]
    fams = collections.Counter(family(g) for g in all_genes)
    lines.append(f"families with >= 2 RBPs in the atlas: {sum(1 for v in fams.values() if v > 1)} "
                 f"({sum(v for v in fams.values() if v > 1)} RBPs)")
    original = R.null_dist
    for label, mode, thr in (("original (only the RBP excluded)", "none", 1.1),
                             ("paralogs excluded", "paralogs", 1.1),
                             ("similar motifs excluded (PCC>=0.75)", "similar", 0.75),
                             ("similar motifs excluded (PCC>=0.60)", "similar", 0.60),
                             ("paralogs + similar (PCC>=0.75)", "both", 0.75)):
        R.null_dist = original if mode == "none" else make_null(by_gene, mode, thr)
        rows = []
        for cohort in ("post", "pre"):
            rows += R._cohort_rows(cohort, by_gene, all_genes=all_genes)
        lines.append(summary(rows, label))
        print(lines[-1], flush=True)
    R.null_dist = original
    out = A.ROOT / "results/null_sensitivity.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
