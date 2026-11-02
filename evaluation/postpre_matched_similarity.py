"""
Similarity-matched pre/post-cutoff comparison of AF3 RNA accuracy.

Instead of comparing the post-cutoff (a2021) and pre-cutoff (b2021) sets as two unpaired groups,
every post-cutoff target is paired with a pre-cutoff counterpart and accuracy is compared WITHIN
the pair, as a function of how similar the two are.

Similarity
  * RNA identity      best identity over all RNA-chain pairs of the two targets (global alignment;
                      identical columns / length of the LONGER sequence, as in Ludaic & Elofsson
                      2026, so a short probe cannot score high against a long RNA by containment).
  * protein identity  (protein-RNA only) the same over all protein-chain pairs (BLOSUM62).
  * template link     whether AF3 used the matched pre-cutoff complex as an RBP template for the
                      post-cutoff target (RBP_PDB_Templates).
Pairwise identity is computed directly (no CD-HIT-EST: cd-hit-est discards sequences shorter than
its word length, and many protein-bound RNAs are 9-12 nt). Candidates are pre-screened (RNA:
normalised alignment score; protein: shared 3-mers) and the top candidates aligned exactly.

Matching is always on the RNA: each post target is paired with the pre target with the highest
RNA identity (ties -> protein identity), because the generalization gap is RNA-specific.

Analyses
  A  RNA-identity bins (both systems): pairs binned by RNA identity (>= 0.8, the CD-HIT-EST 80%
     redundancy level; [0.5, 0.8); < 0.5).
  B  Protein stratification (protein-RNA): the same RNA-matched pairs, each bin split by whether
     the RNA twin also carries the same protein (protein identity >= 0.8) or a different one --
     i.e. does a near-zero gap for seen RNAs depend on AF3 also having seen the protein?
Within every subset the paired difference (post - matched pre) in RNA_LDDT and RNA_TM is
summarised by its median with a bootstrap 95% CI, plus the share of well-folded RNAs (LDDT_r>0.85).

Run in the RNABench env (biopython, numpy):
  python evaluation/postpre_matched_similarity.py
"""

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POSTPRE = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre")
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "postpre_matched")

SYSTEMS = {
    "protein_rna": dict(post=os.path.join(_POSTPRE, "a2021_b2021_protein_rna", "a2021.db"),
                        pre=os.path.join(_POSTPRE, "a2021_b2021_protein_rna", "b2021.db"),
                        table="pred_protein_rna", protein=True),
    "rna_only": dict(post=os.path.join(_POSTPRE, "a2021_b2021_rna_only", "a2021_rna.db"),
                     pre=os.path.join(_POSTPRE, "a2021_b2021_rna_only", "b2021_rna.db"),
                     table="pred_rna_rna", protein=False),
    "rna_multichain": dict(post=os.path.join(_POSTPRE, "a2021_b2021_rna_rna", "a2021_rna_rna.db"),
                           pre=os.path.join(_POSTPRE, "a2021_b2021_rna_rna", "b2021_rna_rna.db"),
                           table="pred_rna_rna", protein=False),
}
BINS = [(0.8, 1.01, ">=80%"), (0.5, 0.8, "50-80%"), (0.0, 0.5, "<50%")]
SAME_PROTEIN = 0.8   # protein identity of the RNA twin counted as "same protein"
GOOD_RNA_LDDT = 0.85
TOP_K_RNA = 25      # RNA candidates re-aligned exactly after the score pre-screen
TOP_K_PROT = 15     # protein candidates re-aligned exactly after the 3-mer pre-screen
N_BOOT = 5000
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Aligners / identity
# --------------------------------------------------------------------------- #
def _rna_aligner():
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score, a.mismatch_score = 2, -3          # blastn-like
    a.open_gap_score, a.extend_gap_score = -5, -2
    return a


def _protein_aligner():
    a = Align.PairwiseAligner()
    a.mode = "global"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score = -10, -0.5
    return a


RNA_AL, PROT_AL = _rna_aligner(), _protein_aligner()


def identity(aligner, s1, s2):
    """Identical aligned columns / length of the longer sequence."""
    if not s1 or not s2:
        return 0.0
    aln = aligner.align(s1, s2)[0]
    same = 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        same += sum(x == y for x, y in zip(s1[a0:a1], s2[b0:b1]))
    return same / max(len(s1), len(s2))


class IdentityCache:
    """Memoised sequence-pair identity."""

    def __init__(self, aligner):
        self.al, self.d = aligner, {}

    def __call__(self, a, b):
        k = (a, b)
        if k not in self.d:
            self.d[k] = identity(self.al, a, b)
        return self.d[k]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _list(raw):
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    except json.JSONDecodeError:
        return [s]


_BLOSUM_AA = set("ARNDCQEGHILKMFPSTWYVBZX")


def _clean(seq, rna):
    seq = "".join(c for c in seq.upper() if c.isalpha())
    if rna:
        return seq.replace("T", "U")
    return "".join(c if c in _BLOSUM_AA else "X" for c in seq)   # e.g. U (Sec), O (Pyl) -> X


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(db, table, protein):
    # all-atom LDDT_r for the paired differences (columns *_RNA_LDDT below hold it); the backbone C3' LDDT (RNA_LDDT)
    # only for the well-folded share (> GOOD_RNA_LDDT), as in the fold classes i-iv
    cols = "exp_db_id, RNASequence, RNA_TM, RNA_LDDT_allAtom, RNA_LDDT"
    if protein:
        cols += ", ProteinSequence, RBP_PDB_Templates"
    out = []
    for row in sqlite3.connect(db).execute(f"SELECT {cols} FROM {table}"):
        rec = dict(id=str(row[0]).strip().upper(),
                   rna=[s for s in (_clean(x, True) for x in _list(row[1])) if s],
                   tm=_f(row[2]), lddt=_f(row[3]), lddt_bb=_f(row[4]))
        if protein:
            rec["prot"] = [s for s in (_clean(x, False) for x in _list(row[5])) if s]
            rec["templates"] = {t.strip().upper()[:4] for t in _list(row[6])}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Candidate pre-screens
# --------------------------------------------------------------------------- #
def rna_candidates(post_seqs, pre_seqs):
    """{post RNA seq: [top pre RNA seqs]} by alignment score normalised by the longer length."""
    out = {}
    for i, u in enumerate(post_seqs):
        scored = sorted(((RNA_AL.score(u, v) / max(len(u), len(v)), v) for v in pre_seqs),
                        reverse=True)[:TOP_K_RNA]
        out[u] = [v for _, v in scored]
        if (i + 1) % 100 == 0:
            print(f"    RNA pre-screen {i + 1}/{len(post_seqs)}", flush=True)
    return out


def protein_candidates(post_seqs, pre_seqs, k=3):
    """{post protein seq: [top pre protein seqs]} by shared k-mers (inverted index)."""
    index = defaultdict(set)
    kmers_pre = {}
    for v in pre_seqs:
        km = {v[i:i + k] for i in range(len(v) - k + 1)}
        kmers_pre[v] = len(km)
        for x in km:
            index[x].add(v)
    out = {}
    for u in post_seqs:
        km = {u[i:i + k] for i in range(len(u) - k + 1)}
        hits = Counter()
        for x in km:
            hits.update(index.get(x, ()))
        # fraction of the larger k-mer set that is shared ~ identity over the longer sequence
        ranked = sorted(hits, key=lambda v: -hits[v] / max(len(km), kmers_pre[v]))
        out[u] = ranked[:TOP_K_PROT]
    return out


# --------------------------------------------------------------------------- #
# Pair similarities for every post target
# --------------------------------------------------------------------------- #
def candidate_pairs(post, pre, protein):
    """For every post target: list of (pre target, rna identity, protein identity|None)."""
    pre_rna = sorted({s for r in pre for s in r["rna"]})
    post_rna = sorted({s for r in post for s in r["rna"]})
    rna_cand = rna_candidates(post_rna, pre_rna)
    by_rna = defaultdict(list)
    for q in pre:
        for s in q["rna"]:
            by_rna[s].append(q)
    rna_id = IdentityCache(RNA_AL)

    prot_cand, by_prot, prot_id = {}, defaultdict(list), None
    if protein:
        pre_p = sorted({s for r in pre for s in r["prot"]})
        post_p = sorted({s for r in post for s in r["prot"]})
        print(f"    protein pre-screen: {len(post_p)} x {len(pre_p)} unique sequences", flush=True)
        prot_cand = protein_candidates(post_p, pre_p)
        for q in pre:
            for s in q["prot"]:
                by_prot[s].append(q)
        prot_id = IdentityCache(PROT_AL)

    result = {}
    for i, p in enumerate(post):
        cand = {}
        for u in p["rna"]:
            for v in rna_cand.get(u, []):
                for q in by_rna[v]:
                    cand[q["id"]] = q
        if protein:
            for u in p["prot"]:
                for v in prot_cand.get(u, []):
                    for q in by_prot[v]:
                        cand[q["id"]] = q
        rows = []
        for q in cand.values():
            r_id = max((rna_id(a, b) for a in p["rna"] for b in q["rna"]), default=0.0)
            p_id = (max((prot_id(a, b) for a in p["prot"] for b in q["prot"]), default=0.0)
                    if protein else None)
            rows.append((q, r_id, p_id))
        result[p["id"]] = rows
        if (i + 1) % 100 == 0:
            print(f"    pair identities {i + 1}/{len(post)}", flush=True)
    return result


def _pair(p, q, r_id, p_id, group=None):
    return dict(post=p, pre=q, rna_id=r_id, prot_id=p_id, group=group,
                template=bool(p.get("templates")) and q["id"][:4] in p.get("templates", set()))


def match_by_rna(post, cands):
    """Analysis A: nearest pre target by RNA identity (ties -> protein identity)."""
    pairs = []
    for p in post:
        rows = cands.get(p["id"])
        if rows:
            q, r_id, p_id = max(rows, key=lambda t: (t[1], t[2] or 0.0))
            pairs.append(_pair(p, q, r_id, p_id))
    return pairs


def bin_name(rna_id):
    return next(name for lo, hi, name in BINS if lo <= rna_id < hi)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def boot_median_ci(d):
    d = np.asarray(d, float)
    if len(d) < 3:
        return np.nan, np.nan
    meds = np.median(RNG.choice(d, size=(N_BOOT, len(d)), replace=True), axis=1)
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def _pct_good(sub, side):
    """share of well-folded RNAs (backbone C3' LDDT_r > GOOD_RNA_LDDT) among pairs with both backbone values"""
    ok = [x for x in sub if x["post"]["lddt_bb"] is not None and x["pre"]["lddt_bb"] is not None]
    return 100 * np.mean([x[side]["lddt_bb"] > GOOD_RNA_LDDT for x in ok]) if ok else np.nan


def summarize_subset(sub, system, analysis, name):
    rows, lines = [], [f"  [{name:>15s}] n={len(sub):3d}"]
    for key, lab in (("lddt", "LDDT_r"), ("tm", "TM_r")):
        ok = [x for x in sub if x["post"][key] is not None and x["pre"][key] is not None]
        if not ok:
            continue
        a = np.array([x["post"][key] for x in ok]); b = np.array([x["pre"][key] for x in ok])
        d = a - b; lo, hi = boot_median_ci(d)
        lines.append(f"{lab}: pre {np.median(b):.3f} -> post {np.median(a):.3f}, "
                     f"paired diff {np.median(d):+.3f} [{lo:+.3f},{hi:+.3f}]")
        rows.append(dict(system=system, analysis=analysis, subset=name, metric=lab, n=len(ok),
                         pre_median=np.median(b), post_median=np.median(a),
                         diff_median=np.median(d), ci_lo=lo, ci_hi=hi,
                         pct_good_pre=_pct_good(sub, "pre") if key == "lddt" else np.nan,
                         pct_good_post=_pct_good(sub, "post") if key == "lddt" else np.nan,
                         median_rna_id=np.median([x["rna_id"] for x in sub]),
                         median_prot_id=(np.median([x["prot_id"] for x in sub])
                                         if sub[0]["prot_id"] is not None else np.nan)))
    print("; ".join(lines))
    ok = [x for x in sub if x["post"]["lddt_bb"] is not None and x["pre"]["lddt_bb"] is not None]
    if ok:
        extra = (f"{'':21s}good RNA (backbone LDDT) pre {_pct_good(sub, 'pre'):.0f}%"
                 f" -> post {_pct_good(sub, 'post'):.0f}%"
                 f"; matched RNA id {np.median([x['rna_id'] for x in sub]):.2f}")
        if sub[0]["prot_id"] is not None:
            extra += (f", protein id {np.median([x['prot_id'] for x in sub]):.2f}"
                      f"; matched pre complex used as RBP template {sum(x['template'] for x in sub)}/{len(sub)}")
        print(extra)
    return rows


def write_pairs(pairs, path):
    with open(path, "w") as fh:
        fh.write("post_id\tpre_id\tgroup\trna_identity\tprotein_identity\tpre_is_rbp_template\t"
                 "post_RNA_LDDT\tpre_RNA_LDDT\tpost_RNA_TM\tpre_RNA_TM\n")
        for x in pairs:
            fh.write("\t".join(str(v) for v in (
                x["post"]["id"], x["pre"]["id"], x["group"] or "", f"{x['rna_id']:.3f}",
                "" if x["prot_id"] is None else f"{x['prot_id']:.3f}", int(x["template"]),
                x["post"]["lddt"], x["pre"]["lddt"], x["post"]["tm"], x["pre"]["tm"])) + "\n")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for system, cfg in SYSTEMS.items():
        print(f"\n### {system}: loading", flush=True)
        post = load(cfg["post"], cfg["table"], cfg["protein"])
        pre = load(cfg["pre"], cfg["table"], cfg["protein"])
        print(f"    post n={len(post)}, pre n={len(pre)}", flush=True)
        cands = candidate_pairs(post, pre, cfg["protein"])

        pairs = match_by_rna(post, cands)
        ids = np.array([x["rna_id"] for x in pairs])
        print(f"\n=== {system} A) nearest pre target by RNA identity (n={len(pairs)}; "
              f"median identity {np.median(ids):.2f}) ===")
        for lo, hi, name in BINS:
            sub = [x for x in pairs if lo <= x["rna_id"] < hi]
            if sub:
                rows += summarize_subset(sub, system, "rna_identity_bin", name)

        if cfg["protein"]:
            for x in pairs:
                x["group"] = (bin_name(x["rna_id"]) + " RNA, "
                              + ("same protein" if x["prot_id"] >= SAME_PROTEIN else "other protein"))
            cnt = Counter(x["group"] for x in pairs)
            print(f"\n=== {system} B) RNA-matched pairs split by the twin's protein "
                  f"(same = identity >= {SAME_PROTEIN:.0%}) ===")
            for lo, hi, name in BINS:
                for prot in ("same protein", "other protein"):
                    g = f"{name} RNA, {prot}"
                    sub = [x for x in pairs if x["group"] == g]
                    if sub:
                        rows += summarize_subset(sub, system, "rna_bin_x_protein", g)
            print("    counts: " + ", ".join(f"{g}={n}" for g, n in sorted(cnt.items())))
        write_pairs(pairs, os.path.join(OUTDIR, f"matched_pairs_{system}.tsv"))

    keys = ["system", "analysis", "subset", "metric", "n", "pre_median", "post_median",
            "diff_median", "ci_lo", "ci_hi", "pct_good_pre", "pct_good_post",
            "median_rna_id", "median_prot_id"]
    with open(os.path.join(OUTDIR, "matched_summary.tsv"), "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
                               for k in keys) + "\n")
    print(f"\nWritten to {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
