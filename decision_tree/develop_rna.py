"""
Single-chain RNA subtree of the decision framework: one branch for predictions made WITH the RNA
alignment and one for predictions made WITHOUT it (AlphaFold Server runs carry no RNA alignment).

Why separate from the protein-RNA fold score (develop_v3): on single-chain RNAs AF3's own confidence
tracks accuracy far better than on protein-RNA complexes, its best measure depends on whether an RNA
alignment was used (pTM_r rho 0.95 with TM_r with, 0.71 without), and the protein-specific questions of
v3 (RBP template, >= 2 RNA chains) are always 0.

Design (fixed before any 2026 value is used):
  development  post-cutoff single-chain RNAs (data/compareDatabases/post_pre/a2021_b2021_rna_only/a2021_rna.db)
               with alignment: the a2021 runs; without alignment: the paired no-alignment runs
               (data/compareDatabases/noMSA/RNAonly_noMSA/noMSA_rna.db with their files in
               a2021_rnaOnly_add/a2021_x_noMSA/rna_rna/af) plus the RNAs without any alignment
               (depth 1), whose single run is both conditions.
  confidence   per branch the AF3 metric (pTM_r, mean RNA pLDDT, intra-RNA PAE) with the best development
               AUC for a well-folded RNA (LDDT_r > 0.85); levels 0 / 1 / 2 at its development tertiles.
  levels       WEAK / MODERATE / STRONG = confidence level 0 / 1 / 2. Confidence only: adding the familiarity
               points of the protein-RNA score (+1 sequence-familiar: >= 80 % identity to a pre-cutoff single-chain
               RNA; +1 fold-familiar: US-align TM >= 0.5 to a pre-cutoff RNA chain, RNA >= 20 nt) lowered the
               development AUC in both branches (0.97 vs 1.00 with, 0.81 vs 0.99 without the alignment); that
               score is still reported for comparison.
  test         applied once to the 2026 single-chain RNAs (released after the benchmark download):
               local runs with and without the alignment, and AlphaFold Server runs (no alignment);
               per entry and with one entry per sequence cluster (>= 80 % identity).

Run from the repo root (RNABench env):  python -m decision_tree.develop_rna
Outputs: results/plots/all_chains/decision_tree/rna_single/ and decision_tree/calibration_rna_single.json
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.chain_pae import intra_chain_pae
from decision_tree.develop import wilson
from decision_tree.lddt_labels import complex_lddt

D = lambda *p: os.path.join(_REPO_ROOT, *p)
OUT = D("results", "plots", "all_chains", "decision_tree", "rna_single")
CALIB = D("decision_tree", "calibration_rna_single.json")
GOOD_LDDT = 0.85
SEQ_FAMILIAR, TM_FAMILIAR, MIN_NT_TM = 0.80, 0.50, 20
PRE_DB = D("data", "compareDatabases", "post_pre", "a2021_b2021_rna_only", "b2021_rna.db")
PRE_CHAINS = D("results", "plots", "all_chains", "structural_similarity", "chains", "pre")
USALIGN = D("evaluation", "USalign", "USalign")
TWINS = D("results", "plots", "all_chains", "structural_similarity", "best_structural_twin_rna_only.tsv")
METRICS = ("ptm", "plddt", "rna_pae")          # higher is better after sign (PAE negated)
RNG = np.random.default_rng(0)

# the published databases and structures (fetch_data.sh; structures with --structures)
ALLDB, ALLST = D("data", "all_data_zip", "all_databases"), D("data", "all_data")
POST1 = os.path.join(ALLST, "a2021_post_cutoff")
SETS = {   # name: (database with accuracy, AF3 output folder, experimental cif folder, aligned folder)
    "design_with": (D("data", "compareDatabases", "post_pre", "a2021_b2021_rna_only", "a2021_rna.db"),
                    os.path.join(POST1, "rna_singlechain", "af"), os.path.join(POST1, "rna_singlechain", "pdb"), None),
    "design_without": (D("data", "compareDatabases", "noMSA", "RNAonly_noMSA", "noMSA_rna.db"),
                       os.path.join(POST1, "rna_singlechain_no_msa", "af"), os.path.join(POST1, "rna_singlechain", "pdb"), None),
    **{name: (os.path.join(ALLDB, "2026", f"{s}.db"), os.path.join(ALLST, "2026", s, "af"),
              os.path.join(ALLST, "2026", s, "pdb"), None)
       for name, s in (("test_local_with", "rna_singlechain_local_with_msa"),
                       ("test_local_without", "rna_singlechain_local_no_msa"),
                       ("test_server", "rna_singlechain_server"))},
}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _first(v):
    if v is None or (isinstance(v, str) and v.strip() in ("", "None", "[]")):
        return None
    try:
        v = json.loads(v) if isinstance(v, str) and v.strip()[:1] in "[{" else v
    except json.JSONDecodeError:
        return None
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _depth(v):
    try:
        d = json.loads(v) if isinstance(v, str) else v
        return max(d.values()) if isinstance(d, dict) and d else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def confidence(afdir, pid):
    """pTM_r, mean RNA pLDDT and intra-RNA PAE (longest RNA chain) of the seed-1 model-0 prediction."""
    s = os.path.join(afdir, f"fold_{pid}_s1_summary_confidences_0.json")
    f = os.path.join(afdir, f"fold_{pid}_s1_full_data_0.json")
    if not (os.path.exists(s) and os.path.exists(f)):
        return None
    S, F = json.load(open(s)), json.load(open(f))
    ic = intra_chain_pae(f)
    return dict(ptm=S.get("ptm"), plddt=float(np.mean(F["atom_plddts"])) / 100,
                rna_pae=-max(ic.values(), key=lambda t: t[1])[0] if ic else None)


def lddt_from_aligned(aligned, pid):
    from evaluation.extractEvalMetrics import calculate_complex_lddt
    ref, mdl = (os.path.join(aligned, f"{pid}_{k}_aligned_rna.pdb") for k in ("pdb", "af"))
    if not (os.path.exists(ref) and os.path.exists(mdl)):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        return calculate_complex_lddt(ref, mdl)


def load_set(name, depth_with=None):
    db, afdir, pdbdir, aligned = SETS[name]
    con = sqlite3.connect(db)
    rows = []
    # well folded: LDDT_r of the chosen label (decision_tree/lddt_labels.py; backbone C3' by default)
    labels = complex_lddt(db)
    for pid, seq, depth in con.execute("SELECT exp_db_id, RNASequence, RNA_msa_size FROM pred_rna_rna"):
        p = pid.lower()
        lddt = labels.get(pid.strip().upper())
        c = confidence(afdir, p)
        rows.append(dict(set=name, pdb=pid.upper(), seq=(seq or "").replace("&", ""), depth=_depth(depth),
                         lddt=lddt, cif=os.path.join(pdbdir, f"{p}.cif"), **(c or {k: None for k in METRICS})))
    return rows


def design_without(with_rows):
    """Paired no-alignment runs with files, plus the RNAs without any alignment (their only run counts for both)."""
    nm = {r["pdb"]: r for r in load_set("design_without")}
    out = []
    for w in with_rows:
        n = nm.get(w["pdb"])
        if n is not None and n["ptm"] is not None and n["lddt"] is not None:
            out.append(dict(n, depth=w["depth"]))
        elif w["depth"] is not None and w["depth"] <= 1 and w["ptm"] is not None:
            out.append(dict(w, set="design_without"))
    return out


def test_without(with_rows):
    """2026 local runs without the alignment; RNAs without any alignment use their single run."""
    nm = {r["pdb"]: r for r in load_set("test_local_without")}
    out = []
    for w in with_rows:
        n = nm.get(w["pdb"])
        if n is not None and n["ptm"] is not None and n["lddt"] is not None:
            out.append(dict(n, depth=w["depth"]))
        elif w["depth"] is not None and w["depth"] <= 1 and w["ptm"] is not None:
            out.append(dict(w, set="test_local_without"))
    return out


# --------------------------------------------------------------------------- #
# familiarity (same method for every set)
# --------------------------------------------------------------------------- #
def _identity(a, b):
    """identical aligned columns / length of the longer sequence (the paper's RNA identity)"""
    from evaluation.postpre_matched_similarity import RNA_AL, identity
    return identity(RNA_AL, a, b)


def seq_identity(seqs):
    """Best identity to any pre-cutoff single-chain RNA, after the paper's alignment-score pre-screen."""
    from evaluation.postpre_matched_similarity import rna_candidates
    pre = sorted({(s or "").replace("&", "") for (s,) in sqlite3.connect(PRE_DB).execute("SELECT RNASequence FROM pred_rna_rna")} - {""})
    qs = sorted({q for q in seqs.values() if q})
    with contextlib.redirect_stdout(io.StringIO()):
        cands = rna_candidates(qs, pre)
    best = {q: max((_identity(q, v) for v in cands.get(q, [])), default=0.0) for q in qs}
    return {pid: best[q] for pid, q in seqs.items() if q}


def structural_tm(cifs, cache_path):
    """Best US-align TM-score (normalised by the target chain) of the longest RNA chain against the pre-cutoff
    RNA chain library, as in evaluation/structural_similarity_to_pre.py; cached."""
    from evaluation.structural_similarity_to_pre import MIN_LEN_RATIO, rna_chains
    import gemmi
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    work = os.path.join(OUT, "chains")
    os.makedirs(work, exist_ok=True)
    pre = None
    for pid, cif in cifs.items():
        if pid in cache or not os.path.exists(cif):
            continue
        ch = rna_chains(cif, work, pid.lower())
        if not ch:
            continue
        name, _, n = max(ch, key=lambda t: t[2])
        if n < MIN_NT_TM:
            cache[pid] = [None, n]
            continue
        if pre is None:     # pre-cutoff chain library, read only when a chain has to be aligned
            from evaluation.postpre_matched_similarity import SYSTEMS, load
            with contextlib.redirect_stdout(io.StringIO()):     # pre-cutoff targets of the three sets, as in the paper
                valid = {r["id"][:4].lower() for cfg in SYSTEMS.values() for r in load(cfg["pre"], cfg["table"], cfg["protein"])}
            pre = [(f, gemmi.read_structure(os.path.join(PRE_CHAINS, f))[0][0].get_polymer().length())
                   for f in sorted(os.listdir(PRE_CHAINS)) if f.endswith(".pdb") and f[:4] in valid]
        lst = os.path.join(work, name + ".list")
        open(lst, "w").write("\n".join(f for f, ln in pre if ln >= MIN_LEN_RATIO * n) + "\n")
        res = subprocess.run([USALIGN, os.path.join(work, name), "-dir2", PRE_CHAINS + "/", lst, "-mol", "RNA",
                              "-outfmt", "2"], capture_output=True, text=True)
        tms = [float(l.split("\t")[2]) for l in res.stdout.splitlines() if l.strip() and not l.startswith("#")]
        cache[pid] = [max(tms) if tms else 0.0, n]
        json.dump(cache, open(cache_path, "w"))
    return cache


def add_familiarity(rows):
    seqs = {r["pdb"]: r["seq"] for r in rows}
    sid = seq_identity(seqs)
    tm = structural_tm({r["pdb"]: r["cif"] for r in rows}, os.path.join(OUT, "tm_cache.json"))
    for r in rows:
        r["seq_id"] = sid.get(r["pdb"])
        t, n = tm.get(r["pdb"], [None, None])
        r["tm"], r["rna_len"] = t, n
        r["seq_familiar"] = r["seq_id"] is not None and r["seq_id"] >= SEQ_FAMILIAR
        r["tm_familiar"] = t is not None and n is not None and n >= MIN_NT_TM and t >= TM_FAMILIAR


# --------------------------------------------------------------------------- #
# design and test
# --------------------------------------------------------------------------- #
def _auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def usable(rows):
    return [r for r in rows if r["lddt"] is not None and all(r[m] is not None for m in METRICS)]


def design(rows):
    y = [r["lddt"] > GOOD_LDDT for r in rows]
    aucs = {m: _auc(y, [r[m] for r in rows]) for m in METRICS}
    metric = max(METRICS, key=lambda m: (round(aucs[m], 3), m == "ptm"))
    cuts = [float(c) for c in np.quantile([r[metric] for r in rows], [1 / 3, 2 / 3])]
    return dict(metric=metric, metric_aucs=aucs, conf_cuts=cuts, n=len(rows), n_good=int(sum(y)))


def conf_level(v, cuts):
    return 0 if v <= cuts[0] else 2 if v > cuts[1] else 1


def score_of(r, d):
    return conf_level(r[d["metric"]], d["conf_cuts"])


def familiarity_score(r, d):
    """the rejected alternative: confidence level + familiarity points"""
    return conf_level(r[d["metric"]], d["conf_cuts"]) + int(r["seq_familiar"]) + int(r["tm_familiar"])


def level_of(s, d):
    return ("WEAK", "MODERATE", "STRONG")[s]


def clusters(rows):
    """one representative per sequence cluster (>= 80 % identity, single linkage)"""
    reps = []
    for r in sorted(rows, key=lambda r: r["pdb"]):
        if not any(_identity(r["seq"], q["seq"]) >= SEQ_FAMILIAR or _identity(q["seq"], r["seq"]) >= SEQ_FAMILIAR for q in reps):
            reps.append(r)
    return reps


def boot_diff(rows, d, n=2000):
    y = np.array([r["lddt"] > GOOD_LDDT for r in rows])
    a = np.array([score_of(r, d) for r in rows], float)
    b = np.array([r[d["metric"]] for r in rows], float)
    diffs = []
    for _ in range(n):
        i = RNG.integers(0, len(rows), len(rows))
        if y[i].all() or not y[i].any():
            continue
        diffs.append(_auc(y[i], a[i]) - _auc(y[i], b[i]))
    return np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)


def report(name, rows, d, lines):
    y = [r["lddt"] > GOOD_LDDT for r in rows]
    sc = [score_of(r, d) for r in rows]
    lines.append(f"\n{name}: n = {len(rows)}, well folded {sum(y)} ({100 * np.mean(y):.0f}%)")
    lines.append(f"  AUC levels {_auc(y, sc):.2f} | continuous {d['metric']} {_auc(y, [r[d['metric']] for r in rows]):.2f}"
                 f" | levels + familiarity points {_auc(y, [familiarity_score(r, d) for r in rows]):.2f}")
    for lv in ("WEAK", "MODERATE", "STRONG"):
        sel = [yy for yy, s in zip(y, sc) if level_of(s, d) == lv]
        if sel:
            p, l, u = wilson(sum(sel), len(sel))
            lines.append(f"  {lv:9s} {sum(sel):3d}/{len(sel):3d} well folded = {100 * p:.0f}% [{100 * l:.0f}-{100 * u:.0f}]")


def main():
    os.makedirs(OUT, exist_ok=True)
    with_rows = usable(load_set("design_with"))
    without_rows = usable(design_without(load_set("design_with")))
    t_with_all = load_set("test_local_with")
    t_with = usable(t_with_all)
    t_without = usable(test_without(t_with_all))
    t_server = usable(load_set("test_server"))
    for rs in (with_rows, without_rows, t_with, t_without, t_server):
        add_familiarity(rs)
    designs = {"with_alignment": design(with_rows), "without_alignment": design(without_rows)}
    for br, rows in (("with_alignment", with_rows), ("without_alignment", without_rows)):
        rates = {}
        for lv in ("WEAK", "MODERATE", "STRONG"):
            y = [r["lddt"] > GOOD_LDDT for r in rows if level_of(score_of(r, designs[br]), designs[br]) == lv]
            if y:
                p, l, u = wilson(sum(y), len(y))
                rates[lv] = dict(rate=p, ci_lo=l, ci_hi=u, n=len(y))
        designs[br]["dev_rates"] = rates          # for the decision-tree tool; cut-offs unchanged
    json.dump(designs, open(CALIB, "w"), indent=1)

    lines = ["Single-chain RNA subtree (decision_tree/develop_rna.py)", "=" * 60]
    for br, d in designs.items():
        lines.append(f"\nBranch {br}: development n = {d['n']} ({d['n_good']} well folded)")
        lines.append("  development AUC per confidence metric: " + ", ".join(f"{m} {a:.2f}" for m, a in d["metric_aucs"].items()))
        lines.append(f"  chosen {d['metric']}: WEAK <= {d['conf_cuts'][0]:.3f} < MODERATE <= {d['conf_cuts'][1]:.3f} < STRONG"
                     + ("  (intra-RNA PAE negated: higher is better)" if d['metric'] == "rna_pae" else ""))
    dw, dn = designs["with_alignment"], designs["without_alignment"]
    for name, rows, d in (("development, with alignment", with_rows, dw), ("development, without alignment", without_rows, dn),
                          ("TEST 2026 local, with alignment", t_with, dw), ("TEST 2026 local, without alignment", t_without, dn),
                          ("TEST 2026 AlphaFold Server (no alignment)", t_server, dn)):
        report(name, rows, d, lines)
        if name.startswith("TEST"):
            report(name + " - one per sequence cluster", clusters(rows), d, lines)
    txt = "\n".join(lines)
    open(os.path.join(OUT, "develop_rna_single.txt"), "w").write(txt + "\n")
    with open(os.path.join(OUT, "rows.tsv"), "w", newline="") as fh:
        keys = ["set", "pdb", "depth", "lddt", "ptm", "plddt", "rna_pae", "seq_id", "tm", "rna_len", "seq_familiar", "tm_familiar"]
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for rs in (with_rows, without_rows, t_with, t_without, t_server):
            w.writerows(rs)
    print(txt)


if __name__ == "__main__":
    main()
