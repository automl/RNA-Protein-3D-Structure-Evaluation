"""
Multi-chain RNA subtree of the decision framework (RNA complexes without protein), designed BLIND to the 2026 set.

Protocol (fixed before any multi-chain 2026 value is computed; the 2026 sets are read only by --validate):
  development  post-cutoff multi-chain RNAs released before the benchmark download:
               the paper's set (data/compareDatabases/post_pre/a2021_b2021_rna_rna/a2021_rna_rna.db, files in
               data/RNA_RNA_ab/a2021_rna_rna/rna_rna) and the 50 added from the SHS collection
               (data/RNA_RNA_ab/a2021_multiRNA_add/rna_rna, released 2022-01 .. 2025-03; 7OUO and 9EN6 skipped for
               multi-letter chain IDs).
               with alignment:    the 46 added RNAs, local runs with the RNA alignment
               without alignment: the same 46 run without it (a2021_multiRNA_add/noMSA/rna_rna) plus the paper's 20,
                                  which were AlphaFold Server runs (no RNA alignment; 8 confirmed by their server
                                  downloads in folds/, byte-identical models; the other 12 from the same batch)
               (8AMK-8AMN, 7OUO and 9EN6 skipped: multi-letter chain IDs)
  outcome      well folded = LDDT_r > 0.85, backbone C3' (RNA_LDDT of the databases, the paper) or all-atom with
               DT_LDDT=allatom (decision_tree/lddt_labels.py).
  confidence   per branch the AF3 metric with the best development AUC among pTM, ipTM, mean pLDDT, global PAE and
               intra-RNA PAE of the longest RNA chain; levels 0 / 1 / 2 at its development tertiles.
  familiarity  +1 sequence-familiar (>= 80 % identity of the longest RNA chain to a pre-cutoff multi-chain RNA) and
               +1 fold-familiar (US-align TM >= 0.5 of the longest RNA chain to a pre-cutoff RNA chain, >= 20 nt) are
               added only if they raise the 5-fold cross-validated development AUC by >= 0.01 (folds grouped by
               sequence cluster, >= 80 % identity; levels re-derived inside each training fold).
  freeze       the design is written to decision_tree/calibration_rna_multi.json.
  validation   --validate applies the frozen design once to the 2026 multi-chain RNAs (local runs with and without
               the alignment, AlphaFold Server runs) and writes the report; it never changes the design.

The subtree used in the paper and by the tool is the duplex version (--v2, calibration_rna_multi_v2.json): the
duplex question and its cut-offs are set on the development data only, and the 2026 set tests it (--v2 --validate).
The form without the duplex question (no --v2) is kept for reproducibility.

Run from the repo root (RNABench env):
  python -m decision_tree.develop_rna_multi              # design only (no 2026 data read)
  python -m decision_tree.develop_rna_multi --validate   # once, after the design is frozen
Outputs: results/plots/all_chains/decision_tree/rna_multi/
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.chain_pae import intra_chain_pae
from decision_tree.develop import wilson
from decision_tree import develop_rna as single
from decision_tree.lddt_labels import complex_lddt

D = lambda *p: os.path.join(_REPO_ROOT, *p)
OUT = D("results", "plots", "all_chains", "decision_tree", "rna_multi")
CALIB = D("decision_tree", "calibration_rna_multi.json")
GOOD_LDDT = 0.85
METRICS = ("ptm", "iptm", "plddt", "gpae", "rna_pae")       # PAE negated: higher is better
PRE_DB = D("data", "compareDatabases", "post_pre", "a2021_b2021_rna_rna", "b2021_rna_rna.db")
RNG = np.random.default_rng(0)

# the published databases and structures (fetch_data.sh; structures with --structures)
ALLDB, ALLST = D("data", "all_data_zip", "all_databases"), D("data", "all_data")
_set = lambda group, s: (os.path.join(ALLDB, group, f"{s}.db"), os.path.join(ALLST, group, s))
DESIGN_SETS = {   # name: (database, folder with af/ and pdb/)
    "paper_with": _set("a2021_post_cutoff", "rna_multichain"),
    "add_with": _set("a2021_post_cutoff", "rna_multichain_added_with_msa"),
    "add_without": _set("a2021_post_cutoff", "rna_multichain_added_no_msa"),
}
TEST_SETS = {
    "test_local_with": _set("2026", "rna_multichain_local_with_msa"),
    "test_local_without": _set("2026", "rna_multichain_local_no_msa"),
    "test_server": _set("2026", "rna_multichain_server"),
}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def confidence(afdir, pid):
    s = os.path.join(afdir, f"fold_{pid}_s1_summary_confidences_0.json")
    f = os.path.join(afdir, f"fold_{pid}_s1_full_data_0.json")
    if not (os.path.exists(s) and os.path.exists(f)):
        return None
    S, F = json.load(open(s)), json.load(open(f))
    pae = np.asarray(F["pae"], float)
    ic = intra_chain_pae(f)
    return dict(ptm=S.get("ptm"), iptm=S.get("iptm"), plddt=float(np.mean(F["atom_plddts"])) / 100,
                gpae=-float(pae.mean()), rna_pae=-max(ic.values(), key=lambda t: t[1])[0] if ic else None)


DUPLEX_FRAC = 0.60     # version 2: a duplex pairs >= 60 % of its nucleotides with another chain


from decision_tree.rna_only import duplex_fraction     # shared with the decision-tree tool


def longest_seq(v):
    try:
        v = json.loads(v) if isinstance(v, str) and v.strip()[:1] in "[{" else v
    except json.JSONDecodeError:
        pass
    if isinstance(v, dict):
        v = list(v.values())
    v = all_seqs(v)
    return max(v, key=len) if v else ""


def load(name, sets):
    db, folder = sets[name]
    con = sqlite3.connect(db)
    rows = []
    cols = {c[1] for c in con.execute("PRAGMA table_info(pred_rna_rna)")}
    msa = "RNA_msa_size" if "RNA_msa_size" in cols else "NULL"
    # well folded: LDDT_r of the chosen label (decision_tree/lddt_labels.py; backbone C3' by default)
    labels = complex_lddt(db)
    for pid, seq, depth in con.execute(f"SELECT exp_db_id, RNASequence, {msa} FROM pred_rna_rna"):
        p = pid.lower()
        lddt = labels.get(pid.strip().upper())
        c = confidence(os.path.join(folder, "af"), p)
        rows.append(dict(set=name, pdb=pid.upper(), seq=longest_seq(seq), depth=single._depth(depth), lddt=lddt,
                         cif=os.path.join(folder, "pdb", f"{p}.cif"),
                         duplex_frac=duplex_fraction(os.path.join(folder, "af", f"fold_{p}_s1_model_0.cif")),
                         **(c or {k: None for k in METRICS})))
    return rows


def multi_letter_chains(r):
    """entries with chain IDs longer than one character in the structure or the AF3 model are skipped
    (scoring writes PDB-format files; user rule 2026-10-14)"""
    import gemmi
    model = os.path.join(os.path.dirname(os.path.dirname(r["cif"])), "af", f"fold_{r['pdb'].lower()}_s1_model_0.cif")
    for f in (r["cif"], model):
        if os.path.exists(f) and any(len(ch.name) > 1 for ch in gemmi.read_structure(f)[0]):
            return True
    return False


SKIPPED = set()


def usable(rows):
    out = []
    for r in rows:
        if multi_letter_chains(r):
            SKIPPED.add(r["pdb"])
            continue
        if r["lddt"] is not None and all(r[m] is not None for m in METRICS):
            out.append(r)
    return out


def check_paper_outcome(rows):
    """LDDT_r used here vs the stored label of the paper's 20"""
    stored = complex_lddt(DESIGN_SETS["paper_with"][0])
    d = [abs(r["lddt"] - stored[r["pdb"]]) for r in rows if r["lddt"] is not None and stored.get(r["pdb"]) is not None]
    return len(d), (max(d) if d else None)


def all_seqs(v):
    try:
        v = json.loads(v) if isinstance(v, str) and v.strip()[:1] in "[{" else v
    except json.JSONDecodeError:
        pass
    if isinstance(v, dict):
        v = list(v.values())
    v = [v] if isinstance(v, str) else list(v or [])
    return [s for x in v for s in str(x).split("&") if s]


def seq_identity(seqs):
    """best identity of the longest RNA chain to any RNA chain of a pre-cutoff multi-chain RNA complex"""
    from evaluation.postpre_matched_similarity import rna_candidates
    pre = sorted({s for (v,) in sqlite3.connect(PRE_DB).execute("SELECT RNASequence FROM pred_rna_rna") for s in all_seqs(v)})
    qs = sorted({q for q in seqs.values() if q})
    with contextlib.redirect_stdout(io.StringIO()):
        cands = rna_candidates(qs, pre)
    best = {q: max((single._identity(q, v) for v in cands.get(q, [])), default=0.0) for q in qs}
    return {pid: best[q] for pid, q in seqs.items() if q}


def add_familiarity(rows):
    sid = seq_identity({r["pdb"]: r["seq"] for r in rows})
    tm = single.structural_tm({r["pdb"]: r["cif"] for r in rows}, os.path.join(OUT, "tm_cache.json"))
    for r in rows:
        r["seq_id"] = sid.get(r["pdb"])
        t, n = tm.get(r["pdb"], [None, None])
        r["tm"], r["rna_len"] = t, n
        r["seq_familiar"] = r["seq_id"] is not None and r["seq_id"] >= single.SEQ_FAMILIAR
        r["tm_familiar"] = t is not None and n is not None and n >= single.MIN_NT_TM and t >= single.TM_FAMILIAR


# --------------------------------------------------------------------------- #
# design
# --------------------------------------------------------------------------- #
_auc = single._auc


def conf_level(v, cuts):
    return 0 if v <= cuts[0] else 2 if v > cuts[1] else 1


def score(r, d):
    s = conf_level(r[d["metric"]], d["conf_cuts"])
    return s + (int(r["seq_familiar"]) + int(r["tm_familiar"]) if d["familiarity"] else 0)


def level(r, d):
    s = score(r, d)
    if not d["familiarity"]:
        return ("WEAK", "MODERATE", "STRONG")[s]
    lo, hi = d["level_cuts"]
    return "WEAK" if s <= lo else "STRONG" if s > hi else "MODERATE"


def fit(rows, metric, familiarity):
    cuts = [float(c) for c in np.quantile([r[metric] for r in rows], [1 / 3, 2 / 3])]
    d = dict(metric=metric, conf_cuts=cuts, familiarity=familiarity)
    if familiarity:
        sc = [score(r, d) for r in rows]
        d["level_cuts"] = [float(c) for c in np.quantile(sc, [1 / 3, 2 / 3])]
    return d


def groups(rows):
    """sequence clusters (>= 80 % identity, single linkage) for grouped cross-validation"""
    gid, reps = {}, []
    for r in sorted(rows, key=lambda r: r["pdb"]):
        for k, q in enumerate(reps):
            if single._identity(r["seq"], q["seq"]) >= single.SEQ_FAMILIAR:
                gid[r["pdb"]] = k
                break
        else:
            gid[r["pdb"]] = len(reps)
            reps.append(r)
    return gid


def cv_auc(rows, metric, familiarity, gid, folds=5, repeats=20):
    y = np.array([r["lddt"] > GOOD_LDDT for r in rows])
    g = np.array([gid[r["pdb"]] for r in rows])
    out = []
    for _ in range(repeats):
        perm = RNG.permutation(sorted(set(g)))
        fold_of = {c: i % folds for i, c in enumerate(perm)}
        pred = np.zeros(len(rows))
        for f in range(folds):
            te = np.array([fold_of[x] == f for x in g])
            d = fit([r for r, t in zip(rows, te) if not t], metric, familiarity)
            for i in np.where(te)[0]:
                pred[i] = score(rows[i], d)
        out.append(_auc(y, pred))
    return float(np.nanmean(out))


def design(rows, name, lines):
    y = [r["lddt"] > GOOD_LDDT for r in rows]
    aucs = {m: _auc(y, [r[m] for r in rows]) for m in METRICS}
    metric = max(METRICS, key=lambda m: (round(aucs[m], 3), m == "rna_pae"))
    gid = groups(rows)
    cv0, cv1 = cv_auc(rows, metric, False, gid), cv_auc(rows, metric, True, gid)
    fam = cv1 >= cv0 + 0.01
    d = dict(fit(rows, metric, fam), metric_aucs=aucs, cv_auc_confidence=cv0, cv_auc_with_familiarity=cv1,
             n=len(rows), n_good=int(sum(y)), n_clusters=len(set(gid.values())))
    lines.append(f"\nBranch {name}: development n = {len(rows)} ({sum(y)} well folded, {d['n_clusters']} sequence clusters)")
    lines.append("  development AUC per confidence metric: " + ", ".join(f"{m} {a:.2f}" for m, a in aucs.items()))
    lines.append(f"  chosen {metric}; cut-offs {d['conf_cuts'][0]:.3f} / {d['conf_cuts'][1]:.3f}"
                 + ("  (PAE negated: higher is better)" if "pae" in metric else ""))
    lines.append(f"  grouped 5-fold CV AUC: confidence levels {cv0:.3f}, + familiarity points {cv1:.3f} -> "
                 f"{'familiarity ADDED' if fam else 'confidence only'}")
    report_levels(rows, d, lines, "  development")
    return d


def report_levels(rows, d, lines, label):
    y = [r["lddt"] > GOOD_LDDT for r in rows]
    lines.append(f"{label}: n = {len(rows)}, well folded {sum(y)}, AUC levels {_auc(y, [score(r, d) for r in rows]):.2f}, "
                 f"continuous {d['metric']} {_auc(y, [r[d['metric']] for r in rows]):.2f}")
    for lv in ("WEAK", "MODERATE", "STRONG"):
        sel = [yy for yy, r in zip(y, rows) if level(r, d) == lv]
        if sel:
            p, l, u = wilson(sum(sel), len(sel))
            lines.append(f"    {lv:9s} {sum(sel):3d}/{len(sel):3d} well folded = {100 * p:.0f}% [{100 * l:.0f}-{100 * u:.0f}]")


def build_design():
    os.makedirs(OUT, exist_ok=True)
    paper = load("paper_with", DESIGN_SETS)
    n_chk, max_diff = check_paper_outcome(paper)
    with_rows = usable(load("add_with", DESIGN_SETS))
    without_rows = usable(load("add_without", DESIGN_SETS)) + [dict(r, set="paper_server") for r in usable(paper)]
    for rs in (with_rows, without_rows):
        add_familiarity(rs)
    lines = ["Multi-chain RNA subtree, design (decision_tree/develop_rna_multi.py; no 2026 data read)", "=" * 70,
             f"outcome check on the paper's set: recomputed vs stored LDDT_r for {n_chk} RNAs, max |difference| = {max_diff}",
             f"skipped (multi-letter chain IDs): {', '.join(sorted(SKIPPED)) or 'none'}"]
    designs = {"with_alignment": design(with_rows, "with_alignment", lines),
               "without_alignment": design(without_rows, "without_alignment", lines)}
    json.dump(designs, open(CALIB, "w"), indent=1)
    lines.append(f"\nFROZEN -> {CALIB}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "design.txt"), "w").write(txt + "\n")
    _write_rows(os.path.join(OUT, "design_rows.tsv"), with_rows + without_rows)
    print(txt)


# --------------------------------------------------------------------------- #
# version 2: duplex question first
# --------------------------------------------------------------------------- #
CALIB_V2 = D("decision_tree", "calibration_rna_multi_v2.json")


def is_duplex(r):
    return r.get("duplex_frac") is not None and r["duplex_frac"] >= DUPLEX_FRAC


def level_v2(r, d):
    if is_duplex(r):
        return "STRONG"
    return ("WEAK", "MODERATE", "STRONG")[conf_level(r[d["metric"]], d["conf_cuts"])]


def score_v2(r, d):
    return 3 if is_duplex(r) else conf_level(r[d["metric"]], d["conf_cuts"])


def report_v2(rows, d, lines, label):
    y = [r["lddt"] > GOOD_LDDT for r in rows]
    lines.append(f"{label}: n = {len(rows)}, well folded {sum(y)}, duplexes {sum(map(is_duplex, rows))}, "
                 f"AUC {_auc(y, [score_v2(r, d) for r in rows]):.2f}")
    for lv in ("WEAK", "MODERATE", "STRONG"):
        sel = [yy for yy, r in zip(y, rows) if level_v2(r, d) == lv]
        if sel:
            p, l, u = wilson(sum(sel), len(sel))
            lines.append(f"    {lv:9s} {sum(sel):3d}/{len(sel):3d} well folded = {100 * p:.0f}% [{100 * l:.0f}-{100 * u:.0f}]")
    dup = [yy for yy, r in zip(y, rows) if is_duplex(r)]
    if dup:
        lines.append(f"    (of which duplex: {sum(dup)}/{len(dup)} well folded)")


def _rates(rows, level_fn):
    """development share of well-folded RNAs per level (Wilson 95% CI), for the decision-tree tool"""
    out = {}
    for lv in ("WEAK", "MODERATE", "STRONG"):
        y = [r["lddt"] > GOOD_LDDT for r in rows if level_fn(r) == lv]
        if y:
            p, l, u = wilson(sum(y), len(y))
            out[lv] = dict(rate=p, ci_lo=l, ci_hi=u, n=len(y))
    return out


def build_design_v2():
    """version 2: an AF3-model duplex goes to STRONG; the other complexes get intra-RNA PAE levels at the tertiles
    of the non-duplex development RNAs (same branches and measure as version 1)"""
    v1 = json.load(open(CALIB))
    paper = load("paper_with", DESIGN_SETS)
    branches = {"with_alignment": usable(load("add_with", DESIGN_SETS)),
                "without_alignment": usable(load("add_without", DESIGN_SETS)) + [dict(r, set="paper_server") for r in usable(paper)]}
    lines = ["Multi-chain RNA subtree, version 2 (duplex question first); development data only", "=" * 70,
             f"duplex = AF3 model pairs >= {DUPLEX_FRAC:.0%} of its nucleotides with another chain (N1-N3 < 3.3 A)"]
    designs = {}
    for br, rows in branches.items():
        metric = v1[br]["metric"]
        nd = [r for r in rows if not is_duplex(r)]
        cuts = [float(c) for c in np.quantile([r[metric] for r in nd], [1 / 3, 2 / 3])]
        designs[br] = dict(metric=metric, conf_cuts=cuts, duplex_frac=DUPLEX_FRAC, n=len(rows), n_duplex=len(rows) - len(nd))
        designs[br]["dev_rates"] = _rates(rows, lambda r, d=designs[br]: level_v2(r, d))
        lines.append(f"\nBranch {br}: n = {len(rows)}, duplexes {len(rows) - len(nd)}; non-duplex {metric} cut-offs "
                     f"{cuts[0]:.3f} / {cuts[1]:.3f}")
        report_v2(rows, designs[br], lines, "  development")
    json.dump(designs, open(CALIB_V2, "w"), indent=1)
    lines.append(f"\nFROZEN -> {CALIB_V2}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "design_v2.txt"), "w").write(txt + "\n")
    print(txt)


def validate_v2():
    """2026 multi-chain RNAs: test of the subtree (not used in its design)"""
    if not os.path.exists(CALIB_V2):
        sys.exit(f"No frozen version-2 design ({CALIB_V2}).")
    d = json.load(open(CALIB_V2))
    lines = ["Multi-chain RNA subtree (duplex question first), 2026 test (frozen design, applied once)", "=" * 70]
    for name, key, br in (("local, with alignment", "test_local_with", "with_alignment"),
                          ("local, without alignment", "test_local_without", "without_alignment"),
                          ("AlphaFold Server (no alignment)", "test_server", "without_alignment")):
        report_v2(usable(load(key, TEST_SETS)), d[br], lines, f"\n{name}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "validation_2026_v2.txt"), "w").write(txt + "\n")
    print(txt)


def validate():
    if not os.path.exists(CALIB):
        sys.exit(f"No frozen design ({CALIB}); run the design step first.")
    designs = json.load(open(CALIB))
    loc_with = usable(load("test_local_with", TEST_SETS))
    rows = {"local, with alignment": (loc_with, designs["with_alignment"]),
            "local, without alignment": (usable(load("test_local_without", TEST_SETS)), designs["without_alignment"]),
            "AlphaFold Server (no alignment)": (usable(load("test_server", TEST_SETS)), designs["without_alignment"])}
    lines = ["Multi-chain RNA subtree, 2026 validation (frozen design, applied once)", "=" * 70]
    allrows = []
    for name, (rs, d) in rows.items():
        add_familiarity(rs)
        report_levels(rs, d, lines, f"\n{name}")
        allrows += rs
    txt = "\n".join(lines)
    open(os.path.join(OUT, "validation_2026.txt"), "w").write(txt + "\n")
    _write_rows(os.path.join(OUT, "validation_rows.tsv"), allrows)
    print(txt)


def _write_rows(path, rows):
    keys = ["set", "pdb", "depth", "lddt", *METRICS, "seq_id", "tm", "rna_len", "seq_familiar", "tm_familiar"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="apply the frozen design once to the 2026 multi-chain RNAs")
    ap.add_argument("--v2", action="store_true", help="version 2 (duplex question first)")
    a = ap.parse_args()
    if a.v2:
        validate_v2() if a.validate else build_design_v2()
    else:
        validate() if a.validate else build_design()
