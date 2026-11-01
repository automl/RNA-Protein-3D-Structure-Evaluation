"""
Combined verdict for a protein-RNA prediction: is this RNA chain usable, i.e. is it well folded AND placed
correctly on its protein?

  outcome   per predicted RNA chain: correct interface (post-cutoff: exact-mapped placement class
            correct_interface; 2026: protein-RNA DockQ >= 0.23, as on the interface side) AND the chain is
            well folded (LDDT_r > 0.85 of the chain; decision_tree/lddt_labels.py): backbone C3' LDDT_r of the
            predicted chain against the experimental RNA chain of the same sequence (the paper; chain_lddt below,
            computed from the structures in WORKERS parallel processes) or, with DT_LDDT=allatom, the all-atom
            RNA_LDDT_AllChains of the database.
  level     fixed rule, nothing fitted: WEAK if the interface level or the fold level is WEAK; STRONG if both
            are STRONG; MODERATE otherwise (interface STRONG, fold MODERATE). Every RNA chain of a complex
            carries the complex's fold level (v3 fold score, longest RNA chain) and its own interface level
            (v2 interface score: WEAK <= 1, STRONG >= 2).
            Tested on post-cutoff targets and not adopted: a decision tree on the two scores (the fold leads,
            the interface vetoes only at a score <= 0; more STRONG chains, but a well-folded short RNA can be
            misplaced) and a hybrid that keeps this rule for RNAs <= 10 nt (CV AUC +0.003 over the tree).
  ranking   combined score = fold score + interface score (fixed, not fitted); the three levels discard the
            order within WEAK, so the levels are the decision and the sum is the ranking (ROC). It is compared
            with AF3's confidence alone (RNA confidence level + interface tier); AF3's ranking_score is not used,
            as it is meant to rank the models of one target (its disorder and clash terms are not quality terms).
  calibrate post-cutoff protein-RNA targets -> decision_tree/calibration_combined.json; the 2026 set is
            scored with the same rule (figure_2026.py).

Run from the repo root (RNABench env):  python -m decision_tree.combined
Outputs: results/plots/all_chains/decision_tree/combined/{post_rows.tsv, 2026_rows.tsv, combined.txt}
"""
from __future__ import annotations

import contextlib
import csv
import glob
import io
import json
import os
import sqlite3
import sys
import tempfile
from difflib import SequenceMatcher

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.develop import wilson
from decision_tree.lddt_labels import LABEL, NAME as LABEL_NAME, chain_lddt as all_atom_chain_lddt

OUT = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "combined")
CAL = os.path.join(_REPO_ROOT, "decision_tree", "calibration_combined.json")
LEVELS = ("WEAK", "MODERATE", "STRONG")
GOOD_LDDT = 0.85


def combined_level(fold_level, interface_level):
    """fold_level WEAK/MODERATE/STRONG, interface_level WEAK/STRONG (ranges 'A..B' give a range)."""
    if not fold_level or not interface_level:
        return None
    f = fold_level.split("..")
    i = interface_level.split("..")
    vals = sorted({_combine(a, b) for a in (f[0], f[-1]) for b in (i[0], i[-1])})
    return LEVELS[vals[0]] if len(vals) == 1 else f"{LEVELS[vals[0]]}..{LEVELS[vals[-1]]}"


def _combine(fold, iface):
    if fold == "WEAK" or iface == "WEAK":
        return 0
    return 2 if fold == "STRONG" and iface == "STRONG" else 1


def calibrated_rate(level, period="post"):
    if not level or ".." in level or not os.path.exists(CAL):
        return None
    return json.load(open(CAL)).get(period, {}).get(level)


# --------------------------------------------------------------------------- #
# per-chain backbone C3' LDDT_r (the paper's label)
# --------------------------------------------------------------------------- #
def _rna_chains(path):
    import gemmi
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = []
    for ch in st[0]:
        pol = ch.get_polymer()
        if len(pol) and pol.check_polymer_type() == gemmi.PolymerType.Rna:
            out.append((ch.name, pol))
    return out


def chain_lddt(exp_cif, model_cif, cache):
    """{predicted RNA chain: C3' LDDT_r against the experimental RNA chain of the same sequence (the best of
    identical copies; the most similar sequence if none is identical)}."""
    key = f"{exp_cif}|{model_cif}"
    if key in cache:
        return cache[key]
    from decision_tree.validate_2026 import _write_chain
    from evaluation.extractEvalMetrics import calculate_complex_lddt
    refs, preds = _rna_chains(exp_cif), _rna_chains(model_cif)
    res = {}
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        for i, r in enumerate(refs):
            _write_chain(r, os.path.join(d, f"r{i}.pdb"))
        seqs = [r[1].make_one_letter_sequence() for r in refs]
        for p in preds:
            _write_chain(p, os.path.join(d, "m.pdb"))
            # only reference chains with the predicted chain's sequence (best-matching if none is identical):
            # an unrelated A-form strand compared position by position can look well folded
            q = p[1].make_one_letter_sequence()
            sim = [SequenceMatcher(None, q, x, autojunk=False).ratio() for x in seqs]
            match = [i for i, x in enumerate(seqs) if x == q] or \
                    ([i for i, v in enumerate(sim) if v == max(sim)] if sim else [])
            vals = [calculate_complex_lddt(os.path.join(d, f"r{i}.pdb"), os.path.join(d, "m.pdb")) for i in match]
            vals = [v for v in vals if v is not None]
            res[p[0]] = float(max(vals)) if vals else None
    cache[key] = res
    return res


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
def _fold_levels(name):
    from decision_tree.develop_v3 import OUT as V3OUT, _read_rows, level, score
    cal3 = json.load(open(os.path.join(_REPO_ROOT, "decision_tree", "calibration_v3.json")))
    qs, cuts = [tuple(q) for q in cal3["fold"]["questions"]], cal3["fold"]["cuts"]
    out = {}
    for r in _read_rows(os.path.join(V3OUT, name)):
        if r["base_score"] is not None:
            s = score(r, qs)
            out[r["pdb"]] = (LEVELS[level(s, cuts)], s, r["base_score"])     # base = AF3 RNA confidence level
    return out


def _exp_index():
    from evaluation.bundle_paths import PROTEIN_RNA_DIR
    idx = {}
    for sub in ("pdb", "check_pdb"):
        for p in sorted(glob.glob(os.path.join(PROTEIN_RNA_DIR["post"], sub, "*.cif"))):
            idx.setdefault(os.path.basename(p)[:-4].upper(), p)
    return idx


def post_rows(cache):
    from decision_tree.develop import interface_rows
    from decision_tree.validate import PP, _model_index
    models = _model_index()
    fn = {str(e).strip().upper(): models.get(os.path.basename(str(f)).lower()) for e, f in
          sqlite3.connect(PP["a2021.db"]).execute("SELECT exp_db_id, FileName FROM pred_protein_rna")}
    exp = _exp_index()
    fold = _fold_levels("post_fold_rows.tsv")
    lddt_post = all_atom_chain_lddt(PP["a2021.db"]) if LABEL == "allatom" else None
    rows = []
    for r in interface_rows():
        pid = r["pdb"].upper()
        if r["period"] != "post" or r["familiar"] is None or pid not in fold:
            continue
        warn = r["wellfolded"] == "False"
        v2 = r["tier"] + int(r["familiar"]) - int(warn)
        if pid not in exp or not fn.get(pid):
            continue
        lddt = (lddt_post.get(pid, {}) if lddt_post is not None else chain_lddt(exp[pid], fn[pid], cache)).get(r["chain"])
        if lddt is None:
            continue
        rows.append(_row(pid, r["chain"], fold[pid], v2, r["outcome"], lddt, r["tier"]))
    return rows


def rows_2026(cache):
    from decision_tree import validate_2026 as V
    from decision_tree.validate_tree import _j
    fold = _fold_levels("2026_fold_rows.tsv")
    lddt_2026 = all_atom_chain_lddt(V.DB26) if LABEL == "allatom" else None
    dq = {}
    for eid, raw, pch in sqlite3.connect(V.DB26).execute("SELECT exp_db_id, RBP_DockQ, ProteinChainIDs FROM pred_protein_rna"):
        pid = str(eid).strip().upper()
        pc = _j(pch, [])
        pc = [str(x) for x in (pc if isinstance(pc, list) else [pc])]
        raw = _j(raw) if isinstance(_j(raw), list) else V.local_dockq(pid)
        dq[pid] = ({k: v for x in (raw or []) if isinstance(x, dict) for k, v in x.items() if v is not None}, pc)
    rows = []
    for r in csv.DictReader(open(os.path.join(V.OUTDIR, "interface.tsv")), delimiter="\t"):
        pid = r["pdb"]
        if r["v2"] in ("", "None") or pid not in fold or pid not in V.EXP_CIFS or pid not in V.MODELS:
            continue
        lddt = (lddt_2026.get(pid, {}) if lddt_2026 is not None else
                chain_lddt(V.EXP_CIFS[pid], V.MODELS[pid], cache)).get(r["chain"])
        if lddt is None:
            continue
        d, pc = dq.get(pid, ({}, []))
        vals = [v for k, v in d.items() if r["chain"] in k.split(":") and any(p in k.split(":") for p in pc)]
        row = _row(pid, r["chain"], fold[pid], float(r["v2"]), r["outcome"] == "True", lddt, float(r["tier"]))
        row["dockq"] = max(vals) if vals else None
        rows.append(row)
    return rows


def _row(pid, chain, fold, v2, iface_ok, lddt, tier):
    flv, fscore, _ = fold
    ilv = "WEAK" if v2 <= 1 else "STRONG"
    cl = combined_level(flv, ilv)
    return dict(pdb=pid, chain=chain, fold_level=flv, fold_score=fscore, interface_score=v2, interface_level=ilv,
                combined_score=fscore + v2, level=cl, level_rank=LEVELS.index(cl), interface_ok=iface_ok, chain_lddt=lddt,
                outcome=bool(iface_ok and lddt > GOOD_LDDT),
                af3_confidence=fold[2] + tier)     # AF3 RNA confidence level + interface tier, no other questions


def rates(rows):
    out = {}
    for lv in LEVELS:
        s = [r["outcome"] for r in rows if r["level"] == lv]
        p, lo, hi = wilson(sum(s), len(s))
        out[lv] = dict(rate=p, ci_lo=lo, ci_hi=hi, n=len(s), k=int(sum(s)))
    return out


def _chain_lddt_job(pair):
    return f"{pair[0]}|{pair[1]}", chain_lddt(pair[0], pair[1], {})


def _fill_cache(cache):
    """backbone per-chain LDDT_r of every post-cutoff and 2026 model not yet in the cache, in parallel"""
    from multiprocessing import Pool
    from decision_tree import validate_2026 as V
    from decision_tree.validate import PP, _model_index
    models = _model_index()
    fn = {str(e).strip().upper(): models.get(os.path.basename(str(f)).lower()) for e, f in
          sqlite3.connect(PP["a2021.db"]).execute("SELECT exp_db_id, FileName FROM pred_protein_rna")}
    exp = _exp_index()
    pairs = [(exp[p], fn[p]) for p in fn if p in exp and fn.get(p)]
    pairs += [(V.EXP_CIFS[p], V.MODELS[p]) for p in V.EXP_CIFS if p in V.MODELS]
    todo = [q for q in pairs if f"{q[0]}|{q[1]}" not in cache]
    if todo:
        with Pool(int(os.environ.get("WORKERS", "4"))) as pool:
            cache.update(pool.imap_unordered(_chain_lddt_job, todo, chunksize=2))


def _auc(rows, key):
    from decision_tree.validate import _auc as A
    d = [r for r in rows if r[key] is not None]
    return A([r[key] for r in d if r["outcome"]], [r[key] for r in d if not r["outcome"]])


def main():
    os.makedirs(OUT, exist_ok=True)
    cpath = os.path.join(OUT, "chain_lddt_cache.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    if LABEL == "backbone":
        _fill_cache(cache)
        json.dump(cache, open(cpath, "w"))
    post = post_rows(cache)
    t26 = rows_2026(cache)
    json.dump(cache, open(cpath, "w"))
    L = []
    for name, rows in (("post-cutoff (calibration)", post), ("2026", t26)):
        L.append(f"{name}: {len(rows)} RNA chains in {len({r['pdb'] for r in rows})} complexes, "
                 f"{sum(r['outcome'] for r in rows)} usable (correct interface and LDDT_r > {GOOD_LDDT})")
        for k in ("combined_score", "level_rank", "interface_score", "fold_score", "af3_confidence"):
            L.append(f"  AUC {k:16s} {_auc(rows, k):.3f}")
        for lv, v in rates(rows).items():
            L.append(f"  {lv:9s} {100 * v['rate']:5.1f}% [{100 * v['ci_lo']:.0f}-{100 * v['ci_hi']:.0f}] n={v['n']}")
        L.append("")
    json.dump(dict(rule="WEAK if fold or interface WEAK; STRONG if both STRONG; else MODERATE",
                   outcome=f"correct interface and {LABEL_NAME} > {GOOD_LDDT} (per RNA chain)",
                   post=rates(post), test_2026=rates(t26)), open(CAL, "w"), indent=1, default=float)
    for name, rows in (("post_rows.tsv", post), ("2026_rows.tsv", t26)):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    open(os.path.join(OUT, "combined.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
