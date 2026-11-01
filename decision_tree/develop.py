"""
Development of the scored decision framework (v2): design choices made on PRE-cutoff targets only,
calibration and test on POST-cutoff targets (held out).

Each side gives a score = AF3 confidence level (0 bad, 1 mid, 2 good) + 1 if the RNA is familiar
(similar RNA in the pre-cutoff PDB) - 1 per selected downgrade, and three levels:
score <= 1 -> low, 2 -> moderate, 3 -> high.

  RNA familiarity  included by design (the paper's matched-pair result: accuracy follows RNA
                   novelty), not tuned. Familiar = a pre-cutoff RNA with >= 80% sequence identity
                   (identical columns / longer length) or US-align TM-score >= 0.5 (RNA >= 20 nt).
                   Pre-cutoff targets are in AF3's training data and count as familiar.
  FOLD confidence  chosen on pre-cutoff targets among: RNA pLDDT alone; RNA pLDDT + global PAE;
                   RNA pLDDT + mean intra-RNA PAE; intra-RNA PAE alone (PAE cut-offs also chosen on
                   pre-cutoff targets).
  INTERFACE        base = interface tier (ipTM_r / chain-pair PAE); downgrades chosen on pre-cutoff
                   targets among: large RBP domain (dsRBD / RNase III / Dicer), no protein-RNA
                   template, protein partner pTM < 0.84, RNA pLDDT < 0.78.

Outcomes: fold = RNA LDDT > 0.85 (single-chain RNA and protein-RNA complexes; longest RNA chain);
interface = correct interface (placement class) per protein-RNA RNA chain.
Writes results/plots/all_chains/decision_tree/{develop_summary.txt, calibration.json, develop_*.tsv}.
Run from the repo root (RNABench env):  python -m decision_tree.develop
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.chain_pae import load_cache
from decision_tree.validate import OUTDIR, PP, _auc, db_path
from decision_tree.validate_tree import RNA_ONLY, _first, _j

SIM = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "structural_similarity")
GOOD_LDDT = 0.85
SEQ_FAMILIAR, TM_FAMILIAR = 0.8, 0.5


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def familiarity(system_file):
    out = {}
    p = os.path.join(SIM, f"best_structural_twin_{system_file}.tsv")
    for r in csv.DictReader(open(p), delimiter="\t"):
        try:
            sid = float(r["seq_twin_identity"])
        except ValueError:
            sid = np.nan
        try:
            tm = float(r["best_tm_any_pre"])
        except ValueError:
            tm = np.nan
        long_enough = r["rna_len"] not in ("", "None") and int(r["rna_len"]) >= 20
        fam = (np.isfinite(sid) and sid >= SEQ_FAMILIAR) or (long_enough and np.isfinite(tm) and tm >= TM_FAMILIAR)
        out[r["post_id"].upper()] = bool(fam)
    return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def fold_rows(cache):
    fam = {"single-chain RNA": familiarity("rna_only"), "protein-RNA": familiarity("protein_rna")}
    sets = [("single-chain RNA", RNA_ONLY, "pre", "b2021_rna.db", "pred_rna_rna"),
            ("single-chain RNA", RNA_ONLY, "post", "a2021_rna.db", "pred_rna_rna"),
            ("protein-RNA", PP, "pre", "b2021.db", "pred_protein_rna"),
            ("protein-RNA", PP, "post", "a2021.db", "pred_protein_rna")]
    rows = []
    for system, folder, period, db, pt in sets:
        con = sqlite3.connect(db_path(folder, db))
        for eid, fn, chains, pl, gpae, lddt in con.execute(
                f"SELECT exp_db_id, FileName, RNAChainIDs, af3_rna_pLDDT_avg, af3_global_pae_avg, RNA_LDDT FROM {pt}"):
            lddt = _first(lddt)
            if lddt is None:
                continue
            stem = os.path.basename(str(fn)).lower().replace("_model_0.cif", "")
            ch = _j(chains, [])
            ch = ch if isinstance(ch, list) else [ch]
            cp = cache.get(stem, {})
            rp = [cp[c] for c in ch if c in cp]
            rna_pae = max(rp, key=lambda t: t[1])[0] if rp else None          # longest RNA chain
            familiar = True if period == "pre" else fam[system].get(eid.upper())
            rows.append(dict(system=system, period=period, pdb=eid, plddt=_first(pl), gpae=_first(gpae),
                             rna_pae=rna_pae, familiar=familiar, outcome=lddt > GOOD_LDDT))
    return rows


def interface_rows():
    fam = familiarity("protein_rna")
    tiers = {(r["period"], r["pdb"], r["chain"]): r for r in
             csv.DictReader(open(os.path.join(OUTDIR, "validation_placement.tsv")), delimiter="\t")}
    tb = lambda v: None if v in ("", "None") else v == "True"
    rows = []
    for r in csv.DictReader(open(os.path.join(OUTDIR, "validation_tree_interface.tsv")), delimiter="\t"):
        t = tiers.get((r["period"], r["pdb"], r["chain"]))
        if not t or not t["tier"]:
            continue
        familiar = True if r["period"] == "pre" else fam.get(r["pdb"].upper())
        rows.append(dict(period=r["period"], pdb=r["pdb"], chain=r["chain"],
                         tier={"bad": 0, "mid": 1, "good": 2}[t["tier"]],
                         large=tb(r["domain"]) is False, no_template=tb(r["template"]) is False,
                         **{k: r[k] for k in ("wellfolded",)}, familiar=familiar,
                         outcome=r["outcome"] == "True"))
    return rows


# --------------------------------------------------------------------------- #
# fold confidence candidates
# --------------------------------------------------------------------------- #
def conf_plddt(r, **_):
    p = r["plddt"]
    return None if p is None else (2 if p >= 0.775 else 0 if p < 0.55 else 1)


def conf_plddt_gpae(r, cut=10.0, **_):
    p, g = r["plddt"], r["gpae"]
    if p is None or g is None:
        return None
    return 2 if (p >= 0.78 and g <= cut) else 0 if p < 0.55 else 1


def conf_plddt_rnapae(r, cut=8.0, **_):
    p, e = r["plddt"], r["rna_pae"]
    if p is None or e is None:
        return None
    return 2 if (p >= 0.78 and e <= cut) else 0 if p < 0.55 else 1


def conf_rnapae(r, lo=6.0, hi=12.0, **_):
    e = r["rna_pae"]
    return None if e is None else (2 if e <= lo else 0 if e > hi else 1)


def auc_rows(rows, score):
    s = [(score(r), r["outcome"]) for r in rows]
    s = [(x, c) for x, c in s if x is not None]
    return _auc([x for x, c in s if c], [x for x, c in s if not c]), len(s)


def level(score):
    return "low" if score <= 1 else ("moderate" if score == 2 else "high")


def calib_table(L, rows, score, names):
    cal = {}
    for period in ("pre", "post"):
        d = [r for r in rows if r["period"] == period and score(r) is not None]
        L.append(f"   {period}: AUC(score) {auc_rows(d, score)[0]:.2f}; " + ", ".join(
            f"{names[lv]} {100 * wilson(sum(r['outcome'] for r in d if level(score(r)) == lv), sum(level(score(r)) == lv for r in d))[0]:.0f}% "
            f"[{100 * wilson(sum(r['outcome'] for r in d if level(score(r)) == lv), sum(level(score(r)) == lv for r in d))[1]:.0f}-"
            f"{100 * wilson(sum(r['outcome'] for r in d if level(score(r)) == lv), sum(level(score(r)) == lv for r in d))[2]:.0f}%] "
            f"(n={sum(level(score(r)) == lv for r in d)})" for lv in ("low", "moderate", "high")))
        cal[period] = {names[lv]: dict(zip(("rate", "ci_lo", "ci_hi"), wilson(
            sum(r["outcome"] for r in d if level(score(r)) == lv), sum(level(score(r)) == lv for r in d))),
            n=sum(level(score(r)) == lv for r in d)) for lv in ("low", "moderate", "high")}
    return cal


def main():
    L, design = [], {}
    cache = load_cache()
    # ---------------- FOLD ----------------
    fr = fold_rows(cache)
    pre = [r for r in fr if r["period"] == "pre"]
    L.append(f"FOLD: {len(fr)} targets ({sum(r['rna_pae'] is not None for r in fr)} with intra-RNA PAE)")
    cands = [("pLDDT_r", conf_plddt, {})]
    cands += [(f"pLDDT_r + global PAE <= {c:g}", conf_plddt_gpae, dict(cut=c)) for c in (8, 10, 12, 15)]
    cands += [(f"pLDDT_r + RNA PAE <= {c:g}", conf_plddt_rnapae, dict(cut=c)) for c in (4, 6, 8, 10)]
    cands += [(f"RNA PAE <= {lo:g} good, > {hi:g} bad", conf_rnapae, dict(lo=lo, hi=hi))
              for lo, hi in ((4, 8), (4, 10), (6, 10), (6, 12), (8, 12), (8, 15))]
    L.append(" confidence candidates (selection on pre-cutoff AUC; post shown for transparency):")
    res = []
    for name, f, kw in cands:
        g = lambda r, f=f, kw=kw: f(r, **kw)
        a_pre, n_pre = auc_rows(pre, g)
        a_post, n_post = auc_rows([r for r in fr if r["period"] == "post"], g)
        res.append((a_pre, -len(name), name, f, kw, a_post, n_pre))
        L.append(f"   {name:36s} pre AUC {a_pre:.3f} (n={n_pre}), post AUC {a_post:.3f} (n={n_post})")
    best = max(res, key=lambda t: (round(t[0], 3), t[1]))
    fold_conf = lambda r, f=best[3], kw=best[4]: f(r, **kw)
    design["fold_confidence"] = dict(name=best[2], function=best[3].__name__, params=best[4])
    L.append(f" selected: {best[2]}")
    fold_score = lambda r: None if fold_conf(r) is None or r["familiar"] is None else fold_conf(r) + int(r["familiar"])
    names_f = {"low": "WEAK", "moderate": "MODERATE", "high": "STRONG"}
    L.append(" fold levels (score = confidence + familiar RNA; <=1 WEAK, 2 MODERATE, 3 STRONG), % well folded [95% CI]:")
    cal_f = {}
    for system in ("single-chain RNA", "protein-RNA"):
        L.append(f"  {system}:")
        cal_f[system] = calib_table(L, [r for r in fr if r["system"] == system], fold_score, names_f)
    L.append("  both systems:")
    cal_f["all"] = calib_table(L, fr, fold_score, names_f)
    post = [r for r in fr if r["period"] == "post"]
    L.append(f"  post-cutoff AUC: confidence only {auc_rows(post, fold_conf)[0]:.2f}, + familiar RNA {auc_rows(post, fold_score)[0]:.2f}")

    # ---------------- INTERFACE ----------------
    ir = interface_rows()
    for r in ir:
        r["low_ptm_or_plddt"] = r["wellfolded"] == "False"
    mods = ("large", "no_template", "low_ptm_or_plddt")
    pre_i = [r for r in ir if r["period"] == "pre"]
    L.append(f"\nINTERFACE: {len(ir)} RNA chains; downgrade subsets (selection on pre-cutoff AUC):")
    res = []
    for k in range(len(mods) + 1):
        for sub in itertools.combinations(mods, k):
            g = lambda r, sub=sub: r["tier"] - sum(r[m] for m in sub)
            a_pre = auc_rows(pre_i, g)[0]
            a_post = auc_rows([r for r in ir if r["period"] == "post"], g)[0]
            res.append((round(a_pre, 3), -k, sub, a_post))
            L.append(f"   tier - {' - '.join(sub) if sub else '(none)':45s} pre AUC {a_pre:.3f}, post AUC {a_post:.3f}")
    best_i = max(res, key=lambda t: (t[0], t[1]))
    sel = best_i[2]
    design["interface_downgrades"] = list(sel)
    L.append(f" selected downgrades: {list(sel) or 'none'}")
    if_score = lambda r: None if r["familiar"] is None else r["tier"] - sum(r[m] for m in sel) + int(r["familiar"])
    names_i = {"low": "UNRELIABLE", "moderate": "CAUTION", "high": "RELIABLE"}
    L.append(" interface levels (score = tier - downgrades + familiar RNA; <=1 UNRELIABLE, 2 CAUTION, 3 RELIABLE), % correct [95% CI]:")
    cal_i = calib_table(L, ir, if_score, names_i)
    post_i = [r for r in ir if r["period"] == "post"]
    L.append(f"  post-cutoff AUC: tier only {auc_rows(post_i, lambda r: r['tier'])[0]:.2f}, "
             f"tier - downgrades {auc_rows(post_i, lambda r: r['tier'] - sum(r[m] for m in sel))[0]:.2f}, "
             f"+ familiar RNA {auc_rows(post_i, if_score)[0]:.2f}")

    json.dump(dict(design=design, fold=cal_f, interface=cal_i, familiar=dict(seq=SEQ_FAMILIAR, tm=TM_FAMILIAR)),
              open(os.path.join(OUTDIR, "calibration.json"), "w"), indent=1, default=float)
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "develop_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
