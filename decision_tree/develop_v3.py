"""
Scored decision framework v3 for protein-RNA predictions, without MSA information.

Why v3: v2 used RNA familiarity on the fold side; on the independent 2026 set (data/2026/2026_a2021.db)
it ranked well-folded RNAs below AF3's own confidence (AUC 0.69 vs 0.74), and AlphaFold Server output
carries no RNA alignment (and protein alignments not comparable to the paper's local runs), so MSA
depth cannot be used.

Design (fixed before v3 is applied to 2026):
  development  POST-cutoff protein-RNA targets (data/compareDatabases/post_pre/.../a2021.db): the regime
               of new predictions (pre-cutoff targets are memorised by AF3 and behave differently, e.g.
               an Rfam family helps before the cutoff and is a warning after it).
  form         points score as in v2: AF3 confidence level (0/1/2; fold: mean intra-RNA PAE <= 4 / > 8 A,
               interface: ipTM_r / chain-pair PAE tier) + 1 or - 1 per selected question.
  candidates   known before prediction or read from the AF3 model; no MSA, no experimental reference:
               sequence- and structure-familiar RNA (separately), Rfam family, pseudoknot / stems /
               hairpins / internal loops / multiloops in the AF3 model, RNA length, numbers of RNA and
               protein chains, GC content, MFE per nucleotide, sequence complexity, AF3 templates (RBP =
               protein-RNA complex template: used / number; protein-only template: used / number), large
               RBP domain, AF3 RNA pLDDT / pTM_r / global PAE / fraction disordered, predicted interface
               area; interface side also the partner pTM, chain ipTM_r and chain-pair PAE.
  questions    boolean: "is true"; continuous: ">= the 25th / 50th / 75th percentile" of the training
               data; each with sign + or -. Missing value: 0 points.
  selection    greedy forward; candidate scored by out-of-fold AUC in 5 x repeated 5-fold cross-
               validation grouped by PDB entry, the question's cut-off and sign chosen on the training
               folds only. A question is added only if it raises the CV AUC by >= 0.01; at most 4.
  levels       three levels at the post-cutoff tertiles of the score.
  test         the frozen v3 is applied once to the 2026 set (decision_tree.validate_2026 labels) and
               compared with AF3 confidence alone and with v2 (paired bootstrap over complexes).


Run from the repo root (RNABench env):  python -m decision_tree.develop_v3   [--calibrate: only rewrite
decision_tree/calibration_v3.json from the saved rows]
Outputs: results/plots/all_chains/decision_tree/v3/{develop_v3.txt, v3_design.json, *_rows.tsv}
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.af3_output import read_prediction
from decision_tree.develop import wilson
from decision_tree.framework import fold_confidence
from decision_tree.rules import interface_tier
from decision_tree.tree import domain_group
from decision_tree.validate import PP, _auc, _model_index
from decision_tree.validate_tree import _family, _first, _j, _pk, _tpl
from evaluation.placement_labels import best_per_rna_chain

OUT = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "v3")
GOOD_LDDT, ACCEPTABLE = 0.85, 0.23
SEQ_FAM, TM_FAM = 0.8, 0.5
MAX_Q, MIN_GAIN, REPEATS, FOLDS = 4, 0.01, 5, 5
TIER = {"bad": 0, "mid": 1, "good": 2}
RNG = np.random.default_rng(0)

FOLD_FEATURES = ["seq_familiar", "tm_familiar", "family", "pseudoknot", "n_stems", "n_hairpins",
                 "n_internal", "n_multiloops", "rna_len", "n_rnas", "n_proteins", "gc", "mfe_per_nt",
                 "complexity", "rbp_template", "n_rbp_templates", "protein_template", "n_protein_templates",
                 "large_domain", "rna_plddt", "rna_ptm", "global_pae",
                 "frac_disordered", "iface_area"]
IFACE_FEATURES = FOLD_FEATURES + ["partner_ptm", "chain_rna_plddt", "chain_iptm", "pae_min"]


def _sim_file(name):
    for d in ("plots", "plots_odl"):
        p = os.path.join(_REPO_ROOT, "results", d, "all_chains", "structural_similarity", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def _n(v):
    v = _j(v)
    return len(v) if isinstance(v, list) else None


def _maxnum(v):
    v = _j(v)
    vals = v if isinstance(v, list) else [v]
    vals = [float(x) for x in vals if x not in (None, "")]
    return max(vals) if vals else None


def _num(v):
    try:
        return float(_first(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# rows (same builder for post-cutoff and 2026)
# --------------------------------------------------------------------------- #
from decision_tree.lddt_labels import COLUMN as LDDT_COLUMN   # well-folded label (DT_LDDT)
QUERY = ("SELECT p.exp_db_id, p.FileName, p.ProteinChainIDs, p.af3_rna_pLDDT_avg, p.af3_rna_pTM, "
         "p.af3_global_pae_avg, p.af3_fraction_disordered, p.RNA_Pseudoknots, p.RNA_Stems, p.RNA_HairpinLoops, "
         "p.RNA_InternalLoops, p.RNA_MultibranchLoops, p.RNALength, p.NumberRNAs, p.NumberProteins, "
         "p.RNA_GC_Content, p.RNA_MFE_Value, p.RNA_SequenceComplexity, p.RBP_PDB_Templates, "
         f"p.ProteinRNAInterfaceArea, p.Protein_PDB_Templates, p.{LDDT_COLUMN}, p.RNA_interface_placement, p.RBP_DockQ, "
         "e.RNAFamily, e.domain_names FROM pred_protein_rna p JOIN exp_protein_rna e "
         "ON upper(trim(p.exp_db_id)) = upper(trim(e.PDBId))")


def build_rows(db, models, familiarity, fold_label=None, iface_labels=None):
    """familiarity: {PDB: (seq identity, TM to pre, rna length)}; fold_label(pid, db_lddt) and
    iface_labels(pid, placement, dockq, protein chains, rna chains) give the outcomes."""
    con = sqlite3.connect(db)
    cols = {c[1] for c in con.execute("PRAGMA table_info(pred_protein_rna)")}
    q = QUERY if "RNA_interface_placement" in cols else QUERY.replace("p.RNA_interface_placement", "NULL")
    fold, iface = [], []
    for rec in con.execute(q):
        (eid, fn, pch, pl, ptm, gpae, dis, pk, st, hp, il, ml, ln, nr, npr, gc, mfe, cx, tpl, area, ptpl,
         lddt, plc, dq, fam, doms) = rec
        pid = str(eid).strip().upper()
        cif = models.get(os.path.basename(str(fn)).lower())
        if cif is None:
            continue
        try:
            pred = read_prediction(cif)
        except Exception:
            continue
        if not pred["rna_chains"]:
            continue
        s, t, rl = familiarity.get(pid, (None, None, None))
        rlen = _maxnum(ln)
        domains = [d for d in str(doms or "").split(";") if d.strip() not in ("", "-")] or None
        grp = domain_group(domains)
        base = dict(pdb=pid,
                    seq_familiar=None if s is None else s >= SEQ_FAM,
                    tm_familiar=None if t is None else (t >= TM_FAM and (rl or 0) >= 20),
                    family=_family(fam), pseudoknot=_pk(pk), n_stems=_n(st), n_hairpins=_n(hp),
                    n_internal=_n(il), n_multiloops=_n(ml), rna_len=rlen, n_rnas=_num(nr), n_proteins=_num(npr),
                    gc=_num(gc), mfe_per_nt=(None if _num(mfe) is None or not rlen else _num(mfe) / rlen),
                    complexity=_num(cx), rbp_template=_tpl(tpl), n_rbp_templates=_n(tpl),
                    protein_template=_tpl(ptpl), n_protein_templates=_n(ptpl), large_domain=None if grp is None else grp == "large",
                    rna_plddt=_num(pl), rna_ptm=_maxnum(ptm), global_pae=_num(gpae), frac_disordered=_num(dis),
                    iface_area=_num(area))
        longest = max(pred["rna_chains"], key=lambda c: c["length"])
        y = fold_label(pid, lddt)
        if y is not None:
            fold.append(dict(base, outcome=y > GOOD_LDDT, base_score=fold_confidence(rna_pae=longest["rna_pae"])))
        pc = _j(pch, [])
        pc = [str(x) for x in (pc if isinstance(pc, list) else [pc])]
        labels = iface_labels(pid, plc, dq, pc, [c["chain"] for c in pred["rna_chains"]])
        if not labels:
            continue
        for c in pred["rna_chains"]:
            tier = interface_tier(c["rna_iptm"], c["pae_min"])
            if tier is None or c["chain"] not in labels:
                continue
            iface.append(dict(base, chain=c["chain"], outcome=labels[c["chain"]], base_score=TIER[tier],
                              partner_ptm=c["partner_ptm"], chain_rna_plddt=c["rna_plddt"],
                              chain_iptm=c["rna_iptm"], pae_min=c["pae_min"]))
    return fold, iface


def post_inputs():
    fam = {}
    for r in csv.DictReader(open(_sim_file("best_structural_twin_protein_rna.tsv")), delimiter="\t"):
        f = lambda k: None if r[k] in ("", "None", "nan") else float(r[k])
        fam[r["post_id"].upper()] = (f("seq_twin_identity"), f("best_tm_any_pre"), f("rna_len"))
    fold_label = lambda pid, lddt: _first(lddt)

    def iface_labels(pid, plc, dq, pc, rc):
        plc = best_per_rna_chain(plc)              # one class per RNA chain: best over its interfaces
        return {k: v == "correct_interface" for k, v in plc.items()} if plc else None
    return PP["a2021.db"], fam, fold_label, iface_labels


def set2026_inputs():
    from decision_tree import validate_2026 as V
    cache = json.load(open(os.path.join(V.OUTDIR, "familiarity_cache.json")))
    fam = {pid: (cache["seq"].get(pid), *(cache["tm"].get(pid) or (None, None))) for pid in cache["ids"]}
    return V.DB26, fam, (lambda pid, lddt: V.fold_label(pid)), \
        (lambda pid, plc, dq, pc, rc: V.interface_labels(dq if isinstance(_j(dq), list) else V.local_dockq(pid), pc, rc))


# --------------------------------------------------------------------------- #
# scorecard
# --------------------------------------------------------------------------- #
def _values(rows, f):
    v = [r[f] for r in rows if r[f] is not None]
    if v and isinstance(v[0], bool):
        return "bool", [True]
    v = np.asarray(v, float)
    return "num", sorted(set(np.round(np.percentile(v, [25, 50, 75]), 6))) if len(v) else []


def _hit(r, q):
    f, kind, thr, sign = q
    x = r[f]
    if x is None:
        return 0
    return sign if (x is True if kind == "bool" else x >= thr) else 0


def score(r, questions):
    b = r["base_score"]
    return None if b is None else b + sum(_hit(r, q) for q in questions)


def auc(rows, questions):
    s = [(score(r, questions), r["outcome"]) for r in rows]
    s = [(x, y) for x, y in s if x is not None]
    return _auc([x for x, y in s if y], [x for x, y in s if not y])


def fit_question(rows, chosen, f):
    """Best cut-off and sign for feature f given the already chosen questions (training rows)."""
    kind, thrs = _values(rows, f)
    best = None
    for thr in thrs:
        for sign in (1, -1):
            q = (f, kind, thr, sign)
            a = auc(rows, chosen + [q])
            if best is None or a > best[0]:
                best = (a, q)
    return best


def cv_auc(rows, features):
    """Mean out-of-fold AUC of the greedy fit restricted to `features` (in this order)."""
    pdbs = sorted({r["pdb"] for r in rows})
    out = []
    for rep in range(REPEATS):
        perm = np.random.default_rng(rep).permutation(len(pdbs))     # same splits for every candidate
        fold_of = {pdbs[i]: k % FOLDS for k, i in enumerate(perm)}
        for k in range(FOLDS):
            tr = [r for r in rows if fold_of[r["pdb"]] != k]
            te = [r for r in rows if fold_of[r["pdb"]] == k]
            if len({r["outcome"] for r in te}) < 2:
                continue
            qs = []
            for f in features:
                b = fit_question(tr, qs, f)
                if b:
                    qs.append(b[1])
            out.append(auc(te, qs))
    return float(np.mean(out))


def develop(rows, candidates, L, name):
    L.append(f"{name}: {len(rows)} rows, {sum(r['outcome'] for r in rows)} positive")
    base = cv_auc(rows, [])
    L.append(f"  CV AUC AF3 confidence only: {base:.3f}")
    chosen, cur = [], base
    while len(chosen) < MAX_Q:
        trials = []
        for f in candidates:
            if f in chosen or all(r[f] is None for r in rows):
                continue
            trials.append((cv_auc(rows, chosen + [f]), f))
        trials.sort(reverse=True)
        L.append("    candidates: " + ", ".join(f"{f} {a:.3f}" for a, f in trials[:6]))
        if not trials or trials[0][0] - cur < MIN_GAIN:
            break
        cur = trials[0][0]
        chosen.append(trials[0][1])
        L.append(f"  + {trials[0][1]}: CV AUC {cur:.3f}")
    qs = []
    for f in chosen:
        qs.append(fit_question(rows, qs, f)[1])
    sc = sorted(score(r, qs) for r in rows if score(r, qs) is not None)
    cuts = [float(np.percentile(sc, 100 / 3)), float(np.percentile(sc, 200 / 3))]
    L.append(f"  final questions (fit on all development rows): {qs}; level cuts (score) {cuts}")
    return dict(questions=[list(q) for q in qs], cuts=cuts, cv_auc=cur, cv_auc_confidence=base)


def level(s, cuts):
    return 0 if s <= cuts[0] else (1 if s <= cuts[1] else 2)


def boot_diff(rows, f1, f2, n=2000):
    by = {}
    for r in rows:
        by.setdefault(r["pdb"], []).append(r)
    keys = list(by)
    d = []
    for _ in range(n):
        s = [r for k in RNG.choice(keys, len(keys)) for r in by[k]]
        if len({r["outcome"] for r in s}) < 2:
            continue
        d.append(f1(s) - f2(s))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def evaluate(rows, design, L, name, v2key=None):
    qs = [tuple(q) for q in design["questions"]]
    rows = [r for r in rows if r["base_score"] is not None]
    a3, a0 = auc(rows, qs), auc(rows, [])
    lo, hi = boot_diff(rows, lambda s: auc(s, qs), lambda s: auc(s, []))
    L.append(f"{name}: n={len(rows)}; v3 AUC {a3:.3f}, AF3 confidence only {a0:.3f}; "
             f"dAUC {a3 - a0:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    for lv, nm in enumerate(("low", "moderate", "high")):
        s = [r["outcome"] for r in rows if level(score(r, qs), design["cuts"]) == lv]
        p, l, h = wilson(sum(s), len(s))
        L.append(f"    {nm:9s} {100 * p:5.1f}% [{100 * l:.0f}-{100 * h:.0f}] n={len(s)}")
    return a3, a0


def main():
    os.makedirs(OUT, exist_ok=True)
    L = []
    models = _model_index()
    db, fam, fl, il = post_inputs()
    pf, pi = build_rows(db, models, fam, fl, il)
    design = {"fold": develop(pf, FOLD_FEATURES, L, "FOLD (post-cutoff, well folded LDDT_r > 0.85)")}
    L.append("")
    design["interface"] = develop(pi, IFACE_FEATURES, L, "INTERFACE (post-cutoff, correct interface)")
    json.dump(design, open(os.path.join(OUT, "v3_design.json"), "w"), indent=1, default=float)
    L.append("\nDESIGN FROZEN -> v3_design.json. Independent test on 2026:")
    db26, fam26, fl26, il26 = set2026_inputs()
    m26 = {os.path.basename(p).lower(): p for p in __import__("glob").glob(
        os.path.join(__import__("decision_tree.validate_2026", fromlist=["DATA26"]).DATA26, "af", "fold_*_s1_model_0.cif"))}
    tf, ti = build_rows(db26, m26, fam26, fl26, il26)
    evaluate(pf, design["fold"], L, "  fold, post-cutoff (development, in-sample)")
    evaluate(tf, design["fold"], L, "  fold, 2026 (test)")
    evaluate(pi, design["interface"], L, "  interface, post-cutoff (development, in-sample)")
    evaluate(ti, design["interface"], L, "  interface, 2026 (test)")
    for name, rows in (("post_fold", pf), ("post_interface", pi), ("2026_fold", tf), ("2026_interface", ti)):
        if rows:
            with open(os.path.join(OUT, f"{name}_rows.tsv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
                w.writeheader()
                w.writerows(rows)
    open(os.path.join(OUT, "develop_v3.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))




# --------------------------------------------------------------------------- #
# packaged calibration of the combined tree (v3 fold side + v2 interface side)
# --------------------------------------------------------------------------- #
def _read_rows(path):
    conv = lambda v: None if v in ("", "None") else (v == "True" if v in ("True", "False") else float(v))
    out = []
    for r in csv.DictReader(open(path), delimiter="\t"):
        out.append({k: (v if k in ("pdb", "chain") else conv(v)) for k, v in r.items()})
    return out


def calibrate():
    """decision_tree/calibration_v3.json: v3 fold questions, level cuts and rates (post-cutoff =
    development, 2026 = held-out) + the v2 interface calibration."""
    from decision_tree.framework import calibration
    design = json.load(open(os.path.join(OUT, "v3_design.json")))
    fold = design["fold"]
    qs = [tuple(q) for q in fold["questions"]]
    rates = {}
    for name, f in (("post_development", "post_fold_rows.tsv"), ("2026_heldout", "2026_fold_rows.tsv")):
        rows = [r for r in _read_rows(os.path.join(OUT, f)) if r["base_score"] is not None]
        rates[name] = {}
        for lv, nm in enumerate(("WEAK", "MODERATE", "STRONG")):
            s = [r["outcome"] for r in rows if level(score(r, qs), fold["cuts"]) == lv]
            rates[name][nm] = dict(zip(("rate", "ci_lo", "ci_hi"), wilson(sum(s), len(s))), n=len(s))
        rates[name]["auc"] = auc(rows, qs)
        rates[name]["auc_confidence_only"] = auc(rows, [])
    v2 = calibration()
    cal = dict(version="v3: fold side v3 (develop_v3.py), interface side v2 (develop.py)",
               fold=dict(confidence=v2["design"]["fold_confidence"], questions=fold["questions"],
                         cuts=fold["cuts"], rates=rates,
                         note="protein-RNA complexes; designed on post-cutoff targets, tested on 2026"),
               interface=dict(design=dict(interface_downgrades=v2["design"]["interface_downgrades"]),
                              rates=v2["interface"]),
               familiar=dict(seq=SEQ_FAM, tm=TM_FAM))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_v3.json")
    json.dump(cal, open(path, "w"), indent=1, default=float)
    print(f"wrote {path}")
    return cal


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        main()
        calibrate()
