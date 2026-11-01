"""
Probability form of the decision tree for protein-RNA predictions, designed on POST-cutoff targets only.

  fold        P(longest RNA chain well folded, LDDT_r > 0.85) by logistic regression on the intra-RNA PAE of the
              longest RNA chain (mean of AF3's `pae` matrix over the token pairs within that chain, full_data JSON)
              and the four questions of the fold score (familiar RNA sequence, familiar RNA fold, RNA-bound protein
              template, >= 2 RNA chains). pLDDT_r is added only if it raises the cross-validated AUC by >= 0.01.
  interface   P(correct interface) per RNA chain on AF3's minimum chain-pair PAE between the RNA chain and its
              protein partners (summary JSON `chain_pair_pae_min`), a familiar RNA and a well-folded partner
              (pTM_p >= 0.84 and pLDDT_r >= 0.78); ipTM_r is added only if it raises the cross-validated AUC by
              >= 0.01. (With both, ipTM_r took a negative weight: r = -0.80 with the chain-pair PAE, CV AUC 0.770 for
              ipTM_r against 0.806 for the PAE with the same two questions.)
  overall     P(usable RNA chain: correct interface and LDDT_r > 0.85 of the chain) = P_fold x P_interface, or a
              logistic model on the two log-odds if that is better calibrated (cross-validated log-loss).
  levels      WEAK < 25% <= MODERATE < 60% <= STRONG, for all three.
  CV          5 x 5-fold, grouped by PDB entry. The design is frozen in decision_tree/calibration_probability.json;
              --test applies it once to the 2026 set.

RNA-only subtrees (single-chain: pLDDT_r with an RNA alignment, intra-RNA PAE without; RNA complexes: duplex in the
AF3 model + intra-RNA PAE of the longest chain) get the same probability form on their development sets (--rna).

Run from the repo root (RNABench env):  python -m decision_tree.develop_probability [--test] [--rna [--test]]
Outputs: results/plots/all_chains/decision_tree/probability/
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

from sklearn.linear_model import LogisticRegression

from decision_tree.af3_output import read_prediction
from decision_tree.develop import wilson
from decision_tree.develop_v3 import OUT as V3OUT, _read_rows
from decision_tree.validate import PP, _auc, _model_index

# PROB_VARIANT=iptm: interface on ipTM_r instead of the chain-pair PAE (outputs and calibration get the suffix)
VARIANT = os.environ.get("PROB_VARIANT", "")
_SFX = f"_{VARIANT}" if VARIANT else ""
BASE_OUT = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "probability")
OUT = BASE_OUT + _SFX
CAL = os.path.join(_REPO_ROOT, "decision_tree", f"calibration_probability{_SFX}.json")
IFACE_CONF, IFACE_ALT = ("iptm", "pae_min") if VARIANT == "iptm" else ("pae_min", "iptm")
COMB = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "combined")
LEVELS = ("WEAK", "MODERATE", "STRONG")
CUTS = (0.25, 0.60)
MIN_GAIN = 0.01
FOLD_Q = ["seq_familiar", "tm_familiar", "rbp_template", "multi_rna"]
IFACE_Q = ["familiar", "wellfolded"]


def level(p):
    return 0 if p < CUTS[0] else (1 if p < CUTS[1] else 2)


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
def _af3_cache():
    path = os.path.join(BASE_OUT, "af3_cache.json")
    return path, (json.load(open(path)) if os.path.exists(path) else {})


def af3_values(cif, cache):
    """per RNA chain: (intra-RNA PAE, ipTM_r, pLDDT_r); longest RNA chain id."""
    if cif not in cache:
        p = read_prediction(cif)
        rc = p["rna_chains"]
        cache[cif] = dict(chains={c["chain"]: [c.get("rna_pae"), c.get("rna_iptm"), c.get("rna_plddt")] for c in rc},
                          longest=max(rc, key=lambda c: c["length"])["chain"] if rc else None)
    return cache[cif]


def _b(v):
    return 1 if v is True or v == "True" else 0


PRED_TM = os.path.join(BASE_OUT, "predicted_fold_tm.json")
FAMILIAR_TM = 0.5


def _pred_tm(name):
    """fold familiarity from the AF3 model (predicted_fold_familiarity.py): {pdb: best TM to pre-cutoff RNAs}"""
    key = "2026" if name.startswith("2026") else "post"
    d = dict(json.load(open(PRED_TM)).get(key, {})) if os.path.exists(PRED_TM) else {}
    extra = PRED_TM.replace(".json", "_extra.json")          # 2026 models computed in parallel (same function)
    if key == "2026" and os.path.exists(extra):
        d.update(json.load(open(extra)))
    return {p: v[0] for p, v in d.items()}


PDB_SEQ = os.path.join(BASE_OUT, "pdb_seq_identity.json")
FAMILIAR_SEQ = 0.8


def _pdb_seq(name):
    """sequence familiarity against every pre-cutoff PDB RNA (pdb_sequence_familiarity.py): {pdb: best identity}"""
    key = "2026" if name.startswith("2026") else "post"
    return (json.load(open(PDB_SEQ)).get(key, {}) if os.path.exists(PDB_SEQ) else {})


def _num(v):
    return None if v is None else float(v)


def _complex_features(r, tm_pred, seq_id=None):
    """candidate inputs known before prediction or read from the AF3 model; none needs the experimental structure"""
    ln, ia = _num(r.get("rna_len")), _num(r.get("iface_area"))
    seq_f = _b(r.get("seq_familiar")) if seq_id is None else int(seq_id >= FAMILIAR_SEQ)
    return dict(seq_familiar=seq_f, tm_familiar=int(tm_pred is not None and tm_pred >= FAMILIAR_TM),
                family=_b(r.get("family")), pseudoknot=_b(r.get("pseudoknot")), n_stems=_num(r.get("n_stems")),
                n_hairpins=_num(r.get("n_hairpins")), n_internal=_num(r.get("n_internal")),
                n_multiloops=_num(r.get("n_multiloops")), log_len=None if not ln else float(np.log(ln)),
                multi_rna=int((r.get("n_rnas") or 0) >= 2), many_proteins=int((r.get("n_proteins") or 0) >= 3),
                gc=_num(r.get("gc")), mfe_per_nt=_num(r.get("mfe_per_nt")), complexity=_num(r.get("complexity")),
                rbp_template=_b(r.get("rbp_template")), n_rbp_templates=_num(r.get("n_rbp_templates")),
                protein_template=_b(r.get("protein_template")), n_protein_templates=_num(r.get("n_protein_templates")),
                rna_ptm=_num(r.get("rna_ptm")), global_pae=_num(r.get("global_pae")),
                frac_disordered=_num(r.get("frac_disordered")), log_iface_area=None if not ia else float(np.log(ia)))


def fold_rows(name, cifs, cache):
    rows, tmp, sid = [], _pred_tm(name), _pdb_seq(name)
    for r in _read_rows(os.path.join(V3OUT, name)):
        cif = cifs.get(r["pdb"])
        if not cif:
            continue
        a = af3_values(cif, cache)
        if a["longest"] is None:
            continue
        pae, _, plddt = a["chains"][a["longest"]]
        if pae is None:
            continue
        rows.append(dict(pdb=r["pdb"], y=bool(r["outcome"]), pae=pae, plddt=plddt, **_complex_features(r, tmp.get(r["pdb"]), sid.get(r["pdb"]))))
    return rows


def iface_rows(name):
    rows, tmp, sid = [], _pred_tm(name), _pdb_seq(name)
    conv = lambda v: None if v in ("", "None") else (v == "True" if v in ("True", "False") else float(v))
    for r in csv.DictReader(open(os.path.join(V3OUT, name)), delimiter="\t"):
        r = {k: (v if k in ("pdb", "chain", "outcome") else conv(v)) for k, v in r.items()}
        ip, pp, pl = r.get("chain_iptm"), r.get("partner_ptm"), r.get("chain_rna_plddt")
        if ip is None:
            continue
        pl = None if pl is None else (pl / 100 if pl > 1 else pl)
        cf = _complex_features(r, tmp.get(r["pdb"]), sid.get(r["pdb"]))
        rows.append(dict(pdb=r["pdb"], chain=r["chain"], y=r["outcome"] == "True", iptm=ip, pae_min=r.get("pae_min"),
                         partner_ptm=pp, chain_plddt=pl, familiar=int(cf["seq_familiar"] or cf["tm_familiar"]),
                         wellfolded=int(not ((pp is not None and pp < 0.84) or (pl is not None and pl < 0.78))), **cf))
    return rows


FOLD_CANDIDATES = ["seq_familiar", "tm_familiar", "family", "pseudoknot", "n_stems", "n_hairpins", "n_internal",
                   "n_multiloops", "log_len", "multi_rna", "many_proteins", "gc", "mfe_per_nt", "complexity",
                   "rbp_template", "n_rbp_templates", "protein_template", "n_protein_templates", "plddt", "rna_ptm",
                   "global_pae", "frac_disordered", "log_iface_area"]
IFACE_CANDIDATES = ["familiar", "seq_familiar", "tm_familiar", "wellfolded", "partner_ptm", "chain_plddt", "iptm",
                    "family", "log_len", "multi_rna", "many_proteins", "rbp_template", "protein_template", "gc",
                    "mfe_per_nt", "complexity", "global_pae", "frac_disordered", "log_iface_area"]
MAX_FEATURES = 6


def impute(rows, feats, medians=None):
    """missing values -> post-cutoff median (stored with the design so that the tool can apply it)"""
    medians = medians or {f: float(np.nanmedian([np.nan if r.get(f) is None else r[f] for r in rows])) for f in feats}
    return [dict(r, **{f: (medians[f] if r.get(f) is None else r[f]) for f in feats}) for r in rows], medians


def _spans(rows, feats):
    """value range used by the tool for an unknown input: {0, 1} for yes/no inputs, 5th-95th percentile otherwise"""
    out = {}
    for f in feats:
        v = np.array([r[f] for r in rows if r.get(f) is not None], float)
        out[f] = [0, 1] if set(np.unique(v)) <= {0.0, 1.0} else [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    return out


def select(rows, base, cands, L, name):
    """greedy forward selection on post-cutoff targets: add the input with the largest cross-validated AUC gain
    while it is >= MIN_GAIN, at most MAX_FEATURES inputs in all"""
    feats = list(base)
    best, _ = cv(rows, feats)
    L.append(f"{name}: {len(rows)} rows, {sum(r['y'] for r in rows)} positive; CV AUC {feats} {best:.3f}")
    while len(feats) < MAX_FEATURES:
        trial = sorted(((cv(rows, feats + [c])[0], c) for c in cands if c not in feats), reverse=True)
        if not trial or trial[0][0] - best < MIN_GAIN:
            if trial:
                L.append(f"  stop: best addition {trial[0][1]} gains {trial[0][0] - best:+.3f}")
            break
        best, c = trial[0]
        feats.append(c)
        L.append(f"  + {c}: CV AUC {best:.3f}")
    return feats


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def X(rows, feats):
    return np.array([[np.nan if r[f] is None else float(r[f]) for f in feats] for r in rows], float)


def fit(rows, feats):
    x, y = X(rows, feats), np.array([r["y"] for r in rows])
    keep = ~np.isnan(x).any(1)
    return LogisticRegression(C=1e6, max_iter=5000).fit(x[keep], y[keep])


def predict(model, rows, feats):
    x = X(rows, feats)
    p = np.full(len(rows), np.nan)
    ok = ~np.isnan(x).any(1)
    p[ok] = model.predict_proba(x[ok])[:, 1]
    return p


def cv(rows, feats, reps=5, folds=5):
    pdbs = sorted({r["pdb"] for r in rows})
    aucs, ll = [], []
    for rep in range(reps):
        perm = np.random.default_rng(rep).permutation(len(pdbs))
        fo = {pdbs[j]: k % folds for k, j in enumerate(perm)}
        oof = np.full(len(rows), np.nan)
        for k in range(folds):
            tr = [r for r in rows if fo[r["pdb"]] != k]
            idx = [i for i, r in enumerate(rows) if fo[r["pdb"]] == k]
            oof[idx] = predict(fit(tr, feats), [rows[i] for i in idx], feats)
        y = np.array([r["y"] for r in rows]); m = ~np.isnan(oof)
        aucs.append(_auc(list(oof[m & y]), list(oof[m & ~y])))
        pr = np.clip(oof[m], 1e-6, 1 - 1e-6)
        ll.append(-np.mean(y[m] * np.log(pr) + (1 - y[m]) * np.log(1 - pr)))
    return float(np.mean(aucs)), float(np.mean(ll))


def choose(rows, base, extra, L, name):
    a0, _ = cv(rows, base)
    L.append(f"{name}: {len(rows)} rows, {sum(r['y'] for r in rows)} positive; CV AUC {base} {a0:.3f}")
    a1, _ = cv(rows, base + [extra])
    L.append(f"  + {extra}: CV AUC {a1:.3f} ({'kept' if a1 - a0 >= MIN_GAIN else 'not kept'}, gain {a1 - a0:+.3f})")
    return base + [extra] if a1 - a0 >= MIN_GAIN else base


def describe(model, feats):
    return dict(intercept=float(model.intercept_[0]), coef={f: float(c) for f, c in zip(feats, model.coef_[0])})


def apply(design, rows):
    z = design["intercept"] + sum(c * np.array([np.nan if r[f] is None else float(r[f]) for r in rows])
                                  for f, c in design["coef"].items())
    return 1 / (1 + np.exp(-z))


def rates(p, y):
    out = {}
    for k, nm in enumerate(LEVELS):
        s = [yy for pp, yy in zip(p, y) if not np.isnan(pp) and level(pp) == k]
        r, lo, hi = wilson(sum(s), len(s)) if s else (np.nan, np.nan, np.nan)
        mp = [pp for pp in p if not np.isnan(pp) and level(pp) == k]
        out[nm] = dict(rate=r, ci_lo=lo, ci_hi=hi, n=len(s), k=int(sum(s)), mean_p=float(np.mean(mp)) if mp else None)
    return out


def fmt(rt):
    return " | ".join(f"{nm} {100 * v['rate']:.0f}% [{100 * v['ci_lo']:.0f}-{100 * v['ci_hi']:.0f}] n={v['n']} "
                      f"(predicted {100 * v['mean_p']:.0f}%)" if v["n"] else f"{nm} n=0" for nm, v in rt.items())


# --------------------------------------------------------------------------- #
def inputs():
    models = _model_index()
    fn = {str(e).strip().upper(): models.get(os.path.basename(str(f)).lower()) for e, f in
          sqlite3.connect(PP["a2021.db"]).execute("SELECT exp_db_id, FileName FROM pred_protein_rna")}
    from decision_tree.validate_2026 import MODELS as M26
    return fn, M26


def combine(pf, pi, rule="product", rho=0.40):
    """P(usable) from P_fold and P_interface: product (independent), min (fully dependent; upper limit) or a
    Gaussian copula with correlation rho between fold and interface success."""
    pf, pi = np.asarray(pf, float), np.asarray(pi, float)
    if rule == "min":
        return np.minimum(pf, pi)
    if rule == "copula":
        from scipy.stats import multivariate_normal, norm
        z = np.column_stack([norm.ppf(np.clip(pf, 1e-9, 1 - 1e-9)), norm.ppf(np.clip(pi, 1e-9, 1 - 1e-9))])
        return np.array([multivariate_normal.cdf(x, mean=[0, 0], cov=[[1, rho], [rho, 1]]) for x in z])
    return pf * pi


def oof_rho(fr, ff, ir, fi, reps=5, folds=5):
    """copula correlation maximising the likelihood of usable chains, from out-of-fold P_fold and P_interface"""
    from scipy.optimize import minimize_scalar
    pdbs = sorted({r["pdb"] for r in fr} | {r["pdb"] for r in ir})
    pairs = []
    for rep in range(reps):
        perm = np.random.default_rng(rep).permutation(len(pdbs))
        fo = {pdbs[j]: k % folds for k, j in enumerate(perm)}
        for k in range(folds):
            mf = fit([r for r in fr if fo[r["pdb"]] != k], ff)
            mi = fit([r for r in ir if fo[r["pdb"]] != k], fi)
            tf = [r for r in fr if fo[r["pdb"]] == k]
            ti = [r for r in ir if fo[r["pdb"]] == k]
            ur = usable_rows("post_rows.tsv", dict(zip([r["pdb"] for r in tf], predict(mf, tf, ff))),
                             dict(zip([(r["pdb"], r["chain"]) for r in ti], predict(mi, ti, fi))))
            pairs += [(r["pf"], r["pi"], r["y"]) for r in ur]
    a, b, y = (np.array(v, float) for v in zip(*pairs))

    def nll(rho):
        p = np.clip(combine(a, b, "copula", rho), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.0, 0.99), method="bounded", options=dict(xatol=1e-3)).x)


def usable_rows(name, F, I):
    rows = []
    for r in csv.DictReader(open(os.path.join(COMB, name)), delimiter="\t"):
        f, i = F.get(r["pdb"]), I.get((r["pdb"], r["chain"]))
        if f is None or i is None:
            continue
        rows.append(dict(pdb=r["pdb"], chain=r["chain"], y=r["outcome"] == "True", pf=f, pi=i))
    return rows


def design():
    os.makedirs(OUT, exist_ok=True)
    fn, _ = inputs()
    cpath, cache = _af3_cache()
    L = ["Probability form of the decision tree, designed on post-cutoff targets", "=" * 70]
    fr = fold_rows("post_fold_rows.tsv", fn, cache)
    json.dump(cache, open(cpath, "w"))
    fr, fmed = impute(fr, FOLD_CANDIDATES + ["plddt"])
    ff = choose(fr, ["pae"] + FOLD_Q, "plddt", L, "FOLD")
    ir = iface_rows("post_interface_rows.tsv")
    ir, imed = impute(ir, IFACE_CANDIDATES + ["pae_min"])
    ir = [r for r in ir if r["pae_min"] is not None]
    fi = choose(ir, [IFACE_CONF] + IFACE_Q, IFACE_ALT, L, "INTERFACE")
    if IFACE_ALT in fi:                     # the two AF3 interface measures are not combined (r = -0.80)
        fi = [IFACE_CONF] + IFACE_Q
    fd, idg = describe(fit(fr, ff), ff), describe(fit(ir, fi), fi)
    pf = dict(zip([r["pdb"] for r in fr], apply(fd, fr)))
    pi = dict(zip([(r["pdb"], r["chain"]) for r in ir], apply(idg, ir)))
    ur = usable_rows("post_rows.tsv", pf, pi)
    # overall: P_fold and P_interface joined by a Gaussian copula; its one correlation is fitted on out-of-fold
    # side probabilities of post-cutoff targets (5 x 5 folds grouped by PDB entry)
    rho = oof_rho(fr, ff, ir, fi)
    y = np.array([r["y"] for r in ur], float)
    po = combine([r["pf"] for r in ur], [r["pi"] for r in ur], "copula", rho)
    L.append(f"OVERALL: {len(ur)} RNA chains; Gaussian copula, correlation fitted out of fold rho = {rho:.3f}")
    cal = dict(levels=dict(names=LEVELS, cuts=CUTS),
               fold=dict(features=ff, medians={f: fmed[f] for f in ff if f in fmed}, spans=_spans(fr, ff), **fd),
               interface=dict(features=fi, medians={f: imed[f] for f in fi if f in imed}, spans=_spans(ir, fi), **idg),
               overall=dict(rule="copula", rho=rho), post={})
    # post-cutoff (in-sample) calibration by level
    for side, prob, yy in (("fold", apply(fd, fr), [r["y"] for r in fr]), ("interface", apply(idg, ir), [r["y"] for r in ir]),
                           ("overall", po, list(y))):
        cal["post"][side] = rates(prob, yy)
        L.append(f"  post {side:9s}: {fmt(cal['post'][side])}")
    L.append(f"\nfold model: {fd}\ninterface model: {idg}\noverall: {cal['overall']}")
    old = json.load(open(CAL)) if os.path.exists(CAL) else {}
    cal.update({k: v for k, v in old.items() if k.startswith("rna_")})      # RNA-only subtrees are designed separately
    json.dump(cal, open(CAL, "w"), indent=1, default=float)
    open(os.path.join(OUT, "design.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def test():
    cal = json.load(open(CAL))
    _, M26 = inputs()
    cpath, cache = _af3_cache()
    fr = fold_rows("2026_fold_rows.tsv", M26, cache)
    json.dump(cache, open(cpath, "w"))
    fr, _ = impute(fr, cal["fold"]["features"], cal["fold"].get("medians"))
    ir = iface_rows("2026_interface_rows.tsv")
    ir, _ = impute(ir, cal["interface"]["features"], cal["interface"].get("medians"))
    pf, pi = apply(cal["fold"], fr), apply(cal["interface"], ir)
    ur = usable_rows("2026_rows.tsv", dict(zip([r["pdb"] for r in fr], pf)), dict(zip([(r["pdb"], r["chain"]) for r in ir], pi)))
    po = combine([r["pf"] for r in ur], [r["pi"] for r in ur], cal["overall"]["rule"], cal["overall"].get("rho", 0.0))
    L = ["2026 test of the frozen probability design", "=" * 70]
    res = {}
    for side, prob, rows in (("fold", pf, fr), ("interface", pi, ir), ("overall", po, ur)):
        y = [r["y"] for r in rows]
        a = _auc([p for p, t in zip(prob, y) if t and not np.isnan(p)], [p for p, t in zip(prob, y) if not t and not np.isnan(p)])
        res[side] = dict(auc=a, levels=rates(prob, y), n=len(rows))
        L.append(f"{side:9s} n={len(rows):3d} AUC {a:.3f} | {fmt(res[side]['levels'])}")
    cal["test_2026"] = res
    json.dump(cal, open(CAL, "w"), indent=1, default=float)
    for name, rows, prob in (("fold", fr, pf), ("interface", ir, pi), ("overall", ur, po)):
        with open(os.path.join(OUT, f"2026_{name}.tsv"), "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["pdb", "chain", "outcome", "probability", "level"])
            for r, p in zip(rows, prob):
                w.writerow([r["pdb"], r.get("chain", ""), r["y"], None if np.isnan(p) else round(float(p), 4),
                            None if np.isnan(p) else LEVELS[level(p)]])
    open(os.path.join(OUT, "test_2026.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------- #
# RNA-only predictions (no protein chain), designed on their development sets
# --------------------------------------------------------------------------- #
RNA_SINGLE = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "rna_single", "rows.tsv")


def _rna_single_rows(sets):
    rows = []
    for r in csv.DictReader(open(RNA_SINGLE), delimiter="\t"):
        if r["set"] not in sets or r["lddt"] in ("", "None"):
            continue
        f = lambda k: None if r[k] in ("", "None") else float(r[k])
        pl, pae = f("plddt"), f("rna_pae")
        rows.append(dict(pdb=r["pdb"], set=r["set"], y=float(r["lddt"]) > 0.85,
                         plddt=None if pl is None else 100 * pl,          # pLDDT_r in %
                         pae=None if pae is None else -pae))             # rows store the negated PAE
    return rows


def _rna_multi_rows(keys):
    import decision_tree.develop_rna_multi as Mu
    rows = []
    for key in keys:
        sets = Mu.DESIGN_SETS if key in Mu.DESIGN_SETS else Mu.TEST_SETS
        for r in Mu.usable(Mu.load(key, sets)):
            if r.get("rna_pae") is None:
                continue
            rows.append(dict(pdb=r["pdb"], set=key, y=r["lddt"] > 0.85, pae=-r["rna_pae"],
                             duplex=int((r.get("duplex_frac") or 0) >= Mu.DUPLEX_FRAC)))
    return rows


def _fit_l2(rows, feats):
    x, y = X(rows, feats), np.array([r["y"] for r in rows])
    keep = ~np.isnan(x).any(1)
    return LogisticRegression(C=1.0, max_iter=5000).fit(x[keep], y[keep])     # mild L2: near-separable sets


RNA_DESIGN = {   # subtree, branch: (development sets, test sets, features)
    ("single", "with_alignment"): (["design_with"], ["test_local_with"], ["plddt"]),
    ("single", "without_alignment"): (["design_without"], ["test_local_without", "test_server"], ["pae"]),
    ("multi", "with_alignment"): (["add_with"], ["test_local_with"], ["duplex", "pae"]),
    ("multi", "without_alignment"): (["add_without", "paper_with"], ["test_local_without", "test_server"], ["duplex", "pae"]),
}


def rna_design(test=False):
    cal = json.load(open(CAL))
    L = [("2026 test" if test else "Development") + " of the RNA-only probability subtrees", "=" * 70]
    for (sub, br), (dev, tst, feats) in RNA_DESIGN.items():
        load = _rna_single_rows if sub == "single" else _rna_multi_rows
        key = f"rna_{sub}"
        if not test:
            rows = load(dev)
            m = _fit_l2(rows, feats)
            cal.setdefault(key, {})[br] = dict(features=feats, intercept=float(m.intercept_[0]),
                                               coef={f: float(c) for f, c in zip(feats, m.coef_[0])})
            p = apply(cal[key][br], rows)
            y = [r["y"] for r in rows]
            cal[key][br]["dev"] = dict(auc=_auc([a for a, t in zip(p, y) if t], [a for a, t in zip(p, y) if not t]),
                                       levels=rates(p, y), n=len(rows))
            L.append(f"{key} {br:17s} dev n={len(rows):3d} AUC {cal[key][br]['dev']['auc']:.3f} | {fmt(cal[key][br]['dev']['levels'])}")
        else:
            for t in tst:
                rows = load([t]); p = apply(cal[key][br], rows); y = [r["y"] for r in rows]
                a = _auc([a for a, tt in zip(p, y) if tt], [a for a, tt in zip(p, y) if not tt]) if 0 < sum(y) < len(y) else float("nan")
                cal[key][br].setdefault("test_2026", {})[t] = dict(auc=a, levels=rates(p, y), n=len(rows))
                L.append(f"{key} {br:17s} {t:18s} n={len(rows):3d} AUC {a:.3f} | {fmt(cal[key][br]['test_2026'][t]['levels'])}")
    json.dump(cal, open(CAL, "w"), indent=1, default=float)
    open(os.path.join(OUT, "rna_test_2026.txt" if test else "rna_design.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--rna" in sys.argv:
        rna_design(test="--test" in sys.argv)
    else:
        test() if "--test" in sys.argv else design()
