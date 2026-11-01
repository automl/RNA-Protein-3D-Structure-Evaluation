"""
Interface placement classes vs RNA base-pairing fidelity and rare RNA conformations.

Complex-level analysis (the RNA geometry columns are per complex), in the style of
Fig. interface_failure_modes D. For every pre-/post-cutoff complex with graded protein-RNA
interfaces (evaluation/interface_rna_pose.py):
  - share of its interfaces in each placement class, and its dominant class (most interfaces;
    ties go to the failure class);
  - base-pair fidelity of the prediction, RNA_INF = [all, WC, non-WC, stacking] (-1 = the target
    has none -> missing);
  - rare conformations of the TARGET (exp_ table): share of syn glycosidic bonds, of sugar puckers
    other than C3'-endo, and of gamma angles other than g+;
  - AF3's error on them: predicted share minus target share (negative = AF3 under-predicts the rare
    conformation, i.e. models it canonically).
Writes results/plots/all_chains/interface_class_rna_geometry/{complexes.tsv, spearman.tsv,
by_dominant_class.tsv}.  Run in the RNABench env:  python evaluation/interface_class_rna_geometry.py
"""

import ast
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from evaluation.bundle_paths import PROTEIN_RNA_DB
PERIODS = dict(PROTEIN_RNA_DB)
POSE_TSV = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "interface_rna_pose", "interface_rna_pose.tsv")
            for d in ("plots", "plots_odl")]
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_class_rna_geometry")

CLASSES = [("correct interface", "Correct interface"),
           ("same place, duplex slid", "Correct site, slid duplex"),
           ("right protein site, wrong RNA pose", "Correct site, incorrect RNA"),
           ("same place, RNA turned", "Correct site, rotated RNA"),
           ("shifted nearby", "Shifted site"),
           ("different region", "Distal site"),
           ("protein binding region mispredicted", "Incorrect binding site")]
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}
FEATURES = [("inf_wc", "WC pair fidelity (INF)"), ("inf_nonwc", "non-WC pair fidelity (INF)"),
            ("inf_stack", "Stacking fidelity (INF)"),
            ("exp_syn", "Target syn share"), ("exp_non_c3endo", "Target non-C3'-endo share"),
            ("exp_non_gplus", "Target non-g+ gamma share"),
            ("d_syn", "AF3 - target syn share"), ("d_non_c3endo", "AF3 - target non-C3'-endo share"),
            ("d_non_gplus", "AF3 - target non-g+ share")]


def _dict(raw):
    if raw in (None, "", "None"):
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(str(raw))
            return v if isinstance(v, dict) else {}
        except (ValueError, SyntaxError):
            continue
    return {}


def _share(raw, keys, invert=False):
    """Share of residues in the states `keys` (or, with invert, in any other state)."""
    d = _dict(raw)
    n = sum(float(v) for v in d.values())
    if not n:
        return np.nan
    k = sum(float(v) for s, v in d.items() if (s in keys) != invert)
    return k / n


def _inf(raw):
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        v = [float(x) for x in v]
    except (TypeError, ValueError):
        return [np.nan] * 4
    v = (v + [np.nan] * 4)[:4]
    return [x if x >= 0 else np.nan for x in v]


def complex_features(db):
    con = sqlite3.connect(db)
    pred = {str(r[0]).upper(): r[1:] for r in con.execute(
        "SELECT exp_db_id, RNA_INF, RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle FROM pred_protein_rna")}
    exp = {str(r[0]).upper(): r[1:] for r in con.execute(
        "SELECT PDBId, RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle FROM exp_protein_rna")}
    con.close()
    out = {}
    for pid, (inf, pg, pp, pgam) in pred.items():
        eg, ep, egam = exp.get(pid, (None, None, None))
        _all, wc, nonwc, stack = _inf(inf)
        f = dict(inf_wc=wc, inf_nonwc=nonwc, inf_stack=stack,
                 exp_syn=_share(eg, {"syn"}), exp_non_c3endo=_share(ep, {"C3'-endo"}, invert=True),
                 exp_non_gplus=_share(egam, {"g+"}, invert=True),
                 exp_exotic_pucker=_share(ep, {"C3'-endo", "C2'-endo"}, invert=True),
                 pred_syn=_share(pg, {"syn"}), pred_non_c3endo=_share(pp, {"C3'-endo"}, invert=True),
                 pred_non_gplus=_share(pgam, {"g+"}, invert=True))
        for k in ("syn", "non_c3endo", "non_gplus"):
            f[f"d_{k}"] = f[f"pred_{k}"] - f[f"exp_{k}"]
        out[pid] = f
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pose = next((p for p in POSE_TSV if os.path.isfile(p)), None)
    d = pd.read_csv(pose, sep="\t")
    d = d[d.dockq.notna()].copy()
    d["placement"] = d["placement"].replace(MERGE)
    rows = []
    for period, db in PERIODS.items():
        feats = complex_features(db)
        for pid, g in d[d.period == period].groupby("pdb"):
            shares = g.placement.value_counts(normalize=True)
            counts = g.placement.value_counts()
            order = [k for k, _ in CLASSES]
            top = counts.max()
            tied = [k for k in order if counts.get(k, 0) == top]
            dominant = next((k for k in tied if k != "correct interface"), tied[0])
            row = dict(period=period, pdb=pid, n_interfaces=len(g), dominant=dominant,
                       **{f"share_{k}": shares.get(k, 0.0) for k, _ in CLASSES})
            row.update(feats.get(str(pid).upper(), {}))
            rows.append(row)
    cx = pd.DataFrame(rows)
    cx.to_csv(os.path.join(OUTDIR, "complexes.tsv"), sep="\t", index=False)

    rho_rows, med_rows = [], []
    for period in ("post", "pre"):
        c = cx[cx.period == period]
        for k, lab in CLASSES:
            r = dict(period=period, placement=lab)
            for f, _ in FEATURES:
                ok = c[[f"share_{k}", f]].dropna()
                if len(ok) >= 10 and ok[f].nunique() > 1:
                    rho, p = spearmanr(ok[f"share_{k}"], ok[f])
                    r[f], r[f"{f}_p"], r[f"{f}_n"] = rho, p, len(ok)
            rho_rows.append(r)
            g = c[c.dominant == k]
            m = dict(period=period, dominant=lab, n_complexes=len(g))
            for f, _ in FEATURES:
                m[f] = g[f].median()
                m[f"{f}_n"] = int(g[f].notna().sum())
            med_rows.append(m)
    rho = pd.DataFrame(rho_rows)
    med = pd.DataFrame(med_rows)
    rho.to_csv(os.path.join(OUTDIR, "spearman.tsv"), sep="\t", index=False)
    med.to_csv(os.path.join(OUTDIR, "by_dominant_class.tsv"), sep="\t", index=False)
    cols = [f for f, _ in FEATURES]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print("Spearman rho: share of a complex's interfaces in the class vs feature")
        print(rho[["period", "placement"] + cols].round(2).to_string(index=False))
        print("\nMedians by dominant class of the complex")
        print(med[["period", "dominant", "n_complexes"] + cols].round(2).to_string(index=False))
    print(f"\n{len(cx)} complexes -> {OUTDIR}")


if __name__ == "__main__":
    main()
