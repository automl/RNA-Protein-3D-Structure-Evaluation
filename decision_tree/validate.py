"""
Held-out validation of the decision-framework reliability tiers.

The tier cutpoints are fixed (decision_tree/rules.py). Here the released code is run on raw
AF3 output and the labels are compared with accuracy labels that played no part in setting
the cutpoints:

  A  interface placement classes (evaluation/interface_rna_pose.py; RNA_interface_placement)
     of every protein-RNA RNA chain, pre- and post-cutoff: does the interface tier separate
     correct interfaces from the failure modes?
  B  the 15 AF3 samples (3 seeds x 5 models) of the 50 seed targets
     (results/plots/all_chains/seed_sample_interfaces/seed_sample_dockq.tsv): interface tier
     of each sampled interface vs its DockQ (here the RNA's protein partner is known);
  C  the multi-chain RNA set (complexes of >= 2 RNA chains, never used for the framework):
     fold tier vs RNA all-atom LDDT and TM-score.

Writes results/plots/all_chains/decision_tree/{validation_*.tsv, validation_summary.txt}.
Run in the RNABench env from the repo root:  python -m decision_tree.validate
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sqlite3
import sys
from collections import Counter

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.af3_output import read_chains, read_prediction
from decision_tree.rules import TIERS, fold_tier, interface_tier
from evaluation.placement_labels import best_per_rna_chain

from evaluation.bundle_paths import ALL_DB, PERIOD_SET, PROTEIN_RNA_DB, seed_dir
# protein-RNA databases by their former file names (Zenodo bundle, data/all_data_zip/all_databases)
PP = {"a2021.db": PROTEIN_RNA_DB["post"], "b2021.db": PROTEIN_RNA_DB["pre"]}
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree")
SEED_TSV = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "seed_sample_interfaces", "seed_sample_dockq.tsv")
MULTI = {p: os.path.join(ALL_DB, PERIOD_SET[p], "rna_multichain.db") for p in ("post", "pre")}
FAILURE_ORDER = ["right_protein_site_wrong_rna_pose", "same_place_same_orientation", "same_place_rna_turned",
                 "shifted_nearby", "different_region", "protein_binding_region_mispredicted"]
ACCEPTABLE = 0.23
GOOD_RNA_LDDT = 0.85


def _auc(pos, neg):
    """ROC-AUC of an ordinal score (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


ORD = {"bad": 0, "mid": 1, "good": 2}


def _model_index():
    """{model file name: path} over the published structures (evaluation.interface_rna_pose.MODEL_ROOTS)."""
    from evaluation.interface_rna_pose import MODEL_ROOTS
    idx = {}
    for p in [q for r in MODEL_ROOTS for q in sorted(glob.glob(os.path.join(r, "*_model_0.cif")))]:
        idx.setdefault(os.path.basename(p).lower(), p)
    return idx


def db_path(folder, name):
    """Database `name` of a folder, or of PP (protein-RNA databases by their former file names)."""
    return folder[name] if isinstance(folder, dict) else os.path.join(folder, name)


# --------------------------------------------------------------------------- #
# A  placement classes
# --------------------------------------------------------------------------- #
def placement_validation():
    idx = _model_index()
    rows, missing = [], 0
    for period, db in (("pre", "b2021.db"), ("post", "a2021.db")):
        con = sqlite3.connect(PP[db])
        for eid, fn, plc in con.execute("SELECT exp_db_id, FileName, RNA_interface_placement FROM pred_protein_rna"):
            plc = best_per_rna_chain(plc)          # one class per RNA chain: best over its interfaces
            cif = idx.get(os.path.basename(str(fn)).lower())
            if not plc or cif is None:
                missing += bool(plc)
                continue
            pred = read_prediction(cif, with_plddt=False)
            for c in pred["rna_chains"]:
                cls = plc.get(c["chain"])
                if not cls:
                    continue
                rows.append(dict(period=period, pdb=eid, chain=c["chain"], placement=cls,
                                 rna_iptm=c["rna_iptm"], pae_min=c["pae_min"],
                                 tier=interface_tier(c["rna_iptm"], c["pae_min"])))
    return rows, missing


# --------------------------------------------------------------------------- #
# B  seed samples
# --------------------------------------------------------------------------- #
def _seed_cif(pid, seed, model):
    stem = f"fold_{pid}_s{seed}_model_{model}.cif"
    c = os.path.join(seed_dir(seed), "af", stem)
    return c if os.path.isfile(c) else None


def seed_validation():
    rows = []
    cache = {}
    for r in csv.DictReader(open(SEED_TSV), delimiter="\t"):
        pid, seed, model = r["pdb"].lower(), int(r["seed"]), int(r["model"])
        key = (pid, seed, model)
        if key not in cache:
            cif = _seed_cif(pid, seed, model)
            cache[key] = None
            if cif:
                summ = cif.replace(f"_model_{model}.cif", f"_summary_confidences_{model}.json")
                if os.path.exists(summ):
                    cache[key] = (read_chains(cif), json.load(open(summ)))
        if cache[key] is None:
            continue
        chains, s = cache[key]
        prot, rna = r["mdl_pair"].split(":")
        if prot not in chains or rna not in chains:
            continue
        i, j = chains[prot]["index"], chains[rna]["index"]
        ip = s["chain_iptm"][j]
        pae = s["chain_pair_pae_min"][i][j]
        rows.append(dict(pdb=r["pdb"], seed=seed, model=model, pair=r["mdl_pair"], ref_pair=r["ref_pair"],
                         dockq=float(r["dockq"]), rna_iptm=ip, pae_min=pae, tier=interface_tier(ip, pae)))
    return rows


# --------------------------------------------------------------------------- #
# C  multi-chain RNA
# --------------------------------------------------------------------------- #
def multichain_validation():
    rows = []
    for period, db in MULTI.items():
        con = sqlite3.connect(db)
        for fn, pl, lddt, tm in con.execute(
                "SELECT FileName, af3_rna_pLDDT_avg, RNA_LDDT_allAtom, RNA_TM FROM pred_rna_rna"):
            def f(v):
                try:
                    return float(json.loads(v)[0]) if str(v).startswith("[") else float(v)
                except (TypeError, ValueError, IndexError):
                    return np.nan
            rows.append(dict(period=period, model=fn, rna_plddt=f(pl), rna_lddt=f(lddt), rna_tm=f(tm),
                             tier=fold_tier(f(pl))))
    return rows


def _write(rows, name):
    if not rows:
        return
    with open(os.path.join(OUTDIR, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    L = []
    # A
    pl, missing = placement_validation()
    _write(pl, "validation_placement.tsv")
    L.append(f"A  interface tier vs placement class (RNA chains; {missing} complexes without model files)")
    for period in ("pre", "post"):
        d = [r for r in pl if r["period"] == period and r["tier"]]
        corr = [r for r in d if r["placement"] == "correct_interface"]
        fail = [r for r in d if r["placement"] != "correct_interface"]
        auc = _auc([ORD[r["tier"]] for r in corr], [ORD[r["tier"]] for r in fail])
        L.append(f"  {period}: n={len(d)}; correct-interface share by tier: " + ", ".join(
            f"{t} {100 * np.mean([r['placement'] == 'correct_interface' for r in d if r['tier'] == t]):.0f}% "
            f"(n={sum(r['tier'] == t for r in d)})" for t in TIERS) + f"; AUC(tier, correct)={auc:.2f}")
        for cls in FAILURE_ORDER:
            sub = [r for r in fail if r["placement"] == cls]
            if sub:
                c = Counter(r["tier"] for r in sub)
                L.append(f"      {cls:38s} n={len(sub):3d}  good {c['good']:3d}  mid {c['mid']:3d}  bad {c['bad']:3d}")
    # B
    sd = seed_validation()
    _write(sd, "validation_seed_samples.tsv")
    d = [r for r in sd if r["tier"]]
    acc = [r for r in d if r["dockq"] >= ACCEPTABLE]
    rej = [r for r in d if r["dockq"] < ACCEPTABLE]
    L.append(f"\nB  interface tier vs DockQ over all sampled interfaces (n={len(d)} interface-samples, "
             f"{len({r['pdb'] for r in d})} targets)")
    L.append("  acceptable (DockQ>=0.23) share by tier: " + ", ".join(
        f"{t} {100 * np.mean([r['dockq'] >= ACCEPTABLE for r in d if r['tier'] == t]):.0f}% "
        f"(n={sum(r['tier'] == t for r in d)}, median DockQ "
        f"{np.median([r['dockq'] for r in d if r['tier'] == t]) if any(r['tier'] == t for r in d) else np.nan:.2f})"
        for t in TIERS) + f"; AUC(tier, acceptable)={_auc([ORD[r['tier']] for r in acc], [ORD[r['tier']] for r in rej]):.2f}")
    # per interface: tier stability across the 15 samples
    by_if = {}
    for r in d:
        by_if.setdefault((r["pdb"], r["ref_pair"]), []).append(r["tier"])
    stable = np.mean([len(set(v)) == 1 for v in by_if.values()])
    L.append(f"  same tier in all samples of an interface: {100 * stable:.0f}% of {len(by_if)} interfaces")
    # C
    mc = multichain_validation()
    _write(mc, "validation_multichain_fold.tsv")
    L.append(f"\nC  fold tier vs accuracy in the multi-chain RNA set (n={sum(1 for r in mc if r['tier'])} complexes)")
    for period in ("pre", "post", None):
        d = [r for r in mc if r["tier"] and (period is None or r["period"] == period)]
        good = [r for r in d if np.isfinite(r["rna_lddt"]) and r["rna_lddt"] > GOOD_RNA_LDDT]
        bad = [r for r in d if np.isfinite(r["rna_lddt"]) and r["rna_lddt"] <= GOOD_RNA_LDDT]
        L.append(f"  {period or 'all':4s}: " + ", ".join(
            f"{t} n={sum(r['tier'] == t for r in d)} median LDDT_r "
            f"{np.nanmedian([r['rna_lddt'] for r in d if r['tier'] == t]) if any(r['tier'] == t for r in d) else np.nan:.2f}"
            for t in TIERS) + f"; AUC(tier, LDDT_r>0.85)={_auc([ORD[r['tier']] for r in good], [ORD[r['tier']] for r in bad]):.2f}")
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "validation_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
