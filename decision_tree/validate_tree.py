"""
Validation of the decision tree (decision_tree/tree.py) and selection of its node order.

Pre-prediction answers come from the databases (RNAFamily, RNA_msa_size, RNA_Pseudoknots of the
AF3 model, RBP_PDB_Templates, InterPro domain_names); interface confidences are read from the raw
AF3 output with the released code.

  FOLD       single-chain RNA and protein-RNA complexes: leaf vs well-folded RNA (LDDT_r > 0.85;
             complex-level values, longest RNA chain).
  INTERFACE  every protein-RNA RNA chain: leaf vs placement class (correct interface) -- labels
             that played no part in building the tree.

Node order: the drawn tree (ORDER_DRAWN) and every order that asks AF3's confidence first are
scored by the AUC of the leaf order. The order is SELECTED ON PRE-CUTOFF TARGETS ONLY and the
post-cutoff targets are the held-out test. Writes results/plots/all_chains/decision_tree/
validation_tree_*.tsv and validation_tree_summary.txt.

Run from the repo root (RNABench env):  python -m decision_tree.validate_tree
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

from decision_tree.af3_output import read_prediction
from decision_tree.tree import (FOLD_LEAVES, INTERFACE_LEAVES, ORDER, ORDER_DRAWN, fold_answers,
                                interface_answers, label, walk)
from decision_tree.validate import OUTDIR, PP, _auc, db_path, _model_index
from evaluation.placement_labels import best_per_rna_chain

RNA_ONLY = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre", "a2021_b2021_rna_only")
GOOD_LDDT = 0.85
NODES = {"fold": ("confident", "family", "msa", "canonical"),
         "interface": ("confident", "domain", "template", "wellfolded")}
LEAVES = {"fold": FOLD_LEAVES, "interface": INTERFACE_LEAVES}


def _j(v, default=None):
    try:
        return json.loads(v) if isinstance(v, str) and v.strip().startswith(("[", "{")) else (v if v is not None else default)
    except json.JSONDecodeError:
        return default


def _first(v):
    v = _j(v)
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _family(v):
    v = _j(v)
    vals = [str(x).strip() for x in (v if isinstance(v, list) else [v]) if x is not None]
    return any(x and x.lower() != "none" for x in vals) if vals else None


def _msa(v):
    v = _j(v, {})
    return max(float(x) for x in v.values()) if isinstance(v, dict) and v else None


def _pk(v):
    s = str(v or "").strip()
    return None if s == "" else s not in ("[]", "None")


def _tpl(v):
    v = _j(v, [])
    return bool(v) if isinstance(v, list) else None


# --------------------------------------------------------------------------- #
def fold_rows():
    rows = []
    sets = [("single-chain RNA", RNA_ONLY, "pre", "b2021_rna.db", "pred_rna_rna", "exp_rna_rna"),
            ("single-chain RNA", RNA_ONLY, "post", "a2021_rna.db", "pred_rna_rna", "exp_rna_rna"),
            ("protein-RNA", PP, "pre", "b2021.db", "pred_protein_rna", "exp_protein_rna"),
            ("protein-RNA", PP, "post", "a2021.db", "pred_protein_rna", "exp_protein_rna")]
    for system, folder, period, db, pt, et in sets:
        con = sqlite3.connect(db_path(folder, db))
        q = (f"SELECT p.exp_db_id, e.RNAFamily, p.RNA_msa_size, p.RNA_Pseudoknots, p.af3_rna_pLDDT_avg, "
             f"p.af3_global_pae_avg, p.RNA_LDDT FROM {pt} p JOIN {et} e "
             f"ON upper(trim(p.exp_db_id)) = upper(trim(e.PDBId))")
        for eid, fam, msa, pk, pl, gpae, lddt in con.execute(q):
            lddt = _first(lddt)
            if lddt is None:
                continue
            a = fold_answers(conserved_family=_family(fam), msa_depth=_msa(msa), pseudoknot=_pk(pk),
                             rna_plddt=_first(pl), global_pae=_first(gpae))
            rows.append(dict(system=system, period=period, pdb=eid, outcome=lddt > GOOD_LDDT, rna_lddt=lddt, **a))
    return rows


def interface_rows():
    idx = _model_index()
    rows = []
    for period, db in (("pre", "b2021.db"), ("post", "a2021.db")):
        con = sqlite3.connect(PP[db])
        q = ("SELECT p.exp_db_id, p.FileName, p.RNA_interface_placement, p.RBP_PDB_Templates, e.domain_names "
             "FROM pred_protein_rna p JOIN exp_protein_rna e ON upper(trim(p.exp_db_id)) = upper(trim(e.PDBId))")
        for eid, fn, plc, tpl, doms in con.execute(q):
            plc = best_per_rna_chain(plc)          # one class per RNA chain: best over its interfaces
            cif = idx.get(os.path.basename(str(fn)).lower())
            if not plc or cif is None:
                continue
            pred = read_prediction(cif)
            domains = [d for d in str(doms or "").split(";") if d.strip() not in ("", "-")] or None
            for c in pred["rna_chains"]:
                cls = plc.get(c["chain"])
                if not cls:
                    continue
                a = interface_answers(protein_ptms=[c["partner_ptm"]] if c["partner_ptm"] is not None else None,
                                      rna_plddt=c["rna_plddt"], rbp_template=_tpl(tpl), domains=domains,
                                      rna_iptm=c["rna_iptm"], pair_pae=c["pae_min"])
                rows.append(dict(system="protein-RNA", period=period, pdb=eid, chain=c["chain"],
                                 outcome=cls == "correct_interface", placement=cls, **a))
    return rows


def leaf_rank(row, order, leaves):
    """Rank of the leaf; a range (unknown answer) counts as its midpoint."""
    reach = walk(row, order, leaves)
    r = [leaves.index(l) for l in reach]
    return (min(r) + max(r)) / 2


def auc_of(rows, order, leaves):
    pos = [leaf_rank(r, order, leaves) for r in rows if r["outcome"]]
    neg = [leaf_rank(r, order, leaves) for r in rows if not r["outcome"]]
    return _auc(pos, neg)


def node_effects(L, rows, side):
    for period in ("pre", "post"):
        d = [r for r in rows if r["period"] == period]
        L.append(f"  {period}: " + "; ".join(
            f"{n} yes {100 * np.mean([r['outcome'] for r in d if r[n] is True]):.0f}% / no "
            f"{100 * np.mean([r['outcome'] for r in d if r[n] is False]) if any(r[n] is False for r in d) else float('nan'):.0f}%"
            for n in NODES[side]))


def order_scan(L, rows, side):
    leaves = LEAVES[side]
    pre = [r for r in rows if r["period"] == "pre"]
    post = [r for r in rows if r["period"] == "post"]
    cands = [ORDER_DRAWN[side]] + [("confident",) + p for p in itertools.permutations(NODES[side][1:])]
    res = [(o, auc_of(pre, o, leaves), auc_of(post, o, leaves)) for o in cands]
    best = max(res[1:], key=lambda t: t[1])
    L.append(f"  drawn order {' > '.join(ORDER_DRAWN[side])}: AUC pre {res[0][1]:.2f}, post {res[0][2]:.2f}")
    for o, a, b in res[1:]:
        L.append(f"  {' > '.join(o):45s} AUC pre {a:.2f}, post {b:.2f}{'   <- selected on pre' if o == best[0] else ''}")
    return best[0]


def leaf_table(L, rows, side, order):
    leaves = LEAVES[side]
    for key in sorted({(r["system"], r["period"]) for r in rows}, key=lambda k: (k[0], k[1] != "pre")):
        d = [r for r in rows if (r["system"], r["period"]) == key]
        parts = []
        for lf in leaves:
            s = [r["outcome"] for r in d if label(walk(r, order, leaves)) == lf]
            if s:
                parts.append(f"{lf} {100 * np.mean(s):.0f}% (n={len(s)})")
        L.append(f"  {key[0]}, {key[1]} (n={len(d)}): " + ", ".join(parts)
                 + f"; AUC {auc_of(d, order, leaves):.2f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fr, ir = fold_rows(), interface_rows()
    L = []
    chosen = {}
    for side, rows, what in (("fold", fr, "well-folded RNA (LDDT_r > 0.85)"),
                             ("interface", ir, "correct interface (placement class)")):
        L.append(f"{side.upper()} side, outcome = {what}")
        L.append(" per question, outcome share when answered yes / no:")
        node_effects(L, rows, side)
        L.append(" node orders (selected on pre-cutoff only, post-cutoff = held-out):")
        chosen[side] = order_scan(L, rows, side)
        L.append(f" leaves with the selected order {' > '.join(chosen[side])}:")
        leaf_table(L, rows, side, chosen[side])
        if tuple(chosen[side]) != tuple(ORDER[side]):
            L.append(f"  NOTE: tree.ORDER['{side}'] = {ORDER[side]} differs from the selected order")
        L.append("")
    for name, rows in (("validation_tree_fold.tsv", fr), ("validation_tree_interface.tsv", ir)):
        with open(os.path.join(OUTDIR, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print("\n".join(L))
    with open(os.path.join(OUTDIR, "validation_tree_summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
