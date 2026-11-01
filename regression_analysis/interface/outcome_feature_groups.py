"""
Shapley shares of five descriptor groups for the two outcomes of the graphical abstract
(post-cutoff targets only: AF3 has seen the pre-cutoff structures during training).

  fold       RNA well folded (RNA_LDDT > 0.85, the paper's class rule), one row per complex
  interface  correct protein-RNA interface (DockQ >= 0.23), one row per interface, every complex
             weighted equally
Groups (as in interface_failure_feature_groups.py, with composition merged into 'structural'):
  experimental, structural (RNA sequence / structure of the target and composition), evolutionary,
  templates, AF3 confidence.
Same classifier and grouped cross-validation as interface_failure_feature_groups.py; exact Shapley
split of the out-of-fold AUC above 0.5.

Writes results/outcome_shapley_post.tsv. Run in the RNABench env:
    python regression_analysis/interface/outcome_feature_groups.py
"""

import itertools
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from interface_failure_feature_groups import JOBS, REPEATS, RES, coalition_auc, shapley

REPO = os.path.dirname(os.path.dirname(HERE))
DB = os.path.join(REPO, "data", "compareDatabases", "post_pre", "a2021_b2021_protein_rna", "a2021.db")
GROUPS5 = [("exp", "Experimental", ["exp"]), ("struct", "Structural", ["seq", "comp"]),
           ("evo", "Evolutionary", ["evo"]), ("tmpl", "Templates", ["tmpl"]),
           ("af3", "AF3 confidence", ["af3"])]


def main():
    df = pd.read_csv(os.path.join(RES, "interface_features_post.tsv"), sep="\t")
    cols = {g: [c for c in df.columns if c.split("__")[0] in parts and "__" in c] for g, _l, parts in GROUPS5}
    keys = [g for g, _l, _p in GROUPS5]
    con = sqlite3.connect(DB)
    lddt = {str(a).strip().upper(): b for a, b in con.execute("SELECT exp_db_id, RNA_LDDT FROM pred_protein_rna")}
    con.close()
    outcomes = {}
    # interface: one row per interface, complexes weighted equally
    outcomes["interface"] = (df, (df["placement"] == "correct interface").astype(int).to_numpy(),
                             (1.0 / df["n_interfaces_of_complex"]).to_numpy())
    # fold: one row per complex
    cx = df.drop_duplicates("pdb").copy()
    cx["rna_lddt"] = pd.to_numeric(cx["pdb"].map(lambda p: lddt.get(p)), errors="coerce")
    cx = cx[cx["rna_lddt"].notna()].reset_index(drop=True)
    outcomes["fold"] = (cx, (cx["rna_lddt"] > 0.85).astype(int).to_numpy(), np.ones(len(cx)))
    coalitions = [frozenset(S) for r in range(len(keys) + 1) for S in itertools.combinations(keys, r)]
    rows = []
    for name, (d, y, w) in outcomes.items():
        X = d[[c for g in keys for c in cols[g]]].astype(float)
        meta = [(S, rep) for S in coalitions for rep in range(REPEATS)]
        aucs = Parallel(n_jobs=JOBS)(delayed(coalition_auc)(X, y, w, d["pdb"].to_numpy(),
                                                            [c for g in S for c in cols[g]], rep) for S, rep in meta)
        per_rep = [shapley({S: a for (S, r), a in zip(meta, aucs) if r == rep}, keys) for rep in range(REPEATS)]
        full = [a for (S, r), a in zip(meta, aucs) if S == frozenset(keys)]
        for g, lab, _p in GROUPS5:
            vals = [p[g] for p in per_rep]
            rows.append(dict(outcome=name, group=g, label=lab, shapley=np.mean(vals), shapley_sd=np.std(vals),
                             full_auc=np.mean(full), n_rows=len(d), n_positive=int(y.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "outcome_shapley_post.tsv"), sep="\t", index=False)
    print(out.pivot_table(index="label", columns="outcome", values="shapley").round(3).to_string())
    print(out.groupby("outcome")[["full_auc", "n_rows", "n_positive"]].first().round(3).to_string())


if __name__ == "__main__":
    main()
