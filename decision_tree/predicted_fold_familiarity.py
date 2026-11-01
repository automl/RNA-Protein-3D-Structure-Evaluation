"""
Fold familiarity from the AF3 model, available without an experimental structure.

For every protein-RNA prediction, the longest RNA chain of the AF3 model is aligned with US-align (-mol RNA)
against the pre-cutoff RNA chain library used for the paper's structural-similarity analysis
(evaluation/structural_similarity_to_pre.py), and the best TM-score (normalised by the query) is kept.
Chains shorter than MIN_NT_TM have no TM-score (None), as for the experimental comparison.

Writes results/plots/all_chains/decision_tree/probability/predicted_fold_tm.json  {set: {pdb: [tm, n_nt]}}
Run (RNABench env):  python -m decision_tree.predicted_fold_familiarity [post] [2026]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_tree.develop_probability import BASE_OUT, inputs
from decision_tree.validate_2026 import PRE_CHAINS, USALIGN, _pre_chain_list
from evaluation.structural_similarity_to_pre import MIN_LEN_RATIO, MIN_NT_TM, rna_chains

OUTFILE = os.path.join(BASE_OUT, "predicted_fold_tm.json")
WORK = os.path.join(BASE_OUT, "predicted_chains")


def best_tm(cif, tag, pre):
    """(best TM-score of the model's longest RNA chain against the pre-cutoff library, its length)"""
    chains = rna_chains(cif, WORK, tag)
    if not chains:
        return None, 0
    name, _, n = max(chains, key=lambda t: t[2])
    if n < MIN_NT_TM:
        return None, n
    cands = [f for f, ln in pre if ln >= MIN_LEN_RATIO * n]
    lst = os.path.join(WORK, name + ".list")
    open(lst, "w").write("\n".join(cands) + "\n")
    res = subprocess.run([USALIGN, os.path.join(WORK, name), "-dir2", PRE_CHAINS + "/", lst, "-mol", "RNA",
                          "-outfmt", "2"], capture_output=True, text=True)
    tms = [float(l.split("\t")[2]) for l in res.stdout.splitlines() if l.strip() and not l.startswith("#")]
    return (max(tms) if tms else 0.0), n


def run(sets=("post", "2026"), limit=None):
    os.makedirs(WORK, exist_ok=True)
    out = json.load(open(OUTFILE)) if os.path.exists(OUTFILE) else {}
    fn, m26 = inputs()
    pre = _pre_chain_list()
    for name in sets:
        models = fn if name == "post" else m26
        done = out.setdefault(name, {})
        todo = [p for p, cif in sorted(models.items()) if cif and p not in done][:limit]
        for k, pid in enumerate(todo):
            done[pid] = list(best_tm(models[pid], f"{name}_{pid.lower()}_model", pre))
            if k % 20 == 0:
                json.dump(out, open(OUTFILE, "w"))
                print(f"  {name}: {len(done)}/{len(models)}", flush=True)
        json.dump(out, open(OUTFILE, "w"))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a in ("post", "2026")] or ["post", "2026"]
    run(args)
