"""
Mean intra-chain PAE of every chain of an AF3 prediction (from *_full_data_<k>.json), cached.

The fold side of the framework can use an RNA-specific error estimate: the mean predicted aligned
error between the tokens of the RNA chain itself, instead of the global PAE of the whole complex
(which is dominated by the protein in protein-RNA complexes).

    python -m decision_tree.chain_pae <folder> [<folder> ...]   # fills the cache

Cache: results/plots/all_chains/decision_tree/chain_pae_cache.tsv (stem, chain, mean_pae, n_tokens).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "chain_pae_cache.tsv")


def intra_chain_pae(full_data: str) -> dict:
    """{chain: (mean intra-chain PAE, n tokens)}; empty on unreadable files."""
    try:
        d = json.load(open(full_data))
    except (json.JSONDecodeError, OSError):
        return {}
    ids = np.array(d.get("token_chain_ids", []))
    pae = d.get("pae")
    if not len(ids) or pae is None:
        return {}
    pae = np.asarray(pae, dtype=np.float32)
    out = {}
    for ch in dict.fromkeys(ids.tolist()):
        m = ids == ch
        sub = pae[np.ix_(m, m)]
        out[ch] = (float(sub.mean()), int(m.sum()))
    return out


def stem_of(path: str) -> str:
    return os.path.basename(path).split("_full_data_")[0].lower()


def load_cache() -> dict:
    out = {}
    if os.path.exists(CACHE):
        for r in csv.DictReader(open(CACHE), delimiter="\t"):
            out.setdefault(r["stem"], {})[r["chain"]] = (float(r["mean_pae"]), int(r["n_tokens"]))
    return out


def _job(path):
    return stem_of(path), intra_chain_pae(path)


def build(folders, workers=4):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    have = load_cache()
    todo = [p for f in folders for p in glob.glob(os.path.join(f, "*_full_data_0.json")) if stem_of(p) not in have]
    new = not os.path.exists(CACHE)
    with open(CACHE, "a", newline="") as fh, ProcessPoolExecutor(max_workers=workers) as ex:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(["stem", "chain", "mean_pae", "n_tokens"])
        for i, (stem, res) in enumerate(ex.map(_job, todo, chunksize=1), 1):
            for ch, (m, n) in res.items():
                w.writerow([stem, ch, f"{m:.3f}", n])
            fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"cache: {CACHE} (+{len(todo)} predictions)")


if __name__ == "__main__":
    build(sys.argv[1:])
