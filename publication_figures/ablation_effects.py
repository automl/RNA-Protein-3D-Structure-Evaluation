"""Effect of removing each input component, from the 16-way factorial.

For one factor X the other three factors give 8 background combinations, and each target was
run under all of them, so X is estimated WITHIN target: for every background the run with X
present is paired with the otherwise identical run with X removed, and the 8 paired
differences are averaged into one value per target. That removes between-target variation,
which is far larger than any ablation effect, and it uses all 16 runs rather than 2.

Only targets that carry the component are included (`ablation_data.carriers`), since removing
an input a target never had cannot change its prediction.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from publication_figures.ablation_data import (FACTORS, METRICS, availability,
                                               carriers, load_factorial)

HIGHER_IS_BETTER = {m: True for m in METRICS}
HIGHER_IS_BETTER["Complex_RMSD"] = False


def paired_differences(fac, factor, metric):
    """-> Series indexed by target: mean over backgrounds of (present - removed)."""
    others = [f for f in FACTORS if f != factor]
    keep = ["exp_db_id", factor, metric] + others
    d = fac[keep].dropna(subset=[metric])
    wide = d.pivot_table(index=["exp_db_id"] + others, columns=factor,
                         values=metric, aggfunc="mean")
    if 0 not in wide.columns or 1 not in wide.columns:
        return pd.Series(dtype=float)
    delta = (wide[1] - wide[0]).dropna()
    if not HIGHER_IS_BETTER[metric]:
        delta = -delta                      # orient so + always means "component helps"
    return delta.groupby("exp_db_id").mean()


def summarise(delta, n_boot=2000, seed=0):
    v = delta.to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return dict(n=v.size, mean=np.nan, lo=np.nan, hi=np.nan, frac_pos=np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return dict(n=v.size, mean=float(v.mean()),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                frac_pos=float((v > 0).mean()))


def factor_effects(fac, av, metrics=("Complex_TM", "RNA_TM", "Complex_LDDT_allAtom",
                                     "RNA_LDDT_allAtom", "Dockq", "Complex_RMSD")):
    rows = []
    cls = av.set_index("exp_db_id")["cls"].to_dict()
    for factor in FACTORS:
        have = carriers(av, factor)
        for metric in metrics:
            delta = paired_differences(fac, factor, metric)
            delta = delta[delta.index.isin(have)]
            for group in ("1v1", "multi", "all"):
                d = delta if group == "all" else delta[[cls.get(t) == group for t in delta.index]]
                rows.append(dict(factor=factor, metric=metric, group=group, **summarise(d)))
    return pd.DataFrame(rows)


def all_four_effect(fac, av, metrics=("Complex_TM", "RNA_TM", "Dockq")):
    """Everything present vs everything removed, on targets carrying all four."""
    have = set(av.loc[av.all_four, "exp_db_id"])
    cls = av.set_index("exp_db_id")["cls"].to_dict()
    full = fac[(fac.rMSA == 1) & (fac.pMSA == 1) & (fac.pTemp == 1) & (fac.cTemp == 1)]
    none = fac[(fac.rMSA == 0) & (fac.pMSA == 0) & (fac.pTemp == 0) & (fac.cTemp == 0)]
    rows = []
    for metric in metrics:
        a = full.set_index("exp_db_id")[metric]
        b = none.set_index("exp_db_id")[metric]
        common = sorted((set(a.index) & set(b.index)) & have)
        delta = (a.loc[common] - b.loc[common]).dropna()
        if not HIGHER_IS_BETTER[metric]:
            delta = -delta
        for group in ("1v1", "multi", "all"):
            d = delta if group == "all" else delta[[cls.get(t) == group for t in delta.index]]
            rows.append(dict(metric=metric, group=group, **summarise(d)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fac = load_factorial(verbose=False)
    av = availability()
    eff = factor_effects(fac, av)
    pd.set_option("display.width", 200)
    print("=== single-factor effect (paired within target; + = component helps) ===")
    for metric in ("Complex_TM", "RNA_TM", "Dockq"):
        print(f"\n--- {metric} ---")
        t = eff[(eff.metric == metric)].pivot_table(index="factor", columns="group",
                                                    values=["mean", "n"])
        print(t.round(4))
    print("\n=== all four present vs all four removed (carriers of all four) ===")
    print(all_four_effect(fac, av).round(4).to_string(index=False))
