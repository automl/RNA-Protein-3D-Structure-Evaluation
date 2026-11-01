"""
Which feature groups predict how a protein-RNA interface fails?

Input: results/interface_features_{period}.tsv (build_interface_features.py).
Targets (one vs rest, per interface):
  correct    correct interface (DockQ >= 0.23)
  csir       correct site, incorrect RNA
  distal     distal or shifted site (RNA placed away from its native site)
  badsite    incorrect binding site (the protein's binding region mispredicted)
Rotated RNAs (18 post-, 8 pre-cutoff interfaces) stay in the 'rest' of every target.

Model: HistGradientBoostingClassifier (native missing values, balanced classes); out-of-fold
predictions from StratifiedGroupKFold grouped by PDB entry (5 folds x REPEATS shuffles); every
complex has equal total weight (sample weight 1 / its number of interfaces) in fitting and in the
AUC. Group importance: exact Shapley values over the six feature groups of the out-of-fold AUC,
v(S) = AUC with the groups in S, v({}) = 0.5, so each target's Shapley values sum to its full-model
AUC - 0.5 (the order-independent partitioning of the regression analysis, for AUC instead of R2).

Writes to results/: shapley_{period}.tsv, coalitions_{period}.tsv and the heatmap
interface_failure_feature_groups.{pdf,png} (post-cutoff; pre-cutoff as replication panel).
Run in the RNABench env:  python regression_analysis/interface/interface_failure_feature_groups.py
(--plot-only redraws the figure from the saved tables)
"""

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GROUPS = [("af3", "AF3 confidence"), ("exp", "Experimental and\nenvironmental"),
          ("seq", "RNA sequence and\nstructure"), ("evo", "Evolutionary"), ("tmpl", "Templates"),
          ("comp", "Composition")]
TARGETS = [("correct", "Correct interface", ["correct interface"]),
           ("csir", "Correct site, incorrect RNA", ["right protein site, wrong RNA pose"]),
           ("distal", "Distal or shifted site", ["different region", "shifted nearby"]),
           ("badsite", "Incorrect binding site", ["protein binding region mispredicted"])]
N_SPLITS, REPEATS, JOBS = 5, 3, 16


def _model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def coalition_auc(X, y, w, groups_col, cols, seed):
    """Out-of-fold weighted AUC of a model on `cols` (one repeat)."""
    if not cols:
        return 0.5
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups_col):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _model().fit(X.iloc[tr][cols], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
    ok = np.isfinite(oof)
    return roc_auc_score(y[ok], oof[ok], sample_weight=w[ok])


def shapley(values, keys):
    """Exact Shapley values from {frozenset(groups): v}."""
    n = len(keys)
    out = {}
    for g in keys:
        s = 0.0
        others = [k for k in keys if k != g]
        for r in range(n):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                s += wgt * (values[frozenset(S) | {g}] - values[frozenset(S)])
        out[g] = s
    return out


def run(period):
    df = pd.read_csv(os.path.join(RES, f"interface_features_{period}.tsv"), sep="\t")
    keys = [g for g, _ in GROUPS if any(c.startswith(g + "__") for c in df.columns)]
    cols = {g: [c for c in df.columns if c.startswith(g + "__")] for g in keys}
    X = df[[c for g in keys for c in cols[g]]].astype(float)
    w = (1.0 / df["n_interfaces_of_complex"]).to_numpy()
    grp = df["pdb"].to_numpy()
    coalitions = [frozenset(S) for r in range(len(keys) + 1) for S in itertools.combinations(keys, r)]
    jobs, meta = [], []
    for tkey, _tlab, classes in TARGETS:
        y = df["placement"].isin(classes).astype(int).to_numpy()
        for S in coalitions:
            for rep in range(REPEATS):
                jobs.append(delayed(coalition_auc)(X, y, w, grp, [c for g in S for c in cols[g]], rep))
                meta.append((tkey, S, rep))
    aucs = Parallel(n_jobs=JOBS, verbose=0)(jobs)
    tab = pd.DataFrame([dict(target=t, coalition="+".join(sorted(S)) or "none", rep=r, auc=a)
                        for (t, S, r), a in zip(meta, aucs)])
    tab.to_csv(os.path.join(RES, f"coalitions_{period}.tsv"), sep="\t", index=False)
    rows = []
    for tkey, tlab, classes in TARGETS:
        n_pos = int(df["placement"].isin(classes).sum())
        per_rep = []
        for rep in range(REPEATS):
            v = {S: a for (t, S, r), a in zip(meta, aucs) if t == tkey and r == rep}
            per_rep.append(shapley(v, keys))
        full = [a for (t, S, r), a in zip(meta, aucs) if t == tkey and S == frozenset(keys)]
        for g in keys:
            vals = [p[g] for p in per_rep]
            single = [a for (t, S, r), a in zip(meta, aucs) if t == tkey and S == frozenset({g})]
            rows.append(dict(period=period, target=tkey, target_label=tlab, n_positive=n_pos,
                             n_interfaces=len(df), full_auc=np.mean(full), full_auc_sd=np.std(full),
                             group=g, shapley=np.mean(vals), shapley_sd=np.std(vals),
                             auc_alone=np.mean(single)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, f"shapley_{period}.tsv"), sep="\t", index=False)
    return out


def figure(post, pre):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "publication_figures"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.apply("plain")
    keys = [g for g, _ in GROUPS]
    tl = [t for t, _l, _c in TARGETS]
    mats = []
    for d in (post, pre):
        mats.append(np.array([[d[(d.target == t) & (d.group == g)].shapley.mean() if g in set(d.group) else np.nan
                               for g in keys] for t in tl]))
    vmax = max(0.15, np.nanmax(np.concatenate([m.ravel() for m in mats])))
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH_2COL, 2.8), gridspec_kw=dict(wspace=0.32))
    for k, (ax, d, mat, title) in enumerate(zip(axes, (post, pre), mats, ("Post-cutoff", "Pre-cutoff"))):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                            color="white" if mat[i, j] > 0.6 * vmax else "black")
        full = [d[d.target == t].full_auc.iloc[0] for t in tl]
        npos = [d[d.target == t].n_positive.iloc[0] for t in tl]
        ax.set_yticks(range(len(tl)))
        if k == 0:
            ax.set_yticklabels([f"{lab}\nn={n}, AUC {a:.2f}" for (_t, lab, _c), n, a in zip(TARGETS, npos, full)],
                               fontsize=6.5)
        else:
            ax.set_yticklabels([f"n={n}\nAUC {a:.2f}" for n, a in zip(npos, full)], fontsize=6.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab for _g, lab in GROUPS], fontsize=6.5, rotation=40, ha="right")
        ax.set_title(title, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Shapley share of AUC above 0.5", fontsize=6.5)
    base = os.path.join(RES, "interface_failure_feature_groups")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print("Saved:", base + ".pdf")


def main():
    os.makedirs(RES, exist_ok=True)
    if "--plot-only" in sys.argv:
        post, pre = (pd.read_csv(os.path.join(RES, f"shapley_{p}.tsv"), sep="\t") for p in ("post", "pre"))
        figure(post, pre)
        return
    post = run("post")
    pre = run("pre")
    for d in (post, pre):
        print(d.pivot_table(index="target", columns="group", values="shapley").round(3).assign(
            full_auc=d.groupby("target").full_auc.first().round(3)).to_string())
    figure(post, pre)


if __name__ == "__main__":
    main()
