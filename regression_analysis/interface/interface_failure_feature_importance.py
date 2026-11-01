"""
Per-feature importance for the interface-failure targets (within-group breakdown of
interface_failure_feature_groups.py).

For each target (correct; correct site, incorrect RNA; distal or shifted; incorrect binding site)
the full model (all six groups, same classifier, weighting and grouped cross-validation) is fitted on
the training folds; in each held-out fold every feature is shuffled N_PERM times and the drop in the
weighted AUC is recorded (out-of-fold permutation importance, mean over folds and repeats).
Direction: Spearman rho between the feature and the target over complexes (each complex's share of
interfaces in the class), + = higher values go with the class.
Correlated features share their importance (shuffling one leaves its correlates in place), so a
feature's importance is a lower bound; read it within its group (shapley_{period}.tsv).

Writes results/feature_importance_{post,pre}.tsv and results/interface_failure_feature_importance.{pdf,png}
(post-cutoff, top features per target) and the heatmap interface_failure_feature_heatmap.{pdf,png}
(all features x targets, post | pre). Run in the RNABench env:
    python regression_analysis/interface/interface_failure_feature_importance.py [--plot-only]
"""

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from interface_failure_feature_groups import GROUPS, JOBS, N_SPLITS, REPEATS, RES, TARGETS, _model

N_PERM = 5
GROUP_COLOR = {"af3": "#4E79A7", "exp": "#B07AA1", "seq": "#59A14F", "evo": "#F28E2B", "tmpl": "#9C755F",
               "comp": "#76B7B2"}


def _fold(X, y, w, tr, te, seed):
    m = _model().fit(X.iloc[tr], y[tr], sample_weight=w[tr])
    Xte, yte, wte = X.iloc[te], y[te], w[te]
    if len(np.unique(yte)) < 2:
        return None
    base = roc_auc_score(yte, m.predict_proba(Xte)[:, 1], sample_weight=wte)
    rng = np.random.default_rng(seed)
    drops = {}
    for c in X.columns:
        d = []
        for _ in range(N_PERM):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            d.append(base - roc_auc_score(yte, m.predict_proba(Xp)[:, 1], sample_weight=wte))
        drops[c] = float(np.mean(d))
    return base, drops, len(te)


def run(period):
    df = pd.read_csv(os.path.join(RES, f"interface_features_{period}.tsv"), sep="\t")
    feats = [c for c in df.columns if "__" in c and c.split("__")[0] in dict(GROUPS)]
    X = df[feats].astype(float)
    w = (1.0 / df["n_interfaces_of_complex"]).to_numpy()
    grp = df["pdb"].to_numpy()
    rows = []
    for tkey, tlab, classes in TARGETS:
        y = df["placement"].isin(classes).astype(int).to_numpy()
        jobs = []
        for rep in range(REPEATS):
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep)
            for k, (tr, te) in enumerate(cv.split(X, y, grp)):
                jobs.append(delayed(_fold)(X, y, w, tr, te, 1000 * rep + k))
        res = [r for r in Parallel(n_jobs=JOBS)(jobs) if r is not None]
        tot = sum(n for _b, _d, n in res)
        # direction at complex level: feature vs the complex's share of interfaces in the class
        share = df.assign(y=y).groupby("pdb").agg(share=("y", "mean"))
        cx = df.drop_duplicates("pdb").set_index("pdb")[feats].join(share)
        for c in feats:
            ok = cx[[c, "share"]].dropna()
            rho = spearmanr(ok[c], ok["share"])[0] if len(ok) >= 10 and ok[c].nunique() > 1 else np.nan
            vals = [d[c] for _b, d, _n in res]
            rows.append(dict(period=period, target=tkey, target_label=tlab, feature=c.split("__", 1)[1],
                             group=c.split("__")[0], importance=sum(d[c] * n for _b, d, n in res) / tot,
                             importance_sd=float(np.std(vals)), direction_rho=rho,
                             coverage=float(df[c].notna().mean()),
                             fold_auc=float(np.mean([b for b, _d, _n in res]))))
    out = pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])
    out.to_csv(os.path.join(RES, f"feature_importance_{period}.tsv"), sep="\t", index=False)
    return out


def figure(d, top=10):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "publication_figures"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import figstyle
    figstyle.apply("plain")
    fig, axes = plt.subplots(1, 4, figsize=(figstyle.WIDTH_2COL, 3.4), gridspec_kw=dict(wspace=1.15))
    for ax, (tkey, tlab, _c) in zip(axes, TARGETS):
        s = d[d.target == tkey].nlargest(top, "importance").iloc[::-1]
        y = np.arange(len(s))
        ax.barh(y, s.importance, xerr=s.importance_sd, color=[GROUP_COLOR[g] for g in s.group],
                height=0.7, error_kw=dict(lw=0.5, capsize=1))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{f} ({'+' if r > 0 else '−' if r < 0 else '0'})"
                            for f, r in zip(s.feature, s.direction_rho)], fontsize=5.5)
        ax.set_title(tlab, fontsize=7)
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_xlabel("AUC drop when shuffled", fontsize=6)
        ax.axvline(0, color="0.5", lw=0.5)
    fig.legend([Patch(color=GROUP_COLOR[g]) for g, _ in GROUPS], [lab.replace("\n", " ") for _g, lab in GROUPS],
               loc="lower center", ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    base = os.path.join(RES, "interface_failure_feature_importance")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print("Saved:", base + ".pdf")


def heatmap(post, pre):
    """Signed importance (AUC drop x sign of the direction rho): features x targets, post | pre."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "publication_figures"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.apply("plain")
    order = [g for g, _ in GROUPS]
    feats = (post.drop_duplicates("feature")[["feature", "group"]]
             .assign(o=lambda x: x.group.map({g: i for i, g in enumerate(order)})).sort_values(["o", "feature"]))
    tk = [t for t, _l, _c in TARGETS]
    def mat(d):
        v = d.assign(s=d.importance * np.sign(d.direction_rho.fillna(0)))
        piv = v.pivot_table(index="feature", columns="target", values="s")
        return piv.reindex(index=feats.feature, columns=tk).to_numpy()
    mats = [mat(post), mat(pre)]
    vmax = np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in mats])), 99)
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH_2COL, 0.13 * len(feats) + 1.4),
                             gridspec_kw=dict(wspace=0.08))
    for ax, m, title in zip(axes, mats, ("Post-cutoff", "Pre-cutoff")):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]) and abs(m[i, j]) >= 0.005:
                    ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center", fontsize=4.6,
                            color="white" if abs(m[i, j]) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(tk)))
        ax.set_xticklabels([lab for _t, lab, _c in TARGETS], rotation=35, ha="right", fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats.feature if ax is axes[0] else [], fontsize=5)
        for k in np.where(feats.group.to_numpy()[1:] != feats.group.to_numpy()[:-1])[0]:
            ax.axhline(k + 0.5, color="black", lw=0.6)
    if True:
        ax0 = axes[0]
        bounds = feats.reset_index(drop=True).groupby("group").apply(lambda g: (g.index.min(), g.index.max()))
        for g, lab in GROUPS:
            if g in bounds:
                lo, hi = bounds[g]
                ax0.text(-0.62, (lo + hi) / 2, lab.replace("\n", " "), transform=ax0.get_yaxis_transform(),
                         rotation=90, ha="center", va="center", fontsize=5.5, color=GROUP_COLOR[g],
                         fontweight="bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("AUC drop when shuffled, signed by direction\n(+ higher values go with the class)", fontsize=6)
    base = os.path.join(RES, "interface_failure_feature_heatmap")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print("Saved:", base + ".pdf")


def main():
    if "--plot-only" in sys.argv:
        post = pd.read_csv(os.path.join(RES, "feature_importance_post.tsv"), sep="\t")
        pre = pd.read_csv(os.path.join(RES, "feature_importance_pre.tsv"), sep="\t")
    else:
        post = run("post")
        pre = run("pre")
    with pd.option_context("display.width", 200):
        for t, g in post.groupby("target"):
            print(f"\n{t}:")
            print(g.head(10)[["feature", "group", "importance", "importance_sd", "direction_rho", "coverage"]]
                  .round(3).to_string(index=False))
    figure(post)
    heatmap(post, pre)


if __name__ == "__main__":
    main()
