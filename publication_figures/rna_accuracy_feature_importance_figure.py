#!/usr/bin/env python3
"""
Which target/input features drive RNA accuracy? For four outcomes --
RNA_TM, RNA_LDDT (allAtom), RNA_RMSD, and the good/bad-RNA classification
(quadrants i+ii vs iii+iv of the decision framework, i.e. RNA_LDDT > 0.85;
see get_sample_audit_global_fold_interface_lists in evaluationMetrics.py and
pub_fig1_classification_framework.py) -- we rank features by

  * univariate Spearman rho (direction), and
  * cross-validated random-forest permutation importance (multivariate).

Structural descriptors present in BOTH exp_ and pred_ tables (interface Hbond/
vdW/area, secondary-structure counts, backbone geometry, electrostatics) are read
from the exp_ table: they must describe the TRUE target, not the AF3 prediction
(the _pred versions are outputs; using them is circular -- see bond_metrics_figure.py).
Sequence-derived descriptors (GC, entropy, MFE, ...) and input-availability features
(MSA depth, templates) come from the pred_ table. Base-pairing accuracy metrics (INF, DI,
mcc, f1, ...) are excluded as they are accuracy measures, and so are the interface
agreement scores AAmatch_score / RNAmatch_score (exact-residue-mapped overlap of predicted
and experimental contact residues) and RNAmotif_score: they score the prediction against
the experiment and are interface-accuracy outcomes, not determinants.

Two datasets:
  * protein-RNA complexes (database/rbpDatabase.db) -> ..._feature_importance.png
  * RNA-only / RNA-RNA (a2021_rna.db)               -> ..._feature_importance_rnaonly.png
    (no protein features/templates; interface features are RNA-RNA; n is small).

Run:  python publication_figures/rna_accuracy_feature_importance_figure.py
"""
import ast
import contextlib
import io
import os
import sqlite3
import sys
import warnings

warnings.filterwarnings("ignore")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import SimpleImputer

_DB_PR = os.path.join(_REPO_ROOT, "database", "rbpDatabase.db")
_DB_RO = os.path.join(_REPO_ROOT, "data", "compareDatabases", "post_pre",
                      "a2021_b2021_rna_only", "a2021_rna.db")
_LDDT_GOOD = 0.85  # RNA_LDDT threshold for good (i+ii) vs bad (iii+iv) RNA


# ---- parsers ---------------------------------------------------------------
def _f(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s.lower() in ("none", "", "null", "nan"):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        x = ast.literal_eval(s)
        if isinstance(x, (list, tuple)) and x:
            return float(x[0])
        if isinstance(x, dict) and x:
            return float(list(x.values())[0])
        return float(x)
    except (ValueError, SyntaxError):
        return np.nan


def _meanlist(v):
    try:
        x = ast.literal_eval(str(v))
        if isinstance(x, (list, tuple)):
            xs = [float(e) for e in x if isinstance(e, (int, float))]
            return float(np.mean(xs)) if xs else np.nan
        return float(x)
    except (ValueError, SyntaxError):
        return _f(v)


def _sumlist(v):
    try:
        x = ast.literal_eval(str(v))
        if isinstance(x, (list, tuple)):
            return float(sum(float(e) for e in x if isinstance(e, (int, float))))
        return float(x)
    except (ValueError, SyntaxError):
        return _f(v)


def _countnested(v):
    try:
        x = ast.literal_eval(str(v))
        return float(len(x)) if isinstance(x, (list, tuple)) else np.nan
    except (ValueError, SyntaxError):
        return np.nan


def _dictfrac(v, keys):
    try:
        x = ast.literal_eval(str(v))
        if isinstance(x, dict) and x:
            tot = sum(float(c) for c in x.values())
            num = sum(float(x.get(k, 0)) for k in keys)
            return num / tot if tot else np.nan
    except (ValueError, SyntaxError):
        return np.nan
    return np.nan


def _msadepth(v):
    try:
        x = ast.literal_eval(str(v))
        if isinstance(x, dict) and x:
            return float(np.mean([float(c) for c in x.values()]))
    except (ValueError, SyntaxError):
        return np.nan
    return np.nan


def _countjson(v):
    try:
        x = ast.literal_eval(str(v))
        return float(len(x)) if isinstance(x, (list, tuple)) else 0.0
    except (ValueError, SyntaxError):
        return 0.0


def _ioncount(v):
    try:
        x = ast.literal_eval(str(v)) if isinstance(v, str) else v
        if isinstance(x, list):
            return float(sum(int(d.get("n", 0)) for d in x if isinstance(d, dict)))
    except (ValueError, SyntaxError):
        return np.nan
    return np.nan


def _logkd(v):
    try:
        fv = float(v)
        return np.log10(fv) if fv > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def _has_family(v):
    try:
        x = ast.literal_eval(str(v))
        vals = x if isinstance(x, (list, tuple)) else [x]
        return 1.0 if any(str(z).lower() not in ("none", "") for z in vals) else 0.0
    except (ValueError, SyntaxError):
        return 0.0


def _inf(v, i):
    """RNA_INF = [INF_all, INF_WC, INF_nonWC, INF_stack]; -1 marks missing."""
    try:
        x = ast.literal_eval(str(v))
        val = float(x[i])
        return np.nan if val == -1 else val
    except (ValueError, SyntaxError, IndexError, TypeError):
        return np.nan


LABELS = {
    "pucker_C2endo": r"sugar puck C2$'$-endo frac (exp)",
    "gamma_gplus": r"$\gamma$ g+ frac (exp)", "glyc_syn": "glycosidic syn frac (exp)",
    "n_Stems": "# stems (exp)", "n_Hairpin": "# hairpins (exp)",
    "n_Internal": "# internal loops (exp)", "n_Multibranch": "# multibranch (exp)",
    "n_Dangling": "# dangling ends (exp)", "n_Pseudoknots": "# pseudoknots (exp)",
    "isDuplex": "is duplex (exp)", "RNAMotif_isDuplex": "motif is duplex (exp)",
    "RNALength": "RNA length (exp)", "NumberRNAs": "# RNA chains (exp)",
    "RNAMotifLength": "motif length (exp)", "Hbond": "interface H-bonds (exp)",
    "vdW": "interface vdW contacts (exp)", "InterfaceArea": "interface area (exp)",
    "InterfaceRatio": "interface ratio (exp)", "Free_energy": "complex free energy (exp)",
    "logKd": r"log$_{10}$ $K_d$ (exp)", "ElectrostaticPot": "RNA electrostatic pot. (exp)",
    "GC_Content": "GC content", "SeqComplexity": "sequence complexity",
    "Shannon_K1": "Shannon entropy (k1)", "Shannon_K2": "Shannon entropy (k2)",
    "GC_Skew": "GC skew", "AU_Skew": "AU skew", "AU_Content": "AU content",
    "MFE": "MFE (folding energy)", "EnsembleDiv": "ensemble diversity",
    "CentroidEnergy": "centroid energy", "RNAmotif_score": "RNA motif score",
    "AAmatch": "protein seq match", "RNAmatch": "RNA seq match",
    "RNA_msa_depth": r"MSA$_r$ depth", "has_RNA_MSA": r"has MSA$_r$",
    "Protein_msa_depth": r"MSA$_p$ depth", "has_Protein_MSA": r"has MSA$_p$",
    "RBP_template_n": "# RBP templates", "Protein_template_n": "# protein templates",
    "exp_pH": "exp pH", "exp_Temp": "exp temperature", "ion_count": "# ions",
    "has_family": "RNA family annotated", "PEG_percent": "PEG %", "buffer_M": "buffer conc.",
    "INF_WC": "WC base-pair fidelity (INF)", "INF_nonWC": "non-WC fidelity (INF)",
    "INF_stack": "stacking fidelity (INF)",
}
# Fixed semantic ordering: related features are grouped and shown together. The
# INF base-pairing-fidelity block stays first (WC-vs-nonWC/stacking contrast; these
# are predicted-vs-true accuracy sub-metrics). Only features present are drawn.
GROUPS = [
    ("base-pair\nfidelity", ["INF_stack", "INF_nonWC", "INF_WC"]),
    ("backbone", ["pucker_C2endo", "gamma_gplus", "glyc_syn"]),
    ("2°\nstructure", ["n_Stems", "n_Hairpin", "n_Internal", "n_Multibranch",
                       "n_Dangling", "n_Pseudoknots", "isDuplex", "RNAMotif_isDuplex"]),
    ("size", ["RNALength", "NumberRNAs"]),
    ("sequence", ["GC_Content", "AU_Content", "GC_Skew", "AU_Skew",
                  "Shannon_K1", "Shannon_K2", "SeqComplexity"]),
    ("thermo-\ndynamics", ["MFE", "EnsembleDiv", "CentroidEnergy", "ElectrostaticPot"]),
    ("interface", ["InterfaceArea", "InterfaceRatio", "Hbond", "vdW", "Free_energy",
                   "logKd", "RNAMotifLength"]),
    ("homology", ["RNA_msa_depth", "has_RNA_MSA", "Protein_msa_depth",
                  "has_Protein_MSA", "RBP_template_n", "Protein_template_n", "has_family"]),
    ("cond.", ["exp_pH", "exp_Temp", "ion_count", "buffer_M", "PEG_percent"]),
]
# target columns: (key, display label, kind)
TARGETS = [("RNA_TM", r"$\mathrm{TM}_{r}$", "reg"),
           ("RNA_LDDT", r"$\mathrm{LDDT}_{r}$", "reg"),
           ("RNA_RMSD", r"$\mathrm{RMSD}_{r}$", "reg"),
           ("good_rna", "RNA class\n(i+ii / iii+iv)", "clf")]


def _common_struct(F, df):
    """Structural + sequence + condition features shared by both datasets."""
    F["Free_energy"] = df["Free_energy"].apply(_f)
    F["logKd"] = df["Binding_affinity_kd"].apply(_logkd)
    F["glyc_syn"] = df["RNA_GlycosidicBond"].apply(lambda v: _dictfrac(v, ["syn"]))
    F["pucker_C2endo"] = df["RNA_SugarPucker"].apply(lambda v: _dictfrac(v, ["C2'-endo"]))
    F["gamma_gplus"] = df["RNA_GammaAngle"].apply(lambda v: _dictfrac(v, ["g+"]))
    for nm, col in [("n_Stems", "RNA_Stems"), ("n_Hairpin", "RNA_HairpinLoops"),
                    ("n_Internal", "RNA_InternalLoops"), ("n_Multibranch", "RNA_MultibranchLoops"),
                    ("n_Dangling", "RNA_DanglingEnds"), ("n_Pseudoknots", "RNA_Pseudoknots")]:
        F[nm] = df[col].apply(_countnested)
    F["isDuplex"] = df["RNA_isDuplex"].apply(_f)
    F["ElectrostaticPot"] = df["RNA_ElectrostaticPotential"].apply(_f)
    F["exp_pH"] = df["exp_pH"].apply(_f)
    F["exp_Temp"] = df["exp_Temp"].apply(_f)
    F["ion_count"] = df["Ions"].apply(_ioncount)
    F["has_family"] = df["RNAFamily"].apply(_has_family)
    for nm, col in [("GC_Content", "RNA_GC_Content"), ("SeqComplexity", "RNA_SequenceComplexity"),
                    ("Shannon_K1", "RNA_Shannon_Entropy_K1"), ("Shannon_K2", "RNA_Shannon_Entropy_K2"),
                    ("GC_Skew", "RNA_GC_Skew"), ("AU_Skew", "RNA_AU_Skew"),
                    ("AU_Content", "RNA_AU_Content"), ("MFE", "RNA_MFE_Value"),
                    ("EnsembleDiv", "RNA_Ensemble_Diversity"), ("CentroidEnergy", "RNA_Centroid_Energy")]:
        F[nm] = df[col].apply(_f)
    F["RNA_msa_depth"] = df["RNA_msa_size"].apply(_msadepth)
    F["has_RNA_MSA"] = (F["RNA_msa_depth"] > 1).astype(float)
    if "RNA_INF" in df.columns:  # base-pairing fidelity components (accuracy sub-metrics)
        F["INF_WC"] = df["RNA_INF"].apply(lambda v: _inf(v, 1))
        F["INF_nonWC"] = df["RNA_INF"].apply(lambda v: _inf(v, 2))
        F["INF_stack"] = df["RNA_INF"].apply(lambda v: _inf(v, 3))


def _finalize(F, y, min_notna):
    X = pd.DataFrame(F)
    keep = [c for c in X.columns
            if X[c].notna().sum() >= min_notna and X[c].nunique(dropna=True) > 1]
    return X[keep], y


def _build_pr():
    """Protein-RNA (rbpDatabase.db). Interface features are protein-RNA."""
    con = sqlite3.connect(_DB_PR)
    pr = pd.read_sql_query(
        "SELECT exp_db_id, RNA_TM, RNA_LDDT, RNA_LDDT_allAtom, RNA_RMSD, "
        "RNA_GC_Content, RNA_SequenceComplexity, RNA_Shannon_Entropy_K1, "
        "RNA_Shannon_Entropy_K2, RNA_GC_Skew, RNA_AU_Skew, RNA_AU_Content, "
        "RNA_MFE_Value, RNA_Ensemble_Diversity, RNA_Centroid_Energy, RNAmotif_score, "
        "RNA_INF, AAmatch_score, RNAmatch_score, RNA_msa_size, Protein_msa_size, "
        "RBP_PDB_Templates, Protein_PDB_Templates FROM pred_protein_rna", con)
    ex = pd.read_sql_query(
        "SELECT PDBId, RNALength, NumberRNAs, RNAMotifLength, Hbond_proteinRNA, "
        "vdWbond_proteinRNA, ProteinRNAInterfaceArea, ProteinRNAInterfaceRatio, "
        "Free_energy, Binding_affinity_kd, RNA_GlycosidicBond, RNA_SugarPucker, "
        "RNA_GammaAngle, RNA_Stems, RNA_HairpinLoops, RNA_InternalLoops, "
        "RNA_MultibranchLoops, RNA_DanglingEnds, RNA_Pseudoknots, RNA_isDuplex, "
        "RNAMotif_isDuplex, RNA_ElectrostaticPotential, exp_pH, exp_Temp, Ions, "
        "RNAFamily FROM exp_protein_rna", con)
    con.close()
    df = pr.merge(ex, left_on="exp_db_id", right_on="PDBId", how="left")
    with contextlib.redirect_stdout(io.StringIO()):
        from evaluation.peg_analysis_plots import PEGAnalysisPlotter
        peg = PEGAnalysisPlotter().df[["PDBId", "PEG_percent", "buffer_M"]].copy()
    for c in ("PEG_percent", "buffer_M"):
        peg[c] = pd.to_numeric(peg[c], errors="coerce")
    df = df.merge(peg, left_on="exp_db_id", right_on="PDBId", how="left", suffixes=("", "_pg"))

    F = {}
    F["RNALength"] = df["RNALength"].apply(_meanlist)
    F["NumberRNAs"] = df["NumberRNAs"].apply(_f)
    F["RNAMotifLength"] = df["RNAMotifLength"].apply(_meanlist)
    F["Hbond"] = df["Hbond_proteinRNA"].apply(_sumlist)
    F["vdW"] = df["vdWbond_proteinRNA"].apply(_sumlist)
    F["InterfaceArea"] = df["ProteinRNAInterfaceArea"].apply(_f)
    F["InterfaceRatio"] = df["ProteinRNAInterfaceRatio"].apply(_f)
    _common_struct(F, df)
    F["RNAMotif_isDuplex"] = df["RNAMotif_isDuplex"].apply(_f)
    F["Protein_msa_depth"] = df["Protein_msa_size"].apply(_msadepth)
    F["has_Protein_MSA"] = (F["Protein_msa_depth"] > 1).astype(float)
    F["RBP_template_n"] = df["RBP_PDB_Templates"].apply(_countjson)
    F["Protein_template_n"] = df["Protein_PDB_Templates"].apply(_countjson)
    F["PEG_percent"] = df["PEG_percent"]
    F["buffer_M"] = df["buffer_M"]

    y = {"RNA_TM": df["RNA_TM"].apply(_f), "RNA_LDDT": df["RNA_LDDT_allAtom"].apply(_f),
         "RNA_RMSD": df["RNA_RMSD"].apply(_f),
         "good_rna": (df["RNA_LDDT"].apply(_f) > _LDDT_GOOD).astype(float)}
    return _finalize(F, y, min_notna=40)


def _build_rnaonly():
    """RNA-only / RNA-RNA (a2021_rna.db). Interface features are RNA-RNA; no protein."""
    con = sqlite3.connect(_DB_RO)
    pr = pd.read_sql_query(
        "SELECT exp_db_id, RNA_TM, RNA_LDDT, RNA_LDDT_allAtom, RNA_RMSD, "
        "RNA_GC_Content, RNA_SequenceComplexity, RNA_Shannon_Entropy_K1, "
        "RNA_Shannon_Entropy_K2, RNA_GC_Skew, RNA_AU_Skew, RNA_AU_Content, "
        "RNA_MFE_Value, RNA_Ensemble_Diversity, RNA_Centroid_Energy, "
        "RNA_INF, RNAmatch_score, RNA_msa_size FROM pred_rna_rna", con)
    ex = pd.read_sql_query(
        "SELECT PDBId, RNALength, NumberRNAs, RNAMotifLength, Hbond_rnaRNA, "
        "vdWbond_rnaRNA, RNArnaInterfaceArea, RNArnaInterfaceRatio, Free_energy, "
        "Binding_affinity_kd, RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle, "
        "RNA_Stems, RNA_HairpinLoops, RNA_InternalLoops, RNA_MultibranchLoops, "
        "RNA_DanglingEnds, RNA_Pseudoknots, RNA_isDuplex, RNA_ElectrostaticPotential, "
        "exp_pH, exp_Temp, Ions, RNAFamily FROM exp_rna_rna", con)
    con.close()
    df = pr.merge(ex, left_on="exp_db_id", right_on="PDBId", how="left")

    F = {}
    F["RNALength"] = df["RNALength"].apply(_meanlist)
    F["NumberRNAs"] = df["NumberRNAs"].apply(_f)
    F["RNAMotifLength"] = df["RNAMotifLength"].apply(_meanlist)
    F["Hbond"] = df["Hbond_rnaRNA"].apply(_sumlist)
    F["vdW"] = df["vdWbond_rnaRNA"].apply(_sumlist)
    F["InterfaceArea"] = df["RNArnaInterfaceArea"].apply(_f)
    F["InterfaceRatio"] = df["RNArnaInterfaceRatio"].apply(_f)
    _common_struct(F, df)

    y = {"RNA_TM": df["RNA_TM"].apply(_f), "RNA_LDDT": df["RNA_LDDT_allAtom"].apply(_f),
         "RNA_RMSD": df["RNA_RMSD"].apply(_f),
         "good_rna": (df["RNA_LDDT"].apply(_f) > _LDDT_GOOD).astype(float)}
    return _finalize(F, y, min_notna=15)


# ---- importance ------------------------------------------------------------
def _cv_perm_importance(X, yv, kind, n_splits=5):
    """Cross-validated permutation importance (mean over folds) + mean CV score.
    Median imputation is fit inside each training fold only, so no test-fold
    information leaks into the imputation (consistent with the in-fold
    preprocessing used elsewhere in this work)."""
    Xa = np.asarray(X, dtype=float)
    yv = np.asarray(yv)
    if kind == "clf":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                       class_weight="balanced", random_state=0, n_jobs=-1)
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=0)
        scoring, split_y = "roc_auc", yv
    else:
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                      random_state=0, n_jobs=-1)
        splitter = KFold(n_splits, shuffle=True, random_state=0)
        scoring, split_y = "r2", None
    imps, scores = [], []
    for tr, te in splitter.split(Xa, split_y):
        imp = SimpleImputer(strategy="median").fit(Xa[tr])
        Xtr, Xte = imp.transform(Xa[tr]), imp.transform(Xa[te])
        model.fit(Xtr, yv[tr])
        pi = permutation_importance(model, Xte, yv[te], scoring=scoring,
                                    n_repeats=5, random_state=0, n_jobs=-1)
        imps.append(pi.importances_mean)
        scores.append(model.score(Xte, yv[te]) if kind == "reg" else np.nan)
    return np.clip(np.mean(imps, axis=0), 0, None), (np.nanmean(scores) if scores else np.nan)


def _analyse(X, y):
    feats = list(X.columns)
    tgt = [t for t in TARGETS if t[0] in y]
    rho = pd.DataFrame(index=feats, columns=[t[0] for t in tgt], dtype=float)
    rfimp = pd.DataFrame(index=feats, columns=[t[0] for t in tgt], dtype=float)
    for key, _lab, kind in tgt:
        yv = y[key].values.astype(float)
        m = np.isfinite(yv)
        Xt, yt = X[m], yv[m]
        for c in feats:
            mm = Xt[c].notna().values & np.isfinite(yt)
            if mm.sum() >= 15 and Xt[c][mm].nunique() > 1 and np.ptp(yt[mm]) > 0:
                rho.loc[c, key] = spearmanr(Xt[c][mm], yt[mm]).correlation
        imp, score = _cv_perm_importance(Xt, yt, kind)
        rfimp[key] = imp
        extra = f"pos={int(yt.sum())}/{len(yt)}" if kind == "clf" else f"R2={score:.2f}"
        print(f"  {key:9s} ({kind}): {extra}, n={int(m.sum())}")
    return rho, rfimp, tgt


def _draw(rho, rfimp, tgt, out_name, suptitle):
    from matplotlib.transforms import blended_transform_factory
    F_TICK_Y, F_TICK_X, F_CELL, F_CBAR, F_LET, F_SUP, F_GRP = 14, 16, 12, 15, 26, 20, 13
    # orient so + = higher accuracy / good RNA (flip RMSD only)
    rho_o = rho.copy()
    if "RNA_RMSD" in rho_o:
        rho_o["RNA_RMSD"] = -rho_o["RNA_RMSD"]
    norm = rfimp / rfimp.max().replace(0, np.nan)
    # fixed semantic ordering; keep only present features; record group spans
    ordered, spans = [], []
    for gname, feats in GROUPS:
        present = [f for f in feats if f in rho_o.index]
        if not present:
            continue
        spans.append((gname, len(ordered), len(ordered) + len(present)))
        ordered += present
    rho_o, norm = rho_o.loc[ordered], norm.loc[ordered]
    ylabels = [LABELS.get(f, f) for f in ordered]
    tlabs = [t[1] for t in tgt]
    x = np.arange(len(tgt)); yv = np.arange(len(ordered))

    H = max(9.5, len(ordered) * 0.42 + 2.2)
    fig = plt.figure(figsize=(15.5, H))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.05, 0.30, 1.0, 0.05], wspace=0.16,
                          top=0.965, bottom=0.05, left=0.26, right=0.965)
    axA = fig.add_subplot(gs[0, 0]); caxA = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[0, 3]); caxB = fig.add_subplot(gs[0, 4])

    def _cells(ax, M, cmap, vmin, vmax, hi):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v) and (cmap != "Greens" or v > 0.02):
                    ax.text(j, i, (f"{v:+.2f}" if cmap != "Greens" else f"{v:.2f}"),
                            ha="center", va="center", fontsize=F_CELL,
                            color="white" if hi(v) else "black")
        ax.set_xticks(x); ax.set_xticklabels(tlabs, fontsize=F_TICK_X)
        for _g, s, e in spans[1:]:
            ax.axhline(s - 0.5, color="0.55", lw=1.0)
        ax.axhline(spans[0][2] - 0.5, color="0.15", lw=1.8)  # bold under INF block
        return im

    im = _cells(axA, rho_o.values.astype(float), "RdBu_r", -0.6, 0.6, lambda v: abs(v) > 0.4)
    axA.set_yticks(yv); axA.set_yticklabels(ylabels, fontsize=F_TICK_Y)
    im2 = _cells(axB, norm.values.astype(float), "Greens", 0, 1, lambda v: v > 0.6)
    axB.set_yticks(yv); axB.set_yticklabels([])

    # group labels in the far-left margin (x in axes frac, y in data coords)
    tr = blended_transform_factory(axA.transAxes, axA.transData)
    for gname, s, e in spans:
        axA.text(-0.82, (s + e - 1) / 2.0, gname, rotation=90, va="center", ha="center",
                 fontsize=F_GRP, fontweight="bold", color="0.2", transform=tr, linespacing=0.9)

    cbA = fig.colorbar(im, cax=caxA)
    cbA.set_label(r"Spearman $\rho$  (+ $\rightarrow$ higher accuracy / good RNA)", fontsize=F_CBAR)
    cbA.ax.tick_params(labelsize=F_CBAR - 3)
    cbB = fig.colorbar(im2, cax=caxB)
    cbB.set_label("random-forest permutation importance", fontsize=F_CBAR)
    cbB.ax.tick_params(labelsize=F_CBAR - 3)
    axA.text(-0.92, 1.008, "A", transform=axA.transAxes, fontsize=F_LET, fontweight="bold",
             ha="left", va="bottom")
    axB.text(-0.02, 1.008, "B", transform=axB.transAxes, fontsize=F_LET, fontweight="bold",
             ha="right", va="bottom")

    out = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{out_name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.join(out, f"{out_name}.png"))


def main():
    print("Protein-RNA complexes:")
    Xp, yp = _build_pr()
    print(f"  features={len(Xp.columns)}")
    rho, rfimp, tgt = _analyse(Xp, yp)
    _draw(rho, rfimp, tgt, "rna_accuracy_feature_importance",
          "Determinants of RNA accuracy — protein–RNA complexes")

    print("RNA-only (RNA–RNA) complexes:")
    Xr, yr = _build_rnaonly()
    print(f"  features={len(Xr.columns)}")
    rho2, rfimp2, tgt2 = _analyse(Xr, yr)
    _draw(rho2, rfimp2, tgt2, "rna_accuracy_feature_importance_rnaonly",
          "Determinants of RNA accuracy — single-chain RNA")


if __name__ == "__main__":
    main()
