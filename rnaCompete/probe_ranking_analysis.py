"""
Per-protein RNAcompete maxprobe-vs-minprobe ranking analyses.

Each pred_protein_rna row is one in-silico competition assay for a protein:
1 maxprobe RNA chain (index 0) + 1-3 minprobe RNA chains (MAX1MIN{1,2,3}).
Per-chain predicted preference signals are taken from AF3 confidence lists
(af3_rna_ipTM, af3_rna_pTM) and the per-chain predicted RNA motif length.
Per-probe experimental ground truth (BindingIntensity, RNAMotif_z_score) is
read from the exp_protein_rna CSV and matched by RNA sequence.

Implements the recommended next analyses:
  (A) per-protein ranking accuracy (within-protein pairwise AUC)
  (B) global + per-protein ROC-AUC (maxprobe=1, minprobe=0)
  (C) per-protein effect-size decomposition + Wilcoxon signed-rank
  (D) logistic regression of correct ranking on structural/sequence features
  (E) length-controlled / matched analyses (equal RNALength, per class type)
  (F) margin-based analysis (predicted margin vs experimental margin)
  (G) agreement with RNACompete experimental signals
  (H) stability under increasing competition complexity (MAX1MIN1->3)

All plots + a statistics text file are written to the output directory via
plots.plotCreator.PlotCreator (standalone mode).

Usage:
  python3 rnacompete/probe_ranking_analysis.py
  python3 rnacompete/probe_ranking_analysis.py --db PATH --csv PATH --outdir PATH
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator

PLOT_TYPE = "rna_compete_ranking"
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_compete_ranking")
DEFAULT_CSV = os.path.join(_REPO_ROOT, "data", "protein_rna", "exp_protein_rna.csv")
STATS_FILENAME = "probe_ranking_statistics.txt"

MAX_COLOR = "#1976D2"
MIN_COLOR = "#D32F2F"
ACCENT = "#2E7D32"

# Per-chain predicted preference scores (higher => model prefers this RNA).
# Only per-RNA-chain fields can rank maxprobe vs minprobe within a protein;
# af3_protein_*/Binding_affinity_kd are per-complex scalars and are used as
# logistic-regression features only (see plot_logistic).
PRED_SCORES = (
    ("af3_rna_ipTM", "AF3 RNA ipTM"),
    ("pred_motif_length", "Predicted RNA motif length"),
)
PRIMARY_SCORE = "af3_rna_ipTM"

# Per-probe experimental signals (higher => stronger experimental binding).
EXP_SIGNALS = (
    ("intensity", "Experimental binding intensity"),
    ("zscore", "RNAcompete motif z-score"),
)
PRIMARY_EXP = "intensity"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def base_protein_id(pdb_id):
    """RNCMPT00001 from a longer competition-assay id."""
    m = re.match(r"(RNCMPT\d{5})", pdb_id, re.IGNORECASE)
    return m.group(1).upper() if m else pdb_id.upper()


def class_type(pdb_id):
    m = re.search(r"MAX\d+MIN\d+", pdb_id, re.IGNORECASE)
    return m.group(0).upper() if m else "unknown"


def _to_list(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, (int, float)):
        return [raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, (int, float)):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    return None


def _to_float(raw):
    if raw is None:
        return np.nan
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


def _scalar_from_cell(raw):
    """Single float from a scalar or a (possibly length-1) list cell.

    af3_protein_pTM / af3_protein_ipTM are stored as length-1 lists like [0.35].
    """
    vals = _to_list(raw)
    if vals is None:
        return _to_float(raw)
    finite = [v for v in (_to_float(x) for x in vals) if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def _log10_kd(raw):
    kd = _to_float(raw)
    if not np.isfinite(kd) or kd <= 0:
        return np.nan
    return float(np.log10(kd))


def _aligned_per_chain(raw, n_chains):
    """Per-chain numeric list aligned to RNAChainIDs order; pad/truncate to n_chains."""
    vals = _to_list(raw)
    if vals is None:
        return [np.nan] * n_chains
    out = []
    for i in range(n_chains):
        out.append(_to_float(vals[i]) if i < len(vals) else np.nan)
    return out


def _motif_lengths_per_chain(motif_raw, chain_pairs_raw, rna_chains):
    """Align predicted RNAMotifLength to RNA chain order via ChainIDpairList_proteinRNA."""
    n = len(rna_chains)
    motif_vals = _to_list(motif_raw)
    pairs = _to_list(chain_pairs_raw)
    if motif_vals is None:
        return [0.0] * n
    if pairs and isinstance(pairs[0], (list, tuple)):
        motif_map = {}
        for i, pair in enumerate(pairs):
            if len(pair) > 1 and i < len(motif_vals):
                motif_map[pair[1]] = _to_float(motif_vals[i])
        return [float(motif_map.get(ch, 0.0) or 0.0) for ch in rna_chains]
    # Fallback: positional, misses -> 0.
    return [float(_to_float(motif_vals[i])) if i < len(motif_vals) else 0.0 for i in range(n)]


def _counts(raw):
    vals = _to_list(raw)
    if vals is None:
        return 0
    return len(vals)


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def rank_auc(pos, neg):
    """Probability a random positive outranks a random negative (ties = 0.5).

    For 1 positive vs k negatives this equals the user's RankAccuracy.
    """
    pos = [v for v in pos if v is not None and np.isfinite(v)]
    neg = [v for v in neg if v is not None and np.isfinite(v)]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def global_auc(labels, scores):
    """Rank-based ROC-AUC (Mann-Whitney). labels in {0,1}."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = scores.argsort().argsort().astype(float) + 1.0
    # tie-correct ranks
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sr = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = sr[i:j + 1].mean()
            sr[i:j + 1] = avg
        i = j + 1
    ranks[order] = sr
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = pos.size, neg.size
    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve_points(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1])
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.concatenate(([0.0], tps / P))
    fpr = np.concatenate(([0.0], fps / N))
    return fpr, tpr


def summarize(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def wilcoxon_greater(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    nonzero = arr[arr != 0]
    if nonzero.size < 1:
        return None
    try:
        stat, p = wilcoxon(nonzero, alternative="greater")
    except ValueError:
        return None
    return {"n": int(arr.size), "n_nonzero": int(nonzero.size), "stat": float(stat), "p": float(p)}


def fmt_summary(s):
    if not s:
        return "no data"
    return (f"n={s['n']}, mean={s['mean']:.4g}, median={s['median']:.4g}, "
            f"std={s['std']:.4g}, min={s['min']:.4g}, max={s['max']:.4g}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_exp_csv(csv_path):
    """(base_id, RNASequence) -> dict(class, intensity, zscore, exp_motif_len, exp_rna_len)."""
    probe_map = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (base_protein_id(row["PDBId"]).lower(), row["RNASequence"].strip().upper())
            probe_map[key] = {
                "class": row.get("Class", "").strip().lower(),
                "intensity": _to_float(row.get("BindingIntensity")),
                "zscore": _to_float(row.get("RNAMotif_z_score")),
                "exp_motif_len": _to_float(row.get("RNAMotifLength")),
                "exp_rna_len": _to_float(row.get("RNALength")),
            }
    return probe_map


PRED_QUERY = """
    SELECT exp_db_id, RNASequence, RNAChainIDs, ChainIDpairList_proteinRNA, RNAMotifLength,
           af3_rna_ipTM, af3_rna_pTM, RNALength,
           af3_protein_ipTM, af3_protein_pTM, af3_rna_pLDDT_avg, af3_global_pae_avg,
           af3_chain_pair_pae_min, af3_ranking_score, RNAmotif_score, RNAMotif_inExp,
           Binding_affinity_kd, RBP_PDB_Templates, Protein_PDB_Templates
    FROM pred_protein_rna
"""


def load_probe_records(db_path, probe_map):
    """Return a flat list of competition-assay dicts with classified probe-chain records.

    Each pred_protein_rna row is one in-silico competition assay (one protein, one
    maxprobe + 1-3 minprobe RNA chains). Currently there is exactly one assay per
    protein, so the assay is the unit of analysis (not repeated within-protein).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(PRED_QUERY)
    rows = cur.fetchall()
    conn.close()

    assays = []
    proteins_seen = set()
    for r in rows:
        (exp_id, rna_seq_raw, rna_chains_raw, chain_pairs_raw, motif_raw,
         iptm_raw, ptm_raw, rna_len_raw, prot_iptm, prot_ptm, rna_plddt, global_pae,
         chain_pae_min, ranking_score, rnamotif_score, rnamotif_inexp,
         kd_raw, rbp_templates, prot_templates) = r

        seqs = _to_list(rna_seq_raw)
        chains = _to_list(rna_chains_raw)
        if not seqs or not chains:
            continue
        n = len(seqs)
        pid = base_protein_id(exp_id)
        base = pid.lower()
        ctype = class_type(exp_id)

        iptm = _aligned_per_chain(iptm_raw, n)
        ptm = _aligned_per_chain(ptm_raw, n)
        motif_pred = _motif_lengths_per_chain(motif_raw, chain_pairs_raw, chains)
        rna_lengths = _aligned_per_chain(rna_len_raw, n)

        job_features = {
            "af3_protein_ipTM": _scalar_from_cell(prot_iptm),
            "af3_protein_pTM": _scalar_from_cell(prot_ptm),
            "log10_Binding_affinity_kd": _log10_kd(kd_raw),
            "af3_rna_pLDDT_avg": _to_float(rna_plddt),
            "af3_global_pae_avg": _to_float(global_pae),
            "af3_chain_pair_pae_min": _to_float(chain_pae_min),
            "af3_ranking_score": _to_float(ranking_score),
            "RNAmotif_score": _to_float(rnamotif_score),
            "RNAMotif_inExp": _to_float(rnamotif_inexp),
            "RBP_PDB_Templates_n": _counts(rbp_templates),
            "Protein_PDB_Templates_n": _counts(prot_templates),
        }

        chain_records = []
        for i in range(n):
            seq = seqs[i].strip().upper() if isinstance(seqs[i], str) else seqs[i]
            probe = probe_map.get((base, seq))
            if probe is None:
                # Fall back: chain 0 is the maxprobe by construction.
                klass = "maxprobe" if i == 0 else "minprobe"
                intensity = zscore = np.nan
                exp_rna_len = rna_lengths[i]
            else:
                klass = probe["class"]
                intensity = probe["intensity"]
                zscore = probe["zscore"]
                exp_rna_len = probe["exp_rna_len"]
            if klass not in ("maxprobe", "minprobe"):
                continue
            chain_records.append({
                "chain_idx": i,
                "class": klass,
                "af3_rna_ipTM": iptm[i],
                "af3_rna_pTM": ptm[i],
                "pred_motif_length": motif_pred[i],
                "intensity": intensity,
                "zscore": zscore,
                "rna_length": exp_rna_len if np.isfinite(exp_rna_len) else rna_lengths[i],
            })

        has_max = any(c["class"] == "maxprobe" for c in chain_records)
        has_min = any(c["class"] == "minprobe" for c in chain_records)
        if not (has_max and has_min):
            continue

        proteins_seen.add(pid)
        assays.append({
            "id": exp_id,
            "protein": pid,
            "class_type": ctype,
            "chains": chain_records,
            "features": job_features,
            "n_min": sum(1 for c in chain_records if c["class"] == "minprobe"),
        })

    return assays, len(proteins_seen)


# --------------------------------------------------------------------------- #
# Per-assay aggregation
# --------------------------------------------------------------------------- #
def assay_score_groups(assay, score_key):
    """Return (max_scores, min_scores) lists for a single competition assay."""
    max_s, min_s = [], []
    for c in assay["chains"]:
        v = c[score_key]
        if v is None or not np.isfinite(v):
            continue
        (max_s if c["class"] == "maxprobe" else min_s).append(v)
    return max_s, min_s


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def analysis_rank_accuracy(assays, score_key):
    """Per-assay rank accuracy (within-assay AUC: maxprobe vs its minprobes)."""
    acc = {}
    for a in assays:
        max_s, min_s = assay_score_groups(a, score_key)
        auc = rank_auc(max_s, min_s)
        if auc is not None:
            acc[a["id"]] = auc
    return acc


def analysis_effect_size(assays, score_key):
    """Per-assay deltaS = mean(max) - mean(min)."""
    deltas = {}
    for a in assays:
        max_s, min_s = assay_score_groups(a, score_key)
        max_s = [v for v in max_s if np.isfinite(v)]
        min_s = [v for v in min_s if np.isfinite(v)]
        if max_s and min_s:
            deltas[a["id"]] = float(np.mean(max_s) - np.mean(min_s))
    return deltas


def assay_margins(assays, pred_key, exp_key):
    """Per-assay (pred_margin, exp_margin) using max - max_j(min) for pred and exp."""
    out = {}
    for a in assays:
        pmax, pmin = assay_score_groups(a, pred_key)
        emax, emin = assay_score_groups(a, exp_key)
        pmax = [v for v in pmax if np.isfinite(v)]
        pmin = [v for v in pmin if np.isfinite(v)]
        emax = [v for v in emax if np.isfinite(v)]
        emin = [v for v in emin if np.isfinite(v)]
        if pmax and pmin and emax and emin:
            pred_margin = max(pmax) - max(pmin)
            exp_margin = max(emax) - float(np.mean(emin))
            out[a["id"]] = (pred_margin, exp_margin)
    return out


# --------------------------------------------------------------------------- #
# Logistic regression (numpy + scipy, no sklearn)
# --------------------------------------------------------------------------- #
def fit_logistic(X, y, l2=1.0):
    """Standardized logistic regression via scipy.optimize. Returns coefs, intercept, auc."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    Xb = np.hstack([np.ones((n, 1)), Xs])

    def nll(w):
        z = Xb @ w
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        reg = l2 * np.sum(w[1:] ** 2) / 2.0
        return -np.sum(ll) + reg

    def grad(w):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y)
        g[1:] += l2 * w[1:]
        return g

    w0 = np.zeros(d + 1)
    res = minimize(nll, w0, jac=grad, method="L-BFGS-B")
    w = res.x
    z = np.clip(Xb @ w, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    auc = global_auc(y, p)
    return w[1:], w[0], auc


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_rank_accuracy(acc_by_score, plotter, sf):
    keys = [k for k, _ in PRED_SCORES if acc_by_score.get(k)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        label = dict(PRED_SCORES)[key]
        vals = np.array(list(acc_by_score[key].values()))
        ax.hist(vals, bins=np.linspace(0, 1, 21), color=MAX_COLOR, alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color=ACCENT, lw=2, label=f"mean={vals.mean():.2f}")
        ax.axvline(np.median(vals), color="black", lw=1.5, ls="--", label=f"median={np.median(vals):.2f}")
        frac_perfect = float(np.mean(vals >= 0.999))
        ax.set_title(f"{label}\nmax>all min in {frac_perfect*100:.0f}% of assays")
        ax.set_xlabel("Within-assay rank accuracy")
        ax.set_ylabel("Competition assays")
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("(A) Per-assay maxprobe ranking accuracy (one assay per protein)")
    plt.tight_layout()
    plotter.save_plot(f"A_rank_accuracy{sf}")
    plt.close(fig)


def plot_roc(assays, plotter, sf):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for key, label in PRED_SCORES:
        labels, scores = [], []
        for a in assays:
            for c in a["chains"]:
                v = c[key]
                if v is not None and np.isfinite(v):
                    labels.append(1 if c["class"] == "maxprobe" else 0)
                    scores.append(v)
        if not labels:
            continue
        fpr, tpr = roc_curve_points(labels, scores)
        auc = global_auc(labels, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(B) Global ROC: maxprobe (1) vs minprobe (0)")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plotter.save_plot(f"B_global_roc{sf}")
    plt.close(fig)


def plot_effect_sizes(eff_by_score, plotter, sf):
    keys = [k for k, _ in PRED_SCORES if eff_by_score.get(k)]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        label = dict(PRED_SCORES)[key]
        vals = np.array(list(eff_by_score[key].values()))
        ax.hist(vals, bins=25, color=MAX_COLOR, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5)
        ax.axvline(vals.mean(), color=ACCENT, lw=2, label=f"mean={vals.mean():.3g}")
        frac_pos = float(np.mean(vals > 0))
        ax.set_title(f"{label}\nmax>min in {frac_pos*100:.0f}% of assays")
        ax.set_xlabel(r"$\Delta S$ = mean(max) - mean(min)")
        ax.set_ylabel("Competition assays")
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("(C) Per-assay effect size")
    plt.tight_layout()
    plotter.save_plot(f"C_effect_size{sf}")
    plt.close(fig)


def plot_margins(assays, plotter, sf, stats_file):
    ncol = len(PRED_SCORES)
    nrow = len(EXP_SIGNALS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.4 * nrow), squeeze=False)
    stats_file.write("(F/G) Prediction margin vs experimental margin (Spearman / Pearson):\n")
    for ri, (exp_key, exp_label) in enumerate(EXP_SIGNALS):
        for ci, (pred_key, pred_label) in enumerate(PRED_SCORES):
            ax = axes[ri][ci]
            margins = assay_margins(assays, pred_key, exp_key)
            if len(margins) < 3:
                ax.set_visible(False)
                continue
            xp = np.array([m[0] for m in margins.values()])
            yp = np.array([m[1] for m in margins.values()])
            ax.scatter(xp, yp, s=22, alpha=0.6, color=MAX_COLOR, edgecolor="white", linewidth=0.4)
            rho, prho = spearmanr(xp, yp)
            r, pr = pearsonr(xp, yp)
            ax.set_title(f"{pred_label}\nrho={rho:.2f} (p={prho:.1e}), r={r:.2f}", fontsize=10)
            ax.set_xlabel(f"Predicted margin\n({pred_label})", fontsize=9)
            ax.set_ylabel(f"Exp margin\n({exp_label})", fontsize=9)
            ax.grid(True, ls="--", alpha=0.4)
            stats_file.write(
                f"  {exp_label} vs {pred_label}: n={len(margins)}, "
                f"Spearman rho={rho:.4f} (p={prho:.3e}), Pearson r={r:.4f} (p={pr:.3e})\n"
            )
    fig.suptitle("(F/G) Per-assay predicted vs experimental separation margins")
    plt.tight_layout()
    plotter.save_plot(f"FG_margin_correlation{sf}")
    plt.close(fig)
    stats_file.write("\n")


def plot_complexity(assays, plotter, sf, stats_file):
    order = ["MAX1MIN1", "MAX1MIN2", "MAX1MIN3"]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    width = 0.35
    x = np.arange(len(order))
    stats_file.write("(H) Rank accuracy by competition complexity:\n")
    for si, (key, label) in enumerate(PRED_SCORES):
        means, ns = [], []
        for ct in order:
            accs = []
            for a in assays:
                if a["class_type"] != ct:
                    continue
                max_s, min_s = assay_score_groups(a, key)
                auc = rank_auc(max_s, min_s)
                if auc is not None:
                    accs.append(auc)
            means.append(np.mean(accs) if accs else 0.0)
            ns.append(len(accs))
        ax.bar(x + (si - 0.5) * width, means, width, label=label)
        stats_file.write(f"  {label}: " + ", ".join(
            f"{ct}: mean acc={m:.3f} (n={n})" for ct, m, n in zip(order, means, ns)) + "\n")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean within-assay rank accuracy")
    ax.set_title("(H) Stability under increasing competition")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plotter.save_plot(f"H_complexity{sf}")
    plt.close(fig)
    stats_file.write("\n")


def plot_length_controlled(all_assays, equal_assays, plotter, stats_file):
    """(E) Compare rank accuracy: all vs equal-RNALength assays, per score."""
    fig, ax = plt.subplots(figsize=(7.6, 5))
    labels = [lab for _, lab in PRED_SCORES]
    x = np.arange(len(PRED_SCORES))
    width = 0.38
    stats_file.write("(E) Length-controlled rank accuracy (all vs equal-RNALength):\n")
    for offset, (subset, name, color) in enumerate(
        [(all_assays, "all assays", MAX_COLOR), (equal_assays, "equal RNALength", ACCENT)]
    ):
        means = []
        for key, label in PRED_SCORES:
            acc = analysis_rank_accuracy(subset, key)
            vals = list(acc.values())
            means.append(np.mean(vals) if vals else 0.0)
            stats_file.write(f"  [{name}] {label}: {fmt_summary(summarize(vals))}\n")
        ax.bar(x + (offset - 0.5) * width, means, width, label=name, color=color, alpha=0.85)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean within-assay rank accuracy")
    ax.set_title("(E) Length-controlled ranking accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plotter.save_plot("E_length_controlled")
    plt.close(fig)
    stats_file.write("\n")


def plot_logistic(assays, plotter, stats_file):
    """(D) Predict correct ranking (per minprobe pair) from job-level + chain-level features.

    Outcome = 1 if the maxprobe outranks this minprobe by af3_rna_ipTM. The ipTM
    *margin* defines the label, so it can't be a feature; instead we include the
    maxprobe's own af3_rna_ipTM (absolute confidence on the true max) as a driver.
    """
    job_feature_names = [
        "RNAmotif_score", "RNAMotif_inExp",
        "af3_protein_ipTM", "af3_protein_pTM", "log10_Binding_affinity_kd",
        "af3_rna_pLDDT_avg", "af3_global_pae_avg",
        "af3_chain_pair_pae_min", "af3_ranking_score",
        "RBP_PDB_Templates_n", "Protein_PDB_Templates_n",
    ]
    feature_names = ["delta_pred_motif_len", "af3_rna_ipTM_maxprobe"] + job_feature_names
    X, y = [], []
    for a in assays:
        maxc = [c for c in a["chains"] if c["class"] == "maxprobe"]
        minc = [c for c in a["chains"] if c["class"] == "minprobe"]
        if not maxc:
            continue
        m = maxc[0]
        for mn in minc:
            if not (np.isfinite(m[PRIMARY_SCORE]) and np.isfinite(mn[PRIMARY_SCORE])):
                continue
            correct = 1 if m[PRIMARY_SCORE] > mn[PRIMARY_SCORE] else 0
            delta_motif = (m["pred_motif_length"] or 0) - (mn["pred_motif_length"] or 0)
            feats = [delta_motif, m["af3_rna_ipTM"]] + [
                _to_float(a["features"].get(fn)) for fn in job_feature_names
            ]
            X.append(feats)
            y.append(correct)
    stats_file.write("(D) Logistic regression: P(maxprobe ranked above a minprobe by AF3 RNA ipTM)\n")
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.size:
        # Drop features that are entirely missing; median-impute sporadic NaNs.
        keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).sum() >= max(20, 0.5 * len(y))]
        feature_names = [feature_names[j] for j in keep]
        X = X[:, keep]
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
            col[~np.isfinite(col)] = med
            X[:, j] = col
    if len(set(y.tolist())) < 2 or len(y) < 20 or X.size == 0:
        stats_file.write(f"  insufficient data (n={len(y)}, classes={set(y.tolist())}); skipped.\n\n")
        return
    coefs, intercept, auc = fit_logistic(X, y)
    stats_file.write(f"  n pairs={len(y)}, positive rate={np.mean(y):.3f}, in-sample AUC={auc:.3f}\n")
    order = np.argsort(-np.abs(coefs))
    for i in order:
        stats_file.write(f"    {feature_names[i]:26s} standardized coef={coefs[i]:+.4f}\n")
    stats_file.write("\n")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    idx = order[::-1]
    colors = [ACCENT if coefs[i] >= 0 else MIN_COLOR for i in idx]
    ax.barh([feature_names[i] for i in idx], [coefs[i] for i in idx], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title(f"(D) Drivers of correct ranking (in-sample AUC={auc:.3f}, n={len(y)})")
    ax.grid(True, ls="--", alpha=0.4, axis="x")
    plt.tight_layout()
    plotter.save_plot("D_logistic_drivers")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_headline(assays, stats_file):
    stats_file.write("=" * 80 + "\n")
    stats_file.write("HEADLINE: per-assay prediction of experimental binding preference\n")
    stats_file.write("=" * 80 + "\n")
    acc = analysis_rank_accuracy(assays, PRIMARY_SCORE)
    vals = np.array(list(acc.values()))
    if vals.size:
        stats_file.write(
            f"Primary predicted score: {dict(PRED_SCORES)[PRIMARY_SCORE]}\n"
            f"  assays evaluated (one per protein): {vals.size}\n"
            f"  maxprobe ranked above ALL minprobes: {np.mean(vals >= 0.999)*100:.1f}% of assays\n"
            f"  mean within-assay rank accuracy: {vals.mean():.3f} (median {np.median(vals):.3f})\n"
        )
    margins = assay_margins(assays, PRIMARY_SCORE, PRIMARY_EXP)
    if len(margins) >= 3:
        xp = np.array([m[0] for m in margins.values()])
        yp = np.array([m[1] for m in margins.values()])
        rho, prho = spearmanr(xp, yp)
        stats_file.write(
            f"  predicted margin vs experimental binding-intensity margin: "
            f"Spearman rho={rho:.3f} (p={prho:.2e}, n={len(margins)})\n"
        )
    stats_file.write("\n")


def run_all(assays, plotter, stats_file, suffix=""):
    acc_by_score = {k: analysis_rank_accuracy(assays, k) for k, _ in PRED_SCORES}
    eff_by_score = {k: analysis_effect_size(assays, k) for k, _ in PRED_SCORES}

    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"(A) Per-assay rank accuracy {'(equal RNALength)' if suffix else '(all assays)'}\n")
    stats_file.write("=" * 80 + "\n")
    for key, label in PRED_SCORES:
        vals = list(acc_by_score[key].values())
        frac_perfect = float(np.mean(np.array(vals) >= 0.999)) if vals else 0.0
        stats_file.write(f"  {label}: {fmt_summary(summarize(vals))}; "
                         f"max>all min in {frac_perfect*100:.1f}% of assays\n")
    stats_file.write("\n")

    stats_file.write("=" * 80 + "\n")
    stats_file.write("(B) Global ROC-AUC (maxprobe=1 vs minprobe=0)\n")
    stats_file.write("=" * 80 + "\n")
    for key, label in PRED_SCORES:
        labels, scores = [], []
        for a in assays:
            for c in a["chains"]:
                if np.isfinite(c[key]):
                    labels.append(1 if c["class"] == "maxprobe" else 0)
                    scores.append(c[key])
        auc = global_auc(labels, scores)
        if auc is not None:
            try:
                u, pmw = mannwhitneyu(
                    [s for s, l in zip(scores, labels) if l == 1],
                    [s for s, l in zip(scores, labels) if l == 0],
                    alternative="greater",
                )
            except ValueError:
                pmw = float("nan")
            stats_file.write(f"  {label}: AUC={auc:.4f}, Mann-Whitney p(max>min)={pmw:.3e}, "
                             f"n_max={sum(labels)}, n_min={len(labels)-sum(labels)}\n")
    stats_file.write("\n")

    stats_file.write("=" * 80 + "\n")
    stats_file.write("(C) Per-assay effect size + Wilcoxon signed-rank (max>min, one-sided)\n")
    stats_file.write("=" * 80 + "\n")
    for key, label in PRED_SCORES:
        vals = list(eff_by_score[key].values())
        w = wilcoxon_greater(vals)
        wtxt = (f"Wilcoxon stat={w['stat']:.4g}, p={w['p']:.3e} (n={w['n']}, nonzero={w['n_nonzero']})"
                if w else "Wilcoxon n/a")
        frac_pos = float(np.mean(np.array(vals) > 0)) if vals else 0.0
        stats_file.write(f"  {label}: {fmt_summary(summarize(vals))}; "
                         f"frac max>min={frac_pos:.3f}; {wtxt}\n")
    stats_file.write("\n")

    plot_rank_accuracy(acc_by_score, plotter, suffix)
    plot_roc(assays, plotter, suffix)
    plot_effect_sizes(eff_by_score, plotter, suffix)
    plot_margins(assays, plotter, suffix, stats_file)
    plot_complexity(assays, plotter, suffix, stats_file)


def assays_equal_rna_length(assays):
    """Subset: assays where every probe chain shares the same experimental RNALength."""
    out = []
    for a in assays:
        lengths = [round(float(c["rna_length"]), 3)
                   for c in a["chains"] if np.isfinite(c["rna_length"])]
        if lengths and len(set(lengths)) == 1:
            out.append(a)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="exp_protein_rna CSV (per-probe experimental values)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots + stats")
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db or StartConfig().get_database_path())
    probe_map = load_exp_csv(args.csv)
    assays, n_proteins = load_probe_records(db_path, probe_map)
    equal_assays = assays_equal_rna_length(assays)

    n_min_total = sum(a["n_min"] for a in assays)
    print(f"competition assays with max+min probes: {len(assays)} (distinct proteins: {n_proteins}, "
          f"minprobes: {n_min_total}); equal-RNALength assays: {len(equal_assays)}")

    plotter = PlotCreator(plot_type=PLOT_TYPE, standalone_outdir=args.outdir)
    plotter.set_plot_style()
    print(f"Saving outputs under {plotter.results_dir}")

    stats_path = os.path.join(plotter.results_dir, STATS_FILENAME)
    with open(stats_path, "w", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write("RNAcompete per-assay maxprobe-vs-minprobe ranking analyses\n")
        sf.write("=" * 80 + "\n\n")
        sf.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        sf.write(f"Database: {db_path}\n")
        sf.write(f"Reference CSV: {args.csv}\n")
        sf.write("Unit of analysis: one in-silico competition assay (1 maxprobe + 1-3 minprobe\n")
        sf.write("RNA chains in one predicted complex). Each protein currently has exactly one\n")
        sf.write("assay, so this is NOT a repeated-measures within-protein design.\n")
        sf.write(f"Competition assays with both max & min probes: {len(assays)}\n")
        sf.write(f"Distinct proteins: {n_proteins}; total minprobes: {n_min_total}\n")
        sf.write(f"Equal-RNALength assays: {len(equal_assays)}\n")
        sf.write(f"Primary predicted score: {dict(PRED_SCORES)[PRIMARY_SCORE]}\n")
        sf.write(f"Primary experimental signal: {dict(EXP_SIGNALS)[PRIMARY_EXP]}\n\n")

        write_headline(assays, sf)
        run_all(assays, plotter, sf, suffix="")
        if equal_assays:
            run_all(equal_assays, plotter, sf, suffix="_equalRNAlength")
        else:
            print("No equal-RNALength assays; skipping that variant.")
        plot_length_controlled(assays, equal_assays, plotter, sf)
        plot_logistic(assays, plotter, sf)

    print(f"Statistics written to: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
