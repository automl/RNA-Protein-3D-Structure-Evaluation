"""
Within-protein RNAcompete maxprobe-vs-minprobe ranking analyses (6-sample dataset).

In this dataset each probe interacts SEPARATELY with the protein: every protein has
6 independent single-RNA complexes - 3 maxprobes and 3 minprobes. Every per-probe
predicted score (af3_rna_ipTM, Binding_affinity_kd, RNAmotif_score,
InterfaceArea_norm, RNAMotif_inExp, predicted motif length) therefore varies
across probes, so this is a genuine within-protein (repeated-measures) design.

Data are loaded by joining pred_protein_rna to exp_protein_rna on PDBId (Class in
{maxprobe, minprobe}) and grouping by ProteinId. The same ranking principles as
rnacompete/probe_ranking_analysis.py are applied, per protein:
  (A) per-protein ranking accuracy (within-protein pairwise AUC, 3 max vs 3 min)
  (B) global + per-protein ROC-AUC (maxprobe=1, minprobe=0)
  (C) per-protein effect-size decomposition + Wilcoxon signed-rank
  (D) logistic regression of correct ranking on per-probe feature differences
  (E) length-controlled analysis (proteins whose 6 probes share one RNA length)
  (F) margin-based analysis (predicted margin vs experimental margin)
  (G) agreement with RNACompete experimental signals
Competition-complexity (H) does not apply here (no MAX1MINk structures).

Usage:
  python3 rnacompete/probe_ranking_within_protein.py
  python3 rnacompete/probe_ranking_within_protein.py --db PATH --outdir PATH
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator
from probe_ranking_analysis import (  # shared, dataset-agnostic helpers
    _counts,
    _log10_kd,
    _scalar_from_cell,
    _to_float,
    base_protein_id,
    fit_logistic,
    fmt_summary,
    global_auc,
    rank_auc,
    roc_curve_points,
    summarize,
    wilcoxon_greater,
)

PLOT_TYPE = "rna_compete_ranking_within_protein"
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_compete_ranking_within_protein")
STATS_FILENAME = "probe_ranking_within_protein_statistics.txt"

MAX_COLOR = "#1976D2"
MIN_COLOR = "#D32F2F"
ACCENT = "#2E7D32"

# Nature-style publication defaults: sans-serif, thin spines, no title (captions
# carry the description), editable text in vector exports.
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _pub_ax(ax, grid_axis=None):
    """Apply Nature-style axis cosmetics: despine top/right, outward ticks, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _panel_letter(ax, letter, dx=-0.1):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")

# Per-probe predicted preference scores with direction (higher => stronger binding).
# Each is oriented at load time so that a larger oriented value always means
# "model prefers this probe".
PRED_SCORES = (
    ("af3_rna_ipTM", "AF3 RNA ipTM", "higher"),
    ("log10_Binding_affinity_kd", "Predicted affinity (-log10 Kd)", "lower"),
    ("RNAmotif_score", "RNA motif score", "higher"),
    ("InterfaceArea_norm", "Normalized interface area", "higher"),
    ("RNAMotif_inExp", "RNA motif in exp", "higher"),
    ("pred_motif_length", "Predicted RNA motif length", "higher"),
)
PRED_LABELS = {k: lab for k, lab, _ in PRED_SCORES}
PRIMARY_SCORE = "af3_rna_ipTM"

EXP_SIGNALS = (
    ("intensity", "Experimental binding intensity"),
    ("zscore", "RNAcompete motif z-score"),
)
EXP_LABELS = dict(EXP_SIGNALS)
PRIMARY_EXP = "intensity"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
JOIN_QUERY = """
    SELECT e.ProteinId, e.PDBId, LOWER(TRIM(e.Class)) AS klass,
           e.BindingIntensity, e.RNAMotif_z_score, e.RNALength AS exp_rna_length,
           p.af3_rna_ipTM, p.af3_protein_ipTM, p.af3_protein_pTM, p.Binding_affinity_kd,
           p.RNAmotif_score, p.InterfaceArea_norm, p.RNAMotif_inExp,
           p.RNAMotifLength, p.RNALength AS pred_rna_length,
           p.af3_rna_pLDDT_avg, p.af3_global_pae_avg, p.af3_chain_pair_pae_min,
           p.af3_ranking_score, p.RBP_PDB_Templates, p.Protein_PDB_Templates
    FROM pred_protein_rna p
    INNER JOIN exp_protein_rna e
        ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))
    WHERE LOWER(TRIM(e.Class)) IN ('maxprobe', 'minprobe')
"""


def _orient(raw_scores):
    """Return oriented preference dict (higher == model prefers this probe)."""
    oriented = {}
    for key, _, direction in PRED_SCORES:
        v = raw_scores.get(key, np.nan)
        if v is None or not np.isfinite(v):
            oriented[key] = np.nan
        else:
            oriented[key] = float(v) if direction == "higher" else -float(v)
    return oriented


def load_proteins(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(JOIN_QUERY)
    rows = cur.fetchall()
    conn.close()

    proteins = {}
    for r in rows:
        (protein_id, pdb_id, klass, intensity, zscore, exp_rna_len,
         iptm, prot_iptm, prot_ptm, kd, rnamotif_score, motif_sim, rnamotif_inexp,
         motif_len, pred_rna_len, rna_plddt, global_pae, chain_pae_min,
         ranking_score, rbp_templates, prot_templates) = r

        pid = (protein_id or base_protein_id(pdb_id)).upper()
        raw_scores = {
            "af3_rna_ipTM": _scalar_from_cell(iptm),
            "log10_Binding_affinity_kd": _log10_kd(kd),
            "RNAmotif_score": _to_float(rnamotif_score),
            "InterfaceArea_norm": _to_float(motif_sim),
            "RNAMotif_inExp": _to_float(rnamotif_inexp),
            "pred_motif_length": _scalar_from_cell(motif_len),
        }
        rna_len = _to_float(exp_rna_len)
        if not np.isfinite(rna_len):
            rna_len = _scalar_from_cell(pred_rna_len)
        probe = {
            "pdb_id": pdb_id,
            "class": klass,
            "oriented": _orient(raw_scores),
            "raw": raw_scores,
            "intensity": _to_float(intensity),
            "zscore": _to_float(zscore),
            "rna_length": rna_len,
            "features": {
                "af3_protein_ipTM": _scalar_from_cell(prot_iptm),
                "af3_protein_pTM": _scalar_from_cell(prot_ptm),
                "af3_rna_pLDDT_avg": _to_float(rna_plddt),
                "af3_global_pae_avg": _to_float(global_pae),
                "af3_chain_pair_pae_min": _to_float(chain_pae_min),
                "af3_ranking_score": _to_float(ranking_score),
                "RBP_PDB_Templates_n": _counts(rbp_templates),
                "Protein_PDB_Templates_n": _counts(prot_templates),
            },
        }
        entry = proteins.setdefault(pid, {"protein": pid, "maxprobes": [], "minprobes": []})
        if klass == "maxprobe":
            entry["maxprobes"].append(probe)
        else:
            entry["minprobes"].append(probe)

    return [p for p in proteins.values() if p["maxprobes"] and p["minprobes"]]


# --------------------------------------------------------------------------- #
# Per-protein aggregation
# --------------------------------------------------------------------------- #
def score_groups(protein, key):
    """Oriented predicted-score lists (max, min) for one protein."""
    mx = [p["oriented"][key] for p in protein["maxprobes"] if np.isfinite(p["oriented"][key])]
    mn = [p["oriented"][key] for p in protein["minprobes"] if np.isfinite(p["oriented"][key])]
    return mx, mn


def exp_groups(protein, key):
    mx = [p[key] for p in protein["maxprobes"] if np.isfinite(p[key])]
    mn = [p[key] for p in protein["minprobes"] if np.isfinite(p[key])]
    return mx, mn


def rank_accuracy(proteins, key):
    out = {}
    for p in proteins:
        mx, mn = score_groups(p, key)
        auc = rank_auc(mx, mn)
        if auc is not None:
            out[p["protein"]] = auc
    return out


def effect_size(proteins, key):
    out = {}
    for p in proteins:
        mx, mn = score_groups(p, key)
        if mx and mn:
            out[p["protein"]] = float(np.mean(mx) - np.mean(mn))
    return out


def margins(proteins, pred_key, exp_key):
    out = {}
    for p in proteins:
        pmx, pmn = score_groups(p, pred_key)
        emx, emn = exp_groups(p, exp_key)
        if pmx and pmn and emx and emn:
            out[p["protein"]] = (max(pmx) - max(pmn), max(emx) - float(np.mean(emn)))
    return out


def proteins_equal_rna_length(proteins):
    out = []
    for p in proteins:
        lengths = [round(float(pr["rna_length"]), 3)
                   for pr in (p["maxprobes"] + p["minprobes"]) if np.isfinite(pr["rna_length"])]
        if lengths and len(set(lengths)) == 1:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _grid(n, ncol=3, panel=(5.0, 4.4)):
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_rank_accuracy(acc_by_score, plotter, sf):
    keys = [k for k, _, _ in PRED_SCORES if acc_by_score.get(k)]
    if not keys:
        return
    with plt.rc_context(PUB_RC):
        fig, axes = _grid(len(keys))
        for ax, key in zip(axes, keys):
            vals = np.array(list(acc_by_score[key].values()))
            ax.hist(vals, bins=np.linspace(0, 1, 21), color=MAX_COLOR, alpha=0.85, edgecolor="white")
            ax.axvline(vals.mean(), color=ACCENT, lw=2, label=f"mean={vals.mean():.2f}")
            ax.axvline(np.median(vals), color="black", lw=1.5, ls="--", label=f"median={np.median(vals):.2f}")
            frac_perfect = float(np.mean(vals >= 0.999))
            ax.text(0.03, 0.97, f"{PRED_LABELS[key]}\nmax>all min: {frac_perfect*100:.0f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8)
            ax.set_xlabel("Within-protein rank accuracy")
            ax.set_ylabel("Proteins")
            ax.legend(fontsize=7, loc="upper right")
            _pub_ax(ax, grid_axis="y")
        fig.tight_layout()
        plotter.save_plot(f"A_rank_accuracy{sf}")
        plt.close(fig)


def plot_roc(proteins, plotter, sf):
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        for key, label, _ in PRED_SCORES:
            labels, scores = [], []
            for p in proteins:
                for pr in p["maxprobes"] + p["minprobes"]:
                    v = pr["oriented"][key]
                    if np.isfinite(v):
                        labels.append(1 if pr["class"] == "maxprobe" else 0)
                        scores.append(v)
            if not labels:
                continue
            fpr, tpr = roc_curve_points(labels, scores)
            auc = global_auc(labels, scores)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        _pub_ax(ax)
        fig.tight_layout()
        plotter.save_plot(f"B_global_roc{sf}")
        plt.close(fig)


def plot_per_protein_auc(acc_by_score, plotter, sf):
    """(B) Distribution of per-protein AUC (== within-protein rank accuracy) per score."""
    keys = [k for k, _, _ in PRED_SCORES if acc_by_score.get(k)]
    if not keys:
        return
    data = [list(acc_by_score[k].values()) for k in keys]
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(1.5 * len(keys) + 2.5, 4.6))
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(MAX_COLOR)
            patch.set_alpha(0.55)
        for i, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, color=MAX_COLOR, alpha=0.35,
                       edgecolor="none", zorder=2)
            ax.scatter([i], [np.mean(vals)], marker="D", color=ACCENT, s=55, zorder=3,
                       label="mean" if i == 1 else None)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels([PRED_LABELS[k] for k in keys], rotation=20, ha="right")
        ax.set_ylabel("Per-protein AUC (within-protein)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        _pub_ax(ax, grid_axis="y")
        fig.tight_layout()
        plotter.save_plot(f"B_per_protein_auc{sf}")
        plt.close(fig)


def plot_effect_sizes(eff_by_score, plotter, sf):
    keys = [k for k, _, _ in PRED_SCORES if eff_by_score.get(k)]
    if not keys:
        return
    with plt.rc_context(PUB_RC):
        fig, axes = _grid(len(keys))
        for ax, key in zip(axes, keys):
            vals = np.array(list(eff_by_score[key].values()))
            ax.hist(vals, bins=25, color=MAX_COLOR, alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", lw=1.5)
            ax.axvline(vals.mean(), color=ACCENT, lw=2, label=f"mean={vals.mean():.3g}")
            frac_pos = float(np.mean(vals > 0))
            ax.text(0.03, 0.97, f"{PRED_LABELS[key]}\nmax>min: {frac_pos*100:.0f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8)
            ax.set_xlabel(r"$\Delta S$ = mean(max) - mean(min), oriented")
            ax.set_ylabel("Proteins")
            ax.legend(fontsize=7, loc="upper right")
            _pub_ax(ax, grid_axis="y")
        fig.tight_layout()
        plotter.save_plot(f"C_effect_size{sf}")
        plt.close(fig)


def plot_margins(proteins, plotter, sf, stats_file):
    npred = len(PRED_SCORES)
    nexp = len(EXP_SIGNALS)
    stats_file.write("(F/G) Prediction margin vs experimental margin (Spearman / Pearson):\n")
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(npred, nexp, figsize=(4.0 * nexp, 3.4 * npred), squeeze=False)
        for ri, (pred_key, pred_label, _) in enumerate(PRED_SCORES):
            for ci, (exp_key, exp_label) in enumerate(EXP_SIGNALS):
                ax = axes[ri][ci]
                mg = margins(proteins, pred_key, exp_key)
                if len(mg) < 3:
                    ax.set_visible(False)
                    continue
                xp = np.array([m[0] for m in mg.values()])
                yp = np.array([m[1] for m in mg.values()])
                ax.scatter(xp, yp, s=20, alpha=0.6, color=MAX_COLOR, edgecolor="white", linewidth=0.4)
                rho, prho = spearmanr(xp, yp)
                r, pr = pearsonr(xp, yp)
                ax.text(0.03, 0.97, f"{pred_label} vs {exp_label}\nrho={rho:.2f}, r={r:.2f}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=7)
                ax.set_xlabel(f"Predicted margin ({pred_label})", fontsize=8)
                ax.set_ylabel(f"Exp margin ({exp_label})", fontsize=8)
                _pub_ax(ax)
                stats_file.write(
                    f"  {exp_label} vs {pred_label}: n={len(mg)}, "
                    f"Spearman rho={rho:.4f} (p={prho:.3e}), Pearson r={r:.4f} (p={pr:.3e})\n"
                )
        fig.tight_layout()
        plotter.save_plot(f"FG_margin_correlation{sf}")
        plt.close(fig)
    stats_file.write("\n")


def plot_length_controlled(all_p, equal_p, plotter, stats_file):
    labels = [lab for _, lab, _ in PRED_SCORES]
    x = np.arange(len(PRED_SCORES))
    width = 0.38
    stats_file.write("(E) Length-controlled rank accuracy (all vs equal-RNALength):\n")
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for offset, (subset, name, color) in enumerate(
            [(all_p, "all proteins", MAX_COLOR), (equal_p, "equal RNALength", ACCENT)]
        ):
            means = []
            for key, label, _ in PRED_SCORES:
                vals = list(rank_accuracy(subset, key).values())
                means.append(np.mean(vals) if vals else 0.0)
                stats_file.write(f"  [{name}] {label}: {fmt_summary(summarize(vals))}\n")
            ax.bar(x + (offset - 0.5) * width, means, width, label=name, color=color)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Mean within-protein rank accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        _pub_ax(ax, grid_axis="y")
        fig.tight_layout()
        plotter.save_plot("E_length_controlled")
        plt.close(fig)
    stats_file.write("\n")


def plot_per_protein_auc_length_comparison(all_p, equal_p, plotter, stats_file):
    """(E) Per-protein AUC distribution: all proteins vs equal-RNALength, side by side."""
    keys = [k for k, _, _ in PRED_SCORES]
    cohorts = [("all proteins", all_p, MAX_COLOR), ("equal RNALength", equal_p, ACCENT)]
    cohorts = [(name, sub, color) for name, sub, color in cohorts if sub]
    if len(cohorts) < 2:
        return
    n_out = len(keys)
    n_coh = len(cohorts)
    width = 0.8 / n_coh
    fig, ax = plt.subplots(figsize=(1.9 * n_out + 4, 6.0))
    rng = np.random.default_rng(0)
    stats_file.write("(E) Per-protein AUC distribution (all vs equal-RNALength):\n")
    for ci, (name, subset, color) in enumerate(cohorts):
        positions = np.arange(1, n_out + 1) + (ci - (n_coh - 1) / 2.0) * width
        data = [list(rank_accuracy(subset, k).values()) for k in keys]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.3))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for oi, vals in enumerate(data):
            if not vals:
                continue
            jitter = rng.normal(0, width * 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), positions[oi]) + jitter, vals, s=6,
                       color=color, alpha=0.3, edgecolor="none", zorder=2)
        ax.scatter([], [], color=color, alpha=0.6, label=f"{name} (n={len(subset)} proteins)")
        for key in keys:
            vals = list(rank_accuracy(subset, key).values())
            stats_file.write(f"  [{name}] {PRED_LABELS[key]}: {fmt_summary(summarize(vals))}\n")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(np.arange(1, n_out + 1))
    ax.set_xticklabels([PRED_LABELS[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Per-protein AUC (maxprobe vs minprobe)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("B_per_protein_auc_length_comparison")
    plt.close(fig)
    stats_file.write("\n")


def _global_auc_for_score(proteins, key):
    """Global AUC pooling every probe across all proteins (maxprobe=1, minprobe=0)."""
    labels, scores = [], []
    for p in proteins:
        for pr in p["maxprobes"] + p["minprobes"]:
            v = pr["oriented"].get(key, np.nan)
            if np.isfinite(v):
                labels.append(1 if pr["class"] == "maxprobe" else 0)
                scores.append(v)
    return global_auc(labels, scores) if labels else None


def plot_per_protein_vs_global_auc(proteins, plotter, stats_file):
    """Within-protein per-protein AUC distribution vs global (pooled) AUC, per score."""
    keys = [k for k, _, _ in PRED_SCORES]
    per_protein = {k: list(rank_accuracy(proteins, k).values()) for k in keys}
    keys = [k for k in keys if per_protein[k]]
    if not keys:
        return
    data = [per_protein[k] for k in keys]
    global_aucs = [_global_auc_for_score(proteins, k) for k in keys]

    fig, ax = plt.subplots(figsize=(1.7 * len(keys) + 3, 5.6))
    bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(MAX_COLOR)
        patch.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color=MAX_COLOR, alpha=0.3,
                   edgecolor="none", zorder=2)
        ax.scatter([i], [np.mean(vals)], marker="D", color=ACCENT, s=55, zorder=3,
                   label="per-protein mean" if i == 1 else None)
    for i, g in enumerate(global_aucs, start=1):
        if g is not None:
            ax.scatter([i], [g], marker="*", color="#D32F2F", s=180, zorder=4,
                       edgecolor="black", linewidth=0.5,
                       label="global AUC (pooled)" if i == 1 else None)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([PRED_LABELS[k] for k in keys], rotation=20, ha="right")
    ax.set_ylabel("AUC (maxprobe vs minprobe)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("B_per_protein_vs_global_auc")
    plt.close(fig)

    stats_file.write("Per-protein (within) vs global (pooled) separation AUC:\n")
    for k, g in zip(keys, global_aucs):
        g_txt = f"{g:.4f}" if g is not None else "n/a"
        stats_file.write(f"  {PRED_LABELS[k]}: per-protein {fmt_summary(summarize(per_protein[k]))}; "
                         f"global pooled AUC={g_txt}\n")
    stats_file.write("\n")


# AF3 confidence deltas used as predictors in (D). Each outcome below is a NON-AF3
# score, so AF3 metrics carry no circularity here. Template counts are dropped (they
# are protein-level constants, so within-protein deltas are ~0) and af3_protein_ipTM
# is dropped because it is effectively identical to af3_rna_ipTM.
D_FEATURE_KEYS = (
    "af3_rna_ipTM",
    "af3_protein_pTM",
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_ranking_score",
)
# Non-AF3 ranking outcomes to be explained by the AF3 + template deltas.
D_OUTCOMES = (
    ("log10_Binding_affinity_kd", "Predicted affinity (-log10 Kd)"),
    ("RNAmotif_score", "RNA motif score"),
    ("InterfaceArea_norm", "Normalized interface area"),
    ("pred_motif_length", "Predicted RNA motif length"),
)


def _feat_value(probe, key):
    """AF3 ipTM lives in raw scores; remaining AF3/template features in `features`."""
    if key in probe["raw"]:
        return probe["raw"].get(key, np.nan)
    return probe["features"].get(key, np.nan)


def _fit_ranking_drivers(proteins, outcome_key, outcome_label, plotter, stats_file):
    """One (D) panel: P(maxprobe outranks minprobe by `outcome_key`) ~ AF3+template deltas."""
    feature_names = [f"delta_{k}" for k in D_FEATURE_KEYS]
    X, y = [], []
    for p in proteins:
        for m in p["maxprobes"]:
            if not np.isfinite(m["oriented"][outcome_key]):
                continue
            for mn in p["minprobes"]:
                if not np.isfinite(mn["oriented"][outcome_key]):
                    continue
                correct = 1 if m["oriented"][outcome_key] > mn["oriented"][outcome_key] else 0
                X.append([_feat_value(m, k) - _feat_value(mn, k) for k in D_FEATURE_KEYS])
                y.append(correct)

    stats_file.write(f"(D) Logistic regression: P(maxprobe outranks minprobe by {outcome_label})\n")
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.size:
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
    coefs, _, auc = fit_logistic(X, y)
    stats_file.write(f"  n pairs={len(y)}, positive rate={np.mean(y):.3f}, in-sample AUC={auc:.3f}\n")
    order = np.argsort(-np.abs(coefs))
    for i in order:
        stats_file.write(f"    {feature_names[i]:28s} standardized coef={coefs[i]:+.4f}\n")
    stats_file.write("\n")

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        idx = order[::-1]
        colors = [ACCENT if coefs[i] >= 0 else MIN_COLOR for i in idx]
        ax.barh([feature_names[i] for i in idx], [coefs[i] for i in idx], color=colors)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Standardized logistic-regression coefficient")
        ax.text(0.97, 0.04, f"outcome: {outcome_label}\nin-sample AUC={auc:.3f}, n={len(y)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        _pub_ax(ax, grid_axis="x")
        fig.tight_layout()
        plotter.save_plot(f"D_logistic_{outcome_key}")
        plt.close(fig)


def plot_logistic(proteins, plotter, stats_file):
    """(D) For each non-AF3 ranking outcome, explain it with AF3 + template deltas."""
    for outcome_key, outcome_label in D_OUTCOMES:
        _fit_ranking_drivers(proteins, outcome_key, outcome_label, plotter, stats_file)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_headline(proteins, stats_file):
    stats_file.write("=" * 80 + "\n")
    stats_file.write("HEADLINE: within-protein prediction of experimental binding preference\n")
    stats_file.write("=" * 80 + "\n")
    acc = rank_accuracy(proteins, PRIMARY_SCORE)
    vals = np.array(list(acc.values()))
    if vals.size:
        stats_file.write(
            f"Primary predicted score: {PRED_LABELS[PRIMARY_SCORE]}\n"
            f"  proteins evaluated: {vals.size}\n"
            f"  maxprobe ranked above ALL minprobes: {np.mean(vals >= 0.999)*100:.1f}% of proteins\n"
            f"  mean within-protein rank accuracy: {vals.mean():.3f} (median {np.median(vals):.3f})\n"
        )
    mg = margins(proteins, PRIMARY_SCORE, PRIMARY_EXP)
    if len(mg) >= 3:
        xp = np.array([m[0] for m in mg.values()])
        yp = np.array([m[1] for m in mg.values()])
        rho, prho = spearmanr(xp, yp)
        stats_file.write(
            f"  predicted margin vs experimental binding-intensity margin: "
            f"Spearman rho={rho:.3f} (p={prho:.2e}, n={len(mg)})\n"
        )
    stats_file.write("\n")


def run_all(proteins, plotter, stats_file, suffix=""):
    acc_by_score = {k: rank_accuracy(proteins, k) for k, _, _ in PRED_SCORES}
    eff_by_score = {k: effect_size(proteins, k) for k, _, _ in PRED_SCORES}

    variant = "(equal RNALength)" if suffix else "(all proteins)"
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"(A) Per-protein rank accuracy {variant}\n")
    stats_file.write("=" * 80 + "\n")
    for key, label, _ in PRED_SCORES:
        vals = list(acc_by_score[key].values())
        frac_perfect = float(np.mean(np.array(vals) >= 0.999)) if vals else 0.0
        stats_file.write(f"  {label}: {fmt_summary(summarize(vals))}; "
                         f"max>all min in {frac_perfect*100:.1f}% of proteins\n")
    stats_file.write("\n")

    stats_file.write("=" * 80 + "\n")
    stats_file.write("(B) Global ROC-AUC (maxprobe=1 vs minprobe=0)\n")
    stats_file.write("=" * 80 + "\n")
    for key, label, _ in PRED_SCORES:
        labels, scores = [], []
        for p in proteins:
            for pr in p["maxprobes"] + p["minprobes"]:
                v = pr["oriented"][key]
                if np.isfinite(v):
                    labels.append(1 if pr["class"] == "maxprobe" else 0)
                    scores.append(v)
        auc = global_auc(labels, scores)
        if auc is not None:
            try:
                _, pmw = mannwhitneyu(
                    [s for s, l in zip(scores, labels) if l == 1],
                    [s for s, l in zip(scores, labels) if l == 0],
                    alternative="greater",
                )
            except ValueError:
                pmw = float("nan")
            stats_file.write(f"  {label}: AUC={auc:.4f}, Mann-Whitney p(max>min)={pmw:.3e}, "
                             f"n_max={sum(labels)}, n_min={len(labels)-sum(labels)}\n")
    stats_file.write("\n")

    stats_file.write("(B) Per-protein AUC (within-protein; == per-protein rank accuracy):\n")
    for key, label, _ in PRED_SCORES:
        vals = list(acc_by_score[key].values())
        stats_file.write(f"  {label}: {fmt_summary(summarize(vals))}\n")
    stats_file.write("\n")

    stats_file.write("=" * 80 + "\n")
    stats_file.write("(C) Per-protein effect size + Wilcoxon signed-rank (max>min, one-sided)\n")
    stats_file.write("=" * 80 + "\n")
    for key, label, _ in PRED_SCORES:
        vals = list(eff_by_score[key].values())
        w = wilcoxon_greater(vals)
        wtxt = (f"Wilcoxon stat={w['stat']:.4g}, p={w['p']:.3e} (n={w['n']}, nonzero={w['n_nonzero']})"
                if w else "Wilcoxon n/a")
        frac_pos = float(np.mean(np.array(vals) > 0)) if vals else 0.0
        stats_file.write(f"  {label}: {fmt_summary(summarize(vals))}; "
                         f"frac max>min={frac_pos:.3f}; {wtxt}\n")
    stats_file.write("\n")

    plot_rank_accuracy(acc_by_score, plotter, suffix)
    plot_roc(proteins, plotter, suffix)
    plot_per_protein_auc(acc_by_score, plotter, suffix)
    plot_effect_sizes(eff_by_score, plotter, suffix)
    plot_margins(proteins, plotter, suffix, stats_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots + stats")
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db or StartConfig().get_database_path())
    proteins = load_proteins(db_path)
    equal_proteins = proteins_equal_rna_length(proteins)

    n_max = sum(len(p["maxprobes"]) for p in proteins)
    n_min = sum(len(p["minprobes"]) for p in proteins)
    print(f"proteins with max+min probes: {len(proteins)} (maxprobes: {n_max}, minprobes: {n_min}); "
          f"equal-RNALength proteins: {len(equal_proteins)}")

    plotter = PlotCreator(plot_type=PLOT_TYPE, standalone_outdir=args.outdir)
    plotter.set_plot_style()
    plt.rcParams.update(PUB_RC)  # Nature-style defaults for any non-rc_context plots
    print(f"Saving outputs under {plotter.results_dir}")

    stats_path = os.path.join(plotter.results_dir, STATS_FILENAME)
    with open(stats_path, "w", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write("RNAcompete within-protein maxprobe-vs-minprobe ranking analyses\n")
        sf.write("=" * 80 + "\n\n")
        sf.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        sf.write(f"Database: {db_path}\n")
        sf.write("Unit of analysis: protein. Each protein has independent single-RNA complexes\n")
        sf.write("(typically 3 maxprobes + 3 minprobes) -> a genuine within-protein design.\n")
        sf.write(f"Proteins with both max & min probes: {len(proteins)}\n")
        sf.write(f"Total maxprobes: {n_max}; total minprobes: {n_min}\n")
        sf.write(f"Equal-RNALength proteins: {len(equal_proteins)}\n")
        sf.write(f"Primary predicted score: {PRED_LABELS[PRIMARY_SCORE]}\n")
        sf.write(f"Primary experimental signal: {EXP_LABELS[PRIMARY_EXP]}\n")
        sf.write("Scores are oriented so higher == model prefers the probe "
                 "(Kd uses -log10, lower Kd = stronger).\n\n")

        write_headline(proteins, sf)
        run_all(proteins, plotter, sf, suffix="")
        if equal_proteins:
            run_all(equal_proteins, plotter, sf, suffix="_equalRNAlength")
        else:
            print("No equal-RNALength proteins; skipping that variant.")
        plot_length_controlled(proteins, equal_proteins, plotter, sf)
        plot_per_protein_auc_length_comparison(proteins, equal_proteins, plotter, sf)
        plot_per_protein_vs_global_auc(proteins, plotter, sf)
        plot_logistic(proteins, plotter, sf)

    print(f"Statistics written to: {stats_path}")
    return 0


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
COMPOSITE_PANEL_NAMES = (
    "B_per_protein_auc",
    "B_global_roc",
    "A_rank_accuracy",
)


def _draw_per_protein_auc(acc_by_score):
    keys = [k for k, _, _ in PRED_SCORES if acc_by_score.get(k)]
    data = [list(acc_by_score[k].values()) for k in keys]

    def draw(ax):
        if not keys:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="grey")
            return
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(MAX_COLOR)
            patch.set_alpha(0.55)
        for i, vals in enumerate(data, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, color=MAX_COLOR,
                       alpha=0.35, edgecolor="none", zorder=2)
            ax.scatter([i], [np.mean(vals)], marker="D", color=ACCENT, s=55,
                       zorder=3, label="mean" if i == 1 else None)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels([PRED_LABELS[k] for k in keys], rotation=20, ha="right")
        ax.set_ylabel("Per-protein AUC (within-protein)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        _pub_ax(ax, grid_axis="y")
    return draw


def _draw_global_roc(proteins):
    def draw(ax):
        for key, label, _ in PRED_SCORES:
            labels, scores = [], []
            for p in proteins:
                for pr in p["maxprobes"] + p["minprobes"]:
                    v = pr["oriented"][key]
                    if np.isfinite(v):
                        labels.append(1 if pr["class"] == "maxprobe" else 0)
                        scores.append(v)
            if not labels:
                continue
            fpr, tpr = roc_curve_points(labels, scores)
            auc = global_auc(labels, scores)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7)
        _pub_ax(ax)
    return draw


def _draw_rank_accuracy_primary(acc_by_score):
    """Histogram of within-protein rank accuracy for the primary score."""
    vals = np.array(list(acc_by_score.get(PRIMARY_SCORE, {}).values()))

    def draw(ax):
        if not len(vals):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="grey")
            return
        ax.hist(vals, bins=np.linspace(0, 1, 21), color=MAX_COLOR, alpha=0.85,
                edgecolor="white")
        ax.axvline(vals.mean(), color=ACCENT, lw=2, label=f"mean={vals.mean():.2f}")
        ax.axvline(np.median(vals), color="black", lw=1.5, ls="--",
                   label=f"median={np.median(vals):.2f}")
        ax.set_xlabel(f"Within-protein rank accuracy\n({PRED_LABELS[PRIMARY_SCORE]})")
        ax.set_ylabel("Proteins")
        ax.legend(fontsize=7, loc="upper left")
        _pub_ax(ax, grid_axis="y")
    return draw


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None):
    """Return {base_name: (draw_fn, title)} for plots.composite_figures."""
    from plots.publication_figure import normalize_panel_name
    wanted = ([normalize_panel_name(n) for n in names]
              if names else list(COMPOSITE_PANEL_NAMES))

    db_path = db_path or StartConfig().get_database_path()
    proteins = load_proteins(db_path)
    acc_by_score = {k: rank_accuracy(proteins, k) for k, _, _ in PRED_SCORES}

    registry = {
        "B_per_protein_auc": (_draw_per_protein_auc(acc_by_score),
                              "Per-protein ranking AUC"),
        "B_global_roc": (_draw_global_roc(proteins),
                         "Global maxprobe-vs-minprobe ROC"),
        "A_rank_accuracy": (_draw_rank_accuracy_primary(acc_by_score),
                            "Within-protein rank accuracy"),
    }
    return {n: registry[n] for n in wanted if n in registry}


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="probe_ranking_composite"))
    raise SystemExit(main())
