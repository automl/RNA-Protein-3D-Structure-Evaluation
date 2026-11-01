"""
RNAcompete AAMotif-vs-domain coverage analysis.

For every protein-RNA complex we ask: of the protein residues that AlphaFold3
places at the RNA interface (``pred_protein_rna.AAMotif``, e.g.
``"ILE(128), ASN(131), GLY(132)"``), how many fall inside an annotated
interacting domain range (``exp_protein_rna.domain_pos``, e.g.
``"IPR000504:48-125,130-207,225-292"``)?

Two questions are answered:
  1. Domain coverage of the predicted interface residues (mirrors the
     domain-coverage logic in evaluation/plotResults.py): per-domain within /
     partial / outside counts and a per-sample category
     (all_in_all / partial_in_all / all_in_one / partial_in_one / outside_all).
  2. How those hits scale with probe class (maxprobe vs minprobe) and with the
     per-probe RNA-motif metrics: binding affinity (log10 Kd), RNA motif score
     and motif similarity score.

Usage:
  python3 rnacompete/domain_analysis.py
  python3 rnacompete/domain_analysis.py --db PATH --outdir PATH
"""

import argparse
import os
import re
import sqlite3
import sys
from ast import literal_eval
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, wilcoxon

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator
from probe_ranking_analysis import (
    _log10_kd, _scalar_from_cell, _to_float, base_protein_id, fmt_summary, rank_auc, summarize,
)

PLOT_TYPE = "rna_compete_domain"
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_compete_domain")
STATS_FILENAME = "domain_analysis_statistics.txt"

MAX_COLOR = "#1976D2"
MIN_COLOR = "#D32F2F"
ACCENT = "#2E7D32"

# Nature-style publication defaults: sans-serif, thin spines, no title (captions
# carry the description), editable text in vector exports.
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
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


# Per-probe metrics (besides domain coverage). Direction note: lower log10 Kd =
# stronger predicted binding; the others are "higher = stronger".
METRICS = (
    ("log10_kd", "Predicted affinity (log10 Kd)"),
    ("RNAmotif_score", "RNA motif score"),
    ("InterfaceArea_norm", "Normalized interface area"),
)
METRIC_LABELS = dict(METRICS)
# Orientation for max/min separation: higher oriented value == maxprobe-preferred.
METRIC_DIRECTION = {
    "log10_kd": "lower",  # lower Kd = stronger binding
    "RNAmotif_score": "higher",
    "InterfaceArea_norm": "higher",
    "af3_rna_ipTM": "higher",
}

# Map InterPro domain names to broad RNA-binding-domain families (keyword match,
# first hit wins). Anything unmatched is "Other".
DOMAIN_FAMILY_RULES = (
    ("RRM", ("rrm",)),
    ("KH", ("kh_dom", "kh_", "_kh", "fmrp_kh")),
    ("PUM-HD", ("pum", "pumilio")),
    ("Zinc finger", ("znf", "zinc", "ranbp2", "cccch", "ccch", "zn_")),
    ("CSD", ("csd", "cold_shock", "cold-shock")),
    ("SAM", ("sam",)),
    ("dsRBD", ("dsrbd", "dsrm", "ds_rna", "dsrna")),
)

# Methods to compare within each domain family: one AF3 structural score vs the
# motif-based scores. direction: higher oriented value == maxprobe-preferred.
HIERARCHY_METHODS = (
    ("af3_rna_ipTM", "higher", "AF3 RNA ipTM", "#1976D2"),
    ("RNAmotif_score", "higher", "RNA motif score", "#2E7D32"),
    ("InterfaceArea_norm", "higher", "Normalized interface area", "#F9A825"),
)


def _family_of_name(name):
    n = str(name).lower()
    for family, keys in DOMAIN_FAMILY_RULES:
        if any(k in n for k in keys):
            return family
    return "Other"


def protein_domain_family(domain_map):
    """Assign a protein to one RBD family from its domain names ('Mixed' if >1)."""
    fams = {_family_of_name(v) for v in domain_map.values()}
    fams.discard("Other")
    if len(fams) == 1:
        return next(iter(fams))
    if len(fams) > 1:
        return "Mixed"
    return "Other"

CATEGORIES = ("all_in_all", "partial_in_all", "all_in_one", "partial_in_one", "outside_all")
CATEGORY_LABELS = {
    "all_in_all": "All residues in all domains",
    "partial_in_all": "Some residues in all domains",
    "all_in_one": "All residues in one domain",
    "partial_in_one": "Some residues in one domain",
    "outside_all": "No residues in any domain",
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_domain_ranges(domain_ids, domain_pos):
    """Return {domain_id: [(start, end), ...]} from the ';'-separated columns.

    Mirrors evaluation/plotResults.py: each domain position may carry an
    ``id:`` prefix and one or more comma-separated ``start-end`` ranges.
    """
    if not domain_ids or not domain_pos:
        return {}
    id_list = str(domain_ids).split(";")
    pos_list = str(domain_pos).split(";")
    if all(p.strip() == "-" for p in pos_list):
        return {}
    ranges = {}
    for d_id, d_pos in zip(id_list, pos_list):
        d_id = d_id.strip()
        d_pos = d_pos.strip()
        if d_pos == "-":
            continue
        range_str = d_pos.split(":", 1)[1] if ":" in d_pos else d_pos
        spans = []
        for r in range_str.split(","):
            r = r.strip()
            try:
                start, end = map(int, r.split("-"))
                spans.append((start, end))
            except ValueError:
                continue
        if spans:
            ranges[d_id] = spans
    return ranges


def parse_aa_indices(aa_motif):
    """Pool all residue indices in AAMotif (numbers inside parentheses)."""
    if aa_motif is None:
        return []
    if isinstance(aa_motif, str) and aa_motif.startswith("[") and aa_motif.endswith("]"):
        try:
            groups = literal_eval(aa_motif)
        except (ValueError, SyntaxError):
            groups = [aa_motif]
    else:
        groups = [aa_motif]
    if not isinstance(groups, (list, tuple)):
        groups = [groups]
    indices = []
    for g in groups:
        indices.extend(int(n) for n in re.findall(r"\((\d+)\)", str(g)))
    return indices


def domain_map_from(domain_ids, domain_names):
    if not domain_ids or not domain_names:
        return {}
    ids = str(domain_ids).split(";")
    names = str(domain_names).split(";")
    if len(ids) != len(names):
        return {}
    return {i.strip(): n.strip() for i, n in zip(ids, names)}


# --------------------------------------------------------------------------- #
# Coverage computation
# --------------------------------------------------------------------------- #
def categorize(aa_indices, domain_ranges):
    """Return (category, coverage_fraction, n_hits, per_domain) for one sample.

    per_domain maps domain_id -> ("within"|"partial"|"outside", fraction).
    Coverage fraction is the share of AAMotif residues inside ANY domain range.
    """
    n_aa = len(aa_indices)
    if n_aa == 0 or not domain_ranges:
        return None
    in_any = set()
    per_domain = {}
    domains_with_all = domains_with_partial = 0
    for d_id, spans in domain_ranges.items():
        inside = [idx for idx in aa_indices if any(s <= idx <= e for s, e in spans)]
        in_any.update(inside)
        frac = len(inside) / n_aa
        if len(inside) == n_aa:
            per_domain[d_id] = ("within", frac)
            domains_with_all += 1
        elif inside:
            per_domain[d_id] = ("partial", frac)
            domains_with_partial += 1
        else:
            per_domain[d_id] = ("outside", frac)

    total = len(domain_ranges)
    if total > 1:
        if domains_with_all == total:
            cat = "all_in_all"
        elif domains_with_partial == total and domains_with_all == 0:
            cat = "partial_in_all"
        elif domains_with_all > 0:
            cat = "all_in_one"
        elif domains_with_partial > 0:
            cat = "partial_in_one"
        else:
            cat = "outside_all"
    else:
        if domains_with_all == 1:
            cat = "all_in_one"
        elif domains_with_partial == 1:
            cat = "partial_in_one"
        else:
            cat = "outside_all"

    coverage = len(in_any) / n_aa
    return cat, coverage, len(in_any), per_domain


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
JOIN_QUERY = """
    SELECT e.PDBId, e.ProteinId, LOWER(TRIM(e.Class)) AS klass,
           e.domain_ids, e.domain_names, e.domain_pos,
           p.AAMotif, p.Binding_affinity_kd, p.RNAmotif_score, p.InterfaceArea_norm,
           p.af3_rna_ipTM
    FROM exp_protein_rna e
    INNER JOIN pred_protein_rna p
        ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))
    WHERE e.domain_pos IS NOT NULL AND p.AAMotif IS NOT NULL
      AND LOWER(TRIM(e.Class)) IN ('maxprobe', 'minprobe')
"""


def load_samples(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(JOIN_QUERY)
    rows = cur.fetchall()
    conn.close()

    samples = []
    for (pdb_id, protein_id, klass, domain_ids, domain_names, domain_pos,
         aa_motif, kd, rnamotif_score, motif_sim, af3_rna_iptm) in rows:
        domain_ranges = parse_domain_ranges(domain_ids, domain_pos)
        aa_indices = parse_aa_indices(aa_motif)
        result = categorize(aa_indices, domain_ranges)
        if result is None:
            continue
        cat, coverage, n_hits, per_domain = result
        domain_map = domain_map_from(domain_ids, domain_names)
        samples.append({
            "pdb_id": pdb_id,
            "protein": (protein_id or base_protein_id(pdb_id)).upper(),
            "class": klass,
            "domain_map": domain_map,
            "domain_family": protein_domain_family(domain_map),
            "per_domain": per_domain,
            "n_domains": len(domain_ranges),
            "n_aa": len(aa_indices),
            "n_hits": n_hits,
            "coverage": coverage,
            "category": cat,
            "log10_kd": _log10_kd(kd),
            "RNAmotif_score": _to_float(rnamotif_score),
            "InterfaceArea_norm": _to_float(motif_sim),
            "af3_rna_ipTM": _scalar_from_cell(af3_rna_iptm),
        })
    return samples


def split_by_class(samples):
    mx = [s for s in samples if s["class"] == "maxprobe"]
    mn = [s for s in samples if s["class"] == "minprobe"]
    return mx, mn


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_category_distribution(samples, plotter, stats_file):
    """Stacked-bar share of per-sample coverage categories, split by class."""
    mx, mn = split_by_class(samples)
    groups = [("maxprobe", mx, MAX_COLOR), ("minprobe", mn, MIN_COLOR), ("all", samples, ACCENT)]
    stats_file.write("Coverage category breakdown (share of samples):\n")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    palette = ["#1B5E20", "#66BB6A", "#FBC02D", "#FB8C00", "#C62828"]
    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))
    for ci, cat in enumerate(CATEGORIES):
        fracs = []
        for _, grp, _ in groups:
            n = len(grp)
            fracs.append(100.0 * sum(1 for s in grp if s["category"] == cat) / n if n else 0.0)
        ax.bar(x, fracs, bottom=bottoms, color=palette[ci], label=CATEGORY_LABELS[cat])
        bottoms += np.array(fracs)
    for gi, (name, grp, _) in enumerate(groups):
        n = len(grp)
        stats_file.write(f"  {name} (n={n}):\n")
        for cat in CATEGORIES:
            c = sum(1 for s in grp if s["category"] == cat)
            stats_file.write(f"    {CATEGORY_LABELS[cat]:32s} {c:5d} ({100.0*c/n if n else 0:.1f}%)\n")
    stats_file.write("\n")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n={len(grp)})" for name, grp, _ in groups])
    ax.set_ylabel("Percentage of samples (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.0, 0.5))
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("domain_category_distribution")
    plt.close(fig)


def plot_domain_coverage_distribution(samples, plotter, stats_file=None):
    """Per-domain within / partial / outside stacked bars (mirrors plotResults)."""
    types = {}
    names = {}
    for s in samples:
        for d_id, (kind, _) in s["per_domain"].items():
            types.setdefault(d_id, {"within": 0, "partial": 0, "outside": 0})[kind] += 1
            names.setdefault(d_id, s["domain_map"].get(d_id, d_id))
    if not types:
        if stats_file is not None:
            stats_file.write("Per-domain AAMotif coverage: no domain data.\n\n")
        return
    domains = sorted(types, key=lambda d: -sum(types[d].values()))
    within = np.array([types[d]["within"] for d in domains], dtype=float)
    partial = np.array([types[d]["partial"] for d in domains], dtype=float)
    outside = np.array([types[d]["outside"] for d in domains], dtype=float)
    totals = within + partial + outside

    if stats_file is not None:
        stats_file.write("Per-domain AAMotif coverage (domain -> within / partial / outside):\n")
        stats_file.write("  within = all AAMotif residues inside this domain; partial = some; "
                         "outside = none.\n")
        for i, d in enumerate(domains):
            tot = int(within[i] + partial[i] + outside[i])
            denom = tot if tot else 1
            stats_file.write(
                f"  {names[d]} [{d}] (n={tot}): "
                f"within={int(within[i])} ({100*within[i]/denom:.1f}%), "
                f"partial={int(partial[i])} ({100*partial[i]/denom:.1f}%), "
                f"outside={int(outside[i])} ({100*outside[i]/denom:.1f}%)\n"
            )
        stats_file.write("\n")
    totals[totals == 0] = 1
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(domains) + 4), 6))
    x = np.arange(len(domains))
    ax.bar(x, 100 * within / totals, color="#2E7D32", label="Within domain")
    ax.bar(x, 100 * partial / totals, bottom=100 * within / totals, color="#FBC02D",
           label="Partially within")
    ax.bar(x, 100 * outside / totals, bottom=100 * (within + partial) / totals,
           color="#C62828", label="Outside domain")
    ax.set_xticks(x)
    ax.set_xticklabels([names[d] for d in domains], rotation=90, ha="right", fontsize=11)
    ax.set_ylabel("Percentage of samples (%)")
    ax.set_xlabel("Protein domain")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.margins(x=0.01)
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("domain_coverage_distribution")
    plt.close(fig)


def plot_coverage_kind_by_class(samples, plotter, stats_file):
    """within / partial / outside coverage breakdown, split by maxprobe vs minprobe."""
    kinds = ("within", "partial", "outside")
    kind_labels = {"within": "Within domain", "partial": "Partially within",
                   "outside": "Outside domain"}
    counts = {"maxprobe": {k: 0 for k in kinds}, "minprobe": {k: 0 for k in kinds}}
    for s in samples:
        cls = s["class"]
        if cls not in counts:
            continue
        for _d_id, (kind, _frac) in s["per_domain"].items():
            counts[cls][kind] += 1
    totals = {cls: sum(counts[cls].values()) for cls in counts}
    if not totals["maxprobe"] or not totals["minprobe"]:
        stats_file.write("Coverage class by probe type: insufficient data.\n\n")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = np.arange(len(kinds))
    width = 0.4
    for offset, (cls, color) in enumerate([("maxprobe", MAX_COLOR), ("minprobe", MIN_COLOR)]):
        pct = [100.0 * counts[cls][k] / totals[cls] for k in kinds]
        ax.bar(x + (offset - 0.5) * width, pct, width, color=color, alpha=0.85,
               label=f"{cls} (n={totals[cls]} domain calls)")
        for xi, (k, p) in enumerate(zip(kinds, pct)):
            ax.text(xi + (offset - 0.5) * width, p + 1, f"{p:.0f}%", ha="center",
                    fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([kind_labels[k] for k in kinds])
    ax.set_ylabel("Percentage of domain calls (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("coverage_kind_by_class")
    plt.close(fig)

    stats_file.write("Coverage class (within/partial/outside) by probe type "
                     "(per sample-domain call):\n")
    for cls in ("maxprobe", "minprobe"):
        parts = ", ".join(
            f"{kind_labels[k]}={counts[cls][k]} ({100.0*counts[cls][k]/totals[cls]:.1f}%)"
            for k in kinds
        )
        stats_file.write(f"  {cls} (n={totals[cls]}): {parts}\n")
    contingency = np.array([[counts["maxprobe"][k] for k in kinds],
                            [counts["minprobe"][k] for k in kinds]], dtype=float)
    if (contingency.sum(axis=0) > 0).all() and (contingency.sum(axis=1) > 0).all():
        chi2, p, dof, _ = chi2_contingency(contingency)
        stats_file.write(f"  Chi-square (probe type x coverage class): "
                         f"chi2={chi2:.4g}, dof={dof}, p={p:.3e}\n")
    stats_file.write("\n")


def plot_coverage_by_class(samples, plotter, stats_file):
    """Distribution of coverage fraction and hit count for max vs min probes."""
    mx, mn = split_by_class(samples)
    fields = [("coverage", "AAMotif fraction inside a domain"),
              ("n_hits", "AAMotif residues inside a domain (count)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    stats_file.write("Coverage by probe class (maxprobe vs minprobe):\n")
    for ax, (field, label) in zip(axes, fields):
        mvals = [s[field] for s in mx if np.isfinite(s[field])]
        nvals = [s[field] for s in mn if np.isfinite(s[field])]
        bp = ax.boxplot([mvals, nvals], showfliers=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bp["boxes"], (MAX_COLOR, MIN_COLOR)):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for i, (vals, color) in enumerate(((mvals, MAX_COLOR), (nvals, MIN_COLOR)), start=1):
            jitter = np.random.default_rng(0).normal(0, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color=color, alpha=0.3,
                       edgecolor="none")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"maxprobe\n(n={len(mvals)})", f"minprobe\n(n={len(nvals)})"])
        ax.set_ylabel(label)
        _pub_ax(ax, grid_axis="y")
        try:
            u, p = mannwhitneyu(mvals, nvals, alternative="two-sided")
        except ValueError:
            p = float("nan")
        ax.text(0.97, 0.97, f"Mann-Whitney\np={p:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=12)
        stats_file.write(f"  {field}: max {fmt_summary(summarize(mvals))}\n")
        stats_file.write(f"  {field}: min {fmt_summary(summarize(nvals))}\n")
        stats_file.write(f"  {field}: Mann-Whitney two-sided p={p:.3e}\n")
    stats_file.write("\n")
    plt.tight_layout()
    plotter.save_plot("coverage_by_class")
    plt.close(fig)


def plot_metric_scaling(samples, plotter, stats_file):
    """Scatter of domain coverage vs each metric, colored by probe class."""
    mx, mn = split_by_class(samples)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 5.4))
    if len(METRICS) == 1:
        axes = [axes]
    stats_file.write("Coverage vs metric correlation (Spearman rho):\n")
    for ax, (key, label) in zip(axes, METRICS):
        for grp, color, name in ((mx, MAX_COLOR, "maxprobe"), (mn, MIN_COLOR, "minprobe")):
            xs = np.array([s[key] for s in grp])
            ys = np.array([s["coverage"] for s in grp])
            ok = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[ok], ys[ok], s=14, color=color, alpha=0.45, edgecolor="none", label=name)
        allx = np.array([s[key] for s in samples])
        ally = np.array([s["coverage"] for s in samples])
        ok = np.isfinite(allx) & np.isfinite(ally)
        rho, p = (spearmanr(allx[ok], ally[ok]) if ok.sum() >= 3 else (float("nan"), float("nan")))
        ax.set_xlabel(label)
        ax.set_ylabel("AAMotif fraction inside a domain")
        ax.text(0.97, 0.97, f"rho={rho:.3f}\np={p:.2e}, n={int(ok.sum())}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        _pub_ax(ax)
        ax.legend(fontsize=11)
        stats_file.write(f"  coverage vs {label}: rho={rho:.4f}, p={p:.3e}, n={int(ok.sum())}\n")
        for grp, name in ((mx, "maxprobe"), (mn, "minprobe")):
            gx = np.array([s[key] for s in grp])
            gy = np.array([s["coverage"] for s in grp])
            gok = np.isfinite(gx) & np.isfinite(gy)
            grho, gp = (spearmanr(gx[gok], gy[gok]) if gok.sum() >= 3 else (float("nan"), float("nan")))
            stats_file.write(f"    {name}: rho={grho:.4f}, p={gp:.3e}, n={int(gok.sum())}\n")
    stats_file.write("\n")
    plt.tight_layout()
    plotter.save_plot("coverage_vs_metrics")
    plt.close(fig)


def _per_protein_separation_vs_coverage(samples, metric):
    """Per protein: (mean domain-hit %, within-protein max/min separation AUC) for metric."""
    direction = METRIC_DIRECTION[metric]
    by_protein = {}
    for s in samples:
        by_protein.setdefault(s["protein"], []).append(s)
    points = []
    for probes in by_protein.values():
        maxs = [p for p in probes if p["class"] == "maxprobe"]
        mins = [p for p in probes if p["class"] == "minprobe"]
        if not maxs or not mins:
            continue
        cov = [p["coverage"] for p in probes if np.isfinite(p["coverage"])]
        if not cov:
            continue
        sign = -1.0 if direction == "lower" else 1.0
        mx = [sign * p[metric] for p in maxs if np.isfinite(p[metric])]
        mn = [sign * p[metric] for p in mins if np.isfinite(p[metric])]
        auc = rank_auc(mx, mn)
        if auc is None:
            continue
        points.append((100.0 * float(np.mean(cov)), auc))
    return points


def plot_separation_vs_coverage(samples, plotter, stats_file):
    """Per-protein max/min separation AUC vs the protein's domain-hit percentage."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.0 * len(METRICS), 5.4))
    if len(METRICS) == 1:
        axes = [axes]
    stats_file.write("Within-protein separation (per-protein AUC) vs domain-hit % "
                     "(Spearman rho):\n")
    for ax, (key, label) in zip(axes, METRICS):
        points = _per_protein_separation_vs_coverage(samples, key)
        if not points:
            ax.set_visible(False)
            stats_file.write(f"  {label}: no paired proteins\n")
            continue
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        ax.scatter(xs, ys, s=18, color=MAX_COLOR, alpha=0.5, edgecolor="white", linewidth=0.3)
        rho, p = (spearmanr(xs, ys) if len(xs) >= 3 else (float("nan"), float("nan")))
        if len(xs) >= 2:
            b, a = np.polyfit(xs, ys, 1)
            xline = np.array([xs.min(), xs.max()])
            ax.plot(xline, a + b * xline, color=ACCENT, lw=2)
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xlabel("Domain-hit % (AAMotif residues inside a domain)")
        ax.set_ylabel("Per-protein separation AUC (max vs min)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-0.02, 1.02)
        ax.text(0.97, 0.97, f"{label}\nrho={rho:.3f} (p={p:.2e}, n={len(xs)})",
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        _pub_ax(ax)
        stats_file.write(f"  {label}: rho={rho:.4f}, p={p:.3e}, n={len(xs)}\n")
    stats_file.write("\n")
    plt.tight_layout()
    plotter.save_plot("separation_vs_domain_coverage")
    plt.close(fig)


def _grouped_separation_aucs(samples, metric, direction, classify):
    """{group: {protein: per-protein AUC}} for one metric, grouping via classify(probes)."""
    by_protein = {}
    for s in samples:
        by_protein.setdefault(s["protein"], []).append(s)
    sign = -1.0 if direction == "lower" else 1.0
    out = {}
    for protein, probes in by_protein.items():
        maxs = [p for p in probes if p["class"] == "maxprobe"]
        mins = [p for p in probes if p["class"] == "minprobe"]
        if not maxs or not mins:
            continue
        group = classify(probes)
        if group is None:
            continue
        mx = [sign * p[metric] for p in maxs if np.isfinite(p[metric])]
        mn = [sign * p[metric] for p in mins if np.isfinite(p[metric])]
        auc = rank_auc(mx, mn)
        if auc is not None:
            out.setdefault(group, {})[protein] = auc
    return out


def _family_separation_aucs(samples, metric, direction):
    """{family: {protein: per-protein AUC}} for one metric/direction."""
    return _grouped_separation_aucs(samples, metric, direction,
                                    lambda probes: probes[0]["domain_family"])


# Per-protein coverage class from mean AAMotif coverage across its probes.
COVERAGE_WITHIN_MIN = 0.8
COVERAGE_OUTSIDE_MAX = 0.2


def _protein_coverage_class(probes):
    cov = [p["coverage"] for p in probes if np.isfinite(p["coverage"])]
    if not cov:
        return None
    m = float(np.mean(cov))
    if m >= COVERAGE_WITHIN_MIN:
        return "within"
    if m <= COVERAGE_OUTSIDE_MAX:
        return "outside"
    return "partial"


def _paired_wilcoxon(motif_by_prot, af3_by_prot):
    """Paired signed-rank: motif AUC vs AF3 AUC over proteins present in both."""
    shared = sorted(set(motif_by_prot) & set(af3_by_prot))
    diffs = np.array([motif_by_prot[p] - af3_by_prot[p] for p in shared], dtype=float)
    if diffs.size < 3 or not np.any(diffs != 0):
        return {"n": int(diffs.size), "median_diff": float(np.median(diffs)) if diffs.size else np.nan,
                "p": np.nan}
    try:
        _, p = wilcoxon(diffs, alternative="greater")  # H1: motif > AF3
    except ValueError:
        p = np.nan
    return {"n": int(diffs.size), "median_diff": float(np.median(diffs)), "p": float(p)}


def plot_domain_family_hierarchy(samples, plotter, stats_file, min_proteins=5):
    """Per-domain-family within-protein separation: AF3 ipTM vs motif-based methods."""
    method_family = {m: _family_separation_aucs(samples, m, d)
                     for m, d, _, _ in HIERARCHY_METHODS}

    protein_family = {}
    for s in samples:
        protein_family[s["protein"]] = s["domain_family"]
    fam_counts = {}
    for fam in protein_family.values():
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    families = [f for f, _ in sorted(fam_counts.items(), key=lambda kv: -kv[1])
                if f not in ("Other", "Mixed") and fam_counts[f] >= min_proteins]
    if not families:
        stats_file.write("Domain-family predictive hierarchy: no family with enough proteins.\n\n")
        return

    stats_file.write("=" * 80 + "\n")
    stats_file.write("Domain-specific predictive hierarchy "
                     "(within-protein max/min separation AUC per family)\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Families shown (>= {min_proteins} proteins): "
                     f"{', '.join(f'{f} (n={fam_counts[f]})' for f in families)}\n")

    n_fam = len(families)
    n_meth = len(HIERARCHY_METHODS)
    width = 0.8 / n_meth
    fig, ax = plt.subplots(figsize=(1.8 * n_fam + 4, 6.0))
    x = np.arange(n_fam)
    for mi, (metric, _dir, label, color) in enumerate(HIERARCHY_METHODS):
        means, errs = [], []
        for fam in families:
            vals = list(method_family.get(metric, {}).get(fam, {}).values())
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        positions = x + (mi - (n_meth - 1) / 2.0) * width
        ax.bar(positions, means, width, yerr=errs, capsize=3, color=color, alpha=0.85,
               label=label, error_kw=dict(lw=1, alpha=0.6))
        stats_file.write(f"\n  {label}:\n")
        for fam in families:
            vals = list(method_family.get(metric, {}).get(fam, {}).values())
            stats_file.write(f"    {fam:14s} {fmt_summary(summarize(vals))}\n")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n(n={fam_counts[f]})" for f in families])
    ax.set_ylabel("Mean per-protein separation AUC (max vs min)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("domain_family_predictive_hierarchy")
    plt.close(fig)

    af3 = method_family.get("af3_rna_ipTM", {})
    stats_file.write("\n  Paired Wilcoxon signed-rank per family (H1: motif method > AF3 RNA ipTM,\n"
                     "  same proteins; one-sided):\n")
    for metric, _dir, label, _ in HIERARCHY_METHODS:
        if metric == "af3_rna_ipTM":
            continue
        for fam in families:
            res = _paired_wilcoxon(method_family.get(metric, {}).get(fam, {}), af3.get(fam, {}))
            p_txt = f"{res['p']:.3e}" if np.isfinite(res["p"]) else "n/a"
            stats_file.write(f"    {label} vs AF3 ipTM | {fam:12s} "
                             f"n={res['n']}, median(motif-AF3)={res['median_diff']:+.3f}, p={p_txt}\n")
    stats_file.write("\n")


def plot_coverage_stratified_hierarchy(samples, plotter, stats_file, min_proteins=5):
    """AF3 vs motif separation AUC stratified by per-protein coverage class."""
    method_cov = {m: _grouped_separation_aucs(samples, m, d, _protein_coverage_class)
                  for m, d, _, _ in HIERARCHY_METHODS}

    ref = method_cov.get("af3_rna_ipTM", {})
    class_counts = {cls: len(d) for cls, d in ref.items()}
    order = ["within", "partial", "outside"]
    classes = [c for c in order if class_counts.get(c, 0) >= min_proteins]
    if not classes:
        stats_file.write("Coverage-stratified predictive hierarchy: no class with enough "
                         "proteins.\n\n")
        return

    class_labels = {"within": "Within (mean cov >= 0.8)",
                    "partial": "Partial (0.2-0.8)",
                    "outside": "Outside (mean cov <= 0.2)"}
    stats_file.write("=" * 80 + "\n")
    stats_file.write("Coverage-stratified predictive hierarchy "
                     "(within-protein max/min separation AUC per coverage class)\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Classes shown (>= {min_proteins} proteins): "
                     f"{', '.join(f'{c} (n={class_counts[c]})' for c in classes)}\n")

    n_cls = len(classes)
    n_meth = len(HIERARCHY_METHODS)
    width = 0.8 / n_meth
    fig, ax = plt.subplots(figsize=(1.9 * n_cls + 4, 6.0))
    x = np.arange(n_cls)
    for mi, (metric, _dir, label, color) in enumerate(HIERARCHY_METHODS):
        means, errs = [], []
        for cls in classes:
            vals = list(method_cov.get(metric, {}).get(cls, {}).values())
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        positions = x + (mi - (n_meth - 1) / 2.0) * width
        ax.bar(positions, means, width, yerr=errs, capsize=3, color=color, alpha=0.85,
               label=label, error_kw=dict(lw=1, alpha=0.6))
        stats_file.write(f"\n  {label}:\n")
        for cls in classes:
            vals = list(method_cov.get(metric, {}).get(cls, {}).values())
            stats_file.write(f"    {cls:8s} {fmt_summary(summarize(vals))}\n")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{class_labels[c]}\n(n={class_counts[c]})" for c in classes])
    ax.set_ylabel("Mean per-protein separation AUC (max vs min)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _pub_ax(ax, grid_axis="y")
    plt.tight_layout()
    plotter.save_plot("coverage_class_predictive_hierarchy")
    plt.close(fig)

    af3 = method_cov.get("af3_rna_ipTM", {})
    stats_file.write("\n  Paired Wilcoxon signed-rank per coverage class "
                     "(H1: motif > AF3 RNA ipTM; one-sided):\n")
    for metric, _dir, label, _ in HIERARCHY_METHODS:
        if metric == "af3_rna_ipTM":
            continue
        for cls in classes:
            res = _paired_wilcoxon(method_cov.get(metric, {}).get(cls, {}), af3.get(cls, {}))
            p_txt = f"{res['p']:.3e}" if np.isfinite(res["p"]) else "n/a"
            stats_file.write(f"    {label} vs AF3 ipTM | {cls:8s} "
                             f"n={res['n']}, median(motif-AF3)={res['median_diff']:+.3f}, p={p_txt}\n")
    stats_file.write("\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output dir for plots + stats")
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db or StartConfig().get_database_path())
    samples = load_samples(db_path)
    mx, mn = split_by_class(samples)
    print(f"samples with domain_pos + AAMotif: {len(samples)} "
          f"(maxprobe: {len(mx)}, minprobe: {len(mn)})")
    if not samples:
        print("No samples to analyze.")
        return 0

    plotter = PlotCreator(plot_type=PLOT_TYPE, standalone_outdir=args.outdir)
    plotter.set_plot_style()
    plt.rcParams.update(PUB_RC)  # Nature-style defaults (no titles; despined axes)
    print(f"Saving outputs under {plotter.results_dir}")

    stats_path = os.path.join(plotter.results_dir, STATS_FILENAME)
    with open(stats_path, "w", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write("AAMotif vs annotated-domain coverage analysis\n")
        sf.write("=" * 80 + "\n\n")
        sf.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        sf.write(f"Database: {db_path}\n")
        sf.write(f"Samples (domain_pos + AAMotif present): {len(samples)} "
                 f"(maxprobe={len(mx)}, minprobe={len(mn)})\n")
        sf.write("Coverage = fraction of AAMotif interface residues inside ANY annotated "
                 "domain range.\n")
        sf.write("Note: lower log10 Kd = stronger predicted binding.\n\n")

        sf.write("Overall coverage: " + fmt_summary(summarize([s["coverage"] for s in samples])) + "\n")
        sf.write("Overall hit count: " + fmt_summary(summarize([s["n_hits"] for s in samples])) + "\n\n")

        plot_category_distribution(samples, plotter, sf)
        plot_coverage_by_class(samples, plotter, sf)
        plot_metric_scaling(samples, plotter, sf)
        plot_separation_vs_coverage(samples, plotter, sf)
        plot_domain_family_hierarchy(samples, plotter, sf)
        plot_coverage_stratified_hierarchy(samples, plotter, sf)
        plot_domain_coverage_distribution(samples, plotter, sf)
        plot_coverage_kind_by_class(samples, plotter, sf)

    print(f"Statistics written to: {stats_path}")
    return 0


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
COMPOSITE_PANEL_NAMES = (
    "domain_family_predictive_hierarchy",
    "domain_coverage_distribution",
)


def _draw_domain_family_hierarchy(samples, min_proteins=5):
    method_family = {m: _family_separation_aucs(samples, m, d)
                     for m, d, _, _ in HIERARCHY_METHODS}
    fam_counts = {}
    for s in samples:
        fam_counts[s["domain_family"]] = fam_counts.get(s["domain_family"], 0) + 1
    families = [f for f, _ in sorted(fam_counts.items(), key=lambda kv: -kv[1])
                if f not in ("Other", "Mixed") and fam_counts[f] >= min_proteins]

    def draw(ax):
        if not families:
            ax.text(0.5, 0.5, "no family with enough proteins",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
            return
        n_meth = len(HIERARCHY_METHODS)
        width = 0.8 / n_meth
        x = np.arange(len(families))
        for mi, (metric, _dir, label, color) in enumerate(HIERARCHY_METHODS):
            means, errs = [], []
            for fam in families:
                vals = list(method_family.get(metric, {}).get(fam, {}).values())
                means.append(float(np.mean(vals)) if vals else np.nan)
                errs.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            positions = x + (mi - (n_meth - 1) / 2.0) * width
            ax.bar(positions, means, width, yerr=errs, capsize=3, color=color,
                   alpha=0.85, label=label, error_kw=dict(lw=1, alpha=0.6))
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}\n(n={fam_counts[f]})" for f in families])
        ax.set_ylabel("Mean per-protein separation AUC")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=7)
        _pub_ax(ax, grid_axis="y")
    return draw


def _draw_domain_coverage(samples):
    types, names = {}, {}
    for s in samples:
        for d_id, (kind, _) in s["per_domain"].items():
            types.setdefault(d_id, {"within": 0, "partial": 0, "outside": 0})[kind] += 1
            names.setdefault(d_id, s["domain_map"].get(d_id, d_id))

    def draw(ax):
        if not types:
            ax.text(0.5, 0.5, "no domain data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
            return
        domains = sorted(types, key=lambda d: -sum(types[d].values()))
        within = np.array([types[d]["within"] for d in domains], dtype=float)
        partial = np.array([types[d]["partial"] for d in domains], dtype=float)
        outside = np.array([types[d]["outside"] for d in domains], dtype=float)
        totals = within + partial + outside
        totals[totals == 0] = 1
        x = np.arange(len(domains))
        ax.bar(x, 100 * within / totals, color="#2E7D32", label="Within domain")
        ax.bar(x, 100 * partial / totals, bottom=100 * within / totals,
               color="#FBC02D", label="Partially within")
        ax.bar(x, 100 * outside / totals, bottom=100 * (within + partial) / totals,
               color="#C62828", label="Outside domain")
        ax.set_xticks(x)
        ax.set_xticklabels([names[d] for d in domains], rotation=90,
                           fontsize=max(5, min(9, 220 // max(1, len(domains)))))
        ax.set_ylabel("Percentage of samples (%)")
        ax.set_xlabel("Protein domain")
        ax.set_ylim(0, 100)
        ax.margins(x=0.01)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=8)
        _pub_ax(ax, grid_axis="y")
    return draw


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None):
    """Return {base_name: (draw_fn, title)} for plots.composite_figures."""
    from plots.publication_figure import normalize_panel_name
    wanted = ([normalize_panel_name(n) for n in names]
              if names else list(COMPOSITE_PANEL_NAMES))

    db_path = db_path or StartConfig().get_database_path()
    samples = load_samples(db_path)

    registry = {
        "domain_family_predictive_hierarchy": (
            _draw_domain_family_hierarchy(samples),
            "Domain-family predictive hierarchy"),
        "domain_coverage_distribution": (
            _draw_domain_coverage(samples),
            "Per-domain AAMotif coverage"),
    }
    return {n: registry[n] for n in wanted if n in registry}


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="domain_composite"))
    raise SystemExit(main())
