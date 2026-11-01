"""
Combined Nature-style publication figure for the RNAcompete analyses.

Brings four existing panels into a single multi-panel figure:
  A  Per-protein ranking AUC per predicted score   (probe_ranking_within_protein: B_per_protein_auc)
  B  Global maxprobe-vs-minprobe ROC per score      (probe_ranking_within_protein: B_global_roc)
  C  Domain-family predictive hierarchy             (domain_analysis: domain_family_predictive_hierarchy)
  D  Per-domain AAMotif coverage distribution       (domain_analysis: domain_coverage_distribution)

This is an *interface* script: it reuses the data-loading and analysis helpers from
``probe_ranking_within_protein.py`` and ``domain_analysis.py`` (no logic is duplicated),
and only re-renders each panel onto a shared GridSpec so the result is a single
vector-editable figure suitable for a Nature-style layout.

Usage:
  python3 rnacompete/combined_publication_figure.py
  python3 rnacompete/combined_publication_figure.py --db PATH --outdir PATH
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator

# Reuse data loading + analysis helpers from the two source scripts (no duplication).
import probe_ranking_within_protein as pr
import domain_analysis as da

PLOT_TYPE = "rna_compete_combined"
DEFAULT_OUTDIR = os.path.join(
    _REPO_ROOT, "results", "plots", "all_chains", "rna_compete_combined"
)
FIG_NAME = "combined_rnacompete_publication_figure"

# Nature-style defaults (sans-serif, thin spines, editable text in vector exports).
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 14.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _pub_ax(ax, grid_axis=None):
    """Nature-style axis cosmetics: despine top/right, outward ticks, faint grid."""
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


def _panel_letter(ax, letter, dx=-0.12):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="right")


# Extra per-probe predicted score overlaid on panels A and B: AF3 chain-pair PAE min.
# A low chain_pair_pae_min has been reported to indicate that two chains interact, so
# we orient it "lower" (higher oriented value == model prefers the probe) to test that
# claim head-to-head against af3_rna_ipTM (ipTM_r) on the same maxprobe-vs-minprobe
# separation task. Scoped to this combined figure only (pr.PRED_SCORES is untouched, so
# the standalone probe_ranking_within_protein.py outputs and stats are unchanged).
PAE_KEY = "af3_chain_pair_pae_min"
PAE_SCORE = (PAE_KEY, "AF3 chain-pair PAE min", "lower")
AB_SCORES = [pr.PRED_SCORES[0], PAE_SCORE, *pr.PRED_SCORES[1:]]  # PAE right after ipTM_r
AB_LABELS = {**pr.PRED_LABELS, PAE_KEY: PAE_SCORE[1]}


def augment_with_pae(proteins):
    """Mirror an oriented af3_chain_pair_pae_min onto every probe for panels A/B.

    pr.load_proteins keeps chain-pair PAE-min only in probe["features"] (raw, positive).
    Panels A/B rank on probe["oriented"][key], so add it there with the "lower is better"
    orientation (negated), matching how the other scores are oriented.
    """
    for p in proteins:
        for probe in p["maxprobes"] + p["minprobes"]:
            v = probe["features"].get(PAE_KEY, np.nan)
            probe["oriented"][PAE_KEY] = -float(v) if np.isfinite(v) else np.nan


# --------------------------------------------------------------------------- #
# Panels (each draws onto a supplied Axes; data via the source-module helpers)
# --------------------------------------------------------------------------- #
def panel_per_protein_auc(ax, acc_by_score, scores=pr.PRED_SCORES, labels=pr.PRED_LABELS):
    """A: per-protein within-protein AUC distribution per predicted score."""
    keys = [k for k, _, _ in scores if acc_by_score.get(k)]
    data = [list(acc_by_score[k].values()) for k in keys]
    bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(pr.MAX_COLOR)
        patch.set_alpha(0.55)
    for i, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=9, color=pr.MAX_COLOR,
                   alpha=0.35, edgecolor="none", zorder=2)
        ax.scatter([i], [np.mean(vals)], marker="D", color=pr.ACCENT, s=45, zorder=3,
                   label="mean" if i == 1 else None)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([labels[k] for k in keys], rotation=25, ha="right")
    ax.set_ylabel("Per-protein AUC (within-protein)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    _pub_ax(ax, grid_axis="y")


def panel_global_roc(ax, proteins, scores=pr.PRED_SCORES):
    """B: global maxprobe(=1)-vs-minprobe(=0) ROC per predicted score."""
    for key, label, _ in scores:
        labels, scores = [], []
        for p in proteins:
            for probe in p["maxprobes"] + p["minprobes"]:
                v = probe["oriented"][key]
                if np.isfinite(v):
                    labels.append(1 if probe["class"] == "maxprobe" else 0)
                    scores.append(v)
        if not labels:
            continue
        fpr, tpr = pr.roc_curve_points(labels, scores)
        auc = pr.global_auc(labels, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)   # dense 7-entry ROC legend: keep compact
    _pub_ax(ax)


def panel_domain_family_hierarchy(ax, samples, min_proteins=5):
    """C: per-domain-family within-protein separation AUC, AF3 vs motif methods."""
    method_family = {m: da._family_separation_aucs(samples, m, d)
                     for m, d, _, _ in da.HIERARCHY_METHODS}
    fam_counts = {}
    for s in samples:
        fam = s["domain_family"]
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    families = [f for f, _ in sorted(fam_counts.items(), key=lambda kv: -kv[1])
                if f not in ("Other", "Mixed") and fam_counts[f] >= min_proteins]
    if not families:
        ax.text(0.5, 0.5, "no family with enough proteins", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return
    n_meth = len(da.HIERARCHY_METHODS)
    width = 0.8 / n_meth
    x = np.arange(len(families))
    for mi, (metric, _dir, label, color) in enumerate(da.HIERARCHY_METHODS):
        means, errs = [], []
        for fam in families:
            vals = list(method_family.get(metric, {}).get(fam, {}).values())
            means.append(float(np.mean(vals)) if vals else np.nan)
            errs.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        positions = x + (mi - (n_meth - 1) / 2.0) * width
        ax.bar(positions, means, width, yerr=errs, capsize=3, color=color, alpha=0.85,
               label=label, error_kw=dict(lw=1, alpha=0.6))
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n(n={fam_counts[f]})" for f in families])
    ax.set_ylabel("Mean per-protein separation AUC")
    ax.set_ylim(0, 1.02)
    # legend on one line below the panel (clears the bars + the 2-line x tick labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=12)
    _pub_ax(ax, grid_axis="y")


def panel_domain_coverage(ax, samples):
    """D: per-domain within/partial/outside AAMotif coverage (stacked bars)."""
    types, names = {}, {}
    for s in samples:
        for d_id, (kind, _) in s["per_domain"].items():
            types.setdefault(d_id, {"within": 0, "partial": 0, "outside": 0})[kind] += 1
            names.setdefault(d_id, s["domain_map"].get(d_id, d_id))
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
    ax.bar(x, 100 * partial / totals, bottom=100 * within / totals, color="#FBC02D",
           label="Partially within")
    ax.bar(x, 100 * outside / totals, bottom=100 * (within + partial) / totals,
           color="#C62828", label="Outside domain")
    ax.set_xticks(x)
    ax.set_xticklabels([names[d] for d in domains], rotation=90,
                       fontsize=max(9, min(11, 300 // max(1, len(domains)))))
    ax.set_ylabel("Percentage of samples (%)")
    ax.set_xlabel("Protein domain")
    ax.set_ylim(0, 100)
    ax.margins(x=0.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=12)
    _pub_ax(ax, grid_axis="y")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_figure(proteins, acc_by_score, samples):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (axes[0, 0], "A", lambda ax: panel_per_protein_auc(ax, acc_by_score, AB_SCORES, AB_LABELS)),
        (axes[0, 1], "B", lambda ax: panel_global_roc(ax, proteins, AB_SCORES)),
        (axes[1, 0], "C", lambda ax: panel_domain_family_hierarchy(ax, samples)),
        (axes[1, 1], "D", lambda ax: panel_domain_coverage(ax, samples)),
    ]
    for ax, letter, draw in panels:
        try:
            draw(ax)
        except Exception as exc:  # keep the rest of the figure if one panel fails
            print(f"Warning: panel {letter} failed: {exc}", file=sys.stderr)
            ax.text(0.5, 0.5, f"panel {letter} unavailable", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
        _panel_letter(ax, letter)

    fig.tight_layout(w_pad=3.0, h_pad=3.0)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Output directory for the combined figure")
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db or StartConfig().get_database_path())

    proteins = pr.load_proteins(db_path)
    augment_with_pae(proteins)  # add oriented af3_chain_pair_pae_min for panels A/B
    acc_by_score = {k: pr.rank_accuracy(proteins, k) for k, _, _ in AB_SCORES}
    samples = da.load_samples(db_path)

    n_max = sum(len(p["maxprobes"]) for p in proteins)
    n_min = sum(len(p["minprobes"]) for p in proteins)
    print(f"proteins (max+min): {len(proteins)} (maxprobes={n_max}, minprobes={n_min}); "
          f"domain samples: {len(samples)}")

    # Head-to-head: chain_pair_pae_min vs ipTM_r on maxprobe-vs-minprobe separation.
    print("maxprobe-vs-minprobe separation (higher AUC = better at telling binder from non-binder):")
    for key in ("af3_rna_ipTM", PAE_KEY):
        gauc = pr._global_auc_for_score(proteins, key)
        pp = list(acc_by_score.get(key, {}).values())
        pp_mean = float(np.mean(pp)) if pp else float("nan")
        print(f"  {AB_LABELS[key]:26s} global-pooled AUC={gauc:.3f}  "
              f"per-protein mean AUC={pp_mean:.3f} (n={len(pp)})")

    plotter = PlotCreator(plot_type=PLOT_TYPE, standalone_outdir=args.outdir)
    plotter.set_plot_style()

    with plt.rc_context(PUB_RC):
        fig = build_figure(proteins, acc_by_score, samples)
        os.makedirs(plotter.results_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            out = os.path.join(plotter.results_dir, f"{FIG_NAME}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Combined figure written to: {plotter.results_dir}/{FIG_NAME}.png (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
