#!/usr/bin/env python3
"""
pub_fig1_classification_framework -- the AF3 protein-RNA classification-framework figures
(pub_fig1..4), extracted from evaluation/evaluationMetrics.py so all publication figures live
under publication_figures/. The i-iv / a/b/c classification itself stays in evaluationMetrics.py
(get_sample_audit_global_fold_interface_lists), which this script imports. pub_fig1 uses the
full data set (n=384); the interface subclass is DockQ_Nr.

Run from the repo root:
  python publication_figures/pub_fig1_classification_framework.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _spearman(x, y):
    """Spearman rho over finite pairs; nan if <3 points or no variance in either."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan
    return float(spearmanr(x[m], y[m]).correlation)

from plots.plotCreator import PlotCreator
from plots.publication_figure import onpage_to_gen
from evaluation.evaluationMetrics import (
    RNAMetrics,
    _audit_load_prediction_by_eid,
    _audit_extract_motif_bin_metrics,
    _longest_rna_index_and_chain,
    _rna_length_at_index,
    _as_chain_list,
    _as_float_list,
    _safe_literal_eval,
    _effective_msa_depth_for_chain,
)


def plot_five_publication_figures(plot_creator, table_source, audit, rna_metrics=None):
    """Generate Nature-style multi-panel summary figures saved to results_dir.

    Violin panels use seaborn (density_norm='width') to match plots/plotCreator.py.
    No figure-level or panel-level titles are drawn (only corner panel letters).

    Figure 1 - Classification framework
        (A) RNA_LDDT vs Protein_LDDT scatter (quadrant colored)
        (B) Stacked bar: a/b/c interface-subclass proportions per quadrant
        (C) Violin DockQ_Nr per quadrant
        (D) LDDT_r and TM_r vs DockQ interface class (a/b/c)
    Figure 2 - Interface subclass validation (RBP_DockQ / QS / fnat / irmsd by a/b/c)
    Figure 3 - RNA global accuracy vs interface quality
    Figure 4 - AF3 confidence metrics (2x3; letters in reading order A-C / D-F)
        (A) af3_protein_pTM by quadrant
        Panels B/C are computed PER RNA CHAIN (one point per predicted RNA chain,
        not per complex): af3_rna_pTM and RNALength are index-aligned to RNAChainIDs,
        per-chain TM comes from RNA_TM_AllChains ({chain: TM}), and MSA depth from the
        per-chain RNA_msa_size value. Multi-RNA complexes contribute several points
        (n ~653 chains vs 384 complexes), which fills the sparse +MSA bins.
        (B) af3_rna_pTM vs per-chain TM scatter, colored by RNA MSA (+MSA / -MSA);
            per-group Pearson r shown in the legend (-MSA ~0.03, +MSA ~0.80).
        (C) mean af3_rna_pTM and mean per-chain TM by RNA MSA depth bin (pTM
            calibration), each bin labelled with n and median RNA length: without an
            MSA pTM collapses to a flat low band; the +MSA bins are length-matched
            (median ~73 nt) so the >=1000 dip is small-n noise, not length.
        (D) af3_rna_ipTM by audit category (quadrant x a/b/ca/cb)
        (E) af3_global_pae_avg vs Complex_RMSD (cRMSD), colored by RNA MSA (+MSA / -MSA)
        (F) confidence-vs-ground-truth Spearman rho heatmap (one value per complex):
            rows are the seven AF3 confidence metrics, columns the axes (protein
            LDDT_p, RNA TM_r, RNA LDDT_r, interface DockQ_Nr, complex TM_c, log RNA
            MSA depth, RNA folding free energy MFE, RBP template count 0–4). The
            accuracy targets are oriented higher=better (Complex_TM, not RMSD) so signs
            are comparable; the last three columns are RNA-intrinsic / input axes (MFE
            is pTM_r's single strongest correlate, rho ~ -0.77, ahead of MSA depth).
            RBP template uses the count (not
            binary presence): it correlates marginally better overall and more clearly
            for the interface metrics. The boxed cell per row is that metric's dominant
            |r| (its "centre"): which confidence metric tracks which axis.
    """
    if not audit or "quad_i_all" not in audit:
        return

    import os as _os
    try:
        import seaborn as sns
    except Exception as _sns_err:
        print(f"Warning: seaborn unavailable, skipping publication figures: {_sns_err}",
              file=sys.stderr)
        return
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    # -- constants --------------------------------------------------------------
    # Nature-style muted qualitative palette (desaturated, print-friendly).
    _QUAD_COLORS = ["#4E79A7", "#59A14F", "#E15759", "#B07AA1"]
    _QUAD_LABELS = ["(i)", "(ii)", "(iii)", "(iv)"]
    _QUAD_KEYS = ["quad_i", "quad_ii", "quad_iii", "quad_iv"]
    _SUB_COLORS = {"a": "#4E79A7", "b": "#F1A340", "c": "#D7301F"}
    _ABC_COLORS = [_SUB_COLORS[k] for k in ("a", "b", "c")]
    _ABC_LABELS = ["(a) good interface", "(b) mid interface", "(c) bad interface"]

    plot_creator.set_plot_style()

    def _finite(vals):
        return [float(v) for v in vals if v is not None and np.isfinite(v)]

    def _extract_field(records, field):
        return _finite([rec.get(field) for rec in (records or [])])

    def _panel_letter(ax, letter, fig_width):
        ax.text(-0.08, 1.04, letter, transform=ax.transAxes,
                fontsize=onpage_to_gen(8, fig_width), fontweight="bold",
                va="bottom", ha="right")

    def _sns_violin(ax, data_values, colors, labels, *, rotation=0, fontsize=None):
        """Seaborn violin (density_norm='width') recolored per group; matches plotCreator.

        Groups with no finite data are dropped so violins, colors and x-tick labels
        stay aligned (seaborn collapses all-NaN groups otherwise).
        """
        keep = [(d, c, l) for d, c, l in zip(data_values, colors, labels)
                if d is not None and len(d) > 0]
        if not keep:
            return
        data_keep, colors_keep, labels_keep = zip(*keep)
        sns.violinplot(data=list(data_keep), density_norm="width", ax=ax,
                       inner="box", cut=0)
        # Recolour by the violin's x-position, not by draw order: a single-point
        # (or zero-variance) group yields no KDE body, so seaborn draws fewer
        # PolyCollections than groups. Zipping colours in order would then shift
        # every colour after the gap (e.g. iiia/ivb miscoloured because iia is a
        # bare line). Mapping each body's centre x -> group index stays aligned.
        bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
        for body in bodies:
            paths = body.get_paths()
            if not paths:
                continue
            idx = int(round(float(np.mean(paths[0].vertices[:, 0]))))
            if 0 <= idx < len(colors_keep):
                body.set_facecolor(colors_keep[idx])
                body.set_edgecolor("black")
                body.set_alpha(0.75)
        ax.set_xticks(range(len(labels_keep)))
        ax.set_xticklabels(labels_keep, rotation=rotation,
                           ha="right" if rotation else "center",
                           fontsize=fontsize)

    def _save_fig(fig, name):
        chain_type = "single_chain" if plot_creator.single_chain_only else "all_chains"
        msa_sfx = f"_{plot_creator.msa_option}" if getattr(plot_creator, "msa_option", None) else ""
        rbp_sfx = f"_{plot_creator.rbp_template_option}" if getattr(plot_creator, "rbp_template_option", None) else ""
        stem = f"{name}_{table_source}_{chain_type}{msa_sfx}{rbp_sfx}"
        for ext in ("png", "pdf", "svg", "eps"):
            fig.savefig(_os.path.join(plot_creator.results_dir, f"{stem}.{ext}"),
                        dpi=600, bbox_inches="tight")
        plt.close(fig)

    # -- collect per-quadrant records -------------------------------------------
    quad_all = [audit.get(f"{p}_all") or [] for p in _QUAD_KEYS]

    # -- load AF3 features per eid (best-effort) --------------------------------
    by_eid = None
    if rna_metrics is not None:
        try:
            by_eid = _audit_load_prediction_by_eid(rna_metrics, table_source)
        except Exception:
            by_eid = None

    def _af3_metric_by_quad(metric_key):
        result = []
        for recs in quad_all:
            vals = []
            for rec in recs:
                eid = str(rec.get("exp_db_id", "")).strip()
                if by_eid and eid in by_eid:
                    fv = _audit_extract_motif_bin_metrics(by_eid[eid])
                    v = fv.get(metric_key)
                    if v is not None and np.isfinite(v):
                        vals.append(float(v))
            result.append(vals)
        return result

    # audit categories: quadrant × interface bin (a / b / c_ca / c_cb), 16 groups
    _CAT_SUFFIXES = [("_a", "a"), ("_b", "b"), ("_c_ca", "ca"), ("_c_cb", "cb")]

    def _af3_metric_by_category(metric_key):
        """Per audit category (quadrant × a/b/ca/cb): values, labels, per-group colors."""
        labels, series, colors = [], [], []
        for qi, (qk, roman) in enumerate(zip(_QUAD_KEYS, _QUAD_LABELS)):
            for sfx, slbl in _CAT_SUFFIXES:
                vals = []
                for rec in audit.get(f"{qk}{sfx}") or []:
                    eid = str(rec.get("exp_db_id", "")).strip()
                    if by_eid and eid in by_eid:
                        fv = _audit_extract_motif_bin_metrics(by_eid[eid])
                        v = fv.get(metric_key)
                        if v is not None and np.isfinite(v):
                            vals.append(float(v))
                labels.append(f"{roman} {slbl}")
                series.append(vals)
                colors.append(_QUAD_COLORS[qi])
        return labels, series, colors

    # MSA-coloured scatter helpers (matches msa_*_all_chains plots: +MSA blue / -MSA orange)
    _MSA_COLORS = {"+MSA": "#1f77b4", "-MSA": "#ff7f0e"}

    def _msa_status(fv):
        v = fv.get("has_RNA_MSA")
        if v is None or not np.isfinite(v):
            return None
        return "+MSA" if float(v) >= 0.5 else "-MSA"

    def _msa_scatter_points(xkey, ykey):
        """Per MSA status ('+MSA'/'-MSA'): (xs, ys) from by_eid features over all audit eids."""
        pts = {"+MSA": ([], []), "-MSA": ([], [])}
        for recs in quad_all:
            for rec in recs:
                eid = str(rec.get("exp_db_id", "")).strip()
                if not by_eid or eid not in by_eid:
                    continue
                fv = _audit_extract_motif_bin_metrics(by_eid[eid])
                s = _msa_status(fv)
                if s is None:
                    continue
                x, y = fv.get(xkey), fv.get(ykey)
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    continue
                pts[s][0].append(float(x))
                pts[s][1].append(float(y))
        return pts

    def _perchain_keyed_metric(raw):
        """Per-predicted-chain RNA TM as a LIST index-aligned to RNAChainIDs.

        Accepts BOTH stored formats of RNA_TM_AllChains:
          * keyed form  [{'B':0.13},{'C':0.15}]      -> [0.13, 0.15]  (value per dict, in chain order)
          * legacy all-vs-all matrix [[0.13,0.0],[0.0,0.15]] -> [0.13, 0.15]
            (predicted chain j = column j, max over reference rows; matches
            evaluationMetrics._extract_longest_rna_matrix_score). The shipped
            rbpDatabase.db stores the MATRIX form, so this fallback is what fills panels B/C.
        """
        parsed = _safe_literal_eval(raw)
        vals = []
        if isinstance(parsed, list) and parsed:
            if all(isinstance(e, dict) for e in parsed):
                for entry in parsed:
                    v = next(iter(entry.values()), np.nan)
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        vals.append(np.nan)
            elif all(isinstance(e, (list, tuple)) for e in parsed):
                ncol = max((len(r) for r in parsed), default=0)
                for j in range(ncol):
                    col = [float(r[j]) for r in parsed
                           if j < len(r) and isinstance(r[j], (int, float))]
                    vals.append(max(col) if col else np.nan)
        return vals

    def _perchain_records():
        """Per-RNA-chain arrays (one row per predicted RNA chain, not per complex).

        af3_rna_pTM and RNALength are lists index-aligned to RNAChainIDs; RNA_TM
        comes from the per-chain keyed RNA_TM_AllChains ({chain: TM}); MSA depth is
        the per-chain RNA_msa_size value (1 -> 0.0 = no MSA). Multi-RNA complexes
        therefore contribute several points. Finite entries only.

        Returns dict of np.arrays: ptm, tm, depth, length, has_msa (bool).
        """
        ptm, tm, dep, length = [], [], [], []
        for recs in quad_all:
            for rec in recs:
                eid = str(rec.get("exp_db_id", "")).strip()
                if not by_eid or eid not in by_eid:
                    continue
                rd = by_eid[eid]
                chains = _as_chain_list(rd.get("RNAChainIDs"))
                lens = _as_float_list(rd.get("RNALength"))
                ptms = _as_float_list(rd.get("af3_rna_pTM"))
                tm_list = _perchain_keyed_metric(rd.get("RNA_TM_AllChains"))
                for i, ch in enumerate(chains):
                    P = ptms[i] if i < len(ptms) else np.nan
                    L = lens[i] if i < len(lens) else np.nan
                    T = tm_list[i] if i < len(tm_list) else np.nan
                    d = _effective_msa_depth_for_chain(rd.get("RNA_msa_size"), ch)
                    if not (np.isfinite(P) and np.isfinite(T)
                            and np.isfinite(d) and np.isfinite(L)):
                        continue
                    ptm.append(float(P)); tm.append(float(T))
                    dep.append(float(d)); length.append(float(L))
        dep = np.array(dep)
        return {"ptm": np.array(ptm), "tm": np.array(tm), "depth": dep,
                "length": np.array(length), "has_msa": dep > 0}

    # AF3 confidence metrics (rows) and the ground-truth "centre" each is asked
    # to track (columns). One value per complex, longest-RNA / longest-protein
    # aligned as in _audit_extract_motif_bin_metrics. RNA MSA depth is log10(1+d).
    _CONF_ROWS = [
        ("af3_protein_pTM",       r"$\mathrm{pTM}_{p}$"),
        ("af3_rna_pTM",           r"$\mathrm{pTM}_{r}$"),
        ("af3_protein_ipTM",      r"$\mathrm{ipTM}_{p}$"),
        ("af3_rna_ipTM",          r"$\mathrm{ipTM}_{r}$"),
        ("af3_protein_pLDDT_avg", r"$\mathrm{pLDDT}_{p}$"),
        ("af3_rna_pLDDT_avg",     r"$\mathrm{pLDDT}_{r}$"),
        ("af3_global_pae_avg",    "PAE"),
    ]
    # (key, label, log10(1+x)?) — the four accuracy targets are oriented
    # "higher = better" so signs are comparable across rows (Complex_TM is used,
    # not Complex_RMSD; PAE's row therefore reads negative where it tracks a good
    # outcome). The last two columns are input-availability axes, not accuracy:
    # RNA MSA depth (log) and RBP template count. Count (0–4) is used rather than
    # binary presence: it correlates marginally better overall (mean |r| 0.19 vs
    # 0.17) and more clearly for the interface metrics (ipTM_r, pLDDT_r, PAE).
    # Bare symbols; the p/r/c subscripts (protein/RNA/complex), MSA_r = log depth,
    # and Temp_c = RBP-template count (0-4) are all defined in the figure caption.
    _CONF_TARGETS = [
        ("Protein_LDDT",     r"$\mathrm{LDDT}_{p}$",   False),
        ("RNA_TM",           r"$\mathrm{TM}_{r}$",     False),
        ("RNA_LDDT_allAtom", r"$\mathrm{LDDT}_{r}$",   False),
        ("DockQ_Nr",         r"DockQ$_{Nr}$",          False),
        ("Complex_TM",       r"$\mathrm{TM}_{c}$",     False),
        ("RNA_msa_depth",    r"$\mathrm{MSA}_{r}$",    True),
        ("RNA_MFE_Value",    "MFE",                    False),
        ("RBP_PDB_n",        r"$\mathrm{Temp}_{c}$",   False),
    ]

    def _conf_target_corr():
        """Pairwise-complete Spearman rho of every confidence metric vs every target.

        Returns (R, N, row_labels, col_labels): R and N are
        (n_conf x n_target) arrays of Spearman rho and sample sizes. The dominant
        |rho| per row is that metric's "centre". (Spearman to match the paper's
        convention; relationships are monotonic but not necessarily linear.)
        """
        seen, rows = set(), []
        for recs in quad_all:
            for rec in recs:
                eid = str(rec.get("exp_db_id", "")).strip()
                if not by_eid or eid not in by_eid or eid in seen:
                    continue
                seen.add(eid)
                fv = _audit_extract_motif_bin_metrics(by_eid[eid])
                # Interface target reuses the classification's DockQ_Nr (from the
                # audit record, as in Fig 1 Panel C), not the feats' RBP_DockQ.
                fv["DockQ_Nr"] = rec.get("dockq_nr")
                rows.append(fv)

        def _col(key, logt):
            out = []
            for fv in rows:
                v = fv.get(key)
                v = float(v) if v is not None else np.nan
                if logt and np.isfinite(v):
                    v = np.log10(1.0 + v) if v >= 0 else np.nan
                out.append(v)
            return np.array(out, dtype=float)

        conf = {k: _col(k, False) for k, _ in _CONF_ROWS}
        tgt = {k: _col(k, lg) for k, _, lg in _CONF_TARGETS}
        nC, nT = len(_CONF_ROWS), len(_CONF_TARGETS)
        R = np.full((nC, nT), np.nan)
        N = np.zeros((nC, nT), dtype=int)
        for i, (ck, _) in enumerate(_CONF_ROWS):
            x = conf[ck]
            for j, (tk, _lbl, _lg) in enumerate(_CONF_TARGETS):
                y = tgt[tk]
                m = np.isfinite(x) & np.isfinite(y)
                N[i, j] = int(m.sum())
                if N[i, j] >= 3 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
                    R[i, j] = float(spearmanr(x[m], y[m]).correlation)
        return (R, N, [lbl for _, lbl in _CONF_ROWS],
                [lbl for _, lbl, _ in _CONF_TARGETS])

    # ===========================================================================
    # FIGURE 1: Classification framework
    # ===========================================================================
    try:
        # 1x3: the a/b/c interface proportions per class (former panel B) are in the
        # main interface figure, so only A (classes), B (DockQ) and C (RNA accuracy) remain.
        fig1, axes1 = plt.subplots(1, 3, figsize=(21, 6.8))

        # Panel A: RNA_LDDT vs Protein_LDDT scatter
        ax = axes1[0]
        for qi, (recs, lbl) in enumerate(zip(quad_all, _QUAD_LABELS)):
            xs = _extract_field(recs, "rna_lddt")
            ys = _extract_field(recs, "protein_lddt")
            n = min(len(xs), len(ys))
            if n:
                ax.scatter(xs[:n], ys[:n], c=_QUAD_COLORS[qi], alpha=0.55, s=18,
                           label=f"Class {lbl} (n={n})", edgecolors="none")
        ax.axhline(0.85, color="gray", lw=0.8, ls="--")
        ax.axvline(0.85, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$\mathrm{LDDT}^{\mathrm{C3}'}_{r}$")       # backbone LDDT (RNA_LDDT), as the fold classes
        ax.set_ylabel(r"$\mathrm{LDDT}^{\mathrm{C}\alpha}_{p}$")   # backbone LDDT (Protein_LDDT)
        ax.legend(fontsize=onpage_to_gen(7, 14), markerscale=2.0, frameon=False)
        _panel_letter(ax, "A", 21)

        # Panel B: Violin DockQ_Nr per quadrant
        ax = axes1[1]
        _sns_violin(ax, [_extract_field(r, "dockq_nr") for r in quad_all], _QUAD_COLORS, _QUAD_LABELS)
        ax.set_ylabel(r"DockQ$_{Nr}$")
        ax.set_ylim(-0.05, 1.05)
        _panel_letter(ax, "B", 21)

        # Panel C: LDDT_r and TM_r vs DockQ interface class (a/b/c), grouped violins
        # Shows how global RNA accuracy depends on interface quality.
        ax = axes1[2]
        rows_d = []
        for sub, sublbl in zip(("a", "b", "c"), _ABC_LABELS):
            for p in _QUAD_KEYS:
                recs = audit.get(f"{p}_{sub}") or []
                for v in _extract_field(recs, "rna_lddt"):
                    rows_d.append((sublbl, r"$\mathrm{LDDT}_{r}$", v))
                for v in _extract_field(recs, "rna_tm"):
                    rows_d.append((sublbl, r"$\mathrm{TM}_{r}$", v))
        if rows_d:
            df_d = pd.DataFrame(rows_d, columns=["cls", "metric", "value"])
            sns.violinplot(data=df_d, x="cls", y="value", hue="metric",
                           order=_ABC_LABELS, hue_order=[r"$\mathrm{LDDT}_{r}$", r"$\mathrm{TM}_{r}$"],
                           density_norm="width", cut=0, inner="box",
                           palette={r"$\mathrm{LDDT}_{r}$": "#4E79A7", r"$\mathrm{TM}_{r}$": "#59A14F"}, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("RNA accuracy")
            ax.set_ylim(-0.05, 1.05)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right",
                     fontsize=onpage_to_gen(7, 14))
            ax.legend(title="", fontsize=onpage_to_gen(7, 14), frameon=False, loc="lower left")
        _panel_letter(ax, "C", 21)

        fig1.tight_layout()
        _save_fig(fig1, "pub_fig1_classification_framework")
    except Exception as _e1:
        print(f"Warning: pub_fig1 failed: {_e1}", file=sys.stderr)
        plt.close("all")

    # ===========================================================================
    # FIGURE 2: Interface subclass validation
    # ===========================================================================
    try:
        def _pool_abc(field):
            result = {"a": [], "b": [], "c": []}
            for p in _QUAD_KEYS:
                for sub in ("a", "b", "c"):
                    result[sub].extend(_extract_field(audit.get(f"{p}_{sub}") or [], field))
            return result

        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
        panel_specs2 = [
            (axes2[0, 0], "rbp_dockq", "RBP DockQ", "A", (-0.05, 1.05), False),
            (axes2[0, 1], "rbp_qs",    "RBP QS",    "B", (-0.05, 1.05), False),
            (axes2[1, 0], "rbp_fnat",  "RBP fnat",  "C", (-0.05, 1.05), False),
            (axes2[1, 1], "rbp_irmsd", "RBP iRMSD (Å)", "D", None,  True),
        ]
        for ax, field, ylabel, letter, ylim, invert in panel_specs2:
            pools = _pool_abc(field)
            _sns_violin(ax, [pools[k] for k in ("a", "b", "c")], _ABC_COLORS,
                        _ABC_LABELS, rotation=15, fontsize=onpage_to_gen(7, 14))
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(ylim)
            if invert:
                ax.invert_yaxis()
            for xi, k in enumerate(k for k in ("a", "b", "c") if pools[k]):
                ax.text(xi, ax.get_ylim()[0], f"n={len(pools[k])}", ha="center",
                        va="bottom", fontsize=onpage_to_gen(6.5, 14), color="gray")
            _panel_letter(ax, letter, 14)

        fig2.tight_layout()
        _save_fig(fig2, "pub_fig2_interface_subclass_validation")
    except Exception as _e2:
        print(f"Warning: pub_fig2 failed: {_e2}", file=sys.stderr)
        plt.close("all")

    # ===========================================================================
    # FIGURE 3: RNA global accuracy vs interface quality
    # ===========================================================================
    try:
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11))

        all_recs = [rec for recs in quad_all for rec in recs]

        # Panel A: RBP DockQ & QS by RNA_LDDT quartile (grouped boxplots)
        ax = axes3[0, 0]
        rna_lddt_all = np.array([rec.get("rna_lddt", np.nan) for rec in all_recs], dtype=float)
        rbp_qs_all = np.array([rec.get("rbp_qs", np.nan) for rec in all_recs], dtype=float)
        rbp_dq_all = np.array([rec.get("rbp_dockq", np.nan) for rec in all_recs], dtype=float)
        mask = np.isfinite(rna_lddt_all) & (np.isfinite(rbp_qs_all) | np.isfinite(rbp_dq_all))
        if mask.sum() >= 8:
            lddt_f = rna_lddt_all[mask]
            try:
                qt = pd.qcut(pd.Series(lddt_f), q=4, duplicates="drop")
                cats = qt.cat.categories
                dq_by_q, qs_by_q = [], []
                for cat in cats:
                    m = (qt == cat).to_numpy()
                    dq_by_q.append(_finite(rbp_dq_all[mask][m].tolist()))
                    qs_by_q.append(_finite(rbp_qs_all[mask][m].tolist()))
                pos = np.arange(1, len(cats) + 1)
                ax.boxplot([d if d else [np.nan] for d in dq_by_q],
                           positions=pos - 0.18, widths=0.28, patch_artist=True,
                           showfliers=False, boxprops=dict(facecolor="#4C72B0", alpha=0.75),
                           medianprops=dict(color="black"))
                ax.boxplot([d if d else [np.nan] for d in qs_by_q],
                           positions=pos + 0.18, widths=0.28, patch_artist=True,
                           showfliers=False, boxprops=dict(facecolor="#DD8452", alpha=0.75),
                           medianprops=dict(color="black"))
                ax.set_xticks(pos)
                ax.set_xticklabels([f"Q{i+1}" for i in range(len(cats))],
                                   fontsize=onpage_to_gen(7, 14))
                ax.legend(handles=[Patch(facecolor="#4C72B0", alpha=0.75, label="RBP DockQ"),
                                   Patch(facecolor="#DD8452", alpha=0.75, label="RBP QS")],
                          fontsize=onpage_to_gen(7, 14), frameon=False)
            except Exception:
                ax.text(0.5, 0.5, "quartile split failed", transform=ax.transAxes, ha="center")
        ax.set_xlabel(r"$\mathrm{LDDT}_{r}$ quartile (Q1 = lowest)")
        ax.set_ylabel("Interface score")
        ax.set_ylim(-0.05, 1.05)
        _panel_letter(ax, "A", 14)

        # Panel B: scatter RNA_LDDT vs RBP_QS colored by quadrant
        ax = axes3[0, 1]
        for qi, (recs, lbl) in enumerate(zip(quad_all, _QUAD_LABELS)):
            xs = _extract_field(recs, "rna_lddt")
            ys = _extract_field(recs, "rbp_qs")
            n = min(len(xs), len(ys))
            if n:
                ax.scatter(xs[:n], ys[:n], c=_QUAD_COLORS[qi], alpha=0.45, s=16,
                           label=f"Class {lbl}", edgecolors="none")
        ax.axvline(0.85, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(r"$\mathrm{LDDT}_{r}$")
        ax.set_ylabel("RBP QS")
        ax.legend(fontsize=onpage_to_gen(7, 14), frameon=False, markerscale=2.0)
        _panel_letter(ax, "B", 14)

        # Panel C: violin RNA_LDDT by a/b/c
        ax = axes3[1, 0]
        lddt_pools = {"a": [], "b": [], "c": []}
        for p in _QUAD_KEYS:
            for sub in ("a", "b", "c"):
                lddt_pools[sub].extend(_extract_field(audit.get(f"{p}_{sub}") or [], "rna_lddt"))
        _sns_violin(ax, [lddt_pools[k] for k in ("a", "b", "c")], _ABC_COLORS,
                    _ABC_LABELS, rotation=15, fontsize=onpage_to_gen(7, 14))
        ax.set_ylabel(r"$\mathrm{LDDT}_{r}$")
        ax.set_ylim(-0.05, 1.05)
        _panel_letter(ax, "C", 14)

        # Panel D: violin RNA_TM by a/b/c
        ax = axes3[1, 1]
        tm_pools = {"a": [], "b": [], "c": []}
        for p in _QUAD_KEYS:
            for sub in ("a", "b", "c"):
                tm_pools[sub].extend(_extract_field(audit.get(f"{p}_{sub}") or [], "rna_tm"))
        _sns_violin(ax, [tm_pools[k] for k in ("a", "b", "c")], _ABC_COLORS,
                    _ABC_LABELS, rotation=15, fontsize=onpage_to_gen(7, 14))
        ax.set_ylabel(r"$\mathrm{TM}_{r}$")
        ax.set_ylim(-0.05, 1.05)
        _panel_letter(ax, "D", 14)

        fig3.tight_layout()
        _save_fig(fig3, "pub_fig3_rna_accuracy_vs_interface")
    except Exception as _e3:
        print(f"Warning: pub_fig3 failed: {_e3}", file=sys.stderr)
        plt.close("all")

    # ===========================================================================
    # FIGURE 4: AF3 confidence metrics (letters in reading order: A-C / D-F)
    #   (A) af3_protein_pTM by quadrant
    #   (B) af3_rna_pTM vs RNA_TM scatter, coloured by RNA MSA (+MSA / -MSA)
    #   (C) mean af3_rna_pTM / per-chain TM by RNA MSA depth bin
    #   (D) af3_rna_ipTM by audit category (quadrant × a/b/ca/cb)
    #   (E) af3_global_pae_avg vs Complex_RMSD (cRMSD), coloured by RNA MSA
    #   (F) confidence-metric vs ground-truth Spearman rho heatmap
    # ===========================================================================
    try:
        fig4, axes4 = plt.subplots(2, 3, figsize=(21, 11))
        # Panel letters follow reading order (A-C top row, D-F bottom row).
        ax_a = axes4[0, 0]    # Panel A: pTM_p by quadrant (violin)
        ax_b = axes4[0, 1]    # Panel B: pTM_r vs TM_r scatter (MSA-coloured)
        ax_cal = axes4[0, 2]  # Panel C: mean pTM_r / TM_r by MSA depth bin
        ax_cat = axes4[1, 0]  # Panel D: ipTM_r by audit category
        ax_d = axes4[1, 1]    # Panel E: Global PAE vs RMSD_c scatter (MSA-coloured)
        ax_hm = axes4[1, 2]   # Panel F: confidence-metric vs ground-truth correlation heatmap
        if by_eid is not None:
            # Panel A: af3_protein_pTM by quadrant
            data_by_quad = _af3_metric_by_quad("af3_protein_pTM")
            _sns_violin(ax_a, data_by_quad, _QUAD_COLORS, _QUAD_LABELS)
            ax_a.set_ylabel(r"$\mathrm{pTM}_{p}$")
            ax_a.set_ylim(-0.05, 1.05)
            for xi, d in enumerate(d for d in data_by_quad if len(d)):
                ax_a.text(xi, ax_a.get_ylim()[0], f"n = {len(d)}", ha="center",
                          va="bottom", fontsize=onpage_to_gen(6.5, 21), color="dimgray")
            _panel_letter(ax_a, "A", 21)

            # Per-RNA-chain dataset (one point per predicted RNA chain) shared by B/E/F.
            pc = _perchain_records()

            # Panel B: pTM_r vs TM_r, coloured by RNA MSA (per RNA chain).
            # Pearson here (not Spearman like the heatmap): this is a calibration plot
            # probing whether predicted pTM_r linearly tracks actual TM_r.
            _stats_b = []
            for s, m in [("+MSA", pc["has_msa"]), ("-MSA", ~pc["has_msa"])]:
                if m.sum():
                    xs, ys = pc["ptm"][m], pc["tm"][m]
                    r_b = float(np.corrcoef(xs, ys)[0, 1]) if m.sum() >= 3 else np.nan
                    ax_b.scatter(xs, ys, c=_MSA_COLORS[s], alpha=0.55, s=20,
                                 label=s, edgecolors="none")
                    _rb = f"$r$ = {r_b:+.2f}" if np.isfinite(r_b) else "$r$ = n/a"
                    _stats_b.append(f"{s}: {_rb} (n = {int(m.sum())})")
            ax_b.set_xlabel(r"$\mathrm{pTM}_{r}$")
            ax_b.set_ylabel(r"$\mathrm{TM}_{r}$")
            ax_b.set_xlim(-0.05, 1.05)
            ax_b.set_ylim(-0.05, 1.05)
            # correlation stats in an in-plot box; the legend below is just the colour key.
            ax_b.text(0.30, 0.97, "\n".join(_stats_b), transform=ax_b.transAxes,
                      va="top", ha="left", fontsize=onpage_to_gen(6.5, 21),
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
            ax_b.legend(fontsize=onpage_to_gen(7, 14), frameon=False, markerscale=2.0,
                        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
            _panel_letter(ax_b, "B", 21)

            # Panel D: af3_rna_ipTM by audit category (16 groups, violin)
            cat_labels, cat_series, cat_colors = _af3_metric_by_category("af3_rna_ipTM")
            _sns_violin(ax_cat, cat_series, cat_colors, cat_labels, rotation=90,
                        fontsize=onpage_to_gen(7, 14))
            ax_cat.set_ylabel(r"$\mathrm{ipTM}_{r}$")
            ax_cat.set_ylim(-0.05, 1.05)
            _panel_letter(ax_cat, "D", 21)

            # Panel E: af3_global_pae_avg vs Complex_RMSD, coloured by RNA MSA.
            # Spearman here (like the heatmap): PAE-vs-RMSD_c saturates (monotonic but
            # nonlinear), and the paper reports PAE-vs-error as rho.
            pts_d = _msa_scatter_points("Complex_RMSD", "af3_global_pae_avg")
            _stats_e = []
            for s in ("+MSA", "-MSA"):
                xs, ys = pts_d[s]
                if xs:
                    r_d = _spearman(xs, ys)
                    ax_d.scatter(xs, ys, c=_MSA_COLORS[s], alpha=0.55, s=20,
                                 label=s, edgecolors="none")
                    _re = f"$\\rho$ = {r_d:+.2f}" if np.isfinite(r_d) else "$\\rho$ = n/a"
                    _stats_e.append(f"{s}: {_re} (n = {len(xs)})")
            ax_d.set_xlabel(r"$\mathrm{RMSD}_{c}$ (Å)")
            ax_d.set_ylabel("Global PAE")
            # correlation stats in an in-plot box; the legend below is just the colour key.
            ax_d.text(0.97, 0.06, "\n".join(_stats_e), transform=ax_d.transAxes,
                      va="bottom", ha="right", fontsize=onpage_to_gen(6.5, 21),
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
            ax_d.legend(fontsize=onpage_to_gen(7, 14), frameon=False, markerscale=2.0,
                        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
            _panel_letter(ax_d, "E", 21)

            # Panel C: mean pTM_r and mean TM_r by RNA MSA depth bin (pTM calibration).
            # Without an MSA pTM collapses to a flat low band (~0.1) that ignores true
            # TM (~0.36) -> the scattered vertical cloud in Panel B.
            xf, yf, df_, lf = pc["ptm"], pc["tm"], pc["depth"], pc["length"]
            f_bins = [
                (df_ == 0, "0"),
                ((df_ > 0) & (df_ < 10), "1–9"),
                ((df_ >= 10) & (df_ < 100), "10–99"),
                ((df_ >= 100) & (df_ < 1000), "100–999"),
                (df_ >= 1000, "≥1000"),
            ]
            def _sem(vals):
                """Standard error of the mean; 0.0 if fewer than 2 values."""
                vals = np.asarray(vals, dtype=float)
                return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) >= 2 else 0.0

            f_x = np.arange(len(f_bins))
            fw = 0.38
            m_pt = [float(xf[m].mean()) if m.any() else np.nan for m, _ in f_bins]
            m_tm = [float(yf[m].mean()) if m.any() else np.nan for m, _ in f_bins]
            e_pt = [_sem(xf[m]) if m.any() else 0.0 for m, _ in f_bins]
            e_tm = [_sem(yf[m]) if m.any() else 0.0 for m, _ in f_bins]
            f_n = [int(m.sum()) for m, _ in f_bins]
            f_medL = [float(np.nanmedian(lf[m])) if m.any() and np.isfinite(lf[m]).any()
                      else np.nan for m, _ in f_bins]
            ax_cal.bar(f_x - fw / 2, m_pt, fw, color="#59A14F", edgecolor="black",
                       yerr=e_pt, capsize=4, error_kw=dict(ecolor="black", lw=1.0),
                       label=r"mean $\mathrm{pTM}_{r}$")
            ax_cal.bar(f_x + fw / 2, m_tm, fw, color="#E15759", edgecolor="black",
                       yerr=e_tm, capsize=4, error_kw=dict(ecolor="black", lw=1.0),
                       label=r"mean $\mathrm{TM}_{r}$")
            # n / median-length annotations: all placed at ONE common height just
            # above the tallest bar (+error) and split onto two lines, so each
            # label stays narrower than the bar-group pitch and never collides
            # with its neighbours (the old per-bar heights let the wide single-line
            # labels of the short bins overlap). The legend moves above the axes so
            # the freed top strip holds the labels without clashing with it.
            all_tops = [v + e for i in range(len(f_bins))
                        for v, e in ((m_pt[i], e_pt[i]), (m_tm[i], e_tm[i]))
                        if np.isfinite(v)]
            lab_y = (max(all_tops) if all_tops else 0.8) + 0.04
            for i, n in enumerate(f_n):
                lab = f"n = {n}"
                if np.isfinite(f_medL[i]):
                    lab += f"\n{f_medL[i]:.0f} nt"
                ax_cal.text(i, lab_y, lab, ha="center", va="bottom",
                            fontsize=onpage_to_gen(6.5, 21), color="dimgray",
                            linespacing=1.05)
            ax_cal.set_xticks(f_x)
            ax_cal.set_xticklabels([l for _, l in f_bins])
            ax_cal.set_xlabel(r"$\mathrm{MSA}_{r}$ depth")
            ax_cal.set_ylabel("Mean value")
            ax_cal.set_ylim(0, lab_y + 0.20)
            ax_cal.legend(fontsize=onpage_to_gen(7, 14), frameon=False,
                          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                          columnspacing=1.2, handletextpad=0.5)
            _panel_letter(ax_cal, "C", 21)

            # Panel F: what each AF3 confidence metric is "centred" on -- Spearman rho
            # of every confidence metric (rows) against each axis (columns: protein /
            # RNA / interface / complex accuracy + RNA MSA depth + RBP template count).
            # The boxed cell in each row is that metric's dominant |r| (its centre);
            # reading down a column shows which metric best tracks that axis.
            R, Nmat, row_lbls, col_lbls = _conf_target_corr()
            im = ax_hm.imshow(R, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
            ax_hm.set_xticks(range(len(col_lbls)))
            # Two-line axis labels are wide for the column pitch: steepen the
            # rotation (smaller horizontal footprint), drop one font step and keep
            # rotation_mode="anchor" so adjacent labels no longer merge.
            xt = ax_hm.set_xticklabels(col_lbls, rotation=55, ha="right",
                                       rotation_mode="anchor",
                                       fontsize=onpage_to_gen(6.5, 21))
            for t in xt:
                t.set_multialignment("center")
            ax_hm.set_yticks(range(len(row_lbls)))
            ax_hm.set_yticklabels(row_lbls, fontsize=onpage_to_gen(7, 14))
            ax_hm.set_xticks(np.arange(-0.5, len(col_lbls), 1), minor=True)
            ax_hm.set_yticks(np.arange(-0.5, len(row_lbls), 1), minor=True)
            ax_hm.grid(which="minor", color="white", lw=1.0)
            ax_hm.tick_params(which="minor", length=0)
            for i in range(R.shape[0]):
                finite = np.where(np.isfinite(R[i]))[0]
                jmax = int(finite[np.argmax(np.abs(R[i][finite]))]) if finite.size else -1
                for j in range(R.shape[1]):
                    if not np.isfinite(R[i, j]):
                        continue
                    ax_hm.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                               fontsize=onpage_to_gen(5.5, 21),
                               fontweight="bold" if j == jmax else "normal",
                               color="white" if abs(R[i, j]) > 0.55 else "black")
                if jmax >= 0:
                    ax_hm.add_patch(plt.Rectangle((jmax - 0.5, i - 0.5), 1, 1,
                                                  fill=False, edgecolor="black", lw=2.2))
            cbar = fig4.colorbar(im, ax=ax_hm, fraction=0.035, pad=0.02)
            cbar.set_label(r"Spearman $\rho$", fontsize=onpage_to_gen(7.5, 21))
            _panel_letter(ax_hm, "F", 21)
        else:
            for ax, letter in [(ax_a, "A"), (ax_b, "B"), (ax_cal, "C"),
                               (ax_cat, "D"), (ax_d, "E"), (ax_hm, "F")]:
                ax.text(0.5, 0.5, "AF3 metrics unavailable", transform=ax.transAxes,
                        ha="center", va="center", fontsize=onpage_to_gen(6.5, 21), color="gray")
                _panel_letter(ax, letter, 21)

        fig4.tight_layout()
        # open the inter-row gap so the below-plot legends of the top-row panels (B, C)
        # do not collide with the bottom row (D/E/F).
        fig4.subplots_adjust(hspace=0.52)
        _save_fig(fig4, "pub_fig4_af3_confidence")
    except Exception as _e4:
        print(f"Warning: pub_fig4 failed: {_e4}", file=sys.stderr)
        plt.close("all")


def plot_publication_figures_only(table, msa_option=None, single_chain_only=False, rbp_template_option=None):
    """Generate ONLY the 4 publication-ready figures (``--pub`` mode).

    Computes just the sample-audit quadrant/interface classification needed by
    ``plot_five_publication_figures`` and skips every other evaluation plot and
    the statistics file.
    """
    print("Processing publication figures only (--pub)...")
    plot_creator = PlotCreator('evaluation_metrics', msa_option, single_chain_only, rbp_template_option)
    rna_metrics = RNAMetrics(msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option)

    if table != "pred_protein_rna":
        print(f"--pub only supports table 'pred_protein_rna'; got '{table}'. Nothing to do.")
        rna_metrics.close_connection()
        return

    audit_lists = rna_metrics.get_sample_audit_global_fold_interface_lists(table)
    if not audit_lists:
        print("No sample-audit data available; cannot build publication figures.")
        rna_metrics.close_connection()
        return

    plot_five_publication_figures(plot_creator, table, audit_lists, rna_metrics)
    print(f"Publication figures written to: {plot_creator.results_dir}")
    rna_metrics.close_connection()


def main():
    # Full-set (n=384) publication figures: no +MSA/+RBPtemp filter.
    plot_publication_figures_only("pred_protein_rna", msa_option=None,
                                  single_chain_only=False, rbp_template_option=None)


if __name__ == "__main__":
    main()
