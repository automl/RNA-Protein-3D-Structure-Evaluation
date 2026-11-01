"""Supporting-information figures for the inputs/confidence analysis.

si_confidence   A  pTM_p by fold class                     (pub_fig4 panel A)
                B  mean pTM_r and TM_r by RNA MSA depth    (pub_fig4 panel C)

si_input_ablation
                A  which inputs each complex carries, 1v1 vs multi-chain
                B  all four inputs present vs all four removed, per complex
                C  effect of each input on the complex (dTM_c)
                D-F  every one of the 16 input combinations, on the complexes that carry
                   all four, for TM_c / TM_r / DockQ_Nr (mean +/- SEM). DockQ_Nr is the
                   mean over the protein-RNA interfaces, so it is not diluted by the
                   protein-protein interfaces of the larger assemblies.
"""
import os
import sys

import numpy as np
import pandas as pd
import sqlite3

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CONF_DB = os.path.join(_REPO, "data", "nar_msa_template_ablation_20261007", "a2021.db")
_CONF_NAME = "config_inputs_and_confidence.json"
if os.path.exists(CONF_DB):
    import json as _json
    with open(os.path.join(_REPO, "database", _CONF_NAME), "w") as _fh:
        _json.dump({"classification": "protein_rna",
                    "base_data_path": os.path.dirname(CONF_DB),
                    "db_path": CONF_DB}, _fh, indent=4)
    os.environ["EVAL_CONFIG_NAME"] = _CONF_NAME

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC
from plots.plotCreator import PH_RANGE_COLORS
from publication_figures.ablation_data import AB, FACTORS, availability, load_factorial
from publication_figures.ablation_effects import paired_differences, summarise
from publication_figures.inputs_and_confidence_figure import FACTOR_LABEL

PUB_RC.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
               "legend.fontsize": 12})
C_1V1, C_MULTI = PH_RANGE_COLORS["5-6"], PH_RANGE_COLORS["6-7"]
QUAD_COLORS = ["#4E79A7", "#59A14F", "#E15759", "#B07AA1"]
QUAD_LABELS = ["(i) both", "(ii) RNA only", "(iii) protein only", "(iv) neither"]
QUAD_KEYS = ["quad_i", "quad_ii", "quad_iii", "quad_iv"]


# --------------------------------------------------------------------------- #
def si_confidence(outdir=None):
    from evaluation.evaluationMetrics import (RNAMetrics, _audit_extract_motif_bin_metrics,
                                              _audit_load_prediction_by_eid)
    rm = RNAMetrics(msa_option=None, single_chain_only=False, rbp_template_option=None)
    audit = rm.get_sample_audit_global_fold_interface_lists("pred_protein_rna")
    by_eid = _audit_load_prediction_by_eid(rm, "pred_protein_rna")
    series = []
    for qk in QUAD_KEYS:
        vals = []
        for rec in audit.get(f"{qk}_all") or []:
            eid = str(rec.get("exp_db_id", "")).strip()
            if by_eid and eid in by_eid:
                v = _audit_extract_motif_bin_metrics(by_eid[eid]).get("af3_protein_pTM")
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        series.append(vals)
    rm.close_connection()

    def panel_ptm_p(ax):
        keep = [(l, s, c) for l, s, c in zip(QUAD_LABELS, series, QUAD_COLORS) if len(s)]
        bp = ax.boxplot([s for _, s, _ in keep], positions=np.arange(len(keep)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="0.15", lw=1.2))
        for box, (_, _, c) in zip(bp["boxes"], keep):
            box.set(facecolor=c, alpha=0.75, lw=0.7, edgecolor="0.3")
        for key in ("whiskers", "caps"):
            for ln in bp[key]:
                ln.set(color="0.45", lw=0.8)
        for xi, (_, s_, _) in enumerate(keep):
            ax.text(xi, -0.02, f"n={len(s_)}", ha="center", va="top", fontsize=8, color="0.45")
        ax.set_xticks(np.arange(len(keep)))
        ax.set_xticklabels([l for l, _, _ in keep], rotation=20, ha="right")
        ax.set_ylabel(r"pTM$_p$")
        ax.set_ylim(-0.05, 1.05)
        style_axis(ax, grid_axis="y")

    def panel_msa_depth(ax):
        """Mean pTM_r and TM_r by RNA MSA depth (pub_fig4 panel C), per RNA chain."""
        from publication_figures.inputs_and_confidence_figure import (confidence_data,
                                                                      MSA_DEPTH_BINS,
                                                                      MSA_COLORS)
        pc = panel_msa_depth.pc
        xs, mp, mt, ns, medL = [], [], [], [], []
        for lo, hi, lab in MSA_DEPTH_BINS:
            m = (pc["depth"] > lo) & (pc["depth"] <= hi)
            if m.sum() < 3:
                continue
            xs.append(lab); mp.append(pc["ptm"][m].mean()); mt.append(pc["tm"][m].mean())
            ns.append(int(m.sum()))
            L = pc["length"][m]
            medL.append(float(np.nanmedian(L)) if np.isfinite(L).any() else np.nan)
        x = np.arange(len(xs))
        ax.plot(x, mp, "o-", color=MSA_COLORS["+MSA"], lw=2, ms=7, label=r"pTM$_r$")
        ax.plot(x, mt, "s--", color=MSA_COLORS["-MSA"], lw=2, ms=7, label=r"TM$_r$")
        # pub_fig4 panel C annotates each bin with its n and median RNA length
        top = max(max(mp), max(mt))
        for xi, (n, L) in enumerate(zip(ns, medL)):
            lab = f"n = {n}" + (f"\n{L:.0f} nt" if np.isfinite(L) else "")
            ax.text(xi, top + 0.04, lab, ha="center", va="bottom", fontsize=10,
                    color="dimgray", linespacing=1.05)
        ax.set_xticks(x); ax.set_xticklabels(xs)
        ax.set_xlabel(r"MSA$_r$ depth"); ax.set_ylabel("Mean value")
        ax.set_ylim(0, top + 0.22)
        # legend below the axes, as in pub_fig4 panel C, so it cannot sit on the n / nt labels
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=2, fontsize=12, columnspacing=1.4, handletextpad=0.5)
        style_axis(ax, grid_axis="y")

    from publication_figures.inputs_and_confidence_figure import confidence_data
    panel_msa_depth.pc = confidence_data()[0]

    fig = PublicationFigure(ncols=2, panel_size=(5.6, 4.5), outdir=outdir)
    for draw in (panel_ptm_p, panel_msa_depth):
        fig.add_panel(draw, title=None)
    return fig.save("si_confidence", formats=("png", "pdf"))


# --------------------------------------------------------------------------- #
def si_input_ablation(outdir=None):
    fac = load_factorial(verbose=False)
    av = availability()
    av = av[av.exp_db_id.isin(set(fac.exp_db_id))].set_index("exp_db_id")
    cls = av["cls"].to_dict()
    carriers = {f: set(av.index[av[f]]) for f in FACTORS}
    ids = {g: {t for t in av.index if cls.get(t) == g} for g in ("1v1", "multi")}
    all_four = list(av.index[av[list(FACTORS)].all(axis=1)])

    def panel_available(ax):
        rows = list(FACTORS) + ["all"]
        labels = [FACTOR_LABEL[f] for f in FACTORS] + ["All four"]
        x = np.arange(len(rows))
        vals_all = []
        for k, (g, col) in enumerate((("1v1", C_1V1), ("multi", C_MULTI))):
            sel = av.loc[av.index.isin(ids[g])]
            vals = [int(sel[list(FACTORS)].all(axis=1).sum()) if r == "all" else int(sel[r].sum())
                    for r in rows]
            vals_all.extend(vals)
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=col, zorder=3,
                          label=f"{g.capitalize()} (n={len(ids[g])})")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + max(vals_all) * 0.015, str(v),
                        ha="center", va="bottom", fontsize=9, color="0.25")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Number of complexes")
        ax.set_ylim(0, max(vals_all) * 1.18)
        ax.legend(frameon=False, loc="upper right")
        style_axis(ax, grid_axis="y")

    def panel_all_four(ax):
        full = fac[(fac[list(FACTORS)] == 1).all(axis=1)].set_index("exp_db_id")["Complex_TM"]
        none = fac[(fac[list(FACTORS)] == 0).all(axis=1)].set_index("exp_db_id")["Complex_TM"]
        common = sorted((set(full.index) & set(none.index)) & set(all_four))
        for g, col in (("1v1", C_1V1), ("multi", C_MULTI)):
            sel = [t for t in common if cls.get(t) == g]
            for t in sel:
                ax.plot([0, 1], [none[t], full[t]], color=col, lw=1.4, alpha=0.75, zorder=3)
            ax.plot([0] * len(sel), [none[t] for t in sel], "o", ms=5, color=col, zorder=4)
            ax.plot([1] * len(sel), [full[t] for t in sel], "o", ms=5, color=col, zorder=4,
                    label=f"{g.capitalize()} (n={len(sel)})")
        ax.set_xlim(-0.25, 1.25); ax.set_xticks([0, 1])
        ax.set_xticklabels(["All four\nremoved", "All four\npresent"])
        ax.set_ylabel(r"TM$_c$")
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax, grid_axis="y")

    # Label order puts MSA_p first, because it is the input that separates the conditions.
    COMBO_ORDER = ["pMSA", "rMSA", "cTemp", "pTemp"]
    COMBO_HEADER = r"MSA$_p$ MSA$_r$ Tmpl$_c$ Tmpl$_p$"

    # One row order shared by all three combination panels, so the same line is the same
    # condition in each (sorting each panel by its own metric scrambled them). The order is
    # lexicographic on the label with "+" before "-", reading MSA_p MSA_r Tmpl_c Tmpl_p, so
    # the rows run +xxx (8, MSA_p present), then -+xx (4), then --xx (4), top to bottom.
    _sub_all4 = fac[fac.exp_db_id.isin(all_four)]
    _tmc = _sub_all4.groupby(list(FACTORS))["Complex_TM"].mean()
    COMBO_KEYS = sorted(_tmc.index,
                        key=lambda k: tuple(0 if dict(zip(FACTORS, k))[f] else 1
                                            for f in COMBO_ORDER),
                        reverse=True)      # barh draws index 0 at the bottom

    def make_combination_panel(metric, xlabel):
        """Mean +/- SEM of `metric` for each of the 16 input combinations, over the
        complexes that carry all four inputs. Rows are grouped by MSA_p, then by the
        number of inputs present, so the MSA_p block reads as one set."""
        def draw(ax):
            sub = fac[fac.exp_db_id.isin(all_four)]
            stat = sub.groupby(list(FACTORS))[metric].agg(["mean", "sem", "size"])
            keys = [k for k in COMBO_KEYS if k in stat.index]
            means = [stat.loc[k, "mean"] for k in keys]
            errs = [stat.loc[k, "sem"] if np.isfinite(stat.loc[k, "sem"]) else 0.0
                    for k in keys]
            labels, colors = [], []
            for k in keys:
                d = dict(zip(FACTORS, k))
                labels.append("".join("+" if d[f] else "\u2212" for f in COMBO_ORDER))
                colors.append(PH_RANGE_COLORS["5-6"] if d["pMSA"] else "0.72")
            y = np.arange(len(keys))
            ax.barh(y, means, xerr=errs, color=colors, zorder=3, height=0.72,
                    error_kw=dict(ecolor="0.35", lw=1.1, capsize=2.5))
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontfamily="monospace", fontsize=10)
            ax.set_ylabel(COMBO_HEADER, fontsize=11)
            ax.set_xlabel(xlabel)
            top = max(m + e for m, e in zip(means, errs))
            ax.set_xlim(0, top * 1.08)
            ax.plot([], [], "s", color=PH_RANGE_COLORS["5-6"], label=r"MSA$_p$ present")
            ax.plot([], [], "s", color="0.72", label=r"MSA$_p$ removed")
            ax.legend(frameon=False, loc="lower right", fontsize=11)
            ax.annotate(f"N={len(all_four)} complexes carrying all four",
                        (0.98, 1.01), xycoords="axes fraction", ha="right",
                        fontsize=10, color="0.4")
            style_axis(ax, grid_axis="x")
        return draw

    def panel_complex_effect(ax):
        ypos = np.arange(len(FACTORS))[::-1]
        for k, (g, col) in enumerate((("1v1", C_1V1), ("multi", C_MULTI))):
            off = (k - 0.5) * 0.3
            for yv, f in zip(ypos, FACTORS):
                d = paired_differences(fac, f, "Complex_TM")
                d = d[d.index.isin(carriers[f] & ids[g])]
                s = summarise(d)
                if not np.isfinite(s["mean"]):
                    continue
                ax.plot([s["lo"], s["hi"]], [yv + off] * 2, color=col, lw=2, zorder=3)
                ax.plot(s["mean"], yv + off, "o", ms=6, color=col, zorder=4,
                        label=f"{g.capitalize()} (n={len(ids[g])})" if f == FACTORS[0] else None)
        ax.axvline(0, color="0.7", lw=1, zorder=1)
        ax.set_yticks(ypos); ax.set_yticklabels([FACTOR_LABEL[f] for f in FACTORS])
        ax.set_xlabel(r"$\Delta$TM$_c$ (input present $-$ removed)")
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax, grid_axis="x")

    fig = PublicationFigure(ncols=3, panel_size=(5.4, 4.6), outdir=outdir)
    for draw in (panel_available, panel_all_four, panel_complex_effect,
                 make_combination_panel("Complex_TM", r"Mean TM$_c$"),
                 make_combination_panel("RNA_TM", r"Mean TM$_r$"),
                 make_combination_panel("DockQ_Nr", r"Mean DockQ$_{Nr}$")):
        fig.add_panel(draw, title=None)
    return fig.save("si_input_ablation", formats=("png", "pdf"))


def main():
    si_confidence()
    si_input_ablation()


if __name__ == "__main__":
    main()
