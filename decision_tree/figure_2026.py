"""
Publication figure: the calibrated probabilities on the independent 2026 set (PDB entries released May-Sep 2026).

  Columns: RNA fold (left), protein-RNA interface (middle), both per RNA chain (right).
  A-C  ROC on 2026: A P_fold vs pLDDT_r and the intra-RNA PAE; B P_interface vs ipTM_r and the chain-pair PAE
       minimum; C P_usable = P_fold x P_interface vs AF3's ranking score and the chain's intra-RNA PAE
  D-F  % outcome per probability level (WEAK < 25% <= MODERATE < 60% <= STRONG) on the post-cutoff development set
       and on 2026, with the mean predicted probability of the 2026 targets in each level
  G-I  ground truth on 2026: G LDDT_r by fold level; H best protein-RNA DockQ by interface level;
       I LDDT_r of each RNA chain against its DockQ, coloured by usable level

Inputs: results/plots/all_chains/decision_tree/probability/2026_{fold,interface,overall}.tsv and
decision_tree/calibration_probability.json (python -m decision_tree.develop_probability --test),
.../v3/2026_{fold,interface}_rows.tsv, .../validation_2026/fold.tsv, .../combined/2026_rows.tsv and
data/2026/protein_rna/2026_a2021.db (DockQ).
Run (RNABench env):  FIG_RENDER=plain python -m decision_tree.figure_2026
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

from decision_tree import combined as CB
from decision_tree import probability as PB
from decision_tree.develop import wilson
from decision_tree.develop_probability import CAL as PROB_CAL, OUT as PROB, VARIANT
from decision_tree.develop_v3 import OUT as V3OUT, _read_rows
from decision_tree.validate import _auc
from decision_tree.validate_2026 import DB26, OUTDIR as V26, local_dockq
from decision_tree.validate_tree import _j

OUTFILE = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "decision_tree", "decision_tree_2026")
LEVEL_COLORS = [figstyle.COLORS["red"], figstyle.COLORS["yellow"], figstyle.COLORS["green"]]
LEVELS = PB.LEVELS


_AF3 = {}
# The AF3 confidence files come with the structures (fetch_data.sh --structures). Without them, the values of a run
# with the structures are read from results/plots_odl (analysis_inputs.zip).
AF3_CACHE = [os.path.join(_REPO_ROOT, "results", d, "all_chains", "decision_tree", "figure_2026_af3.json")
             for d in ("plots", "plots_odl")]
_AF3_SHIPPED = None


def _af3(pid):
    """AF3 values of the 2026 model (per-chain ipTM_r and intra-RNA PAE, ranking score), read once."""
    global _AF3_SHIPPED
    if pid not in _AF3:
        from decision_tree.af3_output import read_prediction
        from decision_tree.validate_2026 import MODELS
        if pid in MODELS and os.path.exists(MODELS[pid]):
            _AF3[pid] = read_prediction(MODELS[pid])
        else:
            if _AF3_SHIPPED is None:
                path = next((p for p in AF3_CACHE if os.path.exists(p)), None)
                _AF3_SHIPPED = json.load(open(path)) if path else {}
            _AF3[pid] = _AF3_SHIPPED.get(pid, {"rna_chains": []})
    return _AF3[pid]


def _save_af3():
    """keep the AF3 values of a run with the structures, for runs without them"""
    if _AF3 and any(v.get("rna_chains") for v in _AF3.values()):
        os.makedirs(os.path.dirname(AF3_CACHE[0]), exist_ok=True)
        paths = ("model", "summary", "full_data")
        json.dump({pid: {k: v for k, v in d.items() if k not in paths} for pid, d in _AF3.items()},
                  open(AF3_CACHE[0], "w"), default=float)


def _tsv(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def _f(v):
    return None if v in ("", "None", None) else float(v)


def roc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    thr = np.unique(s)[::-1]
    tpr = [0.0] + [((s >= t) & y).sum() / y.sum() for t in thr]
    fpr = [0.0] + [((s >= t) & ~y).sum() / (~y).sum() for t in thr]
    return np.array(fpr + [1.0]), np.array(tpr + [1.0])


def _roc_panel(ax, rows, curves, title):
    for key, name, color, ls in curves:
        d = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
        x, y = roc([a for a, _ in d], [b for _, b in d])
        a = _auc([a for a, b in d if b], [a for a, b in d if not b])
        ax.plot(x, y, color=color, ls=ls, lw=1.4 if ls == "-" else 1.0, label=f"{name} ({a:.2f})")
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", frameon=False, fontsize=figstyle.FONT_SMALL - 1, title="AUC on 2026",
              title_fontsize=figstyle.FONT_SMALL - 1, handlelength=1.4, borderaxespad=0.2)


def _rate_panel(ax, names, dev, test, ylabel, dev_label, colors=LEVEL_COLORS, pred=None):
    """observed share per level (95% Wilson CI); pred: 2026 probabilities per level, mean shown as a black bar"""
    x = np.arange(len(names))
    for k, (src, off, hatch) in enumerate(((dev, -0.19, None), (test, 0.19, "///"))):
        for i, nm in enumerate(names):
            p, lo, hi, n = src[nm]
            ax.bar(i + off, 100 * p, width=0.36, color=colors[i], alpha=0.55 if k == 0 else 0.95,
                   hatch=hatch, edgecolor="white", lw=0.5)
            ax.errorbar(i + off, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="0.25", lw=0.8, capsize=2)
            ax.text(i + off, 1.5, f"{n}", ha="center", va="bottom", fontsize=figstyle.FONT_SMALL - 1, color="0.15")
    if pred:
        for i, nm in enumerate(names):
            if pred[nm]:
                ax.plot([i + 0.19 - 0.17, i + 0.19 + 0.17], [100 * np.mean(pred[nm])] * 2, color="k", lw=1.6)
    ax.set_xticks(x, names, fontsize=figstyle.FONT_SMALL)
    ax.set(ylim=(0, 122), ylabel=ylabel)
    ax.set_yticks(range(0, 101, 20))
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(facecolor="0.6", alpha=0.55, label=dev_label),
         Patch(facecolor="0.6", hatch="///", edgecolor="white", label="2026 (test)")]
    if pred:
        h.append(Line2D([], [], color="k", lw=1.6, label="2026 mean predicted"))
    ax.legend(handles=h,
              loc="upper left", frameon=False, fontsize=figstyle.FONT_SMALL)


def _truth_panel(ax, values_by_level, names, ylabel, threshold, colors=LEVEL_COLORS):
    rng = np.random.default_rng(0)
    for i, nm in enumerate(names):
        v = np.asarray(values_by_level[nm], float)
        if len(v):
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=colors[i], alpha=0.35, lw=0.7),
                       medianprops=dict(color="0.1", lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=7, color=colors[i],
                       edgecolor="0.2", lw=0.3, zorder=3)
        ax.text(i, -0.08, f"n={len(v)}", ha="center", va="top", fontsize=figstyle.FONT_SMALL - 1)
    ax.axhline(threshold, color="0.3", lw=0.7, ls="--")
    ax.set_xticks(range(len(names)), names, fontsize=figstyle.FONT_SMALL)
    ax.set(ylim=(-0.12, 1.02), ylabel=ylabel)


def _prob(name):
    return {(r["pdb"], r["chain"]): (_f(r["probability"]), r["level"]) for r in _tsv(os.path.join(PROB, f"2026_{name}.tsv"))}


def main():
    figstyle.apply()
    pct = r"\%" if matplotlib.rcParams.get("text.usetex") else "%"
    cal = json.load(open(PROB_CAL))

    # ---------------- fold rows (2026): longest RNA chain ----------------
    pf = _prob("fold")
    lddt = {r["pdb"]: _f(r["rna_lddt"]) for r in _tsv(os.path.join(V26, "fold.tsv"))}
    fold = []
    for r in _read_rows(os.path.join(V3OUT, "2026_fold_rows.tsv")):
        p, lv = pf.get((r["pdb"], ""), (None, None))
        if p is None:
            continue
        rc = _af3(r["pdb"])["rna_chains"]
        pae = max(rc, key=lambda c: c["length"]).get("rna_pae") if rc else None
        fold.append(dict(y=bool(r["outcome"]), lddt=lddt.get(r["pdb"]), prob=p, lv=lv, plddt=r.get("rna_plddt"),
                         pae=None if pae is None else -pae))          # lower PAE = more confident

    # ---------------- interface rows (2026): every RNA chain with a protein-RNA DockQ ----------------
    dq = {}
    for eid, raw, pch in sqlite3.connect(DB26).execute("SELECT exp_db_id, RBP_DockQ, ProteinChainIDs FROM pred_protein_rna"):
        pc = _j(pch, [])
        pc = [str(x) for x in (pc if isinstance(pc, list) else [pc])]
        raw = _j(raw) if isinstance(_j(raw), list) else local_dockq(str(eid).strip().upper())   # as in validate_2026
        d = {k: v for x in (raw or []) if isinstance(x, dict) for k, v in x.items() if v is not None}
        dq[str(eid).strip().upper()] = (d, pc)
    pi = _prob("interface")
    iface = []
    for r in _tsv(os.path.join(V3OUT, "2026_interface_rows.tsv")):
        p, lv = pi.get((r["pdb"], r["chain"]), (None, None))
        d, pc = dq.get(r["pdb"], ({}, []))
        vals = [v for k, v in d.items() if r["chain"] in k.split(":") and any(q in k.split(":") for q in pc)]
        if p is None or not vals:
            continue
        pm = _f(r["pae_min"])
        iface.append(dict(y=r["outcome"] == "True", dockq=max(vals), prob=p, lv=lv, iptm=_f(r["chain_iptm"]),
                          pae_min=None if pm is None else -pm))

    # ---------------- usable RNA chains (2026) ----------------
    po = _prob("overall")
    rule = os.environ.get("COMBINE", cal["overall"]["rule"])
    if rule != cal["overall"]["rule"]:
        from decision_tree import develop_probability as DP
        fn, m26 = DP.inputs()
        _, cache = DP._af3_cache()
        dev_rows = {}
        for per, F, I, U, cifs in (("post", "post_fold_rows.tsv", "post_interface_rows.tsv", "post_rows.tsv", fn),
                                   ("2026", "2026_fold_rows.tsv", "2026_interface_rows.tsv", "2026_rows.tsv", m26)):
            fr, ir = DP.fold_rows(F, cifs, cache), DP.iface_rows(I)
            ur = DP.usable_rows(U, dict(zip([r["pdb"] for r in fr], DP.apply(cal["fold"], fr))),
                                dict(zip([(r["pdb"], r["chain"]) for r in ir], DP.apply(cal["interface"], ir))))
            pu = DP.combine([r["pf"] for r in ur], [r["pi"] for r in ur], rule, cal["overall"].get("rho", 0.40))
            dev_rows[per] = (ur, pu)
        po = {(r["pdb"], r["chain"]): (float(p), LEVELS[DP.level(p)]) for r, p in zip(*dev_rows["2026"])}
        cal["post"]["overall"] = DP.rates(dev_rows["post"][1], [r["y"] for r in dev_rows["post"][0]])
    comb = []
    for r in _tsv(os.path.join(CB.OUT, "2026_rows.tsv")):
        p, lv = po.get((r["pdb"], r["chain"]), (None, None))
        if p is None:
            continue
        pae = {c["chain"]: c.get("rna_pae") for c in _af3(r["pdb"])["rna_chains"]}.get(r["chain"])
        comb.append(dict(y=r["outcome"] == "True", prob=p, lv=lv, chain_lddt=_f(r["chain_lddt"]), dockq=_f(r["dockq"]),
                         ranking=_af3(r["pdb"]).get("ranking_score"), pae=None if pae is None else -pae))

    fig, axes = plt.subplots(3, 3, figsize=(figstyle.WIDTH_2COL, 8.0))
    (aA, aB, aG), (aC, aD, aH), (aE, aF, aI) = axes
    blue, grey, purple = (figstyle.COLORS[k] for k in ("blue", "neutral", "purple"))
    # tree (blue), AF3 output (grey), derived from AF3's PAE matrix by us (purple, dashed)
    _roc_panel(aA, fold, [("prob", "P$_{fold}$ (tree)", blue, "-"),
                          ("plddt", "pLDDT$_r$ (AF3 output)", grey, "-"),
                          ("pae", "Intra-RNA PAE (from AF3 PAE)", purple, "--")],
               "RNA fold")
    _roc_panel(aB, iface, [("prob", "P$_{interface}$ (tree)", blue, "-"),
                           ("iptm", "ipTM$_r$ (AF3 output)", grey, "-"),
                           ("pae_min", "Chain-pair PAE$_{min}$ (AF3 output)", purple, "--")],
               "Interface")
    _roc_panel(aG, comb, [("prob", "P$_{usable}$ (tree" + ("" if rule == cal["overall"]["rule"] else f", {rule}") + ")", blue, "-"),
                          ("ranking", "Ranking score (AF3 output)", grey, "-"),
                          ("pae", "Intra-RNA PAE (from AF3 PAE)", purple, "--")],
               "Both")

    for ax, side, rows, ylabel in ((aC, "fold", fold, f"RNAs well folded ({pct})"),
                                   (aD, "interface", iface, f"Correct interfaces ({pct})"),
                                   (aH, "overall", comb, f"Usable RNA chains ({pct})")):
        dev = {nm: tuple(cal["post"][side][nm][k] for k in ("rate", "ci_lo", "ci_hi", "n")) for nm in LEVELS}
        test = {}
        for nm in LEVELS:
            s = [r["y"] for r in rows if r["lv"] == nm]
            test[nm] = (*wilson(sum(s), len(s)), len(s)) if s else (0.0, 0.0, 0.0, 0)
        pred = {nm: [r["prob"] for r in rows if r["lv"] == nm] for nm in LEVELS}
        _rate_panel(ax, LEVELS, dev, test, ylabel, "post-cutoff (development)", pred=pred)

    _truth_panel(aE, {nm: [r["lddt"] for r in fold if r["lv"] == nm] for nm in LEVELS},
                 LEVELS, "LDDT$_r$, longest RNA chain (2026)", 0.85)
    _truth_panel(aF, {nm: [r["dockq"] for r in iface if r["lv"] == nm] for nm in LEVELS},
                 LEVELS, "Best DockQ$_{pr}$ (2026)", 0.23)
    for i, nm in enumerate(LEVELS):
        d = [r for r in comb if r["lv"] == nm and r["dockq"] is not None]
        aI.scatter([r["chain_lddt"] for r in d], [r["dockq"] for r in d], s=9, color=LEVEL_COLORS[i],
                   edgecolor="0.2", lw=0.3, zorder=3, label=f"{nm} (n={len(d)})")
    aI.axvline(CB.GOOD_LDDT, color="0.3", lw=0.7, ls="--")
    aI.axhline(0.23, color="0.3", lw=0.7, ls="--")
    aI.set(xlim=(0, 1.02), ylim=(-0.02, 1.02), xlabel="LDDT$_r$, every RNA chain (2026)", ylabel="Best DockQ$_{pr}$ (2026)")
    aI.legend(loc="upper left", frameon=False, fontsize=figstyle.FONT_SMALL, handletextpad=0.2)
    for ax, letter in zip(axes.flat, "ABCDEFGHI"):
        figstyle.panel_letter(ax, letter)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    out = OUTFILE + (f"_{VARIANT}" if VARIANT else "") + ("" if rule == cal["overall"]["rule"] else f"_{rule}")
    figstyle.save(fig, out, formats=("pdf", "png"))
    print(f"wrote {out}.pdf/.png  (fold n={len(fold)}, interface n={len(iface)}, usable n={len(comb)})")


if __name__ == "__main__":
    main()
    _save_af3()
