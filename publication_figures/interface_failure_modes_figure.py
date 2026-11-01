"""
Main figure: how AlphaFold3 protein-RNA interfaces fail.

  A  interface quality (DockQ_Nr subclasses a/b/c) within each global-fold class i-iv
     (get_sample_audit_global_fold_interface_lists; post-cutoff set, database/rbpDatabase.db)
  B  placement of every protein-RNA interface, pre- vs post-cutoff
     (evaluation/interface_rna_pose.py), classes ordered by severity
  C  pose of the incorrect interfaces where AF3 contacts < 10% of the native protein residues and
     the protein binding region itself is right: whole-RNA translation vs rotation in the protein
     frame, with the 10 A / 20 A / 45 deg cut-offs; correct interfaces in grey
  D  Spearman rho of the complex-level protein / RNA contact match and of the share of each
     placement class among a complex's interfaces with fold, confidence, interface and size
     covariates (heatmap of motif_vs_interface_figure.py; post-cutoff complexes), plus the RNA
     base-pair fidelity and conformation measures of evaluation/interface_class_rna_geometry.py and
     the lowest protein-protein DockQ of the complex (assembly placement)

All panels read the post-cutoff database (FIG_DB), not the working database/rbpDatabase.db, which
belongs to whichever folder mainExecution.py processed last.

The per-interface protein vs RNA contact-match panels are the SI figure
interface_motif_match_figure.py. Drawn at the NAR double-column width with the central style
(figstyle.py), so the fonts keep their point size in print.

Usage (RNABench env):  python publication_figures/interface_failure_modes_figure.py
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "publication_figures")):
    if p not in sys.path:
        sys.path.insert(0, p)

import figstyle
import motif_vs_interface_figure as mvi
from evaluation.interface_rna_pose import NEARBY_A, OUTDIR as POSE_DIR, SAME_PLACE_A, TURNED_DEG
from evaluation.interface_class_rna_geometry import complex_features
from evaluation.interface_class_composition import OUTDIR as COMPOSITION_DIR, ost_index, protein_protein_dockq

from evaluation.bundle_paths import PROTEIN_RNA_DB
FIG_DB = PROTEIN_RNA_DB["post"]
mvi._db_path = lambda: FIG_DB
# Panel D: rows = placement-class shares (outcomes); columns = descriptors. The covariates of
# motif_vs_interface_figure.py without the two weakest (MSA_r depth, RBP template), with the
# assembly placement (lowest protein-protein DockQ) and the protein / RNA contact match in the
# interface block, then (right of the line) RNA base-pair fidelity and conformation.
_BASE = [c for c in mvi.HEAT_COLS if c[0] not in ("msa_depth", "has_rbp", "rna_len")]
_k = [c[0] for c in _BASE].index("DockQ_Nr") + 1
HEAT_COLS = _BASE[:_k] + [("pp_dockq_min", "DockQ$_{pp}$"), ("AAmatch_score", "Protein contact match"),
                          ("RNAmatch_score", "RNA contact match"), ("rna_len", "RNA length")] + [
    ("inf_nonwc", "Non-WC INF"), ("inf_stack", "Stacking INF"),
    ("exp_non_c3endo", "Non-C3$'$-endo"), ("exp_syn", "Syn bases"), ("exp_non_gplus", "Non-g$^+$ $\\gamma$"),
]
N_BASE_COLS = [c[0] for c in HEAT_COLS].index("inf_nonwc")

OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "publication_figures")
POSE_TSV = next((p for p in (os.path.join(POSE_DIR, "interface_rna_pose.tsv"),
                              os.path.join(_REPO_ROOT, "results", "plots_odl", "all_chains", "interface_rna_pose",
                                           "interface_rna_pose.tsv")) if os.path.isfile(p)),
                os.path.join(POSE_DIR, "interface_rna_pose.tsv"))
FS, FSS, FSL = figstyle.FONT_BASE, figstyle.FONT_SMALL, figstyle.FONT_LEGEND

# placement classes in severity order: (analysis label, figure label, colour); colours kept
# distinct from panel A's interface grades (blue / orange / red)
CLASSES = [("correct interface", "Correct interface", "#59A14F"),
           ("same place, duplex slid", "Correct site, slid duplex", "#8CD17D"),
           ("right protein site, wrong RNA pose", "Correct site, incorrect RNA", "#EDC948"),
           ("same place, RNA turned", "Correct site, rotated RNA", "#76B7B2"),
           ("shifted nearby", "Shifted site", "#FF9DA7"),
           ("different region", "Distal site", "#9C755F"),
           ("protein binding region mispredicted", "Incorrect binding site", "#B07AA1")]
# An RNA in place and not rotated (<= 10 A, <= 45 deg) that still fails (6 interfaces) is, like
# class "right protein site, wrong RNA pose", a correct-site failure: shown together.
MERGE = {"same place, same orientation": "right protein site, wrong RNA pose"}
LABEL = {k: lab for k, lab, _ in CLASSES}
COLOR = {k: c for k, _, c in CLASSES}
POSE_CLASSES = ["same place, duplex slid", "same place, same orientation", "same place, RNA turned", "shifted nearby",
                "different region"]
SUB_LABELS = ("good", "intermediate", "poor")
QUAD_LABELS = ["(i)\nRNA +\nprotein\naccurate", "(ii)\nRNA\naccurate", "(iii)\nProtein\naccurate",
               "(iv)\nNeither"]
SUB_COLORS = {"a": "#4E79A7", "b": "#F1A340", "c": "#D7301F"}
# placement shares per complex in panel D (column, row label)
SHARE_ROWS = [("share_correct interface", "Correct interface"),
              ("share_same place, duplex slid", "Correct site, slid duplex"),
              ("share_right protein site, wrong RNA pose", "Correct site, incorrect RNA"),
              ("share_same place, RNA turned", "Correct site, rotated RNA"),
              ("share_shifted nearby", "Shifted site"),
              ("share_different region", "Distal site"),
              ("share_protein binding region mispredicted", "Incorrect binding site")]
# Panel D rows: the placement classification of evaluation/placement_all_chains.py reduced to distal site vs at or
# near the site (every other failed pose: wrong segment, in place, rotated, slid duplex, shifted, incorrect site)
PLACEMENT_TSV = os.path.join(POSE_DIR, "placement_all_chains.tsv")
D_CLASSES = [("correct interface", "Correct interface"),
             ("right protein site, wrong RNA pose", "Correct site, incorrect RNA"), ("shifted nearby", "Shifted site"),
             ("protein binding region mispredicted", "Incorrect binding site"), ("different region", "Distal site")]
# the rotated-RNA and slid-duplex classes are too rare per RNA chain for a row of their own (5 and 4 post-cutoff)
# Panels B and C: the pose classes of that classification (analysis label, figure label, colour), site to far
POSE_V2 = [("correct interface", "Correct interface", "#59A14F"),
           ("same place, duplex slid", "Slid duplex", "#8CD17D"),
           ("RNA rotated at the pocket", "Rotated RNA", "#76B7B2"),
           ("correct RNA placement, incorrect contacts", "Correct site, partial RNA contacts", "#FFBE7D"),
           ("wrong RNA segment at the pocket", "Correct site, incorrect RNA contacts", "#EDC948"),
           ("RNA shifted", "Shifted site", "#FF9DA7"),
           ("protein binding region mispredicted", "Incorrect binding site", "#B07AA1"),
           ("distal site", "Distal site", "#9C755F")]
POSE_V2_COLOR = {k: c for k, _, c in POSE_V2}
# Panel B, bars 2 and 3 (incorrect interfaces): protein assembly and RNA fold, coloured like panel A's grades
ASSEMBLY = [("correct", "Correct", SUB_COLORS["a"]), ("incorrect", "Incorr.", SUB_COLORS["c"])]
PROTEIN_STATE = [("correct", "Correct", SUB_COLORS["a"]), ("intra", "Intra", "#FDBF6F"),
                 ("inter", "Inter", SUB_COLORS["c"]), ("both", "Both", "#8B1A10")]
RNA_FOLD = [("good", "Good", SUB_COLORS["a"]), ("intermediate", "Interm.", SUB_COLORS["b"]),
            ("wrong", "Poor", SUB_COLORS["c"])]


def _f(v):
    return float(v) if v not in ("", "nan", None) else np.nan


def _pose_rows():
    rows = list(csv.DictReader(open(POSE_TSV), delimiter="\t"))
    for r in rows:
        for k in ("dockq", "rna_translation", "rna_rotation", "protein_match", "pocket_rmsd"):
            r[k] = _f(r[k])
        if r["placement"] == "same place, duplex slid":       # panel C: pose of the whole duplex, not of the strand
            r["rna_translation"], r["rna_rotation"] = _f(r.get("duplex_shift")), _f(r.get("duplex_axis_tilt"))
        r["shown"] = MERGE.get(r["placement"], r["placement"])
    return [r for r in rows if np.isfinite(r["dockq"])]


def _placement_df():
    """Interfaces with their pose class, protein assembly and RNA fold (evaluation/placement_all_chains.py).
    Protein assembly correct: DockQ >= 0.23 of the interface's protein chain against every protein partner it
    contacts in the deposited structure; without a protein partner, the protein chain's TM-score >= 0.5.
    RNA fold: good = LDDT_r > 0.85 (and TM_r > 0.5 from 60 nt), intermediate = LDDT_r 0.75-0.85, poor otherwise."""
    d = pd.read_csv(PLACEMENT_TSV, sep="\t")
    d = d[d.dockq.notna()].copy()
    ic = pd.read_csv(os.path.join(COMPOSITION_DIR, "interface_class_composition.tsv"), sep="\t")
    d = d.merge(ic[["period", "pdb", "native_pair", "protein_chain_tm"]], on=["period", "pdb", "native_pair"],
                how="left")
    d["pose"] = d.placement_b1.str.split(" / ").str[-1]
    # the paper's placement classes (evaluation/interface_rna_pose.py), as stored in the databases
    d["paper"] = d.placement.replace(MERGE)
    d["placement"] = np.where(d.paper.isin([k for k, _ in D_CLASSES]), d.paper, "at or near the site")
    pp = d.pp_partner_min
    d["assembly"] = np.where(pp.notna(), np.where(pp >= 0.23, "correct", "incorrect"),
                             np.where(d.protein_chain_tm < 0.5, "incorrect", "correct"))
    # protein errors split: inter-protein (misplaced against its protein partners, DockQ_pp < 0.23) and
    # intra-protein (the chain itself less accurately folded, TM_p <= 0.85)
    inter, intra = pp < 0.23, d.protein_chain_tm <= 0.85
    d["protein_state"] = np.select([inter & intra, inter, intra], ["both", "inter", "intra"], "correct")
    d["rna_fold"] = np.where(d.rna_ok, "good", np.where(d.rna_lddt > 0.75, "intermediate", "wrong"))
    return d


def _placement_rows(d=None):
    d = _placement_df() if d is None else d
    return d[["period", "pdb", "placement"]].to_dict("records")


def _audit():
    import sqlite3
    from evaluation.evaluationMetrics import RNAMetrics
    rm = RNAMetrics(msa_option=None, single_chain_only=False, rbp_template_option=None)
    # both connections (DatabaseMethods and PlotCreator's self.db) are opened on config.json's
    # database: point them at FIG_DB
    for obj in (rm, getattr(rm, "db", None)):
        if obj is None:
            continue
        obj.connection.close()
        obj.connection = sqlite3.connect(FIG_DB)
        obj.connection.row_factory = sqlite3.Row
        obj.cursor = obj.connection.cursor()
    try:
        return rm.get_sample_audit_global_fold_interface_lists("pred_protein_rna")
    finally:
        rm.close_connection()


def _style(ax, grid=None):
    if grid:
        ax.grid(axis=grid, color="0.9", lw=figstyle.GRIDWIDTH)
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------- panels
def panel_classes(ax, audit):
    keys = ["quad_i", "quad_ii", "quad_iii", "quad_iv"]
    x, bottom = np.arange(4), np.zeros(4)
    denom = np.array([sum(len(audit.get(f"{q}_{s}") or []) for s in "abc") for q in keys], float)
    for s, lab in zip("abc", SUB_LABELS):
        frac = np.array([len(audit.get(f"{q}_{s}") or []) for q in keys]) / np.maximum(denom, 1) * 100
        ax.bar(x, frac, 0.66, bottom=bottom, color=SUB_COLORS[s], label=lab, edgecolor="white", lw=0.4)
        bottom += frac
    for xi, n in zip(x, denom):
        ax.text(xi, 101.5, f"n={int(n)}", ha="center", va="bottom", fontsize=FSS)
    ax.set_xticks(x)
    ax.set_xticklabels(QUAD_LABELS, fontsize=FSS - 0.5)
    ax.set_ylabel("Predictions (%)")
    ax.set_ylim(0, 112)
    _style(ax, "y")


def panel_placement(ax, rows):
    periods = [("pre", "Pre-\ncutoff"), ("post", "Post-\ncutoff")]
    x, bottom = np.arange(2), np.zeros(2)
    ns = [sum(r["period"] == p for r in rows) for p, _ in periods]
    for key, _, col in CLASSES:
        frac = np.array([sum(r["period"] == p and r["shown"] == key for r in rows) / n * 100
                         for (p, _), n in zip(periods, ns)])
        ax.bar(x, frac, 0.6, bottom=bottom, color=col, edgecolor="white", lw=0.4)
        for xi, fr, b in zip(x, frac, bottom):
            if fr >= 5:
                ax.text(xi, b + fr / 2, f"{fr:.0f}%", ha="center", va="center", fontsize=FSS,
                        color="white" if key in ("correct interface", "different region",
                                                 "protein binding region mispredicted") else "black")
        bottom += frac
    for xi, n in zip(x, ns):
        ax.text(xi, 101.5, f"n={n}", ha="center", va="bottom", fontsize=FSS)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in periods])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Protein-RNA interfaces (%)")
    _style(ax, "y")


def panel_pose(ax, rows):
    clip_t = lambda v: np.clip(v, 0.3, 150)
    cor = [r for r in rows if r["placement"] == "correct interface" and np.isfinite(r["rna_translation"])]
    ax.scatter(clip_t([r["rna_translation"] for r in cor]), [r["rna_rotation"] for r in cor],
               s=2, c="0.85", edgecolor="none", alpha=0.5, zorder=1, rasterized=True)
    n = 0
    for key in POSE_CLASSES:
        s = [r for r in rows if r["placement"] == key and np.isfinite(r["rna_translation"])]
        for per, mk in (("pre", "o"), ("post", "^")):
            q = [r for r in s if r["period"] == per]
            n += len(q)
            ax.scatter(clip_t([r["rna_translation"] for r in q]), [r["rna_rotation"] for r in q],
                       s=7, marker=mk, c=COLOR[MERGE.get(key, key)], edgecolor="black", linewidth=0.15,
                       alpha=0.85, zorder=3, rasterized=True)
    ax.axvline(SAME_PLACE_A, color="0.3", ls="--", lw=0.8)
    ax.axvline(NEARBY_A, color="0.3", ls=":", lw=0.8)
    ax.plot([0.3, SAME_PLACE_A], [TURNED_DEG, TURNED_DEG], color="0.3", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 150)
    ax.set_ylim(-3, 183)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("RNA translation (Å)")
    ax.set_ylabel("RNA rotation (°)")
    ax.text(0.03, 0.97, f"n = {n}\n$\\circ$ pre, $\\triangle$ post", transform=ax.transAxes, ha="left", va="top",
            fontsize=FSS, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    _style(ax, "both")


def panel_causes(ax, d, ylabel="Protein-RNA interfaces (%)", header="Incorrect interfaces"):
    """Placement class of every interface (or RNA chain); protein state (intra- / inter-protein error) and RNA fold
    of the incorrect ones. Wide
    bars post-cutoff, thin bars to their left pre-cutoff."""
    white = ("#59A14F", "#9C755F", "#B07AA1", "#8B1A10", SUB_COLORS["a"], SUB_COLORS["c"])
    for per, dx, w in (("post", 0.1, 0.52), ("pre", -0.29, 0.16)):
        s = d[d.period == per]
        f = s[s.dockq < 0.23]
        bars = [(s.paper, CLASSES), (f.protein_state, PROTEIN_STATE), (f.rna_fold, RNA_FOLD)]
        for xi, (col, classes) in enumerate(bars):
            bottom = 0.0
            for key, lab, c in classes:
                fr = (col == key).mean() * 100
                ax.bar(xi + dx, fr, w, bottom=bottom, color=c, edgecolor="white", lw=0.4)
                if per == "post" and fr >= 5:
                    txt = f"{fr:.0f}%" if xi == 0 or fr < 10 else f"{lab}\n{fr:.0f}%"
                    ax.text(xi + dx, bottom + fr / 2, txt, ha="center", va="center", fontsize=FSS - 1,
                            color="white" if c in white else "black")
                bottom += fr
            if per == "post":
                ax.text(xi + dx, 101.5, f"n={len(col)}", ha="center", va="bottom", fontsize=FSS - 1)
            else:
                ax.text(xi + dx, 101.5, "pre", ha="center", va="bottom", fontsize=FSS - 2, color="0.35")
    ax.axvline(0.5, color="0.4", lw=0.6, ls=":")
    ax.text(1.5, 110, header, ha="center", va="bottom", fontsize=FSS)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Placement", "Protein", "RNA\nfold"], fontsize=FSS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylabel)
    _style(ax, "y")


def panel_pose_v2(ax, d):
    """Incorrect interfaces (pre and post) whose class the geometry decides: no contact with the native protein site
    (protein contact match < 0.1) and the slid duplexes. Interfaces at the site by contact are classed before any
    distance is considered and are left out. Axes: the translation / rotation the pose rules used."""
    clip_t = lambda v: np.clip(v, 0.3, 150)
    cor = d[d.dockq >= 0.23]
    loc = cor.t_loc.notna()                         # same measure as the pose rules: native interface nucleotides
    ax.scatter(clip_t(cor.t_loc.where(loc, cor.rna_translation)), cor.a_loc.where(loc, cor.rna_rotation), s=2, c="0.85", edgecolor="none", alpha=0.5, zorder=1,
               rasterized=True)
    f = d[(d.dockq < 0.23) & d.pose_t.notna() & (~(d.protein_match >= 0.1) | (d.pose == "same place, duplex slid"))]
    rot = f.pose_a.values
    for key, _, col in POSE_V2[1:]:
        for per, mk in (("pre", "o"), ("post", "^")):
            m = ((f.pose == key) & (f.period == per)).values
            ax.scatter(clip_t(f.pose_t.values[m]), rot[m], s=6, marker=mk, c=col, edgecolor="black",
                       linewidth=0.15, alpha=0.85, zorder=3, rasterized=True)
    ax.axvline(SAME_PLACE_A, color="0.3", ls="--", lw=0.8)
    ax.axvline(NEARBY_A, color="0.3", ls=":", lw=0.8)
    ax.plot([0.3, SAME_PLACE_A], [TURNED_DEG, TURNED_DEG], color="0.3", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 150)
    ax.set_ylim(-3, 183)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("RNA translation (Å)")
    ax.set_ylabel("RNA rotation (°)")
    ax.text(0.03, 0.97, f"n = {len(f)}\n$\\circ$ pre, $\\triangle$ post", transform=ax.transAxes, ha="left", va="top",
            fontsize=FSS, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    _style(ax, "both")


def _heatmap_df(rows, share_rows=None):
    df = mvi._load()
    post = pd.DataFrame([r for r in rows if r["period"] == "post"])
    shares = post.groupby("pdb")["placement"].value_counts(normalize=True).unstack(fill_value=0.0)
    shares.columns = [f"share_{c}" for c in shares.columns]
    df["pdb"] = df["exp_db_id"].str.strip().str.upper()
    df = df.merge(shares, left_on="pdb", right_index=True, how="left")
    for c, _ in share_rows or SHARE_ROWS:
        if c not in df:
            df[c] = np.nan
    feats = complex_features(FIG_DB)
    for c in ("inf_nonwc", "inf_stack", "exp_non_c3endo", "exp_syn", "exp_non_gplus"):
        df[c] = df["pdb"].map(lambda p, c=c: feats.get(p, {}).get(c, np.nan))
    idx = ost_index()

    def _pp_min(pid, pcs):
        pcs = set(json.loads(pcs)) if str(pcs).startswith("[") else {str(pcs)}
        q = [v for _a, _b, v in protein_protein_dockq(idx.get(pid), pcs) if np.isfinite(v)]
        return min(q) if q else np.nan
    import sqlite3
    con = sqlite3.connect(FIG_DB)
    pcs = dict((str(a).strip().upper(), b) for a, b in con.execute("SELECT exp_db_id, ProteinChainIDs FROM pred_protein_rna"))
    con.close()
    df["pp_dockq_min"] = df["pdb"].map(lambda p: _pp_min(p, pcs.get(p, "")))
    # Fold accuracy of the chains that form the interfaces, not of the complex's longest chains
    # (the DB's RNA_TM / RNA_LDDT_allAtom / Protein_TM / Protein_LDDT score the longest chain only):
    # mean over a complex's graded interfaces of the per-chain scores of each interface's protein and
    # RNA chain (evaluation/interface_class_composition.py), weighted like the class shares.
    ic = pd.read_csv(os.path.join(COMPOSITION_DIR, "interface_class_composition.tsv"), sep="\t")
    ic = ic[ic["period"] == "post"]
    per_cx = ic.groupby(ic["pdb"].str.upper())[
        ["rna_chain_tm", "rna_chain_lddt", "protein_chain_tm", "protein_chain_lddt"]].mean()
    for col, src in (("RNA_TM", "rna_chain_tm"), ("RNA_LDDT_allAtom", "rna_chain_lddt"),
                     ("Protein_TM", "protein_chain_tm"), ("Protein_LDDT", "protein_chain_lddt")):
        df[col] = df["pdb"].map(per_cx[src])
    return df


def panel_heatmap(ax, df, share_rows=None):
    mvi.HEAT_COLS = HEAT_COLS
    mvi._heatmap(df, rows=share_rows or SHARE_ROWS)(ax)
    for t in ax.texts:                              # cell values: horizontal, without the leading zero
        s = t.get_text()
        try:
            v = float(s)
            t.set_text(("\u2212" if v < 0 else "+") + f"{abs(v):.2f}".lstrip("0"))
        except ValueError:
            pass
        t.set_rotation(0)
        t.set_fontsize(FSS - 2.5)
    rows = [r for r, _ in (share_rows or SHARE_ROWS)]
    cols = [c for c, _ in HEAT_COLS]
    if "share_intra" in rows:
        ax.axhline(rows.index("share_intra"), color="black", lw=0.8)
    from matplotlib.patches import Rectangle
    for r, c in DEFINITIONAL:
        if r in rows and c in cols:
            ax.add_patch(Rectangle((cols.index(c), rows.index(r)), 1, 1, fill=False, ec="black", lw=1.1, zorder=5))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FS)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FS)
    ax.axvline(N_BASE_COLS, color="black", lw=1.0)  # covariates | RNA base pairing and conformation
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=FSS)
    cb.set_label(r"Spearman $\rho$", fontsize=FS)


PROTEIN_ERROR_SHARES = []        # filled by chain_data(): per-complex shares of intra- / inter-protein errors
# cells of panel D that hold by definition (row, column): boxed
DEFINITIONAL = [("share_intra", "Protein_TM"), ("share_inter", "pp_dockq_min")]


def chain_data():
    """B-D per RNA chain: each RNA chain by its main protein partner (placement_per_rna_chain.py).
    Returns the RNA-chain table, the heatmap rows / row labels and the (period, pdb, native pair) of the main-partner
    interfaces."""
    import placement_per_rna_chain as PRC
    chains = PRC.per_rna_chain(PRC.interfaces())
    chains["paper"] = chains["class"].map({lab: k for k, lab in LABEL.items()})
    chains["placement"] = np.where(chains.paper.isin([k for k, _ in D_CLASSES]), chains.paper, "at or near the site")
    d_rows = chains[["period", "pdb", "placement"]].to_dict("records")
    n_post = pd.Series([r["placement"] for r in d_rows if r["period"] == "post"]).value_counts()
    d_share = [(f"share_{k}", f"{lab} (n={n_post.get(k, 0)})") for k, lab in D_CLASSES]
    # protein errors of the failed RNA chains (panel B; "both" counts in each row), per complex as for the classes
    post = chains[chains.period == "post"]
    failed = post.dockq < 0.23
    errs = pd.DataFrame({"pdb": post.pdb, "share_intra": failed & post.protein_state.isin(["intra", "both"]),
                         "share_inter": failed & post.protein_state.isin(["inter", "both"])})
    d_share += [("share_intra", f"Intra-protein error (n={int(errs.share_intra.sum())})"),
                ("share_inter", f"Inter-protein error (n={int(errs.share_inter.sum())})")]
    PROTEIN_ERROR_SHARES[:] = [errs.groupby("pdb")[["share_intra", "share_inter"]].mean()]
    main_pairs = set(zip(chains.period, chains.pdb, chains.main_partner + ":" + chains.rna_chain))
    return chains, d_rows, d_share, main_pairs


def build(base=None):
    figstyle.apply("plain")
    rows = _pose_rows()
    audit = _audit()
    chains, d_rows, d_share, main_pairs = chain_data()
    hdf = _heatmap_df(d_rows, d_share)
    for c in ("share_intra", "share_inter"):
        hdf[c] = hdf["pdb"].map(PROTEIN_ERROR_SHARES[0][c])
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 5.4))
    # A-C span the full width; D is indented by its row labels
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.14], hspace=0.30, wspace=0.36, left=0.06, right=0.995,
                          top=0.98, bottom=0.40, width_ratios=[1.15, 1.05, 1.1])
    gd = fig.add_gridspec(1, 1, left=0.22, right=0.995, top=0.30, bottom=0.08)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axLA = fig.add_subplot(gs[1, 0]); axLA.axis("off")
    axL = fig.add_subplot(gs[1, 1:]); axL.axis("off")
    axD = fig.add_subplot(gd[0, 0])
    panel_classes(axA, audit)
    panel_causes(axB, chains, ylabel="RNA chains (%)", header="Incorrect interfaces")
    panel_pose(axC, [r for r in rows if (r["period"], r["pdb"], r["native_pair"]) in main_pairs])
    # both legends top-aligned in one row: A's title line level with "Correct interface"
    leg_kw = dict(loc="upper center", bbox_to_anchor=(0.5, 0.55), ncol=3, frameon=False, fontsize=FSL,
                  handlelength=1.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    axLA.legend([plt.Rectangle((0, 0), 1, 1, color=SUB_COLORS[s]) for s in "abc"], list(SUB_LABELS),
                title="Interface (DockQ$_{Nr}$)", title_fontsize=FSL, **leg_kw)
    axL.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CLASSES],
               [lab for _, lab, _ in CLASSES], **leg_kw)
    panel_heatmap(axD, hdf, d_share)
    for ax, ltr in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")):
        figstyle.panel_letter(ax, ltr, x=-0.16 if ax is not axD else -0.26, y=1.03)
    base = base or os.path.join(OUTDIR, "interface_failure_modes")
    figstyle.save(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    print("Saved:", base + ".png")
    return base


if __name__ == "__main__":
    build()
