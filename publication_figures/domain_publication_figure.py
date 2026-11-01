"""
Two-panel Nature-style figure on protein domains (protein-RNA set).

Combines two analyses previously emitted separately by evaluation/plotResults.py:
  A  binding_affinity_vs_domains    -- log10 binding affinity Kd vs domain count
  B  domain_coverage_distribution   -- where predicted interface residues fall
                                       relative to annotated protein domains
                                       (within / partially within / outside)

Reads the protein-RNA tables directly (pred_protein_rna / exp_protein_rna), so it
does not depend on the database/config.json 'classification' setting. Colours
follow the shared project palette (plots.plotCreator.PH_RANGE_COLORS).

Usage:  python -m evaluation.domain_publication_figure
"""

import ast
import json
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, style_axis, PUB_RC
from plots.plotCreator import PH_RANGE_COLORS
import figstyle

# domain_binding_coverage is drawn at the NAR double-column width with the central
# figstyle sizes (included at scale 1): ticks/labels FS, legends and domain names FL,
# n annotations FSM.
FS, FL, FSM = figstyle.FONT_BASE, figstyle.FONT_LEGEND, figstyle.FONT_SMALL

# Nature-style large fonts. These scripts run standalone (python -m ...), so
# mutating the shared PUB_RC only affects this figure's process.
PUB_RC.update({
    "axes.labelsize": 18, "axes.titlesize": 15, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 13, "font.size": 14,
})

# post-cutoff protein-RNA database of the data bundle (data/all_data_zip/all_databases); DOMAIN_FIG_DB overrides it
_DB = os.environ.get("DOMAIN_FIG_DB", os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases",
                                                   "a2021_post_cutoff", "protein_rna.db"))


def _panel_letter(ax, s, x=-0.13, y=1.05):
    """Bold panel letter at top-left (manual; figure built with letters=False)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="bottom", ha="right")

# Shared palette: green/orange/red read intuitively for the coverage stack.
WITHIN_C = PH_RANGE_COLORS["7-8"]    # green  — fully within a domain
PARTIAL_C = PH_RANGE_COLORS["6-7"]   # orange — partially within
OUTSIDE_C = PH_RANGE_COLORS["<5"]    # red    — outside all domains
BOX_C = PH_RANGE_COLORS["5-6"]       # blue   — binding-affinity boxes
XRAY_C = PH_RANGE_COLORS["5-6"]      # blue   — X-ray RMSD bars
EM_C = PH_RANGE_COLORS["6-7"]        # orange — EM RMSD bars

TOP_N_DOMAINS = 12                    # most-represented domains shown in panel B


def _domain_count(cell):
    """Sum the ';'-separated per-chain domain counts (e.g. '1;1;1;1' -> 4)."""
    if pd.isna(cell):
        return None
    parts = [int(c) for c in str(cell).split(";") if c.strip().isdigit()]
    return sum(parts) if parts else None


def _log_affinity(cell):
    """log10 of the stored binding affinity Kd; None if non-positive/missing."""
    try:
        if cell is not None and str(cell).lower() != "none":
            v = float(cell)
            return np.log10(v) if v > 0 else None
    except (ValueError, TypeError):
        return None
    return None


def _load_binding_domain():
    con = sqlite3.connect(_DB)
    df = pd.read_sql_query(
        "SELECT p.Binding_affinity_kd, e.domain_counts "
        "FROM pred_protein_rna p JOIN exp_protein_rna e ON p.exp_db_id = e.PDBId "
        "WHERE e.domain_counts IS NOT NULL", con)
    con.close()
    df["ndom"] = df["domain_counts"].apply(_domain_count)
    df["logkd"] = df["Binding_affinity_kd"].apply(_log_affinity)
    return df.dropna(subset=["ndom", "logkd"])


def _load_domain_coverage():
    """Replicates plotResults.plot_domain_coverage's within/partial/outside count
    per protein domain: do predicted interface (AAMotif) residues fall inside the
    annotated domain residue ranges?"""
    con = sqlite3.connect(_DB)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT domain_ids, domain_names FROM exp_protein_rna "
                "WHERE domain_ids IS NOT NULL AND domain_names IS NOT NULL")
    dmap = {}
    for ids, names in cur.fetchall():
        il, nl = ids.split(";"), names.split(";")
        if len(il) == len(nl):
            dmap.update(dict(zip(il, nl)))
    cur.execute(
        "SELECT e.PDBId, e.domain_ids, e.domain_pos, p.AAMotif, e.ProteinSequence "
        "FROM exp_protein_rna e JOIN pred_protein_rna p ON e.PDBId = p.exp_db_id "
        "WHERE e.domain_pos IS NOT NULL AND p.AAMotif IS NOT NULL")
    rows = cur.fetchall()
    con.close()

    cov = {}   # domain_id -> {within, partial, outside}
    for _pdb, domain_ids, domain_pos, aa_motif, prot_seqs in rows:
        # domain_pos holds residue ranges without a chain, so it applies to every protein chain only when the
        # complex has a single distinct protein (identical copies allowed); otherwise one protein's interface
        # residues would be scored against another protein's domains
        try:
            seqs = json.loads(prot_seqs) if str(prot_seqs).startswith("[") else [prot_seqs]
        except (TypeError, ValueError):
            seqs = [prot_seqs]
        if len({str(x) for x in seqs if x}) != 1:
            continue
        try:
            id_list, pos_list = domain_ids.split(";"), domain_pos.split(";")
            if all(p.strip() == "-" for p in pos_list):
                continue
            dranges = {}
            for d_id, d_pos in zip(id_list, pos_list):
                d_id, d_pos = d_id.strip(), d_pos.strip()
                if d_pos == "-":
                    continue
                # each entry carries its own id ("IPR000999:314-487,523-673"); the order of domain_pos need
                # not follow domain_ids, so the ranges are keyed by that id, not by list position
                if ":" in d_pos:
                    d_id, range_str = (x.strip() for x in d_pos.split(":", 1))
                else:
                    range_str = d_pos
                ranges = []
                for r in range_str.split(","):
                    try:
                        s, e = map(int, r.strip().split("-"))
                        ranges.append((s, e))
                    except ValueError:
                        continue
                if ranges:
                    dranges[d_id] = ranges
            if not dranges:
                continue

            if isinstance(aa_motif, str):
                aa_list = (ast.literal_eval(aa_motif)
                           if aa_motif.startswith("[") else [aa_motif])
            else:
                aa_list = [str(aa_motif)]
            if not isinstance(aa_list, list):
                aa_list = [aa_list]

            for motif in aa_list:
                if not isinstance(motif, str):
                    continue
                nums = re.findall(r"\((\d+)\)", motif)
                if not nums:
                    continue
                aa = [int(n) for n in nums]
                for d_id, ranges in dranges.items():
                    c = cov.setdefault(d_id, {"within": 0, "partial": 0, "outside": 0})
                    within = sum(1 for i in aa
                                 if any(s <= i <= e for s, e in ranges))
                    if within == len(aa):
                        c["within"] += 1
                    elif within > 0:
                        c["partial"] += 1
                    else:
                        c["outside"] += 1
        except Exception:
            continue
    return cov, dmap


def _dockq_nr(rbp_cell, pair_cell):
    """Mean RBP_DockQ over the protein-RNA interfaces in ChainIDpairList_proteinRNA
    (the interface-only DockQ_Nr used elsewhere; matches post_pre_comparison_figure.py
    / motif_vs_interface_figure.py). RBP_DockQ is a keyed dict-list
    [{"A:B": 0.5}, ...]; if the pair list is empty/missing, all pairs are used.
    NaN if no scorable DockQ."""
    allowed = set()
    try:
        pl = ast.literal_eval(str(pair_cell)) if pair_cell is not None else None
    except (ValueError, SyntaxError):
        pl = None
    if isinstance(pl, list):
        for pr in pl:
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                allowed.add(frozenset(str(x) for x in pr))
    s = "" if rbp_cell is None else str(rbp_cell).strip()
    if s in ("", "[]", "None", "null"):
        return np.nan
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan
    vals = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                for k, v in el.items():
                    parts = str(k).split(":")
                    if len(parts) == 2 and (not allowed or frozenset(parts) in allowed):
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            pass
    return float(np.mean(vals)) if vals else np.nan


def _load_domain_dockq():
    """Tidy per-domain, per-method interface DockQ_Nr, attributing each complex's
    DockQ_Nr to every domain it carries. Domains use the domain_names column, which
    is populated far more often than the residue-level domain_pos (mostly '-');
    Panel C therefore spans more complexes than the coverage panel, and a given
    domain legitimately has a larger n than in Panel B. DockQ_Nr (0-1, higher =
    better; mean RBP_DockQ over the protein-RNA interfaces) is interface-focused,
    avoiding the global-superposition artefacts of complex RMSD. Returns columns
    [domain, method ('X-ray'|'EM'), val]."""
    con = sqlite3.connect(_DB)
    df = pd.read_sql_query(
        "SELECT e.domain_names, e.Experiment, p.RBP_DockQ, e.ChainIDpairList_proteinRNA "
        "FROM exp_protein_rna e JOIN pred_protein_rna p ON e.PDBId = p.exp_db_id "
        "WHERE e.domain_names IS NOT NULL AND p.RBP_DockQ IS NOT NULL", con)
    con.close()

    method = {"X-RAY DIFFRACTION": "X-ray", "ELECTRON MICROSCOPY": "EM"}
    rec = []
    for _, row in df.iterrows():
        m = method.get(str(row["Experiment"]).strip())
        if m is None:
            continue
        r = _dockq_nr(row["RBP_DockQ"], row["ChainIDpairList_proteinRNA"])
        if not np.isfinite(r):
            continue
        for d in str(row["domain_names"]).split(";"):
            d = d.strip()
            if d and d != "-":
                rec.append((d, m, r))
    return pd.DataFrame(rec, columns=["domain", "method", "val"])


def _load_domain_coverage_dockq():
    """Unified per-(complex, domain) table so Panels B and C share identical n.

    For every complex with domain positions, an interface motif and an RBP_DockQ,
    each of its domains carries BOTH a coverage class (interface residues fully
    within / partially within / outside the domain, pooled over the complex's motifs)
    AND the complex's interface DockQ_Nr. Domain identity comes from the
    domain_ids->domain_names map (same basis as the coverage panel). Returns
    DataFrame [domain, cover ('within'|'partial'|'outside'), dockq]."""
    con = sqlite3.connect(_DB)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT domain_ids, domain_names FROM exp_protein_rna "
                "WHERE domain_ids IS NOT NULL AND domain_names IS NOT NULL")
    dmap = {}
    for ids, names in cur.fetchall():
        il, nl = ids.split(";"), names.split(";")
        if len(il) == len(nl):
            dmap.update(dict(zip(il, nl)))
    cur.execute(
        "SELECT e.domain_ids, e.domain_pos, p.AAMotif, p.RBP_DockQ, "
        "e.ChainIDpairList_proteinRNA "
        "FROM exp_protein_rna e JOIN pred_protein_rna p ON e.PDBId = p.exp_db_id "
        "WHERE e.domain_pos IS NOT NULL AND p.AAMotif IS NOT NULL "
        "AND p.RBP_DockQ IS NOT NULL")
    rows = cur.fetchall()
    con.close()

    rec = []
    for domain_ids, domain_pos, aa_motif, rbp, pairs in rows:
        dq = _dockq_nr(rbp, pairs)
        if not np.isfinite(dq):
            continue
        try:
            id_list, pos_list = domain_ids.split(";"), domain_pos.split(";")
            if all(p.strip() == "-" for p in pos_list):
                continue
            dranges = {}
            for d_id, d_pos in zip(id_list, pos_list):
                d_id, d_pos = d_id.strip(), d_pos.strip()
                if d_pos == "-":
                    continue
                # each entry carries its own id ("IPR000999:314-487,523-673"); the order of domain_pos need
                # not follow domain_ids, so the ranges are keyed by that id, not by list position
                if ":" in d_pos:
                    d_id, range_str = (x.strip() for x in d_pos.split(":", 1))
                else:
                    range_str = d_pos
                ranges = []
                for r in range_str.split(","):
                    try:
                        s, e = map(int, r.strip().split("-"))
                        ranges.append((s, e))
                    except ValueError:
                        continue
                if ranges:
                    dranges[d_id] = ranges
            if not dranges:
                continue
            aa_l = (ast.literal_eval(aa_motif)
                    if isinstance(aa_motif, str) and aa_motif.startswith("[")
                    else [aa_motif])
            if not isinstance(aa_l, list):
                aa_l = [aa_l]
            aa = []
            for motif in aa_l:
                if isinstance(motif, str):
                    aa += [int(n) for n in re.findall(r"\((\d+)\)", motif)]
            if not aa:
                continue
            for d_id, ranges in dranges.items():
                within = sum(1 for i in aa if any(s <= i <= e for s, e in ranges))
                cover = ("within" if within == len(aa)
                         else "partial" if within > 0 else "outside")
                rec.append((dmap.get(d_id, d_id), cover, dq))
        except Exception:
            continue
    return pd.DataFrame(rec, columns=["domain", "cover", "dockq"])


def _median_domains_per_domain():
    """For each domain name, the median number of protein domains (sum of
    domain_counts, as in Panel A) in the complexes that carry it."""
    from collections import defaultdict
    con = sqlite3.connect(_DB)
    d = pd.read_sql_query("SELECT domain_names, domain_counts FROM exp_protein_rna "
                          "WHERE domain_names IS NOT NULL", con)
    con.close()
    by = defaultdict(list)
    for _, row in d.iterrows():
        raw = str(row["domain_names"]).split(";")
        names = [x.strip() for x in raw if x.strip() and x.strip() != "-"]
        counts = [int(c) if c.strip().isdigit() else 1
                  for c in str(row["domain_counts"]).split(";")]
        if not names:
            continue
        total = sum(counts) if len(counts) == len(raw) else len(names)
        for nm in set(names):
            by[nm].append(total)
    return {k: float(np.median(v)) for k, v in by.items()}


def build():
    # ----- Panel A: binding affinity vs domain count -----
    ba = _load_binding_domain()

    order = sorted(int(c) for c in ba["ndom"].unique())
    groups = {k: [] for k in order}
    for n, v in zip(ba["ndom"], ba["logkd"]):
        groups[int(n)].append(v)

    print(f"Panel A — binding affinity vs domain count (n={len(ba)}):")
    for k in order:
        g = groups[k]
        print(f"  domains {k:>2}: n={len(g):<3} median log10(Kd)={np.median(g):+.1f}")

    def drawA(ax):
        data = [groups[k] for k in order]
        bp = ax.boxplot(data, positions=range(len(order)), widths=0.6,
                        patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=BOX_C, alpha=0.85, edgecolor="black", linewidth=0.6)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.1)
        for el in ("whiskers", "caps"):
            for art in bp[el]:
                art.set(color="0.3", linewidth=0.8)
        # jittered points
        rng = np.random.default_rng(0)
        for i, k in enumerate(order):
            g = groups[k]
            if g:
                xj = i + rng.uniform(-0.16, 0.16, len(g))
                ax.scatter(xj, g, s=3, color="0.25", alpha=0.45, zorder=3,
                           edgecolor="none")
                ax.text(i, 1.01, f"n={len(g)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=FSM, rotation=60,
                        color="0.35")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(k) for k in order])
        ax.set_xlabel("Number of protein domains")
        ax.set_ylabel(r"log$_{10}$ binding affinity $K_d$")
        style_axis(ax, grid_axis="y")

    # ----- Panel B: domain coverage on the FULL set (all samples; n shown only
    # here). Top-N by n, plus one fully-outside-domain case even if it is not in C.
    cov, dmap = _load_domain_coverage()
    items = []
    for d_id, c in cov.items():
        tot = c["within"] + c["partial"] + c["outside"]
        if tot > 0:
            items.append((dmap.get(d_id, d_id), c["within"], c["partial"],
                          c["outside"], tot))
    chosen = sorted(items, key=lambda t: -t[4])[:TOP_N_DOMAINS]
    b_core = [t[0] for t in chosen]                 # top-N; referenced by Panel C
    outsiders = [t for t in items if t[3] == t[4] and t[0] not in set(b_core)]
    if outsiders:                                   # add one fully-outside domain (B only)
        chosen = chosen + [max(outsiders, key=lambda t: t[4])]
    chosen.sort(key=lambda t: t[4])                 # largest n at top
    b_order = [t[0] for t in chosen]                # B display order (incl. outside)
    print(f"\nPanel B — domain coverage (full set; top {TOP_N_DOMAINS} by n + 1 outside):")
    for name, w, p, o, tot in chosen[::-1]:
        print(f"  {name:<24} within={100*w/tot:3.0f}%  partial={100*p/tot:3.0f}%  "
              f"outside={100*o/tot:3.0f}%  n={tot}")

    def drawB(ax):
        y = np.arange(len(chosen))
        win = np.array([100 * t[1] / t[4] for t in chosen])
        par = np.array([100 * t[2] / t[4] for t in chosen])
        out = np.array([100 * t[3] / t[4] for t in chosen])
        ax.barh(y, win, color=WITHIN_C, edgecolor="white", linewidth=0.4,
                label="within domain")
        ax.barh(y, par, left=win, color=PARTIAL_C, edgecolor="white",
                linewidth=0.4, label="partially within")
        ax.barh(y, out, left=win + par, color=OUTSIDE_C, edgecolor="white",
                linewidth=0.4, label="outside domain")
        for yi, t in zip(y, chosen):
            ax.text(101, yi, f"n={t[4]}", va="center", ha="left", fontsize=FSM,
                    color="0.35")
        ax.set_yticks(y)
        ax.set_yticklabels(b_order, fontsize=FL)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Samples (%)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False,
                  handlelength=1.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
        style_axis(ax, grid_axis="x")

    # ----- Panel C: interface DockQ_Nr on the fuller domain_names basis (more
    # samples than the coverage subset). Shows every domain in B plus a few
    # better-sampled, informative ones -- notably canonical RNA-processing domains
    # AF3 nonetheless places poorly (e.g. Dicer dimerisation). n is shown only on B,
    # so B and C need not use an identical subset. Ordered by accuracy.
    cstat = _load_domain_dockq().groupby("domain")["val"].agg(["mean", "std", "count"])
    extras = [d for d in cstat.index
              if d not in list(b_order) and cstat.loc[d, "count"] >= 4]
    c_names = [d for d in b_order if d in cstat.index] + extras
    cstat = cstat.loc[c_names].sort_values("mean")   # display ordered by accuracy
    print(f"\nPanel C — per-domain DockQ_Nr (fuller basis; B's domains + "
          f"{len(extras)} extras, no n shown):")
    for d, r in cstat[::-1].iterrows():
        print(f"  {d:<26} DockQ_Nr={r['mean']:.2f}  n={int(r['count'])}")

    dom_n = _median_domains_per_domain()             # domains-per-complex per domain

    def drawC(ax):
        y = np.arange(len(cstat))
        means = cstat["mean"].to_numpy()
        sems = (cstat["std"] / np.sqrt(cstat["count"])).fillna(0.0).to_numpy()
        ax.barh(y, means, height=0.72, xerr=sems, color=BOX_C, alpha=0.9,
                edgecolor="black", linewidth=0.3,
                error_kw=dict(ecolor="0.3", elinewidth=0.6, capsize=1.2))
        ax.set_yticks(y)
        ax.set_yticklabels(list(cstat.index), fontsize=FL)
        ax.set_xlabel(r"DockQ$_{Nr}$")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        style_axis(ax, grid_axis="x")
        # Secondary top axis: number of protein domains per complex, marked with a
        # star per row -- the poorly-modelled domains sit in large multi-domain proteins.
        counts = [dom_n.get(d, np.nan) for d in cstat.index]
        finite = [c for c in counts if np.isfinite(c)]
        mx = int(max(finite)) if finite else 1
        axt = ax.twiny()
        axt.set_xlim(0.5, mx + 0.5)
        axt.set_xticks(range(1, mx + 1))
        axt.tick_params(axis="x", labelsize=FS)
        axt.set_xlabel(r"protein domains per complex ($\bigstar$)", fontsize=FS)
        axt.scatter(counts, y, marker="*", s=28, color="gold",
                    edgecolor="black", linewidth=0.3, zorder=6)
        for sp in ("top",):
            axt.spines[sp].set_visible(True)

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    import matplotlib.pyplot as plt
    figstyle.apply("plain")
    fig, axs = plt.subplots(1, 3, figsize=(figstyle.WIDTH_2COL, 3.1), layout="constrained",
                            gridspec_kw=dict(width_ratios=[0.75, 0.85, 1.0]))
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.0)
    drawA(axs[0])
    drawB(axs[1])
    drawC(axs[2])
    # panel letters at the left edge of each panel's labels, level with the figure top
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axs, "ABC"):
        x0 = inv.transform(ax.get_tightbbox(r).p0)[0]
        fig.text(max(x0, 0.0), 1.0, letter, fontsize=figstyle.PANEL_LETTER_SIZE,
                 fontweight=figstyle.PANEL_LETTER_WEIGHT, ha="left", va="bottom")
    base = figstyle.save(fig, os.path.join(outdir, "domain_binding_coverage"), formats=("png", "pdf"))
    plt.close(fig)
    print("Saved:", base + ".png (+ .pdf)")
    return base


def _load_domain_counts_dockq():
    """Per-complex (n_types, total_copies) and per-domain-occurrence (copies)
    interface DockQ_Nr, using the index-aligned domain_names / domain_counts lists."""
    con = sqlite3.connect(_DB)
    df = pd.read_sql_query(
        "SELECT e.domain_names, e.domain_counts, p.RBP_DockQ, "
        "e.ChainIDpairList_proteinRNA "
        "FROM exp_protein_rna e JOIN pred_protein_rna p ON e.PDBId = p.exp_db_id "
        "WHERE e.domain_names IS NOT NULL AND p.RBP_DockQ IS NOT NULL", con)
    con.close()

    comp, occ = [], []
    for _, row in df.iterrows():
        dq = _dockq_nr(row["RBP_DockQ"], row["ChainIDpairList_proteinRNA"])
        if not np.isfinite(dq):
            continue
        names = str(row["domain_names"]).split(";")
        counts = [int(c) if c.strip().isdigit() else 1
                  for c in str(row["domain_counts"]).split(";")]
        if len(names) != len(counts):
            continue
        names_ok = [(n.strip(), c) for n, c in zip(names, counts)
                    if n.strip() and n.strip() != "-"]
        if not names_ok:
            continue
        comp.append((len(names_ok), sum(c for _, c in names_ok), dq))
        for nm, c in names_ok:
            occ.append((nm, c, dq))
    return (pd.DataFrame(comp, columns=["n_types", "total_copies", "dq"]),
            pd.DataFrame(occ, columns=["domain", "copies", "dq"]))


def build_domain_counts_supplement():
    """Supplementary: does domain multiplicity (domain_counts) affect interface
    DockQ_Nr? Panel A, per-domain copy number; Panel B, total domain copies per
    complex. Both are essentially flat, in contrast to the binding-affinity vs
    domain-count trend of the main figure."""
    from scipy.stats import spearmanr

    comp, occ = _load_domain_counts_dockq()

    def _sem(a):
        return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0

    # Panel A: DockQ by per-domain copy number
    copy_levels = sorted(occ["copies"].unique())
    a_stats = [(c, occ.loc[occ["copies"] == c, "dq"]) for c in copy_levels]
    rho_a, p_a = spearmanr(occ["copies"], occ["dq"])
    print(f"\nSupplement Panel A — DockQ_Nr by per-domain copy number "
          f"(rho={rho_a:+.3f}, p={p_a:.3f}):")
    for c, s in a_stats:
        print(f"  copies={c}: mean={s.mean():.3f} n={len(s)}")

    def drawA(ax):
        x = np.arange(len(a_stats))
        ax.bar(x, [s.mean() for _, s in a_stats],
               yerr=[_sem(s) for _, s in a_stats], width=0.6, color=BOX_C,
               alpha=0.9, edgecolor="black", linewidth=0.4,
               error_kw=dict(capsize=3, ecolor="0.25", elinewidth=0.9))
        for xi, (_, s) in zip(x, a_stats):
            ax.text(xi, s.mean() + _sem(s) + 0.02, f"n={len(s)}", ha="center",
                    va="bottom", fontsize=11, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c, _ in a_stats])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Domain copy number (per domain)")
        ax.set_ylabel(r"DockQ$_{Nr}$")
        ax.text(0.04, 0.96, f"Spearman ρ = {rho_a:+.2f}\n(n.s., n={len(occ)})",
                transform=ax.transAxes, va="top", ha="left", fontsize=13)
        style_axis(ax, grid_axis="y")

    # Panel B: DockQ by total domain copies per complex (binned)
    bins = [0, 1, 2, 4, np.inf]
    labels = ["1", "2", "3-4", "5+"]
    comp = comp.copy()
    comp["bin"] = pd.cut(comp["total_copies"], bins=bins, labels=labels)
    b_stats = [(lab, comp.loc[comp["bin"] == lab, "dq"]) for lab in labels]
    rho_b, p_b = spearmanr(comp["total_copies"], comp["dq"])
    print(f"\nSupplement Panel B — DockQ_Nr by total domain copies "
          f"(rho={rho_b:+.3f}, p={p_b:.3f}):")
    for lab, s in b_stats:
        print(f"  copies {lab}: mean={s.mean():.3f} n={len(s)}")

    def drawB(ax):
        x = np.arange(len(b_stats))
        ax.bar(x, [s.mean() for _, s in b_stats],
               yerr=[_sem(s) for _, s in b_stats], width=0.6, color=BOX_C,
               alpha=0.9, edgecolor="black", linewidth=0.4,
               error_kw=dict(capsize=3, ecolor="0.25", elinewidth=0.9))
        for xi, (_, s) in zip(x, b_stats):
            ax.text(xi, s.mean() + _sem(s) + 0.02, f"n={len(s)}", ha="center",
                    va="bottom", fontsize=11, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in b_stats])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Total domain copies per complex")
        ax.set_ylabel(r"DockQ$_{Nr}$")
        ax.text(0.04, 0.96, f"Spearman ρ = {rho_b:+.2f}\n(n.s., n={len(comp)})",
                transform=ax.transAxes, va="top", ha="left", fontsize=13)
        style_axis(ax, grid_axis="y")

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "publication_figures")
    fig = PublicationFigure(ncols=2, panel_size=(6.4, 5.2), outdir=outdir)
    fig.add_panel(drawA, title=None)
    fig.add_panel(drawB, title=None)
    return fig.save("domain_counts_dockq")


if __name__ == "__main__":
    build()
    build_domain_counts_supplement()
