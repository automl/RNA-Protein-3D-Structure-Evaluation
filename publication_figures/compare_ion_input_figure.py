"""
Does giving AF3 the ions as input improve RNA-only predictions?

Compares two RNA-only prediction runs on the same structures:
  a2021_rna.db       -- predicted WITHOUT ions
  a2021_rna_ions.db  -- predicted WITH ions
(both under data/RNAonly/compare_ions/). Paired on exp_db_id over all 41
complexes; ligand presence is carried as a covariate (exposed in Panel C)
rather than used to filter, so the full ion count/type/complexation range is
retained. NOTE: restricting to truly ligand-free complexes leaves only n=12,
where the ion effect is not significant — the effect is carried by the
ligand-bound complexes, so ligand presence must be reported alongside.

Reports paired tests (Wilcoxon) for RNA TM / LDDT / RMSD and relates the
per-structure improvement to RNA properties (MSA depth, MFE, ...) and to ion
features (count, Mg, multidentate chelation). Writes a three-panel figure:
A overview, B which RNAs benefit (vs MFE/MSA), C ion-feature dependence.

Usage:  python -m evaluation.compare_ion_input_figure
#Folder: /data/compare_ions
"""

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.publication_figure import PublicationFigure, style_axis, onpage_to_gen
from plots.plotCreator import PH_RANGE_COLORS, RBP_TEMPLATE_COLORS

_BASE = os.path.join(_REPO_ROOT, "data", "compare_ions")
# With-ions predictions (ions supplied to AF3).
WITHIONS = "a2021_rna_ions.db"
PROT_IONS = "a2021_protein_rna_ions.db"
# No-ions baseline = the ground-truth sets (ions deposited but NOT given to AF3, i.e.
# ion-naive predictions), restricted to the PDB ids that appear in the with-ions sets.
RNA_GT = "a2021_rna_ONLY.db"            # RNA-only, 66 complexes
PROT_GT = "a2021_protein_rna.db"        # protein-RNA, 385 complexes

COL_NO = PH_RANGE_COLORS["6-7"]    # without ions (orange)
COL_YES = PH_RANGE_COLORS["5-6"]   # with ions (blue)
COL_NOMSA = PH_RANGE_COLORS["<5"]  # no MSA available (red)
METRICS = [("RNA_LDDT_allAtom", "RNA LDDT", "higher"),
           ("RNA_TM", "RNA TM", "higher"),
           ("RNA_RMSD", "RNA RMSD (Å)", "lower")]
PROPS = [("RNA_msa_size", "MSA depth"), ("RNA_MFE_Value", "MFE"),
         ("RNA_Shannon_Entropy_K2", "Shannon entropy"),
         ("RNA_GC_Content", "GC content"), ("RNA_INF", "INF")]


def _num(cell):
    """Robust scalar: float, or first element of a list/RMSD-style string."""
    if cell is None:
        return np.nan
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan
    if isinstance(v, (list, tuple)) and v:
        return float(v[0])
    return np.nan


def _dockq_nr(rbp_dockq_cell, pairlist_cell):
    """Mean DockQ_Nr: average of RBP_DockQ over the protein-RNA interfaces only.

    RBP_DockQ is a JSON keyed dict-list [{"A:B": 0.5}, ...]; ChainIDpairList_proteinRNA
    is [["A","B"], ...]. Only pairs listed there (order-insensitive) contribute; if that
    list is empty/missing, all RBP_DockQ pairs are used. NaN if no scorable DockQ.
    """
    import json
    allowed = set()
    try:
        pl = json.loads(str(pairlist_cell)) if pairlist_cell is not None else None
    except (ValueError, TypeError):
        pl = None
    if isinstance(pl, list):
        for pr in pl:
            if isinstance(pr, list) and len(pr) == 2:
                allowed.add(frozenset(str(x) for x in pr))
    s = "" if rbp_dockq_cell is None else str(rbp_dockq_cell).strip()
    if s in ("", "[]", "None", "null"):
        return np.nan
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return np.nan
    vals = []
    if isinstance(obj, list):
        for el in obj:
            if not isinstance(el, dict):
                continue
            for k, val in el.items():
                parts = str(k).split(":")
                if len(parts) != 2:
                    continue
                if allowed and frozenset(parts) not in allowed:
                    continue
                try:
                    vals.append(float(val))
                except (TypeError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else np.nan


def _msa_depth(cell):
    """RNA_msa_size is {chain: size}; use the deepest chain MSA."""
    if not cell:
        return np.nan
    try:
        d = json.loads(cell) if isinstance(cell, str) else cell
    except (json.JSONDecodeError, TypeError):
        return np.nan
    vals = [float(v) for v in d.values()] if isinstance(d, dict) else []
    return max(vals) if vals else np.nan


def _ion_list(cell):
    """Parse the Ions / Ligands column.

    Current schema is a list of dicts, e.g.
        [{'ions': 'MG', 'n': 4, 'af3_supported': True}, ...]   (Ions)
        [{'ligands': 'SPD', 'n': 1, 'af3_supported': False}]   (Ligands)
    """
    if not cell:
        return []
    try:
        g = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError):
        return []
    return g if isinstance(g, list) else []


def _has_ligand(cell):
    return len(_ion_list(cell)) > 0


def _ion_species(cell):
    """Set of ion type codes present (e.g. {'MG', 'K'})."""
    return {str(d.get("ions", "")).strip().upper()
            for d in _ion_list(cell) if isinstance(d, dict)}


def _ion_total_n(cell):
    """Total ion count summed over species."""
    return sum(int(d.get("n", 0)) for d in _ion_list(cell) if isinstance(d, dict))


def _has_mg(cell):
    """True if Mg is among the ions (regardless of other ions present)."""
    return "MG" in _ion_species(cell)


def _species_ion_count(cell, species):
    """Count of ions of a given species (e.g. 'MG', 'ZN') in the complex."""
    species = species.strip().upper()
    return sum(int(d.get("n", 0)) for d in _ion_list(cell)
               if isinstance(d, dict)
               and str(d.get("ions", "")).strip().upper() == species)


def _n_mg_ions(cell):
    """Total Mg ion count."""
    return _species_ion_count(cell, "MG")


def _has_rbp_template(cell):
    """True if an RBP PDB template was used (RBP_PDB_Templates is a JSON list)."""
    if cell is None:
        return False
    s = str(cell).strip()
    if s in ("", "[]", "None", "nan", "0"):
        return False
    try:
        x = ast.literal_eval(s)
        return len(x) > 0 if isinstance(x, (list, tuple)) else bool(x)
    except (ValueError, SyntaxError):
        return True


# Ionic charge for the common species, rendered as a mathtext superscript on the
# axis labels (e.g. Na -> Na^+, Mg -> Mg^{2+}); species not listed show no charge.
ION_CHARGE = {
    "MG": "2+", "CA": "2+", "ZN": "2+", "MN": "2+", "NI": "2+", "CO": "2+",
    "CD": "2+", "CU": "2+", "FE": "2+", "SR": "2+", "BA": "2+", "PB": "2+", "HG": "2+",
    "NA": "+", "K": "+", "CS": "+", "LI": "+", "RB": "+", "NH4": "+",
    "CL": "-", "BR": "-", "F": "-", "I": "-", "IOD": "-",
    "SO4": "2-", "PO4": "3-",
}


def _fmt_species(sp):
    """Display an ion code as its ionic form: ZN->Zn$^{2+}$, NA->Na$^+$,
    K->K$^+$ (polyatomic codes with digits kept upper-case; charge from ION_CHARGE)."""
    base = sp if any(ch.isdigit() for ch in sp) else sp.capitalize()
    ch = ION_CHARGE.get(sp.strip().upper())
    return f"{base}$^{{{ch}}}$" if ch else base


def _chelating_contacts(cell):
    """All chelating contact dicts across ion species in Ion_Location.

    Ion_Location is {species: {"chelating": [{atom_types, min_distance,
    multidentate, n_contacts, rna_chain, rna_res}, ...]}}.
    """
    if not cell:
        return []
    try:
        d = json.loads(cell) if isinstance(cell, str) else cell
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(d, dict):
        return []
    return [c for info in d.values() for c in info.get("chelating", [])]


def _n_multidentate(cell):
    """Count of multidentate (inner-sphere) chelating ion contacts."""
    return sum(1 for c in _chelating_contacts(cell) if c.get("multidentate"))


def _n_phos_ribose_chelation(cell):
    """Count of ion contacts that chelate a phosphate or ribose atom."""
    return sum(1 for c in _chelating_contacts(cell)
               if "phosphate" in c.get("atom_types", [])
               or "ribose" in c.get("atom_types", []))


def _n_atom_chelation(cell, atom):
    """Count of ion contacts that chelate the given atom type."""
    return sum(1 for c in _chelating_contacts(cell) if atom in c.get("atom_types", []))


def _min_ion_distance(cell):
    """Closest ion–RNA chelating distance (Å), or NaN if none recorded."""
    ds = [c["min_distance"] for c in _chelating_contacts(cell)
          if c.get("min_distance") is not None]
    return float(np.min(ds)) if ds else np.nan


def _load(fname):
    import sqlite3
    con = sqlite3.connect(os.path.join(_BASE, fname))
    df = pd.read_sql_query(
        "SELECT p.exp_db_id, p.RNA_TM, p.RNA_LDDT_allAtom, p.RNA_RMSD, "
        "p.RNA_msa_size, p.RNA_MFE_Value, p.RNA_Shannon_Entropy_K2, "
        "p.RNA_GC_Content, p.RNA_INF, e.Ligands, e.Ions, e.Ion_Location "
        "FROM pred_rna_rna p JOIN exp_rna_rna e ON p.exp_db_id = e.PDBId", con)
    con.close()
    return df


# Ground-truth sets: ions are present in the deposited structure but were NOT
# given to AF3 (ion-naive predictions). Lets us ask whether ion-rich /
# strongly-chelated complexes are intrinsically more predictable.
def _load_rna_groundtruth():
    """RNA-only ground-truth set (66) with derived ion features."""
    import sqlite3
    con = sqlite3.connect(os.path.join(_BASE, RNA_GT))
    df = pd.read_sql_query(
        "SELECT p.exp_db_id, p.RNA_TM, p.RNA_LDDT_allAtom, p.RNA_RMSD, "
        "e.Ligands, e.Ions, e.Ion_Location "
        "FROM pred_rna_rna p JOIN exp_rna_rna e ON p.exp_db_id = e.PDBId", con)
    con.close()
    for c in ("RNA_TM", "RNA_LDDT_allAtom", "RNA_RMSD"):
        df[c] = df[c].apply(_num)
    df["ion count"] = df["Ions"].apply(_ion_total_n)
    df["Mg count"] = df["Ions"].apply(_n_mg_ions)
    df["ion species"] = df["Ions"].apply(lambda c: len(_ion_species(c)))
    df["phosphate chel."] = df["Ion_Location"].apply(lambda c: _n_atom_chelation(c, "phosphate"))
    df["ribose chel."] = df["Ion_Location"].apply(lambda c: _n_atom_chelation(c, "ribose"))
    df["multidentate"] = df["Ion_Location"].apply(_n_multidentate)
    df["min ion dist."] = df["Ion_Location"].apply(_min_ion_distance)
    df["has_ligand"] = df["Ligands"].apply(_has_ligand)
    df["has_mg"] = df["Ions"].apply(_has_mg)
    return df


def _load_protein_groundtruth():
    """Protein-RNA ground-truth set (385): interface metrics + RBP template."""
    import sqlite3
    con = sqlite3.connect(os.path.join(_BASE, PROT_GT))
    df = pd.read_sql_query(
        "SELECT p.exp_db_id, p.Dockq, p.ICS, p.iLDDT, p.QS_global, "
        "p.RBP_PDB_Templates, e.Ions, e.Ion_Location "
        "FROM pred_protein_rna p JOIN exp_protein_rna e ON p.exp_db_id = e.PDBId", con)
    con.close()
    from evaluation.excluded_targets import POST_EXCLUDED     # 6FQL: in AF3 training
    df = df[~df["exp_db_id"].astype(str).str.strip().str.upper().isin(POST_EXCLUDED)]
    for c in ("Dockq", "ICS", "iLDDT", "QS_global"):
        df[c] = df[c].apply(_num)
    df["ion_n"] = df["Ions"].apply(_ion_total_n)
    df["has_rbp"] = df["RBP_PDB_Templates"].apply(_has_rbp_template)
    return df


def _load_protein_paired():
    """Protein-RNA ion comparison: no-ions baseline = protein-RNA GT (restricted below to
    the with-ions PDB ids), with-ions = the ion-input predictions. Interface quality is
    DockQ_Nr = mean RBP_DockQ over the protein-RNA interfaces (ChainIDpairList_proteinRNA);
    the whole-complex Dockq is no longer used. RBP_DockQ is read where present (currently
    the GT set); if a DB lacks it, DockQ_Nr is NaN until that DB is updated."""
    import sqlite3

    num_cols = ("RNA_LDDT_allAtom", "Complex_LDDT_allAtom", "RNA_RMSD", "Complex_RMSD")

    def _one(fname):
        con = sqlite3.connect(os.path.join(_BASE, fname))
        have = {r[1] for r in con.execute("PRAGMA table_info(pred_protein_rna)")}
        sel = ["exp_db_id"] + [c for c in num_cols if c in have]
        sel += [c for c in ("RBP_DockQ", "ChainIDpairList_proteinRNA") if c in have]
        d = pd.read_sql_query("SELECT " + ", ".join(sel) + " FROM pred_protein_rna", con)
        con.close()
        from evaluation.excluded_targets import POST_EXCLUDED     # 6FQL: in AF3 training
        d = d[~d["exp_db_id"].astype(str).str.strip().str.upper().isin(POST_EXCLUDED)]
        for c in num_cols:
            if c in d.columns:
                d[c] = d[c].apply(_num)
        if "RBP_DockQ" in d.columns:
            pairs = (d["ChainIDpairList_proteinRNA"]
                     if "ChainIDpairList_proteinRNA" in d.columns else [None] * len(d))
            d["DockQ_Nr"] = [_dockq_nr(rq, pr) for rq, pr in zip(d["RBP_DockQ"], pairs)]
        else:
            d["DockQ_Nr"] = np.nan
        return d.set_index("exp_db_id")
    return _one(PROT_GT), _one(PROT_IONS)


def build():
    a = _load(RNA_GT).set_index("exp_db_id")    # no-ions baseline = RNA-only GT
    b = _load(WITHIONS).set_index("exp_db_id")   # with ions
    # keep only the ids that appear in the with-ions set (the ion-input subset)
    # Use all paired complexes; ligand presence is carried as a covariate
    # (exposed in Panel C) rather than used to filter, so we retain power and
    # the ion-count/type/complexation range needed to test ion dependence.
    ids = [i for i in b.index if i in a.index]

    rows = {}
    for col, _lab, _better in METRICS:
        x = a.loc[ids, col].apply(_num).values
        y = b.loc[ids, col].apply(_num).values
        rows[col] = (x, y)

    has_ligand = b.loc[ids, "Ligands"].apply(_has_ligand).values
    ion_total = b.loc[ids, "Ions"].apply(_ion_total_n).values.astype(float)
    n_multi = b.loc[ids, "Ion_Location"].apply(_n_multidentate).values.astype(float)

    print(f"Paired RNA-only complexes: n={len(ids)} "
          f"(ligand-bound {int(has_ligand.sum())}, ligand-free "
          f"{int((~has_ligand).sum())})\n")
    print(f"{'metric':<18} {'no-ions':>8} {'+ions':>8} {'Δ mean':>8} "
          f"{'Δ med':>7} {'Wilcoxon p':>11}  better")
    for col, lab, better in METRICS:
        x, y = rows[col]
        m = np.isfinite(x) & np.isfinite(y)
        xa, ya = x[m], y[m]
        d = ya - xa
        try:
            _w, p = wilcoxon(xa, ya)
        except ValueError:
            p = float("nan")
        print(f"{lab:<18} {xa.mean():8.3f} {ya.mean():8.3f} {d.mean():+8.3f} "
              f"{np.median(d):+7.3f} {p:11.4f}  {better}")

    # Which RNAs benefit? Correlate per-structure improvement with properties.
    print("\nSpearman: per-structure improvement vs RNA property "
          "(Δ = +ions − no-ions; for RMSD Δ uses no-ions − +ions so +Δ = better)")
    bprops = pd.DataFrame(index=ids)
    bprops["MSA depth"] = b.loc[ids, "RNA_msa_size"].apply(_msa_depth).values
    for pc, plab in PROPS:
        if pc in ("RNA_msa_size",):
            continue
        bprops[plab] = b.loc[ids, pc].apply(_num).values
    for col, lab, better in METRICS:
        x, y = rows[col]
        improve = (y - x) if better == "higher" else (x - y)
        print(f"  {lab}:")
        for plab in ["MSA depth", "MFE", "Shannon entropy", "GC content", "INF"]:
            pv = bprops[plab].values
            mm = np.isfinite(pv) & np.isfinite(improve)
            if mm.sum() >= 8 and pd.Series(pv[mm]).nunique() > 1:
                r, pval = spearmanr(pv[mm], improve[mm])
                star = "*" if pval < 0.05 else ""
                print(f"      {plab:<16} ρ={r:+.3f} (p={pval:.3f}, n={int(mm.sum())}){star}")

    mg_present = b.loc[ids, "Ions"].apply(_has_mg).values

    # ---- Does the amount / complexation of ions predict the improvement? ----
    # Δ RNA RMSD with +Δ = better (lower RMSD with ions). Report the raw
    # Spearman and the partial correlation controlling for ligand presence.
    def _partial_spearman(x, y, z):
        from scipy.stats import rankdata
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

        def _resid(v, w):
            w1 = np.c_[np.ones_like(w), w]
            coef, *_ = np.linalg.lstsq(w1, v, rcond=None)
            return v - w1 @ coef
        return float(np.corrcoef(_resid(rx, rz), _resid(ry, rz))[0, 1])

    xr, yr = rows["RNA_RMSD"]
    d_rmsd = xr - yr                      # +Δ = RMSD lower with ions = better
    print("\nΔ RNA RMSD (+Δ = better with ions) vs ion features:")
    for vals, lab in [(ion_total, "ion count"), (n_multi, "multidentate chel.")]:
        mm = np.isfinite(vals) & np.isfinite(d_rmsd)
        if mm.sum() >= 8 and pd.Series(vals[mm]).nunique() > 1:
            r, p = spearmanr(vals[mm], d_rmsd[mm])
            pr = _partial_spearman(vals[mm], d_rmsd[mm], has_ligand[mm].astype(float))
            print(f"  {lab:<20} ρ={r:+.3f} (p={p:.3f}, n={int(mm.sum())}) "
                  f"| partial ρ|ligand={pr:+.3f}")

    def _stars(p):
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "n.s.")

    def _finite(col):
        x, y = rows[col]
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    # Shared helpers for the paired bar panel (a).
    from matplotlib.patches import Patch

    def _sem(a):
        return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0

    # ---- Panel B: which RNAs benefit — Δ RNA LDDT vs MFE ----
    # Colour = MSA depth (viridis); no-MSA (RNA_msa_size==1 → MSA 0) forced red.
    # Marker = ion type: circle if Mg present (with or without other ions),
    #          diamond otherwise.
    def drawB(ax):
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.lines import Line2D

        xl, yl = rows["RNA_LDDT_allAtom"]
        d = yl - xl
        mfe = bprops["MFE"].values
        msa = bprops["MSA depth"].values
        mm = np.isfinite(mfe) & np.isfinite(d) & np.isfinite(msa)
        no_msa = mm & (msa <= 1)        # RNA_msa_size == 1 means no MSA available
        has_msa = mm & (msa > 1)

        # Continuous gradient in PH_RANGE_COLORS hues for graded MSA depth
        # (shallow -> deep = orange -> green -> blue -> purple); red is reserved
        # for the no-MSA points so they stand apart from the gradient.
        ph_cmap = LinearSegmentedColormap.from_list(
            "ph_msa",
            [PH_RANGE_COLORS["6-7"], PH_RANGE_COLORS["7-8"],
             PH_RANGE_COLORS["5-6"], PH_RANGE_COLORS[">8"]],
        )
        ph_cmap.set_under(COL_NOMSA)   # MSA depth 0 (no alignment) -> red band
        vmin, vmax = (msa[has_msa].min(), msa[has_msa].max()) if has_msa.any() else (2, 100)
        norm = Normalize(vmin=vmin, vmax=vmax)

        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        # has-MSA: continuous PH-style gradient; circle=Mg, diamond=other
        for marker, has_mg in (("o", True), ("D", False)):
            sel = has_msa & (mg_present == has_mg)
            if sel.any():
                ax.scatter(mfe[sel], d[sel], c=msa[sel], cmap=ph_cmap, norm=norm,
                           marker=marker, s=40, edgecolor="black", linewidth=0.3,
                           zorder=2)
        # no-MSA (MSA size == 1, i.e. 0 alignments): solid red; circle=Mg, diamond=other
        for marker, has_mg in (("o", True), ("D", False)):
            sel = no_msa & (mg_present == has_mg)
            if sel.any():
                ax.scatter(mfe[sel], d[sel], color=COL_NOMSA, marker=marker, s=46,
                           edgecolor="black", linewidth=0.3, zorder=3)

        r, p = spearmanr(mfe[mm], d[mm])
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.04, 0.96, f"ρ = {r:+.2f}\n{ptxt}  (n={int(mm.sum())})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=onpage_to_gen(6.5, 11.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", linewidth=0.8, alpha=0.9))

        sm = ScalarMappable(norm=norm, cmap=ph_cmap)
        sm.set_array([])
        # extend='min' draws a red rectangular band flush at the bottom of the
        # gradient (the set_under colour); it marks MSA depth 0 = no alignment.
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                                extend="min", extendrect=True, extendfrac=0.06)
        cb.set_label("MSA depth", fontsize=onpage_to_gen(6.5, 11.2))
        cb.ax.tick_params(labelsize=onpage_to_gen(6.5, 11.2))
        # Label the red under-extension as "0", with a tick dash matching the
        # other colorbar ticks.
        cb.ax.plot([1.0, 1.25], [-0.03, -0.03], transform=cb.ax.transAxes,
                   color="black", linewidth=0.8, clip_on=False)
        cb.ax.text(1.45, -0.03, "0", transform=cb.ax.transAxes,
                   ha="left", va="center", fontsize=onpage_to_gen(6.5, 11.2))

        legend_handles = [
            Line2D([], [], marker="o", color="0.4", ls="", markersize=7,
                   markeredgecolor="black", label="Mg$^{2+}$ present"),
            Line2D([], [], marker="D", color="0.4", ls="", markersize=7,
                   markeredgecolor="black", label="no Mg$^{2+}$"),
        ]
        # legend below the plot, matching panel A's below-plot position
        ax.legend(handles=legend_handles, fontsize=onpage_to_gen(7, 11.2),
                  loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax.set_xlabel("MFE (kcal/mol)")
        ax.set_ylabel(r"Δ LDDT$_r$ (with − without ions)")
        style_axis(ax, grid_axis="y")

    # ---- Panel C: does ion amount / type / complexation drive the gain? ----
    # Best metric = RNA RMSD. y = Δ RNA RMSD (+ = better with ions);
    # x = total ion count; colour = ligand presence (the exposed confound);
    # marker = Mg (circle) vs no-Mg (diamond); size = multidentate chelation.
    def drawC(ax):
        from matplotlib.lines import Line2D

        xr2, yr2 = rows["RNA_RMSD"]
        d = xr2 - yr2                     # +Δ = lower RMSD with ions = better
        mm = np.isfinite(d) & np.isfinite(ion_total)
        # bubble size from multidentate chelation count (complexation strength)
        mx = n_multi[mm].max() if np.isfinite(n_multi[mm]).any() and n_multi[mm].max() > 0 else 1.0
        size = 30 + 170 * (np.nan_to_num(n_multi) / mx)

        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        # COL_YES (blue) = ligand-free, COL_NO (orange) = ligand-bound
        for lig, col in ((False, COL_YES), (True, COL_NO)):
            for marker, has_mg in (("o", True), ("D", False)):
                sel = mm & (has_ligand == lig) & (mg_present == has_mg)
                if sel.any():
                    ax.scatter(ion_total[sel], d[sel], s=size[sel], color=col,
                               marker=marker, edgecolor="black", linewidth=0.3,
                               alpha=0.85, zorder=2)

        r, p = spearmanr(ion_total[mm], d[mm])
        pr = _partial_spearman(ion_total[mm], d[mm], has_ligand[mm].astype(float))
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.96, 0.96,
                f"ρ = {r:+.2f}  ({ptxt})\npartial ρ|ligand = {pr:+.2f}\n(n={int(mm.sum())})",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=onpage_to_gen(6.5, 11.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", linewidth=0.8, alpha=0.9))

        # legend: colour = ligand, marker = Mg, size = complexation
        # 3 columns, column-major fill (2 rows): col 1 = the two coloured ligand
        # entries, col 2 = Mg present / no Mg, col 3 = "more multidentate" alone.
        handles = [
            Line2D([], [], marker="s", color=COL_YES, ls="", markersize=8,
                   markeredgecolor="black", label="ligand-free"),
            Line2D([], [], marker="s", color=COL_NO, ls="", markersize=8,
                   markeredgecolor="black", label="ligand-bound"),
            Line2D([], [], marker="o", color="0.4", ls="", markersize=7,
                   markeredgecolor="black", label="Mg$^{2+}$ present"),
            Line2D([], [], marker="D", color="0.4", ls="", markersize=7,
                   markeredgecolor="black", label="no Mg$^{2+}$"),
            Line2D([], [], marker="o", color="0.4", ls="", markersize=11,
                   markeredgecolor="black", label="more multidentate"),
        ]
        # 2-column legend placed below the axis so no entry sits on a data point
        # (the dense low-ion-count cluster occupies the lower-left interior).
        ax.legend(handles=handles, fontsize=onpage_to_gen(7, 11.2),
                  loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
                  columnspacing=1.4, handletextpad=0.5, borderpad=0.6)
        ax.set_xlabel("Total ion count")
        ax.set_ylabel(r"Δ RMSD$_r$ (Å, without − with ions)")
        ax.set_ylim(bottom=-5.5)   # zoom in; one outlier at ~-11 is clipped
        style_axis(ax, grid_axis="y")

    # ======================================================================
    # Bottom row: ions deposited in the ground truth but NOT given to AF3.
    # ======================================================================
    from matplotlib.patches import Patch

    # ---- Panel E: protein-RNA — per-ion-species DockQ reversal with RBP ----
    # Mirrors ion_rbp_reversal.py Panel A (per-species Spearman ρ vs ion count,
    # +RBP vs -RBP, ordered by reversal magnitude) but uses DockQ instead of
    # iLDDT, since the interface DockQ matters more than the RNA metrics.
    from collections import Counter
    prgt = _load_protein_groundtruth()
    n_rbp_p = int(prgt["has_rbp"].sum())
    n_rbp_m = int((~prgt["has_rbp"]).sum())
    _spcnt = Counter()
    for c in prgt["Ions"]:
        for sp in _ion_species(c):
            if sp:
                _spcnt[sp] += 1
    E_METRIC, E_YLAB = "iLDDT", r"Spearman $\rho$ (iLDDT vs ions)"
    e_species = [s for s, n in _spcnt.most_common() if n >= 4]
    e_ncontain = dict(_spcnt)   # complexes containing each species

    def _species_iface_rho(species, grp):
        cnts = prgt["Ions"].apply(lambda c: _species_ion_count(c, species))
        sel = (prgt["has_rbp"] == grp).values
        x = cnts.values.astype(float)[sel]
        y = prgt[E_METRIC].values.astype(float)[sel]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() >= 8 and pd.Series(x[mm]).nunique() > 1 and (x[mm] > 0).sum() >= 2:
            return spearmanr(x[mm], y[mm])
        return (np.nan, np.nan)

    e_rows = [(sp, _species_iface_rho(sp, True), _species_iface_rho(sp, False))
              for sp in e_species]
    # order by interface reversal magnitude (most reversed first)
    e_rows.sort(key=lambda r: -abs((r[1][0] or 0) - (r[2][0] or 0)))
    print(f"\nPanel E — protein-RNA per-species {E_METRIC} reversal "
          f"(+RBP n={n_rbp_p}, -RBP n={n_rbp_m}):")
    for sp, (rp, pp), (rm, pm) in e_rows:
        print(f"  {sp:<4} (n={e_ncontain.get(sp, 0)}) +RBP ρ={rp:+.3f} (p={pp:.3f}) "
              f"| -RBP ρ={rm:+.3f} (p={pm:.3f})")

    def drawE(ax):
        ions = [r[0] for r in e_rows]
        plus = [r[1][0] for r in e_rows]
        minus = [r[2][0] for r in e_rows]
        x = np.arange(len(ions))
        w = 0.38
        ax.bar(x - w / 2, plus, w, color=RBP_TEMPLATE_COLORS["+RBP"], alpha=0.9,
               label=f"+RBP (n={n_rbp_p})")
        ax.bar(x + w / 2, minus, w, color=RBP_TEMPLATE_COLORS["-RBP"], alpha=0.9,
               label=f"-RBP (n={n_rbp_m})")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_fmt_species(sp) for sp in ions])
        # sample size (complexes containing the species) under each label
        for xi, sp in zip(x, ions):
            ax.text(xi, -0.12, f"n={e_ncontain.get(sp, 0)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=onpage_to_gen(6.5, 11.2), color="0.35")
        ax.set_ylabel(E_YLAB)
        # legend below the plot, vertically centred on panel C's 2-line legend
        ax.legend(fontsize=onpage_to_gen(7, 11.2), loc="center",
                  bbox_to_anchor=(0.5, -0.29), ncol=2)
        style_axis(ax, grid_axis="y")

    # ---- Panel F: protein-RNA — does GIVING ions help the interface? ----
    # Scores (0-1: RNA LDDT, DockQ_Nr) on the left axis; RMSD (Å) on right. The no-ions
    # baseline is the protein-RNA GT, matched to the with-ions PDB ids.
    pno, pyes = _load_protein_paired()
    p_ids = [i for i in pyes.index if i in pno.index]
    f_left = [("RNA_LDDT_allAtom", r"LDDT$_r$"), ("DockQ_Nr", r"DockQ$_{Nr}$")]
    f_right = [("RNA_RMSD", r"RMSD$_r$"), ("Complex_RMSD", r"RMSD$_c$")]
    f_mets = f_left + f_right
    left_cols = {c for c, _ in f_left}
    print(f"\nJoined panel (a) — protein-RNA ion-input pair, n={len(p_ids)}:")
    f_data = []
    for col, lab in f_mets:
        x = np.array([_num(pno.loc[i, col]) for i in p_ids])
        y = np.array([_num(pyes.loc[i, col]) for i in p_ids])
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        try:
            _w, p = wilcoxon(x, y)
        except ValueError:
            p = float("nan")
        f_data.append((col, lab, x, y, p))
        print(f"  {lab:<13} no={x.mean():.3f} +ions={y.mean():.3f} "
              f"Δ={(y - x).mean():+.3f} (p={p:.3f})")

    # RNA-only side of the joined panel (paired without vs with ions).
    ao_layout = [("RNA_LDDT_allAtom", r"LDDT$_r$", "left"),
                 ("RNA_TM", r"TM$_r$", "left"),
                 ("RNA_RMSD", r"RMSD$_r$", "right")]
    ao_data = []
    for col, lab, side in ao_layout:
        xa, ya = _finite(col)
        try:
            _u, pp = wilcoxon(xa, ya)
        except ValueError:
            pp = float("nan")
        ao_data.append((lab, xa, ya, pp, side))
    # Protein-RNA side from f_data (side = right axis for the RMSD metrics).
    pr_data = [(lab, x, y, p, ("left" if col in left_cols else "right"))
               for col, lab, x, y, p in f_data]

    def drawAF(ax):
        """Joined paired-bar panel: protein-RNA (left) | RNA-only (right), split by a
        dashed line. Left axis = LDDT / TM / DockQ_Nr (0-1); right axis = RMSD (Å)."""
        ax2 = ax.twinx()
        w = 0.36
        ekw = dict(capsize=3, ecolor="0.25", elinewidth=0.9)

        stars_todo = []
        def _entry(xi, xa, ya, p, side):
            tgt = ax if side == "left" else ax2
            tgt.bar(xi - w / 2, np.mean(xa), w, color=COL_NO, alpha=0.9,
                    yerr=_sem(xa), error_kw=ekw)
            tgt.bar(xi + w / 2, np.mean(ya), w, color=COL_YES, alpha=0.9,
                    yerr=_sem(ya), error_kw=ekw)
            if np.isfinite(p) and p < 0.05:
                top = max(np.mean(xa) + _sem(xa), np.mean(ya) + _sem(ya))
                stars_todo.append((xi, top, p, tgt))  # placed after axes are finalized

        ticks, labels = [], []
        xi = 0
        pr_start = xi
        for lab, x, y, p, side in pr_data:
            _entry(xi, x, y, p, side); ticks.append(xi); labels.append(lab); xi += 1
        pr_end = xi - 1
        sep = xi                      # dashed separator sits in the gap
        xi += 1
        ao_start = xi
        for lab, xa, ya, p, side in ao_data:
            _entry(xi, xa, ya, p, side); ticks.append(xi); labels.append(lab); xi += 1
        ao_end = xi - 1

        ax.axvline(sep, color="0.5", ls="--", lw=1.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor",
                           fontsize=onpage_to_gen(7, 11.2))
        ax.set_xlim(-0.7, xi - 0.3)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(r"LDDT$_r$ / TM$_r$ / DockQ$_{Nr}$")
        ax2.set_ylabel("RMSD (Å)")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        # the central rc hides the right spine globally; the twin axis IS the right
        # (RMSD) axis, so re-enable its spine + right ticks or it disappears.
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(axis="y", which="both", right=True, labelright=True)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
        # place significance stars now that the axes are final; cap each within the
        # plotting area (below the section titles) so the tall RMSD_c star is not
        # hidden by the title. Axes limits and bars are unchanged.
        for xi_s, top_s, p_s, tgt_s in stars_todo:
            lo, hi = tgt_s.get_ylim()
            frac = min((top_s * 1.03 - lo) / (hi - lo), 0.96)
            tgt_s.text(xi_s, frac, _stars(p_s), transform=tgt_s.get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=onpage_to_gen(6.5, 11.2))
        ax.text((pr_start + pr_end) / 2, 1.02, f"Protein–RNA (n={len(p_ids)})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=onpage_to_gen(7, 11.2), fontweight="bold")
        ax.text((ao_start + ao_end) / 2, 1.02, f"Single-chain RNA (n={len(ids)})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=onpage_to_gen(7, 11.2), fontweight="bold")
        ax.legend(handles=[Patch(facecolor=COL_NO, label="predicted without ions"),
                           Patch(facecolor=COL_YES, label="predicted with ions")],
                  fontsize=onpage_to_gen(7, 11.2), loc="lower center",
                  bbox_to_anchor=(0.5, -0.32), ncol=2)

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "ion_comparison")
    fig = PublicationFigure(ncols=2, panel_size=(5.6, 4.4), outdir=outdir)
    for draw in (drawAF, drawB, drawC, drawE):
        fig.add_panel(draw, title=None)
    return fig.save("compare_ion_input")


def build_feature_scatter(metric="RNA_TM", metric_label="RNA TM"):
    """Supplement to Panel D: each ground-truth ion feature as its own scatter
    vs an RNA structure metric (RNA TM by default; RNA TM gives the strongest,
    most consistent correlations — see the metric comparison). Colour encodes
    ligand presence, matching the main figure. RNA-only ground-truth set."""
    from matplotlib.lines import Line2D

    gt = _load_rna_groundtruth()
    feats = ["ion count", "Mg count", "phosphate chel.", "ribose chel.",
             "multidentate", "min ion dist."]
    invert = metric == "RNA_RMSD"   # lower RMSD = better -> flip y for readability

    def _make(feat):
        def draw(ax):
            x = gt[feat].values.astype(float)
            y = gt[metric].values.astype(float)
            lig = gt["has_ligand"].values
            mm = np.isfinite(x) & np.isfinite(y)
            for has_lig, c in ((False, COL_YES), (True, COL_NO)):
                sel = mm & (lig == has_lig)
                if sel.any():
                    ax.scatter(x[sel], y[sel], color=c, s=30, alpha=0.85,
                               edgecolor="black", linewidth=0.3, zorder=2)
            r, p = spearmanr(x[mm], y[mm])
            if pd.Series(x[mm]).nunique() > 1:
                b = np.polyfit(x[mm], y[mm], 1)
                xs = np.linspace(x[mm].min(), x[mm].max(), 50)
                ax.plot(xs, np.polyval(b, xs), color="0.35", lw=1.2, ls="--", zorder=1)
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(0.97, 0.04, f"ρ = {r:+.2f}\n{ptxt} (n={int(mm.sum())})",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=onpage_to_gen(6.5, 11.2),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
            ax.set_xlabel(f"ground-truth {feat}")
            ax.set_ylabel(metric_label)
            if invert:
                ax.invert_yaxis()
            style_axis(ax, grid_axis="y")
        return draw

    outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains",
                          "ion_comparison")
    pf = PublicationFigure(ncols=3, panel_size=(4.6, 4.0), outdir=outdir)
    for f in feats:
        pf.add_panel(_make(f), title=None)
    figobj = pf.build()
    # The ligand-presence colour key applies to every panel, so show it once as a single
    # centred row below the bottom row of panels (previously it sat inside panel A only).
    import matplotlib.pyplot as plt
    handles = [
        Line2D([], [], marker="o", color=COL_YES, ls="", markersize=7,
               markeredgecolor="black", label="ligand-free"),
        Line2D([], [], marker="o", color=COL_NO, ls="", markersize=7,
               markeredgecolor="black", label="ligand-bound"),
    ]
    figobj.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015),
                  ncol=2, frameon=False, fontsize=onpage_to_gen(7.5, 13.8),
                  markerscale=1.3, columnspacing=2.4)
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, f"ion_feature_scatter.{ext}")
             for ext in ("png", "pdf", "svg", "eps")]
    for out in paths:
        figobj.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(figobj)
    print(f"Publication figure saved: {os.path.join(outdir, 'ion_feature_scatter')}")
    return paths


if __name__ == "__main__":
    build()
    build_feature_scatter()
