import inspect
import logging
import pandas as pd
import ast
import json
import sys
import os
import matplotlib.pyplot as plt
# allow direct execution: python evaluation/<script>.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.startConfig import StartConfig
from database.databaseMethods import DatabaseMethods
from plots.plotCreator import PlotCreator
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _call_grouped_violin_plot(plot_creator, **kwargs):
    """Call PlotCreator.get_grouped_violin_plot; drop kwargs not in signature (older plotCreator)."""
    sig = inspect.signature(plot_creator.get_grouped_violin_plot)
    plot_creator.get_grouped_violin_plot(
        **{k: v for k, v in kwargs.items() if k in sig.parameters}
    )


def _ion_ligand_entry_code(entry):
    """Ions: {'ions', …}; Ligands: {'ligands', …}; legacy {'code', …}; or list formats."""
    if isinstance(entry, dict):
        for key in ("ions", "ligands", "code"):
            c = entry.get(key)
            if isinstance(c, str) and c.strip():
                return c.strip()
        return None
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    if len(entry) >= 3 and isinstance(entry[0], str) and entry[0].strip():
        return entry[0].strip()
    if isinstance(entry[1], str) and entry[1].strip():
        return entry[1].strip()
    return None


def _ion_ligand_entry_af3_supported(entry):
    if isinstance(entry, dict):
        return entry.get("af3_supported") is True
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return False
    if len(entry) >= 3 and isinstance(entry[2], bool):
        return entry[2] is True
    if len(entry) >= 3 and isinstance(entry[2], str):
        return entry[2].startswith("+")
    if isinstance(entry[0], int):
        return entry[0] == 1
    return False


def extract_items(df, col):
    """Extract all ion/ligand names from a DataFrame column."""
    items = []
    for val in df[col]:
        try:
            group = ast.literal_eval(val) if isinstance(val, str) else val
            for x in group:
                code = _ion_ligand_entry_code(x)
                if code:
                    items.append(code)
        except Exception:
            continue
    return items

def get_samples_with_item(df, col, item):
    """Return a boolean mask for rows where the item is present in the given column."""
    mask = []
    for val in df[col]:
        found = False
        try:
            group = ast.literal_eval(val) if isinstance(val, str) else val
            for x in group:
                if _ion_ligand_entry_code(x) == item:
                    found = True
                    break
        except Exception:
            pass
        mask.append(found)
    return pd.Series(mask, index=df.index)

def classify_ions_ligands(ions, ligands):
    import ast

    try:
        ions = ast.literal_eval(ions) if isinstance(ions, str) else ions
        ligands = ast.literal_eval(ligands) if isinstance(ligands, str) else ligands
    except Exception:
        return 'unknown'

    # Case: both empty
    if not ions and not ligands:
        return 'none'

    def is_disallowed(group):
        for x in group:
            if isinstance(x, dict):
                if x.get("af3_supported") is False:
                    return True
                continue
            if not isinstance(x, (list, tuple)) or not x:
                continue
            if len(x) >= 3 and isinstance(x[2], bool) and x[2] is False:
                return True
            if len(x) >= 3 and isinstance(x[2], str) and x[2].startswith("-"):
                return True
            if isinstance(x, list) and len(x) >= 1 and x[0] == 0:
                return True
        return False

    if is_disallowed(ions) or is_disallowed(ligands):
        return 'unsupported'

    def is_allowed(group):
        if not group:
            return True
        return all(_ion_ligand_entry_af3_supported(x) for x in group if x)

    if is_allowed(ions) and is_allowed(ligands) and (ions or ligands):
        return 'supported'

    return 'other'

def classify_presence(ions, ligands):
    import ast
    try:
        ions_list = ast.literal_eval(ions) if isinstance(ions, str) else ions
        ligands_list = ast.literal_eval(ligands) if isinstance(ligands, str) else ligands
    except Exception:
        return 'unknown'

    has_ions = bool(ions_list)
    has_ligands = bool(ligands_list)

    if not has_ions and not has_ligands:
        return 'none'
    elif has_ions and not has_ligands:
        return 'ions'
    elif not has_ions and has_ligands:
        return 'ligand'
    elif has_ions and has_ligands:
        return 'ions/ligands'
    
    return 'unknown'

def classify_ion_valency(ions):
    MONOVALENT_IONS = {'K', 'NA', 'CL', 'NH4', 'CS'}
    DIVALENT_IONS = {'MG', 'CA', 'ZN', 'FE', 'CU', 'MN', 'CO'}

    import ast
    try:
        ions_list = ast.literal_eval(ions) if isinstance(ions, str) else ions
    except Exception:
        return 'unknown'

    if not ions_list:
        return 'none'

    valency_set = set()
    for ion in ions_list:
        ion_name = _ion_ligand_entry_code(ion)
        if ion_name:
            ion_name = ion_name.upper()
            if ion_name in MONOVALENT_IONS:
                valency_set.add('monovalent')
            elif ion_name in DIVALENT_IONS:
                valency_set.add('divalent')
            else:
                valency_set.add('unknown')

    if not valency_set:
        return 'none'
    elif valency_set == {'monovalent'}:
        return 'monovalent'
    elif valency_set == {'divalent'}:
        return 'divalent'
    elif 'unknown' in valency_set and len(valency_set) == 1:
        return 'unknown'
    else:
        return 'mixed'


def is_empty_list(val):
    try:
        group = ast.literal_eval(val) if isinstance(val, str) else val
        return isinstance(group, list) and len(group) == 0
    except Exception:
        return False

def parse_rna_rmsd(val):
    if pd.isna(val):
        return np.nan
    try:
        if isinstance(val, str):
            if '[' in val:
                values = ast.literal_eval(val)
                if isinstance(values, list) and len(values) > 0:
                    return float(values[0])
            return float(val)
        elif isinstance(val, (list, tuple)):
            return float(val[0]) if len(val) > 0 else np.nan
        return float(val)
    except Exception:
        return np.nan

def calculate_mean_std(values):
    """Calculate mean, std, and median for a list of values."""
    if not values or len(values) == 0:
        return None, None, None
    # Filter out None and invalid values
    valid_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid_values:
        return None, None, None
    mean = np.mean(valid_values)
    std = np.std(valid_values)
    median = np.median(valid_values)
    return mean, std, median


def _safe_load_json_dict(cell):
    """Parse Ion_Location / Ligand_Location TEXT from DB."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        try:
            out = ast.literal_eval(s)
            return out if isinstance(out, dict) else None
        except (ValueError, SyntaxError):
            return None


def _total_n_from_ions_ligands_column(cell):
    """Sum of 'n' fields in new-style Ions/Ligands list; legacy entries count as 1."""
    try:
        group = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    if not group:
        return 0
    total = 0
    for x in group:
        if isinstance(x, dict) and "n" in x:
            try:
                total += int(x["n"])
            except (TypeError, ValueError):
                total += 1
        else:
            total += 1
    return int(total)


def _bin_count(n, bins=(0, 1, 2, 3)):
    """Labels: '0', '1', '2', '3+'."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "unknown"
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


MONOVALENT_IONS_COUNTS = {"K", "NA", "CL", "NH4", "CS"}
DIVALENT_IONS_COUNTS = {"MG", "CA", "ZN", "FE", "CU", "MN", "CO"}


def _ion_valence_counts_from_ions_col(cell):
    """Return (monovalent_n, divalent_n, other_n) summed from Ions list dicts."""
    try:
        group = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError, TypeError):
        return 0, 0, 0
    if not group:
        return 0, 0, 0
    mono = di = other = 0
    for x in group:
        code = _ion_ligand_entry_code(x)
        if not code:
            continue
        cu = code.strip().upper()
        try:
            n = int(x["n"]) if isinstance(x, dict) and "n" in x else 1
        except (TypeError, ValueError):
            n = 1
        if cu in MONOVALENT_IONS_COUNTS:
            mono += n
        elif cu in DIVALENT_IONS_COUNTS:
            di += n
        else:
            other += n
    return mono, di, other


def _ion_count_detail_bucket_and_order(ion_total_series, max_individual=25):
    """
    Categories follow the data: one bin per count 0..max_obs when max_obs <= max_individual;
    otherwise 0..max_individual-1 plus f"{max_individual}+".
    Returns (bucket callable, ordered label list for plotting).
    """
    vals = pd.to_numeric(ion_total_series, errors="coerce").dropna()
    if vals.empty:
        mx = 0
    else:
        mx = int(vals.astype(int).max())
    use_tail = mx > max_individual
    tail_label = f"{max_individual}+" if use_tail else None

    def bucket(n):
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "unknown"
        n = int(n)
        if use_tail:
            if n >= max_individual:
                return tail_label
            return str(n)
        return str(n)

    if use_tail:
        ordered = [str(i) for i in range(0, max_individual)] + [tail_label]
    else:
        ordered = [str(i) for i in range(0, mx + 1)]
    return bucket, ordered


def _pvalue_to_sig_stars(p):
    """Two-sided test stars: *** p<0.001, ** p<0.01, * p<0.05."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _spearman_rho_and_p(x, y, min_n=4):
    """Spearman ρ and two-sided p-value; nan if insufficient variance or samples."""
    sub = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).dropna()
    sub = sub[np.isfinite(sub["x"]) & np.isfinite(sub["y"])]
    if len(sub) < min_n:
        return np.nan, np.nan
    if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return np.nan, np.nan
    r, p = spearmanr(sub["x"], sub["y"])
    r = float(r) if pd.notna(r) and np.isfinite(r) else np.nan
    p = float(p) if pd.notna(p) and np.isfinite(p) else np.nan
    return r, p


def _spearman_rho_pandas(x, y, min_n=4):
    """Signed Spearman ρ; returns nan if insufficient variance or samples."""
    r, _ = _spearman_rho_and_p(x, y, min_n=min_n)
    return r


def _valence_majority_category(mono, di, other):
    """Which valence bucket dominates total ion instances (entity Ions column)."""
    tot = mono + di + other
    if tot == 0:
        return "no_ions"
    if mono > di + other:
        return "mono_majority"
    if di > mono + other:
        return "di_majority"
    return "mixed_valence"


def _chelating_phosphate_category(summary):
    """Stratify RNA-site chelation by involvement of phosphate atom types."""
    if summary["chelating_n"] == 0:
        return "no_chelating"
    if "phosphate" in summary["atom_type_union"]:
        return "phosphate_in_chelation"
    if summary["atom_type_union"]:
        return "chelating_no_phosphate_type"
    return "chelating_untyped_atoms"


def _safe_col_suffix(code):
    s = str(code).strip().upper()
    return "".join(ch if ch.isalnum() else "_" for ch in s) or "X"


def _ion_counts_dict(cell):
    """Map ion code -> total n from Ions column."""
    out = defaultdict(int)
    try:
        group = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError, TypeError):
        return {}
    if not group:
        return {}
    for x in group:
        c = _ion_ligand_entry_code(x)
        if not c:
            continue
        cu = c.strip().upper()
        try:
            n = int(x["n"]) if isinstance(x, dict) and "n" in x else 1
        except (TypeError, ValueError):
            n = 1
        out[cu] += n
    return dict(out)


def _lig_counts_dict(cell):
    """Map ligand code -> total n from Ligands column."""
    return _ion_counts_dict(cell)


def _ligand_location_bucket_list(buckets, canonical_names):
    """Lists for bound / non_binding when keys match or differ by case / hyphen."""
    if not isinstance(buckets, dict):
        return []
    for name in canonical_names:
        if name in buckets:
            items = buckets.get(name) or []
            return items if isinstance(items, list) else []
    alias_to_items = {}
    for k, v in buckets.items():
        if not isinstance(k, str):
            continue
        kn = k.lower().replace("-", "_")
        if isinstance(v, list):
            alias_to_items[kn] = v
    for name in canonical_names:
        kn = name.lower().replace("-", "_")
        if kn in alias_to_items:
            return alias_to_items[kn]
    return []


def _ligand_location_bound_nonbinding_n(cell):
    """
    Count ligand sites: ``bound`` (legacy ``chelating``) vs ``non_binding`` (legacy
    ``non_chelating``), plus ``weakly_bound`` toward bound-like counts. Column names stay
    lig_bound_n / lig_non_binding_n for downstream plots.
    """
    lig_loc = _safe_load_json_dict(cell)
    if not lig_loc or not isinstance(lig_loc, dict):
        return 0, 0
    bound_n = nonb_n = 0
    for _name, buckets in lig_loc.items():
        if not isinstance(buckets, dict):
            continue
        bound_n += len(_ligand_location_bucket_list(buckets, ("bound", "chelating")))
        bound_n += len(_ligand_location_bucket_list(buckets, ("weakly_bound",)))
        nonb_n += len(
            _ligand_location_bucket_list(
                buckets, ("non_binding", "non_chelating", "non-binding")
            )
        )
    return bound_n, nonb_n


def _ion_location_global_min_distance(cell):
    """Smallest min_distance (Å) across all ion entries in Ion_Location JSON."""
    ion_loc = _safe_load_json_dict(cell)
    if not ion_loc or not isinstance(ion_loc, dict):
        return np.nan
    best = None
    for _name, buckets in ion_loc.items():
        if not isinstance(buckets, dict):
            continue
        for bucket in ("chelating", "non_chelating"):
            for e in buckets.get(bucket) or []:
                if not isinstance(e, dict) or "min_distance" not in e:
                    continue
                try:
                    d = float(e["min_distance"])
                except (TypeError, ValueError):
                    continue
                if best is None or d < best:
                    best = d
    return best if best is not None else np.nan


def _bin_min_dist_angstrom(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "unknown"
    if d <= 3.0:
        return "≤3 Å"
    if d <= 5.0:
        return "3–5 Å"
    return ">5 Å"


MIN_DIST_ORD = {"≤3 Å": 0, "3–5 Å": 1, ">5 Å": 2}


def _ion_coord_class_from_summary(s):
    """none: no chelating sites; mono: chelating, all monodentate; multi: any multidentate."""
    if s["chelating_n"] == 0:
        return "none"
    if s["multidentate_any"]:
        return "multi"
    return "mono"


ION_COORD_CLASS_ORD = {"none": 0, "mono": 1, "multi": 2}


def _chel_coord_num_bin_from_max(s):
    """
    Bin max RNA-atom contact count among chelating Ion_Location entries (see `n_contacts`
    in add_ion_location_to_db). 0 = no chelating sites.
    """
    m = s.get("chelating_max_n_contacts", 0) or 0
    try:
        m = int(m)
    except (TypeError, ValueError):
        return "unknown"
    if m <= 0:
        return "0 (no chel site)"
    if m == 1:
        return "1 contact / site (mono)"
    if m == 2:
        return "2 contacts (max)"
    return "3+ contacts (max)"


def _ion_rna_region_cat_from_summary(s):
    """
    Dominant RNA chemical moiety at chelating sites (phosphate / ribose / base / mixed / untyped).
    If chelating_n == 0: non_chelating (no RNA-ion chelation in Ion_Location).
    untyped = chelating sites exist but atom types could not be classified (non_chelating is different).
    """
    if s["chelating_n"] == 0:
        return "non_chelating"
    u = s["atom_type_union"]
    if not u:
        return "untyped"
    ul = {str(x).lower() for x in u}
    if len(ul) > 1:
        return "mixed"
    x = next(iter(ul))
    if x == "phosphate":
        return "phosphate"
    if x == "ribose":
        return "ribose"
    if x == "base":
        return "base"
    return "mixed"


RNA_REGION_ORD = {"phosphate": 0, "ribose": 1, "base": 2, "mixed": 3, "untyped": 4}


def _chelating_has_atom_region(summary, region):
    """region: 'phosphate' | 'ribose' | 'base' (any chelating atom_types entry)."""
    if summary["chelating_n"] == 0:
        return 0
    for x in summary["atom_type_union"]:
        if str(x).lower() == region:
            return 1
    return 0


def _install_species_count_predictors(df, source_col, prefix, min_rows):
    """Add one numeric column per species code with count in source_col. Returns [(col, label), ...]."""
    present = Counter()
    for _, row in df.iterrows():
        d = _ion_counts_dict(row[source_col]) if source_col == "Ions" else _lig_counts_dict(row[source_col])
        for k, v in d.items():
            if v > 0:
                present[k] += 1
    keep = sorted([k for k, c in present.items() if c >= min_rows])
    out = []
    for code in keep:
        cname = f"{prefix}_{_safe_col_suffix(code)}"

        def _count_cell(x, code=code):
            d = _ion_counts_dict(x) if source_col == "Ions" else _lig_counts_dict(x)
            return float(d.get(code, 0))

        df[cname] = df[source_col].apply(_count_cell)
        out.append((cname, str(code)))
    return out


def _records_for_combined_ion_coord_table(df, metric_col):
    """Long table: ion type × coordination class × metric value (expands each PDB row)."""
    y_all = _metric_y_series(df, metric_col)
    rec = []
    for idx in df.index:
        row = df.loc[idx]
        cc = row.get("ion_coord_class", "none")
        y = y_all.loc[idx]
        if pd.isna(y) or not np.isfinite(float(y)):
            continue
        if cc == "none":
            codes = list(_ion_counts_dict(row.get("Ions")).keys())
        else:
            loc = _safe_load_json_dict(row.get("Ion_Location"))
            codes = [str(k).strip().upper() for k in loc.keys()] if loc else []
        for code in sorted(set(codes)):
            if code:
                rec.append({"ion": code, "coord": cc, "val": float(y)})
    return pd.DataFrame(rec)


def _plot_combined_ion_coord_mean_heatmap(
    df, metric_col, metric_label, plot_creator, classification, min_cell_n=3
):
    """Mean metric: rows = none/mono/multi, columns = ion type (Ion_Location or Ions for none)."""
    import matplotlib.pyplot as plt

    t = _records_for_combined_ion_coord_table(df, metric_col)
    if t.empty:
        return
    agg = t.groupby(["coord", "ion"])["val"].agg(["mean", "count"]).reset_index()
    agg.columns = ["coord", "ion", "mean_val", "n"]
    coords_order = ["none", "mono", "multi"]
    ion_freq = t["ion"].value_counts()
    ions_order = sorted(t["ion"].unique(), key=lambda x: (-ion_freq.get(x, 0), x))
    mat = np.full((len(coords_order), len(ions_order)), np.nan)
    for _, r in agg.iterrows():
        if r["coord"] not in coords_order or r["n"] < min_cell_n:
            continue
        i = coords_order.index(r["coord"])
        j = ions_order.index(r["ion"])
        mat[i, j] = r["mean_val"]
    if np.all(np.isnan(mat)):
        return
    plot_creator.set_plot_style()
    fig_w = max(10, min(0.55 * len(ions_order), 32))
    fig_h = max(5.5, 1.1 * len(coords_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = (vmin if np.isfinite(vmin) else 0) - 1e-6
        vmax = (vmax if np.isfinite(vmax) else 1) + 1e-6
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(ions_order)))
    ax.set_xticklabels(ions_order, rotation=45, ha="right")
    ax.set_yticks(range(len(coords_order)))
    ax.set_yticklabels(coords_order)
    ax.set_xlabel("ion type")
    ax.set_ylabel("coordination class")
    cls_safe = _safe_col_suffix(classification or "run")
    ax.set_title(f"Mean {metric_label} (n≥{min_cell_n} per cell) [{classification}]")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label=metric_label)
    plt.tight_layout()
    plot_creator.save_plot(f"combined_ioncoord_mean_{cls_safe}_{_safe_col_suffix(metric_col)}")
    plt.close()


def _combined_mean_metric_tuples(classification, df_columns):
    cls = (classification or "").strip().lower()
    if cls == "protein_rna":
        want = [
            ("iLDDT", "RNA iLDDT"),
            ("RNA_LDDT_allAtom", "RNA LDDT"),
            ("RNA_TM", "RNA TM"),
            ("RNA_DockQ", "RNA DockQ"),
            ("RNA_irmsd", "RNA iRMSD"),
        ]
    elif cls == "rna_rna":
        want = [("RNA_LDDT_allAtom", "RNA LDDT"), ("RNA_TM", "RNA TM")]
    else:
        want = []
    return [(a, b) for a, b in want if a in df_columns]


def _parse_spearman_factor(entry):
    """(col, label) or (col, label, opts dict). opts: ion_close_subset=True → rows with ion_min_dist_A < 3 Å only."""
    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        col, xlab, opts = entry
        return col, xlab, opts if isinstance(opts, dict) else {}
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return entry[0], entry[1], {}
    raise TypeError("factor def must be (col, label) or (col, label, opts)")


def _spearman_factors_present(df, factor_entries):
    out = []
    for entry in factor_entries:
        col, _, opts = _parse_spearman_factor(entry)
        if col not in df.columns:
            continue
        if opts.get("ion_close_subset") and "ion_min_dist_A" not in df.columns:
            continue
        out.append(entry)
    return out


def _plot_spearman_heatmap_matrix(
    df, metrics, plot_creator, factor_defs, title, save_name, stats_dict=None, stats_section=None
):
    """Signed Spearman ρ: rows = metrics, cols = predictors in factor_defs (see _parse_spearman_factor)."""
    import matplotlib.pyplot as plt

    factor_defs = _spearman_factors_present(df, factor_defs)
    if not factor_defs:
        return

    rows = []
    row_labels = []
    store = stats_dict.setdefault(stats_section, {}) if stats_dict is not None and stats_section else None
    x_labels_plot = []
    for metric, label in metrics:
        row = []
        for entry in factor_defs:
            col, xlab, opts = _parse_spearman_factor(entry)
            if opts.get("ion_close_subset"):
                dcol = pd.to_numeric(df["ion_min_dist_A"], errors="coerce")
                sub = df.loc[dcol.notna() & (dcol < 3.0)]
                x = pd.to_numeric(sub[col], errors="coerce")
                y = _metric_y_series(sub, metric)
            else:
                x = pd.to_numeric(df[col], errors="coerce")
                y = _metric_y_series(df, metric)
            rho, p = _spearman_rho_and_p(x, y, min_n=4)
            row.append(rho)
            if store is not None and np.isfinite(rho) and np.isfinite(p):
                xlab_note = xlab
                if opts.get("ion_close_subset"):
                    xlab_note = f"{xlab} (subset: min RNA–ion distance < 3 Å)"
                key = f"{label} | {xlab_note}"
                store[key] = {"rho": float(rho), "p": float(p)}
        rows.append(row)
        row_labels.append(label[:28] + ("…" if len(label) > 28 else ""))

    for entry in factor_defs:
        _, xlab, _opts = _parse_spearman_factor(entry)
        x_labels_plot.append(xlab)

    mat = np.array(rows, dtype=float)
    if np.all(np.isnan(mat)):
        if stats_dict is not None and stats_section and store is not None and not store:
            stats_dict.pop(stats_section, None)
        return

    plot_creator.set_plot_style()
    fig_w = max(10, min(0.55 * len(factor_defs), 40))
    fig_h = max(8, 0.35 * len(metrics))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mat_plot = np.ma.masked_invalid(mat)
    im = ax.imshow(
        mat_plot,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    ax.set_xticks(range(len(factor_defs)))
    ax.set_xticklabels(x_labels_plot, rotation=28, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("ρ")
    plt.tight_layout()
    plot_creator.save_plot(save_name)
    plt.close()


def _plot_factor_impact_heatmap(
    df,
    metrics,
    plot_creator,
    name="ion_impact_spearman_heatmap",
    stats_dict=None,
    stats_section="Spearman ρ: ion predictors vs metrics",
):
    """
    Core ion predictors vs metrics. chelated_bin = 1 if any chelating site (includes chel+nonchel mixed).
    chel_mixed_bin = 1 if both chelating and non-chelating RNA-site ions exist.
    """
    factor_defs = [
        ("ion_total_n", "total ion n"),
        ("ion_mono_n", "monovalent ion n"),
        ("ion_di_n", "divalent ion n"),
        ("chel_phosphate_bin", "phosphate chelation (0/1)"),
        ("chelated_bin", "chelated (0/1)"),
        ("chel_mixed_bin", "chel+nonchel mix (0/1)"),
    ]
    _plot_spearman_heatmap_matrix(
        df,
        metrics,
        plot_creator,
        factor_defs,
        "Spearman ρ: ion predictors vs metrics (red +, blue −)",
        name,
        stats_dict=stats_dict,
        stats_section=stats_section,
    )


# --- Baseline Δ, ion chemistry, RNA topology, adjusted regression (multivariate) ---

def _ensure_rna_rbp_metrics(metrics, df):
    """
    Extend metric list so RNA_DockQ / RNA_irmsd are included when columns exist (panel plots
    previously truncated metrics[:n] and could drop them).
    """
    seen = {m for m, _ in metrics}
    out = list(metrics)
    for col, lab in (("RNA_DockQ", "RNA DockQ"), ("RNA_irmsd", "RNA iRMSD")):
        if col in df.columns and col not in seen:
            out.append((col, lab))
            seen.add(col)
    return out


def _metrics_for_panels(metrics, df, max_n=16):
    """
    Cap metrics to max_n while keeping RNA_DockQ / RNA_irmsd when present—they are last in
    PLOT_METRIC_SPECS and were otherwise excluded by metrics[:16].
    """
    base = _ensure_rna_rbp_metrics(metrics, df)
    priority = []
    for col, default_lab in (("RNA_DockQ", "RNA DockQ"), ("RNA_irmsd", "RNA iRMSD")):
        if col not in df.columns:
            continue
        label = default_lab
        for m, l in base:
            if m == col:
                label = l
                break
        priority.append((col, label))
    prio_ids = {c for c, _ in priority}
    rest = [(m, l) for m, l in base if m not in prio_ids]
    return (priority + rest)[:max_n]


def _bootstrap_median_diff(group_arr, baseline_arr, n_boot=2000, seed=42):
    """Bootstrap distribution: median(g*) - median(b*). Returns point (simple med diff), lo, hi."""
    g = np.asarray(group_arr, dtype=float)
    b = np.asarray(baseline_arr, dtype=float)
    g = g[np.isfinite(g)]
    b = b[np.isfinite(b)]
    if len(g) < 4 or len(b) < 4:
        return np.nan, np.nan, np.nan, np.nan
    point = float(np.median(g) - np.median(b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        mg = np.median(rng.choice(g, size=len(g), replace=True))
        mb = np.median(rng.choice(b, size=len(b), replace=True))
        diffs.append(mg - mb)
    diffs = np.array(diffs)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), diffs


def _ion_chemical_class(ions_cell):
    """
    Coarse chemistry for effect-size contrasts (Mg vs monovalents vs other divalents).
    """
    try:
        ions = ast.literal_eval(ions_cell) if isinstance(ions_cell, str) else ions_cell
    except (ValueError, SyntaxError, TypeError):
        return "unknown"
    if not ions:
        return "no_ions"
    codes = []
    for x in ions:
        c = _ion_ligand_entry_code(x)
        if c:
            codes.append(c.strip().upper())
    if not codes:
        return "no_ions"
    s = set(codes)
    MG = {"MG"}
    ca_family = {"CA", "SR", "BA"}
    mono = {"NA", "K", "CL", "NH4", "CS", "LI", "BR", "I"}
    divalent_other = {"ZN", "FE", "CU", "MN", "CO", "NI", "CD", "HG"}
    if s & MG:
        return "contains_Mg"
    if s & ca_family:
        return "Ca_group_no_Mg"
    if s <= mono or s.issubset(mono):
        return "monovalent_only"
    if s & divalent_other:
        return "other_divalent_no_Mg"
    return "mixed_or_other"


def _exp_n_stems_cell(cell):
    """
    Stem count from experimental `exp_RNA_Stems`.

    - One stem per inner list: ``[[4,5,...], [14,15,...]]`` → 2.
    - Flat residue index list (single stem): ``[4, 5, 6, ...]`` → 1.
    - Empty list → 0.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return np.nan
    try:
        s = ast.literal_eval(cell) if isinstance(cell, str) else cell
        if not isinstance(s, list):
            return np.nan
        if len(s) == 0:
            return 0.0
        n_inner_lists = sum(1 for x in s if isinstance(x, (list, tuple)))
        if n_inner_lists > 0:
            return float(n_inner_lists)
        return 1.0
    except (ValueError, SyntaxError, TypeError):
        pass
    return np.nan


def _deep_msa_flag(val):
    """Match PlotCreator +MSA heuristic: non-trivial RNA MSA depth."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip()
    if s.startswith("[1]") or s.startswith("[1,") or "1,1" in s:
        return 1
    return 0


def _has_rbp_template_flag(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip().lower()
    if not s or s in ("[]", "null"):
        return 0
    return 1


def _rna_stem_count_bin(n):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "unknown"
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def _plot_baseline_delta_bars(df, metrics, plot_creator, stats_dict, section_name):
    """
    Δ median(metric | group) − median(metric | baseline); baseline = no ions & no ligands.
    """
    metrics_use = _metrics_for_panels(metrics, df, 14)
    baseline_df = df[df["presence_category"] == "none"]
    if baseline_df.empty:
        return
    cats = ["ions", "ligand", "ions/ligands"]
    plot_creator.set_plot_style()
    n_met = len(metrics_use)
    fig, axes = plt.subplots(n_met, 1, figsize=(10, max(4, 2.4 * n_met)), sharex=False)
    if n_met == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, metrics_use):
        bvals = _metric_y_series(baseline_df, metric).dropna().values
        rows = []
        for cat in cats:
            sub = df[df["presence_category"] == cat]
            gvals = _metric_y_series(sub, metric).dropna().values
            pt, lo, hi, _ = _bootstrap_median_diff(gvals, bvals, n_boot=1500)
            rows.append((cat, pt, lo, hi, len(gvals)))
        if all(np.isnan(r[1]) for r in rows):
            ax.set_visible(False)
            continue
        ys = np.arange(len(rows))
        pts = [r[1] for r in rows]
        err_lo = [r[1] - r[2] if np.isfinite(r[1]) else 0 for r in rows]
        err_hi = [r[3] - r[1] if np.isfinite(r[1]) else 0 for r in rows]
        ax.barh(ys, pts, xerr=[err_lo, err_hi], capsize=4, color="steelblue", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel(f"Δ median vs baseline (none): {label}")
        sig = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}
        stats_dict.setdefault(section_name, {})[label] = sig
    plt.suptitle("Baseline = no ions & no ligands (experimental inventory)", y=1.01, fontsize=14)
    plt.tight_layout()
    plot_creator.save_plot("baseline_delta_median_bars_presence")
    plt.close()


def _plot_forest_baseline_presence(df, metrics, plot_creator):
    """Single forest-style figure: all metrics × presence contrasts vs baseline."""
    metrics_use = _metrics_for_panels(metrics, df, 16)
    baseline_df = df[df["presence_category"] == "none"]
    if baseline_df.empty or len(baseline_df) < 5:
        return
    cats = ["ions", "ligand", "ions/ligands"]
    records = []
    for metric, label in metrics_use:
        bvals = _metric_y_series(baseline_df, metric).dropna().values
        for cat in cats:
            gvals = _metric_y_series(df[df["presence_category"] == cat], metric).dropna().values
            pt, lo, hi, _ = _bootstrap_median_diff(gvals, bvals, n_boot=1200)
            if np.isfinite(pt):
                records.append(
                    {"contrast": f"{cat} vs none", "metric": label[:32], "delta": pt, "lo": lo, "hi": hi}
                )
    if not records:
        return
    dfp = pd.DataFrame(records)
    plot_creator.set_plot_style()
    dfp = dfp.sort_values(["metric", "contrast"])
    fig_h = max(7, 0.38 * len(dfp))
    fig, ax = plt.subplots(figsize=(10, min(fig_h, 48)))
    y_pos = np.arange(len(dfp))
    ax.errorbar(
        dfp["delta"],
        y_pos,
        xerr=[dfp["delta"] - dfp["lo"], dfp["hi"] - dfp["delta"]],
        fmt="o",
        capsize=3,
        color="darkgreen",
        ecolor="gray",
    )
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y_pos)
    lbls = [f"{row['metric']} | {row['contrast']}" for _, row in dfp.iterrows()]
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel("Δ median vs baseline (none)")
    ax.set_title("Forest: bootstrap 95% CI for median difference")
    plt.tight_layout()
    plot_creator.save_plot("baseline_delta_forest_presence")
    plt.close()


def _plot_violin_stratified(
    df, metrics, plot_creator, cat_col, cat_order, plot_key, stats_dict, stats_key, x_axis_label=None, rotation=0
):
    """Grouped violins for arbitrary categorical column."""
    for metric, label in metrics:
        data, labels_, metric_stats = [], [], {}
        for cat in cat_order:
            if cat not in df[cat_col].values:
                continue
            sub = df[df[cat_col] == cat]
            if sub.empty:
                continue
            vals = _extract_metric_values(sub, metric)
            if vals.empty:
                continue
            data.append(vals)
            labels_.append(cat)
            mean, std, median = calculate_mean_std(vals.tolist())
            metric_stats[cat] = (mean, std, median, len(vals))
        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source=plot_key,
                data_values=data,
                labels=labels_,
                name=f"{metric}_{plot_key}",
                yAxis_label=label,
                xAxis_label=x_axis_label,
                rotation=rotation,
            )
        if metric_stats:
            stats_dict.setdefault(stats_key, {})[label] = metric_stats


def _build_xy_for_adjusted_model(df):
    """Binary ion/ligand + standardized lengths + MSA + template. Returns X, colnames, valid mask."""
    has_ion = (pd.to_numeric(df["ion_total_n"], errors="coerce").fillna(0) > 0).astype(float)
    has_lig = (pd.to_numeric(df["ligand_total_n"], errors="coerce").fillna(0) > 0).astype(float)
    rna_len = pd.to_numeric(df["RNALength"], errors="coerce") if "RNALength" in df.columns else pd.Series(np.nan, index=df.index)
    prot_len = (
        pd.to_numeric(df["ProteinLength"], errors="coerce") if "ProteinLength" in df.columns else pd.Series(np.nan, index=df.index)
    )
    msa = df["af3_rna_MSA"].apply(_deep_msa_flag) if "af3_rna_MSA" in df.columns else pd.Series(0, index=df.index)
    tpl = (
        df["RBP_PDB_Templates"].apply(_has_rbp_template_flag) if "RBP_PDB_Templates" in df.columns else pd.Series(0, index=df.index)
    )
    rna_med = rna_len.median()
    prot_med = prot_len.median()
    if not np.isfinite(rna_med):
        rna_med = 0.0
    if not np.isfinite(prot_med):
        prot_med = 0.0
    rna_f = rna_len.fillna(rna_med)
    prot_f = prot_len.fillna(prot_med)
    RL = rna_f.values.reshape(-1, 1).astype(float)
    PL = prot_f.values.reshape(-1, 1).astype(float)
    if np.nanstd(RL) < 1e-9:
        rna_z = np.zeros(len(df))
    else:
        rna_z = StandardScaler().fit_transform(RL).ravel()
    if np.nanstd(PL) < 1e-9:
        prot_z = np.zeros(len(df))
    else:
        prot_z = StandardScaler().fit_transform(PL).ravel()
    X = np.column_stack(
        [has_ion.values, has_lig.values, rna_z, prot_z, msa.values.astype(float), tpl.values.astype(float)]
    )
    colnames = ["has_ion", "has_lig", "rna_len_z", "prot_len_z", "deep_msa", "has_template"]
    return X, colnames


def _plot_adjusted_coefficient_forest(df, metrics, plot_creator, stats_dict, n_boot=800, seed=42):
    """
    y ~ has_ion + has_lig + rna_len_z + prot_len_z + deep_msa + has_template (OLS).
    Bootstrap CI on coefficients; forest plot for has_ion and has_lig.
    """
    metrics_use = _metrics_for_panels(metrics, df, 16)
    need = {"ion_total_n", "ligand_total_n"}
    if not need.issubset(df.columns):
        return
    X_all, colnames = _build_xy_for_adjusted_model(df)
    target_coef_idx = [colnames.index("has_ion"), colnames.index("has_lig")]
    plot_creator.set_plot_style()
    records = []
    rng = np.random.default_rng(seed)
    n = len(df)
    for metric, label in metrics_use:
        y = _metric_y_series(df, metric).values.astype(float)
        mask = np.isfinite(y) & np.isfinite(X_all).all(axis=1)
        if mask.sum() < max(30, X_all.shape[1] + 5):
            continue
        Xm = X_all[mask]
        ym = y[mask]
        reg = LinearRegression(fit_intercept=True)
        reg.fit(Xm, ym)
        coef_point = reg.coef_
        # residualize bootstrap
        boot_rows = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ym), size=len(ym))
            reg_b = LinearRegression(fit_intercept=True)
            reg_b.fit(Xm[idx], ym[idx])
            boot_rows.append(reg_b.coef_)
        boot_arr = np.array(boot_rows)
        for j in target_coef_idx:
            cname = colnames[j]
            pt = coef_point[j]
            lo, hi = np.percentile(boot_arr[:, j], [2.5, 97.5])
            records.append({"metric": label[:28], "term": cname, "coef": pt, "lo": lo, "hi": hi})
        stats_dict.setdefault("Adjusted OLS (has_ion/has_lig)", {})[label] = {
            colnames[j]: (float(coef_point[j]), float(np.percentile(boot_arr[:, j], 2.5)), float(np.percentile(boot_arr[:, j], 97.5)))
            for j in target_coef_idx
        }
    if not records:
        return
    dfp = pd.DataFrame(records)
    fig_h = max(6, 0.35 * len(dfp))
    fig, ax = plt.subplots(figsize=(9, max(fig_h, 30)))
    y_pos = np.arange(len(dfp))
    # Matplotlib errorbar accepts one color per call, not an array (e.g. mpl 3.9).
    for yi, (_, row) in enumerate(dfp.iterrows()):
        c = "#1f77b4" if row["term"] == "has_ion" else "#ff7f0e"
        ax.errorbar(
            row["coef"],
            yi,
            xerr=[[row["coef"] - row["lo"]], [row["hi"] - row["coef"]]],
            fmt="o",
            capsize=3,
            ecolor="gray",
            color=c,
        )
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['metric']} | {row['term']}" for _, row in dfp.iterrows()], fontsize=9)
    ax.set_xlabel("OLS coefficient (bootstrapped 95% CI; confounders: length z, MSA, template)")
    ax.set_title("Multivariate: ion/ligand presence adjusted for covariates")
    plt.tight_layout()
    plot_creator.save_plot("adjusted_ols_forest_ion_lig")
    plt.close()


def _plot_partial_residual_spearman_panel(
    df,
    metrics,
    plot_creator,
    x_col="chelated_bin",
    save_name="partial_residual_chelation_spearman",
    xlabel=None,
    max_metrics=16,
    bar_color="slateblue",
    plot_title=None,
    stats_dict=None,
    stats_section=None,
):
    """
    After linearly removing RNA_length + ProteinLength + ion_total_n, Spearman ρ(x_binary, y_resid).
    Default x_col: chelated_bin (1 = any chelating RNA-site ion in Ion_Location).
    """
    import matplotlib.pyplot as plt

    if "RNALength" not in df.columns or x_col not in df.columns:
        return
    metrics = _metrics_for_panels(metrics, df, max_n=max_metrics)
    rna = pd.to_numeric(df["RNALength"], errors="coerce")
    prot = pd.to_numeric(df["ProteinLength"], errors="coerce") if "ProteinLength" in df.columns else pd.Series(0.0, index=df.index)
    ion_n = pd.to_numeric(df["ion_total_n"], errors="coerce").fillna(0) if "ion_total_n" in df.columns else pd.Series(0.0, index=df.index)
    Xsize = np.column_stack(
        [
            rna.fillna(rna.median()).values,
            prot.fillna(prot.median()).values,
            ion_n.values,
        ]
    )
    x_pred = pd.to_numeric(df[x_col], errors="coerce").fillna(0)
    plot_creator.set_plot_style()
    rhos = []
    store = stats_dict.setdefault(stats_section, {}) if stats_dict is not None and stats_section else None
    for metric, label in metrics:
        y = _metric_y_series(df, metric).values.astype(float)
        mask = np.isfinite(y) & np.isfinite(Xsize).all(axis=1)
        if mask.sum() < 12:
            continue
        reg = LinearRegression().fit(Xsize[mask], y[mask])
        y_res = y.copy()
        y_res[mask] = y[mask] - reg.predict(Xsize[mask])
        rho, p = _spearman_rho_and_p(x_pred[mask], pd.Series(y_res[mask]), min_n=8)
        if np.isfinite(rho):
            rhos.append((label[:26], rho))
            if store is not None and np.isfinite(p):
                store[f"{label} | residual ~ {x_col}"] = {"rho": float(rho), "p": float(p)}
    if len(rhos) < 2:
        if stats_dict is not None and stats_section and store is not None and not store:
            stats_dict.pop(stats_section, None)
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(rhos))))
    labs, vals = zip(*rhos)
    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals, color=bar_color)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs, fontsize=10)
    if xlabel is None:
        xlabel = f"Spearman ρ: {x_col} vs residual metric (after length + ion_n)"
    ax.set_xlabel(xlabel)
    if plot_title:
        ax.set_title(plot_title, fontsize=11)
    plt.tight_layout()
    plot_creator.save_plot(save_name)
    plt.close()


def _summarize_ion_location(ion_loc):
    """Aggregate chelating / non-chelating instances and chelation geometry from Ion_Location JSON."""
    empty = {
        "chelating_n": 0,
        "nonchelating_n": 0,
        "atom_type_union": frozenset(),
        "multidentate_any": False,
        "chelating_max_n_contacts": 0,
    }
    if not ion_loc or not isinstance(ion_loc, dict):
        return empty
    chel_n = non_n = 0
    atom_union = set()
    mult_any = False
    chel_coord_max = 0
    for _ion_name, buckets in ion_loc.items():
        if not isinstance(buckets, dict):
            continue
        ch = buckets.get("chelating") or []
        nch = buckets.get("non_chelating") or []
        chel_n += len(ch)
        non_n += len(nch)
        for entry in ch:
            if not isinstance(entry, dict):
                continue
            for t in entry.get("atom_types") or []:
                if isinstance(t, str):
                    atom_union.add(t.strip())
            if entry.get("multidentate") is True:
                mult_any = True
            try:
                nc = int(entry.get("n_contacts", 0) or 0)
            except (TypeError, ValueError):
                nc = 0
            if nc > chel_coord_max:
                chel_coord_max = nc
    return {
        "chelating_n": chel_n,
        "nonchelating_n": non_n,
        "atom_type_union": frozenset(atom_union),
        "multidentate_any": mult_any,
        "chelating_max_n_contacts": chel_coord_max,
    }


def _ion_chelation_category(summary):
    if summary["chelating_n"] == 0 and summary["nonchelating_n"] == 0:
        return "no_ion_geometry"
    if summary["chelating_n"] > 0 and summary["nonchelating_n"] > 0:
        return "chelating+non_chelating"
    if summary["chelating_n"] > 0:
        return "chelating_only"
    return "non_chelating_only"


def _ion_atom_types_label(summary):
    if summary["chelating_n"] == 0:
        return "n/a"
    if not summary["atom_type_union"]:
        return "chelating (no typed atoms)"
    return "+".join(sorted(summary["atom_type_union"]))


def _ion_multidentate_label(summary):
    if summary["chelating_n"] == 0:
        return "n/a"
    return "multidentate" if summary["multidentate_any"] else "monodentate_only"


def _ion_chelation_hue_from_summary(s):
    """Categorical chelation state for 2D strip hue alongside ion counts."""
    c, n = s["chelating_n"], s["nonchelating_n"]
    if c == 0 and n == 0:
        return "no_RNA_contact_site"
    if c > 0 and n > 0:
        return "chelating+nonchelating"
    if c > 0:
        return "chelating"
    return "nonchelating"


def _dominant_species_from_loc_keys(loc):
    if not loc or not isinstance(loc, dict):
        return "none"
    keys = sorted({str(k).strip().upper() for k in loc.keys() if str(k).strip()})
    if not keys:
        return "none"
    if len(keys) == 1:
        return keys[0]
    if len(keys) <= 3:
        return "+".join(keys)
    return "multi"


def _dominant_species_from_ions_ligands_col(cell, entry_parser=_ion_ligand_entry_code):
    try:
        group = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError, TypeError):
        return "none"
    if not group:
        return "none"
    codes = []
    for x in group:
        c = entry_parser(x)
        if c:
            codes.append(c.strip().upper())
    if not codes:
        return "none"
    u = sorted(set(codes))
    if len(u) == 1:
        return u[0]
    if len(u) <= 3:
        return "+".join(u)
    return "multi"


def _dominant_ion_species_row(row):
    loc = _safe_load_json_dict(row.get("Ion_Location"))
    d = _dominant_species_from_loc_keys(loc)
    if d != "none":
        return d
    return _dominant_species_from_ions_ligands_col(row.get("Ions"))


def _collapse_rare_categories(series, min_n):
    if series is None or series.empty:
        return series
    vc = series.value_counts()
    rare = set(vc[vc < min_n].index)
    out = series.copy()
    out[out.isin(rare)] = "other"
    return out


def _metric_y_series(df, metric):
    s = df[metric]
    if metric in ("Complex_RMSD", "RNA_RMSD"):
        s = s.apply(parse_rna_rmsd)
    return pd.to_numeric(s, errors="coerce")


def _plot_strip_metric(
    df,
    plot_creator,
    metric,
    y_label,
    x_col,
    hue_col,
    plot_tag,
    x_order=None,
    rotation=0,
):
    """2D strip plot: x = binned count or composite category, y = metric; optional hue (3rd factor)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_creator.set_plot_style()
    plot_df = df.copy()
    plot_df["_ym"] = _metric_y_series(plot_df, metric)
    need = ["_ym", x_col]
    if hue_col:
        need.append(hue_col)
    plot_df = plot_df.dropna(subset=need)
    if plot_df.empty:
        return
    if x_order is None:
        x_order = sorted(plot_df[x_col].dropna().astype(str).unique().tolist())
    else:
        x_order = [x for x in x_order if x in plot_df[x_col].values]
    if not x_order:
        return
    plot_df = plot_df[plot_df[x_col].isin(x_order)]
    fig_w = max(9, min(2.2 * len(x_order), 30))
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    if hue_col:
        sns.stripplot(
            data=plot_df,
            x=x_col,
            y="_ym",
            hue=hue_col,
            order=x_order,
            dodge=True,
            alpha=0.55,
            jitter=0.2,
            linewidth=0,
            ax=ax,
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
    else:
        sns.stripplot(
            data=plot_df,
            x=x_col,
            y="_ym",
            order=x_order,
            alpha=0.55,
            jitter=0.2,
            linewidth=0,
            ax=ax,
        )
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_label)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plot_creator.save_plot(f"{metric}_{plot_tag}")
    plt.close()


def _extract_metric_values(df, metric):
    s = df[metric]
    if metric in ("Complex_RMSD", "RNA_RMSD"):
        s = s.apply(parse_rna_rmsd)
    return pd.to_numeric(s, errors="coerce").dropna()


def _plot_metrics_by_category(
    df,
    metrics,
    plot_creator,
    stats_dict,
    stats_section,
    cat_col,
    categories_order,
    plot_suffix,
    rotation=0,
    x_axis_label=None,
):
    for metric, label in metrics:
        data, labels_, metric_stats = [], [], {}
        for cat in categories_order:
            sub = df[df[cat_col] == cat]
            if sub.empty:
                continue
            vals = _extract_metric_values(sub, metric)
            if vals.empty:
                continue
            data.append(vals)
            labels_.append(cat)
            mean, std, median = calculate_mean_std(vals.tolist())
            metric_stats[cat] = (mean, std, median, len(vals))
        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source=plot_suffix,
                data_values=data,
                labels=labels_,
                name=f"{metric}_{plot_suffix}",
                yAxis_label=label,
                xAxis_label=x_axis_label,
                rotation=rotation,
            )
        if metric_stats:
            stats_dict.setdefault(stats_section, {})[label] = metric_stats


# Metrics to plot: (logical_column_name_in_DataFrame, y-axis label, physical column candidates on pred_* table).
# First existing candidate on the table is selected; aliased to logical_name when names differ (e.g. RNA_LDDT -> RNA_LDDT_allAtom).
PLOT_METRIC_SPECS = [
    ("af3_global_pae_avg", "Global PAE", ("af3_global_pae_avg",)),
    ("af3_chain_pair_pae_min", "Chain Pair PAE", ("af3_chain_pair_pae_min",)),
    ("Complex_RMSD", "Complex RMSD", ("Complex_RMSD",)),
    ("RNA_RMSD", "RNA RMSD", ("RNA_RMSD",)),
    ("RNA_TM", "RNA TM", ("RNA_TM", "RNAComplex_TM")),
    ("RNA_LDDT_allAtom", "RNA LDDT", ("RNA_LDDT_allAtom", "RNA_LDDT")),
    ("RNAmotif_score", "RNA Motif Score", ("RNAmotif_score",)),
    ("Complex_TM", "Complex TM", ("Complex_TM",)),
    ("Complex_LDDT_allAtom", "Complex LDDT", ("Complex_LDDT_allAtom", "Complex_LDDT")),
    ("AAmatch_score", "AA Match Score", ("AAmatch_score",)),
    ("RNAmatch_score", "RNAmatch_score", ("RNAmatch_score",)),
    ("RNA_DI", "RNA DI", ("RNA_DI",)),
    ("ICS", "ICS", ("ICS",)),
    ("iLDDT", "iLDDT", ("iLDDT",)),
    # OST compare-structures DockQ; present on pred_protein_rna (and similar), not rna_rna
    ("Dockq", "DockQ", ("Dockq",)),
    ("QS_global", "QS_global", ("QS_global",)),
    # Filled by _explode_df_per_target_rna_metrics (not loaded from SQL)
    ("RNA_DockQ", "RNA DockQ", ()),
    ("RNA_irmsd", "RNA iRMSD", ()),
]


def _load_json_like_list(cell):
    """Parse JSON / literal list from DB cell."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, list):
        return cell
    if isinstance(cell, tuple):
        return list(cell)
    s = str(cell).strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else None
    except (json.JSONDecodeError, TypeError):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else None
        except (ValueError, SyntaxError):
            return None


def _parse_chain_id_list(cell):
    lst = _load_json_like_list(cell)
    if not lst:
        return []
    out = []
    for x in lst:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _parse_numeric_list_parallel(cell):
    lst = _load_json_like_list(cell)
    if not lst:
        return []
    out = []
    for x in lst:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def _parse_sequence_list(cell):
    lst = _load_json_like_list(cell)
    if not lst:
        return []
    return ["" if x is None else str(x).strip() for x in lst]


def _to_float_matrix2d(cell):
    lst = _load_json_like_list(cell)
    if not lst:
        return None
    try:
        return np.asarray(lst, dtype=float)
    except (ValueError, TypeError):
        return None


def _flatten_rbp_keyed_list(cell):
    """RBP_DockQ / RBP_irmsd as [{\"A:E\": v}, ...] -> merged dict."""
    merged = {}
    lst = _load_json_like_list(cell)
    if not lst:
        return merged
    for part in lst:
        if isinstance(part, dict):
            merged.update(part)
    return merged


def _mean_rbp_for_pred_chain(flat, chain_id):
    if not flat or chain_id is None:
        return np.nan
    cid = str(chain_id).strip()
    if not cid:
        return np.nan
    vals = []
    for k, v in flat.items():
        if not isinstance(k, str) or ":" not in k:
            continue
        parts = [p.strip() for p in k.split(":", 1)]
        if cid in parts:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return float(np.mean(vals)) if vals else np.nan


def _rna_chains_from_location_root(loc_root):
    """All exp rna_chain ids from one location JSON (ion or ligand), every species × bucket."""
    chains = []
    seen = set()

    def push(c):
        if c is None:
            return
        s = str(c).strip()
        if not s or s in seen:
            return
        seen.add(s)
        chains.append(s)

    root = _safe_load_json_dict(loc_root)
    if not root or not isinstance(root, dict):
        return chains
    for _species, buckets in root.items():
        if not isinstance(buckets, dict):
            continue
        for _bname, entries in buckets.items():
            if not isinstance(entries, list):
                continue
            for ent in entries:
                if isinstance(ent, dict) and "rna_chain" in ent:
                    push(ent.get("rna_chain"))
    return chains


def _rna_chains_from_ion_ligand_locations(ion_cell, lig_cell):
    """Union of rna_chain from Ion_Location and Ligand_Location (same rules: all buckets). Order: ions first, then ligands (deduped)."""
    out = []
    seen = set()
    for ch in _rna_chains_from_location_root(ion_cell):
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    for ch in _rna_chains_from_location_root(lig_cell):
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def _length_match_pair(exp_lens, pred_lens, e, p):
    if not exp_lens or not pred_lens or len(exp_lens) <= e or len(pred_lens) <= p:
        return True
    a, b = exp_lens[e], pred_lens[p]
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return True


def _seq_match_pair(exp_seqs, pred_seqs, e, p):
    if not exp_seqs or not pred_seqs or len(exp_seqs) <= e or len(pred_seqs) <= p:
        return False
    se, sp = exp_seqs[e], pred_seqs[p]
    if se is None or sp is None:
        return False
    return str(se).strip() == str(sp).strip() and str(se).strip() != ""


def _build_valid_exp_pred_pairs(n_e, n_p, exp_lens, pred_lens, exp_seqs, pred_seqs):
    valid = np.ones((n_e, n_p), dtype=bool)
    if exp_lens and pred_lens and len(exp_lens) >= n_e and len(pred_lens) >= n_p:
        for e in range(n_e):
            for p in range(n_p):
                valid[e, p] = _length_match_pair(exp_lens, pred_lens, e, p)
    for e in range(n_e):
        ps = [p for p in range(n_p) if valid[e, p]]
        if len(ps) <= 1:
            continue
        if not exp_seqs or len(exp_seqs) <= e:
            continue
        narrowed = [p for p in ps if _seq_match_pair(exp_seqs, pred_seqs, e, p)]
        if narrowed:
            valid[e, :] = False
            for p in narrowed:
                valid[e, p] = True
    return valid


def _assignment_max_tm_lddt(valid, tm, lddt, big=1e9):
    """Return dict exp_idx -> pred_idx maximizing TM+LDDT on valid pairs (Hungarian or greedy)."""
    n_e, n_p = valid.shape
    if tm.shape != (n_p, n_e):
        return {}
    for e in range(n_e):
        if not np.any(valid[e]):
            return {}
    score = tm + lddt
    cost = np.full((n_e, n_p), big, dtype=float)
    for e in range(n_e):
        for p in range(n_p):
            if valid[e, p] and np.isfinite(score[p, e]):
                cost[e, p] = -(score[p, e])
    ri, ci = None, None
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
    except Exception:
        used_p = set()
        greedy = {}
        order = sorted(
            range(n_e),
            key=lambda e: max(
                (score[p, e] if valid[e, p] and np.isfinite(score[p, e]) else -np.inf) for p in range(n_p)
            ),
        )
        for e in order:
            best_p = None
            best_sc = -np.inf
            for p in range(n_p):
                if p in used_p or not valid[e, p]:
                    continue
                sc = score[p, e]
                if np.isfinite(sc) and sc > best_sc:
                    best_sc, best_p = sc, p
            if best_p is not None:
                greedy[e] = best_p
                used_p.add(best_p)
        return greedy if len(greedy) == n_e else {}
    match = {}
    for r, c in zip(ri, ci):
        if cost[r, c] < big - 1:
            match[int(r)] = int(c)
    return match if len(match) == n_e else {}


def _scalar_tm_lddt_from_pred_row(row):
    tm = np.nan
    for c in ("RNA_TM", "RNAComplex_TM"):
        if c in row.index and pd.notna(row.get(c)):
            tm = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(tm):
                break
    lddt = np.nan
    for c in ("RNA_LDDT_allAtom", "RNA_LDDT"):
        if c in row.index and pd.notna(row.get(c)):
            lddt = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(lddt):
                break
    return tm, lddt


def _per_rna_metric_dicts_for_row(row):
    """
    One or more {RNA_TM, RNA_LDDT_allAtom, RNA_DockQ, RNA_irmsd} per sample row (plot labels stay RNA TM / RNA LDDT).

    Multi-chain: exp–pred matching via RNALength, then RNASequence, then max (RNA_TM_AllChains + RNA_LDDT_AllChains);
    matrices are pred (row i) × exp (col j) as stored in the DB.

    Target exp RNAs are those with rna_chain in experimental Ion_Location *or* Ligand_Location (every bucket:
    ion chelating/non_chelating; ligand bound/non_binding / legacy keys). If none are listed, all exp RNA chains are used.
    """
    pred_ids = _parse_chain_id_list(row.get("RNAChainIDs"))
    if not pred_ids:
        return []

    exp_ids = _parse_chain_id_list(row.get("exp_RNAChainIDs"))

    exp_lens = _parse_numeric_list_parallel(row.get("exp_RNALength"))
    pred_lens = _parse_numeric_list_parallel(row.get("RNALength"))
    exp_seqs = _parse_sequence_list(row.get("exp_RNASequence"))
    pred_seqs = _parse_sequence_list(row.get("RNASequence"))

    dock_flat = _flatten_rbp_keyed_list(row.get("RBP_DockQ"))
    irms_flat = _flatten_rbp_keyed_list(row.get("RBP_irmsd"))

    out = []
    if len(pred_ids) == 1:
        cid = pred_ids[0]
        tm, lddt = _scalar_tm_lddt_from_pred_row(row)
        out.append(
            {
                "RNA_TM": tm,
                "RNA_LDDT_allAtom": lddt,
                "RNA_DockQ": _mean_rbp_for_pred_chain(dock_flat, cid),
                "RNA_irmsd": _mean_rbp_for_pred_chain(irms_flat, cid),
            }
        )
        return out

    if not exp_ids or len(exp_ids) != len(pred_ids):
        return []

    tm_m = _to_float_matrix2d(row.get("RNA_TM_AllChains"))
    ld_m = _to_float_matrix2d(row.get("RNA_LDDT_AllChains"))
    if tm_m is None or ld_m is None or tm_m.shape != ld_m.shape:
        return []

    n_p, n_e = tm_m.shape[0], tm_m.shape[1]
    if len(pred_ids) != n_p or len(exp_ids) != n_e:
        return []

    if len(exp_lens) < n_e:
        exp_lens = []
    if len(pred_lens) < n_p:
        pred_lens = []
    if len(exp_seqs) < n_e:
        exp_seqs = []
    if len(pred_seqs) < n_p:
        pred_seqs = []

    valid = _build_valid_exp_pred_pairs(n_e, n_p, exp_lens, pred_lens, exp_seqs, pred_seqs)
    match_e_to_p = _assignment_max_tm_lddt(valid, tm_m, ld_m)
    if len(match_e_to_p) != n_e:
        return []

    targets = _rna_chains_from_ion_ligand_locations(row.get("Ion_Location"), row.get("Ligand_Location"))
    if not targets:
        targets = list(exp_ids)

    exp_to_i = {cid: idx for idx, cid in enumerate(exp_ids)}
    for exp_chain in targets:
        e = exp_to_i.get(str(exp_chain).strip())
        if e is None:
            continue
        p = match_e_to_p.get(e)
        if p is None:
            continue
        try:
            tmv = float(tm_m[p, e])
            ldv = float(ld_m[p, e])
        except (TypeError, ValueError, IndexError):
            continue
        pcid = pred_ids[p]
        out.append(
            {
                "RNA_TM": tmv,
                "RNA_LDDT_allAtom": ldv,
                "RNA_DockQ": _mean_rbp_for_pred_chain(dock_flat, pcid),
                "RNA_irmsd": _mean_rbp_for_pred_chain(irms_flat, pcid),
            }
        )
    return out


def _explode_df_per_target_rna_metrics(df):
    """Duplicate rows per RNA chain that appears in experimental Ion_Location or Ligand_Location (same breadth for both)."""
    if df.empty or "RNAChainIDs" not in df.columns:
        return df
    need_any = (
        "RNA_TM_AllChains" in df.columns
        or "RBP_DockQ" in df.columns
        or "RBP_irmsd" in df.columns
    )
    if not need_any:
        return df

    rows_out = []
    for _, row in df.iterrows():
        recs = _per_rna_metric_dicts_for_row(row)
        base = row.to_dict()
        if not recs:
            rows_out.append(base)
            continue
        for part in recs:
            d = dict(base)
            for k, v in part.items():
                d[k] = v
            rows_out.append(d)
    out = pd.DataFrame(rows_out)
    return out.reset_index(drop=True)


def _build_ions_ligands_sql(pred_table, ref_table, pred_columns, ref_columns):
    """Build SELECT for ions/ligands plots using only columns present on pred_table."""
    ion_loc_sql = "e.Ion_Location" if "Ion_Location" in ref_columns else "NULL AS Ion_Location"
    lig_loc_sql = "e.Ligand_Location" if "Ligand_Location" in ref_columns else "NULL AS Ligand_Location"
    pred_contact_sql = (
        "p.ContactList AS pred_contacts" if "ContactList" in pred_columns else "NULL AS pred_contacts"
    )
    exp_contact_sql = (
        "e.ContactList AS exp_contacts" if "ContactList" in ref_columns else "NULL AS exp_contacts"
    )

    parts = ["p.exp_db_id", pred_contact_sql]
    for meta in ("NumberRNAs", "RNALength", "RNAChainIDs", "RNAMotifLength", "ProteinLength", "RNASequence"):
        if meta in pred_columns:
            parts.append(f"p.{meta}")
    for extra_pred in ("RNA_TM_AllChains", "RNA_LDDT_AllChains", "RBP_DockQ", "RBP_irmsd"):
        if extra_pred in pred_columns:
            parts.append(f"p.{extra_pred}")

    for exp_col, alias in (
        ("RNAChainIDs", "exp_RNAChainIDs"),
        ("RNALength", "exp_RNALength"),
        ("RNASequence", "exp_RNASequence"),
    ):
        if exp_col in ref_columns:
            parts.append(f"e.{exp_col} AS {alias}")

    if "RNA_Stems" in ref_columns:
        parts.append("e.RNA_Stems AS exp_RNA_Stems")
    if "af3_rna_MSA" in pred_columns:
        parts.append("p.af3_rna_MSA")
    if "RBP_PDB_Templates" in pred_columns:
        parts.append("p.RBP_PDB_Templates")

    parts.extend(
        [
            exp_contact_sql,
            "e.Ions",
            "e.Ligands",
            ion_loc_sql,
            lig_loc_sql,
        ]
    )

    for logical, _label, candidates in PLOT_METRIC_SPECS:
        if logical == "Dockq" and str(pred_table).lower() != "pred_protein_rna":
            continue
        if not candidates:
            continue
        phys = next((c for c in candidates if c in pred_columns), None)
        if phys is None:
            continue
        if phys == logical:
            parts.append(f"p.{phys}")
        else:
            parts.append(f"p.{phys} AS {logical}")

    where_clauses = []
    if "ContactList" in pred_columns:
        where_clauses.append("p.ContactList IS NOT NULL")
    if "ContactList" in ref_columns:
        where_clauses.append("e.ContactList IS NOT NULL")
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    select_list = ",\n            ".join(parts)
    return f"""
        SELECT
            {select_list}
        FROM {pred_table} p
        LEFT JOIN {ref_table} e ON p.exp_db_id = e.PDBId
        WHERE {where_sql}
    """


def _is_flat_correlation_section(section_stats):
    """Each entry is {'rho': float, 'p': float} (Spearman / partial-residual tests)."""
    if not isinstance(section_stats, dict) or not section_stats:
        return False
    for v in section_stats.values():
        if not isinstance(v, dict) or "rho" not in v or "p" not in v or len(v) != 2:
            return False
    return True


def _is_adjusted_ols_section(section_stats):
    """Per metric: {term: (coef, boot_lo, boot_hi)} from adjusted ion/ligand OLS."""
    if not isinstance(section_stats, dict) or not section_stats:
        return False
    for term_map in section_stats.values():
        if not isinstance(term_map, dict) or not term_map:
            return False
        for tup in term_map.values():
            if not (isinstance(tup, tuple) and len(tup) == 3):
                return False
    return True


def _is_multivariate_driver_section(section_stats):
    """Structured multivariate results with term-level p-values and FDR."""
    return (
        isinstance(section_stats, dict)
        and isinstance(section_stats.get("models"), list)
        and isinstance(section_stats.get("fdr_method"), str)
    )


def _run_multivariate_driver_models(df, metrics, stats_dict, ion_predictor_cols=None, lig_predictor_cols=None):
    """
    Fit multivariate driver models with confounder adjustment and interaction tests.
    - Primary model: OLS + HC3 robust SE
    - Optional mixed-effects random intercept by exp_db_id when repeated rows exist
    - Global BH-FDR across all tested terms (all metrics, primary model terms)
    - Optional PCA compression for high-dimensional species-count predictors
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests
    except Exception:
        return

    ion_predictor_cols = ion_predictor_cols or []
    lig_predictor_cols = lig_predictor_cols or []

    work = pd.DataFrame(index=df.index)
    work["exp_db_id"] = df["exp_db_id"] if "exp_db_id" in df.columns else np.arange(len(df))

    if "RNALength" in df.columns:
        work["rna_len"] = pd.to_numeric(df["RNALength"], errors="coerce")
    else:
        work["rna_len"] = np.nan
    work["rna_len"] = work["rna_len"].fillna(work["rna_len"].median() if np.isfinite(work["rna_len"].median()) else 0.0)

    if "ProteinLength" in df.columns:
        work["prot_len"] = pd.to_numeric(df["ProteinLength"], errors="coerce")
    else:
        work["prot_len"] = 0.0
    work["prot_len"] = work["prot_len"].fillna(work["prot_len"].median() if np.isfinite(work["prot_len"].median()) else 0.0)

    def _zscore(arr):
        arr = pd.to_numeric(arr, errors="coerce")
        mu = arr.mean()
        sd = arr.std()
        if not np.isfinite(sd) or sd < 1e-12:
            return pd.Series(np.zeros(len(arr)), index=arr.index, dtype=float)
        return ((arr - mu) / sd).astype(float)

    work["rna_len_z"] = _zscore(work["rna_len"])
    work["prot_len_z"] = _zscore(work["prot_len"])
    if "af3_rna_MSA" in df.columns:
        work["deep_msa"] = df["af3_rna_MSA"].apply(_deep_msa_flag).astype(float)
    else:
        work["deep_msa"] = 0.0
    if "RBP_PDB_Templates" in df.columns:
        work["has_template"] = df["RBP_PDB_Templates"].apply(_has_rbp_template_flag).astype(float)
    else:
        work["has_template"] = 0.0

    raw_predictors = [
        "ion_total_n",
        "ligand_total_n",
        "chelated_bin",
        "non_chelated_only_bin",
        "chel_mixed_bin",
        "chel_multi_bin",
        "ion_min_dist_A",
        "ion_per_rna_chain",
        "lig_bound_n",
        "lig_non_binding_n",
    ]
    for c in raw_predictors:
        if c in df.columns:
            work[c] = pd.to_numeric(df[c], errors="coerce")

    base_predictors = [
        c for c in ["ion_total_n", "ligand_total_n", "chelated_bin", "chel_mixed_bin", "chel_multi_bin", "lig_bound_n"] if c in work.columns
    ]
    confounders = ["rna_len_z", "prot_len_z", "deep_msa", "has_template"]

    pca_meta = {"used": False, "n_components": 0, "explained_variance_ratio": []}
    species_cols = [c for c in list(ion_predictor_cols) + list(lig_predictor_cols) if c in df.columns]
    species_cols = [c for c in species_cols if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 12]
    if len(species_cols) >= 3 and len(df) >= 30:
        sp = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in species_cols}).fillna(0.0)
        keep = [c for c in sp.columns if float(sp[c].var()) > 1e-12]
        if len(keep) >= 3:
            sp = sp[keep]
            sp_z = StandardScaler().fit_transform(sp.values.astype(float))
            pca_full = PCA().fit(sp_z)
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            n_pc = int(np.searchsorted(csum, 0.80) + 1)
            n_pc = max(1, min(n_pc, 3))
            pca = PCA(n_components=n_pc).fit(sp_z)
            scores = pca.transform(sp_z)
            for i in range(n_pc):
                work[f"species_pc{i+1}"] = scores[:, i]
            pca_meta = {
                "used": True,
                "n_components": int(n_pc),
                "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
            }
            base_predictors.extend([f"species_pc{i+1}" for i in range(n_pc)])

    if "ion_total_n" in work.columns and "chelated_bin" in work.columns:
        work["ion_total_x_chelated"] = work["ion_total_n"] * work["chelated_bin"]
        base_predictors.append("ion_total_x_chelated")
    if "ligand_total_n" in work.columns and "chelated_bin" in work.columns:
        work["ligand_total_x_chelated"] = work["ligand_total_n"] * work["chelated_bin"]
        base_predictors.append("ligand_total_x_chelated")
    if "chel_multi_bin" in work.columns and "chelated_bin" in work.columns:
        work["chel_multi_x_chelated"] = work["chel_multi_bin"] * work["chelated_bin"]
        base_predictors.append("chel_multi_x_chelated")

    predictors = []
    for c in base_predictors + confounders:
        if c not in work.columns:
            continue
        vals = pd.to_numeric(work[c], errors="coerce")
        if vals.notna().sum() < 12:
            continue
        if float(vals.fillna(vals.median() if np.isfinite(vals.median()) else 0.0).var()) < 1e-12:
            continue
        work[c] = vals
        predictors.append(c)

    if len(predictors) < 3:
        return

    model_rows = []
    fdr_index = []
    fdr_pvals = []
    for metric, label in metrics:
        y = _metric_y_series(df, metric)
        local = work.copy()
        local["y"] = y
        local = local.dropna(subset=["y"] + predictors)
        if len(local) < max(40, len(predictors) + 15):
            continue

        X = sm.add_constant(local[predictors].astype(float), has_constant="add")
        ols_fit = sm.OLS(local["y"].astype(float), X).fit(cov_type="HC3")
        # Standardized effect sizes for cross-predictor comparison.
        Xz = local[predictors].astype(float).copy()
        for c in Xz.columns:
            sd = float(Xz[c].std())
            mu = float(Xz[c].mean())
            if np.isfinite(sd) and sd > 1e-12:
                Xz[c] = (Xz[c] - mu) / sd
            else:
                Xz[c] = 0.0
        yz = local["y"].astype(float)
        y_sd = float(yz.std())
        y_mu = float(yz.mean())
        if np.isfinite(y_sd) and y_sd > 1e-12:
            yz = (yz - y_mu) / y_sd
        else:
            yz = pd.Series(np.zeros(len(yz)), index=yz.index, dtype=float)
        ols_fit_std = sm.OLS(yz, sm.add_constant(Xz, has_constant="add")).fit(cov_type="HC3")
        terms = []
        for term in predictors:
            if term not in ols_fit.params.index:
                continue
            terms.append(
                {
                    "term": term,
                    "coef": float(ols_fit.params[term]),
                    "se": float(ols_fit.bse.get(term, np.nan)),
                    "p": float(ols_fit.pvalues.get(term, np.nan)),
                    "ci_low": float(ols_fit.conf_int().loc[term, 0]) if term in ols_fit.conf_int().index else np.nan,
                    "ci_high": float(ols_fit.conf_int().loc[term, 1]) if term in ols_fit.conf_int().index else np.nan,
                    "beta_std": float(ols_fit_std.params.get(term, np.nan)) if term in ols_fit_std.params.index else np.nan,
                }
            )

        mixed = None
        groups = local["exp_db_id"].astype(str)
        if groups.nunique() >= 8 and groups.value_counts().max() > 1:
            try:
                mixed_fit = sm.MixedLM(local["y"].astype(float), X, groups=groups).fit(reml=False, method="lbfgs", disp=False)
                mixed_terms = {}
                for term in predictors:
                    if term in mixed_fit.params.index:
                        mixed_terms[term] = {
                            "coef": float(mixed_fit.params[term]),
                            "se": float(mixed_fit.bse.get(term, np.nan)),
                            "p": float(mixed_fit.pvalues.get(term, np.nan)),
                        }
                mixed = {
                    "n_groups": int(groups.nunique()),
                    "aic": float(mixed_fit.aic) if np.isfinite(mixed_fit.aic) else np.nan,
                    "converged": bool(getattr(mixed_fit, "converged", False)),
                    "terms": mixed_terms,
                }
            except Exception:
                mixed = None

        for i, t in enumerate(terms):
            p = t.get("p")
            if p is None or not np.isfinite(p):
                continue
            fdr_index.append((len(model_rows), i))
            fdr_pvals.append(float(p))

        model_rows.append(
            {
                "metric": label,
                "n_obs": int(len(local)),
                "n_predictors": int(len(predictors)),
                "model_type": "mixedlm+ols" if mixed else "ols_hc3",
                "r2_adj": float(ols_fit.rsquared_adj) if np.isfinite(ols_fit.rsquared_adj) else np.nan,
                "aic": float(ols_fit.aic) if np.isfinite(ols_fit.aic) else np.nan,
                "terms": terms,
                "mixed": mixed,
            }
        )

    if not model_rows:
        return

    if fdr_pvals:
        _, p_adj, _, _ = multipletests(fdr_pvals, alpha=0.05, method="fdr_bh")
        for (row_i, term_i), p_corr in zip(fdr_index, p_adj):
            model_rows[row_i]["terms"][term_i]["p_fdr"] = float(p_corr)
            model_rows[row_i]["terms"][term_i]["sig_fdr"] = bool(p_corr < 0.05)
    for row in model_rows:
        for t in row["terms"]:
            if "p_fdr" not in t:
                t["p_fdr"] = np.nan
                t["sig_fdr"] = False

    stats_dict["Multivariate driver models (OLS/MixedLM, FDR)"] = {
        "fdr_method": "Benjamini-Hochberg",
        "predictors": predictors,
        "pca": pca_meta,
        "models": model_rows,
    }


def _merge_spearman_sections_for_summary(stats_dict):
    """Single section listing all ρ with p-values, sorted by significance (smallest p first)."""
    merged = {}
    for name, sec in stats_dict.items():
        if name == "Spearman correlations (all panels, sorted by p)":
            continue
        if not _is_flat_correlation_section(sec):
            continue
        if not (
            name.startswith("Spearman")
            or name.startswith("Partial residual Spearman")
        ):
            continue
        for k, v in sec.items():
            merged[f"[{name}] {k}"] = dict(v)
    if not merged:
        return
    ordered = sorted(
        merged.items(),
        key=lambda kv: kv[1]["p"] if np.isfinite(kv[1]["p"]) else 1.0,
    )
    stats_dict["Spearman correlations (all panels, sorted by p)"] = {a: b for a, b in ordered}


def write_statistics_to_file(stats_dict, plot_creator, msa_option=None, single_chain_only=False, rbp_template_option=None):
    """Write statistics to a text file in the same directory as plots."""
    output_path = os.path.join(plot_creator.results_dir, 'ions_ligands_statistics.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Ions and Ligands Analysis Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        # Add filter information
        filter_parts = []
        if single_chain_only:
            filter_parts.append("Single-Chain")
        else:
            filter_parts.append("All-Chains")
        
        if msa_option:
            filter_parts.append(msa_option)
        
        if rbp_template_option:
            # Format RBP template option nicely
            if rbp_template_option.upper() == '+RBPTEMP':
                filter_parts.append("+RBP-template")
            elif rbp_template_option.upper() == '-RBPTEMP':
                filter_parts.append("-RBP-template")
            else:
                filter_parts.append(rbp_template_option)
        
        if filter_parts:
            f.write("Selected Samples by: " + ", ".join(filter_parts) + "\n\n")
        
        f.write(
            "Correlation p-values are two-sided. Stars: *** p<0.001, ** p<0.01, * p<0.05.\n\n"
        )

        # Compact global summary of independent effects from multivariate models.
        mv_key = "Multivariate driver models (OLS/MixedLM, FDR)"
        mv_section = stats_dict.get(mv_key)
        if _is_multivariate_driver_section(mv_section):
            sig_rows = []
            for model in mv_section.get("models", []):
                metric_name = model.get("metric", "unknown")
                for t in model.get("terms", []):
                    p_fdr = t.get("p_fdr", np.nan)
                    beta_std = t.get("beta_std", np.nan)
                    if np.isfinite(p_fdr) and p_fdr < 0.05 and np.isfinite(beta_std):
                        sig_rows.append(
                            {
                                "metric": metric_name,
                                "term": t.get("term", "term"),
                                "beta_std": float(beta_std),
                                "coef": float(t.get("coef", np.nan)),
                                "p_fdr": float(p_fdr),
                            }
                        )

            f.write("=" * 80 + "\n")
            f.write("Top independent drivers (FDR<0.05)\n")
            f.write("=" * 80 + "\n\n")
            f.write(
                "Ranked by absolute standardized effect size |beta_std| from multivariate OLS.\n"
            )
            f.write(
                "Positive beta_std = higher metric as predictor increases; negative = lower metric.\n\n"
            )
            if sig_rows:
                sig_rows = sorted(sig_rows, key=lambda r: abs(r["beta_std"]), reverse=True)
                for i, r in enumerate(sig_rows[:30], start=1):
                    stars = _pvalue_to_sig_stars(r["p_fdr"])
                    f.write(
                        f"{i:2d}. {r['term']} | {r['metric']}: "
                        f"beta_std={r['beta_std']:+.4f}, coef={r['coef']:+.5f}, p_fdr={r['p_fdr']:.4g} {stars}\n"
                    )
                if len(sig_rows) > 30:
                    f.write(f"\n... and {len(sig_rows) - 30} additional FDR-significant term-metric associations.\n")
            else:
                f.write("No term reached FDR<0.05 in the multivariate models.\n")
            f.write("\n")
        
        summary_key = "Spearman correlations (all panels, sorted by p)"
        section_items = list(stats_dict.items())
        if summary_key in stats_dict:
            section_items = [(summary_key, stats_dict[summary_key])] + [
                (k, v) for k, v in section_items if k != summary_key
            ]

        # Write statistics for each metric and category
        for section_name, section_stats in section_items:
            f.write("=" * 80 + "\n")
            f.write(f"{section_name}\n")
            f.write("=" * 80 + "\n\n")

            if _is_flat_correlation_section(section_stats):
                rows = sorted(
                    section_stats.items(),
                    key=lambda kv: kv[1]["p"] if np.isfinite(kv[1]["p"]) else 1.0,
                )
                for desc, d in rows:
                    stars = _pvalue_to_sig_stars(d["p"])
                    rho, p = d["rho"], d["p"]
                    if not np.isfinite(rho) or not np.isfinite(p):
                        f.write(f"  {desc}: ρ = nan, p = nan\n")
                    else:
                        f.write(f"  {desc}: ρ = {rho:+.4f}, p = {p:.4g} {stars}\n")
                f.write("\n")
                continue

            if _is_adjusted_ols_section(section_stats):
                for metric_name, term_map in section_stats.items():
                    f.write(f"{metric_name}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(
                        "  Model: y ~ has_ion + has_lig + rna_len_z + prot_len_z + deep_msa + has_template; "
                        "bootstrap 95% CI on listed coefficients.\n"
                    )
                    for term, tup in sorted(term_map.items()):
                        coef, lo, hi = tup
                        f.write(f"  {term}: {coef:+.6f}  [{lo:.6f}, {hi:.6f}]\n")
                    f.write("\n")
                continue

            if _is_multivariate_driver_section(section_stats):
                f.write(
                    "Modeling strategy: OLS with HC3 robust SEs for independent effects; "
                    "random-intercept MixedLM attempted when repeated exp_db_id structure exists.\n"
                )
                f.write(
                    "Terms include ions/ligands/chelation features, interaction terms with chelation, "
                    "and confounders (RNA/protein length, MSA, template).\n"
                )
                f.write(f"Multiple testing control: {section_stats.get('fdr_method', 'BH-FDR')} across all term tests.\n")
                pca_meta = section_stats.get("pca", {})
                if pca_meta.get("used"):
                    evr = pca_meta.get("explained_variance_ratio", [])
                    evr_txt = ", ".join([f"PC{i+1}={v:.3f}" for i, v in enumerate(evr)])
                    f.write(
                        f"Redundancy reduction: PCA on species-count features "
                        f"({pca_meta.get('n_components', 0)} PCs; variance ratios: {evr_txt}).\n"
                    )
                else:
                    f.write("Redundancy reduction: PCA not used (insufficient/high-quality species-count dimensions).\n")
                f.write("\n")

                for model in section_stats.get("models", []):
                    f.write(f"{model.get('metric', 'unknown metric')}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(
                        f"  Primary model: {model.get('model_type', 'ols_hc3')}; "
                        f"n={model.get('n_obs', 0)}, predictors={model.get('n_predictors', 0)}, "
                        f"adj_R2={model.get('r2_adj', np.nan):.4f}, AIC={model.get('aic', np.nan):.2f}\n"
                    )
                    mixed = model.get("mixed")
                    if isinstance(mixed, dict):
                        f.write(
                            f"  MixedLM check: groups={mixed.get('n_groups', 0)}, "
                            f"converged={mixed.get('converged', False)}, AIC={mixed.get('aic', np.nan):.2f}\n"
                        )
                    f.write("  Term estimates (sorted by FDR):\n")
                    term_rows = sorted(
                        model.get("terms", []),
                        key=lambda t: t.get("p_fdr", 1.0) if np.isfinite(t.get("p_fdr", np.nan)) else 1.0,
                    )
                    for t in term_rows:
                        stars = _pvalue_to_sig_stars(t.get("p_fdr"))
                        f.write(
                            "    "
                            f"{t.get('term', 'term')}: coef={t.get('coef', np.nan):+.5f}, "
                            f"beta_std={t.get('beta_std', np.nan):+.5f}, "
                            f"SE={t.get('se', np.nan):.5f}, "
                            f"95%CI=[{t.get('ci_low', np.nan):.5f}, {t.get('ci_high', np.nan):.5f}], "
                            f"p={t.get('p', np.nan):.4g}, p_fdr={t.get('p_fdr', np.nan):.4g} {stars}\n"
                        )
                    f.write("\n")
                continue
            
            for metric_name, category_stats in section_stats.items():
                f.write(f"{metric_name}:\n")
                f.write("-" * 80 + "\n")
                for category, stats_tuple in category_stats.items():
                    if isinstance(stats_tuple, tuple) and len(stats_tuple) == 4:
                        mean, std, median, count = stats_tuple
                        if mean is not None:
                            f.write(f"  {category}:\n")
                            f.write(f"    Mean: {mean:.4f}\n")
                            f.write(f"    Std: {std:.4f}\n")
                            f.write(f"    Median: {median:.4f}\n")
                            f.write(f"    Count: {count}\n\n")
                f.write("\n")


def main(msa_option=None, single_chain_only=False, rbp_template_option=None,
         heatmaps_only=False, parse_argv=True, return_prepared=False):
    """Run the ion/ligand analysis.

    When ``heatmaps_only`` is True only the two Spearman heatmaps
    (``ion_impact_spearman_heatmap`` and ``per_ion_count_spearman``) are written
    and all other plots are skipped — used by the ``--filter`` sweep so the same
    two summaries can be produced for each +/-MSA and +/-RBP filter. When
    ``parse_argv`` is True (default, direct CLI call) the filter options are read
    from ``sys.argv``; the sweep passes them explicitly with ``parse_argv=False``.
    """
    # Parse command-line arguments (case-insensitive, any order)
    if parse_argv:
        for arg in sys.argv[1:]:
            arg_lower = arg.lower()
            if arg_lower in ['single_chain', 'all']:
                single_chain_only = (arg_lower == 'single_chain')
            elif arg_lower in ['+msa', '-msa', 'msa']:
                msa_option = '+MSA' if arg_lower in ['+msa', 'msa'] else '-MSA'
            elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
                rbp_template_option = arg  # Keep original case

    config = StartConfig()
    db = DatabaseMethods()
    pred_table = config.pred_table
    ref_table = config.ref_table
    
    cursor = db.connection.cursor()
    cursor.execute(f"PRAGMA table_info({ref_table})")
    ref_columns = {row[1] for row in cursor.fetchall()}
    cursor.execute(f"PRAGMA table_info({pred_table})")
    pred_columns = {row[1] for row in cursor.fetchall()}

    query = _build_ions_ligands_sql(pred_table, ref_table, pred_columns, ref_columns)
    
    # Add RBP template filter if specified
    if rbp_template_option:
        if "RBP_PDB_Templates" in pred_columns:
            if rbp_template_option.upper() == '+RBPTEMP':
                query += " AND (p.RBP_PDB_Templates IS NOT NULL AND p.RBP_PDB_Templates != '' AND p.RBP_PDB_Templates != '[]' AND p.RBP_PDB_Templates != 'null')"
            elif rbp_template_option.upper() == '-RBPTEMP':
                query += " AND (p.RBP_PDB_Templates IS NULL OR p.RBP_PDB_Templates = '' OR p.RBP_PDB_Templates = '[]' OR p.RBP_PDB_Templates = 'null')"
    
    query += ";"
    
    df = pd.read_sql_query(query, db.connection)
    
    plot_creator = PlotCreator("ions_ligands_analysis", msa_option, single_chain_only, rbp_template_option)
    filtered_ids = plot_creator.get_filtered_pdb_ids(pred_table)
    if filtered_ids:
        df = df[df["exp_db_id"].isin(filtered_ids)]
    
    db.close_connection()

    df = _explode_df_per_target_rna_metrics(df)

    metrics = [(logical, label) for logical, label, _cands in PLOT_METRIC_SPECS if logical in df.columns]
    if not metrics:
        logging.error("No plottable metric columns on %s after prepare; check schema.", pred_table)
        return

    df["category"] = df.apply(lambda row: classify_ions_ligands(row["Ions"], row["Ligands"]), axis=1)

    df["ion_total_n"] = df["Ions"].apply(_total_n_from_ions_ligands_column)
    df["ligand_total_n"] = df["Ligands"].apply(_total_n_from_ions_ligands_column)
    df["ion_count_bin"] = df["ion_total_n"].apply(_bin_count)
    df["ligand_count_bin"] = df["ligand_total_n"].apply(_bin_count)

    df["ion_loc_sum"] = df["Ion_Location"].apply(
        lambda c: _summarize_ion_location(_safe_load_json_dict(c))
    )
    df["ion_chelation_cat"] = df["ion_loc_sum"].apply(_ion_chelation_category)
    df["ion_atom_types_raw"] = df["ion_loc_sum"].apply(_ion_atom_types_label)
    df["ion_multidentate_cat"] = df["ion_loc_sum"].apply(_ion_multidentate_label)

    at_mask = df["ion_atom_types_raw"].ne("n/a")
    if at_mask.any():
        at_vc = df.loc[at_mask, "ion_atom_types_raw"].value_counts()
        min_n = max(5, int(0.01 * len(df)))
        frequent_at = set(at_vc[at_vc >= min_n].index)

        def _collapse_atom_types(lab):
            if lab == "n/a" or lab == "chelating (no typed atoms)":
                return lab
            return lab if lab in frequent_at else "other_combo"

        df["ion_atom_types_plot"] = df["ion_atom_types_raw"].apply(_collapse_atom_types)
    else:
        df["ion_atom_types_plot"] = df["ion_atom_types_raw"]

    df["ion_chelation_hue"] = df["ion_loc_sum"].apply(_ion_chelation_hue_from_summary)
    df["ion_species_hue"] = df.apply(_dominant_ion_species_row, axis=1)
    min_spec_n = max(8, int(0.02 * max(len(df), 1)))
    df["ion_species_hue_c"] = _collapse_rare_categories(df["ion_species_hue"], min_spec_n)

    df["ion_combo_x"] = df["ion_chelation_hue"].astype(str) + " | n=" + df["ion_count_bin"].astype(str)

    _vcounts = df["Ions"].apply(_ion_valence_counts_from_ions_col)
    df["ion_mono_n"] = _vcounts.apply(lambda t: t[0])
    df["ion_di_n"] = _vcounts.apply(lambda t: t[1])
    df["ion_other_n"] = _vcounts.apply(lambda t: t[2])
    _ion_detail_bucket, ion_count_detail_order = _ion_count_detail_bucket_and_order(df["ion_total_n"])
    df["ion_count_detail"] = df["ion_total_n"].apply(_ion_detail_bucket)
    if (df["ion_count_detail"] == "unknown").any() and "unknown" not in ion_count_detail_order:
        ion_count_detail_order = list(ion_count_detail_order) + ["unknown"]
    df["ion_combo_x_detail"] = (
        df["ion_chelation_hue"].astype(str) + " | n=" + df["ion_count_detail"].astype(str)
    )
    df["ion_chemical_class"] = df["Ions"].apply(_ion_chemical_class)
    if "exp_RNA_Stems" in df.columns:
        df["n_rna_stems_exp"] = df["exp_RNA_Stems"].apply(_exp_n_stems_cell)
        df["rna_stem_bin"] = df["n_rna_stems_exp"].apply(_rna_stem_count_bin)
    df["ion_valence_majority"] = df.apply(
        lambda r: _valence_majority_category(r["ion_mono_n"], r["ion_di_n"], r["ion_other_n"]),
        axis=1,
    )
    df["chelating_phosphate_cat"] = df["ion_loc_sum"].apply(_chelating_phosphate_category)
    df["chel_phosphate_bin"] = (df["chelating_phosphate_cat"] == "phosphate_in_chelation").astype(int)
    df["chelated_bin"] = df["ion_loc_sum"].apply(lambda s: int(s["chelating_n"] > 0))
    # 1 = Ion_Location has non-chelating RNA-site ion entries only (no chelating sites).
    df["non_chelated_only_bin"] = df["ion_loc_sum"].apply(
        lambda s: int(s["nonchelating_n"] > 0 and s["chelating_n"] == 0)
    )
    df["chel_mixed_bin"] = df["ion_loc_sum"].apply(
        lambda s: int(s["chelating_n"] > 0 and s["nonchelating_n"] > 0)
    )
    df["ion_coord_class"] = df["ion_loc_sum"].apply(_ion_coord_class_from_summary)
    df["chelating_max_n_contacts"] = df["ion_loc_sum"].apply(
        lambda s: int(s.get("chelating_max_n_contacts", 0) or 0)
    )
    df["chel_coord_num_bin"] = df["ion_loc_sum"].apply(_chel_coord_num_bin_from_max)
    df["ion_coord_class_ord"] = df["ion_coord_class"].map(
        lambda c: float(ION_COORD_CLASS_ORD[c]) if c in ION_COORD_CLASS_ORD else np.nan
    )
    df["chel_mono_bin"] = df["ion_loc_sum"].apply(
        lambda s: int(s["chelating_n"] > 0 and not s["multidentate_any"])
    )
    df["chel_multi_bin"] = df["ion_loc_sum"].apply(
        lambda s: int(s["chelating_n"] > 0 and s["multidentate_any"])
    )
    df["ion_min_dist_A"] = df["Ion_Location"].apply(_ion_location_global_min_distance)
    df["ion_min_dist_bin"] = df["ion_min_dist_A"].apply(_bin_min_dist_angstrom)
    df["ion_min_dist_ord"] = df["ion_min_dist_bin"].map(
        lambda b: MIN_DIST_ORD[b] if b in MIN_DIST_ORD else np.nan
    )
    df["ion_rna_region_cat"] = df["ion_loc_sum"].apply(_ion_rna_region_cat_from_summary)
    df["ion_rna_region_ord"] = df["ion_rna_region_cat"].map(
        lambda c: RNA_REGION_ORD[c] if c in RNA_REGION_ORD else np.nan
    )
    df["rna_site_phosphate_bin"] = df["ion_loc_sum"].apply(
        lambda s: _chelating_has_atom_region(s, "phosphate")
    )
    df["rna_site_ribose_bin"] = df["ion_loc_sum"].apply(lambda s: _chelating_has_atom_region(s, "ribose"))
    df["rna_site_base_bin"] = df["ion_loc_sum"].apply(lambda s: _chelating_has_atom_region(s, "base"))
    df["chel_phosphate_ribose_bin"] = (
        df["rna_site_phosphate_bin"].eq(1) & df["rna_site_ribose_bin"].eq(1)
    ).astype(int)
    if "NumberRNAs" in df.columns:
        _nr = pd.to_numeric(df["NumberRNAs"], errors="coerce").replace(0, np.nan)
        df["ion_per_rna_chain"] = pd.to_numeric(df["ion_total_n"], errors="coerce") / _nr
    else:
        df["ion_per_rna_chain"] = np.nan

    species_min = max(4, int(0.01 * max(len(df), 1)))
    ion_predictor_cols = _install_species_count_predictors(df, "Ions", "ionct", species_min)
    lig_predictor_cols = _install_species_count_predictors(df, "Ligands", "ligct", species_min)
    _lig_bn = df["Ligand_Location"].apply(_ligand_location_bound_nonbinding_n)
    df["lig_bound_n"] = _lig_bn.apply(lambda t: t[0])
    df["lig_non_binding_n"] = _lig_bn.apply(lambda t: t[1])
    
    # Dictionary to store statistics
    stats_dict = {
        "Ions/Ligands Classification Statistics": {},
        "Presence Category Statistics": {},
        "Frequent Ions Statistics": {},
        "Frequent Ligands Statistics": {},
        "Ion Valency Statistics": {},
        "Ion count bin Statistics": {},
        "Ion count detail Statistics": {},
        "Ion valence majority Statistics": {},
        "Chelating phosphate vs other atom types Statistics": {},
        "Phosphate + ribose chelation (both) Statistics": {},
        "Ligand count bin Statistics": {},
        "Ion_Location chelation Statistics": {},
        "Ion_Location atom_types Statistics": {},
        "Ion_Location multidentate Statistics": {},
        "Ion RNA coordination number (max n_contacts) Statistics": {},
        "Ion coordination geometry (mono vs multi) Statistics": {},
        "RNA chelation dominant moiety Statistics": {},
        "Ion min distance bin Statistics": {},
        "Baseline median delta (vs none)": {},
        "Ion chemical class Statistics": {},
        "RNA stem count (exp) Statistics": {},
    }

    # Hand the fully prepared frame (all derived ion columns + per-species
    # predictors + metric list) back to a caller that wants to compute its own
    # figure (e.g. the +/-RBP Spearman-reversal figure) using identical prep.
    if return_prepared:
        return {
            "df": df,
            "metrics": metrics,
            "ion_predictor_cols": ion_predictor_cols,
            "plot_creator": plot_creator,
        }

    # Heatmaps-only mode (--filter sweep): emit just the two Spearman heatmaps and
    # stop. All derived columns they need (ion_total_n/mono/di, chelation bins,
    # per-species predictor columns, metrics) are already computed above.
    if heatmaps_only:
        _plot_factor_impact_heatmap(df, metrics, plot_creator, stats_dict=stats_dict)
        if ion_predictor_cols:
            _plot_spearman_heatmap_matrix(
                df,
                metrics,
                plot_creator,
                ion_predictor_cols,
                "Spearman ρ: per-ion counts vs metrics (red +, blue −)",
                "per_ion_count_spearman",
                stats_dict=stats_dict,
                stats_section="Spearman ρ: per-ion counts vs metrics",
            )
        return

    count_bin_order_strip = ["0", "1", "2", "3+"]
    ion_combo_x_order = [
        f"{c} | n={b}"
        for c in (
            "chelating",
            "nonchelating",
            "chelating+nonchelating",
            "no_RNA_contact_site",
        )
        for b in count_bin_order_strip
    ]
    _chelation_hues = (
        "chelating",
        "nonchelating",
        "chelating+nonchelating",
        "no_RNA_contact_site",
    )
    ion_combo_x_order_detail = [
        f"{c} | n={b}" for c in _chelation_hues for b in ion_count_detail_order
    ]

    for metric, label in metrics:
        data = []
        labels = []
        metric_stats = {}
        for cat in ['none', 'supported', 'unsupported']:
            if metric in ['Complex_RMSD', 'RNA_RMSD']:
                vals = df[df['category'] == cat][metric].apply(parse_rna_rmsd).dropna()
                vals = pd.to_numeric(vals, errors='coerce').dropna()
            else:
                vals = pd.to_numeric(df[df['category'] == cat][metric], errors='coerce').dropna()
            if len(vals) > 0:
                data.append(vals)
                labels.append(cat)
                # Calculate statistics
                mean, std, median = calculate_mean_std(vals.tolist())
                metric_stats[cat] = (mean, std, median, len(vals))
        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source='ions_ligands',
                data_values=data,
                labels=labels,
                name=f'{metric}_by_ions_ligands',
                yAxis_label=label,
            )
        if metric_stats:
            stats_dict['Ions/Ligands Classification Statistics'][label] = metric_stats

    # --- PLOTS BY PRESENCE CATEGORY ---
    df['presence_category'] = df.apply(lambda row: classify_presence(row['Ions'], row['Ligands']), axis=1)

    for metric, label in metrics:
        data = []
        labels = []
        metric_stats = {}
        for cat in ['none', 'ions', 'ligand', 'ions/ligands']:
            category_df = df[df['presence_category'] == cat]
            if category_df.empty:
                continue

            if metric in ['Complex_RMSD', 'RNA_RMSD']:
                vals = category_df[metric].apply(parse_rna_rmsd).dropna()
            else:
                vals = pd.to_numeric(category_df[metric], errors='coerce').dropna()

            if not vals.empty:
                data.append(vals)
                labels.append(cat)
                # Calculate statistics
                mean, std, median = calculate_mean_std(vals.tolist())
                metric_stats[cat] = (mean, std, median, len(vals))

        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source='ions_ligands_presence',
                data_values=data,
                labels=labels,
                name=f'{metric}_by_presence',
                yAxis_label=label,
            )
        if metric_stats:
            stats_dict['Presence Category Statistics'][label] = metric_stats

    # --- IONS ---
    ion_names = extract_items(df, 'Ions')
    ion_counts = Counter(ion_names)
    frequent_ions = [ion for ion, count in ion_counts.items() if count > 2]

    for metric, label in metrics:
        data = []
        labels_ = []
        metric_stats = {}
        # Frequent ions
        for ion in frequent_ions:
            mask = get_samples_with_item(df, 'Ions', ion)
            vals = df.loc[mask, metric]
            if metric in ['Complex_RMSD', 'RNA_RMSD']:
                vals = vals.apply(parse_rna_rmsd)
            vals = pd.to_numeric(vals, errors='coerce').dropna()
            if len(vals) > 0:
                data.append(vals)
                labels_.append(ion)
                # Calculate statistics
                mean, std, median = calculate_mean_std(vals.tolist())
                metric_stats[ion] = (mean, std, median, len(vals))
        # Add group for samples with no ions
        mask_no_ions = df['Ions'].apply(is_empty_list)
        vals = df.loc[mask_no_ions, metric]
        if metric in ['Complex_RMSD', 'RNA_RMSD']:
            vals = vals.apply(parse_rna_rmsd)
        vals = pd.to_numeric(vals, errors='coerce').dropna()
        if len(vals) > 0:
            data.append(vals)
            labels_.append('none')
            # Calculate statistics
            mean, std, median = calculate_mean_std(vals.tolist())
            metric_stats['none'] = (mean, std, median, len(vals))
        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source='ions_ligands',
                data_values=data,
                labels=labels_,
                name=f'{metric}_by_frequent_ions',
                yAxis_label=label,
            )
        if metric_stats:
            stats_dict['Frequent Ions Statistics'][label] = metric_stats

    # --- LIGANDS ---
    ligand_names = extract_items(df, 'Ligands')
    ligand_counts = Counter(ligand_names)
    frequent_ligands = [lig for lig, count in ligand_counts.items() if count > 4]

    for metric, label in metrics:
        data = []
        labels_ = []
        metric_stats = {}
        # Frequent ligands
        for lig in frequent_ligands:
            mask = get_samples_with_item(df, 'Ligands', lig)
            vals = df.loc[mask, metric]
            if metric in ['Complex_RMSD', 'RNA_RMSD']:
                vals = vals.apply(parse_rna_rmsd)
            vals = pd.to_numeric(vals, errors='coerce').dropna()
            if len(vals) > 0:
                data.append(vals)
                labels_.append(lig)
                # Calculate statistics
                mean, std, median = calculate_mean_std(vals.tolist())
                metric_stats[lig] = (mean, std, median, len(vals))
        # Add group for samples with no ligands
        mask_no_ligands = df['Ligands'].apply(is_empty_list)
        vals = df.loc[mask_no_ligands, metric]
        if metric in ['Complex_RMSD', 'RNA_RMSD']:
            vals = vals.apply(parse_rna_rmsd)
        vals = pd.to_numeric(vals, errors='coerce').dropna()
        if len(vals) > 0:
            data.append(vals)
            labels_.append('none')
            # Calculate statistics
            mean, std, median = calculate_mean_std(vals.tolist())
            metric_stats['none'] = (mean, std, median, len(vals))
        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source='ions_ligands',
                data_values=data,
                labels=labels_,
                name=f'{metric}_by_frequent_ligands',
                yAxis_label=label,
                rotation=90,
            )
        if metric_stats:
            stats_dict['Frequent Ligands Statistics'][label] = metric_stats

    df['ion_valency'] = df['Ions'].apply(classify_ion_valency)
    for metric, label in metrics:
        data = []
        labels = []
        metric_stats = {}
        for cat in ['none', 'monovalent', 'divalent', 'mixed']:
            category_df = df[df['ion_valency'] == cat]
            if category_df.empty:
                continue

            if metric in ['Complex_RMSD', 'RNA_RMSD']:
                vals = category_df[metric].apply(parse_rna_rmsd).dropna()
            else:
                vals = pd.to_numeric(category_df[metric], errors='coerce').dropna()

            if not vals.empty:
                data.append(vals)
                labels.append(cat)
                # Calculate statistics
                mean, std, median = calculate_mean_std(vals.tolist())
                metric_stats[cat] = (mean, std, median, len(vals))

        if data:
            _call_grouped_violin_plot(
                plot_creator,
                table_source='ion_valency',
                data_values=data,
                labels=labels,
                name=f'{metric}_by_ion_valency',
                yAxis_label=label,
            )
        if metric_stats:
            stats_dict['Ion Valency Statistics'][label] = metric_stats
    
    count_bin_order = ["0", "1", "2", "3+"]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion count bin Statistics",
        "ion_count_bin",
        count_bin_order,
        "ion_count_bin",
        x_axis_label="Ion Count",
    )
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion count detail Statistics",
        "ion_count_detail",
        ion_count_detail_order,
        "ion_count_detail",
        x_axis_label="Ion Count",
    )
    ion_val_maj_order = ["no_ions", "mono_majority", "di_majority", "mixed_valence"]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion valence majority Statistics",
        "ion_valence_majority",
        ion_val_maj_order,
        "ion_valence_majority",
        rotation=15,
        x_axis_label="Ion Valence",
    )
    chel_p_order = [
        "no_chelating",
        "phosphate_in_chelation",
        "chelating_no_phosphate_type",
        "chelating_untyped_atoms",
    ]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Chelating phosphate vs other atom types Statistics",
        "chelating_phosphate_cat",
        chel_p_order,
        "chelating_phosphate_category",
        rotation=20,
    )
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Phosphate + ribose chelation (both) Statistics",
        "chel_phosphate_ribose_bin",
        [0, 1],
        "chel_po4_ribose_both",
        rotation=0,
        x_axis_label="0 = not both; 1 = chelating RNA-ion sites include phosphate and ribose",
    )
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ligand count bin Statistics",
        "ligand_count_bin",
        count_bin_order,
        "ligand_count_bin",
    )

    ion_min_dist_bin_order = ["≤3 Å", "3–5 Å", ">5 Å", "unknown"]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion min distance bin Statistics",
        "ion_min_dist_bin",
        ion_min_dist_bin_order,
        "ion_min_distance_bin",
        rotation=15,
        x_axis_label="RNA–ion min distance (Å)",
    )

    ion_chelation_order = [
        "no_ion_geometry",
        "chelating_only",
        "non_chelating_only",
        "chelating+non_chelating",
    ]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion_Location chelation Statistics",
        "ion_chelation_cat",
        ion_chelation_order,
        "ion_location_chelation",
        rotation=15,
    )

    atom_types_order = sorted(df["ion_atom_types_plot"].dropna().unique().tolist())
    if "other_combo" in atom_types_order:
        atom_types_order = [x for x in atom_types_order if x != "other_combo"]
        atom_types_order.append("other_combo")
    if not atom_types_order:
        atom_types_order = ["n/a"]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion_Location atom_types Statistics",
        "ion_atom_types_plot",
        atom_types_order,
        "ion_location_atom_types",
        rotation=25,
    )

    ion_md_order = ["n/a", "monodentate_only", "multidentate"]
    _plot_metrics_by_category(
        df,
        metrics,
        plot_creator,
        stats_dict,
        "Ion_Location multidentate Statistics",
        "ion_multidentate_cat",
        ion_md_order,
        "ion_location_multidentate",
    )

    chel_coord_bins_order = [
        "0 (no chel site)",
        "1 contact / site (mono)",
        "2 contacts (max)",
        "3+ contacts (max)",
        "unknown",
    ]
    chel_coord_bins_order = [c for c in chel_coord_bins_order if c in df["chel_coord_num_bin"].values]
    if chel_coord_bins_order:
        _plot_metrics_by_category(
            df,
            metrics,
            plot_creator,
            stats_dict,
            "Ion RNA coordination number (max n_contacts) Statistics",
            "chel_coord_num_bin",
            chel_coord_bins_order,
            "ion_max_n_contacts_bin",
            rotation=12,
            x_axis_label="Max RNA–atom contacts per chelating site (Ion_Location n_contacts; 0 = no chelation)",
        )

    coord_geo_order = [c for c in ("none", "mono", "multi") if c in df["ion_coord_class"].values]
    if coord_geo_order:
        _plot_metrics_by_category(
            df,
            metrics,
            plot_creator,
            stats_dict,
            "Ion coordination geometry (mono vs multi) Statistics",
            "ion_coord_class",
            coord_geo_order,
            "ion_coord_class_mono_multi",
            rotation=0,
            x_axis_label="none=no chelating sites; mono=all monodentate; multi=any multidentate",
        )

    moiety_order_full = (
        "non_chelating",
        "phosphate",
        "ribose",
        "base",
        "mixed",
        "untyped",
    )
    moiety_order_use = [c for c in moiety_order_full if c in df["ion_rna_region_cat"].values]
    if moiety_order_use:
        _plot_metrics_by_category(
            df,
            metrics,
            plot_creator,
            stats_dict,
            "RNA chelation dominant moiety Statistics",
            "ion_rna_region_cat",
            moiety_order_use,
            "ion_rna_chelation_moiety",
            rotation=15,
            x_axis_label="Dominant RNA chelation type (union of chelating atom_types); non_chelating=none",
        )

    # --- Ion impact: Spearman ρ heatmap (ion predictors vs metrics) ---
    _plot_factor_impact_heatmap(df, metrics, plot_creator, stats_dict=stats_dict)

    # --- Baseline (no ions/ligands) Δ and multivariate adjustment ---
    if len(df[df["presence_category"] == "none"]) >= 5:
        _plot_baseline_delta_bars(df, metrics, plot_creator, stats_dict, "Baseline median delta (vs none)")
        _plot_forest_baseline_presence(df, metrics, plot_creator)
    chem_order = [
        "no_ions",
        "monovalent_only",
        "contains_Mg",
        "Ca_group_no_Mg",
        "other_divalent_no_Mg",
        "mixed_or_other",
        "unknown",
    ]
    chem_order = [c for c in chem_order if c in df["ion_chemical_class"].values]
    if chem_order:
        _plot_violin_stratified(
            df,
            metrics,
            plot_creator,
            "ion_chemical_class",
            chem_order,
            "ion_chemical_class",
            stats_dict,
            "Ion chemical class Statistics",
            x_axis_label="Ion chemistry (inventory)",
            rotation=18,
        )
    # Violin suffix e.g. RNA_TM_rna_stem_count_exp: metric vs bins of experimental stem count
    # (number of DSSR stem loops in exp_RNA_Stems, binned 0/1/2/3+).
    if "rna_stem_bin" in df.columns and df["rna_stem_bin"].notna().any():
        stem_ord = [x for x in ["0", "1", "2", "3+", "unknown"] if x in df["rna_stem_bin"].values]
        if stem_ord:
            _plot_violin_stratified(
                df,
                metrics,
                plot_creator,
                "rna_stem_bin",
                stem_ord,
                "rna_stem_count_exp",
                stats_dict,
                "RNA stem count (exp) Statistics",
                x_axis_label="Experimental RNA stems (DSSR)",
                rotation=0,
            )
    _plot_adjusted_coefficient_forest(df, metrics, plot_creator, stats_dict)
    _run_multivariate_driver_models(
        df,
        metrics,
        stats_dict,
        ion_predictor_cols=ion_predictor_cols,
        lig_predictor_cols=lig_predictor_cols,
    )
    _plot_partial_residual_spearman_panel(
        df,
        metrics,
        plot_creator,
        x_col="chelated_bin",
        save_name="partial_residual_chelation_spearman",
        xlabel="Spearman ρ: chelated_bin vs residual metric (after length + ion_n; 1 = any chelating RNA-site ion)",
        plot_title="Partial residual vs chelation (chelated_bin)",
        stats_dict=stats_dict,
        stats_section="Partial residual Spearman: chelated_bin vs metric residual",
    )
    _plot_partial_residual_spearman_panel(
        df,
        metrics,
        plot_creator,
        x_col="non_chelated_only_bin",
        save_name="partial_residual_nonchel_spearman",
        xlabel=(
            "Spearman ρ: non_chelated_only_bin vs residual metric (after length + ion_n; "
            "1 = non-chelating RNA-site ions only, no chelating sites)"
        ),
        plot_title="Partial residual vs non-chelating-only ion contacts",
        bar_color="coral",
        stats_dict=stats_dict,
        stats_section="Partial residual Spearman: non_chelated_only vs metric residual",
    )

    extra_geom_factors = [
        ("ion_min_dist_A", "RNA-ion <3 Angs", {"ion_close_subset": True}),
        ("chelated_bin", "chelated by RNA (0/1)"),
        ("chel_phosphate_ribose_bin", "PO4+ribose chel both (0/1)"),
        ("rna_site_phosphate_bin", "any PO4 chel site"),
        ("rna_site_ribose_bin", "any ribose chel site"),
        ("rna_site_base_bin", "any base chel site"),
        ("chel_mono_bin", "coord mono (1 contact)"),
        ("chel_multi_bin", "coord multi (≥2 contacts)"),
    ]
    if df["ion_per_rna_chain"].notna().any():
        extra_geom_factors.append(("ion_per_rna_chain", "ions / RNA chain"))
    _plot_spearman_heatmap_matrix(
        df,
        metrics,
        plot_creator,
        extra_geom_factors,
        "Spearman ρ: geometry / distance / density vs metrics (red +, blue −)",
        "ion_geometry_distance_density_spearman",
        stats_dict=stats_dict,
        stats_section="Spearman ρ: geometry / distance / density vs metrics",
    )

    # Coordination # (max n_contacts / site), mono vs multi class, dominant PO4/ribose/base moiety:
    # strong |ρ| with max contacts / multi → geometry/local electrostatics; with moiety ord → chemistry/screening.
    coord_moiety_factors = [
        ("chelating_max_n_contacts", "max RNA contacts / chel site"),
        ("ion_coord_class_ord", "coord: none<mono<multi (ordinal)"),
        ("ion_rna_region_ord", "moiety: PO4<ribose<base<mixed (ordinal; NaN if non-chel)"),
    ]
    coord_moiety_factors = [t for t in coord_moiety_factors if t[0] in df.columns]
    if len(coord_moiety_factors) >= 2:
        _plot_spearman_heatmap_matrix(
            df,
            metrics,
            plot_creator,
            coord_moiety_factors,
            (
                "Spearman ρ: coordination & RNA chelation moiety vs metrics\n"
                "(geometry: max contacts + mono/multi; screening/chemistry: phosphate vs ribose vs base)"
            ),
            "ion_coordination_moiety_spearman",
            stats_dict=stats_dict,
            stats_section="Spearman ρ: coordination & chelation moiety vs metrics",
        )

    if ion_predictor_cols:
        _plot_spearman_heatmap_matrix(
            df,
            metrics,
            plot_creator,
            ion_predictor_cols,
            "Spearman ρ: per-ion counts vs metrics (red +, blue −)",
            "per_ion_count_spearman",
            stats_dict=stats_dict,
            stats_section="Spearman ρ: per-ion counts vs metrics",
        )
    lig_spearman_factors = list(lig_predictor_cols) + [
        ("lig_bound_n", "bound (n sites)"),
        ("lig_non_binding_n", "non_binding (n sites)"),
    ]
    _plot_spearman_heatmap_matrix(
        df,
        metrics,
        plot_creator,
        lig_spearman_factors,
        "Spearman ρ: per-ligand counts vs metrics (red +, blue −)",
        "per_ligand_count_spearman",
        stats_dict=stats_dict,
        stats_section="Spearman ρ: per-ligand counts vs metrics",
    )

    cls = getattr(config, "classification", None)
    for mcol, mlab in _combined_mean_metric_tuples(cls, set(df.columns)):
        _plot_combined_ion_coord_mean_heatmap(df, mcol, mlab, plot_creator, cls)

    # --- 2D strip plots: count × chelation (hue) and count × species code (hue); combo x encodes two factors ---
    for metric, label in metrics:
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_count_bin",
            "ion_chelation_hue",
            "strip2d_ioncount_chelationhue",
            x_order=count_bin_order_strip,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_count_detail",
            "ion_chelation_hue",
            "strip2d_ioncountdetail_chelationhue",
            x_order=ion_count_detail_order,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_count_detail",
            "chelating_phosphate_cat",
            "strip2d_ioncountdetail_phosphatechelcat",
            x_order=ion_count_detail_order,
            rotation=12,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_count_bin",
            "ion_species_hue_c",
            "strip2d_ioncount_ionspecieshue",
            x_order=count_bin_order_strip,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_combo_x",
            None,
            "strip2d_ion_combo_chelation_x_count",
            x_order=ion_combo_x_order,
            rotation=55,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_combo_x",
            "ion_species_hue_c",
            "strip2d_ioncombo_chelationcount_ionspecieshue",
            x_order=ion_combo_x_order,
            rotation=55,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_combo_x_detail",
            None,
            "strip2d_ion_combo_chelation_x_count_detail",
            x_order=ion_combo_x_order_detail,
            rotation=55,
        )
        _plot_strip_metric(
            df,
            plot_creator,
            metric,
            label,
            "ion_combo_x_detail",
            "ion_species_hue_c",
            "strip2d_ioncombo_chelationcountdetail_ionspecieshue",
            x_order=ion_combo_x_order_detail,
            rotation=55,
        )

    # Write statistics to file
    _merge_spearman_sections_for_summary(stats_dict)
    write_statistics_to_file(stats_dict, plot_creator, msa_option, single_chain_only, rbp_template_option)


# --------------------------------------------------------------------------- #
# +RBP vs -RBP ion-dependence reversal figure (2 publication panels)
# --------------------------------------------------------------------------- #
def plot_rbp_reversal_figure(single_chain_only=False, out_name="ion_rbp_reversal"):
    """Two-panel Nature-style figure contrasting +RBP and -RBP template filters.

    A  iLDDT Spearman ρ vs per-ion count  — interface quality. Reverses sign for
       most ions (strongest for Mg/Zn/Mn) while K stays positive in both regimes.
    B  RNA TM Spearman ρ vs per-ion count — RNA-fold accuracy.

    iLDDT (A) and RNA TM (B) are used because, among DockQ/iLDDT and RNA TM/RNA
    LDDT respectively, they show the +RBP/-RBP reversal most strongly and
    consistently. Bars use the shared RBP_TEMPLATE_COLORS palette, matching the
    colour key used in phCorrelation.py.
    """
    from plots.publication_figure import PublicationFigure, style_axis
    from plots.plotCreator import RBP_TEMPLATE_COLORS

    IFACE_METRIC = ("iLDDT", "iLDDT ρ vs ion count")
    RNA_METRIC = ("RNA_TM", "RNA TM ρ vs ion count")
    min_n = 8

    def _prep(rbp):
        o = main(rbp_template_option=rbp, single_chain_only=single_chain_only,
                 return_prepared=True, parse_argv=False)
        return o["df"], {lab: col for col, lab in o["ion_predictor_cols"]}, len(o["df"])

    dfp, cp, n_p = _prep("+RBPtemp")
    dfm, cm, n_m = _prep("-RBPtemp")
    common = sorted(set(cp) & set(cm))

    def _rho(df, col, metric):
        if metric not in df.columns or col not in df.columns:
            return np.nan
        r, _p = _spearman_rho_and_p(df[col], _metric_y_series(df, metric), min_n=min_n)
        return r

    # Order ions by interface-metric reversal magnitude (most reversed first).
    rows = [(ion, _rho(dfp, cp[ion], IFACE_METRIC[0]),
             _rho(dfm, cm[ion], IFACE_METRIC[0])) for ion in common]
    rows.sort(key=lambda r: -(abs((r[1] or 0) - (r[2] or 0))))
    ord_ions = [r[0] for r in rows]

    def _bars(ax, metric_col, ylabel):
        plus = [_rho(dfp, cp[i], metric_col) for i in ord_ions]
        minus = [_rho(dfm, cm[i], metric_col) for i in ord_ions]
        x = np.arange(len(ord_ions))
        w = 0.38
        ax.bar(x - w / 2, plus, w, color=RBP_TEMPLATE_COLORS["+RBP"], alpha=0.9,
               label=f"+RBP (n={n_p})")
        ax.bar(x + w / 2, minus, w, color=RBP_TEMPLATE_COLORS["-RBP"], alpha=0.9,
               label=f"-RBP (n={n_m})")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ord_ions)
        ax.set_xlabel("Ion species")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        style_axis(ax, grid_axis="y")

    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "plots", "all_chains", "publication_figures")
    fig = PublicationFigure(ncols=2, panel_size=(6.6, 4.9), outdir=outdir)
    fig.add_panel(lambda ax: _bars(ax, IFACE_METRIC[0], IFACE_METRIC[1]), title=None)
    fig.add_panel(lambda ax: _bars(ax, RNA_METRIC[0], RNA_METRIC[1]), title=None)
    return fig.save(out_name)


# --------------------------------------------------------------------------- #
# Does the ion effect "rule" the RBP-template effect? — variance partition
# --------------------------------------------------------------------------- #
def plot_ion_vs_rbp_variance_partition(single_chain_only=False, metric="iLDDT",
                                       out_name="ion_vs_rbp_variance_partition",
                                       min_species_n=20):
    """Decompose how much unique variance ion *type* vs RBP templates explain.

    Restricted to ion-containing complexes (ion_total_n > 0) so the comparison is
    about ion identity/amount, not ion presence vs absence (which RBP templates
    already track). For each ion species with at least ``min_species_n``
    complexes, fits the two-way model (confounders partialled out first,
    matching the multivariate driver model)::

        metric ~ rna_len + prot_len + deep_msa + <species count> * RBP

    and reports the semipartial ΔR² (unique variance) added by the species-count
    main effect, the RBP main effect and their interaction. Valency classes are
    deliberately not used because species within a class differ (e.g. Ca vs Mg,
    K vs Na). Bars use the shared phCorrelation hues (PH_RANGE_COLORS).
    """
    import statsmodels.formula.api as smf
    from plots.publication_figure import PublicationFigure, style_axis
    from plots.plotCreator import PH_RANGE_COLORS

    o = main(single_chain_only=single_chain_only, return_prepared=True,
             parse_argv=False)
    df = o["df"]
    if "RBP_PDB_Templates" not in df.columns:
        raise SystemExit("RBP_PDB_Templates column absent; cannot model RBP effect.")

    work = pd.DataFrame(index=df.index)
    work["y"] = _metric_y_series(df, metric)
    # RBP template present (same emptiness rule as the SQL +/-RBP filter).
    t = df["RBP_PDB_Templates"].astype(str).str.strip().str.lower()
    work["RBP"] = (~t.isin(["", "[]", "null", "none", "nan"])).astype(float)
    # RNALength / ProteinLength are sparsely populated; median-impute them as
    # the multivariate driver model does, so they act as confounders without
    # discarding the ~75 % of complexes that lack a stored length.
    work["rna_len"] = pd.to_numeric(df.get("RNALength"), errors="coerce")
    work["rna_len"] = work["rna_len"].fillna(work["rna_len"].median())
    work["prot_len"] = pd.to_numeric(df.get("ProteinLength"), errors="coerce")
    work["prot_len"] = work["prot_len"].fillna(work["prot_len"].median())
    if "af3_rna_MSA" in df.columns:
        work["deep_msa"] = df["af3_rna_MSA"].apply(_deep_msa_flag).astype(float)
    else:
        work["deep_msa"] = 0.0
    # Ion-containing complexes only (ion identity/amount, not presence/absence).
    work["ion_total_n"] = pd.to_numeric(df.get("ion_total_n"), errors="coerce")
    ion_containing = work["ion_total_n"] > 0

    # One predictor per ion species with adequate sample size (well-powered
    # species lead, e.g. Zn, Mg); ordered by n descending.
    species = []
    for col, lab in o["ion_predictor_cols"]:
        cnt = pd.to_numeric(df[col], errors="coerce")
        work[col] = cnt
        n_present = int(((cnt.fillna(0) > 0) & ion_containing).sum())
        if n_present >= min_species_n:
            species.append((col, lab, n_present))
    species.sort(key=lambda s: -s[2])
    ion_preds = [(c, l) for c, l, _n in species]
    species_n = {c: n for c, _l, n in species}   # complexes containing the species
    conf = "rna_len + prot_len + deep_msa"

    def _r2(formula, data):
        return smf.ols(formula, data=data).fit().rsquared

    results = []  # (label, ion_dR2, rbp_dR2, int_dR2, n, R2_full)
    for col, lab in ion_preds:
        d = work.loc[ion_containing,
                     ["y", "RBP", "rna_len", "prot_len", "deep_msa", col]].dropna()
        d = d[np.isfinite(d["y"])]
        if len(d) < 30 or d[col].nunique() < 2 or d["RBP"].nunique() < 2:
            continue
        base_rbp = _r2(f"y ~ {conf} + RBP", d)          # ion added on top of RBP
        base_ion = _r2(f"y ~ {conf} + {col}", d)        # RBP added on top of ion
        add = _r2(f"y ~ {conf} + {col} + RBP", d)       # additive
        full = _r2(f"y ~ {conf} + {col} * RBP", d)      # + interaction
        results.append((lab,
                        max(add - base_rbp, 0.0),       # ion main (beyond RBP)
                        max(add - base_ion, 0.0),       # RBP main (beyond ion)
                        max(full - add, 0.0),           # interaction
                        species_n.get(col, len(d)), full))
    if not results:
        raise SystemExit("Insufficient data to fit the variance-partition model.")

    labels = [r[0] for r in results]

    def draw(ax):
        x = np.arange(len(labels))
        w = 0.26
        ax.bar(x - w, [r[1] * 100 for r in results], w,
               color=PH_RANGE_COLORS["5-6"], label="Ion (main)")
        ax.bar(x, [r[2] * 100 for r in results], w,
               color=PH_RANGE_COLORS["<5"], label="RBP template (main)")
        ax.bar(x + w, [r[3] * 100 for r in results], w,
               color=PH_RANGE_COLORS[">8"], label="Ion × RBP")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r[0]}\n(n={r[4]})" for r in results])
        ax.set_ylabel("Unique variance explained, ΔR² (%)")
        ax.set_xlabel(f"Ion species (ion-containing complexes):  "
                      f"{metric} ~ confounders + species × RBP")
        ax.legend(fontsize=7, loc="best")
        style_axis(ax, grid_axis="y")

    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "plots", "all_chains", "publication_figures")
    fig = PublicationFigure(ncols=1, panel_size=(8.2, 5.0), outdir=outdir,
                            letters=False)
    fig.add_panel(draw, title=None)
    paths = fig.save(out_name)

    print(f"\nVariance partition for {metric} (semipartial ΔR², %):")
    print(f"{'ion predictor':>16}  {'ion':>6} {'RBP':>6} {'ion×RBP':>8} "
          f"{'ion+int':>8} {'fullR2':>7}  n")
    for lab, iod, rbpd, intd, n, full in results:
        print(f"{lab:>16}  {iod*100:6.2f} {rbpd*100:6.2f} {intd*100:8.2f} "
              f"{(iod+intd)*100:8.2f} {full*100:7.2f}  {n}")
    return paths


# --------------------------------------------------------------------------- #
# Phosphate+ribose chelation vs RNA accuracy (Ion_Location required)
# --------------------------------------------------------------------------- #
def plot_phosphate_ribose_chelation_figure(
        single_chain_only=False, out_name="ion_phosphate_ribose_chelation"):
    """RNA accuracy for phosphate+ribose-chelating vs other ion-containing complexes.

    Among ion-containing complexes, ions that chelate *both* the phosphate
    backbone and the ribose are the strongest ion-related predictor of RNA fold
    accuracy (notably in RNA-only structures: RNA TM $\\rho\\approx+0.57$,
    RNA LDDT $\\rho\\approx+0.38$). Two panels (RNA LDDT, RNA TM) compare that
    group against the remaining ion-containing complexes with a Mann--Whitney p.
    Only runs when the Ion_Location-derived chelation columns are populated.
    """
    from scipy.stats import mannwhitneyu
    from plots.publication_figure import PublicationFigure, style_axis, sns_violin
    from plots.plotCreator import PH_RANGE_COLORS

    o = main(single_chain_only=single_chain_only, return_prepared=True,
             parse_argv=False)
    df = o["df"]
    if "chel_phosphate_ribose_bin" not in df.columns:
        raise SystemExit("Ion_Location chelation columns absent; nothing to plot.")
    ion_cont = pd.to_numeric(df.get("ion_total_n"), errors="coerce") > 0
    chel = pd.to_numeric(df["chel_phosphate_ribose_bin"], errors="coerce") == 1
    if int((ion_cont & chel).sum()) < 3 or int((ion_cont & ~chel).sum()) < 3:
        raise SystemExit(
            "Too few complexes per group (Ion_Location empty/sparse); cannot plot.")

    green, grey = PH_RANGE_COLORS["7-8"], "0.6"

    def _panel(metric_col, ylabel):
        def draw(ax):
            g1 = _metric_y_series(df[ion_cont & chel], metric_col).dropna()
            g2 = _metric_y_series(df[ion_cont & ~chel], metric_col).dropna()
            sns_violin(ax, [g1.values, g2.values],
                       [f"phosphate+ribose\n(n={len(g1)})",
                        f"other ion-containing\n(n={len(g2)})"],
                       [green, grey], ylabel=ylabel)
            if len(g1) >= 3 and len(g2) >= 3:
                _u, p = mannwhitneyu(g1, g2)
                ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
                ax.text(0.5, 1.02, f"Mann–Whitney {ptxt}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
            style_axis(ax, grid_axis="y")
        return draw

    metrics = [(m, lab) for m, lab in (("RNA_LDDT_allAtom", "RNA LDDT"),
                                       ("RNA_TM", "RNA TM")) if m in df.columns]
    outdir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "plots", "all_chains", "publication_figures")
    fig = PublicationFigure(ncols=len(metrics), panel_size=(5.4, 4.8), outdir=outdir)
    for mcol, lab in metrics:
        fig.add_panel(_panel(mcol, lab), title=None)
    return fig.save(out_name)


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
# Composition reuses the PNGs this script already saved (run it first).
_COMPOSITE_PLOT_TYPES = ("ions_ligands_analysis",)


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None,
                     rbp_template_option=None):
    """Return {base_name: (draw_fn, title)} from this script's saved PNGs."""
    from plots.publication_figure import saved_png_panels
    return saved_png_panels(
        _COMPOSITE_PLOT_TYPES, names, single_chain_only=single_chain_only,
        msa_option=msa_option, rbp_template_option=rbp_template_option)


if __name__ == '__main__':
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="ions_ligands_composite"))

    _argv_lower = [a.lower() for a in _sys.argv[1:]]
    if 'rbp_reversal' in _argv_lower or '--rbp-reversal' in _argv_lower:
        # Two-panel +RBP vs -RBP ion-dependence reversal figure (iLDDT + RNA TM).
        _single_chain = 'single_chain' in _argv_lower
        plot_rbp_reversal_figure(single_chain_only=_single_chain)
        raise SystemExit(0)

    if 'variance_partition' in _argv_lower or '--variance-partition' in _argv_lower:
        # Variance partition: does the ion effect rule the RBP-template effect?
        _single_chain = 'single_chain' in _argv_lower
        plot_ion_vs_rbp_variance_partition(single_chain_only=_single_chain)
        raise SystemExit(0)

    if 'chelation' in _argv_lower or '--chelation' in _argv_lower:
        # RNA accuracy: phosphate+ribose-chelating vs other ion-containing (needs Ion_Location).
        _single_chain = 'single_chain' in _argv_lower
        plot_phosphate_ribose_chelation_figure(single_chain_only=_single_chain)
        raise SystemExit(0)

    if '--filter' in _argv_lower:
        # Emit only the two Spearman heatmaps, once per +/-MSA and +/-RBP filter
        # (4 filter settings × 2 heatmaps = 8 plots), nothing else.
        _single_chain = 'single_chain' in _argv_lower
        _FILTER_SWEEP = [
            ('+MSA', None),
            ('-MSA', None),
            (None, '+RBPtemp'),
            (None, '-RBPtemp'),
        ]
        for _msa_opt, _rbp_opt in _FILTER_SWEEP:
            print(f"[--filter] Spearman heatmaps for "
                  f"MSA={_msa_opt or 'any'}, RBP={_rbp_opt or 'any'}, "
                  f"single_chain={_single_chain}")
            main(msa_option=_msa_opt, single_chain_only=_single_chain,
                 rbp_template_option=_rbp_opt, heatmaps_only=True, parse_argv=False)
        raise SystemExit(0)

    main()