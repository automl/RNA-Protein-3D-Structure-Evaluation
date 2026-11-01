import os
import sys
import ast
import numpy as np
import os as _os, sys as _sys  # allow direct execution: python evaluation/<script>.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from plots.plotCreator import PlotCreator
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig
import matplotlib.pyplot as plt
import pandas as pd

def max_length(data):
    data_max_length = []
    for exp_db_id, x, rna_length in data:
        if isinstance(rna_length, str) and rna_length.startswith('[') and rna_length.endswith(']'):
            rna_length = max(ast.literal_eval(rna_length))
        data_max_length.append((exp_db_id, x, rna_length))
    return data_max_length


def _safe_literal_eval(value):
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple, int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return value
    return value


def _as_chain_list(value):
    parsed = _safe_literal_eval(value)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, str):
        s = parsed.strip()
        if not s:
            return []
        return [s]
    return []


def _as_float_list(value):
    parsed = _safe_literal_eval(value)
    if isinstance(parsed, list):
        out = []
        for x in parsed:
            try:
                out.append(float(x))
            except Exception:
                out.append(np.nan)
        return out
    try:
        return [float(parsed)]
    except Exception:
        return []


def _first_metric_float(raw):
    """Leading numeric from list-valued RMSD columns (e.g. \"[28.02, 2.49]\" → 28.02)."""
    vals = _as_float_list(raw)
    return vals[0] if vals else np.nan


def _longest_rna_index_and_chain(rna_chain_ids, rna_lengths):
    chains = _as_chain_list(rna_chain_ids)
    lens = _as_float_list(rna_lengths)
    if not chains or not lens:
        return None, None
    n = min(len(chains), len(lens))
    if n <= 0:
        return None, None
    lens = lens[:n]
    chains = chains[:n]
    valid = [(i, v) for i, v in enumerate(lens) if np.isfinite(v)]
    if not valid:
        return None, None
    idx = max(valid, key=lambda t: t[1])[0]
    return idx, chains[idx]


def _extract_longest_rna_matrix_score(raw_matrix, longest_idx):
    """
    RNA_*_AllChains matrices: for predicted RNA column j (aligned to RNAChainIDs),
    take the maximum over reference RNA rows i (matrix[i][j]).
    """
    parsed = _safe_literal_eval(raw_matrix)
    if not isinstance(parsed, list) or longest_idx is None:
        return np.nan
    candidates = []
    try:
        j = int(longest_idx)
        for row in parsed:
            if isinstance(row, list) and j < len(row):
                val = row[j]
                if val is not None:
                    f = float(val)
                    if np.isfinite(f):
                        candidates.append(f)
    except Exception:
        return np.nan
    return max(candidates) if candidates else np.nan


def _extract_rbp_metric_for_rna_chain(raw_pairs, rna_chain, protein_chain_ids):
    """
    RBP_* list of dicts [{'A:B': val}, ...]: average values over pairs where one side
    is rna_chain and the other is a listed protein chain.
    """
    if rna_chain is None:
        return np.nan
    protein_set = set(_as_chain_list(protein_chain_ids))
    parsed = _safe_literal_eval(raw_pairs)
    if not isinstance(parsed, list):
        return np.nan
    vals = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for pair_key, pair_val in item.items():
            if not isinstance(pair_key, str):
                continue
            parts = pair_key.split(":")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a == rna_chain and b in protein_set:
                try:
                    vals.append(float(pair_val))
                except Exception:
                    pass
            elif b == rna_chain and a in protein_set:
                try:
                    vals.append(float(pair_val))
                except Exception:
                    pass
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _extract_rbp_metric_for_longest_rna(raw_pairs, longest_rna_chain, protein_chain_ids):
    """
    RBP_* is a list-like of dicts {'A:E': val}. Keep only pairs involving longest RNA
    and a valid protein chain, then average across proteins.
    """
    return _extract_rbp_metric_for_rna_chain(raw_pairs, longest_rna_chain, protein_chain_ids)


def _rbp_metric_all_protein_rna(raw_pairs, rna_chain_ids, protein_chain_ids):
    """Mean RBP_* over ALL protein-RNA interface pairs (any RNA chain x any protein chain)
    -- the ``_Nr`` reduction used by the publication figures. NaN if no such pair."""
    rset = set(_as_chain_list(rna_chain_ids))
    pset = set(_as_chain_list(protein_chain_ids))
    parsed = _safe_literal_eval(raw_pairs)
    if not isinstance(parsed, list) or not rset or not pset:
        return np.nan
    vals = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for pair_key, pair_val in item.items():
            if not isinstance(pair_key, str):
                continue
            parts = pair_key.split(":")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if (a in rset and b in pset) or (a in pset and b in rset):
                try:
                    vals.append(float(pair_val))
                except (TypeError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else np.nan


def _af3_rna_metric_at_chain_index(raw, chain_idx):
    """af3_rna_pTM / af3_rna_ipTM values listed in RNAChainIDs index order."""
    parsed = _safe_literal_eval(raw)
    if parsed is None:
        return np.nan
    if isinstance(parsed, (list, tuple)):
        if chain_idx < len(parsed):
            try:
                fv = float(parsed[chain_idx])
                return fv if np.isfinite(fv) else np.nan
            except Exception:
                return np.nan
        return np.nan
    try:
        fv = float(parsed)
        if np.isfinite(fv) and chain_idx == 0:
            return fv
    except Exception:
        pass
    return np.nan


def _parse_optional_float_metric(raw):
    """Parse scalar AF3 fields: float, numeric string, or single-element list/tuple."""
    if raw is None:
        return np.nan
    if isinstance(raw, (int, float)):
        fv = float(raw)
        return fv if np.isfinite(fv) else np.nan
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return np.nan
        try:
            fv = float(s)
            return fv if np.isfinite(fv) else np.nan
        except Exception:
            pass
        parsed = _safe_literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            try:
                fv = float(parsed[0])
                return fv if np.isfinite(fv) else np.nan
            except Exception:
                return np.nan
        try:
            fv = float(parsed)
            return fv if np.isfinite(fv) else np.nan
        except Exception:
            return np.nan
    return np.nan


def _audit_af3_entity_plddt_avg(rowdict, col):
    """AF3 entity pLDDT average on 0–1 scale (scalar or list, 0–100 or 0–1)."""
    raw = rowdict.get(col)
    v = _parse_optional_float_metric(raw)
    if np.isfinite(v):
        return float(v / 100.0) if v > 1.0 else float(v)
    parsed = _safe_literal_eval(raw)
    if isinstance(parsed, (list, tuple)) and parsed:
        vals = []
        for x in parsed:
            try:
                fx = float(x)
                if np.isfinite(fx):
                    vals.append(fx / 100.0 if fx > 1.0 else fx)
            except (TypeError, ValueError):
                pass
        if vals:
            return float(np.mean(vals))
    return np.nan


def _scalar_pred_float(raw):
    if raw is None or raw == "":
        return np.nan
    try:
        fv = float(raw)
        return fv if np.isfinite(fv) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _longest_protein_index_and_chain(protein_chain_ids, protein_lengths_raw):
    chains = _as_chain_list(protein_chain_ids)
    lens = _as_float_list(protein_lengths_raw)
    if not chains or not lens:
        return None, None
    n = min(len(chains), len(lens))
    if n <= 0:
        return None, None
    lens = lens[:n]
    chains = chains[:n]
    valid = [(i, v) for i, v in enumerate(lens) if np.isfinite(v)]
    if not valid:
        return None, None
    idx = max(valid, key=lambda t: t[1])[0]
    return idx, chains[idx]


def _effective_msa_depth_for_chain(msa_raw, chain_id):
    if chain_id is None:
        return np.nan
    parsed = _safe_literal_eval(msa_raw)
    if not isinstance(parsed, dict):
        return np.nan
    v = parsed.get(chain_id)
    if v is None:
        v = parsed.get(str(chain_id).strip())
    if v is None:
        return np.nan
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(fv):
        return np.nan
    if fv == 1.0:
        return 0.0
    return fv


def _template_entry_count(raw):
    p = _safe_literal_eval(raw)
    if isinstance(p, list):
        return len(p)
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ("null", "none", "[]"):
            return 0
        try:
            q = ast.literal_eval(s)
            return len(q) if isinstance(q, list) else 0
        except Exception:
            return 0
    return 0


def _rna_length_at_index(rna_lengths_raw, idx):
    lens = _as_float_list(rna_lengths_raw)
    if idx is None or idx >= len(lens):
        return np.nan
    v = lens[idx]
    return float(v) if np.isfinite(v) else np.nan


AUDIT_BIN_SUFFIXES_ORDERED = ("_a", "_b", "_c", "_c_ca", "_c_cb")
AUDIT_AF3_PLOT_QUAD_ROMANS = ("i", "ii", "iii", "iv")
AUDIT_AF3_PLOT_CATEGORY_SUFFIXES = (
    ("_a", "a"),
    ("_b", "b"),
    ("_c_ca", "c_ca"),
    ("_c_cb", "c_cb"),
)
AUDIT_AF3_PLOT_VALUE_METRICS = (
    "af3_ranking_score",
    "af3_protein_ipTM",
    "af3_rna_ipTM",
    "af3_rna_pLDDT_avg",
    "af3_protein_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_protein_pTM",
    "af3_rna_pTM",
    "af3_chain_pair_pae_min",
)
# Engineered / sequence features (same normalization as AF3 value plots).
AUDIT_AF3_PLOT_FEATURE_METRICS = (
    "RBP_PDB_n",
    "RNA_length_longest",
    "RNA_Shannon_Entropy_K2",
    "RNA_MFE_Value",
    "ProteinChain_Count",
    "Protein_length_longest",
    "RNA_chain_load",
    "has_RNA_MSA",
    "RNA_msa_depth",
    "Protein_PDB_n",
    "RNAChain_Count",
)
# Binary features: mean fraction ± SD per category (no min–max scaling).
AUDIT_AF3_PLOT_BINARY_FEATURE_METRICS = frozenset({"has_RNA_MSA"})
# Quadrant scatter in ``af3_metrics/``: (x, y, filename_suffix, filter_key).
# x is typically RNA_length_longest (RNALength); filter_key None = all audit PDBs.
AUDIT_AF3_QUADRANT_SCATTER_SPECS = (
    ("RNA_length_longest", "RNA_Shannon_Entropy_K2", "", None),
    ("RNA_length_longest", "RNA_MFE_Value", "", None),
    ("RNA_TM", "af3_rna_pTM", "_has_RNA_MSA", "has_RNA_MSA"),
    ("RNA_length_longest", "RNA_MFE_Value", "_RBP_PDB_ge1", "rbp_pdb_ge1"),
    ("RNA_length_longest", "RNA_MFE_Value", "_RBP_PDB_eq0", "rbp_pdb_eq0"),
    ("RNA_length_longest", "RNA_Shannon_Entropy_K2", "_RBP_PDB_ge1", "rbp_pdb_ge1"),
    ("RNA_length_longest", "RNA_Shannon_Entropy_K2", "_RBP_PDB_eq0", "rbp_pdb_eq0"),
    ("RNA_length_longest", "RNA_MFE_Value", "_RNAChain_gt1", "rna_chains_gt1"),
    ("RNA_length_longest", "RNA_MFE_Value", "_RNAChain_eq1", "rna_chains_eq1"),
    ("RNA_length_longest", "RNA_Shannon_Entropy_K2", "_RNAChain_gt1", "rna_chains_gt1"),
    ("RNA_length_longest", "RNA_Shannon_Entropy_K2", "_RNAChain_eq1", "rna_chains_eq1"),
)
AUDIT_AF3_QUADRANT_SCATTER_RNA_LENGTH_MAX = 200
AUDIT_AF3_QUADRANT_SCATTER_FILTER_TITLES = {
    "has_RNA_MSA": "has_RNA_MSA=1",
    "rbp_pdb_ge1": "RBP_PDB_Templates≥1",
    "rbp_pdb_eq0": "RBP_PDB_Templates=0",
    "rna_chains_gt1": "RNAChain_Count>1",
    "rna_chains_eq1": "RNAChain_Count=1",
}
AUDIT_QUADRANT_RNA_FEATURE_BREAKDOWN_METRICS = (
    "RNA_Shannon_Entropy_K2",
    "RNA_MFE_Value",
    "af3_rna_pLDDT_avg",
)
_AUDIT_RNA_BREAKDOWN_METRIC_SHORT_LABELS = {
    "RNA_Shannon_Entropy_K2": "Shannon",
    "RNA_MFE_Value": "MFE",
    "af3_rna_pLDDT_avg": "pLDDT",
}
_AUDIT_RNA_INFLUENCE_FILTER_LABELS = {
    "rbp_pdb_eq0": "RBP_PDB_Templates = 0",
    "rbp_pdb_eq1": "RBP_PDB_Templates = 1",
    "rbp_pdb_gt1": "RBP_PDB_Templates > 1",
    "rbp_pdb_ge1": "RBP_PDB_Templates ≥ 1",
    "rna_chains_eq1": "RNAChain_Count = 1",
    "rna_chains_gt1": "RNAChain_Count > 1",
    "has_RNA_MSA": "has_RNA_MSA = 1",
    "has_RNA_MSA_0": "has_RNA_MSA = 0",
    "len_0_40": "RNALength_longest 0–40",
    "len_41_100": "RNALength_longest 41–100",
    "len_gt_100": "RNALength_longest > 100",
}
_AUDIT_RNA_INFLUENCE_COMBO_DIMENSIONS = (
    ("rbp_pdb_eq0", "rbp_pdb_eq1", "rbp_pdb_gt1"),
    ("rna_chains_eq1", "rna_chains_gt1"),
    ("has_RNA_MSA_0", "has_RNA_MSA"),
    ("len_0_40", "len_41_100", "len_gt_100"),
)
AUDIT_AF3_PLOT_ENTITY_TIER_SPECS = (
    ("final", "final_tier", "Final complex tier (Interface + Protein + Protein-RNA; RNA excluded)"),
    ("Interface", "Interface", "Interface entity tier"),
    ("RNA", "RNA", "RNA entity tier"),
    ("Protein", "Protein", "Protein entity tier"),
    ("Protein-RNA", "Protein-RNA", "Protein-RNA entity tier"),
)
AUDIT_DOCKQ_ABC_BIN_SUFFIXES = ("_a", "_b", "_c")
AUDIT_DOCKQ_ABC_ROW_BIN_TITLES = (
    "RBP_DockQ ≥ 0.8 (a)",
    "0.23 ≤ RBP_DockQ < 0.8 (b)",
    "RBP_DockQ < 0.23 (c)",
)
AUDIT_BIN_COLUMN_TITLES = (
    "DockQ ≥ 0.8",
    "0.23≤DQ<0.8",
    "DQ < 0.23",
    "DQ fail\nCRMSD <10Å",
    "DQ fail\nCRMSD ≥10Å",
)

AUDIT_PROFILE_AF3_METRICS = (
    ("af3_rna_pTM", "af3_rna_pTM\n(longest RNA)"),
    ("af3_protein_pTM", "af3_protein_pTM\n(longest prot.)"),
    ("af3_rna_ipTM", "af3_rna_ipTM\n(longest RNA)"),
    ("af3_protein_ipTM", "af3_protein_ipTM\n(longest prot.)"),
    ("af3_rna_pLDDT_avg", "af3_rna_pLDDT_avg"),
    ("af3_protein_pLDDT_avg", "af3_protein_pLDDT_avg"),
    ("af3_global_pae_avg", "af3_global_pae_avg"),
    ("af3_chain_pair_pae_min", "af3_chain_pair_pae_min"),
    ("af3_ranking_score", "af3_ranking_score"),
)

# Core audit heatmap columns: motif/match scores + engineered features + structure / interface metrics.
AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS = (
    ("RNAmotif", "RNAmotif"),
    ("AAmatch_score", "AAmatch_score"),
    ("RNAmatch", "RNAmatch"),
    ("RNAChain_Count", "RNAChain_Count"),
    ("ProteinChain_Count", "ProteinChain_Count"),
    ("RNA_length_longest", "RNA length\n(longest chain)"),
    ("Protein_length_longest", "Protein length\n(longest chain)"),
    ("RNAProtein_Ratio", "RNA/Protein\n(L_longest ratio)"),
    ("RNA_chain_load", "RNA:\nchains × L_longest"),
    ("Protein_chain_load", "Protein:\nchains × L_longest"),
    ("Protein_PDB_n", "Protein_PDB_Templates\ncount"),
    ("RBP_PDB_n", "RBP_PDB_Templates\ncount"),
    ("RNA_msa_depth", "RNA MSA depth\n(longest chain)"),
    ("Protein_msa_depth", "Protein MSA depth\n(longest chain)"),
    ("has_RNA_MSA", "has_RNA_MSA"),
    ("has_RBP_Template", "has_RBP_Template"),
    ("has_Ions", "has_Ions"),
    ("exp_pH", "pH"),
    ("Protein_LDDT", "Protein_LDDT"),
    ("RNA_LDDT_allAtom", "RNA_LDDT"),
    ("RNA_TM", "RNA_TM"),
    ("Protein_TM", "Protein_TM"),
    ("Complex_RMSD", "Complex_RMSD"),
    ("Complex_TM", "Complex_TM"),
    ("Complex_LDDT_allAtom", "Complex_LDDT"),
    ("RBP_DockQ", "RBP_DockQ\n(longest RNA)"),
    ("RBP_QS", "RBP_QS\n(longest RNA)"),
    ("RBP_fnat", "RBP_fnat\n(longest RNA)"),
    ("RBP_irmsd", "RBP_irmsd\n(longest RNA)"),
    ("RNA_iLDDT_AllChains", "RNA_iLDDT_AllChains\n(longest RNA)"),
    ("RNA_ICS_AllChains", "RNA_ICS_AllChains\n(longest RNA)"),
    ("RNA_MFE_Value", "RNA_MFE_Value"),
    ("RNA_SequenceComplexity", "RNA_SequenceComplexity"),
    ("RNA_Shannon_Entropy_K2", "RNA_Shannon_Entropy_K2"),
    ("RNA_GC_Content", "RNA_GC_Content"),
)

AUDIT_PROFILE_CORE_METRICS = AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS

AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS = AUDIT_PROFILE_AF3_METRICS

_LEGACY_COLUMN_ALIASES = {
    "RNAmotif": "RNAmotif_score",
    "RNAmatch": "RNAmatch_score",
    "Complex_LDDT_allAtom": "Complex_LDDT",
}

# Backward-compatible alias for score columns only.
_LEGACY_SCORE_COLUMN_ALIASES = _LEGACY_COLUMN_ALIASES

# Stratification keys for motif-bin audit heatmaps (RNAmotif / RNAmatch only).
AUDIT_MOTIF_STRATIFY_COLUMNS = (
    "RNAmotif",
    "RNAmatch",
)


def _audit_resolve_present_columns(present, wanted):
    """SQL columns to SELECT; prefers canonical names, falls back to legacy *_score columns."""
    out = []
    seen = set()
    for col in wanted:
        if col in present and col not in seen:
            out.append(col)
            seen.add(col)
            continue
        legacy = _LEGACY_COLUMN_ALIASES.get(col)
        if legacy and legacy in present and legacy not in seen:
            out.append(legacy)
            seen.add(legacy)
    return out


def _audit_normalize_prediction_row(d):
    """Map legacy DB column names to canonical audit feature keys."""
    for canonical, legacy in _LEGACY_COLUMN_ALIASES.items():
        if legacy in d and canonical not in d:
            d[canonical] = d[legacy]
    return d


def _audit_stratify_column_resolved(present, stratify_by):
    if stratify_by in present:
        return stratify_by
    legacy = _LEGACY_SCORE_COLUMN_ALIASES.get(stratify_by)
    if legacy and legacy in present:
        return legacy
    return None


def _motif_score_bin_index(score):
    """Bin stratification for motif/match scores in [0, 1].

    - 0: score ≤ 0.5
    - 1: 0.5 < score ≤ 0.8
    - 2: 0.8 < score ≤ 1.0
    Values outside (0, 1] are excluded (returns None), except score ≤ 0.5 includes negatives as bin 0.
    """
    if score is None or not np.isfinite(score):
        return None
    xf = float(score)
    if xf <= 0.5:
        return 0
    if xf <= 0.8:
        return 1
    if xf <= 1.0:
        return 2
    return None


def _compute_motif_bin_ternary_effect_sizes(b0, b1, b2, core_tuples, af3_tuples, *, compare_note):
    """Cohen d (bin 2 vs bin 0) and Spearman(metric, ordinal 1,2,3 across three bins)."""
    rows_out = []
    for panel, tuples in (("core", core_tuples), ("af3", af3_tuples)):
        for mkey, mlbl in tuples:
            v0 = b0.get(mkey) or []
            v1 = b1.get(mkey) or []
            v2 = b2.get(mkey) or []
            stat = _cohens_d_mean_c_minus_mean_a(v0, v2)
            x_ord, y_ord = _ordinal_xy_from_abc_buckets(v0, v1, v2)
            rho_ord, rho_p = _spearman_rho_and_p(x_ord, y_ord)
            abs_rho = abs(float(rho_ord)) if rho_ord is not None and np.isfinite(rho_ord) else None
            comp = None
            if stat.get("abs_cohen_d") is not None and abs_rho is not None:
                comp = float(stat["abs_cohen_d"]) * float(abs_rho)
            rows_out.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    "spearman_rho_ord": rho_ord,
                    "spearman_p_ord": rho_p,
                    "abs_spearman_rho_ord": abs_rho,
                    "n_ord_abc": int(x_ord.size),
                    "n_ord_ab": int(x_ord.size),
                    "composite_abs_d_rho": comp,
                    **stat,
                }
            )
    return {
        "compare": compare_note,
        "formula_cohen": "Cohen d = (mean_bin2 − mean_bin0) / s_pooled; bin0 = score ≤ 0.5, bin2 = (0.8, 1].",
        "formula_spearman": "rho_ord = Spearman(metric, ordinal class 1=bin0, 2=bin1, 3=bin2) on pooled finite samples.",
        "rows": rows_out,
    }


def compute_audit_quadrant_outcome_motif_mean_matrix(rna_metrics, select_table, audit, score_column):
    """
    Same 4×5 layout as ``plot_audit_quadrant_outcome_heatmap``, but each cell is the mean
    RNAmotif or RNAmatch over audit-listed predictions in that bin.
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None
    if score_column not in AUDIT_MOTIF_STRATIFY_COLUMNS:
        return None
    try:
        rna_metrics.db.cursor.execute(f"PRAGMA table_info({select_table})")
        present = {row[1] for row in rna_metrics.db.cursor.fetchall()}
    except Exception:
        return None
    if score_column not in present:
        score_col = _audit_stratify_column_resolved(present, score_column)
        if not score_col:
            return None
    else:
        score_col = score_column
    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    def _mean_score_for_audit_recs(recs):
        vals = []
        for rec in recs or []:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            v = _parse_optional_float_metric(drow.get(score_column))
            if np.isfinite(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    mat = np.full((4, 5), np.nan)
    for i, pref in enumerate(_AUDIT_QUAD_PREFIXES):
        mat[i, 0] = _mean_score_for_audit_recs(audit.get(f"{pref}_a"))
        mat[i, 1] = _mean_score_for_audit_recs(audit.get(f"{pref}_b"))
        mat[i, 2] = _mean_score_for_audit_recs(audit.get(f"{pref}_c"))
        mat[i, 3] = _mean_score_for_audit_recs(audit.get(f"{pref}_c_ca"))
        mat[i, 4] = _mean_score_for_audit_recs(audit.get(f"{pref}_c_cb"))
    if not np.any(np.isfinite(mat)):
        return None
    return mat


def compute_motif_bin_profiles(rna_metrics, select_table, stratify_by):
    """
    Mean metrics over all prediction rows split by RNAmotif or RNAmatch:

    - Row 0: score ≤ 0.5
    - Row 1: 0.5 < score ≤ 0.8
    - Row 2: 0.8 < score ≤ 1.0

    Columns: AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS + AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS.
    Effect table: Cohen d (bin 2 vs bin 0) and Spearman vs ordinal 1–3 across bins.
    """
    if stratify_by not in AUDIT_MOTIF_STRATIFY_COLUMNS:
        return None
    row_dicts = _audit_load_prediction_rows_for_motif_bins(rna_metrics, select_table, stratify_by)
    if not row_dicts:
        return None

    core_keys = [k for k, _ in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS]
    af3_keys = [k for k, _ in AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS]
    bucket_core = {k: [[], [], []] for k in core_keys}
    bucket_af3 = {k: [[], [], []] for k in af3_keys}
    prna = [[], [], []]
    pprot = [[], [], []]
    ppdb = [[], [], []]
    rpdb = [[], [], []]

    for d in row_dicts:
        raw_strat = _parse_optional_float_metric(d.get(stratify_by))
        bix = _motif_score_bin_index(raw_strat)
        if bix is None:
            continue
        fv = _audit_extract_motif_bin_metrics(d)
        for k in core_keys:
            val = fv[k]
            if np.isfinite(val):
                bucket_core[k][bix].append(float(val))
        for k in af3_keys:
            val = fv[k]
            if np.isfinite(val):
                bucket_af3[k][bix].append(float(val))
        if np.isfinite(fv["has_RNA_msa"]):
            prna[bix].append(fv["has_RNA_msa"])
        if np.isfinite(fv["has_Protein_msa"]):
            pprot[bix].append(fv["has_Protein_msa"])
        if np.isfinite(fv["has_Protein_PDB"]):
            ppdb[bix].append(fv["has_Protein_PDB"])
        if np.isfinite(fv["has_RBP_PDB"]):
            rpdb[bix].append(fv["has_RBP_PDB"])

    if stratify_by == "RNAmotif":
        row_labels = (
            "RNAmotif ≤ 0.5",
            "0.5 < RNAmotif ≤ 0.8",
            "0.8 < RNAmotif ≤ 1",
        )
        compare_note = (
            "RNAmotif: bins ≤0.5, (0.5, 0.8], (0.8, 1]; Cohen d = bin2 vs bin0; all prediction rows."
        )
    else:
        row_labels = (
            "RNAmatch ≤ 0.5",
            "0.5 < RNAmatch ≤ 0.8",
            "0.8 < RNAmatch ≤ 1",
        )
        compare_note = (
            "RNAmatch: bins ≤0.5, (0.5, 0.8], (0.8, 1]; Cohen d = bin2 vs bin0; all prediction rows."
        )

    n_bins = 3
    core_mat = np.full((n_bins, len(core_keys)), np.nan)
    af3_mat = np.full((n_bins, len(af3_keys)), np.nan)
    frac_rna_msa = np.full(n_bins, np.nan)
    frac_prot_msa = np.full(n_bins, np.nan)
    frac_prot_pdb = np.full(n_bins, np.nan)
    frac_rbp_pdb = np.full(n_bins, np.nan)
    mean_rna_msa_d = np.full(n_bins, np.nan)
    mean_prot_msa_d = np.full(n_bins, np.nan)
    bin_counts = [0, 0, 0]

    for d in row_dicts:
        raw_strat = _parse_optional_float_metric(d.get(stratify_by))
        bix = _motif_score_bin_index(raw_strat)
        if bix is None:
            continue
        bin_counts[bix] += 1

    for bi in (0, 1, 2):
        for ci, ck in enumerate(core_keys):
            vals = bucket_core[ck][bi]
            core_mat[bi, ci] = float(np.mean(vals)) if vals else np.nan
        for ai, ak in enumerate(af3_keys):
            vals = bucket_af3[ak][bi]
            af3_mat[bi, ai] = float(np.mean(vals)) if vals else np.nan
        frac_rna_msa[bi] = float(np.mean(prna[bi])) if prna[bi] else np.nan
        frac_prot_msa[bi] = float(np.mean(pprot[bi])) if pprot[bi] else np.nan
        frac_prot_pdb[bi] = float(np.mean(ppdb[bi])) if ppdb[bi] else np.nan
        frac_rbp_pdb[bi] = float(np.mean(rpdb[bi])) if rpdb[bi] else np.nan
        rdep = bucket_core["RNA_msa_depth"][bi]
        pdep = bucket_core["Protein_msa_depth"][bi]
        mean_rna_msa_d[bi] = float(np.mean(rdep)) if rdep else np.nan
        mean_prot_msa_d[bi] = float(np.mean(pdep)) if pdep else np.nan

    b0c = {k: bucket_core[k][0] for k in core_keys}
    b1c = {k: bucket_core[k][1] for k in core_keys}
    b2c = {k: bucket_core[k][2] for k in core_keys}
    b0a = {k: bucket_af3[k][0] for k in af3_keys}
    b1a = {k: bucket_af3[k][1] for k in af3_keys}
    b2a = {k: bucket_af3[k][2] for k in af3_keys}
    cohen_core = _compute_motif_bin_ternary_effect_sizes(
        b0c, b1c, b2c, AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS, (), compare_note=compare_note
    )
    cohen_af3 = _compute_motif_bin_ternary_effect_sizes(
        b0a, b1a, b2a, (), AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS, compare_note=compare_note
    )
    cohen_rows = (cohen_core.get("rows") or []) + (cohen_af3.get("rows") or [])
    cohen_rows.sort(
        key=lambda r: (
            r["composite_abs_d_rho"] is None,
            -(r["composite_abs_d_rho"] if r["composite_abs_d_rho"] is not None else 0.0),
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            -(r["abs_spearman_rho_ord"] if r["abs_spearman_rho_ord"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(cohen_rows, start=1):
        r["rank"] = i
    cohen_d_effect_sizes = {
        "compare": compare_note,
        "formula_cohen": cohen_core.get("formula_cohen", ""),
        "formula_spearman": cohen_core.get("formula_spearman", ""),
        "rows": cohen_rows,
    }

    if sum(bin_counts) == 0:
        return None

    return {
        "stratify_by": stratify_by,
        "row_labels": list(row_labels),
        "bin_counts": bin_counts,
        "core_mat": core_mat,
        "core_col_labels": [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS],
        "af3_mat": af3_mat,
        "af3_col_labels": [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS],
        "frac_rna_msa": frac_rna_msa,
        "frac_prot_msa": frac_prot_msa,
        "frac_prot_pdb": frac_prot_pdb,
        "frac_rbp_pdb": frac_rbp_pdb,
        "mean_rna_msa_depth": mean_rna_msa_d,
        "mean_prot_msa_depth": mean_prot_msa_d,
        "cohen_d_effect_sizes": cohen_d_effect_sizes,
    }


AUDIT_CRMSD_BIN_ROW_LABELS = (
    "CRMSD < 2 Å",
    "2 ≤ CRMSD < 10 Å",
    "CRMSD ≥ 10 Å",
)


def _crmsd_bin_index(crmsd):
    """Bin stratification for Complex_RMSD (Å): <2, [2, 10), ≥10."""
    if crmsd is None or not np.isfinite(crmsd):
        return None
    x = float(crmsd)
    if x < 2.0:
        return 0
    if x < 10.0:
        return 1
    return 2


def compute_crmsd_bin_profiles(rna_metrics, select_table):
    """
    Mean metrics over all prediction rows with finite Complex_RMSD, split by CRMSD bin:

    - Row 0: CRMSD < 2 Å
    - Row 1: 2 ≤ CRMSD < 10 Å
    - Row 2: CRMSD ≥ 10 Å

    Columns: AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS + AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS (combined).
    """
    row_dicts = _audit_load_prediction_rows_for_motif_bins(
        rna_metrics, select_table, "RNAmotif"
    )
    if not row_dicts:
        return None

    core_keys = [k for k, _ in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS]
    af3_keys = [k for k, _ in AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS]
    bucket_core = {k: [[], [], []] for k in core_keys}
    bucket_af3 = {k: [[], [], []] for k in af3_keys}
    compare_note = (
        "Complex_RMSD bins: <2 Å, [2, 10) Å, ≥10 Å; Cohen d = bin2 vs bin0; "
        "all prediction rows with finite Complex_RMSD."
    )

    for d in row_dicts:
        fv = _audit_extract_motif_bin_metrics(d)
        bix = _crmsd_bin_index(fv.get("Complex_RMSD"))
        if bix is None:
            continue
        for k in core_keys:
            val = fv[k]
            if np.isfinite(val):
                bucket_core[k][bix].append(float(val))
        for k in af3_keys:
            val = fv[k]
            if np.isfinite(val):
                bucket_af3[k][bix].append(float(val))

    n_bins = 3
    core_mat = np.full((n_bins, len(core_keys)), np.nan)
    af3_mat = np.full((n_bins, len(af3_keys)), np.nan)
    bin_counts = [0, 0, 0]

    for d in row_dicts:
        fv = _audit_extract_motif_bin_metrics(d)
        bix = _crmsd_bin_index(fv.get("Complex_RMSD"))
        if bix is None:
            continue
        bin_counts[bix] += 1

    for bi in (0, 1, 2):
        for ci, ck in enumerate(core_keys):
            vals = bucket_core[ck][bi]
            core_mat[bi, ci] = float(np.mean(vals)) if vals else np.nan
        for ai, ak in enumerate(af3_keys):
            vals = bucket_af3[ak][bi]
            af3_mat[bi, ai] = float(np.mean(vals)) if vals else np.nan

    if sum(bin_counts) == 0:
        return None

    b0c = {k: bucket_core[k][0] for k in core_keys}
    b1c = {k: bucket_core[k][1] for k in core_keys}
    b2c = {k: bucket_core[k][2] for k in core_keys}
    b0a = {k: bucket_af3[k][0] for k in af3_keys}
    b1a = {k: bucket_af3[k][1] for k in af3_keys}
    b2a = {k: bucket_af3[k][2] for k in af3_keys}
    cohen_core = _compute_motif_bin_ternary_effect_sizes(
        b0c, b1c, b2c, AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS, (), compare_note=compare_note
    )
    cohen_af3 = _compute_motif_bin_ternary_effect_sizes(
        b0a, b1a, b2a, (), AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS, compare_note=compare_note
    )
    cohen_rows = (cohen_core.get("rows") or []) + (cohen_af3.get("rows") or [])
    cohen_rows.sort(
        key=lambda r: (
            r["composite_abs_d_rho"] is None,
            -(r["composite_abs_d_rho"] if r["composite_abs_d_rho"] is not None else 0.0),
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            -(r["abs_spearman_rho_ord"] if r["abs_spearman_rho_ord"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(cohen_rows, start=1):
        r["rank"] = i
    cohen_d_effect_sizes = {
        "compare": compare_note,
        "formula_cohen": cohen_core.get("formula_cohen", ""),
        "formula_spearman": cohen_core.get("formula_spearman", ""),
        "rows": cohen_rows,
    }

    combined_col_labels = (
        [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS]
        + [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS]
    )
    combined_mat = np.hstack([core_mat, af3_mat])

    return {
        "stratify_by": "Complex_RMSD",
        "row_labels": list(AUDIT_CRMSD_BIN_ROW_LABELS),
        "bin_counts": bin_counts,
        "core_mat": core_mat,
        "core_col_labels": [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS],
        "af3_mat": af3_mat,
        "af3_col_labels": [lbl for _, lbl in AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS],
        "combined_mat": combined_mat,
        "combined_col_labels": combined_col_labels,
        "n_core_cols": len(core_keys),
        "cohen_d_effect_sizes": cohen_d_effect_sizes,
    }


_AUDIT_MOTIF_QUADRANT_OUTCOME_COLS = (
    "DockQ≥0.8",
    "0.23≤DQ<0.8",
    "DQ<0.23",
    "DQ fail CRMSD<10Å",
    "DQ fail CRMSD≥10Å",
)


def compute_motif_score_table_summaries(rna_metrics, select_table):
    """Distribution of RNAmotif and RNAmatch over all prediction rows."""
    row_dicts = _audit_load_prediction_rows_for_motif_bins(
        rna_metrics, select_table, "RNAmotif"
    )
    if not row_dicts:
        return None

    summaries = {}
    for strat in AUDIT_MOTIF_STRATIFY_COLUMNS:
        vals = [_parse_optional_float_metric(d.get(strat)) for d in row_dicts]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if not vals:
            summaries[strat] = None
            continue
        arr = np.asarray(vals, dtype=float)
        bin_counts = [0, 0, 0]
        for v in arr:
            bix = _motif_score_bin_index(v)
            if bix is not None:
                bin_counts[bix] += 1
        n = int(arr.size)
        summaries[strat] = {
            "n_finite": n,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "bin_n_le_0_5": bin_counts[0],
            "bin_n_mid": bin_counts[1],
            "bin_n_hi": bin_counts[2],
            "bin_frac_le_0_5": bin_counts[0] / n,
            "bin_frac_mid": bin_counts[1] / n,
            "bin_frac_hi": bin_counts[2] / n,
        }
    return summaries


def _motif_quadrant_mean_matrix_to_rows(mat):
    """4×5 quadrant × outcome matrix → list of dicts for txt export."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_TICKLABELS
    except Exception:
        return []
    if mat is None or mat.size == 0:
        return []
    rows = []
    quads = list(_AUDIT_QUAD_TICKLABELS)
    cols = _AUDIT_MOTIF_QUADRANT_OUTCOME_COLS
    for i, qlab in enumerate(quads):
        for j, clab in enumerate(cols):
            v = mat[i, j] if i < mat.shape[0] and j < mat.shape[1] else np.nan
            rows.append(
                {
                    "quadrant": qlab,
                    "outcome_col": clab,
                    "mean_score": float(v) if np.isfinite(v) else None,
                }
            )
    return rows


def _audit_merge_exp_ions_ph_into_row_dicts(rna_metrics, row_dicts):
    """Copy Ions / exp_pH from exp table onto each prediction row dict (may repeat exp_db_id)."""
    if not row_dicts:
        return
    by_eid = {}
    for d in row_dicts:
        eid = str(d.get("exp_db_id", "")).strip()
        if eid:
            by_eid[eid] = d
    _audit_attach_exp_ions_ph(rna_metrics, by_eid)
    for d in row_dicts:
        eid = str(d.get("exp_db_id", "")).strip()
        if not eid:
            continue
        src = by_eid.get(eid)
        if not src:
            continue
        if "Ions" in src:
            d["Ions"] = src.get("Ions")
        if "exp_pH" in src:
            d["exp_pH"] = src.get("exp_pH")


def _audit_load_prediction_rows_for_motif_bins(rna_metrics, select_table, stratify_by):
    """All rows from prediction table with columns needed for motif-bin profiles."""
    try:
        rna_metrics.db.cursor.execute(f"PRAGMA table_info({select_table})")
        present = {row[1] for row in rna_metrics.db.cursor.fetchall()}
    except Exception:
        return None
    avail = _audit_resolve_present_columns(present, _MOTIF_BIN_PROFILE_SQL_COLUMNS)
    if "exp_db_id" not in avail:
        return None
    strat_col = _audit_stratify_column_resolved(present, stratify_by)
    if not strat_col:
        return None
    if strat_col not in avail:
        avail = list(avail) + [strat_col]
    try:
        rows = rna_metrics.get_filtered_values(select_table, avail)
    except Exception:
        return None
    row_dicts = [_audit_normalize_prediction_row(dict(zip(avail, tup))) for tup in rows]
    _audit_merge_exp_ions_ph_into_row_dicts(rna_metrics, row_dicts)
    return row_dicts


def _audit_extract_motif_bin_metrics(rowdict):
    """Engineered features + structure + RBP / RNA interface metrics (longest RNA chain)."""
    feats = _audit_extract_features_from_row(rowdict)
    feats["Protein_LDDT"] = _parse_optional_float_metric(rowdict.get("Protein_LDDT"))
    feats["RNA_LDDT_allAtom"] = _parse_optional_float_metric(
        rowdict.get("RNA_LDDT_allAtom")
        if rowdict.get("RNA_LDDT_allAtom") is not None
        else rowdict.get("RNA_LDDT")
    )
    feats["RNA_TM"] = _parse_optional_float_metric(rowdict.get("RNA_TM"))
    feats["Protein_TM"] = _parse_optional_float_metric(rowdict.get("Protein_TM"))
    feats["Complex_RMSD"] = _parse_optional_float_metric(rowdict.get("Complex_RMSD"))
    feats["Complex_TM"] = _parse_optional_float_metric(rowdict.get("Complex_TM"))
    feats["Complex_LDDT_allAtom"] = _parse_optional_float_metric(
        rowdict.get("Complex_LDDT_allAtom")
        if rowdict.get("Complex_LDDT_allAtom") is not None
        else rowdict.get("Complex_LDDT")
    )
    for col in (
        "RNA_MFE_Value",
        "RNA_SequenceComplexity",
        "RNA_Shannon_Entropy_K2",
        "RNA_GC_Content",
    ):
        feats[col] = _parse_optional_float_metric(rowdict.get(col))

    chain_ids = rowdict.get("RNAChainIDs")
    rna_len_raw = rowdict.get("RNALength")
    prot_ids = rowdict.get("ProteinChainIDs")
    longest_rna_idx, longest_rna_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)

    for col in ("RBP_DockQ", "RBP_QS", "RBP_fnat", "RBP_irmsd"):
        feats[col] = _extract_rbp_metric_for_longest_rna(
            rowdict.get(col), longest_rna_chain, prot_ids
        )
    feats["RNA_iLDDT_AllChains"] = _extract_longest_rna_matrix_score(
        rowdict.get("RNA_iLDDT_AllChains"), longest_rna_idx
    )
    feats["RNA_ICS_AllChains"] = _extract_longest_rna_matrix_score(
        rowdict.get("RNA_ICS_AllChains"), longest_rna_idx
    )
    feats["af3_rna_pLDDT_avg"] = _audit_af3_entity_plddt_avg(rowdict, "af3_rna_pLDDT_avg")
    feats["af3_protein_pLDDT_avg"] = _audit_af3_entity_plddt_avg(rowdict, "af3_protein_pLDDT_avg")
    return feats


def _audit_extract_features_from_row(rowdict):
    """Single-row features aligned to longest RNA / longest protein chains."""
    chain_ids = rowdict.get("RNAChainIDs")
    rna_len_raw = rowdict.get("RNALength")
    prot_ids = rowdict.get("ProteinChainIDs")
    rna_chains_list = _as_chain_list(chain_ids)
    prot_chains_list = _as_chain_list(prot_ids)
    longest_rna_idx, longest_rna_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)
    longest_prot_idx, longest_prot_chain = _longest_protein_index_and_chain(
        prot_ids, rowdict.get("ProteinLength")
    )

    rna_len_longest = _rna_length_at_index(rna_len_raw, longest_rna_idx)
    prot_len_longest = _rna_length_at_index(rowdict.get("ProteinLength"), longest_prot_idx)

    rna_chain_ct = float(len(rna_chains_list))
    prot_chain_ct = float(len(prot_chains_list))

    rna_prot_ratio = np.nan
    if np.isfinite(rna_len_longest) and np.isfinite(prot_len_longest) and prot_len_longest > 0:
        rna_prot_ratio = float(rna_len_longest / prot_len_longest)

    rna_load = np.nan
    if np.isfinite(rna_chain_ct) and np.isfinite(rna_len_longest):
        rna_load = float(rna_chain_ct * rna_len_longest)

    prot_load = np.nan
    if np.isfinite(prot_chain_ct) and np.isfinite(prot_len_longest):
        prot_load = float(prot_chain_ct * prot_len_longest)

    feats = {
        "RNAmotif": _scalar_pred_float(
            rowdict.get("RNAmotif") if rowdict.get("RNAmotif") is not None else rowdict.get("RNAmotif_score")
        ),
        "AAmatch_score": _scalar_pred_float(rowdict.get("AAmatch_score")),
        "RNAmatch": _scalar_pred_float(
            rowdict.get("RNAmatch") if rowdict.get("RNAmatch") is not None else rowdict.get("RNAmatch_score")
        ),
        "RNAChain_Count": rna_chain_ct,
        "ProteinChain_Count": prot_chain_ct,
        "RNA_length_longest": rna_len_longest,
        "Protein_length_longest": prot_len_longest,
        "RNAProtein_Ratio": rna_prot_ratio,
        "RNA_chain_load": rna_load,
        "Protein_chain_load": prot_load,
        "Protein_PDB_n": float(_template_entry_count(rowdict.get("Protein_PDB_Templates"))),
        "RBP_PDB_n": float(_template_entry_count(rowdict.get("RBP_PDB_Templates"))),
        "RNA_msa_depth": np.nan,
        "Protein_msa_depth": np.nan,
        "has_RNA_msa": np.nan,
        "has_Protein_msa": np.nan,
        "has_Protein_PDB": np.nan,
        "has_RBP_PDB": np.nan,
        "has_RNA_MSA": np.nan,
        "has_RBP_Template": np.nan,
        "has_Ions": np.nan,
        "exp_pH": np.nan,
        "af3_rna_pTM": np.nan,
        "af3_protein_pTM": np.nan,
        "af3_rna_ipTM": np.nan,
        "af3_protein_ipTM": np.nan,
        "af3_rna_pLDDT_avg": np.nan,
        "af3_protein_pLDDT_avg": np.nan,
        "af3_global_pae_avg": np.nan,
        "af3_chain_pair_pae_min": np.nan,
        "af3_ranking_score": np.nan,
    }

    rna_dep = _effective_msa_depth_for_chain(rowdict.get("RNA_msa_size"), longest_rna_chain)
    prot_dep = _effective_msa_depth_for_chain(rowdict.get("Protein_msa_size"), longest_prot_chain)
    feats["RNA_msa_depth"] = rna_dep
    feats["Protein_msa_depth"] = prot_dep
    feats["has_RNA_msa"] = (
        1.0 if (np.isfinite(rna_dep) and rna_dep > 0.0) else 0.0 if np.isfinite(rna_dep) else np.nan
    )
    feats["has_RNA_MSA"] = feats["has_RNA_msa"]
    feats["has_Protein_msa"] = (
        1.0 if (np.isfinite(prot_dep) and prot_dep > 0.0) else 0.0 if np.isfinite(prot_dep) else np.nan
    )

    pn = int(_template_entry_count(rowdict.get("Protein_PDB_Templates")))
    rn = int(_template_entry_count(rowdict.get("RBP_PDB_Templates")))
    feats["has_Protein_PDB"] = 1.0 if pn > 0 else 0.0
    feats["has_RBP_PDB"] = 1.0 if rn > 0 else 0.0
    feats["has_RBP_Template"] = 1.0 if rn > 0 else 0.0

    if longest_rna_idx is not None:
        feats["af3_rna_pTM"] = _af3_rna_metric_at_chain_index(rowdict.get("af3_rna_pTM"), longest_rna_idx)
        feats["af3_rna_ipTM"] = _af3_rna_metric_at_chain_index(rowdict.get("af3_rna_ipTM"), longest_rna_idx)
    if longest_prot_idx is not None:
        feats["af3_protein_pTM"] = _af3_rna_metric_at_chain_index(rowdict.get("af3_protein_pTM"), longest_prot_idx)
        feats["af3_protein_ipTM"] = _af3_rna_metric_at_chain_index(rowdict.get("af3_protein_ipTM"), longest_prot_idx)

    feats["af3_rna_pLDDT_avg"] = _audit_af3_entity_plddt_avg(rowdict, "af3_rna_pLDDT_avg")
    feats["af3_protein_pLDDT_avg"] = _audit_af3_entity_plddt_avg(rowdict, "af3_protein_pLDDT_avg")
    feats["af3_global_pae_avg"] = _parse_optional_float_metric(rowdict.get("af3_global_pae_avg"))
    feats["af3_chain_pair_pae_min"] = _parse_optional_float_metric(rowdict.get("af3_chain_pair_pae_min"))
    feats["af3_ranking_score"] = _parse_optional_float_metric(rowdict.get("af3_ranking_score"))

    feats["has_Ions"] = _audit_sum_ions_n_from_cell(rowdict.get("Ions"))
    feats["exp_pH"] = _parse_optional_float_metric(rowdict.get("exp_pH"))

    return feats


def _audit_sum_ions_n_from_cell(raw):
    """Sum of ``n`` fields in exp-table ``Ions`` list-of-dicts; ``[]`` → 0; missing/unparseable → nan."""
    if raw is None:
        return np.nan
    if isinstance(raw, str) and not raw.strip():
        return np.nan
    try:
        group = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    if not group:
        return 0.0
    total = 0
    for x in group:
        if isinstance(x, dict) and "n" in x:
            try:
                total += int(x["n"])
            except (TypeError, ValueError):
                pass
    return float(total)


def _audit_attach_exp_ions_ph(rna_metrics, by_eid):
    """Attach Ions and exp_pH from reference (exp_*) table onto prediction row dicts (keyed by exp_db_id)."""
    if not by_eid:
        return
    cfg = StartConfig()
    ref_table = cfg.ref_table
    if not ref_table:
        return
    try:
        cur = rna_metrics.db.cursor
        cur.execute(f"PRAGMA table_info({ref_table})")
        ref_cols = {row[1] for row in cur.fetchall()}
    except Exception:
        return
    want = []
    if "Ions" in ref_cols:
        want.append("Ions")
    if "exp_pH" in ref_cols:
        want.append("exp_pH")
    if not want:
        return
    ids = list(by_eid.keys())
    cols_sql = ", ".join(["PDBId"] + want)
    exp_map = {}
    chunk = 450
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        if not part:
            continue
        qmarks = ",".join("?" * len(part))
        try:
            cur.execute(f"SELECT {cols_sql} FROM {ref_table} WHERE PDBId IN ({qmarks})", part)
            for row in cur.fetchall():
                pdbid = str(row[0]).strip()
                rec = {want[j]: row[j + 1] for j in range(len(want))}
                exp_map[pdbid] = rec
        except Exception:
            continue
    for eid, d in by_eid.items():
        ek = str(eid).strip()
        rec = exp_map.get(ek)
        if not rec:
            continue
        if "Ions" in want:
            d["Ions"] = rec.get("Ions")
        if "exp_pH" in want:
            d["exp_pH"] = rec.get("exp_pH")


_AUDIT_INTERFACE_PROFILE_COLUMNS = (
    "exp_db_id",
    "RNAChainIDs",
    "RNALength",
    "ProteinChainIDs",
    "ProteinLength",
    "RNAmotif",
    "AAmatch_score",
    "RNAmatch",
    "Protein_PDB_Templates",
    "RBP_PDB_Templates",
    "RNA_msa_size",
    "Protein_msa_size",
    "af3_rna_pTM",
    "af3_protein_pTM",
    "af3_rna_ipTM",
    "af3_protein_ipTM",
    "af3_rna_pLDDT_avg",
    "af3_protein_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_ranking_score",
    "Protein_LDDT",
    "RNA_LDDT_allAtom",
    "RNA_TM",
    "Protein_TM",
    "Complex_RMSD",
    "Complex_TM",
    "Complex_LDDT_allAtom",
    "RBP_DockQ",
    "RBP_QS",
    "RBP_fnat",
    "RBP_irmsd",
    "RNA_TM_AllChains",
    "RNA_iLDDT_AllChains",
    "RNA_ICS_AllChains",
    "RNA_MFE_Value",
    "RNA_SequenceComplexity",
    "RNA_Shannon_Entropy_K2",
    "RNA_GC_Content",
)

_MOTIF_BIN_PROFILE_SQL_COLUMNS = _AUDIT_INTERFACE_PROFILE_COLUMNS


def _audit_load_prediction_by_eid(rna_metrics, select_table):
    """Load prediction-table rows needed for audit bin profiles / Cohen d; exp_db_id → column dict."""
    try:
        rna_metrics.db.cursor.execute(f"PRAGMA table_info({select_table})")
        present = {row[1] for row in rna_metrics.db.cursor.fetchall()}
    except Exception:
        return None

    avail = _audit_resolve_present_columns(present, _AUDIT_INTERFACE_PROFILE_COLUMNS)
    if "exp_db_id" not in avail:
        return None

    try:
        rows = rna_metrics.get_filtered_values(select_table, avail)
    except Exception:
        return None

    by_eid = {}
    for tup in rows:
        d = _audit_normalize_prediction_row(dict(zip(avail, tup)))
        eid = str(d.get("exp_db_id", "")).strip()
        if eid:
            by_eid[eid] = d
    _audit_attach_exp_ions_ph(rna_metrics, by_eid)
    return by_eid


def _cohens_d_mean_c_minus_mean_a(vals_a, vals_c):
    """
    Cohen's d for independent samples, numerator (mean_C - mean_A) / pooled SD.
    Pooled SD uses unbiased variance (ddof=1) per group. Returns dict with effect and diagnostics.
    """
    a = np.asarray(vals_a, dtype=float)
    c = np.asarray(vals_c, dtype=float)
    a = a[np.isfinite(a)]
    c = c[np.isfinite(c)]
    n_a, n_c = int(a.size), int(c.size)
    out = {
        "cohen_d": None,
        "mean_a": None,
        "mean_c": None,
        "n_a": n_a,
        "n_c": n_c,
        "pooled_sd": None,
        "abs_cohen_d": None,
    }
    if n_a < 1 or n_c < 1:
        return out
    mean_a = float(np.mean(a))
    mean_c = float(np.mean(c))
    out["mean_a"] = mean_a
    out["mean_c"] = mean_c
    if n_a < 2 or n_c < 2:
        return out
    var_a = float(np.var(a, ddof=1))
    var_c = float(np.var(c, ddof=1))
    dof = n_a + n_c - 2
    if dof <= 0:
        return out
    pooled_var = ((n_a - 1) * var_a + (n_c - 1) * var_c) / dof
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        if abs(mean_c - mean_a) < 1e-12:
            out["cohen_d"] = 0.0
            out["pooled_sd"] = 0.0
            out["abs_cohen_d"] = 0.0
        return out
    sp = float(np.sqrt(pooled_var))
    out["pooled_sd"] = sp
    d = (mean_c - mean_a) / sp
    out["cohen_d"] = float(d)
    out["abs_cohen_d"] = abs(float(d))
    return out


def _spearman_rho_and_p(x, y):
    """Spearman correlation; returns (rho, p_value) or (None, None)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None, None
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None, None
    try:
        from scipy.stats import spearmanr

        r, p = spearmanr(x, y)
        if not np.isfinite(r):
            return None, None
        return float(r), float(p) if np.isfinite(p) else None
    except Exception:
        return None, None


def _ordinal_xy_from_abc_buckets(vals_a, vals_b, vals_c):
    """Concatenate class-specific value lists with ordinal labels 1,2,3."""
    xa = [float(v) for v in vals_a if np.isfinite(v)]
    xb = [float(v) for v in vals_b if np.isfinite(v)]
    xc = [float(v) for v in vals_c if np.isfinite(v)]
    x = np.asarray(xa + xb + xc, dtype=float)
    y = np.concatenate(
        [np.ones(len(xa), dtype=float), 2.0 * np.ones(len(xb), dtype=float), 3.0 * np.ones(len(xc), dtype=float)]
    )
    return x, y


def _quad_spearman_note(pooled_rho, quad_rhos):
    """Short robustness note comparing per-quadrant Spearman signs to pooled rho."""
    finite = [r for r in quad_rhos if r is not None and np.isfinite(r)]
    if len(finite) < 2:
        return "insufficient finite quadrant rhos"
    if pooled_rho is None or not np.isfinite(pooled_rho):
        return f"{len(finite)}/4 quadrants finite rho"
    ps = int(np.sign(pooled_rho))
    if ps == 0:
        return f"pooled rho~0; {len(finite)}/4 quad rhos finite"
    agree = sum(1 for r in finite if int(np.sign(r)) == ps or int(np.sign(r)) == 0)
    if agree == len(finite):
        return f"sign match pooled ({len(finite)}/4 quad)"
    return f"mixed signs vs pooled ({agree}/{len(finite)} quad match)"


def compute_audit_interface_class_cohen_d(rna_metrics, select_table, audit):
    """
    Complex-level effect summaries for audit heatmap features (core + AF3):

    - Cohen's d: class (a) RBP_DockQ >= 0.8 vs (c) RBP_DockQ < 0.23, pooled over quadrants.
    - Spearman rho: metric vs ordinal DockQ class with (a)=1, (b)=2, (c)=3, pooled over quadrants.
    - Composite (primary rank): |d| * |rho| when both finite.

    Per-quadrant Spearman rhos are computed for validation only (not used for ranking).
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    core_keys = [k for k, _ in AUDIT_PROFILE_CORE_METRICS]
    af3_keys = [k for k, _ in AUDIT_PROFILE_AF3_METRICS]
    all_keys = core_keys + af3_keys

    bucket_a = {k: [] for k in all_keys}
    bucket_b = {k: [] for k in all_keys}
    bucket_c = {k: [] for k in all_keys}
    quad_buckets = [
        {"a": {k: [] for k in all_keys}, "b": {k: [] for k in all_keys}, "c": {k: [] for k in all_keys}}
        for _ in _AUDIT_QUAD_PREFIXES
    ]

    for qi, pref in enumerate(_AUDIT_QUAD_PREFIXES):
        qb = quad_buckets[qi]
        for rec in audit.get(f"{pref}_a") or []:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            for k in all_keys:
                val = fv[k]
                if np.isfinite(val):
                    v = float(val)
                    bucket_a[k].append(v)
                    qb["a"][k].append(v)
        for rec in audit.get(f"{pref}_b") or []:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            for k in all_keys:
                val = fv[k]
                if np.isfinite(val):
                    v = float(val)
                    bucket_b[k].append(v)
                    qb["b"][k].append(v)
        for rec in audit.get(f"{pref}_c") or []:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            for k in all_keys:
                val = fv[k]
                if np.isfinite(val):
                    v = float(val)
                    bucket_c[k].append(v)
                    qb["c"][k].append(v)

    rows_out = []
    for panel, tuples in (
        ("core", AUDIT_PROFILE_CORE_METRICS),
        ("af3", AUDIT_PROFILE_AF3_METRICS),
    ):
        for mkey, mlbl in tuples:
            stat = _cohens_d_mean_c_minus_mean_a(bucket_a[mkey], bucket_c[mkey])
            x_ord, y_ord = _ordinal_xy_from_abc_buckets(bucket_a[mkey], bucket_b[mkey], bucket_c[mkey])
            rho_ord, rho_p = _spearman_rho_and_p(x_ord, y_ord)
            abs_rho = abs(float(rho_ord)) if rho_ord is not None and np.isfinite(rho_ord) else None

            quad_rhos = []
            for qb in quad_buckets:
                qx, qy = _ordinal_xy_from_abc_buckets(qb["a"][mkey], qb["b"][mkey], qb["c"][mkey])
                rq, _ = _spearman_rho_and_p(qx, qy)
                quad_rhos.append(rq)

            comp = None
            if stat.get("abs_cohen_d") is not None and abs_rho is not None:
                comp = float(stat["abs_cohen_d"]) * float(abs_rho)

            rows_out.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    "spearman_rho_ord": rho_ord,
                    "spearman_p_ord": rho_p,
                    "abs_spearman_rho_ord": abs_rho,
                    "n_ord_abc": int(x_ord.size),
                    "composite_abs_d_rho": comp,
                    "rho_quad_i": quad_rhos[0],
                    "rho_quad_ii": quad_rhos[1],
                    "rho_quad_iii": quad_rhos[2],
                    "rho_quad_iv": quad_rhos[3],
                    "quad_validation_note": _quad_spearman_note(rho_ord, quad_rhos),
                    **stat,
                }
            )

    rows_out.sort(
        key=lambda r: (
            r["composite_abs_d_rho"] is None,
            -(r["composite_abs_d_rho"] if r["composite_abs_d_rho"] is not None else 0.0),
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            -(r["abs_spearman_rho_ord"] if r["abs_spearman_rho_ord"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_out, start=1):
        r["rank"] = i

    return {
        "compare": "(c) RBP_DockQ < 0.23 vs (a) RBP_DockQ >= 0.8 for Cohen d; Spearman uses ordinal class "
        "a=1, b=2, c=3 on all complexes in (a)+(b)+(c), quadrants pooled.",
        "formula_cohen": "Cohen d = (mean_C - mean_A) / s_pooled; "
        "s_pooled = sqrt(((n_C-1)*var_C + (n_A-1)*var_A) / (n_A+n_C-2)) with sample variances (ddof=1).",
        "formula_composite": "Primary rank: composite = |d| * |rho_ord|; rho_ord = Spearman(metric, ordinal class).",
        "rows": rows_out,
    }


_AUDIT_UNIT_INTERVAL_METRIC_KEYS = frozenset({
    "RNAmotif", "RNAmatch", "AAmatch_score",
    "Protein_LDDT", "RNA_LDDT_allAtom", "RNA_TM", "Protein_TM", "Complex_TM", "Complex_LDDT_allAtom",
    "RBP_DockQ", "RBP_QS", "RBP_fnat",
    "RNA_iLDDT_AllChains", "RNA_ICS_AllChains",
    "RNA_SequenceComplexity", "RNA_GC_Content",
    "af3_rna_pTM", "af3_protein_pTM", "af3_rna_ipTM", "af3_protein_ipTM",
    "af3_rna_pLDDT_avg", "af3_protein_pLDDT_avg", "af3_ranking_score",
    "has_RNA_MSA", "has_RBP_Template", "has_Ions",
})


def _audit_finite_values(vals):
    return np.asarray([float(v) for v in vals if np.isfinite(v)], dtype=float)


def _audit_class_median(vals):
    a = _audit_finite_values(vals)
    return float(np.median(a)) if a.size else None


def _audit_class_mean(vals):
    a = _audit_finite_values(vals)
    return float(np.mean(a)) if a.size else None


def _audit_tier_cutpoints_from_centers(c_good, c_mid, c_bad):
    """Midpoints between good/mid/bad class centers; None if degenerate."""
    if c_good is None or c_bad is None:
        return None
    if c_mid is None:
        c_mid = (c_good + c_bad) / 2.0
    t_lo_raw = (c_bad + c_mid) / 2.0
    t_hi_raw = (c_mid + c_good) / 2.0
    t_lo, t_hi = (min(t_lo_raw, t_hi_raw), max(t_lo_raw, t_hi_raw))
    if abs(t_hi - t_lo) < 1e-12:
        return None
    return t_lo, t_hi, c_good >= c_bad


# False = lower values are good. Overrides median-based direction when known or when
# class means disagree with class medians (e.g. ProteinChain_Count vs DockQ).
_AUDIT_METRIC_HIGHER_IS_GOOD_OVERRIDES = {
    "ProteinChain_Count": False,
}


def _audit_resolve_metric_higher_is_good(metric_key, median_higher_good, mean_good, mean_bad):
    if metric_key in _AUDIT_METRIC_HIGHER_IS_GOOD_OVERRIDES:
        return bool(_AUDIT_METRIC_HIGHER_IS_GOOD_OVERRIDES[metric_key]), "direction override"
    if mean_good is not None and mean_bad is not None:
        mean_higher = mean_good >= mean_bad
        if mean_higher != bool(median_higher_good):
            return mean_higher, "direction from class means (medians disagreed)"
    return bool(median_higher_good), None


def _audit_is_unit_interval_metric(metric_key, pooled_vals):
    if metric_key in _AUDIT_UNIT_INTERVAL_METRIC_KEYS:
        return True
    a = _audit_finite_values(pooled_vals)
    if a.size == 0:
        return False
    return float(np.min(a)) >= 0.0 and float(np.max(a)) <= 1.0 + 1e-9


def _fmt_audit_tier_band(tier, t_lo, t_hi, higher_good, unit_interval):
    """Human-readable good / mid / bad interval for one metric."""
    if t_lo is None or t_hi is None or not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        return "NA"
    t_lo, t_hi = float(t_lo), float(t_hi)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    cap = "1" if unit_interval else "∞"
    if higher_good:
        if tier == "good":
            return f"[{_fmt_audit_stat_float(t_hi, 4)}, {cap}]"
        if tier == "mid":
            return f"[{_fmt_audit_stat_float(t_lo, 4)}, {_fmt_audit_stat_float(t_hi, 4)})"
        return f"< {_fmt_audit_stat_float(t_lo, 4)}"
    if tier == "good":
        return f"≤ {_fmt_audit_stat_float(t_lo, 4)}"
    if tier == "mid":
        return f"({_fmt_audit_stat_float(t_lo, 4)}, {_fmt_audit_stat_float(t_hi, 4)}]"
    if unit_interval:
        return f"({_fmt_audit_stat_float(t_hi, 4)}, 1.0000]"
    return f"> {_fmt_audit_stat_float(t_hi, 4)}"


def _build_audit_metric_tier_interval_rows(metric_tuples, bucket_good, bucket_mid, bucket_bad, skip_keys=()):
    """Build good/mid/bad band rows from three value buckets (good=bin0, mid=bin1, bad=bin2)."""
    rows_out = []
    skip = set(skip_keys or ())
    for mkey, mlbl in metric_tuples:
        if mkey in skip:
            continue
        vg, vm, vb = bucket_good.get(mkey, []), bucket_mid.get(mkey, []), bucket_bad.get(mkey, [])
        ng, nm, nb = len(vg), len(vm), len(vb)
        med_g = _audit_class_median(vg)
        med_m = _audit_class_median(vm)
        med_b = _audit_class_median(vb)
        mean_g = _audit_class_mean(vg)
        mean_m = _audit_class_mean(vm)
        mean_b = _audit_class_mean(vb)
        pooled = vg + vm + vb
        unit_iv = _audit_is_unit_interval_metric(mkey, pooled)

        if med_g is None or med_b is None:
            rows_out.append({
                "metric_key": mkey,
                "label": mlbl,
                "n_good": ng, "n_mid": nm, "n_bad": nb,
                "tier_note": "insufficient good- or bad-tier samples",
            })
            continue

        if med_m is None:
            med_m = (med_g + med_b) / 2.0

        cut = _audit_tier_cutpoints_from_centers(med_g, med_m, med_b)
        cutpoint_note = None
        if cut is None:
            if mean_m is None and mean_g is not None and mean_b is not None:
                mean_m = (mean_g + mean_b) / 2.0
            cut = _audit_tier_cutpoints_from_centers(mean_g, mean_m, mean_b)
            if cut is not None:
                cutpoint_note = "cutpoints from class means (medians tied)"

        if cut is None:
            rows_out.append({
                "metric_key": mkey,
                "label": mlbl,
                "n_good": ng, "n_mid": nm, "n_bad": nb,
                "med_good": med_g, "med_mid": med_m, "med_bad": med_b,
                "tier_note": "degenerate class medians and means (single cutpoint)",
            })
            continue

        t_lo, t_hi, median_higher_good = cut
        higher_good, dir_note = _audit_resolve_metric_higher_is_good(
            mkey, median_higher_good, mean_g, mean_b
        )
        row = {
            "metric_key": mkey,
            "label": mlbl,
            "unit_interval": unit_iv,
            "higher_is_good": higher_good,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "med_good": med_g,
            "med_mid": med_m,
            "med_bad": med_b,
            "good_band": _fmt_audit_tier_band("good", t_lo, t_hi, higher_good, unit_iv),
            "mid_band": _fmt_audit_tier_band("mid", t_lo, t_hi, higher_good, unit_iv),
            "bad_band": _fmt_audit_tier_band("bad", t_lo, t_hi, higher_good, unit_iv),
            "n_good": ng,
            "n_mid": nm,
            "n_bad": nb,
        }
        if cutpoint_note:
            row["cutpoint_note"] = cutpoint_note
        if dir_note:
            row["direction_note"] = dir_note
        rows_out.append(row)
    return rows_out


def compute_audit_metric_tier_intervals(rna_metrics, select_table, audit):
    """
    Suggested good / mid / bad value bands per core + AF3 audit feature.

    Aligned to RBP_DockQ audit bins: good ≈ (a) ≥0.8, mid ≈ (b), bad ≈ (c) <0.23
    (quadrants pooled). Cutpoints t_lo / t_hi are midpoints between class medians
    (bad|mid) and (mid|good). Direction follows median(good) vs median(bad).
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    metric_tuples = list(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS) + list(AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS)
    bucket_good = {k: [] for k, _ in metric_tuples}
    bucket_mid = {k: [] for k, _ in metric_tuples}
    bucket_bad = {k: [] for k, _ in metric_tuples}

    for pref in _AUDIT_QUAD_PREFIXES:
        for recs, bucket in (
            (audit.get(f"{pref}_a") or [], bucket_good),
            (audit.get(f"{pref}_b") or [], bucket_mid),
            (audit.get(f"{pref}_c") or [], bucket_bad),
        ):
            for rec in recs:
                eid = rec.get("exp_db_id")
                if not eid:
                    continue
                drow = by_eid.get(str(eid).strip())
                if not drow:
                    continue
                fv = _audit_extract_motif_bin_metrics(drow)
                for k, _ in metric_tuples:
                    val = fv.get(k)
                    if np.isfinite(val):
                        bucket[k].append(float(val))

    rows_out = _build_audit_metric_tier_interval_rows(
        metric_tuples, bucket_good, bucket_mid, bucket_bad, skip_keys={"RBP_DockQ"}
    )
    if not rows_out:
        return None

    return {
        "caption": (
            "DockQ-aligned tiers: good ≈ RBP_DockQ (a) ≥0.8, mid ≈ (b), bad ≈ (c) <0.23; "
            "quadrants pooled. t_lo / t_hi = midpoints between class medians (bad|mid) and (mid|good); "
            "if medians tie, cutpoints use class means instead. "
            "Direction: higher_is_good if median(good) ≥ median(bad), else class means when "
            "medians disagree; explicit overrides for known count metrics. "
            "Unit-interval metrics cap good/bad bands at [0, 1] where noted. RBP_DockQ omitted."
        ),
        "rows": rows_out,
    }


def compute_audit_crmsd_metric_tier_intervals(rna_metrics, select_table):
    """
    Good / mid / bad bands per core + AF3 feature stratified by Complex_RMSD bin
    (<2 Å, [2, 10) Å, ≥10 Å); all rows with finite Complex_RMSD.
    """
    row_dicts = _audit_load_prediction_rows_for_motif_bins(rna_metrics, select_table, "RNAmotif")
    if not row_dicts:
        return None

    metric_tuples = list(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS) + list(AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS)
    bucket_good = {k: [] for k, _ in metric_tuples}
    bucket_mid = {k: [] for k, _ in metric_tuples}
    bucket_bad = {k: [] for k, _ in metric_tuples}

    for d in row_dicts:
        fv = _audit_extract_motif_bin_metrics(d)
        bix = _crmsd_bin_index(fv.get("Complex_RMSD"))
        if bix is None:
            continue
        bucket = (bucket_good, bucket_mid, bucket_bad)[bix]
        for k, _ in metric_tuples:
            val = fv.get(k)
            if np.isfinite(val):
                bucket[k].append(float(val))

    rows_out = _build_audit_metric_tier_interval_rows(
        metric_tuples, bucket_good, bucket_mid, bucket_bad, skip_keys={"Complex_RMSD"}
    )
    if not rows_out:
        return None

    return {
        "caption": (
            "Complex_RMSD tiers: good ≈ CRMSD < 2 Å, mid ≈ [2, 10) Å, bad ≈ ≥10 Å; "
            "all prediction rows with finite Complex_RMSD. "
            "t_lo / t_hi = midpoints between class medians (bad|mid) and (mid|good); "
            "if medians tie, cutpoints use class means instead. "
            "Direction: higher_is_good if median(good bin) ≥ median(bad bin). "
            "Unit-interval metrics cap good/bad bands at [0, 1] where noted. Complex_RMSD omitted."
        ),
        "rows": rows_out,
    }


def _audit_metric_buckets_empty(metric_tuples):
    return {k: [] for k, _ in metric_tuples}


def _audit_merge_metric_buckets(*bucket_dicts):
    if not bucket_dicts:
        return {}
    keys = bucket_dicts[0].keys()
    merged = {k: [] for k in keys}
    for bd in bucket_dicts:
        for k in keys:
            merged[k].extend(bd.get(k, []))
    return merged


def _audit_push_audit_recs_to_buckets(recs, by_eid, buckets):
    for rec in recs or []:
        eid = rec.get("exp_db_id")
        if not eid:
            continue
        drow = by_eid.get(str(eid).strip())
        if not drow:
            continue
        fv = _audit_extract_motif_bin_metrics(drow)
        for k in buckets:
            val = fv.get(k)
            if np.isfinite(val):
                buckets[k].append(float(val))


def _audit_tier_rows_by_metric(rows_out):
    return {r["metric_key"]: r for r in (rows_out or []) if r.get("metric_key")}


def _audit_collect_fold_stratification_buckets(rna_metrics, select_table, audit, metric_tuples):
    """Metric value buckets for fold / RNA / protein / CRMSD-failure audit stratifications."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None
    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    quad_prefs = tuple(_AUDIT_QUAD_PREFIXES)
    if len(quad_prefs) != 4:
        return None

    quad_all = [_audit_metric_buckets_empty(metric_tuples) for _ in range(4)]
    for qi, pref in enumerate(quad_prefs):
        _audit_push_audit_recs_to_buckets(audit.get(f"{pref}_all") or [], by_eid, quad_all[qi])

    lt10 = _audit_metric_buckets_empty(metric_tuples)
    ge10 = _audit_metric_buckets_empty(metric_tuples)
    for pref in quad_prefs:
        _audit_push_audit_recs_to_buckets(audit.get(f"{pref}_c_ca") or [], by_eid, lt10)
        _audit_push_audit_recs_to_buckets(audit.get(f"{pref}_c_cb") or [], by_eid, ge10)

    return {
        "fold_ordinal_i_iv": (
            quad_all[0],
            _audit_merge_metric_buckets(quad_all[1], quad_all[2]),
            quad_all[3],
        ),
        "fold_binary_i_vs_iv": (
            quad_all[0],
            _audit_metric_buckets_empty(metric_tuples),
            quad_all[3],
        ),
        "rna_accurate_i_ii_vs_iii_iv": (
            _audit_merge_metric_buckets(quad_all[0], quad_all[1]),
            _audit_metric_buckets_empty(metric_tuples),
            _audit_merge_metric_buckets(quad_all[2], quad_all[3]),
        ),
        "protein_accurate_i_iii_vs_ii_iv": (
            _audit_merge_metric_buckets(quad_all[0], quad_all[2]),
            _audit_metric_buckets_empty(metric_tuples),
            _audit_merge_metric_buckets(quad_all[1], quad_all[3]),
        ),
        "crmsd_fail_lt10_vs_ge10": (
            lt10,
            _audit_metric_buckets_empty(metric_tuples),
            ge10,
        ),
    }


# Per-metric consensus tier source(s). Overlapping metrics merge strictest cutpoints when
# direction matches (↑ good: max t_lo/t_hi; ↓ good: min). Fold ordinal / binary excluded.
AUDIT_CONSENSUS_METRIC_STRATIFICATIONS = {
    # DockQ good/mid/bad (RBP_DockQ a / b / c)
    "RNAmotif": ("dockq",),
    "AAmatch_score": ("dockq",),
    "RBP_PDB_n": ("dockq", "rna_accuracy", "crmsd"),
    "has_RBP_Template": ("dockq", "rna_accuracy", "crmsd"),
    "has_Ions": ("dockq", "rna_accuracy", "crmsd"),
    "exp_pH": ("dockq", "rna_accuracy", "crmsd"),
    "RBP_QS": ("dockq",),
    "RBP_fnat": ("dockq",),
    "RBP_irmsd": ("dockq",),
    "RNA_iLDDT_AllChains": ("dockq",),
    "af3_rna_ipTM": ("dockq",),
    "af3_protein_ipTM": ("dockq",),
    # RNA accuracy (i)+(ii) vs (iii)+(iv)
    "RNAChain_Count": ("rna_accuracy",),
    "RNA_length_longest": ("rna_accuracy",),
    "RNA_chain_load": ("rna_accuracy",),
    "has_RNA_MSA": ("rna_accuracy",),
    "RNA_msa_depth": ("rna_accuracy",),
    "RNA_LDDT_allAtom": ("rna_accuracy",),
    "RNA_TM": ("rna_accuracy",),
    "RNA_MFE_Value": ("rna_accuracy",),
    "RNA_SequenceComplexity": ("rna_accuracy",),
    "RNA_Shannon_Entropy_K2": ("rna_accuracy",),
    "RNA_GC_Content": ("rna_accuracy",),
    "af3_rna_pTM": ("rna_accuracy",),
    "af3_rna_pLDDT_avg": ("rna_accuracy",),
    # Protein accuracy (i)+(iii) vs (ii)+(iv)
    "ProteinChain_Count": ("protein_accuracy",),
    "Protein_length_longest": ("protein_accuracy",),
    "Protein_chain_load": ("protein_accuracy",),
    "Protein_PDB_n": ("protein_accuracy",),
    "Protein_msa_depth": ("protein_accuracy",),
    "Protein_LDDT": ("protein_accuracy",),
    "Protein_TM": ("protein_accuracy",),
    "af3_protein_pTM": ("protein_accuracy",),
    "af3_protein_pLDDT_avg": ("protein_accuracy",),
    # Complex_RMSD (<2 Å / [2,10) / ≥10 Å)
    "RNAProtein_Ratio": ("crmsd",),
    "Complex_RMSD": ("crmsd",),
    "Complex_TM": ("crmsd",),
    "Complex_LDDT_allAtom": ("crmsd",),
    "af3_global_pae_avg": ("crmsd",),
    "af3_chain_pair_pae_min": ("crmsd",),
    "af3_ranking_score": ("crmsd",),
}
_AUDIT_CONSENSUS_STRAT_LABELS = {
    "dockq": "DockQ good/mid/bad",
    "rna_accuracy": "RNA accuracy (i)+(ii) vs (iii)+(iv)",
    "protein_accuracy": "protein accuracy (i)+(iii) vs (ii)+(iv)",
    "crmsd": "Complex_RMSD (<2 / [2,10) / ≥10 Å)",
}
_AUDIT_CONSENSUS_STRAT_PRIORITY = (
    "dockq",
    "rna_accuracy",
    "protein_accuracy",
    "crmsd",
)


def _audit_merge_stricter_tier_cutpoints(base_row, other_rows_by_name):
    """
    Start from DockQ tier cutpoints; tighten using other stratifications with matching direction.
    ↑ good: raise t_lo and t_hi; ↓ good: lower t_lo and t_hi.
    """
    if base_row.get("tier_note") or base_row.get("t_lo") is None or base_row.get("t_hi") is None:
        return None

    t_lo = float(base_row["t_lo"])
    t_hi = float(base_row["t_hi"])
    higher_good = bool(base_row["higher_is_good"])
    unit_iv = base_row.get("unit_interval", False)
    refined_by = []

    for name, orow in other_rows_by_name.items():
        if orow.get("tier_note") or orow.get("t_lo") is None or orow.get("t_hi") is None:
            continue
        if bool(orow.get("higher_is_good")) != higher_good:
            continue
        ot_lo, ot_hi = float(orow["t_lo"]), float(orow["t_hi"])
        if higher_good:
            nt_lo, nt_hi = max(t_lo, ot_lo), max(t_hi, ot_hi)
        else:
            nt_lo, nt_hi = min(t_lo, ot_lo), min(t_hi, ot_hi)
        if nt_lo != t_lo or nt_hi != t_hi:
            refined_by.append(name)
            t_lo, t_hi = nt_lo, nt_hi

    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    if abs(t_hi - t_lo) < 1e-12:
        return None

    out = dict(base_row)
    out["t_lo"] = t_lo
    out["t_hi"] = t_hi
    out["good_band"] = _fmt_audit_tier_band("good", t_lo, t_hi, higher_good, unit_iv)
    out["mid_band"] = _fmt_audit_tier_band("mid", t_lo, t_hi, higher_good, unit_iv)
    out["bad_band"] = _fmt_audit_tier_band("bad", t_lo, t_hi, higher_good, unit_iv)
    out["consensus_refined_by"] = refined_by
    if refined_by:
        out["consensus_note"] = "strictest cutpoints vs DockQ-only: " + ", ".join(refined_by)
    else:
        out["consensus_note"] = "same as DockQ-only (no stricter cutpoint from other audit tables)"
    return out


def _audit_build_stratification_tier_maps(
    rna_metrics, select_table, audit, metric_tuples, *, dockq_tiers=None
):
    """Tier rows keyed by stratification name, then metric_key."""
    dockq_block = dockq_tiers or compute_audit_metric_tier_intervals(
        rna_metrics, select_table, audit
    )
    crmsd_block = compute_audit_crmsd_metric_tier_intervals(rna_metrics, select_table)
    strat_buckets = _audit_collect_fold_stratification_buckets(
        rna_metrics, select_table, audit, metric_tuples
    )
    if not dockq_block or not strat_buckets:
        return None

    out = {}
    if dockq_block.get("rows"):
        out["dockq"] = _audit_tier_rows_by_metric(dockq_block["rows"])
    if crmsd_block and crmsd_block.get("rows"):
        out["crmsd"] = _audit_tier_rows_by_metric(crmsd_block["rows"])

    for strat_name, bucket_key in (
        ("rna_accuracy", "rna_accurate_i_ii_vs_iii_iv"),
        ("protein_accuracy", "protein_accurate_i_iii_vs_ii_iv"),
    ):
        buckets = strat_buckets.get(bucket_key)
        if not buckets:
            continue
        rows = _build_audit_metric_tier_interval_rows(
            metric_tuples, buckets[0], buckets[1], buckets[2], skip_keys={"RBP_DockQ"}
        )
        out[strat_name] = _audit_tier_rows_by_metric(rows)
    return out


def _audit_build_assigned_consensus_tier_row(mkey, strat_names, tiers_by_strat):
    """One consensus row from explicitly assigned classification(s) for a metric."""
    available = []
    for strat_name in strat_names:
        row = (tiers_by_strat.get(strat_name) or {}).get(mkey)
        if row and not row.get("tier_note"):
            available.append((strat_name, row))
    if not available:
        return None

    available.sort(
        key=lambda item: (
            _AUDIT_CONSENSUS_STRAT_PRIORITY.index(item[0])
            if item[0] in _AUDIT_CONSENSUS_STRAT_PRIORITY
            else 99
        )
    )
    base_name, merged = available[0][0], dict(available[0][1])
    refined_by = []
    for other_name, other_row in available[1:]:
        tightened = _audit_merge_stricter_tier_cutpoints(merged, {other_name: other_row})
        if tightened is None:
            continue
        if tightened.get("consensus_refined_by"):
            refined_by.extend(tightened["consensus_refined_by"])
        merged = tightened

    src_labels = [_AUDIT_CONSENSUS_STRAT_LABELS.get(s, s) for s in strat_names]
    if len(available) == 1:
        merged["consensus_note"] = (
            f"from {_AUDIT_CONSENSUS_STRAT_LABELS.get(base_name, base_name)} only"
        )
    elif refined_by:
        merged["consensus_note"] = (
            "strictest cutpoints across assigned classifications: "
            + ", ".join(_AUDIT_CONSENSUS_STRAT_LABELS.get(s, s) for s in refined_by)
        )
    else:
        merged["consensus_note"] = (
            "same across assigned classifications: " + ", ".join(src_labels)
        )
    merged["consensus_sources"] = list(strat_names)
    return merged


def compute_audit_consensus_metric_tier_intervals(rna_metrics, select_table, audit, dockq_tiers=None):
    """
    Per-metric good/mid/bad bands from explicitly assigned audit classifications.

    Each metric uses one or more of: DockQ tiers, RNA-accuracy quadrants, protein-accuracy
    quadrants, Complex_RMSD bins. Overlapping metrics (e.g. has_Ions, pH, RBP_PDB_n) merge
    strictest cutpoints when direction matches. Fold ordinal / binary stratifications excluded.
    """
    metric_tuples = list(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS) + list(
        AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS
    )
    tiers_by_strat = _audit_build_stratification_tier_maps(
        rna_metrics, select_table, audit, metric_tuples, dockq_tiers=dockq_tiers
    )
    if not tiers_by_strat:
        return None

    rows_out = []
    for mkey, strat_names in AUDIT_CONSENSUS_METRIC_STRATIFICATIONS.items():
        row = _audit_build_assigned_consensus_tier_row(mkey, strat_names, tiers_by_strat)
        if row is not None:
            rows_out.append(row)

    if not rows_out:
        return None

    rows_out.sort(key=lambda r: (_audit_metric_display_label(r.get("metric_key", "")),))

    return {
        "caption": (
            "Each metric is assigned to one or more audit classifications (no fold ordinal "
            "(i)→(iv) or fold binary). DockQ good/mid/bad: RNAmotif, AAmatch_score, "
            "RBP_PDB_Templates count, has_RBP_Template, has_Ions, pH, RBP_QS, RBP_fnat, "
            "RBP_irmsd, RNA_iLDDT_AllChains, af3_rna_ipTM, af3_protein_ipTM. "
            "RNA accuracy (i)+(ii) vs (iii)+(iv): RNAChain_Count, RNA length (longest), "
            "RNA chain load, RBP templates, has_RBP_Template, has_RNA_MSA, RNA MSA depth, "
            "has_Ions, pH, RNA_LDDT, RNA_TM, RNA_MFE, RNA_SequenceComplexity, "
            "RNA_Shannon_Entropy_K2, RNA_GC_Content, af3_rna_pTM, af3_rna_pLDDT_avg. "
            "Protein accuracy (i)+(iii) vs (ii)+(iv): ProteinChain_Count, Protein length, "
            "Protein chain load, Protein_PDB templates, Protein MSA depth, Protein_LDDT, "
            "Protein_TM, af3_protein_pTM, af3_protein_pLDDT_avg. "
            "Complex_RMSD (<2 / [2,10) / ≥10 Å): RNA/Protein ratio, RBP templates, "
            "has_RBP_Template, has_Ions, pH, Complex_RMSD, Complex_TM, Complex_LDDT, "
            "af3_global_pae_avg, af3_chain_pair_pae_min, af3_ranking_score. "
            "When a metric appears in multiple lists, adopt strictest cutpoints if direction "
            "matches (↑ good → max(t_lo, t_hi); ↓ good → min). Displayed medians are from "
            "the highest-priority assigned classification (DockQ > RNA > protein > CRMSD)."
        ),
        "rows": rows_out,
    }


# Metrics for consensus-tier PDB id lists (keys = engineered audit feature names).
AUDIT_CONSENSUS_PDB_TIER_METRIC_KEYS = (
    "af3_protein_ipTM",
    "af3_rna_ipTM",
    "af3_rna_pLDDT_avg",
    "af3_protein_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_protein_pTM",
    "RBP_PDB_n",
    "RNA_length_longest",
    "RNA_Shannon_Entropy_K2",
    "RNA_MFE_Value",
    "ProteinChain_Count",
    "af3_chain_pair_pae_min",
    "RNA_SequenceComplexity",
    "RNA_chain_load",
    "af3_ranking_score",
)

# Entity groupings for AF3-only complex prediction (consensus-tier feature keys).
AUDIT_ENTITY_METRIC_GROUPS = {
    "Interface": ("af3_protein_ipTM", "af3_rna_ipTM"),
    "RNA": ("af3_rna_pLDDT_avg",),
    "Protein": ("af3_protein_pTM",),
    "Protein-RNA": ("af3_chain_pair_pae_min", "af3_global_pae_avg"),
}

_AUDIT_TIER_SEVERITY = {"bad": 0, "mid": 1, "good": 2}
_AUDIT_COMPONENT_ENTITIES = ("Interface", "RNA", "Protein")
AUDIT_ENTITY_PRED_SCORE_VALUES = ("bad", "mid", "good")
AUDIT_ENTITY_PRED_SCORE_DB_COLUMNS = {
    "final": "Final_pred_score",
    "Interface": "Interface_pred_score",
    "RNA": "RNA_pred_score",
    "Protein": "Protein_pred_score",
    "Protein-RNA": "Protein_RNA_pred_score",
}
AUDIT_ENTITY_TIER_MODEL_VERSION = 1


def _audit_metric_display_label(mkey):
    for k, lbl in AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS + AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS:
        if k == mkey:
            return str(lbl).replace("\n", " ")
    return mkey


def _audit_unique_eids_from_audit(audit):
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return []
    seen = set()
    out = []
    for pref in _AUDIT_QUAD_PREFIXES:
        for rec in audit.get(f"{pref}_all") or []:
            eid = str(rec.get("exp_db_id", "")).strip()
            if eid and eid not in seen:
                seen.add(eid)
                out.append(eid)
    return sorted(out)


def _audit_classify_value_tier(value, tier_row):
    if tier_row.get("t_lo") is None or tier_row.get("t_hi") is None:
        return None
    if value is None or not np.isfinite(value):
        return None
    v = float(value)
    t_lo, t_hi = float(tier_row["t_lo"]), float(tier_row["t_hi"])
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    higher_good = bool(tier_row.get("higher_is_good"))
    if higher_good:
        if v >= t_hi:
            return "good"
        if v < t_lo:
            return "bad"
        return "mid"
    if v <= t_lo:
        return "good"
    if v > t_hi:
        return "bad"
    return "mid"


def _fmt_audit_eid_list(eids):
    return ", ".join(str(x) for x in (eids or []))


def _audit_aggregate_metric_list_tier(tiers, *, af3_veto=False):
    """Majority tier on a list of good/mid/bad; optional any-bad AF3 veto."""
    finite = [t for t in (tiers or []) if t is not None]
    if not finite:
        return None
    n = len(finite)
    n_good = sum(1 for t in finite if t == "good")
    n_bad = sum(1 for t in finite if t == "bad")
    if af3_veto and n_bad >= 1:
        return "bad"
    if n_bad / n >= 0.5:
        return "bad"
    if n_good / n >= 0.75 and n_bad == 0:
        return "good"
    return "mid"


def _audit_eid_dockq_truth_tiers(audit):
    """Map exp_db_id to DockQ truth tier good/mid/bad from quad_*_{a,b,c} lists."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return {}
    truth = {}
    for pref in _AUDIT_QUAD_PREFIXES:
        for tier, suffix in (("good", "_a"), ("mid", "_b"), ("bad", "_c")):
            for rec in audit.get(f"{pref}{suffix}") or []:
                eid = str(rec.get("exp_db_id", "")).strip()
                if eid and eid not in truth:
                    truth[eid] = tier
    return truth


def _audit_tier_match_score(pred_tier, truth_tier):
    """1 exact, 0.5 adjacent ordinal, 0 opposite; None if either missing."""
    if pred_tier is None or truth_tier is None:
        return None
    if pred_tier == truth_tier:
        return 1.0
    po = _AUDIT_TIER_SEVERITY.get(pred_tier)
    to = _AUDIT_TIER_SEVERITY.get(truth_tier)
    if po is None or to is None:
        return None
    return 0.5 if abs(po - to) == 1 else 0.0


def _audit_aggregate_af3_entity_tier(cls_map, metric_keys):
    """Entity tier from AF3 metric tiers only (any-bad veto)."""
    tiers = [cls_map[k] for k in metric_keys if k in cls_map]
    return _audit_aggregate_metric_list_tier(tiers, af3_veto=True)


# RNA entity tier from influence-stratum logic (pLDDT 0–1 + RBP / chains / MSA / length).
AUDIT_RNA_PLDDT_TIER_GOOD = 0.775
AUDIT_RNA_PLDDT_TIER_BAD = 0.55
AUDIT_RNA_SHANNON_IV_LIKE = 3.0


def _audit_rna_plddt_tier_from_value(plddt):
    """good / mid / bad from af3_rna_pLDDT_avg (DockQ-consensus cutpoints)."""
    if not np.isfinite(plddt):
        return None
    v = float(plddt)
    if v >= AUDIT_RNA_PLDDT_TIER_GOOD:
        return "good"
    if v < AUDIT_RNA_PLDDT_TIER_BAD:
        return "bad"
    return "mid"


def _audit_rna_influence_context(fv):
    """Structural flags matching RNA quadrant × influence strata."""
    ctx = {}
    msa = fv.get("has_RNA_MSA")
    if np.isfinite(msa):
        ctx["has_msa"] = float(msa) >= 0.5
        ctx["no_msa"] = float(msa) < 0.5
    else:
        ctx["has_msa"] = ctx["no_msa"] = None
    rbp = fv.get("RBP_PDB_n")
    if np.isfinite(rbp):
        rv = float(rbp)
        ctx["rbp0"] = rv == 0.0
        ctx["rbp1"] = rv == 1.0
        ctx["rbp_gt1"] = rv > 1.0
        ctx["rbp_ge1"] = rv >= 1.0
    else:
        ctx["rbp0"] = ctx["rbp1"] = ctx["rbp_gt1"] = ctx["rbp_ge1"] = None
    nch = fv.get("RNAChain_Count")
    if np.isfinite(nch):
        nv = float(nch)
        ctx["ch1"] = nv == 1.0
        ctx["ch_gt1"] = nv > 1.0
    else:
        ctx["ch1"] = ctx["ch_gt1"] = None
    ln = fv.get("RNA_length_longest")
    if np.isfinite(ln):
        lv = float(ln)
        ctx["len_le40"] = lv <= 40.0
        ctx["len_41_100"] = 41.0 <= lv <= 100.0
        ctx["len_gt100"] = lv > 100.0
    else:
        ctx["len_le40"] = ctx["len_41_100"] = ctx["len_gt100"] = None
    sh = fv.get("RNA_Shannon_Entropy_K2")
    ctx["sh_high"] = bool(np.isfinite(sh) and float(sh) >= AUDIT_RNA_SHANNON_IV_LIKE)
    return ctx


def _audit_predict_rna_tier_from_influence_rules(fv):
    """
    Per-PDB RNA good/mid/bad from influence-stratum logic.

    Uses af3_rna_pLDDT_avg tiers (≥0.775 good, [0.55, 0.775) mid, <0.55 bad) with
    structural context: RBP_PDB {0, 1, >1}, RNAChain_Count, has_RNA_MSA, RNALength bins.
    High Shannon (≥3) with pLDDT below good cutpoint on short no-MSA single-chain → bad (iv-like).
    """
    pt = _audit_rna_plddt_tier_from_value(fv.get("af3_rna_pLDDT_avg"))
    plddt = fv.get("af3_rna_pLDDT_avg")
    ctx = _audit_rna_influence_context(fv)

    def _iv_like_bump(tier):
        if (
            tier in ("good", "mid")
            and ctx.get("no_msa")
            and ctx.get("len_le40")
            and ctx.get("ch1")
            and ctx.get("sh_high")
            and np.isfinite(plddt)
            and float(plddt) < AUDIT_RNA_PLDDT_TIER_GOOD
        ):
            return "bad"
        return tier

    if ctx.get("has_msa"):
        if pt is None:
            return "bad"
        if ctx.get("len_41_100") and ctx.get("ch1") and ctx.get("rbp0"):
            return pt
        if (
            pt == "good"
            and ctx.get("ch1")
            and (ctx.get("rbp1") or ctx.get("rbp_gt1"))
            and (ctx.get("len_41_100") or ctx.get("len_gt100"))
        ):
            return "mid"
        return pt if pt != "good" else "mid"

    if ctx.get("no_msa") and (ctx.get("len_41_100") or ctx.get("len_gt100")):
        return pt if pt is not None else "bad"

    if ctx.get("no_msa") and ctx.get("len_le40"):
        if ctx.get("ch_gt1") and ctx.get("rbp_ge1"):
            return pt if pt is not None else "mid"
        if ctx.get("ch_gt1") and ctx.get("rbp0"):
            return pt if pt is not None else "mid"
        if ctx.get("ch1"):
            if pt is not None:
                return _iv_like_bump(pt)
            return "bad" if ctx.get("rbp0") else "mid"
        return pt

    return pt


def _audit_build_af3_entity_combination_rows(per_eid, dockq_truth):
    """Interface × RNA × Protein AF3 tiers → observed DockQ good/mid/bad counts."""
    combos = {}
    for rec in per_eid:
        eid = rec.get("exp_db_id")
        truth = dockq_truth.get(eid)
        if not truth:
            continue
        entities = rec.get("entities") or {}
        key = tuple(entities.get(en) for en in _AUDIT_COMPONENT_ENTITIES)
        if any(v is None for v in key):
            continue
        if key not in combos:
            combos[key] = {"dockq_good": 0, "dockq_mid": 0, "dockq_bad": 0, "eids": []}
        combos[key]["eids"].append(eid)
        combos[key][f"dockq_{truth}"] += 1

    rows = []
    for key, counts in combos.items():
        n = len(counts["eids"])
        rows.append({
            "interface": key[0],
            "rna": key[1],
            "protein": key[2],
            "n": n,
            "dockq_good": counts["dockq_good"],
            "dockq_mid": counts["dockq_mid"],
            "dockq_bad": counts["dockq_bad"],
            "dockq_good_frac": counts["dockq_good"] / n if n else None,
            "dockq_mid_frac": counts["dockq_mid"] / n if n else None,
            "dockq_bad_frac": counts["dockq_bad"] / n if n else None,
            "eids": sorted(counts["eids"]),
        })
    rows.sort(
        key=lambda r: (
            -r["n"],
            -(r["dockq_good_frac"] if r["dockq_good_frac"] is not None else 0),
        )
    )
    return rows


def _audit_predict_complex_tier_from_af3_entities(interface, _rna, protein, protein_rna):
    """
    AF3 entity tiers → final complex good/mid/bad.

    RNA tier is computed and reported separately but does not affect ``final``.
    Uses Interface, Protein, and Protein-RNA only.

    good: Interface=Protein=good and Protein-RNA in {good, mid}
    mid:  Interface=good, Protein=bad (any Protein-RNA); or Interface=Protein=good
          but Protein-RNA=bad; or other mixed with ≤1 of {Interface, Protein} bad
    bad:  Interface bad, or both Interface and Protein bad
    """
    if interface is None or protein is None:
        return None

    if interface == "good" and protein == "good":
        if protein_rna in ("good", "mid", None):
            return "good"
        if protein_rna == "bad":
            return "mid"

    if interface == "good" and protein == "bad":
        return "mid"

    n_bad = sum(1 for c in (interface, protein) if c == "bad")
    if n_bad >= 2 or interface == "bad":
        return "bad"

    return "mid"


def _audit_build_consensus_classifications(rna_metrics, select_table, audit, consensus_tiers, metric_keys=None):
    """Per exp_db_id metric tier map using classification-assigned consensus cutpoints."""
    if not consensus_tiers or not consensus_tiers.get("rows"):
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    eids = _audit_unique_eids_from_audit(audit)
    if not eids:
        return None

    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    tier_by_metric = _audit_tier_rows_by_metric(consensus_tiers["rows"])
    if metric_keys is None:
        metric_keys = [k for k in AUDIT_CONSENSUS_PDB_TIER_METRIC_KEYS if k in tier_by_metric]
    else:
        metric_keys = [k for k in metric_keys if k in tier_by_metric]
    if not metric_keys:
        return None

    classifications = {eid: {} for eid in eids}
    for mkey in metric_keys:
        tier_row = tier_by_metric[mkey]
        for eid in eids:
            drow = by_eid.get(eid)
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            tier = _audit_classify_value_tier(fv.get(mkey), tier_row)
            if tier is not None:
                classifications[eid][mkey] = tier

    return {
        "eids": eids,
        "classifications": classifications,
        "tier_by_metric": tier_by_metric,
        "metric_keys": metric_keys,
    }


def _audit_entity_tier_pred_score(tier):
    """DB pred_score columns store tier labels: bad, mid, or good."""
    if tier in AUDIT_ENTITY_PRED_SCORE_VALUES:
        return tier
    return None


def _audit_entity_pred_scores_from_breakdown(entities, final_tier):
    scores = {}
    for entity_key, col in AUDIT_ENTITY_PRED_SCORE_DB_COLUMNS.items():
        tier = final_tier if entity_key == "final" else (entities or {}).get(entity_key)
        scores[col] = _audit_entity_tier_pred_score(tier)
    return scores


def predict_af3_entity_tiers_for_features(fv, tier_by_metric):
    """
    Per-PDB AF3 entity breakdown from classification-assigned consensus cutpoints.

    Same rules as ``compute_audit_entity_complex_predictions`` (RNA influence rules +
    final tier from Interface / Protein / Protein-RNA only).
    """
    entity_metric_keys = tuple(dict.fromkeys(
        [k for keys in AUDIT_ENTITY_METRIC_GROUPS.values() for k in keys]
        + ["af3_ranking_score"]
    ))
    cls_map = {}
    for mkey in entity_metric_keys:
        tier_row = (tier_by_metric or {}).get(mkey)
        if not tier_row:
            continue
        tier = _audit_classify_value_tier((fv or {}).get(mkey), tier_row)
        if tier is not None:
            cls_map[mkey] = tier

    entities = {}
    entity_metric_tiers = {}
    for ename, metric_keys in AUDIT_ENTITY_METRIC_GROUPS.items():
        entities[ename] = _audit_aggregate_af3_entity_tier(cls_map, metric_keys)
        entity_metric_tiers[ename] = {
            k: cls_map[k] for k in metric_keys if k in cls_map
        }

    rna_af3_plddt = entities.get("RNA")
    rna_rule = _audit_predict_rna_tier_from_influence_rules(fv or {})
    entities["RNA"] = rna_rule if rna_rule is not None else rna_af3_plddt
    final_tier = _audit_predict_complex_tier_from_af3_entities(
        entities.get("Interface"),
        entities.get("RNA"),
        entities.get("Protein"),
        entities.get("Protein-RNA"),
    )
    return {
        "entities": entities,
        "entity_metric_tiers": entity_metric_tiers,
        "rna_tier_af3_plddt": rna_af3_plddt,
        "af3_ranking_score_tier": cls_map.get("af3_ranking_score"),
        "final_tier": final_tier,
        "pred_scores": _audit_entity_pred_scores_from_breakdown(entities, final_tier),
    }


def _audit_serialize_tier_row_for_model(row):
    if not row:
        return None
    keys = (
        "metric_key", "label", "unit_interval", "higher_is_good", "t_lo", "t_hi",
        "good_band", "mid_band", "bad_band", "med_good", "med_mid", "med_bad",
        "n_good", "n_mid", "n_bad", "consensus_sources", "consensus_note",
    )
    out = {}
    for k in keys:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        elif isinstance(v, (list, tuple)):
            v = [str(x) for x in v]
        out[k] = v
    return out if out.get("metric_key") else None


def export_audit_entity_tier_model_json(
    path,
    consensus_tiers,
    *,
    source_table=None,
    extra_metadata=None,
):
    """Write portable JSON model from classification-assigned consensus tier intervals."""
    if not consensus_tiers or not consensus_tiers.get("rows"):
        raise ValueError("consensus_tiers has no rows to export")
    from datetime import datetime, timezone

    metrics = {}
    for row in consensus_tiers["rows"]:
        ser = _audit_serialize_tier_row_for_model(row)
        if ser:
            metrics[ser["metric_key"]] = ser

    payload = {
        "model_version": AUDIT_ENTITY_TIER_MODEL_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_table": source_table,
        "caption": consensus_tiers.get("caption"),
        "pred_score_values": list(AUDIT_ENTITY_PRED_SCORE_VALUES),
        "entity_groups": {
            k: list(v) for k, v in AUDIT_ENTITY_METRIC_GROUPS.items()
        },
        "rna_influence_rules": {
            "plddt_good": AUDIT_RNA_PLDDT_TIER_GOOD,
            "plddt_bad": AUDIT_RNA_PLDDT_TIER_BAD,
            "shannon_iv_like": AUDIT_RNA_SHANNON_IV_LIKE,
        },
        "final_tier_rules": (
            "good: Interface=Protein=good and Protein-RNA in {good, mid}; "
            "mid: Interface=good, Protein=bad; or Interface=Protein=good but Protein-RNA=bad; "
            "or other mixed with ≤1 of {Interface, Protein} bad; "
            "bad: Interface bad, or both Interface and Protein bad; RNA excluded from final."
        ),
        "pred_score_columns": dict(AUDIT_ENTITY_PRED_SCORE_DB_COLUMNS),
        "metrics": metrics,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata

    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def load_audit_entity_tier_model_json(path):
    """Load exported entity-tier model; returns tier_by_metric dict for prediction."""
    import json

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics") or {}
    tier_by_metric = {
        mkey: dict(row) for mkey, row in metrics.items() if isinstance(row, dict)
    }
    payload["tier_by_metric"] = tier_by_metric
    return payload


def compute_audit_entity_complex_predictions(rna_metrics, select_table, audit, consensus_tiers):
    """
    AF3-only entity tiers (Interface / RNA / Protein / Protein-RNA) per audit PDB,
    then predict overall complex good/mid/bad from AF3 entity decision rules.
    """
    entity_metric_keys = tuple(dict.fromkeys(
        [k for keys in AUDIT_ENTITY_METRIC_GROUPS.values() for k in keys]
        + ["af3_ranking_score"]
    ))

    base = _audit_build_consensus_classifications(
        rna_metrics, select_table, audit, consensus_tiers, metric_keys=entity_metric_keys
    )
    if not base:
        return None

    eids = base["eids"]
    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    dockq_truth = _audit_eid_dockq_truth_tiers(audit)
    per_eid = []
    final_good, final_mid, final_bad = [], [], []
    insufficient = []

    tier_by_metric = base["tier_by_metric"]
    for eid in eids:
        fv = (
            _audit_extract_motif_bin_metrics(by_eid[eid])
            if by_eid and eid in by_eid
            else {}
        )
        pred = predict_af3_entity_tiers_for_features(fv, tier_by_metric)
        rec = {
            "exp_db_id": eid,
            "dockq_truth": dockq_truth.get(eid),
            "entities": pred.get("entities"),
            "entity_metric_tiers": pred.get("entity_metric_tiers"),
            "rna_tier_af3_plddt": pred.get("rna_tier_af3_plddt"),
            "af3_ranking_score_tier": pred.get("af3_ranking_score_tier"),
            "final_tier": pred.get("final_tier"),
            "pred_scores": pred.get("pred_scores"),
        }
        per_eid.append(rec)
        final_tier = pred.get("final_tier")
        if final_tier == "good":
            final_good.append(eid)
        elif final_tier == "mid":
            final_mid.append(eid)
        elif final_tier == "bad":
            final_bad.append(eid)
        else:
            insufficient.append(eid)

    af3_combos = _audit_build_af3_entity_combination_rows(per_eid, dockq_truth)
    match_scores = [
        sc for rec in per_eid
        for sc in [_audit_tier_match_score(rec.get("final_tier"), rec.get("dockq_truth"))]
        if sc is not None
    ]
    match_rate = float(np.mean(match_scores)) if match_scores else None

    return {
        "caption": (
            "AF3 entity tiers (classification-assigned consensus cutpoints). "
            "RNA tier uses influence-stratum rules: pLDDT good≥0.775, mid≥0.55, bad<0.55, "
            "with RBP {0,1,>1}, RNAChain_Count, has_RNA_MSA, RNALength; "
            "af3_rna_pLDDT-only tier shown in parentheses when it differs. "
            "Final tier uses Interface, Protein, and Protein-RNA only (RNA excluded). "
            "Final good: Interface=Protein=good and Protein-RNA in {good, mid}. "
            "Final mid: Interface=good, Protein=bad (any Protein-RNA); or Interface=Protein=good "
            "but Protein-RNA=bad; or other mixed with ≤1 of {Interface, Protein} bad. "
            "Final bad: Interface bad, or both Interface and Protein bad."
        ),
        "entity_groups": AUDIT_ENTITY_METRIC_GROUPS,
        "af3_combinations": af3_combos,
        "final_tier_match_rate": {"rate": match_rate, "n": len(match_scores)},
        "per_eid": per_eid,
        "final_good_ids": final_good,
        "final_mid_ids": final_mid,
        "final_bad_ids": final_bad,
        "insufficient_ids": insufficient,
    }


def _audit_af3_plot_category_specs():
    """Audit list keys and short labels: ia…ic_ca, iia…, iiia…, iva… (20 categories)."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return []
    if len(_AUDIT_QUAD_PREFIXES) != len(AUDIT_AF3_PLOT_QUAD_ROMANS):
        return []
    specs = []
    for pref, roman in zip(_AUDIT_QUAD_PREFIXES, AUDIT_AF3_PLOT_QUAD_ROMANS):
        for sfx, sfx_lbl in AUDIT_AF3_PLOT_CATEGORY_SUFFIXES:
            specs.append((f"{pref}{sfx}", f"{roman}{sfx_lbl}"))
    return specs


def _audit_coerce_af3_plot_value(value):
    """Map final tier strings to ordinal 1..3; pass through finite floats."""
    if value is None:
        return None
    if isinstance(value, str):
        sev = _AUDIT_TIER_SEVERITY.get(value)
        return float(sev + 1) if sev is not None else None
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _audit_collect_af3_category_plot_series(audit, eid_to_value):
    """Per audit category: list of scalar values (one list per category label)."""
    labels = []
    series = []
    for audit_key, lbl in _audit_af3_plot_category_specs():
        labels.append(lbl)
        vals = []
        for rec in audit.get(audit_key) or []:
            eid = str(rec.get("exp_db_id", "")).strip()
            if not eid:
                continue
            v = _audit_coerce_af3_plot_value(eid_to_value.get(eid))
            if v is not None:
                vals.append(v)
        series.append(vals)
    return labels, series


def _audit_collect_af3_quadrant_plot_series(audit, eid_to_value):
    """Per quadrant (i–iv): pooled values from ``quad_*_all`` (no DockQ/CRMSD sub-bins)."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return [], []
    if len(_AUDIT_QUAD_PREFIXES) != len(AUDIT_AF3_PLOT_QUAD_ROMANS):
        return [], []
    labels = list(AUDIT_AF3_PLOT_QUAD_ROMANS)
    series = []
    for pref in _AUDIT_QUAD_PREFIXES:
        vals = []
        for rec in audit.get(f"{pref}_all") or []:
            eid = str(rec.get("exp_db_id", "")).strip()
            if not eid:
                continue
            v = _audit_coerce_af3_plot_value(eid_to_value.get(eid))
            if v is not None:
                vals.append(v)
        series.append(vals)
    return labels, series


def _audit_collect_af3_crmsd_bin_plot_series(audit, by_eid, eid_to_value):
    """Per CRMSD bin (<2, [2,10), ≥10 Å): values for audit-listed exp_db_id (once each)."""
    labels = list(AUDIT_CRMSD_BIN_ROW_LABELS)
    series = [[], [], []]
    if not by_eid:
        return labels, series
    for eid in _audit_unique_eids_from_audit(audit):
        v = _audit_coerce_af3_plot_value(eid_to_value.get(eid))
        if v is None:
            continue
        drow = by_eid.get(eid)
        if not drow:
            continue
        fv = _audit_extract_motif_bin_metrics(drow)
        bix = _crmsd_bin_index(fv.get("Complex_RMSD"))
        if bix is None:
            continue
        series[bix].append(v)
    return labels, series


def _audit_normalize_af3_metric_for_plot(score_by_eid, *, sentinel_threshold=-99.0):
    """
    Map sentinel scores (e.g. -100) to 0, then min–max scale finite values to [0, 1]
    over the audit population used for plotting.
    """
    pre = {}
    for eid, v in (score_by_eid or {}).items():
        if v is None or not np.isfinite(v):
            continue
        vf = float(v)
        pre[str(eid).strip()] = 0.0 if vf <= sentinel_threshold else vf
    if not pre:
        return {}
    lo, hi = min(pre.values()), max(pre.values())
    if hi <= lo:
        return {eid: 0.0 for eid in pre}
    return {eid: (v - lo) / (hi - lo) for eid, v in pre.items()}


def _audit_binary_feature_mean_std(series):
    """Per category: mean and sample SD of 0/1 values (fraction in [0, 1])."""
    means, stds = [], []
    for vals in series or []:
        v = _audit_finite_values(vals)
        if v.size == 0:
            means.append(np.nan)
            stds.append(0.0)
        elif v.size == 1:
            means.append(float(v[0]))
            stds.append(0.0)
        else:
            means.append(float(np.mean(v)))
            stds.append(float(np.std(v, ddof=1)))
    return means, stds


def _audit_feature_boxplot_ylim(values, *, lo_pct=5.0, hi_pct=92.0):
    """
    Y limits for raw feature boxplots: cap axis at percentiles so one outlier
    does not compress the main distribution (outliers still drawn as fliers).
    """
    v = _audit_finite_values(values)
    if v.size == 0:
        return None
    if v.size < 8:
        lo, hi = float(np.min(v)), float(np.max(v))
    else:
        lo = float(np.percentile(v, lo_pct))
        hi = float(np.percentile(v, hi_pct))
    if hi <= lo:
        pad = max(abs(hi) * 0.05, 0.5) if hi != 0 else 0.5
        return lo - pad, hi + pad
    pad = max(0.04 * (hi - lo), 0.5)
    y0 = lo - pad
    if np.all(v >= 0):
        y0 = max(0.0, y0)
    return y0, hi + pad


def _audit_af3_plot_file_stem(prefix):
    return prefix.replace("-", "_").replace(" ", "_")


def _audit_plot_af3_boxplot(
    plot_creator,
    table_source,
    labels,
    series,
    plot_stem,
    *,
    ylabel=None,
    ytick_positions=None,
    ytick_labels=None,
    y_axis_limit=None,
    xlabel=None,
    figsize=None,
    rotation=90,
):
    if not labels or not any(len(v) > 0 for v in series):
        return
    plot_creator.plot_af3_audit_category_boxplot(
        table_source,
        labels,
        series,
        plot_stem,
        ylabel=ylabel,
        ytick_positions=ytick_positions,
        ytick_labels=ytick_labels,
        y_axis_limit=y_axis_limit,
        xlabel=xlabel,
        figsize=figsize,
        rotation=rotation,
    )


def _audit_plot_af3_entity_tier_boxplot(
    plot_creator, table_source, audit, tier_by_eid, plot_stem, ylabel, *,
    by_eid=None, collect_series=None, xlabel=None, figsize=None, rotation=90,
):
    if collect_series is _audit_collect_af3_crmsd_bin_plot_series:
        labels, series = collect_series(audit, by_eid, tier_by_eid)
    elif collect_series is not None:
        labels, series = collect_series(audit, tier_by_eid)
    else:
        labels, series = _audit_collect_af3_category_plot_series(audit, tier_by_eid)
    _audit_plot_af3_boxplot(
        plot_creator,
        table_source,
        labels,
        series,
        plot_stem,
        ylabel=ylabel,
        ytick_positions=[1, 2, 3],
        ytick_labels=["bad", "mid", "good"],
        xlabel=xlabel,
        figsize=figsize,
        rotation=rotation,
    )


def _audit_plot_af3_metric_value_boxplot(
    plot_creator, table_source, audit, by_eid, metric_key, *,
    stem_suffix="by_audit_category",
    collect_series=None,
    xlabel=None,
    figsize=None,
    rotation=90,
):
    if not by_eid:
        return
    raw_by_eid = {}
    for eid, drow in by_eid.items():
        fv = _audit_extract_motif_bin_metrics(drow)
        v = fv.get(metric_key)
        if np.isfinite(v):
            raw_by_eid[str(eid).strip()] = float(v)

    ytick_positions = None
    ytick_labels = None
    if metric_key in AUDIT_AF3_PLOT_BINARY_FEATURE_METRICS:
        value_by_eid = raw_by_eid
        ylabel = f"{metric_key} (mean fraction)"
        y_axis_limit = (0.0, 1.0)
        ytick_positions = [0, 0.5, 1]
        ytick_labels = ["0", "0.5", "1"]
    elif metric_key in AUDIT_AF3_PLOT_FEATURE_METRICS:
        value_by_eid = raw_by_eid
        ylabel = metric_key
        y_axis_limit = None
    else:
        value_by_eid = _audit_normalize_af3_metric_for_plot(raw_by_eid)
        ylabel = f"{metric_key} (normalized 0–1)"
        y_axis_limit = (0.0, 1.0)

    if collect_series is _audit_collect_af3_crmsd_bin_plot_series:
        labels, series = collect_series(audit, by_eid, value_by_eid)
    elif collect_series is not None:
        labels, series = collect_series(audit, value_by_eid)
    else:
        labels, series = _audit_collect_af3_category_plot_series(audit, value_by_eid)

    stem = f"af3_{_audit_af3_plot_file_stem(metric_key)}_{stem_suffix}"
    if metric_key in AUDIT_AF3_PLOT_BINARY_FEATURE_METRICS:
        means, stds = _audit_binary_feature_mean_std(series)
        plot_creator.plot_af3_audit_category_mean_std(
            table_source,
            labels,
            means,
            stds,
            stem,
            ylabel=ylabel,
            y_axis_limit=y_axis_limit,
            ytick_positions=ytick_positions,
            ytick_labels=ytick_labels,
            xlabel=xlabel,
            figsize=figsize,
            rotation=rotation,
        )
        return

    if y_axis_limit is None and metric_key in AUDIT_AF3_PLOT_FEATURE_METRICS:
        flat = [x for s in series for x in s]
        y_axis_limit = _audit_feature_boxplot_ylim(flat)

    _audit_plot_af3_boxplot(
        plot_creator,
        table_source,
        labels,
        series,
        stem,
        ylabel=ylabel,
        y_axis_limit=y_axis_limit,
        ytick_positions=ytick_positions,
        ytick_labels=ytick_labels,
        xlabel=xlabel,
        figsize=figsize,
        rotation=rotation,
    )


def _audit_quadrant_scatter_passes_filter(fv, filter_key):
    """Subset audit PDBs for quadrant scatter variants."""
    if not filter_key:
        return True
    if filter_key == "has_RNA_MSA_0":
        v = fv.get("has_RNA_MSA")
        return np.isfinite(v) and float(v) < 0.5
    if filter_key == "has_RNA_MSA":
        v = fv.get("has_RNA_MSA")
        return np.isfinite(v) and float(v) >= 0.5
    if filter_key == "rbp_pdb_ge1":
        v = fv.get("RBP_PDB_n")
        return np.isfinite(v) and float(v) >= 1.0
    if filter_key == "rbp_pdb_gt1":
        v = fv.get("RBP_PDB_n")
        return np.isfinite(v) and float(v) > 1.0
    if filter_key == "rbp_pdb_eq1":
        v = fv.get("RBP_PDB_n")
        return np.isfinite(v) and float(v) == 1.0
    if filter_key == "rbp_pdb_eq0":
        v = fv.get("RBP_PDB_n")
        return np.isfinite(v) and float(v) == 0.0
    if filter_key == "rna_chains_gt1":
        v = fv.get("RNAChain_Count")
        return np.isfinite(v) and float(v) > 1.0
    if filter_key == "rna_chains_eq1":
        v = fv.get("RNAChain_Count")
        return np.isfinite(v) and float(v) == 1.0
    return True


def _audit_rna_feature_influence_passes(fv, filter_key):
    """Stratum filter for RNA Shannon / MFE quadrant breakdown (single key or AND tuple)."""
    if not filter_key:
        return True
    if isinstance(filter_key, (tuple, list)):
        return all(_audit_rna_feature_influence_passes(fv, k) for k in filter_key)
    if isinstance(filter_key, str) and filter_key.startswith("len_"):
        rna_len = fv.get("RNA_length_longest")
        if not np.isfinite(rna_len):
            return False
        x = float(rna_len)
        if filter_key == "len_0_40":
            return 0.0 <= x <= 40.0
        if filter_key == "len_41_100":
            return 41.0 <= x <= 100.0
        if filter_key == "len_gt_100":
            return x > 100.0
        return False
    return _audit_quadrant_scatter_passes_filter(fv, filter_key)


def _audit_rna_feature_combination_label(filter_keys):
    if isinstance(filter_keys, (tuple, list)):
        return " AND ".join(_AUDIT_RNA_INFLUENCE_FILTER_LABELS.get(k, k) for k in filter_keys)
    return _AUDIT_RNA_INFLUENCE_FILTER_LABELS.get(filter_keys, str(filter_keys))


def _audit_iter_rna_feature_influence_combinations():
    """All 3×2×2×3 AND combinations of RBP / chains / MSA / RNA length bins."""
    from itertools import product

    combo_idx = 0
    for keys in product(*_AUDIT_RNA_INFLUENCE_COMBO_DIMENSIONS):
        combo_idx += 1
        keys_t = tuple(keys)
        yield (
            "__".join(keys_t),
            _audit_rna_feature_combination_label(keys_t),
            keys_t,
            combo_idx,
        )


def _audit_metric_summary_stats(vals):
    v = _audit_finite_values(vals)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": std,
    }


def compute_audit_quadrant_rna_feature_influence_breakdown(rna_metrics, select_table, audit):
    """
    Per quadrant (i–iv): mean / median / SD of RNA_Shannon_Entropy_K2, RNA_MFE_Value, and
    af3_rna_pLDDT_avg (0–1) for each AND combination of RBP templates {0, 1, >1},
    RNAChain_Count, has_RNA_MSA, RNALength bin. Combinations with zero audit PDBs in all
    quadrants are omitted.
    """
    if not audit or "quad_i_all" not in audit:
        return None
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES, _AUDIT_QUAD_TICKLABELS
    except Exception:
        return None
    if len(_AUDIT_QUAD_PREFIXES) != len(AUDIT_AF3_PLOT_QUAD_ROMANS):
        return None
    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    metrics = AUDIT_QUADRANT_RNA_FEATURE_BREAKDOWN_METRICS
    strata_out = []
    rows_out = []

    for stratum_key, stratum_label, filter_key, combo_idx in _audit_iter_rna_feature_influence_combinations():
        buckets = {
            roman: {m: [] for m in metrics}
            for roman in AUDIT_AF3_PLOT_QUAD_ROMANS
        }
        for pref, roman in zip(_AUDIT_QUAD_PREFIXES, AUDIT_AF3_PLOT_QUAD_ROMANS):
            for rec in audit.get(f"{pref}_all") or []:
                eid = str(rec.get("exp_db_id", "")).strip()
                drow = by_eid.get(eid)
                if not drow:
                    continue
                fv = _audit_extract_motif_bin_metrics(drow)
                if not _audit_rna_feature_influence_passes(fv, filter_key):
                    continue
                for mkey in metrics:
                    val = fv.get(mkey)
                    if np.isfinite(val):
                        buckets[roman][mkey].append(float(val))

        total_samples = sum(len(buckets[r][m]) for r in buckets for m in metrics)
        if total_samples == 0:
            continue

        strata_out.append({
            "key": stratum_key,
            "label": stratum_label,
            "index": int(combo_idx),
        })
        for roman, qlbl in zip(AUDIT_AF3_PLOT_QUAD_ROMANS, _AUDIT_QUAD_TICKLABELS):
            row = {
                "stratum_key": stratum_key,
                "stratum_label": stratum_label,
                "quadrant": roman,
                "quadrant_label": qlbl,
            }
            for mkey in metrics:
                stats = _audit_metric_summary_stats(buckets[roman][mkey])
                row[f"{mkey}_n"] = stats["n"]
                row[f"{mkey}_mean"] = stats["mean"]
                row[f"{mkey}_median"] = stats["median"]
                row[f"{mkey}_std"] = stats["std"]
            rows_out.append(row)

    if not rows_out:
        return None

    return {
        "caption": (
            "RNA_Shannon_Entropy_K2, RNA_MFE_Value, and af3_rna_pLDDT_avg (0–1) by audit quadrant (i–iv). "
            "One table per AND combination of: RBP_PDB_Templates {0, 1, >1}, "
            "RNAChain_Count {=1, >1}, has_RNA_MSA {0, 1}, RNALength_longest {0–40, 41–100, >100}. "
            "Combinations with no audit PDBs in any quadrant are omitted. "
            "Quadrants pool all DockQ/CRMSD sub-bins. "
            "RNA entity tier reference cutpoints (DockQ consensus): "
            "good ≥ 0.775, mid [0.550, 0.775), bad < 0.550."
        ),
        "metrics": list(metrics),
        "strata": strata_out,
        "rows": rows_out,
    }


def _audit_build_quadrant_feature_scatter_rows(audit, by_eid, x_metric, y_metric, *, filter_key=None):
    """One row per audit PDB with finite x/y and quadrant label (i–iv)."""
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES, _AUDIT_QUAD_TICKLABELS
    except Exception:
        return []
    if len(_AUDIT_QUAD_PREFIXES) != len(AUDIT_AF3_PLOT_QUAD_ROMANS):
        return []
    rows = []
    for pref, roman, qlbl in zip(
        _AUDIT_QUAD_PREFIXES, AUDIT_AF3_PLOT_QUAD_ROMANS, _AUDIT_QUAD_TICKLABELS
    ):
        for rec in audit.get(f"{pref}_all") or []:
            eid = str(rec.get("exp_db_id", "")).strip()
            drow = (by_eid or {}).get(eid)
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            if not _audit_quadrant_scatter_passes_filter(fv, filter_key):
                continue
            x = fv.get(x_metric)
            y = fv.get(y_metric)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            rna_len_max = AUDIT_AF3_QUADRANT_SCATTER_RNA_LENGTH_MAX
            if rna_len_max is not None:
                if x_metric == "RNA_length_longest" and float(x) > float(rna_len_max):
                    continue
                if y_metric == "RNA_length_longest" and float(y) > float(rna_len_max):
                    continue
            rows.append({
                "exp_db_id": eid,
                "quadrant": roman,
                "quadrant_label": f"{roman}: {qlbl}",
                x_metric: float(x),
                y_metric: float(y),
            })
    return rows


def _audit_plot_af3_quadrant_feature_scatter(
    plot_creator,
    table_source,
    audit,
    by_eid,
    x_metric,
    y_metric,
    *,
    stem_suffix="",
    filter_key=None,
):
    rows = _audit_build_quadrant_feature_scatter_rows(
        audit, by_eid, x_metric, y_metric, filter_key=filter_key
    )
    if not rows:
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    stem = (
        f"af3_scatter_{_audit_af3_plot_file_stem(x_metric)}_vs_"
        f"{_audit_af3_plot_file_stem(y_metric)}_by_quadrant{stem_suffix or ''}"
    )
    title_parts = [f"{y_metric} vs {x_metric} by quadrant"]
    if filter_key:
        title_parts.append(AUDIT_AF3_QUADRANT_SCATTER_FILTER_TITLES.get(filter_key, filter_key))
    if AUDIT_AF3_QUADRANT_SCATTER_RNA_LENGTH_MAX is not None and (
        x_metric == "RNA_length_longest" or y_metric == "RNA_length_longest"
    ):
        title_parts.append(f"RNA length ≤ {int(AUDIT_AF3_QUADRANT_SCATTER_RNA_LENGTH_MAX)}")
    plot_creator.plot_af3_audit_quadrant_scatter(
        table_source,
        df,
        x_metric,
        y_metric,
        stem,
        xlabel=x_metric,
        ylabel=y_metric,
        title="; ".join(title_parts),
    )


def plot_audit_af3_metrics_figures(
    plot_creator, table_source, audit, rna_metrics, entity_predictions
):
    """Boxplots in ``af3_metrics``: entity tiers and metric values vs audit category, quadrant, CRMSD."""
    if not audit or not plot_creator or entity_predictions is None:
        return

    by_eid = _audit_load_prediction_by_eid(rna_metrics, table_source)
    crmsd_xlabel = "CRMSD bin"
    crmsd_figsize = (8.0, 6.2)
    quad_xlabel = "Quadrant"
    quad_figsize = (7.0, 6.2)
    per_eid = entity_predictions.get("per_eid") or []
    for plot_label, entity_key, ylabel in AUDIT_AF3_PLOT_ENTITY_TIER_SPECS:
        tier_by_eid = {}
        for rec in per_eid:
            eid = str(rec.get("exp_db_id", "")).strip()
            if not eid:
                continue
            if entity_key == "final_tier":
                tier = rec.get("final_tier")
            else:
                tier = (rec.get("entities") or {}).get(entity_key)
            if tier is not None:
                tier_by_eid[eid] = tier
        stem = f"af3_{_audit_af3_plot_file_stem(plot_label)}_entity_tier_by_audit_category"
        if plot_label == "final":
            stem = "af3_final_tier_by_audit_category"
        _audit_plot_af3_entity_tier_boxplot(
            plot_creator, table_source, audit, tier_by_eid, stem, ylabel
        )
        quad_stem = f"af3_{_audit_af3_plot_file_stem(plot_label)}_entity_tier_by_quadrant"
        if plot_label == "final":
            quad_stem = "af3_final_tier_by_quadrant"
        _audit_plot_af3_entity_tier_boxplot(
            plot_creator,
            table_source,
            audit,
            tier_by_eid,
            quad_stem,
            ylabel,
            collect_series=_audit_collect_af3_quadrant_plot_series,
            xlabel=quad_xlabel,
            figsize=quad_figsize,
            rotation=0,
        )
        crmsd_stem = f"af3_{_audit_af3_plot_file_stem(plot_label)}_entity_tier_by_crmsd_bin"
        if plot_label == "final":
            crmsd_stem = "af3_final_tier_by_crmsd_bin"
        _audit_plot_af3_entity_tier_boxplot(
            plot_creator,
            table_source,
            audit,
            tier_by_eid,
            crmsd_stem,
            ylabel,
            by_eid=by_eid,
            collect_series=_audit_collect_af3_crmsd_bin_plot_series,
            xlabel=crmsd_xlabel,
            figsize=crmsd_figsize,
            rotation=0,
        )

    for metric_key in AUDIT_AF3_PLOT_VALUE_METRICS:
        _audit_plot_af3_metric_value_boxplot(
            plot_creator, table_source, audit, by_eid, metric_key
        )
        _audit_plot_af3_metric_value_boxplot(
            plot_creator,
            table_source,
            audit,
            by_eid,
            metric_key,
            stem_suffix="by_quadrant",
            collect_series=_audit_collect_af3_quadrant_plot_series,
            xlabel=quad_xlabel,
            figsize=quad_figsize,
            rotation=0,
        )
        _audit_plot_af3_metric_value_boxplot(
            plot_creator,
            table_source,
            audit,
            by_eid,
            metric_key,
            stem_suffix="by_crmsd_bin",
            collect_series=_audit_collect_af3_crmsd_bin_plot_series,
            xlabel=crmsd_xlabel,
            figsize=crmsd_figsize,
            rotation=0,
        )

    for metric_key in AUDIT_AF3_PLOT_FEATURE_METRICS:
        _audit_plot_af3_metric_value_boxplot(
            plot_creator, table_source, audit, by_eid, metric_key
        )
        _audit_plot_af3_metric_value_boxplot(
            plot_creator,
            table_source,
            audit,
            by_eid,
            metric_key,
            stem_suffix="by_quadrant",
            collect_series=_audit_collect_af3_quadrant_plot_series,
            xlabel=quad_xlabel,
            figsize=quad_figsize,
            rotation=0,
        )
        _audit_plot_af3_metric_value_boxplot(
            plot_creator,
            table_source,
            audit,
            by_eid,
            metric_key,
            stem_suffix="by_crmsd_bin",
            collect_series=_audit_collect_af3_crmsd_bin_plot_series,
            xlabel=crmsd_xlabel,
            figsize=crmsd_figsize,
            rotation=0,
        )

    for x_metric, y_metric, stem_suffix, filter_key in AUDIT_AF3_QUADRANT_SCATTER_SPECS:
        _audit_plot_af3_quadrant_feature_scatter(
            plot_creator,
            table_source,
            audit,
            by_eid,
            x_metric,
            y_metric,
            stem_suffix=stem_suffix,
            filter_key=filter_key,
        )


def compute_audit_consensus_pdb_tier_lists(rna_metrics, select_table, audit, consensus_tiers):
    """
    Classify audit-listed exp_db_id entries into good/mid/bad per metric using
    Classification-assigned consensus tier cutpoints; AF3-primary buckets with subdivisions.
    """
    base = _audit_build_consensus_classifications(rna_metrics, select_table, audit, consensus_tiers)
    if not base:
        return None

    eids = base["eids"]
    classifications = base["classifications"]
    tier_by_metric = base["tier_by_metric"]
    metric_keys = base["metric_keys"]

    per_metric = []
    for mkey in metric_keys:
        tier_row = tier_by_metric[mkey]
        good_ids, mid_ids, bad_ids, na_ids = [], [], [], []
        for eid in eids:
            tier = classifications[eid].get(mkey)
            if tier is None:
                na_ids.append(eid)
                continue
            if tier == "good":
                good_ids.append(eid)
            elif tier == "mid":
                mid_ids.append(eid)
            else:
                bad_ids.append(eid)

        dir_lbl = "↑ good" if tier_row.get("higher_is_good") else "↓ good"
        per_metric.append({
            "metric_key": mkey,
            "label": _audit_metric_display_label(mkey),
            "direction": dir_lbl,
            "good_band": tier_row.get("good_band"),
            "mid_band": tier_row.get("mid_band"),
            "bad_band": tier_row.get("bad_band"),
            "good_ids": good_ids,
            "mid_ids": mid_ids,
            "bad_ids": bad_ids,
            "na_ids": na_ids,
        })

    af3_keys = [k for k in metric_keys if k.startswith("af3_")]
    other_keys = [k for k in metric_keys if not k.startswith("af3_")]

    def _eids_uniform_tier(keys, tier):
        if not keys:
            return []
        out = []
        for eid in eids:
            cls = classifications[eid]
            if any(k not in cls for k in keys):
                continue
            if all(cls[k] == tier for k in keys):
                out.append(eid)
        return out

    af3_all_good = _eids_uniform_tier(af3_keys, "good")
    af3_all_mid = _eids_uniform_tier(af3_keys, "mid")
    af3_all_bad = _eids_uniform_tier(af3_keys, "bad")

    by_af3_tier = {}
    for af3_tier, af3_bucket in (
        ("good", af3_all_good),
        ("mid", af3_all_mid),
        ("bad", af3_all_bad),
    ):
        blocks = []
        for mkey in other_keys:
            tier_row = tier_by_metric[mkey]
            good_ids, mid_ids, bad_ids, na_ids = [], [], [], []
            for eid in af3_bucket:
                tier = classifications[eid].get(mkey)
                if tier is None:
                    na_ids.append(eid)
                    continue
                if tier == "good":
                    good_ids.append(eid)
                elif tier == "mid":
                    mid_ids.append(eid)
                else:
                    bad_ids.append(eid)
            dir_lbl = "↑ good" if tier_row.get("higher_is_good") else "↓ good"
            blocks.append({
                "metric_key": mkey,
                "label": _audit_metric_display_label(mkey),
                "direction": dir_lbl,
                "good_band": tier_row.get("good_band"),
                "mid_band": tier_row.get("mid_band"),
                "bad_band": tier_row.get("bad_band"),
                "good_ids": good_ids,
                "mid_ids": mid_ids,
                "bad_ids": bad_ids,
                "na_ids": na_ids,
            })
        by_af3_tier[af3_tier] = blocks

    af3_labels = ", ".join(_audit_metric_display_label(k) for k in af3_keys)
    return {
        "caption": (
            "Audit-listed exp_db_id (quad_*_all) classified with classification-assigned "
            "consensus tier cutpoints (DockQ / RNA / protein / CRMSD per metric map). "
            "Primary buckets: ALL af3_* metrics same tier (good/mid/bad). "
            "Other metrics are subdivided good/mid/bad within each af3 bucket."
        ),
        "metric_keys": metric_keys,
        "af3_metric_keys": af3_keys,
        "other_metric_keys": other_keys,
        "af3_metric_labels": af3_labels,
        "af3_all_good_ids": af3_all_good,
        "af3_all_mid_ids": af3_all_mid,
        "af3_all_bad_ids": af3_all_bad,
        "by_af3_tier": by_af3_tier,
        "per_metric": per_metric,
    }


def _ordinal_xy_from_four_lists(vals_per_bin):
    """vals_per_bin: four lists of scalar metric values -> x vector, y ordinal 1..4."""
    xs = []
    ys = []
    for qi, vl in enumerate(vals_per_bin):
        ov = float(qi + 1)
        for v in vl:
            if np.isfinite(v):
                xs.append(float(v))
                ys.append(ov)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _dockq_bin_validation_note(pooled_rho, rho_a, rho_b, rho_c):
    """Compare pooled quadrant-ordinal rho to Spearman in DockQ (a), (b), (c) slices only."""
    slice_rhos = [rho_a, rho_b, rho_c]
    finite = [r for r in slice_rhos if r is not None and np.isfinite(r)]
    if len(finite) < 2:
        return "insufficient slice rhos"
    if pooled_rho is None or not np.isfinite(pooled_rho):
        return f"{len(finite)}/3 DockQ slices finite rho"
    ps = int(np.sign(pooled_rho))
    if ps == 0:
        return f"pooled rho~0; {len(finite)}/3 slices finite"
    agree = sum(1 for r in finite if int(np.sign(r)) == ps or int(np.sign(r)) == 0)
    if agree == len(finite):
        return f"sign match pooled ({len(finite)}/3 DockQ bins)"
    return f"mixed vs pooled ({agree}/{len(finite)} slice match)"


def compute_audit_fold_quadrant_and_crmsd_effects(rna_metrics, select_table, audit):
    """
    Additional stratifications on the same core/AF3 engineered features:

    1) Ordinal fold quadrant (i)->(ii)->(iii)->(iv) on quad_*_all: Cohen d (iv vs i),
       Spearman(metric, quadrant ordinal 1..4), composite |d||rho|.
       Validation: Spearman in DockQ (a), (b), (c) slices only (same quadrant ordinal).

    2) Binary quadrant (i) vs (iv) on quad_*_all: Cohen d only (no ordinal middle).

    3) Binary Complex RMSD among DockQ failures: *_c_ca (<10 Å) vs *_c_cb (>=10 Å), Cohen d only.

    4) Binary RNA global fold: accurate RNA = quadrants (i)+(ii) vs inaccurate = (iii)+(iv) on quad_*_all.

    5) Binary protein global fold: accurate protein = (i)+(iii) vs inaccurate = (ii)+(iv) on quad_*_all.

    Code naming: c_ca = CRMSD <10 Å; c_cb = CRMSD >=10 Å (both within DockQ < 0.23).
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    quad_prefs = tuple(_AUDIT_QUAD_PREFIXES)
    if len(quad_prefs) != 4:
        return None

    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    core_keys = [k for k, _ in AUDIT_PROFILE_CORE_METRICS]
    af3_keys = [k for k, _ in AUDIT_PROFILE_AF3_METRICS]
    all_keys = core_keys + af3_keys

    def _push(recs, metric_bucket):
        """Append finite metric values into metric_bucket[k] for each key."""
        for rec in recs or []:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            for k in all_keys:
                val = fv[k]
                if np.isfinite(val):
                    metric_bucket[k].append(float(val))

    # Quadrant ordinal: metric lists per quadrant index 0..3 from quad_*_all
    quad_all_vals = [{k: [] for k in all_keys} for _ in range(4)]
    for qi, pref in enumerate(quad_prefs):
        bucket = {k: [] for k in all_keys}
        _push(audit.get(f"{pref}_all") or [], bucket)
        quad_all_vals[qi] = bucket

    # DockQ (a)/(b)/(c) slices × quadrant ordinal (validation rhos)
    dock_slices = ("_a", "_b", "_c")
    quad_dock_vals = {ds: [{k: [] for k in all_keys} for _ in range(4)] for ds in dock_slices}
    for ds in dock_slices:
        for qi, pref in enumerate(quad_prefs):
            bucket = {k: [] for k in all_keys}
            _push(audit.get(f"{pref}{ds}") or [], bucket)
            quad_dock_vals[ds][qi] = bucket

    # --- CRMSD failure split pools ---
    lt10 = {k: [] for k in all_keys}
    ge10 = {k: [] for k in all_keys}
    for pref in quad_prefs:
        b_lt10 = {k: [] for k in all_keys}
        _push(audit.get(f"{pref}_c_ca") or [], b_lt10)
        b_ge10 = {k: [] for k in all_keys}
        _push(audit.get(f"{pref}_c_cb") or [], b_ge10)
        for k in all_keys:
            lt10[k].extend(b_lt10[k])
            ge10[k].extend(b_ge10[k])

    rows_ord = []
    rows_bin_iv = []
    rows_rna_i_ii_vs_iii_iv = []
    rows_prot_i_iii_vs_ii_iv = []
    rows_crmsd = []

    for panel, tuples in (
        ("core", AUDIT_PROFILE_CORE_METRICS),
        ("af3", AUDIT_PROFILE_AF3_METRICS),
    ):
        for mkey, mlbl in tuples:
            lists_iv_ord = [quad_all_vals[qi][mkey] for qi in range(4)]
            stat_ord = _cohens_d_mean_c_minus_mean_a(lists_iv_ord[0], lists_iv_ord[3])
            x_ord, y_ord = _ordinal_xy_from_four_lists(lists_iv_ord)
            rho_ord, rho_p = _spearman_rho_and_p(x_ord, y_ord)
            abs_rho = abs(float(rho_ord)) if rho_ord is not None and np.isfinite(rho_ord) else None
            comp = None
            if stat_ord.get("abs_cohen_d") is not None and abs_rho is not None:
                comp = float(stat_ord["abs_cohen_d"]) * float(abs_rho)

            rho_a = _spearman_rho_and_p(*_ordinal_xy_from_four_lists([qd[mkey] for qd in quad_dock_vals["_a"]]))[0]
            rho_b = _spearman_rho_and_p(*_ordinal_xy_from_four_lists([qd[mkey] for qd in quad_dock_vals["_b"]]))[0]
            rho_c = _spearman_rho_and_p(*_ordinal_xy_from_four_lists([qd[mkey] for qd in quad_dock_vals["_c"]]))[0]

            rows_ord.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    "spearman_rho_ord": rho_ord,
                    "spearman_p_ord": rho_p,
                    "abs_spearman_rho_ord": abs_rho,
                    "n_ord_levels": int(x_ord.size),
                    "composite_abs_d_rho": comp,
                    "rho_dockq_slice_a": rho_a,
                    "rho_dockq_slice_b": rho_b,
                    "rho_dockq_slice_c": rho_c,
                    "dockq_slice_validation_note": _dockq_bin_validation_note(rho_ord, rho_a, rho_b, rho_c),
                    **stat_ord,
                }
            )

            stat_iv = _cohens_d_mean_c_minus_mean_a(lists_iv_ord[0], lists_iv_ord[3])
            rows_bin_iv.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    **stat_iv,
                }
            )

            rna_acc = lists_iv_ord[0] + lists_iv_ord[1]
            rna_inacc = lists_iv_ord[2] + lists_iv_ord[3]
            stat_rna = _cohens_d_mean_c_minus_mean_a(rna_acc, rna_inacc)
            rows_rna_i_ii_vs_iii_iv.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    **stat_rna,
                }
            )

            prot_acc = lists_iv_ord[0] + lists_iv_ord[2]
            prot_inacc = lists_iv_ord[1] + lists_iv_ord[3]
            stat_prot = _cohens_d_mean_c_minus_mean_a(prot_acc, prot_inacc)
            rows_prot_i_iii_vs_ii_iv.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    **stat_prot,
                }
            )

            stat_cr = _cohens_d_mean_c_minus_mean_a(lt10[mkey], ge10[mkey])
            rows_crmsd.append(
                {
                    "panel": panel,
                    "metric_key": mkey,
                    "label": mlbl,
                    **stat_cr,
                }
            )

    rows_ord.sort(
        key=lambda r: (
            r["composite_abs_d_rho"] is None,
            -(r["composite_abs_d_rho"] if r["composite_abs_d_rho"] is not None else 0.0),
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            -(r["abs_spearman_rho_ord"] if r["abs_spearman_rho_ord"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_ord, start=1):
        r["rank"] = i

    rows_bin_iv.sort(
        key=lambda r: (
            r["cohen_d"] is None,
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_bin_iv, start=1):
        r["rank"] = i

    rows_rna_i_ii_vs_iii_iv.sort(
        key=lambda r: (
            r["cohen_d"] is None,
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_rna_i_ii_vs_iii_iv, start=1):
        r["rank"] = i

    rows_prot_i_iii_vs_ii_iv.sort(
        key=lambda r: (
            r["cohen_d"] is None,
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_prot_i_iii_vs_ii_iv, start=1):
        r["rank"] = i

    rows_crmsd.sort(
        key=lambda r: (
            r["cohen_d"] is None,
            -(r["abs_cohen_d"] if r["abs_cohen_d"] is not None else 0.0),
            str(r.get("panel", "")),
            str(r.get("metric_key", "")),
        )
    )
    for i, r in enumerate(rows_crmsd, start=1):
        r["rank"] = i

    return {
        "audit_effect_fold_quadrant_ordinal_i_iv": {
            "title": "Fold quadrant ordinal (i)->(ii)->(iii)->(iv) on quad_*_all",
            "d_compare": "Cohen d: (iv) minus (i); Spearman: metric vs quadrant ordinal 1..4 (same pools).",
            "validation": "rho_DQA/B/C: Spearman(metric, quadrant ordinal) restricted to DockQ (a)/(b)/(c) slices.",
            "rows": rows_ord,
        },
        "audit_effect_fold_quadrant_binary_i_vs_iv": {
            "title": "Fold quadrant binary: (i) vs (iv) only (quad_*_all)",
            "d_compare": "Cohen d = (mean_iv - mean_i) / s_pooled; no intermediate quadrants in the contrast.",
            "rows": rows_bin_iv,
        },
        "audit_effect_rna_accurate_i_ii_vs_inaccurate_iii_iv": {
            "title": "RNA accuracy (combined): (i)+(ii) vs (iii)+(iv) on quad_*_all",
            "d_compare": "Accurate RNA = RNA_LDDT > 0.85 (quadrants i and ii); inaccurate = RNA_LDDT <= 0.85 "
            "(iii and iv). Cohen d = (mean_inaccurate - mean_accurate) / s_pooled.",
            "rows": rows_rna_i_ii_vs_iii_iv,
        },
        "audit_effect_protein_accurate_i_iii_vs_inaccurate_ii_iv": {
            "title": "Protein accuracy (combined): (i)+(iii) vs (ii)+(iv) on quad_*_all",
            "d_compare": "Accurate protein = Protein_LDDT > 0.85 AND finite Protein_TM > 0.85 (quadrants i and iii); "
            "inaccurate = otherwise (ii and iv). Cohen d = (mean_inaccurate - mean_accurate) / s_pooled.",
            "rows": rows_prot_i_iii_vs_ii_iv,
        },
        "audit_effect_crmsd_fail_lt10_vs_ge10": {
            "title": "DockQ failures: Complex RMSD <10 Å (*_c_ca) vs >=10 Å (*_c_cb), quadrants pooled",
            "d_compare": "Cohen d = (mean_>=10Å - mean_<10Å) / s_pooled among finite Complex_RMSD failures.",
            "rows": rows_crmsd,
        },
    }


def compute_audit_interface_bin_profiles(
    rna_metrics, select_table, audit, bin_suffixes=None, pool_quadrants=False
):
    """
    Mean prediction-table metrics per audit bin (same quadrants × DockQ slices as outcome heatmap).

    ``bin_suffixes`` defaults to full ``AUDIT_BIN_SUFFIXES_ORDERED`` (includes CRMSD fail bins).
    Pass ``AUDIT_DOCKQ_ABC_BIN_SUFFIXES`` for DockQ (a)/(b)/(c) rows only.

    If ``pool_quadrants`` is True, each row pools records from all quadrants (i)–(iv) for that
    DockQ suffix, so rows are only (a)/(b)/(c) — no quadrant subclasses.

    Includes RNAChain_Count / ProteinChain_Count (parsed list lengths), RNAProtein_Ratio as
    longest RNALength / longest ProteinLength, and RNA_chain_load / Protein_chain_load =
    chain count × longest-chain length (crude polymer-extent index; alternatively consider
    total residues sum_j L_j or sqrt(count × L) for scale mixes).
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES, _AUDIT_QUAD_TICKLABELS
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    by_eid = _audit_load_prediction_by_eid(rna_metrics, select_table)
    if not by_eid:
        return None

    suffixes = tuple(bin_suffixes) if bin_suffixes is not None else AUDIT_BIN_SUFFIXES_ORDERED
    use_dockq_abc_labels = suffixes == AUDIT_DOCKQ_ABC_BIN_SUFFIXES

    row_labels = []
    bin_counts = []
    bin_rec_sources = []

    if pool_quadrants:
        for sj, suff in enumerate(suffixes):
            full_idx = AUDIT_BIN_SUFFIXES_ORDERED.index(suff)
            bin_title = (
                AUDIT_DOCKQ_ABC_ROW_BIN_TITLES[sj]
                if use_dockq_abc_labels
                else AUDIT_BIN_COLUMN_TITLES[full_idx]
            )
            row_labels.append(bin_title)
            merged = []
            for pref in _AUDIT_QUAD_PREFIXES:
                merged.extend(audit.get(f"{pref}{suff}") or [])
            bin_rec_sources.append(merged)
            bin_counts.append(len(merged))
    else:
        for qi, pref in enumerate(_AUDIT_QUAD_PREFIXES):
            qtitle = _AUDIT_QUAD_TICKLABELS[qi]
            for sj, suff in enumerate(suffixes):
                full_idx = AUDIT_BIN_SUFFIXES_ORDERED.index(suff)
                key = f"{pref}{suff}"
                bin_title = (
                    AUDIT_DOCKQ_ABC_ROW_BIN_TITLES[sj]
                    if use_dockq_abc_labels
                    else AUDIT_BIN_COLUMN_TITLES[full_idx]
                )
                row_labels.append(f"{qtitle}\n{bin_title}")
                recs = audit.get(key) or []
                bin_rec_sources.append(recs)
                bin_counts.append(len(recs))

    n_bins = len(bin_rec_sources)
    core_keys = [k for k, _ in AUDIT_PROFILE_CORE_METRICS]
    af3_keys = [k for k, _ in AUDIT_PROFILE_AF3_METRICS]

    core_mat = np.full((n_bins, len(core_keys)), np.nan)
    af3_mat = np.full((n_bins, len(af3_keys)), np.nan)
    frac_rna_msa = np.full(n_bins, np.nan)
    frac_prot_msa = np.full(n_bins, np.nan)
    frac_prot_pdb = np.full(n_bins, np.nan)
    frac_rbp_pdb = np.full(n_bins, np.nan)
    mean_rna_msa_d = np.full(n_bins, np.nan)
    mean_prot_msa_d = np.full(n_bins, np.nan)

    for bi, recs in enumerate(bin_rec_sources):
        if not recs:
            continue
        bucket_core = {k: [] for k in core_keys}
        bucket_af3 = {k: [] for k in af3_keys}
        prna, pprot, ppdb, rpdb = [], [], [], []
        for rec in recs:
            eid = rec.get("exp_db_id")
            if not eid:
                continue
            drow = by_eid.get(str(eid).strip())
            if not drow:
                continue
            fv = _audit_extract_motif_bin_metrics(drow)
            for k in core_keys:
                val = fv[k]
                if np.isfinite(val):
                    bucket_core[k].append(float(val))
            for k in af3_keys:
                val = fv[k]
                if np.isfinite(val):
                    bucket_af3[k].append(float(val))
            if np.isfinite(fv["has_RNA_msa"]):
                prna.append(fv["has_RNA_msa"])
            if np.isfinite(fv["has_Protein_msa"]):
                pprot.append(fv["has_Protein_msa"])
            if np.isfinite(fv["has_Protein_PDB"]):
                ppdb.append(fv["has_Protein_PDB"])
            if np.isfinite(fv["has_RBP_PDB"]):
                rpdb.append(fv["has_RBP_PDB"])

        for ci, ck in enumerate(core_keys):
            vals = bucket_core.get(ck) or []
            core_mat[bi, ci] = float(np.mean(vals)) if vals else np.nan

        for ai, ak in enumerate(af3_keys):
            vals = bucket_af3.get(ak) or []
            af3_mat[bi, ai] = float(np.mean(vals)) if vals else np.nan

        frac_rna_msa[bi] = float(np.mean(prna)) if prna else np.nan
        frac_prot_msa[bi] = float(np.mean(pprot)) if pprot else np.nan
        frac_prot_pdb[bi] = float(np.mean(ppdb)) if ppdb else np.nan
        frac_rbp_pdb[bi] = float(np.mean(rpdb)) if rpdb else np.nan

        rdep_vals = bucket_core.get("RNA_msa_depth") or []
        pdep_vals = bucket_core.get("Protein_msa_depth") or []
        mean_rna_msa_d[bi] = float(np.mean(rdep_vals)) if rdep_vals else np.nan
        mean_prot_msa_d[bi] = float(np.mean(pdep_vals)) if pdep_vals else np.nan

    return {
        "row_labels": row_labels,
        "bin_counts": bin_counts,
        "core_mat": core_mat,
        "core_col_labels": [lbl for _, lbl in AUDIT_PROFILE_CORE_METRICS],
        "af3_mat": af3_mat,
        "af3_col_labels": [lbl for _, lbl in AUDIT_PROFILE_AF3_METRICS],
        "frac_rna_msa": frac_rna_msa,
        "frac_prot_msa": frac_prot_msa,
        "frac_prot_pdb": frac_prot_pdb,
        "frac_rbp_pdb": frac_rbp_pdb,
        "mean_rna_msa_depth": mean_rna_msa_d,
        "mean_prot_msa_depth": mean_prot_msa_d,
    }


def build_sample_audit_figure_statistics(
    audit,
    profile,
    ac_cohen_d=None,
    fold_crmsd_effects=None,
    motif_bin_profiles=None,
    motif_quadrant_matrices=None,
    motif_score_summaries=None,
    crmsd_bin_profile=None,
    metric_tier_intervals=None,
    crmsd_metric_tier_intervals=None,
    consensus_metric_tier_intervals=None,
    consensus_pdb_tier_lists=None,
    entity_complex_predictions=None,
    rna_feature_quadrant_breakdown=None,
):
    """
    Tabular summaries for interface audit PNGs (DockQ / CRMSD stacks, outcome heatmap,
    bin-mean heatmaps). Optional ac_cohen_d adds complex-level Cohen's d (good vs bad DockQ)
    for core+AF3 features; fold_crmsd_effects adds quadrant-ordinal / binary / CRMSD-split tables.
    motif_bin_profiles: optional dict strat_key -> profile dict from compute_motif_bin_profiles().
    motif_quadrant_matrices: optional dict strat_key -> 4×5 mean-score matrix.
    motif_score_summaries: optional dict strat_key -> table-wide score distribution.
    Safe with profile=None (heatmap tables omitted).
    """
    try:
        from plots.plotCreator import _AUDIT_QUAD_PREFIXES, _AUDIT_QUAD_TICKLABELS
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None

    prefs = _AUDIT_QUAD_PREFIXES
    quads = list(_AUDIT_QUAD_TICKLABELS)
    tbl = audit.get("pred_table", "")

    rbp_rows = []
    for i, p in enumerate(prefs):
        na = len(audit.get(f"{p}_a") or [])
        nb = len(audit.get(f"{p}_b") or [])
        nc = len(audit.get(f"{p}_c") or [])
        ntot = na + nb + nc
        rbp_rows.append(
            {
                "quadrant": quads[i],
                "n_finite_RBP_DockQ": int(ntot),
                "n_DockQ_ge_0_8": int(na),
                "n_DockQ_0_23_to_0_8": int(nb),
                "n_DockQ_lt_0_23": int(nc),
                "frac_DockQ_ge_0_8": (na / ntot) if ntot else None,
                "frac_DockQ_mid": (nb / ntot) if ntot else None,
                "frac_DockQ_lt_0_23": (nc / ntot) if ntot else None,
            }
        )

    crmsd_rows = []
    for i, p in enumerate(prefs):
        ca_n = len(audit.get(f"{p}_c_ca") or [])
        cb_n = len(audit.get(f"{p}_c_cb") or [])
        n_fin = ca_n + cb_n
        crmsd_rows.append(
            {
                "quadrant": quads[i],
                "n_DockQ_failures_finite_CRMSD": int(n_fin),
                "n_CRMSD_lt_10_A": int(ca_n),
                "n_CRMSD_ge_10_A": int(cb_n),
                "frac_CRMSD_lt_10_A": (ca_n / n_fin) if n_fin else None,
                "frac_CRMSD_ge_10_A": (cb_n / n_fin) if n_fin else None,
            }
        )

    outcome_rows = []
    for i, p in enumerate(prefs):
        n_all = len(audit.get(f"{p}_all") or [])
        na = len(audit.get(f"{p}_a") or [])
        nb = len(audit.get(f"{p}_b") or [])
        nc = len(audit.get(f"{p}_c") or [])
        n_cb = len(audit.get(f"{p}_c_cb") or [])
        n_ca = len(audit.get(f"{p}_c_ca") or [])
        outcome_rows.append(
            {
                "quadrant": quads[i],
                "n_quadrant_fin_lddt": int(n_all),
                "count_DockQ_ge_0_8": int(na),
                "count_DockQ_mid": int(nb),
                "count_DockQ_lt_0_23": int(nc),
                "pct_of_quadrant_DockQ_ge_0_8": (100.0 * na / n_all) if n_all else None,
                "pct_of_quadrant_DockQ_mid": (100.0 * nb / n_all) if n_all else None,
                "pct_of_quadrant_DockQ_lt_0_23": (100.0 * nc / n_all) if n_all else None,
                "pct_among_failures_CRMSD_lt_10_A": (100.0 * n_ca / nc) if nc else None,
                "pct_among_failures_CRMSD_ge_10_A": (100.0 * n_cb / nc) if nc else None,
            }
        )

    core_hm = None
    af3_hm = None
    if profile:
        core_hm = {
            "caption": "Raw bin means (PNG uses per-column 5–95% scaling for color only).",
            "bins": [],
        }
        for ri, rl in enumerate(profile["row_labels"]):
            bc = profile["bin_counts"][ri] if ri < len(profile["bin_counts"]) else None
            entry = {"bin_label": rl, "n_in_bin": int(bc) if bc is not None else None}
            for cj, (mkey, _) in enumerate(AUDIT_PROFILE_CORE_METRICS):
                v = profile["core_mat"][ri, cj]
                entry[mkey] = float(v) if np.isfinite(v) else None
            core_hm["bins"].append(entry)

        af3_hm = {
            "caption": "Raw bin means (PNG uses per-column 5–95% scaling for color only).",
            "bins": [],
        }
        for ri, rl in enumerate(profile["row_labels"]):
            bc = profile["bin_counts"][ri] if ri < len(profile["bin_counts"]) else None
            entry = {"bin_label": rl, "n_in_bin": int(bc) if bc is not None else None}
            for cj, (mkey, _) in enumerate(AUDIT_PROFILE_AF3_METRICS):
                v = profile["af3_mat"][ri, cj]
                entry[mkey] = float(v) if np.isfinite(v) else None
            af3_hm["bins"].append(entry)

    out = {
        "pred_table": tbl,
        "audit_quadrant_RBP_DockQ_stack": rbp_rows,
        "audit_failure_Complex_RMSD_stack": crmsd_rows,
        "audit_quadrant_outcome_heatmap": outcome_rows,
        "audit_bins_core_means_heatmap": core_hm,
        "audit_bins_af3_means_heatmap": af3_hm,
    }
    if ac_cohen_d:
        out["audit_bins_cohen_d_effect_sizes"] = ac_cohen_d
    if fold_crmsd_effects:
        for fk, fv in fold_crmsd_effects.items():
            if fv is not None:
                out[fk] = fv
    if motif_bin_profiles:
        for strat_k, mprof in motif_bin_profiles.items():
            if not mprof:
                continue
            sk = str(strat_k).replace("-", "_")
            m_core_hm = {
                "caption": "Motif score bins (≤0.5, (0.5, 0.8], (0.8, 1]): raw row means "
                "(PNG uses per-column 5–95% scaling for color only).",
                "bins": [],
            }
            for ri, rl in enumerate(mprof["row_labels"]):
                bc = mprof["bin_counts"][ri] if ri < len(mprof["bin_counts"]) else None
                entry = {"bin_label": rl, "n_in_bin": int(bc) if bc is not None else None}
                for cj, (mkey, _) in enumerate(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS):
                    v = mprof["core_mat"][ri, cj]
                    entry[mkey] = float(v) if np.isfinite(v) else None
                m_core_hm["bins"].append(entry)
            m_af3_hm = {
                "caption": "Motif score bins (≤0.5, (0.5, 0.8], (0.8, 1]): raw row means "
                "(PNG uses per-column 5–95% scaling for color only).",
                "bins": [],
            }
            for ri, rl in enumerate(mprof["row_labels"]):
                bc = mprof["bin_counts"][ri] if ri < len(mprof["bin_counts"]) else None
                entry = {"bin_label": rl, "n_in_bin": int(bc) if bc is not None else None}
                for cj, (mkey, _) in enumerate(AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS):
                    v = mprof["af3_mat"][ri, cj]
                    entry[mkey] = float(v) if np.isfinite(v) else None
                m_af3_hm["bins"].append(entry)
            out[f"audit_motif_bins_{sk}_core_means_heatmap"] = m_core_hm
            out[f"audit_motif_bins_{sk}_af3_means_heatmap"] = m_af3_hm
            m_cd = mprof.get("cohen_d_effect_sizes")
            if m_cd and m_cd.get("rows"):
                out[f"audit_motif_bins_{sk}_binary_cohen_d"] = m_cd
    if crmsd_bin_profile:
        crmsd_hm = {
            "caption": "Complex_RMSD bins (<2 Å, [2, 10) Å, ≥10 Å): core + AF3 raw row means "
            "(PNG uses per-column 5–95% scaling for color only).",
            "bins": [],
        }
        metric_tuples = (
            list(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS)
            + list(AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS)
        )
        for ri, rl in enumerate(crmsd_bin_profile["row_labels"]):
            bc = crmsd_bin_profile["bin_counts"][ri] if ri < len(crmsd_bin_profile["bin_counts"]) else None
            entry = {"bin_label": rl, "n_in_bin": int(bc) if bc is not None else None}
            for cj, (mkey, _) in enumerate(metric_tuples):
                v = crmsd_bin_profile["combined_mat"][ri, cj]
                entry[mkey] = float(v) if np.isfinite(v) else None
            crmsd_hm["bins"].append(entry)
        out["audit_crmsd_bins_combined_means_heatmap"] = crmsd_hm
        crmsd_cd = crmsd_bin_profile.get("cohen_d_effect_sizes")
        if crmsd_cd and crmsd_cd.get("rows"):
            out["audit_crmsd_bins_cohen_d_effect_sizes"] = crmsd_cd
    if motif_score_summaries:
        out["audit_motif_score_table_summaries"] = motif_score_summaries
    if motif_quadrant_matrices:
        for strat_k, mat in motif_quadrant_matrices.items():
            if mat is None or not np.any(np.isfinite(mat)):
                continue
            sk = str(strat_k).replace("-", "_")
            out[f"audit_motif_quadrant_mean_{sk}"] = {
                "score_column": strat_k,
                "caption": "Mean motif score by audit quadrant × DockQ / failure class.",
                "rows": _motif_quadrant_mean_matrix_to_rows(mat),
            }
    if metric_tier_intervals and metric_tier_intervals.get("rows"):
        out["audit_metric_tier_intervals"] = metric_tier_intervals
    if crmsd_metric_tier_intervals and crmsd_metric_tier_intervals.get("rows"):
        out["audit_crmsd_metric_tier_intervals"] = crmsd_metric_tier_intervals
    if consensus_metric_tier_intervals and consensus_metric_tier_intervals.get("rows"):
        out["audit_consensus_metric_tier_intervals"] = consensus_metric_tier_intervals
    if consensus_pdb_tier_lists:
        out["audit_consensus_pdb_tier_lists"] = consensus_pdb_tier_lists
    if entity_complex_predictions:
        out["audit_entity_complex_predictions"] = entity_complex_predictions
    if rna_feature_quadrant_breakdown:
        out["audit_rna_feature_quadrant_influence_breakdown"] = rna_feature_quadrant_breakdown
    return out


def _zscore_array(arr):
    """Population z-scores; NaN if zero variance."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    m = np.mean(a)
    s = np.std(a)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(a, np.nan)
    return (a - m) / s


def _delta_distribution_summary(delta_arr):
    d = np.asarray(delta_arr, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "median": float(np.median(d)),
        "pct_gt_0": float(np.mean(d > 0)),
    }


def _rbp_interface_metrics_defined_nonzero(dockq, fnat, irmsd):
    """Exclude undefined (non-finite) or zero-valued RBP interface metrics."""
    for v in (dockq, fnat, irmsd):
        if not np.isfinite(v) or float(v) == 0.0:
            return False
    return True


class ProteinMetrics(DatabaseMethods, PlotCreator):
    def __init__(self, id: str = "", all_rmsd: str = "", protein_rmsd: str = "", protein_lddt: float = None,
                 protein_tm_score: float = None,  **kwargs):
        DatabaseMethods.__init__(self)
        if 'msa_option' in kwargs or 'single_chain_only' in kwargs or 'rbp_template_option' in kwargs:
            PlotCreator.__init__(self, 'evaluation_metrics',
                                 kwargs.get('msa_option'),
                                 kwargs.get('single_chain_only'),
                                 kwargs.get('rbp_template_option'))
        self.id = id
        self.all_rmsd = all_rmsd
        self.protein_rmsd = protein_rmsd
        self.protein_lddt = protein_lddt
        self.protein_tm_score = protein_tm_score

    def load_from_db(self, select_table):
        columns = ["Complex_RMSD", "Protein_RMSD", "Protein_LDDT", "Protein_TM",
                   "Complex_LDDT_allAtom", "iLDDT", "QS_global", "ICS"]
        condition = "exp_db_id = ?"
        params = (self.id,)
        data = self.get_table_columns(select_table, columns, condition, params)

        if data:
            (self.all_rmsd, self.protein_rmsd, self.protein_lddt, self.protein_tm_score,
            self.complex_lddt_allAtom,
            self.ilddt,
            self.qs,
            self.ics) = data

    def get_protein_rmsd_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["Protein_RMSD"], condition=None)
        rows = self.get_filtered_values(select_table, "Protein_RMSD")
        protein_rmsd = [
            value_list[0] for row in rows if row[0]
            for value_list in [ast.literal_eval(row[0])]
            if isinstance(value_list, list) and value_list
        ]
        return protein_rmsd

    def get_protein_lddt_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["Protein_LDDT"], condition=None)
        rows = self.get_filtered_values(select_table, "Protein_LDDT")
        lddt_values = [row[0] for row in rows if row[0] not in (None, '')]
        return lddt_values

    def get_protein_tm_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["Protein_TM"], condition=None)
        rows = self.get_filtered_values(select_table, "Protein_TM")
        protein_tm_score = [row[0] for row in rows if row[0] not in (None, '')]
        return protein_tm_score

class RNAMetrics(DatabaseMethods, PlotCreator):
    def __init__(self, id: str = "", all_rmsd: str = "", rna_rmsd: str = "", rna_lddt: float = None,
                 rna_tm_score: float = None, rna_inf: str = "", rna_di: float = None, **kwargs):
        DatabaseMethods.__init__(self)
        if 'msa_option' in kwargs or 'single_chain_only' in kwargs or 'rbp_template_option' in kwargs:
            PlotCreator.__init__(self, 'evaluation_metrics',
                                 kwargs.get('msa_option'),
                                 kwargs.get('single_chain_only'),
                                 kwargs.get('rbp_template_option'))
        self.id = id
        self.all_rmsd = all_rmsd
        self.rna_rmsd = rna_rmsd
        self.rna_lddt = rna_lddt
        self.rna_tm_score = rna_tm_score
        self.rna_inf = rna_inf
        self.rna_di = rna_di

    def load_from_db(self, select_table):
        columns = ["Complex_RMSD", "RNA_RMSD", "RNA_LDDT", "RNA_TM", "RNA_INF", "RNA_DI", "RNA_LDDT_allAtom"]
        condition = "exp_db_id = ?"
        params = (self.id,)
        data = self.get_table_columns(select_table, columns, condition, params)

        if data:
            (
                self.all_rmsd,
                self.rna_rmsd,
                self.rna_lddt,
                self.rna_tm_score,
                self.rna_inf,
                self.rna_di,
                self.rna_lddt_allAtom
            ) = data

    def get_rna_rmsd_values(self, select_table):
        rows = self.get_filtered_values(select_table, "RNA_RMSD")
        rna_rmsd = [
            value_list[0] for row in rows if row[0]  # row is a tuple (value,)
            for value_list in [ast.literal_eval(row[0])]
            if isinstance(value_list, list) and value_list
        ]
        return rna_rmsd

    def get_rna_rmsd(self):
        return self.rna_rmsd

    def get_rna_lddt_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["RNA_LDDT"], condition=None)
        rows = self.get_filtered_values(select_table, "RNA_LDDT")
        lddt_values = [row[0] for row in rows if row[0] not in (None, '')]
        return lddt_values

    def get_rna_tm_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["RNA_TM"], condition=None)
        rows = self.get_filtered_values(select_table, "RNA_TM")
        rna_tm_score = [row[0] for row in rows if row[0] not in (None, '')]
        return rna_tm_score

    def get_x_and_y(self, x, y, select_table):
        # print("@£$%@£%@", x, y)
        condition = f"{x} IS NOT NULL AND {y} IS NOT NULL"
        data = self.get_filtered_values(
            select_table,
            ["exp_db_id", y, x],
            base_condition=condition
        )
        return max_length(data)

    def get_longest_rna_interaction_pairs(self, x_metric, y_metric, select_table):
        """
        Return tuples (exp_db_id, y_value, x_value) where x_metric is mapped to longest RNA.
        Supports:
          - RNA_*_AllChains matrix metrics (column-wise best score for longest RNA chain)
          - RBP_* pair-list metrics (average over protein interactions with longest RNA chain)
          - RNA_interface_class (mapped to ordinal numeric score)
        """
        needed = ["exp_db_id", "RNAChainIDs", "RNALength", "ProteinChainIDs", x_metric, y_metric]
        try:
            rows = self.get_filtered_values(select_table, needed)
        except Exception:
            return []
        out = []
        class_map = {
            "unknown": 0.0,
            "incorrect_interface": 1.0,
            "partial_interface": 2.0,
            "correct_interface": 3.0,
        }
        for row in rows:
            exp_db_id, rna_chain_ids, rna_len_raw, protein_chain_ids, x_raw, y_raw = row
            longest_idx, longest_chain = _longest_rna_index_and_chain(rna_chain_ids, rna_len_raw)
            if longest_chain is None or longest_idx is None:
                continue

            x_value = np.nan
            if x_metric in {
                "RNA_TM_AllChains", "RNA_LDDT_AllChains", "RNA_iLDDT_AllChains",
                "RNA_QS_global_AllChains", "RNA_Dockq_AllChains", "RNA_ICS_AllChains"
            }:
                x_value = _extract_longest_rna_matrix_score(x_raw, longest_idx)
            elif x_metric in {"RBP_DockQ", "RBP_QS", "RBP_fnat", "RBP_irmsd"}:
                x_value = _extract_rbp_metric_for_longest_rna(x_raw, longest_chain, protein_chain_ids)
            elif x_metric == "RNA_interface_class":
                parsed = _safe_literal_eval(x_raw)
                if isinstance(parsed, dict):
                    cls = parsed.get(longest_chain)
                    if isinstance(cls, str):
                        x_value = class_map.get(cls.strip().lower(), np.nan)
            else:
                try:
                    x_value = float(x_raw)
                except Exception:
                    x_value = np.nan

            try:
                y_value = float(y_raw)
            except Exception:
                y_value = np.nan
            if np.isfinite(x_value) and np.isfinite(y_value):
                out.append((exp_db_id, y_value, x_value))
        return out

    def get_longest_rna_interface_class_correlations(self, select_table, metric_cols):
        """
        Correlate RNA_interface_class with metrics over RNA chains in RNAChainIDs.
        Rows with label missing are skipped. Ordinal scores: incorrect=1, partial=2, correct=3.

        RNA_*_AllChains matrices use column j aligned to RNAChainIDs (predicted RNA index);
        value per chain j is max_i(matrix[i][j]). RBP_* averages pair entries involving that
        RNA chain and ProteinChainIDs. af3_rna_pTM / af3_rna_ipTM use list positions aligned
        with RNAChainIDs. Scalar metrics use the longest RNA chain label only.
        """
        metrics = [m for m in metric_cols if m]
        rbp_metrics = {"RBP_DockQ", "RBP_QS", "RBP_fnat", "RBP_irmsd"}
        matrix_metrics = {
            "RNA_TM_AllChains", "RNA_LDDT_AllChains", "RNA_iLDDT_AllChains",
            "RNA_QS_global_AllChains", "RNA_Dockq_AllChains", "RNA_ICS_AllChains"
        }
        af3_rna_per_chain_metrics = {"af3_rna_pTM", "af3_rna_ipTM"}
        needs_protein = any(m in rbp_metrics for m in metrics)
        base_cols = ["exp_db_id", "RNAChainIDs", "RNALength", "RNA_interface_class"]
        if needs_protein:
            base_cols.append("ProteinChainIDs")
        needed = base_cols + metrics
        try:
            rows = self.get_filtered_values(select_table, needed)
        except Exception:
            return {}
        class_map = {
            "incorrect_interface": 1.0,
            "partial_interface": 2.0,
            "correct_interface": 3.0,
        }
        inv_class_map = {v: k for k, v in class_map.items()}
        by_metric = {m: {"x": [], "y": [], "by_class": {k: [] for k in class_map}} for m in metrics}
        for row in rows:
            if needs_protein:
                exp_db_id, chain_ids, rna_len_raw, iface_raw, protein_chain_ids, *mvals = row
            else:
                exp_db_id, chain_ids, rna_len_raw, iface_raw, *mvals = row
                protein_chain_ids = ""
            chains = _as_chain_list(chain_ids)
            if not chains:
                continue
            longest_idx, longest_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)
            iface_dict = _safe_literal_eval(iface_raw)
            if not isinstance(iface_dict, dict):
                continue
            for chain_idx, rchain in enumerate(chains):
                cls = iface_dict.get(rchain)
                if not isinstance(cls, str):
                    continue
                cls_norm = cls.strip().lower()
                if cls_norm == "unknown":
                    continue
                cls_num = class_map.get(cls_norm, np.nan)
                if not np.isfinite(cls_num):
                    continue
                for m, y_raw in zip(metrics, mvals):
                    if m in matrix_metrics:
                        y_val = _extract_longest_rna_matrix_score(y_raw, chain_idx)
                    elif m in rbp_metrics:
                        y_val = _extract_rbp_metric_for_rna_chain(y_raw, rchain, protein_chain_ids)
                    elif m in af3_rna_per_chain_metrics:
                        y_val = _af3_rna_metric_at_chain_index(y_raw, chain_idx)
                    else:
                        if longest_chain is None or rchain != longest_chain:
                            continue
                        try:
                            y_val = float(y_raw)
                        except Exception:
                            continue
                    if not np.isfinite(y_val):
                        continue
                    by_metric[m]["x"].append(cls_num)
                    by_metric[m]["y"].append(y_val)
                    by_metric[m]["by_class"][inv_class_map[cls_num]].append(y_val)

        try:
            from scipy.stats import spearmanr
        except Exception:
            spearmanr = None
        out = {}
        for m, payload in by_metric.items():
            x = payload["x"]
            y = payload["y"]
            rho, p = np.nan, np.nan
            if spearmanr is not None and len(x) >= 4 and len(set(x)) > 1 and len(set(y)) > 1:
                try:
                    rho, p = spearmanr(x, y)
                except Exception:
                    pass
            class_stats = {}
            for cls_name, vals in payload["by_class"].items():
                if vals:
                    class_stats[cls_name] = {
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "count": int(len(vals)),
                    }
            by_class_values = {k: list(v) for k, v in payload["by_class"].items()}
            out[m] = {"rho": float(rho) if np.isfinite(rho) else np.nan,
                      "p": float(p) if np.isfinite(p) else np.nan,
                      "class_stats": class_stats,
                      "by_class_values": by_class_values}
        return out

    def get_longest_rna_inf_by_interface_class(self, select_table):
        """
        Bucket RNA_INF WC / nonWC / stack components (list indices 1–3) by interface class
        for the longest RNA chain only. Skips missing labels and INF sentinel -1.
        Returns dict metric_label -> {incorrect_interface|partial_interface|correct_interface -> [values]}.
        """
        iface_labels = ("incorrect_interface", "partial_interface", "correct_interface")
        bucket_labels = ("INF-WC", "INF-nonWC", "INF-STACK")
        out = {label: {k: [] for k in iface_labels} for label in bucket_labels}
        needed = ["RNAChainIDs", "RNALength", "RNA_interface_class", "RNA_INF"]
        try:
            rows = self.get_filtered_values(select_table, needed)
        except Exception:
            return out
        inf_slots = (1, 2, 3)
        for row in rows:
            chain_ids, rna_len_raw, iface_raw, inf_raw = row
            _, longest_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)
            if longest_chain is None:
                continue
            iface_dict = _safe_literal_eval(iface_raw)
            if not isinstance(iface_dict, dict):
                continue
            cls = iface_dict.get(longest_chain)
            if not isinstance(cls, str):
                continue
            cls_norm = cls.strip().lower()
            if cls_norm not in iface_labels:
                continue
            vl = _safe_literal_eval(inf_raw)
            if not isinstance(vl, list) or len(vl) < 4:
                continue
            for slot_idx, blabel in zip(inf_slots, bucket_labels):
                try:
                    v = float(vl[slot_idx])
                except Exception:
                    continue
                if v == -1 or not np.isfinite(v):
                    continue
                out[blabel][cls_norm].append(v)
        return out

    def get_single_rna_interface_af3_global_buckets(self, select_table):
        """
        Samples where RNA_interface_class has exactly one chain entry (e.g. {\"C\": \"correct_interface\"}):
        bucket af3_global_pae_avg, af3_ranking_score, af3_chain_pair_pae_min by that label.
        Excludes unknown; only incorrect_interface / partial_interface / correct_interface.
        """
        iface_labels = ("incorrect_interface", "partial_interface", "correct_interface")
        metric_cols = ("af3_global_pae_avg", "af3_ranking_score", "af3_chain_pair_pae_min")
        out = {m: {k: [] for k in iface_labels} for m in metric_cols}
        needed = ["RNA_interface_class"] + list(metric_cols)
        try:
            rows = self.get_filtered_values(select_table, needed)
        except Exception:
            return out
        for row in rows:
            iface_raw = row[0]
            mvals = row[1:]
            iface_dict = _safe_literal_eval(iface_raw)
            if not isinstance(iface_dict, dict) or len(iface_dict) != 1:
                continue
            cls = next(iter(iface_dict.values()))
            if not isinstance(cls, str):
                continue
            cls_norm = cls.strip().lower()
            if cls_norm not in iface_labels:
                continue
            for m, raw in zip(metric_cols, mvals):
                fv = _parse_optional_float_metric(raw)
                if np.isfinite(fv):
                    out[m][cls_norm].append(fv)
        return out

    def get_global_vs_interface_alignment_series(self, select_table):
        """
        Per-row aligned lists: global RNA_TM / RNA_LDDT vs interface RBP_DockQ, RBP_QS (longest RNA),
        RNA_ICS_AllChains (best matrix score for longest RNA column).
        Requires all five finite for each kept sample. Returns None if too few rows or query fails.
        """
        cols = [
            "RNAChainIDs",
            "RNALength",
            "ProteinChainIDs",
            "RNA_TM",
            "RNA_LDDT",
            "RBP_DockQ",
            "RBP_QS",
            "RNA_ICS_AllChains",
        ]
        try:
            rows = self.get_filtered_values(select_table, cols)
        except Exception:
            return None
        series = {"rna_tm": [], "rna_lddt": [], "rbp_dockq": [], "rbp_qs": [], "rna_ics": []}
        for row in rows:
            chain_ids, rna_len_raw, protein_chain_ids, tm_raw, lddt_raw, dockq_raw, qs_raw, ics_raw = row
            longest_idx, longest_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)
            if longest_chain is None or longest_idx is None:
                continue
            try:
                tm = float(tm_raw)
                lddt = float(lddt_raw)
            except Exception:
                continue
            dockq = _extract_rbp_metric_for_longest_rna(dockq_raw, longest_chain, protein_chain_ids)
            qs = _extract_rbp_metric_for_longest_rna(qs_raw, longest_chain, protein_chain_ids)
            ics = _extract_longest_rna_matrix_score(ics_raw, longest_idx)
            if not all(np.isfinite(v) for v in (tm, lddt, dockq, qs, ics)):
                continue
            series["rna_tm"].append(tm)
            series["rna_lddt"].append(lddt)
            series["rbp_dockq"].append(dockq)
            series["rbp_qs"].append(qs)
            series["rna_ics"].append(ics)
        if len(series["rna_tm"]) < 5:
            return None
        return series

    def get_sample_audit_global_fold_interface_lists(self, select_table):
        """
        Sample audit: four quadrants from RNA_LDDT, Protein_LDDT, and Protein_TM.

        (i) RNA_LDDT > 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85 (TM finite);
            if longest RNA length ≥ 60, also require RNA_TM finite and > 0.5.
        (ii) RNA_LDDT > 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85 OR TM non-finite).
        (iii) RNA_LDDT ≤ 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85 (TM finite).
        (iv) RNA_LDDT ≤ 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85 OR TM non-finite).

        Each quadrant lists (a)/(b)/(c) by DockQ_Nr (mean RBP_DockQ over all protein-RNA interfaces);
        (c) splits into (ca) Complex_RMSD < 10 Å / (cb) ≥ 10 Å, then bins (1)/(2)/(3) by RBP_iRMSD and
        RBP_fnat (partial reconstruction / incorrect docking / large-scale). Rows require finite
        RNA_LDDT and Protein_LDDT. RBP_iRMSD/RBP_fnat use the longest RNA chain.
        """
        cols = [
            "exp_db_id",
            "RNAChainIDs",
            "RNALength",
            "ProteinChainIDs",
            "RNA_LDDT",
            "Protein_LDDT",
            "RNA_TM",
            "Protein_TM",
            "RBP_DockQ",
            "RBP_QS",
            "RBP_irmsd",
            "RBP_fnat",
            "iLDDT",
            "Dockq",
            "Complex_RMSD",
            "RNA_RMSD",
        ]
        try:
            rows = self.get_filtered_values(select_table, cols)
        except Exception:
            return None

        store = {}

        for row in rows:
            (
                exp_db_id,
                chain_ids,
                rna_len_raw,
                protein_chain_ids,
                rna_lddt_raw,
                protein_lddt_raw,
                rna_tm_raw,
                protein_tm_raw,
                dockq_raw,
                qs_raw,
                irmsd_raw,
                fnat_raw,
                ilddt_raw,
                dockq_complex_raw,
                complex_rmsd_raw,
                rna_rmsd_raw,
            ) = row
            try:
                rna_lddt = float(rna_lddt_raw)
                protein_lddt = float(protein_lddt_raw)
            except Exception:
                continue
            if not (np.isfinite(rna_lddt) and np.isfinite(protein_lddt)):
                continue

            rna_tm = np.nan
            protein_tm = np.nan
            try:
                if rna_tm_raw is not None:
                    rna_tm = float(rna_tm_raw)
                    if not np.isfinite(rna_tm):
                        rna_tm = np.nan
            except (TypeError, ValueError):
                pass
            try:
                if protein_tm_raw is not None:
                    protein_tm = float(protein_tm_raw)
                    if not np.isfinite(protein_tm):
                        protein_tm = np.nan
            except (TypeError, ValueError):
                pass

            complex_ilddt = np.nan
            complex_dockq = np.nan
            try:
                if ilddt_raw is not None:
                    complex_ilddt = float(ilddt_raw)
                    if not np.isfinite(complex_ilddt):
                        complex_ilddt = np.nan
            except (TypeError, ValueError):
                pass
            try:
                if dockq_complex_raw is not None:
                    complex_dockq = float(dockq_complex_raw)
                    if not np.isfinite(complex_dockq):
                        complex_dockq = np.nan
            except (TypeError, ValueError):
                pass

            complex_rmsd_first = _first_metric_float(complex_rmsd_raw)
            rna_rmsd_first = _first_metric_float(rna_rmsd_raw)

            eid = str(exp_db_id).strip()
            dockq = np.nan
            qs = np.nan
            irmsd = np.nan
            fnat = np.nan
            longest_idx, longest_chain = _longest_rna_index_and_chain(chain_ids, rna_len_raw)
            rna_length_longest = np.nan
            if longest_idx is not None:
                lens = _as_float_list(rna_len_raw)
                if longest_idx < len(lens):
                    lv = lens[longest_idx]
                    if np.isfinite(lv):
                        rna_length_longest = float(lv)
            if longest_chain is not None:
                dockq = _extract_rbp_metric_for_longest_rna(dockq_raw, longest_chain, protein_chain_ids)
                qs = _extract_rbp_metric_for_longest_rna(qs_raw, longest_chain, protein_chain_ids)
                irmsd = _extract_rbp_metric_for_longest_rna(irmsd_raw, longest_chain, protein_chain_ids)
                fnat = _extract_rbp_metric_for_longest_rna(fnat_raw, longest_chain, protein_chain_ids)
            # DockQ_Nr: mean RBP_DockQ over ALL protein-RNA interfaces (any RNA chain). This now
            # drives the a/b/c interface classification and pub_fig1 panels B/C/D. The longest-RNA
            # rbp_dockq is still stored for reference but no longer used for grading.
            dockq_nr = _rbp_metric_all_protein_rna(dockq_raw, chain_ids, protein_chain_ids)

            rec = {
                "exp_db_id": eid,
                "rna_lddt": rna_lddt,
                "protein_lddt": protein_lddt,
                "rna_tm": rna_tm,
                "protein_tm": protein_tm,
                "rbp_dockq": dockq,
                "dockq_nr": dockq_nr,
                "rbp_qs": qs,
                "rbp_irmsd": irmsd,
                "rbp_fnat": fnat,
                "complex_ilddt": complex_ilddt,
                "complex_dockq": complex_dockq,
                "complex_rmsd_first": complex_rmsd_first,
                "rna_rmsd_first": rna_rmsd_first,
                "rna_length_longest": rna_length_longest,
            }
            score = rna_lddt + protein_lddt
            if (
                eid not in store
                or score > store[eid]["rna_lddt"] + store[eid]["protein_lddt"]
            ):
                store[eid] = rec

        def _valid(rec):
            return _rbp_interface_metrics_defined_nonzero(
                rec["dockq_nr"], rec["rbp_fnat"], rec["rbp_irmsd"]
            )

        n_total = len(store)
        n_valid = sum(1 for r in store.values() if _valid(r))

        def _neg_tm_key(tm):
            return -float(tm) if np.isfinite(tm) else float("inf")

        def _fv_sort_key(v):
            return -float(v) if np.isfinite(v) else float("-inf")

        def _sort_quad(rs):
            rs.sort(
                key=lambda r: (
                    -r["rna_lddt"],
                    -r["protein_lddt"],
                    _fv_sort_key(r["dockq_nr"]),
                    _fv_sort_key(r["rbp_qs"]),
                    r["exp_db_id"],
                )
            )

        def _split_rbp_dockq_abc(recs):
            a, b, c, unk = [], [], [], []
            for rec in recs:
                dq = rec["dockq_nr"]
                if not np.isfinite(dq):
                    unk.append(rec)
                elif dq >= 0.8:
                    a.append(rec)
                elif dq >= 0.23:
                    b.append(rec)
                else:
                    c.append(rec)
            return a, b, c, unk

        def _partition_c_by_complex_rmsd(recs_c):
            """Split DockQ failures by Complex_RMSD: c_ca = <10 Å, c_cb = ≥10 Å (finite only)."""
            crmsd_ge10, crmsd_lt10, c_nf = [], [], []
            for rec in recs_c:
                crmsd = rec["complex_rmsd_first"]
                if np.isfinite(crmsd) and crmsd >= 10.0:
                    crmsd_ge10.append(rec)
                elif np.isfinite(crmsd) and crmsd < 10.0:
                    crmsd_lt10.append(rec)
                else:
                    c_nf.append(rec)
            return crmsd_lt10, crmsd_ge10, c_nf

        def _split_rbp_iface_123(recs):
            """Return (bin_1, bin_2, bin_3, other) for labels (1)(2)(3) per audit prose."""
            large_scale, incorrect_dock, partial_recon, bo = [], [], [], []
            for rec in recs:
                irmsd = rec["rbp_irmsd"]
                fnat = rec["rbp_fnat"]
                if not np.isfinite(irmsd) or not np.isfinite(fnat):
                    bo.append(rec)
                    continue
                if irmsd > 10.0:
                    large_scale.append(rec)
                elif irmsd <= 10.0 and 0.0 <= fnat < 0.1:
                    incorrect_dock.append(rec)
                elif irmsd <= 10.0 and 0.1 <= fnat < 0.5:
                    partial_recon.append(rec)
                else:
                    bo.append(rec)
            return partial_recon, incorrect_dock, large_scale, bo

        def _sort_abc(rs):
            rs.sort(
                key=lambda r: (
                    _fv_sort_key(r["dockq_nr"]),
                    _fv_sort_key(r["rbp_qs"]),
                    -r["rna_lddt"],
                    r["exp_db_id"],
                )
            )

        def _sort_c_ca(rs):
            rs.sort(
                key=lambda r: (
                    -r["complex_rmsd_first"]
                    if np.isfinite(r["complex_rmsd_first"])
                    else float("-inf"),
                    r["exp_db_id"],
                )
            )

        def _sort_c_sub(rs):
            rs.sort(
                key=lambda r: (
                    -r["rbp_irmsd"] if np.isfinite(r["rbp_irmsd"]) else float("-inf"),
                    r["rbp_fnat"] if np.isfinite(r["rbp_fnat"]) else float("inf"),
                    r["exp_db_id"],
                )
            )

        def _sort_unknown_dockq(rs):
            rs.sort(
                key=lambda r: (
                    -r["rna_lddt"],
                    -r["protein_lddt"],
                    _neg_tm_key(r["rna_tm"]),
                    r["exp_db_id"],
                )
            )

        def _process_quadrant(parent):
            _sort_quad(parent)
            a, b, c, unk = _split_rbp_dockq_abc(parent)
            crmsd_lt10, crmsd_ge10, c_crmsd_nf = _partition_c_by_complex_rmsd(c)
            _sort_c_ca(crmsd_lt10)
            _sort_c_ca(crmsd_ge10)
            lt10_1, lt10_2, lt10_3, lt10_o = _split_rbp_iface_123(crmsd_lt10)
            ge10_1, ge10_2, ge10_3, ge10_o = _split_rbp_iface_123(crmsd_ge10)
            _sort_abc(a)
            _sort_abc(b)
            _sort_abc(c)
            _sort_c_sub(lt10_1)
            _sort_c_sub(lt10_2)
            _sort_c_sub(lt10_3)
            _sort_c_sub(lt10_o)
            _sort_c_sub(ge10_1)
            _sort_c_sub(ge10_2)
            _sort_c_sub(ge10_3)
            _sort_c_sub(ge10_o)
            _sort_unknown_dockq(c_crmsd_nf)
            _sort_unknown_dockq(unk)
            return {
                "all": parent,
                "a": a,
                "b": b,
                "c": c,
                "unknown_rbp_dockq": unk,
                "c_ca": crmsd_lt10,
                "c_cb": crmsd_ge10,
                "c_crmsd_nonfinite": c_crmsd_nf,
                "c_ca_1": lt10_1,
                "c_ca_2": lt10_2,
                "c_ca_3": lt10_3,
                "c_ca_other": lt10_o,
                "c_cb_1": ge10_1,
                "c_cb_2": ge10_2,
                "c_cb_3": ge10_3,
                "c_cb_other": ge10_o,
            }

        def _quad_i_long_rna_tm_gate(rec):
            """If longest RNA length ≥ 60, require RNA_TM > 0.5 for quadrant (i)."""
            rlen = rec["rna_length_longest"]
            if np.isfinite(rlen) and rlen >= 60.0:
                rt = rec["rna_tm"]
                return np.isfinite(rt) and rt > 0.5
            return True

        quad_i = []
        quad_ii = []
        quad_iii = []
        quad_iv = []
        for rec in store.values():
            rq = rec["rna_lddt"]
            pq = rec["protein_lddt"]
            ptm = rec["protein_tm"]
            tm_good = np.isfinite(ptm) and ptm > 0.85

            if rq > 0.85:
                if pq > 0.85 and tm_good and _quad_i_long_rna_tm_gate(rec):
                    quad_i.append(rec)
                else:
                    quad_ii.append(rec)
            else:
                if pq > 0.85 and tm_good:
                    quad_iii.append(rec)
                else:
                    quad_iv.append(rec)

        qi = _process_quadrant(quad_i)
        qii = _process_quadrant(quad_ii)
        qiii = _process_quadrant(quad_iii)
        qiv = _process_quadrant(quad_iv)

        def _flatten_quad(prefix, qd):
            out = {}
            for k, v in qd.items():
                out[f"{prefix}_{k}"] = v
            return out

        out = {
            "pred_table": select_table,
            "n_predictions_finite_lddt": n_total,
            "n_predictions_valid_interface_metrics": n_valid,
        }
        out.update(_flatten_quad("quad_i", qi))
        out.update(_flatten_quad("quad_ii", qii))
        out.update(_flatten_quad("quad_iii", qiii))
        out.update(_flatten_quad("quad_iv", qiv))
        return out

    def get_x_and_y_condition(self, x, y, condition, select_table):
        # print("@£$%@£%@", x, y)
        condition = f"{condition} AND {x} IS NOT NULL AND {y} IS NOT NULL"
        data = self.get_filtered_values(
            select_table,
            ["exp_db_id", y, x],
            base_condition=condition
        )
        return max_length(data)

    def get_rna_inf_values(self, select_table):
        rows = self.get_filtered_values(select_table, "RNA_INF")
        # rows = self.get_table_columns(select_table, ["RNA_INF"], condition=None)
        all_inf = []
        wc_inf = []
        nonWC_inf = []
        stacking_inf = []
        for row in rows:
            if row[0]:
                try:
                    value_list = ast.literal_eval(row[0])
                    if isinstance(value_list, list) and len(value_list) > 0:
                        all_inf.append(value_list[0])
                        wc_inf.append(value_list[1])
                        nonWC_inf.append(value_list[2])
                        stacking_inf.append(value_list[3])
                except (ValueError, SyntaxError):
                    continue
            else:
                continue
        return all_inf, wc_inf, nonWC_inf, stacking_inf

    def get_rna_inf_values_with_extra(self, select_table, extra_column):
        # query = f"SELECT exp_db_id, {extra_column}, RNA_INF FROM unified_pred WHERE SourceTable = ?"
        # rows = self.get_table_columns(select_table, ["exp_db_id", extra_column, "RNA_INF"], condition=None)
        rows = self.get_filtered_values(select_table, ["exp_db_id", extra_column, "RNA_INF"])

        exp_db_ids = []
        x_values = []  # This will be 'extra_value'
        y_values_all_inf = []  # This will be 'RNA_INF'
        wc_inf = []
        nonWC_inf = []
        stacking_inf = []

        for row in rows:
            exp_db_id = row[0]
            extra_value = row[1]
            rna_inf_value = row[2]

            # Check if both extra_value and rna_inf_value are present
            if extra_value is not None and rna_inf_value:
                try:
                    value_list = ast.literal_eval(rna_inf_value)
                    if isinstance(value_list, list) and len(value_list) > 0:
                        # Append to lists only if valid values are found
                        exp_db_ids.append(exp_db_id)
                        x_values.append(extra_value)
                        y_values_all_inf.append(value_list[0])  # y-values for plotting
                        wc_inf.append(value_list[1])
                        nonWC_inf.append(value_list[2])
                        stacking_inf.append(value_list[3])
                except (ValueError, SyntaxError):
                    continue

        # Ensure all lists are the same length
        min_length = min(len(x_values), len(y_values_all_inf), len(wc_inf), len(nonWC_inf), len(stacking_inf))
        return (exp_db_ids[:min_length], x_values[:min_length], y_values_all_inf[:min_length],
                wc_inf[:min_length], nonWC_inf[:min_length], stacking_inf[:min_length])

    def get_rna_di_values(self, select_table):
        # rows = self.get_table_columns(select_table, ["RNA_DI"], condition=None)
        rows = self.get_filtered_values(select_table, "RNA_DI")
        rna_di = [row[0] for row in rows if row[0] not in (None, '')]
        return rna_di

    def get_di_and_length(self, select_table):
        condition = "RNA_DI IS NOT NULL AND RNA_DI != '' AND RNALength IS NOT NULL"
        data = self.get_filtered_values(
            select_table, 
            ["exp_db_id", "RNA_DI", "RNALength"], 
            base_condition=condition
        )
        return max_length(data)

    def get_x_and_length(self, x, select_table):
        condition = f"{x} IS NOT NULL AND {x} != '' AND RNALength IS NOT NULL"
        data = self.get_filtered_values(
            select_table, 
            ["exp_db_id", x, "RNALength"], 
            base_condition=condition
        )
        return max_length(data)

    def get_rmsd_and_length(self, select_table):
        condition = "RNA_RMSD IS NOT NULL AND RNA_RMSD != '' AND RNALength IS NOT NULL"
        rows = self.get_filtered_values(
            select_table, 
            ["exp_db_id", "RNA_RMSD", "RNALength"], 
            base_condition=condition
        )
        processed_data = []
        for row in rows:
            exp_db_id = row[0]
            rna_rmsd_str = row[1]
            rna_length = row[2]
            try:
                rna_rmsd = float(rna_rmsd_str.strip('[]').split(',')[0]) #pymol: 0, tmscore:1
                processed_data.append((exp_db_id, rna_rmsd, rna_length))
            except (ValueError, IndexError):
                continue
        return max_length(processed_data)

    def get_msa_rmsd_comparison(self, table):
        """Get RMSD values grouped by MSA status"""
        query = f"""
        SELECT RNA_RMSD, af3_rna_MSA 
        FROM {table}
        WHERE RNA_RMSD IS NOT NULL AND af3_rna_MSA IS NOT NULL
        """
        rows = self.execute_query(query)
        
        msa_rmsd = {'+MSA': [], '-MSA': []}
        for rmsd, msa in rows:
            if rmsd and msa:
                try:
                    # Convert string representations to lists if necessary
                    if isinstance(rmsd, str):
                        rmsd_values = eval(rmsd)
                        rmsd = max(rmsd_values) if isinstance(rmsd_values, list) else float(rmsd)
                    if isinstance(msa, str):
                        msa_values = eval(msa)
                        # Classify as -MSA if any value is 0, otherwise +MSA
                        is_msa = 0 if (isinstance(msa_values, list) and 0 in msa_values) else 1
                    else:
                        is_msa = int(msa)
                    
                    # Add to appropriate list
                    if is_msa == 1:
                        msa_rmsd['+MSA'].append(float(rmsd))
                    else:
                        msa_rmsd['-MSA'].append(float(rmsd))
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing values - RMSD: {rmsd}, MSA: {msa}")
                    continue
        
        return msa_rmsd['+MSA'], msa_rmsd['-MSA']

    def get_msa_metric_comparison(self, table, metric_column):
        """Get metric values grouped by MSA status"""
        query = f"""
        SELECT {metric_column}, af3_rna_MSA 
        FROM {table}
        WHERE {metric_column} IS NOT NULL AND af3_rna_MSA IS NOT NULL
        """
        rows = self.execute_query(query)
        
        metric_values = {'+MSA': [], '-MSA': []}
        for value, msa in rows:
            if value and msa:
                try:
                    # Handle RMSD values differently (they're stored as lists)
                    if metric_column == 'RNA_RMSD':
                        if isinstance(value, str):
                            value_list = eval(value)
                            if isinstance(value_list, list) and value_list:
                                value = float(value_list[0])  # Take first value for RMSD
                            else:
                                value = float(value)
                        else:
                            # For other metrics (TM, LDDT)
                            if isinstance(value, str):
                                value = float(value)
                    
                    # Process MSA values
                    if isinstance(msa, str):
                        msa_values = eval(msa)
                        # Classify as -MSA if any value is 0, otherwise +MSA
                        is_msa = 0 if (isinstance(msa_values, list) and 0 in msa_values) else 1
                    else:
                        is_msa = int(msa)
                    
                    # Add to appropriate list
                    if is_msa == 1:
                        metric_values['+MSA'].append(float(value))
                    else:
                        metric_values['-MSA'].append(float(value))
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing values - {metric_column}: {value}, MSA: {msa}")
                    continue
        
        return metric_values['+MSA'], metric_values['-MSA']

    def get_msa_inf_comparison(self, table):
        """Get INF values grouped by MSA status"""
        query = f"""
        SELECT RNA_INF, af3_rna_MSA 
        FROM {table}
        WHERE RNA_INF IS NOT NULL AND af3_rna_MSA IS NOT NULL
        """
        rows = self.execute_query(query)
        
        inf_values = {
            '+MSA': {'all': [], 'wc': [], 'nonwc': [], 'stack': []},
            '-MSA': {'all': [], 'wc': [], 'nonwc': [], 'stack': []}
        }
        
        for inf, msa in rows:
            if inf and msa:
                try:
                    # Process INF values
                    if isinstance(inf, str):
                        inf_list = eval(inf)
                        if isinstance(inf_list, list) and len(inf_list) == 4:
                            # Only include valid INF values (-1 indicates invalid/missing)
                            inf_values_filtered = [v for v in inf_list if v != -1]
                            if len(inf_values_filtered) != len(inf_list):
                                continue  # Skip if any value was -1
                                
                            # Process MSA values
                            if isinstance(msa, str):
                                msa_values = eval(msa)
                                # Classify as -MSA if any value is 0, otherwise +MSA
                                is_msa = 0 if (isinstance(msa_values, list) and 0 in msa_values) else 1
                            else:
                                is_msa = int(msa)
                            
                            # Add to appropriate lists
                            msa_key = '+MSA' if is_msa == 1 else '-MSA'
                            inf_values[msa_key]['all'].append(float(inf_list[0]))
                            inf_values[msa_key]['wc'].append(float(inf_list[1]))
                            inf_values[msa_key]['nonwc'].append(float(inf_list[2]))
                            inf_values[msa_key]['stack'].append(float(inf_list[3]))
                            
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing values - INF: {inf}, MSA: {msa}")
                    continue
        
        return inf_values

    def get_msa_di_comparison(self, table):
        """Get DI values grouped by MSA status"""
        query = f"""
        SELECT RNA_DI, af3_rna_MSA 
        FROM {table}
        WHERE RNA_DI IS NOT NULL AND af3_rna_MSA IS NOT NULL
        """
        rows = self.execute_query(query)
        
        di_values = {'+MSA': [], '-MSA': []}
        for di, msa in rows:
            if di and msa:
                try:
                    # Convert DI value to float
                    if isinstance(di, str):
                        di = float(di)
                    
                    # Process MSA values
                    if isinstance(msa, str):
                        msa_values = eval(msa)
                        # Classify as -MSA if any value is 0, otherwise +MSA
                        is_msa = 0 if (isinstance(msa_values, list) and 0 in msa_values) else 1
                    else:
                        is_msa = int(msa)
                    
                    # Add to appropriate list
                    if is_msa == 1:
                        di_values['+MSA'].append(float(di))
                    else:
                        di_values['-MSA'].append(float(di))
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing values - DI: {di}, MSA: {msa}")
                    continue
        
        return di_values['+MSA'], di_values['-MSA']

class DNAMetrics(DatabaseMethods, PlotCreator):
    def __init__(self, id: str = "", all_rmsd: str = "", dna_rmsd: str = "", dna_lddt: float = None,
                 dna_tm_score: float = None, msa_option=None, single_chain_only=False, rbp_template_option=None):
        DatabaseMethods.__init__(self)
        PlotCreator.__init__(self, 'evaluation_metrics', msa_option, single_chain_only, rbp_template_option)
        self.id = id
        self.all_rmsd = all_rmsd
        self.dna_rmsd = dna_rmsd
        self.dna_lddt = dna_lddt
        self.dna_tm_score = dna_tm_score

    def load_from_db(self, select_table):
        columns = ["Complex_RMSD", "DNA_RMSD", "DNA_LDDT", "DNA_TM"]
        condition = "exp_db_id = ?"
        params = (self.id,)

        data = self.get_table_columns(select_table, columns, condition, params)

        if data:
            (
                self.all_rmsd,
                self.dna_rmsd,
                self.dna_lddt,
                self.dna_tm_score
            ) = data

    def get_dna_rmsd_values(self, select_table):
        rows = self.get_table_columns(select_table, ["DNA_RMSD"], condition=None)
        dna_rmsd = []
        for row in rows:
            if row[0]:
                try:
                    value_list = ast.literal_eval(row[0])
                    if isinstance(value_list, list) and len(value_list) > 0:
                        dna_rmsd.append(value_list[0])  # Take the first value
                except (ValueError, SyntaxError):
                    continue
            else:
                continue
        return dna_rmsd

    def get_dna_lddt_values(self, select_table):
        rows = self.get_table_columns(select_table, ["DNA_LDDT"], condition=None)
        lddt_values = [row[0] for row in rows if row[0] not in (None, '')]
        return lddt_values

    def get_dna_tm_values(self, select_table):
        rows = self.get_table_columns(select_table, ["DNA_TM"], condition=None)
        dna_tm_score = [row[0] for row in rows if row[0] not in (None, '')]
        return dna_tm_score

    def get_x_and_length(self, x, select_table):
        condition = f"{x} IS NOT NULL AND {x} != '' AND RNALength IS NOT NULL"
        data = self.get_table_columns(select_table, ["exp_db_id", x, "DNALength"], condition)
        return max_length(data)

    def get_rmsd_and_length(self, select_table):
        condition = "DNA_RMSD IS NOT NULL AND DNA_RMSD != '' AND DNALength IS NOT NULL"
        rows = self.get_table_columns_all(select_table, ["exp_db_id", "DNA_RMSD", "DNALength"], condition)
        processed_data = []
        for row in rows:
            exp_db_id = row[0]
            dna_rmsd_str = row[1]
            dna_length = row[2]
            try:
                dna_rmsd = float(dna_rmsd_str.strip('[]').split(',')[0]) #pymol: 0, tmscore:1
                processed_data.append((exp_db_id, dna_rmsd, dna_length))
            except (ValueError, IndexError):
                continue
        return max_length(processed_data)

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


def _summarize_interface_class_buckets(by_class_values):
    """
    by_class_values: incorrect_interface | partial_interface | correct_interface -> list of floats.
    Returns dict with Spearman rho/p vs ordinal (1,2,3) and per-class mean/median/count.
    """
    class_ord = {
        "incorrect_interface": 1.0,
        "partial_interface": 2.0,
        "correct_interface": 3.0,
    }
    x, y = [], []
    for cname, ordv in class_ord.items():
        for v in (by_class_values or {}).get(cname, []):
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                x.append(ordv)
                y.append(fv)
    rho, p = np.nan, np.nan
    try:
        from scipy.stats import spearmanr
    except Exception:
        spearmanr = None
    if spearmanr is not None and len(x) >= 4 and len(set(x)) > 1 and len(set(y)) > 1:
        try:
            rho, p = spearmanr(x, y)
        except Exception:
            pass
    class_stats = {}
    for cname in class_ord:
        vals = []
        for v in (by_class_values or {}).get(cname, []):
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
        if vals:
            class_stats[cname] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "count": int(len(vals)),
            }
    return {
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "class_stats": class_stats,
    }


def _analyze_global_vs_interface_alignment(series):
    """
    series keys: rna_tm, rna_lddt, rbp_dockq, rbp_qs, rna_ics (equal-length lists).
    Returns dict for statistics file / plotting coordination.
    """
    tm = np.array(series["rna_tm"], dtype=float)
    lddt = np.array(series["rna_lddt"], dtype=float)
    dockq = np.array(series["rbp_dockq"], dtype=float)
    qs = np.array(series["rbp_qs"], dtype=float)
    ics = np.array(series["rna_ics"], dtype=float)
    n = tm.size
    out = {
        "n_samples": int(n),
        "caption": (
            "Longest RNA used for RBP_DockQ / RBP_QS (mean over protein pairs) and "
            "RNA_ICS_AllChains (best matrix score for that predicted RNA column). "
            "Global scores RNA_TM / RNA_LDDT are whole-molecule metrics."
        ),
    }

    def _spearman(xa, ya, key):
        try:
            from scipy.stats import spearmanr
            m = np.isfinite(xa) & np.isfinite(ya)
            if m.sum() < 4:
                return
            xv, yv = xa[m], ya[m]
            if len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
                return
            rho, p = spearmanr(xv, yv)
            out[f"spearman_{key}_rho"] = float(rho)
            out[f"spearman_{key}_p"] = float(p)
        except Exception:
            pass

    _spearman(lddt, dockq, "rna_lddt_rbp_dockq")
    _spearman(tm, dockq, "rna_tm_rbp_dockq")
    _spearman(lddt, qs, "rna_lddt_rbp_qs")
    _spearman(tm, qs, "rna_tm_rbp_qs")
    _spearman(lddt, ics, "rna_lddt_rna_ics_allchains")
    _spearman(tm, ics, "rna_tm_rna_ics_allchains")

    z_tm = _zscore_array(tm)
    z_lddt = _zscore_array(lddt)
    z_dockq = _zscore_array(dockq)
    z_qs = _zscore_array(qs)
    z_ics = _zscore_array(ics)

    delta_defs = [
        ("delta_z_rna_lddt_minus_rbp_dockq", z_lddt - z_dockq),
        ("delta_z_rna_tm_minus_rbp_dockq", z_tm - z_dockq),
        ("delta_z_rna_lddt_minus_rbp_qs", z_lddt - z_qs),
        ("delta_z_rna_tm_minus_rbp_qs", z_tm - z_qs),
        ("delta_z_rna_lddt_minus_rna_ics", z_lddt - z_ics),
        ("delta_z_rna_tm_minus_rna_ics", z_tm - z_ics),
    ]
    for key, arr in delta_defs:
        summ = _delta_distribution_summary(arr)
        if summ:
            out[key] = summ

    try:
        from scipy.stats import linregress
        lr = linregress(lddt, dockq)
        pred = lr.intercept + lr.slope * lddt
        resid = dockq - pred
        thr = np.percentile(lddt, 75)
        hi = lddt >= thr
        out["ols_rbp_dockq_on_rna_lddt"] = {
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_squared": float(lr.rvalue ** 2),
            "mean_residual": float(np.mean(resid)),
            "pct_residual_negative_rna_lddt_top_quartile": float(np.mean(resid[hi] < 0)) if np.any(hi) else np.nan,
            "n_rna_lddt_top_quartile": int(np.sum(hi)),
        }
        lr_tm = linregress(tm, dockq)
        pred_t = lr_tm.intercept + lr_tm.slope * tm
        resid_t = dockq - pred_t
        thr_t = np.percentile(tm, 75)
        hi_t = tm >= thr_t
        out["ols_rbp_dockq_on_rna_tm"] = {
            "slope": float(lr_tm.slope),
            "intercept": float(lr_tm.intercept),
            "r_squared": float(lr_tm.rvalue ** 2),
            "mean_residual": float(np.mean(resid_t)),
            "pct_residual_negative_rna_tm_top_quartile": float(np.mean(resid_t[hi_t] < 0)) if np.any(hi_t) else np.nan,
            "n_rna_tm_top_quartile": int(np.sum(hi_t)),
        }
    except Exception:
        pass

    try:
        cat = pd.qcut(pd.Series(lddt), q=4, duplicates="drop")
        qdock = {}
        for interval in cat.cat.categories:
            mask = (cat == interval).to_numpy()
            v = dockq[mask]
            v = v[np.isfinite(v)]
            if v.size:
                qdock[str(interval)] = {
                    "n": int(v.size),
                    "mean_rbp_dockq": float(np.mean(v)),
                    "median_rbp_dockq": float(np.median(v)),
                }
        out["rna_lddt_quartile_bins_rbp_dockq"] = qdock
    except Exception:
        pass

    try:
        cat_q = pd.qcut(pd.Series(lddt), q=4, duplicates="drop")
        qqs = {}
        for interval in cat_q.cat.categories:
            mask = (cat_q == interval).to_numpy()
            v = qs[mask]
            v = v[np.isfinite(v)]
            if v.size:
                qqs[str(interval)] = {
                    "n": int(v.size),
                    "mean_rbp_qs": float(np.mean(v)),
                    "median_rbp_qs": float(np.median(v)),
                }
        out["rna_lddt_quartile_bins_rbp_qs"] = qqs
    except Exception:
        pass

    return out


def _plot_global_vs_interface_alignment(plot_creator, table, series):
    tm = series["rna_tm"]
    lddt = series["rna_lddt"]
    dockq = series["rbp_dockq"]
    qs = series["rbp_qs"]
    ics = series["rna_ics"]

    plot_creator.get_scatterplot(
        table, lddt, r"$\mathrm{LDDT}_{r}$", "RBP DockQ (longest RNA)", "globalRNA_LDDT_vs_RBP_DockQ", yAxis_score=dockq
    )
    plot_creator.get_scatterplot(
        table, tm, r"$\mathrm{TM}_{r}$", "RBP DockQ (longest RNA)", "globalRNA_TM_vs_RBP_DockQ", yAxis_score=dockq
    )
    plot_creator.get_scatterplot(
        table, lddt, r"$\mathrm{LDDT}_{r}$", "RBP QS (longest RNA)", "globalRNA_LDDT_vs_RBP_QS", yAxis_score=qs
    )
    plot_creator.get_scatterplot(
        table, tm, r"$\mathrm{TM}_{r}$", "RBP QS (longest RNA)", "globalRNA_TM_vs_RBP_QS", yAxis_score=qs
    )
    plot_creator.get_scatterplot(
        table, lddt, r"$\mathrm{LDDT}_{r}$", "RNA ICS AllChains (longest RNA)", "globalRNA_LDDT_vs_RNA_ICS_AllChains", yAxis_score=ics
    )
    plot_creator.get_scatterplot(
        table, tm, r"$\mathrm{TM}_{r}$", "RNA ICS AllChains (longest RNA)", "globalRNA_TM_vs_RNA_ICS_AllChains", yAxis_score=ics
    )

    z_tm = _zscore_array(np.array(tm, dtype=float))
    z_lddt = _zscore_array(np.array(lddt, dtype=float))
    z_dockq = _zscore_array(np.array(dockq, dtype=float))
    z_qs = _zscore_array(np.array(qs, dtype=float))
    z_ics = _zscore_array(np.array(ics, dtype=float))

    hist_specs = [
        (z_lddt - z_dockq, "DeltaZ_RNA_LDDT_minus_RBP_DockQ", "Δ_z = z(RNA_LDDT) − z(RBP_DockQ)"),
        (z_tm - z_dockq, "DeltaZ_RNA_TM_minus_RBP_DockQ", "Δ_z = z(RNA_TM) − z(RBP_DockQ)"),
        (z_lddt - z_qs, "DeltaZ_RNA_LDDT_minus_RBP_QS", "Δ_z = z(RNA_LDDT) − z(RBP_QS)"),
        (z_tm - z_qs, "DeltaZ_RNA_TM_minus_RBP_QS", "Δ_z = z(RNA_TM) − z(RBP_QS)"),
        (z_lddt - z_ics, "DeltaZ_RNA_LDDT_minus_RNA_ICS", "Δ_z = z(RNA_LDDT) − z(RNA_ICS_AllChains)"),
        (z_tm - z_ics, "DeltaZ_RNA_TM_minus_RNA_ICS", "Δ_z = z(RNA_TM) − z(RNA_ICS_AllChains)"),
    ]
    for arr, fname, xlab in hist_specs:
        finite = arr[np.isfinite(arr)].tolist()
        if len(finite) >= 2:
            plot_creator.get_histogram(table, finite, fname, xAxis_label=xlab, vlines=[0])

    try:
        cat = pd.qcut(pd.Series(lddt), q=4, duplicates="drop")
        data_vals = []
        labels = []
        for interval in cat.cat.categories:
            mask = (cat == interval).to_numpy()
            chunk = np.array(dockq)[mask]
            chunk = chunk[np.isfinite(chunk)].tolist()
            if chunk:
                data_vals.append(chunk)
                labels.append(str(interval))
        if len(data_vals) >= 2:
            plot_creator.get_boxplot(
                table,
                data_vals,
                labels,
                "RBP_DockQ_by_RNA_LDDT_quartile",
                yAxis_label="RBP DockQ",
                xAxis_label=r"$\mathrm{LDDT}_{r}$ quartile (low → high)",
                rotation=15,
            )
    except Exception:
        pass

    try:
        cat = pd.qcut(pd.Series(lddt), q=4, duplicates="drop")
        data_vals = []
        labels = []
        for interval in cat.cat.categories:
            mask = (cat == interval).to_numpy()
            chunk = np.array(qs)[mask]
            chunk = chunk[np.isfinite(chunk)].tolist()
            if chunk:
                data_vals.append(chunk)
                labels.append(str(interval))
        if len(data_vals) >= 2:
            plot_creator.get_boxplot(
                table,
                data_vals,
                labels,
                "RBP_QS_by_RNA_LDDT_quartile",
                yAxis_label="RBP QS",
                xAxis_label=r"$\mathrm{LDDT}_{r}$ quartile (low → high)",
                rotation=15,
            )
    except Exception:
        pass


def compute_sample_audit_quadrant_dockq_chi2(audit):
    """Pearson chi-square independence test for quadrant × RBP_DockQ bin (a, b, c, unknown)."""
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return None
    if not audit or "quad_i_all" not in audit:
        return None
    prefs = ("quad_i", "quad_ii", "quad_iii", "quad_iv")
    row_labels = ["(i)", "(ii)", "(iii)", "(iv)"]
    col_labels = [
        "RBP_DockQ ≥ 0.8 (a)",
        "0.23 ≤ RBP_DockQ < 0.8 (b)",
        "RBP_DockQ < 0.23 (c)",
        "RBP_DockQ non-finite",
    ]
    table = []
    for p in prefs:
        table.append(
            [
                len(audit.get(f"{p}_a") or []),
                len(audit.get(f"{p}_b") or []),
                len(audit.get(f"{p}_c") or []),
                len(audit.get(f"{p}_unknown_rbp_dockq") or []),
            ]
        )
    obs = np.asarray(table, dtype=float)
    if obs.sum() < 1:
        return None
    chi2, p, dof, expected = chi2_contingency(obs)
    return {
        "Pearson_chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "observed": obs,
        "expected": expected,
        "row_labels": row_labels,
        "col_labels": col_labels,
    }


def _write_sample_audit_chi2_block(f, chi_block):
    if not chi_block or "Pearson_chi2" not in chi_block:
        return
    f.write("=" * 80 + "\n")
    f.write("Sample audit: Pearson chi-square (quadrant × RBP_DockQ bin)\n")
    f.write("=" * 80 + "\n")
    f.write(f"H0: quadrant and DockQ class are independent.\n")
    f.write(f"chi2 = {chi_block['Pearson_chi2']:.6g}, dof = {chi_block['dof']}, p = {chi_block['p_value']:.6g}\n\n")
    rows = chi_block["row_labels"]
    cols = chi_block["col_labels"]
    obs = chi_block["observed"]
    exp = chi_block["expected"]
    f.write("Observed counts:\n")
    f.write("       " + " | ".join(f"{c[:28]:>28}" for c in cols) + "\n")
    for i, rl in enumerate(rows):
        f.write(f"{rl:>6} " + " | ".join(f"{int(obs[i, j]):>28}" for j in range(len(cols))) + "\n")
    f.write("\nExpected counts:\n")
    f.write("       " + " | ".join(f"{c[:28]:>28}" for c in cols) + "\n")
    for i, rl in enumerate(rows):
        f.write(f"{rl:>6} " + " | ".join(f"{exp[i, j]:>28.2f}" for j in range(len(cols))) + "\n")
    f.write("\n")


def _fmt_audit_stat_float(x, digits=4):
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(xf):
        return "NA"
    return f"{xf:.{digits}f}"


def _fmt_audit_per_bin_means_line(bin_entry, metric_tuples, digits=4):
    """Comma-separated metric means for one audit bin (single txt line)."""
    parts = [
        f"{mlbl}: {_fmt_audit_stat_float(bin_entry.get(mkey), digits)}"
        for mkey, mlbl in metric_tuples
    ]
    return ", ".join(parts)


def _write_audit_metric_tier_intervals_block(f, tier_block, stem, section_title, med_hdr, n_hdr):
    if not tier_block or not tier_block.get("rows"):
        return
    f.write("-" * 80 + "\n")
    f.write(f"{section_title}{stem}\n")
    f.write(str(tier_block.get("caption", "")) + "\n\n")
    f.write(
        f"feature | dir | t_lo | t_hi | good | mid | bad | "
        f"{med_hdr[0]} | {med_hdr[1]} | {med_hdr[2]} | {n_hdr[0]} | {n_hdr[1]} | {n_hdr[2]}\n"
    )
    for row in tier_block["rows"]:
        lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
        if row.get("tier_note"):
            f.write(
                f"{lbl} | — | — | — | — | — | — | "
                f"{_fmt_audit_stat_float(row.get('med_good'), 4)} | "
                f"{_fmt_audit_stat_float(row.get('med_mid'), 4)} | "
                f"{_fmt_audit_stat_float(row.get('med_bad'), 4)} | "
                f"{row.get('n_good', 0)} | {row.get('n_mid', 0)} | {row.get('n_bad', 0)}  "
                f"# {row['tier_note']}\n"
            )
            continue
        direction = "↑ good" if row.get("higher_is_good") else "↓ good"
        note_parts = []
        if row.get("cutpoint_note"):
            note_parts.append(str(row["cutpoint_note"]))
        if row.get("direction_note"):
            note_parts.append(str(row["direction_note"]))
        if row.get("consensus_note"):
            note_parts.append(str(row["consensus_note"]))
        note_suffix = f"  # {'; '.join(note_parts)}" if note_parts else ""
        f.write(
            f"{lbl} | {direction:>6} | "
            f"{_fmt_audit_stat_float(row.get('t_lo'), 4)} | "
            f"{_fmt_audit_stat_float(row.get('t_hi'), 4)} | "
            f"{row.get('good_band', 'NA')} | {row.get('mid_band', 'NA')} | {row.get('bad_band', 'NA')} | "
            f"{_fmt_audit_stat_float(row.get('med_good'), 4)} | "
            f"{_fmt_audit_stat_float(row.get('med_mid'), 4)} | "
            f"{_fmt_audit_stat_float(row.get('med_bad'), 4)} | "
            f"{row.get('n_good', 0)} | {row.get('n_mid', 0)} | {row.get('n_bad', 0)}{note_suffix}\n"
        )
    f.write("\n")


def any_motif_audit_bins(bundle):
    """True if sample-audit bundle contains motif/match stratified bin stats."""
    if not bundle:
        return False
    if bundle.get("audit_crmsd_bins_combined_means_heatmap", {}).get("bins"):
        return True
    if bundle.get("audit_motif_score_table_summaries"):
        return True
    for sk in AUDIT_MOTIF_STRATIFY_COLUMNS:
        sk_u = str(sk).replace("-", "_")
        if bundle.get(f"audit_motif_quadrant_mean_{sk_u}", {}).get("rows"):
            return True
        mc = bundle.get(f"audit_motif_bins_{sk}_core_means_heatmap")
        ma = bundle.get(f"audit_motif_bins_{sk}_af3_means_heatmap")
        mco = bundle.get(f"audit_motif_bins_{sk}_binary_cohen_d")
        if (mc and mc.get("bins")) or (ma and ma.get("bins")) or (mco and mco.get("rows")):
            return True
    return False


def _audit_rna_breakdown_metric_short_label(mkey):
    return _AUDIT_RNA_BREAKDOWN_METRIC_SHORT_LABELS.get(mkey, mkey)


def _write_audit_rna_feature_quadrant_influence_breakdown(f, block, stem):
    """RNA Shannon / MFE / af3_rna_pLDDT_avg means per quadrant (i–iv) within each influence stratum."""
    if not block or not block.get("rows"):
        return
    metrics = block.get("metrics") or list(AUDIT_QUADRANT_RNA_FEATURE_BREAKDOWN_METRICS)
    f.write("-" * 80 + "\n")
    f.write(
        f"RNA_Shannon_Entropy_K2, RNA_MFE_Value & af3_rna_pLDDT_avg by quadrant × influence{stem}\n"
    )
    if block.get("caption"):
        f.write(str(block["caption"]) + "\n\n")
    hdr_parts = ["quadrant"]
    for mkey in metrics:
        sl = _audit_rna_breakdown_metric_short_label(mkey)
        hdr_parts.extend([
            f"n_{sl}",
            f"{sl}_mean",
            f"{sl}_median",
            f"{sl}_sd",
        ])
    hdr = " | ".join(hdr_parts) + "\n"
    for stratum in block.get("strata") or []:
        sk = stratum.get("key", "")
        sl = stratum.get("label", sk)
        stratum_rows = [r for r in block.get("rows") or [] if r.get("stratum_key") == sk]
        if not any(
            any(int(r.get(f"{mkey}_n") or 0) > 0 for mkey in metrics)
            for r in stratum_rows
        ):
            continue
        idx = stratum.get("index", "")
        prefix = f"{idx}. " if idx != "" else ""
        f.write(f"--- {prefix}Stratum: {sl} ---\n")
        f.write(hdr)
        for row in stratum_rows:
            parts = [f"{row.get('quadrant', ''):>8}"]
            for mkey in metrics:
                msl = _audit_rna_breakdown_metric_short_label(mkey)
                n_w = 9 if msl == "Shannon" else 5
                parts.extend([
                    f"{row.get(f'{mkey}_n', 0):>{n_w}}",
                    f"{_fmt_audit_stat_float(row.get(f'{mkey}_mean'), 4):>14}",
                    f"{_fmt_audit_stat_float(row.get(f'{mkey}_median'), 4):>16}",
                    f"{_fmt_audit_stat_float(row.get(f'{mkey}_std'), 4):>12}",
                ])
            f.write(" | ".join(parts) + "\n")
        f.write("\n")


def _write_sample_audit_figure_statistics(f, bundle):
    """Write numeric tables backing interface audit PNGs (DockQ / CRMSD stacks, outcome heatmap, bin heatmaps)."""
    if not bundle:
        return
    tbl = bundle.get("pred_table") or ""
    stem = f"_{tbl}" if tbl else ""

    f.write("=" * 80 + "\n")
    f.write("Sample audit figure statistics (interface PNGs)\n")
    f.write("=" * 80 + "\n")
    if tbl:
        f.write(f"Prediction table stem: {tbl}\n")
        f.write(f"(Filenames use suffix ...{stem})\n\n")

    f.write("-" * 80 + "\n")
    f.write(f"audit_quadrant_RBP_DockQ_stack{stem}\n")
    f.write("Per quadrant: denominator = complexes with finite RBP_DockQ.\n")
    f.write(
        "Quadrant | n_finite | n_a (DockQ≥0.8) | n_b (mid) | n_c (<0.23) | frac_a | frac_b | frac_c\n"
    )
    for row in bundle.get("audit_quadrant_RBP_DockQ_stack") or []:
        f.write(
            f"{row['quadrant']:>32} | {row['n_finite_RBP_DockQ']:>8} | "
            f"{row['n_DockQ_ge_0_8']:>14} | {row['n_DockQ_0_23_to_0_8']:>9} | "
            f"{row['n_DockQ_lt_0_23']:>11} | {_fmt_audit_stat_float(row['frac_DockQ_ge_0_8'], 4)} | "
            f"{_fmt_audit_stat_float(row['frac_DockQ_mid'], 4)} | {_fmt_audit_stat_float(row['frac_DockQ_lt_0_23'], 4)}\n"
        )
    f.write("\n")

    f.write("-" * 80 + "\n")
    f.write(f"audit_failure_Complex_RMSD_stack{stem}\n")
    f.write("Among DockQ failures (class c) only; denominator = finite Complex_RMSD.\n")
    f.write("Quadrant | n_finite_CRMSD | n_<10Å | n_≥10Å | frac_<10Å | frac_≥10Å\n")
    for row in bundle.get("audit_failure_Complex_RMSD_stack") or []:
        f.write(
            f"{row['quadrant']:>32} | {row['n_DockQ_failures_finite_CRMSD']:>16} | "
            f"{row['n_CRMSD_lt_10_A']:>7} | {row['n_CRMSD_ge_10_A']:>8} | "
            f"{_fmt_audit_stat_float(row['frac_CRMSD_lt_10_A'], 4)} | {_fmt_audit_stat_float(row['frac_CRMSD_ge_10_A'], 4)}\n"
        )
    f.write("\n")

    f.write("-" * 80 + "\n")
    f.write(f"audit_quadrant_outcome_heatmap{stem}\n")
    f.write(
        "Cols 1–3: % of quadrant total (finite global RNA lDDT bins). "
        "Cols 4–5: % among DockQ failures in that quadrant.\n"
    )
    f.write(
        "Quadrant | n_quad | %DockQ≥0.8 | %mid | %<0.23 | %fail CRMSD<10Å | %fail CRMSD≥10Å\n"
    )
    for row in bundle.get("audit_quadrant_outcome_heatmap") or []:
        f.write(
            f"{row['quadrant']:>32} | {row['n_quadrant_fin_lddt']:>6} | "
            f"{_fmt_audit_stat_float(row['pct_of_quadrant_DockQ_ge_0_8'], 2)} | "
            f"{_fmt_audit_stat_float(row['pct_of_quadrant_DockQ_mid'], 2)} | "
            f"{_fmt_audit_stat_float(row['pct_of_quadrant_DockQ_lt_0_23'], 2)} | "
            f"{_fmt_audit_stat_float(row['pct_among_failures_CRMSD_lt_10_A'], 2)} | "
            f"{_fmt_audit_stat_float(row['pct_among_failures_CRMSD_ge_10_A'], 2)}\n"
        )
    f.write("\n")

    core = bundle.get("audit_bins_core_means_heatmap")
    af3 = bundle.get("audit_bins_af3_means_heatmap")
    cohen = bundle.get("audit_bins_cohen_d_effect_sizes")
    fq_ord = bundle.get("audit_effect_fold_quadrant_ordinal_i_iv") or {}
    fq_bin = bundle.get("audit_effect_fold_quadrant_binary_i_vs_iv") or {}
    fq_rna = bundle.get("audit_effect_rna_accurate_i_ii_vs_inaccurate_iii_iv") or {}
    fq_prot = bundle.get("audit_effect_protein_accurate_i_iii_vs_inaccurate_ii_iv") or {}
    fq_cr = bundle.get("audit_effect_crmsd_fail_lt10_vs_ge10") or {}
    has_bins = (core and core.get("bins")) or (af3 and af3.get("bins"))
    has_fold = bool(
        (
            fq_ord.get("rows")
            or fq_bin.get("rows")
            or fq_rna.get("rows")
            or fq_prot.get("rows")
            or fq_cr.get("rows")
        )
    )
    f.write("-" * 80 + "\n")
    f.write(f"audit_bins_core_means_heatmap / audit_bins_af3_means_heatmap{stem}\n")
    if has_bins or (cohen and cohen.get("rows")) or has_fold:
        f.write(
            "Core + AF3 metrics (same engineered features as the bin-mean heatmap PNGs).\n\n"
        )
        if cohen and cohen.get("rows"):
            f.write(
                "Effect-size table — rows sorted by composite (descending).\n"
                "  composite: |d| x |rho_ord| (primary rank; NA if d or rho_ord is missing).\n"
                "  d: Cohen's d = (mean_C - mean_A) / s_pool; (a) RBP_DockQ >= 0.8, (c) RBP_DockQ < 0.23; "
                "complexes pooled over quadrants (positive d => higher mean in (c) than (a)).\n"
                "  s_pool: pooled SD for d from (a) and (c) sample variances (ddof=1).\n"
                "  mean_A, mean_C: metric means in classes (a) and (c).\n"
                "  rho_ord: Spearman correlation of metric with ordinal DockQ class "
                "(a)=1, (b)=2, (c)=3 on finite values in (a)+(b)+(c), quadrants pooled.\n"
                "  |d|, |rho|: absolute values; signed columns report d and rho_ord as in the formulas above.\n"
                "  n_A, n_C: finite-metric counts in (a) and (c); n_ABC: finite counts in (a)+(b)+(c) for rho_ord.\n"
                "  Q_i–Q_iv: Spearman(metric, ordinal class) within that quadrant only (validation; NA if degenerate).\n"
                "  quadrant_validation: short sign-consistency note vs pooled rho (not used for ranking).\n\n"
                "Finite d requires n_A >= 2, n_C >= 2, and s_pool > 0. "
                "Finite rho_ord requires spread in both metric and class labels.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'composite':>9} | {'|d|':>8} | {'|rho|':>8} | {'d':>9} | {'rho':>9} | {'s_pool':>9} | "
                f"{'mean_A':>9} | {'mean_C':>9} | {'n_ABC':>6} | {'n_A':>5} | {'n_C':>5} | "
                f"{'Q_i':>7} | {'Q_ii':>7} | {'Q_iii':>7} | {'Q_iv':>7} | {'quadrant_validation':>36} | feature\n"
            )
            for row in cohen["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('composite_abs_d_rho'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('abs_spearman_rho_ord'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('spearman_rho_ord'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_ord_abc', 0):>6} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | "
                    f"{_fmt_audit_stat_float(row.get('rho_quad_i'), 3):>7} | "
                    f"{_fmt_audit_stat_float(row.get('rho_quad_ii'), 3):>7} | "
                    f"{_fmt_audit_stat_float(row.get('rho_quad_iii'), 3):>7} | "
                    f"{_fmt_audit_stat_float(row.get('rho_quad_iv'), 3):>7} | "
                    f"{str(row.get('quad_validation_note', ''))[:36]:>36} | {lbl}\n"
                )
            f.write("\n")

        if fq_ord.get("rows"):
            f.write("-" * 80 + "\n")
            f.write((fq_ord.get("title") or "Fold quadrant ordinal") + f"{stem}\n")
            if fq_ord.get("d_compare"):
                f.write(fq_ord["d_compare"] + "\n")
            if fq_ord.get("validation"):
                f.write(fq_ord["validation"] + "\n")
            f.write(
                "composite = |d| x |rho|; d = (mean_(iv) - mean_(i)) / s_pool on quad_*_all; "
                "rho = Spearman(metric, quadrant ordinal 1..4); n_ord = finite samples in (i)+(ii)+(iii)+(iv).\n"
                "rho_DQA/DQB/DQC = Spearman(metric, quadrant ordinal) restricted to DockQ (a)/(b)/(c); "
                "dq_val = sign-consistency note (not used for ranking).\n\n"
            )
            f.write(
                f"{'rk':>3} | {'composite':>9} | {'|d|':>8} | {'|rho|':>8} | {'d':>9} | {'rho':>9} | {'s_pool':>9} | "
                f"{'mean_i':>9} | {'mean_iv':>9} | {'n_ord':>6} | {'n_i':>5} | {'n_iv':>5} | "
                f"{'r_DQA':>7} | {'r_DQB':>7} | {'r_DQC':>7} | {'dq_val':>36} | feature\n"
            )
            for row in fq_ord["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('composite_abs_d_rho'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('abs_spearman_rho_ord'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('spearman_rho_ord'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_ord_levels', 0):>6} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | "
                    f"{_fmt_audit_stat_float(row.get('rho_dockq_slice_a'), 3):>7} | "
                    f"{_fmt_audit_stat_float(row.get('rho_dockq_slice_b'), 3):>7} | "
                    f"{_fmt_audit_stat_float(row.get('rho_dockq_slice_c'), 3):>7} | "
                    f"{str(row.get('dockq_slice_validation_note', ''))[:36]:>36} | {lbl}\n"
                )
            f.write("\n")

        if fq_bin.get("rows"):
            f.write("-" * 80 + "\n")
            f.write((fq_bin.get("title") or "Fold quadrant binary") + f"{stem}\n")
            if fq_bin.get("d_compare"):
                f.write(fq_bin["d_compare"] + "\n")
            f.write(
                "Binary contrast only: Cohen's d = (mean_(iv) - mean_(i)) / s_pool (quad_*_all); "
                "sorted by |d|. No ordinal Spearman/composite.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'|d|':>8} | {'d':>9} | {'s_pool':>9} | {'mean_i':>9} | {'mean_iv':>9} | "
                f"{'n_i':>5} | {'n_iv':>5} | feature\n"
            )
            for row in fq_bin["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | {lbl}\n"
                )
            f.write("\n")

        if fq_rna.get("rows"):
            f.write("-" * 80 + "\n")
            f.write((fq_rna.get("title") or "RNA accuracy combined") + f"{stem}\n")
            if fq_rna.get("d_compare"):
                f.write(fq_rna["d_compare"] + "\n")
            f.write(
                "Binary only: mean_acc / mean_inacc / n_acc / n_inacc refer to (i)+(ii) vs (iii)+(iv); "
                "d = (mean_inacc - mean_acc) / s_pool; sorted by |d|.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'|d|':>8} | {'d':>9} | {'s_pool':>9} | {'mean_acc':>9} | {'mean_inacc':>9} | "
                f"{'n_acc':>6} | {'n_inacc':>7} | feature\n"
            )
            for row in fq_rna["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_a', 0):>6} | {row.get('n_c', 0):>7} | {lbl}\n"
                )
            f.write("\n")

        if fq_prot.get("rows"):
            f.write("-" * 80 + "\n")
            f.write((fq_prot.get("title") or "Protein accuracy combined") + f"{stem}\n")
            if fq_prot.get("d_compare"):
                f.write(fq_prot["d_compare"] + "\n")
            f.write(
                "Binary only: mean_acc / mean_inacc / n_acc / n_inacc refer to (i)+(iii) vs (ii)+(iv); "
                "d = (mean_inacc - mean_acc) / s_pool; sorted by |d|.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'|d|':>8} | {'d':>9} | {'s_pool':>9} | {'mean_acc':>9} | {'mean_inacc':>9} | "
                f"{'n_acc':>6} | {'n_inacc':>7} | feature\n"
            )
            for row in fq_prot["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_a', 0):>6} | {row.get('n_c', 0):>7} | {lbl}\n"
                )
            f.write("\n")

        if fq_cr.get("rows"):
            f.write("-" * 80 + "\n")
            f.write((fq_cr.get("title") or "CRMSD failure split") + f"{stem}\n")
            if fq_cr.get("d_compare"):
                f.write(fq_cr["d_compare"] + "\n")
            f.write(
                "Binary contrast among DockQ failures: *_c_ca = Complex_RMSD < 10 Å; *_c_cb = >= 10 Å "
                "(audit key names). d = (mean_>=10 - mean_<10) / s_pool; sorted by |d|.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'|d|':>8} | {'d':>9} | {'s_pool':>9} | {'mean_lt10':>9} | {'mean_ge10':>9} | "
                f"{'n_lt10':>5} | {'n_ge10':>5} | feature\n"
            )
            for row in fq_cr["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | {lbl}\n"
                )
            f.write("\n")

        if core and core.get("bins"):
            f.write("Per-bin raw means — core\n")
            f.write(core.get("caption", "") + "\n")
            for b in core["bins"]:
                bl = b.get("bin_label", "").replace("\n", " / ")
                f.write(
                    f"  [{bl}]  n={b.get('n_in_bin')}: "
                    f"{_fmt_audit_per_bin_means_line(b, AUDIT_PROFILE_CORE_METRICS)}\n"
                )
            f.write("\n")
        if af3 and af3.get("bins"):
            f.write("Per-bin raw means — AF3\n")
            f.write(af3.get("caption", "") + "\n")
            for b in af3["bins"]:
                bl = b.get("bin_label", "").replace("\n", " / ")
                f.write(
                    f"  [{bl}]  n={b.get('n_in_bin')}: "
                    f"{_fmt_audit_per_bin_means_line(b, AUDIT_PROFILE_AF3_METRICS)}\n"
                )
            f.write("\n")

        if not has_bins and ((cohen and cohen.get("rows")) or has_fold):
            f.write("(Per-bin heatmap tables omitted: bin profile unavailable.)\n\n")
    else:
        if not any_motif_audit_bins(bundle):
            f.write(
                "(Not computed: bin profile, DockQ effect table, and fold/CRMSD effect tables unavailable.)\n\n"
            )

    motif_summaries = bundle.get("audit_motif_score_table_summaries")
    if motif_summaries:
        f.write("-" * 80 + "\n")
        f.write(f"RNA motif score table summaries (all prediction rows){stem}\n")
        f.write(
            "RNAmotif: gap-padded global alignment score. "
            "RNAmatch: exact (base, index) Jaccard with triplet shift fallback.\n"
            "Bins: ≤0.5, (0.5, 0.8], (0.8, 1].\n\n"
        )
        f.write(
            "metric | n | mean | std | median | min | max | n_≤0.5 | n_mid | n_hi | frac_≤0.5 | frac_mid | frac_hi\n"
        )
        for strat_k in AUDIT_MOTIF_STRATIFY_COLUMNS:
            s = motif_summaries.get(strat_k)
            if not s:
                continue
            f.write(
                f"{strat_k} | {s['n_finite']:>3} | "
                f"{_fmt_audit_stat_float(s['mean'], 4)} | {_fmt_audit_stat_float(s['std'], 4)} | "
                f"{_fmt_audit_stat_float(s['median'], 4)} | {_fmt_audit_stat_float(s['min'], 4)} | "
                f"{_fmt_audit_stat_float(s['max'], 4)} | "
                f"{s['bin_n_le_0_5']:>7} | {s['bin_n_mid']:>5} | {s['bin_n_hi']:>4} | "
                f"{_fmt_audit_stat_float(s['bin_frac_le_0_5'], 4)} | "
                f"{_fmt_audit_stat_float(s['bin_frac_mid'], 4)} | "
                f"{_fmt_audit_stat_float(s['bin_frac_hi'], 4)}\n"
            )
        f.write("\n")

    for strat_k in AUDIT_MOTIF_STRATIFY_COLUMNS:
        sk = str(strat_k).replace("-", "_")
        qblock = bundle.get(f"audit_motif_quadrant_mean_{sk}")
        if not qblock or not qblock.get("rows"):
            continue
        f.write("-" * 80 + "\n")
        f.write(
            f"audit_quadrant_outcome_heatmap_mean_{strat_k}{stem}\n"
        )
        f.write(
            "Same layout as outcome heatmap PNG; cell = mean score over audit-listed predictions.\n"
        )
        f.write("Quadrant | outcome_col | mean_score\n")
        for row in qblock["rows"]:
            f.write(
                f"{row['quadrant']:>32} | {row['outcome_col']:>22} | "
                f"{_fmt_audit_stat_float(row.get('mean_score'), 4)}\n"
            )
        f.write("\n")

    for strat_k in AUDIT_MOTIF_STRATIFY_COLUMNS:
        mk_core = bundle.get(f"audit_motif_bins_{strat_k}_core_means_heatmap")
        mk_af3 = bundle.get(f"audit_motif_bins_{strat_k}_af3_means_heatmap")
        mk_cohen = bundle.get(f"audit_motif_bins_{strat_k}_binary_cohen_d")
        if not (
            (mk_core and mk_core.get("bins"))
            or (mk_af3 and mk_af3.get("bins"))
            or (mk_cohen and mk_cohen.get("rows"))
        ):
            continue
        f.write("-" * 80 + "\n")
        f.write(f"Motif / match score bins ({strat_k}) — heatmaps + binary effects{stem}\n")
        f.write("Bins: ≤0.5, (0.5, 0.8], (0.8, 1]; all prediction rows in table.\n\n")
        if mk_cohen and mk_cohen.get("rows"):
            if mk_cohen.get("compare"):
                f.write(str(mk_cohen["compare"]) + "\n")
            if mk_cohen.get("formula_cohen"):
                f.write(str(mk_cohen["formula_cohen"]) + "\n")
            if mk_cohen.get("formula_spearman"):
                f.write(str(mk_cohen["formula_spearman"]) + "\n")
            f.write(
                "Sorted by composite |d|·|rho|; Spearman uses ordinal bins 1–3; "
                "Cohen d: mean_hi / n_hi = bin (0.8, 1], mean_lo / n_lo = bin ≤0.5; "
                "d = (mean_hi − mean_lo) / s_pool.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'composite':>9} | {'|d|':>8} | {'|rho|':>8} | {'d':>9} | {'rho':>9} | "
                f"{'s_pool':>9} | {'mean_lo':>9} | {'mean_hi':>9} | {'n_ABC':>6} | {'n_lo':>5} | {'n_hi':>5} | feature\n"
            )
            for row in mk_cohen["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('composite_abs_d_rho'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('abs_spearman_rho_ord'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('spearman_rho_ord'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_ord_abc', row.get('n_ord_ab', 0)):>6} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | {lbl}\n"
                )
            f.write("\n")

    crmsd_combined = bundle.get("audit_crmsd_bins_combined_means_heatmap")
    crmsd_cohen = bundle.get("audit_crmsd_bins_cohen_d_effect_sizes")
    if crmsd_combined or (crmsd_cohen and crmsd_cohen.get("rows")):
        f.write("-" * 80 + "\n")
        f.write(f"Complex_RMSD bins — combined core + AF3 heatmap{stem}\n")
        f.write("Bins: CRMSD < 2 Å, 2 ≤ CRMSD < 10 Å, CRMSD ≥ 10 Å; all rows with finite Complex_RMSD.\n\n")
        if crmsd_cohen and crmsd_cohen.get("rows"):
            if crmsd_cohen.get("compare"):
                f.write(str(crmsd_cohen["compare"]) + "\n")
            if crmsd_cohen.get("formula_cohen"):
                f.write(str(crmsd_cohen["formula_cohen"]) + "\n")
            if crmsd_cohen.get("formula_spearman"):
                f.write(str(crmsd_cohen["formula_spearman"]) + "\n")
            f.write(
                "Sorted by composite |d|·|rho|; Cohen d: bin ≥10 Å vs bin <2 Å.\n\n"
            )
            f.write(
                f"{'rk':>3} | {'composite':>9} | {'|d|':>8} | {'|rho|':>8} | {'d':>9} | {'rho':>9} | "
                f"{'s_pool':>9} | {'mean_lo':>9} | {'mean_hi':>9} | {'n_ABC':>6} | {'n_lo':>5} | {'n_hi':>5} | feature\n"
            )
            for row in crmsd_cohen["rows"]:
                lbl = str(row.get("label", row.get("metric_key", ""))).replace("\n", " ")
                f.write(
                    f"{row.get('rank', ''):>3} | "
                    f"{_fmt_audit_stat_float(row.get('composite_abs_d_rho'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('abs_cohen_d'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('abs_spearman_rho_ord'), 4):>8} | "
                    f"{_fmt_audit_stat_float(row.get('cohen_d'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('spearman_rho_ord'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('pooled_sd'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_a'), 4):>9} | "
                    f"{_fmt_audit_stat_float(row.get('mean_c'), 4):>9} | "
                    f"{row.get('n_ord_abc', row.get('n_ord_ab', 0)):>6} | "
                    f"{row.get('n_a', 0):>5} | {row.get('n_c', 0):>5} | {lbl}\n"
                )
            f.write("\n")
        if crmsd_combined and crmsd_combined.get("bins"):
            f.write("Per-bin raw means — CRMSD bins (core + AF3 combined)\n")
            f.write(crmsd_combined.get("caption", "") + "\n")
            metric_tuples = (
                list(AUDIT_PROFILE_MOTIF_BIN_CORE_METRICS)
                + list(AUDIT_PROFILE_MOTIF_BIN_AF3_METRICS)
            )
            for b in crmsd_combined["bins"]:
                bl = b.get("bin_label", "").replace("\n", " / ")
                f.write(
                    f"  [{bl}]  n={b.get('n_in_bin')}: "
                    f"{_fmt_audit_per_bin_means_line(b, metric_tuples)}\n"
                )
            f.write("\n")

    tier_block = bundle.get("audit_metric_tier_intervals")
    _write_audit_metric_tier_intervals_block(
        f,
        tier_block,
        stem,
        "Metric tier intervals (DockQ good/mid/bad)",
        ("med_a", "med_b", "med_c"),
        ("n_a", "n_b", "n_c"),
    )
    consensus_tier_block = bundle.get("audit_consensus_metric_tier_intervals")
    _write_audit_metric_tier_intervals_block(
        f,
        consensus_tier_block,
        stem,
        "Metric tier intervals (classification-assigned consensus)",
        ("med_a", "med_b", "med_c"),
        ("n_a", "n_b", "n_c"),
    )
    crmsd_tier_block = bundle.get("audit_crmsd_metric_tier_intervals")
    _write_audit_metric_tier_intervals_block(
        f,
        crmsd_tier_block,
        stem,
        "Metric tier intervals (Complex_RMSD good/mid/bad)",
        ("med_<2Å", "med_[2,10)Å", "med_≥10Å"),
        ("n_<2Å", "n_[2,10)Å", "n_≥10Å"),
    )

    pdb_lists = bundle.get("audit_consensus_pdb_tier_lists")
    if pdb_lists:
        f.write("-" * 80 + "\n")
        f.write(f"Consensus tier PDB lists (classification-assigned cutpoints){stem}\n")
        f.write(str(pdb_lists.get("caption", "")) + "\n\n")
        af3_lbl = pdb_lists.get("af3_metric_labels") or "af3_* metrics"
        f.write(f"AF3 primary buckets (ALL {af3_lbl} same tier):\n\n")
        f.write(f"  AF3 ALL good (n={len(pdb_lists.get('af3_all_good_ids') or [])}):\n")
        f.write(f"    {_fmt_audit_eid_list(pdb_lists.get('af3_all_good_ids'))}\n\n")
        f.write(f"  AF3 ALL mid (n={len(pdb_lists.get('af3_all_mid_ids') or [])}):\n")
        f.write(f"    {_fmt_audit_eid_list(pdb_lists.get('af3_all_mid_ids'))}\n\n")
        f.write(f"  AF3 ALL bad (n={len(pdb_lists.get('af3_all_bad_ids') or [])}):\n")
        f.write(f"    {_fmt_audit_eid_list(pdb_lists.get('af3_all_bad_ids'))}\n\n")

        by_af3 = pdb_lists.get("by_af3_tier") or {}
        for af3_tier, tier_title in (
            ("good", "AF3 ALL good"),
            ("mid", "AF3 ALL mid"),
            ("bad", "AF3 ALL bad"),
        ):
            blocks = by_af3.get(af3_tier) or []
            if not blocks:
                continue
            f.write(f"--- Other metrics within {tier_title} ---\n\n")
            for block in blocks:
                lbl = block.get("label", block.get("metric_key", ""))
                f.write(f"{lbl} ({block.get('direction', '')}; good {block.get('good_band', 'NA')})\n")
                f.write(
                    f"  good (n={len(block.get('good_ids') or [])}): "
                    f"{_fmt_audit_eid_list(block.get('good_ids'))}\n"
                )
                f.write(
                    f"  mid  (n={len(block.get('mid_ids') or [])}): "
                    f"{_fmt_audit_eid_list(block.get('mid_ids'))}\n"
                )
                f.write(
                    f"  bad  (n={len(block.get('bad_ids') or [])}): "
                    f"{_fmt_audit_eid_list(block.get('bad_ids'))}\n\n"
                )

        f.write("--- Population-wide per-metric tiers (all audit ids) ---\n\n")
        for block in pdb_lists.get("per_metric") or []:
            lbl = block.get("label", block.get("metric_key", ""))
            f.write(f"{lbl} ({block.get('direction', '')}; good {block.get('good_band', 'NA')})\n")
            f.write(f"  good (n={len(block.get('good_ids') or [])}): {_fmt_audit_eid_list(block.get('good_ids'))}\n")
            f.write(f"  mid  (n={len(block.get('mid_ids') or [])}): {_fmt_audit_eid_list(block.get('mid_ids'))}\n")
            f.write(f"  bad  (n={len(block.get('bad_ids') or [])}): {_fmt_audit_eid_list(block.get('bad_ids'))}\n\n")

    rna_inf = bundle.get("audit_rna_feature_quadrant_influence_breakdown")
    _write_audit_rna_feature_quadrant_influence_breakdown(f, rna_inf, stem)

    entity_pred = bundle.get("audit_entity_complex_predictions")
    if entity_pred:
        f.write("-" * 80 + "\n")
        f.write(f"Entity-based Protein-RNA complex prediction{stem}\n")
        f.write(str(entity_pred.get("caption", "")) + "\n\n")
        f.write(
            f"Final complex good (n={len(entity_pred.get('final_good_ids') or [])}): "
            f"{_fmt_audit_eid_list(entity_pred.get('final_good_ids'))}\n\n"
        )
        f.write(
            f"Final complex mid (n={len(entity_pred.get('final_mid_ids') or [])}): "
            f"{_fmt_audit_eid_list(entity_pred.get('final_mid_ids'))}\n\n"
        )
        f.write(
            f"Final complex bad (n={len(entity_pred.get('final_bad_ids') or [])}): "
            f"{_fmt_audit_eid_list(entity_pred.get('final_bad_ids'))}\n\n"
        )
        insuf = entity_pred.get("insufficient_ids") or []
        if insuf:
            f.write(
                f"Insufficient entity data (n={len(insuf)}): {_fmt_audit_eid_list(insuf)}\n\n"
            )

        blk = entity_pred.get("final_tier_match_rate") or {}
        f.write(
            f"Final-tier match to DockQ (1=exact, 0.5=adjacent, 0=opposite): "
            f"rate={_fmt_audit_stat_float(blk.get('rate'), 4)} (n={blk.get('n', 0)})\n\n"
        )

        af3_combos = entity_pred.get("af3_combinations") or []
        if af3_combos:
            f.write("AF3 entity combinations (Interface × RNA × Protein → DockQ):\n")
            f.write(
                "Interface | RNA | Protein | n | DockQ good | mid | bad | "
                "good_frac | PDB ids\n"
            )
            for row in af3_combos:
                f.write(
                    f"{row.get('interface', 'NA'):>9} | {row.get('rna', 'NA'):>3} | "
                    f"{row.get('protein', 'NA'):>7} | {row.get('n', 0):>2} | "
                    f"{row.get('dockq_good', 0):>10} | {row.get('dockq_mid', 0):>3} | "
                    f"{row.get('dockq_bad', 0):>3} | "
                    f"{_fmt_audit_stat_float(row.get('dockq_good_frac'), 3):>9} | "
                    f"{_fmt_audit_eid_list(row.get('eids'))}\n"
                )
            f.write("\n")

        f.write("Per-PDB AF3 entity breakdown:\n\n")
        for rec in entity_pred.get("per_eid") or []:
            eid = rec.get("exp_db_id", "")
            entities = rec.get("entities") or {}
            final_t = rec.get("final_tier")
            dockq_t = rec.get("dockq_truth", "NA")
            ent_parts = []
            for k in AUDIT_ENTITY_METRIC_GROUPS:
                val = entities.get(k, "NA")
                if k == "RNA":
                    af3_rna = rec.get("rna_tier_af3_plddt")
                    if af3_rna and af3_rna != val:
                        ent_parts.append(f"RNA={val}(pLDDT={af3_rna})")
                    else:
                        ent_parts.append(f"RNA={val}")
                else:
                    ent_parts.append(f"{k}={val}")
            ent_str = ", ".join(ent_parts)
            rank_t = rec.get("af3_ranking_score_tier", "NA")
            f.write(
                f"  {eid} → DockQ={dockq_t} | final={final_t} | {ent_str} | "
                f"af3_ranking_score={rank_t}\n\n"
            )


def plot_sample_audit_publication_figures(
    plot_creator, table_source, audit, rna_metrics=None, bin_profile=None, motif_bin_profiles=None,
    crmsd_bin_profile=None, entity_complex_predictions=None,
):
    """Sample-audit panels (saved under ``results_dir/interface`` via PlotCreator._save_audit_plot)."""
    if not audit or "quad_i_all" not in audit:
        return
    plot_specs = (
        "plot_audit_quadrant_rbp_dockq_stack",
        "plot_audit_failure_complex_rmsd_stack",
        "plot_audit_quadrant_outcome_heatmap",
    )
    missing = []
    for name in plot_specs:
        fn = getattr(plot_creator, name, None)
        if fn is None or not callable(fn):
            missing.append(name)
            continue
        fn(table_source, audit)
    if rna_metrics is not None:
        for scol in AUDIT_MOTIF_STRATIFY_COLUMNS:
            mmat = compute_audit_quadrant_outcome_motif_mean_matrix(
                rna_metrics, table_source, audit, scol
            )
            if mmat is not None:
                plot_creator.plot_audit_quadrant_outcome_motif_mean_heatmap(
                    table_source, mmat, scol
                )
        prof = bin_profile
        if prof is None:
            prof = compute_audit_interface_bin_profiles(rna_metrics, table_source, audit)
        if prof:
            plot_creator.plot_audit_bins_core_means_heatmap(table_source, prof)
            plot_creator.plot_audit_bins_af3_means_heatmap(table_source, prof)
            plot_creator.plot_audit_bins_mean_msa_depths(table_source, prof)
            plot_creator.plot_audit_bins_presence_and_templates(table_source, prof)
            prof_dockq = compute_audit_interface_bin_profiles(
                rna_metrics,
                table_source,
                audit,
                bin_suffixes=AUDIT_DOCKQ_ABC_BIN_SUFFIXES,
                pool_quadrants=True,
            )
            if prof_dockq:
                sfx = "_dockq_abc"
                plot_creator.plot_audit_bins_core_means_heatmap(
                    table_source, prof_dockq, filename_suffix=sfx
                )
                plot_creator.plot_audit_bins_af3_means_heatmap(
                    table_source, prof_dockq, filename_suffix=sfx
                )
        if motif_bin_profiles:
            for strat_k, mp in motif_bin_profiles.items():
                if not mp:
                    continue
                mat_c = mp.get("core_mat")
                mat_a = mp.get("af3_mat")
                if mat_c is None or not np.any(np.isfinite(mat_c)):
                    if mat_a is None or not np.any(np.isfinite(mat_a)):
                        continue
                sfx_m = f"_motif_{strat_k}"
                plot_creator.plot_audit_bins_core_means_heatmap(
                    table_source, mp, figsize=(22, 10), filename_suffix=sfx_m
                )
                plot_creator.plot_audit_bins_af3_means_heatmap(
                    table_source, mp, figsize=(14, 10), filename_suffix=sfx_m
                )
        if crmsd_bin_profile:
            plot_creator.plot_audit_crmsd_bins_combined_means_heatmap(
                table_source, crmsd_bin_profile
            )
        if entity_complex_predictions is not None:
            plot_audit_af3_metrics_figures(
                plot_creator,
                table_source,
                audit,
                rna_metrics,
                entity_complex_predictions,
            )
    # The classification-framework figures (pub_fig1..4) moved to
    # publication_figures/pub_fig1_classification_framework.py; run that script separately.
    if missing:
        print(
            "PlotCreator is missing sample-audit plot methods; skipping "
            + ", ".join(missing)
            + ". Copy plots/plotCreator.py from a repo that includes those definitions.",
            file=sys.stderr,
        )






def _write_sample_audit_section(f, audit):
    """Append sample audit: quadrants × (a)(b) with ids; (c) counts only then (ca)(cb)×(1–3)."""
    if not audit:
        return
    if "quad_i_all" not in audit:
        return
    tbl = audit.get("pred_table", "")
    f.write("=" * 80 + "\n")
    f.write("Sample audit: assembly vs monomer folds (Complex_iLDDT / Complex_Dockq / RBP_* )\n")
    f.write("=" * 80 + "\n")
    if tbl:
        f.write(f"Source prediction table: {tbl}\n")
    f.write("-" * 80 + "\n")
    f.write(
        "Predictions with finite RNA_LDDT and Protein_LDDT are grouped into quadrants (i)–(iv). "
        "After the prediction totals line, a single overview lists class counts for every bin in "
        "(i)–(iv). Under each quadrant only a total count is shown at the title; (a) and (b) list "
        "comma-separated "
        "exp_db_id entries (metrics in parentheses). For (c), only counts are printed "
        "for (c), (ca), and (cb); exp_db_id lists appear under (ca1)–(ca3), (cb1)–(cb3), and optional "
        "\"other\" lines.\n\n"
        "RBP_* values use the longest RNA chain (mean over protein–RNA pairs involving that chain).\n\n"
        "(i) RNA_LDDT > 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85; if longest RNA length ≥ 60, "
        "also RNA_TM finite and > 0.5.\n"
        "(ii) RNA_LDDT > 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85).\n"
        "(iii) RNA_LDDT ≤ 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85.\n"
        "(iv) RNA_LDDT ≤ 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85).\n"
        "Interpretation: Protein_TM must be finite and > 0.85 for (i) and (iii); non-finite Protein_TM "
        "is treated like Protein_TM ≤ 0.85 for placement in (ii) or (iv).\n\n"
        "Shared RBP_DockQ bins:\n"
        "  (a) RBP_DockQ ≥ 0.8.\n"
        "  (b) 0.23 ≤ RBP_DockQ < 0.8.\n"
        "  (c) RBP_DockQ < 0.23.\n"
        "Samples with non-finite RBP_DockQ are listed separately under that quadrant.\n\n"
        "Within (c) by Complex_RMSD (pred table, first numeric):\n"
        "  (ca) Complex_RMSD < 10 Å.\n"
        "  (cb) Complex_RMSD ≥ 10 Å.\n"
        "(c) rows with non-finite Complex_RMSD are counted only (no (ca)/(cb) split).\n\n"
        "Within each non-empty (ca) or (cb), subdivisions (counts + exp_db_id):\n"
        "  (1) RBP_iRMSD ≤ 10 Å AND (0.1 ≤ RBP_fnat < 0.5).\n"
        "  (2) RBP_iRMSD ≤ 10 Å AND 0 ≤ RBP_fnat < 0.1.\n"
        "  (3) RBP_iRMSD > 10 Å.\n"
        'Remainders under "(ca) other" / "(cb) other".\n\n'
    )

    n_tot = audit.get("n_predictions_finite_lddt")
    n_val = audit.get("n_predictions_valid_interface_metrics")
    if n_tot is not None and n_val is not None:
        f.write(
            f"Predictions with finite RNA_LDDT and Protein_LDDT: {n_tot}. "
            f"With finite non-zero RBP_DockQ, RBP_fnat, RBP_iRMSD (subset reference): {n_val}.\n\n"
        )

    def _write_global_class_overview(f_h):
        f_h.write("Overview (class counts):\n")
        for roman, prefix in (
            ("i", "quad_i"),
            ("ii", "quad_ii"),
            ("iii", "quad_iii"),
            ("iv", "quad_iv"),
        ):
            ap = f"{prefix}_"
            f_h.write(f"  ({roman})(a): {len(audit.get(ap + 'a') or [])}\n")
            f_h.write(f"  ({roman})(b): {len(audit.get(ap + 'b') or [])}\n")
            f_h.write(f"  ({roman})(c): {len(audit.get(ap + 'c') or [])}\n")
            f_h.write(f"  ({roman})(ca): {len(audit.get(ap + 'c_ca') or [])}\n")
            f_h.write(f"  ({roman})(cb): {len(audit.get(ap + 'c_cb') or [])}\n")
            f_h.write(f"  ({roman})(ca1): {len(audit.get(ap + 'c_ca_1') or [])}\n")
            f_h.write(f"  ({roman})(ca2): {len(audit.get(ap + 'c_ca_2') or [])}\n")
            f_h.write(f"  ({roman})(ca3): {len(audit.get(ap + 'c_ca_3') or [])}\n")
            f_h.write(f"  ({roman})(ca) other: {len(audit.get(ap + 'c_ca_other') or [])}\n")
            f_h.write(f"  ({roman})(cb1): {len(audit.get(ap + 'c_cb_1') or [])}\n")
            f_h.write(f"  ({roman})(cb2): {len(audit.get(ap + 'c_cb_2') or [])}\n")
            f_h.write(f"  ({roman})(cb3): {len(audit.get(ap + 'c_cb_3') or [])}\n")
            f_h.write(f"  ({roman})(cb) other: {len(audit.get(ap + 'c_cb_other') or [])}\n")
            f_h.write(
                f"  ({roman})(c) Complex_RMSD non-finite: "
                f"{len(audit.get(ap + 'c_crmsd_nonfinite') or [])}\n"
            )
            f_h.write(
                f"  ({roman}) Non-finite RBP_DockQ: "
                f"{len(audit.get(ap + 'unknown_rbp_dockq') or [])}\n"
            )
            f_h.write("\n")

    _write_global_class_overview(f)

    def _fv(x):
        try:
            xf = float(x)
            if np.isfinite(xf):
                return f"{xf:.4f}"
        except (TypeError, ValueError):
            pass
        return "n/a"

    def _line(entries, fmt_fn, indent="  "):
        if not entries:
            return indent + "(none)\n"
        parts = [fmt_fn(e) for e in entries]
        return indent + ", ".join(parts) + "\n"

    def _fmt_base_lddt_tm(r):
        return (
            f"{r['exp_db_id']} (RNALength={_fv(r['rna_length_longest'])}, "
            f"RNA_LDDT={r['rna_lddt']:.4f}, RNA_TM={_fv(r['rna_tm'])}, "
            f"Protein_LDDT={r['protein_lddt']:.4f}, Protein_TM={_fv(r['protein_tm'])}"
        )

    def _fmt_iface_full(r):
        return (
            f"{_fmt_base_lddt_tm(r)}, "
            f"Complex_iLDDT={_fv(r['complex_ilddt'])}, Complex_Dockq={_fv(r['complex_dockq'])}, "
            f"Complex_RMSD={_fv(r['complex_rmsd_first'])}, RNA_RMSD={_fv(r['rna_rmsd_first'])}, "
            f"RBP_iRMSD={_fv(r['rbp_irmsd'])}, RBP_fnat={_fv(r['rbp_fnat'])}, "
            f"RBP_DockQ={_fv(r['rbp_dockq'])}, RBP_QS={_fv(r['rbp_qs'])})"
        )

    def _write_ca_cb_branch(f_h, roman_label, key_prefix, branch):
        specs = (
            (
                "1",
                "RBP_iRMSD ≤ 10 Å AND (0.1 ≤ RBP_fnat < 0.5)",
                f"c_{branch}_1",
            ),
            (
                "2",
                "RBP_iRMSD ≤ 10 Å AND 0 ≤ RBP_fnat < 0.1",
                f"c_{branch}_2",
            ),
            (
                "3",
                "RBP_iRMSD > 10 Å",
                f"c_{branch}_3",
            ),
        )
        branch_recs = audit.get(f"{key_prefix}_c_{branch}") or []
        if not branch_recs:
            return
        f_h.write(
            f"      Subdivision of ({roman_label})({branch}) "
            "(labels (ca1)–(ca3) or (cb1)–(cb3) denote branch + bin):\n"
        )
        for sid, title, sfx in specs:
            bucket = audit.get(f"{key_prefix}_{sfx}") or []
            if not bucket:
                continue
            f_h.write(f"        ({roman_label})({branch}{sid}) {title}\n")
            f_h.write(f"          Count: {len(bucket)}\n")
            f_h.write("          exp_db_id:\n")
            f_h.write(_line(bucket, _fmt_iface_full, indent="            "))
        bo = audit.get(f"{key_prefix}_c_{branch}_other") or []
        if bo:
            f_h.write(f"        ({roman_label})({branch}) other\n")
            f_h.write(f"          Count: {len(bo)}\n")
            f_h.write("          exp_db_id:\n")
            f_h.write(_line(bo, _fmt_iface_full, indent="            "))

    def _write_c_followup(f_h, roman_label, key_prefix):
        ca = audit.get(f"{key_prefix}_c_ca") or []
        cb = audit.get(f"{key_prefix}_c_cb") or []
        c_nf = audit.get(f"{key_prefix}_c_crmsd_nonfinite") or []

        f_h.write(f"    ({roman_label})(ca) Complex_RMSD < 10 Å\n")
        f_h.write(f"      Count: {len(ca)}\n")
        _write_ca_cb_branch(f_h, roman_label, key_prefix, "ca")
        f_h.write("\n")

        f_h.write(f"    ({roman_label})(cb) Complex_RMSD ≥ 10 Å\n")
        f_h.write(f"      Count: {len(cb)}\n")
        _write_ca_cb_branch(f_h, roman_label, key_prefix, "cb")
        f_h.write("\n")

        if c_nf:
            f_h.write(
                f"    ({roman_label})(c) Complex_RMSD missing/non-finite "
                "(excluded from (ca)/(cb))\n"
            )
            f_h.write(f"      Count: {len(c_nf)}\n\n")

    def _write_quadrant(f_h, roman_label, condition_line, key_prefix):
        parent = audit.get(f"{key_prefix}_all") or []
        f_h.write(f"  ({roman_label}) {condition_line}\n")
        f_h.write(f"  Count: {len(parent)}\n\n")
        for abc, title in (
            ("a", "RBP_DockQ ≥ 0.8"),
            ("b", "0.23 ≤ RBP_DockQ < 0.8"),
            ("c", "RBP_DockQ < 0.23"),
        ):
            entries = audit.get(f"{key_prefix}_{abc}") or []
            f_h.write(f"  ({roman_label})({abc}) {title}\n")
            f_h.write(f"    Count: {len(entries)}\n")
            if abc in ("a", "b"):
                f_h.write("    exp_db_id:\n")
                f_h.write(_line(entries, _fmt_iface_full, indent="      "))
            f_h.write("\n")
            if abc == "c":
                _write_c_followup(f_h, roman_label, key_prefix)
        unk = audit.get(f"{key_prefix}_unknown_rbp_dockq") or []
        if unk:
            f_h.write(f"  ({roman_label}) Non-finite RBP_DockQ\n")
            f_h.write(f"    Count: {len(unk)}\n")
            f_h.write("    exp_db_id:\n")
            f_h.write(_line(unk, _fmt_iface_full, indent="      "))
            f_h.write("\n")

    _write_quadrant(
        f,
        "i",
        "RNA_LDDT > 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85 "
        "(if longest RNA ≥ 60 nt: RNA_TM > 0.5)",
        "quad_i",
    )
    _write_quadrant(
        f,
        "ii",
        "RNA_LDDT > 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85)",
        "quad_ii",
    )
    _write_quadrant(
        f,
        "iii",
        "RNA_LDDT ≤ 0.85 AND Protein_LDDT > 0.85 AND Protein_TM > 0.85",
        "quad_iii",
    )
    _write_quadrant(
        f,
        "iv",
        "RNA_LDDT ≤ 0.85 AND (Protein_LDDT ≤ 0.85 OR Protein_TM ≤ 0.85)",
        "quad_iv",
    )


def write_statistics_to_file(stats_dict, plot_creator, msa_option=None, single_chain_only=False, rbp_template_option=None):
    """Write statistics to a text file in the same directory as plots."""
    output_path = os.path.join(plot_creator.results_dir, 'evaluation_metrics_statistics.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Evaluation Metrics Statistics\n")
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
        
        # Process statistics, skipping the _values entries
        for metric_name, stats_tuple in stats_dict.items():
            if metric_name.endswith('_values'):
                continue  # Skip the values entries
            if metric_name.startswith('_'):
                continue  # Reserved (e.g. sample audit appended at end)
            if metric_name == "RNA interface class correlations" and isinstance(stats_tuple, dict):
                f.write("RNA interface class correlations (per RNA chain)\n")
                f.write("-" * 80 + "\n")
                f.write(
                    "Rows labeled missing are excluded. Ordinal mapping for correlation: "
                    "incorrect_interface=1, partial_interface=2, correct_interface=3.\n"
                    "RNA_*_AllChains: per predicted RNA column j, score is max over matrix rows i. "
                    "RBP_*: mean over protein–RNA pair entries involving that RNA chain. "
                    "af3_rna_pTM / af3_rna_ipTM: list elements aligned with RNAChainIDs indices. "
                    "Other scalar metrics: paired with the longest RNA chain label only.\n\n"
                )
                for mname, payload in stats_tuple.items():
                    rho = payload.get("rho", np.nan)
                    p = payload.get("p", np.nan)
                    f.write(f"{mname}:\n")
                    if np.isfinite(rho) and np.isfinite(p):
                        f.write(f"  Spearman rho: {rho:+.4f}, p={p:.4g}\n")
                    else:
                        f.write("  Spearman rho: n/a (insufficient variation/data)\n")
                    class_stats = payload.get("class_stats", {})
                    if class_stats:
                        f.write("  Per-class metric summary:\n")
                        for cname in ["incorrect_interface", "partial_interface", "correct_interface"]:
                            if cname in class_stats:
                                cs = class_stats[cname]
                                f.write(
                                    f"    {cname}: mean={cs['mean']:.4f}, median={cs['median']:.4f}, n={cs['count']}\n"
                                )
                    f.write("\n")
                continue
            if metric_name == "RNA INF vs interface class (longest RNA)" and isinstance(stats_tuple, dict):
                f.write("RNA INF vs interface class (longest RNA)\n")
                f.write("-" * 80 + "\n")
                f.write(
                    "RNA_INF list slots: index 1 = WC, 2 = non-WC, 3 = stack. "
                    "Interface class is taken from the longest RNA chain in RNAChainIDs / RNALength. "
                    "Rows labeled missing are excluded. Ordinal mapping for Spearman: "
                    "incorrect_interface=1, partial_interface=2, correct_interface=3.\n\n"
                )
                for mname, payload in stats_tuple.items():
                    rho = payload.get("rho", np.nan)
                    p = payload.get("p", np.nan)
                    f.write(f"{mname}:\n")
                    if np.isfinite(rho) and np.isfinite(p):
                        f.write(f"  Spearman rho: {rho:+.4f}, p={p:.4g}\n")
                    else:
                        f.write("  Spearman rho: n/a (insufficient variation/data)\n")
                    class_stats = payload.get("class_stats", {})
                    if class_stats:
                        f.write("  Per-class metric summary:\n")
                        for cname in ["incorrect_interface", "partial_interface", "correct_interface"]:
                            if cname in class_stats:
                                cs = class_stats[cname]
                                f.write(
                                    f"    {cname}: mean={cs['mean']:.4f}, median={cs['median']:.4f}, n={cs['count']}\n"
                                )
                    f.write("\n")
                continue
            if metric_name == "RNA interface class vs AF3 globals (single RNA chain)" and isinstance(stats_tuple, dict):
                f.write("RNA interface class vs AF3 globals (single RNA chain)\n")
                f.write("-" * 80 + "\n")
                f.write(
                    "Restricted to samples where RNA_interface_class contains exactly one RNA chain key "
                    "(single predicted RNA). Rows labeled missing are excluded. "
                    "Ordinal mapping for Spearman: incorrect_interface=1, partial_interface=2, correct_interface=3.\n\n"
                )
                for mname, payload in stats_tuple.items():
                    rho = payload.get("rho", np.nan)
                    p = payload.get("p", np.nan)
                    f.write(f"{mname}:\n")
                    if np.isfinite(rho) and np.isfinite(p):
                        f.write(f"  Spearman rho: {rho:+.4f}, p={p:.4g}\n")
                    else:
                        f.write("  Spearman rho: n/a (insufficient variation/data)\n")
                    class_stats = payload.get("class_stats", {})
                    if class_stats:
                        f.write("  Per-class metric summary:\n")
                        for cname in ["incorrect_interface", "partial_interface", "correct_interface"]:
                            if cname in class_stats:
                                cs = class_stats[cname]
                                f.write(
                                    f"    {cname}: mean={cs['mean']:.4f}, median={cs['median']:.4f}, n={cs['count']}\n"
                                )
                    f.write("\n")
                continue
            if metric_name == "Global RNA vs interface alignment" and isinstance(stats_tuple, dict):
                f.write("Global RNA vs interface alignment\n")
                f.write("-" * 80 + "\n")
                cap = stats_tuple.get("caption")
                if cap:
                    f.write(cap + "\n\n")
                n_s = stats_tuple.get("n_samples")
                if n_s is not None:
                    f.write(f"n samples (jointly finite RNA_TM, RNA_LDDT, RBP_DockQ, RBP_QS, RNA_ICS_AllChains): {n_s}\n\n")
                for k in sorted(stats_tuple.keys()):
                    if k.endswith("_rho"):
                        prefix = k[:-4]
                        pk = prefix + "_p"
                        if pk in stats_tuple:
                            rho = stats_tuple[k]
                            p = stats_tuple[pk]
                            f.write(f"{prefix}: Spearman rho={rho:+.4f}, p={p:.4g}\n")
                f.write("\nDelta-z summaries (large positive → global score high vs interface score low, cohort-relative):\n")
                for k in sorted(stats_tuple.keys()):
                    if not k.startswith("delta_z_"):
                        continue
                    blk = stats_tuple[k]
                    if not isinstance(blk, dict):
                        continue
                    f.write(f"  {k}:\n")
                    for kk in ("n", "mean", "std", "median", "pct_gt_0"):
                        if kk in blk:
                            fv = blk[kk]
                            if kk == "pct_gt_0":
                                f.write(f"    fraction Delta > 0: {fv:.4f}\n")
                            else:
                                f.write(f"    {kk}: {fv}\n")
                    f.write("\n")
                for ols_key in ("ols_rbp_dockq_on_rna_lddt", "ols_rbp_dockq_on_rna_tm"):
                    blk = stats_tuple.get(ols_key)
                    if isinstance(blk, dict):
                        f.write(f"{ols_key} (residual = RBP_DockQ - fitted):\n")
                        for kk, vv in blk.items():
                            f.write(f"  {kk}: {vv}\n")
                        f.write("\n")
                for qb_key, qtitle in (
                    ("rna_lddt_quartile_bins_rbp_dockq", "RNA LDDT quartile bins vs RBP_DockQ"),
                    ("rna_lddt_quartile_bins_rbp_qs", "RNA LDDT quartile bins vs RBP_QS"),
                ):
                    qb = stats_tuple.get(qb_key)
                    if isinstance(qb, dict) and qb:
                        f.write(f"{qtitle}:\n")
                        for interval_label, qstats in qb.items():
                            f.write(f"  {interval_label}: {qstats}\n")
                    f.write("\n")
                continue
            if isinstance(stats_tuple, tuple) and len(stats_tuple) == 3:
                mean, std, median = stats_tuple
                if mean is not None:
                    # Get count from corresponding _values entry
                    values_key = metric_name + '_values'
                    count = len([v for v in stats_dict.get(values_key, []) if v is not None]) if values_key in stats_dict else 0
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean: {mean:.4f}\n")
                    f.write(f"  Std: {std:.4f}\n")
                    f.write(f"  Median: {median:.4f}\n")
                    f.write(f"  Count: {count}\n\n")
    
        audit = stats_dict.get("_sample_audit_global_fold_interface")
        if audit:
            _write_sample_audit_section(f, audit)
        chi_block = stats_dict.get("_sample_audit_quadrant_dockq_chi2")
        if chi_block:
            _write_sample_audit_chi2_block(f, chi_block)
        fig_stats = stats_dict.get("_sample_audit_figure_statistics")
        if fig_stats:
            _write_sample_audit_figure_statistics(f, fig_stats)

    print(f"\nStatistics written to: {output_path}")




def plot_metrics(table, msa_option=None, single_chain_only=False, rbp_template_option=None):
    """Plot evaluation metrics with filtering options"""
    print("Processing evaluation metrics...")
    # Create plot creator first
    plot_creator = PlotCreator('evaluation_metrics', msa_option, single_chain_only, rbp_template_option)

    # Initialize metrics classes with plot_creator
    rna_metrics = RNAMetrics(msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option)

    # Dictionary to store statistics
    stats_dict = {}

    if table == "pred_protein_rna" or table == "pred_protein_rna_dna" or table == "pred_protein_dna":
        protein_metrics = ProteinMetrics(msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option)
        name = "Protein"
        prot_rmsd_values = protein_metrics.get_protein_rmsd_values(table)
        # print("!!!protein_rmsd_values", prot_rmsd_values)
        lddt_values = protein_metrics.get_protein_lddt_values(table)
        # print("protein lddt_values", lddt_values)
        tm_values = protein_metrics.get_protein_tm_values(table)
        # print("protein tm_values", tm_values)
        
        # Store statistics
        if prot_rmsd_values:
            stats_dict['Protein RMSD'] = calculate_mean_std(prot_rmsd_values)
            stats_dict['Protein RMSD_values'] = prot_rmsd_values
        if lddt_values:
            stats_dict['Protein LDDT'] = calculate_mean_std(lddt_values)
            stats_dict['Protein LDDT_values'] = lddt_values
        if tm_values:
            stats_dict['Protein TM'] = calculate_mean_std(tm_values)
            stats_dict['Protein TM_values'] = tm_values
        
        data_values = lddt_values, tm_values
        plot_creator.get_violin_plot(table,
                                     data_values=[lddt_values, tm_values],
                                     labels=[f'{name} LDDT', f'{name} TM'],
                                     name = "Protein_LDDTtm"
                                    )

    if table == "pred_protein_rna" or table == "pred_rna_rna":
        name = "RNA"
        rna_rmsd_values = rna_metrics.get_rna_rmsd_values(table)
        # print("rna_rmsd_values", rna_rmsd_values)
        lddt_values = rna_metrics.get_rna_lddt_values(table)
        # print("rna lddt_values", lddt_values)
        tm_values = rna_metrics.get_rna_tm_values(table)
        # print("rna tm_values", tm_values)
        
        # Store statistics
        if rna_rmsd_values:
            stats_dict['RNA RMSD'] = calculate_mean_std(rna_rmsd_values)
            stats_dict['RNA RMSD_values'] = rna_rmsd_values
        if lddt_values:
            stats_dict['RNA LDDT'] = calculate_mean_std(lddt_values)
            stats_dict['RNA LDDT_values'] = lddt_values
        if tm_values:
            stats_dict['RNA TM'] = calculate_mean_std(tm_values)
            stats_dict['RNA TM_values'] = tm_values
        plot_creator.get_violin_plot(table,
                                     data_values=[lddt_values, tm_values],
                                     labels=[f'{name} LDDT', f'{name} TM'],
                                     name="RNA_LDDTtm"
                                     )
        if table == "pred_rna_rna": #protein_metrics.get_protein_rmsd_values(table) == []:
            plot_creator.get_violin_plot(table,
                                         data_values=[rna_rmsd_values],
                                         labels=[f'{name} RMSD'],
                                         name="RNA_RMSD"
                                         )
        if table == "pred_protein_rna" and protein_metrics.get_protein_rmsd_values(table) != []:
            plot_creator.get_violin_plot(table,
                                         data_values=[prot_rmsd_values, rna_rmsd_values],
                                         labels=["Protein RMSD","RNA RMSD"],
                                         name="Protein_RNA_RMSD"
                                         )

        all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values(table)
        all_inf = [value for value in all_inf if value != -1]
        wc_inf = [value for value in wc_inf if value != -1]
        nonWC_inf = [value for value in nonWC_inf if value != -1]
        stacking_inf = [value for value in stacking_inf if value != -1]
        
        # Store INF statistics
        if all_inf:
            stats_dict['RNA INF'] = calculate_mean_std(all_inf)
            stats_dict['RNA INF_values'] = all_inf
        if wc_inf:
            stats_dict['RNA INF-WC'] = calculate_mean_std(wc_inf)
            stats_dict['RNA INF-WC_values'] = wc_inf
        if nonWC_inf:
            stats_dict['RNA INF-nonWC'] = calculate_mean_std(nonWC_inf)
            stats_dict['RNA INF-nonWC_values'] = nonWC_inf
        if stacking_inf:
            stats_dict['RNA INF-STACK'] = calculate_mean_std(stacking_inf)
            stats_dict['RNA INF-STACK_values'] = stacking_inf
        plot_creator.get_boxplot(table,
                                 data_values=[all_inf, wc_inf, nonWC_inf, stacking_inf],
                                 labels=["INF", "INF-WC", "INF-nonWC", "INF-STACK"],
                                 name="RNA_INFs"
                                 )
        rna_di = rna_metrics.get_rna_di_values(table)
        # print("all_inf, wc_inf, nonWC_inf, stacking_inf", all_inf, wc_inf, nonWC_inf, stacking_inf)
        # print("rna_di", rna_di)
        
        # Store DI statistics
        if rna_di:
            stats_dict['RNA DI'] = calculate_mean_std(rna_di)
            stats_dict['RNA DI_values'] = rna_di
        di_length_data = rna_metrics.get_di_and_length(table)
        plot_creator.get_scatterplot(table,
                                     di_length_data,
                                     xAxis_label = "RNA Sequence Length",
                                     yAxis_label = "DI",
                                     name="DI_RNAlength")

        lddt_length_data = rna_metrics.get_x_and_length("RNA_LDDT", table)
        # get_scatterplot(lddt_length_data, "LDDT", table, "RNA Sequence Length")
        plot_creator.get_scatterplot(table,
                                     lddt_length_data,
                                     xAxis_label="RNA Sequence Length",
                                     yAxis_label="LDDT",
                                     name="LDDT_RNAlength")
        rmsd_length_data = rna_metrics.get_rmsd_and_length(table)
        plot_creator.get_scatterplot(table,
                                     rmsd_length_data,
                                     xAxis_label="RNA Sequence Length",
                                     yAxis_label="RMSD",
                                     name="RMSD_RNAlength")
        tm_length_data = rna_metrics.get_x_and_length("RNA_TM", table)
        plot_creator.get_scatterplot(table,
                                     tm_length_data,
                                     xAxis_label="RNA Sequence Length",
                                     yAxis_label="TM-score",
                                     name="TM_RNAlength")

        exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
            table, "RNA_LDDT")
        plot_creator.get_scatterplot(table,
                                     all_inf,
                                     xAxis_label="INF",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score = extra_vals,
                                     name="RNA-LDDT_INF")
        plot_creator.get_scatterplot(table,
                                     wc_inf,
                                     xAxis_label="INF-WC",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF-WC")
        plot_creator.get_scatterplot(table,
                                     nonWC_inf,
                                     xAxis_label="INF-nonWC",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF-nonWC")
        plot_creator.get_scatterplot(table,
                                     stacking_inf,
                                     xAxis_label="INF-STACK",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF-STACK")
        plot_creator.get_scatterplot(table,
                                     stacking_inf,
                                     xAxis_label="INF-STACK",
                                     yAxis_label="INF-nonWC",
                                     yAxis_score=nonWC_inf,
                                     name="RNA-INF-nonWC_stack")
        exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
            table, "RNA_TM")
        plot_creator.get_scatterplot(table,
                                     all_inf,
                                     xAxis_label="INF",
                                     yAxis_label=r"$\mathrm{TM}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-TM_INF")
        plot_creator.get_scatterplot(table,
                                     nonWC_inf,
                                     xAxis_label="INF-nonWC",
                                     yAxis_label=r"$\mathrm{TM}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-TM_INF-nonWC")
        plot_creator.get_scatterplot(table,
                                     stacking_inf,
                                     xAxis_label="INF-STACK",
                                     yAxis_label=r"$\mathrm{TM}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-TM_INF-STACK")

        exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
            table, "RNA_LDDT_allAtom")
        plot_creator.get_scatterplot(table,
                                     all_inf,
                                     xAxis_label="INF",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF")
        plot_creator.get_scatterplot(table,
                                     nonWC_inf,
                                     xAxis_label="INF-nonWC",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF-nonWC")
        plot_creator.get_scatterplot(table,
                                     stacking_inf,
                                     xAxis_label="INF-STACK",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     yAxis_score=extra_vals,
                                     name="RNA-LDDT_INF-STACK")

        if table == "pred_protein_rna":

            exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
                table, "iLDDT")
            plot_creator.get_scatterplot(table,
                                         all_inf,
                                         xAxis_label="INF",
                                         yAxis_label="iLDDT",
                                         yAxis_score=extra_vals,
                                         name="iLDDT_INF")
            plot_creator.get_scatterplot(table,
                                         nonWC_inf,
                                         xAxis_label="INF-nonWC",
                                         yAxis_label="iLDDT",
                                         yAxis_score=extra_vals,
                                         name="iLDDT_INF-nonWC")
            plot_creator.get_scatterplot(table,
                                         stacking_inf,
                                         xAxis_label="INF-STACK",
                                         yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                         yAxis_score=extra_vals,
                                         name="iLDDT_INF-STACK")

            exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
                table, "QS_global")
            plot_creator.get_scatterplot(table,
                                         all_inf,
                                         xAxis_label="INF",
                                         yAxis_label="QS",
                                         yAxis_score=extra_vals,
                                         name="QS_INF")
            plot_creator.get_scatterplot(table,
                                         nonWC_inf,
                                         xAxis_label="INF-nonWC",
                                         yAxis_label="QS",
                                         yAxis_score=extra_vals,
                                         name="QS_INF-nonWC")
            plot_creator.get_scatterplot(table,
                                         stacking_inf,
                                         xAxis_label="INF-STACK",
                                         yAxis_label="QS",
                                         yAxis_score=extra_vals,
                                         name="QS_INF-STACK")

            exp_ids, extra_vals, all_inf, wc_inf, nonWC_inf, stacking_inf = rna_metrics.get_rna_inf_values_with_extra(
                table, "ICS")
            plot_creator.get_scatterplot(table,
                                         all_inf,
                                         xAxis_label="INF",
                                         yAxis_label="ICS",
                                         yAxis_score=extra_vals,
                                         name="ICS_INF")
            plot_creator.get_scatterplot(table,
                                         nonWC_inf,
                                         xAxis_label="INF-nonWC",
                                         yAxis_label="ICS",
                                         yAxis_score=extra_vals,
                                         name="ICS_INF-nonWC")
            plot_creator.get_scatterplot(table,
                                         stacking_inf,
                                         xAxis_label="INF-STACK",
                                         yAxis_label="ICS",
                                         yAxis_score=extra_vals,
                                         name="ICS-STACK")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_ICS_AllChains", "RNA_f1_score", table),
                                        xAxis_label="RNA ICS (longest RNA)",
                                         yAxis_label="RNA_f1_score",
                                         name="ICS_RNA_f1_score")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_ICS_AllChains", "RNA_precision", table),
                                        xAxis_label="RNA ICS (longest RNA)",
                                         yAxis_label="RNA_precision",
                                         name="ICS_RNA_precision")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_ICS_AllChains", "RNA_recall", table),
                                        xAxis_label="RNA ICS (longest RNA)",
                                         yAxis_label="RNA_recall",
                                         name="ICS_RNA_recall")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_ICS_AllChains", "RNA_mcc", table),
                                        xAxis_label="RNA ICS (longest RNA)",
                                         yAxis_label="RNA_mcc",
                                         name="ICS_RNA_mcc")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_ICS_AllChains", "RNA_wl", table),
                                        xAxis_label="RNA ICS (longest RNA)",
                                         yAxis_label="RNA_wl",
                                         name="ICS_RNA_wl")

            plot_creator.get_scatterplot(table,
                                         rna_metrics.get_x_and_y("Complex_LDDT_allAtom", "RNA_wl", table),
                                         xAxis_label="Complex_LDDT_allAtom",
                                         yAxis_label="RNA_wl",
                                         name="Complex_LDDT_allAtom_RNA_wl")

            plot_creator.get_scatterplot(table,
                                         rna_metrics.get_x_and_y("Complex_LDDT_allAtom", "RNA_mcc", table),
                                         xAxis_label="Complex_LDDT_allAtom",
                                         yAxis_label="RNA_mcc",
                                         name="Complex_LDDT_allAtom_RNA_mcc")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_QS_global_AllChains", "RNA_wl", table),
                                        xAxis_label="RNA QS_global (longest RNA)",
                                         yAxis_label="RNA_wl",
                                         name="QS_global_RNA_wl")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_QS_global_AllChains", "RNA_mcc", table),
                                        xAxis_label="RNA QS_global (longest RNA)",
                                         yAxis_label="RNA_mcc",
                                         name="QS_global_RNA_mcc")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_iLDDT_AllChains", "RNA_wl", table),
                                        xAxis_label="RNA iLDDT (longest RNA)",
                                         yAxis_label="RNA_wl",
                                         name="iLDDT_RNA_wl")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RNA_iLDDT_AllChains", "RNA_mcc", table),
                                        xAxis_label="RNA iLDDT (longest RNA)",
                                         yAxis_label="RNA_mcc",
                                         name="iLDDT_RNA_mcc")

            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RBP_DockQ", "RNA_mcc", table),
                                        xAxis_label="RBP DockQ (avg over proteins vs longest RNA)",
                                        yAxis_label="RNA_mcc",
                                        name="RBP_DockQ_RNA_mcc")
            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RBP_QS", "RNA_mcc", table),
                                        xAxis_label="RBP QS (avg over proteins vs longest RNA)",
                                        yAxis_label="RNA_mcc",
                                        name="RBP_QS_RNA_mcc")
            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RBP_fnat", "RNA_mcc", table),
                                        xAxis_label="RBP fnat (avg over proteins vs longest RNA)",
                                        yAxis_label="RNA_mcc",
                                        name="RBP_fnat_RNA_mcc")
            plot_creator.get_scatterplot(table,
                                        rna_metrics.get_longest_rna_interaction_pairs("RBP_irmsd", "RNA_mcc", table),
                                        xAxis_label="RBP iRMSD (avg over proteins vs longest RNA)",
                                        yAxis_label="RNA_mcc",
                                        name="RBP_irmsd_RNA_mcc")

            conditionGC = "RNA_GC_Content >= 0.5"
            gc_pTM_content = rna_metrics.get_x_and_y_condition("af3_rna_ipTM", "RNA_TM", conditionGC, table)
            plot_creator.get_scatterplot(table,
                                         gc_pTM_content,
                                         xAxis_label="RNA pTM",
                                         yAxis_label=r"$\mathrm{TM}_{r}$",
                                         name="RNA-GC>05_pTM_TM")

            au_lddt_content = rna_metrics.get_longest_rna_interaction_pairs("af3_rna_ipTM", "RNA_iLDDT_AllChains", table)
            plot_creator.get_scatterplot(table,
                                         au_lddt_content,
                                         xAxis_label="AF3 RNA ipTM",
                                        yAxis_label="RNA iLDDT (longest RNA)",
                                         name="ipTM_iLDDT")

        gc_lddt_content = rna_metrics.get_x_and_y("RNA_GC_Content", "RNA_LDDT", table)
        plot_creator.get_scatterplot(table,
                                     gc_lddt_content,
                                     xAxis_label="RNA GC-Content",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     name="RNA-GC_LDDT")

        plot_creator.get_scatterplot(table,
                                     rna_metrics.get_x_and_y("RNA_LDDT_allAtom", "RNA_wl", table),
                                     xAxis_label="RNA_LDDT_allAtom",
                                     yAxis_label="RNA_wl",
                                     name="RNA_LDDT_allAtom_RNA_wl")

        plot_creator.get_scatterplot(table,
                                     rna_metrics.get_x_and_y("RNA_TM", "RNA_wl", table),
                                     xAxis_label="RNA_TM",
                                     yAxis_label="RNA_wl",
                                     name="RNA_TM_RNA_wl")

        plot_creator.get_scatterplot(table,
                                     rna_metrics.get_x_and_y("RNA_LDDT_allAtom", "RNA_mcc", table),
                                     xAxis_label="RNA_LDDT_allAtom",
                                     yAxis_label="RNA_mcc",
                                     name="RNA_LDDT_RNA_mcc")

        plot_creator.get_scatterplot(table,
                                     rna_metrics.get_x_and_y("RNA_TM", "RNA_mcc", table),
                                     xAxis_label="RNA_TM",
                                     yAxis_label="RNA_mcc",
                                     name="RNA_TM_RNA_mcc")

        au_lddt_content = rna_metrics.get_x_and_y("RNA_AU_Content", "RNA_LDDT", table)
        plot_creator.get_scatterplot(table,
                                     au_lddt_content,
                                     xAxis_label="RNA AU-Content",
                                     yAxis_label=r"$\mathrm{LDDT}_{r}$",
                                     name="RNA-AU_LDDT")

        gc_tm_content = rna_metrics.get_x_and_y("RNA_GC_Content", "RNA_TM", table)
        plot_creator.get_scatterplot(table,
                                     gc_tm_content,
                                     xAxis_label="RNA GC-Content",
                                     yAxis_label=r"$\mathrm{TM}_{r}$",
                                     name="RNA-GC_TM")

        plot_creator.get_scatterplot(table,
                                     rna_metrics.get_x_and_y("RNA_GC_Content", "RNA_LDDT_allAtom", table),
                                     xAxis_label="RNA GC-Content",
                                     yAxis_label="RNA_LDDT_allAtom",
                                     name="RNA-GC_RNA_LDDT_allAtom")

        au_tm_content = rna_metrics.get_x_and_y("RNA_AU_Content", "RNA_TM", table)
        plot_creator.get_scatterplot(table,
                                     au_tm_content,
                                     xAxis_label="RNA AU-Content",
                                     yAxis_label=r"$\mathrm{TM}_{r}$",
                                     name="RNA-AU_TM")

        if table == "pred_protein_rna":
            # Add MSA comparison violin plots for all metrics
            metrics_to_compare = [
                ('RNA_RMSD', 'RMSD [Å]'),
                ('RNA_TM', r"$\mathrm{TM}_{r}$"),
                ('RNA_LDDT', r"$\mathrm{LDDT}_{r}$"),
                ('RNA_LDDT_allAtom', r"$\mathrm{LDDT}_{r}$ allAtom"),
                ('Complex_LDDT_allAtom', 'Complex LDDT allAtom'),
                ('iLDDT', 'iLDDT'),
                ('QS_global', 'QS'),
                ('ICS', 'ICS')
            ]
        elif table == "pred_rna_rna":
            metrics_to_compare = [
                ('RNA_RMSD', 'RMSD [Å]'),
                ('RNA_TM', r"$\mathrm{TM}_{r}$"),
                ('RNA_LDDT', r"$\mathrm{LDDT}_{r}$"),
                ('RNA_LDDT_allAtom', r"$\mathrm{LDDT}_{r}$ allAtom")
            ]

        for metric_col, ylabel in metrics_to_compare:
            msa_values, no_msa_values = rna_metrics.get_msa_metric_comparison(table, metric_col)
            if msa_values and no_msa_values:  # Only create plot if we have data for both cases
                plot_creator.get_violin_plot(
                    table_source='rna_metrics',
                    data_values=[msa_values, no_msa_values],
                    labels=['+MSA', '-MSA'],
                    name=f'{metric_col}_MSA_comparison',
                    yAxis_label=ylabel
                )

        # Add MSA comparison boxplots for INF metrics
        inf_values = rna_metrics.get_msa_inf_comparison(table)
        if inf_values['+MSA']['all'] and inf_values['-MSA']['all']:  # Check if we have data
            # Create boxplot for each INF type
            inf_types = ['all', 'wc', 'nonwc', 'stack']
            labels = ['INF', 'INF-WC', 'INF-nonWC', 'INF-STACK']

            for inf_type, label in zip(inf_types, labels):
                plot_creator.get_boxplot(
                    table_source='rna_metrics',
                    data_values=[inf_values['+MSA'][inf_type], inf_values['-MSA'][inf_type]],
                    labels=['+MSA', '-MSA'],
                    name=f'{label}_MSA_comparison',
                    yAxis_label=label
                )

        # Add MSA comparison for DI
        msa_di, no_msa_di = rna_metrics.get_msa_di_comparison(table)
        if msa_di and no_msa_di:  # Only create plot if we have data for both cases
            plot_creator.get_boxplot(
                table_source='rna_metrics',
                data_values=[msa_di, no_msa_di],
                labels=['+MSA', '-MSA'],
                name='DI_MSA_comparison',
                yAxis_label='Deformation Index'
            )

        if table == "pred_protein_rna":
            iface_metrics = [
                "RNA_TM",
                "RNA_LDDT",
                "RNA_LDDT_allAtom",
                "RNA_wl",
                "RNA_mcc",
                "RNA_f1_score",
                "RNA_precision",
                "RNA_recall",
                "RNA_TM_AllChains",
                "RNA_LDDT_AllChains",
                "RNA_iLDDT_AllChains",
                "RNA_QS_global_AllChains",
                "RNA_Dockq_AllChains",
                "RNA_ICS_AllChains",
                "RBP_DockQ",
                "RBP_QS",
                "RBP_fnat",
                "RBP_irmsd",
                "af3_rna_pTM",
                "af3_rna_ipTM",
            ]
            stats_dict["RNA interface class correlations"] = rna_metrics.get_longest_rna_interface_class_correlations(
                table, iface_metrics
            )
            iface_corr = stats_dict.get("RNA interface class correlations") or {}
            iface_class_order = [
                ("incorrect_interface", "incorrect (1)"),
                ("partial_interface", "partial (2)"),
                ("correct_interface", "correct (3)"),
            ]
            for mname, payload in iface_corr.items():
                byv = payload.get("by_class_values") or {}
                data_use = []
                labels_use = []
                for key, lab in iface_class_order:
                    vals = byv.get(key, [])
                    if vals:
                        data_use.append(vals)
                        labels_use.append(lab)
                if sum(len(v) for v in data_use) < 2:
                    continue
                safe_plot_name = mname.replace("/", "_").replace(" ", "_")
                plot_creator.get_violin_plot(
                    table,
                    data_values=data_use,
                    labels=labels_use,
                    name=f"RNA_iface_class_{safe_plot_name}",
                    yAxis_label=mname,
                    xAxis_label="Interface class (per RNA chain)",
                    rotation=15,
                )

            inf_iface_buckets = rna_metrics.get_longest_rna_inf_by_interface_class(table)
            inf_plot_specs = [
                ("INF-WC", "RNA INF-WC"),
                ("INF-nonWC", "RNA INF-nonWC"),
                ("INF-STACK", "RNA INF-STACK"),
            ]
            stats_dict["RNA INF vs interface class (longest RNA)"] = {
                y_label: _summarize_interface_class_buckets(inf_iface_buckets.get(bucket_key) or {})
                for bucket_key, y_label in inf_plot_specs
            }
            for bucket_key, y_label in inf_plot_specs:
                byv = inf_iface_buckets.get(bucket_key) or {}
                inf_data_use = []
                inf_labels_use = []
                for key, lab in iface_class_order:
                    vals = byv.get(key, [])
                    if vals:
                        inf_data_use.append(vals)
                        inf_labels_use.append(lab)
                if sum(len(v) for v in inf_data_use) < 2:
                    continue
                inf_safe_name = bucket_key.replace("-", "_")
                plot_creator.get_violin_plot(
                    table,
                    data_values=inf_data_use,
                    labels=inf_labels_use,
                    name=f"RNA_iface_class_{inf_safe_name}_longestRNA",
                    yAxis_label=y_label,
                    xAxis_label="Interface class (longest RNA)",
                    rotation=15,
                )

            single_af3_buckets = rna_metrics.get_single_rna_interface_af3_global_buckets(table)
            single_af3_cols = (
                "af3_global_pae_avg",
                "af3_ranking_score",
                "af3_chain_pair_pae_min",
            )
            stats_dict["RNA interface class vs AF3 globals (single RNA chain)"] = {
                col: _summarize_interface_class_buckets(single_af3_buckets.get(col) or {})
                for col in single_af3_cols
            }
            for col in single_af3_cols:
                byv = single_af3_buckets.get(col) or {}
                s_data_use = []
                s_labels_use = []
                for key, lab in iface_class_order:
                    vals = byv.get(key, [])
                    if vals:
                        s_data_use.append(vals)
                        s_labels_use.append(lab)
                if sum(len(v) for v in s_data_use) < 2:
                    continue
                plot_creator.get_violin_plot(
                    table,
                    data_values=s_data_use,
                    labels=s_labels_use,
                    name=f"RNA_iface_class_{col}_singleRNAchain",
                    yAxis_label=col,
                    xAxis_label="Interface class (single RNA chain)",
                    rotation=15,
                )

            align_series = rna_metrics.get_global_vs_interface_alignment_series(table)
            if align_series is not None:
                stats_dict["Global RNA vs interface alignment"] = _analyze_global_vs_interface_alignment(
                    align_series
                )
                _plot_global_vs_interface_alignment(plot_creator, table, align_series)

            audit_lists = rna_metrics.get_sample_audit_global_fold_interface_lists(table)
            if audit_lists is not None:
                stats_dict["_sample_audit_global_fold_interface"] = audit_lists
                chi_audit = compute_sample_audit_quadrant_dockq_chi2(audit_lists)
                if chi_audit is not None:
                    stats_dict["_sample_audit_quadrant_dockq_chi2"] = chi_audit
                bin_prof = compute_audit_interface_bin_profiles(rna_metrics, table, audit_lists)
                cohen_d_stats = compute_audit_interface_class_cohen_d(rna_metrics, table, audit_lists)
                fold_eff = compute_audit_fold_quadrant_and_crmsd_effects(rna_metrics, table, audit_lists)
                motif_prof = {}
                for strat in AUDIT_MOTIF_STRATIFY_COLUMNS:
                    mp = compute_motif_bin_profiles(rna_metrics, table, strat)
                    if mp:
                        motif_prof[strat] = mp
                motif_quadrant_mats = {}
                for strat in AUDIT_MOTIF_STRATIFY_COLUMNS:
                    mmat = compute_audit_quadrant_outcome_motif_mean_matrix(
                        rna_metrics, table, audit_lists, strat
                    )
                    if mmat is not None:
                        motif_quadrant_mats[strat] = mmat
                motif_summaries = compute_motif_score_table_summaries(rna_metrics, table)
                crmsd_prof = compute_crmsd_bin_profiles(rna_metrics, table)
                tier_intervals = compute_audit_metric_tier_intervals(rna_metrics, table, audit_lists)
                consensus_tier_intervals = compute_audit_consensus_metric_tier_intervals(
                    rna_metrics, table, audit_lists, dockq_tiers=tier_intervals
                )
                consensus_pdb_lists = compute_audit_consensus_pdb_tier_lists(
                    rna_metrics, table, audit_lists, consensus_tier_intervals
                )
                entity_predictions = compute_audit_entity_complex_predictions(
                    rna_metrics, table, audit_lists, consensus_tier_intervals
                )
                try:
                    # written to results/ (git-ignored), not over the tracked copy in database/
                    _model_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
                    )
                    os.makedirs(_model_dir, exist_ok=True)
                    export_audit_entity_tier_model_json(
                        os.path.join(_model_dir, f"audit_entity_tier_model_{table}.json"),
                        consensus_tier_intervals,
                        source_table=table,
                    )
                except Exception as _model_export_err:
                    print(f"Warning: entity tier model export failed: {_model_export_err}")
                crmsd_tier_intervals = compute_audit_crmsd_metric_tier_intervals(rna_metrics, table)
                rna_quad_breakdown = compute_audit_quadrant_rna_feature_influence_breakdown(
                    rna_metrics, table, audit_lists
                )
                stats_dict["_sample_audit_figure_statistics"] = build_sample_audit_figure_statistics(
                    audit_lists,
                    bin_prof,
                    ac_cohen_d=cohen_d_stats,
                    fold_crmsd_effects=fold_eff,
                    motif_bin_profiles=motif_prof if motif_prof else None,
                    motif_quadrant_matrices=motif_quadrant_mats if motif_quadrant_mats else None,
                    motif_score_summaries=motif_summaries,
                    crmsd_bin_profile=crmsd_prof,
                    metric_tier_intervals=tier_intervals,
                    crmsd_metric_tier_intervals=crmsd_tier_intervals,
                    consensus_metric_tier_intervals=consensus_tier_intervals,
                    consensus_pdb_tier_lists=consensus_pdb_lists,
                    entity_complex_predictions=entity_predictions,
                    rna_feature_quadrant_breakdown=rna_quad_breakdown,
                )
                plot_sample_audit_publication_figures(
                    plot_creator,
                    table,
                    audit_lists,
                    rna_metrics,
                    bin_profile=bin_prof,
                    motif_bin_profiles=motif_prof if motif_prof else None,
                    crmsd_bin_profile=crmsd_prof,
                    entity_complex_predictions=entity_predictions,
                )

    if table == "pred_protein_dna" or table == "pred_protein_rna_dna":
        dna_metrics = DNAMetrics(msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option)
        name = "DNA"
        dna_rmsd_values = dna_metrics.get_dna_rmsd_values(table)
        print("dna_rmsd_values", rna_rmsd_values)
        lddt_values = dna_metrics.get_dna_lddt_values(table)
        print("dna lddt_values", lddt_values)
        tm_values = dna_metrics.get_dna_tm_values(table)
        print("dna tm_values", tm_values)
        
        # Store DNA statistics
        if dna_rmsd_values:
            stats_dict['DNA RMSD'] = calculate_mean_std(dna_rmsd_values)
            stats_dict['DNA RMSD_values'] = dna_rmsd_values
        if lddt_values:
            stats_dict['DNA LDDT'] = calculate_mean_std(lddt_values)
            stats_dict['DNA LDDT_values'] = lddt_values
        if tm_values:
            stats_dict['DNA TM'] = calculate_mean_std(tm_values)
            stats_dict['DNA TM_values'] = tm_values
        plot_creator.get_violin_plot(table,
                                     data_values=[lddt_values, tm_values],
                                     labels=[f'{name} LDDT', f'{name} TM'],
                                     name="DNA_LDDTtm"
                                     )
        if rna_metrics.get_rna_rmsd_values(table) != []:
            plot_creator.get_violin_plot(table,
                                         data_values=[prot_rmsd_values, rna_rmsd_values, dna_rmsd_values],
                                         labels=["Protein RMSD", "RNA RMSD", "DNA RMSD"],
                                         name="Protein-RNA-DNA_RMSD"
                                         )
        lddt_length_data = dna_metrics.get_x_and_length("DNA_LDDT", table)
        plot_creator.get_scatterplot(table,
                                     lddt_length_data,
                                     xAxis_label="DNA Sequence Length",
                                     yAxis_label="LDDT",
                                     name="LDDT_DNAlength")
        tm_length_data = dna_metrics.get_x_and_length("DNA_TM", table)
        plot_creator.get_scatterplot(table,
                                     tm_length_data,
                                     xAxis_label="DNA Sequence Length",
                                     yAxis_label="TM",
                                     name="TM_DNAlength")
        rmsd_length_data = dna_metrics.get_rmsd_and_length(table)
        plot_creator.get_scatterplot(table,
                                     rmsd_length_data,
                                     xAxis_label="DNA Sequence Length",
                                     yAxis_label="RMSD",
                                     name="RMSD_DNAlength")

    # Write statistics to file
    write_statistics_to_file(stats_dict, plot_creator, msa_option, single_chain_only, rbp_template_option)
    
    if table == "pred_protein_rna" or table == "pred_protein_dna" or table == "pred_protein_rna_dna":
        protein_metrics.close_connection()
    if table == "pred_rna_rna":
        rna_metrics.close_connection()


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
# evaluationMetrics builds whole multi-panel figures, so composition here reuses
# the PNGs it already saved (run the script first). Panel names are the saved
# plot base names (suffixes like _all_chains optional).
_COMPOSITE_PLOT_TYPES = ("evaluation_metrics",)


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None,
                     rbp_template_option=None):
    """Return {base_name: (draw_fn, title)} from this script's saved PNGs."""
    from plots.publication_figure import saved_png_panels
    return saved_png_panels(
        _COMPOSITE_PLOT_TYPES, names, single_chain_only=single_chain_only,
        msa_option=msa_option, rbp_template_option=rbp_template_option)


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="evaluation_metrics_composite"))
    # Get table names from startConfig
    config = StartConfig()
    pred_table = config.pred_table
    
    # Parse command-line arguments
    single_chain_only = False
    msa_option = None
    rbp_template_option = None
    pub_only = False

    # Parse arguments (case-insensitive, any order)
    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in ['single_chain', 'all']:
            single_chain_only = (arg_lower == 'single_chain')
        elif arg_lower in ['+msa', '-msa', 'msa']:
            msa_option = arg  # Keep original case
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case
        elif arg_lower in ['--pub', 'pub']:
            pub_only = True

    # Normalize MSA option format
    if msa_option and msa_option.upper() not in ['+MSA', '-MSA']:
        msa_lower = msa_option.lower()
        msa_option = '+MSA' if msa_lower in ['+msa', 'msa'] else '-MSA' if msa_lower == '-msa' else None

    if pub_only:
        print("The --pub classification-framework figures moved to a standalone script.\n"
              "Run:  python publication_figures/pub_fig1_classification_framework.py")
    else:
        plot_metrics(pred_table, msa_option, single_chain_only, rbp_template_option)
