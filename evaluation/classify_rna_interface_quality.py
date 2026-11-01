#!/usr/bin/env python3
"""
Classify predicted protein–RNA complexes in pred_protein_rna using existing DB metrics.

Interprets (all metric columns are JSON keyed dict-lists):
  - ProteinChainIDs / RNAChainIDs: JSON lists or single chain id (optional ``//`` comment stripped).
  - RNA_TM_AllChains / RNA_LDDT_AllChains: per predicted RNA chain, e.g. [{"B": 0.13}, {"C": 0.15}];
    used directly as that chain's score.
  - RNA_iLDDT_AllChains / RNA_ICS_AllChains: per protein–RNA interface, e.g. [{"A:B": 0.02}, ...];
    reduced to a per predicted RNA-chain score = max over interfaces involving that chain.
  - RBP_fnat, RBP_irmsd, RBP_DockQ, RBP_QS: per protein–RNA interface, e.g. [{"A:E": 0.5}, {"B:E": 0.1}, ...]
    Per RNA chain, all entries involving that chain (protein–RNA and RNA–RNA) are aggregated
    (mean across pairs) and combined with that chain's RNA_*_AllChains scores.

``RNA_interface_class`` is a JSON object mapping each predicted RNA chain id to a category
(e.g. ``{"A": "correct_interface", "B": "partial_interface"}``), using per-chain matrix
columns and all RBP interfaces involving that chain with proteins or other RNAs.

DockQ/QS CAPRI bins (OpenStructure cutoffs) are computed per chain for classification and
detail JSON. CSV is written to the StartConfig parent folder (same level as ``af/`` and
``pdb/``) as ``rna_interface_classification.csv``. Only ``RNA_interface_class`` is written to the SQLite DB on each run (unless ``--dry-run`` or ``--skip-db``).

A **full-table run** (no ``--where`` / ``--limit``) first clears ``RNA_interface_class`` for **every** row in the table, then writes fresh values for each row returned by the query (so stale classifications are removed). Per-row failures still update that row with an ``_error`` JSON object instead of skipping the row.

OpenStructure (CAPRI-like) bins — DockQ:
  Incorrect < 0.23 | Acceptable [0.23, 0.49) | Medium [0.49, 0.8) | High ≥ 0.8

OpenStructure (CAPRI-like) bins — QS-score:
  Incorrect < 0.1 | Acceptable [0.1, 0.3) | Medium [0.3, 0.7) | High ≥ 0.7

Categories (DockQ-based, complying with the a/b/c scheme in evaluationMetrics.py):
  - correct_interface:   RBP_DockQ >= 0.8          (a, very good)
  - partial_interface:   0.23 <= RBP_DockQ < 0.8   (b, medium)
  - incorrect_interface: RBP_DockQ < 0.23          (c, bad)
  - unknown: the RNA chain has no scorable protein–RNA interface (all RBP columns empty, the chain
    is absent from every ``X:Y`` key, or its DockQ is non-finite).
Per RNA chain, RBP_DockQ is averaged over the protein–RNA interfaces involving that chain. fnat, iRMSD,
iLDDT, TM and QS are not used for the label (QS is not used at all); the same three category names are
kept so the ordinal (incorrect=1, partial=2, correct=3) read by evaluationMetrics.py stays consistent.

Usage:
  python evaluation/classify_rna_interface_quality.py
  python evaluation/classify_rna_interface_quality.py --limit 20 --dry-run
  python evaluation/classify_rna_interface_quality.py --skip-db   # CSV only, no DB writes
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.startConfig import StartConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Written next to ``af/`` and ``pdb/`` (StartConfig parent folder).
DEFAULT_CSV_NAME = "rna_interface_classification.csv"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_inline_comment(s: str) -> str:
    if "//" in s:
        s = s.split("//", 1)[0]
    return s.strip()


def parse_chain_ids_field(raw: Any) -> List[str]:
    """Parse ProteinChainIDs / RNAChainIDs from DB (JSON list, python literal, or single id)."""
    if raw is None:
        return []
    s = _strip_inline_comment(str(raw).strip())
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            data = s
    if isinstance(data, (list, tuple)):
        return [str(x).strip() for x in data if x is not None and str(x).strip()]
    return [str(data).strip()]


def per_rna_chain_from_interfaces(
    merged: Dict[str, float], rna_chain_ids: List[str]
) -> Dict[str, float]:
    """
    Reduce an interface dict keyed by ``"protein:RNA"`` pairs (RNA_iLDDT_AllChains /
    RNA_ICS_AllChains) to a per predicted RNA-chain best score: for each RNA chain,
    max over all interface entries that involve it.
    """
    out: Dict[str, float] = {}
    for r in rna_chain_ids:
        sub = filter_rbp_pairs_involving_chain(merged, r)
        if sub:
            out[r] = max(sub.values())
    return out


def dict_max(d: Dict[str, float]) -> float:
    if not d:
        return float("nan")
    return max(d.values())


def nanmax2(a: float, b: float) -> float:
    """Max of two floats; if one is NaN, return the other."""
    if math.isnan(a) and math.isnan(b):
        return float("nan")
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def parse_rbp_list(raw: Any) -> Dict[str, float]:
    """Merge [{"A:B": x}, {"C:D": y}] into {"A:B": x, "C:D": y}."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return {}
    else:
        data = raw
    if not isinstance(data, list):
        return {}
    merged: Dict[str, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            try:
                merged[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return merged


def _split_pair(key: str) -> Optional[Tuple[str, str]]:
    if ":" not in key:
        return None
    a, b = key.split(":", 1)
    return a.strip(), b.strip()


def filter_rbp_pairs_involving_chain(merged: Dict[str, float], chain_id: str) -> Dict[str, float]:
    """All RBP interface entries where ``chain_id`` is one of the two chains (protein–RNA or RNA–RNA)."""
    out: Dict[str, float] = {}
    for k, v in merged.items():
        sp = _split_pair(k)
        if sp is None:
            continue
        c1, c2 = sp
        if c1 == chain_id or c2 == chain_id:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# OpenStructure CAPRI-like bins (compare-structures DockQ / QS-score)
# ---------------------------------------------------------------------------


def dockq_capri_bin(dockq: float) -> str:
    """
    4-bin scheme (OpenStructure): Incorrect, Acceptable, Medium, High.
    Returns 'missing' if NaN.
    """
    if math.isnan(dockq):
        return "missing"
    if dockq < 0.23:
        return "Incorrect"
    if dockq < 0.49:
        return "Acceptable"
    if dockq < 0.8:
        return "Medium"
    return "High"


def qs_capri_bin(qs: float) -> str:
    """
    4-bin scheme (OpenStructure): Incorrect, Acceptable, Medium, High.
    Returns 'missing' if NaN.
    """
    if math.isnan(qs):
        return "missing"
    if qs < 0.1:
        return "Incorrect"
    if qs < 0.3:
        return "Acceptable"
    if qs < 0.7:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    fnat_correct: float = 0.5
    irmsd_correct: float = 4.0
    ilddt_high: float = 0.7
    fnat_near_zero: float = 0.05
    irmsd_shifted_max: float = 6.0
    ilddt_shifted_min: float = 0.6
    tm_high: float = 0.75
    irmsd_far: float = 12.0
    fnat_partial_low: float = 0.1
    fnat_partial_high: float = 0.5
    # correct_position_wrong_alignment when RNA_TM is unavailable: use RBP iRMSD + weak Fnat + iLDDT band
    cpwa_irmsd_min: float = 4.0
    cpwa_irmsd_max: float = 10.0
    cpwa_ilddt_floor: float = 0.05


def classify_row(
    fnat_max_rp: float,
    irmsd_min_rp: float,
    irmsd_max_rp: float,
    ilddt_max: float,
    tm_max: float,
    qs_max: float,
    dockq_max: float,
    th: Thresholds,
) -> str:
    """
    DockQ-based interface grade, complying with the a/b/c scheme in evaluationMetrics.py
    (``get_sample_audit_global_fold_interface_lists``): the interface quality of an RNA chain is
    read straight off its RBP_DockQ (averaged over the protein partners upstream):

        (a) RBP_DockQ >= 0.8            -> ``correct_interface``   (very good)
        (b) 0.23 <= RBP_DockQ < 0.8     -> ``partial_interface``   (medium)
        (c) RBP_DockQ < 0.23            -> ``incorrect_interface`` (bad)
        no finite DockQ                 -> ``missing``

    fnat/iRMSD/iLDDT/TM/QS are ignored here (they only refine the *bad* bucket into failure modes
    in the per-complex evaluationMetrics analysis; QS is not used at all). The three category names
    are kept so the RNA_interface_class ordinal (incorrect=1, partial=2, correct=3) consumed by
    evaluationMetrics.py continues to match the DockQ tier.
    """
    _ = (fnat_max_rp, irmsd_min_rp, irmsd_max_rp, ilddt_max, tm_max, qs_max, th)  # unused
    if math.isnan(dockq_max):
        return "unknown"
    if dockq_max >= 0.8:
        return "correct_interface"
    if dockq_max >= 0.23:
        return "partial_interface"
    return "incorrect_interface"


def _finite_float(x: float) -> bool:
    return not math.isnan(x) and not math.isinf(x)


def rbp_pair_stats(sub: Dict[str, float]) -> Dict[str, Any]:
    """Mean / min / max / count for one RBP metric dict (e.g. all pairs involving chain C)."""
    if not sub:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
            "keys": [],
        }
    vals = list(sub.values())
    ks = list(sub.keys())
    return {
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "count": len(vals),
        "keys": ks,
    }


def per_chain_has_processable_data(
    chain_id: str,
    fn_sub: Dict[str, float],
    ir_sub: Dict[str, float],
    q_sub: Dict[str, float],
    d_sub: Dict[str, float],
    per_tm: Dict[str, float],
    per_lddt: Dict[str, float],
    per_ilddt: Dict[str, float],
    per_qs: Dict[str, float],
    per_dq: Dict[str, float],
    per_ics: Dict[str, float],
) -> bool:
    """
    True if this RNA chain has any RBP interface entry and/or any finite AllChains matrix-derived value.
    If False, the chain was not meaningfully processed — classify as ``missing``, not ``uncertain``.
    """
    if fn_sub or ir_sub or q_sub or d_sub:
        return True
    for d in (per_tm, per_lddt, per_ilddt, per_qs, per_dq, per_ics):
        v = d.get(chain_id, float("nan"))
        if _finite_float(v):
            return True
    return False


def chain_appears_in_any_rbp_pair(
    fn_m: Dict[str, float],
    ir_m: Dict[str, float],
    qs_m: Dict[str, float],
    dq_m: Dict[str, float],
    chain_id: str,
) -> bool:
    """
    True if ``chain_id`` appears in at least one merged RBP key.
    Empty metrics (DB ``[]`` → merged ``{}``) are skipped — a chain cannot be “in” an empty list.
    """
    for merged in (fn_m, ir_m, qs_m, dq_m):
        if not merged:
            continue
        if filter_rbp_pairs_involving_chain(merged, chain_id):
            return True
    return False


def classify_per_rna_chains(
    rna_ids: List[str],
    m: Dict[str, Any],
    th: Thresholds,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Per predicted RNA chain: matrix-derived scores for that chain + RBP pairs involving
    that chain with proteins or other RNAs. When a chain appears in several interfaces
    (e.g. ``A:C`` and ``B:C``), Fnat, iRMSD, QS, and DockQ are **averaged** across those
    pairs before combining with matrix scores and running classification rules.
    """
    fn_m = m["RBP_fnat_all"]
    ir_m = m["RBP_irmsd_all"]
    qs_bp = m["RBP_QS_all"]
    dq_bp = m["RBP_DockQ_all"]
    per_tm = m["RNA_TM_max_per_chain"]
    per_lddt = m["RNA_LDDT_max_per_chain"]
    per_ilddt = m["RNA_iLDDT_max_per_chain"]
    per_qs = m["RNA_QS_max_per_chain"]
    per_dq = m["RNA_DockQ_max_per_chain"]
    per_ics = m["RNA_ICS_max_per_chain"]

    rbp_present = bool(fn_m or ir_m or qs_bp or dq_bp)

    classes: Dict[str, str] = {}
    detail: Dict[str, Dict[str, Any]] = {}

    # All RBP columns empty ([] / null) — no interface dictionary at all; every RNA chain is unknown.
    if not rbp_present:
        for r in rna_ids:
            fn_sub = filter_rbp_pairs_involving_chain(fn_m, r)
            ir_sub = filter_rbp_pairs_involving_chain(ir_m, r)
            q_sub = filter_rbp_pairs_involving_chain(qs_bp, r)
            d_sub = filter_rbp_pairs_involving_chain(dq_bp, r)
            fn_st = rbp_pair_stats(fn_sub)
            ir_st = rbp_pair_stats(ir_sub)
            qs_st = rbp_pair_stats(q_sub)
            dq_st = rbp_pair_stats(d_sub)
            ild = per_ilddt.get(r, float("nan"))
            tm = per_tm.get(r, float("nan"))
            qs_m_val = per_qs.get(r, float("nan"))
            dq_m_val = per_dq.get(r, float("nan"))
            qs_comb = nanmax2(qs_m_val, qs_st["mean"])
            dq_comb = nanmax2(dq_m_val, dq_st["mean"])
            detail[r] = {
                "class": "unknown",
                "has_processable_data": False,
                "in_rbp_interface_dictionary": False,
                "missing_reason": "rbp_columns_empty",
                "rbp_fnat_empty": not bool(fn_m),
                "rbp_irmsd_empty": not bool(ir_m),
                "rbp_qs_empty": not bool(qs_bp),
                "rbp_dockq_empty": not bool(dq_bp),
                "RBP_fnat_by_pair": fn_st,
                "RBP_irmsd_by_pair": ir_st,
                "RBP_QS_by_pair": qs_st,
                "RBP_DockQ_by_pair": dq_st,
                "RNA_iLDDT_max": ild,
                "RNA_TM_max": tm,
                "QS_combined_max": qs_comb,
                "DockQ_combined_max": dq_comb,
                "DockQ_CAPRI_bin": dockq_capri_bin(dq_comb),
                "QS_CAPRI_bin": qs_capri_bin(qs_comb),
            }
            classes[r] = "unknown"
        return classes, detail

    for r in rna_ids:
        fn_sub = filter_rbp_pairs_involving_chain(fn_m, r)
        ir_sub = filter_rbp_pairs_involving_chain(ir_m, r)
        q_sub = filter_rbp_pairs_involving_chain(qs_bp, r)
        d_sub = filter_rbp_pairs_involving_chain(dq_bp, r)

        fn_st = rbp_pair_stats(fn_sub)
        ir_st = rbp_pair_stats(ir_sub)
        qs_st = rbp_pair_stats(q_sub)
        dq_st = rbp_pair_stats(d_sub)
        fnat_mean = fn_st["mean"]
        irmsd_mean = ir_st["mean"]
        qs_rbp_mean = qs_st["mean"]
        dq_rbp_mean = dq_st["mean"]

        ild = per_ilddt.get(r, float("nan"))
        tm = per_tm.get(r, float("nan"))
        qs_m_val = per_qs.get(r, float("nan"))
        dq_m_val = per_dq.get(r, float("nan"))
        qs_comb = nanmax2(qs_m_val, qs_rbp_mean)
        dq_comb = nanmax2(dq_m_val, dq_rbp_mean)

        in_rbp_dict = chain_appears_in_any_rbp_pair(fn_m, ir_m, qs_bp, dq_bp, r)

        # If OST returned any RBP interfaces for this model but this chain is not in any X:Y key, it is unknown.
        if rbp_present and not in_rbp_dict:
            final = "unknown"
            has_data = False
            missing_reason = "not_in_rbp_interface_dictionary"
        else:
            missing_reason = None
            has_data = per_chain_has_processable_data(
                r,
                fn_sub,
                ir_sub,
                q_sub,
                d_sub,
                per_tm,
                per_lddt,
                per_ilddt,
                per_qs,
                per_dq,
                per_ics,
            )

            if not has_data:
                final = "unknown"
            else:
                # Multiple interfaces (e.g. A:C and B:C) → use mean across pairs for Fnat, iRMSD, QS, DockQ.
                raw = classify_row(
                    fnat_mean,
                    irmsd_mean,
                    irmsd_mean,
                    ild,
                    tm,
                    qs_comb,
                    dq_comb,
                    th,
                )
                final = raw

        classes[r] = final
        row_detail: Dict[str, Any] = {
            "class": final,
            "has_processable_data": has_data,
            "in_rbp_interface_dictionary": in_rbp_dict,
            "RBP_fnat_by_pair": fn_st,
            "RBP_irmsd_by_pair": ir_st,
            "RBP_QS_by_pair": qs_st,
            "RBP_DockQ_by_pair": dq_st,
            "RNA_iLDDT_max": ild,
            "RNA_TM_max": tm,
            "QS_combined_max": qs_comb,
            "DockQ_combined_max": dq_comb,
            "DockQ_CAPRI_bin": dockq_capri_bin(dq_comb),
            "QS_CAPRI_bin": qs_capri_bin(qs_comb),
        }
        if missing_reason:
            row_detail["missing_reason"] = missing_reason
        row_detail["rbp_fnat_empty"] = not bool(fn_m)
        row_detail["rbp_irmsd_empty"] = not bool(ir_m)
        row_detail["rbp_qs_empty"] = not bool(qs_bp)
        row_detail["rbp_dockq_empty"] = not bool(dq_bp)
        detail[r] = row_detail

    return classes, detail


def sanitize_nan_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan_for_json(x) for x in obj]
    return obj


def build_row_metrics(
    _protein_ids: List[str],
    rna_ids: List[str],
    rbp_fnat_raw: Any,
    rbp_irmsd_raw: Any,
    rbp_qs_raw: Any,
    rbp_dockq_raw: Any,
    tm_raw: Any,
    lddt_raw: Any,
    ilddt_raw: Any,
    ics_raw: Any,
) -> Dict[str, Any]:
    # RBP_* and the RNA_*_AllChains columns are all JSON keyed dict-lists now:
    #   per-chain  ->  [{"B": 0.13}, {"C": 0.15}]      (RNA_TM/LDDT_AllChains)
    #   interface  ->  [{"A:B": 0.02}, {"A:C": 0.05}]  (RNA_iLDDT/ICS_AllChains, RBP_*)
    fn_m = parse_rbp_list(rbp_fnat_raw)
    ir_m = parse_rbp_list(rbp_irmsd_raw)
    qs_rbp_m = parse_rbp_list(rbp_qs_raw)
    dq_rbp_m = parse_rbp_list(rbp_dockq_raw)

    per_tm = parse_rbp_list(tm_raw)      # already keyed by predicted RNA chain
    per_lddt = parse_rbp_list(lddt_raw)  # already keyed by predicted RNA chain
    per_ilddt = per_rna_chain_from_interfaces(parse_rbp_list(ilddt_raw), rna_ids)
    per_ics = per_rna_chain_from_interfaces(parse_rbp_list(ics_raw), rna_ids)
    # QS/DockQ are only available as protein-RNA interface dicts (RBP_QS / RBP_DockQ);
    # there are no per-RNA-chain AllChains columns for them.
    per_qs: Dict[str, float] = {}
    per_dq: Dict[str, float] = {}

    return {
        "RBP_fnat_all": fn_m,
        "RBP_irmsd_all": ir_m,
        "RBP_QS_all": qs_rbp_m,
        "RBP_DockQ_all": dq_rbp_m,
        "RNA_TM_max_per_chain": per_tm,
        "RNA_LDDT_max_per_chain": per_lddt,
        "RNA_iLDDT_max_per_chain": per_ilddt,
        "RNA_QS_max_per_chain": per_qs,
        "RNA_DockQ_max_per_chain": per_dq,
        "RNA_ICS_max_per_chain": per_ics,
        "RNA_TM_max": dict_max(per_tm),
        "RNA_LDDT_max": dict_max(per_lddt),
        "RNA_iLDDT_max": dict_max(per_ilddt),
        "RNA_QS_max": dict_max(per_qs),
        "RNA_DockQ_max": dict_max(per_dq),
        "RNA_ICS_max": dict_max(per_ics),
    }


SELECT_COLS = """
Id, exp_db_id, FileName,
ProteinChainIDs, RNAChainIDs,
"RNA_TM_AllChains", "RNA_LDDT_AllChains", "RNA_iLDDT_AllChains", "RNA_ICS_AllChains",
"RBP_DockQ", "RBP_QS", "RBP_fnat", "RBP_irmsd"
""".replace("\n", " ")


def ensure_columns(cursor: sqlite3.Cursor, table: str) -> None:
    try:
        cursor.execute(f'ALTER TABLE {table} ADD COLUMN "RNA_interface_class" TEXT')
        logger.info('Added column %s."RNA_interface_class"', table)
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify pred_protein_rna rows by combined RBP + RNA_AllChains metrics."
    )
    parser.add_argument("--db", default=None, help="SQLite DB path (default: StartConfig)")
    parser.add_argument("--table", default="pred_protein_rna", help="Table name")
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Do not write RNA_interface_class to the database (CSV still written if parent folder is set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write DB or CSV; only log what would be updated",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows")
    parser.add_argument("--where", default=None, help='Extra SQL WHERE clause (without "WHERE")')
    # Thresholds
    parser.add_argument("--fnat-correct", type=float, default=0.5)
    parser.add_argument("--irmsd-correct", type=float, default=4.0)
    parser.add_argument("--ilddt-high", type=float, default=0.7)
    parser.add_argument("--fnat-near-zero", type=float, default=0.05)
    parser.add_argument("--irmsd-shifted-max", type=float, default=6.0)
    parser.add_argument("--ilddt-shifted-min", type=float, default=0.6)
    parser.add_argument("--tm-high", type=float, default=0.75)
    parser.add_argument("--irmsd-far", type=float, default=12.0)
    parser.add_argument(
        "--cpwa-irmsd-min",
        type=float,
        default=4.0,
        help="Min RBP iRMSD (Å) for correct_position_wrong_alignment when TM is weak/missing (default: 4)",
    )
    parser.add_argument(
        "--cpwa-irmsd-max",
        type=float,
        default=10.0,
        help="Max RBP iRMSD (Å) for that case (default: 10)",
    )
    parser.add_argument(
        "--cpwa-ilddt-floor",
        type=float,
        default=0.05,
        help="Min RNA_iLDDT (AllChains) for that case when TM is weak/missing (default: 0.05)",
    )
    args = parser.parse_args()

    cfg = StartConfig()
    db_path = args.db or cfg.get_database_path()
    if not os.path.isfile(db_path):
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    th = Thresholds(
        fnat_correct=args.fnat_correct,
        irmsd_correct=args.irmsd_correct,
        ilddt_high=args.ilddt_high,
        fnat_near_zero=args.fnat_near_zero,
        irmsd_shifted_max=args.irmsd_shifted_max,
        ilddt_shifted_min=args.ilddt_shifted_min,
        tm_high=args.tm_high,
        irmsd_far=args.irmsd_far,
        cpwa_irmsd_min=args.cpwa_irmsd_min,
        cpwa_irmsd_max=args.cpwa_irmsd_max,
        cpwa_ilddt_floor=args.cpwa_ilddt_floor,
    )

    sql = f"SELECT {SELECT_COLS} FROM {args.table}"
    if args.where:
        sql += f" WHERE ({args.where})"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(sql)
    except sqlite3.OperationalError as e:
        logger.error("Query failed: %s\nSQL: %s", e, sql)
        sys.exit(1)
    rows = cur.fetchall()

    full_table_run = args.where is None and args.limit is None

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        try:
            protein_ids = parse_chain_ids_field(row["ProteinChainIDs"])
            rna_ids = parse_chain_ids_field(row["RNAChainIDs"])
            m = build_row_metrics(
                protein_ids,
                rna_ids,
                row["RBP_fnat"],
                row["RBP_irmsd"],
                row["RBP_QS"],
                row["RBP_DockQ"],
                row["RNA_TM_AllChains"],
                row["RNA_LDDT_AllChains"],
                row["RNA_iLDDT_AllChains"],
                row["RNA_ICS_AllChains"],
            )
            classes, detail = classify_per_rna_chains(rna_ids, m, th)
            class_json = json.dumps(classes, sort_keys=True)
            detail_json = json.dumps(sanitize_nan_for_json(detail), sort_keys=True)
            rec = {
                "Id": row["Id"],
                "exp_db_id": row["exp_db_id"],
                "FileName": row["FileName"],
                "ProteinChainIDs": row["ProteinChainIDs"],
                "RNAChainIDs": row["RNAChainIDs"],
                "RNA_interface_class": class_json,
                "RNA_interface_detail_json": detail_json,
                "RNA_TM_max_per_chain_json": json.dumps(m["RNA_TM_max_per_chain"], sort_keys=True),
                "RNA_iLDDT_max_per_chain_json": json.dumps(m["RNA_iLDDT_max_per_chain"], sort_keys=True),
                "RNA_ICS_max_per_chain_json": json.dumps(m["RNA_ICS_max_per_chain"], sort_keys=True),
                "RNA_TM_max_all": m["RNA_TM_max"],
                "RNA_LDDT_max_all": m["RNA_LDDT_max"],
                "RNA_iLDDT_max_all": m["RNA_iLDDT_max"],
                "RNA_QS_max_all": m["RNA_QS_max"],
                "RNA_DockQ_max_all": m["RNA_DockQ_max"],
                "RNA_ICS_max_all": m["RNA_ICS_max"],
            }
            out_rows.append(rec)
        except Exception as e:
            logger.exception(
                "Classification failed for Id=%s FileName=%s — storing error in RNA_interface_class",
                row["Id"],
                row["FileName"],
            )
            err_payload = {"_classification_error": str(e)}
            out_rows.append(
                {
                    "Id": row["Id"],
                    "exp_db_id": row["exp_db_id"],
                    "FileName": row["FileName"],
                    "ProteinChainIDs": row["ProteinChainIDs"],
                    "RNAChainIDs": row["RNAChainIDs"],
                    "RNA_interface_class": json.dumps(err_payload, sort_keys=True),
                    "RNA_interface_detail_json": "{}",
                    "RNA_TM_max_per_chain_json": "{}",
                    "RNA_iLDDT_max_per_chain_json": "{}",
                    "RNA_ICS_max_per_chain_json": "{}",
                    "RNA_TM_max_all": "",
                    "RNA_LDDT_max_all": "",
                    "RNA_iLDDT_max_all": "",
                    "RNA_QS_max_all": "",
                    "RNA_DockQ_max_all": "",
                    "RNA_ICS_max_all": "",
                }
            )

    parent = cfg.get_parent_folder()
    csv_path = os.path.join(parent, DEFAULT_CSV_NAME) if parent else None
    if not args.dry_run:
        if csv_path:
            if not out_rows:
                logger.warning("No rows to write to CSV")
            else:
                fieldnames = list(out_rows[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for r in out_rows:
                        w.writerow(r)
                logger.info("Wrote %d rows to %s", len(out_rows), csv_path)
        else:
            logger.warning(
                "StartConfig parent folder is not set; skipping CSV (set config.json / run with a configured project)."
            )
    else:
        logger.info("[DRY-RUN] would write %d row(s) to CSV and DB", len(out_rows))

    if not args.skip_db and not args.dry_run:
        ensure_columns(cur, args.table)
        if full_table_run and rows:
            cur.execute(f'UPDATE {args.table} SET "RNA_interface_class" = NULL')
            logger.info(
                'Cleared "%s"."RNA_interface_class" for all rows (full-table run before refresh)',
                args.table,
            )
        for r in out_rows:
            cur.execute(
                f'UPDATE {args.table} SET "RNA_interface_class" = ? WHERE Id = ?',
                (r["RNA_interface_class"], r["Id"]),
            )
        conn.commit()
        logger.info(
            'Updated %d row(s) "%s"."RNA_interface_class" in %s',
            len(out_rows),
            args.table,
            db_path,
        )
    elif args.dry_run and not args.skip_db:
        logger.info(
            "[DRY-RUN] would set RNA_interface_class JSON on %d row(s) in %s (e.g. {\"B\": \"uncertain\", \"C\": \"uncertain\"})",
            len(out_rows),
            args.table,
        )
    elif args.skip_db:
        logger.info("Skipping DB update (--skip-db)")

    # Summary counts (per RNA chain, all rows)
    from collections import Counter

    if out_rows:
        flat: List[str] = []
        for r in out_rows:
            try:
                d = json.loads(r["RNA_interface_class"])
                if isinstance(d, dict):
                    flat.extend(d.values())
            except json.JSONDecodeError:
                pass
        logger.info("Per-chain classification counts: %s", dict(Counter(flat)))
    conn.close()


if __name__ == "__main__":
    main()
