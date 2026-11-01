"""
Combined RNAcompete maxprobe/minprobe probe-class analysis.

Merges exp_probeMax.py (class-stratified motif-length boxplots, average motif
lengths, violin distributions) and exp_probeMax_new.py (hit/miss boxplots,
hit/miss counts, RNAmotif score boxplots) into a single script.

Competition complexes (pred ``exp_db_id`` containing MAX1MIN*) hold one protein
chain (A) and 2–4 RNA chains. ``RNAChainIDs`` order defines probe class:
index 0 = maxprobe, indices 1+ = minprobe(s). ``ChainIDpairList_proteinRNA`` gives
the protein–RNA pair per index; ``RNAMotifLength[i] > 0`` means chain *i* binds.

Per-chain ``RNAmotif_score``, ``RNAMotif_inExp`` and ``InterfaceArea_norm`` are
computed by matching each RNA sequence to ``exp_protein_rna.csv`` (same lookup as
class boxplots). Row-level DB columns store the **maxprobe-chain** values; NaN
when the maxprobe RNA does not interact (motif length 0).

For every figure two variants are produced:
  - all matched rows
  - rows where every RNALength value is equal (suffix ``_equalRNAlength``)

All plots are saved via plots.plotCreator.PlotCreator (standalone mode) to:
  results/plots/all_chains/rna_compete_minmax/
and a statistics text file is written next to them.

Usage:
  python rnacompete/exp_probeMax_combined.py
  python rnacompete/exp_probeMax_combined.py --csv /path/exp_protein_rna.csv --outdir /path/out
  python rnacompete/exp_probeMax_combined.py --no-update-db
"""

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator

PLOT_TYPE = "rna_compete_minmax"
DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "rna_compete_minmax")
STATS_FILENAME = "probe_minmax_statistics.txt"

MAXPROBE_COLOR = "#1976D2"
MINPROBE_COLOR = "#D32F2F"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_base_pdb_id(pdb_id):
    """Extract RNCMPT##### base id (case-insensitive)."""
    match = re.search(r"(?i)(RNCMPT\d{5})", str(pdb_id))
    if match:
        return match.group(1).upper()
    return str(pdb_id)


def base_protein_key(pdb_id):
    """Lowercase RNCMPT##### key for joining pred chains to exp CSV rows."""
    return extract_base_pdb_id(pdb_id).lower()


def get_base_pdb_id(pdb_id):
    """Extract base PDB ID from full PDB ID by removing max/min and everything after."""
    parts = re.split(r"(?:max|min)", pdb_id, 1)
    return parts[0]


def get_class_type(pdb_id):
    """Extract class type (MAX1MIN1, MAX1MIN2, MAX1MIN3) from PDB ID."""
    match = re.search(r"MAX\d+MIN\d+", pdb_id)
    if match:
        return match.group(0)
    return "unknown"


def _parse_json_cell(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return raw


def _rna_lengths_all_equal(rna_length_raw) -> bool:
    """True when all RNALength values are equal (e.g. [38, 38, 38] or scalar 38)."""
    parsed = _parse_json_cell(rna_length_raw)
    if isinstance(parsed, (int, float)) and np.isfinite(parsed):
        return True
    if not isinstance(parsed, (list, tuple)) or not parsed:
        return False
    try:
        vals = [float(v) for v in parsed]
    except (TypeError, ValueError):
        return False
    return len(set(vals)) == 1


def _passes_new_template_filter(rbp_raw, prot_raw) -> bool:
    """RBP_PDB_Templates has exactly one entry and Protein_PDB_Templates is empty."""
    rbp = _parse_json_cell(rbp_raw)
    prot = _parse_json_cell(prot_raw)
    rbp_ok = isinstance(rbp, (list, tuple)) and len(rbp) == 1
    prot_ok = (prot is None) or (isinstance(prot, (list, tuple)) and len(prot) == 0)
    return bool(rbp_ok and prot_ok)


def _summary_stats(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_stats_block(f, label, stats):
    if not stats:
        f.write(f"  {label}: no data\n")
        return
    f.write(
        f"  {label}: n={stats['n']}, mean={stats['mean']:.4g}, std={stats['std']:.4g}, "
        f"median={stats['median']:.4g}, min={stats['min']:.4g}, max={stats['max']:.4g}\n"
    )


# --------------------------------------------------------------------------- #
# Per-chain motif / interface scoring (logic aligned with rnaCompete.py)
# --------------------------------------------------------------------------- #
def _probe_role_for_chain_index(index):
    """RNAChainIDs index 0 = maxprobe; all other indices = minprobe (see run_hits_misses)."""
    return "maxprobe" if index == 0 else "minprobe"


def _get_motif_positions(motif):
    """Predicted RNAMotif positions as (1-based position, base) tuples."""
    if not motif or "(" not in str(motif):
        return None
    positions = []
    for part in str(motif).split(","):
        part = part.strip()
        if "(" in part and ")" in part:
            nuc = part.split("(")[0].strip()
            pos = int(part.split("(")[1].split(")")[0])
            positions.append((pos, nuc))
    return sorted(positions) if positions else None


def _calculate_motif_score(exp_motif, pred_motif, rna_sequence):
    """Fuzzy overlap score in [0, 1] (same algorithm as rnaCompete.calculate_motif_score)."""
    if not exp_motif or not pred_motif or not rna_sequence:
        return np.nan
    exp_start = rna_sequence.find(exp_motif)
    if exp_start == -1:
        return np.nan
    exp_positions = [(exp_start + i + 1, base) for i, base in enumerate(exp_motif)]
    pred_positions = _get_motif_positions(pred_motif)
    if not pred_positions:
        return np.nan
    matches = 0.0
    total_positions = max(len(exp_positions), len(pred_positions))
    exp_pos_dict = {base: set() for base in "ACGU"}
    pred_pos_dict = {base: set() for base in "ACGU"}
    for pos, base in exp_positions:
        exp_pos_dict[base].add(pos)
    for pos, base in pred_positions:
        pred_pos_dict[base].add(pos)
    for base in "ACGU":
        exp_set = set(exp_pos_dict[base])
        pred_set = set(pred_pos_dict[base])
        while exp_set and pred_set:
            exp_pos = exp_set.pop()
            closest_pred = min(pred_set, key=lambda x: abs(x - exp_pos))
            distance = abs(closest_pred - exp_pos)
            if distance == 0:
                matches += 1.0
            elif distance <= 3:
                matches += 0.5
            else:
                matches += 0.2
            pred_set.remove(closest_pred)
    score = matches / total_positions if total_positions > 0 else 0.0
    return float(max(0.0, min(1.0, score)))


def _calculate_motif_in_exp_score(exp_positions, pred_positions):
    """Exact (position, base) overlap fraction (rnaCompete.calculate_motif_in_exp_score)."""
    if not exp_positions or not pred_positions:
        return np.nan
    exp_set = set((pos, base) for pos, base in exp_positions)
    pred_set = set((pos, base) for pos, base in pred_positions)
    matches = exp_set.intersection(pred_set)
    return round(len(matches) / len(exp_positions), 2)


def _interface_area_norm(interface_area, protein_length):
    """Weight-neutral interface score: area / sqrt(protein length)."""
    try:
        area_f = float(interface_area) if interface_area else 0.0
        plen_f = float(protein_length) if protein_length else 0.0
    except (ValueError, TypeError):
        return np.nan
    if not area_f or plen_f <= 0:
        return np.nan
    return area_f / math.sqrt(plen_f)


def _to_float_or_nan(raw):
    if raw is None or raw == "":
        return np.nan
    try:
        return float(raw)
    except (ValueError, TypeError):
        return np.nan


def _contacts_per_chain(chain_pairs, hbond_pairs, vdw_pairs):
    """Map RNA chain ID -> (H-bond + vdW) contact count via ChainIDpairList order.

    Hbond_proteinRNA / vdWbond_proteinRNA are per-interacting-chain lists aligned to
    ChainIDpairList_proteinRNA. Chains absent from the pair list contribute 0.
    """
    if not chain_pairs:
        return {}

    def _vals(raw):
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if isinstance(raw, (list, tuple)):
            return [float(v) if v is not None else 0.0 for v in raw]
        return []

    hb = _vals(hbond_pairs)
    vd = _vals(vdw_pairs)
    out = {}
    for i, pair in enumerate(chain_pairs):
        if len(pair) > 1:
            h = hb[i] if i < len(hb) else 0.0
            v = vd[i] if i < len(vd) else 0.0
            out[pair[1]] = h + v
    return out


def _motif_length_map(chain_pairs, motif_length_raw):
    """Map RNA chain ID -> motif length via ChainIDpairList_proteinRNA index order."""
    if not chain_pairs:
        return {}
    if isinstance(motif_length_raw, (int, float)):
        lengths = [float(motif_length_raw)] * len(chain_pairs)
    else:
        lengths = [float(v) for v in motif_length_raw]
    motif_map = {}
    for i, pair in enumerate(chain_pairs):
        if len(pair) > 1:
            motif_map[pair[1]] = lengths[i] if i < len(lengths) else 0.0
    return motif_map


def _parse_list_cell(raw):
    parsed = _parse_json_cell(raw)
    if parsed is None:
        return None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def _align_pred_motifs_per_chain(raw, rna_chains, chain_pairs, motif_length_raw):
    """Return predicted RNAMotif string per RNAChainIDs index.

    Handles three storage formats in pred_protein_rna:
      - JSON list, one motif per chain: ``["motif_B", "motif_C", ...]``
      - Plain-text single motif (one binding chain): ``"A(23), U(24), ..."``
      - Empty string ``""`` when only one chain binds and no motif text is stored
    """
    n = len(rna_chains) if rna_chains else 0
    out = [None] * n
    if n == 0 or not chain_pairs:
        return out

    motif_map = _motif_length_map(chain_pairs, motif_length_raw)
    hit_indices = [
        i for i, ch in enumerate(rna_chains)
        if float(motif_map.get(ch, 0.0)) > 0
    ]

    if raw is None:
        return out
    text = str(raw).strip()
    if not text:
        return out

    if text.startswith("["):
        motif_list = _parse_list_cell(raw)
        if motif_list is not None:
            for i in range(n):
                if i < len(motif_list):
                    val = motif_list[i]
                    if val is not None and str(val).strip():
                        out[i] = str(val).strip()
            return out

    # Plain-text single motif: assign to the binding chain (unique hit in this dataset).
    if "(" in text:
        if len(hit_indices) == 1:
            out[hit_indices[0]] = text
        elif len(hit_indices) > 1:
            best = max(hit_indices, key=lambda i: float(motif_map.get(rna_chains[i], 0.0)))
            out[best] = text
    return out

def _is_competition_id(pdb_id):
    return bool(re.search(r"MAX\d+MIN\d+", str(pdb_id), re.IGNORECASE))


def _parse_competition_row(pred, rna_exp_map):
    """Return per-chain interaction + score dicts for a multi-RNA competition row."""
    pdb_id = pred[0]
    rna_seqs = _parse_list_cell(pred[1])
    rna_chains = _parse_list_cell(pred[2])
    chain_pairs = _parse_list_cell(pred[3])
    motif_length_raw = _parse_json_cell(pred[4])
    pred_motifs = _align_pred_motifs_per_chain(
        pred[10] if len(pred) > 10 else None,
        rna_chains,
        chain_pairs,
        motif_length_raw,
    )
    interface_area = pred[11] if len(pred) > 11 else None
    protein_length = pred[12] if len(pred) > 12 else None
    iptm_per_chain = _parse_list_cell(pred[13]) if len(pred) > 13 else None
    hbond_pairs = _parse_list_cell(pred[14]) if len(pred) > 14 else None
    vdw_pairs = _parse_list_cell(pred[15]) if len(pred) > 15 else None
    kd_complex = _to_float_or_nan(pred[16]) if len(pred) > 16 else np.nan
    # Per-RNA-chain protein-RNA min PAE, aligned to RNAChainIDs (lower == interaction).
    pae_per_chain = _parse_list_cell(pred[17]) if len(pred) > 17 else None

    if not rna_chains or not chain_pairs or not rna_seqs:
        return None

    base_pdb = base_protein_key(pdb_id)
    class_type = get_class_type(pdb_id)
    motif_map = _motif_length_map(chain_pairs, motif_length_raw)
    contacts_map = _contacts_per_chain(chain_pairs, hbond_pairs, vdw_pairs)
    iface_norm_all = _interface_area_norm(interface_area, protein_length)

    chains_out = []
    for i, chain in enumerate(rna_chains):
        seq = str(rna_seqs[i]) if i < len(rna_seqs) else ""
        role = _probe_role_for_chain_index(i)
        rank_label = "maxprobe" if i == 0 else f"min{i}"
        exp_info = rna_exp_map.get((base_pdb, seq.strip()), {})
        exp_class = exp_info.get("class") or role
        exp_motif = exp_info.get("exp_motif") or ""
        pred_motif = pred_motifs[i] if i < len(pred_motifs) else None
        motif_len = float(motif_map.get(chain, 0.0))
        hit = motif_len > 0

        iptm = np.nan
        if iptm_per_chain is not None and i < len(iptm_per_chain):
            iptm = _to_float_or_nan(iptm_per_chain[i])

        pae = np.nan
        if pae_per_chain is not None and i < len(pae_per_chain):
            pae = _to_float_or_nan(pae_per_chain[i])

        contacts = float(contacts_map.get(chain, 0.0))

        rna_score = np.nan
        in_exp = np.nan
        if hit and exp_motif and pred_motif and seq:
            rna_score = round(_calculate_motif_score(exp_motif, pred_motif, seq), 2)
            exp_start = seq.find(exp_motif)
            if exp_start != -1:
                exp_positions = [(exp_start + j + 1, base) for j, base in enumerate(exp_motif)]
                pred_positions = _get_motif_positions(pred_motif)
                if pred_positions:
                    in_exp = _calculate_motif_in_exp_score(exp_positions, pred_positions)

        chains_out.append({
            "chain": chain,
            "index": i,
            "role": role,
            "rank_label": rank_label,
            "exp_class": exp_class,
            "hit": hit,
            "motif_length": motif_len,
            "rna_ipTM": iptm,
            # Oriented per-chain PAE: higher == maxprobe-preferred (lower raw PAE).
            "pae_oriented": -pae if np.isfinite(pae) else np.nan,
            "interface_contacts": contacts,
            # motif score for ranking: missing/non-hit -> 0 (no predicted motif)
            "RNAmotif_score": rna_score,
            "RNAmotif_score_rank": rna_score if np.isfinite(rna_score) else 0.0,
            "RNAMotif_inExp": in_exp,
            "InterfaceArea_norm": iface_norm_all if hit else np.nan,
        })

    max_chain = chains_out[0] if chains_out else {}
    return {
        "pdb_id": pdb_id,
        "class_type": class_type,
        "domain_family": None,  # filled by caller via fam_by_base
        "chains": chains_out,
        "maxprobe": max_chain,
        "n_hits": sum(1 for c in chains_out if c["hit"]),
        "maxprobe_hit": bool(max_chain.get("hit")),
        "kd_complex": kd_complex,
        "interface_area_complex": _to_float_or_nan(interface_area),
        "InterfaceArea_norm_complex": iface_norm_all,
    }


def update_competition_db_scores(db_path, pred_rows, rna_exp_map):
    """Write maxprobe-chain RNAmotif_score, RNAMotif_inExp, InterfaceArea_norm to DB."""
    updates = []
    for pred in pred_rows:
        if not _is_competition_id(pred[0]):
            continue
        parsed = _parse_competition_row(pred, rna_exp_map)
        if parsed is None:
            continue
        mx = parsed["maxprobe"]
        updates.append((
            mx["RNAmotif_score"] if np.isfinite(mx["RNAmotif_score"]) else None,
            mx["RNAMotif_inExp"] if np.isfinite(mx["RNAMotif_inExp"]) else None,
            mx["InterfaceArea_norm"] if np.isfinite(mx["InterfaceArea_norm"]) else None,
            parsed["pdb_id"],
        ))

    if not updates:
        print("No competition rows to update.")
        return 0

    conn = sqlite3.connect(db_path)
    with conn:
        conn.executemany(
            """
            UPDATE pred_protein_rna
            SET RNAmotif_score = ?,
                RNAMotif_inExp = ?,
                InterfaceArea_norm = ?
            WHERE exp_db_id = UPPER(?)
            """,
            updates,
        )
    conn.close()
    print(f"Updated {len(updates)} competition rows with maxprobe-chain scores.")
    return len(updates)


def collect_competition_summary(pred_rows, rna_exp_map):
    """Aggregate interaction counts, preferential binding, and maxprobe-chain scores."""
    rows = [p for p in pred_rows if _is_competition_id(p[0])]
    summary = {
        "n_input": len(rows),
        "n_parsed": 0,
        "parse_failed": 0,
        "hit_counts": defaultdict(int),
        "single_hit_role": defaultdict(int),
        "preferential": {"max_only": 0, "min_only": 0, "both": 0, "neither": 0},
        "per_class": defaultdict(
            lambda: {"max_hit": 0, "max_miss": 0, "min_hit": 0, "min_miss": 0, "n": 0}
        ),
        "maxprobe_outcomes": {"miss": 0, "hit_scored": 0, "hit_no_motif": 0},
        "maxprobe_scores": {
            "RNAmotif_score": [],
            "RNAMotif_inExp": [],
            "InterfaceArea_norm": [],
        },
        "maxprobe_hit_scores": {
            "RNAmotif_score": [],
            "RNAMotif_inExp": [],
            "InterfaceArea_norm": [],
        },
    }
    if not rows:
        return summary

    for pred in rows:
        parsed = _parse_competition_row(pred, rna_exp_map)
        if parsed is None:
            summary["parse_failed"] += 1
            continue
        summary["n_parsed"] += 1
        ct = parsed["class_type"]
        summary["per_class"][ct]["n"] += 1
        summary["hit_counts"][parsed["n_hits"]] += 1

        max_hit = parsed["maxprobe_hit"]
        min_hits = [c["hit"] for c in parsed["chains"] if c["role"] == "minprobe"]
        any_min_hit = any(min_hits)

        if max_hit and not any_min_hit:
            summary["preferential"]["max_only"] += 1
        elif any_min_hit and not max_hit:
            summary["preferential"]["min_only"] += 1
        elif max_hit and any_min_hit:
            summary["preferential"]["both"] += 1
        else:
            summary["preferential"]["neither"] += 1

        if parsed["n_hits"] == 1:
            sole = next(c for c in parsed["chains"] if c["hit"])
            summary["single_hit_role"][sole["role"]] += 1

        summary["per_class"][ct]["max_hit" if max_hit else "max_miss"] += 1
        for c in parsed["chains"]:
            if c["role"] != "minprobe":
                continue
            if c["hit"]:
                summary["per_class"][ct]["min_hit"] += 1
            else:
                summary["per_class"][ct]["min_miss"] += 1

        mx = parsed["maxprobe"]
        if not mx["hit"]:
            summary["maxprobe_outcomes"]["miss"] += 1
        elif np.isfinite(mx["RNAmotif_score"]):
            summary["maxprobe_outcomes"]["hit_scored"] += 1
        else:
            summary["maxprobe_outcomes"]["hit_no_motif"] += 1

        for key in summary["maxprobe_scores"]:
            val = mx[key]
            if np.isfinite(val):
                summary["maxprobe_scores"][key].append(val)
                if mx["hit"]:
                    summary["maxprobe_hit_scores"][key].append(val)

    return summary


def _write_interaction_stats(summary, stats_file, filename_suffix=""):
    if summary["n_input"] == 0:
        stats_file.write("No competition complexes for interaction analysis.\n\n")
        return

    hit_counts = summary["hit_counts"]
    single_hit_role = summary["single_hit_role"]
    preferential = summary["preferential"]
    per_class = summary["per_class"]
    outcomes = summary["maxprobe_outcomes"]

    variant = "equal RNALength rows only" if filename_suffix else "all competition rows"
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Competition complex interactions ({variant})\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(
        "Probe class by RNAChainIDs index: index 0 = maxprobe, indices 1+ = minprobe. "
        "Hit = RNAMotifLength > 0 for that chain via ChainIDpairList_proteinRNA.\n\n"
    )
    stats_file.write(
        f"Complexes analysed: {summary['n_parsed']} "
        f"(parse failed: {summary['parse_failed']})\n"
    )
    stats_file.write("Total interacting chains per complex:\n")
    for k in sorted(hit_counts):
        stats_file.write(f"  {k} chain(s) bind: {hit_counts[k]}\n")
    stats_file.write(
        "\nWhen exactly one chain binds, which probe class:\n"
        f"  maxprobe: {single_hit_role['maxprobe']}\n"
        f"  minprobe: {single_hit_role['minprobe']}\n"
    )
    stats_file.write(
        "\nPreferential binding pattern (maxprobe chain vs any minprobe chain):\n"
        f"  maxprobe only: {preferential['max_only']}\n"
        f"  minprobe only: {preferential['min_only']}\n"
        f"  both max and min: {preferential['both']}\n"
        f"  neither: {preferential['neither']}\n\n"
    )
    stats_file.write("Hit / miss counts by competition class:\n")
    for ct in sorted(per_class):
        d = per_class[ct]
        rate = 100.0 * d["max_hit"] / d["n"] if d["n"] else 0.0
        stats_file.write(
            f"  {ct} (n={d['n']}): maxprobe hit={d['max_hit']}, miss={d['max_miss']} "
            f"({rate:.0f}% hit); minprobe chain hits={d['min_hit']}, misses={d['min_miss']}\n"
        )
    stats_file.write(
        "\nMaxprobe-chain score availability:\n"
        f"  no interaction (miss): {outcomes['miss']}\n"
        f"  hit with motif scores: {outcomes['hit_scored']}\n"
        f"  hit without motif scores: {outcomes['hit_no_motif']}\n\n"
    )


def run_interaction_analysis(pred_rows, rna_exp_map, stats_file, filename_suffix=""):
    """Summarise how many RNA chains bind per complex and whether maxprobe is preferred."""
    summary = collect_competition_summary(pred_rows, rna_exp_map)
    _write_interaction_stats(summary, stats_file, filename_suffix)
    return summary


def run_maxprobe_chain_score_stats(pred_rows, rna_exp_map, stats_file, filename_suffix=""):
    """Stats for maxprobe-chain scores (NaN when maxprobe does not bind)."""
    summary = collect_competition_summary(pred_rows, rna_exp_map)
    if summary["n_parsed"] == 0:
        stats_file.write("No competition complexes for maxprobe score stats.\n\n")
        return summary

    scores = summary["maxprobe_scores"]
    hit_scores = summary["maxprobe_hit_scores"]
    outcomes = summary["maxprobe_outcomes"]
    n_max_hit = outcomes["hit_scored"] + outcomes["hit_no_motif"]
    n_max_miss = outcomes["miss"]

    variant = "equal RNALength rows only" if filename_suffix else "all competition rows"
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Maxprobe-chain scores ({variant})\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(
        "Scores are computed per RNA chain using exp_protein_rna.csv (sequence match). "
        "Row-level values apply to the maxprobe chain (index 0); NaN when it does not bind.\n\n"
    )
    stats_file.write(
        f"Complexes: {summary['n_parsed']}; maxprobe binds: {n_max_hit}; "
        f"maxprobe miss: {n_max_miss}\n\n"
    )
    for key, label in (
        ("RNAmotif_score", "RNAmotif_score (maxprobe chain, finite values)"),
        ("RNAMotif_inExp", "RNAMotif_inExp (maxprobe chain, finite values)"),
        ("InterfaceArea_norm", "InterfaceArea_norm (maxprobe chain, finite values)"),
    ):
        _write_stats_block(stats_file, label, _summary_stats(scores[key]))
        _write_stats_block(
            stats_file,
            f"{label} — maxprobe hits only",
            _summary_stats(hit_scores[key]),
        )
    stats_file.write("\n")
    return summary


def plot_competition_findings(plotter, summary, filename_suffix=""):
    """Plot interaction counts, preferential binding, hit rates, and score availability."""
    if summary["n_parsed"] == 0:
        print(f"No competition data to plot (suffix='{filename_suffix}').")
        return

    plotter.set_plot_style()
    suffix = filename_suffix or ""

    # --- 2x2 overview panel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    hit_counts = summary["hit_counts"]
    chain_labels = [str(k) for k in sorted(hit_counts)]
    chain_vals = [hit_counts[int(k)] for k in chain_labels]
    axes[0, 0].bar(chain_labels, chain_vals, color="#546E7A", alpha=0.85)
    axes[0, 0].set_title("Interacting chains per complex")
    axes[0, 0].set_xlabel("Number of RNA chains with RNAMotifLength > 0")
    axes[0, 0].set_ylabel("Complex count")
    for i, v in enumerate(chain_vals):
        axes[0, 0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    single = summary["single_hit_role"]
    axes[0, 1].bar(
        ["maxprobe", "minprobe"],
        [single["maxprobe"], single["minprobe"]],
        color=[MAXPROBE_COLOR, MINPROBE_COLOR],
        alpha=0.85,
    )
    axes[0, 1].set_title("When exactly one chain binds")
    axes[0, 1].set_ylabel("Complex count")
    for i, v in enumerate([single["maxprobe"], single["minprobe"]]):
        axes[0, 1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

    pref = summary["preferential"]
    pref_labels = ["max only", "min only", "both", "neither"]
    pref_vals = [pref["max_only"], pref["min_only"], pref["both"], pref["neither"]]
    pref_colors = [MAXPROBE_COLOR, MINPROBE_COLOR, "#7B1FA2", "#9E9E9E"]
    axes[1, 0].bar(pref_labels, pref_vals, color=pref_colors, alpha=0.85)
    axes[1, 0].set_title("Preferential binding (predicted)")
    axes[1, 0].set_ylabel("Complex count")
    axes[1, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(pref_vals):
        axes[1, 0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    per_class = summary["per_class"]
    class_types = sorted(per_class)
    hit_rates = [
        100.0 * per_class[ct]["max_hit"] / per_class[ct]["n"] if per_class[ct]["n"] else 0.0
        for ct in class_types
    ]
    bars = axes[1, 1].bar(class_types, hit_rates, color=MAXPROBE_COLOR, alpha=0.85)
    axes[1, 1].set_title("Maxprobe hit rate by competition class")
    axes[1, 1].set_ylabel("Maxprobe hit rate (%)")
    axes[1, 1].set_ylim(0, 105)
    for bar, ct in zip(bars, class_types):
        d = per_class[ct]
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{d['max_hit']}/{d['n']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Competition complex binding summary (RNAMotifLength > 0 = predicted interaction)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plotter.save_plot(f"competition_interaction_overview{suffix}")
    plt.close(fig)

    # --- Maxprobe score availability ---
    outcomes = summary["maxprobe_outcomes"]
    fig = plt.figure(figsize=(8, 5))
    avail_labels = ["maxprobe miss\n(no interaction)", "hit + motif scores", "hit, no motif score"]
    avail_vals = [outcomes["miss"], outcomes["hit_scored"], outcomes["hit_no_motif"]]
    avail_colors = ["#9E9E9E", "#2E7D32", "#F9A825"]
    plt.bar(avail_labels, avail_vals, color=avail_colors, alpha=0.85)
    plt.title("Maxprobe-chain score availability")
    plt.ylabel("Complex count")
    for i, v in enumerate(avail_vals):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plotter.save_plot(f"competition_maxprobe_score_availability{suffix}")
    plt.close(fig)

    # --- Score boxplots (maxprobe hits only) ---
    hit_scores = summary["maxprobe_hit_scores"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    score_specs = (
        ("RNAmotif_score", "RNA motif score"),
        ("RNAMotif_inExp", "RNA motif in exp"),
        ("InterfaceArea_norm", "Normalized interface area"),
    )
    for ax, (key, label) in zip(axes, score_specs):
        data = hit_scores[key]
        if data:
            ax.boxplot([data], labels=["maxprobe hit"])
            ax.scatter([1] * len(data), data, alpha=0.25, color=MAXPROBE_COLOR, s=12)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Maxprobe-chain scores (complexes where maxprobe binds)", y=1.02)
    plt.tight_layout()
    plotter.save_plot(f"competition_maxprobe_scores_boxplots{suffix}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Extended competition analyses (within-complex ranking of competing probes)
# --------------------------------------------------------------------------- #
# Per-chain metrics that genuinely vary across competing RNA chains within one
# AF3 complex. Higher value == maxprobe-preferred for all three. Kd and total
# interface area are complex-level scalars and therefore NOT used for within-
# complex ranking (interface strength is proxied by per-chain H-bond+vdW counts).
COMP_RANK_METRICS = (
    ("rna_ipTM", "RNA ipTM", MAXPROBE_COLOR),
    ("RNAmotif_score_rank", "RNA motif score", "#2E7D32"),
    ("interface_contacts", "Interface contacts (H-bond+vdW)", "#F9A825"),
)
# Publication figure (pub_competition_outcomes_family) also shows per-chain PAE, which
# needs the af3_chain_pair_pae_min_perchain column (the old scalar af3_chain_pair_pae_min
# is one value per complex and cannot rank competing chains). PAE is oriented (-PAE, key
# "pae_oriented") so that higher == maxprobe-preferred, matching the other three metrics.
PUB_RANK_METRICS = COMP_RANK_METRICS + (
    ("pae_oriented", "PAE", "#6A1B9A"),
)
COMP_CLASS_ORDER = ("MAX1MIN1", "MAX1MIN2", "MAX1MIN3")
OUTCOME_ORDER = ("max_only", "min_only", "both", "neither")
OUTCOME_LABELS = {
    "max_only": "maxprobe only",
    "min_only": "minprobe only",
    "both": "both",
    "neither": "neither",
}
OUTCOME_COLORS = {
    "max_only": MAXPROBE_COLOR,
    "min_only": MINPROBE_COLOR,
    "both": "#7B1FA2",
    "neither": "#9E9E9E",
}

# Nature-style publication defaults: sans-serif, thin spines, no title (captions
# carry the description), editable text in vector exports.
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _pub_ax(ax, grid_axis=None):
    """Apply Nature-style axis cosmetics: despine top/right, outward ticks, light grid."""
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


def load_domain_family_map(db_path):
    """Map base RNCMPT id -> RBD family (RRM/KH/PUM-HD/...) from exp_protein_rna domains."""
    try:
        import domain_analysis as da  # sibling module
    except ImportError:
        sys.path.insert(0, _SCRIPT_DIR)
        import domain_analysis as da
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT PDBId, domain_ids, domain_names FROM exp_protein_rna WHERE domain_names IS NOT NULL"
    )
    fam = {}
    for pid, dids, dnames in cur.fetchall():
        base = extract_base_pdb_id(pid)
        if base in fam:
            continue
        domain_map = da.domain_map_from(dids, dnames)
        if domain_map:
            fam[base] = da.protein_domain_family(domain_map)
    conn.close()
    return fam


def build_competition_records(pred_rows, rna_exp_map, fam_by_base=None):
    """Per-complex records enriched with per-chain metrics + domain family."""
    fam_by_base = fam_by_base or {}
    records = []
    for pred in pred_rows:
        if not _is_competition_id(pred[0]):
            continue
        parsed = _parse_competition_row(pred, rna_exp_map)
        if parsed is None:
            continue
        parsed["domain_family"] = fam_by_base.get(base_protein_key(pred[0]).upper())
        records.append(parsed)
    return records


def _metric_values(record, metric_key):
    """Ordered per-chain values (index 0 = maxprobe) for a metric."""
    return [c.get(metric_key, np.nan) for c in record["chains"]]


def _rank_first_credit(values, max_index=0):
    """Fractional credit that the maxprobe chain ranks first (ties split evenly)."""
    finite = [(i, float(v)) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    top = max(v for _, v in finite)
    tied = [i for i, v in finite if v == top]
    if max_index in tied:
        return 1.0 / len(tied)
    return 0.0


def _max_min_margin(values, max_index=0):
    """S_max - max(S_min) for one complex (None if not computable)."""
    if max_index >= len(values):
        return None
    vmax = values[max_index]
    if vmax is None or not np.isfinite(vmax):
        return None
    mins = [float(v) for i, v in enumerate(values) if i != max_index and v is not None and np.isfinite(v)]
    if not mins:
        return None
    return float(vmax) - max(mins)


def _maxprobe_rank(values, max_index=0):
    """Average-rank position of the maxprobe chain (1 = best) for a descending metric."""
    if max_index >= len(values):
        return None
    vmax = values[max_index]
    if vmax is None or not np.isfinite(vmax):
        return None
    others = [float(v) for i, v in enumerate(values) if i != max_index and v is not None and np.isfinite(v)]
    greater = sum(1 for v in others if v > vmax)
    tied = sum(1 for v in others if v == vmax)
    return 1.0 + greater + 0.5 * tied


def _roc_auc(pos, neg):
    """Mann-Whitney ROC-AUC: P(random positive > random negative); ties = 0.5."""
    pos = [float(v) for v in pos if v is not None and np.isfinite(v)]
    neg = [float(v) for v in neg if v is not None and np.isfinite(v)]
    if not pos or not neg:
        return None, 0, 0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg)), len(pos), len(neg)


# ---- Analysis 1: maxprobe win rate ---------------------------------------- #
def competition_win_rates(records, metrics=COMP_RANK_METRICS):
    out = {}
    for key, label, _ in metrics:
        credits = []
        for rec in records:
            c = _rank_first_credit(_metric_values(rec, key))
            if c is not None:
                credits.append(c)
        out[key] = {
            "label": label,
            "rate": float(np.mean(credits)) if credits else float("nan"),
            "n": len(credits),
        }
    return out


# ---- Analysis 2: max-min margins ------------------------------------------ #
def competition_margins(records):
    out = {}
    for key, label, _ in COMP_RANK_METRICS:
        margins = []
        for rec in records:
            m = _max_min_margin(_metric_values(rec, key))
            if m is not None:
                margins.append(m)
        out[key] = {"label": label, "margins": margins}
    return out


# ---- Analysis 3: P(interaction | rank) ------------------------------------ #
def competition_interaction_by_rank(records):
    hits = defaultdict(int)
    totals = defaultdict(int)
    for rec in records:
        for c in rec["chains"]:
            totals[c["rank_label"]] += 1
            if c["hit"]:
                hits[c["rank_label"]] += 1
    labels = ["maxprobe", "min1", "min2", "min3"]
    return {
        lab: {"p": hits[lab] / totals[lab] if totals[lab] else float("nan"),
              "hits": hits[lab], "n": totals[lab]}
        for lab in labels if totals[lab] > 0
    }


# ---- Analysis 4: competition ROC-AUC -------------------------------------- #
def competition_auc(records, metrics=COMP_RANK_METRICS):
    out = {}
    for key, label, _ in metrics:
        pos, neg = [], []
        for rec in records:
            for c in rec["chains"]:
                v = c.get(key, np.nan)
                if c["role"] == "maxprobe":
                    pos.append(v)
                else:
                    neg.append(v)
        auc, npos, nneg = _roc_auc(pos, neg)
        out[key] = {"label": label, "auc": auc, "n_pos": npos, "n_neg": nneg}
    return out


# ---- Analysis 5: outcome frequencies by class ----------------------------- #
def competition_outcomes_by_class(records):
    table = {ct: {o: 0 for o in OUTCOME_ORDER} for ct in COMP_CLASS_ORDER}
    for rec in records:
        ct = rec["class_type"]
        if ct not in table:
            table[ct] = {o: 0 for o in OUTCOME_ORDER}
        max_hit = rec["maxprobe_hit"]
        any_min = any(c["hit"] for c in rec["chains"] if c["role"] == "minprobe")
        if max_hit and not any_min:
            table[ct]["max_only"] += 1
        elif any_min and not max_hit:
            table[ct]["min_only"] += 1
        elif max_hit and any_min:
            table[ct]["both"] += 1
        else:
            table[ct]["neither"] += 1
    return table


# ---- Analysis 7: ranking metrics ------------------------------------------ #
def competition_ranking_metrics(records):
    """Top-1 accuracy, MRR, nDCG treating the maxprobe as the single relevant probe."""
    out = {}
    for key, label, _ in COMP_RANK_METRICS:
        top1, mrr, ndcg = [], [], []
        for rec in records:
            vals = _metric_values(rec, key)
            credit = _rank_first_credit(vals)
            rank = _maxprobe_rank(vals)
            if credit is None or rank is None:
                continue
            top1.append(credit)
            mrr.append(1.0 / rank)
            ndcg.append(1.0 / math.log2(rank + 1.0))
        out[key] = {
            "label": label,
            "top1": float(np.mean(top1)) if top1 else float("nan"),
            "mrr": float(np.mean(mrr)) if mrr else float("nan"),
            "ndcg": float(np.mean(ndcg)) if ndcg else float("nan"),
            "n": len(top1),
        }
    return out


def _p_max_interaction(records):
    n = len(records)
    if not n:
        return float("nan"), 0
    hits = sum(1 for r in records if r["maxprobe_hit"])
    return hits / n, n


# --------------------------------------------------------------------------- #
# Plotting + stats for the extended analyses
# --------------------------------------------------------------------------- #
def plot_competition_extended(records, plotter, stats_file, filename_suffix=""):
    """Run + plot analyses 1-5, 7 for a record set; return summary dict for analysis 6.

    Analysis 9 (alluvial class -> outcome) was removed: it visualizes the same
    class x outcome contingency as Analysis 5 but with lower quantitative
    readability, so the stacked-bar version is kept instead.
    """
    suffix = filename_suffix or ""
    variant = "equal RNALength" if filename_suffix else "all competition rows"
    if not records:
        stats_file.write(f"No competition records for extended analyses ({variant}).\n\n")
        return None

    plotter.set_plot_style()
    win = competition_win_rates(records)
    margins = competition_margins(records)
    by_rank = competition_interaction_by_rank(records)
    auc = competition_auc(records)
    outcomes = competition_outcomes_by_class(records)
    ranking = competition_ranking_metrics(records)
    p_max_int, n_rec = _p_max_interaction(records)

    metric_keys = [k for k, _, _ in COMP_RANK_METRICS]
    metric_labels = [lab for _, lab, _ in COMP_RANK_METRICS]
    metric_colors = [col for _, _, col in COMP_RANK_METRICS]

    classes = [ct for ct in COMP_CLASS_ORDER if sum(outcomes.get(ct, {}).values()) > 0]

    with plt.rc_context(PUB_RC):
        # --- Analysis 1 + 4 + 7: bar panel (win rate, AUC, ranking metrics) ---
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        x = np.arange(len(metric_keys))

        win_vals = [win[k]["rate"] for k in metric_keys]
        axes[0].bar(x, win_vals, color=metric_colors, width=0.6)
        axes[0].axhline(0.5, color="0.4", ls="--", lw=0.8)
        axes[0].set_ylabel("P(maxprobe ranks first)")
        for i, v in enumerate(win_vals):
            axes[0].text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        auc_vals = [auc[k]["auc"] if auc[k]["auc"] is not None else 0 for k in metric_keys]
        axes[1].bar(x, auc_vals, color=metric_colors, width=0.6)
        axes[1].axhline(0.5, color="0.4", ls="--", lw=0.8)
        axes[1].set_ylabel("ROC-AUC (maxprobe vs minprobe)")
        for i, v in enumerate(auc_vals):
            axes[1].text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        width = 0.25
        for j, (mlabel, attr) in enumerate((("Top-1", "top1"), ("MRR", "mrr"), ("nDCG", "ndcg"))):
            vals = [ranking[k][attr] for k in metric_keys]
            axes[2].bar(x + (j - 1) * width, vals, width, label=mlabel)
        axes[2].set_ylabel("Ranking metric")
        axes[2].legend()

        panel_letters = ("a", "b", "c")
        for ax, letter in zip(axes, panel_letters):
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, rotation=20, ha="right")
            ax.set_ylim(0, 1)
            _pub_ax(ax, grid_axis="y")
            ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="right")
        fig.tight_layout()
        plotter.save_plot(f"competition_discrimination_metrics{suffix}")
        plt.close(fig)

        # --- Analysis 2: margin distributions ---
        fig, axarr = plt.subplots(1, len(metric_keys), figsize=(4.4 * len(metric_keys), 3.6))
        if len(metric_keys) == 1:
            axarr = [axarr]
        for ax, (key, label, color), letter in zip(axarr, COMP_RANK_METRICS, ("a", "b", "c")):
            data = margins[key]["margins"]
            if data:
                ax.hist(data, bins=30, color=color)
                frac_pos = float(np.mean([1.0 if d > 0 else 0.0 for d in data]))
                ax.axvline(0, color="0.2", ls="--", lw=0.8)
                ax.text(0.97, 0.95, f"{label}\nΔS>0: {frac_pos:.0%} (n={len(data)})",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("ΔS = S(maxprobe) − max S(minprobe)")
            ax.set_ylabel("Complexes")
            _pub_ax(ax)
            ax.text(-0.1, 1.04, letter, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="right")
        fig.tight_layout()
        plotter.save_plot(f"competition_score_margins{suffix}")
        plt.close(fig)

        # --- Analysis 3: P(interaction | rank) ---
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        rank_labels = [lab for lab in ["maxprobe", "min1", "min2", "min3"] if lab in by_rank]
        rank_p = [by_rank[lab]["p"] for lab in rank_labels]
        colors = [MAXPROBE_COLOR] + [MINPROBE_COLOR] * (len(rank_labels) - 1)
        ax.bar(rank_labels, rank_p, color=colors, width=0.65)
        for i, lab in enumerate(rank_labels):
            ax.text(i, rank_p[i] + 0.015, f"{rank_p[i]:.2f}\n(n={by_rank[lab]['n']})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("P(predicted interaction)")
        ax.set_ylim(0, 1.05)
        _pub_ax(ax, grid_axis="y")
        fig.tight_layout()
        plotter.save_plot(f"competition_interaction_by_rank{suffix}")
        plt.close(fig)

        # --- Analysis 5: outcome frequencies stacked by class ---
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        bottoms = np.zeros(len(classes))
        totals = [sum(outcomes[ct].values()) for ct in classes]
        for outcome in OUTCOME_ORDER:
            fracs = [outcomes[ct][outcome] / totals[i] if totals[i] else 0 for i, ct in enumerate(classes)]
            ax.bar(classes, fracs, bottom=bottoms, label=OUTCOME_LABELS[outcome],
                   color=OUTCOME_COLORS[outcome], width=0.7)
            bottoms += np.array(fracs)
        ax.set_ylabel("Fraction of complexes")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
        _pub_ax(ax)
        fig.tight_layout()
        plotter.save_plot(f"competition_outcomes_by_class{suffix}")
        plt.close(fig)

    # --- stats ---
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Extended competition analyses ({variant})\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(
        "Within-complex ranking metrics are per-chain: RNA ipTM, RNA motif score, and "
        "interface contacts (H-bond+vdW). Kd and total interface area are complex-level "
        "and excluded from within-complex ranking.\n\n"
    )
    stats_file.write(f"Complexes: {n_rec}; P(maxprobe interaction) = {p_max_int:.3f}\n\n")

    stats_file.write("Analysis 1 - maxprobe win rate (P maxprobe ranks first):\n")
    for k in metric_keys:
        stats_file.write(f"  {win[k]['label']}: {win[k]['rate']:.3f} (n={win[k]['n']})\n")

    stats_file.write("\nAnalysis 2 - max-min margins (ΔS):\n")
    for k in metric_keys:
        data = margins[k]["margins"]
        if data:
            stats_file.write(
                f"  {margins[k]['label']}: mean={np.mean(data):+.3f}, median={np.median(data):+.3f}, "
                f"ΔS>0 in {np.mean([d>0 for d in data]):.1%} (n={len(data)})\n"
            )

    stats_file.write("\nAnalysis 3 - P(interaction | experimental rank):\n")
    for lab in ["maxprobe", "min1", "min2", "min3"]:
        if lab in by_rank:
            d = by_rank[lab]
            stats_file.write(f"  {lab}: {d['p']:.3f} ({d['hits']}/{d['n']})\n")

    stats_file.write("\nAnalysis 4 - competition ROC-AUC (maxprobe=pos, minprobe=neg):\n")
    for k in metric_keys:
        a = auc[k]
        astr = f"{a['auc']:.3f}" if a["auc"] is not None else "n/a"
        stats_file.write(f"  {a['label']}: AUC={astr} (pos={a['n_pos']}, neg={a['n_neg']})\n")

    stats_file.write("\nAnalysis 5 - outcome frequencies by class:\n")
    for ct in classes:
        d = outcomes[ct]
        tot = sum(d.values())
        stats_file.write(
            f"  {ct} (n={tot}): "
            + ", ".join(f"{OUTCOME_LABELS[o]}={d[o]}" for o in OUTCOME_ORDER) + "\n"
        )

    stats_file.write("\nAnalysis 7 - ranking metrics (maxprobe = relevant probe):\n")
    for k in metric_keys:
        r = ranking[k]
        stats_file.write(
            f"  {r['label']}: Top-1={r['top1']:.3f}, MRR={r['mrr']:.3f}, nDCG={r['ndcg']:.3f} "
            f"(n={r['n']})\n"
        )
    stats_file.write("\n")

    return {
        "win": win,
        "auc": auc,
        "p_max_int": p_max_int,
        "n": n_rec,
    }


def plot_competition_domain_family(records, plotter, stats_file, min_complexes=5, filename_suffix=""):
    """Analysis 8: competition win rate + ROC-AUC per RNA-binding domain family."""
    suffix = filename_suffix or ""
    variant = "equal RNALength" if filename_suffix else "all competition rows"
    by_family = defaultdict(list)
    for rec in records:
        fam = rec.get("domain_family")
        if fam:
            by_family[fam].append(rec)

    families = sorted(
        (f for f, recs in by_family.items() if len(recs) >= min_complexes),
        key=lambda f: -len(by_family[f]),
    )
    if not families:
        stats_file.write(f"Analysis 8 - no domain family with >= {min_complexes} complexes ({variant}).\n\n")
        return

    metric_keys = [k for k, _, _ in COMP_RANK_METRICS]
    metric_labels = [lab for _, lab, _ in COMP_RANK_METRICS]

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(families))
        width = 0.8 / len(metric_keys)
        for j, (key, label, color) in enumerate(COMP_RANK_METRICS):
            win_vals, auc_vals = [], []
            for fam in families:
                recs = by_family[fam]
                w = competition_win_rates(recs)[key]["rate"]
                a = competition_auc(recs)[key]["auc"]
                win_vals.append(w if np.isfinite(w) else 0)
                auc_vals.append(a if a is not None else 0)
            off = (j - (len(metric_keys) - 1) / 2) * width
            axes[0].bar(x + off, win_vals, width, label=label, color=color)
            axes[1].bar(x + off, auc_vals, width, label=label, color=color)

        fam_ticks = [f"{f}\n(n={len(by_family[f])})" for f in families]
        for ax, ylab, letter in (
            (axes[0], "P(maxprobe ranks first)", "a"),
            (axes[1], "ROC-AUC (maxprobe vs minprobe)", "b"),
        ):
            ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(fam_ticks)
            ax.set_ylabel(ylab); ax.set_ylim(0, 1)
            _pub_ax(ax, grid_axis="y")
            ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="right")
        axes[1].legend()
        fig.tight_layout()
        plotter.save_plot(f"competition_domain_family{suffix}")
        plt.close(fig)

    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Analysis 8 - domain-family-specific competition performance ({variant})\n")
    stats_file.write("=" * 80 + "\n")
    for fam in families:
        recs = by_family[fam]
        stats_file.write(f"{fam} (n={len(recs)} complexes):\n")
        for key, label, _ in COMP_RANK_METRICS:
            w = competition_win_rates(recs)[key]["rate"]
            a = competition_auc(recs)[key]["auc"]
            astr = f"{a:.3f}" if a is not None else "n/a"
            stats_file.write(f"  {label}: win={w:.3f}, AUC={astr}\n")
    stats_file.write("\n")


def write_length_controlled_comparison(res_all, res_equal, stats_file, plotter):
    """Analysis 6: all complexes vs equal-RNA-length subset."""
    if res_all is None:
        return
    stats_file.write("=" * 80 + "\n")
    stats_file.write("Analysis 6 - length-controlled vs unrestricted comparison\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write(
        f"P(maxprobe interaction): all={res_all['p_max_int']:.3f} (n={res_all['n']})"
    )
    if res_equal:
        stats_file.write(
            f"  |  equal-length={res_equal['p_max_int']:.3f} (n={res_equal['n']})\n"
        )
    else:
        stats_file.write("  |  equal-length=n/a\n")

    metric_keys = [k for k, _, _ in COMP_RANK_METRICS]
    stats_file.write("\nP(maxprobe ranks first):\n")
    for k in metric_keys:
        a = res_all["win"][k]
        e = res_equal["win"][k]["rate"] if res_equal else float("nan")
        stats_file.write(f"  {a['label']}: all={a['rate']:.3f}  |  equal-length={e:.3f}\n")
    stats_file.write("\nCompetition ROC-AUC:\n")
    for k in metric_keys:
        a = res_all["auc"][k]
        astr = f"{a['auc']:.3f}" if a["auc"] is not None else "n/a"
        if res_equal and res_equal["auc"][k]["auc"] is not None:
            estr = f"{res_equal['auc'][k]['auc']:.3f}"
        else:
            estr = "n/a"
        stats_file.write(f"  {a['label']}: all={astr}  |  equal-length={estr}\n")
    stats_file.write("\n")

    if not res_equal:
        return
    metric_labels = [lab for _, lab, _ in COMP_RANK_METRICS]
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        x = np.arange(len(metric_keys))
        width = 0.38
        for ax, getter, ylab, letter in (
            (axes[0], lambda d, k: d["win"][k]["rate"], "P(maxprobe ranks first)", "a"),
            (axes[1], lambda d, k: (d["auc"][k]["auc"] or 0), "ROC-AUC (maxprobe vs minprobe)", "b"),
        ):
            all_vals = [getter(res_all, k) for k in metric_keys]
            eq_vals = [getter(res_equal, k) for k in metric_keys]
            ax.bar(x - width / 2, all_vals, width, label="all complexes", color="#455A64")
            ax.bar(x + width / 2, eq_vals, width, label="equal RNA length", color="#26A69A")
            ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(metric_labels, rotation=20, ha="right")
            ax.set_ylabel(ylab); ax.set_ylim(0, 1)
            _pub_ax(ax, grid_axis="y")
            ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="bottom", ha="right")
        axes[1].legend()
        fig.tight_layout()
        plotter.save_plot("competition_length_controlled_comparison")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Combined publication figure
# --------------------------------------------------------------------------- #
def plot_publication_competition_figure(records, plotter, min_complexes=5):
    """Nature-style 1x3 figure combining outcomes-by-class and domain-family performance.

    Panel A — Stacked bar: proportion of complexes per predicted binding outcome
               (maxprobe only / minprobe only / both / neither) for each competition
               class (MAX1MIN1, MAX1MIN2, MAX1MIN3).
    Panel B — Grouped bar: P(maxprobe ranks first) per RNA-binding domain family,
               for each within-complex ranking metric.
    Panel C — Grouped bar: ROC-AUC (maxprobe vs minprobe chains) per domain family.

    Saves PNG (300 dpi) and PDF to plotter.results_dir.
    """
    if not records:
        print("plot_publication_competition_figure: no records, skipping.")
        return

    # ---- data -------------------------------------------------- #
    outcomes = competition_outcomes_by_class(records)
    classes = [ct for ct in COMP_CLASS_ORDER if sum(outcomes.get(ct, {}).values()) > 0]

    by_family = defaultdict(list)
    for rec in records:
        fam = rec.get("domain_family")
        if fam:
            by_family[fam].append(rec)
    families = sorted(
        (f for f, recs in by_family.items() if len(recs) >= min_complexes),
        key=lambda f: -len(by_family[f]),
    )

    # ---- figure ------------------------------------------------ #
    # Drawn at the NAR double-column width with the central figstyle sizes, so the
    # text keeps its point size in the paper (included at scale 1).
    import figstyle
    figstyle.apply("plain")
    small = figstyle.FONT_SMALL
    fig = plt.figure(figsize=(figstyle.WIDTH_2COL, 3.1))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1, 2.3, 2.3])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ---- A: binding outcomes by competition class ----------- #
    bottoms = np.zeros(len(classes))
    totals = [sum(outcomes[ct].values()) for ct in classes]
    for outcome in OUTCOME_ORDER:
        fracs = [
            outcomes[ct][outcome] / totals[i] if totals[i] else 0.0
            for i, ct in enumerate(classes)
        ]
        ax_a.bar(
            classes, fracs, bottom=bottoms,
            label=OUTCOME_LABELS[outcome],
            color=OUTCOME_COLORS[outcome], width=0.7,
        )
        bottoms += np.array(fracs)
    for i, (ct, tot) in enumerate(zip(classes, totals)):
        ax_a.text(i, 0.97, f"n={tot}", ha="center", va="top", fontsize=small,
                  color="white", rotation=90)
    ax_a.set_ylabel("Fraction of complexes")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xticks(range(len(classes)))
    ax_a.set_xticklabels(classes, rotation=35, ha="right")
    # outcome legend joins the metric legend in one row below the figure; "neither"
    # (no complex) is left out of it
    out_handles, out_labels = [
        list(t) for t in zip(*[(h, l) for h, l in zip(*ax_a.get_legend_handles_labels())
                               if l != OUTCOME_LABELS["neither"]])]
    _pub_ax(ax_a)

    # ---- B + C: domain-family metrics ---------------------- #
    if families:
        x = np.arange(len(families))
        # figure-only restyle: ipTM_r grey (not the blue of "maxprobe only"), PAE teal (not the purple of "both")
        pub_style = {"rna_ipTM": (r"ipTM$_r$", "#616161"), "pae_oriented": ("PAE", "#26A69A")}
        n_met = len(PUB_RANK_METRICS)
        width = 0.8 / n_met

        win_vals_by_key = {}
        auc_vals_by_key = {}
        for key, _lbl, _col in PUB_RANK_METRICS:
            wv, av = [], []
            for fam in families:
                recs = by_family[fam]
                w = competition_win_rates(recs, PUB_RANK_METRICS)[key]["rate"]
                a = competition_auc(recs, PUB_RANK_METRICS)[key]["auc"]
                wv.append(float(w) if np.isfinite(w) else 0.0)
                av.append(float(a) if a is not None else 0.0)
            win_vals_by_key[key] = wv
            auc_vals_by_key[key] = av

        fam_ticks = [f"{f} (n={len(by_family[f])})" for f in families]

        for ax, vals_by_key, ylab in (
            (ax_b, win_vals_by_key, r"$P$(maxprobe ranks first)"),
            (ax_c, auc_vals_by_key, "ROC-AUC (max vs min)"),
        ):
            for j, (key, label, color) in enumerate(PUB_RANK_METRICS):
                label, color = pub_style.get(key, (label, color))
                off = (j - (n_met - 1) / 2.0) * width
                ax.bar(
                    x + off, vals_by_key[key], width,
                    label=label, color=color, alpha=0.85,
                )
            ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(fam_ticks, rotation=40, ha="right")
            ax.set_ylabel(ylab)
            ax.set_ylim(0, 1.0)
            _pub_ax(ax, grid_axis="y")
        # one legend row below all panels: A's outcomes, then the B/C metrics
        handles, labels = ax_c.get_legend_handles_labels()
        fig.legend(out_handles + handles, out_labels + labels, loc="upper center",
                   ncol=len(out_labels) + len(labels), frameon=False,
                   bbox_to_anchor=(0.5, -0.12), handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.0)
    else:
        for ax in (ax_b, ax_c):
            ax.text(
                0.5, 0.5,
                f"no family ≥ {min_complexes} complexes",
                transform=ax.transAxes, ha="center", va="center",
                color="grey", fontsize=small,
            )
            _pub_ax(ax)

    # ---- panel letters ------------------------------------- #
    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        figstyle.panel_letter(ax, letter, x=-0.18 if ax is ax_a else -0.12)

    os.makedirs(plotter.results_dir, exist_ok=True)
    figstyle.save(fig, os.path.join(plotter.results_dir, "pub_competition_outcomes_family"),
                  formats=("png", "pdf"))
    plt.close(fig)
    print(
        f"Publication figure saved: "
        f"{plotter.results_dir}/pub_competition_outcomes_family.png (+ .pdf)"
    )


# --------------------------------------------------------------------------- #
# Motif-length aggregation (from exp_probeMax.py)
# --------------------------------------------------------------------------- #
def calculate_average_motif_lengths(data, class_types):
    avg_data = {"maxprobe": defaultdict(list), "minprobe": defaultdict(list)}
    for class_type in class_types:
        for probe in ("maxprobe", "minprobe"):
            lengths = []
            lengths.extend(data[probe]["max_index"][class_type])
            lengths.extend(data[probe]["min_index"][class_type])
            if lengths:
                avg_data[probe][class_type] = float(np.mean(lengths))
    return avg_data


def get_combined_motif_lengths(data, class_types):
    combined_data = {"maxprobe": defaultdict(list), "minprobe": defaultdict(list)}
    for class_type in class_types:
        for probe in ("maxprobe", "minprobe"):
            lengths = []
            lengths.extend(data[probe]["max_index"][class_type])
            lengths.extend(data[probe]["min_index"][class_type])
            if lengths:
                combined_data[probe][class_type] = lengths
    return combined_data


# --------------------------------------------------------------------------- #
# Analysis 1: class-stratified motif-length boxplots / averages / violins
# --------------------------------------------------------------------------- #
def run_class_boxplots(pred_rows, rna_class_map, plotter, stats_file, filename_suffix=""):
    data = {
        "maxprobe": {"max_index": defaultdict(list), "min_index": defaultdict(list)},
        "minprobe": {"max_index": defaultdict(list), "min_index": defaultdict(list)},
    }

    for pred in pred_rows:
        try:
            pdb_id = pred[0]
            rna_seqs = json.loads(pred[1])
            rna_chains = json.loads(pred[2])
            chain_pairs = json.loads(pred[3])
            motif_lengths = json.loads(pred[4])
            class_type = get_class_type(pdb_id)

            if isinstance(motif_lengths, (int, float)):
                if chain_pairs and len(chain_pairs[0]) > 1:
                    target_rna_seq = rna_seqs[0]
                    score_value = float(motif_lengths)
                else:
                    continue
            else:
                max_index = motif_lengths.index(max(motif_lengths))
                min_index = motif_lengths.index(min(motif_lengths))
                max_seq = rna_seqs[max_index]
                min_seq = rna_seqs[min_index]
                max_score = float(motif_lengths[max_index])
                min_score = float(motif_lengths[min_index])

            base_pdb_id = base_protein_key(pdb_id)

            if isinstance(motif_lengths, (int, float)):
                rna_class = rna_class_map.get((base_pdb_id, target_rna_seq), "unknown")
                if rna_class == "maxprobe":
                    data["maxprobe"]["max_index"][class_type].append(score_value)
            else:
                rna_class = rna_class_map.get((base_pdb_id, max_seq), "unknown")
                if rna_class == "maxprobe":
                    data["maxprobe"]["max_index"][class_type].append(max_score)
                rna_class = rna_class_map.get((base_pdb_id, min_seq), "unknown")
                if rna_class == "maxprobe":
                    data["maxprobe"]["min_index"][class_type].append(min_score)

                rna_class = rna_class_map.get((base_pdb_id, max_seq), "unknown")
                if rna_class == "minprobe":
                    data["minprobe"]["max_index"][class_type].append(max_score)
                rna_class = rna_class_map.get((base_pdb_id, min_seq), "unknown")
                if rna_class == "minprobe":
                    data["minprobe"]["min_index"][class_type].append(min_score)
        except Exception as e:
            print(f"Error processing {pred[0]}: {e}")
            continue

    if not any(data["maxprobe"]["max_index"].values()) and not any(
        data["minprobe"]["min_index"].values()
    ):
        print(f"No class-boxplot data to plot (suffix='{filename_suffix}').")
        return

    class_types = sorted(
        set(list(data["maxprobe"]["max_index"].keys()) + list(data["minprobe"]["min_index"].keys()))
    )

    plotter.set_plot_style()

    # --- 4-panel boxplots with counts ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    panels = [
        (ax1, "maxprobe", "max_index", "Maxprobe - Max Index Motif Lengths", MAXPROBE_COLOR),
        (ax2, "maxprobe", "min_index", "Maxprobe - Min Index Motif Lengths", MAXPROBE_COLOR),
        (ax3, "minprobe", "max_index", "Minprobe - Max Index Motif Lengths", MINPROBE_COLOR),
        (ax4, "minprobe", "min_index", "Minprobe - Min Index Motif Lengths", MINPROBE_COLOR),
    ]
    for ax, probe, idx, title, color in panels:
        box_data = [data[probe][idx][ct] for ct in class_types]
        ax.boxplot(box_data, labels=class_types)
        ax.set_title(title)
        ax.set_ylabel("Motif Length")
        ax.grid(True, linestyle="--", alpha=0.7)
        for i, ct in enumerate(class_types):
            x = i + 1
            ax.scatter([x] * len(data[probe][idx][ct]), data[probe][idx][ct], alpha=0.5, color=color)
            ax.text(x, ax.get_ylim()[0], f"n={len(data[probe][idx][ct])}", ha="center", va="top")
    plt.tight_layout()
    plotter.save_plot(f"probe_class_boxplots_with_counts{filename_suffix}")
    plt.close(fig)

    # --- average motif length bar plot ---
    avg_data = calculate_average_motif_lengths(data, class_types)
    fig = plt.figure(figsize=(10, 6))
    x = np.arange(len(class_types))
    width = 0.35
    maxprobe_avgs = [avg_data["maxprobe"][ct] if avg_data["maxprobe"][ct] else 0 for ct in class_types]
    minprobe_avgs = [avg_data["minprobe"][ct] if avg_data["minprobe"][ct] else 0 for ct in class_types]
    plt.bar(x - width / 2, maxprobe_avgs, width, label="Maxprobe", color=MAXPROBE_COLOR, alpha=0.8)
    plt.bar(x + width / 2, minprobe_avgs, width, label="Minprobe", color=MINPROBE_COLOR, alpha=0.8)
    plt.xlabel("Class Type")
    plt.ylabel("Average Motif Length")
    plt.title("Average Motif Lengths by Class Type")
    plt.xticks(x, class_types)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    for i, v in enumerate(maxprobe_avgs):
        plt.text(i - width / 2, v, f"{v:.1f}", ha="center", va="bottom")
    for i, v in enumerate(minprobe_avgs):
        plt.text(i + width / 2, v, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plotter.save_plot(f"average_motif_lengths{filename_suffix}")
    plt.close(fig)

    # --- violin distributions ---
    combined_data = get_combined_motif_lengths(data, class_types)
    fig = plt.figure(figsize=(12, 6))
    positions = np.arange(len(class_types))
    width = 0.35
    maxprobe_data = [combined_data["maxprobe"][ct] or [0] for ct in class_types]
    minprobe_data = [combined_data["minprobe"][ct] or [0] for ct in class_types]
    v1 = plt.violinplot(maxprobe_data, positions=positions - width / 2, showmeans=True, showmedians=True)
    v2 = plt.violinplot(minprobe_data, positions=positions + width / 2, showmeans=True, showmedians=True)
    for pc in v1["bodies"]:
        pc.set_facecolor(MAXPROBE_COLOR)
        pc.set_alpha(0.7)
    for pc in v2["bodies"]:
        pc.set_facecolor(MINPROBE_COLOR)
        pc.set_alpha(0.7)
    for i, ct in enumerate(class_types):
        if combined_data["maxprobe"][ct]:
            plt.scatter([i - width / 2] * len(combined_data["maxprobe"][ct]),
                        combined_data["maxprobe"][ct], color=MAXPROBE_COLOR, alpha=0.3, s=20)
        if combined_data["minprobe"][ct]:
            plt.scatter([i + width / 2] * len(combined_data["minprobe"][ct]),
                        combined_data["minprobe"][ct], color=MINPROBE_COLOR, alpha=0.3, s=20)
    plt.xlabel("Class Type")
    plt.ylabel("Motif Length")
    plt.title("Motif Length Distribution by Class Type")
    plt.xticks(positions, class_types)
    plt.legend(handles=[
        Patch(facecolor=MAXPROBE_COLOR, alpha=0.7, label="Maxprobe"),
        Patch(facecolor=MINPROBE_COLOR, alpha=0.7, label="Minprobe"),
    ])
    plt.grid(True, linestyle="--", alpha=0.7)
    for i, ct in enumerate(class_types):
        plt.text(i - width / 2, plt.ylim()[0], f"n={len(combined_data['maxprobe'][ct])}", ha="center", va="top")
        plt.text(i + width / 2, plt.ylim()[0], f"n={len(combined_data['minprobe'][ct])}", ha="center", va="top")
    plt.tight_layout()
    plotter.save_plot(f"motif_length_distributions{filename_suffix}")
    plt.close(fig)

    # --- stats ---
    variant = "equal RNALength rows only" if filename_suffix else "all rows"
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Class-stratified motif lengths ({variant})\n")
    stats_file.write("=" * 80 + "\n")
    for probe in ("maxprobe", "minprobe"):
        for ct in class_types:
            for idx in ("max_index", "min_index"):
                _write_stats_block(
                    stats_file, f"{probe} | {ct} | {idx}", _summary_stats(data[probe][idx][ct])
                )
    stats_file.write("\n")


# --------------------------------------------------------------------------- #
# Analysis 2: hits/misses + RNAmotif scores (from exp_probeMax_new.py)
# --------------------------------------------------------------------------- #
def run_hits_misses(pred_rows, plotter, stats_file, filename_suffix=""):
    data = {
        "maxprobe": defaultdict(list),
        "minprobe": defaultdict(list),
        "maxprobe_miss": defaultdict(list),
        "minprobe_miss": defaultdict(list),
    }
    motif_scores = {"RNAmotif_score": defaultdict(list), "RNAMotif_inExp": defaultdict(list)}
    counts = {
        "maxprobe": defaultdict(int),
        "minprobe": defaultdict(int),
        "maxprobe_miss": defaultdict(int),
        "minprobe_miss": defaultdict(int),
    }

    for pred in pred_rows:
        try:
            pdb_id = pred[0]
            rna_chains = json.loads(pred[2])
            chain_pairs = json.loads(pred[3])
            motif_length_raw = json.loads(pred[4])
            class_type = get_class_type(pdb_id)
            if pred[5] is not None and pred[5] != "":
                motif_scores["RNAMotif_inExp"][class_type].append(float(pred[5]))
            if pred[6] is not None and pred[6] != "":
                motif_scores["RNAmotif_score"][class_type].append(float(pred[6]))

            if not rna_chains or not chain_pairs:
                continue

            if isinstance(motif_length_raw, (int, float)):
                motif_lengths = [float(motif_length_raw)] * len(chain_pairs)
            else:
                motif_lengths = [float(v) for v in motif_length_raw]

            motif_map = {}
            for i, pair in enumerate(chain_pairs):
                chain = pair[1]
                val = motif_lengths[i] if i < len(motif_lengths) else 0.0
                motif_map[chain] = float(val)

            aligned = [float(motif_map.get(chain, 0.0)) for chain in rna_chains]
            for i, val in enumerate(aligned):
                if i == 0:
                    key = "maxprobe_miss" if val == 0 else "maxprobe"
                else:
                    key = "minprobe_miss" if val == 0 else "minprobe"
                data[key][class_type].append(val)
                counts[key][class_type] += 1
        except Exception as e:
            print(f"Error processing {pred[0]}: {e}")
            continue

    class_types = sorted(
        set(
            list(data["maxprobe"].keys())
            + list(data["minprobe"].keys())
            + list(data["maxprobe_miss"].keys())
            + list(data["minprobe_miss"].keys())
        )
    )
    if not class_types:
        print(f"No hits/misses data to plot (suffix='{filename_suffix}').")
        return

    plotter.set_plot_style()

    # --- hits + misses boxplots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    max_box, min_box = [], []
    for ct in class_types:
        max_box.append(data["maxprobe"][ct] + data["maxprobe_miss"][ct])
        min_box.append(data["minprobe"][ct] + data["minprobe_miss"][ct])
    axes[0].boxplot([b or [0] for b in max_box], labels=class_types, showfliers=True)
    axes[0].set_title("Maxprobe Motif Lengths (Hits + Misses)")
    axes[0].set_ylabel("Motif Length")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[1].boxplot([b or [0] for b in min_box], labels=class_types, showfliers=True)
    axes[1].set_title("Minprobe Motif Lengths (Hits + Misses)")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plotter.save_plot(f"probe_hits_misses_boxplots{filename_suffix}")
    plt.close(fig)

    # --- hit/miss counts bar plot ---
    fig = plt.figure(figsize=(10, 6))
    x = range(len(class_types))
    width = 0.2
    plt.bar(x, [counts["maxprobe"][ct] for ct in class_types], width, label="Maxprobe Hit")
    plt.bar([p + width for p in x], [counts["maxprobe_miss"][ct] for ct in class_types], width, label="Maxprobe Miss")
    plt.bar([p + 2 * width for p in x], [counts["minprobe"][ct] for ct in class_types], width, label="Minprobe Hit")
    plt.bar([p + 3 * width for p in x], [counts["minprobe_miss"][ct] for ct in class_types], width, label="Minprobe Miss")
    plt.title("Hit / Miss Probe Counts per Class")
    plt.ylabel("Counts")
    plt.xlabel("Class Type")
    plt.xticks([p + 1.5 * width for p in x], class_types, rotation=30)
    plt.legend()
    plt.tight_layout()
    plotter.save_plot(f"hit_miss_counts{filename_suffix}")
    plt.close(fig)

    # --- RNAmotif score boxplots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([motif_scores["RNAmotif_score"][ct] or [0] for ct in class_types],
                    labels=class_types, showfliers=True)
    axes[0].set_title("RNAMotif Score per Class")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].boxplot([motif_scores["RNAMotif_inExp"][ct] or [0] for ct in class_types],
                    labels=class_types, showfliers=True)
    axes[1].set_title("RNAMotif InExp Score per Class")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plotter.save_plot(f"rnamotif_score_boxplots{filename_suffix}")
    plt.close(fig)

    # --- stats ---
    variant = "equal RNALength rows only" if filename_suffix else "all rows"
    stats_file.write("=" * 80 + "\n")
    stats_file.write(f"Hits/misses + RNAmotif scores (1 RBP template, no protein template; {variant})\n")
    stats_file.write("=" * 80 + "\n")
    stats_file.write("Hit/miss counts per class:\n")
    for ct in class_types:
        stats_file.write(
            f"  {ct}: maxprobe hit={counts['maxprobe'][ct]}, miss={counts['maxprobe_miss'][ct]}; "
            f"minprobe hit={counts['minprobe'][ct]}, miss={counts['minprobe_miss'][ct]}\n"
        )
    stats_file.write("Motif lengths (hits + misses):\n")
    for ct in class_types:
        _write_stats_block(stats_file, f"maxprobe | {ct}",
                           _summary_stats(data["maxprobe"][ct] + data["maxprobe_miss"][ct]))
        _write_stats_block(stats_file, f"minprobe | {ct}",
                           _summary_stats(data["minprobe"][ct] + data["minprobe_miss"][ct]))
    stats_file.write("RNAmotif_score per class:\n")
    for ct in class_types:
        _write_stats_block(stats_file, ct, _summary_stats(motif_scores["RNAmotif_score"][ct]))
    stats_file.write("RNAMotif_inExp per class:\n")
    for ct in class_types:
        _write_stats_block(stats_file, ct, _summary_stats(motif_scores["RNAMotif_inExp"][ct]))
    stats_file.write("\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_rna_exp_map(csv_file_path):
    """Map (base_pdb_id, RNASequence) -> Class and experimental RNAMotif from exp CSV."""
    rna_exp_map = {}
    with open(csv_file_path, mode="r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            pdb_base = base_protein_key(row["PDBId"])
            rna_seq = row["RNASequence"].strip()
            rna_exp_map[(pdb_base, rna_seq)] = {
                "class": row["Class"].strip(),
                "exp_motif": (row.get("RNAMotif") or "").strip(),
            }
    return rna_exp_map


def load_rna_class_map(csv_file_path):
    """Backward-compatible Class-only map used by class boxplots."""
    exp_map = load_rna_exp_map(csv_file_path)
    return {key: val["class"] for key, val in exp_map.items()}


def load_pred_rows(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT exp_db_id, RNASequence, RNAChainIDs, ChainIDpairList_proteinRNA, RNAMotifLength,
               RNAMotif_inExp, RNAmotif_score, RNALength, RBP_PDB_Templates, Protein_PDB_Templates,
               RNAMotif, ProteinRNAInterfaceArea, ProteinLength,
               af3_rna_ipTM, Hbond_proteinRNA, vdWbond_proteinRNA, Binding_affinity_kd,
               af3_chain_pair_pae_min_perchain
        FROM pred_protein_rna
        """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", help="SQLite DB path (default: config.json db_path)")
    parser.add_argument(
        "--csv",
        default=os.path.join(_REPO_ROOT, "data", "protein_rna", "exp_protein_rna.csv"),
        help="Reference exp_protein_rna CSV (PDBId, RNASequence, Class)",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots + stats")
    parser.add_argument(
        "--no-update-db",
        action="store_true",
        help="Skip writing maxprobe-chain RNAmotif_score / RNAMotif_inExp / InterfaceArea_norm to DB",
    )
    parser.add_argument(
        "--pub",
        action="store_true",
        help="Generate only the combined publication figure (outcomes by class + domain family)",
    )
    args = parser.parse_args(argv)

    cfg = StartConfig()
    db_path = os.path.abspath(args.db or cfg.get_database_path())

    rna_exp_map = load_rna_exp_map(args.csv)
    rna_class_map = {key: val["class"] for key, val in rna_exp_map.items()}
    pred_rows = load_pred_rows(db_path)

    if not args.no_update_db:
        update_competition_db_scores(db_path, pred_rows, rna_exp_map)
        pred_rows = load_pred_rows(db_path)

    # Equal-RNALength subset (col index 7 = RNALength).
    equal_rows = [r for r in pred_rows if _rna_lengths_all_equal(r[7])]
    competition_rows = [r for r in pred_rows if _is_competition_id(r[0])]
    competition_equal_rows = [r for r in equal_rows if _is_competition_id(r[0])]
    # Hits/misses analysis uses the 1-RBP-template / no-protein-template subset (cols 8, 9).
    template_rows = [r for r in pred_rows if _passes_new_template_filter(r[8], r[9])]
    template_equal_rows = [r for r in template_rows if _rna_lengths_all_equal(r[7])]

    print(
        f"pred rows: {len(pred_rows)} (competition: {len(competition_rows)}, "
        f"equal RNALength: {len(equal_rows)}); "
        f"template subset: {len(template_rows)} (equal RNALength: {len(template_equal_rows)})"
    )

    plotter = PlotCreator(plot_type=PLOT_TYPE, standalone_outdir=args.outdir)
    print(f"Saving plots under {plotter.results_dir}")

    if args.pub:
        fam_by_base = load_domain_family_map(db_path)
        records_all = build_competition_records(competition_rows, rna_exp_map, fam_by_base)
        plotter.set_plot_style()
        plot_publication_competition_figure(records_all, plotter)
        return 0

    stats_path = os.path.join(plotter.results_dir, STATS_FILENAME)
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        stats_file.write("=" * 80 + "\n")
        stats_file.write("RNAcompete maxprobe/minprobe combined statistics\n")
        stats_file.write("=" * 80 + "\n\n")
        stats_file.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        stats_file.write(f"Database: {db_path}\n")
        stats_file.write(f"Reference CSV: {args.csv}\n")
        stats_file.write(f"Total pred rows: {len(pred_rows)}\n")
        stats_file.write(f"  equal-RNALength rows: {len(equal_rows)}\n")
        stats_file.write(f"  1-RBP-template / no-protein-template rows: {len(template_rows)}\n")
        stats_file.write(f"    of those, equal-RNALength: {len(template_equal_rows)}\n")
        stats_file.write(f"  competition complexes (MAX1MIN*): {len(competition_rows)}\n\n")

        # Competition interaction + maxprobe-chain score summaries and plots.
        summary_all = run_interaction_analysis(competition_rows, rna_exp_map, stats_file, filename_suffix="")
        run_maxprobe_chain_score_stats(competition_rows, rna_exp_map, stats_file, filename_suffix="")
        plot_competition_findings(plotter, summary_all, filename_suffix="")
        if competition_equal_rows:
            summary_eq = run_interaction_analysis(
                competition_equal_rows, rna_exp_map, stats_file, filename_suffix="_equalRNAlength"
            )
            run_maxprobe_chain_score_stats(
                competition_equal_rows, rna_exp_map, stats_file, filename_suffix="_equalRNAlength"
            )
            plot_competition_findings(plotter, summary_eq, filename_suffix="_equalRNAlength")

        # Extended competition analyses (within-complex ranking of competing probes).
        fam_by_base = load_domain_family_map(db_path)
        records_all = build_competition_records(competition_rows, rna_exp_map, fam_by_base)
        records_equal = build_competition_records(competition_equal_rows, rna_exp_map, fam_by_base)

        res_all = plot_competition_extended(records_all, plotter, stats_file, filename_suffix="")
        res_equal = (
            plot_competition_extended(records_equal, plotter, stats_file, filename_suffix="_equalRNAlength")
            if records_equal else None
        )
        plot_competition_domain_family(records_all, plotter, stats_file, filename_suffix="")
        write_length_controlled_comparison(res_all, res_equal, stats_file, plotter)

        # Analysis 1: class boxplots (all matched rows).
        run_class_boxplots(pred_rows, rna_class_map, plotter, stats_file, filename_suffix="")
        if equal_rows:
            run_class_boxplots(equal_rows, rna_class_map, plotter, stats_file, filename_suffix="_equalRNAlength")
        else:
            print("Skipping equal-RNALength class boxplots (no matching rows).")

        # Analysis 2: hits/misses (template subset).
        run_hits_misses(template_rows, plotter, stats_file, filename_suffix="")
        if template_equal_rows:
            run_hits_misses(template_equal_rows, plotter, stats_file, filename_suffix="_equalRNAlength")
        else:
            print("Skipping equal-RNALength hits/misses (no matching rows).")

    print(f"Statistics written to: {stats_path}")
    return 0


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
COMPOSITE_PANEL_NAMES = (
    "competition_outcomes_by_class",
    "competition_domain_family_winrate",
    "competition_domain_family_auc",
)


def _default_competition_csv(db_path):
    d = os.path.dirname(os.path.abspath(db_path))
    for cand in (os.path.join(d, "protein_rna", "exp_protein_rna.csv"),
                 os.path.join(d, "exp_protein_rna.csv"),
                 os.path.join(os.path.dirname(d), "exp_protein_rna.csv")):
        if os.path.exists(cand):
            return cand
    return None


def _build_competition_records_for(db_path, csv_path):
    rna_exp_map = load_rna_exp_map(csv_path)
    pred_rows = load_pred_rows(db_path)
    fam_by_base = load_domain_family_map(db_path)
    comp_rows = [r for r in pred_rows if _is_competition_id(r[0])]
    return build_competition_records(comp_rows, rna_exp_map, fam_by_base)


def _draw_outcomes_by_class(records):
    outcomes = competition_outcomes_by_class(records)
    classes = [ct for ct in COMP_CLASS_ORDER if sum(outcomes.get(ct, {}).values()) > 0]

    def draw(ax):
        if not classes:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color="grey")
            return
        bottoms = np.zeros(len(classes))
        totals = [sum(outcomes[ct].values()) for ct in classes]
        for outcome in OUTCOME_ORDER:
            fracs = [outcomes[ct][outcome] / totals[i] if totals[i] else 0.0
                     for i, ct in enumerate(classes)]
            ax.bar(classes, fracs, bottom=bottoms, label=OUTCOME_LABELS[outcome],
                   color=OUTCOME_COLORS[outcome], width=0.65)
            bottoms += np.array(fracs)
        for i, (ct, tot) in enumerate(zip(classes, totals)):
            ax.text(i, 1.02, f"n={tot}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Fraction of complexes")
        ax.set_ylim(0, 1.12)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7)
        _pub_ax(ax)
    return draw


def _draw_competition_domain_family(records, which, min_complexes=5):
    from collections import defaultdict as _dd
    by_family = _dd(list)
    for rec in records:
        fam = rec.get("domain_family")
        if fam:
            by_family[fam].append(rec)
    families = sorted((f for f, r in by_family.items() if len(r) >= min_complexes),
                      key=lambda f: -len(by_family[f]))

    def draw(ax):
        if not families:
            ax.text(0.5, 0.5, f"no family ≥ {min_complexes} complexes",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
            return
        x = np.arange(len(families))
        n = len(COMP_RANK_METRICS)
        width = 0.8 / n
        for j, (key, label, color) in enumerate(COMP_RANK_METRICS):
            vals = []
            for fam in families:
                recs = by_family[fam]
                if which == "win":
                    v = competition_win_rates(recs)[key]["rate"]
                else:
                    a = competition_auc(recs)[key]["auc"]
                    v = a if a is not None else 0.0
                vals.append(v if np.isfinite(v) else 0.0)
            off = (j - (n - 1) / 2.0) * width
            ax.bar(x + off, vals, width, label=label, color=color, alpha=0.85)
        ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}\n(n={len(by_family[f])})" for f in families])
        ax.set_ylabel("P(maxprobe ranks first)" if which == "win"
                      else "ROC-AUC (maxprobe vs minprobe)")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7)
        _pub_ax(ax, grid_axis="y")
    return draw


def composite_panels(names=None, single_chain_only=False, msa_option=None,
                     db_path=None, csv_path=None):
    """Return {base_name: (draw_fn, title)} for plots.composite_figures.

    Needs the competition (MAX1MIN*) dataset; point ``db`` at it (e.g.
    data/rnaCompete_v5_MINMAX/rbpDatabase.db). The matching exp CSV is located
    next to the DB unless ``csv_path`` is given.
    """
    from plots.publication_figure import normalize_panel_name
    wanted = ([normalize_panel_name(n) for n in names]
              if names else list(COMPOSITE_PANEL_NAMES))

    db_path = db_path or StartConfig().get_database_path()
    csv_path = csv_path or _default_competition_csv(db_path)
    if csv_path is None:
        raise SystemExit(
            "exp_probeMax_combined: could not find exp_protein_rna.csv next to the "
            "DB. Point --db at the rnaCompete_v5_MINMAX dataset.")
    records = _build_competition_records_for(db_path, csv_path)

    registry = {
        "competition_outcomes_by_class": (
            _draw_outcomes_by_class(records),
            "Binding outcome by competition class"),
        "competition_domain_family_winrate": (
            _draw_competition_domain_family(records, "win"),
            "Maxprobe win rate by domain family"),
        "competition_domain_family_auc": (
            _draw_competition_domain_family(records, "auc"),
            "ROC-AUC by domain family"),
    }
    return {n: registry[n] for n in wanted if n in registry}


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="competition_composite"))
    raise SystemExit(main())
