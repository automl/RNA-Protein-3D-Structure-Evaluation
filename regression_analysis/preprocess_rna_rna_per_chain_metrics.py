#!/usr/bin/env python
"""Per-RNA-chain analysis table for MULTI-CHAIN RNA complexes (two or more RNA chains, no
protein), parsed from the rna_rna databases (tables pred_rna_rna / exp_rna_rna).

Mirrors preprocess_rna_only_per_chain_metrics.py (single-chain RNAs) but expands every
complex into one row per predicted RNA chain, using the chain-keyed metrics the evaluation
stores (RNA_TM_AllChains / RNA_LDDT_AllChains as [{"A": 0.69}, ...]) and the per-chain
AF3 confidences (af3_rna_pTM / af3_rna_ipTM as lists in RNAChainIDs order).

Column names follow the per-chain registry of single_chain_level_regression_analysis_final.py
so the same grouped regression, variance partition and stability selection run unchanged:
  G0  af3_rna_pLDDT_avg, af3_global_pae_avg, af3_rna_pTM_list_value, af3_rna_ipTM_list_value
      (chain-pair PAE and all protein-side confidences are absent -> dropped by the screen)
  G1  rna_family, rna_family_is_known, n_unique_rna_families (no RNA alignment depth is
      stored for these predictions, so log10_rna_msa_size is NaN; no templates)
  G2  NumberRNAs, pred_rna_chain_length, rna_total_length, total_complex_length,
      max_rna_fraction_of_total_rna (protein terms are 0 and drop as constants)
  G3  RNA_GC_Content, RNA_SequenceComplexity, RNA_Shannon_Entropy_K1/K2 computed from the
      predicted chain sequence (no thermodynamic descriptors: not stored and ViennaRNA is
      not a dependency of this pipeline); secondary-structure counts are NaN
  G4  RNA-RNA interface of the PREDICTED model, per chain: chain_n_unique_rna_interface_residues,
      chain_interface_contact_distance_mean, chain_frac_contacts_lt_3_5A (from ContactList),
      plus chain_n_partner_chains_in_contact and chain_n_hbonds_rna_rna (informative extras)
  G5  not available for this set (no ions, pH or temperature are stored) -> group empty

Outputs: <outdir>/<db>/<db>__per_chain_all.csv and <outdir>/combined/combined__per_chain_all.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocess_rna_only_per_chain_metrics import (
    clean_rna_sequence,
    infer_post_cutoff_from_db_tag,
    load_table_columns,
    safe_literal_eval,
    sequence_complexity,
    shannon_entropy_kmers,
    to_float,
)

EXP_COLUMNS = ["PDBId", "RNAFamily", "RNAModified", "RNASequence", "RNAChainIDs", "RNALength",
               "NumberRNAs", "Experiment", "XRayResolution", "Deposited_date", "RNADescription"]
PRED_COLUMNS = ["exp_db_id", "RNASequence", "RNAChainIDs", "RNALength", "NumberRNAs", "RNAModified",
                "ContactList", "ChainIDpairList_rnaRNA", "Hbond_rnaRNA", "vdWbond_rnaRNA",
                "af3_rna_pTM", "af3_rna_ipTM", "af3_rna_pLDDT_avg", "af3_global_pae_avg",
                "af3_chain_pair_pae_min", "af3_fraction_disordered", "af3_has_clash", "af3_ranking_score",
                "af3_rna_MSA", "Complex_RMSD", "RNA_RMSD", "RNA_LDDT", "Complex_LDDT", "RNA_TM", "Complex_TM",
                "RNA_LDDT_allAtom", "Complex_LDDT_allAtom", "RNAComplex_TM",
                "RNA_TM_AllChains", "RNA_LDDT_AllChains", "RNA_chain_assignment"]

_CONTACT_RE = re.compile(r"^\s*([A-Za-z]+)\((-?\d+)\)-([A-Za-z]+)\((-?\d+)\)\s*:\s*([-\d.]+)")


def _as_list(x: Any) -> List[Any]:
    v = safe_literal_eval(x)
    if v is None:
        try:
            v = json.loads(x) if isinstance(x, str) and x.strip() else None
        except Exception:
            v = None
    return list(v) if isinstance(v, (list, tuple)) else ([] if v is None else [v])


def _keyed(x: Any) -> Dict[str, float]:
    """[{"A": 0.69}, {"B": 0.74}] -> {"A": 0.69, "B": 0.74}"""
    out: Dict[str, float] = {}
    for item in _as_list(x):
        if isinstance(item, dict):
            for k, v in item.items():
                fv = to_float(v)
                if not np.isnan(fv):
                    out[str(k)] = float(fv)
    return out


def _per_chain_value(lst: Any, idx: int) -> float:
    vals = _as_list(lst)
    if idx < len(vals):
        return to_float(vals[idx])
    return np.nan


def _family_per_chain(fam: Any, idx: int) -> Optional[str]:
    vals = _as_list(fam)
    if not vals:
        return None
    v = vals[idx] if idx < len(vals) else None
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None", "nan", "null") else s


def chain_interface_features(chain_ids: List[str], pair_list: Any, contact_list: Any,
                             hbond_list: Any) -> Dict[str, Dict[str, float]]:
    """Per-chain RNA-RNA interface descriptors of the predicted model.

    ChainIDpairList_rnaRNA lists ordered pairs ([X, Y] and [Y, X] both occur); ContactList
    holds one block per pair with entries 'C(1)-G(13): 2.68' (left residue on X, right on Y,
    distance in A). Each unordered pair is counted once (first occurrence)."""
    pairs = [tuple(str(t) for t in p) for p in _as_list(pair_list) if isinstance(p, (list, tuple)) and len(p) == 2]
    blocks = _as_list(contact_list)
    hb = [to_float(v) for v in _as_list(hbond_list)]
    seen = set()
    res = {c: {"pos": set(), "dists": [], "partners": set(), "hbonds": 0.0} for c in chain_ids}
    for i, (x, y) in enumerate(pairs):
        key = frozenset((x, y))
        if key in seen or x not in res or y not in res:
            continue
        seen.add(key)
        block = blocks[i] if i < len(blocks) and isinstance(blocks[i], (list, tuple)) else []
        n_contacts = 0
        for entry in block:
            m = _CONTACT_RE.match(str(entry))
            if not m:
                continue
            d = to_float(m.group(5))
            res[x]["pos"].add(int(m.group(2))); res[y]["pos"].add(int(m.group(4)))
            if not np.isnan(d):
                res[x]["dists"].append(d); res[y]["dists"].append(d)
            n_contacts += 1
        if n_contacts:
            res[x]["partners"].add(y); res[y]["partners"].add(x)
        if i < len(hb) and not np.isnan(hb[i]):
            res[x]["hbonds"] += hb[i]; res[y]["hbonds"] += hb[i]
    out: Dict[str, Dict[str, float]] = {}
    for c, r in res.items():
        d = np.array(r["dists"], dtype=float)
        out[c] = {
            "chain_n_unique_rna_interface_residues": float(len(r["pos"])),
            "chain_interface_contact_distance_mean": float(d.mean()) if d.size else np.nan,
            "chain_frac_contacts_lt_3_5A": float((d < 3.5).mean()) if d.size else np.nan,
            "chain_n_partner_chains_in_contact": float(len(r["partners"])),
            "chain_n_hbonds_rna_rna": float(r["hbonds"]),
        }
    return out


def build_rows(pred: pd.DataFrame, exp: pd.DataFrame, db_tag: str) -> pd.DataFrame:
    merged = pred.merge(exp, left_on="exp_db_id", right_on="PDBId", how="left", suffixes=("_pred", "_exp"))
    is_post = infer_post_cutoff_from_db_tag(db_tag)
    rows: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        chain_ids = [str(c) for c in _as_list(r.get("RNAChainIDs_pred"))]
        lengths = [to_float(v) for v in _as_list(r.get("RNALength_pred"))]
        seqs = [clean_rna_sequence(s) for s in _as_list(r.get("RNASequence_pred"))]
        tm = _keyed(r.get("RNA_TM_AllChains")); lddt = _keyed(r.get("RNA_LDDT_AllChains"))
        assign = safe_literal_eval(r.get("RNA_chain_assignment")) or {}
        exp_chain_ids = [str(c) for c in _as_list(r.get("RNAChainIDs_exp"))]
        # pred-only columns keep their bare name after the merge; shared ones get the _pred suffix
        g = lambda c: r.get(c + "_pred", r.get(c))
        iface = chain_interface_features(chain_ids, g("ChainIDpairList_rnaRNA"), g("ContactList"), g("Hbond_rnaRNA"))
        total_len = float(np.nansum([v for v in lengths if not np.isnan(v)])) if lengths else np.nan
        fams = []
        for j in range(len(exp_chain_ids)):
            f = _family_per_chain(r.get("RNAFamily"), j)
            if f: fams.append(f)
        n_fam = len(set(fams))
        for i, ch in enumerate(chain_ids):
            exp_ch = str(assign.get(ch, ch)) if isinstance(assign, dict) else ch
            exp_idx = exp_chain_ids.index(exp_ch) if exp_ch in exp_chain_ids else i
            seq = seqs[i] if i < len(seqs) else ""
            L = lengths[i] if i < len(lengths) else (float(len(seq)) if seq else np.nan)
            fam = _family_per_chain(r.get("RNAFamily"), exp_idx)
            row: Dict[str, Any] = {
                "exp_db_id": r.get("exp_db_id"), "PDBId": r.get("PDBId"), "pdb_id": str(r.get("exp_db_id")),
                "source_db": db_tag, "is_post_cutoff": is_post, "Experiment": r.get("Experiment"),
                "RNADescription": r.get("RNADescription"),
                # chain identity
                "pred_rna_chain_index": i, "pred_rna_chain_id": ch, "exp_rna_chain_id": exp_ch,
                "pred_rna_sequence": seq, "pred_rna_chain_length": L,
                "pred_rna_is_modified": int(bool(re.search(r"[^ACGUN]", seq))) if seq else 0,
                # targets
                "assigned_rna_tm": tm.get(ch, np.nan), "assigned_rna_lddt": lddt.get(ch, np.nan),
                "is_matched": int(ch in tm), "assignment_method": "eval_assigned",
                # G0
                "af3_rna_pLDDT_avg": to_float(r.get("af3_rna_pLDDT_avg")),
                "af3_global_pae_avg": to_float(r.get("af3_global_pae_avg")),
                "af3_chain_pair_pae_min": np.nan,
                "af3_rna_pTM_list_value": _per_chain_value(r.get("af3_rna_pTM"), i),
                "af3_rna_ipTM_list_value": _per_chain_value(r.get("af3_rna_ipTM"), i),
                "af3_ranking_score": to_float(r.get("af3_ranking_score")),
                "af3_fraction_disordered": to_float(r.get("af3_fraction_disordered")),
                "af3_has_clash": to_float(r.get("af3_has_clash")),
                "af3_protein_pLDDT_avg": np.nan, "af3_protein_pTM_list_value": np.nan, "af3_protein_ipTM_list_value": np.nan,
                # G1
                "rna_family": fam, "rna_family_is_known": int(fam is not None), "n_unique_rna_families": n_fam,
                "rna_family_summary": fam, "log10_rna_msa_size": np.nan, "rna_msa_size": np.nan,
                "log10_protein_msa_size": np.nan, "n_rbp_templates": np.nan, "n_protein_templates": np.nan,
                # G2
                "NumberRNAs": to_float(r.get("NumberRNAs_pred")), "NumberProteins": 0,
                "rna_total_length": total_len, "total_complex_length": total_len,
                "protein_total_length": 0.0, "protein_to_rna_length_ratio": 0.0,
                "max_rna_fraction_of_total_rna": (max(v for v in lengths if not np.isnan(v)) / total_len) if lengths and total_len else np.nan,
                "rna_chain_fraction_of_total_rna": (L / total_len) if total_len and not np.isnan(L) else np.nan,
                # G3 (sequence-derived)
                "RNA_GC_Content": (sum(seq.count(b) for b in "GC") / max(len(seq), 1)) if seq else np.nan,
                "RNA_SequenceComplexity": sequence_complexity(seq) if seq else np.nan,
                "RNA_Shannon_Entropy_K1": shannon_entropy_kmers(seq, 1) if seq else np.nan,
                "RNA_Shannon_Entropy_K2": shannon_entropy_kmers(seq, 2) if seq else np.nan,
                "RNA_MFE_Value": np.nan, "RNA_Ensemble_Diversity": np.nan, "RNA_Centroid_Energy": np.nan, "log10_kd": np.nan,
                "num_stems": np.nan, "num_hairpin_loops": np.nan, "num_internal_loops": np.nan,
                "num_multibranch_loops": np.nan, "num_pseudoknots": np.nan,
                # G4 (predicted RNA-RNA interface)
                "pred_ProteinRNAInterfaceArea": np.nan, "pred_ProteinRNAInterfaceRatio": np.nan,
                "chain_interface_rna_motif_length_sum": np.nan,
                **iface.get(ch, {}),
                # G5: not available
                "n_ions_listed": np.nan, "n_ligands_listed": np.nan, "ionic_strength": np.nan, "peg_percent": np.nan,
                "buffer_m": np.nan, "has_peg": np.nan, "peg_concentration_category": np.nan, "buffer_category": np.nan,
                "exp_pH": np.nan, "exp_Temp": np.nan,
                # complex-level metrics carried along (descriptive)
                "RNA_TM": to_float(r.get("RNA_TM")), "RNA_LDDT_allAtom": to_float(r.get("RNA_LDDT_allAtom")),
                "Complex_TM": to_float(r.get("Complex_TM")), "Complex_LDDT_allAtom": to_float(r.get("Complex_LDDT_allAtom")),
                "RNAComplex_TM": to_float(r.get("RNAComplex_TM")),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def process_db(db_path: Path, out_root: Path) -> pd.DataFrame:
    db_tag = db_path.stem
    out_dir = out_root / db_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        exp = load_table_columns(conn, "exp_rna_rna", EXP_COLUMNS)
        pred = load_table_columns(conn, "pred_rna_rna", PRED_COLUMNS)
    df = build_rows(pred, exp, db_tag)
    out_csv = out_dir / f"{db_tag}__per_chain_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{db_tag}] complexes: {len(pred)}; per-chain rows: {len(df)}; matched (TM available): {int(df['is_matched'].sum())}; "
          f"AF3 pTM available: {int(df['af3_rna_pTM_list_value'].notna().sum())} -> {out_csv}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dbs", nargs="+", required=True, help="multi-chain RNA SQLite db files (pred/exp_rna_rna tables)")
    ap.add_argument("--outdir", default="csvs4analysis_rna_rna", help="output folder")
    args = ap.parse_args()
    out_root = Path(args.outdir)
    dfs = [process_db(Path(p), out_root) for p in args.dbs]
    comb = out_root / "combined"
    comb.mkdir(parents=True, exist_ok=True)
    per_chain = pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()
    per_chain.to_csv(comb / "combined__per_chain_all.csv", index=False)
    pd.DataFrame([{"subset": "per_chain_all", "n_rows": len(per_chain)},
                  {"subset": "per_chain_matched", "n_rows": int(per_chain["is_matched"].sum()) if len(per_chain) else 0}]
                 ).to_csv(comb / "combined__per_chain_summary.csv", index=False)
    print(f"[combined] per-chain rows: {len(per_chain)}; matched: {int(per_chain['is_matched'].sum()) if len(per_chain) else 0}; "
          f"complexes: {per_chain['pdb_id'].nunique() if len(per_chain) else 0}")


if __name__ == "__main__":
    main()
