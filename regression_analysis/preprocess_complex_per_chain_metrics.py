#!/usr/bin/env python3
"""
dbs2single_chain_analysis_csvs_v2.py

Create analysis-ready PER-CHAIN CSVs from AF3-eval SQLite DBs.

Compared with the older version, this one additionally:
- keeps all old parsing behavior
- parses new RNA all-chain metrics:
    * RNA_Dockq_AllChains
    * RNA_ICS_AllChains
    * RNA_iLDDT_AllChains
    * RNA_QS_global_AllChains
- parses new RBP pair metrics:
    * RBP_irmsd
    * RBP_DockQ
    * RBP_fnat
    * RBP_QS
  as complex-level summary features
- adds explicit missingness / availability flags for assigned per-chain metrics
- records matrix-shape diagnostics so malformed entries can be excluded later
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


# =============================================================================
# Small helpers
# =============================================================================
def safe_literal_eval(x: Any) -> Any:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, dict, tuple)):
        return x
    if not isinstance(x, str):
        return x
    s = x.strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return x


def to_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() in {"", ".", "None", "nan", "NaN"}:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def first_list_value(x: Any) -> float:
    v = safe_literal_eval(x)
    if isinstance(v, list) and len(v) > 0:
        return to_float(v[0])
    return to_float(v)


def count_pdb_ids(x: Any) -> int:
    v = safe_literal_eval(x)
    if isinstance(v, list):
        return sum(1 for a in v if isinstance(a, str) and str(a).strip())
    return 0


def id_in_template_list(sample_id: Any, x: Any) -> int:
    sid = str(sample_id).strip().upper() if sample_id is not None else ""
    v = safe_literal_eval(x)
    if isinstance(v, list):
        vals = {str(a).strip().upper() for a in v if a is not None}
        return int(sid in vals)
    return 0


def parse_pairs_list(x: Any) -> List[Tuple[Any, str]]:
    v = safe_literal_eval(x)
    out: List[Tuple[Any, str]] = []
    if isinstance(v, list):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((item[0], str(item[1]).strip()))
    return out


def ensure_list(x: Any) -> List[Any]:
    v = safe_literal_eval(x)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def ensure_dict(x: Any) -> Dict[str, Any]:
    v = safe_literal_eval(x)
    if isinstance(v, dict):
        return v
    return {}


def normalize_name(text: str) -> str:
    s = str(text).strip().lower()

    replacements = {
        "polyethylene glycol": "peg",
        "peg ": "peg_",
        "peg-": "peg_",
        "peg(": "peg_",
        "poly ethylene glycol": "peg",
        "tris-hcl": "tris_hcl",
        "tris-cl": "tris_cl",
        "bis-tris": "bis_tris",
        "bis tris": "bis_tris",
        "hepes-naoh": "hepes_naoh",
        "phosphate-citrate": "phosphate_citrate",
        "sodium-potassium phosphate": "sodium_potassium_phosphate",
        "mono-methyl-ether": "mme",
        "mono methyl ether": "mme",
        "w/v": "",
        "v/v": "",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    s = s.replace("%", "")
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace("/", " ")
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")

    return s or "unknown"


def unit_to_molar(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "m":
        return value
    if u == "mm":
        return value * 1e-3
    if u in {"um", "µm"}:
        return value * 1e-6
    if u == "nm":
        return value * 1e-9
    return np.nan


def infer_post_cutoff_from_db_tag(db_tag: str) -> int:
    tag = str(db_tag).strip().lower()
    return int(tag.startswith("a"))


def as_clean_string_list(x: Any) -> List[str]:
    vals = ensure_list(x)
    out: List[str] = []
    for v in vals:
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


def normalize_missing_token(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return s


_ALLOWED_RNA_CHARS = set("ACGUTN")


def sequence_is_modified(seq: Any) -> float:
    s = normalize_missing_token(seq)
    if s is None:
        return np.nan
    letters = re.sub(r"\s+", "", s).upper()
    if letters == "":
        return np.nan
    return float(any(ch not in _ALLOWED_RNA_CHARS and ch not in {"(", ")"} and not ch.isdigit() for ch in letters))


def unique_in_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_family_entry(x: Any) -> List[Optional[str]]:
    vals = ensure_list(x)
    out: List[Optional[str]] = []
    for v in vals:
        out.append(normalize_missing_token(v))
    return out


def map_family_to_sequences(
    sequences: List[str],
    family_value: Any,
    n_expected: Optional[int] = None,
) -> List[Optional[str]]:
    seqs = [str(s) if s is not None else "" for s in sequences]
    if n_expected is not None and len(seqs) < n_expected:
        seqs = seqs + [""] * (n_expected - len(seqs))
    if n_expected is not None and len(seqs) > n_expected:
        seqs = seqs[:n_expected]

    if len(seqs) == 0:
        return []

    fams = parse_family_entry(family_value)
    n_chain = len(seqs)
    uniq_seqs = unique_in_order(seqs)
    n_uniq = len(uniq_seqs)

    if len(fams) == 0:
        return [None] * n_chain
    if len(fams) == n_chain:
        return fams
    if len(fams) == n_uniq:
        seq_to_family = {seq: fams[i] for i, seq in enumerate(uniq_seqs)}
        return [seq_to_family.get(seq) for seq in seqs]
    if len(fams) == 1:
        if n_uniq == 1:
            return [fams[0]] * n_chain
        return [None] * n_chain
    return [None] * n_chain


def summarize_family_list(families: List[Optional[str]]) -> Dict[str, Any]:
    clean = [f for f in families if normalize_missing_token(f) is not None]
    uniq = unique_in_order(clean)
    return {
        "n_unique": len(uniq),
        "is_known": int(len(clean) > 0),
        "joined": "|".join(uniq) if uniq else np.nan,
    }


def list_value_at(x: Any, idx: int) -> float:
    vals = ensure_list(x)
    if idx < 0 or idx >= len(vals):
        return np.nan
    return to_float(vals[idx])


def parse_numeric_list(x: Any) -> List[float]:
    vals = ensure_list(x)
    out: List[float] = []
    for v in vals:
        fv = to_float(v)
        if not np.isnan(fv):
            out.append(float(fv))
    return out


def clean_rna_sequence(seq: Any) -> str:
    s = normalize_missing_token(seq)
    if s is None:
        return ""
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^A-Za-z]", "", s).upper().replace("T", "U")
    return "".join(ch for ch in s if ch in {"A", "C", "G", "U", "N"})


def clean_protein_sequence(seq: Any) -> str:
    s = normalize_missing_token(seq)
    if s is None:
        return ""
    s = re.sub(r"[^A-Za-z]", "", s).upper()
    return s


def parse_sequence_list(x: Any, kind: str) -> List[str]:
    vals = ensure_list(x)
    if not vals and x is not None and not isinstance(x, list):
        vals = [x]
    if kind == "rna":
        return [clean_rna_sequence(v) for v in vals if clean_rna_sequence(v) != ""]
    return [clean_protein_sequence(v) for v in vals if clean_protein_sequence(v) != ""]


def total_length_from_fields(length_field: Any, sequence_field: Any, kind: str) -> Tuple[List[float], float]:
    lengths = parse_numeric_list(length_field)
    if lengths:
        return lengths, float(np.sum(lengths))
    seqs = parse_sequence_list(sequence_field, kind=kind)
    lengths = [float(len(s)) for s in seqs if len(s) > 0]
    return lengths, float(np.sum(lengths)) if lengths else np.nan


def safe_div(a: Any, b: Any) -> float:
    a = to_float(a)
    b = to_float(b)
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return a / b


def compute_chain_length_stats(lengths: List[float], prefix: str) -> Dict[str, float]:
    arr = np.array([x for x in lengths if not np.isnan(x)], dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_total_length": np.nan,
            f"{prefix}_mean_chain_length": np.nan,
            f"{prefix}_max_chain_length": np.nan,
            f"{prefix}_min_chain_length": np.nan,
            f"{prefix}_std_chain_length": np.nan,
            f"{prefix}_chain_length_cv": np.nan,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr)) if arr.size > 1 else 0.0
    return {
        f"{prefix}_total_length": float(np.sum(arr)),
        f"{prefix}_mean_chain_length": mean,
        f"{prefix}_max_chain_length": float(np.max(arr)),
        f"{prefix}_min_chain_length": float(np.min(arr)),
        f"{prefix}_std_chain_length": std,
        f"{prefix}_chain_length_cv": (std / mean) if mean > 0 else np.nan,
    }


def shannon_entropy_kmers(seq: str, k: int = 1) -> float:
    seq = seq or ""
    if len(seq) < k or k <= 0:
        return np.nan
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    if not kmers:
        return np.nan
    counts = {}
    for kmer in kmers:
        counts[kmer] = counts.get(kmer, 0) + 1
    probs = np.array(list(counts.values()), dtype=float) / len(kmers)
    return float(-(probs * np.log2(probs)).sum())


def sequence_complexity(seq: str) -> float:
    seq = seq or ""
    if len(seq) == 0:
        return np.nan
    return len(set(seq)) / len(seq)


def compute_rna_sequence_features(seqs: List[str], prefix: str) -> Dict[str, float]:
    seqs = [s for s in seqs if s]
    concat = "".join(seqs)
    if concat == "":
        return {
            f"{prefix}_gc_content": np.nan,
            f"{prefix}_au_content": np.nan,
            f"{prefix}_gc_skew": np.nan,
            f"{prefix}_au_skew": np.nan,
            f"{prefix}_sequence_complexity": np.nan,
            f"{prefix}_shannon_entropy_k1": np.nan,
            f"{prefix}_shannon_entropy_k2": np.nan,
        }
    counts = {b: concat.count(b) for b in "ACGUN"}
    non_n = max(counts["A"] + counts["C"] + counts["G"] + counts["U"], 1)
    gc = counts["G"] + counts["C"]
    au = counts["A"] + counts["U"]
    return {
        f"{prefix}_gc_content": gc / non_n,
        f"{prefix}_au_content": au / non_n,
        f"{prefix}_gc_skew": safe_div(counts["G"] - counts["C"], counts["G"] + counts["C"]),
        f"{prefix}_au_skew": safe_div(counts["A"] - counts["U"], counts["A"] + counts["U"]),
        f"{prefix}_sequence_complexity": sequence_complexity(concat),
        f"{prefix}_shannon_entropy_k1": shannon_entropy_kmers(concat, 1),
        f"{prefix}_shannon_entropy_k2": shannon_entropy_kmers(concat, 2),
    }


def parse_loop_count(x: Any) -> float:
    v = safe_literal_eval(x)
    return float(len(v)) if isinstance(v, list) else np.nan


# =============================================================================
# Generic exp_Solution / exp_Sample_Conc parser
# =============================================================================
_COMPONENT_AFTER = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|uM|µM|nM|%)\s*(?:\([wv]/[wv]\)\s*)?(?P<comp>[A-Za-z0-9\-\+\./ ]{2,80})",
    flags=re.IGNORECASE,
)

_COMPONENT_BEFORE = re.compile(
    r"(?P<comp>(?:PEG|Polyethylene glycol|Tris(?:-HCl|-Cl)?|Bis-Tris|HEPES(?:-NaOH)?|MES(?:/imidazole)?|citric acid|sodium acetate|potassium phosphate|sodium phosphate|ammonium sulfate|lithium sulfate|magnesium chloride(?: hexahydrate)?|sodium chloride|NaCl|KCl|LiCl|glycerol|ethylene glycol|MPD|Tacsimate|Bicine|succinic acid|sodium formate|ammonium acetate|ammonium phosphate dibasic|phosphate-citrate)[A-Za-z0-9\-\+\./ ]{0,40})\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|uM|µM|nM|%)",
    flags=re.IGNORECASE,
)

_PH_RE = re.compile(r"\bpH\s*=?\s*(\d+(?:\.\d+)?)", flags=re.IGNORECASE)
_TEMP_RE = re.compile(r"(?:temperature\s*)?(\d+(?:\.\d+)?)\s*(K|C)\b", flags=re.IGNORECASE)


def split_solution_text(text: str) -> List[str]:
    s = str(text).replace("\n", ", ")
    s = re.sub(r"\band\b", ",", s, flags=re.IGNORECASE)
    return [p.strip(" .;:") for p in re.split(r",|;", s) if p.strip(" .;:")]


def parse_solution_like(text: Any, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return out
    s = str(text)
    parts = split_solution_text(s)
    ph_all = _PH_RE.findall(s)
    if ph_all:
        out[f"{prefix}_ph_global"] = to_float(ph_all[0])
    temp_match = _TEMP_RE.search(s)
    if temp_match:
        temp_val = float(temp_match.group(1))
        temp_unit = temp_match.group(2).upper()
        out[f"{prefix}_temp_k"] = temp_val + 273.15 if temp_unit == "C" else temp_val
    for part in parts:
        local_ph = _PH_RE.findall(part)
        m = _COMPONENT_AFTER.search(part)
        if m:
            val = float(m.group("val"))
            unit = m.group("unit")
            comp = normalize_name(m.group("comp"))
            comp = re.sub(r"_(ph|temperature).*", "", comp).strip("_") or "unknown"
            if unit == "%":
                out[f"{prefix}_pct_{comp}"] = val
            else:
                out[f"{prefix}_conc_{comp}_m"] = unit_to_molar(val, unit)
            if local_ph:
                out[f"{prefix}_ph_{comp}"] = to_float(local_ph[0])
            continue
        m2 = _COMPONENT_BEFORE.search(part)
        if m2:
            comp = normalize_name(m2.group("comp"))
            val = float(m2.group("val"))
            unit = m2.group("unit")
            if unit == "%":
                out[f"{prefix}_pct_{comp}"] = val
            else:
                out[f"{prefix}_conc_{comp}_m"] = unit_to_molar(val, unit)
            if local_ph:
                out[f"{prefix}_ph_{comp}"] = to_float(local_ph[0])
    return out


# =============================================================================
# Buffer / PEG / ionic strength parsing
# =============================================================================
PEG_MW_PATTERN = re.compile(
    r"(?:PEG|PEMME|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]{2,5})(?=[^\d]|$)",
    flags=re.I,
)
PEG_K_PATTERN = re.compile(r"(?:PEG|polyethylene\s+glycol)[-_\s]?([0-9]{1,2})\s*[Kk](?=\b|[^0-9])", flags=re.I)

SALTS = {
    "sodium chloride": (1, -1), "potassium chloride": (1, -1), "lithium chloride": (1, -1),
    "magnesium chloride": (2, -1), "manganese chloride": (2, -1), "calcium chloride": (2, -1),
    "ammonium sulfate": (1, -2), "lithium sulfate": (1, -2), "sodium sulfate": (1, -2),
    "potassium sulfate": (1, -2), "sodium acetate": (1, -1), "potassium acetate": (1, -1),
    "ammonium acetate": (1, -1), "magnesium acetate": (2, -1), "calcium acetate": (2, -1),
    "sodium formate": (1, -1), "magnesium formate": (2, -1), "sodium citrate": (1, -3),
    "trisodium citrate": (1, -3), "ammonium citrate": (1, -3), "ammonium tartrate": (1, -2),
    "potassium sodium tartrate": (1, -2), "sodium phosphate": (1, -2), "potassium phosphate": (1, -2),
    "ammonium phosphate": (1, -2), "sodium carbonate": (1, -2), "potassium carbonate": (1, -2),
    "ammonium carbonate": (1, -2), "disodium succinate": (1, -2), "sodium succinate": (1, -2),
    "potassium bromide": (1, -1), "sodium bromide": (1, -1), "potassium iodide": (1, -1),
    "sodium iodide": (1, -1), "ammonium fluoride": (1, -1),
}

ALIASES = {
    "nacl": "sodium chloride", "na cl": "sodium chloride", "kcl": "potassium chloride", "k cl": "potassium chloride",
    "licl": "lithium chloride", "li cl": "lithium chloride", "mgcl2": "magnesium chloride", "mg cl2": "magnesium chloride",
    "mncl2": "manganese chloride", "mn cl2": "manganese chloride", "cacl2": "calcium chloride", "ca cl2": "calcium chloride",
    "(nh4)2so4": "ammonium sulfate", "nh4 sulfate": "ammonium sulfate", "ammonium sulfate": "ammonium sulfate",
    "liso4": "lithium sulfate", "li2so4": "lithium sulfate", "lis04": "lithium sulfate", "na2so4": "sodium sulfate",
    "k2so4": "potassium sulfate", "na acetate": "sodium acetate", "naacetate": "sodium acetate", "k acetate": "potassium acetate",
    "ammonium acetate": "ammonium acetate", "magnesium acetate tetrahydrate": "magnesium acetate",
    "na formate": "sodium formate", "mg formate": "magnesium formate", "mg-formate": "magnesium formate",
    "na citrate": "sodium citrate", "tri-sodium citrate": "trisodium citrate", "citrate": "sodium citrate",
    "ammonium tartrate": "ammonium tartrate", "nh4/tartrate": "ammonium tartrate", "nh4 tartrate": "ammonium tartrate",
    "potassium sodium tartrate": "potassium sodium tartrate", "potasium sodium tartrate": "potassium sodium tartrate",
    "sodium potassium tartrate": "potassium sodium tartrate", "sodium phosphate": "sodium phosphate",
    "na h2po4": "sodium phosphate", "na2 hpo4": "sodium phosphate", "k phosphate": "potassium phosphate",
    "na2co3": "sodium carbonate", "k2co3": "potassium carbonate", "na2 succinate": "disodium succinate",
    "kbr": "potassium bromide", "nabr": "sodium bromide", "ki": "potassium iodide", "nai": "sodium iodide", "naf": "ammonium fluoride",
}

BUFFERS = [
    "tris", "tris-hcl", "tris(hydroxymethyl)aminomethane", "hepes", "mes", "bicine", "imidazole",
    "cacodylate", "cacodyl", "sodium cacodylate", "sodium cacodylate trihydrate",
    "bis-tris", "bis-tris propane", "bis-trispropane", "citric", "citrate", "phosphate",
    "sodium phosphate", "potassium phosphate", "acetate", "formate", "tartrate", "pH",
    "mib", "ches", "ada", "succinate"
]
HYDRATE_PATTERN = re.compile(r"\b(mono|di|tri|tetra|penta|hexa)hydrate\b", flags=re.I)


def normalize_condition_name(raw: str) -> Optional[str]:
    r = raw.lower().strip()
    r = HYDRATE_PATTERN.sub("", r).strip()
    r = re.sub(r"\bph\s*\d+\.?\d*\b", "", r, flags=re.I).strip()
    r = re.sub(r"\s+", " ", r).strip()
    for key, val in ALIASES.items():
        if key in r:
            return val
    for salt in SALTS.keys():
        if salt in r:
            return salt
    return None


def parse_concentration_m(text: str) -> Optional[float]:
    text = text.lower()
    m = re.search(r"(\d+\.?\d*)\s*(m|mm)\b", text)
    if not m:
        return None
    val, unit = m.groups()
    val = float(val)
    return val if unit == "m" else val / 1000.0


def ionic_strength_from_condition(condition_text: str) -> Optional[float]:
    total_I = 0.0
    pattern_with_space = re.compile(r"(\d+\.?\d*)\s+(m|mm)\s+([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)", flags=re.I)
    pattern_no_space = re.compile(r"(\d+\.?\d*)(m|mm)([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)", flags=re.I)
    for pattern in (pattern_with_space, pattern_no_space):
        for match in pattern.finditer(condition_text):
            val_str, unit, tail = match.groups()
            if "peg" in tail.lower():
                continue
            conc = float(val_str) / 1000.0 if unit.lower() == "mm" else float(val_str)
            salt_name = normalize_condition_name(tail.strip())
            if salt_name is None:
                continue
            z_plus, z_minus = SALTS[salt_name]
            total_I += 0.5 * conc * (z_plus ** 2 + z_minus ** 2)
    if total_I == 0.0:
        for part in re.split(r"[;,]", condition_text.lower()):
            conc = parse_concentration_m(part)
            if conc is None:
                continue
            salt_name = normalize_condition_name(part)
            if salt_name is None:
                continue
            z_plus, z_minus = SALTS[salt_name]
            total_I += 0.5 * conc * (z_plus ** 2 + z_minus ** 2)
    return total_I if total_I > 0.0 else np.nan


def ionic_strength_category(ionic_strength: Any) -> str:
    v = to_float(ionic_strength)
    if np.isnan(v) or v == 0:
        return "Unknown"
    if v >= 1.0:
        return "ge_1_0_m"
    if v >= 0.5:
        return "0_5_to_1_0_m"
    return "0_to_0_5_m"


def peg_category_from_percent(percent: Any) -> str:
    p = to_float(percent)
    if np.isnan(p):
        return "no_peg"
    if p >= 30:
        return "peg_30_50_pct"
    if p >= 20:
        return "peg_20_30_pct"
    if p >= 10:
        return "peg_10_20_pct"
    if p > 0:
        return "peg_0_10_pct"
    return "no_peg"


def peg_mw_category(mw: Any) -> str:
    m = to_float(mw)
    if np.isnan(m):
        return "unknown"
    if m < 1000:
        return "peg_0_1k"
    if m < 2000:
        return "peg_1k_2k"
    if m < 10000:
        return "peg_2k_10k"
    return "peg_10k_plus"


def find_peg_percent(text: str) -> Optional[float]:
    peg_percent_patterns = [
        re.compile(r"(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{1,2})\s*[Kk]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:\(w/v\))?\s*(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{2,5})", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*Polyethylene\s+glycol\s+([0-9]{2,5})", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:.*?\s)?(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)", flags=re.I),
        re.compile(r"PEG.*?([0-9]{1,3}(?:\.[0-9]+)?)\s*%", flags=re.I),
    ]
    for pattern in peg_percent_patterns:
        m = pattern.search(text)
        if not m:
            continue
        for g in [x for x in m.groups() if x is not None]:
            try:
                val = float(g.replace(",", ""))
                if 0 <= val <= 100:
                    return val
            except Exception:
                pass
    return np.nan


def find_peg_mw(text: str) -> Optional[float]:
    m = PEG_MW_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            pass
    m2 = PEG_K_PATTERN.search(text)
    if m2:
        try:
            return float(m2.group(1)) * 1000.0
        except Exception:
            pass
    m3 = re.search(r"Polyethylene\s+glycol\s+([0-9]{2,5})", text, flags=re.I)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            pass
    return np.nan


def find_buffer_keyword(text: str) -> Optional[str]:
    text_low = text.lower()
    for buf in BUFFERS:
        if buf.lower() in text_low:
            return normalize_name(buf)
    return None


def parse_buffer_features(text: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ionic_strength": np.nan,
        "ionic_strength_category": "Unknown",
        "peg_percent": np.nan,
        "peg_mw": np.nan,
        "peg_concentration_category": "no_peg",
        "peg_mw_category": "unknown",
        "buffer_m": np.nan,
        "buffer_category": "unknown",
        "buffer_name": np.nan,
        "has_peg": 0,
    }
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return out
    s = str(text).replace("\n", " ").strip()
    ionic_strength = ionic_strength_from_condition(s)
    out["ionic_strength"] = ionic_strength
    out["ionic_strength_category"] = ionic_strength_category(ionic_strength)
    peg_pct = find_peg_percent(s)
    peg_mw = find_peg_mw(s)
    out["peg_percent"] = peg_pct
    out["peg_mw"] = peg_mw
    out["peg_concentration_category"] = peg_category_from_percent(peg_pct)
    out["peg_mw_category"] = peg_mw_category(peg_mw)
    out["has_peg"] = int(not (np.isnan(to_float(peg_pct)) and np.isnan(to_float(peg_mw))))

    pattern_molarity_first = re.compile(r"([0-9]*\.?[0-9]+)\s*(mM|M)\s*([A-Za-z0-9\-/\s\(\)\.]*)", flags=re.I)
    pattern_buffer_first = re.compile(r"([A-Za-z0-9\-/\s\(\)\.]+?)\s*([0-9]*\.?[0-9]+)\s*(mM|M)", flags=re.I)

    def normalize_conc(val_str: str, unit: str) -> float:
        v = float(val_str)
        return v / 1000.0 if unit.lower() == "mm" else v

    buffer_matches = []
    for match in pattern_molarity_first.finditer(s):
        val_str, unit, tail = match.groups()
        buf = find_buffer_keyword(tail)
        if buf:
            buffer_matches.append((normalize_conc(val_str, unit), buf))
    for match in pattern_buffer_first.finditer(s):
        buf_name, val_str, unit = match.groups()
        buf = find_buffer_keyword(buf_name)
        if buf:
            buffer_matches.append((normalize_conc(val_str, unit), buf))

    if buffer_matches:
        buffer_M, buffer_name = max(buffer_matches, key=lambda x: x[0])
        out["buffer_m"] = buffer_M
        out["buffer_name"] = buffer_name
        if buffer_M < 0.1:
            out["buffer_category"] = "0_0_1_m"
        elif buffer_M < 0.5:
            out["buffer_category"] = "0_1_0_5_m"
        elif buffer_M < 5:
            out["buffer_category"] = "0_5_5_0_m"
        else:
            out["buffer_category"] = "gt_5_0_m"
    return out


# =============================================================================
# Ions / ligands
# =============================================================================
def safe_int_flag(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def summarize_support_pairs(x: Any, kind: str) -> Dict[str, Any]:
    pairs = parse_pairs_list(x)
    n_listed = len(pairs)
    supported = 0
    unsupported = 0
    unknown = 0
    names: Set[str] = set()

    for flag_raw, code in pairs:
        names.add(normalize_name(code))
        flag = safe_int_flag(flag_raw)
        if flag == 1:
            supported += 1
        elif flag == 0:
            unsupported += 1
        else:
            unknown += 1

    frac_supported = supported / n_listed if n_listed > 0 else np.nan

    return {
        f"n_{kind}_listed": n_listed,
        f"n_supported_{kind}": supported,
        f"n_unsupported_{kind}": unsupported,
        f"n_unknown_{kind}": unknown,
        f"has_any_{kind[:-1]}": int(n_listed > 0),
        f"has_any_unsupported_{kind[:-1]}": int(unsupported > 0),
        f"has_any_supported_{kind[:-1]}": int(supported > 0),
        f"frac_supported_{kind}": frac_supported,
        f"n_unique_{kind}": len(names),
    }


def collect_dynamic_codes(series: pd.Series, prefix: str) -> List[str]:
    names: Set[str] = set()
    for x in series.dropna():
        for _, code in parse_pairs_list(x):
            names.add(f"{prefix}_{normalize_name(code)}")
    return sorted(names)


def fill_support_code_columns(series: pd.Series, colnames: List[str], prefix: str) -> pd.DataFrame:
    code_to_col = {c[len(prefix) + 1:]: c for c in colnames}
    rows = []
    for x in series:
        row = {c: 0 for c in colnames}
        for flag_raw, code in parse_pairs_list(x):
            key = normalize_name(code)
            col = code_to_col.get(key)
            flag = safe_int_flag(flag_raw)
            if col is not None and flag == 0:
                row[col] = 1
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# New RBP pair-metric parsing
# =============================================================================
def parse_metric_pair_list(x: Any) -> Dict[str, float]:
    """
    Example input:
      '[{"A:B": 1.806}, {"C:D": 2.1}]'
    Returns:
      {"A:B": 1.806, "C:D": 2.1}
    """
    v = safe_literal_eval(x)
    out: Dict[str, float] = {}
    if not isinstance(v, list):
        return out
    for item in v:
        if isinstance(item, dict):
            for k, val in item.items():
                fv = to_float(val)
                if not np.isnan(fv):
                    out[str(k)] = float(fv)
    return out


def summarize_rbp_metric_dict(metric_name: str, x: Any) -> Dict[str, Any]:
    d = parse_metric_pair_list(x)
    vals = np.array(list(d.values()), dtype=float) if d else np.array([], dtype=float)

    rec: Dict[str, Any] = {
        f"{metric_name}_n_pairs": int(len(d)),
        f"{metric_name}_has_any": int(len(d) > 0),
        f"{metric_name}_raw": repr(d) if d else np.nan,
    }

    if vals.size == 0:
        rec.update({
            f"{metric_name}_mean": np.nan,
            f"{metric_name}_median": np.nan,
            f"{metric_name}_max": np.nan,
            f"{metric_name}_min": np.nan,
            f"{metric_name}_std": np.nan,
            f"{metric_name}_best": np.nan,
            f"{metric_name}_worst": np.nan,
        })
        return rec

    rec.update({
        f"{metric_name}_mean": float(np.mean(vals)),
        f"{metric_name}_median": float(np.median(vals)),
        f"{metric_name}_max": float(np.max(vals)),
        f"{metric_name}_min": float(np.min(vals)),
        f"{metric_name}_std": float(np.std(vals)) if vals.size > 1 else 0.0,
    })

    # "best" / "worst" are metric-aware
    if metric_name.lower().endswith("irmsd"):
        rec[f"{metric_name}_best"] = float(np.min(vals))
        rec[f"{metric_name}_worst"] = float(np.max(vals))
    else:
        rec[f"{metric_name}_best"] = float(np.max(vals))
        rec[f"{metric_name}_worst"] = float(np.min(vals))

    return rec

# =============================================================================
# Interface parsing / feature engineering
# =============================================================================
AA_CLASSES = {
    "ARG": "positive", "LYS": "positive", "HIS": "positive",
    "PHE": "aromatic", "TYR": "aromatic", "TRP": "aromatic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar", "CYS": "polar",
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic", "ILE": "hydrophobic",
    "MET": "hydrophobic", "PRO": "hydrophobic", "GLY": "hydrophobic",
    "ASP": "acidic", "GLU": "acidic",
}

CONTACT_RE = re.compile(
    r'^\s*([A-Z]{3})\((\d+)\)-([A-Z0-9]{1,4})\((\d+)\):\s*([0-9]+(?:\.[0-9]+)?)\s*$'
)


def _summary_stats(values: List[float], prefix: str) -> Dict[str, float]:
    arr = np.array([float(v) for v in values if not np.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_sum": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }
    return {
        f"{prefix}_sum": float(np.sum(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_std": float(np.std(arr)) if arr.size > 1 else 0.0,
    }


def _normalize_chain_ids(chain_field: Any, expected_n: Optional[int] = None) -> List[str]:
    vals = ensure_list(chain_field)
    out: List[str] = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s != "":
            out.append(s)
    if expected_n is not None and len(out) == 1 and expected_n == 1:
        return out
    return out


def _normalize_lengths(length_field: Any, chain_ids: List[str], seq_field: Any, kind: str) -> Dict[str, float]:
    lengths = parse_numeric_list(length_field)
    if not lengths:
        seqs = parse_sequence_list(seq_field, kind=kind)
        lengths = [float(len(s)) for s in seqs if len(s) > 0]
    if len(chain_ids) == len(lengths):
        return {str(cid): float(l) for cid, l in zip(chain_ids, lengths)}
    if len(chain_ids) == 1 and len(lengths) == 1:
        return {str(chain_ids[0]): float(lengths[0])}
    return {str(cid): np.nan for cid in chain_ids}


def _normalize_pair_list(x: Any) -> List[Tuple[str, str]]:
    vals = safe_literal_eval(x)
    out: List[Tuple[str, str]] = []
    if isinstance(vals, list):
        for item in vals:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]).strip(), str(item[1]).strip()))
    return out


def _normalize_nested_by_pairs(x: Any, n_pairs: int) -> Tuple[List[List[str]], bool]:
    vals = safe_literal_eval(x)
    if vals is None:
        return [[] for _ in range(n_pairs)], False
    if not isinstance(vals, list):
        return [[] for _ in range(n_pairs)], False

    if n_pairs == 1:
        if len(vals) == 0:
            return [[]], False
        if all(isinstance(v, str) for v in vals):
            return [list(vals)], True
        if len(vals) == 1 and isinstance(vals[0], list):
            return [list(vals[0])], False

    if all(isinstance(v, list) for v in vals):
        return [list(v) for v in vals], False

    if all(isinstance(v, str) for v in vals):
        if n_pairs == 1:
            return [list(vals)], True
        return [[] for _ in range(n_pairs)], False

    out: List[List[str]] = []
    for v in vals:
        if isinstance(v, list):
            out.append(list(v))
        elif isinstance(v, str):
            out.append([v])
        else:
            out.append([])
    return out, False


def _normalize_scalar_or_list_by_pairs(x: Any, n_pairs: int) -> List[Any]:
    vals = safe_literal_eval(x)

    if n_pairs <= 0:
        return []

    if vals is None:
        return [None] * n_pairs

    if not isinstance(vals, list):
        if n_pairs == 1:
            return [vals]
        return [None] * n_pairs

    vals = list(vals)

    if len(vals) == n_pairs:
        return vals
    if len(vals) < n_pairs:
        return vals + [None] * (n_pairs - len(vals))
    return vals[:n_pairs]

def _parse_contact_item(item: str) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    s = str(item).strip()
    m = CONTACT_RE.match(s)
    if not m:
        return None
    aa, aa_pos, nt_name, nt_pos, dist = m.groups()
    nt_class = nt_name if nt_name in {"A", "C", "G", "U"} else "modified"
    aa_class = AA_CLASSES.get(aa, "other")
    return {
        "aa_name": aa,
        "aa_pos": int(aa_pos),
        "nt_name": nt_name,
        "nt_pos": int(nt_pos),
        "distance": float(dist),
        "nt_class": nt_class,
        "aa_class": aa_class,
    }


def compute_interface_features_for_subset(
    pair_list: List[Tuple[str, str]],
    hbond_list: List[Any],
    vdw_list: List[Any],
    motif_len_list: List[Any],
    contact_blocks: List[List[str]],
    protein_len_map: Dict[str, float],
    rna_len_map: Dict[str, float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_interface_pairs": len(pair_list),
        "n_pairs_with_contacts": 0,
        "frac_pairs_with_contacts": np.nan,
        "qc_n_pairs_failed_contact_parse": 0,
        "qc_frac_pairs_failed_contact_parse": np.nan,
        "qc_any_interface_parse_problem": 0,
    }

    if len(pair_list) == 0:
        empty_cols = [
            "total_hbond_proteinRNA", "total_vdwbond_proteinRNA", "total_contact_entries",
            "n_unique_protein_interface_residues", "n_unique_rna_interface_residues",
            "protein_interface_coverage", "rna_interface_coverage",
            "interface_contact_distance_mean", "interface_contact_distance_median",
            "interface_contact_distance_min", "interface_contact_distance_max",
            "interface_contact_distance_std",
            "frac_contacts_lt_2_5A", "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
            "interface_hbond_per_contact", "interface_vdw_per_contact", "interface_hbond_vdw_ratio",
            "interface_rna_motif_length_sum", "interface_rna_motif_length_mean",
            "interface_rna_motif_length_max", "interface_rna_motif_coverage_of_total_rna",
            "interface_frac_contacts_to_A", "interface_frac_contacts_to_C",
            "interface_frac_contacts_to_G", "interface_frac_contacts_to_U",
            "interface_frac_contacts_to_modified",
            "interface_frac_positive_contacts", "interface_frac_aromatic_contacts",
            "interface_frac_polar_contacts", "interface_frac_hydrophobic_contacts",
            "interface_frac_acidic_contacts", "interface_frac_other_contacts",
        ]
        for c in empty_cols:
            out[c] = np.nan
        return out

    pair_hbonds: List[float] = []
    pair_vdws: List[float] = []
    pair_contact_counts: List[float] = []
    pair_motif_lengths: List[float] = []

    global_protein_residues: Set[Tuple[str, int]] = set()
    global_rna_residues: Set[Tuple[str, int]] = set()
    all_distances: List[float] = []

    nt_class_counts = {"A": 0, "C": 0, "G": 0, "U": 0, "modified": 0}
    aa_class_counts = {"positive": 0, "aromatic": 0, "polar": 0, "hydrophobic": 0, "acidic": 0, "other": 0}

    pairs_with_contacts = 0
    pairs_failed_contact_parse = 0

    for i, (p_chain, r_chain) in enumerate(pair_list):
        hbond = to_float(hbond_list[i]) if i < len(hbond_list) else np.nan
        vdw = to_float(vdw_list[i]) if i < len(vdw_list) else np.nan
        motif_len = to_float(motif_len_list[i]) if i < len(motif_len_list) else np.nan
        raw_contacts = contact_blocks[i] if i < len(contact_blocks) else []

        parsed_contacts = []
        for item in raw_contacts:
            parsed = _parse_contact_item(item)
            if parsed is not None:
                parsed_contacts.append(parsed)

        if len(raw_contacts) > 0 and len(parsed_contacts) == 0:
            pairs_failed_contact_parse += 1

        n_contacts = len(parsed_contacts)
        if n_contacts > 0:
            pairs_with_contacts += 1

        pair_hbonds.append(hbond)
        pair_vdws.append(vdw)
        pair_contact_counts.append(float(n_contacts))
        pair_motif_lengths.append(motif_len)

        for c in parsed_contacts:
            global_protein_residues.add((p_chain, c["aa_pos"]))
            global_rna_residues.add((r_chain, c["nt_pos"]))
            all_distances.append(c["distance"])
            nt_class_counts[c["nt_class"]] += 1
            aa_class_counts[c["aa_class"]] += 1

    out["n_pairs_with_contacts"] = pairs_with_contacts
    out["frac_pairs_with_contacts"] = safe_div(pairs_with_contacts, len(pair_list))
    out["qc_n_pairs_failed_contact_parse"] = pairs_failed_contact_parse
    out["qc_frac_pairs_failed_contact_parse"] = safe_div(pairs_failed_contact_parse, len(pair_list))
    out["qc_any_interface_parse_problem"] = int(pairs_failed_contact_parse > 0)

    out["total_hbond_proteinRNA"] = np.nansum(pair_hbonds) if pair_hbonds else np.nan
    out["total_vdwbond_proteinRNA"] = np.nansum(pair_vdws) if pair_vdws else np.nan
    out["total_contact_entries"] = np.nansum(pair_contact_counts) if pair_contact_counts else np.nan

    total_prot_len = np.nansum(list(protein_len_map.values())) if protein_len_map else np.nan
    total_rna_len = np.nansum(list(rna_len_map.values())) if rna_len_map else np.nan

    out["n_unique_protein_interface_residues"] = len(global_protein_residues)
    out["n_unique_rna_interface_residues"] = len(global_rna_residues)
    out["protein_interface_coverage"] = safe_div(len(global_protein_residues), total_prot_len)
    out["rna_interface_coverage"] = safe_div(len(global_rna_residues), total_rna_len)

    if len(all_distances) > 0:
        arr = np.array(all_distances, dtype=float)
        out["interface_contact_distance_mean"] = float(np.mean(arr))
        out["interface_contact_distance_median"] = float(np.median(arr))
        out["interface_contact_distance_min"] = float(np.min(arr))
        out["interface_contact_distance_max"] = float(np.max(arr))
        out["interface_contact_distance_std"] = float(np.std(arr)) if arr.size > 1 else 0.0
        out["frac_contacts_lt_2_5A"] = float(np.mean(arr < 2.5))
        out["frac_contacts_lt_3_0A"] = float(np.mean(arr < 3.0))
        out["frac_contacts_lt_3_5A"] = float(np.mean(arr < 3.5))
        out["frac_contacts_lt_4_0A"] = float(np.mean(arr < 4.0))
    else:
        for col in [
            "interface_contact_distance_mean", "interface_contact_distance_median",
            "interface_contact_distance_min", "interface_contact_distance_max",
            "interface_contact_distance_std",
            "frac_contacts_lt_2_5A", "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
        ]:
            out[col] = np.nan

    total_contacts = out["total_contact_entries"]
    total_h = out["total_hbond_proteinRNA"]
    total_v = out["total_vdwbond_proteinRNA"]

    out["interface_hbond_per_contact"] = safe_div(total_h, total_contacts)
    out["interface_vdw_per_contact"] = safe_div(total_v, total_contacts)
    out["interface_hbond_vdw_ratio"] = safe_div(total_h, total_v)

    motif_arr = np.array([v for v in pair_motif_lengths if not np.isnan(v)], dtype=float)
    out["interface_rna_motif_length_sum"] = float(np.sum(motif_arr)) if motif_arr.size else np.nan
    out["interface_rna_motif_length_mean"] = float(np.mean(motif_arr)) if motif_arr.size else np.nan
    out["interface_rna_motif_length_max"] = float(np.max(motif_arr)) if motif_arr.size else np.nan
    out["interface_rna_motif_coverage_of_total_rna"] = safe_div(out["interface_rna_motif_length_sum"], total_rna_len)

    total_nt = sum(nt_class_counts.values())
    for nt in ["A", "C", "G", "U", "modified"]:
        out[f"interface_frac_contacts_to_{nt}"] = safe_div(nt_class_counts[nt], total_nt)

    total_aa = sum(aa_class_counts.values())
    for aa_cls in ["positive", "aromatic", "polar", "hydrophobic", "acidic", "other"]:
        out[f"interface_frac_{aa_cls}_contacts"] = safe_div(aa_class_counts[aa_cls], total_aa)

    return out


def compute_per_rna_chain_interface_features(base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    protein_chain_ids = _normalize_chain_ids(
        base.get("ProteinChainIDs_pred", base.get("ProteinChainIDs")),
        base.get("NumberProteins"),
    )
    rna_chain_ids = _normalize_chain_ids(
        base.get("RNAChainIDs_pred", base.get("RNAChainIDs")),
        base.get("NumberRNAs"),
    )

    protein_len_map = _normalize_lengths(
        base.get("ProteinLength_pred", base.get("ProteinLength")),
        protein_chain_ids,
        base.get("ProteinSequence_pred", base.get("ProteinSequence")),
        kind="protein",
    )
    rna_len_map = _normalize_lengths(
        base.get("RNALength_pred", base.get("RNALength")),
        rna_chain_ids,
        base.get("RNASequence_pred", base.get("RNASequence")),
        kind="rna",
    )
    pair_list = _normalize_pair_list(base.get("ChainIDpairList_proteinRNA"))
    n_pairs = len(pair_list)

    hbond_list = _normalize_scalar_or_list_by_pairs(base.get("Hbond_proteinRNA"), n_pairs)
    vdw_list = _normalize_scalar_or_list_by_pairs(base.get("vdWbond_proteinRNA"), n_pairs)
    motif_len_list = _normalize_scalar_or_list_by_pairs(base.get("RNAMotifLength"), n_pairs)
    contact_blocks, _ = _normalize_nested_by_pairs(base.get("ContactList"), n_pairs)

    # extra safety in case malformed nested parser still returns wrong length
    if len(contact_blocks) < n_pairs:
        contact_blocks = contact_blocks + [[] for _ in range(n_pairs - len(contact_blocks))]
    elif len(contact_blocks) > n_pairs:
        contact_blocks = contact_blocks[:n_pairs]

    out: Dict[str, Dict[str, Any]] = {}

    for r_chain in rna_chain_ids:
        idx = [i for i, (_, rc) in enumerate(pair_list) if rc == r_chain]
        sub_pairs = [pair_list[i] for i in idx]
        sub_h = [hbond_list[i] if i < len(hbond_list) else None for i in idx]
        sub_v = [vdw_list[i] if i < len(vdw_list) else None for i in idx]
        sub_motif = [motif_len_list[i] if i < len(motif_len_list) else None for i in idx]
        sub_contacts = [contact_blocks[i] if i < len(contact_blocks) else [] for i in idx]

        rec = compute_interface_features_for_subset(
            pair_list=sub_pairs,
            hbond_list=sub_h,
            vdw_list=sub_v,
            motif_len_list=sub_motif,
            contact_blocks=sub_contacts,
            protein_len_map=protein_len_map,
            rna_len_map={r_chain: rna_len_map.get(r_chain, np.nan)},
        )
        out[str(r_chain)] = rec

    return out

def compute_global_interface_features(base: Dict[str, Any]) -> Dict[str, Any]:
    protein_chain_ids = _normalize_chain_ids(
        base.get("ProteinChainIDs_pred", base.get("ProteinChainIDs")),
        base.get("NumberProteins"),
    )
    rna_chain_ids = _normalize_chain_ids(
        base.get("RNAChainIDs_pred", base.get("RNAChainIDs")),
        base.get("NumberRNAs"),
    )
    
    protein_len_map = _normalize_lengths(
        base.get("ProteinLength_pred", base.get("ProteinLength")),
        protein_chain_ids,
        base.get("ProteinSequence_pred", base.get("ProteinSequence")),
        kind="protein",
    )
    rna_len_map = _normalize_lengths(
        base.get("RNALength_pred", base.get("RNALength")),
        rna_chain_ids,
        base.get("RNASequence_pred", base.get("RNASequence")),
        kind="rna",
    )
    pair_list = _normalize_pair_list(base.get("ChainIDpairList_proteinRNA"))
    n_pairs = len(pair_list)

    hbond_list = _normalize_scalar_or_list_by_pairs(base.get("Hbond_proteinRNA"), n_pairs)
    vdw_list = _normalize_scalar_or_list_by_pairs(base.get("vdWbond_proteinRNA"), n_pairs)
    motif_len_list = _normalize_scalar_or_list_by_pairs(base.get("RNAMotifLength"), n_pairs)
    contact_blocks, contactlist_was_flat_single_pair = _normalize_nested_by_pairs(base.get("ContactList"), n_pairs)

    rec = compute_interface_features_for_subset(
        pair_list=pair_list,
        hbond_list=hbond_list,
        vdw_list=vdw_list,
        motif_len_list=motif_len_list,
        contact_blocks=contact_blocks,
        protein_len_map=protein_len_map,
        rna_len_map=rna_len_map,
    )
    rec["qc_contactlist_was_flat_single_pair"] = int(contactlist_was_flat_single_pair)
    rec["qc_interface_lengths_match"] = int(
        len(hbond_list) == n_pairs and
        len(vdw_list) == n_pairs and
        len(motif_len_list) == n_pairs and
        len(contact_blocks) == n_pairs
    )
    return rec


def add_global_interface_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [compute_global_interface_features(row.to_dict()) for _, row in df.iterrows()]
    rows_df = pd.DataFrame(rows, index=df.index)

    rename_map = {
        "n_interface_pairs": "n_protein_rna_interface_pairs",
        "n_unique_protein_interface_residues": "n_unique_protein_interface_residues_global",
        "n_unique_rna_interface_residues": "n_unique_rna_interface_residues_global",
        "protein_interface_coverage": "protein_interface_coverage_global",
        "rna_interface_coverage": "rna_interface_coverage_global",
    }
    rows_df = rows_df.rename(columns=rename_map)
    return pd.concat([df, rows_df], axis=1)

# =============================================================================
# Table loading
# =============================================================================
BASE_PRED_COLUMNS = [
    "exp_db_id",
    "ProteinSequence",
    "ProteinChainIDs",
    "ProteinLength",
    "NumberProteins",
    "RNASequence",
    "RNAChainIDs",
    "RNALength",
    "NumberRNAs",
    "RBP_PDB_Templates",
    "Protein_PDB_Templates",
    "af3_protein_pLDDT_avg",
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_ranking_score",
    "Complex_RMSD",
    "Complex_TM",
    "Complex_LDDT",
    "Complex_LDDT_allAtom",
    "iLDDT",
    "QS_global",
    "Dockq",
    "ICS",
    "RNAComplex_TM",
    "RNAmotif_score",
    "ProteinRNAInterfaceArea",
    "ProteinRNAInterfaceRatio",

    # per-chain / RNA
    "RNA_msa_size",
    "af3_rna_MSA",
    "af3_protein_pTM",
    "af3_rna_pTM",
    "af3_protein_ipTM",
    "af3_rna_ipTM",
    "RNA_TM_AllChains",
    "RNA_LDDT_AllChains",

    # new all-chain RNA metrics
    "RNA_Dockq_AllChains",
    "RNA_ICS_AllChains",
    "RNA_iLDDT_AllChains",
    "RNA_QS_global_AllChains",

    # new RBP pair metrics
    "RBP_irmsd",
    "RBP_DockQ",
    "RBP_fnat",
    "RBP_QS",

    # RNA intrinsic / structure
    "RNA_GlycosidicBond",
    "RNA_SugarPucker",
    "RNA_GammaAngle",
    "RNA_Stems",
    "RNA_HairpinLoops",
    "RNA_InternalLoops",
    "RNA_MultibranchLoops",
    "RNA_DanglingEnds",
    "RNA_Pseudoknots",
    "RNA_isDuplex",
    "RNAMotif_isDuplex",
    "RNAMotif_inExp",
    "RNA_ElectrostaticPotential",
    "RNA_GC_Content",
    "RNA_SequenceComplexity",
    "RNA_Shannon_Entropy_K1",
    "RNA_Shannon_Entropy_K2",
    "RNA_GC_Skew",
    "RNA_AU_Skew",
    "RNA_AU_Content",
    "RNA_MFE_Value",
    "RNA_MFE_Structure",
    "RNA_Ensemble_Diversity",
    "RNA_Centroid_Energy",
    "Protein_msa_size",
    "AAmatch_score",
    "RNAmatch_score",

    # raw interface fields for global + per-chain predicted interface summaries
    "AAMotif",
    "RNAMotif",
    "RNAMotifLength",
    "ContactList",
    "ChainIDpairList_proteinRNA",
    "Hbond_proteinRNA",
    "vdWbond_proteinRNA",
    "rna_prot_interface_atom_ids",
    "interface_rna_atom_ids",
]

BASE_EXP_COLUMNS = [
    "PDBId",
    "ProteinSequence",
    "ProteinChainIDs",
    "ProteinLength",
    "NumberProteins",
    "Binding_affinity_kd",
    "RNA_ElectrostaticPotential",
    "Ions",
    "Ligands",
    "exp_Solution",
    "exp_Sample_Conc",
    "Experiment",
    "XRayResolution",
    "exp_pH",
    "exp_Temp",
    "RNASequence",
    "ProteinSequence",
    "RNAFamily",
    "ProteinFamily",
    "ProteinRNAInterfaceArea",
    "ProteinRNAInterfaceRatio",
]


def load_table_columns(conn: sqlite3.Connection, table: str, wanted: List[str]) -> pd.DataFrame:
    existing = pd.read_sql_query(f"PRAGMA table_info({table})", conn)["name"].tolist()
    keep = [c for c in wanted if c in existing]
    if not keep:
        raise ValueError(f"No requested columns found in table '{table}'.")
    df = pd.read_sql_query(f"SELECT {', '.join(keep)} FROM {table}", conn)
    return df.loc[:, ~df.columns.duplicated()].copy()


def collect_dynamic_names(series: pd.Series, parser) -> List[str]:
    names: Set[str] = set()
    for x in series.dropna():
        names.update(parser(x).keys())
    return sorted(names)


def add_experiment_flags(df: pd.DataFrame) -> pd.DataFrame:
    exp = df.get("Experiment", pd.Series(index=df.index, dtype=object)).astype(str)
    df["is_xray"] = exp.str.contains("X-RAY", case=False, na=False).astype(int)
    df["is_em"] = exp.str.contains("ELECTRON", case=False, na=False).astype(int)
    df["is_nmr"] = exp.str.contains("NMR", case=False, na=False).astype(int)
    return df


def add_solution_features(df: pd.DataFrame) -> pd.DataFrame:
    sol_cols = collect_dynamic_names(df["exp_Solution"], lambda x: parse_solution_like(x, "sol")) if "exp_Solution" in df.columns else []
    sample_cols = collect_dynamic_names(df["exp_Sample_Conc"], lambda x: parse_solution_like(x, "sample")) if "exp_Sample_Conc" in df.columns else []

    if sol_cols:
        sol_rows = [parse_solution_like(x, "sol") for x in df["exp_Solution"]]
        df = pd.concat([df, pd.DataFrame(sol_rows).reindex(columns=sol_cols)], axis=1)

    if sample_cols:
        sample_rows = [parse_solution_like(x, "sample") for x in df["exp_Sample_Conc"]]
        df = pd.concat([df, pd.DataFrame(sample_rows).reindex(columns=sample_cols)], axis=1)

    return df


def add_buffer_peg_features(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_Solution" in df.columns:
        df = pd.concat([df, pd.DataFrame([parse_buffer_features(x) for x in df["exp_Solution"]])], axis=1)
    return df


def add_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    comp_rows = []
    for _, row in df.iterrows():
        prot_lengths, _ = total_length_from_fields(row.get("ProteinLength_pred"), row.get("ProteinSequence_pred"), kind="protein")
        rna_lengths, _ = total_length_from_fields(row.get("RNALength_pred"), row.get("RNASequence_pred"), kind="rna")
        rec = {}
        rec.update(compute_chain_length_stats(prot_lengths, "protein"))
        rec.update(compute_chain_length_stats(rna_lengths, "rna"))
        rec["total_complex_length"] = (
            (rec["protein_total_length"] if not np.isnan(rec["protein_total_length"]) else 0.0) +
            (rec["rna_total_length"] if not np.isnan(rec["rna_total_length"]) else 0.0)
        )
        rec["protein_residue_fraction"] = safe_div(rec["protein_total_length"], rec["total_complex_length"])
        rec["rna_residue_fraction"] = safe_div(rec["rna_total_length"], rec["total_complex_length"])
        rec["protein_to_rna_length_ratio"] = safe_div(rec["protein_total_length"], rec["rna_total_length"])
        rec["max_rna_fraction_of_total_rna"] = safe_div(rec["rna_max_chain_length"], rec["rna_total_length"])
        rec["min_rna_fraction_of_total_rna"] = safe_div(rec["rna_min_chain_length"], rec["rna_total_length"])
        rna_seqs = parse_sequence_list(row.get("RNASequence_pred"), kind="rna")
        rec.update(compute_rna_sequence_features(rna_seqs, "agg_rna"))
        comp_rows.append(rec)
    return pd.concat([df, pd.DataFrame(comp_rows)], axis=1)


def add_interface_agreement_features(df: pd.DataFrame) -> pd.DataFrame:
    if "pred_ProteinRNAInterfaceArea" in df.columns and "exp_ProteinRNAInterfaceArea" in df.columns:
        df["ProteinRNAInterfaceArea_diff_exp_minus_pred"] = df["exp_ProteinRNAInterfaceArea"] - df["pred_ProteinRNAInterfaceArea"]
        df["ProteinRNAInterfaceArea_abs_diff"] = (df["exp_ProteinRNAInterfaceArea"] - df["pred_ProteinRNAInterfaceArea"]).abs()
        df["ProteinRNAInterfaceArea_rel_error"] = df["ProteinRNAInterfaceArea_abs_diff"] / df["exp_ProteinRNAInterfaceArea"].replace(0, np.nan)
        df["ProteinRNAInterfaceArea_pred_over_exp"] = df["pred_ProteinRNAInterfaceArea"] / df["exp_ProteinRNAInterfaceArea"].replace(0, np.nan)
    if "pred_ProteinRNAInterfaceRatio" in df.columns and "exp_ProteinRNAInterfaceRatio" in df.columns:
        df["ProteinRNAInterfaceRatio_diff_exp_minus_pred"] = df["exp_ProteinRNAInterfaceRatio"] - df["pred_ProteinRNAInterfaceRatio"]
        df["ProteinRNAInterfaceRatio_abs_diff"] = (df["exp_ProteinRNAInterfaceRatio"] - df["pred_ProteinRNAInterfaceRatio"]).abs()
        df["ProteinRNAInterfaceRatio_rel_error"] = df["ProteinRNAInterfaceRatio_abs_diff"] / df["exp_ProteinRNAInterfaceRatio"].replace(0, np.nan)
        df["ProteinRNAInterfaceRatio_pred_over_exp"] = df["pred_ProteinRNAInterfaceRatio"] / df["exp_ProteinRNAInterfaceRatio"].replace(0, np.nan)
    return df


def parse_features_and_metrics(merged: pd.DataFrame, db_tag: str) -> pd.DataFrame:
    df = merged.copy()
    df["pdb_id"] = df["exp_db_id"].astype(str)
    df["source_db"] = db_tag
    df["is_post_cutoff"] = infer_post_cutoff_from_db_tag(db_tag)

    for c in ["NumberProteins", "NumberRNAs"]:
        if f"{c}_pred" in df.columns:
            df[c] = pd.to_numeric(df[f"{c}_pred"], errors="coerce").astype("Int64")
        elif f"{c}_exp" in df.columns:
            df[c] = pd.to_numeric(df[f"{c}_exp"], errors="coerce").astype("Int64")

    if "RBP_PDB_Templates" in df.columns:
        df["n_rbp_templates"] = df["RBP_PDB_Templates"].apply(count_pdb_ids)
        df["id_is_rbp_template"] = [id_in_template_list(pid, x) for pid, x in zip(df["exp_db_id"], df["RBP_PDB_Templates"])]
    else:
        df["n_rbp_templates"] = np.nan
        df["id_is_rbp_template"] = np.nan

    if "Protein_PDB_Templates" in df.columns:
        df["n_protein_templates"] = df["Protein_PDB_Templates"].apply(count_pdb_ids)
        df["id_is_protein_template"] = [id_in_template_list(pid, x) for pid, x in zip(df["exp_db_id"], df["Protein_PDB_Templates"])]
    else:
        df["n_protein_templates"] = np.nan
        df["id_is_protein_template"] = np.nan

    numeric_cols = [
        "af3_protein_pLDDT_avg", "af3_rna_pLDDT_avg", "af3_global_pae_avg", "af3_chain_pair_pae_min",
        "af3_ranking_score", "Complex_TM", "Complex_LDDT", "Complex_LDDT_allAtom", "iLDDT", "QS_global",
        "Dockq", "ICS", "RNAComplex_TM", "RNAmotif_score", "pred_ProteinRNAInterfaceArea",
        "pred_ProteinRNAInterfaceRatio", "exp_ProteinRNAInterfaceArea", "exp_ProteinRNAInterfaceRatio",
        "Binding_affinity_kd", "RNA_ElectrostaticPotential", "XRayResolution", "exp_pH", "exp_Temp"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Complex_RMSD" in df.columns:
        df["Complex_RMSD_first"] = df["Complex_RMSD"].apply(first_list_value)

    if "Ions" in df.columns:
        ion_summary = pd.DataFrame([summarize_support_pairs(x, "ions") for x in df["Ions"]])
        df = pd.concat([df, ion_summary], axis=1)
        ion_unsupported_cols = collect_dynamic_codes(df["Ions"], "has_unsupported_ion")
        if ion_unsupported_cols:
            df = pd.concat([df, fill_support_code_columns(df["Ions"], ion_unsupported_cols, "has_unsupported_ion")], axis=1)

    if "Ligands" in df.columns:
        lig_summary = pd.DataFrame([summarize_support_pairs(x, "ligands") for x in df["Ligands"]])
        df = pd.concat([df, lig_summary], axis=1)
        lig_unsupported_cols = collect_dynamic_codes(df["Ligands"], "has_unsupported_ligand")
        if lig_unsupported_cols:
            df = pd.concat([df, fill_support_code_columns(df["Ligands"], lig_unsupported_cols, "has_unsupported_ligand")], axis=1)

    df = add_solution_features(df)
    df = add_buffer_peg_features(df)
    df = add_experiment_flags(df)
    df = add_composition_features(df)
    df = add_interface_agreement_features(df)
    df = add_global_interface_summary_features(df)

    if "Binding_affinity_kd" in df.columns:
        df["log10_kd"] = np.log10(df["Binding_affinity_kd"].replace(0, np.nan))
    if "Dockq" in df.columns and "af3_ranking_score" in df.columns:
        df["DockQ_minus_AF3rank"] = df["Dockq"] - df["af3_ranking_score"]

    # New RBP pair-metric summaries
    for raw_col, prefix in [
        ("RBP_irmsd", "RBP_irmsd"),
        ("RBP_DockQ", "RBP_DockQ"),
        ("RBP_fnat", "RBP_fnat"),
        ("RBP_QS", "RBP_QS"),
    ]:
        if raw_col in df.columns:
            recs = [summarize_rbp_metric_dict(prefix, x) for x in df[raw_col]]
            df = pd.concat([df, pd.DataFrame(recs)], axis=1)
        else:
            df[f"{prefix}_n_pairs"] = np.nan
            df[f"{prefix}_has_any"] = 0
            df[f"{prefix}_raw"] = np.nan
            for suffix in ["mean", "median", "max", "min", "std", "best", "worst"]:
                df[f"{prefix}_{suffix}"] = np.nan

    return df


# =============================================================================
# Per-chain extraction and assignment
# =============================================================================
def extract_matrix(x: Any) -> Optional[np.ndarray]:
    v = safe_literal_eval(x)
    if not isinstance(v, list) or len(v) == 0:
        return None
    try:
        arr = np.array(v, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    return arr


def get_matrix_status(matrix: Optional[np.ndarray], n_pred: int) -> Tuple[str, float, float]:
    """
    Returns:
      (status, n_rows, n_cols)
    status in {"ok","missing","bad_shape"}
    """
    if matrix is None:
        return "missing", np.nan, np.nan
    nr, nc = matrix.shape
    if nc != n_pred:
        return "bad_shape", float(nr), float(nc)
    return "ok", float(nr), float(nc)


def hungarian_assign(score_pred_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if score_pred_gt.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if linear_sum_assignment is None:
        used_cols: Set[int] = set()
        rows: List[int] = []
        cols: List[int] = []
        order = np.argsort(-np.nanmax(score_pred_gt, axis=1))
        for r in order:
            best_c = None
            best_v = -np.inf
            for c in range(score_pred_gt.shape[1]):
                if c in used_cols:
                    continue
                v = score_pred_gt[r, c]
                if np.isnan(v):
                    continue
                if v > best_v:
                    best_v = v
                    best_c = c
            if best_c is not None:
                rows.append(int(r))
                cols.append(int(best_c))
                used_cols.add(int(best_c))
        return np.array(rows, dtype=int), np.array(cols, dtype=int)
    cost = np.where(np.isnan(score_pred_gt), 1e6, -score_pred_gt)
    r, c = linear_sum_assignment(cost)
    return r.astype(int), c.astype(int)


def assign_by_same_length(pred_lengths: List[Any], tm_gt_pred: Optional[np.ndarray], lddt_gt_pred: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    n_pred = len(pred_lengths)
    out = [{
        "pred_rna_chain_index": i,
        "gt_rna_chain_index": np.nan,
        "assigned_rna_tm": np.nan,
        "assigned_rna_lddt": np.nan,
        "is_matched": 0,
        "assignment_method": "no_matrix",
    } for i in range(n_pred)]

    if tm_gt_pred is None:
        return out

    n_gt, n_pred_mat = tm_gt_pred.shape
    if n_pred_mat != n_pred:
        for r in out:
            r["assignment_method"] = "bad_shape"
        return out

    tm_pred_gt = tm_gt_pred.T.copy()
    lddt_pred_gt = lddt_gt_pred.T.copy() if lddt_gt_pred is not None and lddt_gt_pred.shape == tm_gt_pred.shape else None

    pred_len_to_idx: Dict[Any, List[int]] = {}
    for i, L in enumerate(pred_lengths):
        pred_len_to_idx.setdefault(L, []).append(i)

    gt_groups: Dict[Any, List[int]] = {}
    for gt_i in range(n_gt):
        compatible_lengths = set()
        row = tm_gt_pred[gt_i, :]
        for pred_i, L in enumerate(pred_lengths):
            v = row[pred_i]
            if not np.isnan(v) and v > 0:
                compatible_lengths.add(L)
        if len(compatible_lengths) == 1:
            L = list(compatible_lengths)[0]
            gt_groups.setdefault(L, []).append(gt_i)

    can_group = True
    for L, pred_idx in pred_len_to_idx.items():
        if len(gt_groups.get(L, [])) != len(pred_idx):
            can_group = False
            break

    assigned_pred: Set[int] = set()
    assigned_gt: Set[int] = set()

    if can_group and len(pred_len_to_idx) > 0:
        for L, pred_idx in pred_len_to_idx.items():
            gt_idx = gt_groups.get(L, [])
            if len(pred_idx) == 0 or len(gt_idx) == 0:
                continue
            sub_tm = tm_pred_gt[np.ix_(pred_idx, gt_idx)]
            rr, cc = hungarian_assign(sub_tm)
            for r_sub, c_sub in zip(rr, cc):
                p = pred_idx[int(r_sub)]
                g = gt_idx[int(c_sub)]
                out[p]["gt_rna_chain_index"] = g
                out[p]["assigned_rna_tm"] = tm_pred_gt[p, g]
                if lddt_pred_gt is not None and p < lddt_pred_gt.shape[0] and g < lddt_pred_gt.shape[1]:
                    out[p]["assigned_rna_lddt"] = lddt_pred_gt[p, g]
                out[p]["is_matched"] = 1
                out[p]["assignment_method"] = "same_length_grouped"
                assigned_pred.add(p)
                assigned_gt.add(g)

        rem_pred = [i for i in range(n_pred) if i not in assigned_pred]
        rem_gt = [j for j in range(n_gt) if j not in assigned_gt]
        if rem_pred and rem_gt:
            sub_tm = tm_pred_gt[np.ix_(rem_pred, rem_gt)]
            rr, cc = hungarian_assign(sub_tm)
            for r_sub, c_sub in zip(rr, cc):
                p = rem_pred[int(r_sub)]
                g = rem_gt[int(c_sub)]
                out[p]["gt_rna_chain_index"] = g
                out[p]["assigned_rna_tm"] = tm_pred_gt[p, g]
                if lddt_pred_gt is not None and p < lddt_pred_gt.shape[0] and g < lddt_pred_gt.shape[1]:
                    out[p]["assigned_rna_lddt"] = lddt_pred_gt[p, g]
                out[p]["is_matched"] = 1
                out[p]["assignment_method"] = "global_leftover"
    else:
        rr, cc = hungarian_assign(tm_pred_gt)
        for p, g in zip(rr, cc):
            out[int(p)]["gt_rna_chain_index"] = int(g)
            out[int(p)]["assigned_rna_tm"] = tm_pred_gt[int(p), int(g)]
            if lddt_pred_gt is not None and int(p) < lddt_pred_gt.shape[0] and int(g) < lddt_pred_gt.shape[1]:
                out[int(p)]["assigned_rna_lddt"] = lddt_pred_gt[int(p), int(g)]
            out[int(p)]["is_matched"] = 1
            out[int(p)]["assignment_method"] = "global"

    return out


def get_assigned_value_from_matrix(
    matrix_gt_pred: Optional[np.ndarray],
    pred_idx: int,
    gt_idx: Any,
    n_pred: int,
) -> float:
    if matrix_gt_pred is None:
        return np.nan
    if pd.isna(gt_idx):
        return np.nan
    gt_idx_i = int(gt_idx)
    if matrix_gt_pred.ndim != 2:
        return np.nan
    nr, nc = matrix_gt_pred.shape
    if nc != n_pred:
        return np.nan
    if gt_idx_i < 0 or gt_idx_i >= nr or pred_idx < 0 or pred_idx >= nc:
        return np.nan
    return to_float(matrix_gt_pred[gt_idx_i, pred_idx])


def _perchain_assigned_rna_metrics(base: Dict[str, Any], pred_chain_ids: List[str],
                                   pred_lengths: List[Any]) -> List[Dict[str, Any]]:
    """
    Per-predicted-RNA-chain assigned metrics (TM, LDDT, DockQ, ICS, iLDDT, QS), handling BOTH storage
    formats so old and new DBs both work:

    * NEW keyed-dict format (current eval): RNA_TM_AllChains / RNA_LDDT_AllChains are stored as
      chain-keyed single-key dicts [{"B":0.47},...] whose values are ALREADY the one-to-one assigned
      scores (the eval runs the Hungarian assignment and stores raw values). iLDDT/ICS are keyed by
      the predicted "protein:RNA" pair; DockQ/QS come from RBP_DockQ / RBP_QS (also "protein:RNA").
      We read TM/LDDT directly by predicted chain id, and reduce the pair-keyed metrics to the max
      over protein partners for each RNA chain.

    * LEGACY matrix format (published DBs): RNA_*_AllChains are full ref x pred numeric matrices; we
      run the original Hungarian assignment (assign_by_same_length + get_assigned_value_from_matrix)
      so existing DBs reproduce exactly.
    """
    n_pred = len(pred_chain_ids)
    tm_raw = base.get("RNA_TM_AllChains")
    tm_matrix = extract_matrix(tm_raw)

    if tm_matrix is not None:
        # ---- legacy matrix path (unchanged behaviour) ----
        lddt_m = extract_matrix(base.get("RNA_LDDT_AllChains"))
        dockq_m = extract_matrix(base.get("RNA_Dockq_AllChains"))
        ics_m = extract_matrix(base.get("RNA_ICS_AllChains"))
        ilddt_m = extract_matrix(base.get("RNA_iLDDT_AllChains"))
        qs_m = extract_matrix(base.get("RNA_QS_global_AllChains"))
        assignments = assign_by_same_length(pred_lengths, tm_matrix, lddt_m)
        per = []
        for i in range(n_pred):
            a = assignments[i]
            gt_idx = a["gt_rna_chain_index"]
            per.append({
                "assigned_rna_tm": a["assigned_rna_tm"],
                "assigned_rna_lddt": a["assigned_rna_lddt"],
                "assigned_rna_dockq": get_assigned_value_from_matrix(dockq_m, i, gt_idx, n_pred),
                "assigned_rna_ics": get_assigned_value_from_matrix(ics_m, i, gt_idx, n_pred),
                "assigned_rna_ilddt": get_assigned_value_from_matrix(ilddt_m, i, gt_idx, n_pred),
                "assigned_rna_qs_global": get_assigned_value_from_matrix(qs_m, i, gt_idx, n_pred),
                "gt_rna_chain_index": gt_idx,
                "is_matched": a["is_matched"],
                "assignment_method": a["assignment_method"],
            })
        return per

    # ---- new keyed-dict path (eval already assigned one-to-one) ----
    tm_keyed = parse_metric_pair_list(tm_raw)                               # {pred_chain: tm}
    lddt_keyed = parse_metric_pair_list(base.get("RNA_LDDT_AllChains"))     # {pred_chain: lddt}
    ilddt_pairs = parse_metric_pair_list(base.get("RNA_iLDDT_AllChains"))   # {"prot:rna": v}
    ics_pairs = parse_metric_pair_list(base.get("RNA_ICS_AllChains"))
    dockq_pairs = parse_metric_pair_list(base.get("RBP_DockQ"))
    qs_pairs = parse_metric_pair_list(base.get("RBP_QS"))

    def _pair_max_for_chain(pairs: Dict[str, float], ch: str) -> float:
        # keys are "protein:RNA"; keep entries whose RNA suffix is this predicted chain, take the max
        # over protein partners (a chain-quality target: the best interface it participates in).
        vals = [v for k, v in pairs.items() if str(k).split(":")[-1] == ch]
        return float(max(vals)) if vals else np.nan

    per = []
    for ch in pred_chain_ids:
        tm = tm_keyed.get(ch, np.nan)
        per.append({
            "assigned_rna_tm": float(tm) if pd.notna(tm) else np.nan,
            "assigned_rna_lddt": float(lddt_keyed[ch]) if ch in lddt_keyed else np.nan,
            "assigned_rna_dockq": _pair_max_for_chain(dockq_pairs, ch),
            "assigned_rna_ics": _pair_max_for_chain(ics_pairs, ch),
            "assigned_rna_ilddt": _pair_max_for_chain(ilddt_pairs, ch),
            "assigned_rna_qs_global": _pair_max_for_chain(qs_pairs, ch),
            "gt_rna_chain_index": np.nan,  # eval did the assignment; per-chain gt index not meaningful here
            "is_matched": int(ch in tm_keyed),
            "assignment_method": "eval_assigned",
        })
    return per


def build_per_chain_rows(parsed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in parsed.iterrows():
        base = row.to_dict()

        pred_chain_ids = [str(x) for x in ensure_list(base.get("RNAChainIDs_pred", base.get("RNAChainIDs")))]
        pred_lengths = ensure_list(base.get("RNALength_pred", base.get("RNALength")))
        pred_sequences = [
            str(x) if x is not None else ""
            for x in ensure_list(base.get("RNASequence_pred", base.get("RNASequence")))
        ]
        msa_size_dict = ensure_dict(base.get("RNA_msa_size"))
        af3_rna_msa_list = ensure_list(base.get("af3_rna_MSA"))

        n_pred = len(pred_chain_ids)
        if len(pred_lengths) == 0 and n_pred > 0:
            pred_lengths = [np.nan] * n_pred
        if len(pred_sequences) == 0 and n_pred > 0:
            pred_sequences = [""] * n_pred
        if len(pred_lengths) < n_pred:
            pred_lengths = list(pred_lengths) + [np.nan] * (n_pred - len(pred_lengths))
        if len(pred_sequences) < n_pred:
            pred_sequences = list(pred_sequences) + [""] * (n_pred - len(pred_sequences))
        if len(af3_rna_msa_list) < n_pred:
            af3_rna_msa_list = list(af3_rna_msa_list) + [np.nan] * (n_pred - len(af3_rna_msa_list))

        exp_rna_sequences = as_clean_string_list(base.get("RNASequence_exp"))
        if len(exp_rna_sequences) == 0:
            exp_rna_sequences = pred_sequences[:]
        exp_rna_sequences = (exp_rna_sequences + [""] * max(0, n_pred - len(exp_rna_sequences)))[:n_pred]

        rna_families_per_chain = map_family_to_sequences(exp_rna_sequences, base.get("RNAFamily"), n_expected=n_pred)
        rna_family_summary = summarize_family_list(rna_families_per_chain)

        protein_sequences = as_clean_string_list(base.get("ProteinSequence_exp"))
        protein_families_per_chain = map_family_to_sequences(
            protein_sequences,
            base.get("ProteinFamily"),
            n_expected=int(base.get("NumberProteins")) if pd.notna(base.get("NumberProteins")) else None,
        )
        protein_family_summary = summarize_family_list(protein_families_per_chain)

        # Core matching matrices
        tm_gt_pred = extract_matrix(base.get("RNA_TM_AllChains"))
        lddt_gt_pred = extract_matrix(base.get("RNA_LDDT_AllChains"))

        # New RNA all-chain matrices
        dockq_gt_pred = extract_matrix(base.get("RNA_Dockq_AllChains"))
        ics_gt_pred = extract_matrix(base.get("RNA_ICS_AllChains"))
        ilddt_gt_pred = extract_matrix(base.get("RNA_iLDDT_AllChains"))
        qs_gt_pred = extract_matrix(base.get("RNA_QS_global_AllChains"))

        # Assigned per-chain metrics, handling BOTH the new keyed-dict eval output and legacy matrices.
        per_chain_metrics = _perchain_assigned_rna_metrics(base, pred_chain_ids, pred_lengths)

        tm_status, tm_nrow, tm_ncol = get_matrix_status(tm_gt_pred, n_pred)
        lddt_status, lddt_nrow, lddt_ncol = get_matrix_status(lddt_gt_pred, n_pred)
        dockq_status, dockq_nrow, dockq_ncol = get_matrix_status(dockq_gt_pred, n_pred)
        ics_status, ics_nrow, ics_ncol = get_matrix_status(ics_gt_pred, n_pred)
        ilddt_status, ilddt_nrow, ilddt_ncol = get_matrix_status(ilddt_gt_pred, n_pred)
        qs_status, qs_nrow, qs_ncol = get_matrix_status(qs_gt_pred, n_pred)
        per_rna_chain_interface = compute_per_rna_chain_interface_features(base)

        for i in range(n_pred):
            chain_row = dict(base)
            ch = pred_chain_ids[i]
            chain_row["pred_rna_chain_index"] = i
            chain_row["pred_rna_chain_id"] = ch
            chain_row["pred_rna_chain_length"] = to_float(pred_lengths[i])
            chain_row["pred_rna_sequence"] = pred_sequences[i]
            chain_row["pred_rna_is_modified"] = sequence_is_modified(pred_sequences[i])

            chain_row["exp_rna_sequence"] = exp_rna_sequences[i] if i < len(exp_rna_sequences) else np.nan
            chain_row["rna_family"] = rna_families_per_chain[i] if i < len(rna_families_per_chain) else np.nan
            chain_row["rna_family_is_known"] = int(normalize_missing_token(chain_row["rna_family"]) is not None)
            chain_row["n_unique_rna_families"] = rna_family_summary["n_unique"]
            chain_row["rna_family_summary"] = rna_family_summary["joined"]

            chain_row["protein_family_summary"] = protein_family_summary["joined"]
            chain_row["protein_family_is_known"] = protein_family_summary["is_known"]
            chain_row["n_unique_protein_families"] = protein_family_summary["n_unique"]

            chain_row["rna_msa_size"] = to_float(msa_size_dict.get(ch))
            chain_row["log10_rna_msa_size"] = np.log10(chain_row["rna_msa_size"]) if not np.isnan(chain_row["rna_msa_size"]) and chain_row["rna_msa_size"] > 0 else np.nan
            chain_row["af3_rna_MSA_list_value"] = to_float(af3_rna_msa_list[i])

            # Protein MSA depth: geometric-mean alignment depth over the complex's
            # protein chains (mean of log10 per chain), mirroring the RNA MSA log10
            # transform above. For a single-protein complex this equals log10 of that
            # protein's own MSA depth (the RNA analogue). The geometric mean is
            # decoupled from protein-chain count (unlike a sum, which the complexity
            # group already captures) and robust to one deep/shallow chain.
            _protein_msa_dict = ensure_dict(base.get("Protein_msa_size"))
            _pmsa_log = [np.log10(to_float(v)) for v in _protein_msa_dict.values()
                         if to_float(v) == to_float(v) and to_float(v) > 0]
            chain_row["log10_protein_msa_size"] = float(np.mean(_pmsa_log)) if _pmsa_log else np.nan
            chain_row["protein_msa_size"] = float(10.0 ** np.mean(_pmsa_log)) if _pmsa_log else np.nan

            chain_row["af3_protein_pTM_list_value"] = list_value_at(base.get("af3_protein_pTM"), i)
            chain_row["af3_rna_pTM_list_value"] = list_value_at(base.get("af3_rna_pTM"), i)
            chain_row["af3_protein_ipTM_list_value"] = list_value_at(base.get("af3_protein_ipTM"), i)
            chain_row["af3_rna_ipTM_list_value"] = list_value_at(base.get("af3_rna_ipTM"), i)

            m = per_chain_metrics[i]
            chain_row["gt_rna_chain_index"] = m["gt_rna_chain_index"]
            chain_row["assigned_rna_tm"] = m["assigned_rna_tm"]
            chain_row["assigned_rna_lddt"] = m["assigned_rna_lddt"]
            chain_row["is_matched"] = m["is_matched"]
            chain_row["assignment_method"] = m["assignment_method"]

            # Diagnostics for matrix availability / shapes
            chain_row["RNA_TM_AllChains_status"] = tm_status
            chain_row["RNA_TM_AllChains_n_gt"] = tm_nrow
            chain_row["RNA_TM_AllChains_n_pred"] = tm_ncol
            chain_row["RNA_LDDT_AllChains_status"] = lddt_status
            chain_row["RNA_LDDT_AllChains_n_gt"] = lddt_nrow
            chain_row["RNA_LDDT_AllChains_n_pred"] = lddt_ncol
            chain_row["RNA_Dockq_AllChains_status"] = dockq_status
            chain_row["RNA_Dockq_AllChains_n_gt"] = dockq_nrow
            chain_row["RNA_Dockq_AllChains_n_pred"] = dockq_ncol
            chain_row["RNA_ICS_AllChains_status"] = ics_status
            chain_row["RNA_ICS_AllChains_n_gt"] = ics_nrow
            chain_row["RNA_ICS_AllChains_n_pred"] = ics_ncol
            chain_row["RNA_iLDDT_AllChains_status"] = ilddt_status
            chain_row["RNA_iLDDT_AllChains_n_gt"] = ilddt_nrow
            chain_row["RNA_iLDDT_AllChains_n_pred"] = ilddt_ncol
            chain_row["RNA_QS_global_AllChains_status"] = qs_status
            chain_row["RNA_QS_global_AllChains_n_gt"] = qs_nrow
            chain_row["RNA_QS_global_AllChains_n_pred"] = qs_ncol

            # Assigned interface metrics (dual-format): eval keyed-dicts (DockQ/QS from RBP_DockQ/QS)
            # or legacy ref x pred matrices, resolved in _perchain_assigned_rna_metrics above.
            chain_row["assigned_rna_dockq"] = m["assigned_rna_dockq"]
            chain_row["assigned_rna_ics"] = m["assigned_rna_ics"]
            chain_row["assigned_rna_ilddt"] = m["assigned_rna_ilddt"]
            chain_row["assigned_rna_qs_global"] = m["assigned_rna_qs_global"]

            # explicit missing flags
            for metric_col in [
                "assigned_rna_tm",
                "assigned_rna_lddt",
                "assigned_rna_dockq",
                "assigned_rna_ics",
                "assigned_rna_ilddt",
                "assigned_rna_qs_global",
            ]:
                chain_row[f"is_missing_{metric_col}"] = int(pd.isna(chain_row.get(metric_col)))

            # more explicit analysis-friendly flags
            chain_row["has_valid_assigned_rna_tm"] = int(not pd.isna(chain_row["assigned_rna_tm"]))
            chain_row["has_valid_assigned_rna_lddt"] = int(not pd.isna(chain_row["assigned_rna_lddt"]))
            chain_row["has_valid_assigned_rna_dockq"] = int(not pd.isna(chain_row["assigned_rna_dockq"]))
            chain_row["has_valid_assigned_rna_ics"] = int(not pd.isna(chain_row["assigned_rna_ics"]))
            chain_row["has_valid_assigned_rna_ilddt"] = int(not pd.isna(chain_row["assigned_rna_ilddt"]))
            chain_row["has_valid_assigned_rna_qs_global"] = int(not pd.isna(chain_row["assigned_rna_qs_global"]))

            # predicted RNA-chain-specific interface features
            iface_rec = per_rna_chain_interface.get(str(ch), {})
            chain_row["chain_n_interface_pairs"] = iface_rec.get("n_interface_pairs", np.nan)
            chain_row["chain_n_pairs_with_contacts"] = iface_rec.get("n_pairs_with_contacts", np.nan)
            chain_row["chain_frac_pairs_with_contacts"] = iface_rec.get("frac_pairs_with_contacts", np.nan)
            chain_row["chain_total_hbond_proteinRNA"] = iface_rec.get("total_hbond_proteinRNA", np.nan)
            chain_row["chain_total_vdwbond_proteinRNA"] = iface_rec.get("total_vdwbond_proteinRNA", np.nan)
            chain_row["chain_total_contact_entries"] = iface_rec.get("total_contact_entries", np.nan)

            chain_row["chain_n_unique_protein_interface_residues"] = iface_rec.get("n_unique_protein_interface_residues", np.nan)
            chain_row["chain_n_unique_rna_interface_residues"] = iface_rec.get("n_unique_rna_interface_residues", np.nan)
            chain_row["chain_protein_interface_coverage"] = iface_rec.get("protein_interface_coverage", np.nan)
            chain_row["chain_rna_interface_coverage"] = iface_rec.get("rna_interface_coverage", np.nan)

            chain_row["chain_interface_contact_distance_mean"] = iface_rec.get("interface_contact_distance_mean", np.nan)
            chain_row["chain_interface_contact_distance_median"] = iface_rec.get("interface_contact_distance_median", np.nan)
            chain_row["chain_interface_contact_distance_min"] = iface_rec.get("interface_contact_distance_min", np.nan)
            chain_row["chain_interface_contact_distance_max"] = iface_rec.get("interface_contact_distance_max", np.nan)
            chain_row["chain_interface_contact_distance_std"] = iface_rec.get("interface_contact_distance_std", np.nan)

            chain_row["chain_frac_contacts_lt_2_5A"] = iface_rec.get("frac_contacts_lt_2_5A", np.nan)
            chain_row["chain_frac_contacts_lt_3_0A"] = iface_rec.get("frac_contacts_lt_3_0A", np.nan)
            chain_row["chain_frac_contacts_lt_3_5A"] = iface_rec.get("frac_contacts_lt_3_5A", np.nan)
            chain_row["chain_frac_contacts_lt_4_0A"] = iface_rec.get("frac_contacts_lt_4_0A", np.nan)

            chain_row["chain_interface_hbond_per_contact"] = iface_rec.get("interface_hbond_per_contact", np.nan)
            chain_row["chain_interface_vdw_per_contact"] = iface_rec.get("interface_vdw_per_contact", np.nan)
            chain_row["chain_interface_hbond_vdw_ratio"] = iface_rec.get("interface_hbond_vdw_ratio", np.nan)

            chain_row["chain_interface_rna_motif_length_sum"] = iface_rec.get("interface_rna_motif_length_sum", np.nan)
            chain_row["chain_interface_rna_motif_length_mean"] = iface_rec.get("interface_rna_motif_length_mean", np.nan)
            chain_row["chain_interface_rna_motif_length_max"] = iface_rec.get("interface_rna_motif_length_max", np.nan)
            chain_row["chain_interface_rna_motif_coverage_of_chain"] = iface_rec.get("interface_rna_motif_coverage_of_total_rna", np.nan)

            chain_row["chain_interface_frac_contacts_to_A"] = iface_rec.get("interface_frac_contacts_to_A", np.nan)
            chain_row["chain_interface_frac_contacts_to_C"] = iface_rec.get("interface_frac_contacts_to_C", np.nan)
            chain_row["chain_interface_frac_contacts_to_G"] = iface_rec.get("interface_frac_contacts_to_G", np.nan)
            chain_row["chain_interface_frac_contacts_to_U"] = iface_rec.get("interface_frac_contacts_to_U", np.nan)
            chain_row["chain_interface_frac_contacts_to_modified"] = iface_rec.get("interface_frac_contacts_to_modified", np.nan)

            chain_row["chain_interface_frac_positive_contacts"] = iface_rec.get("interface_frac_positive_contacts", np.nan)
            chain_row["chain_interface_frac_aromatic_contacts"] = iface_rec.get("interface_frac_aromatic_contacts", np.nan)
            chain_row["chain_interface_frac_polar_contacts"] = iface_rec.get("interface_frac_polar_contacts", np.nan)
            chain_row["chain_interface_frac_hydrophobic_contacts"] = iface_rec.get("interface_frac_hydrophobic_contacts", np.nan)
            chain_row["chain_interface_frac_acidic_contacts"] = iface_rec.get("interface_frac_acidic_contacts", np.nan)
            chain_row["chain_interface_frac_other_contacts"] = iface_rec.get("interface_frac_other_contacts", np.nan)

            chain_row["chain_qc_n_pairs_failed_contact_parse"] = iface_rec.get("qc_n_pairs_failed_contact_parse", np.nan)
            chain_row["chain_qc_frac_pairs_failed_contact_parse"] = iface_rec.get("qc_frac_pairs_failed_contact_parse", np.nan)
            chain_row["chain_qc_any_interface_parse_problem"] = iface_rec.get("qc_any_interface_parse_problem", np.nan)


            # Single-RNA-only scalar intrinsic features
            if pd.notna(chain_row.get("NumberRNAs")) and int(chain_row["NumberRNAs"]) == 1:
                chain_row["RNA_GC_Content"] = to_float(base.get("RNA_GC_Content"))
                chain_row["RNA_SequenceComplexity"] = to_float(base.get("RNA_SequenceComplexity"))
                chain_row["RNA_Shannon_Entropy_K1"] = to_float(base.get("RNA_Shannon_Entropy_K1"))
                chain_row["RNA_Shannon_Entropy_K2"] = to_float(base.get("RNA_Shannon_Entropy_K2"))
                chain_row["RNA_GC_Skew"] = to_float(base.get("RNA_GC_Skew"))
                chain_row["RNA_AU_Skew"] = to_float(base.get("RNA_AU_Skew"))
                chain_row["RNA_AU_Content"] = to_float(base.get("RNA_AU_Content"))
                chain_row["RNA_MFE_Value"] = to_float(base.get("RNA_MFE_Value"))
                chain_row["RNA_MFE_Structure"] = base.get("RNA_MFE_Structure")
                chain_row["RNA_Ensemble_Diversity"] = to_float(base.get("RNA_Ensemble_Diversity"))
                chain_row["RNA_Centroid_Energy"] = to_float(base.get("RNA_Centroid_Energy"))
                chain_row["Protein_msa_size"] = to_float(base.get("Protein_msa_size"))
                chain_row["AAmatch_score"] = to_float(base.get("AAmatch_score"))
                chain_row["RNAmatch_score"] = to_float(base.get("RNAmatch_score"))
                chain_row["num_stems"] = parse_loop_count(base.get("RNA_Stems"))
                chain_row["num_hairpin_loops"] = parse_loop_count(base.get("RNA_HairpinLoops"))
                chain_row["num_internal_loops"] = parse_loop_count(base.get("RNA_InternalLoops"))
                chain_row["num_multibranch_loops"] = parse_loop_count(base.get("RNA_MultibranchLoops"))
                chain_row["num_dangling_ends"] = parse_loop_count(base.get("RNA_DanglingEnds"))
                chain_row["num_pseudoknots"] = parse_loop_count(base.get("RNA_Pseudoknots"))

            else:
                for c in [
                    "RNA_GC_Content", "RNA_SequenceComplexity", "RNA_Shannon_Entropy_K1", "RNA_Shannon_Entropy_K2",
                    "RNA_GC_Skew", "RNA_AU_Skew", "RNA_AU_Content", "RNA_MFE_Value", "RNA_MFE_Structure",
                    "RNA_Ensemble_Diversity", "RNA_Centroid_Energy", "Protein_msa_size", "AAmatch_score",
                    "RNAmatch_score", "num_stems", "num_hairpin_loops", "num_internal_loops",
                    "num_multibranch_loops", "num_dangling_ends", "num_pseudoknots",
                ]:
                    chain_row[c] = np.nan

            for base_col, new_col in [
                ("num_stems", "stems_per_100nt"),
                ("num_hairpin_loops", "hairpin_loops_per_100nt"),
                ("num_internal_loops", "internal_loops_per_100nt"),
                ("num_multibranch_loops", "multibranch_loops_per_100nt"),
                ("num_dangling_ends", "dangling_ends_per_100nt"),
                ("num_pseudoknots", "pseudoknots_per_100nt"),
            ]:
                denom = chain_row["pred_rna_chain_length"] / 100.0 if pd.notna(chain_row["pred_rna_chain_length"]) and chain_row["pred_rna_chain_length"] > 0 else np.nan
                chain_row[new_col] = safe_div(chain_row.get(base_col), denom)

            rows.append(chain_row)

    return pd.DataFrame(rows)


def select_output_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "pdb_id", "exp_db_id", "PDBId", "source_db", "is_post_cutoff", "NumberProteins", "NumberRNAs",

        # composition
        "protein_total_length", "rna_total_length", "total_complex_length",
        "protein_mean_chain_length", "protein_max_chain_length", "protein_min_chain_length",
        "protein_std_chain_length", "protein_chain_length_cv",
        "rna_mean_chain_length", "rna_max_chain_length", "rna_min_chain_length",
        "rna_std_chain_length", "rna_chain_length_cv", "protein_residue_fraction", "rna_residue_fraction",
        "protein_to_rna_length_ratio", "max_rna_fraction_of_total_rna", "min_rna_fraction_of_total_rna",
        "agg_rna_gc_content", "agg_rna_au_content", "agg_rna_gc_skew", "agg_rna_au_skew",
        "agg_rna_sequence_complexity", "agg_rna_shannon_entropy_k1", "agg_rna_shannon_entropy_k2",

        # per-chain identity/features
        "pred_rna_chain_index", "pred_rna_chain_id", "pred_rna_chain_length", "pred_rna_sequence", "pred_rna_is_modified",
        "exp_rna_sequence", "rna_family", "rna_family_is_known", "n_unique_rna_families", "rna_family_summary",
        "protein_family_summary", "protein_family_is_known", "n_unique_protein_families",
        "rna_msa_size", "log10_rna_msa_size", "af3_rna_MSA_list_value",
        "protein_msa_size", "log10_protein_msa_size",
        "af3_protein_pTM_list_value", "af3_rna_pTM_list_value", "af3_protein_ipTM_list_value", "af3_rna_ipTM_list_value",

        # assignment / per-chain targets
        "gt_rna_chain_index", "assigned_rna_tm", "assigned_rna_lddt",
        "assigned_rna_dockq", "assigned_rna_ics", "assigned_rna_ilddt", "assigned_rna_qs_global",
        "is_matched", "assignment_method",

        # explicit missingness / availability flags
        "is_missing_assigned_rna_tm", "is_missing_assigned_rna_lddt",
        "is_missing_assigned_rna_dockq", "is_missing_assigned_rna_ics",
        "is_missing_assigned_rna_ilddt", "is_missing_assigned_rna_qs_global",
        "has_valid_assigned_rna_tm", "has_valid_assigned_rna_lddt",
        "has_valid_assigned_rna_dockq", "has_valid_assigned_rna_ics",
        "has_valid_assigned_rna_ilddt", "has_valid_assigned_rna_qs_global",

        # matrix diagnostics
        "RNA_TM_AllChains_status", "RNA_TM_AllChains_n_gt", "RNA_TM_AllChains_n_pred",
        "RNA_LDDT_AllChains_status", "RNA_LDDT_AllChains_n_gt", "RNA_LDDT_AllChains_n_pred",
        "RNA_Dockq_AllChains_status", "RNA_Dockq_AllChains_n_gt", "RNA_Dockq_AllChains_n_pred",
        "RNA_ICS_AllChains_status", "RNA_ICS_AllChains_n_gt", "RNA_ICS_AllChains_n_pred",
        "RNA_iLDDT_AllChains_status", "RNA_iLDDT_AllChains_n_gt", "RNA_iLDDT_AllChains_n_pred",
        "RNA_QS_global_AllChains_status", "RNA_QS_global_AllChains_n_gt", "RNA_QS_global_AllChains_n_pred",

        # intrinsic RNA features single-RNA only
        "RNA_GC_Content", "RNA_SequenceComplexity", "RNA_Shannon_Entropy_K1", "RNA_Shannon_Entropy_K2",
        "RNA_GC_Skew", "RNA_AU_Skew", "RNA_AU_Content", "RNA_MFE_Value", "RNA_MFE_Structure",
        "RNA_Ensemble_Diversity", "RNA_Centroid_Energy", "Protein_msa_size", "AAmatch_score", "RNAmatch_score",
        "num_stems", "num_hairpin_loops", "num_internal_loops", "num_multibranch_loops", "num_dangling_ends", "num_pseudoknots",
        "stems_per_100nt", "hairpin_loops_per_100nt", "internal_loops_per_100nt",
        "multibranch_loops_per_100nt", "dangling_ends_per_100nt", "pseudoknots_per_100nt",

        # template / AF3 / complex features
        "n_rbp_templates", "n_protein_templates", "id_is_rbp_template", "id_is_protein_template",
        "af3_protein_pLDDT_avg", "af3_rna_pLDDT_avg", "af3_global_pae_avg", "af3_chain_pair_pae_min", "af3_ranking_score",

        # experimental / biochemical
        "Binding_affinity_kd", "log10_kd", "RNA_ElectrostaticPotential", "XRayResolution", "exp_pH", "exp_Temp",
        "is_xray", "is_em", "is_nmr",

        # ions/ligands
        "n_ions_listed", "n_supported_ions", "n_unsupported_ions", "n_unknown_ions",
        "has_any_ion", "has_any_supported_ion", "has_any_unsupported_ion", "frac_supported_ions", "n_unique_ions",
        "n_ligands_listed", "n_supported_ligands", "n_unsupported_ligands", "n_unknown_ligands",
        "has_any_ligand", "has_any_supported_ligand", "has_any_unsupported_ligand", "frac_supported_ligands", "n_unique_ligands",

        # chemistry / condition
        "ionic_strength", "peg_percent", "peg_mw", "buffer_m", "has_peg",

        # interface features
        "exp_ProteinRNAInterfaceArea", "pred_ProteinRNAInterfaceArea", "ProteinRNAInterfaceArea_diff_exp_minus_pred",
        "ProteinRNAInterfaceArea_abs_diff", "ProteinRNAInterfaceArea_rel_error", "ProteinRNAInterfaceArea_pred_over_exp",
        "exp_ProteinRNAInterfaceRatio", "pred_ProteinRNAInterfaceRatio", "ProteinRNAInterfaceRatio_diff_exp_minus_pred",
        "ProteinRNAInterfaceRatio_abs_diff", "ProteinRNAInterfaceRatio_rel_error", "ProteinRNAInterfaceRatio_pred_over_exp",

        # global predicted interface summary features
        "n_protein_rna_interface_pairs", "n_pairs_with_contacts", "frac_pairs_with_contacts",
        "total_hbond_proteinRNA", "total_vdwbond_proteinRNA", "total_contact_entries",
        "n_unique_protein_interface_residues_global", "n_unique_rna_interface_residues_global",
        "protein_interface_coverage_global", "rna_interface_coverage_global",
        "interface_contact_distance_mean", "interface_contact_distance_median",
        "interface_contact_distance_min", "interface_contact_distance_max",
        "interface_contact_distance_std",
        "frac_contacts_lt_2_5A", "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
        "interface_hbond_per_contact", "interface_vdw_per_contact", "interface_hbond_vdw_ratio",
        "interface_rna_motif_length_sum", "interface_rna_motif_length_mean",
        "interface_rna_motif_length_max", "interface_rna_motif_coverage_of_total_rna",
        "interface_frac_contacts_to_A", "interface_frac_contacts_to_C", "interface_frac_contacts_to_G",
        "interface_frac_contacts_to_U", "interface_frac_contacts_to_modified",
        "interface_frac_positive_contacts", "interface_frac_aromatic_contacts",
        "interface_frac_polar_contacts", "interface_frac_hydrophobic_contacts",
        "interface_frac_acidic_contacts", "interface_frac_other_contacts",
        "qc_contactlist_was_flat_single_pair", "qc_interface_lengths_match",
        "qc_n_pairs_failed_contact_parse", "qc_frac_pairs_failed_contact_parse", "qc_any_interface_parse_problem",

        # chain-specific predicted interface features
        "chain_n_interface_pairs", "chain_n_pairs_with_contacts", "chain_frac_pairs_with_contacts",
        "chain_total_hbond_proteinRNA", "chain_total_vdwbond_proteinRNA", "chain_total_contact_entries",
        "chain_n_unique_protein_interface_residues", "chain_n_unique_rna_interface_residues",
        "chain_protein_interface_coverage", "chain_rna_interface_coverage",
        "chain_interface_contact_distance_mean", "chain_interface_contact_distance_median",
        "chain_interface_contact_distance_min", "chain_interface_contact_distance_max",
        "chain_interface_contact_distance_std",
        "chain_frac_contacts_lt_2_5A", "chain_frac_contacts_lt_3_0A",
        "chain_frac_contacts_lt_3_5A", "chain_frac_contacts_lt_4_0A",
        "chain_interface_hbond_per_contact", "chain_interface_vdw_per_contact", "chain_interface_hbond_vdw_ratio",
        "chain_interface_rna_motif_length_sum", "chain_interface_rna_motif_length_mean",
        "chain_interface_rna_motif_length_max", "chain_interface_rna_motif_coverage_of_chain",
        "chain_interface_frac_contacts_to_A", "chain_interface_frac_contacts_to_C",
        "chain_interface_frac_contacts_to_G", "chain_interface_frac_contacts_to_U",
        "chain_interface_frac_contacts_to_modified",
        "chain_interface_frac_positive_contacts", "chain_interface_frac_aromatic_contacts",
        "chain_interface_frac_polar_contacts", "chain_interface_frac_hydrophobic_contacts",
        "chain_interface_frac_acidic_contacts", "chain_interface_frac_other_contacts",
        "chain_qc_n_pairs_failed_contact_parse", "chain_qc_frac_pairs_failed_contact_parse",
        "chain_qc_any_interface_parse_problem",

        # complex metrics
        "Complex_RMSD_first", "Complex_TM", "Complex_LDDT", "Complex_LDDT_allAtom",
        "iLDDT", "QS_global", "Dockq", "DockQ_minus_AF3rank", "ICS", "RNAComplex_TM", "RNAmotif_score",

        # new RBP summary metrics
        "RBP_irmsd_n_pairs", "RBP_irmsd_has_any", "RBP_irmsd_mean", "RBP_irmsd_median", "RBP_irmsd_max", "RBP_irmsd_min",
        "RBP_irmsd_std", "RBP_irmsd_best", "RBP_irmsd_worst", "RBP_irmsd_raw",
        "RBP_DockQ_n_pairs", "RBP_DockQ_has_any", "RBP_DockQ_mean", "RBP_DockQ_median", "RBP_DockQ_max", "RBP_DockQ_min",
        "RBP_DockQ_std", "RBP_DockQ_best", "RBP_DockQ_worst", "RBP_DockQ_raw",
        "RBP_fnat_n_pairs", "RBP_fnat_has_any", "RBP_fnat_mean", "RBP_fnat_median", "RBP_fnat_max", "RBP_fnat_min",
        "RBP_fnat_std", "RBP_fnat_best", "RBP_fnat_worst", "RBP_fnat_raw",
        "RBP_QS_n_pairs", "RBP_QS_has_any", "RBP_QS_mean", "RBP_QS_median", "RBP_QS_max", "RBP_QS_min",
        "RBP_QS_std", "RBP_QS_best", "RBP_QS_worst", "RBP_QS_raw",
    ]

    dynamic = sorted(c for c in df.columns if c.startswith(("has_unsupported_ion_", "has_unsupported_ligand_", "sol_", "sample_")))
    categorical = [c for c in ["ionic_strength_category", "peg_concentration_category", "peg_mw_category", "buffer_category", "buffer_name"] if c in df.columns]
    return [c for c in preferred if c in df.columns] + categorical + dynamic


# =============================================================================
# Main processing
# =============================================================================
def process_db(db_path: Path, out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_tag = db_path.stem
    out_dir = out_root / db_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        pred = load_table_columns(conn, "pred_protein_rna", BASE_PRED_COLUMNS)
        exp = load_table_columns(conn, "exp_protein_rna", BASE_EXP_COLUMNS)

    if "exp_db_id" not in pred.columns:
        raise ValueError(f"[{db_tag}] pred_protein_rna is missing 'exp_db_id'")
    if "PDBId" not in exp.columns:
        raise ValueError(f"[{db_tag}] exp_protein_rna is missing 'PDBId'")

    pred = pred.rename(columns={
        "ProteinRNAInterfaceArea": "pred_ProteinRNAInterfaceArea",
        "ProteinRNAInterfaceRatio": "pred_ProteinRNAInterfaceRatio",
    })
    exp = exp.rename(columns={
        "ProteinRNAInterfaceArea": "exp_ProteinRNAInterfaceArea",
        "ProteinRNAInterfaceRatio": "exp_ProteinRNAInterfaceRatio",
    })

    merged = pred.merge(exp, left_on="exp_db_id", right_on="PDBId", how="inner", suffixes=("_pred", "_exp"))
    parsed = parse_features_and_metrics(merged, db_tag=db_tag)
    per_chain = build_per_chain_rows(parsed)

    keep_cols = select_output_columns(per_chain)
    per_chain = per_chain[keep_cols].copy()

    is_rna_protein = (per_chain["NumberProteins"] >= 1) & (per_chain["NumberRNAs"] >= 1)
    is_1v1 = (per_chain["NumberProteins"] == 1) & (per_chain["NumberRNAs"] == 1)
    is_multichain = is_rna_protein & ~is_1v1

    one_v_one = per_chain[is_1v1].copy()
    multi = per_chain[is_multichain].copy()

    all_path = out_dir / f"{db_tag}__per_chain_all.csv"
    one_v_one_path = out_dir / f"{db_tag}__per_chain_rna_protein_1v1.csv"
    multi_path = out_dir / f"{db_tag}__per_chain_multichain_complex.csv"

    per_chain.to_csv(all_path, index=False)
    one_v_one.to_csv(one_v_one_path, index=False)
    multi.to_csv(multi_path, index=False)

    summary = pd.DataFrame([
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_all", "n_rows": len(per_chain)},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_rna_protein_1v1", "n_rows": len(one_v_one)},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_multichain_complex", "n_rows": len(multi)},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_matched", "n_rows": int(per_chain["is_matched"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_tm", "n_rows": int(per_chain["has_valid_assigned_rna_tm"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_lddt", "n_rows": int(per_chain["has_valid_assigned_rna_lddt"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_dockq", "n_rows": int(per_chain["has_valid_assigned_rna_dockq"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_ics", "n_rows": int(per_chain["has_valid_assigned_rna_ics"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_ilddt", "n_rows": int(per_chain["has_valid_assigned_rna_ilddt"].fillna(0).sum())},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_valid_qs_global", "n_rows": int(per_chain["has_valid_assigned_rna_qs_global"].fillna(0).sum())},
    ])
    summary.to_csv(out_dir / f"{db_tag}__per_chain_summary.csv", index=False)

    print(f"[{db_tag}] per-chain rows: {len(per_chain)} -> {all_path}")
    print(f"[{db_tag}] 1v1 rows: {len(one_v_one)} -> {one_v_one_path}")
    print(f"[{db_tag}] multichain rows: {len(multi)} -> {multi_path}")
    print(f"[{db_tag}] matched rows: {int(per_chain['is_matched'].fillna(0).sum())}")
    print(f"[{db_tag}] valid TM rows: {int(per_chain['has_valid_assigned_rna_tm'].fillna(0).sum())}")
    print(f"[{db_tag}] valid LDDT rows: {int(per_chain['has_valid_assigned_rna_lddt'].fillna(0).sum())}")
    print(f"[{db_tag}] valid DockQ rows: {int(per_chain['has_valid_assigned_rna_dockq'].fillna(0).sum())}")
    print(f"[{db_tag}] valid ICS rows: {int(per_chain['has_valid_assigned_rna_ics'].fillna(0).sum())}")
    print(f"[{db_tag}] valid iLDDT rows: {int(per_chain['has_valid_assigned_rna_ilddt'].fillna(0).sum())}")
    print(f"[{db_tag}] valid QS rows: {int(per_chain['has_valid_assigned_rna_qs_global'].fillna(0).sum())}")
    print("Done.")

    return per_chain, one_v_one, multi


def write_combined_outputs(out_root: Path, all_per_chain: List[pd.DataFrame], all_one_v_one: List[pd.DataFrame], all_multi: List[pd.DataFrame]) -> None:
    comb_dir = out_root / "combined"
    comb_dir.mkdir(parents=True, exist_ok=True)

    per_chain = pd.concat(all_per_chain, axis=0, ignore_index=True) if all_per_chain else pd.DataFrame()
    one_v_one = pd.concat(all_one_v_one, axis=0, ignore_index=True) if all_one_v_one else pd.DataFrame()
    multi = pd.concat(all_multi, axis=0, ignore_index=True) if all_multi else pd.DataFrame()

    if not per_chain.empty:
        per_chain.to_csv(comb_dir / "combined__per_chain_all.csv", index=False)
    if not one_v_one.empty:
        one_v_one.to_csv(comb_dir / "combined__per_chain_rna_protein_1v1.csv", index=False)
    if not multi.empty:
        multi.to_csv(comb_dir / "combined__per_chain_multichain_complex.csv", index=False)

    if not per_chain.empty:
        summary = pd.DataFrame([
            {"subset": "per_chain_all", "n_rows": len(per_chain)},
            {"subset": "per_chain_rna_protein_1v1", "n_rows": len(one_v_one)},
            {"subset": "per_chain_multichain_complex", "n_rows": len(multi)},
            {"subset": "per_chain_matched", "n_rows": int(per_chain["is_matched"].fillna(0).sum())},
            {"subset": "per_chain_valid_tm", "n_rows": int(per_chain["has_valid_assigned_rna_tm"].fillna(0).sum())},
            {"subset": "per_chain_valid_lddt", "n_rows": int(per_chain["has_valid_assigned_rna_lddt"].fillna(0).sum())},
            {"subset": "per_chain_valid_dockq", "n_rows": int(per_chain["has_valid_assigned_rna_dockq"].fillna(0).sum())},
            {"subset": "per_chain_valid_ics", "n_rows": int(per_chain["has_valid_assigned_rna_ics"].fillna(0).sum())},
            {"subset": "per_chain_valid_ilddt", "n_rows": int(per_chain["has_valid_assigned_rna_ilddt"].fillna(0).sum())},
            {"subset": "per_chain_valid_qs_global", "n_rows": int(per_chain["has_valid_assigned_rna_qs_global"].fillna(0).sum())},
        ])
        summary.to_csv(comb_dir / "combined__per_chain_summary.csv", index=False)

    print(f"[combined] per-chain rows: {len(per_chain)}")
    print(f"[combined] 1v1 rows: {len(one_v_one)}")
    print(f"[combined] multichain rows: {len(multi)}")
    if not per_chain.empty:
        print(f"[combined] matched rows: {int(per_chain['is_matched'].fillna(0).sum())}")
        print(f"[combined] valid TM rows: {int(per_chain['has_valid_assigned_rna_tm'].fillna(0).sum())}")
        print(f"[combined] valid LDDT rows: {int(per_chain['has_valid_assigned_rna_lddt'].fillna(0).sum())}")
        print(f"[combined] valid DockQ rows: {int(per_chain['has_valid_assigned_rna_dockq'].fillna(0).sum())}")
        print(f"[combined] valid ICS rows: {int(per_chain['has_valid_assigned_rna_ics'].fillna(0).sum())}")
        print(f"[combined] valid iLDDT rows: {int(per_chain['has_valid_assigned_rna_ilddt'].fillna(0).sum())}")
        print(f"[combined] valid QS rows: {int(per_chain['has_valid_assigned_rna_qs_global'].fillna(0).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", required=True, help="SQLite db files")
    ap.add_argument("--outdir", default="csvs4analysis_per_chain", help="Output folder")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_per_chain: List[pd.DataFrame] = []
    all_one_v_one: List[pd.DataFrame] = []
    all_multi: List[pd.DataFrame] = []

    for db in args.dbs:
        per_chain, one_v_one, multi = process_db(Path(db), out_root)
        all_per_chain.append(per_chain)
        all_one_v_one.append(one_v_one)
        all_multi.append(multi)

    write_combined_outputs(out_root, all_per_chain, all_one_v_one, all_multi)


if __name__ == "__main__":
    main()