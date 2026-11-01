
#!/usr/bin/env python3
"""
dbs2analysis_csvs_with_interface.py

Builds the complex-level analysis CSVs, preserving the previous output CSVs and
schema while adding complex-level interface summary features derived from
pred_protein_rna interface fields.

New interface-derived features are computed ONLY from pred_protein_rna, so they
can be used for downstream prediction models/tools.

Outputs remain:
- <dbtag>__rna_protein_1v1.csv
- <dbtag>__multichain_complex.csv
- combined/combined__rna_protein_1v1.csv
- combined/combined__multichain_complex.csv

The new interface summary columns are appended to those CSVs.
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


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
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
        return sum(1 for a in v if isinstance(a, str) and a.strip())
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


def ensure_list(x: Any) -> List[Any]:
    v = safe_literal_eval(x)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def normalize_missing_token(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return s


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
    uniq = len(set(seq))
    return uniq / len(seq)


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
            f"{prefix}_n_modified_bases": np.nan,
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
        f"{prefix}_n_modified_bases": np.nan,
    }


def parse_loop_count(x: Any) -> float:
    v = safe_literal_eval(x)
    return float(len(v)) if isinstance(v, list) else np.nan


def single_rna_only(series: pd.Series, n_rna: pd.Series) -> pd.Series:
    out = series.copy()
    mask = pd.to_numeric(n_rna, errors="coerce") != 1
    out.loc[mask] = np.nan
    return out


# -----------------------------------------------------------------------------
# Generic exp_Solution / exp_Sample_Conc parser
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Buffer / PEG / ionic strength parsing
# -----------------------------------------------------------------------------
PEG_MW_PATTERN = re.compile(
    r"(?:PEG|PEMME|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)"
    r"[-_\s]?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]{2,5})(?=[^\d]|$)",
    flags=re.I
)

PEG_K_PATTERN = re.compile(
    r"(?:PEG|polyethylene\s+glycol)[-_\s]?([0-9]{1,2})\s*[Kk](?=\b|[^0-9])",
    flags=re.I
)

SALTS = {
    "sodium chloride": (1, -1),
    "potassium chloride": (1, -1),
    "lithium chloride": (1, -1),
    "magnesium chloride": (2, -1),
    "manganese chloride": (2, -1),
    "calcium chloride": (2, -1),
    "ammonium sulfate": (1, -2),
    "lithium sulfate": (1, -2),
    "sodium sulfate": (1, -2),
    "potassium sulfate": (1, -2),
    "sodium acetate": (1, -1),
    "potassium acetate": (1, -1),
    "ammonium acetate": (1, -1),
    "magnesium acetate": (2, -1),
    "calcium acetate": (2, -1),
    "sodium formate": (1, -1),
    "magnesium formate": (2, -1),
    "sodium citrate": (1, -3),
    "trisodium citrate": (1, -3),
    "ammonium citrate": (1, -3),
    "ammonium tartrate": (1, -2),
    "potassium sodium tartrate": (1, -2),
    "sodium phosphate": (1, -2),
    "potassium phosphate": (1, -2),
    "ammonium phosphate": (1, -2),
    "sodium carbonate": (1, -2),
    "potassium carbonate": (1, -2),
    "ammonium carbonate": (1, -2),
    "disodium succinate": (1, -2),
    "sodium succinate": (1, -2),
    "potassium bromide": (1, -1),
    "sodium bromide": (1, -1),
    "potassium iodide": (1, -1),
    "sodium iodide": (1, -1),
    "ammonium fluoride": (1, -1),
}

ALIASES = {
    "nacl": "sodium chloride", "na cl": "sodium chloride",
    "kcl": "potassium chloride", "k cl": "potassium chloride",
    "licl": "lithium chloride", "li cl": "lithium chloride",
    "mgcl2": "magnesium chloride", "mg cl2": "magnesium chloride",
    "mncl2": "manganese chloride", "mn cl2": "manganese chloride",
    "cacl2": "calcium chloride", "ca cl2": "calcium chloride",
    "(nh4)2so4": "ammonium sulfate", "nh4 sulfate": "ammonium sulfate",
    "ammonium sulfate": "ammonium sulfate",
    "liso4": "lithium sulfate", "li2so4": "lithium sulfate", "lis04": "lithium sulfate",
    "na2so4": "sodium sulfate",
    "k2so4": "potassium sulfate",
    "na acetate": "sodium acetate", "naacetate": "sodium acetate",
    "k acetate": "potassium acetate",
    "ammonium acetate": "ammonium acetate",
    "magnesium acetate tetrahydrate": "magnesium acetate",
    "na formate": "sodium formate", "mg formate": "magnesium formate",
    "mg-formate": "magnesium formate",
    "na citrate": "sodium citrate", "tri-sodium citrate": "trisodium citrate",
    "citrate": "sodium citrate",
    "ammonium tartrate": "ammonium tartrate",
    "nh4/tartrate": "ammonium tartrate", "nh4 tartrate": "ammonium tartrate",
    "potassium sodium tartrate": "potassium sodium tartrate",
    "potasium sodium tartrate": "potassium sodium tartrate",
    "sodium potassium tartrate": "potassium sodium tartrate",
    "sodium phosphate": "sodium phosphate",
    "na h2po4": "sodium phosphate",
    "na2 hpo4": "sodium phosphate",
    "k phosphate": "potassium phosphate",
    "na2co3": "sodium carbonate", "k2co3": "potassium carbonate",
    "na2 succinate": "disodium succinate",
    "kbr": "potassium bromide", "nabr": "sodium bromide",
    "ki": "potassium iodide", "nai": "sodium iodide",
    "naf": "ammonium fluoride",
}

BUFFERS = [
    "tris", "tris-hcl", "tris(hydroxymethyl)aminomethane",
    "hepes", "mes", "bicine", "imidazole",
    "cacodylate", "cacodyl", "sodium cacodylate", "sodium cacodylate trihydrate",
    "bis-tris", "bis-tris propane", "bis-trispropane",
    "citric", "citrate",
    "phosphate", "sodium phosphate", "potassium phosphate",
    "acetate", "formate", "tartrate",
    "pH", "mib", "ches", "ada", "succinate",
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

    pattern_with_space = re.compile(
        r"(\d+\.?\d*)\s+(m|mm)\s+([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)",
        flags=re.I,
    )
    pattern_no_space = re.compile(
        r"(\d+\.?\d*)(m|mm)([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)",
        flags=re.I,
    )

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
        re.compile(
            r"(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)"
            r"[-_\s]?([0-9]{1,2})\s*[Kk]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
            flags=re.I,
        ),
        re.compile(
            r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:\(w/v\))?\s*"
            r"(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)"
            r"[-_\s]?([0-9]{2,5})",
            flags=re.I,
        ),
        re.compile(
            r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*Polyethylene\s+glycol\s+([0-9]{2,5})",
            flags=re.I,
        ),
        re.compile(
            r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:.*?\s)?"
            r"(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)",
            flags=re.I,
        ),
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

    pattern_molarity_first = re.compile(
        r"([0-9]*\.?[0-9]+)\s*(mM|M)\s*([A-Za-z0-9\-/\s\(\)\.]*)",
        flags=re.I,
    )
    pattern_buffer_first = re.compile(
        r"([A-Za-z0-9\-/\s\(\)\.]+?)\s*([0-9]*\.?[0-9]+)\s*(mM|M)",
        flags=re.I,
    )

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


# -----------------------------------------------------------------------------
# Ions / ligands: support features
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Interface parsing / feature engineering
# -----------------------------------------------------------------------------
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
    """
    Returns (list_of_contact_lists, was_flat_single_pair).
    Handles:
    - [["...","..."], ["..."]]  -> unchanged
    - ["...","..."] with n_pairs == 1 -> [["...","..."]]
    """
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

    # Fallback: best-effort; wrap flat items as a single pair if lengths disagree
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
    if vals is None:
        return [None] * n_pairs
    if isinstance(vals, list):
        return list(vals)
    if n_pairs == 1:
        return [vals]
    return [None] * n_pairs


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


def compute_interface_features_for_row(row: pd.Series) -> Dict[str, Any]:
    # Chain-level maps
    protein_chain_ids = _normalize_chain_ids(row.get("ProteinChainIDs"), row.get("NumberProteins"))
    rna_chain_ids = _normalize_chain_ids(row.get("RNAChainIDs"), row.get("NumberRNAs"))
    protein_len_map = _normalize_lengths(row.get("ProteinLength"), protein_chain_ids, row.get("ProteinSequence"), kind="protein")
    rna_len_map = _normalize_lengths(row.get("RNALength"), rna_chain_ids, row.get("RNASequence"), kind="rna")

    pair_list = _normalize_pair_list(row.get("ChainIDpairList_proteinRNA"))
    n_pairs = len(pair_list)

    hbond_list = _normalize_scalar_or_list_by_pairs(row.get("Hbond_proteinRNA"), n_pairs)
    vdw_list = _normalize_scalar_or_list_by_pairs(row.get("vdWbond_proteinRNA"), n_pairs)
    motif_len_list = _normalize_scalar_or_list_by_pairs(row.get("RNAMotifLength"), n_pairs)
    contact_blocks, contactlist_was_flat_single_pair = _normalize_nested_by_pairs(row.get("ContactList"), n_pairs)

    out: Dict[str, Any] = {
        "n_protein_rna_interface_pairs": n_pairs,
        "qc_contactlist_was_flat_single_pair": int(contactlist_was_flat_single_pair),
        "qc_interface_lengths_match": np.nan,
        "qc_n_pairs_failed_contact_parse": 0,
        "qc_frac_pairs_failed_contact_parse": np.nan,
        "qc_any_interface_parse_problem": 0,
    }

    if n_pairs == 0:
        # Keep a wide but sparse interface schema
        for col in [
            "n_pairs_with_contacts", "frac_pairs_with_contacts",
            "total_hbond_proteinRNA", "total_vdwbond_proteinRNA", "total_contact_entries",
            "n_unique_protein_interface_residues_global", "n_unique_rna_interface_residues_global",
            "protein_interface_coverage_global", "rna_interface_coverage_global",
            "interface_contact_distance_mean", "interface_contact_distance_median",
            "interface_contact_distance_min", "interface_contact_distance_max",
            "interface_contact_distance_std", "frac_contacts_lt_2_5A",
            "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
            "interface_max_pair_contact_fraction", "interface_hbond_per_contact",
            "interface_vdw_per_contact", "interface_hbond_vdw_ratio",
            "interface_rna_motif_length_sum", "interface_rna_motif_length_mean",
            "interface_rna_motif_length_max", "interface_rna_motif_coverage_of_total_rna",
        ]:
            out[col] = np.nan
        return out

    qc_lengths_ok = (
        len(hbond_list) == n_pairs and
        len(vdw_list) == n_pairs and
        len(contact_blocks) == n_pairs and
        len(motif_len_list) == n_pairs
    )
    out["qc_interface_lengths_match"] = int(qc_lengths_ok)

    pair_hbonds: List[float] = []
    pair_vdws: List[float] = []
    pair_contact_counts: List[float] = []
    pair_unique_prot_counts: List[float] = []
    pair_unique_rna_counts: List[float] = []
    pair_prot_spans: List[float] = []
    pair_rna_spans: List[float] = []
    pair_mean_distances: List[float] = []
    pair_min_distances: List[float] = []
    pair_motif_lengths: List[float] = []
    pair_prot_coverages: List[float] = []
    pair_rna_coverages: List[float] = []

    global_protein_residues: Set[Tuple[str, int]] = set()
    global_rna_residues: Set[Tuple[str, int]] = set()
    all_distances: List[float] = []

    nt_class_counts = {"A": 0, "C": 0, "G": 0, "U": 0, "modified": 0}
    aa_class_counts = {"positive": 0, "aromatic": 0, "polar": 0, "hydrophobic": 0, "acidic": 0, "other": 0}

    pairs_with_contacts = 0
    pairs_failed_contact_parse = 0

    for i, pair in enumerate(pair_list):
        p_chain, r_chain = pair
        hbond = to_float(hbond_list[i]) if i < len(hbond_list) else np.nan
        vdw = to_float(vdw_list[i]) if i < len(vdw_list) else np.nan
        motif_len = to_float(motif_len_list[i]) if i < len(motif_len_list) else np.nan
        raw_contacts = contact_blocks[i] if i < len(contact_blocks) else []
        parsed_contacts = []
        parse_failures = 0

        for item in raw_contacts:
            parsed = _parse_contact_item(item)
            if parsed is None:
                parse_failures += 1
                continue
            parsed_contacts.append(parsed)

        if len(raw_contacts) > 0 and len(parsed_contacts) == 0:
            pairs_failed_contact_parse += 1

        n_contacts = len(parsed_contacts)
        if n_contacts > 0:
            pairs_with_contacts += 1

        protein_positions = [c["aa_pos"] for c in parsed_contacts]
        rna_positions = [c["nt_pos"] for c in parsed_contacts]
        distances = [c["distance"] for c in parsed_contacts]

        unique_prot = len(set(protein_positions))
        unique_rna = len(set(rna_positions))
        prot_span = (max(protein_positions) - min(protein_positions)) if protein_positions else np.nan
        rna_span = (max(rna_positions) - min(rna_positions)) if rna_positions else np.nan
        mean_dist = float(np.mean(distances)) if distances else np.nan
        min_dist = float(np.min(distances)) if distances else np.nan

        p_len = protein_len_map.get(p_chain, np.nan)
        r_len = rna_len_map.get(r_chain, np.nan)

        pair_hbonds.append(hbond)
        pair_vdws.append(vdw)
        pair_contact_counts.append(float(n_contacts))
        pair_unique_prot_counts.append(float(unique_prot) if n_contacts > 0 else np.nan)
        pair_unique_rna_counts.append(float(unique_rna) if n_contacts > 0 else np.nan)
        pair_prot_spans.append(float(prot_span) if n_contacts > 0 else np.nan)
        pair_rna_spans.append(float(rna_span) if n_contacts > 0 else np.nan)
        pair_mean_distances.append(mean_dist)
        pair_min_distances.append(min_dist)
        pair_motif_lengths.append(motif_len)
        pair_prot_coverages.append(safe_div(unique_prot, p_len))
        pair_rna_coverages.append(safe_div(unique_rna, r_len))

        for c in parsed_contacts:
            global_protein_residues.add((p_chain, c["aa_pos"]))
            global_rna_residues.add((r_chain, c["nt_pos"]))
            all_distances.append(c["distance"])
            nt_class_counts[c["nt_class"]] = nt_class_counts.get(c["nt_class"], 0) + 1
            aa_class_counts[c["aa_class"]] = aa_class_counts.get(c["aa_class"], 0) + 1

    out["qc_n_pairs_failed_contact_parse"] = pairs_failed_contact_parse
    out["qc_frac_pairs_failed_contact_parse"] = safe_div(pairs_failed_contact_parse, n_pairs)
    out["qc_any_interface_parse_problem"] = int((not qc_lengths_ok) or pairs_failed_contact_parse > 0)

    out["n_pairs_with_contacts"] = pairs_with_contacts
    out["frac_pairs_with_contacts"] = safe_div(pairs_with_contacts, n_pairs)

    # Basic pair-count summaries
    out["total_hbond_proteinRNA"] = np.nansum(pair_hbonds) if pair_hbonds else np.nan
    out["total_vdwbond_proteinRNA"] = np.nansum(pair_vdws) if pair_vdws else np.nan
    out["total_contact_entries"] = np.nansum(pair_contact_counts) if pair_contact_counts else np.nan

    for k, v in _summary_stats(pair_hbonds, "pair_hbond_count").items():
        out[k] = v
    for k, v in _summary_stats(pair_vdws, "pair_vdw_count").items():
        out[k] = v
    for k, v in _summary_stats(pair_contact_counts, "pair_contact_entries").items():
        out[k] = v
    for k, v in _summary_stats(pair_unique_prot_counts, "pair_unique_protein_interface_residues").items():
        out[k] = v
    for k, v in _summary_stats(pair_unique_rna_counts, "pair_unique_rna_interface_residues").items():
        out[k] = v
    for k, v in _summary_stats(pair_prot_spans, "pair_protein_interface_span").items():
        out[k] = v
    for k, v in _summary_stats(pair_rna_spans, "pair_rna_interface_span").items():
        out[k] = v
    for k, v in _summary_stats(pair_mean_distances, "pair_mean_contact_distance").items():
        out[k] = v
    for k, v in _summary_stats(pair_min_distances, "pair_min_contact_distance").items():
        out[k] = v
    for k, v in _summary_stats(pair_motif_lengths, "interface_rna_motif_length").items():
        out[k] = v
    for k, v in _summary_stats(pair_prot_coverages, "pair_protein_interface_coverage").items():
        out[k] = v
    for k, v in _summary_stats(pair_rna_coverages, "pair_rna_interface_coverage").items():
        out[k] = v

    # Global unique residue coverage
    total_prot_len = np.nansum(list(protein_len_map.values())) if protein_len_map else np.nan
    total_rna_len = np.nansum(list(rna_len_map.values())) if rna_len_map else np.nan

    out["n_unique_protein_interface_residues_global"] = len(global_protein_residues)
    out["n_unique_rna_interface_residues_global"] = len(global_rna_residues)
    out["protein_interface_coverage_global"] = safe_div(len(global_protein_residues), total_prot_len)
    out["rna_interface_coverage_global"] = safe_div(len(global_rna_residues), total_rna_len)

    # Distances
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
            "interface_contact_distance_std", "frac_contacts_lt_2_5A",
            "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
        ]:
            out[col] = np.nan

    # Aggregate ratios / imbalance
    total_contacts = out["total_contact_entries"]
    total_h = out["total_hbond_proteinRNA"]
    total_v = out["total_vdwbond_proteinRNA"]

    out["interface_max_pair_contact_fraction"] = safe_div(np.nanmax(pair_contact_counts) if len(pair_contact_counts) else np.nan, total_contacts)
    out["interface_max_pair_hbond_fraction"] = safe_div(np.nanmax(pair_hbonds) if len(pair_hbonds) else np.nan, total_h)
    out["interface_max_pair_vdw_fraction"] = safe_div(np.nanmax(pair_vdws) if len(pair_vdws) else np.nan, total_v)
    out["interface_hbond_per_contact"] = safe_div(total_h, total_contacts)
    out["interface_vdw_per_contact"] = safe_div(total_v, total_contacts)
    out["interface_hbond_vdw_ratio"] = safe_div(total_h, total_v)
    out["interface_rna_motif_coverage_of_total_rna"] = safe_div(out["interface_rna_motif_length_sum"], total_rna_len)

    # Composition features
    total_obs = sum(nt_class_counts.values())
    for nt in ["A", "C", "G", "U", "modified"]:
        out[f"interface_n_contacts_to_{nt}"] = nt_class_counts[nt]
        out[f"interface_frac_contacts_to_{nt}"] = safe_div(nt_class_counts[nt], total_obs)

    total_aa_obs = sum(aa_class_counts.values())
    for aa_cls in ["positive", "aromatic", "polar", "hydrophobic", "acidic", "other"]:
        out[f"interface_n_{aa_cls}_contacts"] = aa_class_counts[aa_cls]
        out[f"interface_frac_{aa_cls}_contacts"] = safe_div(aa_class_counts[aa_cls], total_aa_obs)

    return out



def _dockq_nr_from_row(rbp_cell: Any, pair_cell: Any) -> Tuple[float, int]:
    """Complex-level DockQ_Nr: mean of RBP_DockQ over the protein-RNA chain pairs only.

    RBP_DockQ is a JSON keyed dict-list [{"A:B": 0.5}, ...]; ChainIDpairList_proteinRNA is
    [["A","B"], ...]. Only pairs listed there (order-insensitive) contribute; if that list is
    empty/missing, all RBP_DockQ pairs are used. Returns (mean, n_pairs); (NaN, 0) if none.
    Same definition as publication_figures/post_pre_comparison_figure._dockq_nr and the
    interface classification (Appendix H.4)."""
    allowed = set()
    pl = safe_literal_eval(pair_cell)
    if isinstance(pl, list):
        for pr in pl:
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                allowed.add(frozenset(str(x).strip() for x in pr))
    obj = safe_literal_eval(rbp_cell)
    vals: List[float] = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict):
                for k, v in el.items():
                    parts = [t.strip() for t in str(k).split(":")]
                    if len(parts) == 2 and (not allowed or frozenset(parts) in allowed):
                        fv = to_float(v)
                        if not np.isnan(fv):
                            vals.append(float(fv))
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)

def add_interface_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    row_df = pd.DataFrame(index=df.index)

    # After merging pred/exp, these source fields are typically suffixed with _pred.
    source_map = {
        "ProteinChainIDs": "ProteinChainIDs_pred" if "ProteinChainIDs_pred" in df.columns else "ProteinChainIDs",
        "ProteinLength": "ProteinLength_pred" if "ProteinLength_pred" in df.columns else "ProteinLength",
        "ProteinSequence": "ProteinSequence_pred" if "ProteinSequence_pred" in df.columns else "ProteinSequence",
        "RNAChainIDs": "RNAChainIDs_pred" if "RNAChainIDs_pred" in df.columns else "RNAChainIDs",
        "RNALength": "RNALength_pred" if "RNALength_pred" in df.columns else "RNALength",
        "RNASequence": "RNASequence_pred" if "RNASequence_pred" in df.columns else "RNASequence",
        "NumberProteins": "NumberProteins",
        "NumberRNAs": "NumberRNAs",
        "ChainIDpairList_proteinRNA": "ChainIDpairList_proteinRNA",
        "ContactList": "ContactList",
        "Hbond_proteinRNA": "Hbond_proteinRNA",
        "vdWbond_proteinRNA": "vdWbond_proteinRNA",
        "RNAMotifLength": "RNAMotifLength",
    }

    missing = [src for src in source_map.values() if src not in df.columns]
    if missing:
        stub_cols = {
            "n_protein_rna_interface_pairs": np.nan,
            "n_pairs_with_contacts": np.nan,
            "frac_pairs_with_contacts": np.nan,
            "total_hbond_proteinRNA": np.nan,
            "total_vdwbond_proteinRNA": np.nan,
            "total_contact_entries": np.nan,
        }
        return pd.concat([df, pd.DataFrame([stub_cols] * len(df), index=df.index)], axis=1)

    for canonical, src in source_map.items():
        row_df[canonical] = df[src]

    rows = [compute_interface_features_for_row(row) for _, row in row_df.iterrows()]
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


# -----------------------------------------------------------------------------
# Table loading and feature extraction
# -----------------------------------------------------------------------------
PRED_COLUMNS = [
    "exp_db_id",
    "ProteinSequence",
    "ProteinChainIDs",
    "ProteinLength",
    "RNASequence",
    "RNAChainIDs",
    "RNALength",
    "NumberProteins",
    "NumberRNAs",
    "AAMotif",
    "AAC",
    "AAPproteinRNA",
    "RNAMotif",
    "RNAMotifLength",
    "RNAModified",
    "ContactList",
    "ChainIDpairList_proteinRNA",
    "RBP_DockQ",
    "Hbond_proteinRNA",
    "vdWbond_proteinRNA",
    "RBP_PDB_Templates",
    "Protein_PDB_Templates",
    "af3_protein_pTM",
    "af3_rna_pTM",
    "af3_protein_ipTM",
    "af3_rna_ipTM",
    "af3_protein_pLDDT_avg",
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_ranking_score",
    "ProteinRNAInterfaceArea",
    "ProteinRNAInterfaceRatio",
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
    # RNA intrinsic / structure / thermodynamics
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
    "RNA_msa_size",
    "Protein_msa_size",
    "AAmatch_score",
    "RNAmatch_score",
]

EXP_COLUMNS = [
    "PDBId",
    "ProteinSequence",
    "ProteinChainIDs",
    "ProteinLength",
    "RNASequence",
    "RNAChainIDs",
    "RNALength",
    "NumberProteins",
    "NumberRNAs",
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
    "RNAFamily",
    "ProteinFamily",
    "ProteinRNAInterfaceArea",
    "ProteinRNAInterfaceRatio",
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
]


def load_table_columns(conn: sqlite3.Connection, table: str, wanted: List[str]) -> pd.DataFrame:
    existing = pd.read_sql_query(f"PRAGMA table_info({table})", conn)["name"].tolist()
    keep = [c for c in wanted if c in existing]
    if not keep:
        raise ValueError(f"No requested columns found in table '{table}'.")
    return pd.read_sql_query(f"SELECT {', '.join(keep)} FROM {table}", conn)


def collect_dynamic_names(series: pd.Series, parser) -> List[str]:
    names: Set[str] = set()
    for x in series.dropna():
        parsed = parser(x)
        names.update(parsed.keys())
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
        sol_df = pd.DataFrame(sol_rows).reindex(columns=sol_cols)
        df = pd.concat([df, sol_df], axis=1)

    if sample_cols:
        sample_rows = [parse_solution_like(x, "sample") for x in df["exp_Sample_Conc"]]
        sample_df = pd.DataFrame(sample_rows).reindex(columns=sample_cols)
        df = pd.concat([df, sample_df], axis=1)

    return df


def add_buffer_peg_features(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_Solution" in df.columns:
        rows = [parse_buffer_features(x) for x in df["exp_Solution"]]
        df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    return df


def add_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    prot_lengths_all = []
    rna_lengths_all = []
    prot_seq_feat = []
    rna_seq_feat = []

    for _, row in df.iterrows():
        prot_lengths, _ = total_length_from_fields(row.get("ProteinLength_pred"), row.get("ProteinSequence_pred"), kind="protein")
        rna_lengths, _ = total_length_from_fields(row.get("RNALength_pred"), row.get("RNASequence_pred"), kind="rna")

        prot_lengths_all.append(prot_lengths)
        rna_lengths_all.append(rna_lengths)
        prot_seqs = parse_sequence_list(row.get("ProteinSequence_pred"), kind="protein")
        rna_seqs = parse_sequence_list(row.get("RNASequence_pred"), kind="rna")
        prot_seq_feat.append(prot_seqs)
        rna_seq_feat.append(rna_seqs)

    comp_rows = []
    for prot_lengths, rna_lengths, prot_seqs, rna_seqs in zip(prot_lengths_all, rna_lengths_all, prot_seq_feat, rna_seq_feat):
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
        rec.update(compute_rna_sequence_features(rna_seqs, "agg_rna"))
        comp_rows.append(rec)

    df = pd.concat([df, pd.DataFrame(comp_rows)], axis=1)
    return df


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


def add_single_rna_intrinsic_features(df: pd.DataFrame) -> pd.DataFrame:
    intrinsic_cols = [
        "RNA_GlycosidicBond", "RNA_SugarPucker", "RNA_GammaAngle",
        "RNA_Stems", "RNA_HairpinLoops", "RNA_InternalLoops", "RNA_MultibranchLoops",
        "RNA_DanglingEnds", "RNA_Pseudoknots", "RNA_isDuplex", "RNAMotif_isDuplex",
        "RNAMotif_inExp", "RNA_ElectrostaticPotential", "RNA_GC_Content",
        "RNA_SequenceComplexity", "RNA_Shannon_Entropy_K1", "RNA_Shannon_Entropy_K2",
        "RNA_GC_Skew", "RNA_AU_Skew", "RNA_AU_Content", "RNA_MFE_Value",
        "RNA_MFE_Structure", "RNA_Ensemble_Diversity", "RNA_Centroid_Energy",
        "RNA_msa_size", "Protein_msa_size", "AAmatch_score", "RNAmatch_score",
    ]
    for c in intrinsic_cols:
        if c not in df.columns:
            df[c] = np.nan

    for c in [
        "RNA_ElectrostaticPotential", "RNA_GC_Content", "RNA_SequenceComplexity",
        "RNA_Shannon_Entropy_K1", "RNA_Shannon_Entropy_K2", "RNA_GC_Skew",
        "RNA_AU_Skew", "RNA_AU_Content", "RNA_MFE_Value", "RNA_Ensemble_Diversity",
        "RNA_Centroid_Energy", "AAmatch_score", "RNAmatch_score"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = single_rna_only(df[c], df["NumberRNAs"])

    # MSA depth (RNA + protein) for the complex-level evolution group (G1):
    # parse each per-chain JSON dict of alignment sizes and reduce to the
    # geometric-mean depth = mean of log10 over the chains. Computed from the RAW
    # dict columns BEFORE any numeric coercion/overwrite below. Unlike the previous
    # behaviour, MSA is NOT restricted to single-RNA complexes -- multi-chain
    # complexes get a real value. The geometric mean is decoupled from chain count
    # (captured by the complexity group) and mirrors the per-chain RNA log10 form.
    def _geom_mean_log10_depth(x):
        v = safe_literal_eval(x)
        vals = list(v.values()) if isinstance(v, dict) else [v]
        logs = [np.log10(to_float(t)) for t in vals if to_float(t) == to_float(t) and to_float(t) > 0]
        return float(np.mean(logs)) if logs else np.nan
    df["log10_rna_msa_size"] = df["RNA_msa_size"].apply(_geom_mean_log10_depth)
    df["log10_protein_msa_size"] = df["Protein_msa_size"].apply(_geom_mean_log10_depth)
    df["protein_msa_size"] = df["log10_protein_msa_size"].apply(lambda z: float(10.0 ** z) if z == z else np.nan)

    # legacy single-RNA-only raw scalars kept only for the output whitelist (not
    # used by the regression, which reads the log10_* columns above)
    df["RNA_msa_size"] = single_rna_only(df["RNA_msa_size"].apply(first_list_value if df["RNA_msa_size"].dtype == object else lambda x: to_float(x)), df["NumberRNAs"])
    df["Protein_msa_size"] = pd.to_numeric(df["Protein_msa_size"], errors="coerce")
    df["Protein_msa_size"] = single_rna_only(df["Protein_msa_size"], df["NumberRNAs"])

    # Complex-level protein/RNA confidence scalars: AF3 emits per-chain pTM/ipTM as
    # JSON lists; reduce each to its mean over chains so G0 can include the
    # protein-side confidence (protein pTM/ipTM) and RNA interface confidence
    # (rna ipTM) that the confidence analysis flags as the strongest complex-accuracy
    # correlates (protein ipTM vs cTM rho~0.69; protein pLDDT vs cLDDT rho~0.75).
    def _mean_list(x):
        v = safe_literal_eval(x)
        vals = v if isinstance(v, list) else [v]
        nums = [to_float(t) for t in vals if to_float(t) == to_float(t)]
        return float(np.mean(nums)) if nums else np.nan
    df["af3_protein_pTM_mean"] = df["af3_protein_pTM"].apply(_mean_list)
    df["af3_protein_ipTM_mean"] = df["af3_protein_ipTM"].apply(_mean_list)
    df["af3_rna_ipTM_mean"] = df["af3_rna_ipTM"].apply(_mean_list)

    df["num_stems"] = single_rna_only(df["RNA_Stems"].apply(parse_loop_count), df["NumberRNAs"])
    df["num_hairpin_loops"] = single_rna_only(df["RNA_HairpinLoops"].apply(parse_loop_count), df["NumberRNAs"])
    df["num_internal_loops"] = single_rna_only(df["RNA_InternalLoops"].apply(parse_loop_count), df["NumberRNAs"])
    df["num_multibranch_loops"] = single_rna_only(df["RNA_MultibranchLoops"].apply(parse_loop_count), df["NumberRNAs"])
    df["num_dangling_ends"] = single_rna_only(df["RNA_DanglingEnds"].apply(parse_loop_count), df["NumberRNAs"])
    df["num_pseudoknots"] = single_rna_only(df["RNA_Pseudoknots"].apply(parse_loop_count), df["NumberRNAs"])

    for base, new in [
        ("num_stems", "stems_per_100nt"),
        ("num_hairpin_loops", "hairpin_loops_per_100nt"),
        ("num_internal_loops", "internal_loops_per_100nt"),
        ("num_multibranch_loops", "multibranch_loops_per_100nt"),
        ("num_dangling_ends", "dangling_ends_per_100nt"),
        ("num_pseudoknots", "pseudoknots_per_100nt"),
    ]:
        df[new] = df[base] / (df["rna_total_length"].replace(0, np.nan) / 100.0)

    return df


def parse_features_and_metrics(merged: pd.DataFrame, db_tag: str) -> pd.DataFrame:
    df = merged.copy()

    df["pdb_id"] = df["exp_db_id"].astype(str)
    df["source_db"] = db_tag
    df["is_post_cutoff"] = infer_post_cutoff_from_db_tag(db_tag)

    df["NumberProteins"] = pd.to_numeric(df.get("NumberProteins_pred", df.get("NumberProteins_exp")), errors="coerce").astype("Int64")
    df["NumberRNAs"] = pd.to_numeric(df.get("NumberRNAs_pred", df.get("NumberRNAs_exp")), errors="coerce").astype("Int64")

    df["n_rbp_templates"] = df["RBP_PDB_Templates"].apply(count_pdb_ids) if "RBP_PDB_Templates" in df.columns else np.nan
    df["n_protein_templates"] = df["Protein_PDB_Templates"].apply(count_pdb_ids) if "Protein_PDB_Templates" in df.columns else np.nan
    df["id_is_rbp_template"] = [
        id_in_template_list(pid, x) for pid, x in zip(df["exp_db_id"], df.get("RBP_PDB_Templates", pd.Series([None] * len(df))))
    ]
    df["id_is_protein_template"] = [
        id_in_template_list(pid, x) for pid, x in zip(df["exp_db_id"], df.get("Protein_PDB_Templates", pd.Series([None] * len(df))))
    ]

    # Numeric metrics
    for c in [
        "af3_protein_pLDDT_avg", "af3_rna_pLDDT_avg", "af3_global_pae_avg",
        "af3_chain_pair_pae_min", "af3_ranking_score", "Complex_TM",
        "Complex_LDDT", "Complex_LDDT_allAtom", "iLDDT", "QS_global",
        "Dockq", "ICS", "RNAComplex_TM", "RNAmotif_score",
        "pred_ProteinRNAInterfaceArea", "pred_ProteinRNAInterfaceRatio",
        "exp_ProteinRNAInterfaceArea", "exp_ProteinRNAInterfaceRatio",
        "Binding_affinity_kd", "RNA_ElectrostaticPotential", "XRayResolution",
        "exp_pH", "exp_Temp",
    ]:
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
    df = add_interface_summary_features(df)
    df = add_single_rna_intrinsic_features(df)

    if "Binding_affinity_kd" in df.columns:
        df["log10_kd"] = np.log10(df["Binding_affinity_kd"].replace(0, np.nan))

    if "Dockq" in df.columns and "af3_ranking_score" in df.columns:
        df["DockQ_minus_AF3rank"] = df["Dockq"] - df["af3_ranking_score"]

    # protein-RNA-only interface DockQ (mean over protein-RNA chain pairs), the DockQ_Nr used
    # in the figures and the classification; whole-complex Dockq is kept alongside.
    if "RBP_DockQ" in df.columns:
        pair_src = "ChainIDpairList_proteinRNA" if "ChainIDpairList_proteinRNA" in df.columns else None
        vals = [_dockq_nr_from_row(r, (df[pair_src].iloc[i] if pair_src else None)) for i, r in enumerate(df["RBP_DockQ"])]
        df["DockQ_Nr"] = [v[0] for v in vals]
        df["DockQ_Nr_n_pairs"] = [v[1] for v in vals]

    return df


def select_output_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "pdb_id", "exp_db_id", "PDBId", "source_db", "is_post_cutoff",
        "NumberProteins", "NumberRNAs",
        # new composition features
        "protein_total_length", "rna_total_length", "total_complex_length",
        "protein_mean_chain_length", "protein_max_chain_length", "protein_min_chain_length",
        "protein_std_chain_length", "protein_chain_length_cv",
        "rna_mean_chain_length", "rna_max_chain_length", "rna_min_chain_length",
        "rna_std_chain_length", "rna_chain_length_cv",
        "protein_residue_fraction", "rna_residue_fraction",
        "protein_to_rna_length_ratio", "max_rna_fraction_of_total_rna", "min_rna_fraction_of_total_rna",
        "agg_rna_gc_content", "agg_rna_au_content", "agg_rna_gc_skew", "agg_rna_au_skew",
        "agg_rna_sequence_complexity", "agg_rna_shannon_entropy_k1", "agg_rna_shannon_entropy_k2",
        # existing template/AF3
        "n_rbp_templates", "n_protein_templates", "id_is_rbp_template", "id_is_protein_template",
        "af3_protein_pLDDT_avg", "af3_rna_pLDDT_avg", "af3_global_pae_avg",
        "af3_chain_pair_pae_min", "af3_ranking_score",
        "af3_protein_pTM_mean", "af3_protein_ipTM_mean", "af3_rna_ipTM_mean",
        "Binding_affinity_kd", "log10_kd", "RNA_ElectrostaticPotential",
        "XRayResolution", "exp_pH", "exp_Temp", "is_xray", "is_em", "is_nmr",
        # ions / ligands
        "n_ions_listed", "n_supported_ions", "n_unsupported_ions", "n_unknown_ions",
        "has_any_ion", "has_any_supported_ion", "has_any_unsupported_ion", "frac_supported_ions", "n_unique_ions",
        "n_ligands_listed", "n_supported_ligands", "n_unsupported_ligands", "n_unknown_ligands",
        "has_any_ligand", "has_any_supported_ligand", "has_any_unsupported_ligand", "frac_supported_ligands", "n_unique_ligands",
        "ionic_strength", "peg_percent", "peg_mw", "buffer_m", "has_peg",
        # interface agreement
        "exp_ProteinRNAInterfaceArea", "pred_ProteinRNAInterfaceArea", "ProteinRNAInterfaceArea_diff_exp_minus_pred",
        "ProteinRNAInterfaceArea_abs_diff", "ProteinRNAInterfaceArea_rel_error", "ProteinRNAInterfaceArea_pred_over_exp",
        "exp_ProteinRNAInterfaceRatio", "pred_ProteinRNAInterfaceRatio", "ProteinRNAInterfaceRatio_diff_exp_minus_pred",
        "ProteinRNAInterfaceRatio_abs_diff", "ProteinRNAInterfaceRatio_rel_error", "ProteinRNAInterfaceRatio_pred_over_exp",
        # new interface summary features
        "n_protein_rna_interface_pairs", "n_pairs_with_contacts", "frac_pairs_with_contacts",
        "total_hbond_proteinRNA", "total_vdwbond_proteinRNA", "total_contact_entries",
        "pair_hbond_count_mean", "pair_hbond_count_median", "pair_hbond_count_min", "pair_hbond_count_max", "pair_hbond_count_std",
        "pair_vdw_count_mean", "pair_vdw_count_median", "pair_vdw_count_min", "pair_vdw_count_max", "pair_vdw_count_std",
        "pair_contact_entries_mean", "pair_contact_entries_median", "pair_contact_entries_min", "pair_contact_entries_max", "pair_contact_entries_std",
        "pair_unique_protein_interface_residues_mean", "pair_unique_protein_interface_residues_max", "pair_unique_protein_interface_residues_std",
        "pair_unique_rna_interface_residues_mean", "pair_unique_rna_interface_residues_max", "pair_unique_rna_interface_residues_std",
        "pair_protein_interface_span_mean", "pair_protein_interface_span_max", "pair_protein_interface_span_std",
        "pair_rna_interface_span_mean", "pair_rna_interface_span_max", "pair_rna_interface_span_std",
        "pair_mean_contact_distance_mean", "pair_mean_contact_distance_median", "pair_mean_contact_distance_min", "pair_mean_contact_distance_max", "pair_mean_contact_distance_std",
        "pair_min_contact_distance_mean", "pair_min_contact_distance_median", "pair_min_contact_distance_min", "pair_min_contact_distance_max", "pair_min_contact_distance_std",
        "interface_contact_distance_mean", "interface_contact_distance_median", "interface_contact_distance_min", "interface_contact_distance_max", "interface_contact_distance_std",
        "frac_contacts_lt_2_5A", "frac_contacts_lt_3_0A", "frac_contacts_lt_3_5A", "frac_contacts_lt_4_0A",
        "n_unique_protein_interface_residues_global", "n_unique_rna_interface_residues_global",
        "protein_interface_coverage_global", "rna_interface_coverage_global",
        "pair_protein_interface_coverage_mean", "pair_rna_interface_coverage_mean",
        "interface_max_pair_contact_fraction", "interface_max_pair_hbond_fraction", "interface_max_pair_vdw_fraction",
        "interface_hbond_per_contact", "interface_vdw_per_contact", "interface_hbond_vdw_ratio",
        "interface_rna_motif_length_sum", "interface_rna_motif_length_mean", "interface_rna_motif_length_max",
        "interface_rna_motif_coverage_of_total_rna",
        "interface_n_contacts_to_A", "interface_n_contacts_to_C", "interface_n_contacts_to_G", "interface_n_contacts_to_U", "interface_n_contacts_to_modified",
        "interface_frac_contacts_to_A", "interface_frac_contacts_to_C", "interface_frac_contacts_to_G", "interface_frac_contacts_to_U", "interface_frac_contacts_to_modified",
        "interface_frac_positive_contacts", "interface_frac_aromatic_contacts", "interface_frac_polar_contacts",
        "interface_frac_hydrophobic_contacts", "interface_frac_acidic_contacts", "interface_frac_other_contacts",
        "qc_contactlist_was_flat_single_pair", "qc_interface_lengths_match",
        "qc_n_pairs_failed_contact_parse", "qc_frac_pairs_failed_contact_parse", "qc_any_interface_parse_problem",
        # intrinsic RNA single-RNA-only features
        "RNA_GC_Content", "RNA_SequenceComplexity", "RNA_Shannon_Entropy_K1", "RNA_Shannon_Entropy_K2",
        "RNA_GC_Skew", "RNA_AU_Skew", "RNA_AU_Content", "RNA_MFE_Value", "RNA_MFE_Structure",
        "RNA_Ensemble_Diversity", "RNA_Centroid_Energy", "RNA_msa_size", "Protein_msa_size",
        "protein_msa_size", "log10_protein_msa_size", "log10_rna_msa_size",
        "num_stems", "num_hairpin_loops", "num_internal_loops", "num_multibranch_loops",
        "num_dangling_ends", "num_pseudoknots", "stems_per_100nt", "hairpin_loops_per_100nt",
        "internal_loops_per_100nt", "multibranch_loops_per_100nt", "dangling_ends_per_100nt",
        "pseudoknots_per_100nt",
        "AAmatch_score", "RNAmatch_score",
        # metrics
        "Complex_RMSD_first", "Complex_TM", "Complex_LDDT", "Complex_LDDT_allAtom", "iLDDT",
        "QS_global", "Dockq", "DockQ_minus_AF3rank", "DockQ_Nr", "DockQ_Nr_n_pairs", "ICS", "RNAComplex_TM", "RNAmotif_score",
    ]

    dynamic = sorted(
        c for c in df.columns
        if c.startswith(("has_unsupported_ion_", "has_unsupported_ligand_", "sol_", "sample_"))
    )

    categorical = [
        c for c in ["ionic_strength_category", "peg_concentration_category", "peg_mw_category", "buffer_category", "buffer_name"]
        if c in df.columns
    ]

    return [c for c in preferred if c in df.columns] + categorical + dynamic


def process_db(db_path: Path, out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_tag = db_path.stem
    out_dir = out_root / db_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        pred = load_table_columns(conn, "pred_protein_rna", PRED_COLUMNS)
        exp = load_table_columns(conn, "exp_protein_rna", EXP_COLUMNS)

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

    keep_cols = select_output_columns(parsed)
    parsed = parsed[keep_cols].copy()

    print("\nComposition counts:")
    print(
        parsed.groupby(["NumberProteins", "NumberRNAs"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["NumberProteins", "NumberRNAs"])
        .to_string(index=False)
    )

    is_rna_protein = (parsed["NumberProteins"] >= 1) & (parsed["NumberRNAs"] >= 1)
    is_1v1 = (parsed["NumberProteins"] == 1) & (parsed["NumberRNAs"] == 1)
    is_multichain = is_rna_protein & ~is_1v1

    one_v_one = parsed[is_1v1].copy()
    multi = parsed[is_multichain].copy()

    one_v_one_path = out_dir / f"{db_tag}__rna_protein_1v1.csv"
    multi_path = out_dir / f"{db_tag}__multichain_complex.csv"

    one_v_one.to_csv(one_v_one_path, index=False)
    multi.to_csv(multi_path, index=False)

    summary = pd.DataFrame(
        [
            {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "all_merged", "n_rows": len(parsed)},
            {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "rna_protein_any", "n_rows": int(is_rna_protein.sum())},
            {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "rna_protein_1v1", "n_rows": len(one_v_one)},
            {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "multichain_complex", "n_rows": len(multi)},
        ]
    )
    summary.to_csv(out_dir / f"{db_tag}__subset_summary.csv", index=False)

    print(f"[{db_tag}] merged rows: {len(parsed)}")
    print(f"[{db_tag}] any RNA-protein rows: {int(is_rna_protein.sum())}")
    print(f"[{db_tag}] 1v1 rows: {len(one_v_one)} -> {one_v_one_path}")
    print(f"[{db_tag}] multichain rows: {len(multi)} -> {multi_path}")
    print("Done.")

    return parsed, one_v_one, multi


def write_combined_outputs(
    out_root: Path,
    all_parsed: List[pd.DataFrame],
    all_one_v_one: List[pd.DataFrame],
    all_multi: List[pd.DataFrame],
) -> None:
    comb_dir = out_root / "combined"
    comb_dir.mkdir(parents=True, exist_ok=True)

    parsed = pd.concat(all_parsed, axis=0, ignore_index=True) if all_parsed else pd.DataFrame()
    one_v_one = pd.concat(all_one_v_one, axis=0, ignore_index=True) if all_one_v_one else pd.DataFrame()
    multi = pd.concat(all_multi, axis=0, ignore_index=True) if all_multi else pd.DataFrame()

    if not one_v_one.empty:
        one_v_one.to_csv(comb_dir / "combined__rna_protein_1v1.csv", index=False)
    if not multi.empty:
        multi.to_csv(comb_dir / "combined__multichain_complex.csv", index=False)

    if not parsed.empty:
        summary = pd.DataFrame(
            [
                {"subset": "all_merged", "n_rows": len(parsed)},
                {"subset": "rna_protein_any", "n_rows": int(((parsed["NumberProteins"] >= 1) & (parsed["NumberRNAs"] >= 1)).sum())},
                {"subset": "rna_protein_1v1", "n_rows": len(one_v_one)},
                {"subset": "multichain_complex", "n_rows": len(multi)},
            ]
        )
        summary.to_csv(comb_dir / "combined__subset_summary.csv", index=False)

    print(f"[combined] all_merged rows: {len(parsed)}")
    print(f"[combined] 1v1 rows: {len(one_v_one)} -> {comb_dir / 'combined__rna_protein_1v1.csv'}")
    print(f"[combined] multichain rows: {len(multi)} -> {comb_dir / 'combined__multichain_complex.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", required=True, help="SQLite db files")
    ap.add_argument("--outdir", default="csvs4analysis", help="Output folder")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_parsed: List[pd.DataFrame] = []
    all_one_v_one: List[pd.DataFrame] = []
    all_multi: List[pd.DataFrame] = []

    for db in args.dbs:
        parsed, one_v_one, multi = process_db(Path(db), out_root)
        all_parsed.append(parsed)
        all_one_v_one.append(one_v_one)
        all_multi.append(multi)

    write_combined_outputs(out_root, all_parsed, all_one_v_one, all_multi)


if __name__ == "__main__":
    main()
