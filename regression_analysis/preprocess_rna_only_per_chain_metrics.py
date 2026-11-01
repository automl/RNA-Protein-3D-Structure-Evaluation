#!/usr/bin/env python3
"""
dbs2single_chain_analysis_csvs_rna_only.py

Create analysis-ready PER-CHAIN CSVs from AF3-eval RNA-only SQLite DBs.

Updated version:
- preserves the existing schema
- adds composition / size features aligned with protein-RNA per-chain CSVs
- keeps rich RNA intrinsic / structure / thermodynamic features
- adds normalized structural counts per 100 nt
"""
from __future__ import annotations
import argparse, ast, re, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

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

def nonblank_count(series: pd.Series) -> int:
    s = series.astype(object)
    return int((((~pd.isna(s)) & (s.astype(str).str.strip() != ""))).sum())

def print_af3_availability(df: pd.DataFrame, db_tag: str) -> None:
    af3_cols = [
        "af3_rna_pLDDT_avg", "af3_global_pae_avg", "af3_chain_pair_pae_min",
        "af3_ranking_score", "af3_rna_pTM", "af3_rna_ipTM", "af3_rna_MSA", "RNA_msa_size",
    ]
    present = [c for c in af3_cols if c in df.columns]
    if not present:
        print(f"[{db_tag}] AF3 availability: no AF3 columns present in pred_rna_rna")
        return
    stats = ", ".join(f"{c}={nonblank_count(df[c])}/{len(df)}" for c in present)
    print(f"[{db_tag}] AF3 availability: {stats}")

def first_list_value(x: Any) -> float:
    v = safe_literal_eval(x)
    if isinstance(v, list) and len(v) > 0:
        return to_float(v[0])
    return to_float(v)

def ensure_list(x: Any) -> List[Any]:
    v = safe_literal_eval(x)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]

def normalize_name(text: str) -> str:
    s = str(text).strip().lower()
    replacements = {
        "polyethylene glycol": "peg", "peg ": "peg_", "peg-": "peg_", "peg(": "peg_",
        "poly ethylene glycol": "peg", "tris-hcl": "tris_hcl", "tris-cl": "tris_cl",
        "bis-tris": "bis_tris", "bis tris": "bis_tris", "hepes-naoh": "hepes_naoh",
        "phosphate-citrate": "phosphate_citrate", "sodium-potassium phosphate": "sodium_potassium_phosphate",
        "mono-methyl-ether": "mme", "mono methyl ether": "mme", "w/v": "", "v/v": "",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = s.replace("%", "")
    s = s.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def infer_post_cutoff_from_db_tag(db_tag: str) -> int:
    return int(str(db_tag).strip().lower().startswith("a"))

def clean_rna_sequence(seq: Any) -> str:
    if seq is None:
        return ""
    s = str(seq)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^A-Za-z]", "", s).upper().replace("T", "U")
    return "".join(ch for ch in s if ch in {"A", "C", "G", "U", "N"})

def scalar_or_first_from_dict_or_list(x: Any, chain_id: Optional[str] = None) -> float:
    v = safe_literal_eval(x)
    if isinstance(v, dict):
        if chain_id is not None and chain_id in v:
            return to_float(v.get(chain_id))
        if len(v) > 0:
            first_key = next(iter(v.keys()))
            return to_float(v[first_key])
        return np.nan
    if isinstance(v, list):
        return to_float(v[0]) if len(v) > 0 else np.nan
    return to_float(v)

def as_single_chain_id(x: Any) -> str:
    vals = ensure_list(x)
    if len(vals) == 0:
        return "A"
    return str(vals[0])

def as_single_sequence(x: Any) -> str:
    vals = ensure_list(x)
    if len(vals) == 0:
        return "" if x is None else str(x)
    return "" if vals[0] is None else str(vals[0])

_ALLOWED_RNA_CHARS = set("ACGTUNacgtun")
def sequence_has_modified_bases(seq: Any) -> int:
    s = "" if seq is None else str(seq)
    return int(any(ch not in _ALLOWED_RNA_CHARS for ch in s)) if s else 0

def parse_modified_flag(seq: Any, modified_field: Any) -> int:
    mod = safe_literal_eval(modified_field)
    if isinstance(mod, list) and len(mod) > 0:
        return 1
    if isinstance(mod, str) and mod.strip() not in {"", "[]", "None", "nan"}:
        return 1
    return sequence_has_modified_bases(seq)

def parse_single_family(x: Any) -> Optional[str]:
    v = safe_literal_eval(x)
    if isinstance(v, list):
        v = v[0] if len(v) else None
    if v is None:
        return None
    s = str(v).strip()
    return None if s == "" or s.lower() == "none" else s

def shannon_entropy_kmers(seq: str, k: int = 1) -> float:
    seq = seq or ""
    if len(seq) < k or k <= 0:
        return np.nan
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
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

def compute_rna_sequence_features(seq: str, prefix: str) -> Dict[str, float]:
    seq = clean_rna_sequence(seq)
    if seq == "":
        return {
            f"{prefix}_gc_content": np.nan, f"{prefix}_au_content": np.nan,
            f"{prefix}_gc_skew": np.nan, f"{prefix}_au_skew": np.nan,
            f"{prefix}_sequence_complexity": np.nan,
            f"{prefix}_shannon_entropy_k1": np.nan,
            f"{prefix}_shannon_entropy_k2": np.nan,
        }
    counts = {b: seq.count(b) for b in "ACGUN"}
    non_n = max(counts["A"] + counts["C"] + counts["G"] + counts["U"], 1)
    gc = counts["G"] + counts["C"]
    au = counts["A"] + counts["U"]
    return {
        f"{prefix}_gc_content": gc / non_n,
        f"{prefix}_au_content": au / non_n,
        f"{prefix}_gc_skew": ((counts["G"] - counts["C"]) / (counts["G"] + counts["C"])) if (counts["G"] + counts["C"]) > 0 else np.nan,
        f"{prefix}_au_skew": ((counts["A"] - counts["U"]) / (counts["A"] + counts["U"])) if (counts["A"] + counts["U"]) > 0 else np.nan,
        f"{prefix}_sequence_complexity": sequence_complexity(seq),
        f"{prefix}_shannon_entropy_k1": shannon_entropy_kmers(seq, 1),
        f"{prefix}_shannon_entropy_k2": shannon_entropy_kmers(seq, 2),
    }

_COMPONENT_AFTER = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|uM|µM|nM|%)\s*(?:\([wv]/[wv]\)\s*)?(?P<comp>[A-Za-z0-9\-\+\./ ]{2,80})", flags=re.IGNORECASE)
_COMPONENT_BEFORE = re.compile(r"(?P<comp>(?:PEG|Polyethylene glycol|Tris(?:-HCl|-Cl)?|Bis-Tris|HEPES(?:-NaOH)?|MES(?:/imidazole)?|citric acid|sodium acetate|potassium phosphate|sodium phosphate|ammonium sulfate|lithium sulfate|magnesium chloride(?: hexahydrate)?|sodium chloride|NaCl|KCl|LiCl|glycerol|ethylene glycol|MPD|Tacsimate|Bicine|succinic acid|sodium formate|ammonium acetate|ammonium phosphate dibasic|phosphate-citrate)[A-Za-z0-9\-\+\./ ]{0,40})\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|uM|µM|nM|%)", flags=re.IGNORECASE)
_PH_RE = re.compile(r"\bpH\s*=?\s*(\d+(?:\.\d+)?)", flags=re.IGNORECASE)
_TEMP_RE = re.compile(r"(?:temperature\s*)?(\d+(?:\.\d+)?)\s*(K|C)\b", flags=re.IGNORECASE)

def split_solution_text(text: str) -> List[str]:
    s = str(text).replace("\n", ", ")
    s = re.sub(r"\band\b", ",", s, flags=re.IGNORECASE)
    return [p.strip(" .;:") for p in re.split(r",|;", s) if p.strip(" .;:")]

def unit_to_molar(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "m": return value
    if u == "mm": return value * 1e-3
    if u in {"um", "µm"}: return value * 1e-6
    if u == "nm": return value * 1e-9
    return np.nan

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
        temp_val = float(temp_match.group(1)); temp_unit = temp_match.group(2).upper()
        out[f"{prefix}_temp_k"] = temp_val + 273.15 if temp_unit == "C" else temp_val
    for part in parts:
        local_ph = _PH_RE.findall(part)
        m = _COMPONENT_AFTER.search(part)
        if m:
            val = float(m.group("val")); unit = m.group("unit")
            comp = normalize_name(m.group("comp"))
            comp = re.sub(r"_(ph|temperature).*", "", comp).strip("_") or "unknown"
            if unit == "%": out[f"{prefix}_pct_{comp}"] = val
            else: out[f"{prefix}_conc_{comp}_m"] = unit_to_molar(val, unit)
            if local_ph: out[f"{prefix}_ph_{comp}"] = to_float(local_ph[0])
            continue
        m2 = _COMPONENT_BEFORE.search(part)
        if m2:
            comp = normalize_name(m2.group("comp")); val = float(m2.group("val")); unit = m2.group("unit")
            if unit == "%": out[f"{prefix}_pct_{comp}"] = val
            else: out[f"{prefix}_conc_{comp}_m"] = unit_to_molar(val, unit)
            if local_ph: out[f"{prefix}_ph_{comp}"] = to_float(local_ph[0])
    return out

PEG_MW_PATTERN = re.compile(r"(?:PEG|PEMME|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]{2,5})(?=[^\d]|$)", flags=re.I)
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
    "magnesium acetate tetrahydrate": "magnesium acetate", "na formate": "sodium formate", "mg formate": "magnesium formate",
    "na citrate": "sodium citrate", "tri-sodium citrate": "trisodium citrate", "citrate": "sodium citrate",
    "ammonium tartrate": "ammonium tartrate", "nh4/tartrate": "ammonium tartrate", "nh4 tartrate": "ammonium tartrate",
    "potassium sodium tartrate": "potassium sodium tartrate", "sodium potassium tartrate": "potassium sodium tartrate",
    "sodium phosphate": "sodium phosphate", "na h2po4": "sodium phosphate", "na2 hpo4": "sodium phosphate",
    "k phosphate": "potassium phosphate", "na2co3": "sodium carbonate", "k2co3": "potassium carbonate",
    "na2 succinate": "disodium succinate", "kbr": "potassium bromide", "nabr": "sodium bromide",
    "ki": "potassium iodide", "nai": "sodium iodide", "naf": "ammonium fluoride",
}
BUFFERS = ["tris","tris-hcl","tris(hydroxymethyl)aminomethane","hepes","mes","bicine","imidazole","cacodylate","cacodyl","sodium cacodylate","sodium cacodylate trihydrate","bis-tris","bis-tris propane","bis-trispropane","citric","citrate","phosphate","sodium phosphate","potassium phosphate","acetate","formate","tartrate","pH","mib","ches","ada","succinate"]
HYDRATE_PATTERN = re.compile(r"\b(mono|di|tri|tetra|penta|hexa)hydrate\b", flags=re.I)

def normalize_condition_name(raw: str) -> Optional[str]:
    r = raw.lower().strip()
    r = HYDRATE_PATTERN.sub("", r).strip()
    r = re.sub(r"\bph\s*\d+\.?\d*\b", "", r, flags=re.I).strip()
    r = re.sub(r"\s+", " ", r).strip()
    for key, val in ALIASES.items():
        if key in r: return val
    for salt in SALTS.keys():
        if salt in r: return salt
    return None

def parse_concentration_m(text: str) -> Optional[float]:
    m = re.search(r"(\d+\.?\d*)\s*(m|mm)\b", text.lower())
    if not m: return None
    val, unit = m.groups(); val = float(val)
    return val if unit == "m" else val / 1000.0

def ionic_strength_from_condition(condition_text: str) -> Optional[float]:
    total_I = 0.0
    for pattern in (
        re.compile(r"(\d+\.?\d*)\s+(m|mm)\s+([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)", flags=re.I),
        re.compile(r"(\d+\.?\d*)(m|mm)([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)", flags=re.I),
    ):
        for match in pattern.finditer(condition_text):
            val_str, unit, tail = match.groups()
            if "peg" in tail.lower(): continue
            conc = float(val_str) / 1000.0 if unit.lower() == "mm" else float(val_str)
            salt_name = normalize_condition_name(tail.strip())
            if salt_name is None: continue
            z_plus, z_minus = SALTS[salt_name]
            total_I += 0.5 * conc * (z_plus ** 2 + z_minus ** 2)
    if total_I == 0.0:
        for part in re.split(r"[;,]", condition_text.lower()):
            conc = parse_concentration_m(part)
            if conc is None: continue
            salt_name = normalize_condition_name(part)
            if salt_name is None: continue
            z_plus, z_minus = SALTS[salt_name]
            total_I += 0.5 * conc * (z_plus ** 2 + z_minus ** 2)
    return total_I if total_I > 0.0 else np.nan

def ionic_strength_category(v: Any) -> str:
    v = to_float(v)
    if np.isnan(v) or v == 0: return "Unknown"
    if v >= 1.0: return "ge_1_0_m"
    if v >= 0.5: return "0_5_to_1_0_m"
    return "0_to_0_5_m"

def peg_category_from_percent(p: Any) -> str:
    p = to_float(p)
    if np.isnan(p): return "no_peg"
    if p >= 30: return "peg_30_50_pct"
    if p >= 20: return "peg_20_30_pct"
    if p >= 10: return "peg_10_20_pct"
    if p > 0: return "peg_0_10_pct"
    return "no_peg"

def peg_mw_category(m: Any) -> str:
    m = to_float(m)
    if np.isnan(m): return "unknown"
    if m < 1000: return "peg_0_1k"
    if m < 2000: return "peg_1k_2k"
    if m < 10000: return "peg_2k_10k"
    return "peg_10k_plus"

def find_peg_percent(text: str) -> Optional[float]:
    patterns = [
        re.compile(r"(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{1,2})\s*[Kk]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:\(w/v\))?\s*(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{2,5})", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*Polyethylene\s+glycol\s+([0-9]{2,5})", flags=re.I),
        re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:.*?\s)?(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)", flags=re.I),
        re.compile(r"PEG.*?([0-9]{1,3}(?:\.[0-9]+)?)\s*%", flags=re.I),
    ]
    for pattern in patterns:
        m = pattern.search(text)
        if not m: continue
        for g in [x for x in m.groups() if x is not None]:
            try:
                val = float(g.replace(",", ""))
                if 0 <= val <= 100: return val
            except Exception: pass
    return np.nan

def find_peg_mw(text: str) -> Optional[float]:
    for m in (PEG_MW_PATTERN.search(text), PEG_K_PATTERN.search(text), re.search(r"Polyethylene\s+glycol\s+([0-9]{2,5})", text, flags=re.I)):
        if m:
            try:
                return float(m.group(1).replace(",", "")) * (1000.0 if m.re is PEG_K_PATTERN else 1.0)
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
    out: Dict[str, Any] = {"ionic_strength": np.nan, "ionic_strength_category": "Unknown", "peg_percent": np.nan, "peg_mw": np.nan, "peg_concentration_category": "no_peg", "peg_mw_category": "unknown", "buffer_m": np.nan, "buffer_category": "unknown", "buffer_name": np.nan, "has_peg": 0}
    if text is None or (isinstance(text, float) and np.isnan(text)): return out
    s = str(text).replace("\n", " ").strip()
    ionic_strength = ionic_strength_from_condition(s)
    peg_pct = find_peg_percent(s); peg_mw = find_peg_mw(s)
    out.update({
        "ionic_strength": ionic_strength,
        "ionic_strength_category": ionic_strength_category(ionic_strength),
        "peg_percent": peg_pct,
        "peg_mw": peg_mw,
        "peg_concentration_category": peg_category_from_percent(peg_pct),
        "peg_mw_category": peg_mw_category(peg_mw),
        "has_peg": int(not (np.isnan(to_float(peg_pct)) and np.isnan(to_float(peg_mw)))),
    })
    pattern_molarity_first = re.compile(r"([0-9]*\.?[0-9]+)\s*(mM|M)\s*([A-Za-z0-9\-/\s\(\)\.]*)", flags=re.I)
    pattern_buffer_first = re.compile(r"([A-Za-z0-9\-/\s\(\)\.]+?)\s*([0-9]*\.?[0-9]+)\s*(mM|M)", flags=re.I)
    def normalize_conc(val_str: str, unit: str) -> float:
        v = float(val_str)
        return v / 1000.0 if unit.lower() == "mm" else v
    buffer_matches = []
    for match in pattern_molarity_first.finditer(s):
        val_str, unit, tail = match.groups(); buf = find_buffer_keyword(tail)
        if buf: buffer_matches.append((normalize_conc(val_str, unit), buf))
    for match in pattern_buffer_first.finditer(s):
        buf_name, val_str, unit = match.groups(); buf = find_buffer_keyword(buf_name)
        if buf: buffer_matches.append((normalize_conc(val_str, unit), buf))
    if buffer_matches:
        buffer_M, buffer_name = max(buffer_matches, key=lambda x: x[0])
        out["buffer_m"] = buffer_M; out["buffer_name"] = buffer_name
        out["buffer_category"] = "0_0_1_m" if buffer_M < 0.1 else ("0_1_0_5_m" if buffer_M < 0.5 else ("0_5_5_0_m" if buffer_M < 5 else "gt_5_0_m"))
    return out

def safe_int_flag(x: Any) -> Optional[int]:
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        return int(float(x))
    except Exception:
        return None

def parse_pairs_list(x: Any) -> List[Tuple[Any, str]]:
    v = safe_literal_eval(x); out: List[Tuple[Any, str]] = []
    if isinstance(v, list):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((item[0], str(item[1]).strip()))
    return out

def summarize_support_pairs(x: Any, kind: str) -> Dict[str, Any]:
    pairs = parse_pairs_list(x); n_listed = len(pairs); supported = unsupported = unknown = 0; names: Set[str] = set()
    for flag_raw, code in pairs:
        names.add(normalize_name(code)); flag = safe_int_flag(flag_raw)
        if flag == 1: supported += 1
        elif flag == 0: unsupported += 1
        else: unknown += 1
    frac_supported = supported / n_listed if n_listed > 0 else np.nan
    return {f"n_{kind}_listed": n_listed, f"n_supported_{kind}": supported, f"n_unsupported_{kind}": unsupported, f"n_unknown_{kind}": unknown, f"has_any_{kind[:-1]}": int(n_listed > 0), f"has_any_unsupported_{kind[:-1]}": int(unsupported > 0), f"has_any_supported_{kind[:-1]}": int(supported > 0), f"frac_supported_{kind}": frac_supported, f"n_unique_{kind}": len(names)}

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
            key = normalize_name(code); col = code_to_col.get(key); flag = safe_int_flag(flag_raw)
            if col is not None and flag == 0: row[col] = 1
        rows.append(row)
    return pd.DataFrame(rows)

EXP_BASE_COLUMNS = ["PDBId","RNAFamily","RNAModified","RNASequence","RNAChainIDs","RNALength","NumberRNAs","Experiment","XRayResolution","exp_pH","exp_Temp","RNA_ElectrostaticPotential","Ions","Ligands","exp_Solution","Binding_affinity_kd","RNA_Stems","RNA_HairpinLoops","RNA_InternalLoops","RNA_MultibranchLoops","RNA_DanglingEnds","RNA_Pseudoknots","RNA_isDuplex","RNArnaInterfaceArea","RNArnaInterfaceRatio","Free_energy","RNA_GlycosidicBond","RNA_SugarPucker","RNA_GammaAngle"]
PRED_BASE_COLUMNS = [
    "exp_db_id",
    "RNASequence",
    "RNAChainIDs",
    "RNALength",
    "NumberRNAs",
    "RNAModified",

    # AF3 confidence / ranking / MSA
    "af3_rna_pTM",
    "af3_rna_ipTM",
    "af3_rna_pLDDT_avg",
    "af3_global_pae_avg",
    "af3_chain_pair_pae_min",
    "af3_fraction_disordered",
    "af3_has_clash",
    "af3_ranking_score",
    "af3_rna_MSA",
    "RNA_msa_size",

    # core scores
    "Complex_RMSD",
    "RNA_RMSD",
    "RNA_LDDT",
    "Complex_LDDT",
    "RNA_TM",
    "Complex_TM",
    "RNA_INF",
    "RNA_DI",
    "RNA_f1_score",
    "RNA_precision",
    "RNA_recall",
    "RNA_mcc",
    "RNA_wl",
    "RNA_LDDT_allAtom",
    "Complex_LDDT_allAtom",

    # interface-ish RNA-RNA fields
    "RNArnaInterfaceArea",
    "RNArnaInterfaceRatio",

    # thermodynamics / affinity / structure / intrinsic
    "Free_energy",
    "Binding_affinity_kd",
    "RNA_Stems",
    "RNA_HairpinLoops",
    "RNA_InternalLoops",
    "RNA_MultibranchLoops",
    "RNA_DanglingEnds",
    "RNA_Pseudoknots",
    "RNA_isDuplex",
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
    "AAmatch_score",
    "RNAmatch_score",
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
    if sol_cols:
        sol_rows = [parse_solution_like(x, "sol") for x in df["exp_Solution"]]
        sol_df = pd.DataFrame(sol_rows).reindex(columns=sol_cols)
        df = pd.concat([df, sol_df], axis=1)
    return df

def add_buffer_peg_features(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_Solution" in df.columns:
        df = pd.concat([df, pd.DataFrame([parse_buffer_features(x) for x in df["exp_Solution"]])], axis=1)
    return df

def parse_loop_count(x: Any) -> float:
    v = safe_literal_eval(x)
    return float(len(v)) if isinstance(v, list) else np.nan

def build_reference_columns(reference_csv: Optional[Path]) -> List[str]:
    return pd.read_csv(reference_csv, nrows=0).columns.tolist() if reference_csv else []

def parse_features(merged: pd.DataFrame, db_tag: str) -> pd.DataFrame:
    df = merged.copy()
    df["pdb_id"] = df["exp_db_id"].astype(str)
    df["source_db"] = db_tag
    df["is_post_cutoff"] = infer_post_cutoff_from_db_tag(db_tag)
    df["NumberProteins"] = 0
    df["pred_rna_chain_index"] = 0
    df["pred_rna_chain_id"] = df["RNAChainIDs_pred"].apply(as_single_chain_id) if "RNAChainIDs_pred" in df.columns else "A"
    df["pred_rna_chain_length"] = pd.to_numeric(df.get("RNALength_pred", df.get("RNALength_exp")), errors="coerce")
    df["protein_total_length"] = 0.0
    df["protein_mean_chain_length"] = np.nan
    df["protein_max_chain_length"] = np.nan
    df["protein_min_chain_length"] = np.nan
    df["protein_std_chain_length"] = np.nan
    df["protein_chain_length_cv"] = np.nan
    df["rna_total_length"] = df["pred_rna_chain_length"]
    df["total_complex_length"] = df["rna_total_length"]
    df["rna_mean_chain_length"] = df["pred_rna_chain_length"]
    df["rna_max_chain_length"] = df["pred_rna_chain_length"]
    df["rna_min_chain_length"] = df["pred_rna_chain_length"]
    df["rna_std_chain_length"] = 0.0
    df["rna_chain_length_cv"] = 0.0
    df["protein_residue_fraction"] = 0.0
    df["rna_residue_fraction"] = 1.0
    df["protein_to_rna_length_ratio"] = 0.0
    df["max_rna_fraction_of_total_rna"] = 1.0
    df["min_rna_fraction_of_total_rna"] = 1.0
    df["pred_rna_sequence"] = df["RNASequence_pred"].apply(as_single_sequence) if "RNASequence_pred" in df.columns else ""
    df["exp_rna_sequence"] = df["RNASequence_exp"].apply(as_single_sequence) if "RNASequence_exp" in df.columns else ""
    df["pred_rna_is_modified"] = [parse_modified_flag(seq, mod) for seq, mod in zip(df["pred_rna_sequence"], df.get("RNAModified_pred", pd.Series([None] * len(df))))]
    df["rna_family"] = df.get("RNAFamily", pd.Series([None] * len(df))).apply(parse_single_family)
    df["rna_family_is_known"] = df["rna_family"].notna().astype(int)
    df["n_unique_rna_families"] = df["rna_family_is_known"].astype(int)
    df["rna_family_summary"] = df["rna_family"]
    df["protein_family_summary"] = np.nan
    df["protein_family_is_known"] = 0
    df["n_unique_protein_families"] = 0
    df["gt_rna_chain_index"] = 0
    df["assigned_rna_tm"] = pd.to_numeric(df.get("RNA_TM"), errors="coerce")
    df["assigned_rna_lddt"] = pd.to_numeric(df.get("RNA_LDDT"), errors="coerce")
    df["is_matched"] = df["assigned_rna_tm"].notna().astype(int)
    df["assignment_method"] = "identity_single_chain"
    df["n_rbp_templates"] = np.nan
    df["n_protein_templates"] = np.nan
    df["id_is_rbp_template"] = np.nan
    df["id_is_protein_template"] = np.nan
    # ------------------------------------------------------------------
    # AF3 metrics
    # ------------------------------------------------------------------
    df["af3_protein_pLDDT_avg"] = np.nan
    df["af3_protein_pTM_list_value"] = np.nan
    df["af3_protein_ipTM_list_value"] = np.nan

    af3_raw_cols = [
        "af3_rna_pLDDT_avg",
        "af3_global_pae_avg",
        "af3_chain_pair_pae_min",
        "af3_ranking_score",
        "af3_rna_pTM",
        "af3_rna_ipTM",
        "af3_rna_MSA",
        "RNA_msa_size",
        "af3_fraction_disordered",
        "af3_has_clash",
    ]
    for c in af3_raw_cols:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = df[c].replace({"": np.nan, " ": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan})

    # direct scalar AF3 columns
    for c in [
        "af3_rna_pLDDT_avg",
        "af3_global_pae_avg",
        "af3_ranking_score",
        "af3_fraction_disordered",
        "af3_has_clash",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # list-like / dict-like AF3 fields -> scalar analysis columns
    df["af3_chain_pair_pae_min"] = df["af3_chain_pair_pae_min"].apply(first_list_value)
    df["af3_rna_pTM_list_value"] = df["af3_rna_pTM"].apply(first_list_value)
    df["af3_rna_ipTM_list_value"] = df["af3_rna_ipTM"].apply(first_list_value)
    df["af3_rna_MSA_list_value"] = df["af3_rna_MSA"].apply(first_list_value)

    # chain-specific RNA MSA size
    df["rna_msa_size"] = [
        scalar_or_first_from_dict_or_list(x, ch)
        for x, ch in zip(df["RNA_msa_size"], df["pred_rna_chain_id"])
    ]
    df["rna_msa_size"] = pd.to_numeric(df["rna_msa_size"], errors="coerce")
    df["log10_rna_msa_size"] = np.where(df["rna_msa_size"] > 0, np.log10(df["rna_msa_size"]), np.nan) 
    for c in ["Binding_affinity_kd","RNA_ElectrostaticPotential","XRayResolution","exp_pH","exp_Temp"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log10_kd"] = np.log10(df["Binding_affinity_kd"].replace(0, np.nan)) if "Binding_affinity_kd" in df.columns else np.nan
    if "Ions" in df.columns:
        df = pd.concat([df, pd.DataFrame([summarize_support_pairs(x, "ions") for x in df["Ions"]])], axis=1)
        ion_cols = collect_dynamic_codes(df["Ions"], "has_unsupported_ion")
        if ion_cols: df = pd.concat([df, fill_support_code_columns(df["Ions"], ion_cols, "has_unsupported_ion")], axis=1)
    if "Ligands" in df.columns:
        df = pd.concat([df, pd.DataFrame([summarize_support_pairs(x, "ligands") for x in df["Ligands"]])], axis=1)
        lig_cols = collect_dynamic_codes(df["Ligands"], "has_unsupported_ligand")
        if lig_cols: df = pd.concat([df, fill_support_code_columns(df["Ligands"], lig_cols, "has_unsupported_ligand")], axis=1)
    df = add_solution_features(df)
    df = add_buffer_peg_features(df)
    df = add_experiment_flags(df)
    for c in ["exp_ProteinRNAInterfaceArea","pred_ProteinRNAInterfaceArea","ProteinRNAInterfaceArea_diff_exp_minus_pred","ProteinRNAInterfaceArea_abs_diff","ProteinRNAInterfaceArea_rel_error","ProteinRNAInterfaceArea_pred_over_exp","exp_ProteinRNAInterfaceRatio","pred_ProteinRNAInterfaceRatio","ProteinRNAInterfaceRatio_diff_exp_minus_pred","ProteinRNAInterfaceRatio_abs_diff","ProteinRNAInterfaceRatio_rel_error","ProteinRNAInterfaceRatio_pred_over_exp"]:
        df[c] = np.nan
    df["Complex_RMSD_first"] = df.get("Complex_RMSD", pd.Series([np.nan] * len(df))).apply(first_list_value)
    for c in ["Complex_TM","Complex_LDDT","Complex_LDDT_allAtom"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce") if c in df.columns else np.nan
    df["iLDDT"] = np.nan; df["QS_global"] = np.nan; df["Dockq"] = np.nan; df["DockQ_minus_AF3rank"] = np.nan; df["ICS"] = np.nan
    df["RNAComplex_TM"] = pd.to_numeric(df.get("RNA_TM"), errors="coerce")
    df["RNAmotif_score"] = np.nan
    extra_numeric = ["RNA_RMSD","RNA_LDDT","RNA_LDDT_allAtom","RNA_TM","RNA_INF","RNA_DI","RNA_f1_score","RNA_precision","RNA_recall","RNA_mcc","RNA_wl","af3_fraction_disordered","af3_has_clash","Free_energy","RNA_GC_Content","RNA_SequenceComplexity","RNA_Shannon_Entropy_K1","RNA_Shannon_Entropy_K2","RNA_GC_Skew","RNA_AU_Skew","RNA_AU_Content","RNA_MFE_Value","RNA_Ensemble_Diversity","RNA_Centroid_Energy","AAmatch_score","RNAmatch_score"]
    for c in extra_numeric:
        df[c] = pd.to_numeric(df.get(c), errors="coerce") if c in df.columns else np.nan
    df["RNA_MFE_Structure"] = df.get("RNA_MFE_Structure", np.nan)
    df["num_stems"] = df.get("RNA_Stems", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    df["num_hairpin_loops"] = df.get("RNA_HairpinLoops", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    df["num_internal_loops"] = df.get("RNA_InternalLoops", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    df["num_multibranch_loops"] = df.get("RNA_MultibranchLoops", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    df["num_dangling_ends"] = df.get("RNA_DanglingEnds", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    df["num_pseudoknots"] = df.get("RNA_Pseudoknots", pd.Series([np.nan] * len(df))).apply(parse_loop_count)
    for base_col, new_col in [("num_stems","stems_per_100nt"),("num_hairpin_loops","hairpin_loops_per_100nt"),("num_internal_loops","internal_loops_per_100nt"),("num_multibranch_loops","multibranch_loops_per_100nt"),("num_dangling_ends","dangling_ends_per_100nt"),("num_pseudoknots","pseudoknots_per_100nt")]:
        df[new_col] = df[base_col] / (df["pred_rna_chain_length"].replace(0, np.nan) / 100.0)
    # aggregate RNA sequence features aligned with protein-RNA outputs
    agg_rows = [compute_rna_sequence_features(seq, "agg_rna") for seq in df["pred_rna_sequence"]]
    df = pd.concat([df, pd.DataFrame(agg_rows)], axis=1)
    return df

def order_columns(df: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
    for c in reference_cols:
        if c not in df.columns:
            df[c] = np.nan
    extras = [c for c in df.columns if c not in reference_cols]
    return df[reference_cols + sorted(extras)].copy()

def process_db(db_path: Path, out_root: Path, reference_cols: List[str]) -> pd.DataFrame:
    db_tag = db_path.stem
    out_dir = out_root / db_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        exp = load_table_columns(conn, "exp_rna_rna", EXP_BASE_COLUMNS)
        pred = load_table_columns(conn, "pred_rna_rna", PRED_BASE_COLUMNS)
    print_af3_availability(pred, db_tag)
    merged = pred.merge(exp, left_on="exp_db_id", right_on="PDBId", how="left", suffixes=("_pred", "_exp"))
    parsed = parse_features(merged, db_tag=db_tag)
    if reference_cols:
        parsed = order_columns(parsed, reference_cols)
    out_csv = out_dir / f"{db_tag}__per_chain_all.csv"
    parsed.to_csv(out_csv, index=False)
    summary = pd.DataFrame([
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_all", "n_rows": len(parsed)},
        {"source_db": db_tag, "is_post_cutoff": infer_post_cutoff_from_db_tag(db_tag), "subset": "per_chain_matched", "n_rows": int(parsed["is_matched"].fillna(0).sum())},
    ])
    summary.to_csv(out_dir / f"{db_tag}__per_chain_summary.csv", index=False)
    print(f"[{db_tag}] per-chain rows: {len(parsed)} -> {out_csv}")
    print(f"[{db_tag}] matched rows: {int(parsed['is_matched'].fillna(0).sum())}")
    return parsed

def write_combined_outputs(out_root: Path, all_dfs: List[pd.DataFrame]) -> None:
    comb_dir = out_root / "combined"
    comb_dir.mkdir(parents=True, exist_ok=True)
    per_chain = pd.concat(all_dfs, axis=0, ignore_index=True) if all_dfs else pd.DataFrame()
    if not per_chain.empty:
        per_chain.to_csv(comb_dir / "combined__per_chain_all.csv", index=False)
        pd.DataFrame([
            {"subset": "per_chain_all", "n_rows": len(per_chain)},
            {"subset": "per_chain_matched", "n_rows": int(per_chain["is_matched"].fillna(0).sum())},
        ]).to_csv(comb_dir / "combined__per_chain_summary.csv", index=False)
    print(f"[combined] per-chain rows: {len(per_chain)}")
    if not per_chain.empty:
        print(f"[combined] matched rows: {int(per_chain['is_matched'].fillna(0).sum())}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", required=True, help="RNA-only SQLite db files")
    ap.add_argument("--outdir", default="csvs4analysis_per_chain_rnaonly", help="Output folder")
    ap.add_argument("--reference_csv", default=None, help="Optional reference per-chain CSV for schema/column ordering")
    args = ap.parse_args()
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)
    reference_cols = build_reference_columns(Path(args.reference_csv)) if args.reference_csv else []
    all_dfs: List[pd.DataFrame] = []
    for db in args.dbs:
        all_dfs.append(process_db(Path(db), out_root, reference_cols))
    write_combined_outputs(out_root, all_dfs)

if __name__ == "__main__":
    main()
