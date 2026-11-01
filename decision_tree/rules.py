"""
Rules of the accuracy-calibrated decision framework (paper Section "A practical,
accuracy-calibrated decision framework", Fig. decision tree, Table tab:decision_accuracy).

Two layers, both usable without an experimental reference structure:

1. Expected accuracy (lookup). The observed mean per-chain RNA TM-score of AF3 in the
   paper's PDB benchmark, conditioned only on properties known before prediction:
   system class, RNA chain length and whether the target post-dates the AF3 training
   cutoff (30 Sep 2021). For a novel target the post-cutoff value applies.

2. Reliability tiers (per structure). Fixed cutpoints on AF3's own confidence:
     fold tier       RNA pLDDT (0-1):      good >= 0.775, bad < 0.55, else mid
     interface tier  RNA ipTM + chain-pair PAE (A):
                     good: ipTM_r >= 0.5 and PAE <= 8
                     bad:  ipTM_r < 0.25 or PAE > 15
                     else mid
   The interface tier applies to protein-RNA interfaces only.

The values are copied from regression_analysis/decision_branch_stats.csv and
regression_analysis/analysis_R6_tier_validation.py so the package has no data dependency.
"""

from __future__ import annotations

import math
from typing import Optional

# --------------------------------------------------------------------------- #
# Layer 2: reliability tiers
# --------------------------------------------------------------------------- #
FOLD_GOOD = 0.775          # RNA pLDDT (0-1)
FOLD_BAD = 0.55
IF_IPTM_GOOD = 0.5         # RNA ipTM
IF_PAE_GOOD = 8.0          # chain-pair PAE min (A)
IF_IPTM_BAD = 0.25
IF_PAE_BAD = 15.0
TIERS = ("good", "mid", "bad")


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fold_tier(rna_plddt: Optional[float]) -> Optional[str]:
    """RNA-fold tier from the mean RNA pLDDT on the 0-1 scale (AF3 reports 0-100)."""
    if not _finite(rna_plddt):
        return None
    p = rna_plddt / 100.0 if rna_plddt > 1.0 else rna_plddt
    return "good" if p >= FOLD_GOOD else ("bad" if p < FOLD_BAD else "mid")


def interface_tier(rna_iptm: Optional[float], pae_min: Optional[float]) -> Optional[str]:
    """Protein-RNA interface tier from the RNA chain ipTM and the protein-RNA chain-pair PAE."""
    if not (_finite(rna_iptm) and _finite(pae_min)):
        return None
    if rna_iptm >= IF_IPTM_GOOD and pae_min <= IF_PAE_GOOD:
        return "good"
    if rna_iptm < IF_IPTM_BAD or pae_min > IF_PAE_BAD:
        return "bad"
    return "mid"


# --------------------------------------------------------------------------- #
# Layer 1: expected per-chain RNA TM-score (paper Table tab:decision_accuracy)
# --------------------------------------------------------------------------- #
SYSTEM_CLASSES = ("single-chain RNA", "RNA-protein 1v1", "RNA-protein multichain")
# class -> (combined mean, n, pre-cutoff mean, post-cutoff mean)
EXPECTED_BY_CLASS = {
    "single-chain RNA":       (0.66, 163, 0.70, 0.59),
    "RNA-protein 1v1":        (0.58, 345, 0.64, 0.35),
    "RNA-protein multichain": (0.50, 1993, 0.54, 0.40),
}
LENGTH_BINS = ("<=40", "41-100", "101-200", ">200")
# (class, length bin) -> (combined mean, n, post-cutoff mean or None, post n)
EXPECTED_BY_LENGTH = {
    ("single-chain RNA", "<=40"): (0.37, 34, 0.31, 10),
    ("single-chain RNA", "41-100"): (0.69, 84, 0.41, 22),
    ("single-chain RNA", "101-200"): (0.68, 8, 0.52, 3),
    ("single-chain RNA", ">200"): (0.86, 37, 0.82, 29),
    ("RNA-protein multichain", "<=40"): (0.46, 1669, 0.37, 479),
    ("RNA-protein multichain", "41-100"): (0.61, 267, 0.48, 100),
    ("RNA-protein multichain", "101-200"): (0.74, 12, 0.39, 3),
    ("RNA-protein multichain", ">200"): (0.96, 45, 0.88, 4),
}
THIN_BIN = 15   # bins with fewer chains are flagged


def length_bin(n_nt: Optional[int]) -> Optional[str]:
    if n_nt is None:
        return None
    return "<=40" if n_nt <= 40 else "41-100" if n_nt <= 100 else "101-200" if n_nt <= 200 else ">200"


def system_class(n_protein_chains: int, n_rna_chains: int) -> Optional[str]:
    """Paper system classes; None for systems outside the calibrated classes (e.g. RNA-RNA)."""
    if n_rna_chains < 1:
        return None
    if n_protein_chains == 0:
        return "single-chain RNA" if n_rna_chains == 1 else None
    if n_protein_chains == 1 and n_rna_chains == 1:
        return "RNA-protein 1v1"
    return "RNA-protein multichain"


def expected_rna_tm(cls: Optional[str], n_nt: Optional[int], novel: bool = True) -> dict:
    """Expected per-chain RNA TM-score for a class / length; `novel` reads the post-cutoff value."""
    out = dict(system_class=cls, length_bin=length_bin(n_nt), expected_tm=None, basis=None, thin=False)
    if cls not in EXPECTED_BY_CLASS:
        out["basis"] = "not calibrated for this system class"
        return out
    comb, n, pre, post = EXPECTED_BY_CLASS[cls]
    out.update(expected_tm=post if novel else comb, basis=f"class ({'post-cutoff' if novel else 'combined'})")
    key = (cls, out["length_bin"])
    if key in EXPECTED_BY_LENGTH:
        c, cn, p, pn = EXPECTED_BY_LENGTH[key]
        val, nn = (p, pn) if novel else (c, cn)
        if val is not None:
            out.update(expected_tm=val, basis=f"class x length ({'post-cutoff' if novel else 'combined'})",
                       thin=nn < THIN_BIN)
    return out
