"""
Calibrated probabilities of the decision framework (design: develop_probability.py, frozen in
calibration_probability.json). Weights fitted on post-cutoff targets alone; no input needs an experimental structure.

  protein-RNA   P_fold       longest RNA chain well folded (LDDT_r > 0.85); intra-RNA PAE of that chain, a familiar
                             RNA sequence (>= 80% identity to an RNA of a PDB entry released before 30 Sep 2021), a
                             familiar fold of the AF3 model (US-align TM >= 0.5 of the model's longest RNA chain to a
                             pre-cutoff RNA chain), an RNA-bound protein template used by AF3 and >= 2 RNA chains
                P_interface  correct interface of one RNA chain; AF3's chain-pair PAE minimum to the protein, a
                             familiar RNA (sequence or fold) and a well-folded partner (pTM_p >= 0.84, pLDDT_r >= 0.78)
                P_usable     P_fold and P_interface joined by a Gaussian copula (one correlation, fitted out of
                             fold on post-cutoff targets; 0 would be the product, 1 the smaller of the two)
  RNA only      P_fold       single chain: pLDDT_r with the RNA alignment, intra-RNA PAE without;
                             RNA complex: a duplex in the AF3 model and the intra-RNA PAE of the longest chain

The inputs and weights are read from the calibration file, so a redesign needs no change here. intra-RNA PAE = mean
of AF3's `pae` matrix (full_data JSON) over the token pairs within one RNA chain, in A. pLDDT_r is on the 0-1 scale
here (the RNA-only design used %). Levels: WEAK < 25% <= MODERATE < 60% <= STRONG. An unknown input gives the
range of probabilities over its span (yes/no inputs: both answers; counts and scores: 5th-95th percentile of the
post-cutoff targets).
"""

from __future__ import annotations

import itertools
import json
import math
import os
from functools import lru_cache
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(_HERE, "calibration_probability.json")
CUTS = (0.25, 0.60)
LEVELS = ("WEAK", "MODERATE", "STRONG")
WELL_FOLDED_PTM, WELL_FOLDED_PLDDT = 0.84, 0.78


@lru_cache(maxsize=1)
def calibration() -> dict:
    return json.load(open(CAL))


def level(p: Optional[float]) -> Optional[str]:
    if p is None:
        return None
    return LEVELS[0 if p < CUTS[0] else (1 if p < CUTS[1] else 2)]


def level_range(lo: Optional[float], hi: Optional[float]) -> Optional[str]:
    if lo is None:
        return None
    a, b = level(lo), level(hi)
    return a if a == b else f"{a}..{b}"


def _logit(design: dict, values: dict) -> Optional[float]:
    z = design["intercept"]
    for f, c in design["coef"].items():
        if values.get(f) is None:
            return None
        z += c * float(values[f])
    return 1 / (1 + math.exp(-z))


def _range(design: dict, values: dict):
    """(min, max) probability over the unknown (None) inputs, each taken at both ends of its span"""
    spans = design.get("spans", {})
    unknown = [f for f in design["coef"] if values.get(f) is None]
    if any(f not in spans for f in unknown):
        return None, None
    ps = [_logit(design, dict(values, **dict(zip(unknown, combo))))
          for combo in itertools.product(*[spans[f] for f in unknown])]
    ps = [p for p in ps if p is not None]
    return (min(ps), max(ps)) if ps else (None, None)


def _b(v):
    return None if v is None else int(bool(v))


def fold_probability(rna_pae, n_rna=None, tm_familiar=None, seq_familiar=None, rbp_template=None, **extra):
    """protein-RNA: P(longest RNA chain well folded) as (min, max) over unknown inputs."""
    if rna_pae is None:
        return None, None
    v = dict(pae=rna_pae, multi_rna=None if n_rna is None else int(n_rna >= 2), tm_familiar=_b(tm_familiar),
             seq_familiar=_b(seq_familiar), rbp_template=_b(rbp_template), **extra)
    return _range(calibration()["fold"], v)


def well_folded_partner(partner_ptm, rna_plddt):
    """pTM_p >= 0.84 and pLDDT_r >= 0.78 (a missing value does not count against it)."""
    return not ((partner_ptm is not None and partner_ptm < WELL_FOLDED_PTM) or
                (rna_plddt is not None and rna_plddt < WELL_FOLDED_PLDDT))


def interface_probability(pae_min, partner_ptm=None, tm_familiar=None, seq_familiar=None, rna_plddt=None, **extra):
    """protein-RNA: P(correct interface) of one RNA chain as (min, max) over unknown inputs; familiar = familiar
    sequence or familiar fold (unknown if neither is known to be yes and either is unknown)."""
    if pae_min is None:
        return None, None
    fam = 1 if (seq_familiar or tm_familiar) else (None if (seq_familiar is None or tm_familiar is None) else 0)
    v = dict(pae_min=pae_min, familiar=fam, wellfolded=int(well_folded_partner(partner_ptm, rna_plddt)),
             partner_ptm=partner_ptm, tm_familiar=_b(tm_familiar), **extra)
    return _range(calibration()["interface"], v)


def joint(p_fold: float, p_interface: float, rho: Optional[float] = None) -> float:
    """P(both) from the two probabilities, joined by a Gaussian copula with correlation rho (calibration file):
    rho = 0 gives the product (independent sides), rho -> 1 the smaller of the two."""
    from scipy.stats import multivariate_normal, norm
    rho = calibration()["overall"]["rho"] if rho is None else rho
    z = [norm.ppf(min(max(p, 1e-9), 1 - 1e-9)) for p in (p_fold, p_interface)]
    return float(multivariate_normal.cdf(z, mean=[0, 0], cov=[[1, rho], [rho, 1]]))


def usable_probability(fold, interface):
    """P(usable RNA chain: well folded and placed correctly) from (min, max) pairs of P_fold and P_interface; the
    copula is increasing in both, so the range maps to a range."""
    if fold[0] is None or interface[0] is None:
        return None, None
    return joint(fold[0], interface[0]), joint(fold[1], interface[1])


def rna_only_probability(n_rna: int, rna_alignment: Optional[bool], rna_plddt=None, rna_pae=None, duplex=None):
    """RNA-only prediction: P(longest RNA chain well folded); unknown alignment -> range over both branches."""
    key = "rna_single" if n_rna == 1 else "rna_multi"
    branches = ([{True: "with_alignment", False: "without_alignment"}[bool(rna_alignment)]]
                if rna_alignment is not None else ["with_alignment", "without_alignment"])
    v = dict(plddt=None if rna_plddt is None else 100 * rna_plddt, pae=rna_pae, duplex=_b(duplex))
    ps = [p for br in branches for p in _range(calibration()[key][br], v) if p is not None]
    return (min(ps), max(ps)) if ps else (None, None)


def fmt(lo, hi) -> Optional[str]:
    if lo is None:
        return None
    a, b = int(100 * lo), int(100 * hi)              # truncated, so a shown 60% is always STRONG
    return f"{a}%" if a == b else f"{a}-{b}%"
