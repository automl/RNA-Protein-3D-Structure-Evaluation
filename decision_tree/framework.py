"""
Scored decision framework (v2): the version validated and calibrated in the paper.

Each side adds up three kinds of evidence and maps the score to three calibrated levels.

  FOLD       score = AF3 RNA confidence level (0 bad, 1 mid, 2 good)  + 1 if the RNA is familiar
  INTERFACE  score = interface tier (0 bad, 1 mid, 2 good)            + 1 if the RNA is familiar
                     - 1 for each warning sign (selected on pre-cutoff data, see calibration.json)
  level      score <= 1 -> WEAK / UNRELIABLE, 2 -> MODERATE / CAUTION, 3 -> STRONG / RELIABLE
             the paper reports the interface at two levels (interface_final): CAUTION and RELIABLE did not
             separate on post-cutoff targets and are merged into STRONG, UNRELIABLE is WEAK

Familiar RNA: a pre-cutoff PDB RNA with >= 80% sequence identity (identical aligned columns over
the longer sequence) or a US-align RNA TM-score >= 0.5 (RNAs >= 20 nt). Unknown familiarity gives a
range of levels; for a training-era target pass familiar=True.

The design choices (which RNA confidence measure, which warning signs) were made on pre-cutoff
targets only and the levels calibrated on post-cutoff (held-out) targets by decision_tree.develop;
they are read from calibration.json (rates with Wilson 95% CIs).
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from decision_tree.rules import interface_tier
from decision_tree.tree import PROTEIN_PTM_HIGH, PLDDT_CONFIDENT, domain_group

_CAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "plots", "all_chains", "decision_tree", "calibration.json")
_PACKAGED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

FOLD_LEVELS = ("WEAK", "MODERATE", "STRONG")
INTERFACE_LEVELS = ("UNRELIABLE", "CAUTION", "RELIABLE")
INTERFACE_FINAL = {"UNRELIABLE": "WEAK", "CAUTION": "STRONG", "RELIABLE": "STRONG"}
TIER_SCORE = {"bad": 0, "mid": 1, "good": 2}


def calibration() -> dict:
    for p in (_PACKAGED, _CAL):
        if os.path.exists(p):
            return json.load(open(p))
    return {}


def _design():
    return calibration().get("design", {})


# --------------------------------------------------------------------------- #
# fold confidence (the measure chosen in decision_tree.develop)
# --------------------------------------------------------------------------- #
def fold_confidence(rna_plddt=None, global_pae=None, rna_pae=None) -> Optional[int]:
    d = _design().get("fold_confidence", {"function": "conf_plddt", "params": {}})
    f, kw = d.get("function"), d.get("params", {})
    p = None if rna_plddt is None else (rna_plddt / 100 if rna_plddt > 1 else rna_plddt)
    if f == "conf_plddt":
        return None if p is None else (2 if p >= 0.775 else 0 if p < 0.55 else 1)
    if f == "conf_plddt_gpae":
        if p is None or global_pae is None:
            return None
        return 2 if (p >= 0.78 and global_pae <= kw.get("cut", 10)) else 0 if p < 0.55 else 1
    if f == "conf_plddt_rnapae":
        if p is None or rna_pae is None:
            return None
        return 2 if (p >= 0.78 and rna_pae <= kw.get("cut", 8)) else 0 if p < 0.55 else 1
    if f == "conf_rnapae":
        if rna_pae is None:
            return None
        return 2 if rna_pae <= kw.get("lo", 6) else 0 if rna_pae > kw.get("hi", 12) else 1
    raise ValueError(f"unknown fold confidence {f}")


def warnings(domains: Optional[Iterable[str]] = None, rbp_template: Optional[bool] = None,
             partner_ptm: Optional[float] = None, rna_plddt: Optional[float] = None) -> list:
    """Warning signs selected on pre-cutoff data (names as in decision_tree.develop)."""
    sel = _design().get("interface_downgrades", [])
    out = []
    if "large" in sel and domain_group(domains) == "large":
        out.append("large RBP domain")
    if "no_template" in sel and rbp_template is False:
        out.append("no protein-RNA template")
    if "low_ptm_or_plddt" in sel:
        p = None if rna_plddt is None else (rna_plddt / 100 if rna_plddt > 1 else rna_plddt)
        if (partner_ptm is not None and partner_ptm < PROTEIN_PTM_HIGH) or (p is not None and p < PLDDT_CONFIDENT):
            out.append("partner or RNA not well folded")
    return out


def _levels(scores, names):
    lv = sorted({0 if s <= 1 else (1 if s == 2 else 2) for s in scores})
    return names[lv[0]] if len(lv) == 1 else f"{names[lv[0]]}..{names[lv[-1]]}"


def _familiar_values(familiar):
    return (0, 1) if familiar is None else (int(bool(familiar)),)


def fold_level(rna_plddt=None, global_pae=None, rna_pae=None, familiar=None) -> Optional[str]:
    c = fold_confidence(rna_plddt, global_pae, rna_pae)
    return None if c is None else _levels([c + f for f in _familiar_values(familiar)], FOLD_LEVELS)


def interface_level(rna_iptm=None, pair_pae=None, familiar=None, domains=None, rbp_template=None,
                    partner_ptm=None, rna_plddt=None) -> Optional[str]:
    t = interface_tier(rna_iptm, pair_pae)
    if t is None:
        return None
    base = TIER_SCORE[t] - len(warnings(domains, rbp_template, partner_ptm, rna_plddt))
    return _levels([base + f for f in _familiar_values(familiar)], INTERFACE_LEVELS)


def calibrated_rate(side: str, level: str, system: str = "all", period: str = "post") -> Optional[dict]:
    """Held-out rate (+ 95% CI) of the outcome for a level: fold = % RNAs with LDDT_r > 0.85,
    interface = % correct interfaces. Ranges and unknown levels return None."""
    cal = calibration().get(side, {})
    cal = cal.get(system, cal) if side == "fold" else cal
    return cal.get(period, {}).get(level)


def interface_final(level: Optional[str]) -> Optional[str]:
    """Interface level as reported in the paper (WEAK / STRONG) from the development level; ranges are kept."""
    if level is None:
        return None
    parts = level.split("..")
    lo, hi = INTERFACE_FINAL[parts[0]], INTERFACE_FINAL[parts[-1]]
    return lo if lo == hi else f"{lo}..{hi}"


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def calibrated_rate_final(level: str, period: str = "post") -> Optional[dict]:
    """Held-out % correct interfaces for a WEAK / STRONG level, pooling the development levels it merges."""
    cal = calibration().get("interface", {}).get(period, {})
    parts = [cal.get(x) for x, f in INTERFACE_FINAL.items() if f == level]
    if not parts or any(p is None or not p.get("n") for p in parts):
        return None
    n = sum(p["n"] for p in parts)
    k = sum(round(p["rate"] * p["n"]) for p in parts)
    lo, hi = _wilson(k, n)
    return dict(rate=k / n, ci_lo=lo, ci_hi=hi, n=n)


# --------------------------------------------------------------------------- #
# familiarity from sequence (optional helper)
# --------------------------------------------------------------------------- #
def sequence_familiar(seq: str, reference_fasta: str, threshold: float = 0.8) -> Optional[bool]:
    """True if any reference RNA reaches >= threshold identity (identical aligned columns over the
    longer sequence); reference_fasta = pre-cutoff PDB RNA sequences (e.g. pdb_seqres RNA entries)."""
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score, al.open_gap_score, al.extend_gap_score = 2, -3, -5, -2
    seq = seq.upper().replace("T", "U")
    refs, cur = [], []
    for line in open(reference_fasta):
        if line.startswith(">"):
            if cur:
                refs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip().upper().replace("T", "U"))
    if cur:
        refs.append("".join(cur))
    for r in refs:
        if min(len(r), len(seq)) / max(len(r), len(seq)) < threshold:
            continue
        a = al.align(seq, r)[0]
        same = sum(sum(x == y for x, y in zip(seq[a0:a1], r[b0:b1])) for (a0, a1), (b0, b1) in zip(*a.aligned))
        if same / max(len(seq), len(r)) >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# v3 decision tree: fold side v3 (decision_tree.develop_v3), interface side v2 (above)
# --------------------------------------------------------------------------- #
_PACKAGED_V3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_v3.json")


def calibration_v3() -> dict:
    return json.load(open(_PACKAGED_V3)) if os.path.exists(_PACKAGED_V3) else {}


def _v3_hit(q, answers):
    """Points of one v3 question; None if its answer is unknown."""
    f, kind, thr, sign = q
    x = answers.get(f)
    if x is None:
        return None
    return sign if (bool(x) if kind == "bool" else float(x) >= thr) else 0


def fold_level_v3(rna_pae=None, n_rna=None, rbp_template=None, seq_familiar=None, tm_familiar=None,
                  rna_plddt=None, global_pae=None) -> Optional[str]:
    """RNA fold level of a protein-RNA prediction (v3; no MSA).

    score = AF3 RNA confidence level (mean intra-RNA PAE: <= 4 A 2, 4-8 A 1, > 8 A 0)
            + 1 if >= 2 RNA chains in the model
            + 1 if AF3 used a protein-RNA (RBP) complex template, for any protein chain
            + 1 if a pre-cutoff PDB RNA has the same fold (US-align TM >= 0.5, RNA >= 20 nt)
            + 1 if a pre-cutoff PDB RNA has >= 80% sequence identity
    score <= 2 WEAK, 3 MODERATE, >= 4 STRONG (cut-offs and rates in calibration_v3.json).
    Unknown answers give a range of levels (e.g. "WEAK..MODERATE").
    """
    cal = calibration_v3().get("fold", {})
    if not cal:
        return None
    base = fold_confidence(rna_plddt, global_pae, rna_pae)
    if base is None:
        return None
    answers = dict(n_rnas=n_rna, rbp_template=rbp_template, seq_familiar=seq_familiar, tm_familiar=tm_familiar)
    lo = hi = base
    for q in cal["questions"]:
        h = _v3_hit(q, answers)
        if h is None:                      # unknown: either 0 or the question's points
            lo, hi = lo + min(0, q[3]), hi + max(0, q[3])
        else:
            lo, hi = lo + h, hi + h
    c1, c2 = cal["cuts"]
    lv = lambda s: 0 if s <= c1 else (1 if s <= c2 else 2)
    a, b = lv(lo), lv(hi)
    return FOLD_LEVELS[a] if a == b else f"{FOLD_LEVELS[a]}..{FOLD_LEVELS[b]}"


def calibrated_rate_v3(side: str, level: str, dataset: str = "2026_heldout") -> Optional[dict]:
    """v3 fold: rates on the post-cutoff development set or the held-out 2026 set; interface: v2."""
    cal = calibration_v3().get(side, {})
    if side == "fold":
        return cal.get("rates", {}).get(dataset, {}).get(level)
    return cal.get("rates", {}).get("post", {}).get(level)
