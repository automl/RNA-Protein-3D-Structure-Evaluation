"""
RNA-only subtrees of the decision framework (predictions without a protein chain).

  single-chain RNA      AF3 confidence only; the measure depends on whether the RNA alignment was used
                          with alignment     RNA pLDDT (0-1 scale)         WEAK <= c1 < MODERATE <= c2 < STRONG
                          without alignment  intra-RNA PAE (A)             WEAK >= c1 > MODERATE >= c2 > STRONG
  RNA complex           a duplex in the AF3 model (>= 60 % of the nucleotides paired with another chain) is
                          STRONG; otherwise the intra-RNA PAE of the longest RNA chain at the cut-offs of the
                          non-duplex development RNAs, per alignment branch

Cut-offs and development rates are read from decision_tree/calibration_rna_single.json (develop_rna.py) and
decision_tree/calibration_rna_multi_v2.json (develop_rna_multi.py --v2). An unknown alignment answer gives a
range of levels when the two branches disagree.
"""
from __future__ import annotations

import json
import os
from typing import Optional

LEVELS = ("WEAK", "MODERATE", "STRONG")
DUPLEX_FRAC = 0.60
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAL = {"single": os.path.join(_HERE, "calibration_rna_single.json"),
        "multi": os.path.join(_HERE, "calibration_rna_multi_v2.json")}
_BRANCH = {True: "with_alignment", False: "without_alignment"}


def calibration(subtree: str) -> dict:
    p = _CAL[subtree]
    return json.load(open(p)) if os.path.exists(p) else {}


def duplex_fraction(cif: str) -> Optional[float]:
    """Share of RNA nucleotides with a Watson-Crick-like hydrogen bond (purine N1 - pyrimidine N3 < 3.3 A) to a
    nucleotide of another chain, read from the AF3 model (no experimental structure needed)."""
    import gemmi
    if not os.path.exists(cif):
        return None
    st = gemmi.read_structure(cif)
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    atoms = []
    for ch in st[0]:
        pol = ch.get_polymer()
        if pol.check_polymer_type() != gemmi.PolymerType.Rna:
            continue
        for r in pol:
            a = r.find_atom("N1", "*") if r.name in ("A", "G") else r.find_atom("N3", "*") if r.name in ("C", "U") else None
            if a:
                atoms.append((ch.name, r.seqid.num, r.name in ("A", "G"), a.pos))
    paired = set()
    for i, (c1, s1, pu1, p1) in enumerate(atoms):
        for c2, s2, pu2, p2 in atoms[i + 1:]:
            if c1 != c2 and pu1 != pu2 and p1.dist(p2) < 3.3:
                paired.update({(c1, s1), (c2, s2)})
    return len(paired) / len(atoms) if atoms else None


def _conf_level(value, cuts):
    # rounded so that a value exactly at a tertile cut-off (cut-offs are observed values) falls on the same side
    # as in development whatever the summation order of the pLDDT mean
    v, c = round(value, 6), [round(x, 6) for x in cuts]
    return 0 if v <= c[0] else 2 if v > c[1] else 1


def _branch_level(subtree, branch, rna_plddt, rna_pae, duplex):
    d = calibration(subtree).get(branch)
    if not d:
        return None
    if subtree == "multi" and duplex:
        return 2
    metric = d["metric"]
    if metric == "plddt":
        v = None if rna_plddt is None else (rna_plddt / 100 if rna_plddt > 1 else rna_plddt)
    elif metric == "rna_pae":
        v = None if rna_pae is None else -rna_pae          # cut-offs are stored for the negated PAE
    else:
        raise ValueError(f"RNA-only subtree: unsupported measure {metric}")
    return None if v is None else _conf_level(v, d["conf_cuts"])


def rna_only_level(n_rna: int, rna_alignment: Optional[bool], rna_plddt=None, rna_pae=None,
                   duplex_frac=None) -> dict:
    """Level of an RNA-only prediction. rna_plddt: mean RNA pLDDT (0-1 or 0-100); rna_pae: mean intra-RNA PAE (A)
    of the longest RNA chain; duplex_frac: duplex_fraction() of the model (multi-chain only)."""
    subtree = "single" if n_rna == 1 else "multi"
    duplex = subtree == "multi" and duplex_frac is not None and duplex_frac >= DUPLEX_FRAC
    branches = [_BRANCH[rna_alignment]] if rna_alignment is not None else list(_BRANCH.values())
    lv = [x for x in (_branch_level(subtree, b, rna_plddt, rna_pae, duplex) for b in branches) if x is not None]
    if not lv:
        level = None
    else:
        lo, hi = min(lv), max(lv)
        level = LEVELS[lo] if lo == hi else f"{LEVELS[lo]}..{LEVELS[hi]}"
    rate = None
    if level and ".." not in level and rna_alignment is not None:
        rate = calibration(subtree).get(_BRANCH[rna_alignment], {}).get("dev_rates", {}).get(level)
    return dict(subtree="single-chain RNA" if subtree == "single" else "RNA complex", level=level,
                duplex=duplex if subtree == "multi" else None, rate=rate)
