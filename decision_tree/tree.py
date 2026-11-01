"""
The paper's decision tree, implemented node by node.

Each side asks four yes/no questions in a fixed order; the first "no" ends at the leaf of that
step, four "yes" reach the top leaf.

RNA FOLD nodes (single-chain RNA, or the RNA chain of a complex)
  confident   AF3 confident? RNA pLDDT >= 0.78 and global PAE <= 10 A
  family      conserved RNA family? (Rfam family assigned: tRNA, rRNA, ribozyme, ... vs synthetic)
  msa         deep RNA MSA? (> 1,000 aligned sequences)
  canonical   canonical fold? (no pseudoknot in the predicted structure)
  leaves      WEAK < LOW-MOD < MODERATE < MOD-HIGH < STRONG

INTERFACE nodes (protein-RNA complexes only)
  confident   AF3 interface confident? RNA ipTM >= 0.5 and chain-pair PAE <= 8 A
  domain      compact RBP domain? (no dsRBD / RNase III / Dicer domain)
  template    protein-RNA (RBP) template? (AF3 used a protein-RNA complex template)
  wellfolded  interface chains well folded? (protein partner pTM >= 0.84 and RNA pLDDT >= 0.78)
  leaves      UNRELIABLE < CAUTION < MODERATE < LIKELY OK < RELIABLE

Node order. The tree first drawn for the paper (ORDER_DRAWN) put the pre-prediction questions
first and AF3's confidence last; validated on held-out labels it ranked targets close to chance
(decision_tree.validate_tree). The default (ORDER) asks AF3's confidence first; the order of the
remaining questions was chosen on pre-cutoff targets only and tested on post-cutoff ones.

Every answer may be True, False or None (unknown). An unknown answer is followed down both
branches and the result is the range of reachable leaves (e.g. "MODERATE..STRONG").
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

FOLD_LEAVES = ("WEAK", "LOW-MOD", "MODERATE", "MOD-HIGH", "STRONG")
INTERFACE_LEAVES = ("UNRELIABLE", "CAUTION", "MODERATE", "LIKELY OK", "RELIABLE")

PLDDT_CONFIDENT = 0.78        # RNA pLDDT, 0-1
GLOBAL_PAE_LOW = 10.0         # A
PROTEIN_PTM_HIGH = 0.84
MSA_DEEP = 1000               # aligned RNA sequences
IPTM_CONFIDENT = 0.5
PAIR_PAE_CONFIDENT = 8.0      # A

ORDER_DRAWN = {"fold": ("family", "msa", "canonical", "confident"),
               "interface": ("wellfolded", "template", "domain", "confident")}
# confidence first; remaining order selected on pre-cutoff data (decision_tree.validate_tree)
ORDER = {"fold": ("confident", "msa", "canonical", "family"),
         "interface": ("confident", "wellfolded", "template", "domain")}

LARGE_DOMAINS = {"dsRBD_dom", "RNase_III_dom", "Dicer_dimerisation_dom", "Dicer_dimerisation",          # InterPro short names
                 "PF00035", "PF14709", "PF00636", "PF14622", "PF03368",            # Pfam
                 "dsRBD", "dsrm", "RNase III", "Ribonuclease_3", "Dicer_dimer"}
COMPACT_DOMAINS = {"RRM_dom", "Pumilio_RNA-bd_rpt", "PUM-HD", "Piwi", "PAZ_dom", "Argonaute_Mid_dom",
                   "PF00076", "PF14259", "PF13893", "PF00806", "PF02171", "PF02170",
                   "RRM", "PUM", "Pumilio", "PAZ"}


def _gt(x, t):
    return None if x is None else x >= t


def _le(x, t):
    return None if x is None else x <= t


def _and(*vals):
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def _plddt01(p):
    return None if p is None else (p / 100 if p > 1 else p)


def domain_group(domains: Optional[Iterable[str]]) -> Optional[str]:
    """'large', 'compact', 'other' or None (no annotation)."""
    if domains is None:
        return None
    s = {d.strip() for d in domains if d and d.strip() not in ("", "-")}
    if not s:
        return None
    if s & LARGE_DOMAINS:
        return "large"
    return "compact" if s & COMPACT_DOMAINS else "other"


def fold_answers(conserved_family=None, msa_depth=None, pseudoknot=None, rna_plddt=None,
                 global_pae=None) -> Dict[str, Optional[bool]]:
    return {"confident": _and(_gt(_plddt01(rna_plddt), PLDDT_CONFIDENT), _le(global_pae, GLOBAL_PAE_LOW)),
            "family": conserved_family,
            "msa": None if msa_depth is None else msa_depth > MSA_DEEP,
            "canonical": None if pseudoknot is None else not pseudoknot}


def interface_answers(protein_ptms=None, rna_plddt=None, rbp_template=None, domains=None,
                      rna_iptm=None, pair_pae=None) -> Dict[str, Optional[bool]]:
    ptm_ok = None if not protein_ptms else min(protein_ptms) >= PROTEIN_PTM_HIGH
    grp = domain_group(domains)
    return {"confident": _and(_gt(rna_iptm, IPTM_CONFIDENT), _le(pair_pae, PAIR_PAE_CONFIDENT)),
            "domain": None if grp is None else grp != "large",
            "template": rbp_template,
            "wellfolded": _and(ptm_ok, _gt(_plddt01(rna_plddt), PLDDT_CONFIDENT))}


def walk(answers: Dict[str, Optional[bool]], order: Sequence[str], leaves: Sequence[str]) -> List[str]:
    """Reachable leaves: the i-th question answered 'no' exits to leaves[i]; all 'yes' -> leaves[-1]."""
    reach = []

    def go(i):
        if i == len(order):
            reach.append(leaves[-1])
            return
        a = answers[order[i]]
        if a is not True:
            reach.append(leaves[i])
        if a is not False:
            go(i + 1)
    go(0)
    rank = {l: k for k, l in enumerate(leaves)}
    return sorted(set(reach), key=rank.get)


def label(reach: List[str]) -> str:
    return reach[0] if len(reach) == 1 else f"{reach[0]}..{reach[-1]}"


def fold_leaf(order: Sequence[str] = ORDER["fold"], **kw) -> str:
    """RNA-fold leaf. Keyword arguments as in fold_answers."""
    return label(walk(fold_answers(**kw), order, FOLD_LEAVES))


def interface_leaf(order: Sequence[str] = ORDER["interface"], **kw) -> str:
    """Interface leaf. Keyword arguments as in interface_answers."""
    return label(walk(interface_answers(**kw), order, INTERFACE_LEAVES))
