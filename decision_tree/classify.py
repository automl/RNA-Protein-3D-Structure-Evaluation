"""
Classify AF3 predictions with the decision framework.

    python -m decision_tree <model.cif | folder> [--rbp-template yes|no] [--familiar-seq yes|no] [--familiar-tm yes|no] [--rna-alignment yes|no] [--tsv out.tsv]

For every RNA chain of every model found, reports the subtree used (protein-RNA, single-chain RNA or RNA
complex) and the calibrated probabilities of probability.py with their levels (WEAK < 25% <= MODERATE < 60% <=
STRONG): P_fold (RNA well folded, LDDT_r > 0.85) and, for protein-RNA systems, P_interface (correct interface)
and P_usable = P_fold x P_interface. They are reliability estimates for the structure, not predictions of
binding affinity or motif register.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Iterable, List

from decision_tree.af3_output import read_prediction
from decision_tree.rules import expected_rna_tm, fold_tier, interface_tier, system_class
from decision_tree import framework as fw
from decision_tree import probability as pb
from decision_tree import rna_only

FIELDS = ["model", "chain", "length", "system_class", "subtree", "rna_alignment", "duplex_frac", "familiar",
          "rna_plddt", "rna_pae", "rna_iptm", "pae_min", "pae_partner", "partner_ptm", "warnings",
          "p_fold", "fold_level", "p_interface", "interface_level", "p_usable", "overall_level"]


def _yn(v):
    if v is None:
        return None
    return str(v).strip().lower() in ("1", "yes", "y", "true", "t")


def classify_values(n_protein: int, n_rna: int, length: int, rna_plddt=None, rna_iptm=None,
                    pae_min=None, novel: bool = True) -> dict:
    """System class and expected RNA TM-score of a chain from values alone."""
    cls = system_class(n_protein, n_rna)
    exp = expected_rna_tm(cls, length, novel=novel)
    return dict(system_class=cls, expected_rna_tm=exp["expected_tm"], expected_basis=exp["basis"],
                thin_bin=exp["thin"], fold_tier=fold_tier(rna_plddt),
                interface_tier=interface_tier(rna_iptm, pae_min) if n_protein else None)


def _span(pairs):
    pairs = [p for p in pairs if p[0] is not None]
    return (min(p[0] for p in pairs), max(p[1] for p in pairs)) if pairs else (None, None)


def _labels(prefix, rng):
    return {f"p_{prefix}": pb.fmt(*rng), f"{prefix}_level": pb.level_range(*rng)}


def fold_familiar_from_model(cif):
    """True/False: the AF3 model's longest RNA chain has a pre-cutoff RNA with US-align TM >= 0.5 (False below 20 nt,
    as in the design); None when the pre-cutoff chain library is not available"""
    try:
        from decision_tree.predicted_fold_familiarity import WORK, best_tm
        from decision_tree.validate_2026 import PRE_CHAINS, _pre_chain_list
        if not os.path.isdir(PRE_CHAINS):
            return None
        os.makedirs(WORK, exist_ok=True)
        tm, _ = best_tm(cif, os.path.basename(cif).rsplit(".", 1)[0], _pre_chain_list())
        return tm is not None and tm >= 0.5          # < 20 nt: no fold comparison, not familiar (as in the design)
    except Exception:
        return None


PDB_RNA_FASTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pdb_reference",
                             "pdb_rna_pre2021.fasta")


def classify_model(cif: str, novel: bool = True, domains=None, familiar_tm=None, familiar_seq=None, rbp_template=None,
                   rna_alignment=None, auto_familiar=True, reference_fasta=None) -> List[dict]:
    """Calibrated probabilities and levels for every RNA chain of one AF3 model (probability.py). Inputs not read from
    the model are optional; an unknown input gives the range of probabilities over its span. familiar_tm (fold
    familiarity of the AF3 model) is computed from the model when the pre-cutoff RNA chain library is available."""
    pred = read_prediction(cif)
    out = []
    rna_only_fold = None
    dup = None
    if pred["n_protein"] == 0 and pred["rna_chains"]:
        # RNA-only: one probability per prediction, from the longest RNA chain
        longest = max(pred["rna_chains"], key=lambda c: c["length"])
        dup = rna_only.duplex_fraction(cif) if pred["n_rna"] > 1 else None
        rna_only_fold = pb.rna_only_probability(pred["n_rna"], rna_alignment, longest["rna_plddt"],
                                                longest.get("rna_pae"),
                                                None if dup is None else dup >= rna_only.DUPLEX_FRAC)
    longest_pae, fam, seq_f = None, familiar_tm, familiar_seq
    if pred["n_protein"] and pred["rna_chains"]:
        longest_pae = max(pred["rna_chains"], key=lambda c: c["length"]).get("rna_pae")
        if fam is None and auto_familiar:
            fam = fold_familiar_from_model(cif)
        ref = reference_fasta or (PDB_RNA_FASTA if os.path.exists(PDB_RNA_FASTA) else None)
        if seq_f is None and ref:
            seq_f = any(fw.sequence_familiar(c["sequence"], ref) for c in pred["rna_chains"] if c.get("sequence"))
    for c in pred["rna_chains"]:
        lab = {k: v for k, v in classify_values(pred["n_protein"], pred["n_rna"], c["length"]).items()
               if k == "system_class"}
        if rna_only_fold is not None:
            lab.update(subtree="single-chain RNA" if pred["n_rna"] == 1 else "RNA complex",
                       rna_alignment=rna_alignment, duplex_frac=dup, **_labels("fold", rna_only_fold))
        elif pred["n_protein"]:
            lab["subtree"] = "protein-RNA"
            lab["familiar"] = bool(fam or seq_f) if (fam is not None and seq_f is not None) or fam or seq_f else None
            fold = pb.fold_probability(longest_pae, pred["n_rna"], fam, seq_f, rbp_template)
            iface = pb.interface_probability(c["pae_min"], c["partner_ptm"], fam, seq_f, c["rna_plddt"])
            w = fw.warnings(domains, rbp_template, c["partner_ptm"], c["rna_plddt"])
            lab.update(warnings="; ".join(w) if w else "", **_labels("fold", fold), **_labels("interface", iface),
                       p_usable=pb.fmt(*pb.usable_probability(fold, iface)),
                       overall_level=pb.level_range(*pb.usable_probability(fold, iface)))
        out.append(dict(model=os.path.basename(cif), chain=c["chain"], length=c["length"], rna_pae=c.get("rna_pae"),
                        rna_plddt=c["rna_plddt"], rna_iptm=c["rna_iptm"], pae_min=c["pae_min"],
                        pae_partner=c["partner"], partner_ptm=c["partner_ptm"], **lab))
    return out


def find_models(paths: Iterable[str]) -> List[str]:
    models = []
    for p in paths:
        if os.path.isdir(p):
            models += sorted(glob.glob(os.path.join(p, "**", "*_model_*.cif"), recursive=True))
        else:
            models.append(p)
    return models


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="AF3 model mmCIF files or folders containing them")
    ap.add_argument("--tsv", help="write the table to this file instead of stdout")
    tq = ap.add_argument_group("answers known before prediction (optional; unknown -> range of probabilities)")
    tq.add_argument("--rbp-template", help="yes/no: AF3 used a template from a protein-RNA complex (any protein chain)")
    tq.add_argument("--familiar-seq", help="yes/no: an RNA of a PDB entry released before 30 Sep 2021 has >= 80%% identity; "
                                           "computed from data/pdb_reference/pdb_rna_pre2021.fasta when present")
    tq.add_argument("--familiar-tm", help="yes/no: a pre-cutoff PDB RNA has the fold of the model's longest RNA chain "
                                          "(US-align TM >= 0.5); computed from the model when the pre-cutoff chain "
                                          "library is available")
    tq.add_argument("--domains", help="comma-separated InterPro or Pfam domains of the protein (flag only)")
    tq.add_argument("--rna-alignment", help="RNA-only predictions: yes/no, the RNA alignment was used (local AF3 with MSA); "
                                            "AlphaFold Server runs use none (no); unknown -> range of levels")
    a = ap.parse_args(argv)
    doms = a.domains.split(",") if a.domains else None
    rows = [r for m in find_models(a.paths)
            for r in classify_model(m, domains=doms, familiar_tm=_yn(a.familiar_tm), familiar_seq=_yn(a.familiar_seq),
                                    rbp_template=_yn(a.rbp_template), rna_alignment=_yn(a.rna_alignment))]
    fh = open(a.tsv, "w", newline="") if a.tsv else sys.stdout
    w = csv.DictWriter(fh, FIELDS, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    if a.tsv:
        fh.close()
        print(f"{len(rows)} RNA chains -> {a.tsv}")
    return 0
