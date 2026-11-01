"""
Read one AlphaFold3 prediction (AlphaFold Server or local AF3 output) without any reference.

Inputs per prediction (same folder, same stem):
  <stem>_model_<k>.cif                  chain types and lengths
  <stem>_summary_confidences_<k>.json   chain_iptm, chain_pair_pae_min
  <stem>_full_data_<k>.json             atom_plddts, atom_chain_ids (RNA pLDDT; optional)

Per RNA chain this yields exactly the AF3 values used in the paper:
  rna_iptm  = chain_iptm of the RNA chain;
  pae_min   = chain_pair_pae_min[protein][RNA], minimised over the protein chains
              (the paper takes the protein chain that contacts the RNA in the deposited
              structure; without a reference the best-scoring partner is used);
  rna_plddt = mean atom pLDDT of the RNA chain, 0-1 scale.
Chain order in the JSON matrices is the order of first appearance in the mmCIF atom_site.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

RNA_RES = {"A", "C", "G", "U", "N", "PSU", "5MC", "OMG", "OMC", "OMU", "1MA", "2MG", "7MG", "M2G", "H2U",
           "5MU", "4SU", "I", "INO", "6MZ"}
DNA_RES = {"DA", "DC", "DG", "DT", "DU"}
AA_RES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
          "PRO", "SER", "THR", "TRP", "TYR", "VAL", "SEC", "PYL", "MSE", "UNK"}


def _atom_site(cif: str):
    cols, rows = [], []
    with open(cif) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1])
            elif cols and line.startswith(("ATOM", "HETATM")):
                rows.append(line.split())
            elif cols and rows and line.startswith(("#", "loop_", "_")):
                break
    return cols, rows


def read_chains(cif: str) -> Dict[str, dict]:
    """{chain: {type, length, index}} in order of first appearance (= AF3 JSON order)."""
    cols, rows = _atom_site(cif)
    ia, ir, iseq = cols.index("auth_asym_id"), cols.index("label_comp_id"), cols.index("label_seq_id")
    chains: Dict[str, dict] = {}
    for r in rows:
        ch = r[ia]
        c = chains.setdefault(ch, dict(index=len(chains), residues=set(), names={}))
        c["residues"].add(r[iseq])
        c["names"][r[iseq]] = r[ir]
    for ch, c in chains.items():
        by_pos = c.pop("names")
        names = list(by_pos.values())
        c["sequence"] = "".join(n if n in "ACGU" else "N" for _, n in sorted(
            by_pos.items(), key=lambda t: int(t[0]) if t[0].lstrip("-").isdigit() else 0))
        c["length"] = len(c.pop("residues"))
        n_rna = sum(n in RNA_RES for n in names)
        n_dna = sum(n in DNA_RES for n in names)
        n_aa = sum(n in AA_RES for n in names)
        c["type"] = max((("rna", n_rna), ("dna", n_dna), ("protein", n_aa), ("other", 0.5)),
                        key=lambda t: t[1])[0]
    return chains


def companion(cif: str, kind: str) -> Optional[str]:
    """Path of the summary_confidences / full_data JSON that belongs to a model mmCIF."""
    m = re.search(r"_model_(\d+)\.cif$", cif)
    if not m:
        return None
    name = os.path.basename(cif[: m.start()]) + f"_{kind}_{m.group(1)}.json"
    folder = os.path.dirname(cif)
    for d in (folder, os.path.dirname(folder)):       # JSONs may sit one level above the mmCIFs
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def read_prediction(cif: str, summary: Optional[str] = None, full_data: Optional[str] = None,
                    with_plddt: bool = True) -> dict:
    """AF3 values per RNA chain. with_plddt=False skips the (large) full_data JSON."""
    chains = read_chains(cif)
    summary = summary or companion(cif, "summary_confidences")
    full_data = (full_data or companion(cif, "full_data")) if with_plddt else None
    s = json.load(open(summary)) if summary else {}
    chain_iptm = s.get("chain_iptm") or []
    pae = s.get("chain_pair_pae_min") or []
    plddt, global_pae, intra = {}, None, {}
    if full_data:
        try:
            f = json.load(open(full_data))
        except (json.JSONDecodeError, OSError):      # truncated download: no pLDDT
            f = {}
        acc = {}
        for ch, v in zip(f.get("atom_chain_ids", []), f.get("atom_plddts", [])):
            acc.setdefault(ch, []).append(v)
        plddt = {ch: sum(v) / len(v) / 100.0 for ch, v in acc.items() if v}
        pae_m = f.get("pae") or []
        vals = [x for row in pae_m for x in row if x is not None]
        global_pae = sum(vals) / len(vals) if vals else None
        tok = f.get("token_chain_ids") or []
        intra = {}
        if pae_m and tok:
            import numpy as np
            arr, ids = np.asarray(pae_m, dtype=np.float32), np.asarray(tok)
            for ch in dict.fromkeys(tok):
                m = ids == ch
                intra[ch] = float(arr[np.ix_(m, m)].mean())
    prot = [c for c, d in chains.items() if d["type"] == "protein"]
    rna: List[dict] = []
    for ch, d in chains.items():
        if d["type"] != "rna":
            continue
        i = d["index"]
        ip = chain_iptm[i] if i < len(chain_iptm) else None
        pairs = [(pae[chains[p]["index"]][i], p) for p in prot
                 if chains[p]["index"] < len(pae) and i < len(pae[chains[p]["index"]])]
        best = min(pairs) if pairs else (None, None)
        rna.append(dict(chain=ch, length=d["length"], sequence=d.get("sequence", ""), rna_iptm=ip,
                        pae_min=best[0], partner=best[1], rna_plddt=plddt.get(ch), rna_pae=intra.get(ch)))
    chain_ptm = s.get("chain_ptm") or []
    ptm_by_chain = {p: chain_ptm[chains[p]["index"]] for p in prot if chains[p]["index"] < len(chain_ptm)}
    protein_ptms = list(ptm_by_chain.values())
    for c in rna:
        c["partner_ptm"] = ptm_by_chain.get(c["partner"])
    return dict(model=cif, summary=summary, full_data=full_data, n_protein=len(prot),
                n_rna=sum(d["type"] == "rna" for d in chains.values()), rna_chains=rna,
                protein_ptms=[x for x in protein_ptms if x is not None], global_pae=global_pae,
                ranking_score=s.get("ranking_score"))
