"""
Exact residue mapping between experimental and AF3-predicted protein-RNA complexes, and the
per-interface AAmatch / RNAmatch scores built on it (no motif sliding).

Why: AAMotif / RNAMotif (the paper's interface residues: H-bond + van der Waals contacts) are
numbered differently on the two sides -- predicted motifs use AF3 sequence positions, experimental
AAMotif uses PDB author numbering and experimental RNAMotif uses positions in the modelled chain --
and predicted and experimental chain IDs differ. Instead of sliding each motif along the sequence,
every experimental residue is carried to its predicted counterpart through the mapping that
OpenStructure used when it scored the complex (the full-complex `ost compare-structures` JSON,
sequence mapping, the same run that produced RBP_DockQ):

  experimental auth chain + residue  --mmCIF _atom_site-->  label_asym_id chain + polymer index
  --OST `aln` (reference vs model alignment)-->  model chain index  --=-->  AF3 sequence position
  with the chain pairing taken from OST `chain_mapping` (label_asym_id -> model chain).

Per interface (one protein chain, one RNA chain; union of experimental and predicted interfaces,
keyed by the PREDICTED pair as RBP_DockQ) the protein and RNA scores are the Jaccard overlap of the
mapped experimental and the predicted contact residues. A complex-level score is the
union-size-weighted mean over its interfaces (the same weighting as the previous AAmatch/RNAmatch).
"""

import ast
import glob
import json
import os
import re
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
# Modified nucleotides -> parent (same table as PHCorrelationAnalyzer.update_rna_match_scores)
MODIFIED_NT = {
    '6MZ': 'A', 'PSU': 'U', '5MC': 'C', 'OMC': 'C', '4OC': 'C', '5MU': 'U',
    'OMU': 'U', 'UR3': 'U', 'A2M': 'A', 'MA6': 'A', '2MG': 'G', 'OMG': 'G',
    '7MG': 'G', 'RSQ': 'G', '5CM': 'C', 'C34': 'C', '5HC': 'C', '6OG': 'G',
    '6MA': 'A', '1CC': 'C', '8OG': 'G', '5FC': 'C', '3DR': 'T'
}
MIN_NAME_AGREEMENT = 0.8   # min share of motif residues whose names agree with the structure
MIN_FALLBACK_IDENTITY = 0.9   # min share of experimental residues identical in the AF3 chain

_AL = Align.PairwiseAligner()
_AL.mode = "global"
_AL.match_score, _AL.mismatch_score = 2, -1
_AL.open_gap_score, _AL.extend_gap_score = -4, -0.5


def one_letter(name):
    name = name.upper()
    if name in AA3:
        return AA3[name]
    if name in MODIFIED_NT:
        return MODIFIED_NT[name]
    if name in ("A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU"):
        return name[-1]
    return "X"


# --------------------------------------------------------------------------- #
# Locating inputs
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _file_index(root):
    """basename -> path for every reference mmCIF (*/pdb/*.cif) and OST JSON (*/lddt/*_lddt.json)."""
    idx = {}
    for pattern in ("**/pdb/*.cif", "**/lddt/*_lddt.json"):
        for p in glob.glob(os.path.join(root, pattern), recursive=True):
            idx.setdefault(os.path.basename(p).lower(), p)
    return idx


def locate_inputs(exp_db_id, pred_file_name, search_roots):
    """(reference mmCIF, full-complex OST JSON) for one complex, or (None, None)."""
    pid = str(exp_db_id).strip().lower()[:4]
    stem = os.path.splitext(os.path.basename(str(pred_file_name or "")))[0].lower()
    cif = js = None
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        idx = _file_index(os.path.abspath(root))
        cif = cif or idx.get(f"{pid}.cif")
        js = js or (idx.get(f"{stem}_lddt.json") if stem else None) or idx.get(f"fold_{pid}_s1_model_0_lddt.json")
        if cif and js:
            break
    return cif, js


# --------------------------------------------------------------------------- #
# Mapping one complex
# --------------------------------------------------------------------------- #
class ComplexMapping:
    """Experimental (auth chain, residue) -> predicted (chain, sequence position) for one complex.

    Chains come from OpenStructure's chain_mapping and reference/model alignment. OST cannot map
    U-rich / U-repeat chains (it drops them from its mapping or fails on the whole complex) and
    leaves out very short chains; those chains are paired with a free predicted chain by sequence
    identity (identical copies: the copy whose contacts overlap best, as OST/DockQ choose symmetric
    copies) and their residues mapped by exact global alignment to the full AF3 sequence
    (`assign_fallback`). `source[label chain]` records "ost" or "sequence".
    """

    def __init__(self, cif_path, ost_json_path, pred_seqs=None):
        d = json.load(open(ost_json_path)) if ost_json_path else {}
        self.chain_mapping = dict(d.get("chain_mapping") or {})    # label_asym_id -> model chain
        self._aln = {}
        for blk in d.get("aln", []) or []:
            lines = blk.strip().split("\n")
            if len(lines) >= 4:
                self._aln[lines[0].split(":", 1)[1]] = (lines[1], lines[3])
        self._read_cif(cif_path)
        self.pred_seqs = dict(pred_seqs or {})                      # predicted chain -> sequence
        self.source = {la: "ost" for la in self.chain_mapping}
        self._index = {}                                            # label chain -> {polymer idx: model idx}

    def _read_cif(self, path):
        m = MMCIF2Dict(path)
        first_model = m["_atom_site.pdbx_PDB_model_num"][0]
        ins = m.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(m["_atom_site.label_asym_id"]))
        res = OrderedDict()
        for la, au, sid, ic, lsid, comp, mdl in zip(
                m["_atom_site.label_asym_id"], m["_atom_site.auth_asym_id"], m["_atom_site.auth_seq_id"],
                ins, m["_atom_site.label_seq_id"], m["_atom_site.label_comp_id"],
                m["_atom_site.pdbx_PDB_model_num"]):
            if mdl != first_model or lsid in (".", "?"):            # first model, polymer residues
                continue
            res.setdefault((la, sid, "" if ic in ("?", ".") else ic), (au, comp))
        self.label_residues = OrderedDict()                         # label chain -> [(auth num, icode, name)]
        self.auth_to_label = {}
        for (la, sid, ic), (au, comp) in res.items():
            self.label_residues.setdefault(la, []).append((sid, ic, comp))
            self.auth_to_label.setdefault(au, la)

    def _seq(self, label_chain):
        return "".join(one_letter(n) for _, _, n in self.label_residues.get(label_chain, []))

    @staticmethod
    def _exact_alignment(a, b):
        """{i: j} for identical residues in the best global alignment of a and b."""
        out = {}
        if not a or not b:
            return out
        aln = _AL.align(a, b)[0]
        for (s0, s1), (t0, t1) in zip(*aln.aligned):
            for k in range(s1 - s0):
                if a[s0 + k] == b[t0 + k]:
                    out[s0 + k] = t0 + k
        return out

    def _model_index(self, label_chain):
        """{polymer index in label chain: model residue index}."""
        if label_chain in self._index:
            return self._index[label_chain]
        out = {}
        if self.source.get(label_chain) == "ost" and label_chain in self._aln:
            ref_al, mdl_al = self._aln[label_chain]                 # OST reference/model alignment
            ref_to_mdl, i, j = {}, 0, 0
            for a, b in zip(ref_al, mdl_al):
                if a != "-" and b != "-":
                    ref_to_mdl[i] = j
                i += a != "-"
                j += b != "-"
            ost_ref, cif_seq = ref_al.replace("-", ""), self._seq(label_chain)
            # usual case identical residue lists; otherwise OST dropped a few residues
            cif_to_ref = ({k: k for k in range(len(cif_seq))} if cif_seq == ost_ref
                          else self._exact_alignment(cif_seq, ost_ref))
            out = {c: ref_to_mdl[r] for c, r in cif_to_ref.items() if r in ref_to_mdl}
        elif self.source.get(label_chain) == "sequence":
            out = self._exact_alignment(self._seq(label_chain),
                                        self.pred_seqs.get(self.chain_mapping[label_chain], ""))
        self._index[label_chain] = out
        return out

    def pred_chain(self, auth_chain):
        return self.chain_mapping.get(self.auth_to_label.get(str(auth_chain)))

    def _motif_indices(self, label_chain, motif, prefer):
        """(polymer indices, numbering mode, name agreement) of a motif in a label chain."""
        residues = self.label_residues[label_chain]
        by_auth = {}
        for k, (sid, ic, name) in enumerate(residues):
            if not ic:
                by_auth.setdefault(sid, (k, name))
        best = None
        for mode in (prefer, "position" if prefer == "author" else "author"):
            idx, agree = [], 0
            for name, num in motif:
                if mode == "author":
                    hit = by_auth.get(str(num))
                else:
                    hit = (num - 1, residues[num - 1][2]) if 0 < num <= len(residues) else None
                if hit is not None:
                    idx.append(hit[0])
                    agree += one_letter(hit[1]) == one_letter(name)
            frac = agree / len(motif) if motif else 1.0
            if best is None or frac > best[2]:
                best = (idx, mode, frac)
        return best

    def map_motif(self, auth_chain, motif, prefer="author"):
        """Experimental motif [(name, number)] -> set of predicted sequence positions (1-based).

        The motif may be numbered by author residue number (AAMotif) or by position in the
        modelled chain (RNAMotif); the numbering whose residue names agree best with the structure
        is used, `prefer` on a tie. Returns (positions, mode) or (None, reason).
        """
        la = self.auth_to_label.get(str(auth_chain))
        if la is None or la not in self.chain_mapping:
            return None, "chain not mapped"
        idx, mode, frac = self._motif_indices(la, motif, prefer)
        if frac < MIN_NAME_AGREEMENT:
            return None, f"residue names disagree ({frac:.0%})"
        to_model = self._model_index(la)
        return {to_model[k] + 1 for k in idx if k in to_model}, mode

    def assign_fallback(self, exp_contacts, pred_contacts):
        """Pair experimental chains OST did not map with free predicted chains.

        exp_contacts: {auth chain: ([(name, number)], prefer)} contact residues of that chain over
        all its interfaces; pred_contacts: {predicted chain: set of contact positions}.
        """
        from scipy.optimize import linear_sum_assignment
        todo = [la for la in dict.fromkeys(self.auth_to_label.get(str(c)) for c in exp_contacts)
                if la is not None and la not in self.chain_mapping]
        free = [c for c in self.pred_seqs if c not in set(self.chain_mapping.values())]
        if not todo or not free:
            return
        label_contacts = {}
        for auth, (motif, prefer) in exp_contacts.items():
            la = self.auth_to_label.get(str(auth))
            if la in todo:
                label_contacts.setdefault(la, []).append((motif, prefer))
        score = np.zeros((len(todo), len(free)))
        for i, la in enumerate(todo):
            seq = self._seq(la)
            exp_idx = set()
            for motif, prefer in label_contacts.get(la, []):
                idx, _, frac = self._motif_indices(la, motif, prefer)
                if frac >= MIN_NAME_AGREEMENT:
                    exp_idx |= set(idx)
            for j, pc in enumerate(free):
                amap = self._exact_alignment(seq, self.pred_seqs[pc])
                ident = len(amap) / len(seq) if seq else 0.0
                if ident < MIN_FALLBACK_IDENTITY:
                    continue
                overlap = jaccard({amap[k] + 1 for k in exp_idx if k in amap}, pred_contacts.get(pc, set()))
                score[i, j] = round(ident, 2) + 1e-3 * (0.0 if overlap != overlap else overlap)
        rows, cols = linear_sum_assignment(-score)
        for i, j in zip(rows, cols):
            if score[i, j] > 0:
                self.chain_mapping[todo[i]] = free[j]
                self.source[todo[i]] = "sequence"


# --------------------------------------------------------------------------- #
# Motif parsing (same formats as PHCorrelationAnalyzer.parse_motifs / parse_rna_motifs)
# --------------------------------------------------------------------------- #
def _list(raw):
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s == "None":
        return []
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            pass
    return [s]


def parse_motif_list(raw):
    """Motif cell -> list (one per chain pair) of [(name, number)]."""
    if raw is None or str(raw).strip() in ("", "None"):
        return []
    s = str(raw).strip()
    items = _list(s) if s.startswith("[") else [s]
    out = []
    for item in items:
        out.append([(n, int(k)) for n, k in re.findall(r"([A-Z0-9]+)\((-?\d+)\)", str(item))])
    return out


def interfaces_of(pairs_raw, aa_raw, rna_raw):
    """[(protein chain, RNA chain, AA motif, RNA motif)] in ChainIDpairList order."""
    pairs, aa, rna = _list(pairs_raw), parse_motif_list(aa_raw), parse_motif_list(rna_raw)
    out = []
    for i, pr in enumerate(pairs):
        if isinstance(pr, (list, tuple)) and len(pr) == 2:
            out.append((str(pr[0]), str(pr[1]),
                        aa[i] if i < len(aa) else [], rna[i] if i < len(rna) else []))
    return out


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def interface_scores(pred_row, exp_row, mapping):
    """Per-interface exact AA/RNA match for one complex.

    pred_row / exp_row: dicts with ChainIDpairList_proteinRNA, AAMotif, RNAMotif.
    Returns (list of dicts keyed by the predicted "protein:RNA" pair, list of skip reasons).
    Predicted-only pairs are kept when both predicted chains have an experimental counterpart.
    """
    pred = {(p, r): (set(k for _, k in aa), set(k for _, k in rna))
            for p, r, aa, rna in interfaces_of(pred_row["ChainIDpairList_proteinRNA"],
                                               pred_row["AAMotif"], pred_row["RNAMotif"])}
    exp_ifaces = interfaces_of(exp_row["ChainIDpairList_proteinRNA"], exp_row["AAMotif"], exp_row["RNAMotif"])

    # chains OST could not map: pair them by sequence (+ contact overlap among identical copies)
    exp_contacts, pred_contacts = {}, {}
    for P, R, aa, rna in exp_ifaces:
        for ch, motif, prefer in ((P, aa, "author"), (R, rna, "position")):
            prev = exp_contacts.get(ch, ([], prefer))
            exp_contacts[ch] = (prev[0] + list(motif), prefer)
    for (p, r), (aa, rna) in pred.items():
        pred_contacts[p] = pred_contacts.get(p, set()) | aa
        pred_contacts[r] = pred_contacts.get(r, set()) | rna
    mapping.assign_fallback(exp_contacts, pred_contacts)

    exp, src, skipped = {}, {}, []
    for P, R, aa, rna in exp_ifaces:
        p, r = mapping.pred_chain(P), mapping.pred_chain(R)
        if p is None or r is None:
            skipped.append(f"{P}:{R} chain not mapped")
            continue
        pa, ma = mapping.map_motif(P, aa, prefer="author")        # AAMotif: author numbering
        pr_, mr = mapping.map_motif(R, rna, prefer="position")    # RNAMotif: position in the chain
        if pa is None or pr_ is None:
            skipped.append(f"{P}:{R} {ma if pa is None else mr}")
            continue
        prev = exp.get((p, r), (set(), set()))
        exp[(p, r)] = (prev[0] | pa, prev[1] | pr_)
        kinds = {mapping.source[mapping.auth_to_label[str(c)]] for c in (P, R)}
        src[(p, r)] = "ost" if kinds == {"ost"} else "sequence"
    mapped = set(mapping.chain_mapping.values())
    rows = []
    for key in sorted(set(pred) | set(exp)):
        if key not in exp and not (key[0] in mapped and key[1] in mapped):
            continue                                             # no experimental counterpart
        ps, ex = pred.get(key, (set(), set())), exp.get(key, (set(), set()))
        rows.append(dict(pair=f"{key[0]}:{key[1]}", native=key in exp, predicted=key in pred,
                         mapping=src.get(key, ""),
                         aa_match=jaccard(ps[0], ex[0]), rna_match=jaccard(ps[1], ex[1]),
                         aa_union=len(ps[0] | ex[0]), rna_union=len(ps[1] | ex[1])))
    return rows, skipped


def complex_scores(rows):
    """Union-size-weighted mean over interfaces -> (AAmatch, RNAmatch)."""
    def wmean(key, wkey):
        num = sum(r[key] * r[wkey] for r in rows if r[wkey] and r[key] == r[key])
        den = sum(r[wkey] for r in rows if r[wkey] and r[key] == r[key])
        return num / den if den else None
    return wmean("aa_match", "aa_union"), wmean("rna_match", "rna_union")
