"""
Per-interface protein and RNA sequence match of AF3 protein-RNA interfaces (exact residue mapping).

Interface residues are the paper's own contacts (H-bond + vdW, PyMOL; AAMotif / RNAMotif, the same
definition as the motif_vs_interface figure). For each protein-RNA interface (one protein chain,
one RNA chain) the experimental contact residues are carried to the predicted structure through
OpenStructure's chain mapping and reference/model alignment (evaluation/exact_residue_mapping.py,
the same mapping that produced RBP_DockQ), and compared position by position -- no motif sliding,
no index tolerance:

  protein sequence match = |P_pred & P_exp| / |P_pred | P_exp|   (protein contact residues)
  RNA sequence match     = |R_pred & R_exp| / |R_pred | R_exp|   (RNA contact nucleotides)

Each interface carries its own interface DockQ (RBP_DockQ of that chain pair, either key order).
Native interfaces that AF3 did not form score 0; interfaces only AF3 formed are kept in the TSV
(native = False) but have no DockQ. Chains OpenStructure could not map (it fails on or drops
U-repeat RNAs, and leaves out very short chains) are paired by sequence and mapped by exact
alignment to the AF3 sequence (column mapping = "sequence"; see exact_residue_mapping.py).

Run in the RNABench env:  python evaluation/interface_motif_match.py
"""

import ast
import json
import os
import sqlite3
import sys
from collections import Counter

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.exact_residue_mapping import (ComplexMapping, interface_scores, interfaces_of,
                                              locate_inputs, one_letter)

from evaluation.bundle_paths import PROTEIN_RNA_DB, PROTEIN_RNA_DIR
PERIODS = dict(PROTEIN_RNA_DB)
SEARCH_ROOTS = [PROTEIN_RNA_DIR["post"], PROTEIN_RNA_DIR["pre"]]
OUTDIR = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", "interface_motif_match")
SPLIT = 0.5


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


def dockq_of(raw, pair):
    p, r = pair.split(":")
    for item in _list(raw):
        if isinstance(item, dict):
            for k in (f"{p}:{r}", f"{r}:{p}"):
                if k in item:
                    return float(item[k])
    return np.nan


def _pred_sequences(row):
    seqs = {}
    for ch_col, seq_col in (("p_pch", "p_pseq"), ("p_rch", "p_rseq")):
        chs, ss = [str(c) for c in _list(row[ch_col])], [str(s) for s in _list(row[seq_col])]
        for i, c in enumerate(chs):
            seqs[c] = ss[i] if len(ss) > 1 else (ss[0] if ss else "")
    return seqs


def _name_qc(row, mapping, seqs, qc):
    """Share of mapped experimental contact residues whose name equals the predicted residue."""
    for P, R, aa, rna in interfaces_of(row["e_pairs"], row["e_aa"], row["e_rna"]):
        for ch, motif, prefer in ((P, aa, "author"), (R, rna, "position")):
            pc = mapping.pred_chain(ch)
            if pc is None or pc not in seqs:
                continue
            la = mapping.auth_to_label[ch]
            for name, num in motif:
                pos, _ = mapping.map_motif(ch, [(name, num)], prefer=prefer)
                if pos:
                    k = next(iter(pos))
                    qc["n"] += 1
                    qc["same"] += 0 < k <= len(seqs[pc]) and seqs[pc][k - 1] == one_letter(name)


def run():
    os.makedirs(OUTDIR, exist_ok=True)
    rows_out, stats = [], {}
    for period, db in PERIODS.items():
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT p.exp_db_id AS id, p.FileName AS fn, p.RBP_DockQ AS dockq,
                   p.AAMotif AS p_aa, p.RNAMotif AS p_rna, p.ChainIDpairList_proteinRNA AS p_pairs,
                   p.ProteinChainIDs AS p_pch, p.ProteinSequence AS p_pseq,
                   p.RNAChainIDs AS p_rch, p.RNASequence AS p_rseq,
                   e.AAMotif AS e_aa, e.RNAMotif AS e_rna, e.ChainIDpairList_proteinRNA AS e_pairs
            FROM pred_protein_rna p JOIN exp_protein_rna e
              ON UPPER(TRIM(p.exp_db_id)) = UPPER(TRIM(e.PDBId))""").fetchall()
        con.close()
        st = Counter(complexes=len(rows))
        qc = Counter()
        for row in rows:
            cif, js = locate_inputs(row["id"], row["fn"], SEARCH_ROOTS)
            if not (cif and js):
                st["no structure / OST file"] += 1
                continue
            seqs = _pred_sequences(row)
            mapping = ComplexMapping(cif, js, pred_seqs=seqs)
            st["OST failed (U-repeat bug)"] += not mapping.chain_mapping
            res, skipped = interface_scores(
                dict(ChainIDpairList_proteinRNA=row["p_pairs"], AAMotif=row["p_aa"], RNAMotif=row["p_rna"]),
                dict(ChainIDpairList_proteinRNA=row["e_pairs"], AAMotif=row["e_aa"], RNAMotif=row["e_rna"]),
                mapping)
            st["native interfaces unmapped"] += len(skipped)
            st["native interfaces via sequence fallback"] += sum(r["mapping"] == "sequence" for r in res)
            _name_qc(row, mapping, seqs, qc)
            for r in res:
                rna_ch = r["pair"].split(":")[1]
                rows_out.append(dict(period=period, id=str(row["id"]).strip().upper(), pair=r["pair"],
                                     mapping=r["mapping"], native=r["native"], predicted=r["predicted"],
                                     aa_match=r["aa_match"], rna_match=r["rna_match"],
                                     n_aa=r["aa_union"], n_rna=r["rna_union"],
                                     dockq=dockq_of(row["dockq"], r["pair"]) if r["native"] else np.nan,
                                     rna_len=len(seqs.get(rna_ch, ""))))
        st["name agreement"] = f"{qc['same']}/{qc['n']}"
        stats[period] = st

    keys = ["period", "id", "pair", "mapping", "native", "predicted", "aa_match", "rna_match", "n_aa", "n_rna",
            "dockq", "rna_len"]
    with open(os.path.join(OUTDIR, "interface_motif_match.tsv"), "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows_out:
            fh.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    from scipy.stats import spearmanr
    for period in ("pre", "post"):
        sub = [r for r in rows_out if r["period"] == period and r["native"]
               and r["aa_match"] == r["aa_match"] and r["rna_match"] == r["rna_match"]]
        aa = np.array([r["aa_match"] for r in sub]); rn = np.array([r["rna_match"] for r in sub])
        dq = np.array([r["dockq"] for r in sub])
        print(f"\n=== {period}: {dict(stats[period])}")
        print(f"  {len(sub)} native protein-RNA interfaces, {np.isfinite(dq).sum()} with interface DockQ, "
              f"{sum(not r['predicted'] for r in sub)} not formed by AF3; "
              f"{sum(1 for r in rows_out if r['period'] == period and not r['native'])} AF3-only interfaces")
        print(f"  median protein match {np.median(aa):.2f}, RNA match {np.median(rn):.2f}")
        hp, hr = aa >= SPLIT, rn >= SPLIT
        for name, m in (("both right (P>=.5, R>=.5)", hp & hr), ("protein right, RNA wrong", hp & ~hr),
                        ("RNA right, protein wrong", ~hp & hr), ("both wrong", ~hp & ~hr)):
            dm = np.nanmedian(dq[m]) if np.isfinite(dq[m]).any() else np.nan
            print(f"  {name:28s} {m.sum():4d} ({100 * m.mean():4.1f}%)  median DockQ {dm:.2f}")
        ok = np.isfinite(dq)
        print(f"  Spearman with interface DockQ: protein match rho={spearmanr(aa[ok], dq[ok])[0]:.2f}, "
              f"RNA match rho={spearmanr(rn[ok], dq[ok])[0]:.2f}")
        bad = ok & (dq < 0.23)
        print(f"  incorrect interfaces (DockQ<0.23, n={bad.sum()}): protein match >=0.5 {100 * np.mean(aa[bad] >= .5):.0f}%; "
              f"RNA match >=0.5 {100 * np.mean(rn[bad] >= .5):.0f}%")
    print(f"\nWritten to {OUTDIR}/interface_motif_match.tsv")


if __name__ == "__main__":
    run()
