"""Placement per RNA chain: each RNA chain is described by its main protein partner only.

Main partner: the protein chain with the largest native interface (most protein residues in contact with the RNA
chain in the deposited structure, exp_protein_rna AAMotif); ties go to the higher DockQ. No interfaces are merged:
placement class, DockQ, protein side, RNA side and RNA distance / rotation are those of the main-partner interface.

  placement       the paper's placement class of that interface (evaluation/interface_rna_pose.py, as in the DBs)
  protein side    share of the main partner's native interface residues contacted by the AF3 RNA (protein match)
  RNA side        share of the RNA's native interface nucleotides (with that partner) contacted by the AF3 protein
                  (RNA match); reported separately for RNAs of > 15 and <= 15 nt (short RNAs are matched easily)
  no contact      protein and RNA side both < 0.1: whole-RNA distance from its native place in the main partner's
                  frame (<= 10 / 10-20 / > 20 A)
  partner accurate  the main partner's per-chain LDDT_p > 0.85 and TM_p > 0.85 (panel A's protein accuracy)
  RNA fold        good: LDDT_r > 0.85 (and TM_r > 0.5 from 60 nt); intermediate: 0.75-0.85; poor otherwise
Input: results/plots/all_chains/interface_rna_pose/placement_all_chains.tsv, interface_rna_pose.tsv,
       interface_class_composition.tsv, the post_pre a2021 / b2021 databases
Output: results/plots/all_chains/interface_rna_pose/placement_per_rna_chain.tsv (one row per RNA chain) and
        results/plots/all_chains/publication_figures/placement_per_rna_chain_summary.{tsv,tex}
Run in the RNABench env:  python publication_figures/placement_per_rna_chain.py
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interface_failure_modes_figure as F

from evaluation.bundle_paths import PROTEIN_RNA_DB
DBS = dict(PROTEIN_RNA_DB)
SPLIT_NT = 15
OUT_TSV = os.path.join(F.POSE_DIR, "placement_per_rna_chain.tsv")


def native_sizes():
    """(period, pdb, native pair) -> number of native protein interface residues."""
    size = {}
    for per, db in DBS.items():
        for pid, pairs, aa in sqlite3.connect(db).execute(
                "SELECT PDBId, ChainIDpairList_proteinRNA, AAMotif FROM exp_protein_rna"):
            try:
                pairs = json.loads(pairs)
                aa = json.loads(aa) if str(aa).startswith("[") else [aa]
            except (TypeError, ValueError):
                continue
            for (p, r), m in zip(pairs, aa):
                size[(per, str(pid).strip().upper()[:4], f"{p}:{r}")] = len([x for x in str(m).split(",") if x.strip()])
    return size


def interfaces():
    d = F._placement_df()
    ic = pd.read_csv(os.path.join(F.COMPOSITION_DIR, "interface_class_composition.tsv"), sep="\t")[
        ["period", "pdb", "native_pair", "protein_chain_lddt"]]
    nt = pd.read_csv(os.path.join(F.POSE_DIR, "interface_rna_pose.tsv"), sep="\t")[
        ["period", "pdb", "native_pair", "rna_n_nt"]]
    d = d.merge(ic, on=["period", "pdb", "native_pair"], how="left").merge(nt, on=["period", "pdb", "native_pair"],
                                                                          how="left")
    size = native_sizes()
    d["native_size"] = [size.get((r.period, r.pdb, r.native_pair), 0) for r in d.itertuples()]
    d["rna_chain"] = d.native_pair.str.split(":").str[1]
    d["partner"] = d.native_pair.str.split(":").str[0]
    d["partner_accurate"] = (d.protein_chain_lddt > 0.85) & (d.protein_chain_tm > 0.85)
    return d


def per_rna_chain(d):
    d = d.sort_values(["native_size", "dockq"], ascending=False)
    main = d.groupby(["period", "pdb", "rna_chain"], sort=False).head(1).copy()
    agg = d.groupby(["period", "pdb", "rna_chain"]).agg(n_partners=("partner", "size"),
                                                         n_correct=("dockq", lambda x: int((x >= 0.23).sum())))
    main = main.merge(agg, on=["period", "pdb", "rna_chain"])
    main["class"] = main.paper.map(F.LABEL)
    main["no_contact"] = (main.protein_match < 0.1) & (main.rna_match < 0.1)
    main["distance"] = pd.cut(main.rna_translation, [-1, 10, 20, np.inf], labels=["<= 10 A", "10-20 A", "> 20 A"])
    main["length"] = np.where(main.rna_n_nt > SPLIT_NT, f"> {SPLIT_NT} nt", f"<= {SPLIT_NT} nt")
    cols = ["period", "pdb", "rna_chain", "partner", "native_size", "n_partners", "n_correct", "class", "dockq",
            "protein_match", "rna_match", "rna_translation", "rna_rotation", "pocket_rmsd", "partner_accurate",
            "assembly", "protein_state", "rna_lddt", "rna_tm", "rna_fold", "rna_n_nt", "length", "no_contact", "distance"]
    return main[cols].rename(columns={"partner": "main_partner", "protein_match": "protein_side",
                                      "rna_match": "rna_side"})


def summary(c, period="post"):
    q = c[c.period == period]
    order = [lab for _, lab, _ in F.CLASSES]
    g = q.groupby("class")
    t = pd.DataFrame({"n": g.size()})
    for f in ("good", "intermediate", "wrong"):
        t[f"RNA {f} %"] = g.rna_fold.apply(lambda x, f=f: 100 * (x == f).mean())
    t["partner accurate %"] = 100 * g.partner_accurate.mean()
    t["protein side"] = g.protein_side.mean()
    for L in (f"> {SPLIT_NT} nt", f"<= {SPLIT_NT} nt"):
        t[f"RNA side ({L})"] = q[q.length == L].groupby("class").rna_side.mean()
        t[f"n ({L})"] = q[q.length == L].groupby("class").size()
    t["no contact"] = g.no_contact.sum()
    for b in ("<= 10 A", "10-20 A", "> 20 A"):
        t[f"no contact {b}"] = q[q.no_contact].groupby("class").distance.apply(lambda x, b=b: int((x == b).sum()))
    return t.reindex(order).fillna(0).round(2)


def main():
    c = per_rna_chain(interfaces())
    c.to_csv(OUT_TSV, sep="\t", index=False)
    base = os.path.join(F.OUTDIR, "placement_per_rna_chain_summary")
    tabs = {per: summary(c, per) for per in ("post", "pre")}
    pd.concat(tabs).to_csv(base + ".tsv", sep="\t")
    with open(base + ".tex", "w") as fh:
        t = tabs["post"]
        fh.write("\\begin{tabular}{lrrrrrrrr}\n\\hline\n")
        fh.write("Placement (main partner) & $n$ & RNA good & RNA poor & Partner accurate & Protein side & "
                 f"RNA side ($>{SPLIT_NT}$~nt) & RNA side ($\\leq {SPLIT_NT}$~nt) & No contact \\\\\n"
                 "& & (\\%) & (\\%) & (\\%) & & & & \\\\\n\\hline\n")
        for name, r in t.iterrows():
            fh.write(f"{name} & {int(r.n)} & {r['RNA good %']:.0f} & {r['RNA wrong %']:.0f} & "
                     f"{r['partner accurate %']:.0f} & {r['protein side']:.2f} & "
                     f"{r[f'RNA side (> {SPLIT_NT} nt)']:.2f} & {r[f'RNA side (<= {SPLIT_NT} nt)']:.2f} & "
                     f"{int(r['no contact'])} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    pd.set_option("display.width", 250)
    for per, t in tabs.items():
        print(f"== {per}: {(c.period == per).sum()} RNA chains")
        print(t.to_string())
    print("Saved:", OUT_TSV, "|", base + ".tsv / .tex")


if __name__ == "__main__":
    main()
