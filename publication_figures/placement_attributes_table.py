"""SI table: attributes of the incorrect protein-RNA interfaces in each placement class (post-cutoff).

Per placement class (evaluation/interface_rna_pose.py, as in the databases) the share of incorrect interfaces with
  site contact        AF3 contacts >= 10% of the native protein interface residues (protein contact match)
  pocket intact       Calpha RMSD of the native binding pocket <= 4 A after superposing the protein chain
  whole RNA <= 10 A / > 20 A, turned > 45 deg   whole-RNA translation / rotation in the protein frame
  RNA well folded     LDDT_r > 0.85 (and TM_r > 0.5 from 60 nt)
  assembly correct    protein-protein DockQ >= 0.23 to every contacted protein partner (single chain: TM_p >= 0.5)
  protein side / RNA side   protein / RNA contact match >= 0.5 (correct), 0.1-0.5 (partial), < 0.1 (none)
Input: results/plots/all_chains/interface_rna_pose/placement_all_chains.tsv (evaluation/placement_all_chains.py)
Output: results/plots/all_chains/publication_figures/placement_attributes_table.{tsv,tex}
Run in the RNABench env:  python publication_figures/placement_attributes_table.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interface_failure_modes_figure as F

COLS = [("site", "Site contact"), ("pocket", "Pocket intact"), ("near", "Whole RNA $\\leq 10$~\\AA"),
        ("far", "Whole RNA $> 20$~\\AA"), ("turned", "Turned $> 45^\\circ$"), ("fold", "RNA well folded"),
        ("assembly_ok", "Assembly correct"), ("prot_ok", "Protein side correct"), ("rna_ok_side", "RNA side correct"),
        ("rna_none", "RNA side none")]


def build(period="post"):
    d = F._placement_df()
    f = d[(d.period == period) & (d.dockq < 0.23)].copy()
    f["site"] = f.protein_match >= 0.1
    f["pocket"] = ~(f.pocket_rmsd > 4)
    f["near"], f["far"] = f.rna_translation <= 10, f.rna_translation > 20
    f["turned"] = f.rna_rotation > 45
    f["fold"] = f.rna_fold == "good"
    f["assembly_ok"] = f.assembly == "correct"
    f["prot_ok"], f["rna_ok_side"] = f.protein_match >= 0.5, f.rna_match >= 0.5
    f["rna_none"] = f.rna_match < 0.1
    keys = [k for k, lab, _ in F.CLASSES if k != "correct interface"]
    t = f.groupby("paper")[[c for c, _ in COLS]].mean().mul(100).round(0).reindex(keys)
    t.insert(0, "n", f.paper.value_counts().reindex(keys))
    t.loc["all"] = [len(f)] + list(f[[c for c, _ in COLS]].mean().mul(100).round(0))
    t.index = [F.LABEL.get(k, "All incorrect interfaces") for k in t.index]
    base = os.path.join(F.OUTDIR, "placement_attributes_table")
    t.to_csv(base + ".tsv", sep="\t")
    with open(base + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "r" * (len(COLS) + 1) + "}\n\\hline\n")
        fh.write("Placement class & $n$ & " + " & ".join(lab for _, lab in COLS) + " \\\\\n\\hline\n")
        for name, r in t.iterrows():
            fh.write(f"{name} & {int(r.n)} & " + " & ".join(f"{int(r[c])}" for c, _ in COLS) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    print(t.to_string())
    print("Saved:", base + ".tsv / .tex")


SIDES = ["correct", "partial", "none"]


def build_sides(period="post", split_nt=15):
    """Protein side x RNA side of the incorrect interfaces, for RNAs of > split_nt and <= split_nt nucleotides
    (on short RNAs almost any contact covers every nucleotide, so the RNA side is easily matched)."""
    d = F._placement_df()
    nt = pd.read_csv(os.path.join(F.POSE_DIR, "interface_rna_pose.tsv"), sep="\t")[
        ["period", "pdb", "native_pair", "rna_n_nt"]]
    d = d.merge(nt, on=["period", "pdb", "native_pair"], how="left")
    f = d[(d.period == period) & (d.dockq < 0.23)].copy()
    side = lambda x: pd.Series(np.where(x >= 0.5, "correct", np.where(x >= 0.1, "partial", "none")), index=x.index)
    f["prot"], f["rna"] = side(f.protein_match), side(f.rna_match)
    groups = [(f"> {split_nt} nt", f[f.rna_n_nt > split_nt]), (f"<= {split_nt} nt", f[f.rna_n_nt <= split_nt])]
    tabs = {g: pd.crosstab(s.prot, s.rna).reindex(index=SIDES, columns=SIDES).fillna(0).astype(int) for g, s in groups}
    base = os.path.join(F.OUTDIR, "placement_sides_table")
    pd.concat(tabs, axis=1).to_csv(base + ".tsv", sep="\t")
    with open(base + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "rrr" * len(groups) + "}\n\\hline\n")
        tex = {">": "$>$", "<=": "$\\leq$"}
        heads = []
        for g, s in groups:
            op, rest = g.split(" ", 1)
            heads.append("\\multicolumn{3}{c}{RNA " + tex[op] + " " + rest + f" ($n={len(s)}$)" + "}")
        fh.write(" & " + " & ".join(heads) + " \\\\\n")
        fh.write("Protein side & " + " & ".join(["RNA correct & RNA partial & RNA none"] * len(groups)) + " \\\\\n\\hline\n")
        for p in SIDES:
            fh.write(f"{p} & " + " & ".join(" & ".join(str(tabs[g].loc[p, r]) for r in SIDES) for g, _ in groups)
                     + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    for g, s in groups:
        print(f"{g} (n={len(s)})"); print(tabs[g].to_string())
    print("Saved:", base + ".tsv / .tex")


if __name__ == "__main__":
    build()
    build_sides()
