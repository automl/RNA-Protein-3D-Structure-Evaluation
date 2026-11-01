"""RNA chains bound by several proteins: placement with the main partner against the other partners.

For every RNA chain with at least two native protein partners, the main partner is the protein with the largest
native interface (placement_per_rna_chain.py); the other partners are compared with it.
  summary       RNA chains with all / none / some interfaces correct, all interfaces in one placement class,
                correct rate and native interface size of the main and the other partners, the correct rate of the
                other partners given the main partner, and the protein assembly of the other partners that fail
                next to a correct main partner (assembly incorrect: DockQ_pp < 0.23, or TM_p < 0.5 without partners)
  cross table   placement class of the main partner against that of each other partner
Output: results/plots/all_chains/publication_figures/placement_multi_partner.{tsv,tex}
Run in the RNABench env:  python publication_figures/placement_multi_partner.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interface_failure_modes_figure as F
import placement_per_rna_chain as PRC

SHORT = {"correct interface": "Correct", "right protein site, wrong RNA pose": "Correct site, incorrect RNA",
         "different region": "Distal", "protein binding region mispredicted": "Incorrect binding site",
         "shifted nearby": "Shifted", "same place, RNA turned": "Rotated or slid",
         "same place, duplex slid": "Rotated or slid"}
COLS = ["Correct", "Correct site, incorrect RNA", "Distal", "Incorrect binding site", "Shifted", "Rotated or slid"]


def partners():
    d = PRC.interfaces().sort_values(["native_size", "dockq"], ascending=False)
    d["rank"] = d.groupby(["period", "pdb", "rna_chain"]).cumcount()         # 0 = main partner
    d = d[d.groupby(["period", "pdb", "rna_chain"]).partner.transform("size") > 1].copy()
    d["cls"] = d.paper.map(SHORT)
    main = d[d["rank"] == 0].set_index(["period", "pdb", "rna_chain"])
    other = d[d["rank"] > 0].join(main[["cls", "dockq"]], on=["period", "pdb", "rna_chain"], rsuffix="_main")
    return d, main, other


def summary(d, main, other, period):
    s, m, o = d[d.period == period], main.loc[period], other[other.period == period]
    g = s.groupby(["pdb", "rna_chain"])
    ok = g.dockq.apply(lambda x: (x >= 0.23).all()).sum()
    none = g.dockq.apply(lambda x: (x < 0.23).all()).sum()
    mc, mf = o[o.dockq_main >= 0.23], o[o.dockq_main < 0.23]
    fail_next = mc[mc.dockq < 0.23]
    return pd.Series({
        "RNA chains with >= 2 protein partners": g.ngroups, "interfaces": len(s),
        "all interfaces correct": ok, "no interface correct": none, "some interfaces correct": g.ngroups - ok - none,
        "all interfaces in one class": int((g.cls.nunique() == 1).sum()),
        "main partner correct (%)": round(100 * (m.dockq >= 0.23).mean()),
        "other partners correct (%)": round(100 * (o.dockq >= 0.23).mean()),
        "native interface, main partner (median residues)": m.native_size.median(),
        "native interface, other partners (median residues)": o.native_size.median(),
        "other partner correct | main correct (%)": round(100 * (mc.dockq >= 0.23).mean()),
        "other partner correct | main failed (%)": round(100 * (mf.dockq >= 0.23).mean()),
        "other partners failing next to a correct main partner": len(fail_next),
        "  of these with incorrect protein assembly (%)": round(100 * (fail_next.assembly == "incorrect").mean()),
        "  correct other partners with incorrect assembly (%)":
            round(100 * (mc[mc.dockq >= 0.23].assembly == "incorrect").mean())})


def main_():
    d, main, other = partners()
    tab = pd.DataFrame({p: summary(d, main, other, p) for p in ("post", "pre")})
    o = other[other.period == "post"]
    cross = pd.crosstab(o.cls_main, o.cls).reindex(index=COLS, columns=COLS).fillna(0).astype(int)
    base = os.path.join(F.OUTDIR, "placement_multi_partner")
    with open(base + ".tsv", "w") as fh:
        tab.to_csv(fh, sep="\t")
        fh.write("\npost-cutoff: main partner class (rows) x other partner class (columns)\n")
        cross.to_csv(fh, sep="\t")
    with open(base + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "r" * len(COLS) + "}\n\\hline\n")
        fh.write("Main partner & " + " & ".join(COLS) + " \\\\\n\\hline\n")
        for name, r in cross.iterrows():
            fh.write(f"{name} & " + " & ".join(str(v) for v in r) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
    pd.set_option("display.width", 200)
    print(tab.to_string()); print(cross.to_string())
    print("Saved:", base + ".tsv / .tex")


if __name__ == "__main__":
    main_()
