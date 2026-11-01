"""DockQ_Nr aggregation-robustness check (reviewer response / SI table).

DockQ_Nr, the protein--RNA-only DockQ used throughout the paper, is aggregated per RNA
chain as the MEAN over that chain's protein partners (column ``RBP_DockQ_mean``). DockQ
itself is defined for a single two-chain interface, so packing several protein--RNA
interfaces into one number is a choice. This script quantifies how sensitive (a) the
continuous metric and (b) the a/b/c interface classification and (c) the paper's central
fold-vs-interface decoupling result are to that choice, by comparing the mean aggregation
to the median / max / min aggregations that are already stored per chain.

Data source (produced by the preprocessing stage of ``reproduce_all.sh``):
    csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv
    -- carries RBP_DockQ_{mean,median,max,min,n_pairs} and assigned_rna_{lddt,tm} per RNA chain.

Outputs (CSV tables, for the SI / Methods robustness statement):
    paper_new_analyses/DockQ_aggregation_robustness/
        dockq_aggregation_robustness.csv          -- rho + a/b/c class agreement vs the mean
        dockq_fold_decoupling_by_aggregation.csv  -- fold-vs-interface rho under each aggregation

Fully reproducible: read-only on the analysis CSV, no network, no DB writes, deterministic.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_new_analyses", "DockQ_aggregation_robustness")
os.makedirs(OUT, exist_ok=True)

# Canonical per-chain analysis table. Prefer the exact path; fall back to a glob so a
# renamed preprocessing dir does not silently break reproduction.
_CANDIDATES = [os.path.join(HERE, "csvs4analysis_per_chain_complex", "combined",
                            "combined__per_chain_all.csv")]
_CANDIDATES += sorted(glob.glob(os.path.join(HERE, "csvs4analysis*", "combined",
                                             "combined__per_chain_all.csv")))
csv = next((p for p in _CANDIDATES if os.path.exists(p)), None)
if csv is None:
    sys.exit("[dockq_robustness] per-chain analysis CSV not found; run preprocessing first.")

df = pd.read_csv(csv, low_memory=False)
AGGS = ["RBP_DockQ_mean", "RBP_DockQ_median", "RBP_DockQ_max", "RBP_DockQ_min"]
FOLD = ["assigned_rna_lddt", "assigned_rna_tm"]
for c in AGGS + FOLD + ["RBP_DockQ_n_pairs"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# keep RNA chains that actually have a protein--RNA interface to aggregate
d = df[df["RBP_DockQ_n_pairs"].fillna(0) > 0].copy()
n, n_multi = len(d), int((d["RBP_DockQ_n_pairs"] > 1).sum())


def abc(x):
    """CAPRI-style DockQ classes (thresholds borrowed from protein docking):
    a = very good (>=0.8), b = medium ([0.23, 0.8)), c = bad (<0.23)."""
    return np.where(x >= 0.8, "a", np.where(x >= 0.23, "b", "c"))


base = "RBP_DockQ_mean"
rows = []
for alt in [a for a in AGGS if a != base]:
    m = d[[base, alt]].dropna()
    rho, _ = spearmanr(m[base], m[alt])
    agree = float((abc(m[base].values) == abc(m[alt].values)).mean() * 100.0)
    rows.append({"comparison": f"{base} vs {alt}", "n": len(m),
                 "spearman_rho": round(float(rho), 3),
                 "abc_class_agreement_pct": round(agree, 1)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "dockq_aggregation_robustness.csv"), index=False)

drows = []
for fold in FOLD:
    for agg in ["RBP_DockQ_mean", "RBP_DockQ_median", "RBP_DockQ_max"]:
        m = d[[fold, agg]].dropna()
        rho, _ = spearmanr(m[fold], m[agg])
        drows.append({"fold_metric": fold, "dockq_aggregation": agg, "n": len(m),
                      "spearman_rho": round(float(rho), 3),
                      "rho_squared": round(float(rho) ** 2, 3)})
pd.DataFrame(drows).to_csv(os.path.join(OUT, "dockq_fold_decoupling_by_aggregation.csv"), index=False)

print(f"[dockq_robustness] per-chain rows with >=1 protein-RNA interface: {n} "
      f"(multi-interface: {n_multi})")
print(f"[dockq_robustness] wrote dockq_aggregation_robustness.csv + "
      f"dockq_fold_decoupling_by_aggregation.csv to {OUT}/")
