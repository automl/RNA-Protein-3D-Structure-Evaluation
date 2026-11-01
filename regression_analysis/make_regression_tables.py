#!/usr/bin/env python3
"""
make_regression_tables.py

Generate publication-ready LaTeX tables for the multivariate regression analysis
DIRECTLY from the raw result CSVs, so values are copied verbatim and never
re-typed by hand. Outputs into tex/tables/.

Tables produced:
  table1_full_vs_pruned.tex   <- feature_selection_final/*/.../metrics(.csv|_summary.csv)
                                 (fold mean +/- s.d. of R2, Spearman, RMSE;
                                  full vs pruned, per dataset, GX_all + G0..G5)
  table_shapley_partition.tex <- variance_partitioning_final/_summary/summary_overview.csv
                                 (Shapley-R2 share of explained variance per group)

Run: conda run -n af3_eval python make_regression_tables.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = ROOT / "tex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Table 1: full vs pruned
# ---------------------------------------------------------------------------
# (label, full_metrics_csv, pruned_metrics_csv)
FS = ROOT / "feature_selection_final"
T1_DATASETS = [
    ("RNA-only (SC)",
     FS / "rnaOnly_per_chain/pca/rnaOnly_full/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv",
     FS / "rnaOnly_per_chain/pca/rnaOnly_pruned/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv"),
    ("1v1 (SC)",
     FS / "rp_1v1_per_chain/pca/rp_1v1_full/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv",
     FS / "rp_1v1_per_chain/pca/rp_1v1_pruned/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv"),
    ("Multichain (SC)",
     FS / "multi_per_chain/pca/multi_full/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv",
     FS / "multi_per_chain/pca/multi_pruned/tm__a10p0__gpdb__cv5/f/csvs/metrics.csv"),
    ("1v1 complex",
     FS / "rp_1v1_complex/cx/rp_1v1_full/Complex_TM__a10p0__grppdb_id__cv5/f/c/metrics_summary.csv",
     FS / "rp_1v1_complex/cx/rp_1v1_pruned/Complex_TM__a10p0__grppdb_id__cv5/f/c/metrics_summary.csv"),
    ("Multichain complex",
     FS / "multi_complex/cx/multi_full/Complex_TM__a10p0__grppdb_id__cv5/f/c/metrics_summary.csv",
     FS / "multi_complex/cx/multi_pruned/Complex_TM__a10p0__grppdb_id__cv5/f/c/metrics_summary.csv"),
]
MODEL_ORDER = ["GX_all", "G0_af3", "G1_evolution", "G2_complexity",
               "G3_rna_intrinsic", "G4_interface", "G5_experimental"]
MODEL_LABEL = {
    "GX_all": r"$G_X$ all", "G0_af3": "G0 af3", "G1_evolution": "G1 evolution",
    "G2_complexity": "G2 complexity", "G3_rna_intrinsic": "G3 rna intrinsic",
    "G4_interface": "G4 interface", "G5_experimental": "G5 experimental",
}


def fmt_pm(s):
    """Copy the raw 'mean±std' string verbatim, only swapping the unicode sign
    for a LaTeX \\pm. Returns '--' for missing/blank/NaN cells."""
    if s is None:
        return "--"
    if isinstance(s, float) and np.isnan(s):
        return "--"
    s = str(s).strip()
    if s == "" or s.lower() == "nan" or "±nan" in s or s.startswith("nan"):
        return "--"
    return "$" + s.replace("±", r"\pm") + "$"


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index("model")


def t1_cell(df: pd.DataFrame, model: str, col: str) -> str:
    if model not in df.index:
        return "--"
    row = df.loc[model]
    # if a model was skipped, status != 'ok' and fold_* is blank
    if "status" in df.columns and str(row.get("status", "")).lower() != "ok":
        return "--"
    return fmt_pm(row.get(col, None))


def build_table1() -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance of grouped regression models across datasets, "
                 r"comparing the initial full feature set with the dataset-specific pruned "
                 r"feature set. Values are mean~$\pm$~s.d.\ across grouped (\texttt{pdb\_id}) "
                 r"5-fold cross-validation. Selection and evaluation use the same folds, so "
                 r"the comparison reflects in-sample stability/parsimony of the pruned set "
                 r"rather than an out-of-sample improvement. SC, single-chain (per-chain) "
                 r"analysis; $G_X$ all, full combined model; G0--G5, individual feature "
                 r"groups. ``--'' indicates no usable features remained for that group after "
                 r"filtering/pruning.}")
    lines.append(r"\label{tab:fullvspruned}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\hline")
    lines.append(r" & & \multicolumn{3}{c}{Full feature set} & "
                 r"\multicolumn{3}{c}{Pruned feature set}\\")
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    lines.append(r"Dataset & Model & $R^2$ & Spearman & RMSE & $R^2$ & Spearman & RMSE\\")
    lines.append(r"\hline")
    for label, full_p, pruned_p in T1_DATASETS:
        full = load_metrics(full_p)
        pruned = load_metrics(pruned_p)
        for i, m in enumerate(MODEL_ORDER):
            ds = label if i == 0 else ""
            cells = [
                t1_cell(full, m, "fold_r2"), t1_cell(full, m, "fold_spearman"), t1_cell(full, m, "fold_rmse"),
                t1_cell(pruned, m, "fold_r2"), t1_cell(pruned, m, "fold_spearman"), t1_cell(pruned, m, "fold_rmse"),
            ]
            lines.append(f"{ds} & {MODEL_LABEL[m]} & " + " & ".join(cells) + r"\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shapley partition table
# ---------------------------------------------------------------------------
GROUPS = ["G0_af3", "G1_evolution", "G2_complexity", "G3_rna_intrinsic", "G4_interface", "G5_experimental"]
GROUP_HDR = {"G0_af3": "G0", "G1_evolution": "G1", "G2_complexity": "G2",
             "G3_rna_intrinsic": "G3", "G4_interface": "G4", "G5_experimental": "G5"}
DS_LABEL = {
    "all_complex": "All complexes", "multi_complex": "Multichain complexes",
    "rp_1v1_complex": "RNA--protein 1v1", "multi_per_chain": "Multichain (per-chain)",
    "rnaOnly_per_chain": "RNA-only (per-chain)", "rna_rna_per_chain": "Multi-chain RNA (per-chain)", "rp_1v1_per_chain": "RNA--protein 1v1 (per-chain)",
}
TARGET_LABEL = {"Complex_TM": "cTM", "Dockq": "DockQ", "DockQ_Nr": "DockQ$_{Nr}$", "assigned_rna_tm": "rTM"}

# rows to show (direct mode), in a sensible order; exclude the unstable 1v1 per-chain
T_SHAP_ORDER = [
    ("all_complex", "Complex_TM"), ("multi_complex", "Complex_TM"), ("rp_1v1_complex", "Complex_TM"),
    ("all_complex", "Dockq"), ("multi_complex", "Dockq"),
    ("all_complex", "DockQ_Nr"), ("multi_complex", "DockQ_Nr"), ("rp_1v1_complex", "DockQ_Nr"),
    ("multi_per_chain", "assigned_rna_tm"), ("rnaOnly_per_chain", "assigned_rna_tm"), ("rna_rna_per_chain", "assigned_rna_tm"),
]


def build_shapley_table(summary_csv: Path) -> str:
    s = pd.read_csv(summary_csv)
    s = s[s["mode"] == "direct"].copy()
    key = {(r.dataset, r.target): r for r in s.itertuples()}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Shapley-$R^2$ partition of grouped cross-validated explained "
                 r"variance over the six feature groups. $R^2_{\mathrm{AF3}}$, "
                 r"confidence-only model; $R^2_{\mathrm{full}}$, full combined model (equal to "
                 r"the sum of group contributions). Group columns give each group's share of "
                 r"$R^2_{\mathrm{full}}$ (\%), averaged over all group orderings. $n$ is the "
                 r"number of analysed rows; per-chain rows are non-independent within a complex. "
                 r"The RNA--protein 1v1 per-chain subset is omitted: its full model did not "
                 r"exceed the confidence-only baseline out-of-fold ($R^2_{\mathrm{full}}<"
                 r"R^2_{\mathrm{AF3}}$), so a group ranking is not meaningful there.}")
    lines.append(r"\label{tab:shapley}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Dataset & Target & $n$ & $R^2_{\mathrm{AF3}}$ & $R^2_{\mathrm{full}}$ & "
                 + " & ".join(GROUP_HDR[g] for g in GROUPS) + r"\\")
    lines.append(r"\hline")
    for ds, tg in T_SHAP_ORDER:
        if (ds, tg) not in key:
            continue
        r = key[(ds, tg)]
        shares = []
        for g in GROUPS:
            val = getattr(r, f"share_{g}")
            shares.append("--" if pd.isna(val) else f"{round(val):d}")
        af3 = getattr(r, "af3_only_r2")
        full = getattr(r, "full_r2")
        af3_s = "--" if pd.isna(af3) else f"{af3:.2f}"
        lines.append(
            f"{DS_LABEL.get(ds, ds)} & {TARGET_LABEL.get(tg, tg)} & {int(r.n)} & "
            f"{af3_s} & {full:.2f} & " + " & ".join(shares) + r"\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[2pt]")
    lines.append(r"\footnotesize G0 AF3 confidence; G1 evolution; G2 complexity; "
                 r"G3 RNA-intrinsic; G4 interface (predicted); G5 experimental.")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    if FS.is_dir():
        (OUT / "table1_full_vs_pruned.tex").write_text(build_table1(), encoding="utf-8")
        print("wrote", OUT / "table1_full_vs_pruned.tex")
    else:
        print("skip table1_full_vs_pruned.tex (no feature_selection_final/: run run_full_vs_pruned.sh "
              "for the hand-curated full-vs-pruned comparison; the paper table uses feature_selection_stability/)")

    summ = ROOT / "variance_partitioning_final" / "_summary" / "summary_overview.csv"
    (OUT / "table_shapley_partition.tex").write_text(build_shapley_table(summ), encoding="utf-8")
    print("wrote", OUT / "table_shapley_partition.tex")


if __name__ == "__main__":
    main()
