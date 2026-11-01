#!/bin/bash
# =============================================================================
# reproduce_all.sh  --  one-command reproduction of the AF3 RNA / RNA-protein
# reliability analysis (regressions, order-independent variance partition, and
# automated feature selection). Run from the repository root.
#
#   bash reproduce_all.sh
#
# Requires the conda env in environment.yml:  conda env create -f environment.yml
# then either `conda activate af3_eval` before running, or set
#   PY="conda run -n af3_eval python"   (the run_*.sh scripts honour $PY-> python)
#
# Outputs:
#   csvs4analysis_*/            per-analysis feature tables (from the dbs)
#   final_regression_runs/      grouped-CV ridge regressions (all subsets/targets)
#   variance_partitioning_final/  Shapley-R^2 group shares (the ranking claims)
#   feature_selection_stability/  automated stability-selection results + nested R^2
#   figures_final/              publication figures (only if plots/+evaluation/ present)
# =============================================================================
set -e
cd "$(dirname "$0")"
# repository root, resolved BEFORE the optional X: remap below (afterwards ".." of the
# mapped drive root is the drive root itself, not the repository)
REPO_ROOT="$(cd .. && pwd)"

# Optional Windows MAX_PATH helper (maps a short drive X: to this folder). No-op
# on Linux/macOS, where deep output paths are fine.
if [ "$OS" = "Windows_NT" ] && [ -f ensure_x_drive.sh ]; then
  source ensure_x_drive.sh
fi

# Default: RESUME -- skip stages already completed (via each script's .run_complete
# marker). Set FORCE_RERUN=1 to recompute everything from scratch.
# (the scripts test the variable for presence, so it is exported only when set to 1)
if [ "${FORCE_RERUN:-0}" = "1" ]; then export FORCE_RERUN=1; else unset FORCE_RERUN; fi

# X-ray-only features (structure-determination method + resolution) are experimental
# "oracle" metadata (not available from a prediction) -> EXCLUDED by default. Set
# INCLUDE_XRAY_FEATURES=1 for the dedicated X-ray-vs-EM importance analysis.
export INCLUDE_XRAY_FEATURES="${INCLUDE_XRAY_FEATURES:-0}"

# Central figure style: point matplotlib at the generated rc at the repo root so
# every figure inherits the serif doc font, colour scheme, and sizes (see figstyle.py).
export MATPLOTLIBRC="${MATPLOTLIBRC:-$REPO_ROOT/matplotlibrc}"
# the SI analysis scripts import the repository packages (plots, figstyle, evaluation)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "########## 1/4  database preprocessing ##########"
bash run_preprocessing.sh

echo "########## 2/4  regressions + Shapley variance partition ##########"
bash run_final_complex_level_regression.sh
bash run_final_full_per_chain_regression.sh
bash run_final_single_chain_regression_without_interface.sh
bash run_final_multichain_per_chain_complex_comparison.sh
bash run_final_rna_rna_per_chain.sh
bash run_final_variance_partitioning.sh

echo "########## 3/4  automated stability-selection feature selection ##########"
bash _run_stability.sh

echo "########## 4/4  regression figures (Fig1/Fig2 + feature stability) ##########"
# best-effort: these read the regression outputs produced above
set +e
bash plot_regression_results_complexes_section1.sh
bash plot_final_single_chain_system_comparison_results.sh
# Fig2 panel E is the principled Shapley Delta-R^2 panel (fig_delta_r2_complex_vs_perchain,
# "multichain complex - per-chain"), NOT the raw coefficient-shift fallback. Build the Shapley
# summary from variance_partitioning_final/ (produced in step 2/4 above), then render the
# publication panels (titled + no-titles). assemble_figures.py (USE_DELTA_FOR_E=True) selects
# publication_figures/notitles/fig_delta_r2_complex_vs_perchain.png for panel E. Must precede assemble.
python plot_variance_partitioning_summary.py
python plot_publication_variance_partitioning.py --outdir publication_figures
python plot_publication_variance_partitioning.py --outdir publication_figures/notitles --no-titles
bash plot_feature_selection.sh
# SI analyses (read the per-chain CSVs from preprocessing): pre/post-cutoff difficulty
# controls figure (R1) + decision-accuracy tables (R6) + pre/post composition &
# confound-control figure (postpre_family_comparison) -> paper_new_analyses/
python analysis_R1_composition.py
python analysis_R6_decision_table.py
python analysis_postpre_composition.py
# Dataset descriptive overview (SI): whole-dataset composition (counts, method,
# lengths, ions/additives, RNA families) + experimental/computational covariates
# (MSA depth, templates, temperature) and a single before/after pre/post panel.
python analysis_dataset_overview.py
# assemble the multi-panel composites Figure1/Figure2 from the panels above. Runs
# here under the short X: drive so the deep panel paths stay under Windows MAX_PATH
# (Path.exists() silently fails on >260-char paths, dropping every panel).
python assemble_figures.py
set -e

echo
echo "Reproduction complete."
echo "  regressions      -> final_regression_runs/"
echo "  variance shares  -> variance_partitioning_final/  (*/shapley_r2.csv)"
echo "  feature selection-> feature_selection_stability/  (*/selection_summary.json)"
echo "  regression figs  -> tex/figures/ (collected by the top-level reproduce_all.sh)"
echo
echo "The collaborator publication figures (R1/R3/R4/R5/SI) are produced by the"
echo "top-level ../reproduce_all.sh via ../run_figures.sh."
