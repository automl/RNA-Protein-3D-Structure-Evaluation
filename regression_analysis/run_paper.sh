#! /bin/bash
#
# run_paper.sh -- single-command reproduction of the full multivariate-regression
# analysis, all figures (panels + assembled Figure 1 / Figure 2), and all
# publication tables.
#
# Usage:
#   conda activate af3_eval        # the env with python/sklearn/pandas/seaborn
#   ./run_paper.sh                 # full pipeline (analysis -> figures -> tables)
#   ./run_paper.sh --figs-only     # skip the heavy analysis; rebuild figs+tables
#
# All Python is invoked via "$PY" (default: python); override to use any
# interpreter, e.g.:   PY="conda run -n af3_eval python" ./run_paper.sh
#
set -e
PY="${PY:-python}"
export PY                          # so sub-scripts (run_final_*.sh) inherit it

FIGS_ONLY=0
[ "$1" = "--figs-only" ] && FIGS_ONLY=1

echo "=============================================================="
echo " AF3 regression analysis -- full build"
echo " Python: $PY"
echo "=============================================================="

if [ "$FIGS_ONLY" -eq 0 ]; then
  echo "[1/6] Preprocessing, feature selection, regressions, base plots,"
  echo "      and variance partitioning (run_all.sh) ..."
  bash run_all.sh
else
  echo "[1/6] --figs-only: skipping analysis (run_all.sh)."
fi

echo "[2/6] Aggregating Shapley variance partition ..."
$PY plot_variance_partitioning_summary.py

echo "[3/6] Publication figures (Shapley panels; titled + no-titles) ..."
$PY plot_publication_variance_partitioning.py --outdir publication_figures
$PY plot_publication_variance_partitioning.py --outdir publication_figures/notitles --no-titles

echo "[4/6] Publication tables (generated directly from raw result CSVs) ..."
$PY make_regression_tables.py

echo "[5/6] Assembling composite Figure 1 and Figure 2 ..."
$PY assemble_figures.py

echo "[6/7] Collecting all figures into tex/figures/ (png600/pdf/svg/eps) ..."
$PY collect_figures.py

echo "[7/7] Decision framework: branch statistics (Table 1) and Figure 3 ..."
$PY decision_branch_stats.py
$PY make_decision_tree_figure.py

echo "=============================================================="
echo " Done. All figures in png(600dpi)/pdf/svg/eps."
echo "   Submission figures : tex/figures/{Figure1,Figure2,Fig3_decision_tree}.* + Shapley figs"
echo "   Individual panels  : tex/figures/panels/"
echo "   Tables             : tex/tables/*.tex (incl. table_decision_accuracy.tex = Table 1)"
echo "   Branch statistics  : decision_branch_stats.csv"
echo "   Manuscript text    : tex/{methods,results,discussion,supplementary,figure_captions}.tex"
echo "=============================================================="
