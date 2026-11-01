#!/bin/bash
# Reproduce the paper from the published databases: regression analyses, all figure scripts, and the collection of
# the paper's figures into figures_final/ (under the file names of the manuscript) with a check that none is missing.
#
#   conda activate af3_eval      # conda env create -f environment.yml
#   bash fetch_data.sh           # databases and analysis inputs from Zenodo
#   bash reproduce_all.sh
set -e
cd "$(dirname "$0")"

# UTF-8 console output (Greek letters in the scripts' prints) also on Windows
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# X-ray-only features stay out of the regression (set 1 for the dedicated X-ray analysis)
export INCLUDE_XRAY_FEATURES="${INCLUDE_XRAY_FEATURES:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
python -c "import figstyle; figstyle.write_matplotlibrc('matplotlibrc')" 2>/dev/null || true
export MATPLOTLIBRC="${MATPLOTLIBRC:-$PWD/matplotlibrc}"

if [ ! -f regression_analysis/data/post_pre/a2021_b2021_protein_rna/a2021.db ]; then
  echo "The databases are not in place yet. Download them first:  bash fetch_data.sh"
  exit 1
fi

# start from clean figure folders; tables computed by evaluation/make_analysis_tables.sh and the structure renders
# (site_chosen/, ga_cartoons/; shipped in analysis_inputs.zip, made from the structures) are kept
rm -rf figures_final raw_plots regression_analysis/publication_figures/combined regression_analysis/publication_figures/notitles
[ -d results/plots ] && find results/plots -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' -o -name '*.eps' \) \
  -not -path '*/site_chosen/*' -not -path '*/ga_cartoons/*' -delete
# intermediate tables not computed here come from analysis_inputs.zip (results/plots_odl)
mkdir -p results/plots
[ -d results/plots_odl/all_chains ] && cp -rn results/plots_odl/all_chains results/plots/

echo "############################## 1/3  regression analysis ##############################"
( cd regression_analysis && bash reproduce_all.sh )

echo "############################## 2/3  figures ##############################"
bash run_figures.sh

echo "############################## 3/3  collect ##############################"
mkdir -p figures_final/tables
cp regression_analysis/paper_new_analyses/R1_composition/*.csv \
   regression_analysis/paper_new_analyses/R6_decision_table/*.csv \
   regression_analysis/paper_new_analyses/dataset_overview/*.csv figures_final/tables/ 2>/dev/null || true
if python utils/collect_paper_figures.py; then
  echo "Reproduction complete: the figures of the paper are in figures_final/."
else
  echo "Reproduction finished, but some figures are missing (listed above); the log above shows the failing script."
  exit 1
fi
