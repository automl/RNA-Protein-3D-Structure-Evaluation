#! /bin/bash
# Multi-chain RNA complexes (two or more RNA chains, no protein), analysed SEPARATELY from
# the protein-RNA and single-chain RNA sets at the per-RNA-chain level.
# Input : csvs4analysis_rna_rna/combined/combined__per_chain_all.csv (run_preprocessing.sh)
# Output: final_regression_runs/rna_rna_per_chain/pca/rna_rna/{tm__..., e_tm__err__...}
# Feature groups follow the per-chain registry; protein-side descriptors, alignment depth,
# thermodynamic descriptors and experimental context are absent for this set and are
# dropped by the unsupervised missingness screen (see preprocess_rna_rna_per_chain_metrics.py).
set -e
PY="${PY:-python}"
CSV=csvs4analysis_rna_rna/combined/combined__per_chain_all.csv
if [ ! -f "$CSV" ]; then
  echo "### SKIP multi-chain RNA regression: $CSV not present (no rna_rna databases)"; exit 0
fi
COMMON="--csv $CSV --ycol assigned_rna_tm --task regress --group_by pdb_id --cv 5 --seed 0 --ridge_alpha 10 \
  --dataset rna_rna_per_chain --subset rna_rna --outdir final_regression_runs \
  --min_rows 50 --min_rows_model 40 --min_non_na_feature 10 --max_cat_levels 25 --min_category_count 10 \
  --max_missing_frac_feature 0.8 --variance_threshold 1e-8 --full_model_universe"

echo "=============================================================================================="
echo "multi-chain RNA per chain TM regression pdb id cv"
echo "=============================================================================================="
$PY single_chain_level_regression_analysis_final.py $COMMON --target_mode direct

echo "=============================================================================================="
echo "multi-chain RNA per chain TM residual regression pdb id cv"
echo "=============================================================================================="
$PY single_chain_level_regression_analysis_final.py $COMMON --target_mode af3_error

echo "########## multi-chain RNA regressions DONE ##########"
