#! /bin/bash

# ==============================================================================
# Multichain per-chain
# ==============================================================================

echo "=============================================================================================="
echo "Multichain complexes per chain TM regression pdb id cv"
echo "=============================================================================================="

python single_chain_regression_analysis_comparison_with_complex.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi \
  --outdir final_multichain_comparison \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes per chain TM residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_regression_analysis_comparison_with_complex.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi \
  --outdir final_multichain_comparison \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes per chain TM regression rna family cv"
echo "=============================================================================================="

python single_chain_regression_analysis_comparison_with_complex.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by rna_family \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi \
  --outdir final_multichain_comparison \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes per chain Dockq regression pdb id cv"
echo "=============================================================================================="

python single_chain_regression_analysis_comparison_with_complex.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi \
  --outdir final_multichain_comparison \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes per chain Dockq residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_regression_analysis_comparison_with_complex.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi \
  --outdir final_multichain_comparison \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

# ==============================================================================
# Multichain complex level
# ==============================================================================

echo "=============================================================================================="
echo "Multichain complexes complex level TM regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_comparison_with_single_chain.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --task regress \
  --ridge_alpha 10 \
  --min_rows 50 \
  --min_rows_model 40 \
  --ycol Complex_TM \
  --dataset multi_complex \
  --subset multi \
  --target_mode direct \
  --outdir final_multichain_comparison \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level TM residual regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_comparison_with_single_chain.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --task regress \
  --ridge_alpha 10 \
  --min_rows 50 \
  --min_rows_model 40 \
  --ycol Complex_TM \
  --dataset multi_complex \
  --subset multi \
  --target_mode af3_error \
  --outdir final_multichain_comparison \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_comparison_with_single_chain.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --task regress \
  --ridge_alpha 10 \
  --min_rows 50 \
  --min_rows_model 40 \
  --ycol Dockq \
  --dataset multi_complex \
  --subset multi \
  --target_mode direct \
  --outdir final_multichain_comparison \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq residual regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_comparison_with_single_chain.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --task regress \
  --ridge_alpha 10 \
  --min_rows 50 \
  --min_rows_model 40 \
  --ycol Dockq \
  --dataset multi_complex \
  --subset multi \
  --target_mode af3_error \
  --outdir final_multichain_comparison \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq minus TM regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_comparison_with_single_chain.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --task regress \
  --ridge_alpha 10 \
  --min_rows 50 \
  --min_rows_model 40 \
  --ycol Dockq \
  --dataset multi_complex \
  --subset multi \
  --target_mode dockq_minus_tm \
  --outdir final_multichain_comparison \
  --full_model_universe

