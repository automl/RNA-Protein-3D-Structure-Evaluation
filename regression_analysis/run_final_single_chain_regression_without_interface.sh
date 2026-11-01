#! /bin/bash

#! /bin/bash

# Fully harmonized per-chain runs across the paper
# Common setup:
#   --cv 5
#   --seed 0
#   --ridge_alpha 10
#   --min_rows 50
#   --min_rows_model 40
#   --min_non_na_feature 10
#   --max_cat_levels 25
#   --min_category_count 10
#   --max_missing_frac_feature 0.8
#   --variance_threshold 1e-8

# ==============================================================================
# RNA-only per-chain
# ==============================================================================

echo "=============================================================================================="
echo "rnaonly single chain TM regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rnaOnly_per_chain \
  --subset rnaOnly \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "rnaonly single chain TM residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rnaOnly_per_chain \
  --subset rnaOnly \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "rnaonly single chain TM regression per family cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode af3_error \
  --group_by rna_family \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rnaOnly_per_chain \
  --subset rnaOnly \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe


# ==============================================================================
# RP 1v1 per-chain
# ==============================================================================

echo "=============================================================================================="
echo "RP 1v1 per chain TM regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1 \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 per chain TM residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1 \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 per chain TM regression rna family cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by rna_family \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1 \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 per chain Dockq regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1 \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 per chain Dockq residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1 \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe


# ==============================================================================
# Multichain per-chain
# ==============================================================================

echo "=============================================================================================="
echo "Multichain complexes per chain TM regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
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
  --outdir runs_no_interface \
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

python single_chain_level_regression_analysis_no_interface.py \
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
  --outdir runs_no_interface \
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

python single_chain_level_regression_analysis_no_interface.py \
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
  --outdir runs_no_interface \
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

python single_chain_level_regression_analysis_no_interface.py \
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
  --outdir runs_no_interface \
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

python single_chain_level_regression_analysis_no_interface.py \
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
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe


# ==============================================================================
# All complex chains per-chain
# ==============================================================================

echo "=============================================================================================="
echo "All Complexes per chain TM regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset all_complex_chains \
  --subset all_complex_per_chain \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes per chain TM residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset all_complex_chains \
  --subset all_complex_per_chain \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes per chain TM regression rna family cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by rna_family \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset all_complex_chains \
  --subset all_complex_per_chain \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes per chain Dockq regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset all_complex_chains \
  --subset all_complex_per_chain \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes per chain Dockq residual regression pdb id cv"
echo "=============================================================================================="

python single_chain_level_regression_analysis_no_interface.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_dockq \
  --task regress \
  --target_mode af3_error \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset all_complex_chains \
  --subset all_complex_per_chain \
  --outdir runs_no_interface \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8 \
  --full_model_universe