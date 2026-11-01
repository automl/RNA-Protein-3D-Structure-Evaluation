#! /bin/bash

# Fully harmonized feature-selection runs across the whole paper
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
#   no robust scaler anywhere

# ==============================================================================
# Per-chain analysis
# ==============================================================================

# RNA-only
python single_chain_level_regression_analysis_full_feature_sets.py \
  --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rnaOnly_per_chain \
  --subset rnaOnly_full \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

python single_chain_level_regression_analysis_selected_features_rnaonly_final.py \
  --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rnaOnly_per_chain \
  --subset rnaOnly_pruned \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

# RNA-protein 1v1
python single_chain_level_regression_analysis_full_feature_sets.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1_full \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

python single_chain_level_regression_analysis_selected_features_rp1v1.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_per_chain \
  --subset rp_1v1_pruned \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

# Multichain per-chain
python single_chain_level_regression_analysis_full_feature_sets.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi_full \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

python single_chain_level_regression_analysis_selected_features_multichain_final.py \
  --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
  --ycol assigned_rna_tm \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_per_chain \
  --subset multi_pruned \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8


# ==============================================================================
# Complex-level analysis
# ==============================================================================

# RNA-protein 1v1 complex
python complex_level_regression_analysis_with_interface_full_feature_set.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
  --ycol Complex_TM \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_complex \
  --subset rp_1v1_full \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

python complex_level_regression_analysis_pruned_features_rp1v1.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
  --ycol Complex_TM \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset rp_1v1_complex \
  --subset rp_1v1_pruned \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

# Multichain complex
python complex_level_regression_analysis_with_interface_full_feature_set.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --ycol Complex_TM \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_complex \
  --subset multi_full \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8

python complex_level_regression_analysis_with_interface_pruned_features_multichain_final.py \
  --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
  --ycol Complex_TM \
  --task regress \
  --target_mode direct \
  --group_by pdb_id \
  --cv 5 \
  --seed 0 \
  --ridge_alpha 10 \
  --dataset multi_complex \
  --subset multi_pruned \
  --outdir feature_selection_final \
  --min_rows 50 \
  --min_rows_model 40 \
  --min_non_na_feature 10 \
  --max_cat_levels 25 \
  --min_category_count 10 \
  --max_missing_frac_feature 0.8 \
  --variance_threshold 1e-8