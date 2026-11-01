#! /bin/bash

# Fully harmonized complex-level runs across the paper
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
#   no robust scaler

# ==============================================================================
# RP 1v1 complex level
# ==============================================================================

echo "=============================================================================================="
echo "RP 1v1 complex level TM regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
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
  --dataset rp_1v1_complex \
  --subset rp_1v1 \
  --target_mode direct \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 complex level TM residual regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
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
  --dataset rp_1v1_complex \
  --subset rp_1v1 \
  --target_mode af3_error \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 complex level Dockq regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
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
  --dataset rp_1v1_complex \
  --subset rp_1v1 \
  --target_mode direct \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 complex level Dockq residual regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
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
  --dataset rp_1v1_complex \
  --subset rp_1v1 \
  --target_mode af3_error \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "RP 1v1 complex level Dockq minus TM regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
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
  --dataset rp_1v1_complex \
  --subset rp_1v1 \
  --target_mode dockq_minus_tm \
  --outdir final_regression_runs \
  --full_model_universe


# ==============================================================================
# Multichain complex level
# ==============================================================================

echo "=============================================================================================="
echo "Multichain complexes complex level TM regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
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
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level TM residual regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
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
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
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
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq residual regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
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
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "Multichain complexes complex level Dockq minus TM regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
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
  --outdir final_regression_runs \
  --full_model_universe


# ==============================================================================
# All complexes complex level
# ==============================================================================

echo "=============================================================================================="
echo "All Complexes complex level TM regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
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
  --dataset all_complex \
  --subset all \
  --target_mode direct \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes complex level TM residual regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
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
  --dataset all_complex \
  --subset all \
  --target_mode af3_error \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes complex level Dockq regression pdb id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
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
  --dataset all_complex \
  --subset all \
  --target_mode direct \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes complex level Dockq residual regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
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
  --dataset all_complex \
  --subset all \
  --target_mode af3_error \
  --outdir final_regression_runs \
  --full_model_universe

echo "=============================================================================================="
echo "All Complexes complex level Dockq minus TM regression pdb_id cv"
echo "=============================================================================================="

python complex_level_regression_analysis_final.py \
  --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
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
  --dataset all_complex \
  --subset all \
  --target_mode dockq_minus_tm \
  --outdir final_regression_runs \
  --full_model_universe

# =============================================================================================
# DockQ_Nr: protein-RNA-only interface DockQ (mean over protein-RNA chain pairs) as the
# complex-level interface target. The whole-complex Dockq runs above are kept as a
# sensitivity analysis. Column DockQ_Nr is written by dbs2analysis_csvs_with_interface.py.
# =============================================================================================
dockqnr_run() {  # $1 csv  $2 dataset  $3 subset  $4 target_mode
  echo "=============================================================================================="
  echo "$2 complex level DockQ_Nr $4 regression pdb_id cv"
  echo "=============================================================================================="
  python complex_level_regression_analysis_final.py \
    --csv "$1" --group_by pdb_id --cv 5 --seed 0 \
    --min_non_na_feature 10 --max_cat_levels 25 --min_category_count 10 \
    --max_missing_frac_feature 0.8 --variance_threshold 1e-8 \
    --task regress --ridge_alpha 10 --min_rows 50 --min_rows_model 40 \
    --ycol DockQ_Nr --dataset "$2" --subset "$3" --target_mode "$4" \
    --outdir final_regression_runs --full_model_universe
}
dockqnr_run csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv   rp_1v1_complex rp_1v1 direct
dockqnr_run csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv   rp_1v1_complex rp_1v1 af3_error
dockqnr_run csvs4analysis_with_interface/combined/combined__multichain_complex.csv multi_complex  multi  direct
dockqnr_run csvs4analysis_with_interface/combined/combined__multichain_complex.csv multi_complex  multi  af3_error
dockqnr_run csvs4analysis_with_interface/combined/complex_level_all.csv            all_complex    all    direct
dockqnr_run csvs4analysis_with_interface/combined/complex_level_all.csv            all_complex    all    af3_error
