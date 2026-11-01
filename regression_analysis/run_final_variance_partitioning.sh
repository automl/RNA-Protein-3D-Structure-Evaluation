#! /bin/bash
#
# Order-independent (Shapley-R^2 / LMG) variance partitioning over the six
# feature groups, for the sections-1/2 ranking claims. Reuses the SAME fixed
# feature registry, harmonized pipeline and GroupKFold-by-pdb_id used in the
# main analysis (see variance_partitioning_shapley.py). All data pooled.
#
# Common setup (matches run_final_complex_level_regression.sh):
#   --cv 5 --seed 0 --ridge_alpha 10 --variance_threshold 1e-8
#   --min_category_count 10 --min_non_na_feature 10 --max_cat_levels 25
#   --max_missing_frac_feature 0.8 --group_by pdb_id
#
# Set PY to whatever launches the analysis python (env with sklearn/pandas).
PY="${PY:-python}"

CX_MOD="complex_level_regression_analysis_final.py"
PC_MOD="single_chain_level_regression_analysis_final.py"
OUT="variance_partitioning_final"

run() { echo; echo "### $*"; $PY variance_partitioning_shapley.py "$@" --outdir "$OUT"; }

# ==============================================================================
# Complex level (direct): rank groups for whole-complex quality
# ==============================================================================
for ycol in Complex_TM Dockq DockQ_Nr; do
  run --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
      --ycol $ycol --registry_module $CX_MOD --dataset multi_complex --subset multi --target_mode direct
  run --csv csvs4analysis_with_interface/combined/combined__rna_protein_1v1.csv \
      --ycol $ycol --registry_module $CX_MOD --dataset rp_1v1_complex --subset rp_1v1 --target_mode direct
  run --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
      --ycol $ycol --registry_module $CX_MOD --dataset all_complex --subset all --target_mode direct
done

# ==============================================================================
# Complex level (af3_error): what explains the residual AF3 confidence misses
# (partitions residual variance over the 5 non-AF3 groups)
# ==============================================================================
for ycol in Complex_TM Dockq DockQ_Nr; do
  run --csv csvs4analysis_with_interface/combined/combined__multichain_complex.csv \
      --ycol $ycol --registry_module $CX_MOD --dataset multi_complex --subset multi --target_mode af3_error
  run --csv csvs4analysis_with_interface/combined/complex_level_all.csv \
      --ycol $ycol --registry_module $CX_MOD --dataset all_complex --subset all --target_mode af3_error
done

# ==============================================================================
# Per-chain level (direct): rank groups for single-RNA-chain quality
# ==============================================================================
run --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
    --ycol assigned_rna_tm --registry_module $PC_MOD --dataset multi_per_chain --subset multi --target_mode direct
run --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_rna_protein_1v1.csv \
    --ycol assigned_rna_tm --registry_module $PC_MOD --dataset rp_1v1_per_chain --subset rp_1v1 --target_mode direct
run --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv \
    --ycol assigned_rna_tm --registry_module $PC_MOD --dataset rnaOnly_per_chain --subset rnaOnly --target_mode direct

# per-chain residual
run --csv csvs4analysis_per_chain_complex/combined/combined__per_chain_multichain_complex.csv \
    --ycol assigned_rna_tm --registry_module $PC_MOD --dataset multi_per_chain --subset multi --target_mode af3_error

# multi-chain RNA complexes (separate set; skipped when the rna_rna csv is absent)
RR=csvs4analysis_rna_rna/combined/combined__per_chain_all.csv
if [ -f "$RR" ]; then
  run --csv $RR --ycol assigned_rna_tm --registry_module $PC_MOD --dataset rna_rna_per_chain --subset rna_rna --target_mode direct
  run --csv $RR --ycol assigned_rna_tm --registry_module $PC_MOD --dataset rna_rna_per_chain --subset rna_rna --target_mode af3_error
fi
echo; echo "Done. Outputs under $OUT/"
