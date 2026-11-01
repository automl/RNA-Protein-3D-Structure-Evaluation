#!/bin/bash
# Automated stability selection across the key datasets/targets (Step D).
set +e
CX=complex_level_regression_analysis_final.py
SC=single_chain_level_regression_analysis_final.py
D=csvs4analysis_per_chain_complex/combined
X=csvs4analysis_with_interface/combined
SS() {
  # resume: skip a dataset/subset that already has a summary, unless FORCE_RERUN=1
  local args=("$@") ds="" sub="" i
  for ((i=0; i<${#args[@]}; i++)); do
    [ "${args[$i]}" = "--dataset" ] && ds="${args[$((i+1))]}"
    [ "${args[$i]}" = "--subset" ]  && sub="${args[$((i+1))]}"
  done
  if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "feature_selection_stability/$ds/$sub/selection_summary.json" ]; then
    echo "[skip] stability selection $ds/$sub (already done; FORCE_RERUN=1 to redo)"; return 0
  fi
  python stability_selection.py --outdir feature_selection_stability --group_by pdb_id --n_boot 100 "$@"
}

SS --module $CX --csv $X/complex_level_all.csv            --ycol Complex_TM --dataset all_complex        --subset all
SS --module $CX --csv $X/combined__multichain_complex.csv --ycol Complex_TM --dataset multi_complex      --subset multi
SS --module $CX --csv $X/complex_level_all.csv            --ycol Dockq      --dataset all_complex_dockq  --subset all
SS --module $CX --csv $X/complex_level_all.csv            --ycol DockQ_Nr   --dataset all_complex_dockqnr --subset all
SS --module $SC --csv $D/combined__per_chain_multichain_complex.csv --ycol assigned_rna_tm --dataset multi_per_chain --subset multi
SS --module $SC --csv $D/combined__per_chain_rna_protein_1v1.csv    --ycol assigned_rna_tm --dataset rp_1v1_per_chain --subset rp_1v1
SS --module $SC --csv csvs4analysis_rnaonly_with_af3/combined/combined__per_chain_all.csv --ycol assigned_rna_tm --dataset rnaOnly_per_chain --subset rnaOnly
[ -f csvs4analysis_rna_rna/combined/combined__per_chain_all.csv ] && SS --module $SC --csv csvs4analysis_rna_rna/combined/combined__per_chain_all.csv --ycol assigned_rna_tm --dataset rna_rna_per_chain --subset rna_rna
echo "########## stability selection DONE ##########"
