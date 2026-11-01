#! /bin/bash

# rnaonly
python plot_final_single_chain_system_comparison_results.py \
  --run-dir final_regression_runs/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5 \
  --target-label assigned_rna_tm \
  --mode performance \
  --base-filename-prefix "Fig2A_rnaonly" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 24 \
  --xtick-labelsize 22 \
  --ytick-labelsize 24 \
  --performance-models \
    G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
    A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
    A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
  --performance-error-bars \
  --performance-sort-within fixed \
  --performance-show-section-labels \
  --performance-primary-title "" \
  --performance-fig-width 7.4 \
  --performance-fig-height-min 5.8 \
  --performance-row-height 0.33 \
  --performance-section-gap 0.95 \
  --performance-section-label-fontsize 14

# RP 1v1 per chain
python plot_final_single_chain_system_comparison_results.py \
  --run-dir final_regression_runs/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5 \
  --target-label assigned_rna_tm \
  --mode performance \
  --base-filename-prefix "Fig2B_rp1v1" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 24 \
  --xtick-labelsize 22 \
  --ytick-labelsize 24 \
  --performance-models \
    G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
    A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
    A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
  --performance-error-bars \
  --performance-sort-within fixed \
  --performance-show-section-labels \
  --performance-primary-title "" \
  --performance-fig-width 7.4 \
  --performance-fig-height-min 5.8 \
  --performance-row-height 0.33 \
  --performance-section-gap 0.95 \
  --performance-section-label-fontsize 14

# multichain complexes per chain
python plot_final_single_chain_system_comparison_results.py \
  --run-dir final_regression_runs/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5 \
  --target-label assigned_rna_tm \
  --mode performance \
  --base-filename-prefix "Fig2C_multichain" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 24 \
  --xtick-labelsize 22 \
  --ytick-labelsize 24 \
  --performance-models \
    G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
    A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
    A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
  --performance-error-bars \
  --performance-sort-within fixed \
  --performance-show-section-labels \
  --performance-primary-title "" \
  --performance-fig-width 7.4 \
  --performance-fig-height-min 5.8 \
  --performance-row-height 0.33 \
  --performance-section-gap 0.95 \
  --performance-section-label-fontsize 14

# heatmap across systems without interface features
# python plot_final_d-f_system_comparison.py \
#   --mode heatmap \
#   --run "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5" \
#   --run "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5" \
#   --run "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5" \
#   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
#   --feature-set union \
#   --drop-all-zero-rows \
#   --sort-features max_abs \
#   --top-n 30 \
#   --out Fig2D_heatmap.png \
#   --context paper \
#   --dpi 600 \
#   --heatmap-width 5.8 \
#   --heatmap-row-height 0.34 \
#   --heatmap-min-height 6.2 \
#   --axes-labelsize 18 \
#   --xtick-labelsize 15 \
#   --ytick-labelsize 11

# python plot_final_d-f_system_comparison.py \
#   --mode heatmap \
#   --run "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5" \
#   --run "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5" \
#   --run "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5" \
#   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
#   --feature-set union \
#   --sort-features max_abs \
#   --top-k-per-group 2 \
#   --drop-all-zero-rows \
#   --heatmap-show-group-separators \
#   --heatmap-show-group-names \
#   --heatmap-width 4.9 \
#   --heatmap-row-height 0.42 \
#   --ytick-labelsize 8.5 \
#   --xtick-labelsize 9 \
#   --axes-labelsize 11 \
#   --heatmap-vmin -0.12 \
#   --heatmap-vmax 0.12 \
#   --out Fig2D_heatmap_refined.png

# heatmap

# python plot_final_d-f_system_comparison.py   --mode heatmap   --run "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5"   --run "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5"   --run "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5"   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental   --feature-set union   --sort-features max_abs   --top-k-per-group 2   --drop-all-zero-rows   --heatmap-show-group-separators   --heatmap-show-group-names   --heatmap-width 5.3   --heatmap-row-height 0.42   --heatmap-label-col-width 1.30   --heatmap-group-strip-width 0.20   --heatmap-strip-pad 0.01   --heatmap-feature-label-fontsize 8.8   --heatmap-group-name-fontsize 7.6   --heatmap-vmin -0.12   --heatmap-vmax 0.12   --out Fig2D_heatmap_vertical_groups.png --ylabel "" --heatmap-left 0.002 --heatmap-label-col-width 0.85 --heatmap-feature-label-x 0.995 --heatmap-label-separator-x0 0.40 --heatmap-label-separator-x1 0.98 --axes-labelsize 14

# python plot_final_d-f_system_comparison.py   --mode heatmap   --run "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5"   --run "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5"   --run "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5"   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental   --feature-set union   --sort-features max_abs   --top-k-per-group 2   --drop-all-zero-rows   --heatmap-show-group-separators   --heatmap-show-group-names   --heatmap-width 5.3   --heatmap-row-height 0.42   --heatmap-label-col-width 1.30   --heatmap-group-strip-width 0.20   --heatmap-strip-pad 0.01   --heatmap-feature-label-fontsize 12   --heatmap-group-name-fontsize 12   --heatmap-vmin -0.12   --heatmap-vmax 0.12   --out Fig2D_heatmap_vertical_groups.png --ylabel "" --heatmap-left 0.002 --heatmap-label-col-width 0.85 --heatmap-feature-label-x 0.995 --heatmap-label-separator-x0 0.40 --heatmap-label-separator-x1 0.98

python plot_final_d-f_system_comparison.py   --mode heatmap   --run "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5"   --run "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5"   --run "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5"   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental   --feature-set union   --sort-features max_abs   --top-k-per-group 2   --drop-all-zero-rows   --heatmap-show-group-separators   --heatmap-show-group-names   --heatmap-width 5.3   --heatmap-row-height 0.42   --heatmap-label-col-width 1.30   --heatmap-group-strip-width 0.20   --heatmap-strip-pad 0.01   --heatmap-feature-label-fontsize 16   --heatmap-group-name-fontsize 9   --heatmap-vmin -0.12   --heatmap-vmax 0.12   --out Fig2D_heatmap_vertical_groups.png --ylabel "" --heatmap-left 0.002 --heatmap-label-col-width 0.85 --heatmap-feature-label-x 0.995 --heatmap-label-separator-x0 0.20 --heatmap-label-separator-x1 0.98 --axes-labelsize 14 --xtick-labelsize 12

# multichain complex vs multichain per chain comparison
python plot_final_d-f_system_comparison.py   --mode multichain_shift   --run-a "RNA chain TM=final_multichain_comparison/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5"   --run-b "Complex TM=final_multichain_comparison/multi_complex/cx/multi/Complex_TM__a10p0__grppdb_id__cv5"   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental   --top-n 15   --delta-xlabel "Coefficient Change"   --bar-width 5.4   --bar-row-height 0.34   --bar-min-height 4.9   --single-bar-height 0.80   --show-group-legend   --legend-loc "upper left"   --out Fig2E_multichain_shift_n15.png   --context paper   --dpi 600 --ylabel "" --axes-labelsize 20 --ytick-labelsize 18 --legend-fontsize 14 --xtick-labelsize 16
# python plot_final_d-f_system_comparison.py \
#   --mode multichain_shift \
#   --run-a "RNA chain TM=final_multichain_comparison/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5" \
#   --run-b "Complex TM=final_multichain_comparison/multi_complex/cx/multi/Complex_TM__a10p0__grppdb_id__cv5" \
#   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
#   --top-n 15 \
#   --delta-xlabel "Coefficient Change" \  #  (Complex TM − RNA Chain TM)" \
#   --bar-width 5.4 \
#   --bar-row-height 0.34 \
#   --bar-min-height 4.9 \
#   --single-bar-height 0.80 \
#   --show-group-legend \
#   --legend-loc "upper left" \
#   --out Fig2E_multichain_shift_n15.png \
#   --context paper \
#   --dpi 600 \
#   --hide-ylabel


 python plot_final_d-f_system_comparison.py   --mode system_delta   --run-rnaonly "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5"   --run-1v1 "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5"   --run-multi "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5"   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental   --top-n 15   --system-delta-sort score   --delta-xlabel "Coefficient Change"   --bar-width 5.4   --bar-row-height 0.34   --bar-min-height 4.9   --system-delta-bar-height 0.28   --transition1-color "#4C78A8"   --transition2-color "#F28E2B"   --transition1-label "1v1 − RNA-only"   --transition2-label "Multichain − 1v1"   --legend-loc "upper center" --legend-bbox 0.5 -0.15 --legend-ncol 2   --out Fig2F_system_deltas_n15.png   --context paper   --dpi 600 --ylabel "" --axes-labelsize 20 --ytick-labelsize 18 --legend-fontsize 16 --xtick-labelsize 16
 
 # python plot_final_d-f_system_comparison.py \
#   --mode system_delta \
#   --run-rnaonly "RNA only=runs_no_interface/rnaOnly_per_chain/pca/rnaOnly/tm__a10p0__gpdb__cv5" \
#   --run-1v1 "1v1=runs_no_interface/rp_1v1_per_chain/pca/rp_1v1/tm__a10p0__gpdb__cv5" \
#   --run-multi "Multichain=runs_no_interface/multi_per_chain/pca/multi/tm__a10p0__gpdb__cv5" \
#   --coef-model A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
#   --top-n 15 \
#   --system-delta-sort score \
#   --delta-xlabel "Coefficient Change" \
#   --bar-width 5.4 \
#   --bar-row-height 0.34 \
#   --bar-min-height 4.9 \
#   --system-delta-bar-height 0.28 \
#   --transition1-color "#4C78A8" \
#   --transition2-color "#F28E2B" \
#   --transition1-label "1v1 − RNA-only" \
#   --transition2-label "Multichain − 1v1" \
#   --legend-loc "lower right" \
#   --out Fig2F_system_deltas_n15.png \
#   --context paper \
#   --dpi 600 \
#   --hide-ylabel

# ---- multi-chain RNA complexes (separate set): per-chain ladders, direct and residual
if [ -d final_regression_runs/rna_rna_per_chain ]; then
for spec in "assigned_rna_tm FigS_rnarna_direct" "AF3_ERROR__assigned_rna_tm FigS_rnarna_residual"; do
  set -- $spec
  python plot_final_regression_analysis_results.py \
    --runs-root final_regression_runs/rna_rna_per_chain/pca/rna_rna \
    --target-label "$1" --mode performance --base-filename-prefix "$2" --context paper --dpi 600 \
    --axes-labelsize 16 --xtick-labelsize 15 --ytick-labelsize 15 \
    --performance-models G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
      A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
      A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
    --performance-error-bars --performance-sort-within value_asc --performance-show-section-labels \
    --performance-primary-title "" --performance-secondary-title "" \
    --performance-fig-width 7.4 --performance-fig-height-min 5.8 --performance-row-height 0.33 \
    --performance-section-gap 0.95 --performance-section-label-fontsize 13
done
fi
