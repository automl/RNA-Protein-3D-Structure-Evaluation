python plot_final_regression_analysis_results.py \
  --runs-root final_regression_runs/all_complex/cx/all \
  --target-label Complex_TM \
  --mode performance \
  --base-filename-prefix "Fig1A_direct" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 16 \
  --xtick-labelsize 15 \
  --ytick-labelsize 15 \
  --performance-models \
    G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
    A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
    A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
  --performance-error-bars \
  --performance-sort-within value_asc \
  --performance-show-section-labels \
  --performance-primary-title "" \
  --performance-secondary-title "" \
  --performance-fig-width 7.4 \
  --performance-fig-height-min 5.8 \
  --performance-row-height 0.33 \
  --performance-section-gap 0.95 \
  --performance-section-label-fontsize 13

python plot_final_regression_analysis_results.py \
  --runs-root final_regression_runs/all_complex/cx/all \
  --target-label AF3_ERROR__Complex_TM \
  --mode performance \
  --base-filename-prefix "Fig1B_residual" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 16 \
  --xtick-labelsize 15 \
  --ytick-labelsize 15 \
  --performance-models \
    G0_af3 G1_evolution G2_complexity G3_rna_intrinsic G4_interface G5_experimental \
    A1_af3_evolution A2_af3_complexity A3_af3_rna_intrinsic A4_af3_interface A5_af3_experimental \
    A31_af3_evolution_complexity_rna_intrinsic_interface_experimental \
  --performance-error-bars \
  --performance-sort-within value_asc \
  --performance-show-section-labels \
  --performance-primary-title "" \
  --performance-secondary-title "" \
  --performance-fig-width 7.4 \
  --performance-fig-height-min 5.8 \
  --performance-row-height 0.33 \
  --performance-section-gap 0.95 \
  --performance-section-label-fontsize 13

# python plot_final_regression_analysis_results.py \
#   --runs-root final_regression_runs/all_complex/cx/all \
#   --target-label Complex_TM \
#   --mode coefs \
#   --coef-model a31 \
#   --base-filename-prefix "Fig1C_coef" \
#   --context paper \
#   --dpi 600 \
#   --axes-labelsize 20 \
#   --xtick-labelsize 16 \
#   --ytick-labelsize 16 \
#   --coef-top-n 8 \
#   --coef-fig-width 7.8 \
#   --coef-fig-height-min 5.8 \
#   --coef-row-height 0.42 \
#   --hide-coef-direction-note

python plot_final_regression_analysis_results.py \
  --runs-root final_regression_runs/all_complex/cx/all \
  --target-label Complex_TM \
  --mode coefs \
  --coef-model a31 \
  --base-filename-prefix "Fig1C_coef_hor" \
  --coef-horizontal \
  --coef-top-n 8 \
  --coef-horizontal-fig-width 12 \
  --coef-horizontal-fig-height 6.5 \
  --xtick-labelsize 22 \
  --ytick-labelsize 18 \
  --axes-labelsize 24 \
  --coef-xlabel "Std. Ridge Coeff." \
  --coef-horizontal-label-dx 0.05 \
  --hide-coef-direction-note
  
python plot_final_regression_analysis_results.py \
  --runs-root final_regression_runs/all_complex/cx/all \
  --target-label Complex_TM \
  --raw-csv csvs4analysis_with_interface/combined/complex_level_all.csv \
  --mode scatter \
  --base-filename-prefix "Fig1D_scatter" \
  --context paper \
  --dpi 600 \
  --axes-labelsize 20 \
  --xtick-labelsize 18 \
  --ytick-labelsize 18 \
  --scatter-vars \
    "max_rna_fraction_of_total_rna|Complex_TM|Largest RNA Fraction|Complex TM|Accuracy vs RNA Fraction" \
  --scatter-fig-width 5.8 \
  --scatter-fig-height 5.0 \
  --scatter-point-size 18 \
  --scatter-point-alpha 0.50 \
  --scatter-reg-linewidth 2.1 \
  --scatter-annot-fontsize 16 \
  --scatter-jitter-frac 0.05 \
  --scatter-annot-loc upper_left \
  --hide-n

# ---- interface target: DockQ_Nr (main) and whole-complex DockQ (sensitivity), all complexes
for spec in "DockQ_Nr FigS_dockqnr_direct" "AF3_ERROR__DockQ_Nr FigS_dockqnr_residual" "Dockq FigS_dockq_direct"; do
  set -- $spec
  python plot_final_regression_analysis_results.py \
    --runs-root final_regression_runs/all_complex/cx/all \
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
