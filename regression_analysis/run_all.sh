#! /bin/bash
# Full analysis: preprocessing -> feature selection -> regressions -> base panel
# plots -> variance partitioning. Ordering matters for a from-scratch run: the
# per-chain comparison plots depend on the no-interface and multichain-comparison
# runs, so those are produced BEFORE the plotting step.
set -e

# Windows MAX_PATH workaround: map a short drive (X:) to this project and run from
# it so deep output paths stay under the 260-char limit. Auto-applied every run.
# Per-run resume is handled inside the analysis scripts via a .run_complete marker
# (set FORCE_RERUN=1 to recompute everything regardless).
source "$(dirname "$0")/ensure_x_drive.sh"

# 1. database preprocessing (data/*.db -> csvs4analysis_*)
bash run_preprocessing.sh

# 2. feature selection (full vs pruned) + stability figure (Fig 15)
bash run_full_vs_pruned.sh
bash plot_feature_selection.sh

# 3. complex-level regressions (section 1) + Figure 1 panels
bash run_final_complex_level_regression.sh
bash plot_regression_results_complexes_section1.sh

# 4. per-chain regressions (section 2) and the runs the comparison plots need
bash run_final_full_per_chain_regression.sh
bash run_final_single_chain_regression_without_interface.sh
bash run_final_multichain_per_chain_complex_comparison.sh

# 5. Figure 2 panels (need: per-chain runs, no-interface runs, multichain comparison)
bash plot_final_single_chain_system_comparison_results.sh

# 6. order-independent (Shapley-R2) variance partitioning for group-ranking claims
bash run_final_variance_partitioning.sh
