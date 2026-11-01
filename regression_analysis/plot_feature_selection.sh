#! /bin/bash
# Feature-stability barplot of the hand-curated full-vs-pruned comparison (run_full_vs_pruned.sh,
# optional; not part of reproduce_all.sh). Skipped when its inputs are absent: the paper's
# feature-selection table is built from feature_selection_stability/ (stability selection).
if [ ! -d feature_selection_final ]; then
  echo "[plot_feature_selection] skip: feature_selection_final/ absent (optional hand-curated comparison)"
  exit 0
fi

python plot_feature_stability_summary.py   --input-dir feature_selection_final   --outdir feature_stability_outputs   --selected-features final_feature_set.py   --top-pos-k 10   --top-neg-k 10   --display-max-per-group 5   --group-gap 1.2   --barplot-normalize   --highlight-selected   --barplot-width 14   --barplot-row-height 0.34   --title-fontsize 15   --tick-fontsize 12.0   --pos-color "#4C78A8"   --neg-color "#E45756"   --group-header-mode inside   --hide-barplot-ylabel --legend-loc "lower center" --group-header-y-pad 0.68 --barplot-title "" --label-fontsize 15.0 --tick-fontsize 15.0 --legend-fontsize 15.0 --legend-ncol 3 --group-header-fontsize 15.0 --unselected-color "#6E6E6E" --legend-anchor-x 0.5 --legend-anchor-y -0.09 --legend-frameon