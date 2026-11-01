#!/bin/bash
# All figure scripts of the paper (main text and appendix), run from the repository root by reproduce_all.sh.
# They read the published databases (fetch_data.sh) and the intermediate tables in results/plots/all_chains/,
# which reproduce_all.sh takes from analysis_inputs.zip unless evaluation/make_analysis_tables.sh recomputed them.
# A failing script is reported and the next one runs; reproduce_all.sh then reports which figures are missing.
set +e
cd "$(dirname "$0")"
PY="${PY:-python}"
failed=()
run() { echo; echo "### $*"; "$PY" "$@" || { echo "### FAILED: $*"; failed+=("$*"); }; }

# main text and appendix figures
for s in post_pre_comparison_figure postpre_family_comparison_figure postpre_matched_figure \
         fig2_input_ablation si_inputs_and_confidence msa_temp_influence \
         compare_ion_input_figure rna_family_publication_figure \
         pub_fig1_classification_framework interface_failure_modes_figure interface_motif_match_figure \
         interface_vs_complex_metrics per_nucleotide_confidence si_interaction_swaps si_new_panels \
         rna_rna_interfaces_figure rna_secondary_structure_figure bond_metrics_figure \
         domain_publication_figure rna_accuracy_feature_importance_figure ph_peg_composite \
         af3_clip_atlas_motif_figure graphical_abstract; do
  run "publication_figures/$s.py"
done
run seeds/seed_model_comparison_figure.py
FIG_RENDER=plain run -m decision_tree.figure_2026

# RNAcompete
RNC_DIR=data/rnacompete; RNC_OUT=results/rnacompete
if [ -f "$RNC_DIR/rnaCompete_v4.db" ] && [ -f "$RNC_DIR/rnaCompete_v5_MINMAX.db" ]; then
  mkdir -p "$RNC_OUT"
  run rnaCompete/prep_rnacompete_dbs.py --dir "$RNC_DIR"
  run rnaCompete/combined_publication_figure.py --db "$RNC_DIR/rnaCompete_v4.db" --outdir "$RNC_OUT"
  run rnaCompete/exp_probeMax_combined.py --db "$RNC_DIR/rnaCompete_v5_MINMAX.db" \
      --csv "$RNC_DIR/exp_protein_rna.csv" --outdir "$RNC_OUT" --pub
fi

echo
if [ ${#failed[@]} -eq 0 ]; then echo "all figure scripts finished"; else printf 'figure scripts that failed:\n'; printf '  %s\n' "${failed[@]}"; fi
