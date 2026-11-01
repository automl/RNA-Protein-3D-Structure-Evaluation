#!/bin/bash
# Intermediate tables of the paper's analyses, computed from the published databases and structures
# (bash fetch_data.sh --structures). The figure scripts read them from results/plots/all_chains/; reproduce_all.sh
# uses the copies shipped in analysis_inputs.zip (results/plots_odl/) when they have not been computed here.
# None of these steps changes a database.
#
# Run in the RNABench environment (conda activate RNABench); the OpenStructure tool of the `ost` environment is found
# automatically (or set OST_BIN). The CLIP motif comparison downloads four motif atlases and queries UniProt, so it
# needs internet access.
#
#   bash evaluation/make_analysis_tables.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
WORKERS="${WORKERS:-4}"          # parallel US-align / OpenStructure jobs
failed=()
step() {  # $1 = label, rest = command; a failing step is reported and the next one runs
  local label=$1; shift
  echo; echo "=== $label"
  "$@" || { echo "*** $label failed"; failed+=("$label"); }
}

# start from the published interface tables (pose, placement), which the later steps read
mkdir -p results/plots
[ -d results/plots_odl/all_chains ] && cp -rn results/plots_odl/all_chains results/plots/

step "symmetry-equivalent chains"         $PY evaluation/interface_symmetry_check.py
step "placement class composition"        $PY evaluation/interface_class_composition.py
step "contact match per interface"        $PY evaluation/interface_motif_match.py
step "placement and RNA geometry"         $PY evaluation/interface_class_rna_geometry.py
step "placement threshold sensitivity"    $PY evaluation/placement_threshold_sensitivity.py
step "RNA-RNA interfaces"                 $PY evaluation/rna_rna_interfaces.py
step "similarity-matched post/pre pairs"  $PY evaluation/postpre_matched_similarity.py   # before the next step, which reads the sequence twins from it
step "structural similarity to pre-cutoff RNAs (US-align)"  $PY evaluation/structural_similarity_to_pre.py --align --workers "$WORKERS"
step "seed and model samples"             $PY evaluation/seed_sample_interfaces.py --workers "$WORKERS"
step "sequence familiarity (decision tree)"     $PY -m decision_tree.pdb_sequence_familiarity
step "fold familiarity (decision tree)"         $PY -m decision_tree.predicted_fold_familiarity
step "decision tree: held-out validation"     $PY -m decision_tree.validate
step "decision tree: tree validation"         $PY -m decision_tree.validate_tree
step "decision tree: 2026 validation"           $PY -m decision_tree.validate_2026
step "decision tree: point-score framework"     $PY -m decision_tree.develop_v3
step "decision tree: RNA-only subtree"          $PY -m decision_tree.develop_rna
step "decision tree: combined levels"           $PY -m decision_tree.combined
step "decision tree: design"                    $PY -m decision_tree.develop_probability
step "decision tree: 2026 test"                 $PY -m decision_tree.develop_probability --test
step "decision tree: RNA-only design"           $PY -m decision_tree.develop_probability --rna
step "decision tree: RNA-only 2026 test"        $PY -m decision_tree.develop_probability --rna --test
step "CLIP motif comparison"              bash -c "cd clip/scripts && PATH=\"$(dirname "$(command -v "$PY")"):\$PATH\" bash run_all.sh"
step "RNAcompete tables"                        $PY rnaCompete/prep_rnacompete_dbs.py --dir data/rnacompete
step "decision tree: applications"              $PY -m decision_tree.applications --probability

echo
if [ ${#failed[@]} -eq 0 ]; then echo "all analysis tables computed"; else printf 'failed steps:\n'; printf '  %s\n' "${failed[@]}"; exit 1; fi
