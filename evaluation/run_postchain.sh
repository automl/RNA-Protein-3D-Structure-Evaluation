#!/bin/bash
# Interface analyses that run after the evaluation pipeline (evaluation/mainExecution.py), in the order they
# depend on each other. They work on the published layout (evaluation/bundle_paths.py):
#   structures   data/all_data/<set>/protein_rna/{pdb,af,lddt}        (bash fetch_data.sh --structures)
#   databases    data/all_data_zip/all_databases/<set>/protein_rna.db  (rewritten in place: interface columns)
# and write their tables to results/plots/all_chains/. Steps marked (ost) run in the `ost` environment, the others
# in `RNABench`. Override the interpreters with PY_RNABENCH / PY_OST if your environments have other names.
#
#   bash evaluation/run_postchain.sh
set -eu
cd "$(dirname "$0")/.."
PY_RNABENCH="${PY_RNABENCH:-conda run --no-capture-output -n RNABench python}"
PY_OST="${PY_OST:-conda run --no-capture-output -n ost python}"
step() { echo; echo "=== $*"; }

step "exact-match scores";            $PY_RNABENCH evaluation/update_exact_match_scores.py \
                                        data/all_data_zip/all_databases/a2021_post_cutoff/protein_rna.db \
                                        data/all_data_zip/all_databases/b2021_pre_cutoff/protein_rna.db
step "RNA pose of every interface";   $PY_RNABENCH evaluation/interface_rna_pose.py
step "exact-mapping DockQ: interfaces OpenStructure did not score"
$PY_RNABENCH evaluation/interface_own_dockq.py export
$PY_OST      evaluation/interface_own_dockq.py score
step "exact-mapping DockQ: validation quarter and the remaining interfaces"
$PY_RNABENCH evaluation/interface_own_dockq.py export --validate
$PY_OST      evaluation/interface_own_dockq.py score --validate
$PY_RNABENCH evaluation/interface_own_dockq.py export --rest
$PY_OST      evaluation/interface_own_dockq.py score --rest
step "whole-complex scores with all mapped chains"
$PY_RNABENCH evaluation/interface_own_dockq.py export-complex
$PY_OST      evaluation/interface_own_dockq.py score-complex
step "write the fallback values into the databases"
$PY_RNABENCH evaluation/interface_own_dockq.py write-db
$PY_RNABENCH evaluation/interface_own_dockq.py write-metrics
step "placement classes with the final DockQ"
$PY_RNABENCH evaluation/interface_rna_pose.py --relabel
$PY_RNABENCH evaluation/interface_own_dockq.py write-metrics
step "symmetry-equivalent chains";    $PY_RNABENCH evaluation/interface_symmetry_check.py
step "slid duplexes";                 $PY_RNABENCH evaluation/apply_duplex_pose.py
step "placement tables"
$PY_RNABENCH evaluation/placement_all_chains.py
$PY_RNABENCH evaluation/interface_class_composition.py
$PY_RNABENCH publication_figures/placement_per_rna_chain.py
echo; echo "post-chain finished; tables in results/plots/all_chains/"
