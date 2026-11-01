"""
Collect the figures of the paper into figures_final/, under the file names the manuscript uses, and report the
ones that were not produced. Run by reproduce_all.sh after the analyses and figure scripts.

    python utils/collect_paper_figures.py
"""
import glob
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures_final")
SEARCH = ["results/plots", "results/rnacompete", "regression_analysis/publication_figures",
          "regression_analysis/paper_new_analyses", "regression_analysis/final_regression_runs", "regression_analysis"]
EXTS = ("png", "pdf", "svg", "eps")

# file name in the manuscript -> file name the script writes (when they differ)
PAPER = {
    # main text
    "graphical_abstract": None, "postpre_matched": None, "fig2_input_ablation": None,
    "interface_failure_modes": None, "af3_clip_atlas_motifs": None, "decision_tree_2026": None,
    "Fig1_regression": "Figure1", "Fig2_regression": "Figure2",
    # appendix
    "dataset_overview_composition": None, "dataset_overview_covariates": None, "postpre_family_comparison": None,
    "postpre_combined_multichain": None, "si_twin_gap": None, "si_input_ablation": None, "si_confidence_panels": None,
    "compare_ion_input": None, "rnafamily_msa_heatmap": None, "pub_fig1_classification_framework_pred_protein_rna_all_chains": None,
    "interface_motif_match": None, "interface_vs_complex_metrics": None, "interface_gap_vs_composition": None,
    "per_nucleotide_confidence": None, "si_interaction_swaps": None, "si_positional": None, "si_sampling_recovery": None,
    "rna_rna_interfaces": None, "rna_secondary_structure": None, "bond_metrics": None, "domain_binding_coverage": None,
    "rna_accuracy_feature_importance": None, "rna_accuracy_feature_importance_rnaonly": None, "ph_peg_composite": None,
    "seed_model_invariance": None, "combined_rnacompete_publication_figure": None, "pub_competition_outcomes_family": None,
    "FigS_dockq_direct": "FigS_dockq_direct_Figure_model_family_performance__primary",
    "FigS_dockqnr_direct": "FigS_dockqnr_direct_Figure_model_family_performance__primary",
    "FigS_dockqnr_residual": "FigS_dockqnr_residual_Figure_model_family_performance__primary",
    "FigS_rnarna_direct": "FigS_rnarna_direct_Figure_model_family_performance__primary",
    "FigS_rnarna_residual": "FigS_rnarna_residual_Figure_model_family_performance__primary", "fig_shapley_complex": None, "fig_shapley_residual": None,
}
# The appendix figures of single example structures are renders of hand-picked complexes; their panels were partly
# made interactively in PyMOL (publication_figures/si_examples/README.md), so they are not regenerated here.
ILLUSTRATIONS = ["si_ex_torsion", "si_ex_pairing", "si_ex_elements", "si_ex_interface", "si_ex_msa", "si_ex_ions",
                 "si_ex_ablation"]


def newest(name, ext):
    hits = []
    for d in SEARCH:
        hits += glob.glob(os.path.join(REPO, d, "**", f"{name}.{ext}"), recursive=True)
    hits = [h for h in hits if "/notitles/" in h] or hits         # regression figures: the versions without titles
    return max(hits, key=os.path.getmtime) if hits else None


def main():
    os.makedirs(OUT, exist_ok=True)
    missing = []
    for paper_name, produced in PAPER.items():
        found = 0
        for ext in EXTS:
            src = newest(produced or paper_name, ext)
            if src:
                shutil.copyfile(src, os.path.join(OUT, f"{paper_name}.{ext}"))
                found += 1
        if not found:
            missing.append(paper_name)
    print(f"{len(PAPER) - len(missing)} of {len(PAPER)} figures of the paper collected in figures_final/ "
          f"(not regenerated: the {len(ILLUSTRATIONS)} example-structure illustrations of the appendix)")
    for m in missing:
        print(f"  missing: {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
