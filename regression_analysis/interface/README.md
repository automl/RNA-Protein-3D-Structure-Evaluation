# Interface failure: which feature groups predict how a protein-RNA interface fails

Two steps, both in the RNABench env, run from the repository root:

```bash
python regression_analysis/interface/build_interface_features.py
python regression_analysis/interface/interface_failure_feature_groups.py
```

1. `build_interface_features.py` writes one row per native protein-RNA interface
   (`results/interface_features_{post,pre}.tsv`): its placement class
   (`evaluation/interface_rna_pose.py`) and complex-level descriptors in six groups (AF3 confidence;
   experimental and environmental; RNA sequence and structure of the target; evolutionary;
   templates; composition). Structural descriptors come from the experimental structure; nothing
   that defines the classes and no accuracy of the AF3 model is used.
2. `interface_failure_feature_groups.py` fits one-vs-rest gradient-boosting classifiers (correct
   interface; correct site with an incorrect RNA; distal or shifted site; incorrect binding site)
   with cross-validation grouped by PDB entry (5 folds x 3 repeats, every complex weighted equally),
   and splits each target's out-of-fold AUC above 0.5 among the six groups by exact Shapley values.
   Outputs: `results/shapley_{post,pre}.tsv`, `results/coalitions_{post,pre}.tsv` (AUC of every
   group combination) and the heatmap `results/interface_failure_feature_groups.{pdf,png}`.

Inputs: `data/compareDatabases/post_pre/a2021_b2021_protein_rna/{a2021,b2021}.db`, the regression
tables in `regression_analysis/csvs4analysis_with_interface/`, the placement table
`interface_rna_pose.tsv` and, for post-cutoff targets, the familiarity to pre-cutoff RNAs used by
the decision tree (`decision_tree/develop_v3.post_inputs`).
