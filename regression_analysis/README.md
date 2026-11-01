# Reliability regimes for RNA structure prediction in AlphaFold 3 — analysis code

Self-contained reproduction of the regression, variance-partition and feature-selection
analyses behind the paper *"Defining Reliability Regimes for RNA Structure Prediction in
AlphaFold 3"*. Everything is regenerated from the SQLite databases with a single command.

## Quick start

```bash
conda env create -f environment.yml      # creates the 'af3_eval' env
conda activate af3_eval
bash reproduce_all.sh                     # preprocessing -> regressions -> Shapley -> selection
```

Outputs land in:

| directory | contents |
|---|---|
| `csvs4analysis_*/` | per-analysis feature tables parsed from the databases |
| `final_regression_runs/` | grouped-CV ridge regressions for every subset/target |
| `variance_partitioning_final/` | order-independent **Shapley-R²** group shares (`*/shapley_r2.csv`) |
| `feature_selection_stability/` | **automated stability-selection** results + nested-CV R² (`*/selection_summary.json`) |
| `figures_final/` | publication figures (only if the plotting packages are present, see below) |

## What the pipeline does

1. **Preprocessing** (`run_preprocessing.sh`): parses the paired prediction/experiment
   databases in `data/` into per-chain and complex-level feature tables. The six combined
   source databases live in `data/post_pre/` (protein-RNA, single-chain RNA and multi-chain
   RNA, each pre- and post-cutoff); the protein-RNA ones are split by experimental method on
   the fly (`split_dbs_by_experiment.py`). The complex-level table carries both the
   whole-complex `Dockq` and `DockQ_Nr`, the mean DockQ over the protein-RNA chain pairs only
   (`dbs2analysis_csvs_with_interface.py`).
2. **Regressions** (`run_final_*regression*.sh`): grouped 5-fold cross-validated ridge
   (folds by `pdb_id`, `seed=0`, `alpha=10`), for RNA-only chains, single (1v1) and
   multichain RNA–protein complexes, targeting per-chain RNA TM-score, complex TM-score
   and, at the complex level, the protein-RNA interface DockQ (`DockQ_Nr`; the whole-complex
   `Dockq` runs are kept as a sensitivity analysis). Six feature groups: **G0** AF3
   confidence, **G1** evolution (MSA depth, templates, family), **G2** complexity, **G3**
   RNA-intrinsic, **G4** predicted-interface, **G5** experimental context.
   **Multi-chain RNA complexes** (two or more RNA chains, no protein) are analysed as a
   separate set at the per-chain level (`preprocess_rna_rna_per_chain_metrics.py`,
   `run_final_rna_rna_per_chain.sh`, outputs under `rna_rna_per_chain/`): the same registry
   applies, but protein-side descriptors, alignment depth, thermodynamic descriptors and the
   experimental-context group are not available for this set and drop out of the screen;
   G3 is sequence-derived (GC content, entropy, complexity) and G4 is the predicted RNA-RNA
   interface (contacts per chain, contact distances, partner chains, hydrogen bonds).
3. **Variance partition** (`variance_partitioning_shapley.py`): an order-independent
   Shapley-R² decomposition of the cross-validated R² over the six groups (the group-ranking
   claims). Reuses the exact same registry, pipeline and grouped CV.
4. **Feature selection** (`stability_selection.py`): data-driven selection replacing any
   hand-curated feature list. For each subset/target it runs **stability selection** (ElasticNet
   over 100 grouped resamples, keep features selected in ≥70%) on the full non-leaky candidate
   pool, and reports both the fixed pooled-selected set and an honest **nested-CV R²**
   (selection redone inside each fold). Accuracy-derived columns (INF, DI, match scores, …)
   are excluded up front to prevent circular selection.

Methodological notes: all reported R² are cross-validated **explained variance within the
analysed distribution** (associational, not out-of-sample prediction). Preprocessing is fit
strictly within each training fold (no leakage), and the group ranking is robust to the ridge
`alpha` (shares are essentially unchanged for `alpha ∈ {1, 10, 100}`).

## Publication figures

`bash plot_*.sh` and `assemble_figures.py` draw the regression figures (Figures 1 and 2 and the supplementary
regression and Shapley figures); they run as part of `reproduce_all.sh`. The other figures of the paper are drawn by
the scripts in `../publication_figures/`, run by the top-level `../reproduce_all.sh`.

## Data

`data/post_pre/` holds the six combined databases (pre-/post-2021-cutoff, protein-RNA,
single-chain RNA and multi-chain RNA), placed by `../fetch_data.sh` from the final database
bundle (`../DATA.md`). `data/compare_ions/` and `data/compareDatabases/noMSA/…` hold the
ion-input and MSA/template-ablation databases used by the figures. `database/rbpDatabase.db`
is the post-cutoff protein-RNA database used by several figure scripts.

## Environment

See `environment.yml`. Windows note: the analysis writes deep output paths; on Windows (where
long paths are off) `ensure_x_drive.sh` maps a short drive `X:` to this folder and is sourced
automatically. It is a no-op on Linux/macOS.
