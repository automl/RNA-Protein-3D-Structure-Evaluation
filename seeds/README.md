# Seed and model variability

The paper compares the five AF3 models of three seeds per post-cutoff protein–RNA target. The published seed
databases (`other_analyses/seeds/protein_rna_seed{1,2,3}.db`) were built in four steps, because the evaluation
pipeline itself only knows one model per target:

1. **One folder per seed.** `distribute_model_files.py` moves the model 1–4 files of every AF3 output folder into
   `seed<N>/af/` (model 0 is evaluated with the main sets).
2. **Evaluate model 0** of each seed with the normal pipeline (`evaluation/mainExecution.py`), which creates the
   database with one row per target.
3. **Add rows for models 1–4.** `populate_models.py` clones the model 0 rows, so the later steps, which update rows
   by file name, find a row for every model.
4. **Confidence and accuracy of all models.** Run `af3metrics_summary_fallback.py <seed folder>/af` (instead of
   `evaluation/af3Metrics_new.py`, which stops at the first cloned row) and then `evaluation/extractEvalMetrics.py`
   for the folder.

`seed_model_comparison_figure.py` draws the seed figure of the paper from the three databases.
