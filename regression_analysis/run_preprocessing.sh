#! /bin/bash
# Database preprocessing: parse the paired prediction/experiment SQLite databases
# in data/ into the per-analysis CSVs consumed by the regression scripts.
set -e

# Step 0: split the 6 combined source dbs (data/post_pre/) into the 10 method/era
# split dbs the steps below consume. Future data may arrive combined rather than
# pre-split by experimental method, so this regenerates data/{a2021,b2021,a2021x,
# a2021nonX,b2021x,b2021nonX,a2021_rnaOnly_with_af3,b2021_rnaOnly_with_af3,a2021_rna_rna,b2021_rna_rna}.db
# from data/post_pre/ on every run.
python split_dbs_by_experiment.py

# Pass each step only the split dbs that actually exist, so a partial database family
# (e.g. only the post-cutoff era, or protein-RNA only) still preprocesses what it can
# instead of failing on a missing --dbs path. A step whose dbs are all absent is skipped.
present() { for f in "$@"; do [ -f "$f" ] && printf '%s ' "$f"; done; }
prep_step() {  # $1=label  $2=script  $3=outdir  rest=candidate dbs
  local label="$1" script="$2" out="$3"; shift 3
  local dbs; dbs="$(present "$@")"
  if [ -z "$dbs" ]; then echo "### SKIP preprocessing: $label (no input DBs present)"; return; fi
  echo "### $label"
  # shellcheck disable=SC2086
  python "$script" --dbs $dbs --outdir "$out"
}

# RNA-only single chains
prep_step "RNA-only single chains" preprocess_rna_only_per_chain_metrics.py \
  csvs4analysis_rnaonly_with_af3 \
  data/a2021_rnaOnly_with_af3.db data/b2021_rnaOnly_with_af3.db

# complexes, per-RNA-chain view (multichain + RNA-protein 1v1)
prep_step "complexes per-RNA-chain" preprocess_complex_per_chain_metrics.py \
  csvs4analysis_per_chain_complex \
  data/a2021x.db data/a2021nonX.db data/b2021x.db data/b2021nonX.db

# complexes, complex-level view (with predicted-interface features)
prep_step "complexes complex-level" dbs2analysis_csvs_with_interface.py \
  csvs4analysis_with_interface \
  data/a2021.db data/b2021.db

prep_step "multi-chain RNA per-chain (separate set)" preprocess_rna_rna_per_chain_metrics.py \
  csvs4analysis_rna_rna \
  data/a2021_rna_rna.db data/b2021_rna_rna.db

# merge into the combined "all complexes" / per-chain CSVs (best-effort: tolerate a
# partial set of per-analysis CSVs from the steps above)
python merge_csvs.py || echo "### WARNING: merge_csvs.py failed (partial CSV set?) -- continuing"
