#!/bin/bash
# =============================================================================
# capabilities.sh -- detect which databases are present and export HAS_* flags so
# the reproduction drivers run only the figure/analysis groups their inputs support
# (and assert only the figures those groups produce). Source it, do not execute it:
#
#     source "$(dirname "$0")/capabilities.sh"     # from the repo root
#     source "$(dirname "$0")/../capabilities.sh"  # from regression_analysis/
#
# Full DB family present  -> every flag 1 -> the complete paper figure set.
# Only database/rbpDatabase.db present -> HAS_RBP=1, the rest 0 -> the single-DB RBP
# figures only (no pre/post, ion, MSA-ablation or RNAcompete groups).
#
# The caller may pre-set CAP_ROOT to the repo root; otherwise it is inferred from
# this file's location so the flags are correct regardless of the caller's cwd.
# =============================================================================

# Repo root = the directory holding this file (works whether sourced from the root
# or from regression_analysis/). BASH_SOURCE[0] is capabilities.sh itself.
CAP_ROOT="${CAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- DB locations (repo-root-relative) ---------------------------------------------
_DB_MAIN="$CAP_ROOT/database/rbpDatabase.db"
_FIG_PP="$CAP_ROOT/data/compareDatabases/post_pre"          # figure-side pre/post source
_RA_PP="$CAP_ROOT/regression_analysis/data/post_pre"        # regression-side pre/post source
_IONS="$CAP_ROOT/data/compare_ions"
_NOMSA="$CAP_ROOT/data/compareDatabases/noMSA"
_RNC="$CAP_ROOT/data/rnacompete"

# --- helpers -----------------------------------------------------------------------
cap_have()     { [ -f "$1" ]; }                                   # single file
cap_have_all() { for f in "$@"; do [ -f "$f" ] || return 1; done; return 0; }  # every file
cap_glob()     { compgen -G "$1" >/dev/null 2>&1; }               # any match for a glob

# --- flags -------------------------------------------------------------------------
# Regression-side pre/post source DBs (drive Fig1/Fig2, the regressions, the Shapley
# variance partition, and the R1 / pre-post / dataset-overview SI analyses).
HAS_POSTPRE_PR=0
cap_have_all "$_RA_PP/a2021_b2021_protein_rna/a2021.db" \
             "$_RA_PP/a2021_b2021_protein_rna/b2021.db" && HAS_POSTPRE_PR=1
HAS_POSTPRE_RNA=0
cap_have_all "$_RA_PP/a2021_b2021_rna_only/a2021_rna.db" \
             "$_RA_PP/a2021_b2021_rna_only/b2021_rna.db" && HAS_POSTPRE_RNA=1
HAS_RA_POSTPRE=0
[ "$HAS_POSTPRE_PR" = 1 ] && [ "$HAS_POSTPRE_RNA" = 1 ] && HAS_RA_POSTPRE=1

# Figure-side pre/post source DBs (some publication figures read data/ directly rather
# than the regression split outputs).
HAS_FIG_POSTPRE_PR=0
cap_have_all "$_FIG_PP/a2021_b2021_protein_rna/a2021.db" \
             "$_FIG_PP/a2021_b2021_protein_rna/b2021.db" && HAS_FIG_POSTPRE_PR=1
HAS_FIG_POSTPRE_RNA=0
cap_have_all "$_FIG_PP/a2021_b2021_rna_only/a2021_rna.db" \
             "$_FIG_PP/a2021_b2021_rna_only/b2021_rna.db" && HAS_FIG_POSTPRE_RNA=1

# The single a2021 protein-RNA DB behind the RBP interface / motif / pH / bond / domain
# / secondary-structure / RNA-family publication figures.
HAS_RBP=0
cap_have "$_DB_MAIN" && HAS_RBP=1

# Ion-input comparison DBs (compare_ion_input + ion_feature_scatter figures).
HAS_IONS=0
cap_have_all "$_IONS/a2021_protein_rna.db" "$_IONS/a2021_protein_rna_ions.db" && HAS_IONS=1

# MSA / template ablation DBs (msa_temp_influence + msa_template figures).
HAS_NOMSA=0
cap_glob "$_NOMSA/single_protANDRNA_template_remove_experiment/*.db" && HAS_NOMSA=1

# RNAcompete probe DBs (combined_rnacompete + pub_competition_outcomes_family figures).
HAS_RNC=0
cap_have_all "$_RNC/rnaCompete_v4.db" "$_RNC/rnaCompete_v5_MINMAX.db" && HAS_RNC=1

# Compound flags for figures that read more than one dataset (see the figure->DB map):
#  - post_pre_comparison_figure.py needs BOTH figure-side pre/post sets.
#  - rna_family_publication_figure.py loads rbpDatabase.db AND compare_ions/a2021_rna_ONLY.db.
#  - rna_accuracy_feature_importance_figure.py's RNA-only panel loads the figure-side
#    rna-only pre/post DB after the protein-RNA panel (so it needs RBP too).
HAS_FIG_POSTPRE=0
[ "$HAS_FIG_POSTPRE_PR" = 1 ] && [ "$HAS_FIG_POSTPRE_RNA" = 1 ] && HAS_FIG_POSTPRE=1
HAS_RBP_IONS=0
[ "$HAS_RBP" = 1 ] && [ "$HAS_IONS" = 1 ] && HAS_RBP_IONS=1
HAS_RBP_FIGRNA=0
[ "$HAS_RBP" = 1 ] && [ "$HAS_FIG_POSTPRE_RNA" = 1 ] && HAS_RBP_FIGRNA=1

export HAS_POSTPRE_PR HAS_POSTPRE_RNA HAS_RA_POSTPRE \
       HAS_FIG_POSTPRE_PR HAS_FIG_POSTPRE_RNA HAS_FIG_POSTPRE \
       HAS_RBP HAS_IONS HAS_NOMSA HAS_RNC \
       HAS_RBP_IONS HAS_RBP_FIGRNA

# --- run only a gated step; log + skip when its capability flag is 0 ----------------
# usage: cap_gate <FLAG_NAME> <human label> <command...>
cap_gate() {
  local flag_name="$1" label="$2"; shift 2
  if [ "${!flag_name}" = "1" ]; then
    "$@"
  else
    echo "### SKIP ${label} (needs ${flag_name})"
  fi
}

cap_summary() {
  echo "[capabilities] DB detection under $CAP_ROOT:"
  local f
  for f in HAS_POSTPRE_PR HAS_POSTPRE_RNA HAS_RA_POSTPRE \
           HAS_FIG_POSTPRE_PR HAS_FIG_POSTPRE_RNA HAS_FIG_POSTPRE \
           HAS_RBP HAS_IONS HAS_NOMSA HAS_RNC \
           HAS_RBP_IONS HAS_RBP_FIGRNA; do
    printf '  %-20s = %s\n' "$f" "${!f}"
  done
}
