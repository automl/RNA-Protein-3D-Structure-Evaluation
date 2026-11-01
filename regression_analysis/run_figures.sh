#!/bin/bash
# =============================================================================
# run_figures.sh -- regenerate the publication figures from the wired databases
# and collect them into figures_final/.
#
# REQUIRES the collaborator support packages (not shipped in this repo):
#   plots/        (plots.publication_figure: PublicationFigure, style_axis, PUB_RC)
#   evaluation/   (evaluationMetrics.RNAMetrics, phCorrelation, peg_analysis_plots)
#   database/config.json  (real db_path, e.g. database/rbpDatabase.db)
#
# The figure databases are already placed by the repo:
#   data/compareDatabases/post_pre/         (R1 pre/post-cutoff)
#   data/compareDatabases/noMSA/...         (R3 MSA/template ablation)
#   data/compare_ions/                      (R5 ion input)
#   database/rbpDatabase.db                 (R3 family, R4 classification/motif, SI)
# =============================================================================
set +e
cd "$(dirname "$0")"

if [ ! -d plots ] || [ ! -d evaluation ]; then
  echo "[run_figures] Missing collaborator 'plots/' and/or 'evaluation/' packages."
  echo "              Add them (+ database/config.json), then re-run this script."
  exit 1
fi

PY="${PY:-python}"
OUT=figures_final
mkdir -p "$OUT"
FIGDIR=publication_figures

run() {  # run <script> ; collect any produced png/pdf
  echo "### $1"
  $PY "$FIGDIR/$1"
}

# Figures that read the databases directly (no rbpDatabase.db needed):
run post_pre_comparison_figure.py          # R1  pre/post-cutoff gap
run msa_temp_influence.py                  # R3  MSA/template ablation (2^4 factorial)
run compare_ion_input_figure.py            # R5  ion input
run rna_family_publication_figure.py       # R3  RNA-family heatmap

# Figures that go through database/config.json -> rbpDatabase.db + evaluation/:
run pub_fig1_classification_framework.py   # R4  fold x interface classification
run motif_vs_interface_figure.py           # R4  right-residues-wrong-geometry
run interface_vs_complex_metrics.py        # R4  per-interface vs whole-complex
run ph_peg_composite.py                    # R5  pH / buffer / PEG
run ph_composite.py                        # R5  pH composite
run rna_secondary_structure_figure.py      # SI  secondary structure
run rna_descriptors_figure.py              # SI  RNA descriptors
run bond_metrics_figure.py                 # SI  backbone bond geometry
run domain_publication_figure.py           # SI  protein-domain context

# collect
find "$FIGDIR" -maxdepth 3 -type f \( -name '*.png' -o -name '*.pdf' \) -newer "$OUT" \
  -exec cp {} "$OUT"/ \; 2>/dev/null
echo "Figures collected into $OUT/"
