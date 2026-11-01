#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results
./fetch_motifs.sh
python3 fetch_uniprot.py            # both cohorts
python3 build_overlap.py
python3 run_analysis.py
python3 validate_classes.py | tee ../results/class_validation.txt
python3 stats.py | tee ../results/stats_report.txt
python3 null_sensitivity.py         # null without paralogs / similar motifs -> ../results/null_sensitivity.txt
python3 positional.py               # position of the motif hits in the bound RNA -> ../results/positional.tsv
python3 domain_breakdown.py         # by InterPro domain family; needs ../results/positional.tsv
