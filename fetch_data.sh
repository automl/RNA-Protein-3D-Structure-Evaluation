#!/bin/bash
# Download the published databases and analysis inputs from Zenodo and put them where the code expects them.
# The work is done by utils/fetch_zenodo.py (which also runs on Windows: python utils/fetch_zenodo.py).
#
#   bash fetch_data.sh                  # databases and analysis inputs (about 100 MB): enough to reproduce the paper
#   bash fetch_data.sh --structures     # also the structures (about 14 GB): for the full recomputation
#   bash fetch_data.sh --local DIR      # use files you already downloaded from Zenodo into DIR
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"
"$PY" utils/fetch_zenodo.py "$@"
