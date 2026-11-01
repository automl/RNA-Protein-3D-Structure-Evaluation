"""
Write the exact-residue-mapped AAmatch_score / RNAmatch_score (and the per-interface columns
AAmatch_per_interface / RNAmatch_per_interface, one {"protein:RNA": score} entry per pair of
ChainIDpairList_proteinRNA) into one or more protein-RNA databases, without running the rest of
the pH-correlation analysis (that is `python evaluation/phCorrelation.py all db_update`).

Usage (RNABench env):
  python evaluation/update_exact_match_scores.py database/rbpDatabase.db \
      data/compareDatabases/post_pre/a2021_b2021_protein_rna/a2021.db \
      data/compareDatabases/post_pre/a2021_b2021_protein_rna/b2021.db
"""

import os
import sqlite3
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.startConfig import StartConfig
from evaluation.phCorrelation import PHCorrelationAnalyzer


def update(db_path):
    analyzer = PHCorrelationAnalyzer.__new__(PHCorrelationAnalyzer)   # scoring only, no plots
    analyzer.config = StartConfig()
    analyzer.connection = sqlite3.connect(db_path)
    try:
        print(f"\n== {db_path}")
        analyzer.update_aa_match_scores()
        analyzer.update_rna_match_scores()
    finally:
        analyzer.connection.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        update(path)
