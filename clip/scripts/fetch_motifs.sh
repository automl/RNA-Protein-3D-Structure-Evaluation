#!/usr/bin/env bash
# Download the four RBP motif atlases into ../motifs/. Idempotent.
set -euo pipefail
mkdir -p "$(dirname "$0")/../motifs"
cd "$(dirname "$0")/../motifs"

# 1. RBPBench catalog (catRAPID_omics v2.1 ext) -- the only CLIP-provenance-annotated set
[ -f rbpbench.tar.bz2 ] || curl -sSL -o rbpbench.tar.bz2 \
  "https://conda.anaconda.org/bioconda/noarch/rbpbench-1.0.6-pyhdfd78af_0.tar.bz2"
tar -xjf rbpbench.tar.bz2 -C . --strip-components=3 \
  site-packages/rbpbench/content/catRAPID_omics_v2.1_6plus_motifs.weak_rounded.ext.meme \
  site-packages/rbpbench/content/Ray2013_rbp_Homo_sapiens.dna_encoded.meme \
  site-packages/rbpbench/content/rbp_motif2name_ids.1.txt

# 2. CISBP-RNA human (shipped in the RBPBench GitHub test dir)
[ -f cisbp_rna_human.meme ] || curl -sSL -o cisbp_rna_human.meme \
  "https://raw.githubusercontent.com/michauhl/RBPBench/main/test/cisbp_rna_human_rbp_motifs.meme"

# 3. ATtRACT
[ -f ATtRACT.zip ] || curl -sSL -o ATtRACT.zip "https://attract.cnic.es/attract/static/ATtRACT.zip"
mkdir -p attract && unzip -oq ATtRACT.zip -d attract

# 4. RBPDB human matrices
[ -f RBPDB_matrices_human.zip ] || curl -sSL -o RBPDB_matrices_human.zip \
  "http://rbpdb.ccbr.utoronto.ca/downloads/matrices_human.zip"
mkdir -p rbpdb && unzip -oq RBPDB_matrices_human.zip -d rbpdb
echo "motif atlases ready"
