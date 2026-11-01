# decision_tree: accuracy-calibrated decision framework for AlphaFold3 RNA predictions

Runnable implementation of the decision framework of the paper (Section "A practical,
accuracy-calibrated decision framework", decision-tree figure, Table `tab:decision_accuracy`,
Appendix `app:framework`). It needs no experimental reference structure.

## Calibrated probabilities (the version in the paper)

The tree reports probabilities (`probability.py`; design in `develop_probability.py`, frozen in
`calibration_probability.json`; figure `2025_NAR_AF3_Evaluation/figures2/decision_tree/decision_tree_21.tex`).
Levels for every probability: WEAK < 25% <= MODERATE < 60% <= STRONG. Intra-RNA PAE = mean of AF3's `pae` matrix
(full_data JSON) over the token pairs within one RNA chain; it is derived from AF3's output, not reported by AF3.

| Probability | Model (log-odds z, P = 1/(1+e^-z)) | Post-cutoff (development) | 2026 test |
|---|---|---|---|
| P_fold, longest RNA chain LDDT_r > 0.85 | -0.879 - 0.207 intra-RNA PAE + 1.063 familiar sequence + 1.668 familiar fold + 0.978 RBP template + 1.555 (>= 2 RNA chains) | CV AUC 0.853; 10% / 41% / 83% | AUC 0.886; 4% / 33% / 71% |
| P_interface, RNA chain placed correctly | 0.709 - 0.201 chain_pair_pae_min + 1.524 familiar RNA + 0.651 well-folded partner (pTM_p >= 0.84 and pLDDT_r >= 0.78) | CV AUC 0.844; 11% / 41% / 83% | AUC 0.921; 11% / 35% / 96% |
| P_usable, both | P_fold and P_interface joined by a Gaussian copula, correlation 0.66 (fitted out of fold on post-cutoff; 0 = product, 1 = smaller of the two) | 7% / 45% / 78% | AUC 0.910; 3% / 23% / 67% |
| RNA-only, single chain | with alignment: -32.7 + 0.403 pLDDT_r (%); without: 4.67 - 1.03 intra-RNA PAE | dev AUC 0.995 / 0.992 | see `rna_test_2026.txt` |
| RNA-only, RNA complex | with alignment: 1.15 + 2.33 duplex - 0.352 PAE; without: 2.29 + 1.84 duplex - 0.469 PAE | dev AUC 0.978 / 0.953 | AUC 1.00 in all three 2026 sets |

Familiarity needs no experimental structure: familiar sequence = >= 80% identity to an RNA chain of a PDB entry
released before 30 Sep 2021 (`data/pdb_reference/pdb_rna_pre2021.fasta`, built by `pdb_sequence_familiarity.py` from
wwPDB pdb_seqres.txt and an RCSB release-date query); familiar fold = US-align TM >= 0.5 between the AF3 model's
longest RNA chain and a pre-cutoff RNA chain of the benchmark (`predicted_fold_familiarity.py`; < 20 nt = not
familiar). The tool computes both automatically when these references are present. ipTM_r and pLDDT_r were tested as
extra inputs and not kept. Weights and the copula correlation are fitted on post-cutoff targets; 2026 is the test. Refitted 2026-10-27 with the same
procedure and questions after the AF3 model files of 19 post-cutoff complexes were replaced by the published ones
(`data/all_data`; older local copies of some predictions had missing protein copies).
The point-score versions below are kept for reproducibility.

**Which LDDT_r defines "well folded".** As in the paper, the trees use the backbone C3′ LDDT_r
(`RNA_LDDT` of the databases; per RNA chain computed from the structures in `combined.py`). The all-atom LDDT_r of
OpenStructure (`RNA_LDDT_allAtom`, `RNA_LDDT_AllChains`) can be used instead by setting `DT_LDDT=allatom` before
running the design and test scripts (`lddt_labels.py`; rerun the design steps listed in
`evaluation/make_analysis_tables.sh`, since the shipped calibrations are those of the paper). The backbone values of
the 2026 sets and of the multi-chain RNA sets added for the RNA-only subtree are not in the published databases; they
come from the re-run of the Zenodo record and are shipped in `analysis_inputs.zip`
(`results/plots_odl/all_chains/decision_tree/backbone_lddt_supplement.tsv`).

```bash
python -m decision_tree fold_xxx_model_0.cif --rbp-template yes             # p_fold, p_interface, p_usable + levels
python -m decision_tree.pdb_sequence_familiarity        # sequence familiarity reference (pre-cutoff PDB RNAs)
python -m decision_tree.predicted_fold_familiarity      # fold familiarity of the AF3 models (development and 2026)
python -m decision_tree.validate_2026 && python -m decision_tree.develop_v3   # per-target rows the design reads
python -m decision_tree.develop_rna && python -m decision_tree.combined       # RNA-only rows, combined levels
python -m decision_tree.develop_probability           # design on post-cutoff (writes calibration_probability.json)
python -m decision_tree.develop_probability --test    # 2026 test
python -m decision_tree.develop_probability --rna [--test]   # RNA-only subtrees
python -m decision_tree.applications --probability    # RNAcompete v4 / v5 and CLIP atlas (after clip/scripts/run_all.sh)
FIG_RENDER=plain python -m decision_tree.figure_2026  # Fig. decision_tree_2026
```

## Decision tree v3 (protein-RNA; no MSA needed; superseded by the probabilities)

v3 keeps the v2 interface side and replaces the fold side by a score designed on post-cutoff
targets only (`develop_v3.py`) and tested once on PDB entries released in 2026 (`validate_2026.py`,
AlphaFold Server predictions, which carry no RNA alignment). Figure: `tikz/decision_tree_18.tex`;
calibration: `calibration_v3.json`.

| Side | Score | Levels (95% CI) |
|---|---|---|
| Fold (v3) | AF3 RNA confidence (mean intra-RNA PAE <= 4 A: 2, 4-8 A: 1, > 8 A: 0) + 1 familiar RNA sequence (>= 80% identity to a pre-cutoff PDB RNA) + 1 familiar RNA fold (US-align TM >= 0.5) + 1 RBP template used (any protein chain) + 1 if >= 2 RNA chains | 2026 held-out: WEAK (<= 2) 13% [7-24], MODERATE (3) 31% [13-58], STRONG (>= 4) 80% [55-93] well folded; AUC 0.83 vs 0.74 for AF3 confidence alone |
| Interface (v2) | as below | WEAK 36%, STRONG 76% correct (post-cutoff); 2026 AUC 0.875 |

```bash
python -m decision_tree fold_xxx_model_0.cif --familiar-seq no --familiar-tm no --rbp-template yes   # column fold_level_v3
python -m decision_tree.develop_v3            # design on post-cutoff, test on 2026, write calibration_v3.json
python -m decision_tree.validate_2026         # independent 2026 test of all trees
python -m decision_tree.applications --v3     # RNAcompete v5 / CLIP (v4: interface side only, no model files)
```

## RNA-only subtrees (predictions without a protein chain)

RNA-only predictions are scored by AF3's per-residue confidence alone; the measure depends on whether the RNA
alignment was used (`--rna-alignment yes|no`; AlphaFold Server runs use none, so `no`; unknown -> range of levels).
The level is written to the `fold_level_v3` column with the development rate; `subtree` names the path taken.

| Subtree | Branch | WEAK / MODERATE / STRONG |
|---|---|---|
| Single-chain RNA (`calibration_rna_single.json`, `develop_rna.py`) | with alignment | RNA pLDDT <= 0.603 / <= 0.755 / > 0.755 |
| | without alignment | intra-RNA PAE >= 28.4 / >= 12.2 / < 12.2 A |
| RNA complex (`calibration_rna_multi_v2.json`, `develop_rna_multi.py --v2`) | any | duplex in the AF3 model (>= 60% of nucleotides paired with another chain) -> STRONG |
| | with alignment | otherwise intra-RNA PAE (longest RNA chain) >= 13.2 / >= 8.2 / < 8.2 A |
| | without alignment | otherwise >= 15.0 / >= 6.4 / < 6.4 A |

```bash
python -m decision_tree fold_xxx_model_0.cif --rna-alignment no      # RNA-only AlphaFold Server prediction
```

The tool reproduces the development levels of all 225 development predictions (`rna_only.py`). The RNA-only models are chosen and fitted on
the development sets and tested on the 2026 RNAs. The RNA-complex subtree (duplex question, then intra-RNA PAE)
was designed on development data only and tested on the 2026 RNA complexes (`--v2 --validate`). The earlier form
without the duplex question (`calibration_rna_multi.json`, no `--v2`) is kept for reproducibility only.

## Scored framework (v2, superseded)

Each side adds up three kinds of evidence and maps the score to three calibrated levels
(`framework.py`; design and calibration in `calibration.json`, made by `develop.py`):

| Side | Score | Levels (novel targets, held-out, 95% CI) |
|---|---|---|
| Fold | AF3 RNA confidence (mean intra-RNA PAE <= 4 A: 2, 4-8 A: 1, > 8 A: 0) + 1 if the RNA is familiar | WEAK (<= 1) 26% [21-32], MODERATE (2) 59% [48-69], STRONG (3) 67% [57-76] well folded |
| Interface | interface tier (2/1/0) + 1 if familiar - 1 if the partner is not well folded (pTM < 0.84 or RNA pLDDT < 0.78) - 1 for a large RBP domain (dsRBD / RNase III / Dicer) | WEAK (<= 1) 39% [35-44], STRONG (>= 2) 88% correct. Development levels (column `interface_level_dev`): UNRELIABLE (<= 1), CAUTION (2) 89% [80-94] and RELIABLE (3) 88% [80-94]; the upper two did not separate and are merged into STRONG |
| Overall (per RNA chain) | WEAK if the fold or the interface level is WEAK; STRONG if both are STRONG; MODERATE otherwise (`combined.py`, `calibration_combined.json`) | usable (correct interface and LDDT_r > 0.85): WEAK 27%, MODERATE 36%, STRONG 77% (post-cutoff); 2026 14% / 33% / 86% |

Familiar RNA: a pre-cutoff PDB RNA with >= 80% sequence identity or US-align TM-score >= 0.5
(`--familiar yes/no`, or `--reference-fasta` with pre-cutoff PDB RNA sequences for the sequence
test). Unknown familiarity gives a range of levels. The confidence measure and the warning signs
were chosen on pre-cutoff targets only; post-cutoff targets are the held-out test
(post-cutoff AUC: fold 0.75, interface 0.80; AF3 confidence alone 0.71 / 0.72; gated tree 0.60 / 0.67).
Rerun 2026-10-27 on interface labels that include the exact-mapping DockQ where OpenStructure gives none;
the pre-cutoff selection then also keeps the large-domain warning.
Entries 6FQL (post, released 2018) and 7PMQ (pre, released 2022) were removed from the period sets
(`evaluation/excluded_targets.py`).

```bash
python -m decision_tree fold_xxx_model_0.cif --familiar no --domains RRM_dom
python -m decision_tree.chain_pae <af3 folders>   # intra-chain PAE cache used by develop.py
python -m decision_tree.develop                   # design selection (pre) + calibration (post)
```

The gated trees (`tree.py`: as first drawn, `ORDER_DRAWN`; confidence first, `ORDER`) are kept for
reference and still reported as `fold_leaf` / `interface_leaf`.

## What it reports

For every RNA chain of an AF3 model:

| Output | Rule |
|---|---|
| `system_class` | single-chain RNA / RNA-protein 1v1 / RNA-protein multichain (from the chains in the model) |
| `expected_rna_tm` | observed mean per-chain RNA TM-score of AF3 for that class and RNA length; the post-cutoff (novel-target) value unless `--pre-cutoff` |
| `fold_tier` | RNA pLDDT (0-1): good >= 0.775, bad < 0.55, else mid |
| `interface_tier` | protein-RNA only. good: RNA ipTM >= 0.5 and chain-pair PAE <= 8 A; bad: ipTM < 0.25 or PAE > 15 A; else mid |
| `fold_leaf` | decision tree, fold side: AF3 confident (pLDDT_r >= 0.78, global PAE <= 10 A) > deep RNA MSA (> 1,000) > canonical fold (no pseudoknot) > conserved Rfam family; leaves WEAK < LOW-MOD < MODERATE < MOD-HIGH < STRONG |
| `interface_leaf` | decision tree, interface side: AF3 confident (ipTM_r >= 0.5, chain-pair PAE <= 8 A) > interface chains well folded (partner pTM >= 0.84, pLDDT_r >= 0.78) > protein-RNA template > compact RBP domain (not dsRBD / RNase III / Dicer); leaves UNRELIABLE < CAUTION < MODERATE < LIKELY OK < RELIABLE |

The tree questions known before prediction (`--rna-family`, `--msa-depth`, `--pseudoknot`,
`--rbp-template`, `--domains`) are optional; an unknown answer is followed down both branches and
the leaf is reported as a range (e.g. `MODERATE..STRONG`). The tree asks AF3's confidence first:
the order first drawn for the paper (`tree.ORDER_DRAWN`, pre-prediction questions first) ranked
held-out targets close to chance, the confidence-first order (`tree.ORDER`, remaining order
selected on pre-cutoff targets only) does not (`python -m decision_tree.validate_tree`). The TikZ
sources of both trees are in `tikz/` (`decision_tree_16c.tex` = current, `16b` = first drawn).

The tiers are reliability labels for the predicted structure. They do not predict binding
affinity or which RNA a protein prefers. The cutpoints are fixed (`rules.py`) and were
calibrated on the PDB benchmark of the paper; re-derive them if the reference corpus changes.

AF3 values used (as in the paper): `chain_iptm` of the RNA chain and `chain_pair_pae_min`
(protein row, RNA column) from `*_summary_confidences_<k>.json`; mean atom pLDDT of the
RNA chain from `*_full_data_<k>.json`. Without a reference structure the PAE is taken
for the protein chain with the lowest value (the paper used the protein chain that
contacts the RNA in the deposited structure).

## Usage

```bash
# one model, or every *_model_*.cif under a folder (AlphaFold Server or local AF3 output)
python -m decision_tree path/to/fold_xxx_model_0.cif
python -m decision_tree path/to/af_output_folder --tsv labels.tsv
```

```python
from decision_tree import classify_model, classify_values
classify_model("fold_8rn1_s1_model_0.cif")
classify_values(n_protein=1, n_rna=1, length=24, rna_plddt=0.81, rna_iptm=0.62, pae_min=3.1)
```

## Validation and applications (paper SI)

```bash
python -m decision_tree.validate       # held-out validation of the tiers (placement classes, 15 AF3 samples, multi-chain RNA)
python -m decision_tree.validate_tree  # node-by-node validation of the tree and selection of its node order
python -m decision_tree.applications   # tree leaves applied to RNAcompete v4 / v5 and the CLIP motif atlas
python -m decision_tree.figure         # SI figure decision_tree_validation.{png,pdf}
```

Outputs go to `results/plots/all_chains/decision_tree/`. `validate` and `applications` read
the repository databases and AF3 outputs under `data/`; `classify` needs only AF3 output.
