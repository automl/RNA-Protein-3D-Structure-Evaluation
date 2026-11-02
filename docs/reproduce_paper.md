# Reproducing the paper and recomputing the metrics

This is the reference for the two paper-oriented uses of the repository: regenerating all figures,
tables and numbers from the published databases, and recomputing every metric from the structures.
For evaluating your own predictions see the top-level [`README.md`](../README.md) and
[`evaluate_your_own_predictions.md`](evaluate_your_own_predictions.md).

The code and data belong to the paper *AlphaFold 3 Prediction Accuracy and Reliability across RNA
Structures and Protein–RNA Interfaces* (Tennie, Uhl, Franke, Backofen and Runge). The evaluated
structures, the AlphaFold 3 predictions and the databases with all computed metrics are archived on
Zenodo ([doi:10.5281/zenodo.23455210](https://doi.org/10.5281/zenodo.23455210)); the code downloads
them for you.

To reproduce the exact paper version, check it out first:

```bash
git clone https://github.com/automl/RNA-Protein-3D-Structure-Evaluation.git
cd RNA-Protein-3D-Structure-Evaluation
git checkout v1.0.1
```

## Installation

### For reproducing the paper

A single environment, `af3_eval`, holds everything the analyses and figures need:

```bash
conda env create -f environment.yml
conda activate af3_eval
```

This is the same command on every system: macOS (Intel and Apple Silicon), Linux and Windows. The
scripts that tie the steps together are shell scripts; on Windows, run them from Git Bash (installed
with [Git for Windows](https://gitforwindows.org/)) or from WSL. The Python steps also run directly
from the Anaconda Prompt.

### For the full evaluation pipeline

The pipeline uses two environments. `RNABench` holds the parsers, the metric scripts and the RNA tools
(ViennaRNA, forgi, PyMOL, FreeSASA, ...). `ost` holds OpenStructure, which computes the all-atom LDDT
and the interface scores; it is kept separate because of its pinned dependencies. The pipeline calls it
by itself, so you only ever activate `RNABench`. The installer creates both environments, builds the
structure aligner US-align for your machine and registers the repository with both environments:

```bash
bash install.sh
```

On macOS the installer uses the exact package builds used during development (`env/intel/`; Apple
Silicon Macs run the same Intel builds through Rosetta, `env/silicon/`). On Linux it uses `env/linux/`,
which has the same versions of every package the scripts use; the full re-run of the paper described
below was done with these environments. On Windows, use WSL2 with Ubuntu and run the Linux installer
there: OpenStructure and several RNA tools are not built for Windows. [`env/README.md`](../env/README.md)
explains how to create the environments by hand.

## Data

`fetch_data.sh` downloads the data from Zenodo, checks every file against its published checksum and
unpacks it where the scripts expect it:

```bash
bash fetch_data.sh                  # databases and analysis inputs, about 100 MB
bash fetch_data.sh --structures     # in addition the structures, about 14 GB
```

On Windows without a shell, `python utils/fetch_zenodo.py` does the same. If you downloaded the files
from Zenodo yourself, `bash fetch_data.sh --local <folder>` takes them from that folder instead.

The code is tied to one version of the Zenodo record (record 23870599, set in `utils/fetch_zenodo.py`).
The record contains:

- `all_databases.zip`: the SQLite databases from which the paper was computed, one per prediction set;
- the structures (`a2021_post_cutoff.zip`, `b2021_pre_cutoff.zip`, `2026.zip`, `other_analyses.zip` and
  the two ablation archives): experimental structures, AF3 models with their confidence files and DSSR
  annotations;
- `analysis_inputs.zip`: inputs of the interface analyses that are not stored in the databases, such as
  the OpenStructure comparison files, the residue mappings used for the interface fallback, the
  InterProScan domain annotations and intermediate tables;
- `noMSA.zip`: the databases of the MSA ablation runs, used by the RNA-only subtree of the decision tree
  and the regression analysis;
- `no_msa_confidences.zip`: the AF3 confidence files (PAE, pTM, pLDDT) of the post-cutoff single-chain
  RNAs predicted without the RNA alignment, which the RNA-only part of the decision tree reads
  (downloaded with `--structures`);
- `rerun_2026-10.zip`: the databases of our complete re-run of the evaluation (see Software versions
  below).

The record's README describes every database and column.

## Reproducing the paper

With the environment active and the data downloaded, one command regenerates everything:

```bash
bash reproduce_all.sh
```

It first runs the regression and variance-partition analyses, then all figure scripts, and finally
collects the figures in `figures_final/`, under the file names the manuscript uses (each as PNG and PDF,
most also as SVG and EPS), and the tables in `figures_final/tables/`. At the end it checks that every
figure was produced and reports any that is missing. A complete run takes about 45 minutes (measured on
8 CPU cores, with a fresh environment and only the downloaded data). The seven appendix figures that
show renders of single example structures (`si_ex_*`) are illustrations of hand-picked complexes, partly
composed by hand in PyMOL, and are not regenerated; their scripts are in
`publication_figures/si_examples/`.

The regression results can also be explored in a notebook, which runs the same analysis with one
command and shows the main tables and Figures:

```bash
jupyter lab notebooks/regression_analysis.ipynb
```

All cross-validation uses a fixed seed and all configurations are written to disk, so a rerun gives the
same numbers.

## Recomputing the metrics from the structures

This repeats the evaluation that produced the published databases. Download the structures first
(`bash fetch_data.sh --structures`). Each prediction set lives in its own folder,
`data/all_data/<group>/<set>/` with the experimental structures in `pdb/`, the AF3 models in `af/` and
the DSSR annotations in `dssr/`. For one set, the evaluation runs in three steps:

1. **The evaluation pipeline** (`evaluation/mainExecution.py`) parses both structures of every complex
   into a database and computes the structural and interface metrics. Which folder and database it works
   on is set in `database/config.json`. A first pass (`-dssr`) computes the structural scores, a second
   pass (`+dssr --db-update`) adds the base-pair and RNA-conformation analyses from the DSSR files and
   the sequence and experimental descriptors.
2. **The interface analyses** (`bash evaluation/run_postchain.sh`, protein–RNA sets) place every
   predicted RNA relative to its protein partner and score the interfaces OpenStructure could not map on
   the exact residue mapping.
3. **The paper's analyses** then run on the resulting databases as described above.

Two sets need an extra step. The ion sets were predicted with the deposited ions, which AF3 names with
two-letter chain identifiers once there are more than 26 of them; `dataPrep/rename_ion_chains.py` gives
them single-letter identifiers before the evaluation. The seed sets hold five models per target;
[`seeds/README.md`](../seeds/README.md) describes how they are evaluated.

[`docs/pipeline_reference.md`](pipeline_reference.md) documents the options of the pipeline, the expected
file names and which step runs in which environment. The evaluation of one set takes from minutes to
about a day for the 923 pre-cutoff protein–RNA complexes; large sets can be split into parts and merged
afterwards.

### Interface scores, OpenStructure and the SWISS-MODEL key

The per-interface DockQ and QS scores of the protein–RNA complexes come from OpenStructure's
`compare-structures`. The pipeline runs it in one of two ways, without any additional setup:

- **Without a key** (the default), it runs locally in the `ost` environment, with the OpenStructure
  version of `env/` (2.12; 2.11.1 in tag `v1.0.1`).
- **With a SWISS-MODEL API key**, it submits the structures to the SWISS-MODEL OpenStructure service, as
  was done for the published interface scores. A key is free with a
  [SWISS-MODEL account](https://swissmodel.expasy.org/account). Put it in the environment variable
  `SWISSMODEL_OST_TOKEN` (or pass `--token`); if neither is set, the pipeline asks for it once at the
  start, and pressing Enter scores locally.

Both ways run the same comparison. The difference is the OpenStructure version, explained next.

### Software versions

In the databases, TM-scores were computed with US-align (version 20241108), RMSDs with PyMOL `align`,
and the all-atom LDDT with OpenStructure 2.11.1, the version pinned in tag `v1.0.1` (later versions of the
repository install 2.12, which maps short RNA chains better). The per-interface
scores of the protein–RNA complexes were obtained from the SWISS-MODEL OpenStructure service at the time
of the evaluation; the service now runs OpenStructure 2.12 at the time this README was produced. The two
versions can assign the copies of a repeated RNA chain (poly-U strands, for example) to different
experimental chains, which changes the score of a small number of interfaces. OpenStructure 2.11.1 also
cannot map some short or poly-U RNA chains at all (it drops them from its chain mapping or fails on the
whole complex); 2.12 maps most of them. For the complexes OpenStructure cannot map, the interface scores
come from the exact-residue-mapping fallback (`evaluation/interface_own_dockq.py`, run by
`run_postchain.sh`).

A recomputation therefore reproduces the published databases closely but not to the last digit; to come
closest, use the environments of tag `v1.0.1` (OpenStructure 2.11.1). In our complete re-run (October 2026, fresh environments, every metric recomputed from
the archived structures) the structure-level values agree with the published ones, and individual
interface scores of repeated RNAs differ; this moves some percentages in the interface-placement tables
by a few points and changes no conclusion of the paper. The re-run is archived on Zenodo next to the
published databases (`rerun_2026-10.zip`, with a number-by-number comparison to the paper). The published
databases are the reference: the paper was computed from them, and they are what `reproduce_all.sh` uses.

### Known issues in the published databases

- In `a2021_post_cutoff/protein_rna_with_ions.db`, the whole-complex `Complex_RMSD` values stem from an
  earlier version of the evaluation (two decimals, e.g. 32.76 Å for 7W0E). The re-run gives 13.45 Å
  (without ions) against 13.25 Å (with ions): supplying ions does not make the complex RMSD worse.
- In `b2021_pre_cutoff/rna_singlechain_with_ions.db`, `RNA_LDDT` is too low for most entries (for example
  0.41 for 1Q9A at an RMSD of 0.44 Å); the re-run gives values consistent with the RMSD and TM-score. The
  paper does not use this database.
- The single-chain RNA ion baseline (`a2021_post_cutoff/rna_singlechain_ion_baseline.db`) holds 63
  entries, the re-run 66.
