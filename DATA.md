# Where the data goes

The databases and structures are not part of the repository; `bash fetch_data.sh` (or `python utils/fetch_zenodo.py`)
downloads them from Zenodo ([doi:10.5281/zenodo.23455210](https://doi.org/10.5281/zenodo.23455210), version pinned in
`utils/fetch_zenodo.py`) and puts them in place. This page lists what ends up where, for anyone who wants to place
the files by hand.

## The published layout

The newer analysis scripts read the archives in their published layout (`evaluation/bundle_paths.py`):

| Path | From | Contents |
|---|---|---|
| `data/all_data_zip/all_databases/<group>/<set>.db` | `all_databases.zip` | all databases of the paper |
| `data/all_data/<group>/<set>/{pdb,af,dssr}/` | the structure archives (`--structures`) | experimental structures, AF3 models, DSSR annotations |
| `data/all_data/<group>/protein_rna*/lddt/` | `analysis_inputs.zip` | OpenStructure comparison files of the protein–RNA sets |
| `data/interproscan_cache/` | `analysis_inputs.zip` | InterProScan results per protein sequence (domain annotation) |
| `data/nar_msa_template_ablation_20261007/availability_from_runs.json` | `analysis_inputs.zip` | which inputs each ablation target carries |
| `results/plots_odl/all_chains/` | `analysis_inputs.zip` | intermediate tables of the interface analyses (used when the analyses have not been rerun) |

## Copies under the names of the older scripts

The regression analysis and part of the figure scripts read the same databases under other names.
`fetch_data.sh` places these copies; they are identical to the published files.

| Path | Copy of |
|---|---|
| `regression_analysis/data/post_pre/` and `data/compareDatabases/post_pre/`: `a2021_b2021_protein_rna/{a2021,b2021}.db`, `a2021_b2021_rna_only/{a2021_rna,b2021_rna}.db`, `a2021_b2021_rna_rna/{a2021_rna_rna,b2021_rna_rna}.db` | `protein_rna.db`, `rna_singlechain.db` and `rna_multichain.db` of the post- and pre-cutoff sets |
| `database/rbpDatabase.db` | `a2021_post_cutoff/protein_rna.db` |
| `data/compare_ions/a2021_protein_rna{,_ions}.db`, `a2021_rna_ONLY.db`, `a2021_rna_ions.db` | the ion baseline and ion databases of the post-cutoff set |
| `data/nar_msa_template_ablation_20261007/by_condition/<side>/<condition>/protein_rna/rbpDatabase.db` and `a2021.db` | `ablation_studies/msa_template_two_sided/<side>/<condition>.db` and the post-cutoff protein–RNA database |
| `data/rnacompete/rnaCompete_v4.db`, `rnaCompete_v5_MINMAX.db` | `other_analyses/rnacompete/rnacompete_v4_single_probe.db`, `rnacompete_v5_competition.db` |
| `data/compareDatabases/noMSA/...` | the databases in `noMSA.zip` (earlier MSA ablation runs) |

## Excluded targets

6FQL (post-cutoff) is part of the AF3 training set and 7PMQ (pre-cutoff) was released in 2022; both are removed
from the analyses (`evaluation/excluded_targets.py`).
