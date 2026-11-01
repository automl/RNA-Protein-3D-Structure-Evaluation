# Evaluation pipeline reference

Details of `evaluation/mainExecution.py`: configuration, input layout, flags and the steps it runs. The README explains how to use it to recompute the paper's metrics.

## 1. Requirements / setup

Two conda environments are used (kept separate on purpose — OpenStructure has a heavy pinned
dependency stack):

| Env | Used for |
|-----|----------|
| **RNABench** | the main pipeline: parsing, `extractEvalMetrics`, plotting, DB/CSV scripts |
| **ost** | the OpenStructure scoring scripts (`ost_extractAllAtomLDDT.py`, `ost_calcRNA_AllChains.py`, `ost_RBPinterfaces.py`) |

`mainExecution.py` runs the OST scripts automatically via
`conda run --no-capture-output -n ost ...`, so you only ever activate **RNABench** yourself.

Recreate both environments from the specs in [`env/`](env/):

```bash
# macOS Intel (osx-64, as used by the authors) — exact version + build pins
conda env create -f env/intel/environment-rnabench.yml
conda env create -f env/intel/environment-ost.yml

# Apple Silicon — same osx-64 packages, run under Rosetta
CONDA_SUBDIR=osx-64 conda env create -f env/silicon/environment-rnabench.yml
CONDA_SUBDIR=osx-64 conda env create -f env/silicon/environment-ost.yml

# Linux (and Windows via WSL2)
conda env create -f env/linux/environment-rnabench.yml
conda env create -f env/linux/environment-ost.yml
```

`install.sh` picks the right variant automatically.

See [`env/README.md`](env/README.md) for details.

Scripts use package-style imports such as `from database.startConfig import StartConfig`, so the
repo root must be importable. `install.sh` / `install.ps1` register it once in both envs (a `.pth` file
in site-packages), so no `PYTHONPATH` export is needed on macOS, Linux or Windows. Without the
installers, run once per env:

```bash
conda run -n RNABench python env/register_repo_path.py
conda run -n ost      python env/register_repo_path.py
```

An exported `PYTHONPATH` still takes precedence, so remove old exports that point at another checkout.

---

## 2. Configuration

All table names and data/DB paths come from **`database/config.json`** (no paths are passed on
the command line):

```json
{
    "classification": "protein_rna",
    "base_data_path": "/path/to/AF3InterfaceEval/data",
    "db_path": "/path/to/AF3InterfaceEval/database/rbpDatabase.db"
}
```

- **`classification`** — the dataset name. It selects both the data folder
  (`base_data_path/<classification>`) and the DB tables:
  - predicted table → `pred_<classification>` (e.g. `pred_protein_rna`)
  - reference table → `exp_<classification>` (e.g. `exp_protein_rna`)
- **`base_data_path`** — directory holding the per-classification data folders.
- **`db_path`** — SQLite database file (created/updated in place).

Supported classifications include `protein_rna` and `rna_rna`. Some steps are protein–RNA
specific (see the pipeline table below).

---

## 3. Input files — how to save them

For a given `classification`, create a data folder with **`pdb/`** and **`af/`** subfolders:

```
<base_data_path>/<classification>/
├── pdb/        # reference / experimental structures (mmCIF)
├── af/         # predicted structures (mmCIF)
├── aligned/    # (auto-created) predicted structures superposed onto the reference
└── dssr/       # (only for +dssr runs) DSSR outputs
```

**Naming convention** (the code derives the reference id from the predicted filename):

| Folder | Example filename | Rule |
|--------|------------------|------|
| `af/`  | `fold_6nbj_s1_model_0.cif` | AF3-style: the PDB id is the **second `_`-separated token** (`6nbj`). |
| `pdb/` | `6nbj.cif` | Named by the PDB id only, matching the token above. |

Practically:
- Every prediction in `af/` must have a matching reference in `pdb/` whose base name is the
  PDB id embedded in the AF filename. Missing references are skipped with a warning.
- Keep the id casing consistent between the two files (case matters on Linux).
- Files must be **mmCIF** (`.cif`). The `af/` files are the raw AF3 model CIFs; predicted
  metadata (pTM/ipTM/PAE/etc.) is read from the accompanying AF3 `*_data.json` when present.

The `aligned/` folder is produced by the pipeline (needed by `ost_extractAllAtomLDDT`); you do
not create it by hand.

---

## 4. Running the pipeline

Run with **RNABench** active (`mainExecution.py` puts the repo root on the path itself):

```bash
conda activate RNABench
python evaluation/mainExecution.py [flags]
```

The first positional argument selects the DSSR phase; **it defaults to `-dssr`**:

| Flag | Meaning |
|------|---------|
| `-dssr` *(default)* | Parse structures + compute structural/interface metrics + OST scoring. |
| `+dssr` | Second phase: DSSR-based RNA conformation/secondary-structure analysis and plotting (requires a populated `dssr/` folder). |

### `-dssr` options

| Flag | Effect |
|------|--------|
| `-basicEval` | **Minimal run**: `parsePDBCIF2DB` + `parseAFcif2DB` → `extractEvalMetrics` → OST scoring. Skips `extractInteractions`, interface area, binding affinity and AF3 metrics. |
| `--token TOKEN` | SWISS-MODEL API token for `ost_RBPinterfaces` (overrides `SWISSMODEL_OST_TOKEN`). Without a token the interfaces are scored with the local OpenStructure. |
| `-noRBP` | Skip `ost_RBPinterfaces` and suppress the token prompt. |
| `-interface` | Skip `extractInteractions`, `extractInterfaceArea`, `estimateBindingAffinity`. |
| `-af3` | Skip `af3Metrics_new`. |
| `-noDomains` | Skip the InterPro domain annotation (protein–RNA only; needs internet). Implied by `-basicEval`. |

### `+dssr` options

| Flag | Effect |
|------|--------|
| `-addMSA` | Run `addMSA_fromJSON.py`. |
| `--db-update` | Run the DSSR/pH/electrostatics/ligand DB-update steps. |
| `--plot [<+msa/-msa>] <all/single_chain>` | Generate plots with the given filters. |
| `--rbp {+rbptemp,-rbptemp}` | RBP template filter (used with `--plot`). |
| `--allArgs` | Generate plots for all filter combinations. |

### Examples

```bash
# Minimal evaluation; interfaces scored with the local OpenStructure (no token)
python evaluation/mainExecution.py -basicEval

# Minimal evaluation, run RBP interfaces with a token
python evaluation/mainExecution.py -basicEval --token <YOUR_TOKEN>

# Full non-DSSR run (default -dssr), skip AF3 metrics
python evaluation/mainExecution.py -af3

# DSSR phase with MSA processing and all-combination plots
python evaluation/mainExecution.py +dssr -addMSA --allArgs
```

---

## 5. Pipeline steps (which env, when)

| Step | Script | Env | Runs when |
|------|--------|-----|-----------|
| Parse reference CIFs | `database/parsePDBCIF2DB.py` | RNABench | `-dssr` |
| Parse predicted CIFs | `database/parseAFcif2DB.py` | RNABench | `-dssr` |
| Chain interactions/contacts | `evaluation/extractInteractions.py` | RNABench | `-dssr`, not `-interface`/`-basicEval` |
| Interface area / binding affinity | `extractInterfaceArea.py`, `estimateBindingAffinity.py` | RNABench | `-dssr`, not `-interface`/`-basicEval` |
| MSA depth + template ids (from `folds/` or `*_data.json`) | `utils/addMSA_fromJSON.py`, `database/extractTemplatesAF3.py` | RNABench | `-dssr` |
| InterPro protein domains (via UniProt) | `database/fetchInterProDomains.py` | RNABench | `-dssr` on `pred_protein_rna`, not `-noDomains`/`-basicEval` |
| AF3 confidence metrics | `evaluation/af3Metrics_new.py` | RNABench | `-dssr`, not `-af3`/`-basicEval` |
| Core eval metrics (RMSD/TM/INF, backbone LDDT) | `evaluation/extractEvalMetrics.py` | RNABench | `-dssr` |
| All-atom LDDT | `evaluation/ost_extractAllAtomLDDT.py` | **ost** | `-dssr` (both tables) |
| RNA AllChains LDDT (+ interfaces on protein–RNA) | `evaluation/ost_calcRNA_AllChains.py` | **ost** | `-dssr`; `--lddt` only for `pred_rna_rna` |
| RBP interface DockQ/QS (SWISS-MODEL service with a token, local OpenStructure without) | `evaluation/ost_RBPinterfaces.py` | **ost** | `-dssr` + `pred_protein_rna`, unless `-noRBP` |
| Export tables to CSV | `database/exportDB2CSV.py` | RNABench | always (end of run) |

---

## 6. `ost_RBPinterfaces` and the SWISS-MODEL token

`ost_RBPinterfaces.py` compares the reference and predicted CIFs with OpenStructure's
**compare-structures** action and stores per-interface DockQ / QS / fnat / iRMSD. With a SWISS-MODEL
API token it submits them to the SWISS-MODEL OpenStructure service (as for the published values);
without one it runs the same action locally in the `ost` environment (`--local`).

- Create/regenerate a token at <https://swissmodel.expasy.org/account>.
- Provide it via `--token`, or export `SWISSMODEL_OST_TOKEN`. If neither is set, the pipeline
  prompts once up front (press Enter to score locally).
- The service is **network-bound**: each structure is submitted and then polled every ~10 s until the
  remote job finishes (up to 2 h per structure), so it can appear to "hang" while waiting.
- Only relevant for `pred_protein_rna`; on other classifications it is skipped automatically.

> **Never commit your token.** Keep it in `SWISSMODEL_OST_TOKEN`, not in the repo.

---

## 7. Output

- The SQLite database at `db_path` is populated/updated in place (tables `pred_<classification>`
  and `exp_<classification>`).
- At the end of every run, both tables are exported to CSV via `exportDB2CSV.py`.
- Plots (for `+dssr` runs) are written under the project's plotting output folders.

---

## Repository layout

```
database/      config.json, parsers, DB helpers, CSV export
evaluation/    mainExecution.py, metric + OST scripts, plotting
dataPrep/      structure preparation utilities
structures/    RNA/Protein/DNA DB-object classes
utils/         MSA and misc helpers
env/           conda environment specs (see env/README.md)
data/          per-classification pdb/ and af/ inputs
```
