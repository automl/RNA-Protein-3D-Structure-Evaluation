# Evaluate RNA and protein–RNA structure predictions

[![DOI](https://zenodo.org/badge/1427994901.svg)](https://doi.org/10.5281/zenodo.23890211)

Code for the paper *AlphaFold 3 Prediction Accuracy and Reliability across RNA Structures and Protein–RNA
Interfaces* (Tennie, Uhl, Franke, Backofen and Runge). [`run_eval.py`](run_eval.py) scores predicted RNA and
protein–RNA structures in mmCIF or PDB format (from AlphaFold 3 or any other method) against experimental
structures with the metrics of the paper and writes them to an SQLite database.

## Install

```bash
bash install.sh                                      # Linux / macOS
powershell -ExecutionPolicy Bypass -File install.ps1 # Windows (WSL2 recommended)
```

This creates the conda environments `RNABench` and `ost` (OpenStructure) and builds US-align.

## Evaluate your predictions

```bash
conda activate RNABench
python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics full
```

- `--pred`: predicted structures as mmCIF (`.cif`) or PDB (`.pdb`, `.ent`), from any method.
- `--gt`: experimental structures as mmCIF or PDB; without it, only the AF3 confidence is read.
- `--metrics`: `quick` (TM-score, RMSD, backbone LDDT), `basic` (+ all-atom LDDT) or `full` (+ interface
  scores and interactions).
- Predictions and experimental structures are paired by their PDB id or a shared part of the file name;
  `--pair-map` takes a CSV for any other naming.
- AF3 confidence files (`*_summary_confidences*.json`, `*_full_data*.json`) next to the predictions add the
  `af3_*` columns. Predictions of other methods need none.

The results are in `run/run.db`, with a CSV per table.

| metric | tool |
|---|---|
| TM-score | US-align |
| RMSD | PyMOL |
| backbone LDDT (C3′ for RNA, Cα for protein) | own implementation |
| all-atom LDDT, iLDDT, DockQ, QS-score, ICS | OpenStructure |
| pTM, ipTM, pLDDT, PAE | AF3 confidence files |

Inputs, options, all columns and how interfaces are scored:
[`docs/evaluate_your_own_predictions.md`](docs/evaluate_your_own_predictions.md).

## Reproduce the paper

The paper version is tag `v1.0.1`. All figures and tables regenerate from the published data:

```bash
git checkout v1.0.1
conda env create -f environment.yml && conda activate af3_eval
bash fetch_data.sh        # databases and analysis inputs from Zenodo
bash reproduce_all.sh     # figures in figures_final/
```

Details, and how to recompute the metrics from the structures: [`docs/reproduce_paper.md`](docs/reproduce_paper.md).

## Citation

Please cite the paper, the code ([doi:10.5281/zenodo.23890212](https://doi.org/10.5281/zenodo.23890212)) and
the data ([doi:10.5281/zenodo.23455210](https://doi.org/10.5281/zenodo.23455210)).

## License

Apache License 2.0 (see `LICENSE`). US-align (Zhang lab, `evaluation/USalign/`) is included under its own
license. DSSR, used for the base-pair annotations, is not included and needs a license from
[x3dna.org](https://x3dna.org); the DSSR annotations of all structures of the paper are in the Zenodo data.
