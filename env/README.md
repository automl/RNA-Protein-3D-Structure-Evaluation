# Conda environments

Two conda environments are used by this project:

- **RNABench** — the main pipeline (parsing, `extractEvalMetrics`, plotting, DB scripts).
- **ost** — the OpenStructure-based scoring scripts (`ost_extractAllAtomLDDT.py`,
  `ost_calcRNA_AllChains.py`, `ost_RBPinterfaces.py`). `mainExecution.py` runs these via
  `conda run --no-capture-output -n ost ...`, so the two environments stay separate.

Both environments are **osx-64 (macOS Intel)**. There is no native Apple-Silicon (arm64)
build: RNABench pins Intel-only MKL (`blas=1.0=mkl`) and pulls PyMOL from the `schrodinger`
channel, neither of which ships arm64. Apple-Silicon Macs therefore run the **same Intel
builds under Rosetta** (`CONDA_SUBDIR=osx-64`), which Miniconda runs transparently.

## Which folder to use

| Folder | Use on | Pinning |
|--------|--------|---------|
| `intel/`   | **Intel Macs (osx-64)** | exact **version + build** hashes — bit-for-bit reproduction |
| `silicon/` | **Apple-Silicon Macs** (via Rosetta) | exact **versions**, build hashes dropped so conda re-solves per platform |
| `linux/`   | **Linux**, and Windows via WSL2 | the same versions where Linux builds exist; the few substitutions are listed at the top of each file |

The `silicon/` files are the old `*.cross.yml` files. `silicon/environment-rnabench.yml`
also has `rigimol=1.3` removed (a commercial incentive-PyMOL morphing tool that is no longer
downloadable; not used by any script here).

## Recreate

First time only — accept the Anaconda channel Terms of Service:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Apple-Silicon Mac (M1–M5, via Rosetta)

```bash
CONDA_SUBDIR=osx-64 conda env create -f env/silicon/environment-rnabench.yml
CONDA_SUBDIR=osx-64 conda env create -f env/silicon/environment-ost.yml
```

Then pin each env to osx-64 so later `conda install`s into it stay Intel:

```bash
conda activate RNABench && conda config --env --set subdir osx-64 && conda deactivate
conda activate ost      && conda config --env --set subdir osx-64 && conda deactivate
```

### Intel Mac (osx-64, exact)

```bash
conda env create -f env/intel/environment-rnabench.yml
conda env create -f env/intel/environment-ost.yml
```

### Linux

Use the `linux/` files. The macOS files do not solve on `linux-64` (macOS-only libraries, the licensed
Schrödinger PyMOL build); the Linux files keep the versions of every package the scripts use and are the
environments the full recomputation was verified with (October 2026):

```bash
conda env create -f env/linux/environment-rnabench.yml
conda env create -f env/linux/environment-ost.yml
```

### Windows

Use **WSL2** (Ubuntu) and follow the Linux instructions (`bash install.sh` inside WSL2).
Native Windows cannot solve these environments: `ost` (`openstructure`) and several
RNABench packages (`forgi`, `cd-hit`, `viennarna`, `rnastructure`, `bellmans-gapc`,
`apbs`) come from bioconda, which publishes no Windows builds. `install.ps1` still tries
the `silicon/` files but will warn and fail at that step.

## Run

The repo root is registered in both envs by `install.sh` / `install.ps1`
(`python env/register_repo_path.py` in each env), so no `PYTHONPATH` is needed, e.g.:

```bash
conda activate ost
python evaluation/ost_calcRNA_AllChains.py --lddt
```

## Verify

```bash
conda run -n RNABench python -c "import matplotlib, numpy, scipy, pandas, Bio, pymol; print('RNABench OK')"
conda run -n ost python -c "import ost; print('ost OK')"
```

## Notes

- If an exact version pin fails to resolve on your platform, relax it (drop the `=version`
  suffix) or install the key package by name — `openstructure` (bioconda) for `ost`, and the
  packages under the `pip:` section for RNABench.
- `openstructure` comes from the **bioconda** channel; keep `bioconda` and `conda-forge` in
  the channel list.
- The `silicon/` files still include the full `pip:` section, so nothing installed via pip is
  lost.
