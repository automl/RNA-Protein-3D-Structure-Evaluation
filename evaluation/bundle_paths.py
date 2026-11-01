"""
Published inputs of the analyses (Zenodo bundle).

Structures: data/all_data/<set>/<system>/{pdb,af,lddt,dssr}  (reference mmCIF, AF3 model, OpenStructure JSON, DSSR)
Databases:  data/all_data_zip/all_databases/<set>/<system>.db

Older working copies elsewhere in data/ (compareDatabases/post_pre, a2021*, b2021*, seeds, 2026) are not read.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DATA = os.path.join(_REPO_ROOT, "data", "all_data")
ALL_DB = os.path.join(_REPO_ROOT, "data", "all_data_zip", "all_databases")

PERIOD_SET = {"post": "a2021_post_cutoff", "pre": "b2021_pre_cutoff", "2026": "2026"}
PROTEIN_RNA_DIR = {p: os.path.join(ALL_DATA, s, "protein_rna") for p, s in PERIOD_SET.items()}
PROTEIN_RNA_DB = {p: os.path.join(ALL_DB, PERIOD_SET[p], "protein_rna.db") for p in ("post", "pre")}
Y2026_DB = os.path.join(ALL_DB, "2026", "protein_rna.db")
IONS_DIR = os.path.join(ALL_DATA, "a2021_post_cutoff", "protein_rna_with_ions")
IONS_DB = os.path.join(ALL_DB, "a2021_post_cutoff", "protein_rna_with_ions.db")
SEEDS_DIR = os.path.join(ALL_DATA, "other_analyses", "seeds")
SEED_DB = {s: os.path.join(ALL_DB, "other_analyses", "seeds", f"protein_rna_seed{s}.db") for s in (1, 2, 3)}


def seed_dir(seed):
    """Structure folder of one seed (seed 1 model 0 is the default prediction, also in PROTEIN_RNA_DIR['post'])."""
    return os.path.join(SEEDS_DIR, f"protein_rna_seed{seed}")


def ost_binary():
    """The OpenStructure command-line tool: $OST_BIN if set, else `ost` on the PATH, else the conda environment
    named `ost` next to the running one (or under the conda installation)."""
    import shutil
    import sys
    if os.environ.get("OST_BIN"):
        return os.environ["OST_BIN"]
    if shutil.which("ost"):
        return shutil.which("ost")
    candidates = [os.path.join(os.path.dirname(sys.prefix), "ost", "bin", "ost")]
    if os.environ.get("CONDA_EXE"):
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.environ["CONDA_EXE"])), "envs", "ost", "bin", "ost"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "ost"
