"""
Download the published data from Zenodo and unpack it where the code expects it.

The record is pinned (ZENODO_RECORD below), so the code always gets exactly the files it was released with.
Every file is checked against the MD5 checksum Zenodo publishes for it; files that are already downloaded and
intact are not fetched again.

    python utils/fetch_zenodo.py                    # databases + analysis inputs (about 100 MB), enough for the paper
    python utils/fetch_zenodo.py --structures       # also the structures (about 14 GB), for the full recomputation
    python utils/fetch_zenodo.py --local DIR        # use files already downloaded into DIR instead of the internet

What ends up where:
  data/all_data_zip/all_databases/<set>/<name>.db   the published databases (all_databases.zip)
  data/all_data/<set>/<system>/{pdb,af,dssr,lddt}   structures (--structures) and OpenStructure JSONs
  data/compareDatabases/..., regression_analysis/data/..., data/compare_ions/..., database/rbpDatabase.db,
  data/nar_msa_template_ablation_20261007/..., data/rnacompete/...
                                                    copies of the same databases under the names the figure and
                                                    regression scripts read
  data/interproscan_cache/, results/plots_odl/...   analysis inputs (analysis_inputs.zip)
"""
import argparse
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

ZENODO_RECORD = "23870599"            # pinned version of the data record; see README.md, "Data"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.23455210"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = os.path.join(REPO, "data", "downloads")

DATABASE_FILES = ["all_databases.zip", "noMSA.zip", "analysis_inputs.zip"]
STRUCTURE_FILES = ["a2021_post_cutoff.zip", "b2021_pre_cutoff.zip", "2026.zip", "other_analyses.zip",
                   "ablation_studies_protein_side.zip", "ablation_studies_rna_side.zip", "no_msa_confidences.zip"]


def record_files(record):
    """{file name: (download url, md5)} of the Zenodo record."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        meta = json.load(r)
    out = {}
    for f in meta.get("files", []):
        name = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or f"https://zenodo.org/records/{record}/files/{name}?download=1"
        out[name] = (url, (f.get("checksum") or "").replace("md5:", ""))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(name, files, local):
    """Path of a verified copy of <name> in data/downloads/."""
    os.makedirs(DOWNLOADS, exist_ok=True)
    target = os.path.join(DOWNLOADS, name)
    url, checksum = files.get(name, (None, ""))
    if os.path.exists(target) and (not checksum or md5(target) == checksum):
        print(f"  {name}: already downloaded")
        return target
    if local:
        src = os.path.join(local, name)
        if not os.path.exists(src):
            sys.exit(f"{name} is not in {local}")
        shutil.copyfile(src, target)
    else:
        if url is None:
            sys.exit(f"{name} is not part of Zenodo record {ZENODO_RECORD}")
        print(f"  {name}: downloading ...", flush=True)
        tmp = target + ".part"
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 22)
        os.replace(tmp, target)
    if checksum and md5(target) != checksum:
        sys.exit(f"{name}: MD5 checksum does not match the Zenodo record; delete {target} and try again")
    return target


def _skip(name):
    return name.startswith("__MACOSX") or "/__MACOSX" in name or name.endswith(".DS_Store") or name.endswith("/")


def extract(zf, dest):
    """Extract a zip into dest; nested zips are unpacked next to where they sit in the archive."""
    n = 0
    for info in zf.infolist():
        if _skip(info.filename):
            continue
        if info.filename.endswith(".zip"):
            with zf.open(info) as fh:
                n += extract(zipfile.ZipFile(io.BytesIO(fh.read())), os.path.join(dest, os.path.dirname(info.filename)))
            continue
        target = os.path.join(dest, info.filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        n += 1
    return n


def copy(src, dst):
    if not os.path.exists(src):
        print(f"  (not in the bundle: {os.path.relpath(src, REPO)})")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def place_databases(db):
    """Copies of the published databases under the names the analysis scripts read."""
    post, pre = os.path.join(db, "a2021_post_cutoff"), os.path.join(db, "b2021_pre_cutoff")
    for root in (os.path.join(REPO, "regression_analysis", "data", "post_pre"),
                 os.path.join(REPO, "data", "compareDatabases", "post_pre")):
        copy(f"{post}/protein_rna.db", f"{root}/a2021_b2021_protein_rna/a2021.db")
        copy(f"{pre}/protein_rna.db", f"{root}/a2021_b2021_protein_rna/b2021.db")
        copy(f"{post}/rna_singlechain.db", f"{root}/a2021_b2021_rna_only/a2021_rna.db")
        copy(f"{pre}/rna_singlechain.db", f"{root}/a2021_b2021_rna_only/b2021_rna.db")
        copy(f"{post}/rna_multichain.db", f"{root}/a2021_b2021_rna_rna/a2021_rna_rna.db")
        copy(f"{pre}/rna_multichain.db", f"{root}/a2021_b2021_rna_rna/b2021_rna_rna.db")
    copy(f"{post}/protein_rna.db", os.path.join(REPO, "database", "rbpDatabase.db"))
    ions = os.path.join(REPO, "data", "compare_ions")
    copy(f"{post}/protein_rna_ion_baseline.db", f"{ions}/a2021_protein_rna.db")
    copy(f"{post}/protein_rna_with_ions.db", f"{ions}/a2021_protein_rna_ions.db")
    copy(f"{post}/rna_singlechain_ion_baseline.db", f"{ions}/a2021_rna_ONLY.db")
    copy(f"{post}/rna_singlechain_with_ions.db", f"{ions}/a2021_rna_ions.db")
    # input ablation: one folder per side and condition, plus the natural run
    ab = os.path.join(REPO, "data", "nar_msa_template_ablation_20261007")
    for path in glob.glob(os.path.join(db, "ablation_studies", "msa_template_two_sided", "*", "*.db")):
        side, cond = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-3]
        copy(path, os.path.join(ab, "by_condition", side, cond, "protein_rna", "rbpDatabase.db"))
    copy(f"{post}/protein_rna.db", os.path.join(ab, "a2021.db"))
    rnc = os.path.join(REPO, "data", "rnacompete")
    copy(os.path.join(db, "other_analyses", "rnacompete", "rnacompete_v4_single_probe.db"), f"{rnc}/rnaCompete_v4.db")
    copy(os.path.join(db, "other_analyses", "rnacompete", "rnacompete_v5_competition.db"), f"{rnc}/rnaCompete_v5_MINMAX.db")


def place_nomsa(folder):
    """The MSA ablation databases of the RNA-only and template-removal experiments (noMSA.zip)."""
    nm, cd = os.path.join(folder, "noMSA"), os.path.join(REPO, "data", "compareDatabases", "noMSA")
    for d in ("RNAonly_noMSA", "single_protANDRNA_template_remove_experiment"):
        for f in glob.glob(os.path.join(nm, d, "*.db")):
            copy(f, os.path.join(cd, d, os.path.basename(f)))
    for f in glob.glob(os.path.join(nm, "RNAonly_noMSA", "*.db")):
        copy(f, os.path.join(cd, "RNA", "RNAonly_noMSA", os.path.basename(f)))
    for f in glob.glob(os.path.join(nm, "single_protANDRNA_template_remove_experiment", "*.db")):
        copy(f, os.path.join(REPO, "data", "single_protANDRNA_template_remove_experiment", "compareDatabases",
                             os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser(description="Download the published data from Zenodo (record "
                                             f"{ZENODO_RECORD}) and unpack it into data/.")
    ap.add_argument("--structures", action="store_true",
                    help="also download the structures (about 14 GB), needed only for the full recomputation")
    ap.add_argument("--local", metavar="DIR", help="take the files from DIR instead of downloading them")
    ap.add_argument("--record", default=ZENODO_RECORD, help=argparse.SUPPRESS)
    a = ap.parse_args()

    files = {} if a.local else record_files(a.record)
    wanted = DATABASE_FILES + (STRUCTURE_FILES if a.structures else [])
    print(f"Zenodo record {a.record} (all versions: doi:{ZENODO_CONCEPT_DOI})")

    print("databases")
    db_root = os.path.join(REPO, "data", "all_data_zip")
    if os.path.isdir(os.path.join(db_root, "all_databases")):
        shutil.rmtree(os.path.join(db_root, "all_databases"))
    extract(zipfile.ZipFile(download("all_databases.zip", files, a.local)), db_root)
    place_databases(os.path.join(db_root, "all_databases"))

    print("MSA ablation databases")
    tmp = os.path.join(DOWNLOADS, "noMSA")
    shutil.rmtree(tmp, ignore_errors=True)
    extract(zipfile.ZipFile(download("noMSA.zip", files, a.local)), tmp)
    place_nomsa(tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    print("analysis inputs")
    n = extract(zipfile.ZipFile(download("analysis_inputs.zip", files, a.local)), REPO)
    print(f"  {n} files")

    if a.structures:
        print("structures")
        for name in STRUCTURE_FILES:
            n = extract(zipfile.ZipFile(download(name, files, a.local)), os.path.join(REPO, "data", "all_data"))
            print(f"  {name}: {n} files")
    print("Done. Next: bash reproduce_all.sh (see README.md)")


if __name__ == "__main__":
    main()
