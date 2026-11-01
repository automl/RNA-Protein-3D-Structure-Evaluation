# Evaluating your own predictions with run_eval.py

This is the reference for the drop-in evaluation of new predictions. The README gives the short version.

[`run_eval.py`](run_eval.py) is a thin wrapper around the same sub-scripts `mainExecution.py`
drives, packaged as a clone → install → run drop-in: give it a folder of predictions (and,
optionally, ground truth) and it stages the inputs, runs the supported metrics, and writes a
self-contained `run.db` (plus CSV exports) that the plotting pipeline can consume. It never
touches the repo's `rbpDatabase.db`/`config.json` (each run is isolated), and anything whose
tool or input is missing is **skipped and logged**, not treated as an error.

## Install

```bash
bash install.sh          # Linux / macOS  (creates the RNABench + ost conda envs, builds US-align)
# or, on Windows:
powershell -ExecutionPolicy Bypass -File install.ps1
```

`install.sh --help` lists the flags (`--skip-envs`, `--skip-usalign`, `--force-usalign`,
`--cross`). The script finishes by reporting which tools/envs it found. See [`env/README.md`](env/README.md)
to create the environments by hand.

## Quick start

```bash
# quick : parse + AF3 confidence + US-align TM/RMSD/LDDT (no OST, no DSSR, no APIs)
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics quick

# basic : + OpenStructure all-atom LDDT and RNA AllChains scoring
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics basic

# full  : + interfaces, binding, and RBP DockQ/QS (needs a SWISS-MODEL token, or pass -noRBP)
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics full --token TOKEN

# no ground truth -> AF3 confidence only (all GT-dependent metrics are skipped)
conda run -n RNABench python run_eval.py --pred ./preds --out ./run --metrics quick
```

Presets are cumulative (`quick` ⊂ `basic` ⊂ `full`); `--metrics parse_af,af3,eval_metrics` selects
individual steps by name. Each step is gated on its tool/input and skipped-and-logged when
unavailable (`--strict` turns a skip into an error). The metric groups and what they need:

| preset / step | produces | needs |
|---------------|----------|-------|
| `quick`  | parse, AF3 `af3_*` confidence, US-align TM/RMSD/LDDT | US-align (built by `install.sh`); GT for TM/RMSD/LDDT; confidence JSONs for `af3_*` |
| `basic`  | + all-atom LDDT, RNA-AllChains LDDT/interface | the `ost` conda env + GT |
| `full`   | + interactions, interface area, binding affinity, RBP DockQ/QS | PyMOL + freesasa (interface area); SWISS-MODEL token (RBP; or `-noRBP`) |

Optional metadata add-ons (off by default, each gated on its input): `--msa` / `--templates`
(need the AF3 `*_data.json` sidecars), `--db-update` (pH / ions / ligands / GC-MFE / secondary
structure / electrostatics — needs a populated `dssr/` folder and ViennaRNA / forgi / PDB2PQR+APBS
/ DSSR), `--domains --domain-tsv FILE`, `--protein-family --family-email you@example.org`, and
`--motif-csv FILE`.

## Ground-truth matching and identity

Each prediction gets a shared id (the DB join key). Without a `--pair-map`, `run_eval.py` derives
it by stripping the AF3 `fold_<id>_s1_model_0` scaffolding from the prediction filename, matches
ground truth by (in order) an explicit `--pair-map`, an `--id-regex` capture, a classic PDB-id
token (`[0-9][A-Za-z0-9]{3}`, case-insensitive), or a **learned common prefix/suffix** shared
across a folder's filenames (so `job_<id>_...` vs `<id>_reference.cif` still pair). Ground-truth
`_entry.id` is rewritten to the shared id so the DB join works (disable with `--no-rewrite-entry-id`).

If ground truth was given but some predictions don't match, `run_eval.py` shows the unmatched ids
and prompts: continue confidence-only, supply/update a pair-map, enter a new `--id-regex` and
re-match live, re-scan after renaming, or abort. In a non-interactive shell (or with
`--non-interactive` / `--yes`) it continues confidence-only for the unmatched predictions — unless
`--strict`, which aborts instead.

## Era tagging (pre/post cutoff)

With real-PDB ground truth, each complex is tagged from its deposition date against `--cutoff-date`
(default `2021-09-30`): on/after → `a` (post), before → `b` (pre). `--era-split` additionally
writes `a_run.db` / `b_run.db` so the downstream pre/post analysis can split the eras (which it
infers from the `a`/`b` DB-name prefix). Non-PDB ground truth has no dates → a single lane.

## Inputs

All inputs are optional except the predictions. Absent sidecars/tools only reduce the metric set.

**Predictions (`--pred DIR`).** Predicted structures as mmCIF (`.cif`). Classification
(`protein_rna` vs `rna_rna`) is auto-detected from chain content (override with
`--classification`).

**Ground truth (`--gt DIR`).** Deposited mmCIF per complex. Optional — omit for a
confidence-only run.

**AF3 sidecars.** Placed beside each prediction or in a folder given with `--confidence` /
`--data-json`, matched to their prediction by the shared AF3 job stem:
- `*_summary_confidences_[0-4].json` + `*_full_data_[0-4].json` → the `af3_*` confidence metrics
  (core; automatic when present).
- `*_data.json` (the AF3 **input** JSON) → MSA depth (`--msa`) and template counts
  (`--templates`). These two steps identify the structure by the **first four characters** of the
  `*_data.json` name, so they only apply to classic 4-character PDB ids.

**pair-map CSV (`--pair-map FILE`).** The one file that maps arbitrary prediction filenames to
their ground truth and per-id sidecars — no naming convention required. Header row, column `pred`
required; the rest optional:

```csv
pred,gt,id,data_json,confidence,motif,domain
preds/model_001.cif,truth/6XYZ.cif,6XYZ,sidecars/6xyz_s1_data.json,sidecars/,,
preds/model_002.cif,truth/1ABC.cif,1ABC,,,,
```

Paths may be absolute or relative to the CSV's own directory. `confidence`/`data_json` may be a
file or a folder (searched for this id's sidecars).

**domain TSV (`--domains --domain-tsv FILE`).** Tab-separated, header required, columns exactly:

```
PDBId	domain_ids	domain_names	domain_counts	domain_pos
6XYZ	PF00076;PF08423	RRM_1;Rad51	2	12-84;110-330
```

Matched on `PDBId`; the four domain cells are stored verbatim as TEXT (any internal multi-domain
delimiter, e.g. `;`, is your choice and is not parsed).

**motif CSV (`--motif-csv FILE`).** The generalized drop-in path for the RNAcompete motif metrics
(`RNAmotif_score`, `RNAMotif_inExp`, `InterfaceArea_norm`), computed by
[`rnaCompete/motif_from_csv.py`](rnaCompete/motif_from_csv.py). Header required, columns
`pred_id, rna_sequence, exp_motif` (required) `[, pred_motif, interface_area, protein_length]`:

```csv
pred_id,rna_sequence,exp_motif,pred_motif,interface_area,protein_length
6XYZ,AAGUUCGAA,GUUCG,"G(3), U(4), U(5), C(6), G(7)",800,256
1ABC,CCAUGGCC,AUGG,,,
```

`exp_motif` must be a literal substring of `rna_sequence`; `pred_motif` is in `Base(pos)` format.
When `pred_motif` / `interface_area` / `protein_length` are omitted, the values already in the DB
(`RNAMotif` from the eval's interaction extraction, `ProteinRNAInterfaceArea`, `ProteinLength`) are
used — so on a `full` run you can supply just the three required columns.

**External tools/resources** (only what your chosen metrics use): US-align (built by `install.sh`),
the `ost` conda env (OpenStructure scoring), a SWISS-MODEL token (RBP DockQ/QS — free, from your
[SWISS-MODEL account](https://swissmodel.expasy.org/account); or `-noRBP`), PyMOL + freesasa
(interface area), and for `--db-update` a DSSR binary, ViennaRNA, forgi and PDB2PQR+APBS. Anything
absent is skipped and logged.

## Output

`<out>/run.db` (the metrics DB), `<out>/pred_<class>.csv` + `<out>/exp_<class>.csv` (CSV exports),
and `<out>/staging_manifest.csv` (the resolved id → prediction / ground truth / sidecar / era map).
Feed `run.db` into the plotting pipeline's preprocessing to regenerate figures on your data.

