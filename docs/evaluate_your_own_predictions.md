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
# quick : parse + AF3 confidence + TM-score, RMSD, backbone LDDT (no OST, no DSSR, no APIs)
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics quick

# basic : + OpenStructure all-atom LDDT and RNA AllChains scoring
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics basic

# full  : + interactions, interface area, binding, and RBP DockQ/QS (no token needed; pass -noRBP to skip RBP)
conda run -n RNABench python run_eval.py --pred ./preds --gt ./gt --out ./run --metrics full

# no ground truth -> AF3 confidence only (all GT-dependent metrics are skipped)
conda run -n RNABench python run_eval.py --pred ./preds --out ./run --metrics quick
```

Presets are cumulative (`quick` ⊂ `basic` ⊂ `full`); `--metrics parse_af,af3,eval_metrics` selects
individual steps by name. Each step is gated on its tool/input and skipped-and-logged when
unavailable (`--strict` turns a skip into an error). The metric groups and what they need:

| preset / step | produces | needs |
|---------------|----------|-------|
| `quick`  | parse, AF3 `af3_*` confidence, TM-score (US-align), RMSD (PyMOL), backbone (C3'/CA) LDDT | US-align (built by `install.sh`); GT for TM/RMSD/LDDT; confidence JSONs for `af3_*` |
| `basic`  | + OpenStructure all-atom LDDT, RNA-AllChains LDDT/interface | the `ost` conda env + GT |
| `full`   | + interactions, interface area, binding affinity, RBP DockQ/QS, interface fallback | PyMOL + freesasa (interface area); the `ost` conda env (RBP; or `-noRBP`) |

The backbone (C3'/CA) `RNA_LDDT` — the fold label of the decision tree — is computed at every preset,
not only `quick`. `basic` adds the all-atom LDDT columns (`*_allAtom`) on top. A SWISS-MODEL token is
**not** required for the RBP DockQ/QS interface scores: without one they are scored by the local
OpenStructure install in the `ost` env (`--local`); a token only routes them to the SWISS-MODEL service
instead (`--token`), as was done for the paper.

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
(`protein_rna` vs `rna_rna`) is auto-detected from the `_entity_poly.type` records, so the mmCIF must
carry them (AF3 and RCSB mmCIF do). PDB-format files are not read directly — convert them to mmCIF first
(e.g. `gemmi convert in.pdb out.cif`, gemmi is in the `RNABench` env), checking that the result has
`_entity_poly.type`.

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
the `ost` conda env (OpenStructure scoring, incl. the RBP DockQ/QS interface scores — run locally, no
token needed), PyMOL + freesasa (interface area), and for `--db-update` a DSSR binary, ViennaRNA, forgi
and PDB2PQR+APBS. A SWISS-MODEL token (free, from your
[SWISS-MODEL account](https://swissmodel.expasy.org/account)) is **optional**: with `--token` the RBP
interfaces are scored by the SWISS-MODEL OpenStructure service instead of the local install. Anything
absent is skipped and logged.

## What each metric is computed with

| metric (DB columns) | tool |
|---|---|
| `*_TM`, `RNAComplex_TM`, `RNA_TM_AllChains`, `Protein_TM_AllChains` | **US-align** 20241108 (`evaluation/USalign/`) |
| `*_RMSD` | **PyMOL** `align` (the first value of the RMSD list) |
| backbone `Protein_LDDT` / `RNA_LDDT` / `Complex_LDDT` (C3' for RNA, CA for protein) | own code, `extractEvalMetrics.calculate_complex_lddt` |
| `*_LDDT_allAtom`, `iLDDT`, `ICS`, `QS_global`, `Dockq`, `RBP_DockQ/QS/fnat/irmsd`, `RNA_iLDDT/ICS_AllChains` | **OpenStructure** `compare-structures` (the `ost` env; version pinned in `env/`, see the README's *Software versions*) |
| `af3_*` (pTM/ipTM, pLDDT, PAE, ranking score, clash) | parsed from the AF3 confidence JSONs |

### Interface fallback (step `interface_fallback`, preset `full`)

When OpenStructure's `compare-structures` fails on a complex, the OpenStructure steps repeat the comparison
once without the SEQRES records of the reference (`evaluation/ost_compare_noseqres.py`, the same action and JSON);
this avoids a chain-mapping error of OpenStructure 2.12 on a few complexes (e.g. 7QDY).
`ost_extractAllAtomLDDT` records `OST_unmapped_chains`: the model chains OpenStructure dropped from a
complex, `1` when the whole comparison failed, `0` when every chain was mapped. Interfaces without an
OpenStructure DockQ or per-chain iLDDT and the complexes flagged there are rescored by `evaluation/interface_own_dockq.py` on the
exact residue mapping (the paper's fallback): experimental and predicted chains are cut to the residues the
mapping pairs, renumbered, and scored with OpenStructure's DockQ and scorer with the chain mapping fixed. The
chain pairing comes from OpenStructure's whole-complex comparison (`<out>/data/protein_rna/lddt/`) where it is
complete, and from sequence identity otherwise (`--check-alignment`: an OpenStructure alignment that pairs
clearly fewer residues than the exact sequence alignment is replaced). Its maps and tables are kept in
`<out>/interface_fallback/`. The results go into the same columns and only where OpenStructure gave no value
(`RBP_DockQ/QS/fnat/irmsd`, `RNA_iLDDT/ICS_AllChains` per interface); the whole-complex values (`Dockq`,
`iLDDT`, `ICS`, `QS_global`, `Complex_LDDT_allAtom`) are also replaced for flagged complexes and for
complexes with an interface whose DockQ is the fallback's, as their OpenStructure values cover only part of the
complex (the paper's rule). `Interface_score_source` (`ost` / `exact_mapping`, complex
values from the fallback) and `Fallback_interfaces` (chain pairs whose DockQ is the fallback's) record which
values are the fallback's. A metric
OpenStructure cannot compute even on the fixed mapping (the LDDT-based ones when it discards an RNA of a few
nucleotides) stays empty.

## Messages you can ignore

The pipeline reuses the paper's sub-scripts, which print some messages that are not errors:

- `Parse failed (sequence file sequences.fasta)` — the parse step tries an Rfam family annotation with
  Infernal; it is skipped when Infernal/Rfam is not installed and does not affect any score.
- FreeSASA `Found no matches to resn ...` — the interface-area step asks FreeSASA to ignore residue
  names a given structure does not contain; harmless.
- OpenStructure `... without_chem_mapping ...` / `no OST JSON returned` — OpenStructure could not map a
  chain (see `OST_unmapped_chains` above); with `full` the fallback scores it.
- `Target chain specified in chain_mapping (...) does not exist` in the fallback's log — OpenStructure
  discarded a very short chain for one metric (usually the LDDT-based ones); that metric stays empty.

## Output

`<out>/run.db` (the metrics DB), `<out>/pred_<class>.csv` + `<out>/exp_<class>.csv` (CSV exports),
and `<out>/staging_manifest.csv` (the resolved id → prediction / ground truth / sidecar / era map).
Feed `run.db` into the plotting pipeline's preprocessing to regenerate figures on your data.

