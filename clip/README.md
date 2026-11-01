# CLIP-motif agreement of experimental vs AF3-predicted protein–RNA interfaces

Does AlphaFold3 place the protein–RNA interface on the sequence the RBP actually
binds? This compares the **interface footprints** of `exp_protein_rna` (ground
truth, PDB) and `pred_protein_rna` (AF3) against known RBP binding motifs.

## The idea

`RNAMotif` in both tables is not a sequence motif — it is the set of RNA residues
contacting protein, one list entry per chain pair in `ChainIDpairList_proteinRNA`,
with **1-based positions into that pair's RNA chain sequence**. A maximal run of
consecutive contacted positions, rendered back as sequence, is a *footprint*: the
structural analogue of a CLIP k-mer, and directly scoreable against a PWM.

Each footprint is scored against its RBP's motifs with partial-overlap PWM
log-odds (the footprint is often shorter than the motif), rescaled to [0,1] by
what the aligned columns could attain. The null is the **same footprints scored
against other RBPs' motifs** — so `z` measures motif specificity, not footprint
composition.

## Layout

```
scripts/
  af3clip.py         footprint parsing from rbpDatabase.db; MEME parsing; PWM scan/scoring
  atlas.py           unified motif atlas (RBPBench + CISBP-RNA + ATtRACT + RBPDB)
  fetch_motifs.sh    download the four atlases into motifs/
  fetch_uniprot.py   structure-DB accessions -> gene symbols via UniProt REST
  build_overlap.py   which structure-DB genes have motifs        -> results/overlap.tsv
  run_analysis.py    per-entry exp/pred motif scores + null      -> results/entry_scores.tsv
  stats.py           hypothesis tests                            -> results/stats_report.txt
  run_all.sh         the whole pipeline
motifs/              downloaded atlases (see provenance below)
results/             outputs
```

Reproduce with `scripts/run_all.sh`. Requires numpy + scipy; no other install.

## Cohorts

Both deposition cohorts of the post/pre comparison set are analysed, split on the AF3
training cutoff (2021-09-30):

| cohort | file | entries | date range | meaning |
|---|---|---|---|---|
| `post` | `data/compareDatabases/post_pre/a2021_b2021_protein_rna/a2021.db` | 385 | 2018-2024 | novel to AF3 (byte-identical to `database/rbpDatabase.db`) |
| `pre`  | `.../b2021.db` | 924 | 1989-2022 | AF3 was trained on these |

Every entry carries a `cohort` column and every comparison is reported pooled and split.

## Coverage

Of 1309 entries, 159 (57 genes) map to a protein with a motif in the atlas: 34
post-cutoff (15 genes) and 125 pre-cutoff (51 genes); 146 yield a scorable
experimental footprint and 134 are scorable in both exp and pred. The limit is
the structures, not the atlas -- most human/mouse entries are enzymes and RNP
machines with no sequence preference (TSEN, ELP, PUS3, METTL1/15, QTRT1/2,
SKIC2/3/8, ...), and no atlas contains a viral RBP motif, so the large SARS-CoV-2,
influenza and Thogoto subsets cannot be scored at all.

## Results

**1. Experimental interfaces sit on the known motif.** Mean z = +1.70 vs the
other-RBP null (t = 35.0, p = 1.5e-72; Wilcoxon p = 2.0e-25, n = 146). 99/146
entries at p < 0.05 against 7.3 expected (binomial p = 7.1e-92). Recovers
textbook motifs unprompted: U1A/SNRPA `AUUGCAC`, PUM1/PUM2 `UGUANAUA`, NOVA
`YCAY`, HNRNPA1 `UAGGGA`, LIN28A `GGAG`, WDR33 `AAUAAA`.

**2. AF3 reproduces it, and training exposure does not help.** Over 134 paired
entries the shift is negligible (z_exp +1.68 -> z_pred +1.64, paired t p = 0.17)
and the two track tightly (Pearson r = 0.885, p = 9.6e-46). Split by cohort:

| cohort | n_exp | z_exp | n_paired | z_pred | shift | r(exp,pred) |
|---|---|---|---|---|---|---|
| post (novel) | 30 | +1.62 | 26 | +1.55 | -0.025 | 0.896 |
| pre (trained on) | 116 | +1.72 | 108 | +1.66 | -0.041 | 0.882 |

No cohort difference in z_exp (Mann-Whitney p = 0.385) or in the AF3 shift
(p = 0.671). AF3 lands on the right sequence about as well for structures it has
never seen as for ones it was trained on.

**3. Specificity is carried by sequence-reading RBPs, replicated in both cohorts.**
Pooled z_exp: sequence +1.89 (n = 93), structure-driven +1.44 (n = 46),
guide-directed +0.85 (n = 7); Mann-Whitney p = 9.5e-06. Ordering holds separately
pre-cutoff (+1.90 / +1.49 / +0.96) and post-cutoff (+1.86 / +1.17 / +0.55). The
cleanest negative control is **RBM8A** (EJC core, deposited sequence-independently
on spliced mRNA): the only gene with negative mean specificity (z_exp = -0.81).
Class annotation is in `scripts/rbp_class.tsv`, assigned a priori. It is **manual
expert judgement**, not a database field -- no motif atlas records *how* a protein
achieves specificity. `scripts/validate_classes.py` re-derives every gene from its
UniProt Pfam/InterPro domains and reports 95% concordance (54/57). The three
exceptions are the expected failure mode of a domain rule, and the measured data
favours the manual call in each: WDR33 (WD40 propeller reading AAUAAA, z = +1.89)
and NUDT21/CFIm25 (Nudix fold reading UGUA, z = +1.78) have no domain the
vocabulary marks as sequence-reading; RBM8A carries an RRM but is deposited by the
spliceosome and has the only negative z (-0.81). Output: `results/class_validation.txt`.

**4. AF3 quality metrics carry no information about motif correctness.** At
n = 134 nothing predicts z_pred: DockQ r = -0.030 (p = 0.73), RNAmatch_score
r = -0.024 (p = 0.78), RNAmotif_score r = -0.098 (p = 0.26), iLDDT r = 0.005
(p = 0.95); within-gene and between-gene components separately null (all |r| <
0.12, p > 0.28). *(At the earlier n = 26 this looked like a strong negative
correlation confounded by RBP class; with adequate n there is simply no
relationship. Motif correctness is orthogonal to the geometric and compositional
scores.)*

**5. The informative failure is local contact dropout.** AF3 shortens the longest
contiguous interface run in 39/159 entries; in 12 it falls below the 4 nt needed
to read a motif. Such entries do score lower on average (RNAmatch 0.54 vs 0.74,
MW p = 2.0e-05; DockQ 0.48 vs 0.63, p = 0.012), but the *degree* of shortening is
uncorrelated with any score (RNAmatch r = 0.129 p = 0.11; RNAmotif r = 0.031
p = 0.70; DockQ r = 0.039 p = 0.62). The blindness is specific and reproducible
in **WDR33/CPSF**, where AF3 deletes the single contact at the central adenosine
of AAUAAA in **all five** structures across both cohorts:

| PDB | cohort | RNAmotif | RNAmatch | DockQ |
|---|---|---|---|---|
| 8E3I | post | 1.00 | 0.66 | 0.86 |
| 8R8R | post | 1.00 | 0.60 | 0.77 |
| 6DNH | pre | 1.00 | 0.57 | 0.85 |
| 6URO | pre | 1.00 | 0.59 | 0.80 |
| 6FUW | pre | 1.00 | 0.55 | 0.89 |

`RNAmotif_score` is exactly 1.00 in every case. These are interface *composition*
measures, so a contact set that is nearly correct but broken in the middle of the
recognition element scores as nearly correct.

**6. Motif provenance matters, in the direction the assay predicts.** PDB
crystallography and an AF3 prediction of it are both in-vitro-like observations of
a binary complex, so in-vitro motifs are the assay-matched comparator and CLIP the
mismatched one. Experimental footprints score significantly higher against the
in-vitro arm: z_exp +1.81 (n = 135) vs +1.52 (n = 38), Mann-Whitney p = 6.4e-05;
paired on the 27 entries scored by both arms, +1.82 vs +1.47, Wilcoxon p = 2.2e-05.
*(At n = 34 this was null, p = 0.365; the power came from the pre-cutoff cohort.)*

**7. Organism: still no claim.** Human vs mouse looks significant pooled (+1.73 vs
+1.47, p = 0.028) but is confounded by class and does not survive restriction to
sequence-specific RBPs (+1.92 n = 86 vs +1.59 n = 7, p = 0.229).

**8. Viral entries uncovered, n = 0.** No atlas contains a viral RBP motif;
ATtRACT spans 38 organisms, all cellular, and RBPBench/cisBP-RNA/RBPDB are host
RBPs only. Scoring them would need motifs derived de novo from viral CLIP.

**Caveat.** Entries cluster by gene (134 paired over 51 genes; SNRPA 21, IFIH1 16,
PUM1 11), so metric correlations are reported with a within/between-gene
decomposition.

## Figures

Generated into `results/plots/all_chains/publication_figures/` and copied to
`2025_NAR_AF3_Evaluation/figures2/`. Run with the RNABench env (needs
matplotlib/seaborn), from the repo root:

```
~/miniconda3/envs/RNABench/bin/python publication_figures/af3_clip_motif_loss_figure.py
~/miniconda3/envs/RNABench/bin/python publication_figures/af3_clip_atlas_motif_figure.py
```

- `af3_clip_motif_loss` -- A contact ladders for 8E3I/8R8R; B fragmentation vs
  RNAmatch_score; C per-entry z, experimental -> AF3.
- `af3_clip_atlas_motifs` -- A z_exp vs z_pred; B by recognition class and cohort;
  C in-vitro vs CLIP provenance; D atlas motif vs PDB and AF3 footprints, 10 genes
  with a CLIP-derived motif and 10 with an in-vitro motif.

Colour convention follows the PyMOL structure figures in the draft: **PDB orange**
(`#D55E00`), **AF3 green** (`#009E73`), an Okabe-Ito colourblind-safe pair. RBP
recognition class uses a separate blue/purple/red ramp so it never competes with
the PDB-vs-AF3 distinction.

### Sub-threshold contact runs

Footprints shorter than 4 nt cannot be aligned to a PWM and are not scored, but
they are still real contacts. `anyrun_exp` / `anyrun_pred` in `entry_scores.tsv`
record the longest contact run at *any* length, and panel D displays it dimmed
and marked when no scorable footprint exists -- so a short interface is never
reported as "none". Examples: 6SJD (ZC3H12B) AF3 = `GU` (2 nt), 4N48 (CMTR1)
AF3 = `GAU` (3 nt), 8E3I (WDR33) AF3 = `AAC` (3 nt).

## Manuscript

Inserted into `2025_NAR_AF3_Evaluation/main.tex`: a results block after the
RNAcompete paragraph (`\paragraph{Independent RBP motif atlases...}`, with both
figure environments) and a methods subsection
`\subsection{RBP motif-atlas comparison of interface footprints}`
(`sec:clip_motif_eval`) before `Feature extraction`. A pre-existing xcolor option
clash prevents a full compile of `main.tex`; both inserted blocks compile cleanly
in isolation.

## Motif atlas provenance

| atlas | motifs | CLIP-derived | source |
|---|---|---|---|
| RBPBench (catRAPID_omics v2.1 ext) | 598 | 294 | bioconda `rbpbench-1.0.6`; `rbp_motif2name_ids.1.txt` gives per-motif assay |
| CISBP-RNA human | 598 | 0 | RBPBench GitHub `test/` |
| ATtRACT (human+mouse) | 1253 | 0 | attract.cnic.es (2016; SELEX/RNAcompete) |
| RBPDB human | 7 | 0 | rbpdb.ccbr.utoronto.ca |
| **merged** | **2456** (359 genes) | **294** (134 genes) | |

CLIP provenance in RBPBench: 206 eCLIP_ENCODE, 46 PAR-CLIP, 26 HITS-CLIP,
15 eCLIP, 14 iCLIP, plus CLIP/PAR-iCLIP/dCLIP/FLASH.

**Not used:** mCrossBase (download URL 404s — the GitHub repo is the discovery
tool only, not the motif database), POSTAR (binding sites, not motifs), ENCORI.
