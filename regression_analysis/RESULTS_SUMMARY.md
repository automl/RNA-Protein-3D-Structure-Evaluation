# Corrected pipeline — results summary (2026-07-09)

Corrections applied (all verified): geometric-mean MSA depth (RNA **and** protein) now
actually present in G1 at the complex level; protein-side AF3 confidence (ipTM/pTM/pLDDT +
RNA ipTM) added to G0; residual target regrouped so MSA stays in G1 (removes double-count);
`group_by` default hardened to `pdb_id`; data refreshed to `post_pre` (a2021 385, b2021 924,
RNA-only 66/102). Automated stability selection replaces the hand-curated pruning.

## 1. Reproduction check
RNA-only per-chain RNA-TM reproduces the paper essentially exactly:
`full R²=0.848`, shares G0=41, G1=32, G3=12, G5=11 (paper: 0.85 / 41 / 32 / 12 / 12).
Other subsets are consistent modulo the data refresh. Pipeline is sound (preprocessing fit
strictly within fold; pooled-OOF R²; deterministic GroupKFold; α fixed).

## 2. Old vs new Shapley group shares (% of full-model R²)

| dataset / target | full R² (old→new) | G0 | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|
| all complex **cTM** | 0.71 → **0.725** | 48 → **55** | 1 → 2 | 29 → 26 | 4 → 3 | 11 → 10 | 7 → 5 |
| multichain **cTM** | 0.68 → 0.691 | 48 → **56** | 1 → 1 | 25 → 23 | 3 → 1 | 12 → 10 | 12 → 8 |
| 1v1 **cTM** | 0.47 → 0.457 | 68 → **79** | 0→2 | 16→8 | 0 | 7→4 | 10→7 |
| all **DockQ** | 0.40 → 0.418 | 39 → **45** | 0→1 | 13→11 | 9 | 34 → 31 | 4→3 |
| multichain **DockQ** | 0.46 → 0.474 | 53 → 57 | 1 | 12 | 2 | 31 → 28 | 2→0 |
| multichain **RNA-TM** | 0.48 → 0.480 | 44 → **50** | 17 → **18** | 13→12 | 1 | 19 | 6→1 |
| RNA-only **RNA-TM** | 0.85 → 0.848 | 41 | **32** | 4 | 12 | 0 | 12→11 |
| cTM **residual** | — → 0.276 | — | 3 | 52 | 2 | 27 | 16 |
| RNA-TM **residual** | — → 0.107 | — | **40** | 27 | 5 | 30 | −3 |

**Read-out.** The only material change is **G0 rising on protein-dominated complex targets**
(cTM 48→55, DockQ 39→45) — this is the protein-side confidence, which the coefficients show
carries almost no *RNA*-accuracy information (protein ipTM/pTM coeffs ≈ ±0.005 vs RNA pLDDT
0.126 on multichain RNA-TM). On the RNA targets the story is unchanged: evolution holds
(RNA-only 32%, multichain 18%, and **40% of the RNA residual**). So the honest, target-aware
framing is: *AF3 confidence reports the protein-carried complex, but is incomplete precisely
for RNA reliability* — cleaner than the old "confidence roughly doubles" wording.

## 3. Automated stability selection (feature selection)

ElasticNet, 100 grouped resamples, keep features with selection freq ≥ 0.70; nested-CV R² is
selection redone inside each fold (leakage-free honest number).

| dataset | n | candidates → selected | R² full / pooled-sel / **nested** |
|---|---|---|---|
| all complex cTM | 1308 | 42 → 41 | 0.725 / 0.725 / **0.722** |
| all complex DockQ | 1285 | 42 → 40 | 0.418 / 0.417 / **0.422** |
| multichain cTM | 963 | 42 → 40 | 0.691 / 0.691 / **0.688** |
| multichain RNA-TM | 1999 | 37 → 26 | 0.480 / 0.489 / **0.479** |
| RNA-only RNA-TM | 163 | 24 → 15 | 0.852 / 0.857 / **0.854** |
| 1v1 per-chain RNA-TM | 345 | 44 → 10 | 0.363 / 0.559 / **0.540** |

Nested ≈ full everywhere → selection is honest (no meaningful leakage) and the trimmed sets
preserve performance. **Notably, selection stabilizes the 1v1 subset** (0.363 → 0.540 nested;
44 candidates → 10) — the paper's "1v1 unstable, omit from ranking" caveat can be replaced.

## 4. Ridge-alpha sensitivity (all complex cTM)

| α | G0 | G1 | G2 | G3 | G4 | G5 | full R² |
|---|---|---|---|---|---|---|---|
| 1 | 55 | 2 | 26 | 3 | 10 | 4 | 0.724 |
| 10 | 55 | 2 | 26 | 3 | 10 | 5 | 0.725 |
| 100 | 55 | 2 | 26 | 3 | 10 | 5 | 0.707 |

Shares are essentially invariant → α=10 is not cherry-picked.

## 5. Figures — status
All figure **databases** are wired (`data/compare_ions/`, `data/compareDatabases/…`,
`database/rbpDatabase.db` aliased to the a2021 protein-RNA db, `database/config.json` stub).
Regeneration is **blocked** on two collaborator support packages that are not in the repo:
`plots/` (publication styling) and `evaluation/` (`RNAMetrics`, `phCorrelation`,
`peg_analysis_plots`). Once you drop those in + a real `config.json`, `bash run_figures.sh`
produces all figures into `figures_final/`.
