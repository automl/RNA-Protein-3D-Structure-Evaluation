# SI structural-example figures

Composite SI figures `figures2/si_ex_<theme>.png` (torsion, pairing, elements, interface, msa, ions) and
`figures2/si_ex_ablation.{png,pdf}`. Run everything from the repository root in the RNABench environment (PyMOL
via `pymol2`, gemmi, Pillow, matplotlib). Panels are written to
`results/plots/all_chains/publication_figures/site_chosen/{rnafold,interface,ablation}/`; target in orange, AF3 in green.

| Script | Renders |
|---|---|
| `render_failure_modes.py` | first example of each RNA-fold and interface failure mode (a plain script: running or importing it re-renders all its panels) |
| `render_si_examples.py` | second examples (`python publication_figures/si_examples/render_si_examples.py [rna] [arcs] [interface] [pocket]`), pseudoknot / stacking arc diagrams, conformer tracks |
| `render_rnafold_extra.py` | RNA-only pairs (MSA, ions) superposed with US-align, non-WC arc diagrams (imported by `render_si_examples.py`) |
| `render_badsite.py` | incorrect-binding-site panels with the misfolded protein segment (called by `render_si_examples.py pocket`) |
| `render_slid_examples.py` | rotated example 7TJ2 and slid-duplex examples 7VG2, 8S9X (imports `render_si_examples.py`) |
| `render_ablation_examples.py` | controlled-ablation panels 7YFQ, 8SNX, 8PDS (`data/nar_msa_template_ablation_20261007/by_condition`) |

| Script | Composes |
|---|---|
| `compose_si_examples.py` | `si_ex_<theme>.png` from the rendered panels (`python .../compose_si_examples.py [theme ...]`) |
| `compose_ablation_examples.py` | `si_ex_ablation.{png,pdf}`: the ablation panels (7UO5 from `interface/msa_7UO5_*`) and the 4 x 4 score tile of each example (`publication_figures/ablation_data.py`) |

The scripts were developed outside the repository and restored here on 2026-10-19 from the session record; recomposing `si_ex_torsion` and `si_ex_elements` from panels re-rendered with the restored `render_failure_modes.py` reproduces the published figures pixel for pixel.
