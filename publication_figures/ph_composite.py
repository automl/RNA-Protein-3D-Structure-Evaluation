#!/usr/bin/env python3
"""
ph_composite -- pH-dependence composite figure, as a publication_figures entry point.

The individual pH panels are analyses that live in evaluation/phCorrelation.py (its
``composite_panels`` provider and ``_draw_*`` callbacks reuse the pH data loaders there); this
script only assembles the chosen panels into ``ph_composite`` via plots.composite_figures, so the
figure is launched from publication_figures/ like the others.

Default panels: charged-residue occurrences and DockQ_Nr, both by experimental pH.

Run from the repo root:
  python publication_figures/ph_composite.py
  python publication_figures/ph_composite.py <panel> [<panel> ...] [out:NAME ncols:N]   # override
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.phCorrelation import composite_panels
from plots.composite_figures import run_composite_cli

DEFAULT_PANELS = ["charged_residue_occurrences_by_ph", "dockq_by_ph"]


def main(argv=None):
    argv = list(argv) if argv else (DEFAULT_PANELS + ["ncols:2"])
    if not any(str(a).startswith("out:") for a in argv):
        argv.append("out:ph_composite")
    return run_composite_cli(composite_panels, argv, default_out="ph_composite")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
