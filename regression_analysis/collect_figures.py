#!/usr/bin/env python3
"""
collect_figures.py -- gather every submission figure into tex/figures/ in all
formats (png 600 dpi, pdf, svg, eps), so the whole figure set lives in one place.

Layout produced:
  tex/figures/                Figure1.*, Figure2.* (assembled main figures)
                              + standalone Shapley figures (fig_shapley_*, fig_delta_*)
  tex/figures/notitles/       the no-titles variants (for re-assembly)
  tex/figures/panels/         the individual panels that make up Figure 1 and 2

Idempotent: safe to run repeatedly. Only copies; never deletes.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(".")
TEX = ROOT / "tex" / "figures"
PANELS = TEX / "panels"
EXTS = ("png", "pdf", "svg", "eps")


TEX_ABS = (ROOT / "tex" / "figures").resolve()


def copy_globs(patterns, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    seen = set()
    n = 0
    for pat in patterns:
        for p in ROOT.glob(pat):
            if not p.is_file():
                continue
            rp = p.resolve()
            # never re-copy something that already lives under tex/figures/
            if TEX_ABS in rp.parents or rp == TEX_ABS:
                continue
            if rp in seen:
                continue
            target = (dest / p.name).resolve()
            if rp == target:
                continue
            shutil.copy2(p, dest / p.name)
            seen.add(rp)
            n += 1
    return n


def main():
    # 1. Assembled main figures
    n_main = copy_globs([f"publication_figures/combined/Figure*.{e}" for e in EXTS], TEX)
    # 2. Standalone Shapley / Delta-R^2 figures (titled)
    n_stand = copy_globs([f"publication_figures/*.{e}" for e in EXTS], TEX)
    # 3. No-titles variants
    n_nt = copy_globs([f"publication_figures/notitles/*.{e}" for e in EXTS], TEX / "notitles")
    # 4. Individual panels that compose Figure 1 and Figure 2
    panel_pats = []
    for e in EXTS:
        panel_pats += [
            f"**/Fig1A_direct*primary*.{e}", f"**/Fig1B_residual*primary*.{e}",
            f"**/Fig1C_coef*a31*.{e}", f"**/Fig1D_scatter*.{e}",
            f"**/Fig2A_rnaonly*primary*.{e}", f"**/Fig2B_rp1v1*primary*.{e}",
            f"**/Fig2C_multichain*primary*.{e}",
            f"Fig2D_heatmap_vertical_groups.{e}",
            f"Fig2E_multichain_shift_n15.{e}",
            f"Fig2F_system_deltas_n15.{e}",
        ]
    n_panels = copy_globs(panel_pats, PANELS)

    print(f"collected -> {TEX}")
    print(f"  main composite files : {n_main}")
    print(f"  standalone figures   : {n_stand}")
    print(f"  no-titles variants   : {n_nt}")
    print(f"  individual panels    : {n_panels}  (in {PANELS})")
    # quick format inventory of the main folder
    for stem in ("Figure1", "Figure2"):
        have = [e for e in EXTS if (TEX / f"{stem}.{e}").exists()]
        print(f"  {stem}: {', '.join(have) if have else 'MISSING'}")


if __name__ == "__main__":
    main()
