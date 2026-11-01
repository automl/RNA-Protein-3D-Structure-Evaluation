#!/usr/bin/env python3
"""
assemble_figures.py

Compose the multi-panel main figures (Figure 1 and Figure 2) for the manuscript
from the individual panel images produced by the plotting scripts, matching the
layout in the current draft and sized for BMC Bioinformatics (double-column,
max width 170 mm).

Panels are placed at their native aspect ratio (no stretching); each row shares a
common height and panel widths are proportional to their aspect ratios, so the
montage is tight and undistorted. Bold panel letters (A, B, ...) are added.

Inputs are resolved by glob so the script is robust to the hashed run-directory
names. Missing panels are drawn as labelled placeholders (the script never
crashes), making it easy to see what still needs to be generated.

Figure 1 (complex level):   A R^2 ladder | B residual R^2
                            C full-model coefficients | D accuracy-vs-RNA-fraction
Figure 2 (per-chain):       A RNA-only | B 1v1 | C multichain      (R^2 ladders)
                            D coefficient heatmap | E Delta-R^2 (complex vs chain)
                            | F stepwise coefficient change

Outputs (PDF + PNG, 600 dpi) to --outdir (default publication_figures/) and are
also copied to tex/figures/ for the manuscript.

Run:  python assemble_figures.py
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from figsave import save_all_formats, DEFAULT_FORMATS

ROOT = Path(".")

# ---------------------------------------------------------------------------
# Panel sources. Each entry is a list of glob patterns / explicit paths searched
# in order; the first existing match (most recently modified) is used.
# ---------------------------------------------------------------------------
# For Figure 2 panel E we default to the principled Delta-R^2 figure (Shapley,
# complex vs per-chain) rather than the raw coefficient-shift panel, which mixes
# two target scales. Set USE_DELTA_FOR_E = False to fall back to the original.
# The preferred delta-R^2 panel (Shapley complex-vs-per-chain) depends on the
# publication-table summary chain (make_regression_tables.py -> summary_overview.csv)
# that lives in run_paper.sh, not the lean reproduce_all pipeline. Default to the
# available coefficient-shift panel so Figure 2 is complete from reproduce_all; set
# True (and wire that chain) to restore the delta panel.
USE_DELTA_FOR_E = True

FIG1_PANELS = {
    "A": ["**/Fig1A_direct*primary*.png"],
    "B": ["**/Fig1B_residual*primary*.png"],
    "C": ["**/Fig1C_coef*a31*.png", "**/Fig1C_coef*.png"],
    "D": ["**/Fig1D_scatter*.png"],
}

FIG2_PANELS = {
    "A": ["**/Fig2A_rnaonly*primary*.png"],
    "B": ["**/Fig2B_rp1v1*primary*.png"],
    "C": ["**/Fig2C_multichain*primary*.png"],
    "D": ["Fig2D_heatmap_vertical_groups.png", "Fig2D_heatmap*.png"],
    "E": (["publication_figures/notitles/fig_delta_r2_complex_vs_perchain.png",
           "publication_figures/fig_delta_r2_complex_vs_perchain.png",
           "**/fig_delta_r2_complex_vs_perchain.png"]
          if USE_DELTA_FOR_E else
          ["Fig2E_multichain_shift_n15.png", "Fig2E_multichain_shift*.png"]),
    "F": ["Fig2F_system_deltas_n15.png", "Fig2F_system_deltas*.png"],
}

# row groupings (which panels share a row), top to bottom
FIG1_ROWS = [["A", "B"], ["C", "D"]]
FIG2_ROWS = [["A", "B", "C"], ["D", "E", "F"]]

# layout constants (inches)
FIG_W = 6.9            # ~ 175 mm; trim slightly for BMC 170 mm if needed
MARGIN = 0.08          # outer top/bottom margin
GAP = 0.16             # horizontal gap between panels in a row
ROW_GAP = 0.16         # gap between a row's images and the next row's letter band
LETTER_BAND = 0.17     # reserved whitespace ABOVE each row to hold the panel letters
                       # (sized for the ~8 pt letters; was 0.28 for the old 16 pt)
LETTER_PAD = 0.03      # gap between the letter baseline and the image top edge
LETTER_FS = 8       # montage is ~7 in wide (placed ~1:1 in the paper) -> 8 pt on page,
                    # matching every other figure's panel letter (was 16 -> looked huge)
OUT_DPI = 600          # output resolution; panels are downsampled to their placement size at this dpi
OVERSAMPLE = 1.3       # decode panels at 1.3x their placement pixel size to stay crisp


def resolve(patterns) -> Path | None:
    """Return the best match, respecting pattern PRIORITY: the first pattern that
    yields any existing file wins; within that pattern the most-recent file is
    used. This makes the choice deterministic and order-controlled."""
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            # use glob.glob (not Path.glob): recursive "**/" is reliable across
            # Python versions here, whereas Path('.').glob('**/..') can miss matches.
            # Pass the pattern verbatim (ROOT is "."); joining via Path mangles "**/".
            import glob as _glob
            cands = [Path(c) for c in _glob.glob(pat, recursive=True) if Path(c).exists()]
        else:
            p = ROOT / pat
            cands = [p] if p.exists() else []
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def get_aspect(path: Path | None):
    """Return aspect = w/h without decoding the pixels (PIL reads the header only)."""
    if path is None:
        return 1.3
    with Image.open(path) as im:
        w, h = im.size
    return w / h


def load_downsampled(path: Path, target_w_px: int, target_h_px: int):
    """Decode a panel as uint8 RGBA, downsampled to just above its placement size.

    Panels are generated at 600 dpi on their native canvas, but in the montage each
    is placed at a fraction of that. Decoding every full-res panel to float32 at once
    exhausted memory on a loaded machine, so we (a) use uint8 not matplotlib's float32
    imread, and (b) shrink each panel to ~OVERSAMPLE x its final pixel footprint. imshow
    renders the result identically at print resolution; peak memory drops by ~10x.
    """
    with Image.open(path) as im:
        im = im.convert("RGBA")
        tw = max(1, int(target_w_px))
        th = max(1, int(target_h_px))
        if im.width > tw or im.height > th:
            im.thumbnail((tw, th), Image.LANCZOS)
        return np.asarray(im)


def assemble(panels: dict, rows: list, out_base: Path, copy_to: Path | None):
    usable_w = FIG_W - 2 * MARGIN

    # resolve panels and read aspects (header only, no pixel decode yet)
    resolved = {k: resolve(v) for k, v in panels.items()}
    aspects = {k: get_aspect(resolved[k]) for k in panels}

    # compute per-row common height
    row_heights = []
    for row in rows:
        asp = [aspects[k] for k in row]
        h = (usable_w - GAP * (len(row) - 1)) / sum(asp)
        row_heights.append(h)

    # each row reserves a LETTER_BAND of whitespace above its images for the letters
    fig_h = (2 * MARGIN + len(rows) * LETTER_BAND + sum(row_heights)
             + ROW_GAP * (len(rows) - 1))
    fig = plt.figure(figsize=(FIG_W, fig_h))

    y_cursor = fig_h - MARGIN
    for row, h in zip(rows, row_heights):
        img_top = y_cursor - LETTER_BAND      # images start below the letter band
        img_bottom = img_top - h
        x = MARGIN
        for k in row:
            asp = aspects[k]
            w = asp * h
            ax = fig.add_axes([x / FIG_W, img_bottom / fig_h, w / FIG_W, h / fig_h])
            path = resolved[k]
            if path is not None:
                # decode only now, downsampled to this panel's actual footprint
                img = load_downsampled(path, w * OUT_DPI * OVERSAMPLE, h * OUT_DPI * OVERSAMPLE)
                ax.imshow(img)
                del img
            else:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="0.92", ec="0.6", lw=0.8))
                ax.text(0.5, 0.5, f"[panel {k}\nnot found]", ha="center", va="center",
                        fontsize=9, color="0.3")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            # bold panel letter in the band ABOVE the panel: never overlaps the plot,
            # and all letters in a row share the same baseline (img_top) -> aligned.
            fig.text(x / FIG_W, (img_top + LETTER_PAD) / fig_h, k,
                     fontsize=LETTER_FS, fontweight="bold", ha="left", va="bottom")
            x += w + GAP
        y_cursor = img_bottom - ROW_GAP

    out_base.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None: keep the designed canvas (margins host the panel letters)
    written = save_all_formats(fig, out_base, dpi=OUT_DPI, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out_base.name}.{{{','.join(DEFAULT_FORMATS)}}}  ({FIG_W:.1f} x {fig_h:.1f} in)")
    for k, p in resolved.items():
        print(f"   panel {k}: {'MISSING' if p is None else p}")
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        for f in written:
            shutil.copy2(f, copy_to / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="publication_figures/combined")
    ap.add_argument("--tex-figdir", default="tex/figures", help="also copy composites here")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    texdir = Path(args.tex_figdir) if args.tex_figdir else None

    assemble(FIG1_PANELS, FIG1_ROWS, outdir / "Figure1", texdir)
    assemble(FIG2_PANELS, FIG2_ROWS, outdir / "Figure2", texdir)


if __name__ == "__main__":
    main()
