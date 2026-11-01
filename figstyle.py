"""Central figure style for ALL AF3-evaluation figures.

ONE place to edit fonts, colours, sizes, layout and output format. Every figure
script does, right after importing matplotlib:

    import figstyle; figstyle.apply()

and saves with:

    figstyle.save(fig, "path/without/extension")     # writes .pdf (+ .png)

------------------------------------------------------------------------------
Font
------------------------------------------------------------------------------
NAR (Nucleic Acids Research) requires a plain SANS-SERIF figure font (Arial /
Helvetica), no smaller than 5 pt at final reproduction, and <=178 mm wide. Figures
therefore use a sans-serif font (Arial -> Helvetica -> DejaVu Sans fallback),
independent of the OUP manuscript template's serif body font.

Rendering mode -- pick via the FIG_RENDER env var (default "auto"):

    auto   (default) real LaTeX (usetex, Helvetica) IF a latex+dvipng toolchain
                     is on PATH, else a portable sans-serif via mathtext.
    tex              force usetex (Helvetica via helvet + sansmath; needs a toolchain).
    plain            never usetex -- sans-serif (DejaVu Sans), fully portable.
    pgf              pgf backend -> save .pgf and \\input into the document, so the
                     text is typeset by the DOCUMENT in whatever font it uses.
                     figstyle.save() then writes .pgf (+ a .pdf preview).

Everything below the DASHED line is meant to be edited: COLORS, CYCLE, sizes,
column widths. Change a value here and every figure follows on the next run.
"""
from __future__ import annotations
import os
import shutil

import matplotlib

# ============================================================================
#  EDIT HERE  -- colours, sizes, layout
# ============================================================================

# --- Colour scheme ----------------------------------------------------------
# Semantic, named colours used across the paper. Edit these to recolour every
# figure consistently. (Seeded from the Tableau-10 palette already in use.)
COLORS = {
    # categorical cycle base (Tableau 10)
    "blue":   "#4E79A7", "orange": "#F28E2B", "red":    "#E15759",
    "teal":   "#76B7B2", "green":  "#59A14F", "yellow": "#EDC948",
    "purple": "#B07AA1", "pink":   "#FF9DA7", "brown":  "#9C755F",
    "grey":   "#BAB0AC",
    # domain roles
    "rna":       "#4E79A7",   # RNA side
    "protein":   "#F28E2B",   # protein side
    "pre":       "#76B7B2",   # pre-cutoff / training-era
    "post":      "#E15759",   # post-cutoff / held-out
    "positive":  "#4E79A7",   # good / higher accuracy
    "negative":  "#E15759",   # poor / lower accuracy
    "neutral":   "#7F7F7F",
    # variance-partition groups G0..G5
    "G0": "#4E79A7", "G1": "#59A14F", "G2": "#EDC948",
    "G3": "#B07AA1", "G4": "#F28E2B", "G5": "#9C755F",
}

# Categorical colour cycle (order matters -- first N series get these).
CYCLE = [COLORS[k] for k in
         ("blue", "orange", "green", "red", "purple",
          "brown", "pink", "grey", "teal", "yellow")]

# Named colourmaps (keep central so every heatmap/gradient matches).
CMAP_SEQ = "viridis"     # sequential
CMAP_DIV = "RdBu_r"      # diverging (e.g. correlation)

# --- Type sizes (points) ----------------------------------------------------
FONT_BASE   = 8      # base / tick label
FONT_LABEL  = 8      # axis labels
FONT_TITLE  = 9      # axes title
FONT_LEGEND = 7      # legend
FONT_SMALL  = 6.5    # annotations

# Panel letters (A, B, ...) -- one size + weight for EVERY figure, so composites
# and standalone panels match. Use figstyle.panel_letter(ax, "A") to draw them.
PANEL_LETTER_SIZE   = 9
PANEL_LETTER_WEIGHT = "bold"

# --- Line / marker weights --------------------------------------------------
LINEWIDTH   = 1.0
GRIDWIDTH   = 0.5
MARKERSIZE  = 4.0
AXESWIDTH   = 0.8

# --- Figure geometry (inches). NAR limits: single column 84 mm = 3.31 in,
# double column 178 mm = 7.01 in, max height 230 mm = 9.06 in. Generate figures
# AT these widths so LaTeX includes them at scale 1 (fonts keep their pt size;
# NAR requires >= 5 pt at final reproduction).
WIDTH_1COL = 3.31   # 84 mm  single column
WIDTH_1_5  = 5.0
WIDTH_2COL = 7.01   # 178 mm double column
MAX_HEIGHT = 9.06   # 230 mm
GOLDEN     = 0.618

# --- Output -----------------------------------------------------------------
# Vector (pdf/svg) + eps (some journals) + high-dpi png preview. EPS flattens
# transparency and rasterizes gradients, so pdf/svg are the faithful vector pair.
SAVE_FORMATS = ("pdf", "svg", "eps", "png")
DPI          = 600              # raster fallback / png resolution

# --- Directory convention ---------------------------------------------------
# Individual panels land in RAW_DIR; assembled main/supp composites in FINAL_DIR.
# Switching a panel between main and supplement = edit the assembler mapping only.
RAW_DIR   = "raw_plots"
FINAL_DIR = "figures_final"

# ============================================================================
#  ---- machinery below; usually no need to edit ----
# ============================================================================

def _latex_available() -> bool:
    return bool(shutil.which("latex") and (shutil.which("dvipng") or shutil.which("dvips")))


def _resolve_mode() -> str:
    mode = os.environ.get("FIG_RENDER", "auto").strip().lower()
    if mode == "auto":
        return "tex" if _latex_available() else "plain"
    if mode not in ("tex", "plain", "pgf"):
        return "plain"
    return mode


def figsize(width=WIDTH_1COL, ratio=GOLDEN, height=None):
    """(width, height) in inches. Default height = width*golden ratio."""
    return (width, height if height is not None else width * ratio)


def panel_letter(ax, letter, x=-0.02, y=1.04, **kw):
    """Draw a panel label (A, B, ...) at the ONE central size + weight, so every
    figure and composite matches. Placed just outside the top-left of the axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_SIZE, fontweight=PANEL_LETTER_WEIGHT,
            va="bottom", ha="right", **kw)


def apply(mode=None):
    """Install the central style into matplotlib rcParams. Call once per script,
    right after `import matplotlib`/`pyplot`. `mode` overrides FIG_RENDER."""
    mode = (mode or _resolve_mode())

    if mode == "pgf":
        matplotlib.use("pgf")

    rc = {
        # --- fonts (NAR requires a plain sans-serif figure font) ---
        "font.family": "sans-serif",
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_BASE,
        "ytick.labelsize": FONT_BASE,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_TITLE,
        # --- lines / axes ---
        "axes.linewidth": AXESWIDTH,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "grid.linewidth": GRIDWIDTH,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        # --- output ---
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,   # embed TrueType (editable text in Illustrator)
        "ps.fonttype": 42,
    }

    if mode == "tex":
        rc.update({
            "text.usetex": True,
            # NAR figures use sans-serif -> Helvetica (helvet) + sansmath under usetex
            "text.latex.preamble": (r"\usepackage{amsmath}\usepackage{amssymb}"
                                    r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
                                    r"\usepackage{sansmath}\sansmath"),
        })
    elif mode == "pgf":
        # let the DOCUMENT choose the fonts at \input time (rcfonts=False)
        rc.update({
            "text.usetex": False,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:  # plain -- NAR-compliant sans-serif, no LaTeX toolchain needed
        rc.update({
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        })

    matplotlib.rcParams.update(rc)
    return mode


def write_matplotlibrc(path):
    """Emit a matplotlibrc from these central values (portable 'plain' mode:
    CM-like serif, no LaTeX toolchain needed). Point matplotlib at it with the
    MATPLOTLIBRC env var so EVERY figure script inherits the style with no code
    edits. Values come from the constants above -> still one source of truth.
    (For exact usetex / pgf inheritance, a script calls figstyle.apply() instead.)"""
    import os as _os
    cyc = ", ".join(f"'{c.lstrip('#')}'" for c in CYCLE)
    lines = [
        "## AUTO-GENERATED from figstyle.py -- edit figstyle.py, not this file.",
        "font.family: sans-serif",
        "font.sans-serif: Arial, Helvetica, DejaVu Sans",
        "mathtext.fontset: dejavusans",
        "text.usetex: False",
        f"font.size: {FONT_BASE}",
        f"axes.titlesize: {FONT_TITLE}",
        f"axes.labelsize: {FONT_LABEL}",
        f"xtick.labelsize: {FONT_BASE}",
        f"ytick.labelsize: {FONT_BASE}",
        f"legend.fontsize: {FONT_LEGEND}",
        f"figure.titlesize: {FONT_TITLE}",
        f"axes.linewidth: {AXESWIDTH}",
        f"lines.linewidth: {LINEWIDTH}",
        f"lines.markersize: {MARKERSIZE}",
        f"grid.linewidth: {GRIDWIDTH}",
        "axes.grid: False",
        "axes.spines.top: False",
        "axes.spines.right: False",
        f"axes.prop_cycle: cycler('color', [{cyc}])",
        f"savefig.dpi: {DPI}",
        "savefig.bbox: tight",
        "savefig.pad_inches: 0.06",
        "pdf.fonttype: 42",
        "ps.fonttype: 42",
    ]
    d = _os.path.dirname(str(path))
    if d:
        _os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def save(fig, path_no_ext, formats=None, **kw):
    """Save `fig` to `path_no_ext` in every SAVE_FORMATS (or `formats`). In pgf
    mode a .pgf is written (for \\input) plus a .pdf preview."""
    import os as _os
    d = _os.path.dirname(str(path_no_ext))
    if d:
        _os.makedirs(d, exist_ok=True)
    base = str(path_no_ext)
    for ext in (list(formats) if formats else list(SAVE_FORMATS)):
        fig.savefig(f"{base}.{ext}", **kw)
    if matplotlib.get_backend().lower() == "pgf":
        fig.savefig(f"{base}.pgf")
    return base
