"""
Generic Nature-style multi-panel publication figure composer.

Stitches several individual plots — typically 3-6, but 1-8 are supported — into a
single publication-ready figure. Each panel is supplied as a small ``draw(ax)``
callback that renders onto a Matplotlib ``Axes``; the composer takes care of the
shared concerns: grid layout, Nature-style styling (sans-serif, despined axes,
editable text in vector exports), bold panel letters (A, B, C, ...), and writing
both a 300-dpi PNG and a vector PDF.

This is the composition layer that sits on top of ``plots.plotCreator``: any
analysis script in the project that can render its plot onto an ``Axes`` can
contribute a panel, so a single figure may freely mix panels coming from
different scripts (phCorrelation, evaluationMetrics, the rnaCompete analyses ...).

Typical use
-----------
    from plots.publication_figure import PublicationFigure

    fig = PublicationFigure(ncols=2)
    fig.add_panel(draw_panel_a, title="Charged residue occurrences")
    fig.add_panel(draw_panel_b, title="Top mispredicted residues")
    paths = fig.save("ph_residue_composite")     # -> .../ph_residue_composite.{png,pdf}

A ``draw`` callback has signature ``draw(ax)`` and may use the axis freely
(bars, scatter, legend, ...). After it runs, the composer despines the axis and
adds the panel letter and optional title, so callbacks should not set their own
panel letter.
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection

from plots.plotCreator import PlotCreator

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Nature-style defaults: sans-serif, thin spines, editable text in PDF/SVG.
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

# --------------------------------------------------------------------------- #
# Cross-figure font normalization.
#
# Every figure is placed at ~7 in (\textwidth) in the manuscript, but the
# generation canvases range 7-21 in, so identical point sizes render at
# different on-page sizes. To make fonts look uniform across figures we set the
# GENERATION size = target_on_page x (figure_width / 7). Target on-page sizes
# (pt): panel letter 8 (bold), axis label 7.5, tick 7, legend 7, annotation 6.5.
# --------------------------------------------------------------------------- #
PAGE_WIDTH_IN = 7.0
ONPAGE = {"letter": 8.0, "axis": 7.5, "tick": 7.0, "legend": 7.0, "annot": 6.5}


def onpage_to_gen(target_pt, fig_width_in):
    """Generation pt that renders at ``target_pt`` when the figure is scaled to
    PAGE_WIDTH_IN in the paper. Floored so nothing drops below ~5 pt on page."""
    return max(5.0, round(target_pt * fig_width_in / PAGE_WIDTH_IN, 1))


def pub_rc_for_width(fig_width_in, base=None):
    """PUB_RC with axis/tick/legend sizes scaled so they render at the common
    on-page target for a figure generated at ``fig_width_in`` inches wide."""
    rc = dict(PUB_RC if base is None else base)
    rc.update({
        "axes.labelsize": onpage_to_gen(ONPAGE["axis"], fig_width_in),
        "axes.titlesize": onpage_to_gen(ONPAGE["axis"], fig_width_in),
        "xtick.labelsize": onpage_to_gen(ONPAGE["tick"], fig_width_in),
        "ytick.labelsize": onpage_to_gen(ONPAGE["tick"], fig_width_in),
        "legend.fontsize": onpage_to_gen(ONPAGE["legend"], fig_width_in),
    })
    return rc

# n_panels -> (nrows, ncols) when the caller does not pin ``ncols``.
_DEFAULT_GRID = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),
    6: (2, 3),
    7: (2, 4),
    8: (2, 4),
}


# Suffixes that PlotCreator.save_plot appends to a base plot name; stripped so a
# panel can be referenced either by its base name or by its saved-PNG name.
_PANEL_SUFFIXES = (
    "_all_chains", "_single_chain",
    "_+MSA", "_-MSA", "_+RBPtemp", "_-RBPtemp",
)


def normalize_panel_name(name):
    """Strip PlotCreator filter suffixes so 'foo_all_chains' -> 'foo'.

    Lets the composer accept the exact saved-PNG name a user sees on disk.
    """
    out = str(name)
    if out.endswith(".png") or out.endswith(".pdf"):
        out = out.rsplit(".", 1)[0]
    changed = True
    while changed:
        changed = False
        for suf in _PANEL_SUFFIXES:
            if out.endswith(suf):
                out = out[: -len(suf)]
                changed = True
    return out


def sns_violin(ax, data_values, labels, colors=None, *, rotation=0,
               ylabel=None, xlabel=None, hline=None, fontsize=None):
    """Draw a seaborn violin plot onto ``ax`` in the project's standard style.

    Matches plotCreator.get_violin_plot (``sns.violinplot(density_norm='width')``)
    with an inner box, and optionally recolours each violin from ``colors``.
    Empty groups are dropped so colours stay aligned with the kept groups.
    """
    keep = [(d, labels[i], (colors[i] if colors else None))
            for i, d in enumerate(data_values)
            if d is not None and len(d) > 0]
    if not keep:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", color="grey")
        return
    datas, labs, cols = zip(*keep)
    sns.violinplot(data=list(datas), density_norm="width", inner="box",
                   cut=0, ax=ax)
    if any(c is not None for c in cols):
        bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
        for body, col in zip(bodies, cols):
            if col is not None:
                body.set_facecolor(col)
                body.set_edgecolor("black")
                body.set_alpha(0.8)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=rotation,
                       ha="right" if rotation else "center", fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if hline is not None:
        ax.axhline(hline, color="0.4", ls="--", lw=0.8)
    style_axis(ax, grid_axis="y")


def image_draw(path):
    """Return a ``draw(ax)`` that embeds an existing PNG as a panel (axis off).

    Lets the composer assemble panels from plots a script already saved to disk,
    for scripts whose plotting is too coupled to redraw onto a bare Axes.
    """
    def draw(ax):
        try:
            img = plt.imread(path)
        except Exception as exc:
            ax.text(0.5, 0.5, f"image unavailable\n{os.path.basename(path)}",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
            return
        ax.imshow(img)
        ax.axis("off")
    return draw


def saved_png_panels(plot_types, names=None, *, single_chain_only=False,
                     msa_option=None, rbp_template_option=None):
    """Build image panels from PNGs a script already saved via PlotCreator.

    Resolves the results directory for each ``plot_type`` (with the same filter
    options the script used), then maps each saved ``*.png`` to its base panel
    name (suffixes like ``_all_chains`` stripped). Returns
    ``{base_name: (draw_fn, None)}`` for the requested ``names`` (or all found).
    """
    dirs = []
    for pt in plot_types:
        try:
            pc = PlotCreator(pt, msa_option, single_chain_only, rbp_template_option)
            dirs.append(pc.results_dir)
        except Exception:
            continue
    found = {}
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, "*.png"))):
            base = normalize_panel_name(os.path.basename(fp))
            found.setdefault(base, fp)
    wanted = ([normalize_panel_name(n) for n in names] if names else list(found))
    return {b: (image_draw(found[b]), None) for b in wanted if b in found}


def style_axis(ax, grid_axis=None):
    """Apply Nature-style cosmetics to a single axis (despine, outward ticks).

    ``grid_axis`` may be ``"x"``, ``"y"`` or ``"both"`` to add a faint grid.
    Exposed as a helper so panel callbacks can match the composer's look.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.4, alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return ax


class _Panel:
    __slots__ = ("draw", "title", "letter")

    def __init__(self, draw, title=None, letter=None):
        self.draw = draw
        self.title = title
        self.letter = letter


class PublicationFigure:
    """Compose multiple ``draw(ax)`` panels into one Nature-style figure.

    Parameters
    ----------
    ncols : int, optional
        Number of columns. If omitted, a sensible grid is chosen from the panel
        count (see ``_DEFAULT_GRID``).
    panel_size : (float, float)
        Width, height in inches allotted to each panel cell.
    letters : bool
        Draw bold panel letters (A, B, ...). Set False to suppress.
    plot_type : str
        Sub-folder name used when ``outdir`` is not given.
    outdir : str, optional
        Output directory. Defaults to
        ``results/plots/all_chains/<plot_type>`` under the repo root.
    """

    def __init__(self, ncols=None, panel_size=(5.0, 4.0), letters=True,
                 plot_type="publication_figures", outdir=None, width_ratios=None,
                 tight_pad=(2.0, 2.5)):
        self.ncols = ncols
        self.panel_w, self.panel_h = panel_size
        self.letters = letters
        # tight_layout (w_pad, h_pad) in font-size units; lower w_pad packs panels
        # closer together so they fill more horizontal space.
        self.tight_w_pad, self.tight_h_pad = tight_pad
        # optional per-column width ratios (length == ncols). A ratio of 1.0 == one
        # panel_w; e.g. [1.5, 1, 1] makes the first panel 50% wider. The figure width
        # grows with sum(width_ratios) so other panels keep their size.
        self.width_ratios = width_ratios
        self.panels = []
        if outdir is None:
            outdir = os.path.join(_REPO_ROOT, "results", "plots", "all_chains", plot_type)
        self._plotter = PlotCreator(plot_type=plot_type, standalone_outdir=outdir)
        self.results_dir = self._plotter.results_dir

    # -- panel registration -------------------------------------------------- #
    def add_panel(self, draw, title=None, letter=None):
        """Register a panel. ``draw`` must accept a single Matplotlib ``Axes``.

        Returns ``self`` so calls can be chained.
        """
        if not callable(draw):
            raise TypeError("draw must be callable with signature draw(ax)")
        self.panels.append(_Panel(draw, title=title, letter=letter))
        return self

    # -- layout -------------------------------------------------------------- #
    def _grid(self, n):
        if self.ncols:
            ncols = max(1, int(self.ncols))
            nrows = math.ceil(n / ncols)
            return nrows, ncols
        if n in _DEFAULT_GRID:
            return _DEFAULT_GRID[n]
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        return nrows, ncols

    @staticmethod
    def _letter_for(i):
        # A..Z then AA, AB, ... (more than 26 panels is never expected, but safe)
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    # -- build / save -------------------------------------------------------- #
    def build(self):
        """Render all panels and return the Matplotlib ``Figure``."""
        if not self.panels:
            raise ValueError("No panels added; call add_panel(...) first.")
        n = len(self.panels)
        nrows, ncols = self._grid(n)

        wr = self.width_ratios
        total_units = sum(wr) if wr else ncols
        fig_width_in = self.panel_w * total_units
        letter_fs = onpage_to_gen(ONPAGE["letter"], fig_width_in)
        with plt.rc_context(pub_rc_for_width(fig_width_in)):
            fig, axarr = plt.subplots(
                nrows, ncols,
                figsize=(fig_width_in, self.panel_h * nrows),
                squeeze=False,
                gridspec_kw=({"width_ratios": wr} if wr else None),
            )
            axes = axarr.flatten()

            for i, panel in enumerate(self.panels):
                ax = axes[i]
                try:
                    panel.draw(ax)
                except Exception as exc:  # keep the rest of the figure usable
                    import sys
                    print(f"Warning: panel {i} ('{panel.title}') failed: {exc}",
                          file=sys.stderr)
                    ax.text(0.5, 0.5, "panel unavailable", transform=ax.transAxes,
                            ha="center", va="center", color="grey")
                # Despine without clobbering any grid the callback set itself.
                if ax.get_visible():
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                if panel.title:
                    ax.set_title(panel.title)
                if self.letters:
                    letter = panel.letter or self._letter_for(i)
                    ax.text(-0.13, 1.05, letter, transform=ax.transAxes,
                            fontsize=letter_fs, fontweight="bold", va="bottom", ha="right")

            # Hide any unused cells in the last row.
            for j in range(n, len(axes)):
                axes[j].set_visible(False)

            fig.tight_layout(w_pad=self.tight_w_pad, h_pad=self.tight_h_pad)
        return fig

    def save(self, name, dpi=600, formats=("png", "pdf", "svg", "eps")):
        """Build the figure and write it to ``results_dir`` as ``name.<ext>``.

        Returns the list of written file paths.
        """
        fig = self.build()
        os.makedirs(self.results_dir, exist_ok=True)
        paths = []
        for ext in formats:
            out = os.path.join(self.results_dir, f"{name}.{ext}")
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)
        print(f"Publication figure saved: {os.path.join(self.results_dir, name)}.{'/'.join(formats)}")
        return paths
