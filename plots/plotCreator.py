import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import os
import sys
import pandas as pd
from database.startConfig import StartConfig
from utils.plots_dir import plots_root
from database.databaseMethods import DatabaseMethods

# Typography aligned with Nature-style figures: sans-serif, minimum ~7–8 pt when printed
# at column width — use larger source sizes so labels stay legible vs. figure area.
_NATURE_LABEL = 18
_NATURE_TITLE = 21
_NATURE_TICK = 16
_NATURE_LEGEND = 16

# Shared Nature-style colour palettes (muted, colour-blind-friendly Tableau-10
# variant). Use these for any new publication figure so colours stay consistent
# across scripts. NATURE_PH_COLORS is an ordered, perceptually-graded sequence for
# the five pH bins used in the pH-correlation analyses.
NATURE_PALETTE = [
    "#4E79A7",  # blue
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#F1A340",  # orange
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#9C755F",  # brown
]
# Low -> high pH, sequential (cool acidic -> warm basic).
NATURE_PH_COLORS = {
    "<5": "#2C7BB6",
    "5-6": "#5AA7C8",
    "6-7": "#7FB77E",
    "7-8": "#F4A259",
    ">8": "#D7191C",
}
# Interface-quality classes aligned with evaluationMetrics.py RBP_DockQ bins.
NATURE_DOCKQ_CLASS_COLORS = {
    "high": "#1A9850",    # DockQ >= 0.8   (a)
    "medium": "#FDAE61",  # 0.23-0.8       (b)
    "incorrect": "#D73027",  # < 0.23      (c)
}
# Categorical pH-range palette used by the pH-correlation scatter plots
# (get_colour_scatterplot). Reused across phCorrelation.py so every pH-binned
# figure shares one colour key.
PH_RANGE_COLORS = {
    "<5": "#d62728",   # Red
    "5-6": "#1f77b4",  # Blue
    "6-7": "#ff7f0e",  # Orange
    "7-8": "#228B22",  # Green
    ">8": "#9467bd",   # Purple
}
# Two-condition RBP-template-filter palette (+RBP vs -RBP), drawn from the same
# categorical hues as PH_RANGE_COLORS so all figures share one colour key.
RBP_TEMPLATE_COLORS = {
    "+RBP": "#1f77b4",  # Blue (matches PH_RANGE_COLORS '5-6')
    "-RBP": "#d62728",  # Red  (matches PH_RANGE_COLORS '<5')
}

_AUDIT_QUAD_PREFIXES = ("quad_i", "quad_ii", "quad_iii", "quad_iv")
_AUDIT_QUAD_TICKLABELS = (
    "Accurate RNA / Accurate protein",
    "Accurate RNA / Inaccurate protein",
    "Inaccurate RNA / Accurate protein",
    "Inaccurate RNA / Inaccurate protein",
)
# Quadrant scatter colors: i/ii accurate (green/blue), iii/iv inaccurate (pink/purple).
_AUDIT_AF3_QUADRANT_SCATTER_COLORS = {
    "i": "#2ca02c",
    "ii": "#1f77b4",
    "iii": "#e377c2",
    "iv": "#9467bd",
}


def _audit_heatmap_format_cell(v):
    """Compact annotation string for mixed-scale bin means."""
    if not np.isfinite(v):
        return "—"
    x = float(v)
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 1e5 or ax < 1e-4:
        return f"{x:.3g}"
    if ax >= 100:
        return f"{x:.4g}"
    if ax >= 10:
        return f"{x:.4g}"
    if ax >= 1:
        return f"{x:.3g}"
    return f"{x:.4g}"


def _audit_heatmap_format_cell_core(v):
    """Bin mean for core heatmap: at most one decimal, or integer when rounded."""
    if not np.isfinite(v):
        return "—"
    x = float(v)
    xr = round(x, 1)
    if abs(xr - round(xr)) < 1e-9:
        return str(int(round(xr)))
    return f"{xr:.1f}"


def _audit_heatmap_norm_columns(mat, lo_pct=5.0, hi_pct=95.0):
    """
    Per-column robust scaling to [0, 1] for heatmap color only (annotations stay raw).

    Uses percentile bounds across bins within each column so 0–1 scores and 1e4 MSA
    depths are visually comparable. Constant columns map to 0.5.
    """
    out = np.full(mat.shape, np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = np.asarray(mat[:, j], dtype=float)
        mask = np.isfinite(col)
        if not np.any(mask):
            continue
        vals = col[mask]
        lo = np.percentile(vals, lo_pct)
        hi = np.percentile(vals, hi_pct)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[mask, j] = 0.5
            continue
        scaled = (col - lo) / (hi - lo)
        out[:, j] = np.clip(scaled, 0.0, 1.0)
    return out


class PlotCreator:
    def __init__(
        self,
        plot_type,
        msa_option=None,
        single_chain_only=False,
        rbp_template_option=None,
        standalone_outdir=None,
    ):
        """Initialize plot creator with type and filtering options.

        If ``standalone_outdir`` is set, figures are written under that directory only
        (no MSA/chain subfolders, no DB). Filenames are ``{name}.png``. Used for
        ion-site and other one-off exports outside ``results/plots/``.
        """
        self.style_set = False
        if standalone_outdir is not None:
            self._standalone_mode = True
            self.plot_type = plot_type
            self.single_chain_only = False
            self.msa_option = None
            self.msa_folder = None
            self.rbp_template_option = None
            self.rbp_folder = None
            self.results_dir = os.path.abspath(standalone_outdir)
            os.makedirs(self.results_dir, exist_ok=True)
            self.db = None
            return

        self._standalone_mode = False
        self.plot_type = plot_type
        self.single_chain_only = single_chain_only
        
        # Standardize MSA option format - use lowercase for folder names
        if msa_option:
            # Normalize to lowercase for comparison
            msa_lower = msa_option.lower()
            if msa_lower in ['+msa', 'msa']:
                self.msa_option = '+MSA'
                self.msa_folder = '+msa'
            else:
                self.msa_option = '-MSA'
                self.msa_folder = '-msa'
        else:
            self.msa_option = None
            self.msa_folder = None
        
        # Standardize RBP template option format - use lowercase for folder names
        if rbp_template_option:
            rbp_lower = rbp_template_option.lower()
            if rbp_lower in ['+rbptemp', 'rbptemp', '+rbp']:
                self.rbp_template_option = '+RBPtemp'
                self.rbp_folder = '+rbptemp'
            else:
                self.rbp_template_option = '-RBPtemp'
                self.rbp_folder = '-rbptemp'
        else:
            self.rbp_template_option = None
            self.rbp_folder = None
        
        self.style_set = False
        
        # Set up results directory with filter-specific subdirectories
        base_dir = plots_root()   # <--folder>/plots when started by mainExecution.py, else results/plots
        
        # Create subdirectory based on chain, MSA, and RBP template filters
        chain_dir = 'single_chain' if single_chain_only else 'all_chains'
        
        # Build filter folder name combining MSA and RBP options
        filter_parts = []
        if self.msa_folder:
            filter_parts.append(self.msa_folder)  # +msa or -msa
        if self.rbp_folder:
            filter_parts.append(self.rbp_folder)  # +rbptemp or -rbptemp
        
        # Create filter folder name (e.g., "+msa+rbptemp", "-msa-rbptemp", "+msa", etc.)
        if filter_parts:
            filter_folder = ''.join(filter_parts)
            self.results_dir = os.path.join(base_dir, chain_dir, filter_folder, plot_type)
        else:
            self.results_dir = os.path.join(base_dir, chain_dir, plot_type)
            
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize database connection
        self.db = DatabaseMethods()

    def _create_results_dir(self):
        """Create and return appropriate results directory based on plot type and options"""
        # Start with metric type folder
        plot_dir = os.path.join(self.base_dir, self.plot_type)
        
        if self.single_chain_only:
            # For single_chain, create single_chain folder first, then MSA subfolder if needed
            plot_dir = os.path.join(plot_dir, 'single_chain')
            if self.msa_option:
                msa_folder = '+msa' if self.msa_option == '+MSA' else '-msa'
                plot_dir = os.path.join(plot_dir, msa_folder)
        else:
            # For 'all', just add MSA subfolder if needed
            if self.msa_option:
                msa_folder = '+msa' if self.msa_option == '+MSA' else '-msa'
                plot_dir = os.path.join(plot_dir, msa_folder)
            
        # Create directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir

    def set_plot_style(self):
        if not self.style_set:
            plt.rcParams.update({
                # Nature / broad journal practice: Arial-like sans-serif, no bold titles
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Arial",
                    "Helvetica",
                    "Helvetica Neue",
                    "DejaVu Sans",
                    "Liberation Sans",
                    "Nimbus Sans",
                ],
                "font.size": 15,
                "axes.titlesize": _NATURE_TITLE,
                "axes.titleweight": "normal",
                "axes.labelsize": _NATURE_LABEL,
                "axes.labelweight": "normal",
                "axes.titlepad": 14,
                "axes.labelpad": 10,
                "xtick.labelsize": _NATURE_TICK,
                "ytick.labelsize": _NATURE_TICK,
                "legend.fontsize": _NATURE_LEGEND,
                "legend.title_fontsize": _NATURE_LEGEND,
                "figure.titlesize": _NATURE_TITLE,
                "figure.figsize": (7.2, 5.4),
                "figure.dpi": 120,
                "figure.autolayout": True,
                "axes.linewidth": 1.0,
                "xtick.major.width": 0.9,
                "ytick.major.width": 0.9,
                "lines.linewidth": 1.6,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "mathtext.fontset": "dejavusans",
            })
            self.style_set = True

    def save_plot(self, name):
        """Save plot with filter-aware naming"""
        if getattr(self, "_standalone_mode", False):
            filename = f"{name}.png"
        else:
            chain_type = 'single_chain' if self.single_chain_only else 'all_chains'
            msa_type = f'_{self.msa_option}' if self.msa_option else ''
            rbp_type = f'_{self.rbp_template_option}' if self.rbp_template_option else ''
            filename = f'{name}_{chain_type}{msa_type}{rbp_type}.png'
        # print("filename", filename)
        # print("results_dir", self.results_dir)
        plt.savefig(
            os.path.join(self.results_dir, filename),
            bbox_inches="tight",
            dpi=plt.rcParams.get("savefig.dpi", 300),
        )

    def _save_audit_plot(self, name):
        """Save sample-audit figures under ``results_dir/interface``."""
        if getattr(self, "_standalone_mode", False):
            filename = f"{name}.png"
        else:
            chain_type = "single_chain" if self.single_chain_only else "all_chains"
            msa_type = f"_{self.msa_option}" if self.msa_option else ""
            rbp_type = f"_{self.rbp_template_option}" if self.rbp_template_option else ""
            filename = f"{name}_{chain_type}{msa_type}{rbp_type}.png"
        iface_dir = os.path.join(self.results_dir, "interface")
        os.makedirs(iface_dir, exist_ok=True)
        plt.savefig(
            os.path.join(iface_dir, filename),
            bbox_inches="tight",
            dpi=plt.rcParams.get("savefig.dpi", 300),
        )

    def _save_af3_metrics_plot(self, name):
        """Save AF3 metric audit figures under ``results_dir/af3_metrics``."""
        if getattr(self, "_standalone_mode", False):
            filename = f"{name}.png"
        else:
            chain_type = "single_chain" if self.single_chain_only else "all_chains"
            msa_type = f"_{self.msa_option}" if self.msa_option else ""
            rbp_type = f"_{self.rbp_template_option}" if self.rbp_template_option else ""
            filename = f"{name}_{chain_type}{msa_type}{rbp_type}.png"
        out_dir = os.path.join(self.results_dir, "af3_metrics")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            os.path.join(out_dir, filename),
            bbox_inches="tight",
            dpi=plt.rcParams.get("savefig.dpi", 300),
        )

    def plot_af3_audit_category_boxplot(
        self,
        table_source,
        category_labels,
        data_by_category,
        name,
        *,
        ylabel=None,
        ytick_positions=None,
        ytick_labels=None,
        y_axis_limit=None,
        figsize=None,
        rotation=90,
        xlabel=None,
    ):
        """Boxplot of a scalar metric across audit quadrant × DockQ/CRMSD categories."""
        if not category_labels or not data_by_category:
            return
        self.set_plot_style()
        plot_data = []
        plot_pos = []
        n_cat = len(category_labels)
        for i, vals in enumerate(data_by_category):
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            plot_data.append(v)
            plot_pos.append(i + 1)
        if not plot_data:
            return
        w = max(14.0, n_cat * 0.55)
        fig, ax = plt.subplots(figsize=figsize or (w, 6.2))
        ax.boxplot(plot_data, positions=plot_pos, showfliers=True, widths=0.55)
        ax.set_xticks(list(range(1, n_cat + 1)))
        ax.set_xticklabels(
            category_labels,
            rotation=rotation,
            ha="right" if rotation else "center",
            fontsize=_NATURE_TICK - 3,
        )
        if ylabel:
            ax.set_ylabel(ylabel)
        if ytick_positions is not None and ytick_labels is not None:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels)
        if y_axis_limit is not None:
            ax.set_ylim(y_axis_limit)
        ax.set_xlabel(
            xlabel or "Audit category (quadrant × DockQ / CRMSD bin)"
        )
        fig.tight_layout()
        self._save_af3_metrics_plot(f"{name}_{table_source}")
        plt.close(fig)

    def plot_af3_audit_category_mean_std(
        self,
        table_source,
        category_labels,
        means,
        stds,
        name,
        *,
        ylabel=None,
        ytick_positions=None,
        ytick_labels=None,
        y_axis_limit=None,
        figsize=None,
        rotation=90,
        xlabel=None,
    ):
        """Bar chart of category means with sample SD error bars (e.g. binary fractions)."""
        if not category_labels:
            return
        self.set_plot_style()
        positions = []
        bar_means = []
        bar_stds = []
        for i, (m, s) in enumerate(zip(means, stds)):
            if m is None or not np.isfinite(m):
                continue
            positions.append(i + 1)
            bar_means.append(float(m))
            bar_stds.append(float(s) if s is not None and np.isfinite(s) else 0.0)
        if not positions:
            return
        n_cat = len(category_labels)
        w = max(14.0, n_cat * 0.55)
        fig, ax = plt.subplots(figsize=figsize or (w, 6.2))
        ax.bar(
            positions,
            bar_means,
            yerr=bar_stds,
            capsize=4,
            width=0.55,
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_xticks(list(range(1, n_cat + 1)))
        ax.set_xticklabels(
            category_labels,
            rotation=rotation,
            ha="right" if rotation else "center",
            fontsize=_NATURE_TICK - 3,
        )
        if ylabel:
            ax.set_ylabel(ylabel)
        if ytick_positions is not None and ytick_labels is not None:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels)
        if y_axis_limit is not None:
            ax.set_ylim(y_axis_limit)
        ax.set_xlabel(
            xlabel or "Audit category (quadrant × DockQ / CRMSD bin)"
        )
        fig.tight_layout()
        self._save_af3_metrics_plot(f"{name}_{table_source}")
        plt.close(fig)

    def plot_af3_audit_quadrant_scatter(
        self,
        table_source,
        df,
        x_col,
        y_col,
        name,
        *,
        quadrant_col="quadrant",
        quadrant_label_col="quadrant_label",
        xlabel=None,
        ylabel=None,
        title=None,
        figsize=(8.5, 6.5),
        s=36,
        alpha=0.65,
        quadrant_colors=None,
    ):
        """Scatter of two features in ``af3_metrics/``, one color per quadrant (i–iv)."""
        import pandas as pd

        self.set_plot_style()
        d = df.copy()
        d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        need = [x_col, y_col, quadrant_col]
        if quadrant_label_col:
            need.append(quadrant_label_col)
        d = d.dropna(subset=need)
        if d.empty:
            return
        qcolors = quadrant_colors or _AUDIT_AF3_QUADRANT_SCATTER_COLORS
        label_by_roman = {}
        if quadrant_label_col and quadrant_label_col in d.columns:
            for roman, grp in d.groupby(quadrant_col):
                label_by_roman[str(roman)] = str(grp[quadrant_label_col].iloc[0])
        fig, ax = plt.subplots(figsize=figsize)
        for roman in ("i", "ii", "iii", "iv"):
            g = d[d[quadrant_col].astype(str) == roman]
            if g.empty:
                continue
            color = qcolors.get(roman, "#888888")
            legend_lbl = label_by_roman.get(roman, roman)
            ax.scatter(
                g[x_col],
                g[y_col],
                s=s,
                alpha=alpha,
                label=legend_lbl,
                color=color,
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel or y_col)
        if title:
            ax.set_title(title, fontsize=_NATURE_TICK)
        ax.legend(
            title="Quadrant",
            fontsize=_NATURE_TICK - 4,
            title_fontsize=_NATURE_TICK - 3,
            loc="best",
            framealpha=0.9,
        )
        fig.tight_layout()
        self._save_af3_metrics_plot(f"{name}_{table_source}")
        plt.close(fig)

    def scatter_dataframe_hue(
        self,
        df,
        x,
        y,
        hue,
        name,
        xlabel=None,
        ylabel=None,
        title=None,
        *,
        s=12,
        alpha=0.35,
        figsize=(8, 6),
    ):
        """Scatter from a DataFrame, one color per ``hue`` column."""
        self.set_plot_style()
        d = df.copy()
        d[x] = pd.to_numeric(d[x], errors="coerce")
        d[y] = pd.to_numeric(d[y], errors="coerce")
        d = d.dropna(subset=[x, y, hue])
        if d.empty:
            return
        fig, ax = plt.subplots(figsize=figsize)
        for lab, g in d.groupby(hue):
            ax.scatter(g[x], g[y], s=s, alpha=alpha, label=str(lab))
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if title:
            ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        self.save_plot(name)
        plt.close(fig)

    def violin_arrays(
        self, arrays, labels, name, ylabel="", title=None, figsize=None
    ):
        """Side-by-side violin plots for numeric arrays (same order as ``labels``)."""
        self.set_plot_style()
        clean = []
        labs = []
        for a, lb in zip(arrays, labels):
            v = np.asarray(a, dtype=float)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            clean.append(v)
            labs.append(lb)
        if not clean:
            return
        w = max(6.0, len(clean) * 1.3)
        fig, ax = plt.subplots(figsize=(figsize or (w, 5.5)))
        pos = list(range(1, len(clean) + 1))
        ax.violinplot(clean, positions=pos, showmeans=True, showmedians=True)
        ax.set_xticks(pos)
        ax.set_xticklabels(labs)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        self.save_plot(name)
        plt.close(fig)

    def bar_labeled(
        self,
        labels,
        heights,
        name,
        xlabel="",
        ylabel="",
        title=None,
        rotation=0,
        color="steelblue",
    ):
        """Bar chart with string labels on x."""
        self.set_plot_style()
        labels = list(labels)
        heights = list(heights)
        if not labels:
            return
        fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.55), 5.0))
        xpos = np.arange(len(labels))
        ax.bar(xpos, heights, color=color)
        ax.set_xticks(xpos)
        ha = "right" if rotation else "center"
        ax.set_xticklabels(labels, rotation=rotation, ha=ha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        self.save_plot(name)
        plt.close(fig)

    def scatter_xy(self, x, y, name, xlabel="", ylabel="", title=None, **scatter_kw):
        """Single-series scatter (numeric x, y)."""
        self.set_plot_style()
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        y = pd.to_numeric(pd.Series(y), errors="coerce")
        m = ~(x.isna() | y.isna())
        x = x[m].to_numpy()
        y = y[m].to_numpy()
        if x.size < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, **{"s": 20, "alpha": 0.5, **scatter_kw})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        self.save_plot(name)
        plt.close(fig)

    def analyse_correlation(self, name, xAxisScore, yAxisScore):
        print(name)
        length = np.array(yAxisScore)
        score = np.array(xAxisScore)
        try:
            pearson_corr, p_value = stats.pearsonr(length, score)
            print(f"Pearson Correlation Coefficient: {pearson_corr:.2f}, p-value: {p_value:.4f}")
        except ValueError as e:
            print(f"Pearson correlation could not be computed: {e}")

        try:
            from sklearn.linear_model import LinearRegression

            length_reshape = length.reshape(-1, 1)  # Reshape for sklearn
            model = LinearRegression().fit(length_reshape, score)
            r_squared = model.score(length_reshape, score)
            print(f"R² (Coefficient of Determination): {r_squared:.2f}")
        except ValueError as e:
            print(f"R² could not be computed: {e}")

        try:
            spearman_corr, spearman_p_value = stats.spearmanr(length, score)
            print(f"Spearman Rank Correlation: {spearman_corr:.2f}, p-value: {spearman_p_value:.4f}")
        except ValueError as e:
            print(f"Spearman correlation could not be computed: {e}")

        # return {
        #     "pearson_corr": pearson_corr,
        #     "pearson_p_value": p_value,
        #     "r_squared": r_squared,
        #     "spearman_corr": spearman_corr,
        #     "spearman_p_value": spearman_p_value
        # }

    def scatter_plot(self, x, y, ax=None, xlabel=None, ylabel=None, title=None):
        if ax is None:
            ax = plt.gca()
            
        # Convert to numeric arrays
        x = pd.to_numeric(x, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.5)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        # Add trend line and calculate correlation only if we have valid data
        if len(x) > 1 and len(y) > 1:
            try:
                # Calculate correlation
                correlation = np.corrcoef(x, y)[0, 1]
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8)
                
                # Add correlation to legend
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {correlation:.2f}",
                    transform=ax.transAxes,
                    fontsize=_NATURE_LEGEND,
                    bbox=dict(facecolor="white", alpha=0.8),
                    verticalalignment="top",
                )
                
            except np.linalg.LinAlgError:
                print("Could not compute trend line")
        
        return ax

    def get_scatterplot(self, table_source, xAxis_score, xAxis_label, yAxis_label, name, yAxis_score=None, yAxis_limit=None,
                        xAxis_limit=None):
        self.set_plot_style()
        if yAxis_score is None:
            if not xAxis_score:
                plt.close()
                return
            try:
                id, yAxis_score, xAxis_score = zip(*xAxis_score)
            except ValueError:
                plt.close()
                return

        # Convert to numeric arrays and filter out invalid values
        x_values = []
        y_values = []
        for x, y in zip(xAxis_score, yAxis_score):
            try:
                x_val = float(x) if x != '' else float('nan')
                y_val = float(y) if y != '' else float('nan')
                if not (np.isnan(x_val) or np.isnan(y_val)):
                    x_values.append(x_val)
                    y_values.append(y_val)
            except (ValueError, TypeError):
                continue

        if x_values and y_values:  # Only plot if we have valid data
            plt.scatter(x_values, y_values, color='blue', alpha=0.5)
            if yAxis_limit:
                plt.ylim(yAxis_limit)
            if xAxis_limit:
                plt.xlim(xAxis_limit)
            plt.xlabel(f"{xAxis_label}")
            plt.ylabel(f"{yAxis_label}")

            # Remove table_source from title if it's af3_metrics or pae_metrics
            if table_source in ['af3_metrics', 'pae_metrics']:
                title = name
            else:
                title = f"{name}_{table_source}"

            self.save_plot(title)
        plt.close()
        # self.set_plot_style()
        # if yAxis_score is None:
        #     id, yAxis_score, xAxis_score = zip(*xAxis_score)
        # # self.analyse_correlation(yAxis_label, xAxis_score, yAxis_score)
        #
        # plt.scatter(xAxis_score, yAxis_score, color='blue', alpha=0.5)
        # if yAxis_limit:
        #     plt.ylim(yAxis_limit)
        # plt.xlabel(f"{xAxis_label}")
        # plt.ylabel(f"{yAxis_label}")
        #
        # # Remove table_source from title if it's af3_metrics or pae_metrics
        # if table_source in ['af3_metrics', 'pae_metrics']:
        #     title = name
        # else:
        #     title = f"{name}_{table_source}"
        #
        # self.save_plot(title)
        # plt.close()

    def get_violin_plot(self, table_source, data_values, labels, name, yAxis_label=None, xAxis_label=None, rotation=0):
        """
        Create a violin plot
        Args:
            table_source: source of the data
            data_values: list of data arrays to plot
            labels: list of labels for x-axis
            name: name for the plot file
            yAxis_label: label for y-axis (optional)
            rotation: rotation angle for x-axis labels (default: 0)
        """
        self.set_plot_style()
        self.data_values = [np.array(values) for values in data_values]
        sns.violinplot(data=data_values, density_norm='width')

        plt.xticks(range(len(labels)), labels, rotation=rotation)
        if yAxis_label:
            plt.ylabel(yAxis_label)
        if xAxis_label:
            plt.xlabel(xAxis_label)

        plt.tight_layout()

        # Remove table_source from title if it's af3_metrics or pae_metrics
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_boxplot(self, table_source, data_values, labels, name, yAxis_limit=None, yAxis_label=None, xAxis_label=None, rotation=0):
        self.set_plot_style()
        self.data_values = [np.array(values) for values in data_values]
        sns.boxplot(data=data_values)

        plt.xticks(range(len(labels)), labels, rotation=rotation)
        if yAxis_label:
            plt.ylabel(yAxis_label)
        if xAxis_label:
            plt.xlabel(xAxis_label)
        if yAxis_limit:
            plt.ylim(yAxis_limit)

        title = f"{name}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_histogram(self, table_source, values, name, xAxis_label=None, yAxis_label="Count", vlines=None):
        """Histogram; vlines is iterable of x positions (e.g. [0] for Delta=0)."""
        self.set_plot_style()
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            plt.close()
            return
        plt.figure(figsize=(7.2, 5.4))
        sns.histplot(arr, kde=True, stat="count", edgecolor="white")
        if vlines:
            for xv in vlines:
                if xv is not None and np.isfinite(xv):
                    plt.axvline(float(xv), color="#C62828", linestyle="--", linewidth=1.2, alpha=0.85)
        plt.xlabel(xAxis_label or name)
        plt.ylabel(yAxis_label)
        plt.tight_layout()
        title = f"{name}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_boxplot_group(self, table_source, output_df, column, group_by, name, yAxis_limit=None, yAxis_label=None):
        self.set_plot_style()
        sns.boxplot(data=output_df, x=group_by, y=column)
        plt.grid(False)
        plt.xlabel(group_by.replace("_", " ").title())
        if yAxis_label:
            plt.ylabel(yAxis_label)
        if yAxis_limit:
            plt.ylim(yAxis_limit)

        title = f"{name}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_barplot(self, table_source, data, title, xlabel, ylabel, bar_color='#FF8F00', rotation=90): #FF8F00 48A860
        self.set_plot_style()
        sorted_data = dict(sorted(data.items()))
        labels = list(sorted_data.keys())
        values = list(sorted_data.values())
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=bar_color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        underscore_title = title.replace(' ', '_').replace('/', '_')
        title = f"{underscore_title}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_grouped_barplot(self, table_source, groups, group_labels, group_width=0.35, colors=None, xlabel=None, ylabel=None,
                               title=None, rotation=90, legend_labels=None, yaxis_range=None):
        self.set_plot_style()
        indices = np.arange(len(group_labels))
        num_series = len(groups)
        colors = colors or plt.cm.tab10(range(num_series))  # Default to a range of colors
        offsets = [i * group_width for i in range(num_series)]

        for i, group in enumerate(groups):
            plt.bar(
                indices + offsets[i] - group_width * (num_series - 1) / 2,
                group,
                group_width,
                label=legend_labels[i] if legend_labels else None,
                color=colors[i % len(colors)]
            )

        plt.xlabel(xlabel if xlabel else '')
        plt.ylabel(ylabel if ylabel else '')
        plt.xticks(indices, group_labels, rotation=rotation)
        if legend_labels:
            plt.legend()
        if yaxis_range:
            plt.ylim(yaxis_range[0], yaxis_range[1])
        plt.tight_layout()
        underscore_title = title.replace(' ', '_').replace('/', '_')
        title = f"{underscore_title}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_heatmap(self, dataframes, x_labels, y_labels, titles):
        self.set_plot_style()
        if not (len(dataframes) == len(x_labels) == len(y_labels) == len(titles)):
            raise ValueError("All input lists must have the same length.")

        for i, df in enumerate(dataframes):
            plt.figure(figsize=figure_size)
            sns.barplot(x=df.iloc[:, 0], y=df.iloc[:, 1], data=df)
            plt.xlabel(x_labels[i])
            plt.ylabel(y_labels[i])
            plt.title(titles[i])
            plt.xticks(rotation=90)
            plt.tight_layout()

        self.save_plot(titles[0])
    
    def get_correlation_heatmap(self, table_source, corr_matrix, name, title=None, 
                                figsize=(24, 20), annot=True, cmap='coolwarm', center=0, 
                                fmt='.1f', square=True, linewidths=0.5, cbar_shrink=0.5,
                                n_samples=None):
        """
        Create a correlation matrix heatmap using seaborn.
        
        Args:
            table_source: source of the data
            corr_matrix: pandas DataFrame with correlation matrix
            name: plot name for saving
            title: optional title for the plot (if None, uses name)
            figsize: figure size tuple
            annot: whether to annotate cells with values
            cmap: colormap name
            center: center value for colormap
            fmt: format string for annotations
            square: whether to make cells square
            linewidths: width of lines between cells
            cbar_shrink: shrink factor for colorbar
            n_samples: optional number of samples to include in title
        """
        self.set_plot_style()
        plt.figure(figsize=figsize)
        
        sns.heatmap(corr_matrix, annot=annot, cmap=cmap, center=center, fmt=fmt,
                   square=square, linewidths=linewidths, 
                   cbar_kws={"shrink": cbar_shrink})
        
        if title:
            plot_title = title
        else:
            plot_title = f'Correlation Matrix - {name}'
        
        if n_samples is not None:
            plot_title += f' (n={n_samples})'
        
        plt.title(plot_title, pad=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        self.save_plot(name)
        plt.close()

    def filter_data(self, table_name):
        """Filter data based on MSA and chain options"""
        base_query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        # Add MSA filter
        if self.msa_option == '+MSA':
            base_query += " AND (af3_rna_MSA = '[1]' OR af3_rna_MSA LIKE '[1,1%')"
        elif self.msa_option == '-MSA':
            base_query += " AND (af3_rna_MSA = '[0]' OR af3_rna_MSA LIKE '%0%')"
        
        # Add chain filter
        if self.single_chain_only:
            base_query += """ 
                AND (
                    (json_array_length(ProteinChainIDs) = 1 OR 
                     length(ProteinChainIDs) - length(replace(ProteinChainIDs, ',', '')) = 0)
                    AND 
                    (json_array_length(RNAChainIDs) = 1 OR 
                     length(RNAChainIDs) - length(replace(RNAChainIDs, ',', '')) = 0)
                )
            """
        
        return pd.read_sql_query(base_query, self.database.connection)

    def get_filtered_values(self, table_name, columns, base_condition=None):
        """
        Get filtered values for one or more columns using DatabaseMethods
        Args:
            table_name: name of the table to query
            columns: single column name (str) or list of column names
            base_condition: additional WHERE conditions (optional)
        """
        conditions = []
        params = []
        
        # Handle columns parameter
        if isinstance(columns, list):
            column_names = ", ".join(columns)
        else:
            column_names = columns
        
        # Check if MSA data exists
        if self.msa_option:
            msa_check_query = f"SELECT COUNT(*) FROM {table_name} WHERE af3_rna_MSA IS NOT NULL"
            self.db.cursor.execute(msa_check_query)
            msa_count = self.db.cursor.fetchone()[0]
            if msa_count == 0:
                raise ValueError("No MSA data available in the database")
        
        # Add MSA filter
        if self.msa_option == '+MSA':
            conditions.append("(af3_rna_MSA = '[1]' OR af3_rna_MSA LIKE '[1,1%')")
        elif self.msa_option == '-MSA':
            conditions.append("(af3_rna_MSA = '[0]' OR af3_rna_MSA LIKE '%0%')")
        
        # Add RBP template filter
        if self.rbp_template_option:
            # Check if RBP_PDB_Templates column exists in the table
            try:
                rbp_check_query = f"PRAGMA table_info({table_name})"
                self.db.cursor.execute(rbp_check_query)
                columns = [row[1] for row in self.db.cursor.fetchall()]
                
                if 'RBP_PDB_Templates' in columns:
                    if self.rbp_template_option == '+RBPtemp':
                        # Has templates: not empty, not NULL, not '[]'
                        conditions.append("(RBP_PDB_Templates IS NOT NULL AND RBP_PDB_Templates != '' AND RBP_PDB_Templates != '[]' AND RBP_PDB_Templates != 'null')")
                    elif self.rbp_template_option == '-RBPtemp':
                        # No templates: NULL, empty, or '[]'
                        conditions.append("(RBP_PDB_Templates IS NULL OR RBP_PDB_Templates = '' OR RBP_PDB_Templates = '[]' OR RBP_PDB_Templates = 'null')")
            except:
                pass  # If column doesn't exist, skip RBP template filtering
        
        # Add chain filter
        if self.single_chain_only:
            conditions.append("""
                (
                    (
                        ProteinChainIDs LIKE '[_]' 
                        OR length(ProteinChainIDs) = 1
                        OR ProteinChainIDs LIKE '["_"]'
                    )
                    AND 
                    (
                        RNAChainIDs LIKE '[_]'
                        OR length(RNAChainIDs) = 1
                        OR RNAChainIDs LIKE '["_"]'
                    )
                )
            """)
        
        # Build query
        query = f"SELECT {column_names} FROM {table_name}"
        
        # Add conditions
        where_conditions = []
        if base_condition:
            where_conditions.append(f"({base_condition})")
        if conditions:
            where_conditions.extend(conditions)
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        # Execute query through DatabaseMethods
        self.db.cursor.execute(query, params)
        rows = self.db.cursor.fetchall()
        
        return rows

    def get_filtered_pdb_ids(self, pred_table_name):
        """
        Get filtered PDB IDs based on MSA, chain, and RBP template options
        Args:
            pred_table_name: name of the prediction table
        Returns:
            list of filtered PDB IDs
        """
        conditions = []
        params = []
        
        # Check if MSA data exists
        if self.msa_option:
            msa_check_query = f"SELECT COUNT(*) FROM {pred_table_name} WHERE af3_rna_MSA IS NOT NULL"
            self.db.cursor.execute(msa_check_query)
            msa_count = self.db.cursor.fetchone()[0]
            if msa_count == 0:
                raise ValueError("No MSA data available in the database")
        
        # Add MSA filter
        if self.msa_option == '+MSA':
            conditions.append("(af3_rna_MSA = '[1]' OR af3_rna_MSA LIKE '[1,1%')")
        elif self.msa_option == '-MSA':
            conditions.append("(af3_rna_MSA = '[0]' OR af3_rna_MSA LIKE '%0%')")
        
        # Add RBP template filter
        if self.rbp_template_option:
            # Check if RBP_PDB_Templates column exists
            rbp_check_query = f"PRAGMA table_info({pred_table_name})"
            self.db.cursor.execute(rbp_check_query)
            columns = [row[1] for row in self.db.cursor.fetchall()]
            
            if 'RBP_PDB_Templates' in columns:
                if self.rbp_template_option == '+RBPtemp':
                    # Has templates: not empty, not NULL, not '[]'
                    conditions.append("(RBP_PDB_Templates IS NOT NULL AND RBP_PDB_Templates != '' AND RBP_PDB_Templates != '[]' AND RBP_PDB_Templates != 'null')")
                elif self.rbp_template_option == '-RBPtemp':
                    # No templates: NULL, empty, or '[]'
                    conditions.append("(RBP_PDB_Templates IS NULL OR RBP_PDB_Templates = '' OR RBP_PDB_Templates = '[]' OR RBP_PDB_Templates = 'null')")
        
        # Add chain filter
        if self.single_chain_only:
            conditions.append("""
                (
                    (
                        ProteinChainIDs LIKE '[_]' 
                        OR length(ProteinChainIDs) = 1
                        OR ProteinChainIDs LIKE '["_"]'
                    )
                    AND 
                    (
                        RNAChainIDs LIKE '[_]'
                        OR length(RNAChainIDs) = 1
                        OR RNAChainIDs LIKE '["_"]'
                    )
                )
            """)
        
        # Build query
        query = f"SELECT DISTINCT exp_db_id FROM {pred_table_name}"
        
        # Add conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute query through DatabaseMethods
        self.db.cursor.execute(query, params)
        rows = self.db.cursor.fetchall()
        
        # Return list of PDB IDs
        return [row[0] for row in rows if row[0]]

    def get_bubble_scatter(self, table_source, xAxis_score, yAxis_score, xAxis_label, yAxis_label, name, pdb_ids=None):
        """Create a bubble scatter plot with size and color based on point frequency"""
        plt.figure(figsize=(12, 10))  # Increased figure size for legend

        # Create a dictionary to count occurrences of each (x,y) pair
        from collections import Counter
        points = Counter(zip(xAxis_score, yAxis_score))

        # Separate the unique points and their counts
        unique_points = np.array(list(points.keys()))
        counts = np.array(list(points.values()))

        # Create discrete colormap with as many colors as we have different counts
        unique_counts = sorted(set(counts))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_counts)))  # Use tab20 for discrete colors

        # Create color mapping
        color_dict = dict(zip(unique_counts, colors))

        # Create scatter plot with different colors for different counts
        for count in unique_counts:
            mask = counts == count
            points_with_count = unique_points[mask]
            plt.scatter(points_with_count[:, 0], points_with_count[:, 1],
                        s=50 * count, c=[color_dict[count]], alpha=0.6,
                        label=f'Count: {count}')

        # Add labels and title
        plt.xlabel(xAxis_label)
        plt.ylabel(yAxis_label)

        # Add diagonal line for reference
        max_val = max(max(unique_points[:, 0]), max(unique_points[:, 1]))
        min_val = min(min(unique_points[:, 0]), min(unique_points[:, 1]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        # Add legend with better positioning
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add PDB IDs if provided
        if pdb_ids:
            for point, ids in zip(unique_points, pdb_ids):
                if abs(point[0] - point[1]) >= 1:  # Only annotate points where difference >= 1
                    plt.annotate(f'{ids}', (point[0], point[1]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=_NATURE_TICK,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Adjust layout with specific margins
        plt.subplots_adjust(right=0.85)  # Make room for legend
        self.save_plot(name)
        plt.close()

    def get_3d_scatter(self, table_source, xAxis_score, yAxis_score, zAxis_score, xAxis_label, yAxis_label, zAxis_label,
                       name):
        """Create a 3D scatter plot"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a dictionary to count occurrences of each (x,y) pair
        from collections import Counter
        points = Counter(zip(xAxis_score, yAxis_score))

        # Convert to arrays for plotting
        unique_points = np.array(list(points.keys()))
        counts = np.array(list(points.values()))

        # Get z values for each unique point
        z_values = []
        for x, y in unique_points:
            # Find indices where this x,y pair occurs
            indices = [i for i, (xs, ys) in enumerate(zip(xAxis_score, yAxis_score)) if xs == x and ys == y]
            # Take mean of z values at these indices
            z_values.append(np.mean([zAxis_score[i] for i in indices]))

        z_values = np.array(z_values)

        # print("\n3D Plot Data Summary:")
        # print(f"Total unique points: {len(unique_points)}")
        # print(f"X range: {min(unique_points[:, 0]):.2f} to {max(unique_points[:, 0]):.2f}")
        # print(f"Y range: {min(unique_points[:, 1]):.2f} to {max(unique_points[:, 1]):.2f}")
        # print(f"Z range: {min(z_values):.2f} to {max(z_values):.2f}")
        # print("\nPoint distribution:")
        # for (x, y), count, z in zip(unique_points, counts, z_values):
        #     print(f"Exp count: {x}, Pred count: {y}, Frequency: {count}, Avg Length: {z:.2f}")

        # Create scatter plot with size indicating frequency
        scatter = ax.scatter(unique_points[:, 0],
                             unique_points[:, 1],
                             z_values,
                             s=50 * (counts / counts.min()),  # Size by frequency
                             c=z_values,  # Color by z-value
                             cmap='viridis',
                             alpha=0.6)

        # Set axis limits with some padding
        ax.set_xlim(0, max(unique_points[:, 0]) + 0.5)
        ax.set_ylim(0, max(unique_points[:, 1]) + 0.5)
        ax.set_zlim(0, max(z_values) + 0.5)

        # Set integer ticks for x and y axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add labels
        ax.set_xlabel(xAxis_label)
        ax.set_ylabel(yAxis_label)
        ax.set_zlabel(zAxis_label)

        # Add colorbar
        plt.colorbar(scatter, label='Average Stem Length')

        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Save plot
        plt.tight_layout()
        self.save_plot(name)
        plt.close()

    def get_stem_length_count_plot(self, table_source, exp_counts, exp_lengths,
                                   pred_counts, pred_lengths, name, feature_name='Stem'):
        """Create a scatter plot comparing counts vs average lengths"""
        plt.figure(figsize=(12, 6))

        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot experimental data
        scatter1 = ax1.scatter(exp_counts, exp_lengths,
                               alpha=0.6, c='blue', label='Experimental')
        ax1.set_xlabel(f'Number of {feature_name}s')
        ax1.set_ylabel(f'{feature_name} Length')
        ax1.set_title('Experimental Structures')

        # Plot predicted data
        scatter2 = ax2.scatter(pred_counts, pred_lengths,
                               alpha=0.6, c='red', label='Predicted')
        ax2.set_xlabel(f'Number of {feature_name}s')
        ax2.set_ylabel(f'{feature_name} Length')
        ax2.set_title('Predicted Structures')

        # Add count annotations in top right corner
        ax1.text(0.98, 0.98, f'n = {len(exp_counts)}',
                 transform=ax1.transAxes,
                 va='top', ha='right',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        ax2.text(0.98, 0.98, f'n = {len(pred_counts)}',
                 transform=ax2.transAxes,
                 va='top', ha='right',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.tight_layout()
        self.save_plot(name)
        plt.close()

    def get_segmented_barplot(self, table_source, data_dict, category_labels, segment_labels,
                              xlabel=None, ylabel=None, name=None, rotation=0):
        """
        Create a segmented bar plot where each bar shows the distribution of segments

        Args:
            table_source: source of the data
            data_dict: dictionary with categories as keys and lists of segment counts as values
            category_labels: labels for the bars (x-axis)
            segment_labels: labels for the segments within each bar
            xlabel: label for x-axis
            ylabel: label for y-axis
            name: name for saving the plot
            rotation: rotation angle for x-axis labels
        """
        self.set_plot_style()

        # Prepare data for stacking
        data_matrix = np.zeros((len(segment_labels), len(category_labels)))
        for i, category in enumerate(category_labels):
            counts = data_dict[category]
            for j, segment in enumerate(segment_labels):
                data_matrix[j, i] = counts.count(segment)

        # Create the stacked bar plot
        plt.figure(figsize=(10, 6))
        bottom = np.zeros(len(category_labels))

        # Create color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_labels)))

        bars = []
        for i, row in enumerate(data_matrix):
            bar = plt.bar(category_labels, row, bottom=bottom, color=colors[i])
            bottom += row
            bars.append(bar)

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.xticks(rotation=rotation)

        # Add legend
        plt.legend(bars, [f'{s} Domain{"s" if s > 1 else ""}' for s in segment_labels],
                   title='Domain Count', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if name:
            self.save_plot(name)
        plt.close()

    def create_spider_plot(self, correlations_dict, title, filter_type, use_custom_colors=False):
        """
        Create a spider plot comparing correlations across different subsets.
        
        Args:
            correlations_dict (dict): Dictionary mapping subset names to either:
                - lists of (variable_pair, correlation) tuples (if use_custom_colors=False)
                - dict with 'correlations' and 'color' keys (if use_custom_colors=True)
            title (str): Title for the plot
            filter_type (str): Type of filter being used (e.g., 'rna_length', 'domain', etc.)
            use_custom_colors (bool): Whether to use custom colors from the correlations_dict
        """
        # Set up the figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='polar')

        # Get all unique correlation pairs
        all_pairs = set()
        for subset_data in correlations_dict.values():
            if use_custom_colors:
                subset_correlations = subset_data['correlations']
            else:
                subset_correlations = subset_data
            all_pairs.update(pair for pair, _ in subset_correlations)
        
        # Convert to list and sort for consistent ordering
        all_pairs = sorted(list(all_pairs))
        num_vars = len(all_pairs)

        # Calculate angles for each variable
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle

        # Set up colors if not using custom colors
        if not use_custom_colors:
            colors = plt.cm.tab10(np.linspace(0, 1, len(correlations_dict)))
            color_dict = dict(zip(correlations_dict.keys(), colors))

        # Plot data for each subset
        for subset_name, subset_data in correlations_dict.items():
            if use_custom_colors:
                color = subset_data['color']
                subset_correlations = subset_data['correlations']
            else:
                color = color_dict[subset_name]
                subset_correlations = subset_data
            
            # Create a mapping of correlation pairs to their values
            pair_to_value = dict(subset_correlations)
            
            # Get values for all pairs
            values = [abs(pair_to_value[pair]) for pair in all_pairs]
            values += values[:1]  # Complete the circle
            
            # Plot the subset data
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=subset_name, alpha=0.7)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Set chart properties
        ax.set_xticks(angles[:-1])
        # Format labels to show only variable pairs
        labels = [f"{pair[0]}\n{pair[1]}" for pair in all_pairs]
        
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        plt.title(title, pad=20, size=14)

        # Add legend
        plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.tight_layout()

        # Save the plot with a unique name based on the filter type
        self.save_plot(f'spider_plot_{filter_type}_correlations')

    def get_difference_scatterplot(self, table_source, xAxis_score, yAxis_score, xAxis_label, yAxis_label, name, diff_values):
        """
        Create a scatter plot with points color-coded based on the difference between predicted and reference values.
        
        Args:
            table_source: source of the data
            xAxis_score: x-axis values
            yAxis_score: y-axis values
            xAxis_label: label for x-axis
            yAxis_label: label for y-axis
            name: name for saving the plot
            diff_values: array of differences between predicted and reference values
        """
        self.set_plot_style()
        
        # Convert inputs to numpy arrays and get absolute differences
        x_values = np.array(xAxis_score)
        y_values = np.array(yAxis_score)
        diff_values = np.array(diff_values)
        abs_diff = np.abs(diff_values)
        
        # Create color mask based on whether predicted > reference
        pred_larger = diff_values > 0
        
        plt.figure(figsize=(10, 8))
        
        # Plot points where predicted is larger in one color
        plt.scatter(x_values[pred_larger], y_values[pred_larger], 
                   c='red', alpha=0.6, label='Predicted > Reference')
        
        # Plot points where reference is larger in another color
        plt.scatter(x_values[~pred_larger], y_values[~pred_larger], 
                   c='blue', alpha=0.6, label='Reference > Predicted')
        
        plt.xlabel(xAxis_label)
        plt.ylabel(yAxis_label)
        plt.legend()
        
        # Add correlation coefficient
        if len(x_values) > 1:
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        self.save_plot(name)
        plt.close()

    def get_msa_scatterplot(self, table_source, xAxis_score, xAxis_label, name, yAxis_label, yAxis_score, color_by,
                               color_map, legend_title=None, custom_save_path=None):
        """
        Create scatter plot with points colored by a given category.
        
        Args:
            table_source: source of the data
            xAxis_score: x-axis values
            xAxis_label: label for x-axis
            name: plot name for saving
            yAxis_label: label for y-axis
            yAxis_score: y-axis values
            color_by: list of category values for each point
            color_map: dictionary mapping category values to colors
            legend_title: optional title for the legend
            custom_save_path: optional custom path to save the plot (if None, uses default save_plot)
        """
        self.set_plot_style()
        plt.figure(figsize=(10, 8))

        # Create a DataFrame for easy plotting
        plot_df = pd.DataFrame({
            'x': xAxis_score,
            'y': yAxis_score,
            'category': color_by
        })

        handles = []
        for category, color in color_map.items():
            subset = plot_df[plot_df["category"] == category]
            if not subset.empty:
                # Convert color to proper format if it's an array
                plot_color = color
                if isinstance(color, (list, tuple, np.ndarray)) and len(color) > 1:
                    # If it's an RGBA array, convert to tuple
                    if len(color) == 4:
                        plot_color = tuple(color[:3])  # Use RGB only
                    else:
                        plot_color = tuple(color) if len(color) == 3 else color
                plt.scatter(subset['x'], subset['y'], c=plot_color, label=category, alpha=0.7)
                handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', label=category, 
                              markerfacecolor=plot_color, markersize=10))

        plotted_cats = set(color_map.keys())
        other = plot_df[~plot_df["category"].isin(plotted_cats)]
        if not other.empty:
            plt.scatter(other["x"], other["y"], c="gray", label="other", alpha=0.5, s=20)

        plt.xlabel(xAxis_label)
        plt.ylabel(yAxis_label)
        if legend_title:
            plt.legend(handles=handles, title=legend_title)
        else:
            plt.legend(handles=handles)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save plot
        if custom_save_path:
            plt.savefig(custom_save_path, bbox_inches='tight')
        else:
            self.save_plot(name)
        plt.close()

    def get_colour_scatterplot(self, table_source, xAxis_score, xAxis_label, name, yAxis_label, yAxis_score, pH_ranges, lim = None):
        """Create scatter plot with points colored by pH range"""
        plt.figure(figsize=(10, 8))
        
        # Shared categorical pH-range palette (single source of truth).
        colors = PH_RANGE_COLORS

        # Create scatter plot with discrete pH ranges
        for pH_range in sorted(set(pH_ranges)):
            mask = [r == pH_range for r in pH_ranges]
            if any(mask):  # Only plot if there are points in this pH range
                plt.scatter([x for i, x in enumerate(xAxis_score) if mask[i]],
                          [y for i, y in enumerate(yAxis_score) if mask[i]],
                          label=f'pH {pH_range}',
                          color=colors.get(pH_range, 'gray'),
                          alpha=0.8,
                          s=50)
        
        # # Add trend line
        # sns.regplot(x=xAxis_score, y=yAxis_score,
        #            scatter=False,
        #            line_kws={'color': 'red', 'alpha': 0.5})
        
        plt.title(f'{yAxis_label} vs {xAxis_label}')
        plt.xlabel(xAxis_label)
        plt.ylabel(yAxis_label)
        plt.legend()

        if lim:
            plt.xlim(lim)
            plt.ylim(lim)
        
        # Save plot
        self.save_plot(name)
        plt.close()

    def get_combined_barplot(self, table_source, data1, data2, labels, title, xlabel, ylabel, name):
        """Create a bar plot comparing two sets of data side by side"""
        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width/2, data1, width, label='Predicted')
        plt.bar(x + width/2, data2, width, label='Experimental')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x, labels, rotation=45)
        plt.legend()

        self.save_plot(name)
        plt.close()

    def get_range_barplot(self, table_source, data_dict, labels, title, xlabel, ylabel, name):
        """Create a grouped bar plot comparing predicted vs experimental data across pH ranges"""
        plt.figure(figsize=(15, 8))
        
        # Shared categorical pH-range palette: each pH keeps one colour; predicted
        # bars are solid, experimental bars use the same colour hatched + outlined.
        ph_colors = PH_RANGE_COLORS

        # Set up the plot
        x = np.arange(len(labels)) * 2  # Double the spacing between residue types
        width = 0.1  # Narrower bars
        n_ranges = len(data_dict)
        spacing = 0.005  # Space between predicted and experimental bars
        group_spacing = 0.1  # Increased space between different pH range groups

        # Plot bars for each pH range
        for i, (pH_range, data) in enumerate(data_dict.items()):
            # Calculate position for this pH range's bars
            pos = x + (i - (n_ranges-1)/2) * (width * 2 + spacing + group_spacing)
            color = ph_colors.get(pH_range, 'gray')

            # Plot predicted bars (solid)
            plt.bar(pos, data['pred'], width,
                   label=f'Predicted (pH {pH_range})',
                   color=color, alpha=0.9)

            # Plot experimental bars (same pH colour, hatched + outlined)
            plt.bar(pos + width + spacing, data['exp'], width,
                   label=f'Experimental (pH {pH_range})',
                   color=color, alpha=0.5, hatch='//', edgecolor='black',
                   linewidth=0.5)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x, labels, rotation=45)
        
        # Create a custom legend that groups by pH range
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(right=0.85)  # Make room for legend
        
        # Save the plot
        self.save_plot(name)
        plt.close()

    def get_3d_scatterplotA(self, table_source, xAxis_score, yAxis_score, zAxis_score, xAxis_label, yAxis_label,
                           zAxis_label, name, xAxis_limit=None, yAxis_limit=None, zAxis_limit=None):
        """Create and save a 3D scatter plot."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xAxis_score, yAxis_score, zAxis_score, c='b', marker='o', alpha=0.7)
        ax.set_xlabel(xAxis_label)
        ax.set_ylabel(yAxis_label)
        ax.set_zlabel(zAxis_label)
        if xAxis_limit:
            ax.set_xlim(xAxis_limit)
        if yAxis_limit:
            ax.set_ylim(yAxis_limit)
        if zAxis_limit:
            ax.set_zlim(zAxis_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def get_grouped_violin_plot(self, table_source, data_values, labels, name, yAxis_label=None, xAxis_label=None, rotation=0):
        """filters out empty arrays and aligns labels."""
        self.set_plot_style()
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Filter out empty arrays and their labels
        filtered = [(np.array(vals), lab) for vals, lab in zip(data_values, labels) if len(vals) > 0]
        if not filtered:
            print("No data to plot!")
            return
        data_values, labels = zip(*filtered)
        data_values = list(data_values)
        # Plot
        sns.violinplot(data=data_values, density_norm='width')
        plt.xticks(range(len(labels)), labels, rotation=rotation)
        if yAxis_label:
            plt.ylabel(yAxis_label)
        if xAxis_label:
            plt.xlabel(xAxis_label)
        plt.tight_layout()
        # Remove table_source from title if it's af3_metrics or pae_metrics
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        self.save_plot(title)
        plt.close()

    def get_3d_scatterplot_hue(self, table_source, xAxis_score, yAxis_score, zAxis_score,
                               database_column, xAxis_label, yAxis_label, zAxis_label,
                               name, colors=None, xAxis_limit=None, yAxis_limit=None, zAxis_limit=None):
        """
        3D scatter plot with numeric x, y, z and points colored by database/hue.

        Parameters:
            table_source : pd.DataFrame
            xAxis_score, yAxis_score, zAxis_score : numeric arrays
            database_column : column name with database labels
            colors : dict {db_label: color}, optional
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # Numeric ranges before plotting
        x_min, x_max = table_source[xAxis_score].min(), table_source[xAxis_score].max()
        y_min, y_max = table_source[yAxis_score].min(), table_source[yAxis_score].max()
        z_min, z_max = table_source[zAxis_score].min(), table_source[zAxis_score].max()

        # Add small margins for better visibility
        xAxis_limit = (0, x_max * 1.05) if xAxis_limit is None else xAxis_limit
        yAxis_limit = (0, y_max * 1.05) if yAxis_limit is None else yAxis_limit
        if zAxis_limit is None:
            # Expand downward and upward by 5%
            z_range = z_max - z_min
            zAxis_limit = (z_min - 0.05 * abs(z_range), z_max + 0.05 * abs(z_range))

        fig = plt.figure(figsize=(30, 16))
        ax = fig.add_subplot(111, projection='3d')

        # If no colors given, use default cycle
        if colors is None:
            colors = {db: c for db, c in zip(table_source[database_column].unique(),
                                             ['b', 'r', 'g', 'orange', 'purple'])}

        # Plot each database separately
        for db_label, color in colors.items():
            df_db = table_source[table_source[database_column] == db_label].copy()
            df_db = df_db.dropna(subset=[xAxis_score, yAxis_score, zAxis_score])
            if df_db.empty:
                continue

            ax.scatter(
                df_db[xAxis_score].values,
                df_db[yAxis_score].values,
                df_db[zAxis_score].values,
                label=db_label,
                c=color,
                alpha=0.4,  # <-- more transparent
                s=40
            )

        # Axis labels
        ax.set_xlabel(xAxis_label, labelpad=10)
        ax.set_ylabel(yAxis_label, labelpad=10)
        ax.set_zlabel(zAxis_label, labelpad=10)

        if xAxis_limit:
            ax.set_xlim(xAxis_limit)
        if yAxis_limit:
            ax.set_ylim(yAxis_limit)
        if zAxis_limit:
            ax.set_zlim(zAxis_limit)

        ax.tick_params(axis='both', which='major', labelsize=_NATURE_TICK, pad=5)
        ax.tick_params(axis='z', labelsize=_NATURE_TICK, pad=5)

        ax.legend(title="")

        plt.tight_layout()
        self.save_plot(name)
        plt.close()

    def get_3d_scatter_with_groups(self, table_source, x, y, z, xlabel, ylabel, zlabel, name, 
                                    groups=None, group_colors=None, group_labels=None):
        """
        Create a 3D scatter plot with optional color coding by groups.
        
        Args:
            table_source: source of the data
            x, y, z: arrays of x, y, z values
            xlabel, ylabel, zlabel: axis labels
            name: plot name for saving
            groups: array of group labels (same length as x/y/z), optional
            group_colors: dict mapping group labels to colors, optional
            group_labels: list of group labels for legend, optional
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        
        self.set_plot_style()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Default colors if not provided
        if group_colors is None:
            group_colors = {'+MSA': 'blue', '-MSA': 'orange', '+rMSA': 'blue', '-rMSA': 'orange'}
        
        if groups is not None:
            # Plot each group separately
            unique_groups = set(groups) if groups is not None else set()
            for group in unique_groups:
                mask = np.array([g == group for g in groups])
                if mask.any():
                    color = group_colors.get(group, 'gray')
                    label = group if group_labels is None else group_labels.get(group, group)
                    ax.scatter(x[mask], y[mask], z[mask], c=color, label=label, alpha=0.7, s=50)
            
            if unique_groups:
                ax.legend()
        else:
            # Single color plot
            ax.scatter(x, y, z, c='blue', alpha=0.7, s=50)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.tight_layout()
        self.save_plot(name)
        plt.close()

    def get_violin_plot_with_annotations(self, table_source, data_values, labels, name, 
                                         yAxis_label=None, xAxis_label=None, rotation=0,
                                         annotations=None, annotation_pos='below'):
        """
        Create a violin plot with optional annotations below x-axis ticks.
        
        Args:
            table_source: source of the data
            data_values: list of data arrays
            labels: list of labels for x-axis
            name: plot name
            yAxis_label: y-axis label
            xAxis_label: x-axis label
            rotation: rotation angle for x-axis labels
            annotations: list of annotation strings (same length as labels), optional
            annotation_pos: 'below' or 'above' for annotation position
        """
        self.set_plot_style()
        
        # Filter out empty arrays
        filtered = [(np.array(vals), lab, ann) for vals, lab, ann in 
                   zip(data_values, labels, annotations or [None]*len(labels)) if len(vals) > 0]
        if not filtered:
            print("No data to plot!")
            return
        
        data_values, labels, annotations = zip(*filtered)
        data_values = list(data_values)
        labels = list(labels)
        annotations = list(annotations) if annotations else None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        parts = ax.violinplot(data_values, showmeans=False, showmedians=True)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=rotation)
        if yAxis_label:
            ax.set_ylabel(yAxis_label)
        if xAxis_label:
            ax.set_xlabel(xAxis_label)
        
        # Add annotations if provided
        if annotations:
            y_pos = -0.08 if annotation_pos == 'below' else 1.08
            for i, ann in enumerate(annotations):
                if ann:
                    ax.text(i + 1, y_pos, str(ann), ha='center', 
                           va='top' if annotation_pos == 'below' else 'bottom',
                           fontsize=_NATURE_TICK, color='black',
                           transform=ax.get_xaxis_transform())
        
        plt.tight_layout()
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_grouped_barplot_with_errors(self, table_source, groups, group_labels, name,
                                        errors=None, colors=None, legend_labels=None,
                                        xlabel=None, ylabel=None, rotation=0, ylim=None):
        """
        Create a grouped bar plot with error bars.
        
        Args:
            table_source: source of the data
            groups: list of data arrays (one per group)
            group_labels: list of labels for x-axis categories
            name: plot name
            errors: list of error arrays (same length as groups), optional
            colors: list of colors for each group, optional
            legend_labels: list of legend labels, optional
            xlabel: x-axis label
            ylabel: y-axis label
            rotation: rotation angle for x-axis labels
            ylim: tuple (min, max) for y-axis limits, optional
        """
        self.set_plot_style()
        
        if not groups or not group_labels:
            print("No data to plot!")
            return
        
        num_groups = len(groups)
        num_categories = len(group_labels)
        width = 0.35
        x = np.arange(num_categories)
        
        # Default colors
        if colors is None:
            colors = plt.cm.tab10(range(num_groups))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, (group_data, color) in enumerate(zip(groups, colors)):
            offset = (i - (num_groups - 1) / 2) * width
            error_data = errors[i] if errors and i < len(errors) else None
            label = legend_labels[i] if legend_labels and i < len(legend_labels) else None
            
            ax.bar(x + offset, group_data, width, label=label, color=color,
                  yerr=error_data, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        ax.set_xlabel(xlabel if xlabel else '')
        ax.set_ylabel(ylabel if ylabel else '')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=rotation, ha='right' if rotation else 'center')
        
        if legend_labels:
            handles, labels_legend = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        
        if ylim:
            ax.set_ylim(ylim)
        else:
            # Auto-adjust ylim to show all data
            max_val = max([max(g) for g in groups if len(g) > 0] + [0])
            if max_val > 0:
                ax.set_ylim([0, max_val * 1.2])
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout(pad=2.0)
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_boxplot_with_colors(self, table_source, data_values, labels, name,
                                colors=None, yAxis_label=None, xAxis_label=None,
                                rotation=45, legend_elements=None):
        """
        Create a box plot with custom colors for each box.
        
        Args:
            table_source: source of the data
            data_values: list of data arrays
            labels: list of labels for x-axis
            name: plot name
            colors: list of colors (same length as data_values), optional
            yAxis_label: y-axis label
            xAxis_label: x-axis label
            rotation: rotation angle for x-axis labels
            legend_elements: list of matplotlib patches for legend, optional
        """
        self.set_plot_style()
        
        # Filter out empty arrays
        filtered = [(np.array(vals), lab, col) for vals, lab, col in 
                   zip(data_values, labels, colors or [None]*len(labels)) if len(vals) > 0]
        if not filtered:
            print("No data to plot!")
            return
        
        data_values, labels, colors = zip(*filtered)
        data_values = list(data_values)
        labels = list(labels)
        colors = list(colors) if colors else None
        
        fig, ax = plt.subplots(figsize=(16, 8))
        bp = ax.boxplot(data_values, labels=labels, patch_artist=True)
        
        # Apply colors if provided
        if colors:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        if yAxis_label:
            ax.set_ylabel(yAxis_label)
        if xAxis_label:
            ax.set_xlabel(xAxis_label)
        
        ax.tick_params(axis='x', rotation=rotation, labelsize=_NATURE_TICK)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_stacked_barplot(self, table_source, categories, category_labels, segment_data, 
                           segment_labels, name, colors=None, xlabel=None, ylabel=None,
                           rotation=0):
        """
        Create a stacked bar plot.
        
        Args:
            table_source: source of the data
            categories: list of category names
            category_labels: list of labels for x-axis
            segment_data: dict mapping categories to lists of segment values
            segment_labels: list of segment labels for legend
            name: plot name
            colors: list of colors for segments, optional
            xlabel: x-axis label
            ylabel: y-axis label
            rotation: rotation angle for x-axis labels
        """
        self.set_plot_style()
        
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(segment_labels)))
        
        fig, ax = plt.subplots(figsize=(20, 10))
        x = range(len(category_labels))
        bottom = np.zeros(len(category_labels))
        
        bars = []
        for i, segment_label in enumerate(segment_labels):
            values = [segment_data.get(cat, {}).get(segment_label, 0) for cat in categories]
            bar = ax.bar(x, values, bottom=bottom, label=segment_label, color=colors[i])
            bottom += np.array(values)
            bars.append(bar)
        
        ax.set_xlabel(xlabel if xlabel else '', labelpad=10)
        ax.set_ylabel(ylabel if ylabel else '', labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, rotation=rotation, ha='right' if rotation else 'center')
        ax.legend()
        ax.margins(x=0.01)
        plt.tight_layout()
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_barplot_with_errors(self, table_source, values, labels, name, errors=None,
                                color='#FF8F00', xlabel=None, ylabel=None, rotation=90,
                                figsize=(20, 10)):
        """
        Create a bar plot with error bars.
        
        Args:
            table_source: source of the data
            values: array of bar heights
            labels: list of labels for x-axis
            name: plot name
            errors: array of error values, optional
            color: bar color
            xlabel: x-axis label
            ylabel: y-axis label
            rotation: rotation angle for x-axis labels
            figsize: figure size tuple
        """
        self.set_plot_style()
        
        fig, ax = plt.subplots(figsize=figsize)
        x_pos = range(len(labels))
        
        ax.bar(x_pos, values, yerr=errors, capsize=5, color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=rotation, ha='right')
        
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=10)
        
        ax.margins(x=0.01)
        plt.tight_layout()
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    def get_side_by_side_barplot(self, table_source, data1, data2, labels, name,
                                 label1='Experimental', label2='Predicted',
                                 color1='orange', color2='green', alpha2=0.7,
                                 xlabel=None, ylabel=None, rotation=45, figsize=(8, 6)):
        """
        Create a side-by-side bar plot comparing two datasets.
        
        Args:
            table_source: source of the data
            data1: first dataset (array of values)
            data2: second dataset (array of values)
            labels: list of category labels
            name: plot name
            label1, label2: legend labels
            color1, color2: bar colors
            alpha2: alpha for second dataset
            xlabel: x-axis label
            ylabel: y-axis label
            rotation: rotation angle for x-axis labels
            figsize: figure size tuple
        """
        self.set_plot_style()
        
        fig, ax = plt.subplots(figsize=figsize)
        x = range(len(labels))
        width = 0.35
        
        ax.bar(x, data1, width, label=label1, color=color1)
        ax.bar([i + width for i in x], data2, width, label=label2, color=color2, alpha=alpha2)
        
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(labels, rotation=rotation, ha='right' if rotation else 'center')
        
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        handles, labels_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        
        plt.tight_layout(pad=2.0)
        
        if table_source in ['af3_metrics', 'pae_metrics']:
            title = name
        else:
            title = f"{name}_{table_source}"
        
        self.save_plot(title)
        plt.close()

    # --- Sample audit publication figures (saved under results_dir/interface) ---

    def plot_audit_quadrant_rbp_dockq_stack(self, table_source, audit, figsize=(10.5, 5.8)):
        """Stacked fractions of RBP_DockQ (a)(b)(c) among complexes with finite RBP_DockQ per quadrant."""
        prefs = _AUDIT_QUAD_PREFIXES
        xlab = list(_AUDIT_QUAD_TICKLABELS)
        na = np.array([len(audit.get(f"{p}_a") or []) for p in prefs], dtype=float)
        nb = np.array([len(audit.get(f"{p}_b") or []) for p in prefs], dtype=float)
        nc = np.array([len(audit.get(f"{p}_c") or []) for p in prefs], dtype=float)
        ntot = na + nb + nc
        if not np.any(ntot > 0):
            return
        fa = np.divide(na, ntot, out=np.zeros_like(ntot), where=ntot > 0)
        fb = np.divide(nb, ntot, out=np.zeros_like(ntot), where=ntot > 0)
        fc = np.divide(nc, ntot, out=np.zeros_like(ntot), where=ntot > 0)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(xlab))
        colors = ("#0072B2", "#E69F00", "#D55E00")
        labels = ("RBP_DockQ ≥ 0.8", "0.23 ≤ RBP_DockQ < 0.8", "RBP_DockQ < 0.23")
        bars_c = ax.bar(x, fc, label=labels[2], color=colors[2])
        bars_b = ax.bar(x, fb, bottom=fc, label=labels[1], color=colors[1])
        bars_a = ax.bar(x, fa, bottom=fc + fb, label=labels[0], color=colors[0])
        ax.set_xticks(x)
        ax.set_xticklabels(xlab, rotation=22, ha="right")
        ax.set_ylabel("Fraction (DockQ in bin)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", frameon=False)
        for bars in (bars_c, bars_b, bars_a):
            for rect in bars:
                h = rect.get_height()
                if h >= 0.06:
                    fc_rgba = rect.get_facecolor()
                    lum = 0.299 * fc_rgba[0] + 0.587 * fc_rgba[1] + 0.114 * fc_rgba[2]
                    txt_color = "black" if lum > 0.72 else "white"
                    ax.annotate(
                        f"{h:.0%}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + h / 2),
                        ha="center",
                        va="center",
                        fontsize=_NATURE_TICK - 2,
                        color=txt_color,
                    )
        fig.tight_layout()
        title = f"audit_quadrant_RBP_DockQ_stack_{table_source}"
        self._save_audit_plot(title)
        plt.close(fig)

    def plot_audit_failure_complex_rmsd_stack(self, table_source, audit, figsize=(10.5, 5.8)):
        """Among DockQ < 0.23: fractions with finite Complex_RMSD <10 Å vs ≥10 Å only."""
        prefs = _AUDIT_QUAD_PREFIXES
        xlab = list(_AUDIT_QUAD_TICKLABELS)
        ca_n = np.array([len(audit.get(f"{p}_c_ca") or []) for p in prefs], dtype=float)
        cb_n = np.array([len(audit.get(f"{p}_c_cb") or []) for p in prefs], dtype=float)
        n_fin = ca_n + cb_n
        if not np.any(n_fin > 0):
            return
        p_ca = np.divide(ca_n, n_fin, out=np.zeros_like(n_fin), where=n_fin > 0)
        p_cb = np.divide(cb_n, n_fin, out=np.zeros_like(n_fin), where=n_fin > 0)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(xlab))
        color_ca = "#009E73"
        color_cb = "#CC79A7"
        lab_ca = "CRMSD < 10 Å"
        lab_cb = "CRMSD ≥ 10 Å"
        bars_ca = ax.bar(x, p_ca, label=lab_ca, color=color_ca)
        bars_cb = ax.bar(x, p_cb, bottom=p_ca, label=lab_cb, color=color_cb)
        ax.set_xticks(x)
        ax.set_xticklabels(xlab, rotation=22, ha="right")
        ax.set_ylabel("Fraction of DockQ failures")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", frameon=False)
        for bars in (bars_ca, bars_cb):
            for rect in bars:
                h = rect.get_height()
                if h >= 0.06:
                    fc_rgba = rect.get_facecolor()
                    lum = 0.299 * fc_rgba[0] + 0.587 * fc_rgba[1] + 0.114 * fc_rgba[2]
                    txt_color = "black" if lum > 0.72 else "white"
                    ax.annotate(
                        f"{h:.0%}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + h / 2),
                        ha="center",
                        va="center",
                        fontsize=_NATURE_TICK - 2,
                        color=txt_color,
                    )
        fig.tight_layout()
        title = f"audit_failure_Complex_RMSD_stack_{table_source}"
        self._save_audit_plot(title)
        plt.close(fig)

    def plot_audit_quadrant_outcome_heatmap(self, table_source, audit, figsize=(13.2, 5.6)):
        """Heatmap: monomer-accuracy classes vs DockQ % and failure RMSD split among (c)."""
        prefs = _AUDIT_QUAD_PREFIXES
        rows_lab = list(_AUDIT_QUAD_TICKLABELS)
        cols_lab = [
            "DockQ≥0.8\n(% of class)",
            "0.23≤DQ<0.8\n(% of class)",
            "DQ<0.23\n(% of class)",
            "DQ<0.23\n & CRMSD<10 Å",
            "DQ<0.23\n & CRMSD≥10 Å",
        ]
        mat = np.full((4, 5), np.nan)
        for i, p in enumerate(prefs):
            n_all = len(audit.get(f"{p}_all") or [])
            if n_all <= 0:
                continue
            na = len(audit.get(f"{p}_a") or [])
            nb = len(audit.get(f"{p}_b") or [])
            nc = len(audit.get(f"{p}_c") or [])
            mat[i, 0] = 100.0 * na / n_all
            mat[i, 1] = 100.0 * nb / n_all
            mat[i, 2] = 100.0 * nc / n_all
            if nc > 0:
                mat[i, 3] = 100.0 * len(audit.get(f"{p}_c_ca") or []) / nc
                mat[i, 4] = 100.0 * len(audit.get(f"{p}_c_cb") or []) / nc
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor="white",
            xticklabels=cols_lab,
            yticklabels=rows_lab,
            cbar_kws={"label": "Percentage"},
        )
        #ax.set_xlabel("Outcome class")
        ax.tick_params(axis="y", labelsize=_NATURE_TICK - 2)
        fig.tight_layout()
        title = f"audit_quadrant_outcome_heatmap_{table_source}"
        self._save_audit_plot(title)
        plt.close(fig)

    def plot_audit_quadrant_outcome_motif_mean_heatmap(
        self, table_source, mat, score_column, figsize=(13.2, 5.6), score_label=None
    ):
        """Same layout as outcome heatmap; cells = mean ``score_column`` (not %)."""
        if mat is None or mat.size == 0 or not np.any(np.isfinite(mat)):
            return
        display_label = score_label if score_label else score_column
        rows_lab = list(_AUDIT_QUAD_TICKLABELS)
        cols_lab = [
            "DockQ≥0.8",
            "0.23≤DQ<0.8",
            "DQ<0.23",
            "DQ fail\nCRMSD<10Å",
            "DQ fail\nCRMSD≥10Å",
        ]
        mask = ~np.isfinite(mat)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            xticklabels=cols_lab,
            yticklabels=rows_lab,
            cbar_kws={"label": f"Mean {display_label}"},
            mask=mask,
        )
        ax.set_title(
            f"Mean {display_label} by quadrant × DockQ / failure class",
            fontsize=_NATURE_TICK - 2,
            pad=12,
        )
        ax.tick_params(axis="y", labelsize=_NATURE_TICK - 2)
        fig.tight_layout()
        safe = str(score_column).replace(" ", "_")
        self._save_audit_plot(f"audit_quadrant_outcome_heatmap_mean_{safe}_{table_source}")
        plt.close(fig)

    def plot_audit_crmsd_bins_combined_means_heatmap(
        self, table_source, profile, figsize=(28, 8)
    ):
        """Core + AF3 metrics per Complex_RMSD bin in one heatmap."""
        mat = profile.get("combined_mat")
        if mat is None or mat.size == 0 or not np.any(np.isfinite(mat)):
            return
        rows = profile["row_labels"]
        cols = profile["combined_col_labels"]
        counts = profile.get("bin_counts") or []
        n_core = profile.get("n_core_cols") or 0
        row_ann = []
        for i, lab in enumerate(rows):
            c = counts[i] if i < len(counts) else ""
            row_ann.append(f"{lab}\n(n={c})")
        annot = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if j < n_core:
                    row.append(_audit_heatmap_format_cell_core(v))
                else:
                    row.append(_audit_heatmap_format_cell(v))
            annot.append(row)
        color_mat = _audit_heatmap_norm_columns(mat)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        if not np.any(np.isfinite(color_mat)):
            plt.close(fig)
            return
        sns.heatmap(
            color_mat,
            ax=ax,
            annot=np.array(annot),
            fmt="",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.35,
            linecolor="white",
            xticklabels=cols,
            yticklabels=row_ann,
            cbar_kws={"label": "Within-column level (0–1)"},
            mask=~np.isfinite(color_mat),
            annot_kws={
                "rotation": 90,
                "ha": "center",
                "va": "center",
                "fontsize": _NATURE_TICK - 5,
            },
        )
        if n_core and n_core < len(cols):
            ax.axvline(n_core, color="white", linewidth=2.0)
        ax.set_xlabel("Metric (core | AF3)")
        ax.set_title(
            "Complex_RMSD bins: core + AF3 means (column 5–95% scaled)",
            fontsize=_NATURE_TICK - 2,
            pad=14,
        )
        fig.tight_layout()
        self._save_audit_plot(f"audit_crmsd_bins_combined_means_heatmap_{table_source}")
        plt.close(fig)

    def plot_audit_bins_core_means_heatmap(
        self, table_source, profile, figsize=(34, 22), filename_suffix=""
    ):
        """Mean prediction-table metrics per audit bin (heatmap)."""
        mat = profile.get("core_mat")
        if mat is None or mat.size == 0 or not np.any(np.isfinite(mat)):
            return
        rows = profile["row_labels"]
        cols = profile["core_col_labels"]
        counts = profile.get("bin_counts") or []
        row_ann = []
        for i, lab in enumerate(rows):
            c = counts[i] if i < len(counts) else ""
            row_ann.append(f"{lab}\n(n={c})")
        annot = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                v = mat[i, j]
                row.append(_audit_heatmap_format_cell_core(v))
            annot.append(row)
        color_mat = _audit_heatmap_norm_columns(mat)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        if not np.any(np.isfinite(color_mat)):
            plt.close(fig)
            return
        sns.heatmap(
            color_mat,
            ax=ax,
            annot=np.array(annot),
            fmt="",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.35,
            linecolor="white",
            xticklabels=cols,
            yticklabels=row_ann,
            cbar_kws={"label": "Within-column level (0–1)"},
            mask=~np.isfinite(color_mat),
            annot_kws={
                "rotation": 90,
                "ha": "center",
                "va": "center",
                "fontsize": _NATURE_TICK - 4,
            },
        )
        ax.set_xlabel("Metric")
        ax.set_title(
            "Colors: each column scaled by 5th–95th percentile across bins; cells = raw mean",
            fontsize=_NATURE_TICK - 3,
            pad=14,
        )
        fig.tight_layout()
        stem = f"audit_bins_core_means_heatmap_{table_source}"
        if filename_suffix:
            stem = f"{stem}{filename_suffix}"
        self._save_audit_plot(stem)
        plt.close(fig)

    def plot_audit_bins_af3_means_heatmap(
        self, table_source, profile, figsize=(14, 22), filename_suffix=""
    ):
        """Mean AF3-derived metrics per audit bin."""
        mat = profile.get("af3_mat")
        if mat is None or mat.size == 0 or not np.any(np.isfinite(mat)):
            return
        rows = profile["row_labels"]
        cols = profile["af3_col_labels"]
        counts = profile.get("bin_counts") or []
        row_ann = []
        for i, lab in enumerate(rows):
            c = counts[i] if i < len(counts) else ""
            row_ann.append(f"{lab}\n(n={c})")
        annot = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                v = mat[i, j]
                row.append(_audit_heatmap_format_cell(v))
            annot.append(row)
        color_mat = _audit_heatmap_norm_columns(mat)
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        if not np.any(np.isfinite(color_mat)):
            plt.close(fig)
            return
        sns.heatmap(
            color_mat,
            ax=ax,
            annot=np.array(annot),
            fmt="",
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.35,
            linecolor="white",
            xticklabels=cols,
            yticklabels=row_ann,
            cbar_kws={"label": "Within-column level (0–1)"},
            mask=~np.isfinite(color_mat),
        )
        ax.set_xlabel("AF3 metric")
        ax.set_title(
            "Colors: each column scaled by 5th–95th percentile across bins; cells = raw mean",
            fontsize=_NATURE_TICK - 3,
            pad=14,
        )
        fig.tight_layout()
        stem = f"audit_bins_af3_means_heatmap_{table_source}"
        if filename_suffix:
            stem = f"{stem}{filename_suffix}"
        self._save_audit_plot(stem)
        plt.close(fig)

    def plot_audit_bins_mean_msa_depths(self, table_source, profile, figsize=(18, 6)):
        """Grouped bars: mean RNA vs protein MSA depth (longest chains) per audit bin."""
        rna_d = profile.get("mean_rna_msa_depth")
        prot_d = profile.get("mean_prot_msa_depth")
        if rna_d is None or prot_d is None:
            return
        if not (
            np.any(np.isfinite(rna_d))
            or np.any(np.isfinite(prot_d))
        ):
            return
        n = len(rna_d)
        x = np.arange(n)
        w = 0.38
        self.set_plot_style()
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n):
            if np.isfinite(rna_d[i]):
                ax.bar(i - w / 2, rna_d[i], width=w, color="#0072B2")
            if np.isfinite(prot_d[i]):
                ax.bar(i + w / 2, prot_d[i], width=w, color="#D55E00")
        ax.bar([], [], width=w, color="#0072B2", label="RNA MSA depth (mean)")
        ax.bar([], [], width=w, color="#D55E00", label="Protein MSA depth (mean)")
        ax.set_xticks(x)
        rl = profile["row_labels"]
        ct = profile.get("bin_counts") or []
        ticklabs = [f"{rl[i]}\n(n={ct[i]})" if i < len(ct) else rl[i] for i in range(n)]
        ax.set_xticklabels(ticklabs, rotation=75, ha="right", fontsize=_NATURE_TICK - 4)
        ax.set_ylabel("Mean depth (1 → 0)")
        ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.02), borderaxespad=0, frameon=False)
        fig.subplots_adjust(bottom=0.38, right=0.78)
        fig.tight_layout(rect=[0.02, 0.05, 0.78, 0.96])
        self._save_audit_plot(f"audit_bins_mean_MSA_depth_{table_source}")
        plt.close(fig)

    def plot_audit_bins_presence_and_templates(self, table_source, profile, figsize=(18, 10)):
        """Stacked fractions: MSA presence (RNA / protein) and PDB template presence."""
        frna = profile.get("frac_rna_msa")
        fprot = profile.get("frac_prot_msa")
        fpdb = profile.get("frac_prot_pdb")
        frbp = profile.get("frac_rbp_pdb")
        if any(a is None for a in (frna, fprot, fpdb, frbp)):
            return
        n = len(frna)
        x = np.arange(n)
        self.set_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

        def _stack(ax_h, frac, title, c_no, c_yes):
            mask = np.isfinite(frac)
            xh = x[mask]
            fy = np.clip(np.asarray(frac, dtype=float)[mask], 0.0, 1.0)
            ax_h.bar(xh, 1.0 - fy, width=0.82, label="No / zero", color=c_no)
            ax_h.bar(xh, fy, width=0.82, bottom=1.0 - fy, label="Yes / >0", color=c_yes)
            ax_h.set_ylim(0, 1)
            ax_h.set_ylabel("Fraction")
            ax_h.set_title(title)
            ax_h.legend(loc="lower left", bbox_to_anchor=(1.02, 0.15), borderaxespad=0, frameon=False)

        _stack(axes[0, 0], frna, "RNA MSA (>0 effective depth)", "#BBBBBB", "#0072B2")
        _stack(axes[0, 1], fprot, "Protein MSA (>0 effective depth)", "#BBBBBB", "#E69F00")
        _stack(axes[1, 0], fpdb, "Protein PDB templates (count ≥1)", "#BBBBBB", "#009E73")
        _stack(axes[1, 1], frbp, "RBP PDB templates (count ≥1)", "#BBBBBB", "#CC79A7")

        rl = profile["row_labels"]
        ct = profile.get("bin_counts") or []
        ticklabs = [f"{rl[i]}\n(n={ct[i]})" if i < len(ct) else rl[i] for i in range(n)]
        for axh in axes.flat:
            axh.set_xticks(x)
            axh.set_xticklabels(
                ticklabs,
                rotation=90,
                ha="center",
                va="top",
                fontsize=_NATURE_TICK - 5,
            )

        fig.subplots_adjust(bottom=0.42, right=0.82)
        fig.tight_layout(rect=[0.02, 0.06, 0.82, 0.96])
        self._save_audit_plot(f"audit_bins_MSA_and_template_fractions_{table_source}")
        plt.close(fig)

    # def __del__(self):
    #     """Close database connection when object is destroyed"""
    #     if hasattr(self, 'db'):
    #         self.db.close_connection()
