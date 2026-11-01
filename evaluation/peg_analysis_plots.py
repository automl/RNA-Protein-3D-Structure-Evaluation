#!/usr/bin/env python3
"""
PEG Analysis Plotting Script
- Reads predicted metrics from the database
- Classifies buffer categories using database/classifyBufferConditions.py
- Joins experimental categories with predicted metrics
- Produces violin plots via plots/plotCreator.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

# Suppress matplotlib INFO messages about categorical units
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning, message='.*categorical units.*')

# allow direct execution: python evaluation/<script>.py
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.startConfig import StartConfig
from database import classifyBufferConditions as cbc
from plots.plotCreator import PlotCreator


class PEGAnalysisPlotter:
    def __init__(self, msa_option=None, single_chain_only=False, rbp_template_option=None, rnalength_filter: str = None, condition_filter: str = None, create_multi_factor_plots: bool = False):
        self.config = StartConfig()
        self.db_path = self.config.get_database_path()
        self.msa_option = None
        if isinstance(msa_option, str) and msa_option in ['+MSA', '-MSA', '+msa', '-msa']:
            self.msa_option = '+MSA' if msa_option[0] == '+' else '-MSA'
        self.single_chain_only = bool(single_chain_only)
        self.rbp_template_option = rbp_template_option
        self.rnalength_filter = rnalength_filter
        self.condition_filter = condition_filter
        self.create_multi_factor_plots = create_multi_factor_plots

        # Determine which predicted table to use
        self.pred_table = self._detect_predicted_table()
        
        # Check if this is an RNA-only table
        is_rna_only = (self.pred_table == "pred_rna_rna")

        # Map desired metric labels -> actual DB column names
        # Use different metrics based on table type
        if is_rna_only:
            # RNA-only metrics
            self.metric_column_map = {
                'rTM': 'RNA_TM',
                'rLDDT': 'RNA_LDDT_allAtom',
                r'RNA$_{wl}$': 'RNA_wl'
            }
        else:
            # Protein-RNA metrics
            self.metric_column_map = {
                                'rTM': 'RNA_TM',
                                'rLDDT$_{C3}$': 'RNA_LDDT',
                                'cTM': 'Complex_TM',
                                'cLDDT$_{BB}$': 'Complex_LDDT',
                'iLDDT': 'iLDDT',
                                'RNA Match Score': 'RNAmatch_score',
                                'rLDDT': 'RNA_LDDT_allAtom',
                                'cLDDT': 'Complex_LDDT_allAtom',
                                'crTM': 'RNAComplex_TM',
                # 'QS_global': 'QS_global',
                                 r'RNA$_{wl}$': 'RNA_wl'
                # 'ICS': 'ICS',
            }

        # Category orders
        self.category_orders = {
            'PEG_concentration': ['No PEG', 'PEG (0-10%)', 'PEG (10-20%)', 'PEG (20-30%)', 'PEG (30-50%)'],
            'PEG_MW_category': ['No PEG', 'PEG 0-1k', 'PEG 1k-2k', 'PEG 2k-10k', 'PEG 10k-20k'],
            'buffer_category': ['Unknown', '0-0.1 M', '0.1-0.5 M', '0.5-5.0 M', '>5.0 M'],
            'buffer_matches': ['acetate', 'bicine', 'citrate', 'cacodylate', 'formate', 'hepes', 'phosphate', 'tris'],
            'ionic_strength_category': ['Unknown', '0-0.5 M', '0.5-1.0 M', '≥1.0 M']
        }

        # Load data
        self.metrics_df = self._load_predicted_metrics()
        self.categories_df = self._load_buffer_categories()

        # Merge by experimental identifier
        # Normalize keys to maximize join success (case-insensitive)
        self.metrics_df['exp_db_id_norm'] = self.metrics_df['exp_db_id'].astype(str).str.upper()
        self.categories_df['PDBId_norm'] = self.categories_df['PDBId'].astype(str).str.upper()
        self.df = self.metrics_df.merge(
            self.categories_df,
            left_on='exp_db_id_norm',
            right_on='PDBId_norm',
            how='inner',
        )

        # Classify MSA sizes into bins
        self._classify_msa_sizes()
        
        # Classify RBP template presence
        self._classify_rbp_template_presence()

        # Optional filtering by MSA/single-chain/RBP template using PlotCreator's DB filters
        if self.msa_option or self.single_chain_only or self.rbp_template_option:
            try:
                _pc = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
                allowed_ids = set(_pc.get_filtered_pdb_ids(self.pred_table))
                if allowed_ids:
                    self.df = self.df[self.df['exp_db_id'].isin(allowed_ids)]
            except Exception:
                pass

        # Determine available metrics present in the selected table
        self.metrics = [label for label, col in self.metric_column_map.items() if col in self.df.columns]
        self.categories = ['PEG_concentration', 'PEG_MW_category', 'ionic_strength_category', 'buffer_category', 'buffer_matches']

        # Results directory - use PlotCreator to get the correct filtered directory
        _pc = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
        self.results_dir = Path(_pc.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _detect_predicted_table(self) -> str:
        """Detect and return the predicted table to use.
        """
        candidates = ['pred_protein_rna', 'pred_rna_rna']
        # Include config-provided name if not already covered
        cfg_name = self.config.get_predicted_table_name()
        if cfg_name and cfg_name not in candidates:
            candidates.append(cfg_name)

        with sqlite3.connect(self.config.get_database_path()) as conn:
            existing = set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'])
        for name in candidates:
            if name in existing:
                return name
        raise RuntimeError('No predicted table found. Checked: ' + ', '.join(candidates))

    def _load_predicted_metrics(self) -> pd.DataFrame:
        """Load predicted metrics and identifiers from the chosen table."""
        cols_needed = ['exp_db_id', 'RNALength'] + list(self.metric_column_map.values())  # Include RNALength
        
        # Check if RNA_msa_size and RBP_PDB_Templates columns exist in the table
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.pred_table})")
            available_columns = [row[1] for row in cursor.fetchall()]
            
            # Add RNA_msa_size if it exists in the table
            if 'RNA_msa_size' in available_columns and 'RNA_msa_size' not in cols_needed:
                cols_needed.append('RNA_msa_size')
            
            # Add RBP_PDB_Templates if it exists in the table
            if 'RBP_PDB_Templates' in available_columns and 'RBP_PDB_Templates' not in cols_needed:
                cols_needed.append('RBP_PDB_Templates')
        
        select_cols = ', '.join(cols_needed)

        query = f"SELECT {select_cols} FROM {self.pred_table}"

        filters = []
        if self.rnalength_filter:
            filters.append(f"RNALength {self.rnalength_filter}")
        if self.condition_filter:
            filters.append(self.condition_filter)

        if filters:
            query += " WHERE " + " AND ".join(filters)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    def _classify_msa_sizes(self):
        """Classify RNA MSA sizes into bins: no_msa, small_msa, medium_msa, large_msa."""
        import json
        import ast
        
        def get_max_msa_size(msa_size_str):
            """Extract maximum MSA size from JSON string like '{"A": 1, "B": 3208}'."""
            if pd.isna(msa_size_str) or msa_size_str == '' or msa_size_str is None:
                return 0
            
            try:
                # Try to parse as JSON
                msa_dict = json.loads(msa_size_str)
                if isinstance(msa_dict, dict):
                    # Get maximum value from dictionary
                    values = [v for v in msa_dict.values() if isinstance(v, (int, float))]
                    return max(values) if values else 0
                return 0
            except (json.JSONDecodeError, ValueError):
                # Try to parse as Python literal (e.g., if stored as string representation)
                try:
                    msa_dict = ast.literal_eval(msa_size_str)
                    if isinstance(msa_dict, dict):
                        values = [v for v in msa_dict.values() if isinstance(v, (int, float))]
                        return max(values) if values else 0
                except (ValueError, SyntaxError):
                    pass
                return 0
        
        if 'RNA_msa_size' in self.df.columns:
            # Get maximum MSA size for each sample
            self.df['max_msa_size'] = self.df['RNA_msa_size'].apply(get_max_msa_size)
            
            # Classify into bins
            bins = [-float('inf'), 1, 20, 1000, float('inf')]
            labels = ['0 MSA', '0-20 MSA', '20-1000 MSA', '1000+ MSA']
            self.df['msa_class'] = pd.cut(
                self.df['max_msa_size'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            
            # Add MSA class to category orders
            self.category_orders['msa_class'] = labels
        else:
            print("Warning: RNA_msa_size column not found. MSA classification plots will be skipped.")
            self.df['msa_class'] = None
    
    def _classify_rbp_template_presence(self):
        """Classify RBP_PDB_Templates into binary presence: +RBPtemp or -RBPtemp."""
        import json
        import ast
        
        def has_rbp_templates(rbp_templates_str):
            """Check if RBP_PDB_Templates contains any templates."""
            if pd.isna(rbp_templates_str) or rbp_templates_str == '' or rbp_templates_str is None:
                return False
            
            # Check for empty string representations
            if str(rbp_templates_str).strip() in ['[]', 'null', 'NULL', '']:
                return False
            
            try:
                # Try to parse as JSON
                templates = json.loads(rbp_templates_str)
                if isinstance(templates, list):
                    return len(templates) > 0
                return False
            except (json.JSONDecodeError, ValueError):
                # Try to parse as Python literal
                try:
                    templates = ast.literal_eval(str(rbp_templates_str))
                    if isinstance(templates, list):
                        return len(templates) > 0
                    return False
                except (ValueError, SyntaxError):
                    return False
        
        if 'RBP_PDB_Templates' in self.df.columns:
            self.df['rbp_template_presence'] = self.df['RBP_PDB_Templates'].apply(
                lambda x: '+RBPtemp' if has_rbp_templates(x) else '-RBPtemp'
            )
            # Add to category orders
            self.category_orders['rbp_template_presence'] = ['-RBPtemp', '+RBPtemp']
        else:
            print("Warning: RBP_PDB_Templates column not found. RBP template classification plots will be skipped.")
            self.df['rbp_template_presence'] = None

    def _load_buffer_categories(self) -> pd.DataFrame:
        """Use classifyBufferConditions to get categories directly from the DB."""
        entries = cbc.get_exp_solution_data()
        classified = cbc.classify_buffer_conditions(entries)
        # Convert list of dicts to DataFrame, keep only needed
        df = pd.DataFrame(classified)[[
            'PDBId', 'PEG_concentration', 'PEG_MW_category', 'PEG_MW', 'PEG_percent',
            'buffer_category', 'buffer_M', 'buffer_matches',
            'ionic_strength', 'ionic_strength_category'
        ]]
        return df

    def _prepare_violin_data(self, metric_label: str, category: str):
        col = self.metric_column_map[metric_label]
        df = self.df[[col, category]].dropna()
        if df.empty:
            return [], []

        order = self.category_orders.get(category)
        data_values, labels = [], []
        if order:
            for cat in order:
                vals = df[df[category] == cat][col].values
                if len(vals) > 0:
                    data_values.append(vals)
                    labels.append(cat)
        else:
            for cat in sorted(df[category].unique()):
                vals = df[df[category] == cat][col].values
                if len(vals) > 0:
                    data_values.append(vals)
                    labels.append(cat)
        return data_values, labels

    def create_violin_plot(self, metric_label: str, category: str):
        data_values, labels = self._prepare_violin_data(metric_label, category)
        if not data_values:
            return
        plotter = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
        
        # Add descriptive labels for specific categories
        # metric_label is already the display label from metric_column_map
        y_axis_label = metric_label
        if category == 'buffer_matches':
            x_axis_label = "Buffer Name"
        if category == 'buffer_category':
            x_axis_label = "Buffer Concentration"
        elif category == 'ionic_strength_category':
            x_axis_label = "Ionic Strength"
        elif category == 'PEG_concentration':
            x_axis_label = "PEG Concentration"
        elif category == 'PEG_MW_category':
            x_axis_label = "PEG MW"
        
        plotter.get_violin_plot(
            table_source=self.pred_table,
            data_values=data_values,
            labels=labels,
            name=f"{metric_label}_by_{category}",
            yAxis_label=y_axis_label,
            xAxis_label=x_axis_label,
            rotation=90,
        )

    def calculate_mean_std(self, values):
        """Calculate mean, std, and median for a list of values."""
        if not values or len(values) == 0:
            return None, None, None
        # Filter out None and invalid values
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not valid_values:
            return None, None, None
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        median = np.median(valid_values)
        return mean, std, median

    def write_statistics_to_file(self):
        """Write statistics to a text file in the same directory as plots."""
        output_path = self.results_dir / 'peg_analysis_statistics.txt'
        
        from scipy.stats import pearsonr, spearmanr
        
        # Helper function to format p-value
        def format_p_value(p_value):
            """Format p-value, using scientific notation for very small values"""
            if np.isnan(p_value) or np.isinf(p_value):
                return "N/A"
            if p_value == 0.0:
                return "0.000"
            if p_value < 1e-4:
                exp = int(np.floor(np.log10(abs(p_value))))
                coeff = p_value / (10 ** exp)
                return f"{coeff:.3f}e{exp}"
            elif p_value < 0.001:
                return f"{p_value:.3e}"
            else:
                return f"{p_value:.3f}"
        
        # Helper function to get significance stars
        def get_stars(p_value):
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            else:
                return ""

        # Helper to safely normalize categorical/string labels (works for pandas Categorical too)
        def norm_labels(series):
            s = series.astype('object')
            s = s.where(~pd.isna(s), 'Unknown')
            s = s.astype(str).str.strip()
            return s.replace({'': 'Unknown', 'nan': 'Unknown', 'None': 'Unknown', 'NaN': 'Unknown'})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PEG Analysis Statistics\n")
            f.write("=" * 80 + "\n\n")
            
            # Add filter information
            filter_parts = []
            if self.single_chain_only:
                filter_parts.append("Single-Chain")
            else:
                filter_parts.append("All-Chains")
            
            if self.msa_option:
                filter_parts.append(self.msa_option)
            
            if self.rbp_template_option:
                # Format RBP template option nicely
                if self.rbp_template_option.upper() == '+RBPTEMP':
                    filter_parts.append("+RBP-template")
                elif self.rbp_template_option.upper() == '-RBPTEMP':
                    filter_parts.append("-RBP-template")
                else:
                    filter_parts.append(self.rbp_template_option)
            
            if filter_parts:
                f.write("Selected Samples by: " + ", ".join(filter_parts) + "\n\n")

            f.write("Stratified PEG/Buffer effects by MSA and RBP status\n")
            f.write("=" * 80 + "\n")
            f.write(
                "This section asks whether PEG MW, PEG concentration, and buffer effects differ when\n"
                "MSA and/or RBP templates are present or absent.\n\n"
            )

            # Build binary MSA presence (+MSA/-MSA) from max_msa_size or msa_class if available.
            msa_presence = None
            if 'max_msa_size' in self.df.columns:
                msa_num = pd.to_numeric(self.df['max_msa_size'], errors='coerce')
                msa_presence = pd.Series(np.where(msa_num > 1, '+MSA', '-MSA'), index=self.df.index)
                msa_presence = msa_presence.where(~msa_num.isna(), 'Unknown')
            elif 'msa_class' in self.df.columns:
                mclass = norm_labels(self.df['msa_class'])
                msa_presence = pd.Series(np.where(mclass == '0 MSA', '-MSA', '+MSA'), index=self.df.index)
                msa_presence = msa_presence.where(~mclass.isin(['Unknown']), 'Unknown')

            rbp_presence = norm_labels(self.df['rbp_template_presence']) if 'rbp_template_presence' in self.df.columns else None

            strata = [('All samples', pd.Series(True, index=self.df.index))]
            if msa_presence is not None:
                strata.append(('+MSA only', msa_presence == '+MSA'))
                strata.append(('-MSA only', msa_presence == '-MSA'))
            if rbp_presence is not None:
                strata.append(('+RBPtemp only', rbp_presence == '+RBPtemp'))
                strata.append(('-RBPtemp only', rbp_presence == '-RBPtemp'))
            if msa_presence is not None and rbp_presence is not None:
                strata.extend([
                    ('+MSA & +RBPtemp', (msa_presence == '+MSA') & (rbp_presence == '+RBPtemp')),
                    ('+MSA & -RBPtemp', (msa_presence == '+MSA') & (rbp_presence == '-RBPtemp')),
                    ('-MSA & +RBPtemp', (msa_presence == '-MSA') & (rbp_presence == '+RBPtemp')),
                    ('-MSA & -RBPtemp', (msa_presence == '-MSA') & (rbp_presence == '-RBPtemp')),
                ])

            try:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
            except Exception:
                sm = None
                ols = None
                SMValueWarning = Warning

            try:
                from scipy.stats import kruskal
            except Exception:
                kruskal = None

            for stratum_name, stratum_mask in strata:
                sub = self.df[stratum_mask].copy()
                f.write(f"{stratum_name} (n={len(sub)}):\n")
                if len(sub) < 3:
                    f.write("  Too few samples for robust analysis.\n\n")
                    continue

                for metric_label in self.metrics:
                    metric_col = self.metric_column_map.get(metric_label)
                    if not metric_col or metric_col not in sub.columns:
                        continue
                    metric_vals = pd.to_numeric(sub[metric_col], errors='coerce')
                    f.write(f"  {metric_label}:\n")

                    # Numeric PEG and buffer correlations
                    for x_col, x_label in [('PEG_MW', 'PEG_MW'), ('PEG_percent', 'PEG_percent'), ('buffer_M', 'buffer_M')]:
                        if x_col not in sub.columns:
                            continue
                        x_vals = pd.to_numeric(sub[x_col], errors='coerce')
                        mask = ~(metric_vals.isna() | x_vals.isna())
                        if mask.sum() >= 3:
                            try:
                                p_corr, p_p = pearsonr(metric_vals[mask], x_vals[mask])
                                s_corr, s_p = spearmanr(metric_vals[mask], x_vals[mask])
                                f.write(
                                    f"    vs {x_label}: Pearson={p_corr:.3f}{get_stars(p_p)} (p={format_p_value(p_p)}), "
                                    f"Spearman={s_corr:.3f}{get_stars(s_p)} (p={format_p_value(s_p)}), n={mask.sum()}\n"
                                )
                            except Exception:
                                pass

                    # Buffer type and concentration grouped effects
                    if 'buffer_matches' in sub.columns and 'buffer_category' in sub.columns and kruskal is not None:
                        bw = sub[['buffer_matches', 'buffer_category', metric_col]].copy()
                        bw['buffer_matches'] = norm_labels(bw['buffer_matches'])
                        bw['buffer_category'] = norm_labels(bw['buffer_category'])
                        bw[metric_col] = pd.to_numeric(bw[metric_col], errors='coerce')
                        bw = bw.dropna(subset=[metric_col])

                        by_buf = [vals.values for _, vals in bw.groupby('buffer_matches')[metric_col] if len(vals) >= 2]
                        if len(by_buf) >= 2:
                            try:
                                h, p = kruskal(*by_buf)
                                f.write(f"    buffer type effect (Kruskal): H={h:.3f}{get_stars(p)} (p={format_p_value(p)})\n")
                            except Exception:
                                pass

                        by_buf_cat = [vals.values for _, vals in bw.groupby('buffer_category')[metric_col] if len(vals) >= 2]
                        if len(by_buf_cat) >= 2:
                            try:
                                h, p = kruskal(*by_buf_cat)
                                f.write(f"    buffer concentration effect (Kruskal): H={h:.3f}{get_stars(p)} (p={format_p_value(p)})\n")
                            except Exception:
                                pass

                    # PEG MW x PEG concentration interaction inside this stratum
                    if 'PEG_MW_category' in sub.columns and 'PEG_concentration' in sub.columns and sm is not None and ols is not None:
                        aw = sub[['PEG_MW_category', 'PEG_concentration', metric_col]].copy()
                        aw['PEG_MW_category'] = norm_labels(aw['PEG_MW_category'])
                        aw['PEG_concentration'] = norm_labels(aw['PEG_concentration'])
                        aw = aw[(aw['PEG_MW_category'] != 'Unknown') & (aw['PEG_concentration'] != 'Unknown')]
                        aw[metric_col] = pd.to_numeric(aw[metric_col], errors='coerce')
                        aw = aw.dropna(subset=[metric_col])
                        if len(aw) >= 6 and aw['PEG_MW_category'].nunique() >= 2 and aw['PEG_concentration'].nunique() >= 2:
                            cell_counts = aw.groupby(['PEG_MW_category', 'PEG_concentration']).size()
                            if int((cell_counts > 0).sum()) >= 4:
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", category=RuntimeWarning)
                                        warnings.simplefilter("ignore", category=SMValueWarning)
                                        model = ols(f'Q("{metric_col}") ~ C(PEG_concentration) * C(PEG_MW_category)', data=aw).fit()
                                        anova_tbl = sm.stats.anova_lm(model, typ=2)
                                    key = 'C(PEG_concentration):C(PEG_MW_category)'
                                    if key in anova_tbl.index and 'Residual' in anova_tbl.index:
                                        f_stat = float(anova_tbl.loc[key, 'F'])
                                        p_val = float(anova_tbl.loc[key, 'PR(>F)'])
                                        ss_inter = float(anova_tbl.loc[key, 'sum_sq'])
                                        ss_resid = float(anova_tbl.loc['Residual', 'sum_sq'])
                                        partial_eta_sq = (ss_inter / (ss_inter + ss_resid)) if (ss_inter + ss_resid) > 0 else np.nan
                                        if np.isfinite(f_stat) and np.isfinite(p_val):
                                            f.write(
                                                f"    PEG_MW x PEG% interaction: F={f_stat:.3f}{get_stars(p_val)} "
                                                f"(p={format_p_value(p_val)}), partial_eta^2={partial_eta_sq:.4f}\n"
                                            )
                                except Exception:
                                    pass
                    f.write("\n")
                f.write("\n")

            f.write("Buffer Type Statistics\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total merged samples: {len(self.df)}\n\n")

            if 'buffer_category' in self.df.columns:
                f.write("Buffer concentration category distribution:\n")
                cat_counts = norm_labels(self.df['buffer_category']).value_counts(dropna=False)
                for cat, n in cat_counts.items():
                    pct = (n / len(self.df) * 100.0) if len(self.df) else 0.0
                    f.write(f"  {cat}: {n} ({pct:.1f}%)\n")
                f.write("\n")

            if 'buffer_matches' in self.df.columns:
                f.write("Buffer name distribution:\n")
                buf_counts = norm_labels(self.df['buffer_matches']).value_counts(dropna=False)
                for buf, n in buf_counts.items():
                    pct = (n / len(self.df) * 100.0) if len(self.df) else 0.0
                    f.write(f"  {buf}: {n} ({pct:.1f}%)\n")
                f.write("\n")

            def write_grouped_metric_stats(group_col: str, group_label: str):
                if group_col not in self.df.columns:
                    return
                f.write(f"{group_label} effects by metric:\n")
                f.write("-" * 80 + "\n")
                try:
                    from scipy.stats import kruskal
                except Exception:
                    kruskal = None

                for metric_label in self.metrics:
                    metric_col = self.metric_column_map.get(metric_label)
                    if not metric_col or metric_col not in self.df.columns:
                        continue
                    work = self.df[[group_col, metric_col]].copy()
                    work[group_col] = norm_labels(work[group_col])
                    work[metric_col] = pd.to_numeric(work[metric_col], errors='coerce')
                    work = work.dropna(subset=[metric_col])
                    if work.empty:
                        continue

                    grouped = []
                    for grp, vals in work.groupby(group_col)[metric_col]:
                        arr = pd.to_numeric(vals, errors='coerce').dropna().values
                        if len(arr) > 0:
                            grouped.append((grp, arr))
                    if not grouped:
                        continue

                    f.write(f"{metric_label}:\n")
                    for grp, arr in sorted(grouped, key=lambda x: len(x[1]), reverse=True):
                        f.write(
                            f"  {grp}: n={len(arr)}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, median={np.median(arr):.4f}\n"
                        )

                    # For each buffer name, compares concentration categories using Kruskal-Wallis.
                    if kruskal is not None:
                        valid_groups = [arr for _, arr in grouped if len(arr) >= 2]
                        if len(valid_groups) >= 2:
                            try:
                                h_stat, p_value = kruskal(*valid_groups)
                                stars = get_stars(p_value)
                                f.write(
                                    f"  Kruskal-Wallis across {group_label}: H={h_stat:.3f}{stars} (p={format_p_value(p_value)})\n"
                                )
                            except Exception:
                                pass
                    f.write("\n")

            write_grouped_metric_stats('buffer_matches', 'Buffer names')
            write_grouped_metric_stats('buffer_category', 'Buffer concentration categories')

            f.write("Explicit concentration-type interaction (buffer identity x concentration)\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Interpretation note: this section tests whether concentration effects differ by buffer identity.\n"
                "Two-way ANOVA interaction asks if the buffer-type and concentration factors interact (non-additive effect).\n"
                "Kruskal-Wallis within each buffer tests if metric distributions differ across that buffer's concentration bins.\n\n"
            )
            if 'buffer_matches' in self.df.columns and 'buffer_category' in self.df.columns:
                try:
                    from scipy.stats import kruskal
                except Exception:
                    kruskal = None
                try:
                    import statsmodels.api as sm  # noqa: F401
                    from statsmodels.formula.api import ols
                    from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
                except Exception:
                    ols = None
                    SMValueWarning = Warning

                for metric_label in self.metrics:
                    metric_col = self.metric_column_map.get(metric_label)
                    if not metric_col or metric_col not in self.df.columns:
                        continue

                    work = self.df[['buffer_matches', 'buffer_category', metric_col]].copy()
                    work['buffer_matches'] = norm_labels(work['buffer_matches'])
                    work['buffer_category'] = norm_labels(work['buffer_category'])
                    work[metric_col] = pd.to_numeric(work[metric_col], errors='coerce')
                    work = work.dropna(subset=[metric_col])
                    if len(work) < 6:
                        continue

                    f.write(f"{metric_label}:\n")

                    # Formal interaction model when statsmodels is available
                    if ols is not None and work['buffer_matches'].nunique() >= 2 and work['buffer_category'].nunique() >= 2:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                warnings.simplefilter("ignore", category=SMValueWarning)
                                model = ols(f"{metric_col} ~ C(buffer_matches) * C(buffer_category)", data=work).fit()
                                anova_tbl = sm.stats.anova_lm(model, typ=2)
                            if 'C(buffer_matches):C(buffer_category)' in anova_tbl.index:
                                p_int = float(anova_tbl.loc['C(buffer_matches):C(buffer_category)', 'PR(>F)'])
                                f_stat = float(anova_tbl.loc['C(buffer_matches):C(buffer_category)', 'F'])
                                if np.isfinite(f_stat) and np.isfinite(p_int):
                                    stars = get_stars(p_int)
                                    f.write(
                                        f"  Two-way ANOVA interaction: F={f_stat:.3f}{stars} (p={format_p_value(p_int)})\n"
                                    )
                                else:
                                    f.write("  Two-way ANOVA interaction: not estimable (rank-deficient design).\n")
                        except Exception:
                            pass

                    # Within-buffer concentration effect (non-parametric)
                    if kruskal is not None:
                        tested = 0
                        significant = []
                        for buf, sub in work.groupby('buffer_matches'):
                            grouped = [vals.values for _, vals in sub.groupby('buffer_category')[metric_col] if len(vals) >= 2]
                            if len(grouped) >= 2:
                                try:
                                    h_stat, p_val = kruskal(*grouped)
                                    tested += 1
                                    if p_val < 0.05:
                                        significant.append((buf, h_stat, p_val))
                                except Exception:
                                    continue

                        f.write(f"  Within-buffer concentration tests run: {tested}\n")
                        if significant:
                            f.write("  Buffers with significant low-vs-high concentration differences:\n")
                            for buf, h_stat, p_val in sorted(significant, key=lambda x: x[2]):
                                stars = get_stars(p_val)
                                f.write(f"    {buf}: H={h_stat:.3f}{stars} (p={format_p_value(p_val)})\n")
                        else:
                            f.write("  No significant within-buffer concentration effect detected (p < 0.05).\n")
                    f.write("\n")
            else:
                f.write("Insufficient columns (requires both buffer_matches and buffer_category).\n\n")

            f.write("PEG concentration vs PEG molecular weight\n")
            f.write("-" * 80 + "\n")
            if 'PEG_percent' in self.df.columns and 'PEG_MW' in self.df.columns:
                peg_percent_vals = pd.to_numeric(self.df['PEG_percent'], errors='coerce')
                peg_mw_vals = pd.to_numeric(self.df['PEG_MW'], errors='coerce')
                mask = ~(peg_percent_vals.isna() | peg_mw_vals.isna())
                if mask.sum() >= 2:
                    try:
                        pearson_corr, pearson_p = pearsonr(peg_percent_vals[mask], peg_mw_vals[mask])
                        spearman_corr, spearman_p = spearmanr(peg_percent_vals[mask], peg_mw_vals[mask])
                        pearson_stars = get_stars(pearson_p)
                        spearman_stars = get_stars(spearman_p)
                        f.write(f"  Number of samples: {mask.sum()}\n")
                        f.write(f"  Pearson correlation: {pearson_corr:.3f}{pearson_stars} (p={format_p_value(pearson_p)})\n")
                        f.write(f"  Spearman correlation: {spearman_corr:.3f}{spearman_stars} (p={format_p_value(spearman_p)})\n\n")
                    except Exception:
                        f.write("  Could not compute PEG_percent vs PEG_MW correlation.\n\n")
                else:
                    f.write("  Not enough valid samples (need at least 2).\n\n")
            else:
                f.write("  Missing required columns (PEG_percent and/or PEG_MW).\n\n")

            f.write("Influence of individual PEG MW and PEG percentage bins on metrics\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Interpretation note: each line below is one combined bin (PEG_MW_category | PEG_concentration).\n"
                "This helps assess whether specific PEG chemistry/concentration windows are associated with metric shifts.\n\n"
            )
            if 'PEG_MW_category' in self.df.columns and 'PEG_concentration' in self.df.columns:
                try:
                    from scipy.stats import kruskal
                except Exception:
                    kruskal = None

                for metric_label in self.metrics:
                    metric_col = self.metric_column_map.get(metric_label)
                    if not metric_col or metric_col not in self.df.columns:
                        continue

                    work = self.df[['PEG_MW_category', 'PEG_concentration', metric_col]].copy()
                    work['PEG_MW_category'] = norm_labels(work['PEG_MW_category'])
                    work['PEG_concentration'] = norm_labels(work['PEG_concentration'])
                    work['peg_bin_combo'] = work['PEG_MW_category'] + " | " + work['PEG_concentration']
                    work[metric_col] = pd.to_numeric(work[metric_col], errors='coerce')
                    work = work.dropna(subset=[metric_col])
                    if work.empty:
                        continue

                    grouped = []
                    for combo, vals in work.groupby('peg_bin_combo')[metric_col]:
                        arr = pd.to_numeric(vals, errors='coerce').dropna().values
                        if len(arr) > 0:
                            grouped.append((combo, arr))
                    if not grouped:
                        continue

                    f.write(f"{metric_label}:\n")
                    for combo, arr in sorted(grouped, key=lambda x: len(x[1]), reverse=True):
                        f.write(
                            f"  {combo}: n={len(arr)}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, median={np.median(arr):.4f}\n"
                        )

                    if kruskal is not None:
                        valid_groups = [arr for _, arr in grouped if len(arr) >= 2]
                        if len(valid_groups) >= 2:
                            try:
                                h_stat, p_val = kruskal(*valid_groups)
                                stars = get_stars(p_val)
                                f.write(
                                    f"  Kruskal-Wallis across PEG bin combinations: H={h_stat:.3f}{stars} (p={format_p_value(p_val)})\n"
                                )
                            except Exception:
                                pass
                    f.write("\n")
            else:
                f.write("  Missing required categorical columns (PEG_MW_category and/or PEG_concentration).\n\n")

            f.write("Two-factor interaction model: PEG MW x PEG concentration\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Model per metric: metric ~ C(PEG_concentration) + C(PEG_MW_category) + "
                "C(PEG_concentration):C(PEG_MW_category)\n"
                "Reported interaction terms include p-value and effect size estimates "
                "(eta^2 and partial eta^2 / partial R^2).\n\n"
            )
            if 'PEG_MW_category' in self.df.columns and 'PEG_concentration' in self.df.columns:
                try:
                    import statsmodels.api as sm
                    from statsmodels.formula.api import ols
                    from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
                except Exception:
                    sm = None
                    ols = None
                    SMValueWarning = Warning

                if sm is None or ols is None:
                    f.write("  statsmodels not available; skipping two-factor ANOVA interaction models.\n\n")
                else:
                    for metric_label in self.metrics:
                        metric_col = self.metric_column_map.get(metric_label)
                        if not metric_col or metric_col not in self.df.columns:
                            continue
                        work = self.df[['PEG_MW_category', 'PEG_concentration', metric_col]].copy()
                        work['PEG_MW_category'] = norm_labels(work['PEG_MW_category'])
                        work['PEG_concentration'] = norm_labels(work['PEG_concentration'])
                        work[metric_col] = pd.to_numeric(work[metric_col], errors='coerce')
                        work = work.dropna(subset=[metric_col])

                        # Need multiple levels in each factor to estimate interaction.
                        if len(work) < 6 or work['PEG_MW_category'].nunique() < 2 or work['PEG_concentration'].nunique() < 2:
                            continue

                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                warnings.simplefilter("ignore", category=SMValueWarning)
                                model = ols(f'Q("{metric_col}") ~ C(PEG_concentration) * C(PEG_MW_category)', data=work).fit()
                                anova_tbl = sm.stats.anova_lm(model, typ=2)
                            interaction_key = 'C(PEG_concentration):C(PEG_MW_category)'
                            if interaction_key not in anova_tbl.index or 'Residual' not in anova_tbl.index:
                                continue

                            ss_inter = float(anova_tbl.loc[interaction_key, 'sum_sq'])
                            f_inter = float(anova_tbl.loc[interaction_key, 'F'])
                            p_inter = float(anova_tbl.loc[interaction_key, 'PR(>F)'])
                            ss_resid = float(anova_tbl.loc['Residual', 'sum_sq'])
                            ss_total = float(anova_tbl['sum_sq'].sum())

                            eta_sq = (ss_inter / ss_total) if ss_total > 0 else np.nan
                            partial_eta_sq = (ss_inter / (ss_inter + ss_resid)) if (ss_inter + ss_resid) > 0 else np.nan
                            f.write(f"{metric_label}:\n")
                            if np.isfinite(f_inter) and np.isfinite(p_inter):
                                stars = get_stars(p_inter)
                                f.write(f"  Interaction F-statistic: {f_inter:.3f}{stars}\n")
                                f.write(f"  Interaction p-value: {format_p_value(p_inter)}\n")
                                f.write(f"  Interaction eta^2: {eta_sq:.4f}\n")
                                f.write(f"  Interaction partial eta^2 (partial R^2): {partial_eta_sq:.4f}\n")
                            else:
                                f.write("  Interaction model not estimable (rank-deficient PEG_MW x PEG% design).\n")
                            f.write(f"  Samples used: {len(work)}\n\n")
                        except Exception:
                            continue
            else:
                f.write("  Missing required categorical columns (PEG_MW_category and/or PEG_concentration).\n\n")

            f.write("Correlation Analysis\n")
            f.write("=" * 80 + "\n")
            f.write("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001\n\n")
            
            # For each metric, calculate correlations with PEG, ions, and buffers
            for metric_label in self.metrics:
                metric_col = self.metric_column_map.get(metric_label)
                if not metric_col or metric_col not in self.df.columns:
                    continue
                
                # Get metric values
                metric_vals = pd.to_numeric(self.df[metric_col], errors='coerce')
                
                # 1. Correlations with PEG_MW
                if 'PEG_MW' in self.df.columns:
                    peg_mw_vals = pd.to_numeric(self.df['PEG_MW'], errors='coerce')
                    mask = ~(metric_vals.isna() | peg_mw_vals.isna())
                    if mask.sum() >= 2:
                        try:
                            pearson_corr, pearson_p = pearsonr(metric_vals[mask], peg_mw_vals[mask])
                            spearman_corr, spearman_p = spearmanr(metric_vals[mask], peg_mw_vals[mask])
                            pearson_stars = get_stars(pearson_p)
                            spearman_stars = get_stars(spearman_p)
                            f.write(f"{metric_label} vs PEG_MW:\n")
                            f.write(f"  Number of samples: {mask.sum()}\n")
                            f.write(f"  Pearson correlation: {pearson_corr:.3f}{pearson_stars} (p={format_p_value(pearson_p)})\n")
                            f.write(f"  Spearman correlation: {spearman_corr:.3f}{spearman_stars} (p={format_p_value(spearman_p)})\n\n")
                        except Exception:
                            pass
                
                # 2. Correlations with PEG_percent
                if 'PEG_percent' in self.df.columns:
                    peg_percent_vals = pd.to_numeric(self.df['PEG_percent'], errors='coerce')
                    mask = ~(metric_vals.isna() | peg_percent_vals.isna())
                    if mask.sum() >= 2:
                        try:
                            pearson_corr, pearson_p = pearsonr(metric_vals[mask], peg_percent_vals[mask])
                            spearman_corr, spearman_p = spearmanr(metric_vals[mask], peg_percent_vals[mask])
                            pearson_stars = get_stars(pearson_p)
                            spearman_stars = get_stars(spearman_p)
                            f.write(f"{metric_label} vs PEG_percent:\n")
                            f.write(f"  Number of samples: {mask.sum()}\n")
                            f.write(f"  Pearson correlation: {pearson_corr:.3f}{pearson_stars} (p={format_p_value(pearson_p)})\n")
                            f.write(f"  Spearman correlation: {spearman_corr:.3f}{spearman_stars} (p={format_p_value(spearman_p)})\n\n")
                        except Exception:
                            pass
                
                # 3. Correlations with ionic strength
                if 'ionic_strength' in self.df.columns:
                    ionic_vals = pd.to_numeric(self.df['ionic_strength'], errors='coerce')
                    mask = ~(metric_vals.isna() | ionic_vals.isna())
                    if mask.sum() >= 2:
                        try:
                            pearson_corr, pearson_p = pearsonr(metric_vals[mask], ionic_vals[mask])
                            spearman_corr, spearman_p = spearmanr(metric_vals[mask], ionic_vals[mask])
                            pearson_stars = get_stars(pearson_p)
                            spearman_stars = get_stars(spearman_p)
                            f.write(f"{metric_label} vs Ionic Strength:\n")
                            f.write(f"  Number of samples: {mask.sum()}\n")
                            f.write(f"  Pearson correlation: {pearson_corr:.3f}{pearson_stars} (p={format_p_value(pearson_p)})\n")
                            f.write(f"  Spearman correlation: {spearman_corr:.3f}{spearman_stars} (p={format_p_value(spearman_p)})\n\n")
                        except Exception:
                            pass
                
                # 4. Correlations with buffers (using numeric buffer_M)
                if 'buffer_M' in self.df.columns:
                    buffer_vals = pd.to_numeric(self.df['buffer_M'], errors='coerce')
                    mask = ~(metric_vals.isna() | buffer_vals.isna())
                    if mask.sum() >= 2:
                        try:
                            pearson_corr, pearson_p = pearsonr(metric_vals[mask], buffer_vals[mask])
                            spearman_corr, spearman_p = spearmanr(metric_vals[mask], buffer_vals[mask])
                            pearson_stars = get_stars(pearson_p)
                            spearman_stars = get_stars(spearman_p)
                            f.write(f"{metric_label} vs Buffers:\n")
                            f.write(f"  Number of samples: {mask.sum()}\n")
                            f.write(f"  Pearson correlation: {pearson_corr:.3f}{pearson_stars} (p={format_p_value(pearson_p)})\n")
                            f.write(f"  Spearman correlation: {spearman_corr:.3f}{spearman_stars} (p={format_p_value(spearman_p)})\n\n")
                        except Exception:
                            pass
                
                f.write("\n")
        
        print(f"\nStatistics written to: {output_path}")

    def create_all_plots(self):
        for metric_label in self.metrics:
            for category in self.categories:
                try:
                    self.create_violin_plot(metric_label, category)
                except Exception:
                    # Continue on plotting errors to finish the batch
                    pass

        # Additionally: plots conditioned on both PEG_MW and PEG_concentration (MW first, then concentration)
        if 'PEG_concentration' in self.df.columns and 'PEG_MW_category' in self.df.columns:
            for metric_label in self.metrics:
                try:
                    self.create_two_factor_violin(metric_label, primary_cat='PEG_MW_category', secondary_cat='PEG_concentration')
                except Exception:
                    pass

        # MSA class vs PEG plots
        if 'msa_class' in self.df.columns and self.df['msa_class'].notna().any():
            for metric_label in self.metrics:
                try:
                    # MSA class vs PEG MW
                    if 'PEG_MW_category' in self.df.columns:
                        self.create_two_factor_violin(metric_label, primary_cat='msa_class', secondary_cat='PEG_MW_category')
                    # MSA class vs PEG concentration
                    if 'PEG_concentration' in self.df.columns:
                        self.create_two_factor_violin(metric_label, primary_cat='msa_class', secondary_cat='PEG_concentration')
                    # MSA class vs Ionic Strength
                    if 'ionic_strength_category' in self.df.columns:
                        self.create_two_factor_violin(metric_label, primary_cat='msa_class', secondary_cat='ionic_strength_category')
                    # MSA class vs Buffers
                    if 'buffer_category' in self.df.columns:
                        self.create_two_factor_violin(metric_label, primary_cat='msa_class', secondary_cat='buffer_category')
                    # MSA class vs Buffer names
                    if 'buffer_matches' in self.df.columns:
                        self.create_two_factor_violin(metric_label, primary_cat='msa_class', secondary_cat='buffer_matches')
                    # Triple-factor and quadruple-factor plots only when create_multi_factor_plots is True
                    if self.create_multi_factor_plots:
                        # Triple-factor: MSA class / PEG MW / PEG concentration
                        if 'PEG_MW_category' in self.df.columns and 'PEG_concentration' in self.df.columns:
                            self.create_three_factor_violin(metric_label, cat1='msa_class', cat2='PEG_MW_category', cat3='PEG_concentration')
                        # Triple-factor: MSA class / Buffer names / Buffer concentration
                        if 'buffer_matches' in self.df.columns and 'buffer_category' in self.df.columns:
                            self.create_three_factor_violin(metric_label, cat1='msa_class', cat2='buffer_matches', cat3='buffer_category')
                        # Quadruple-factor: MSA class / RBP template / PEG MW / PEG concentration
                        if ('msa_class' in self.df.columns and 
                            'rbp_template_presence' in self.df.columns and 
                            'PEG_MW_category' in self.df.columns and 
                            'PEG_concentration' in self.df.columns):
                            self.create_four_factor_violin(
                                metric_label, 
                                cat1='msa_class', 
                                cat2='rbp_template_presence', 
                                cat3='PEG_MW_category', 
                                cat4='PEG_concentration'
                            )
                except Exception:
                    pass

        # Buffer-name specific plots grouped by buffer_category
        # self._create_buffer_name_plots()

        # Salt type vs salt concentration plots (excluding Multivalent)
        if 'salt_type' in self.df.columns and 'salt_concentration_category' in self.df.columns:
            for metric_label in self.metrics:
                try:
                    # Filter out Multivalent before plotting
                    df_filtered = self.df[self.df['salt_type'] != 'Multivalent'].copy()
                    if not df_filtered.empty:
                        original_df = self.df
                        self.df = df_filtered
                        try:
                            # Create 2-factor violin plot: salt_type x salt_concentration_category
                            self.create_two_factor_violin(metric_label, primary_cat='salt_type', secondary_cat='salt_concentration_category')
                        finally:
                            self.df = original_df
                except Exception:
                    pass

        # Numeric scatter plots for salts, buffers, and PEG MW
        self._create_numeric_scatter_plots()
        
        # Write statistics to file
        self.write_statistics_to_file()

    def _create_buffer_name_plots(self):
        if 'buffer_matches' not in self.df.columns or 'buffer_category' not in self.df.columns:
            return
        allowed_buffers = ['acetate', 'bicine', 'citrate', 'cacodylate', 'formate', 'hepes', 'phosphate', 'tris']
        # Normalize matches column to string
        buf_series = self.df['buffer_matches'].fillna('').astype(str).str.lower()
        for buf in allowed_buffers:
            mask = buf_series.str.contains(buf)
            sub = self.df[mask]
            if sub.empty:
                continue
            for metric_label, metric_col in self.metric_column_map.items():
                if metric_col not in sub.columns:
                    continue
                # Build grouped data by buffer_category
                data_values = []
                labels = []
                order = self.category_orders.get('buffer_category')
                work = sub[[metric_col, 'buffer_category']].dropna()
                if work.empty:
                    continue
                cats = order if order else sorted(work['buffer_category'].unique())
                for cat in cats:
                    vals = work[work['buffer_category'] == cat][metric_col].values
                    if len(vals) > 0:
                        data_values.append(vals)
                        labels.append(cat)
                if not data_values:
                    continue
                plotter = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
                # metric_label is already the display label from metric_column_map
                plotter.get_violin_plot(
                    table_source=self.pred_table,
                    data_values=data_values,
                    labels=labels,
                    name=f"{metric_label}_by_buffer_category_for_{buf}",
                    yAxis_label=metric_label,
                    xAxis_label='Buffer Concentration',
                    rotation=90,
                )

    def _create_numeric_scatter_plots(self):
        # Prepare numeric columns if available
        num_specs = [
            ('ionic_strength', 'Ionic strength (M)'),
            ('buffer_M', 'Buffer molarity (M)'),
            ('PEG_MW', 'PEG molecular weight'),
            ('PEG_percent', 'PEG concentration (%)'),
        ]
        for x_col, x_label in num_specs:
            if x_col not in self.df.columns:
                continue
            x_vals = pd.to_numeric(self.df[x_col], errors='coerce')
            for metric_label, metric_col in self.metric_column_map.items():
                if metric_col not in self.df.columns:
                    continue
                y_vals = pd.to_numeric(self.df[metric_col], errors='coerce')
                mask = ~(x_vals.isna() | y_vals.isna())
                if mask.sum() == 0:
                    continue
                plotter = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
                # metric_label is already the display label from metric_column_map
                plotter.get_scatterplot(
                    table_source=self.pred_table,
                    xAxis_score=x_vals[mask].tolist(),
                    xAxis_label=x_label,
                    yAxis_label=metric_label,
                    name=f"{metric_label}_vs_{x_col}",
                    yAxis_score=y_vals[mask].tolist(),
                )

    def create_two_factor_violin(self, metric_label: str, primary_cat: str, secondary_cat: str):
        """Create a box plot where groups are combinations of two categorical variables.
        Order is nested: iterate primary category order, inside iterate secondary category order.
        Colors by buffer_category, ionic_strength_category, PEG_concentration, or buffer_matches (same color).
        """
        if metric_label not in self.metric_column_map:
            return
        metric_col = self.metric_column_map[metric_label]
        if metric_col not in self.df.columns:
            return
        if primary_cat not in self.df.columns or secondary_cat not in self.df.columns:
            return

        df = self.df[[metric_col, primary_cat, secondary_cat]].dropna()
        if df.empty:
            return

        primary_order = self.category_orders.get(primary_cat)
        secondary_order = self.category_orders.get(secondary_cat)

        # Build groups
        data_values = []
        labels = []

        prim_vals = primary_order if primary_order else sorted(df[primary_cat].unique())
        sec_vals = secondary_order if secondary_order else sorted(df[secondary_cat].unique())

        for p in prim_vals:
            sub_p = df[df[primary_cat] == p]
            if sub_p.empty:
                continue
            for s in sec_vals:
                vals = sub_p[sub_p[secondary_cat] == s][metric_col].values
                if len(vals) > 0:
                    data_values.append(vals)
                    labels.append(f"{p} | {s}")

        if not data_values:
            return

        # Determine what to color by: buffer_category > ionic_strength_category > PEG_concentration > 
        # secondary_cat if it's PEG_MW_category or buffer_matches > last_factor
        color_by = None
        if 'buffer_category' in [primary_cat, secondary_cat]:
            color_by = 'buffer_category'
        elif 'ionic_strength_category' in [primary_cat, secondary_cat]:
            color_by = 'ionic_strength_category'
        elif 'PEG_concentration' in [primary_cat, secondary_cat]:
            color_by = 'PEG_concentration'
        elif secondary_cat == 'PEG_MW_category':
            # Color by PEG_MW_category values (different colors for each PEG_MW_category value)
            color_by = 'last_factor'
        elif secondary_cat == 'buffer_matches':
            # Color by buffer_matches values (different colors for each buffer name)
            color_by = 'last_factor'
        else:
            # Default: color by last factor (secondary_cat values)
            color_by = 'last_factor'
        
        # Create box plot with coloring
        self._create_box_plot_with_colors(
            metric_label=metric_label,
            data_values=data_values,
            labels=labels,
            name=f"{metric_label}_by_{primary_cat}_and_{secondary_cat}",
            color_by=color_by,
            primary_cat=primary_cat,
            secondary_cat=secondary_cat,
            df=df
        )

    def _create_wide_grouped_violin_plot(self, metric_label: str, data_values: list, labels: list, name: str, color_by_last_factor: bool = False):
        """Create a wide grouped violin plot for triple-factor and quadruple-factor plots with more space for labels.
        
        Args:
            metric_label: Label for the metric being plotted
            data_values: List of arrays containing data for each group
            labels: List of labels for each group
            name: Name for the plot file
            color_by_last_factor: If True, color violins by the last factor in the label (e.g., PEG_concentration)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        plotter = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
        
        # Filter out empty arrays
        filtered = [(np.array(vals), lab) for vals, lab in zip(data_values, labels) if len(vals) > 0]
        if not filtered:
            print("No data to plot!")
            return
        data_values_filtered, labels_filtered = zip(*filtered)
        data_values_filtered = list(data_values_filtered)
        
        # Detect if this is a quadruple plot (check name or label format)
        is_quadruple = 'cat4' in name or any(' | ' in label and label.count(' | ') == 3 for label in labels_filtered)
        
        # Create a figure - taller for quadruple plots
        # Width based on number of labels, but keep it reasonable
        num_labels = len(labels_filtered)
        fig_width = max(16, min(24, num_labels * 0.6))  # Between 16-24 inches, or 0.6 inches per label
        fig_height = 12 if is_quadruple else 10  # Taller for quadruple plots
        
        # Set figure size before creating plot
        plt.rcParams['figure.figsize'] = (fig_width, fig_height)
        plotter.set_plot_style()
        
        # Override figure size after set_plot_style
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Use seaborn violinplot with list of arrays
        # Create a long-format DataFrame for better control
        plot_data_list = []
        for i, (vals, label) in enumerate(zip(data_values_filtered, labels_filtered)):
            # Extract the last factor for coloring if requested
            color_category = None
            if color_by_last_factor and ' | ' in label:
                color_category = label.split(' | ')[-1]  # Get the last factor
            
            for val in vals:
                row = {'Group': label, 'Value': val}
                if color_category is not None:
                    row['ColorCategory'] = color_category
                plot_data_list.append(row)
        
        plot_df = pd.DataFrame(plot_data_list)
        
        # Plot with seaborn - add color if requested
        if color_by_last_factor and 'ColorCategory' in plot_df.columns:
            # Get unique color categories and create a palette
            unique_colors = sorted(plot_df['ColorCategory'].unique())
            # Use seaborn's default color palette (same as violin plots)
            palette_colors = sns.color_palette(n_colors=len(unique_colors))
            color_map = dict(zip(unique_colors, palette_colors))
            
            # Create a list of colors in the same order as labels
            color_list = []
            for label in labels_filtered:
                if ' | ' in label:
                    last_factor = label.split(' | ')[-1]
                    color_list.append(color_map.get(last_factor, sns.color_palette()[0]))
                else:
                    color_list.append(sns.color_palette()[0])
            
            # Plot with colors - use box plots for better color control
            ax = plt.gca()
            bp = ax.boxplot(data_values_filtered, positions=range(len(labels_filtered)), 
                           widths=0.6, patch_artist=True)
            
            # Apply colors to each box (same style as violin plots)
            for i, (box, color) in enumerate(zip(bp['boxes'], color_list)):
                # Convert color to RGB tuple if it's a matplotlib color
                if isinstance(color, tuple) and len(color) == 3:
                    box.set_facecolor(color)
                else:
                    box.set_facecolor(color)
                box.set_alpha(0.7)
                box.set_edgecolor('black')
                box.set_linewidth(1)
            
            # Style other box plot elements
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                if element in bp:
                    for item in bp[element]:
                        item.set_color('black')
                        item.set_linewidth(1)
            
            # Set x-axis labels below ticks
            ax.set_xticks(range(len(labels_filtered)))
            ax.set_xticklabels(labels_filtered, rotation=90, ha='center', va='top')
        else:
            # Use seaborn violinplot with default colors (same as regular violin plots)
            ax = sns.violinplot(data=plot_df, x='Group', y='Value', density_norm='width', order=labels_filtered)
            # Set x-axis labels below ticks for violin plots
            ax.set_xticks(range(len(labels_filtered)))
            ax.set_xticklabels(labels_filtered, rotation=90, ha='center', va='top')
        
        plt.ylabel(metric_label)
        plt.xlabel('')  # Remove default x-axis label
        
        # Adjust layout with more bottom margin for rotated labels
        # Use subplots_adjust instead of tight_layout to avoid warnings
        plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)
        
        # Save plot
        title = f"{name}_{self.pred_table}"
        plotter.save_plot(title)
        plt.close(fig)

    def _create_box_plot_with_colors(self, metric_label: str, data_values: list, labels: list, name: str, 
                                     color_by: str = None, primary_cat: str = None, secondary_cat: str = None,
                                     tertiary_cat: str = None, quaternary_cat: str = None, df: pd.DataFrame = None):
        """Create a box plot with colors based on buffer_category, ionic_strength_category, PEG_concentration, or buffer_matches.
        
        Args:
            metric_label: Label for the metric being plotted
            data_values: List of arrays containing data for each group
            labels: List of labels for each group (format: "cat1 | cat2 | ...")
            name: Name for the plot file
            color_by: What to color by ('buffer_category', 'ionic_strength_category', 'PEG_concentration', 'buffer_matches_same', 'last_factor', or None)
            primary_cat, secondary_cat, etc.: Category names to extract values from labels
            df: DataFrame with original data (for extracting category values)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        plotter = PlotCreator(plot_type='PEG_plots', msa_option=self.msa_option, single_chain_only=self.single_chain_only, rbp_template_option=self.rbp_template_option)
        
        # Filter out empty arrays
        filtered = [(np.array(vals), lab) for vals, lab in zip(data_values, labels) if len(vals) > 0]
        if not filtered:
            print("No data to plot!")
            return
        data_values_filtered, labels_filtered = zip(*filtered)
        data_values_filtered = list(data_values_filtered)
        
        # Determine figure size
        num_labels = len(labels_filtered)
        fig_width = max(16, min(24, num_labels * 0.6))
        fig_height = 10
        
        # Set figure size
        plt.rcParams['figure.figsize'] = (fig_width, fig_height)
        plotter.set_plot_style()
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Determine colors based on color_by
        color_list = []
        if color_by == 'buffer_category':
            # Extract buffer_category from labels
            unique_categories = set()
            for label in labels_filtered:
                parts = label.split(' | ')
                if 'buffer_category' == primary_cat and len(parts) > 0:
                    unique_categories.add(parts[0])
                elif 'buffer_category' == secondary_cat and len(parts) > 1:
                    unique_categories.add(parts[1])
                elif 'buffer_category' == tertiary_cat and len(parts) > 2:
                    unique_categories.add(parts[2])
                elif 'buffer_category' == quaternary_cat and len(parts) > 3:
                    unique_categories.add(parts[3])
            unique_categories = sorted(unique_categories)
            palette_colors = sns.color_palette(n_colors=len(unique_categories))
            color_map = dict(zip(unique_categories, palette_colors))
            
            for label in labels_filtered:
                parts = label.split(' | ')
                cat_value = None
                if 'buffer_category' == primary_cat and len(parts) > 0:
                    cat_value = parts[0]
                elif 'buffer_category' == secondary_cat and len(parts) > 1:
                    cat_value = parts[1]
                elif 'buffer_category' == tertiary_cat and len(parts) > 2:
                    cat_value = parts[2]
                elif 'buffer_category' == quaternary_cat and len(parts) > 3:
                    cat_value = parts[3]
                color_list.append(color_map.get(cat_value, sns.color_palette()[0]))
        
        elif color_by == 'ionic_strength_category':
            # Extract ionic_strength_category from labels
            unique_categories = set()
            for label in labels_filtered:
                parts = label.split(' | ')
                if 'ionic_strength_category' == primary_cat and len(parts) > 0:
                    unique_categories.add(parts[0])
                elif 'ionic_strength_category' == secondary_cat and len(parts) > 1:
                    unique_categories.add(parts[1])
                elif 'ionic_strength_category' == tertiary_cat and len(parts) > 2:
                    unique_categories.add(parts[2])
                elif 'ionic_strength_category' == quaternary_cat and len(parts) > 3:
                    unique_categories.add(parts[3])
            unique_categories = sorted(unique_categories)
            palette_colors = sns.color_palette(n_colors=len(unique_categories))
            color_map = dict(zip(unique_categories, palette_colors))
            
            for label in labels_filtered:
                parts = label.split(' | ')
                cat_value = None
                if 'ionic_strength_category' == primary_cat and len(parts) > 0:
                    cat_value = parts[0]
                elif 'ionic_strength_category' == secondary_cat and len(parts) > 1:
                    cat_value = parts[1]
                elif 'ionic_strength_category' == tertiary_cat and len(parts) > 2:
                    cat_value = parts[2]
                elif 'ionic_strength_category' == quaternary_cat and len(parts) > 3:
                    cat_value = parts[3]
                color_list.append(color_map.get(cat_value, sns.color_palette()[0]))
        
        elif color_by == 'PEG_concentration':
            # Extract PEG_concentration from labels
            unique_categories = set()
            for label in labels_filtered:
                parts = label.split(' | ')
                if 'PEG_concentration' == primary_cat and len(parts) > 0:
                    unique_categories.add(parts[0])
                elif 'PEG_concentration' == secondary_cat and len(parts) > 1:
                    unique_categories.add(parts[1])
                elif 'PEG_concentration' == tertiary_cat and len(parts) > 2:
                    unique_categories.add(parts[2])
                elif 'PEG_concentration' == quaternary_cat and len(parts) > 3:
                    unique_categories.add(parts[3])
            unique_categories = sorted(unique_categories)
            palette_colors = sns.color_palette(n_colors=len(unique_categories))
            color_map = dict(zip(unique_categories, palette_colors))
            
            for label in labels_filtered:
                parts = label.split(' | ')
                cat_value = None
                if 'PEG_concentration' == primary_cat and len(parts) > 0:
                    cat_value = parts[0]
                elif 'PEG_concentration' == secondary_cat and len(parts) > 1:
                    cat_value = parts[1]
                elif 'PEG_concentration' == tertiary_cat and len(parts) > 2:
                    cat_value = parts[2]
                elif 'PEG_concentration' == quaternary_cat and len(parts) > 3:
                    cat_value = parts[3]
                color_list.append(color_map.get(cat_value, sns.color_palette()[0]))
        
        elif color_by == 'buffer_matches_same':
            # All buffer_matches get the same color
            default_color = sns.color_palette()[0]
            color_list = [default_color] * len(labels_filtered)
        
        elif color_by == 'last_factor':
            # Color by last factor in label
            unique_last_factors = set()
            for label in labels_filtered:
                if ' | ' in label:
                    last_factor = label.split(' | ')[-1]
                    unique_last_factors.add(last_factor)
            unique_last_factors = sorted(unique_last_factors)
            palette_colors = sns.color_palette(n_colors=len(unique_last_factors))
            color_map = dict(zip(unique_last_factors, palette_colors))
            
            for label in labels_filtered:
                if ' | ' in label:
                    last_factor = label.split(' | ')[-1]
                    color_list.append(color_map.get(last_factor, sns.color_palette()[0]))
                else:
                    color_list.append(sns.color_palette()[0])
        
        else:
            # No coloring - all same color
            default_color = sns.color_palette()[0]
            color_list = [default_color] * len(labels_filtered)
        
        # Create box plot
        ax = plt.gca()
        bp = ax.boxplot(data_values_filtered, positions=range(len(labels_filtered)), 
                       widths=0.6, patch_artist=True)
        
        # Apply colors to each box
        for i, (box, color) in enumerate(zip(bp['boxes'], color_list)):
            box.set_facecolor(color)
            box.set_alpha(0.7)
            box.set_edgecolor('black')
            box.set_linewidth(1)
        
        # Style other box plot elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in bp:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1)
        
        # Set x-axis labels below ticks
        ax.set_xticks(range(len(labels_filtered)))
        ax.set_xticklabels(labels_filtered, rotation=90, ha='center', va='top')
        
        plt.ylabel(metric_label)
        
        # Set x-axis label based on the last factor (secondary_cat for 2-factor plots)
        x_axis_label = ''
        if secondary_cat and not tertiary_cat and not quaternary_cat:
            # This is a 2-factor plot, use secondary_cat as x-axis label
            # Convert category name to readable label
            label_map = {
                'PEG_MW_category': 'PEG Molecular Weight',
                'PEG_concentration': 'PEG Concentration',
                'ionic_strength_category': 'Ionic Strength',
                'buffer_category': 'Buffer Concentration',
                'buffer_matches': 'Buffer',
                'msa_class': 'MSA Class'
            }
            x_axis_label = label_map.get(secondary_cat, secondary_cat.replace('_', ' ').title())
        plt.xlabel(x_axis_label)
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)
        
        # Save plot
        title = f"{name}_{self.pred_table}"
        plotter.save_plot(title)
        plt.close(fig)

    def create_three_factor_violin(self, metric_label: str, cat1: str, cat2: str, cat3: str):
        """Create a violin plot where groups are combinations of three categorical variables.
        Order is nested: iterate cat1 order, inside iterate cat2 order, inside iterate cat3 order.
        """
        if metric_label not in self.metric_column_map:
            return
        metric_col = self.metric_column_map[metric_label]
        if metric_col not in self.df.columns:
            return
        if cat1 not in self.df.columns or cat2 not in self.df.columns or cat3 not in self.df.columns:
            return

        df = self.df[[metric_col, cat1, cat2, cat3]].dropna()
        if df.empty:
            return

        order1 = self.category_orders.get(cat1)
        order2 = self.category_orders.get(cat2)
        order3 = self.category_orders.get(cat3)

        # Build groups
        data_values = []
        labels = []

        vals1 = order1 if order1 else sorted(df[cat1].unique())
        vals2 = order2 if order2 else sorted(df[cat2].unique())
        vals3 = order3 if order3 else sorted(df[cat3].unique())

        for v1 in vals1:
            sub1 = df[df[cat1] == v1]
            if sub1.empty:
                continue
            for v2 in vals2:
                sub2 = sub1[sub1[cat2] == v2]
                if sub2.empty:
                    continue
                for v3 in vals3:
                    vals = sub2[sub2[cat3] == v3][metric_col].values
                    if len(vals) > 0:
                        data_values.append(vals)
                        labels.append(f"{v1} | {v2} | {v3}")

        if not data_values:
            return

        # Determine what to color by: buffer_category > ionic_strength_category > PEG_concentration > buffer_matches (same color)
        color_by = None
        if 'buffer_category' in [cat1, cat2, cat3]:
            color_by = 'buffer_category'
        elif 'ionic_strength_category' in [cat1, cat2, cat3]:
            color_by = 'ionic_strength_category'
        elif 'PEG_concentration' in [cat1, cat2, cat3]:
            color_by = 'PEG_concentration'
        elif 'buffer_matches' in [cat1, cat2, cat3]:
            color_by = 'buffer_matches_same'  # Special flag for same color
        else:
            # Default: color by last factor (cat3)
            color_by = 'last_factor'
        
        # Create box plot with coloring
        self._create_box_plot_with_colors(
            metric_label=metric_label,
            data_values=data_values,
            labels=labels,
            name=f"{metric_label}_by_{cat1}_and_{cat2}_and_{cat3}",
            color_by=color_by,
            primary_cat=cat1,
            secondary_cat=cat2,
            tertiary_cat=cat3,
            df=df
        )
    
    def create_four_factor_violin(self, metric_label: str, cat1: str, cat2: str, cat3: str, cat4: str):
        """Create a violin plot where groups are combinations of four categorical variables.
        Order is nested: iterate cat1 order, inside iterate cat2 order, inside iterate cat3 order, inside iterate cat4 order.
        """
        if metric_label not in self.metric_column_map:
            return
        metric_col = self.metric_column_map[metric_label]
        if metric_col not in self.df.columns:
            return
        if cat1 not in self.df.columns or cat2 not in self.df.columns or cat3 not in self.df.columns or cat4 not in self.df.columns:
            return

        df = self.df[[metric_col, cat1, cat2, cat3, cat4]].dropna()
        if df.empty:
            return

        order1 = self.category_orders.get(cat1)
        order2 = self.category_orders.get(cat2)
        order3 = self.category_orders.get(cat3)
        order4 = self.category_orders.get(cat4)

        # Build groups
        data_values = []
        labels = []

        vals1 = order1 if order1 else sorted(df[cat1].unique())
        vals2 = order2 if order2 else sorted(df[cat2].unique())
        vals3 = order3 if order3 else sorted(df[cat3].unique())
        vals4 = order4 if order4 else sorted(df[cat4].unique())

        for v1 in vals1:
            sub1 = df[df[cat1] == v1]
            if sub1.empty:
                continue
            for v2 in vals2:
                sub2 = sub1[sub1[cat2] == v2]
                if sub2.empty:
                    continue
                for v3 in vals3:
                    sub3 = sub2[sub2[cat3] == v3]
                    if sub3.empty:
                        continue
                    for v4 in vals4:
                        vals = sub3[sub3[cat4] == v4][metric_col].values
                        if len(vals) > 0:
                            data_values.append(vals)
                            labels.append(f"{v1} | {v2} | {v3} | {v4}")

        if not data_values:
            return

        # Determine what to color by: buffer_category > ionic_strength_category > PEG_concentration > buffer_matches (same color)
        color_by = None
        if 'buffer_category' in [cat1, cat2, cat3, cat4]:
            color_by = 'buffer_category'
        elif 'ionic_strength_category' in [cat1, cat2, cat3, cat4]:
            color_by = 'ionic_strength_category'
        elif 'PEG_concentration' in [cat1, cat2, cat3, cat4]:
            color_by = 'PEG_concentration'
        elif 'buffer_matches' in [cat1, cat2, cat3, cat4]:
            color_by = 'buffer_matches_same'  # Special flag for same color
        else:
            # Default: color by last factor (cat4)
            color_by = 'last_factor'
        
        # Create box plot with coloring
        self._create_box_plot_with_colors(
            metric_label=metric_label,
            data_values=data_values,
            labels=labels,
            name=f"{metric_label}_by_{cat1}_and_{cat2}_and_{cat3}_and_{cat4}",
            color_by=color_by,
            primary_cat=cat1,
            secondary_cat=cat2,
            tertiary_cat=cat3,
            quaternary_cat=cat4,
            df=df
        )


def run_peg_analysis(msa_option=None, single_chain_only=False, rbp_template_option=None, rnalength_filter: str = None, condition_filter: str = None, create_multi_factor_plots: bool = False):
    plotter = PEGAnalysisPlotter(msa_option=msa_option, single_chain_only=single_chain_only,
                                 rbp_template_option=rbp_template_option, rnalength_filter=rnalength_filter, 
                                 condition_filter=condition_filter, create_multi_factor_plots=create_multi_factor_plots)
    plotter.create_all_plots()
    print("Plots saved under results/plots/... by PlotCreator (all_chains/PEG_plots)")


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
# Composition reuses the PNGs this script already saved (run it first).
_COMPOSITE_PLOT_TYPES = ("PEG_plots",)


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None,
                     rbp_template_option=None):
    """Return {base_name: (draw_fn, title)} from this script's saved PNGs."""
    from plots.publication_figure import saved_png_panels
    return saved_png_panels(
        _COMPOSITE_PLOT_TYPES, names, single_chain_only=single_chain_only,
        msa_option=msa_option, rbp_template_option=rbp_template_option)


if __name__ == "__main__":
    import sys
    import argparse

    if sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, sys.argv[2:],
                                           default_out="peg_composite"))

    # Parse command-line arguments
    single_chain_only = False
    msa_option = None
    rbp_template_option = None
    rnalength_filter = None
    condition_filter = None
    
    # Parse arguments (case-insensitive, any order)
    # Handle argparse-style arguments first
    parser = argparse.ArgumentParser(description='PEG analysis plots from DB', prefix_chars='-+')
    parser.add_argument('--rnalength', type=str, help='Optional RNALength filter, e.g., "< 100" or "== 150"')
    parser.add_argument('--condition', type=str, help='Optional condition filter, e.g., "RNA_isDuplex = 1"')
    parser.add_argument('+msa', dest='msa_plus', action='store_true', help='Filter to +MSA entries')
    parser.add_argument('-msa', dest='msa_minus', action='store_true', help='Filter to -MSA entries')
    parser.add_argument('--single-chain', dest='single_chain', action='store_true', help='Only single-chain complexes')

    # Parse known args to handle argparse-style arguments
    args, unknown = parser.parse_known_args()

    # Initialize create_multi_factor_plots
    create_multi_factor_plots = False

    # Handle argparse-style MSA arguments
    if args.msa_plus and args.msa_minus:
        print('Both -msa and +msa provided; ignoring both and using all entries.')
    elif args.msa_plus:
        msa_option = '+MSA'
        create_multi_factor_plots = True  # Enable multi-factor plots when MSA option is specified
    elif args.msa_minus:
        msa_option = '-MSA'
        create_multi_factor_plots = True  # Enable multi-factor plots when MSA option is specified

    # Handle argparse-style single-chain argument
    if args.single_chain:
        single_chain_only = True
    
    # Handle argparse-style filters
    if args.rnalength:
        rnalength_filter = args.rnalength
    if args.condition:
        condition_filter = args.condition
    
    # Also check unknown args and sys.argv for RBP template and other formats
    all_args = unknown + sys.argv[1:]
    # create_multi_factor_plots already initialized above
    for arg in all_args:
        arg_lower = arg.lower()
        if arg_lower in ['single_chain', 'all']:
            single_chain_only = (arg_lower == 'single_chain')
            if arg_lower == 'all':
                create_multi_factor_plots = True
        elif arg_lower in ['+msa', '-msa', 'msa']:
            if msa_option is None:
                msa_option = '+MSA' if arg_lower in ['+msa', 'msa'] else '-MSA'
            # Allow multi-factor plots when MSA option is specified
            create_multi_factor_plots = True
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case
    
    # If no arguments provided (only script name) or "all" is explicitly provided, create multi-factor plots
    if len(sys.argv) == 1 or 'all' in [a.lower() for a in all_args]:
        create_multi_factor_plots = True

    run_peg_analysis(msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option,
                     rnalength_filter=rnalength_filter, condition_filter=condition_filter, create_multi_factor_plots=create_multi_factor_plots)

# python peg_analysis_plots.py -msa --rnalength "< 10"
# --rnalength "BETWEEN 10 AND 20"
# python peg_analysis_plots.py +msa --rnalength ">10"
# python peg_analysis_plots.py --single-chain --rnalength ">10"
# python peg_analysis_plots.py --condition "RNA_isDuplex = 1"



