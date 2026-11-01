import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
from typing import List, Dict, Tuple, Optional
from scipy import stats
from database.databaseMethods import DatabaseMethods
from plots.plotCreator import PlotCreator
from database.startConfig import StartConfig
from utils.statistics_utils import get_significance_stars, format_p_value
from utils.column_labels import get_column_label
from utils.plotting_utils import (
    process_rna_inf, process_rmsd, process_tm, process_lddt, 
    process_pae, process_plddt, process_ptm, process_rna_length, 
    process_msa_sizes
)

# Color scheme dictionary - expanded to 16 colors
COLOR_SCHEME = {
    'green': '#2ecc71',
    'orange': '#ff7f0e',
    'blue': '#1f77b4',
    'magenta': '#e73c8e',
    'red': '#e74c3c',
    'purple': '#9b59b6',
    'cyan': '#1abc9c',
    'yellow': '#f1c40f',
    'pink': '#ec7063',
    'teal': '#16a085',
    'lime': '#7dcea0',
    'indigo': '#5b2c6f',
    'coral': '#ff6b6b',
    'navy': '#34495e',
    'olive': '#95a5a6',
    'maroon': '#8b0000'
}

# Default color order for automatic assignment (16 colors)
DEFAULT_COLOR_ORDER = [
    'green', 'orange', 'blue', 'magenta', 'red', 'purple', 'cyan', 'yellow',
    'pink', 'teal', 'lime', 'indigo', 'coral', 'navy', 'olive', 'maroon'
]

class MetricsComparisonPlotter3DB:
    def __init__(
        self,
        db_paths: List[str],
        db_labels: List[str],
        db_colors: List[str],
        single_chain_only: bool = False,
        msa_option: Optional[str] = None,
        rbp_template_option: Optional[str] = None
    ):
        if len(db_paths) < 2 or len(db_paths) > 16:
            raise ValueError(f"Number of databases must be between 2 and 16, got {len(db_paths)}")
        if len(db_paths) != len(db_labels) or len(db_paths) != len(db_colors):
            raise ValueError("Number of paths, labels, and colors must match")
        
        self.db_paths = db_paths
        self.db_labels = db_labels
        self.num_databases = len(db_paths)
        self.db_methods = DatabaseMethods()
        self.config = StartConfig()
        self.msa_option = msa_option
        self.rbp_template_option = rbp_template_option

        # Use PlotCreator for saving plots in the correct directory
        self.plot_creator = PlotCreator(plot_type="compare_databases", msa_option=msa_option, single_chain_only=single_chain_only, rbp_template_option=rbp_template_option)
        
        # Define metric groups
        # Check if protein metrics and interface metrics should be included
        if self.config.pred_table == "pred_rna_rna":
            protein_metrics = []
            interface_metrics = []
        else:
            protein_metrics = ['Protein_RMSD', 'Protein_LDDT', 'Protein_TM']
            interface_metrics = ['iLDDT', 'QS_global', 'ICS']
        
        self.metric_groups = {
            'a': ['RNA_RMSD'],
            'b': ['RNA_LDDT', 'RNA_TM', 'RNA_LDDT_allAtom', 'af3_rna_pTM', 'af3_rna_ipTM', 'af3_rna_pLDDT_avg'],# 'RNAComplex_TM'],
            'c': ['RNA_inf', 'RNA_WC', 'RNA_nonWC', 'RNA_STACK'],
            'd': ['RNA_wl'],
            'e': ['af3_global_pae_avg', 'af3_chain_pair_pae_min'],
            'f': ['Complex_RMSD', 'Complex_TM', 'Complex_LDDT', 'Complex_LDDT_allAtom'] + interface_metrics,
            'g': protein_metrics + (['Dockq'] if interface_metrics else [])
        }
        
        # Define colors for plots - use list for seaborn compatibility
        self.colors = {label: color for label, color in zip(db_labels, db_colors)}
        # Also create a list version in the order of db_labels for seaborn palette
        self.colors_list = db_colors
        
        # Define length classes
        self.length_classes = {
            '0-20': (0, 20),
            '20-200': (20, 200),
            '200+': (200, float('inf'))
        }
        
        # Load data upon initialization
        self.dataframes = []
        for db_path in db_paths:
            df = self._load_and_process_data(db_path, single_chain_only, msa_option, rbp_template_option)
            self.dataframes.append(df)
        
        # Initialize statistics file
        self.stats_file_path = os.path.join(self.plot_creator.results_dir, 'database_comparison_statistics.txt')
        self._initialize_stats_file()

    def _load_and_process_data(self, db_path: str, single_chain_only: bool = False, 
                               msa_option: Optional[str] = None, rbp_template_option: Optional[str] = None) -> pd.DataFrame:
        """Load data from SQLite and process all metrics."""
        conn = sqlite3.connect(db_path)
        
        # Get table name from StartConfig
        pred_table = self.config.pred_table
        
        # Get all column names from the table automatically
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({pred_table})")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]  # Column name is at index 1
        
        # Build query with all available columns
        columns_str = ", ".join(column_names)
        
        query = f"""
        SELECT {columns_str}
        FROM {pred_table}
        """
        if single_chain_only:
            query += " WHERE NumberRNAs = 1"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Define metric processors - centralized processing configuration
        from utils.plotting_utils import process_dataframe_metrics
        
        metric_processors = {
            # RMSD metrics
            'RNA_RMSD': process_rmsd,
            'Complex_RMSD': process_rmsd,
            'Protein_RMSD': process_rmsd,
            # TM metrics
            'RNA_TM': process_tm,
            'Protein_TM': process_tm,
            'Complex_TM': process_tm,
            #'RNAComplex_TM': process_tm,
            # LDDT metrics
            'RNA_LDDT': process_lddt,
            'Protein_LDDT': process_lddt,
            'Complex_LDDT': process_lddt,
            # PAE metrics
            'af3_global_pae_avg': process_pae,
            'af3_chain_pair_pae_min': process_pae,
            # pTM/ipTM metrics (need RNALength dependency)
            'af3_rna_pTM': (process_ptm, 'RNALength'),
            'af3_rna_ipTM': (process_ptm, 'RNALength'),
            # pLDDT metrics
            'af3_rna_pLDDT_avg': process_plddt,
        }
        
        # Process all metrics using centralized function
        df = process_dataframe_metrics(df, metric_processors)
        
        # Process classifications - using centralized functions
        df['RNALength'] = df['RNALength'].apply(process_rna_length)
        df['MSAsize'] = df.apply(lambda row: process_msa_sizes(row, 'RNA_msa_size', 'RNAChainIDs', 'RNALength'), axis=1)
        
        df['length_class'] = pd.cut(
            df['RNALength'],
            bins=[0, 20, 200, float('inf')],
            labels=['0-20', '20-200', '200+']
        )
        
        df['msa_class'] = pd.cut(
            df['MSAsize'],
            bins=[-float('inf'), 1, 1000, float('inf')],
            labels=['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$']
        )
        
        # Process RNA_INF using centralized function
        if 'RNA_INF' in df.columns:
            rna_inf_df = df['RNA_INF'].apply(process_rna_inf)
            df = pd.concat([df, rna_inf_df], axis=1)
        df['RNA_wl'] = pd.to_numeric(df['RNA_wl'], errors='coerce')
        
        # Apply MSA filter
        if msa_option:
            df = self._apply_msa_filter(df, msa_option)
        
        # Apply RBP template filter
        if rbp_template_option:
            df = self._apply_rbp_filter(df, rbp_template_option)
        
        return df
    
    def _apply_msa_filter(self, df: pd.DataFrame, msa_option: str) -> pd.DataFrame:
        """Apply MSA filter to dataframe."""
        if msa_option not in ['+MSA', '-MSA']:
            return df
        
        # Try to use RNA_msa_size first (dictionary format)
        if 'RNA_msa_size' in df.columns:
            from utils.plotting_utils import classify_msa
            df['msa_status'] = df.apply(lambda row: classify_msa(row, 'RNA_msa_size', 'RNAChainIDs', 'RNALength'), axis=1)
            if msa_option == '+MSA':
                df = df[df['msa_status'] == '+rMSA']
            else:  # -MSA
                df = df[df['msa_status'] == '-rMSA']
            df = df.drop(columns=['msa_status'], errors='ignore')
        # Fallback to af3_rna_MSA if available
        elif 'af3_rna_MSA' in df.columns:
            def get_msa_value(msa_str):
                try:
                    if pd.isna(msa_str):
                        return None
                    if isinstance(msa_str, str):
                        from utils.plotting_utils import safe_literal_eval
                        values = safe_literal_eval(msa_str)
                        if isinstance(values, list):
                            return 1 if any(v == 1 for v in values) else 0
                    return msa_str
                except:
                    return None
            
            df['msa_value'] = df['af3_rna_MSA'].apply(get_msa_value)
            if msa_option == '+MSA':
                df = df[df['msa_value'] == 1]
            else:  # -MSA
                df = df[df['msa_value'] == 0]
            df = df.drop(columns=['msa_value'], errors='ignore')
        
        return df
    
    def _apply_rbp_filter(self, df: pd.DataFrame, rbp_template_option: str) -> pd.DataFrame:
        """Apply RBP template filter to dataframe."""
        if 'RBP_PDB_Templates' not in df.columns:
            return df
        
        rbp_upper = rbp_template_option.upper()
        if rbp_upper in ['+RBPTEMP', '+RBP']:
            # Filter for samples with RBP templates
            df = df[
                (df['RBP_PDB_Templates'].notna()) &
                (df['RBP_PDB_Templates'] != '') &
                (df['RBP_PDB_Templates'] != '[]') &
                (df['RBP_PDB_Templates'] != 'null')
            ]
        elif rbp_upper in ['-RBPTEMP', '-RBP']:
            # Filter for samples without RBP templates
            df = df[
                (df['RBP_PDB_Templates'].isna()) |
                (df['RBP_PDB_Templates'] == '') |
                (df['RBP_PDB_Templates'] == '[]') |
                (df['RBP_PDB_Templates'] == 'null')
            ]
        
        return df

    def _get_display_label(self, metric: str) -> str:
        """Convert metric name to display label using centralized labels."""
        return get_column_label(metric)
    
    def _initialize_stats_file(self):
        """Initialize the statistics file with header information."""
        with open(self.stats_file_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Database Comparison Statistics\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of databases compared: {self.num_databases}\n")
            f.write(f"Databases: {', '.join(self.db_labels)}\n")
            filter_parts = []
            if self.plot_creator.single_chain_only:
                filter_parts.append("Single-chain RNAs only")
            else:
                filter_parts.append("All chains")
            if self.msa_option:
                filter_parts.append(self.msa_option)
            if self.rbp_template_option:
                rbp_upper = self.rbp_template_option.upper()
                if rbp_upper in ['+RBPTEMP', '+RBP']:
                    filter_parts.append("+RBP-template")
                elif rbp_upper in ['-RBPTEMP', '-RBP']:
                    filter_parts.append("-RBP-template")
                else:
                    filter_parts.append(self.rbp_template_option)
            f.write(f"Filter: {', '.join(filter_parts)}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    def _calculate_database_statistics(self, combined_df: pd.DataFrame, metric: str) -> Dict:
        """Calculate statistics and significance tests for a metric across databases."""
        # Filter out NaN values
        df_clean = combined_df[[metric, 'database']].dropna()
        
        if df_clean.empty:
            return None
        
        # Convert to numeric
        df_clean[metric] = pd.to_numeric(df_clean[metric], errors='coerce')
        df_clean = df_clean[df_clean[metric].notna()]
        
        if df_clean.empty:
            return None
        
        # Calculate statistics for each database
        db_stats = {}
        for db_label in self.db_labels:
            values = df_clean[df_clean['database'] == db_label][metric].values
            if len(values) > 0:
                db_stats[db_label] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'count': len(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Calculate pairwise p-values using Mann-Whitney U test
        p_values = {}
        db_labels_list = list(self.db_labels)
        
        for i, db1 in enumerate(db_labels_list):
            for db2 in db_labels_list[i+1:]:
                if db1 in db_stats and db2 in db_stats:
                    values1 = db_stats[db1]['values']
                    values2 = db_stats[db2]['values']
                    
                    if len(values1) > 0 and len(values2) > 0:
                        try:
                            statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            comparison_key = f"{db1} vs {db2}"
                            p_values[comparison_key] = p_value
                        except Exception:
                            p_values[f"{db1} vs {db2}"] = None
        
        # Find most significant differences
        significant_comparisons = []
        if p_values:
            sorted_p_values = sorted([(k, v) for k, v in p_values.items() if v is not None], 
                                    key=lambda x: x[1])
            significant_comparisons = sorted_p_values[:5]  # Top 5 most significant
        
        return {
            'db_stats': db_stats,
            'p_values': p_values,
            'most_significant': significant_comparisons
        }
    
    def _write_metric_statistics(self, metric: str, stats_dict: Dict):
        """Write statistics for a metric to the statistics file."""
        if stats_dict is None:
            return
        
        with open(self.stats_file_path, 'a') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Metric: {self._get_display_label(metric)} ({metric})\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Write statistics for each database
            f.write("Database Statistics:\n")
            f.write("-" * 80 + "\n")
            for db_label in self.db_labels:
                if db_label in stats_dict['db_stats']:
                    stats = stats_dict['db_stats'][db_label]
                    f.write(f"\n{db_label}:\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std: {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Min: {stats['min']:.4f}\n")
                    f.write(f"  Max: {stats['max']:.4f}\n")
            
            # Write pairwise comparisons
            if stats_dict['p_values']:
                f.write(f"\n\nPairwise Comparisons (Mann-Whitney U test):\n")
                f.write("-" * 80 + "\n")
                for comparison, p_value in sorted(stats_dict['p_values'].items(), key=lambda x: x[1] if x[1] is not None else float('inf')):
                    if p_value is not None:
                        stars = get_significance_stars(p_value)
                        p_str = format_p_value(p_value)
                        f.write(f"  {comparison}: p = {p_str} {stars}\n")
                    else:
                        f.write(f"  {comparison}: p = N/A\n")
                
                # Write most significant differences
                if stats_dict['most_significant']:
                    f.write(f"\nMost Significant Differences:\n")
                    f.write("-" * 80 + "\n")
                    for comparison, p_value in stats_dict['most_significant']:
                        stars = get_significance_stars(p_value)
                        p_str = format_p_value(p_value)
                        f.write(f"  {comparison}: p = {p_str} {stars}\n")
            else:
                f.write("\n(No pairwise comparisons available)\n")

    def _create_comparison_box_plot(self, metric: str, group_by: str):
        """Creates a grouped box plot comparing the databases."""
        self.plot_creator.set_plot_style()
        
        # Create copies and add database labels
        dfs_with_labels = []
        for df, label in zip(self.dataframes, self.db_labels):
            df_copy = df.copy()
            df_copy['database'] = label
            # Filter out empty/all-NA columns to avoid FutureWarning
            df_copy = df_copy.dropna(axis=1, how='all')
            dfs_with_labels.append(df_copy)

        # Concatenate with explicit handling of empty columns
        combined_df = pd.concat(dfs_with_labels, ignore_index=True, sort=False)
        
        # Define order for plotting
        group_order = None
        if group_by == 'length_class':
            group_order = ['0-20', '20-200', '200+']
        elif group_by == 'msa_class':
            group_order = ['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$']

        # Filter and clean data
        if group_order:
            combined_df = combined_df[combined_df[group_by].isin(group_order)]
        combined_df.dropna(subset=[metric, group_by], inplace=True)

        if combined_df.empty:
            print(f"No data to plot for metric {metric} grouped by {group_by}")
            return
        
        # Calculate and write statistics for this metric
        stats_dict = self._calculate_database_statistics(combined_df, metric)
        if stats_dict:
            self._write_metric_statistics(metric, stats_dict)

        plt.figure(figsize=(15, 8))
        # Use colors_list to ensure all databases are plotted
        ax = sns.boxplot(
            data=combined_df, x=group_by, y=metric, hue='database',
            palette=self.colors_list, order=group_order, hue_order=self.db_labels
        )
        plt.ylabel(self._get_display_label(metric))
        if group_by == 'length_class':
            plt.xlabel("RNA Length")
        else:
            plt.xlabel("MSA$_r$ Size")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        # tm_lddt_metrics = ['RNA_TM', 'RNAComplex_TM', 'RNA_LDDT', 'af3_rna_pTM', 'af3_rna_ipTM', 'af3_rna_pLDDT_avg', 'Complex_TM']
        tm_lddt_metrics = ['RNA_TM', 'RNA_LDDT', 'af3_rna_pTM', 'af3_rna_ipTM', 'af3_rna_pLDDT_avg',
                           'Complex_TM']
        if 'Protein_TM' in combined_df.columns:
            tm_lddt_metrics.append('Protein_TM')
        # if metric in tm_lddt_metrics:
        #      plt.ylim(0, 1.05)
        if metric not in {'RNA_RMSD', 'Complex_RMSD', 'Protein_RMSD'}:
            plt.ylim(0, 1)
        # Handle metrics that range from -1 to 1
        elif metric in ['RNA_inf', 'RNA_WC', 'RNA_nonWC', 'RNA_STACK']:
            plt.ylim(-1.05, 1.05)
            # Safety: ensure axis not inverted
            ymin, ymax = ax.get_ylim()
            if ymin > ymax:
                ax.invert_yaxis()

        for i, group in enumerate(ax.get_xticklabels()):
            group_name = group.get_text()
            counts = []
            for label in self.db_labels:
                count = len(combined_df[(combined_df[group_by] == group_name) & (combined_df['database'] == label)])
                counts.append(str(count))
            group.set_text(f"{group_name}\n(n={'/'.join(counts)})")

        # Place legend outside if more than 4 databases
        # Get existing legend handles and update labels
        handles, original_labels = ax.get_legend_handles_labels()
        formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
        if self.num_databases > 4:
            ax.legend(handles=handles, labels=formatted_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend(handles=handles, labels=formatted_labels)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
            plt.tight_layout()

        filename = f'{metric}_by_{group_by}'
        self.plot_creator.save_plot(filename)
        plt.close()

        # Combined metrics box plot
        rna_metrics = ["RNA_LDDT", "RNA_TM", "RNA_LDDT_allAtom"]
        # rna_metrics = ["RNA_LDDT", "RNA_TM", "RNA_LDDT_allAtom", "RNAComplex_TM"]
        rna_df = combined_df.melt(
            id_vars=["database"],
            value_vars=[m for m in rna_metrics if m in combined_df.columns],
            var_name="metric",
            value_name="value"
        ).dropna(subset=["value"])

        if not rna_df.empty:
            plt.figure(figsize=(12, 7))
            ax = sns.boxplot(
                data=rna_df, x="metric", y="value", hue="database",
                palette=self.colors_list, hue_order=self.db_labels
            )
            plt.ylabel("")
            plt.xlabel("")
            plt.ylim(0, 1)
            # Update x-axis labels to use display labels
            current_labels = [label.get_text() for label in ax.get_xticklabels()]
            display_labels = [self._get_display_label(label) for label in current_labels]
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(display_labels, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            # Place legend outside if more than 4 databases
            # Get existing legend handles and update labels
            handles, original_labels = ax.get_legend_handles_labels()
            formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
            if self.num_databases > 4:
                legend = ax.legend(handles=handles, labels=formatted_labels, title="", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                legend = ax.legend(handles=handles, labels=formatted_labels, title="")
            if legend:
                legend.set_title(None)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
                plt.tight_layout()
            self.plot_creator.save_plot("RNA_metrics_comparison")
            plt.close()

        # Complex metrics box plot
        complex_metrics = ["Complex_TM", "Complex_LDDT_allAtom", "RNA_TM", "RNA_LDDT_allAtom"]
        # complex_metrics = ["Complex_TM", "Complex_LDDT_allAtom", "RNAComplex_TM", "RNA_TM", "RNA_LDDT_allAtom"]
        # Add interface metrics only if they exist in the dataframe
        if 'iLDDT' in combined_df.columns:
            complex_metrics.insert(2, "iLDDT")
        complex_df = combined_df.melt(
            id_vars=["database"],
            value_vars=[m for m in complex_metrics if m in combined_df.columns],
            var_name="metric",
            value_name="value"
        ).dropna(subset=["value"])

        if not complex_df.empty:
            # Calculate statistics for each complex metric
            for metric_name in complex_df['metric'].unique():
                metric_df = complex_df[complex_df['metric'] == metric_name].copy()
                metric_df = metric_df.rename(columns={'value': metric_name})
                stats_dict = self._calculate_database_statistics(metric_df, metric_name)
                if stats_dict:
                    self._write_metric_statistics(metric_name, stats_dict)

            plt.figure(figsize=(12, 7))
            ax = sns.boxplot(
                data=complex_df, x="metric", y="value", hue="database",
                palette=self.colors_list, hue_order=self.db_labels
            )
            plt.ylabel("")
            plt.xlabel("")
            plt.ylim(0, 1)
            # Update x-axis labels to use display labels
            current_labels = [label.get_text() for label in ax.get_xticklabels()]
            display_labels = [self._get_display_label(label) for label in current_labels]
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(display_labels, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            # Place legend outside if more than 4 databases
            # Get existing legend handles and update labels
            handles, original_labels = ax.get_legend_handles_labels()
            formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
            if self.num_databases > 4:
                legend = ax.legend(handles=handles, labels=formatted_labels, title="", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                legend = ax.legend(handles=handles, labels=formatted_labels, title="")
            if legend:
                legend.set_title(None)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
                plt.tight_layout()
            self.plot_creator.save_plot("Complex_metrics_comparison")
            plt.close()

        if group_by == 'msa_class' and "RNALength" in combined_df.columns:
            length_df = combined_df.dropna(subset=["RNALength", "msa_class"]).copy()
            length_df["RNALength"] = length_df["RNALength"].apply(process_rna_length)
            length_df["RNALength"] = pd.to_numeric(length_df["RNALength"], errors='coerce')
            length_df = length_df[length_df["msa_class"].isin(['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$'])]

            if not length_df.empty:
                plt.figure(figsize=(12, 7))
                ax = sns.boxplot(
                    data=length_df, x="msa_class", y="RNALength", hue="database",
                    palette=self.colors_list, hue_order=self.db_labels,
                    order=['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$']
                )
                plt.title("RNA Length (all) vs MSA$_r$ Size")
                plt.ylabel("RNA Length")
                plt.xlabel("MSA$_r$ Size")
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7)
                # Place legend outside if more than 4 databases
                # Get existing legend handles and update labels
                handles, original_labels = ax.get_legend_handles_labels()
                formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
                if self.num_databases > 4:
                    legend = ax.legend(handles=handles, labels=formatted_labels, title="", bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    legend = ax.legend(handles=handles, labels=formatted_labels, title="")
                if legend:
                    legend.set_title(None)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
                    plt.tight_layout()
                self.plot_creator.save_plot("RNA_Length_vs_MSA_all")
                plt.close()

            # # ---- Plot RNALength < 200 ----
            # length_lt200 = length_df[length_df["RNALength"] < 200]
            # if not length_lt200.empty:
            #     plt.figure(figsize=(12, 7))
            #     ax = sns.boxplot(
            #         data=length_lt200, x="msa_class", y="RNALength", hue="database",
            #         palette=self.colors_list, hue_order=self.db_labels,
            #         order=['0-20 MSA$_r$', '21-1000 MSA$_r$', '+1000 MSA$_r$']
            #     )
            #     plt.title("RNA Length (<200) vs MSA$_r$ Size")
            #     plt.ylabel("RNA Length")
            #     plt.xlabel("MSA$_r$ Size")
            #     plt.xticks(rotation=45, ha='right')
            #     plt.grid(True, linestyle='--', alpha=0.7)
            #     # Place legend outside if more than 4 databases
            #     # Get existing legend handles and update labels
            #     handles, original_labels = ax.get_legend_handles_labels()
            #     formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
            #     if self.num_databases > 4:
            #         legend = ax.legend(handles=handles, labels=formatted_labels, title="", bbox_to_anchor=(1.05, 1), loc='upper left')
            #     else:
            #         legend = ax.legend(handles=handles, labels=formatted_labels, title="")
            #     if legend:
            #         legend.set_title(None)
            #     with warnings.catch_warnings():
            #         warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
            #         plt.tight_layout()
            #     self.plot_creator.save_plot("RNA_Length_vs_MSA_lt200")
            #     plt.close()

        # Combined violin plots with simplified classes:
        combined_df['length_class_simple'] = pd.cut(
            combined_df['RNALength'],
            bins=[0, 20, 200, float('inf')],
            labels=['0-20', '20-200', '200+']
        )
        # Simplified MSA classes
        combined_df['msa_class_simple'] = pd.cut(
            combined_df['MSAsize'],
            bins=[-float('inf'), 1, 1000, float('inf')],
            labels=['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$']
        )

        # Combined label
        combined_df['combined_class'] = (
                combined_df['msa_class_simple'].astype(str) + " | " + combined_df['length_class_simple'].astype(str)
        )

        # Order for plotting
        combined_order = [
            f"{msa} | {length}" for msa in ['0 MSA$_r$', '1-1000 MSA$_r$', '+1000 MSA$_r$'] for length in ['0-20', '20-200', '200+']
        ]
        combined_order = [g for g in combined_order if g in combined_df['combined_class'].unique()]

        # Plot box plot
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=combined_df,
            x='combined_class',
            y=metric,
            hue='database',
            palette=self.colors_list,
            hue_order=self.db_labels,
            order=combined_order
        )
        if metric not in {'RNA_RMSD', 'Complex_RMSD', 'Protein_RMSD'}:
            plt.ylim(0, 1)
        plt.xlabel("MSA$_r$ | RNA Length")
        plt.ylabel(self._get_display_label(metric))
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Remove legend title and place outside if more than 4 databases
        # Make legend smaller for combined box plots
        # Get existing legend handles and update labels
        handles, original_labels = ax.get_legend_handles_labels()
        formatted_labels = [format_db_label_for_legend(label) for label in self.db_labels]
        if self.num_databases > 4:
            legend = ax.legend(handles=handles, labels=formatted_labels, title="", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 9})
        else:
            legend = ax.legend(handles=handles, labels=formatted_labels, title="", prop={'size': 9})
        if legend:
            legend.set_title(None)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
            plt.tight_layout()
        self.plot_creator.save_plot(f"{metric}_combined_box")
        plt.close()

    def run_all_plots(self):
        """Generate all comparison plots."""
        all_metrics = [metric for group in self.metric_groups.values() for metric in group]
            
        for metric in all_metrics:
            self._create_comparison_box_plot(metric, 'length_class')
            self._create_comparison_box_plot(metric, 'msa_class')
        
        # Write final message to statistics file
        with open(self.stats_file_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Statistics file complete.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nStatistics written to: {self.stats_file_path}")
    

def extract_db_name(db_path: str) -> str:
    """Extract database name from file path (e.g., 'a2021.db' -> 'a2021')."""
    filename = os.path.basename(db_path)
    # Remove .db extension if present
    if filename.endswith('.db'):
        return filename[:-3]
    return filename


def format_db_label_for_legend(label: str) -> str:
    """
    Format database label for legend display.
    Replaces rMSA, pMSA, pTemp, cTemp with LaTeX subscript notation and removes underscores.
    
    Examples:
        '-rMSA_+pMSA_+pTemp_-cTemp' -> '-MSA$_r$+MSA$_p$+Temp$_p$-Temp$_c$'
        'base' -> 'base'
    """
    # Replace patterns: rMSA -> MSA$_r$, pMSA -> MSA$_p$, pTemp -> Temp$_p$, cTemp -> Temp$_c$
    formatted = label.replace('rMSA', 'MSA$_r$')
    formatted = formatted.replace('pMSA', 'MSA$_p$')
    formatted = formatted.replace('pTemp', 'Temp$_p$')
    formatted = formatted.replace('cTemp', 'Temp$_c$')
    
    # Remove underscores between labels
    # Pattern: _+ or _- or +_ or -_ -> just + or -
    formatted = formatted.replace('_+', '+')
    formatted = formatted.replace('_-', '-')
    formatted = formatted.replace('+_', '+')
    formatted = formatted.replace('-_', '-')
    
    return formatted

def list_db_files(db_folder: str) -> list:
    """List all .db files in the specified folder."""
    if not os.path.isdir(db_folder):
        return []
    db_files = [f for f in os.listdir(db_folder) if f.endswith('.db')]
    return sorted(db_files)

def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison box plots between 2-16 databases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic usage - automatically finds and uses .db files in folder (auto-assigns colors)
  python plot_metrics_comparison_3db.py --db_folder /path/to/databases
  
  # Manually choose colors for specific databases
  python plot_metrics_comparison_3db.py --db_folder /path/to/databases \\
    --color1 green --color2 orange --color3 blue --color4 magenta
  
  # Available colors: {', '.join(COLOR_SCHEME.keys())}
  
  # Single chain only
  python plot_metrics_comparison_3db.py --db_folder /path/to/databases --single_chain

Note: The folder must contain between 2 and 16 .db files. Database names and labels are extracted automatically from filenames.
If you don't specify colors, they will be automatically assigned from the default color scheme.
        """
    )
    
    parser.add_argument('--db_folder', type=str, required=True,
                       help='Path to folder containing 2-16 database files (.db)')
    # Add color arguments dynamically using a loop
    for i in range(1, 17):  # 1 to 16
        parser.add_argument(f'--color{i}', type=str, default=None,
                           choices=list(COLOR_SCHEME.keys()),
                           help=f'Color for database {i} from scheme: {", ".join(COLOR_SCHEME.keys())}')
    parser.add_argument('--single_chain', action='store_true',
                       help='Only process single chain RNAs')
    
    # Parse known args first, then check for MSA/RBP options in remaining args
    args, unknown_args = parser.parse_known_args()
    
    # Parse MSA and RBP options from command line (can be --msa +msa or just +msa/-msa)
    msa_option = None
    rbp_template_option = None
    
    # Check in unknown_args and sys.argv for MSA/RBP patterns
    import sys
    all_args = sys.argv[1:]
    
    for arg in all_args:
        arg_lower = arg.lower()
        if arg_lower in ['+msa', '-msa', 'msa']:
            msa_option = '+MSA' if arg_lower in ['+msa', 'msa'] else '-MSA'
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case
    
    # Validate folder exists
    if not os.path.isdir(args.db_folder):
        raise NotADirectoryError(f"Database folder not found: {args.db_folder}")
    
    # Automatically find all .db files in the folder
    db_files = list_db_files(args.db_folder)
    
    # Validate number of databases
    if len(db_files) == 0:
        raise ValueError(f"No .db files found in folder: {args.db_folder}")
    elif len(db_files) == 1:
        raise ValueError(f"Only 1 .db file found in folder '{args.db_folder}': {db_files[0]}\n"
                        f"Need at least 2 .db files for comparison.")
    elif len(db_files) > 16:
        raise ValueError(f"Found {len(db_files)} .db files. Maximum 16 files supported.\n"
                        f"Please ensure the folder contains 2-16 .db files.")
    
    # Use the found databases (sorted alphabetically)
    num_dbs = len(db_files)
    
    # Construct full paths and extract labels
    db_paths = []
    db_labels = []
    db_colors = []
    
    # Collect color arguments dynamically
    color_args = [getattr(args, f'color{i}', None) for i in range(1, 17)]
    
    # Use provided colors or default to automatic assignment
    for i, db_file in enumerate(db_files):
        db_path = os.path.join(args.db_folder, db_file)
        db_label = extract_db_name(db_file)
        
        # Use provided color if available, otherwise use default color order
        if i < len(color_args) and color_args[i] is not None:
            db_color = COLOR_SCHEME[color_args[i]]
            color_name = color_args[i]
        else:
            # Auto-assign from default color order
            color_name = DEFAULT_COLOR_ORDER[i % len(DEFAULT_COLOR_ORDER)]
            db_color = COLOR_SCHEME[color_name]
        
        db_paths.append(db_path)
        db_labels.append(db_label)
        db_colors.append(db_color)
    
    print(f"\nFound {num_dbs} database file(s) in '{args.db_folder}':")
    for i, (db_file, db_label) in enumerate(zip(db_files, db_labels), 1):
        color_name = color_args[i-1] if (i-1 < len(color_args) and color_args[i-1] is not None) else DEFAULT_COLOR_ORDER[(i-1) % len(DEFAULT_COLOR_ORDER)]
        print(f"  Database {i}: {db_file} (label: {db_label}, color: {color_name} = {db_colors[i-1]})")
    filter_info = []
    if args.single_chain:
        filter_info.append("single chain RNAs only")
    else:
        filter_info.append("all chains")
    if msa_option:
        filter_info.append(msa_option)
    if rbp_template_option:
        filter_info.append(rbp_template_option)
    print(f"\nGenerating plots for {', '.join(filter_info)}...")
    
    plotter = MetricsComparisonPlotter3DB(
        db_paths=db_paths,
        db_labels=db_labels,
        db_colors=db_colors,
        single_chain_only=args.single_chain,
        msa_option=msa_option,
        rbp_template_option=rbp_template_option
    )
    plotter.run_all_plots()
    
    print("\nPlot generation complete!")

if __name__ == "__main__":
    main()

