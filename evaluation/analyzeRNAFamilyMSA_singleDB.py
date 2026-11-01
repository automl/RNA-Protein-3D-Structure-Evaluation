"""
Simplified script for single-database mode analysis.
Analyzes RNAFamily and RNA_msa_size correlation with metric differences from average.
"""

import sqlite3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator

# Available metrics for analysis
AVAILABLE_METRICS = ['RNA_LDDT_allAtom', 'RNAComplex_TM', 'RNA_TM', 'Complex_LDDT_allAtom', 'Complex_TM', 'RNA_LDDT',
                     'Complex_LDDT']

# Display name mapping for metrics
METRIC_DISPLAY_NAMES = {
    'RNA_LDDT_allAtom': 'rLDDT',
    'RNA_TM': 'rTM',
    'RNAComplex_TM': 'rcTM',
    'Complex_LDDT_allAtom': 'cLDDT',
    'Complex_TM': 'cTM',
    'RNA_LDDT': r"rLDDT$_{C3'}$",
    'Complex_LDDT': r'cLDDT$_{BB}$'
}


def process_rna_family(rna_family_value):
    """Process RNAFamily value: handle None, strings, and lists."""
    if rna_family_value is None:
        return "None"
    
    try:
        # If it's a string, try to evaluate it
        if isinstance(rna_family_value, str):
            try:
                values = eval(rna_family_value)
                if isinstance(values, list):
                    # If all values in list are "None" or None, return "None"
                    if all(str(v).lower() == "none" or v is None for v in values):
                        return "None"
                    # Otherwise, take the first non-None value
                    for v in values:
                        if v is not None and str(v).lower() != "none":
                            return str(v)
                    return "None"
                return str(values)
            except:
                # If eval fails, use as string
                if rna_family_value.lower() == "none":
                    return "None"
                return str(rna_family_value)
        elif isinstance(rna_family_value, list):
            # If all values in list are "None" or None, return "None"
            if all(str(v).lower() == "none" or v is None for v in rna_family_value):
                return "None"
            # Otherwise, take the first non-None value
            for v in rna_family_value:
                if v is not None and str(v).lower() != "none":
                    return str(v)
            return "None"
        else:
            return str(rna_family_value)
    except:
        return "None"


def process_rna_msa_size(rna_msa_size_value):
    """Process RNA_msa_size value: extract MSA size value."""
    if rna_msa_size_value is None:
        return None
    try:
        if isinstance(rna_msa_size_value, str):
            try:
                values = eval(rna_msa_size_value)
                if isinstance(values, dict):
                    if len(values) > 0:
                        return list(values.values())[0]
                    return None
                return float(values) if isinstance(values, (int, float)) else None
            except:
                return None
        elif isinstance(rna_msa_size_value, dict):
            if len(rna_msa_size_value) > 0:
                return list(rna_msa_size_value.values())[0]
            return None
        elif isinstance(rna_msa_size_value, (int, float)):
            return float(rna_msa_size_value)
        else:
            return None
    except:
        return None


def process_metric_value(value_str):
    """Process metric string to get the first numeric value."""
    if value_str is None:
        return None
    try:
        if isinstance(value_str, (int, float)):
            return float(value_str)
        if isinstance(value_str, str):
            try:
                values = eval(value_str)
                if isinstance(values, list):
                    return float(values[0]) if len(values) > 0 else None
                return float(values)
            except:
                return float(value_str)
        return float(value_str)
    except (ValueError, TypeError):
        return None


def get_data_from_database(db_path, pred_table, exp_table, metric):
    """
    Get data from database and calculate differences from the average.
    
    Returns:
        pd.DataFrame with columns: exp_db_id, RNAFamily, RNA_msa_size, metric_value, metric_diff
        where metric_diff = metric_value - average(metric_value)
    """
    conn = sqlite3.connect(db_path)
    
    try:
        query = f"""
        SELECT e.PDBId, e.RNAFamily, p.RNA_msa_size, p.{metric}
        FROM {exp_table} e
        INNER JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Process columns
        df['RNAFamily'] = df['RNAFamily'].apply(process_rna_family)
        df['RNA_msa_size'] = df['RNA_msa_size'].apply(process_rna_msa_size)
        df[f'{metric}_value'] = df[metric].apply(process_metric_value)
        
        # Rename PDBId
        df = df.rename(columns={'PDBId': 'exp_db_id'})
        
        # Calculate average of the metric
        metric_avg = df[f'{metric}_value'].mean()
        
        # Calculate difference from average
        df[f'{metric}_diff'] = df[f'{metric}_value'] - metric_avg
        
        # Reset index to ensure unique indices
        df = df.reset_index(drop=True)
        
        return df, metric_avg
        
    finally:
        conn.close()


def create_plots(df, plot_creator, db_name, metric, metric_display_name, metric_avg):
    """Create plots using PlotCreator for single-database mode."""
    plot_creator.set_plot_style()
    
    metric_value_col = f'{metric}_value'
    metric_diff_col = f'{metric}_diff'
    
    # Filter valid data
    valid_metric = df[df[metric_value_col].notna()].copy()
    
    if len(valid_metric) == 0:
        print(f"\nWarning: No valid {metric_display_name} data to plot!")
        return
    
    # Reset index to ensure no duplicate labels
    valid_metric = valid_metric.reset_index(drop=True)
    
    ylabel = f'{metric_display_name} Difference from Average\n(+ = above average, - = below average)'
    
    # Plot 1: RNAFamily vs metric_diff
    plt.figure(figsize=(12, 8))
    
    # Group by RNAFamily and calculate average difference
    family_stats = valid_metric.groupby('RNAFamily', observed=False)[metric_diff_col].agg(['mean', 'count'])
    family_stats = family_stats.reset_index()
    
    rna_families = family_stats['RNAFamily'].values
    family_diffs = family_stats['mean'].values
    family_counts = family_stats['count'].values
    
    plt.bar(range(len(rna_families)), family_diffs)
    plt.xlabel('RNA Family')
    plt.ylabel(ylabel)
    plt.xticks(range(len(rna_families)), rna_families, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    for i, (diff, count) in enumerate(zip(family_diffs, family_counts)):
        plt.text(i, diff, f'n={count}', ha='center', va='bottom' if diff >= 0 else 'top')
    plt.tight_layout()
    plot_creator.save_plot(f'RNAFamily_vs_{metric}_diff')
    plt.close()
    
    # Plot 2: RNA_msa_size vs metric_diff (scatter)
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_metric['RNA_msa_size'], valid_metric[metric_diff_col], alpha=0.5)
    plt.xlabel('RNA MSA Size')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_creator.save_plot(f'RNA_msa_size_vs_{metric}_diff')
    plt.close()
    
    # Create MSA_size_bin column
    valid_metric['MSA_size_bin'] = pd.cut(
        valid_metric['RNA_msa_size'], 
        bins=[0, 1, 20, 100, 1000, float('inf')],
        labels=['0-1', '1-20', '20-100', '100-1000', '+1000']
    )
    
    # Plot 3: RNA_msa_size binned vs metric_diff
    bin_stats = valid_metric.groupby('MSA_size_bin', observed=False)[metric_diff_col].agg(['mean', 'count'])
    bin_stats = bin_stats.reset_index()
    
    bin_labels = bin_stats['MSA_size_bin'].astype(str).values
    bin_diffs = bin_stats['mean'].values
    bin_counts = bin_stats['count'].values
    
    # Get family counts per bin
    bin_family_counts = []
    for bin_label in bin_stats['MSA_size_bin']:
        bin_data = valid_metric[valid_metric['MSA_size_bin'] == bin_label]
        bin_family_counts.append(bin_data['RNAFamily'].nunique())
    
    if len(bin_diffs) > 0:
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(bin_labels)), bin_diffs)
        plt.xlabel('RNA MSA Size')
        plt.ylabel(ylabel)
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        for i, (diff, count, family_count) in enumerate(zip(bin_diffs, bin_counts, bin_family_counts)):
            label = f'n={count}\nfamilies={family_count}'
            plt.text(i, diff, label, ha='center', va='bottom' if diff >= 0 else 'top', fontsize=9)
        plt.tight_layout()
        plot_creator.save_plot(f'RNA_msa_size_binned_vs_{metric}_diff')
        plt.close()
    
    # Plot 4: RNAFamily vs RNA_msa_size correlation (heatmap)
    plt.figure(figsize=(14, 10))
    
    # Calculate percentage change from overall average
    overall_avg = valid_metric[metric_value_col].mean()
    
    # Use pivot_table with proper handling
    try:
        pivot_data_values = valid_metric.pivot_table(
            values=metric_value_col,
            index='RNAFamily',
            columns='MSA_size_bin',
            aggfunc='mean',
            observed=False
        )
        
        pivot_counts = valid_metric.pivot_table(
            values=metric_value_col,
            index='RNAFamily',
            columns='MSA_size_bin',
            aggfunc='count',
            observed=False
        )
        
        # Calculate percentage change from overall average
        pivot_data_pct = ((pivot_data_values - overall_avg) / overall_avg * 100.0)
        
        # Create annotation matrix
        annot_data = pivot_data_pct.astype(object).copy()
        for i in range(len(pivot_data_pct.index)):
            for j in range(len(pivot_data_pct.columns)):
                count = pivot_counts.iloc[i, j]
                pct_value = pivot_data_pct.iloc[i, j]
                if pd.notna(pct_value) and pd.notna(count):
                    sign = '+' if pct_value >= 0 else ''
                    annot_data.iloc[i, j] = f"{int(count)}|{sign}{pct_value:.2f}%"
                else:
                    annot_data.iloc[i, j] = ""
        
        sns.heatmap(pivot_data_pct, annot=annot_data, fmt='', cmap='coolwarm', center=0, 
                    cbar_kws={'label': f'{metric_display_name} % Change from Overall Average'})
        plt.xlabel('RNA MSA Size')
        plt.ylabel('RNA Family')
        plt.tight_layout()
        plot_creator.save_plot(f'RNAFamily_MSA_size_heatmap_{metric}')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
        plt.close()
    
    print(f"\nPlots saved to: {plot_creator.results_dir}")


def calculate_statistics_with_pvalues(df, metric_diff_col, group_col):
    """
    Calculate statistics and p-values for metric differences grouped by a categorical variable.
    """
    # Filter out NaN values
    df_clean = df[[metric_diff_col, group_col]].copy()
    df_clean = df_clean.dropna()
    
    if df_clean.empty:
        return None
    
    # Convert to numeric
    df_clean[metric_diff_col] = pd.to_numeric(df_clean[metric_diff_col], errors='coerce')
    df_clean = df_clean[df_clean[metric_diff_col].notna()]
    
    if df_clean.empty:
        return None
    
    # Get unique groups
    groups = sorted(df_clean[group_col].unique())
    
    # Calculate statistics for each group
    group_stats = {}
    for group in groups:
        values = df_clean[df_clean[group_col] == group][metric_diff_col].values
        if len(values) > 0:
            group_stats[group] = {
                'mean': np.mean(values),
                'count': len(values),
                'values': values
            }
    
    # Calculate p-values between groups (pairwise comparisons)
    p_values = {}
    groups_list = list(groups)
    
    for i, group1 in enumerate(groups_list):
        for group2 in groups_list[i+1:]:
            if group1 in group_stats and group2 in group_stats:
                values1 = group_stats[group1]['values']
                values2 = group_stats[group2]['values']
                
                if len(values1) > 0 and len(values2) > 0:
                    try:
                        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        comparison_key = f"{group1} vs {group2}"
                        p_values[comparison_key] = p_value
                    except Exception as e:
                        p_values[f"{group1} vs {group2}"] = None
    
    return {
        'groups': groups,
        'group_stats': group_stats,
        'p_values': p_values
    }


def write_all_statistics(all_metrics_data, plot_creator, db_name):
    """
    Write statistics with p-values for all metrics to a single text file.
    """
    output_path = os.path.join(plot_creator.results_dir, 'all_metrics_statistics.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Statistical Analysis: All Metrics\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Mode: Single-database (differences from average)\n")
        f.write(f"Database: {db_name}\n")
        f.write(f"Total metrics analyzed: {len(all_metrics_data)}\n\n")
        
        for metric, data in all_metrics_data.items():
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            df = data['df']
            metric_avg = data['metric_avg']
            metric_diff_col = f'{metric}_diff'
            
            f.write("=" * 80 + "\n")
            f.write(f"Metric: {metric} ({metric_display_name})\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Average {metric_display_name}: {metric_avg:.4f}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Valid samples: {df[metric_diff_col].notna().sum()}\n\n")
            
            # Statistics by RNAFamily
            if 'RNAFamily' in df.columns:
                family_stats = calculate_statistics_with_pvalues(df, metric_diff_col, 'RNAFamily')
                if family_stats:
                    f.write("--- Statistics by RNA Family ---\n")
                    f.write("-" * 80 + "\n")
                    f.write("Average Differences by Family:\n")
                    for family in family_stats['groups']:
                        if family in family_stats['group_stats']:
                            stats_data = family_stats['group_stats'][family]
                            f.write(f"  {family:30s}: Average difference = {stats_data['mean']:8.4f}, "
                                   f"n = {stats_data['count']}\n")
                    f.write("\n")
                    
                    f.write("P-values (Mann-Whitney U test) between Families:\n")
                    f.write("-" * 80 + "\n")
                    if family_stats['p_values']:
                        for comparison, p_val in sorted(family_stats['p_values'].items()):
                            if p_val is not None:
                                significance = ""
                                if p_val < 0.001:
                                    significance = "***"
                                elif p_val < 0.01:
                                    significance = "**"
                                elif p_val < 0.05:
                                    significance = "*"
                                f.write(f"  {comparison:50s}: p = {p_val:.6f} {significance}\n")
                            else:
                                f.write(f"  {comparison:50s}: p = N/A\n")
                    f.write("\n")
                    f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n\n")
            
            # Statistics by MSA size bins
            # Check if MSA_size_bin exists in df (it might be created during plotting)
            msa_bin_col = None
            if 'MSA_size_bin' in df.columns:
                msa_bin_col = 'MSA_size_bin'
            elif 'RNA_msa_size' in df.columns:
                # Create MSA_size_bin if it doesn't exist
                df['MSA_size_bin'] = pd.cut(
                    df['RNA_msa_size'], 
                    bins=[0, 1, 20, 100, 1000, float('inf')],
                    labels=['0-1', '1-20', '20-100', '100-1000', '+1000']
                )
                msa_bin_col = 'MSA_size_bin'
            
            if msa_bin_col:
                msa_stats = calculate_statistics_with_pvalues(df, metric_diff_col, msa_bin_col)
                if msa_stats:
                    f.write("--- Statistics by MSA Size Bin ---\n")
                    f.write("-" * 80 + "\n")
                    f.write("Average Differences by MSA Size Bin:\n")
                    for bin_label in msa_stats['groups']:
                        if bin_label in msa_stats['group_stats']:
                            stats_data = msa_stats['group_stats'][bin_label]
                            f.write(f"  {bin_label:30s}: Average difference = {stats_data['mean']:8.4f}, "
                                   f"n = {stats_data['count']}\n")
                    f.write("\n")
                    
                    f.write("P-values (Mann-Whitney U test) between MSA Size Bins:\n")
                    f.write("-" * 80 + "\n")
                    if msa_stats['p_values']:
                        for comparison, p_val in sorted(msa_stats['p_values'].items()):
                            if p_val is not None:
                                significance = ""
                                if p_val < 0.001:
                                    significance = "***"
                                elif p_val < 0.01:
                                    significance = "**"
                                elif p_val < 0.05:
                                    significance = "*"
                                f.write(f"  {comparison:50s}: p = {p_val:.6f} {significance}\n")
                            else:
                                f.write(f"  {comparison:50s}: p = N/A\n")
                    f.write("\n")
                    f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n\n")
            
            # Overall statistics
            f.write("--- Overall Statistics ---\n")
            f.write("-" * 80 + "\n")
            overall_values = pd.to_numeric(df[metric_diff_col], errors='coerce').dropna()
            if len(overall_values) > 0:
                f.write(f"  Mean difference: {np.mean(overall_values):.4f}\n")
                f.write(f"  Median difference: {np.median(overall_values):.4f}\n")
                f.write(f"  Std deviation: {np.std(overall_values):.4f}\n")
                f.write(f"  Min: {np.min(overall_values):.4f}\n")
                f.write(f"  Max: {np.max(overall_values):.4f}\n")
                f.write(f"  Total samples: {len(overall_values)}\n")
            
            f.write("\n\n")
    
    print(f"\nStatistics written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RNAFamily and RNA_msa_size correlation with metric differences from average (single-database mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script analyzes how RNAFamily and RNA_msa_size correlate with metric differences
from the average in a single database.

Available metrics: {', '.join(AVAILABLE_METRICS)}

Differences are calculated as: value - average(value)
- Positive difference = above average
- Negative difference = below average

Examples:
  # Analyze all metrics
  python analyzeRNAFamilyMSA_singleDB.py
  
  # Analyze specific metric
  python analyzeRNAFamilyMSA_singleDB.py --metric RNA_TM
        """
    )
    parser.add_argument('--metric', type=str, default=None, choices=AVAILABLE_METRICS,
                       help=f'Metric to analyze (default: all metrics). Available: {", ".join(AVAILABLE_METRICS)}')
    parser.add_argument('--plot_type', type=str, default='rna_family_msa_analysis',
                       help='Plot type for PlotCreator (default: rna_family_msa_analysis)')
    
    args = parser.parse_args()
    
    # Get database path and table names from StartConfig
    config = StartConfig()
    db_path = config.get_database_path()
    pred_table = config.pred_table
    exp_table = config.ref_table
    
    if not os.path.exists(db_path):
        raise ValueError(f"Database file not found: {db_path}")
    
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    
    print(f"Analyzing single database:")
    print(f"  Database: {os.path.basename(db_path)}")
    print(f"  Tables: {pred_table} and {exp_table}")
    print(f"  Mode: Single-database (differences from average)")
    
    # Determine which metrics to analyze
    metrics_to_analyze = [args.metric] if args.metric else AVAILABLE_METRICS
    
    # Initialize plot_creator once
    plot_creator = PlotCreator(plot_type=args.plot_type, msa_option=None, single_chain_only=False)
    
    # Store data for all metrics for statistics
    all_metrics_data = {}
    
    for metric in metrics_to_analyze:
        metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        print(f"\n{'='*80}")
        print(f"Analyzing metric: {metric} ({metric_display_name})")
        print(f"{'='*80}")
        
        # Get data
        print("\nLoading data...")
        try:
            df, metric_avg = get_data_from_database(db_path, pred_table, exp_table, metric)
            print(f"Loaded {len(df)} samples")
            print(f"Average {metric_display_name}: {metric_avg:.4f}")
            
            # Store data for statistics
            all_metrics_data[metric] = {
                'df': df,
                'metric_avg': metric_avg
            }
            
            # Create plots
            print("\nCreating plots...")
            create_plots(df, plot_creator, db_name, metric, metric_display_name, metric_avg)
        except Exception as e:
            print(f"Error analyzing {metric}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write statistics for all metrics to a single file
    if all_metrics_data:
        print("\nWriting statistics for all metrics...")
        write_all_statistics(all_metrics_data, plot_creator, db_name)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
