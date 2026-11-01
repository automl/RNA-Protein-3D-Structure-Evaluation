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
    'RNA_LDDT': 'rLDDT_',
    'Complex_LDDT': 'cLDDT_'
}


def process_rna_family(rna_family_value):
    """Process RNAFamily value: if list, take first index; if string, use as is."""
    if rna_family_value is None:
        return "None"
    try:
        if isinstance(rna_family_value, str):
            # Try to evaluate if it's a list representation
            try:
                values = eval(rna_family_value)
                if isinstance(values, list):
                    return str(values[0]) if len(values) > 0 else "None"
                return str(values)
            except:
                # If eval fails, use as string
                return str(rna_family_value)
        elif isinstance(rna_family_value, list):
            return str(rna_family_value[0]) if len(rna_family_value) > 0 else "None"
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
            # Try to evaluate if it's a dict representation
            try:
                values = eval(rna_msa_size_value)
                if isinstance(values, dict):
                    # If multiple values, take the first one (or sum, or max)
                    # Taking the first value for now
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


def get_data_from_databases(db1_path, db2_path, pred_table, exp_table, metric):
    """
    Get data from both databases: RNAFamily, RNA_msa_size, and specified metric.
    
    Returns:
        pd.DataFrame with columns: exp_db_id, RNAFamily, RNA_msa_size, metric_db1, metric_db2
    """
    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)
    
    try:
        # Get data from exp_table (RNAFamily) and pred_table (RNA_msa_size, metric)
        query1 = f"""
        SELECT e.PDBId, e.RNAFamily, p.RNA_msa_size, p.{metric}
        FROM {exp_table} e
        INNER JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        """
        
        query2 = f"""
        SELECT e.PDBId, e.RNAFamily, p.RNA_msa_size, p.{metric}
        FROM {exp_table} e
        INNER JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        """
        
        df1 = pd.read_sql_query(query1, conn1)
        df2 = pd.read_sql_query(query2, conn2)
        
        # Process RNAFamily and RNA_msa_size
        df1['RNAFamily_processed'] = df1['RNAFamily'].apply(process_rna_family)
        df1['RNA_msa_size_processed'] = df1['RNA_msa_size'].apply(process_rna_msa_size)
        df1[f'{metric}_db1'] = df1[metric].apply(process_metric_value)
        
        df2['RNAFamily_processed'] = df2['RNAFamily'].apply(process_rna_family)
        df2['RNA_msa_size_processed'] = df2['RNA_msa_size'].apply(process_rna_msa_size)
        df2[f'{metric}_db2'] = df2[metric].apply(process_metric_value)
        
        # Merge on PDBId (exp_db_id)
        merged = pd.merge(
            df1[['PDBId', 'RNAFamily_processed', 'RNA_msa_size_processed', f'{metric}_db1']],
            df2[['PDBId', 'RNA_msa_size_processed', f'{metric}_db2']],
            on='PDBId',
            suffixes=('', '_db2')
        )
        
        # Rename columns for clarity
        merged = merged.rename(columns={
            'PDBId': 'exp_db_id',
            'RNAFamily_processed': 'RNAFamily',
            'RNA_msa_size_processed': 'RNA_msa_size_db1',
            'RNA_msa_size_processed_db2': 'RNA_msa_size_db2'
        })
        
        # Calculate difference
        merged[f'{metric}_diff'] = merged[f'{metric}_db1'] - merged[f'{metric}_db2']
        merged[f'{metric}_diff_percent'] = (
            (merged[f'{metric}_db1'] - merged[f'{metric}_db2']) / 
            ((merged[f'{metric}_db1'] + merged[f'{metric}_db2']) / 2.0) * 100.0
        )
        merged[f'{metric}_diff_percent'] = merged[f'{metric}_diff_percent'].replace([np.inf, -np.inf], np.nan)
        
        return merged
        
    finally:
        conn1.close()
        conn2.close()


def get_data_from_single_database(db_path, pred_table, exp_table, metric):
    """
    Get data from a single database and calculate differences from the average.
    
    Returns:
        pd.DataFrame with columns: exp_db_id, RNAFamily, RNA_msa_size, metric_value, metric_diff
        where metric_diff = metric_value - average(metric_value)
    """
    conn = sqlite3.connect(db_path)
    
    try:
        # Get data from exp_table (RNAFamily) and pred_table (RNA_msa_size, metric)
        query = f"""
        SELECT e.PDBId, e.RNAFamily, p.RNA_msa_size, p.{metric}
        FROM {exp_table} e
        INNER JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Process RNAFamily and RNA_msa_size
        df['RNAFamily_processed'] = df['RNAFamily'].apply(process_rna_family)
        df['RNA_msa_size_processed'] = df['RNA_msa_size'].apply(process_rna_msa_size)
        df[f'{metric}_value'] = df[metric].apply(process_metric_value)
        
        # Rename columns for clarity
        df = df.rename(columns={
            'PDBId': 'exp_db_id',
            'RNAFamily_processed': 'RNAFamily',
            'RNA_msa_size_processed': 'RNA_msa_size'
        })
        
        # Calculate average of the metric
        metric_avg = df[f'{metric}_value'].mean()
        
        # Calculate difference from average
        df[f'{metric}_diff'] = df[f'{metric}_value'] - metric_avg
        df[f'{metric}_diff_percent'] = (
            (df[f'{metric}_value'] - metric_avg) / metric_avg * 100.0
        )
        df[f'{metric}_diff_percent'] = df[f'{metric}_diff_percent'].replace([np.inf, -np.inf], np.nan)
        
        return df, metric_avg
        
    finally:
        conn.close()


def print_statistics(df, db1_name, db2_name, metric, metric_display_name, base_db=None, single_db_mode=False, metric_avg=None):
    """Print statistics to terminal."""
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    if single_db_mode:
        print(f"\nNote: Analyzing single database, differences calculated from average")
        print(f"  Average {metric_display_name}: {metric_avg:.4f}")
        print(f"  Positive difference = value > average (above average)")
        print(f"  Negative difference = value < average (below average)")
    elif base_db:
        other_db = db2_name if base_db == db1_name else db1_name
        print(f"\nNote: {base_db} is the base/reference database")
        print(f"  Positive difference = {other_db} > {base_db} ({other_db} is higher)")
        print(f"  Negative difference = {other_db} < {base_db} ({other_db} is lower)")
    
    if single_db_mode:
        metric_value_col = f'{metric}_value'
        metric_diff_col = f'{metric}_diff'
        metric_diff_percent_col = f'{metric}_diff_percent'
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Samples with valid {metric_display_name}: {df[metric_value_col].notna().sum()}")
        
        # RNAFamily distribution
        print("\n--- RNAFamily Distribution ---")
        rna_family_counts = df['RNAFamily'].value_counts()
        for family, count in rna_family_counts.items():
            print(f"  {family}: {count}")
        
        # RNA_msa_size statistics
        print("\n--- RNA MSA Size Statistics ---")
        print(f"  Mean: {df['RNA_msa_size'].mean():.2f}, Median: {df['RNA_msa_size'].median():.2f}")
        
        # Metric statistics
        print(f"\n--- {metric_display_name} Statistics ---")
        valid_metric = df[df[metric_value_col].notna()].copy()
        if len(valid_metric) > 0:
            print(f"  Overall Average: {metric_avg:.4f}")
            print(f"  Mean: {valid_metric[metric_value_col].mean():.4f}")
            print(f"    Median: {valid_metric[metric_value_col].median():.4f}")
            print(f"    Std: {valid_metric[metric_value_col].std():.4f}")
            print(f"  Average Difference from Mean:")
            print(f"    Mean diff: {valid_metric[metric_diff_col].mean():.4f}")
            print(f"    Median diff: {valid_metric[metric_diff_col].median():.4f}")
            print(f"    Mean diff %: {valid_metric[metric_diff_percent_col].mean():.2f}%")
            print(f"    Median diff %: {valid_metric[metric_diff_percent_col].median():.2f}%")
        
        # RNAFamily vs metric_diff
        print(f"\n--- Average {metric_display_name} Difference from Mean by RNAFamily ---")
        for family in df['RNAFamily'].unique():
            family_data = valid_metric[valid_metric['RNAFamily'] == family]
            if len(family_data) > 0:
                print(f"  {family} (n={len(family_data)}):")
                print(f"    Mean diff: {family_data[metric_diff_col].mean():.4f}")
                print(f"    Mean diff %: {family_data[metric_diff_percent_col].mean():.2f}%")
        
        # RNA_msa_size vs metric_diff
        print(f"\n--- Average {metric_display_name} Difference from Mean by RNA MSA Size (binned) ---")
        valid_metric['MSA_size_bin'] = pd.cut(valid_metric['RNA_msa_size'], 
                                             bins=[0, 1, 20, 100, 1000, float('inf')],
                                             labels=['0-1', '1-20', '20-100', '100-1000', '+1000'])
        for bin_label in valid_metric['MSA_size_bin'].cat.categories:
            bin_data = valid_metric[valid_metric['MSA_size_bin'] == bin_label]
            if len(bin_data) > 0:
                unique_families = bin_data['RNAFamily'].nunique()
                print(f"  {bin_label} (n={len(bin_data)}, families={unique_families}):")
                print(f"    Mean diff: {bin_data[metric_diff_col].mean():.4f}")
                print(f"    Mean diff %: {bin_data[metric_diff_percent_col].mean():.2f}%")
    else:
        metric_db1_col = f'{metric}_db1'
        metric_db2_col = f'{metric}_db2'
        metric_diff_col = f'{metric}_diff'
        metric_diff_percent_col = f'{metric}_diff_percent'
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Samples with valid {metric_display_name} in both databases: {df[[metric_db1_col, metric_db2_col]].notna().all(axis=1).sum()}")
        
        # RNAFamily distribution
        print("\n--- RNAFamily Distribution ---")
        rna_family_counts = df['RNAFamily'].value_counts()
        for family, count in rna_family_counts.items():
            print(f"  {family}: {count}")
        
        # RNA_msa_size statistics
        print("\n--- RNA MSA Size Statistics ---")
        print(f"  Database 1 ({db1_name}) - Mean: {df['RNA_msa_size_db1'].mean():.2f}, Median: {df['RNA_msa_size_db1'].median():.2f}")
        print(f"  Database 2 ({db2_name}) - Mean: {df['RNA_msa_size_db2'].mean():.2f}, Median: {df['RNA_msa_size_db2'].median():.2f}")
        
        # Metric statistics
        print(f"\n--- {metric_display_name} Statistics ---")
        valid_metric = df[df[[metric_db1_col, metric_db2_col]].notna().all(axis=1)]
        if len(valid_metric) > 0:
            print(f"  Database 1 ({db1_name}):")
            print(f"    Mean: {valid_metric[metric_db1_col].mean():.4f}")
            print(f"    Median: {valid_metric[metric_db1_col].median():.4f}")
            print(f"    Std: {valid_metric[metric_db1_col].std():.4f}")
            print(f"  Database 2 ({db2_name}):")
            print(f"    Mean: {valid_metric[metric_db2_col].mean():.4f}")
            print(f"    Median: {valid_metric[metric_db2_col].median():.4f}")
            print(f"    Std: {valid_metric[metric_db2_col].std():.4f}")
            print(f"  Average Difference (db1 - db2):")
            print(f"    Mean: {valid_metric[metric_diff_col].mean():.4f}")
            print(f"    Median: {valid_metric[metric_diff_col].median():.4f}")
            print(f"    Mean %: {valid_metric[metric_diff_percent_col].mean():.2f}%")
            print(f"    Median %: {valid_metric[metric_diff_percent_col].median():.2f}%")
        
        # RNAFamily vs metric_diff
        print(f"\n--- Average {metric_display_name} Difference by RNAFamily ---")
        for family in df['RNAFamily'].unique():
            family_data = valid_metric[valid_metric['RNAFamily'] == family]
            if len(family_data) > 0:
                print(f"  {family} (n={len(family_data)}):")
                print(f"    Mean diff: {family_data[metric_diff_col].mean():.4f}")
                print(f"    Mean diff %: {family_data[metric_diff_percent_col].mean():.2f}%")
        
        # RNA_msa_size vs metric_diff
        print(f"\n--- Average {metric_display_name} Difference by RNA MSA Size (binned) ---")
        valid_metric['MSA_size_bin'] = pd.cut(valid_metric['RNA_msa_size_db1'], 
                                             bins=[0, 1, 20, 100, 1000, float('inf')],
                                             labels=['0-1', '1-20', '20-100', '100-1000', '+1000'])
        for bin_label in valid_metric['MSA_size_bin'].cat.categories:
            bin_data = valid_metric[valid_metric['MSA_size_bin'] == bin_label]
            if len(bin_data) > 0:
                unique_families = bin_data['RNAFamily'].nunique()
                print(f"  {bin_label} (n={len(bin_data)}, families={unique_families}):")
                print(f"    Mean diff: {bin_data[metric_diff_col].mean():.4f}")
                print(f"    Mean diff %: {bin_data[metric_diff_percent_col].mean():.2f}%")


def calculate_statistics_with_pvalues(df, metric_diff_col, group_col, group_name):
    """
    Calculate statistics and p-values for metric differences grouped by a categorical variable.
    
    Args:
        df: DataFrame with metric differences
        metric_diff_col: Column name for metric differences
        group_col: Column name for grouping variable
        group_name: Display name for the grouping variable
    
    Returns:
        Dictionary with statistics and p-values
    """
    # Filter out NaN values
    df_clean = df[[metric_diff_col, group_col]].dropna()
    
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
                'median': np.median(values),
                'std': np.std(values),
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
                        # Mann-Whitney U test (non-parametric)
                        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        comparison_key = f"{group1} vs {group2}"
                        p_values[comparison_key] = p_value
                    except Exception as e:
                        p_values[f"{group1} vs {group2}"] = None
    
    return {
        'group_name': group_name,
        'groups': groups,
        'group_stats': group_stats,
        'p_values': p_values
    }


def write_statistics_to_file(df, plot_creator, db1_name, db2_name, metric, metric_display_name, 
                            base_db=None, single_db_mode=False, metric_avg=None):
    """
    Write statistics with p-values to a text file in the same directory as plots.
    """
    if single_db_mode:
        metric_diff_col = f'{metric}_diff'
        ylabel_info = "Difference from Average"
    else:
        metric_diff_col = f'{metric}_diff'
        if base_db:
            other_db = db2_name if base_db == db1_name else db1_name
            ylabel_info = f"Difference ({other_db} - {base_db})"
        else:
            ylabel_info = f"Difference ({db1_name} - {db2_name})"
    
    # Check if metric_diff_col exists and has data
    if metric_diff_col not in df.columns:
        print(f"Warning: {metric_diff_col} column not found. Skipping statistics export.")
        return
    
    # Filter out NaN values
    df_clean = df[[metric_diff_col]].dropna()
    if df_clean.empty:
        print(f"Warning: No valid data for {metric_diff_col}. Skipping statistics export.")
        return
    
    # Create output file path
    output_path = os.path.join(plot_creator.results_dir, f'{metric}_statistics.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Statistical Analysis: {metric} ({metric_display_name})\n")
        f.write("=" * 80 + "\n\n")
        
        if single_db_mode:
            f.write(f"Mode: Single-database (differences from average)\n")
            f.write(f"Database: {db1_name}\n")
            f.write(f"Average {metric_display_name}: {metric_avg:.4f}\n")
        else:
            f.write(f"Mode: Two-database comparison\n")
            f.write(f"Database 1: {db1_name}\n")
            f.write(f"Database 2: {db2_name}\n")
            if base_db:
                other_db = db2_name if base_db == db1_name else db1_name
                f.write(f"Base database: {base_db}\n")
                f.write(f"Comparison: {other_db} vs {base_db}\n")
        
        f.write(f"Metric difference: {ylabel_info}\n")
        f.write(f"Total samples: {len(df_clean)}\n\n")
        
        # Statistics by RNAFamily
        if 'RNAFamily' in df.columns:
            family_stats = calculate_statistics_with_pvalues(df, metric_diff_col, 'RNAFamily', 'RNAFamily')
            if family_stats:
                f.write("=" * 80 + "\n")
                f.write("Statistics by RNA Family\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Average Differences by Family:\n")
                f.write("-" * 80 + "\n")
                for family in family_stats['groups']:
                    if family in family_stats['group_stats']:
                        stats_data = family_stats['group_stats'][family]
                        f.write(f"  {family:30s}: Mean = {stats_data['mean']:8.4f}, "
                               f"Median = {stats_data['median']:8.4f}, "
                               f"Std = {stats_data['std']:8.4f}, "
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
        # Check for both possible column names
        msa_bin_col = None
        if 'MSA_size_bin' in df.columns:
            msa_bin_col = 'MSA_size_bin'
        elif 'RNA_msa_size_bin' in df.columns:
            msa_bin_col = 'RNA_msa_size_bin'
        
        if msa_bin_col:
            msa_stats = calculate_statistics_with_pvalues(df, metric_diff_col, msa_bin_col, 'MSA Size Bin')
            if msa_stats:
                f.write("=" * 80 + "\n")
                f.write("Statistics by MSA Size Bin\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Average Differences by MSA Size Bin:\n")
                f.write("-" * 80 + "\n")
                for bin_label in msa_stats['groups']:
                    if bin_label in msa_stats['group_stats']:
                        stats_data = msa_stats['group_stats'][bin_label]
                        f.write(f"  {bin_label:30s}: Mean = {stats_data['mean']:8.4f}, "
                               f"Median = {stats_data['median']:8.4f}, "
                               f"Std = {stats_data['std']:8.4f}, "
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
        f.write("=" * 80 + "\n")
        f.write("Overall Statistics\n")
        f.write("=" * 80 + "\n\n")
        overall_values = pd.to_numeric(df[metric_diff_col], errors='coerce').dropna()
        if len(overall_values) > 0:
            f.write(f"  Mean difference: {np.mean(overall_values):.4f}\n")
            f.write(f"  Median difference: {np.median(overall_values):.4f}\n")
            f.write(f"  Std deviation: {np.std(overall_values):.4f}\n")
            f.write(f"  Min: {np.min(overall_values):.4f}\n")
            f.write(f"  Max: {np.max(overall_values):.4f}\n")
            f.write(f"  Total samples: {len(overall_values)}\n")
    
    print(f"\nStatistics written to: {output_path}")


def create_plots(df, plot_creator, db1_name, db2_name, metric, metric_display_name, base_db=None, single_db_mode=False, metric_avg=None):
    """Create plots using PlotCreator."""
    plot_creator.set_plot_style()
    
    if single_db_mode:
        metric_value_col = f'{metric}_value'
        metric_diff_col = f'{metric}_diff'
        valid_metric = df[df[metric_value_col].notna()].copy()
        
        if len(valid_metric) == 0:
            print(f"\nWarning: No valid {metric_display_name} data to plot!")
            return
        
        ylabel = f'{metric_display_name} Difference from Average\n(+ = above average, - = below average)'
        plot_diff_col = metric_diff_col
        msa_size_col = 'RNA_msa_size'
    else:
        metric_db1_col = f'{metric}_db1'
        metric_db2_col = f'{metric}_db2'
        metric_diff_col = f'{metric}_diff'
        
        valid_metric = df[df[[metric_db1_col, metric_db2_col]].notna().all(axis=1)].copy()
        
        if len(valid_metric) == 0:
            print(f"\nWarning: No valid {metric_display_name} data to plot!")
            return
        
        # Determine if we need to invert differences based on base_db
        # Difference is calculated as db1 - db2
        # If base_db = db1, we want to show db2 - db1 (invert)
        # If base_db = db2, we use db1 - db2 (keep as is)
        invert_diff = False
        other_db = None
        if base_db:
            if base_db == db1_name:
                # base_db is db1, so we're comparing db2 to db1
                # Need to invert: show db2 - db1 instead of db1 - db2
                invert_diff = True
                other_db = db2_name
                ylabel = f'{metric_display_name} Difference ({other_db} - {base_db})\n(+ = {other_db} > {base_db}, - = {other_db} < {base_db})'
            else:
                # base_db is db2, so we're comparing db1 to db2
                # Use as is: db1 - db2
                invert_diff = False
                other_db = db1_name
                ylabel = f'{metric_display_name} Difference ({other_db} - {base_db})\n(+ = {other_db} > {base_db}, - = {other_db} < {base_db})'
        else:
            ylabel = f'{metric_display_name} Difference ({db1_name} - {db2_name})'
        
        # Create a column with potentially inverted differences for plotting
        if invert_diff:
            plot_diff_col = f'{metric}_diff_plot'
            valid_metric[plot_diff_col] = -valid_metric[metric_diff_col]
        else:
            plot_diff_col = metric_diff_col
        msa_size_col = 'RNA_msa_size_db1'
    
    # Plot 1: RNAFamily vs metric_diff
    plt.figure(figsize=(12, 8))
    rna_families = valid_metric['RNAFamily'].unique()
    family_diffs = [valid_metric[valid_metric['RNAFamily'] == f][plot_diff_col].mean() 
                    for f in rna_families]
    family_counts = [len(valid_metric[valid_metric['RNAFamily'] == f]) for f in rna_families]
    
    plt.bar(range(len(rna_families)), family_diffs)
    plt.xlabel('RNA Family')
    plt.ylabel(ylabel)
    plt.xticks(range(len(rna_families)), rna_families, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    # Add count labels
    for i, (diff, count) in enumerate(zip(family_diffs, family_counts)):
        plt.text(i, diff, f'n={count}', ha='center', va='bottom' if diff >= 0 else 'top')
    plt.tight_layout()
    plot_creator.save_plot(f'RNAFamily_vs_{metric}_diff')
    plt.close()
    
    # Plot 2: RNA_msa_size vs metric_diff (scatter)
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_metric[msa_size_col], valid_metric[plot_diff_col], alpha=0.5)
    plt.xlabel('RNA MSA Size' + (' (Database 1)' if not single_db_mode else ''))
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_creator.save_plot(f'RNA_msa_size_vs_{metric}_diff')
    plt.close()
    
    # Create MSA_size_bin column for both Plot 3 and Plot 4
    if 'MSA_size_bin' not in valid_metric.columns:
        valid_metric['MSA_size_bin'] = pd.cut(valid_metric[msa_size_col], 
                                             bins=[0, 1, 20, 100, 1000, float('inf')],
                                             labels=['0-1', '1-20', '20-100', '100-1000', '+1000'])
    
    # Plot 3: RNA_msa_size binned vs metric_diff
    bin_diffs = []
    bin_labels = []
    bin_counts = []
    bin_family_counts = []  # Number of unique RNA families per bin
    for bin_label in valid_metric['MSA_size_bin'].cat.categories:
        bin_data = valid_metric[valid_metric['MSA_size_bin'] == bin_label]
        if len(bin_data) > 0:
            bin_diffs.append(bin_data[plot_diff_col].mean())
            bin_labels.append(str(bin_label))
            bin_counts.append(len(bin_data))
            # Count unique RNA families in this bin
            unique_families = bin_data['RNAFamily'].nunique()
            bin_family_counts.append(unique_families)
    
    if bin_diffs:
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(bin_labels)), bin_diffs)
        plt.xlabel('RNA MSA Size')
        plt.ylabel(ylabel)
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        # Add count labels with RNA family count
        for i, (diff, count, family_count) in enumerate(zip(bin_diffs, bin_counts, bin_family_counts)):
            label = f'n={count}\nfamilies={family_count}'
            plt.text(i, diff, label, ha='center', va='bottom' if diff >= 0 else 'top', fontsize=9)
        plt.tight_layout()
        plot_creator.save_plot(f'RNA_msa_size_binned_vs_{metric}_diff')
        plt.close()
    
    # Plot 4: RNAFamily vs RNA_msa_size correlation (heatmap)
    plt.figure(figsize=(14, 10))
    
    if single_db_mode:
        # Single-database mode: show percentage change from overall average
        metric_value_col_for_heatmap = metric_value_col
        overall_avg = valid_metric[metric_value_col_for_heatmap].mean()
        
        # Get mean metric values for each family/MSA combination
        pivot_data_values = valid_metric.pivot_table(
            values=metric_value_col_for_heatmap,
            index='RNAFamily',
            columns='MSA_size_bin',
            aggfunc='mean',
            observed=False
        )
        
        # Calculate percentage change from overall average
        pivot_data_pct = ((pivot_data_values - overall_avg) / overall_avg * 100.0)
        
        # Get counts for each cell
        pivot_counts = valid_metric.pivot_table(
            values=metric_value_col_for_heatmap,
            index='RNAFamily',
            columns='MSA_size_bin',
            aggfunc='count',
            observed=False
        )
        
        # Create annotation matrix with format "count|%value"
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
    else:
        # Two-database mode: show percentage change from base database for each family/MSA bin
        # Get base database values (db1 or db2 depending on base_db)
        if base_db:
            if base_db == db1_name:
                base_value_col = metric_db1_col
                comp_value_col = metric_db2_col
                base_db_label = db1_name
                comp_db_label = db2_name
            else:
                base_value_col = metric_db2_col
                comp_value_col = metric_db1_col
                base_db_label = db2_name
                comp_db_label = db1_name
        else:
            # No base_db specified, use db1 as base
            base_value_col = metric_db1_col
            comp_value_col = metric_db2_col
            base_db_label = db1_name
            comp_db_label = db2_name
        
        # Filter out rows where base value is 0 or NaN to avoid division issues
        valid_for_heatmap = valid_metric[
            (valid_metric[base_value_col].notna()) & 
            (valid_metric[comp_value_col].notna()) &
            (valid_metric[base_value_col] != 0)
        ].copy()
        
        if len(valid_for_heatmap) == 0:
            print(f"\nWarning: No valid data for heatmap (base values are all 0 or NaN)!")
            plt.close()
            return
        
        # Debug: Check overall averages first
        print(f"\nDebug: Heatmap calculation for {metric_display_name}")
        print(f"  Base database: {base_db_label}, Comparison database: {comp_db_label}")
        print(f"  Overall average - Base: {valid_for_heatmap[base_value_col].mean():.4f}, Comp: {valid_for_heatmap[comp_value_col].mean():.4f}")
        print(f"  Overall % change: {((valid_for_heatmap[comp_value_col].mean() - valid_for_heatmap[base_value_col].mean()) / valid_for_heatmap[base_value_col].mean() * 100.0):.2f}%")
        print(f"  Total valid samples: {len(valid_for_heatmap)}")
        print(f"  Unique families: {valid_for_heatmap['RNAFamily'].nunique()}")
        print(f"  Unique MSA bins: {valid_for_heatmap['MSA_size_bin'].nunique()}")
        
        # Group by family and MSA bin, then calculate averages for base and comp separately
        # Step 1: Group by family/MSA bin
        grouped = valid_for_heatmap.groupby(['RNAFamily', 'MSA_size_bin'], observed=False)
        print("grouped", grouped)
        
        # Step 2: Calculate mean for base database for each family/MSA bin combination
        base_means = grouped[base_value_col].mean()
        print("base_means", base_means)
        
        # Step 3: Calculate mean for comparison database for each family/MSA bin combination
        comp_means = grouped[comp_value_col].mean()
        print("comp_means", comp_means)
        
        # Step 4: Get counts for each combination
        counts = grouped[base_value_col].count()
        
        # Convert to pivot tables for visualization
        pivot_data_base = base_means.unstack(level='MSA_size_bin')
        pivot_data_comp = comp_means.unstack(level='MSA_size_bin')
        pivot_counts = counts.unstack(level='MSA_size_bin')
        
        # Debug: Check if pivot tables have different values
        print(f"  Pivot table shape: {pivot_data_base.shape}")
        print(f"  Base pivot table - non-NaN cells: {pivot_data_base.notna().sum().sum()}")
        print(f"  Comp pivot table - non-NaN cells: {pivot_data_comp.notna().sum().sum()}")
        
        # Step 5: Calculate percentage change between the bin averages: ((comp_avg - base_avg) / base_avg) * 100
        pivot_data_pct = ((pivot_data_comp - pivot_data_base) / pivot_data_base * 100.0)
        pivot_data_pct = pivot_data_pct.replace([np.inf, -np.inf], np.nan)
        
        # Debug: Check percentage change statistics
        pct_values = pivot_data_pct.values.flatten()
        pct_values = pct_values[~np.isnan(pct_values)]
        if len(pct_values) > 0:
            print(f"  Percentage change stats - Min: {pct_values.min():.2f}%, Max: {pct_values.max():.2f}%, Mean: {pct_values.mean():.2f}%, Std: {pct_values.std():.2f}%")
        
        # Debug: Print some sample values
        print(f"  Sample cells (first 5 non-NaN):")
        sample_count = 0
        for i in range(len(pivot_data_pct.index)):
            for j in range(len(pivot_data_pct.columns)):
                if sample_count >= 5:
                    break
                base_val = pivot_data_base.iloc[i, j]
                comp_val = pivot_data_comp.iloc[i, j]
                pct_value = pivot_data_pct.iloc[i, j]
                if pd.notna(base_val) and pd.notna(comp_val) and pd.notna(pct_value):
                    family = pivot_data_pct.index[i]
                    msa_bin = pivot_data_pct.columns[j]
                    count = pivot_counts.iloc[i, j]
                    print(f"    {family} / {msa_bin} (n={count}): base={base_val:.4f}, comp={comp_val:.4f}, %change={pct_value:.2f}%")
                    sample_count += 1
            if sample_count >= 5:
                break
        
        # Calculate summary row: aggregate all samples per MSA bin (regardless of family)
        # For each MSA bin, directly aggregate all samples in that bin (ignoring family)
        # Step 1: Group by MSA bin only (all samples, regardless of family)
        grouped_by_msa = valid_for_heatmap.groupby('MSA_size_bin', observed=False)
        
        # Step 2: Calculate mean for base database for each MSA bin (direct average of all samples in that bin)
        msa_base_avg = grouped_by_msa[base_value_col].mean()
        
        # Step 3: Calculate mean for comparison database for each MSA bin (direct average of all samples in that bin)
        msa_comp_avg = grouped_by_msa[comp_value_col].mean()
        
        # Step 4: Get total counts per MSA bin (all samples, regardless of family)
        msa_bin_counts = grouped_by_msa[base_value_col].count()
        
        # Step 5: For each MSA bin, calculate percentage change between the averages: ((comp_avg - base_avg) / base_avg) * 100
        # Create a Series with the same index as pivot_data_pct columns to ensure alignment
        msa_bin_pct = pd.Series(index=pivot_data_pct.columns, dtype=float)
        print(f"  Summary row calculation (All samples per MSA bin, regardless of family):")
        for msa_bin in pivot_data_pct.columns:
            if msa_bin in msa_base_avg.index:
                base_avg = msa_base_avg[msa_bin]
                comp_avg = msa_comp_avg[msa_bin]
                total_count = msa_bin_counts[msa_bin]
                if pd.notna(base_avg) and pd.notna(comp_avg) and base_avg != 0:
                    pct_change = ((comp_avg - base_avg) / base_avg) * 100.0
                    msa_bin_pct[msa_bin] = pct_change
                    print(f"    {msa_bin}: base_avg={base_avg:.4f}, comp_avg={comp_avg:.4f}, %change={pct_change:.2f}%, count={total_count} (all samples)")
                else:
                    msa_bin_pct[msa_bin] = np.nan
                    print(f"    {msa_bin}: base_avg={base_avg}, comp_avg={comp_avg} (skipped)")
            else:
                msa_bin_pct[msa_bin] = np.nan
                print(f"    {msa_bin}: no data in grouped_by_msa")
        
        # Add summary row for "All Families" average per MSA bin
        pivot_data_pct_with_summary = pivot_data_pct.copy()
        pivot_data_pct_with_summary.loc['All Families'] = msa_bin_pct
        
        # Create annotation matrix with format "count|%value"
        annot_data = pivot_data_pct.astype(object).copy()
        for i in range(len(pivot_data_pct.index)):
            for j in range(len(pivot_data_pct.columns)):
                count = pivot_counts.iloc[i, j]
                pct_value = pivot_data_pct.iloc[i, j]
                base_val = pivot_data_base.iloc[i, j]
                comp_val = pivot_data_comp.iloc[i, j]
                
                if pd.notna(pct_value) and pd.notna(count) and pd.notna(base_val) and pd.notna(comp_val):
                    sign = '+' if pct_value >= 0 else ''
                    annot_data.iloc[i, j] = f"{int(count)}|{sign}{pct_value:.2f}%"
                else:
                    annot_data.iloc[i, j] = ""
        
        # Add summary row annotations
        annot_data_with_summary = annot_data.copy()
        annot_data_with_summary.loc['All Families'] = [None] * len(pivot_data_pct.columns)
        for msa_bin in pivot_data_pct.columns:
            if msa_bin in msa_bin_pct.index:
                avg_pct = msa_bin_pct[msa_bin]
                total_count = msa_bin_counts[msa_bin]
                if pd.notna(avg_pct) and pd.notna(total_count):
                    sign = '+' if avg_pct >= 0 else ''
                    annot_data_with_summary.loc['All Families', msa_bin] = f"{int(total_count)}|{sign}{avg_pct:.2f}%"
                else:
                    annot_data_with_summary.loc['All Families', msa_bin] = ""
            else:
                annot_data_with_summary.loc['All Families', msa_bin] = ""
        
        sns.heatmap(pivot_data_pct_with_summary, annot=annot_data_with_summary, fmt='', cmap='coolwarm', center=0, 
                    cbar_kws={'label': f'{metric_display_name} % Change ({comp_db_label} vs {base_db_label})'},
                    vmin=None, vmax=None)
    
    plt.xlabel('RNA MSA Size')
    plt.ylabel('RNA Family')
    plt.tight_layout()
    plot_creator.save_plot(f'RNAFamily_MSA_size_heatmap_{metric}')
    plt.close()
    
    print(f"\nPlots saved to: {plot_creator.results_dir}")
    
    # Ensure MSA_size_bin column exists in df for statistics
    # Use valid_metric which has the MSA_size_bin column created
    df_for_stats = df.copy()
    if 'MSA_size_bin' not in df_for_stats.columns and 'MSA_size_bin' in valid_metric.columns:
        # Merge the MSA_size_bin from valid_metric back to df_for_stats
        df_for_stats = df_for_stats.merge(
            valid_metric[['exp_db_id', 'MSA_size_bin']],
            on='exp_db_id',
            how='left'
        )
    
    # Calculate and save statistics with p-values
    write_statistics_to_file(df_for_stats, plot_creator, db1_name, db2_name, metric, metric_display_name, 
                            base_db=base_db, single_db_mode=single_db_mode, metric_avg=metric_avg)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RNAFamily and RNA_msa_size correlation with metric differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script analyzes how RNAFamily and RNA_msa_size correlate with metric differences
between two databases, or differences from average in a single database.

Available metrics: {', '.join(AVAILABLE_METRICS)}

Two modes (automatically detected):
  1. Single-database mode: If folder contains 1 database, calculate differences from the average
  2. Two-database mode: If folder contains 2 databases, compare metrics between databases

The --base_db argument specifies the reference/base database for comparison (two-database mode only):
  - Positive difference = other database > base_db (other is higher)
  - Negative difference = other database < base_db (other is lower)
  - Difference is calculated as (db1 - db2), where db1 and db2 are sorted alphabetically
  - You can specify either the database filename (without .db) or 'db1'/'db2'

In single-database mode:
  - Differences are calculated as: value - average(value)
  - Positive difference = above average
  - Negative difference = below average

Examples:
  # Two-database mode: Analyze all metrics (default)
  python analyzeRNAFamilyMSA.py /path/to/database/folder
  
  # Two-database mode: Analyze specific metric
  python analyzeRNAFamilyMSA.py /path/to/database/folder --metric RNA_TM
  
  # Two-database mode: Specify base database
  python analyzeRNAFamilyMSA.py /path/to/database/folder --base_db af3
  
  # Single-database mode: Analyze all metrics
  python analyzeRNAFamilyMSA.py /path/to/database/folder --single_db
  
  # Single-database mode: Analyze specific metric
  python analyzeRNAFamilyMSA.py /path/to/database/folder --single_db --metric RNA_TM
        """
    )
    parser.add_argument('folder', type=str, help='Folder containing .db file(s)')
    parser.add_argument('--metric', type=str, default=None, choices=AVAILABLE_METRICS,
                       help=f'Metric to analyze (default: all metrics). Available: {", ".join(AVAILABLE_METRICS)}')
    parser.add_argument('--plot_type', type=str, default='rna_family_msa_analysis',
                       help='Plot type for PlotCreator (default: rna_family_msa_analysis)')
    parser.add_argument('--base_db', type=str,
                       help='Base/reference database for comparison (two-database mode only). '
                            'Specify database name without .db extension, e.g., "af3". '
                            'Differences show whether the other database is higher (+) or lower (-) compared to base_db.')
    parser.add_argument('--single_db', action='store_true',
                       help='Force single-database mode (use first database only, even if 2 databases are present)')
    
    args = parser.parse_args()
    
    # Get table names from StartConfig
    config = StartConfig()
    pred_table = config.pred_table
    exp_table = config.ref_table
    
    # Find SQLite databases in the folder
    db_files = sorted([f for f in os.listdir(args.folder) 
                      if f.endswith('.db') and not f.endswith('.backup')])
    
    # Automatically detect mode based on number of databases
    # If --single_db is explicitly set, force single-database mode (even with 2 databases)
    # Otherwise, auto-detect: 1 database = single mode, 2 databases = two-database mode
    if args.single_db or len(db_files) == 1:
        # Single database mode
        if len(db_files) == 0:
            raise ValueError(f"No database files found in {args.folder}")
        if len(db_files) > 1 and not args.single_db:
            raise ValueError(f"Found {len(db_files)} databases. Single-database mode requires exactly 1 database file. Use --single_db to force single-database mode with multiple databases.")
        
        db_path = os.path.join(args.folder, db_files[0])
        db_name = os.path.splitext(db_files[0])[0]
        
        print(f"Analyzing single database:")
        print(f"  Database: {db_files[0]}")
        print(f"  Tables: {pred_table} and {exp_table}")
        print(f"  Mode: Single-database (differences from average)")
        
        # Determine which metrics to analyze
        metrics_to_analyze = [args.metric] if args.metric else AVAILABLE_METRICS
        
        for metric in metrics_to_analyze:
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            print(f"\n{'='*80}")
            print(f"Analyzing metric: {metric} ({metric_display_name})")
            print(f"{'='*80}")
            
            # Get data
            print("\nLoading data...")
            try:
                df, metric_avg = get_data_from_single_database(db_path, pred_table, exp_table, metric)
                print(f"Loaded {len(df)} samples")
                print(f"Average {metric_display_name}: {metric_avg:.4f}")
                
                # Print statistics
                print_statistics(df, None, None, metric, metric_display_name, 
                               single_db_mode=True, metric_avg=metric_avg)
                
                # Create plots
                print("\nCreating plots...")
                plot_creator = PlotCreator(plot_type=args.plot_type, msa_option=None, single_chain_only=False)
                create_plots(df, plot_creator, db_name, db_name, metric, metric_display_name, 
                           single_db_mode=True, metric_avg=metric_avg)
            except Exception as e:
                print(f"Error analyzing {metric}: {e}")
                continue
    elif len(db_files) == 2:
        # Two database mode
        db1_path = os.path.join(args.folder, db_files[0])
        db2_path = os.path.join(args.folder, db_files[1])
        db1_name = os.path.splitext(db_files[0])[0]
        db2_name = os.path.splitext(db_files[1])[0]
        
        # Determine which database name to use for base_db
        base_db_name = None
        if args.base_db:
            # Check if the provided name matches either database
            if args.base_db.lower() == db1_name.lower():
                base_db_name = db1_name
            elif args.base_db.lower() == db2_name.lower():
                base_db_name = db2_name
            else:
                # Also check if user provided db1 or db2
                if args.base_db.lower() == 'db1':
                    base_db_name = db1_name
                elif args.base_db.lower() == 'db2':
                    base_db_name = db2_name
                else:
                    raise ValueError(f"Database name '{args.base_db}' not found. Available databases: {db1_name}, {db2_name}")
        
        print(f"Analyzing databases:")
        print(f"  Database 1: {db_files[0]}")
        print(f"  Database 2: {db_files[1]}")
        print(f"  Tables: {pred_table} and {exp_table}")
        if base_db_name:
            other_db = db2_name if base_db_name == db1_name else db1_name
            print(f"  Base database: {base_db_name}")
            print(f"  Comparing: {other_db} vs {base_db_name} (positive = {other_db} higher, negative = {other_db} lower)")
        
        # Determine which metrics to analyze
        metrics_to_analyze = [args.metric] if args.metric else AVAILABLE_METRICS
        
        for metric in metrics_to_analyze:
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
            print(f"\n{'='*80}")
            print(f"Analyzing metric: {metric} ({metric_display_name})")
            print(f"{'='*80}")
            
            # Get data
            print("\nLoading data...")
            try:
                df = get_data_from_databases(db1_path, db2_path, pred_table, exp_table, metric)
                print(f"Loaded {len(df)} samples")
                
                # Print statistics
                print_statistics(df, db1_name, db2_name, metric, metric_display_name, base_db=base_db_name)
                
                # Create plots
                print("\nCreating plots...")
                plot_creator = PlotCreator(plot_type=args.plot_type, msa_option=None, single_chain_only=False)
                create_plots(df, plot_creator, db1_name, db2_name, metric, metric_display_name, base_db=base_db_name)
            except Exception as e:
                print(f"Error analyzing {metric}: {e}")
                continue
    else:
        raise ValueError(f"Found {len(db_files)} database files in {args.folder}. Expected 1 or 2 databases.")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

