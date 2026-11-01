import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig
import os
from plots.plotCreator import PlotCreator
from utils.plots_dir import plots_root

config = StartConfig()
# Change plot directory to be in the same folder as plots
plot_dir = os.path.join(plots_root(), 'correlation_analysis')
os.makedirs(plot_dir, exist_ok=True)
plot_creator = PlotCreator("correlation_analysis")

# Define metric groups for organizing scatter plots
METRIC_GROUPS = {
    'sequence': ["RNA_GC_Content", "RNA_SequenceComplexity", "domain_counts"],
    'length': ["ProteinLength", "NumberProteins", "RNALength", "NumberRNAs"],
    'motif': ["RNAMotifLength", "RNAmotif_score"],
    'interaction': ["Hbond_proteinRNA", "vdWbond_proteinRNA", "ProteinRNAInterfaceArea", 
                   "ProteinRNAInterfaceRatio", "Free_energy", "Binding_affinity_kd", 
                   "RNA_ElPotential", "RNA_ElPotential_Diff", "iLDDT", "QS_global", "Dockq", "ICS"],
    'bond': ["GlycosidicBond_Similarity", "SugarPucker_Similarity", "GammaAngle_Similarity"],
    'af3': ["af3_protein_pTM", "af3_rna_pTM", "af3_protein_ipTM", "af3_rna_ipTM", 
            "af3_protein_pLDDT_avg", "af3_rna_pLDDT_avg", "af3_global_pae_avg", 
            "af3_chain_pair_pae_min", "af3_fraction_disordered", "af3_has_clash", 
            "af3_ranking_score", "af3_rna_MSA"],
    'structure': ["Complex_RMSD", "Protein_RMSD", "RNA_RMSD", "RNA_LDDT_allAtom", "Complex_LDDT_allAtom",
                 "Protein_LDDT_allAtom", "RNAComplex_TM",
                 "Complex_TM", "RNA_TM", "Protein_TM",  "RNA_LDDT", "Complex_LDDT", "Protein_LDDT"],
    'rna_secondary': ["RNA_DI", "RNA_inf", "RNA_wc", "RNA_nonWc", "RNA_stack", 
                     "RNA_f1_score", "RNA_precision", "RNA_recall", "RNA_mcc", "RNA_wl", 
                     "RNAMotif_isDuplex", "RNA_isDuplex", "RNADuplex_Mismatch", 
                     "RNAMotifDuplex_Mismatch"],
    'other': []  # Empty list - metrics not in other groups will be categorized as 'other'
}


def get_metric_group(metric_name):
    """Determine which metric group a metric belongs to based on x-axis"""
    for group_name, metrics in METRIC_GROUPS.items():
        if metric_name in metrics:
            return group_name
    return 'other'  # Default group for metrics not in any group


def process_list_column(value, column_name=None):
    """Convert string representations of lists to numeric value
    For RNALength, takes the maximum value if it's a list
    For pTM and ipTM columns, takes average if multiple values
    For af3_rna_MSA, returns 0 if any element is 0
    """
    if pd.isna(value) or value == '':
        return None
    try:
        if isinstance(value, str) and value.startswith('['):
            # Parse string representation of list
            lst = ast.literal_eval(value)
            if isinstance(lst, (list, tuple)):
                if column_name == 'RNALength':
                    return float(max(lst))  # Take maximum for RNALength
                if len(lst) == 1:
                    return float(lst[0])
                return sum(float(x) for x in lst) / len(lst)
            # For af3_rna_MSA, return 0 if any element is 0
            if 0 in lst:
                return 0
            return 1
        return float(value)
    except:
        return None


def process_rna_inf(value):
    """Process RNA_INF list into its four components"""
    if pd.isna(value) or value == '':
        return None, None, None, None
    try:
        if isinstance(value, str) and value.startswith('['):
            lst = ast.literal_eval(value)
            if len(lst) == 4:
                return float(lst[0]), float(lst[1]), float(lst[2]), float(lst[3])
    except:
        pass
    return None, None, None, None


def calculate_conformation_similarity(pred_dict, exp_dict):
    """Calculate similarity score between predicted and experimental conformational distributions"""
    if not pred_dict or not exp_dict:
        return None
    
    try:
        # Convert string representations to dictionaries if needed
        if isinstance(pred_dict, str):
            pred_dict = ast.literal_eval(pred_dict)
        if isinstance(exp_dict, str):
            exp_dict = ast.literal_eval(exp_dict)
        
        # Get all unique keys
        all_keys = set(pred_dict.keys()) | set(exp_dict.keys())
        
        # Get total counts for normalization
        pred_total = sum(pred_dict.values())
        exp_total = sum(exp_dict.values())
        
        if pred_total == 0 or exp_total == 0:
            return None
        
        # Calculate normalized difference for each key
        differences = []
        for key in all_keys:
            pred_freq = pred_dict.get(key, 0) / pred_total
            exp_freq = exp_dict.get(key, 0) / exp_total
            differences.append(abs(pred_freq - exp_freq))
        
        # Return similarity score (1 - average difference)
        # Score ranges from 0 (completely different) to 1 (identical distributions)
        return 1 - (sum(differences) / len(differences))
    except:
        return None


def create_correlation_matrix(df, subset_name):
    """Create and save correlation matrix for a data subset"""
    # Skip correlation analysis if there's only one sample
    if len(df) <= 1:
        print(f"Skipping correlation analysis for {subset_name} (n={len(df)}): insufficient samples")
        return None, None

    # Calculate correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Create heatmap
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.1f',
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title(f'Correlation Matrix - {subset_name} (n={len(df)})', pad=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot in plot directory
    plot_creator.save_plot(f'correlation_matrix_{subset_name.lower().replace(" ", "_")}')
    plt.close()

    # Save correlation matrix to CSV in plot directory
    csv_path = os.path.join(plot_dir, f'correlation_matrix_{subset_name.lower().replace(" ", "_")}.csv')
    corr_matrix.to_csv(csv_path)

    # Define excluded correlation pairs
    excluded_pairs = {
        frozenset(['Complex_RMDS', 'Protein_RMSD']),
        frozenset(['RNA_RMSD', 'Protein_RMSD']),
        frozenset(['Complex_RMDS', 'RNA_RMSD']),
        frozenset(['Binding_affinity_kd', 'Free_energy']),
        frozenset(['af3_protein_ipTM', 'af3_rna_ipTM']),
        frozenset(['af3_global_pae_avg', 'af3_chain_pair_pae_min']),
        frozenset(['NumberRNAs', 'ProteinLength'])
    }
    
    # Add RNA metric combinations to excluded pairs
    rna_metrics = ['RNA_f1_score', 'RNA_precision', 'RNA_recall', 'RNA_mcc', 'RNA_wl']
    for i in range(len(rna_metrics)):
        for j in range(i + 1, len(rna_metrics)):
            excluded_pairs.add(frozenset([rna_metrics[i], rna_metrics[j]]))

    # Add af3_ranking_score combinations to excluded pairs
    af3_metrics = [
        'af3_protein_pTM', 'af3_rna_pTM', 'af3_protein_ipTM', 'af3_rna_ipTM',
        'af3_protein_pLDDT_avg', 'af3_rna_pLDDT_avg', 'af3_global_pae_avg',
        'af3_chain_pair_pae_min', 'af3_fraction_disordered', 'af3_has_clash'
    ]
    for metric in af3_metrics:
        excluded_pairs.add(frozenset(['af3_ranking_score', metric]))

    # Find and save strong correlations (using absolute values)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.index[j]
            corr_value = corr_matrix.iloc[i, j]
            
            # Check if correlation pair is in excluded pairs using absolute value
            # Use a lower threshold (0.0) to include all correlations, filtering happens later
            if (abs(corr_value) > 0.4 and abs(corr_value) < 1 and 
                frozenset([var1, var2]) not in excluded_pairs):
                correlations.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Correlation': corr_value,
                    'Abs_Correlation': abs(corr_value)
                })

    corr_df = pd.DataFrame(correlations)
    if not correlations:
        print(f"No strong correlations found for {subset_name}")
        return corr_matrix, None

    # Sort by absolute correlation value
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Save correlations to CSV and text file in plot directory
    csv_path = os.path.join(plot_dir, f'correlations_{subset_name.lower().replace(" ", "_")}.csv')
    txt_path = os.path.join(plot_dir, f'correlations_{subset_name.lower().replace(" ", "_")}.txt')
    
    corr_df.to_csv(csv_path, index=False)

    with open(txt_path, 'w') as f:
        f.write(f"Strong correlations (|correlation| > 0.5) for {subset_name}:\n\n")
        for _, row in corr_df.iterrows():
            sign = "+" if row['Correlation'] > 0 else "-"
            f.write(f"{row['Variable 1']} - {row['Variable 2']}: {sign}{abs(row['Correlation']):.3f}\n")

    return corr_matrix, corr_df


def get_correlation_matrices(filter_type='rna_length'):
    """
    Create correlation matrices based on different filters.
    
    Args:
        filter_type (str): Type of filter to apply ('all', 'rna_length', 'rna_family', 'domain_names', 
                          'rna_complexity', 'gc_content', 'rna_msa', 'complex_type')
    """
    # Connect to database
    db = DatabaseMethods()

    # Add domain_names and RNAFamily to the query
    pred_columns = [
        "p.exp_db_id", "p.ProteinLength", "p.NumberProteins", "p.RNALength", "p.NumberRNAs",
        "p.RNAMotifLength", "p.Hbond_proteinRNA", "p.vdWbond_proteinRNA",
        "p.ProteinRNAInterfaceArea", "p.ProteinRNAInterfaceRatio", "p.Free_energy",
        "p.Binding_affinity_kd", "p.af3_protein_pTM", "p.af3_rna_pTM", "p.af3_protein_ipTM",
        "p.af3_rna_ipTM", "p.af3_protein_pLDDT_avg", "p.af3_rna_pLDDT_avg",
        "p.af3_global_pae_avg", "p.af3_chain_pair_pae_min", "p.af3_fraction_disordered",
        "p.af3_has_clash", "p.af3_ranking_score", "p.af3_rna_MSA", "p.Complex_RMSD",
        "p.Protein_RMSD", "p.RNA_RMSD", "p.Protein_LDDT", "p.RNA_LDDT", "p.Complex_LDDT",
        "p.RNA_TM", "p.RNA_DI", "p.RNA_INF",
        "p.RNA_f1_score", "p.RNA_precision", "p.RNA_recall", "p.RNA_mcc", "p.RNA_wl",
        "p.RNAmotif_score", "p.RNAMotif_isDuplex", "e.RNAMotif_isDuplex as e_RNAMotif_isDuplex",
        "p.RNA_isDuplex", "e.RNA_isDuplex as e_RNA_isDuplex",
        "p.RNA_GlycosidicBond as p_RNA_GlycosidicBond", "e.RNA_GlycosidicBond as e_RNA_GlycosidicBond",
        "p.RNA_SugarPucker as p_RNA_SugarPucker", "e.RNA_SugarPucker as e_RNA_SugarPucker",
        "p.RNA_GammaAngle as p_RNA_GammaAngle", "e.RNA_GammaAngle as e_RNA_GammaAngle",
        "e.domain_counts", "e.domain_names", "e.RNAFamily",
        "p.RNA_ElectrostaticPotential as RNA_ElPotential", 
        "e.RNA_ElectrostaticPotential as e_RNA_ElPotential",
        "p.RNA_GC_Content as RNA_GC_Content",
        "p.RNA_AU_Content as RNA_AU_Content",
        "p.RNA_SequenceComplexity", "e.exp_pH as exp_pH",
        "p.RNA_LDDT_allAtom", "p.Complex_LDDT_allAtom", "p.iLDDT", "p.QS_global", "p.Dockq", "p.ICS",
        "p.RNAmatch_score", "p.AAmatch_score", "p.Complex_TM", "p.Protein_LDDT_allAtom", "p.RNAComplex_TM"
    ]

    # Query data with JOIN
    query = f"""
    SELECT {', '.join(pred_columns)}
    FROM pred_protein_rna p
    JOIN exp_protein_rna e ON p.exp_db_id = e.PDBId
    """
    df = pd.read_sql_query(query, db.connection)

    # Process the data as before
    df[['RNA_inf', 'RNA_wc', 'RNA_nonWc', 'RNA_stack']] = pd.DataFrame(
        df['RNA_INF'].apply(process_rna_inf).tolist(),
        index=df.index
    )

    # Calculate RNA_ElPotential_Diff
    def calculate_elpotential_diff(row):
        pred_val = row['RNA_ElPotential']
        exp_val = row['e_RNA_ElPotential']
        if pd.isna(pred_val) or pd.isna(exp_val):
            return None
        # Calculate percentage difference and normalize to 0-1
        diff = abs(pred_val - exp_val) / max(abs(pred_val), abs(exp_val))
        return 1 / (1 + np.exp(-5 * (diff - 0.5)))  # Sigmoid normalization

    df['RNA_ElPotential_Diff'] = df.apply(calculate_elpotential_diff, axis=1)

    list_columns = ['Complex_RMSD', 'Protein_RMSD', 'RNA_RMSD', 'Hbond_proteinRNA',
                    'vdWbond_proteinRNA', 'af3_chain_pair_pae_min',
                    'af3_protein_pTM', 'af3_rna_pTM', 'af3_protein_ipTM', 'af3_rna_ipTM',
                    'af3_rna_MSA', 'RNALength']

    for col in list_columns:
        df[col] = df[col].apply(lambda x: process_list_column(x, col))

    # Process other columns as before
    df['domain_counts'] = df['domain_counts'].apply(process_domain_counts)
    df['RNADuplex_Mismatch'] = df.apply(calculate_duplex_mismatch, axis=1)
    df['RNAMotifDuplex_Mismatch'] = df.apply(calculate_motif_duplex_mismatch, axis=1)
    df['GlycosidicBond_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_GlycosidicBond'], row['e_RNA_GlycosidicBond']), axis=1)
    df['SugarPucker_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_SugarPucker'], row['e_RNA_SugarPucker']), axis=1)
    df['GammaAngle_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_GammaAngle'], row['e_RNA_GammaAngle']), axis=1)

    # Drop non-numeric columns
    columns_to_drop = ['exp_db_id', 'e_RNA_isDuplex', 'e_RNAMotif_isDuplex',
                      'p_RNA_GlycosidicBond', 'e_RNA_GlycosidicBond',
                      'p_RNA_SugarPucker', 'e_RNA_SugarPucker',
                      'p_RNA_GammaAngle', 'e_RNA_GammaAngle',
                      'RNA_INF', 'domain_names', 'RNAFamily',
                      'e_RNA_ElPotential']  # Drop the experimental ElPotential after calculating diff
    numeric_df = df.drop(columns_to_drop, axis=1)
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

    # First, process all data without filtering
    if filter_type == 'all':
        create_correlation_matrix(numeric_df, "All_Data")

    elif filter_type == 'rna_length':
        filters = [
            ('RNA Length 0-10', lambda x: x <= 10),
            ('RNA Length 10-20', lambda x: 10 < x <= 20),
            ('RNA Length 20-60', lambda x: 20 < x <= 60),
            ('RNA Length >60', lambda x: x > 60)
        ]
        for name, condition in filters:
            subset = numeric_df[numeric_df['RNALength'].apply(lambda x: condition(x) if pd.notna(x) else False)]
            if len(subset) > 1:
                create_correlation_matrix(subset, name)

    elif filter_type == 'rna_family':
        # Process RNA families
        def normalize_rna_family(x):
            if pd.isna(x):
                return 'None'
            if isinstance(x, str):
                if x.startswith('['):
                    try:
                        lst = ast.literal_eval(x)
                        if all(item == 'None' for item in lst):
                            return 'None'
                        return next((item for item in lst if item != 'None'), 'None')
                    except:
                        return x
            return str(x)

        # Get unique normalized RNA families and their counts
        df['normalized_rna_family'] = df['RNAFamily'].apply(normalize_rna_family)
        family_counts = df['normalized_rna_family'].value_counts()
        
        # Filter families with 3 or more samples and get top 5
        common_families = family_counts[family_counts >= 3].head(5).index.tolist()
        print(f"\nMost common RNA families (min 3 samples):")
        for family in common_families:
            print(f"  {family}: {family_counts[family]} samples")
        
        for family in common_families:
            subset = numeric_df[df['normalized_rna_family'] == family]
            if len(subset) > 1:  # Only process if we have more than one sample
                create_correlation_matrix(subset, f"RNA_Family_{family}")

    elif filter_type == 'domain_names':
        # Process domains
        def extract_domains(x):
            if pd.isna(x):
                return []
            # Filter out "-" and empty strings, and strip space
            return [d.strip() for d in str(x).split(';') if d.strip() and d.strip() != '-']

        # Get all domains and their counts
        domain_counts = {}
        for domains in df['domain_names'].dropna():
            for domain in extract_domains(domains):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Filter domains with 3 or more samples and get top 5
        common_domains = sorted([(count, domain) for domain, count in domain_counts.items() if count >= 3], 
                              reverse=True)[:5]
        
        print(f"\nMost common domains (min 3 samples):")
        for count, domain in common_domains:
            print(f"  {domain}: {count} samples")

        # Process each common domain
        for _, domain in common_domains:
            subset = numeric_df[df['domain_names'].apply(
                lambda x: domain in extract_domains(x) if pd.notna(x) else False
            )]
            
            if len(subset) > 1:  # Only process if we have more than one sample
                create_correlation_matrix(subset, f"Domain_{domain}")

    elif filter_type == 'rna_complexity':
        # Process RNA sequence complexity ranges
        complexity_ranges = [
            ('RNA Complexity 0.00-0.39', lambda x: 0.00 <= x <= 0.39),
            ('RNA Complexity 0.40-0.69', lambda x: 0.40 <= x <= 0.69),
            ('RNA Complexity 0.70-1.00', lambda x: 0.70 <= x <= 1.00)
        ]
        
        for name, condition in complexity_ranges:
            subset = numeric_df[numeric_df['RNA_SequenceComplexity'].apply(lambda x: condition(x) if pd.notna(x) else False)]
            if len(subset) > 1:
                create_correlation_matrix(subset, name)

    elif filter_type == 'gc_content':
        # Process GC content ranges
        gc_ranges = [
            ('GC Content 0.00-0.50', lambda x: 0.00 <= x <= 0.50),
            ('GC Content 0.51-1.00', lambda x: 0.51 <= x <= 1.00)
        ]
        
        for name, condition in gc_ranges:
            subset = numeric_df[numeric_df['RNA_GC_Content'].apply(lambda x: condition(x) if pd.notna(x) else False)]
            if len(subset) > 1:
                create_correlation_matrix(subset, name)

    elif filter_type == 'au_content':
        # Process GC content ranges
        au_ranges = [
            ('AU Content 0.00-0.50', lambda x: 0.00 <= x <= 0.50),
            ('AU Content 0.51-1.00', lambda x: 0.51 <= x <= 1.00)
        ]

        for name, condition in au_ranges:
            subset = numeric_df[numeric_df['RNA_AU_Content'].apply(lambda x: condition(x) if pd.notna(x) else False)]
            if len(subset) > 1:
                create_correlation_matrix(subset, name)

    elif filter_type == 'rna_msa':
        # Process based on RNA MSA value
        def get_msa_value(x):
            if pd.isna(x):
                return None
            try:
                if isinstance(x, str) and x.startswith('['):
                    lst = ast.literal_eval(x)
                    return 0 if 0 in lst else 1
                return float(x)
            except:
                return None

        df['msa_value'] = df['af3_rna_MSA'].apply(get_msa_value)
        
        # Create subsets for MSA=0 and MSA=1
        for msa_value in [0, 1]:
            subset = numeric_df[df['msa_value'] == msa_value]
            if len(subset) > 1:
                create_correlation_matrix(subset, f"RNA_MSA_{msa_value}")

    elif filter_type == 'complex_type':
        # Process based on number of proteins and RNAs
        def is_single_complex(row):
            return 1 if pd.notna(row['NumberProteins']) and pd.notna(row['NumberRNAs']) and row['NumberProteins'] == 1 and row['NumberRNAs'] == 1 else 0

        df['is_single_complex'] = df.apply(is_single_complex, axis=1)
        
        # Create subsets for single and multi-complex
        for complex_type in [0, 1]:
            subset = numeric_df[df['is_single_complex'] == complex_type]
            if len(subset) > 1:
                name = "Single_Complex" if complex_type == 1 else "Multi_Complex"
                create_correlation_matrix(subset, name)

    return db  # Return the database connection to be closed in main()


def process_domain_counts(x):
    """Process domain_counts column to return total number of domains"""
    if pd.notna(x):
        try:
            # Split by semicolon and filter out any non-numeric values
            counts = [int(count) for count in x.split(';') if count.isdigit()]
            return sum(counts) if counts else None
        except:
            return None
    return None


def calculate_duplex_mismatch(row):
    """Calculate mismatch between predicted and experimental RNA_isDuplex"""
    pred_duplex = row['RNA_isDuplex']
    exp_duplex = row['e_RNA_isDuplex']
    if pd.isna(pred_duplex) or pd.isna(exp_duplex):
        return None
    return 0 if pred_duplex == exp_duplex else 1


def calculate_motif_duplex_mismatch(row):
    """Calculate mismatch between predicted and experimental RNAMotif_isDuplex"""
    pred_duplex = row['RNAMotif_isDuplex']
    exp_duplex = row['e_RNAMotif_isDuplex']
    if pd.isna(pred_duplex) or pd.isna(exp_duplex):
        return None
    return 0 if pred_duplex == exp_duplex else 1


def get_full_dataframe():
    """Get the full processed dataframe with all metrics and metadata"""
    db = DatabaseMethods()

    pred_columns = [
        "p.exp_db_id", "p.ProteinLength", "p.NumberProteins", "p.RNALength", "p.NumberRNAs",
        "p.RNAMotifLength", "p.Hbond_proteinRNA", "p.vdWbond_proteinRNA",
        "p.ProteinRNAInterfaceArea", "p.ProteinRNAInterfaceRatio", "p.Free_energy",
        "p.Binding_affinity_kd", "p.af3_protein_pTM", "p.af3_rna_pTM", "p.af3_protein_ipTM",
        "p.af3_rna_ipTM", "p.af3_protein_pLDDT_avg", "p.af3_rna_pLDDT_avg",
        "p.af3_global_pae_avg", "p.af3_chain_pair_pae_min", "p.af3_fraction_disordered",
        "p.af3_has_clash", "p.af3_ranking_score", "p.af3_rna_MSA", "p.Complex_RMSD",
        "p.Protein_RMSD", "p.RNA_RMSD", "p.Protein_LDDT", "p.RNA_LDDT", "p.Protein_TM",
        "p.RNA_TM", "p.RNA_DI", "p.RNA_INF",
        "p.RNA_f1_score", "p.RNA_precision", "p.RNA_recall", "p.RNA_mcc", "p.RNA_wl",
        "p.RNAmotif_score", "p.RNAMotif_isDuplex", "e.RNAMotif_isDuplex as e_RNAMotif_isDuplex",
        "p.RNA_isDuplex", "e.RNA_isDuplex as e_RNA_isDuplex",
        "p.RNA_GlycosidicBond as p_RNA_GlycosidicBond", "e.RNA_GlycosidicBond as e_RNA_GlycosidicBond",
        "p.RNA_SugarPucker as p_RNA_SugarPucker", "e.RNA_SugarPucker as e_RNA_SugarPucker",
        "p.RNA_GammaAngle as p_RNA_GammaAngle", "e.RNA_GammaAngle as e_RNA_GammaAngle",
        "e.domain_counts", "e.domain_names", "e.RNAFamily",
        "p.RNA_ElectrostaticPotential as RNA_ElPotential",
        "e.RNA_ElectrostaticPotential as e_RNA_ElPotential",
        "p.RNA_GC_Content as RNA_GC_Content",
        "p.RNA_AU_Content as RNA_AU_Content",
        "p.RNA_SequenceComplexity", "e.exp_pH as exp_pH", "p.RNAComplex_TM",
        "p.RNA_LDDT_allAtom", "p.Complex_LDDT_allAtom", "p.iLDDT", "p.QS_global", "p.Dockq", "p.ICS",
        "p.RNAmatch_score", "p.AAmatch_score", "p.Complex_TM", "p.Complex_LDDT", "p.Protein_LDDT_allAtom",
        "e.Experiment as Experiment",
        "e.Deposited_date as Deposited_date"
    ]

    query = f"""
    SELECT {', '.join(pred_columns)}
    FROM pred_protein_rna p
    JOIN exp_protein_rna e ON p.exp_db_id = e.PDBId
    """
    df = pd.read_sql_query(query, db.connection)
    db.connection.close()

    # Process the data
    df[['RNA_inf', 'RNA_wc', 'RNA_nonWc', 'RNA_stack']] = pd.DataFrame(
        df['RNA_INF'].apply(process_rna_inf).tolist(),
        index=df.index
    )

    def calculate_elpotential_diff(row):
        pred_val = row['RNA_ElPotential']
        exp_val = row['e_RNA_ElPotential']
        if pd.isna(pred_val) or pd.isna(exp_val):
            return None
        diff = abs(pred_val - exp_val) / max(abs(pred_val), abs(exp_val))
        return 1 / (1 + np.exp(-5 * (diff - 0.5)))

    df['RNA_ElPotential_Diff'] = df.apply(calculate_elpotential_diff, axis=1)

    list_columns = ['Complex_RMSD', 'Protein_RMSD', 'RNA_RMSD', 'Hbond_proteinRNA',
                    'vdWbond_proteinRNA', 'af3_chain_pair_pae_min',
                    'af3_protein_pTM', 'af3_rna_pTM', 'af3_protein_ipTM', 'af3_rna_ipTM',
                    'af3_rna_MSA', 'RNALength']

    for col in list_columns:
        df[col] = df[col].apply(lambda x: process_list_column(x, col))

    df['domain_counts'] = df['domain_counts'].apply(process_domain_counts)
    df['RNADuplex_Mismatch'] = df.apply(calculate_duplex_mismatch, axis=1)
    df['RNAMotifDuplex_Mismatch'] = df.apply(calculate_motif_duplex_mismatch, axis=1)
    df['GlycosidicBond_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_GlycosidicBond'], row['e_RNA_GlycosidicBond']), axis=1)
    df['SugarPucker_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_SugarPucker'], row['e_RNA_SugarPucker']), axis=1)
    df['GammaAngle_Similarity'] = df.apply(lambda row: calculate_conformation_similarity(row['p_RNA_GammaAngle'], row['e_RNA_GammaAngle']), axis=1)

    # Process MSA value for categorization
    def get_msa_value(x):
        if pd.isna(x):
            return None
        try:
            if isinstance(x, str) and x.startswith('['):
                lst = ast.literal_eval(x)
                return 0 if 0 in lst else 1
            return float(x)
        except:
            return None

    df['msa_value'] = df['af3_rna_MSA'].apply(get_msa_value)
    
    # Categorize RNA length
    def categorize_rna_length(x):
        if pd.isna(x):
            return None
        if x <= 10:
            return '0-10'
        elif x <= 20:
            return '10-20'
        elif x <= 60:
            return '20-60'
        else:
            return '>60'
    
    df['rna_length_category'] = df['RNALength'].apply(categorize_rna_length)
    
    # Categorize RNA complexity
    def categorize_rna_complexity(x):
        if pd.isna(x):
            return None
        if x <= 0.39:
            return '0.00-0.39'
        elif x <= 0.69:
            return '0.40-0.69'
        else:
            return '0.70-1.00'
    
    df['rna_complexity_category'] = df['RNA_SequenceComplexity'].apply(categorize_rna_complexity)
    
    # Categorize GC content
    def categorize_gc_content(x):
        if pd.isna(x):
            return None
        if x <= 0.50:
            return '0.00-0.50'
        else:
            return '0.51-1.00'
    
    df['gc_content_category'] = df['RNA_GC_Content'].apply(categorize_gc_content)
    
    # Categorize Experiment type
    def categorize_experiment(x):
        if pd.isna(x):
            return None
        x_str = str(x).upper()
        if 'X-RAY' in x_str or 'DIFFRACTION' in x_str:
            return 'X-RAY DIFFRACTION'
        elif 'ELECTRON' in x_str or 'MICROSCOPY' in x_str:
            return 'ELECTRON MICROSCOPY'
        return None
    
    df['experiment_category'] = df['Experiment'].apply(categorize_experiment)
    
    # Categorize Deposited_date
    def categorize_deposited_date(x):
        if pd.isna(x):
            return None
        try:
            # Try to parse the date
            if isinstance(x, str):
                date_obj = pd.to_datetime(x, errors='coerce')
            else:
                date_obj = pd.to_datetime(x, errors='coerce')
            
            if pd.isna(date_obj):
                return None
            
            # Compare with 2021-10-01
            cutoff_date = pd.to_datetime('2021-10-01')
            if date_obj >= cutoff_date:
                return 'a2021'
            else:
                return 'b2021'
        except:
            return None
    
    df['deposited_date_category'] = df['Deposited_date'].apply(categorize_deposited_date)
    
    # Categorize RNA Family
    def normalize_rna_family(x):
        if pd.isna(x):
            return 'None'
        if isinstance(x, str):
            if x.startswith('['):
                try:
                    lst = ast.literal_eval(x)
                    if all(item == 'None' for item in lst):
                        return 'None'
                    return next((item for item in lst if item != 'None'), 'None')
                except:
                    return x if x != 'None' else 'None'
        return str(x) if str(x) != 'None' else 'None'
    
    df['rna_family_category'] = df['RNAFamily'].apply(normalize_rna_family)
    
    # Categorize Domain Names
    def extract_domains(x):
        if pd.isna(x):
            return 'None'
        # Filter out "-" and empty strings, and strip space
        domains = [d.strip() for d in str(x).split(';') if d.strip() and d.strip() != '-']
        if not domains:
            return 'None'
        # Return the first domain (or join if multiple)
        return domains[0] if len(domains) == 1 else '; '.join(domains[:3])  # Limit to first 3 domains
    
    df['domain_names_category'] = df['domain_names'].apply(extract_domains)
    
    # Keep exp_db_id for reference but don't include in numeric operations
    return df


def get_excluded_pairs():
    """Get the set of excluded correlation pairs (same logic as in create_correlation_matrix)"""
    excluded_pairs = {
        frozenset(['Complex_LDDT_allAtom', 'RNA_LDDT_allAtom']),
        frozenset(['Complex_RMDS', 'Protein_RMSD']),
        frozenset(['RNA_RMSD', 'Protein_RMSD']),
        frozenset(['Complex_RMDS', 'RNA_RMSD']),
        frozenset(['Binding_affinity_kd', 'Free_energy']),
        frozenset(['af3_protein_ipTM', 'af3_rna_ipTM']),
        frozenset(['af3_global_pae_avg', 'af3_chain_pair_pae_min']),
        frozenset(['NumberRNAs', 'ProteinLength'])
    }
    
    # Add RNA metric combinations to excluded pairs
    rna_metrics = ['RNA_f1_score', 'RNA_precision', 'RNA_recall', 'RNA_mcc', 'RNA_wl']
    for i in range(len(rna_metrics)):
        for j in range(i + 1, len(rna_metrics)):
            excluded_pairs.add(frozenset([rna_metrics[i], rna_metrics[j]]))

    interface_metrics = ['iLDDT', 'QS_global', 'Dockq', 'ICS']
    for i in range(len(interface_metrics)):
        for j in range(i + 1, len(interface_metrics)):
            excluded_pairs.add(frozenset([interface_metrics[i], interface_metrics[j]]))

    # Add af3_ranking_score combinations to excluded pairs
    af3_metrics = [
        'af3_protein_pTM', 'af3_rna_pTM', 'af3_protein_ipTM', 'af3_rna_ipTM',
        'af3_protein_pLDDT_avg', 'af3_rna_pLDDT_avg', 'af3_global_pae_avg',
        'af3_chain_pair_pae_min', 'af3_fraction_disordered', 'af3_has_clash', 'af3_ranking_score'
    ]
    for i in range(len(af3_metrics)):
        for j in range(i + 1, len(af3_metrics)):
            excluded_pairs.add(frozenset([af3_metrics[i], af3_metrics[j]]))
    
    return excluded_pairs


def generate_correlation_scatter_plots(custom_pairs=None, top_n=20, min_correlation=0.6, 
                                      plot_msa=False, plot_length=False, plot_complexity=False, 
                                      plot_gc=False, plot_experiment=False, plot_date=False,
                                      plot_rna_family=False, plot_domain_names=False):
    """
    Generate scatter plots for significant correlations, colored by different categories.
    
    Args:
        custom_pairs (list): List of tuples containing pairs to exclude (e.g., [('RNA_GC_Content', 'RNA_AU_Content'), ...])
        top_n (int): Maximum number of top correlations to plot (if None, plots all above min_correlation)
        min_correlation (float): Minimum absolute correlation value to include (default: 0.6)
        plot_msa (bool): Whether to generate plots colored by RNA MSA
        plot_length (bool): Whether to generate plots colored by RNA Length
        plot_complexity (bool): Whether to generate plots colored by RNA Complexity
        plot_gc (bool): Whether to generate plots colored by RNA GC Content
        plot_experiment (bool): Whether to generate plots colored by Experiment Type
        plot_date (bool): Whether to generate plots colored by Deposited Date
        plot_rna_family (bool): Whether to generate plots colored by RNA Family
        plot_domain_names (bool): Whether to generate plots colored by Domain Names
    """
    if custom_pairs is None:
        custom_pairs = []
    
    # Read correlations from All_Data
    corr_csv_path = os.path.join(plot_dir, 'correlations_all_data.csv')
    if not os.path.exists(corr_csv_path):
        print(f"Correlation file not found: {corr_csv_path}")
        print("Please run correlation analysis first.")
        return
    
    corr_df = pd.read_csv(corr_csv_path)
    
    # Get excluded pairs (same as in correlation analysis)
    excluded_pairs = get_excluded_pairs()
    
    # Combine excluded pairs with custom pairs
    custom_pairs_set = {frozenset(pair) for pair in custom_pairs}
    all_excluded_pairs = excluded_pairs | custom_pairs_set
    
    # Metrics to exclude (any pair containing these will be excluded)
    excluded_metrics = ['RNA_isDuplex', 'RNAMotif_isDuplex', 'RNA_precision', 'RNA_recall', 'RNA_f1_score', 'af3_rna_MSA',
                       'RNADuplex_Mismatch', 'RNAMotifDuplex_Mismatch', 'RNA_ElPotential', 'Binding_affinity_kd']
    
    # Additional specific pairs to exclude
    additional_excluded_pairs = {
        frozenset(['RNALength', 'RNA_ElPotential']),
        frozenset(['RNA_DI', 'RNA_inf']),  # RNA_inf is the processed version of RNA_INF
        frozenset(['RNA_wc', 'RNA_nonWc']),
        frozenset(['RNA_inf', 'RNA_wc']),  # RNA_inf is the processed version of RNA_INF
        frozenset(['RNA_inf', 'RNA_nonWc']),  # RNA_inf is the processed version of RNA_INF
        frozenset(['RNA_inf', 'RNA_stack']),  # RNA_inf is the processed version of RNA_INF
        frozenset(['RNA_DI', 'RNA_wc']),
        frozenset(['RNA_DI', 'RNA_nonWc']),
        frozenset(['RNA_DI', 'RNA_stack']),
        frozenset(['RNA_mcc', 'RNA_wc']),
        frozenset(['RNA_mcc', 'RNA_nonWc']),
        frozenset(['RNA_mcc', 'RNA_stack'])
    }
    
    # Combine all excluded pairs
    all_excluded_pairs = all_excluded_pairs | additional_excluded_pairs
    
    # Filter out excluded pairs and excluded metrics
    filtered_correlations = []
    for _, row in corr_df.iterrows():
        var1 = row['Variable 1']
        var2 = row['Variable 2']
        abs_corr = abs(row['Correlation'])
        
        # Skip if either variable is in excluded metrics
        if var1 in excluded_metrics or var2 in excluded_metrics:
            continue
        
        # Skip if pair is in excluded pairs
        pair = frozenset([var1, var2])
        if pair in all_excluded_pairs:
            continue
        
        # Only include correlations with absolute value >= min_correlation
        if abs_corr >= min_correlation and abs_corr < 1.0:
            filtered_correlations.append(row)
    
    if not filtered_correlations:
        print(f"No correlations found after filtering (min correlation: {min_correlation}).")
        return
    
    # Sort by absolute correlation and optionally limit to top N
    filtered_df = pd.DataFrame(filtered_correlations)
    filtered_df = filtered_df.sort_values('Abs_Correlation', ascending=False)
    
    if top_n is not None and len(filtered_df) > top_n:
        filtered_df = filtered_df.head(top_n)
        print(f"\nGenerating scatter plots for top {len(filtered_df)} correlations (|correlation| >= {min_correlation})...")
    else:
        print(f"\nGenerating scatter plots for {len(filtered_df)} correlations (|correlation| >= {min_correlation})...")
    
    # Get full dataframe with all data
    df = get_full_dataframe()
    
    # Define color maps for different categories
    msa_color_map = {'-MSA': 'orange', '+MSA': 'blue'}
    length_color_map = {'0-10': '#1f77b4', '10-20': '#2ca02c', '20-60': '#ff7f0e', '>60': '#d62728'}
    complexity_color_map = {'0.00-0.39': '#9467bd', '0.40-0.69': '#8c564b', '0.70-1.00': '#e377c2'}
    gc_color_map = {'0.00-0.50': '#7f7f7f', '0.51-1.00': '#bcbd22'}
    experiment_color_map = {'X-RAY DIFFRACTION': '#2ca02c', 'ELECTRON MICROSCOPY': '#d62728'}
    date_color_map = {'a2021': '#17becf', 'b2021': '#393b79'}
    
    # Generate scatter plots for each correlation pair
    for idx, row in filtered_df.iterrows():
        var1 = row['Variable 1']
        var2 = row['Variable 2']
        corr_value = row['Correlation']
        
        # Check if both variables exist in dataframe
        if var1 not in df.columns or var2 not in df.columns:
            print(f"Skipping {var1} vs {var2}: columns not found in dataframe")
            continue
        
        # Work on a copy to avoid modifying original dataframe
        plot_df = df[[var1, var2, 'msa_value', 'rna_length_category', 
                     'rna_complexity_category', 'gc_content_category',
                     'experiment_category', 'deposited_date_category',
                     'rna_family_category', 'domain_names_category']].copy()
        
        # Convert to numeric and filter out invalid values (including empty strings)
        plot_df[var1] = pd.to_numeric(plot_df[var1], errors='coerce')
        plot_df[var2] = pd.to_numeric(plot_df[var2], errors='coerce')
        
        # Get valid data points (non-null and numeric for both variables)
        valid_mask = plot_df[var1].notna() & plot_df[var2].notna()
        plot_df = plot_df[valid_mask].copy()
        
        if len(plot_df) == 0:
            print(f"Skipping {var1} vs {var2}: no valid data points")
            continue
        
        # Create safe filename
        safe_var1 = var1.replace('/', '_').replace(' ', '_')
        safe_var2 = var2.replace('/', '_').replace(' ', '_')
        
        # Determine metric group based on x-axis (var1)
        metric_group = get_metric_group(var1)
        
        # Helper function to save plot to appropriate subfolder based on x-axis metric group
        def save_plot_to_group(plot_name, metric_group):
            """Save plot to subfolder based on metric group"""
            # Use the same directory structure as plot_creator (includes all_chains)
            # plot_creator.results_dir is something like: .../results/plots/all_chains/correlation_analysis
            scatter_plots_base = os.path.join(plot_creator.results_dir, 'scatter_plots')
            group_dir = os.path.join(scatter_plots_base, metric_group)
            os.makedirs(group_dir, exist_ok=True)
            filename = f'{plot_name}.png'
            plt.savefig(os.path.join(group_dir, filename), bbox_inches='tight')
        
        # Helper function to create scatter plot with proper data filtering
        def create_scatter_plot(category_col, category_valid_mask, color_map, plot_name_suffix, legend_title):
            """Helper function to create scatter plot with proper filtering"""
            if category_valid_mask.sum() == 0:
                return
            
            # Get valid numeric values
            x_vals = plot_df[var1][category_valid_mask].copy()
            y_vals = plot_df[var2][category_valid_mask].copy()
            categories = plot_df[category_col][category_valid_mask].copy()
            
            # Convert to numeric again to ensure no strings or empty values
            x_vals = pd.to_numeric(x_vals, errors='coerce')
            y_vals = pd.to_numeric(y_vals, errors='coerce')
            
            # Filter out any remaining invalid values
            final_valid = x_vals.notna() & y_vals.notna() & categories.notna()
            if final_valid.sum() == 0:
                return
            
            x_vals = x_vals[final_valid].tolist()
            y_vals = y_vals[final_valid].tolist()
            categories = categories[final_valid].tolist()
            
            # Use categories directly (without prefix) - legend title will be set separately
            # Create color map with just the category values (no prefix)
            
            # Create the plot directly to have control over save location
            plt.figure(figsize=(10, 8))
            plot_df_temp = pd.DataFrame({
                'x': x_vals,
                'y': y_vals,
                'category': categories
            })
            
            handles = []
            for cat, color in color_map.items():
                subset = plot_df_temp[plot_df_temp['category'] == cat]
                if not subset.empty:
                    # Convert color to proper format if it's an array
                    plot_color = color
                    if isinstance(color, (list, tuple, np.ndarray)) and len(color) > 1:
                        # If it's an RGBA array, convert to tuple
                        if len(color) == 4:
                            plot_color = tuple(color[:3])  # Use RGB only
                        else:
                            plot_color = tuple(color) if len(color) == 3 else color
                    plt.scatter(subset['x'], subset['y'], color=plot_color, label=cat, alpha=0.7)
                    handles.append(
                        plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                                  markerfacecolor=plot_color, markersize=10))
            
            plt.xlabel(var1)
            plt.ylabel(var2)
            # Set legend with title
            plt.legend(handles=handles, title=legend_title)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Save to appropriate subfolder
            plot_name = f'scatter_{safe_var1}_vs_{safe_var2}_by_{plot_name_suffix}'
            save_plot_to_group(plot_name, metric_group)
            plt.close()
        
        # Plot 1: Colored by RNA MSA
        if plot_msa:
            # First create MSA category column
            plot_df['msa_category'] = plot_df['msa_value'].apply(lambda x: '+MSA' if x == 1 else '-MSA' if x == 0 else 'Unknown')
            msa_valid = plot_df['msa_category'] != 'Unknown'
            if msa_valid.sum() > 0:
                x_vals = plot_df[var1][msa_valid].copy()
                y_vals = plot_df[var2][msa_valid].copy()
                categories = plot_df['msa_category'][msa_valid].copy()
                
                # Convert to numeric to ensure no strings
                x_vals = pd.to_numeric(x_vals, errors='coerce')
                y_vals = pd.to_numeric(y_vals, errors='coerce')
                
                # Final filter
                final_valid = x_vals.notna() & y_vals.notna()
                if final_valid.sum() > 0:
                    x_vals = x_vals[final_valid].tolist()
                    y_vals = y_vals[final_valid].tolist()
                    categories = categories[final_valid].tolist()
                    
                    # Use categories directly (without prefix) - legend title will be set separately
                    
                    # Create the plot
                    plt.figure(figsize=(10, 8))
                    plot_df_temp = pd.DataFrame({
                        'x': x_vals,
                        'y': y_vals,
                        'category': categories
                    })
                    
                    handles = []
                    for cat, color in msa_color_map.items():
                        subset = plot_df_temp[plot_df_temp['category'] == cat]
                        if not subset.empty:
                            # Convert color to proper format if it's an array
                            plot_color = color
                            if isinstance(color, (list, tuple, np.ndarray)) and len(color) > 1:
                                # If it's an RGBA array, convert to tuple
                                if len(color) == 4:
                                    plot_color = tuple(color[:3])  # Use RGB only
                                else:
                                    plot_color = tuple(color) if len(color) == 3 else color
                            plt.scatter(subset['x'], subset['y'], color=plot_color, label=cat, alpha=0.7)
                            handles.append(
                                plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                                          markerfacecolor=plot_color, markersize=10))
                    
                    plt.xlabel(var1)
                    plt.ylabel(var2)
                    # Set legend with title
                    plt.legend(handles=handles, title='RNA MSA')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    # Save to appropriate subfolder
                    plot_name = f'scatter_{safe_var1}_vs_{safe_var2}_by_msa'
                    save_plot_to_group(plot_name, metric_group)
                    plt.close()
        
        # Plot 2: Colored by RNA Length
        if plot_length:
            length_valid = plot_df['rna_length_category'].notna()
            create_scatter_plot('rna_length_category', length_valid, length_color_map, 'length', 'RNA Length')
        
        # Plot 3: Colored by RNA Complexity
        if plot_complexity:
            complexity_valid = plot_df['rna_complexity_category'].notna()
            create_scatter_plot('rna_complexity_category', complexity_valid, complexity_color_map, 'complexity', 'RNA Complexity')
        
        # Plot 4: Colored by GC Content
        if plot_gc:
            gc_valid = plot_df['gc_content_category'].notna()
            create_scatter_plot('gc_content_category', gc_valid, gc_color_map, 'gc', 'RNA GC Content')
        
        # Plot 5: Colored by Experiment Type
        if plot_experiment:
            experiment_valid = plot_df['experiment_category'].notna()
            create_scatter_plot('experiment_category', experiment_valid, experiment_color_map, 'experiment', 'Experiment Type')
        
        # Plot 6: Colored by Deposited Date
        if plot_date:
            date_valid = plot_df['deposited_date_category'].notna()
            create_scatter_plot('deposited_date_category', date_valid, date_color_map, 'deposited_date', 'Deposited Date')
        
        # Plot 7: Colored by RNA Family
        if plot_rna_family:
            # Get unique RNA families and create color map
            unique_families = plot_df['rna_family_category'].dropna().unique()
            if len(unique_families) > 0:
                # Generate colors for each family - convert RGBA to RGB tuples
                color_array = plt.cm.tab20(np.linspace(0, 1, len(unique_families)))
                rna_family_color_map = {}
                for i, family in enumerate(unique_families):
                    # Convert RGBA array to RGB tuple
                    rgba = color_array[i]
                    rna_family_color_map[family] = tuple(rgba[:3])
                # Include 'None' if present (will be gray)
                if 'None' in unique_families:
                    rna_family_color_map['None'] = (0.5, 0.5, 0.5)  # Gray for None (RGB tuple)
                
                family_valid = plot_df['rna_family_category'].notna()
                create_scatter_plot('rna_family_category', family_valid, rna_family_color_map, 'rna_family', 'RNA Family')
        
        # # Plot 8: Colored by Domain Names
        # if plot_domain_names:
        #     # Get unique domain names and create color map
        #     unique_domains = plot_df['domain_names_category'].dropna().unique()
        #     if len(unique_domains) > 0:
        #         # Generate colors for each domain
        #         colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
        #         domain_names_color_map = {domain: colors[i] for i, domain in enumerate(unique_domains)}
        #         # Include 'None' if present (will be gray)
        #         if 'None' in unique_domains:
        #             domain_names_color_map['None'] = '#808080'  # Gray for None
        #
        #         domain_valid = plot_df['domain_names_category'].notna()
        #         create_scatter_plot('domain_names_category', domain_valid, domain_names_color_map, 'domain_names', 'Domain Names')

        print(f"Created scatter plots for {var1} vs {var2} (correlation: {corr_value:.3f})")


def main():
    """Main function to create correlation matrices"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate correlation matrices and scatter plots')
    parser.add_argument('--msa', action='store_true', help='Generate scatter plots colored by RNA MSA')
    parser.add_argument('--length', action='store_true', help='Generate scatter plots colored by RNA Length')
    parser.add_argument('--complexity', action='store_true', help='Generate scatter plots colored by RNA Complexity')
    parser.add_argument('--gc', action='store_true', help='Generate scatter plots colored by RNA GC Content')
    parser.add_argument('--experiment', action='store_true', help='Generate scatter plots colored by Experiment Type')
    parser.add_argument('--date', action='store_true', help='Generate scatter plots colored by Deposited Date')
    parser.add_argument('--rna_family', action='store_true', help='Generate scatter plots colored by RNA Family')
    parser.add_argument('--min_correlation', type=float, default=0.6, 
                       help='Minimum absolute correlation value to include in scatter plots (default: 0.6)')
    
    args = parser.parse_args()
    
    try:
        db = None
        print("\nCreating correlation matrix for all data...")
        db = get_correlation_matrices('all')
        
        print("\nCreating correlation matrices by RNA Length...")
        db = get_correlation_matrices('rna_length')
        
        print("\nCreating correlation matrices by RNA Family...")
        db = get_correlation_matrices('rna_family')
        
        print("\nCreating correlation matrices by Domain Names...")
        db = get_correlation_matrices('domain_names')
        
        print("\nCreating correlation matrices by RNA Sequence Complexity...")
        db = get_correlation_matrices('rna_complexity')
        
        print("\nCreating correlation matrices by GC Content...")
        db = get_correlation_matrices('gc_content')

        # print("\nCreating correlation matrices by AU Content...")
        # db = get_correlation_matrices('au_content')

        print("\nCreating correlation matrices by RNA MSA...")
        db = get_correlation_matrices('rna_msa')
        
        print("\nCreating correlation matrices by Complex Type...")
        db = get_correlation_matrices('complex_type')
        
        # Generate scatter plots for significant correlations (only if at least one flag is set)
        if args.msa or args.length or args.complexity or args.gc or args.experiment or args.date or args.rna_family:
            print("\nGenerating scatter plots for significant correlations...")
            # Define custom pairs to exclude from scatter plots
            # These are pairs that should not be considered for scatter plot generation
            # Example: pairs that are trivially correlated (e.g., GC and AU content are complementary)
            custom_pairs = [
                ('RNA_GC_Content', 'RNA_AU_Content'),
                # Add more custom pairs as needed based on METRIC_GROUPS or domain knowledge
                # Example: pairs from the same metric group that are expected to be correlated
            ]
            # Include all correlations with |correlation| >= min_correlation (not just top N)
            generate_correlation_scatter_plots(
                custom_pairs=custom_pairs, 
                top_n=None, 
                min_correlation=args.min_correlation,
                plot_msa=args.msa,
                plot_length=args.length,
                plot_complexity=args.complexity,
                plot_gc=args.gc,
                plot_experiment=args.experiment,
                plot_date=args.date,
                plot_rna_family=args.rna_family
            )
        else:
            print("\nNo scatter plot flags specified. Use --msa, --length, --complexity, --gc, --experiment, --date, --rna_family to generate scatter plots.")
        
    finally:
        if db:
            db.connection.close()


if __name__ == "__main__":
    main()