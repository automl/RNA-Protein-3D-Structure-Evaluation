import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import sys as _sys  # allow direct execution: python evaluation/<script>.py
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator
from utils.plots_dir import plots_root
import sys
import json
import numpy as np
import re
import argparse
from collections import OrderedDict
import ast


def safe_eval(val):
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}')):
            try:
                return ast.literal_eval(val)
            except Exception as e:
                print(f"Error in safe_eval: {e} for value: {val}")
                return None
        return val
    return val

def motif_ipTM_match_score(row):
    """
    For multi-chain complexes (NumberRNAs > 1), check if the chain with the largest experimental motif length
    matches the chain with the highest predicted RNA ipTM, based on sequence identity.
    Returns 1 if they match, otherwise returns (100 - percent difference)/100,
    where percent difference is relative to the highest predicted RNA ipTM.
    Returns np.nan for rows with NumberRNAs <= 1 or on error.
    """

    try:
        # Only process if NumberRNAs > 1
        if 'NumberRNAs' not in row or row['NumberRNAs'] <= 1:
            return np.nan

        # Use experimental columns
        motif_col = None
        for col in row.index:
            if 'RNAMotifLength_exp' in col and not pd.isna(row[col]):
                motif_col = col
                break
        if motif_col is None:
            return np.nan
        motif_lengths = safe_eval(row[motif_col]) if isinstance(row[motif_col], str) else row[motif_col]

        chain_pairs_col = None
        for col in row.index:
            if 'ChainIDpairList_proteinRNA_exp' in col and not pd.isna(row[col]):
                chain_pairs_col = col
                break
        if chain_pairs_col is None:
            return np.nan
        chain_pairs = safe_eval(row[chain_pairs_col]) if isinstance(row[chain_pairs_col], str) else row[chain_pairs_col]

        rna_chain_ids = safe_eval(row['RNAChainIDs']) if isinstance(row['RNAChainIDs'], str) else row['RNAChainIDs']
        exp_rna_seqs = safe_eval(row['RNASequence_exp']) if isinstance(row['RNASequence_exp'], str) else row['RNASequence_exp']
        pred_rna_seqs = safe_eval(row['RNASequence_pred']) if isinstance(row['RNASequence_pred'], str) else row['RNASequence_pred']
        af3_rna_iptm = safe_eval(row['af3_rna_ipTM']) if isinstance(row['af3_rna_ipTM'], str) else row['af3_rna_ipTM']
        protein_chain_ids = safe_eval(row['ProteinChainIDs']) if isinstance(row['ProteinChainIDs'], str) else row['ProteinChainIDs']

        # 1. Find index of max motif length
        motif_max_idx = motif_lengths.index(max(motif_lengths))

        # 2. Get the chain pair at that index
        chain_pair = chain_pairs[motif_max_idx]
        # 3. Find the RNA chain ID (not in protein chains)
        rna_chain_id = [cid for cid in chain_pair if cid not in protein_chain_ids][0]
        # 4. Find the index of this RNA chain in RNAChainIDs
        rna_chain_idx = rna_chain_ids.index(rna_chain_id)
        # 5. Get the corresponding sequence
        exp_seq = exp_rna_seqs[rna_chain_idx]
        # 6. Find the index of this sequence in pred_rna_seqs
        pred_seq_idx = pred_rna_seqs.index(exp_seq)
        # 7. Find the index of max ipTM
        iptm_max_idx = af3_rna_iptm.index(max(af3_rna_iptm))
        # 8. If pred_seq_idx matches iptm_max_idx, return 1
        if pred_seq_idx == iptm_max_idx:
            # print("Raw: 100.0 Normalized: 1.0")
            return 1.0
        # 9. Otherwise, return (100 - percent difference)/100, relative to max ipTM
        iptm_at_motif = af3_rna_iptm[pred_seq_idx]
        iptm_max = af3_rna_iptm[iptm_max_idx]
        if iptm_max == 0 or iptm_at_motif is None or iptm_max is None:
            return 0.0
        percent_diff = abs(iptm_max - iptm_at_motif) / abs(iptm_max) * 100
        normalized = (100 - percent_diff) / 100
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        return normalized
    except Exception as e:
        print(f"Error in motif_ipTM_match_score: {e}")
        return np.nan


class ResultPlotter:
    def __init__(self, database, config, msa_option=None, single_chain_only=False, rbp_template_option=None):
        """Initialize database connection and configuration"""
        self.database = database
        self.config = config
        self.msa_option = msa_option
        self.single_chain_only = single_chain_only
        self.rbp_template_option = rbp_template_option

        # Set up base results directory - remove MSA folder creation at root level
        self.results_dir = plots_root()   # <--folder>/plots when started by mainExecution.py

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Get data from database
        self.pred_df = self.get_predicted_data()
        self.exp_df = self.get_experimental_data()
        
        # Apply filters if specified
        if single_chain_only:
            # Filter based on ChainIDpairList_proteinRNA instead of ProteinChainIDs
            self.pred_df = self.pred_df[self.pred_df['ChainIDpairList_proteinRNA'].apply(self.is_single_chain)]
            # Filter experimental data to match
            self.exp_df = self.exp_df[self.exp_df['PDBId'].isin(self.pred_df['exp_db_id'])]
            
        if self.msa_option:
            self.pred_df = self.database.filter_by_msa(self.pred_df, self.msa_option)
            # Filter experimental data to match
            self.exp_df = self.exp_df[self.exp_df['PDBId'].isin(self.pred_df['exp_db_id'])]
        
        # Apply RBP template filter if specified
        if self.rbp_template_option:
            # Check if RBP_PDB_Templates column exists
            if 'RBP_PDB_Templates' in self.pred_df.columns:
                rbp_upper = self.rbp_template_option.upper()
                if rbp_upper == '+RBPTEMP' or rbp_upper == '+RBP':
                    # Has templates: not empty, not NULL, not '[]'
                    self.pred_df = self.pred_df[
                        (self.pred_df['RBP_PDB_Templates'].notna()) &
                        (self.pred_df['RBP_PDB_Templates'] != '') &
                        (self.pred_df['RBP_PDB_Templates'] != '[]') &
                        (self.pred_df['RBP_PDB_Templates'] != 'null')
                    ]
                elif rbp_upper == '-RBPTEMP' or rbp_upper == '-RBP':
                    # No templates: NULL, empty, or '[]'
                    self.pred_df = self.pred_df[
                        (self.pred_df['RBP_PDB_Templates'].isna()) |
                        (self.pred_df['RBP_PDB_Templates'] == '') |
                        (self.pred_df['RBP_PDB_Templates'] == '[]') |
                        (self.pred_df['RBP_PDB_Templates'] == 'null')
                    ]
            # Filter experimental data to match
            self.exp_df = self.exp_df[self.exp_df['PDBId'].isin(self.pred_df['exp_db_id'])]
            
        self.merged_df = self.merge_data()

    def get_predicted_data(self):
        """Get predicted data from database"""

        def get_msa_filter(msa_option):
            if msa_option == '+MSA':
                return "AND (af3_rna_MSA = '[1]' OR af3_rna_MSA LIKE '[1,1%')"
            elif msa_option == '-MSA':
                return "AND (af3_rna_MSA = '[0]' OR af3_rna_MSA LIKE '%0%')"
            return ""

        msa_filter = get_msa_filter(self.msa_option)

        if self.config.pred_table == 'pred_rna_rna':
            query = f"""
                    SELECT exp_db_id, FileName, 
                           af3_rna_pTM, af3_rna_ipTM,af3_rna_pLDDT_avg,
                           af3_global_pae_avg, af3_chain_pair_pae_min,
                           af3_fraction_disordered, af3_rna_MSA,
                           Free_energy, Binding_affinity_kd,
                           RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle,
                           RNALength, RNA_GC_Content, RNA_SequenceComplexity, RNA_Shannon_Entropy_K2, RNA_MFE_Value,
                           RNA_RMSD, RNA_TM, RNA_LDDT,
                           RNA_f1_score, RNA_precision, RNA_recall, RNA_mcc, RNA_wl
                    FROM {self.config.pred_table}
                    WHERE 1=1 {msa_filter}
                    """
        else:
            query = f"""
            SELECT exp_db_id, FileName, 
                   af3_protein_pTM, af3_rna_pTM, 
                   af3_protein_ipTM, af3_rna_ipTM,
                   af3_protein_pLDDT_avg, af3_rna_pLDDT_avg,
                   af3_global_pae_avg, af3_chain_pair_pae_min,
                   af3_fraction_disordered, af3_rna_MSA,
                   Free_energy, Binding_affinity_kd,
                   RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle,
                   ChainIDpairList_proteinRNA,
                   ProteinRNAInterfaceRatio as pred_interface_ratio, 
                   RNAmotif_score,
                   RNALength, ProteinLength,
                   RNA_GC_Content, RNA_SequenceComplexity, RNA_Shannon_Entropy_K2, RNA_MFE_Value,
                   RNA_RMSD, RNA_TM, RNA_TM_AllChains, Protein_TM, Complex_TM, Complex_RMSD, RNA_LDDT_allAtom, Complex_LDDT_allAtom,
                   RNA_f1_score, RNA_precision, RNA_recall, RNA_mcc, RNA_wl,
                   NumberRNAs, RNA_msa_size, RNAChainIDs as RNAChainIDs_pred,
                   RNASequence as RNASequence_pred, AAmatch_score, iLDDT, ICS, QS_global,
                   RBP_PDB_Templates
            FROM {self.config.pred_table}
            WHERE 1=1 {msa_filter}
            """
        return pd.read_sql_query(query, self.database.connection)

    def get_experimental_data(self):
        """Get experimental data from database"""
        query = f"""
        SELECT PDBId,
               Free_energy, Binding_affinity_kd,
               RNA_GlycosidicBond, RNA_SugarPucker, RNA_GammaAngle,
               ProteinRNAInterfaceRatio as exp_interface_ratio,
               RNAMotifLength as RNAMotifLength_exp, ChainIDpairList_proteinRNA as ChainIDpairList_proteinRNA_exp, RNAChainIDs, 
               RNASequence as RNASequence_exp, ProteinChainIDs
        FROM {self.config.ref_table}
        """
        return pd.read_sql_query(query, self.database.connection)

    def is_single_chain(self, chain_pairs):
        """Check if the structure has a single chain pair"""
        try:
            pairs = literal_eval(chain_pairs)
            return len(pairs) == 1
        except:
            return False

    def filter_chains(self, df, single_chain_only):
        """Filter data based on chain pairs"""
        if single_chain_only:
            return df[df['ChainIDpairList_proteinRNA'].apply(self.is_single_chain)]
        return df

    def merge_data(self):
        """Merge predicted and experimental data based on exp_db_id/PDBId"""
        return pd.merge(
            self.pred_df,
            self.exp_df,
            left_on='exp_db_id',
            right_on='PDBId',
            suffixes=('_pred', '_exp')
        )

    def get_max_value_from_list(self, value):
        """Extract maximum value from a string list or return float value"""
        try:
            if isinstance(value, str):
                # Handle list format - strip brackets and split
                value = value.strip('[]').replace(' ', '')
                if ',' in value:
                    values = [float(x) for x in value.split(',')]
                    return max(values)
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            return None
        except (ValueError, TypeError):
            return None

    def plot_af3_metrics(self, single_chain_only=False):
        """Create violin plots for AF3 metrics (excluding PAE) and scatter plots vs RMSD"""
        # Get filtered data first
        df = self.filter_data(self.pred_df, single_chain_only)

        # Create violin plot using PlotCreator with af3_metrics type
        plot_creator = PlotCreator('af3_metrics', self.msa_option, single_chain_only, self.rbp_template_option)

        # Add new plot for Chain-Pair PAE vs RNAmotif_score
        if not df['af3_chain_pair_pae_min'].isna().all() and not df['RNAmotif_score'].isna().all():
            # Handle empty strings and None values properly
            def process_pae(x):
                if pd.isna(x) or x == '':
                    return None
                try:
                    return float(x) if isinstance(x, str) else float(x)
                except (ValueError, TypeError):
                    return None

            pae_values = df['af3_chain_pair_pae_min'].apply(process_pae).dropna()
            valid_indices = pae_values.index.intersection(df.dropna(subset=['RNAmotif_score']).index)

            if len(valid_indices) > 0:
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    yAxis_score=df.loc[valid_indices, 'RNAmotif_score'],
                    yAxis_label='RNA Motif Score',
                    xAxis_label='Chain-Pair PAE',
                    name='chain_pair_pae_vs_motif_score',
                    xAxis_score=pae_values[valid_indices]
                )

        # Merge with experimental data for interface ratio comparison
        merged_df = pd.merge(
            df,
            self.exp_df[['PDBId', 'exp_interface_ratio']],
            left_on='exp_db_id',
            right_on='PDBId',
            how='left'
        )

        # Convert interface ratios to float
        merged_df['pred_interface_ratio'] = pd.to_numeric(merged_df['pred_interface_ratio'], errors='coerce')
        merged_df['exp_interface_ratio'] = pd.to_numeric(merged_df['exp_interface_ratio'], errors='coerce')

        # Calculate interface ratio difference
        merged_df['r_is_diff'] = ((merged_df['pred_interface_ratio'] - merged_df['exp_interface_ratio']) /
                                  merged_df['exp_interface_ratio'] * 100).round(2)

        # Add new plots for interface ratio difference
        # Plot Chain-Pair PAE vs r_is_diff
        if not merged_df['af3_chain_pair_pae_min'].isna().all() and not merged_df['r_is_diff'].isna().all():
            pae_values = merged_df['af3_chain_pair_pae_min'].apply(process_pae).dropna()
            valid_indices = pae_values.index.intersection(merged_df.dropna(subset=['r_is_diff']).index)

            if len(valid_indices) > 0:
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=merged_df.loc[valid_indices, 'r_is_diff'],
                    xAxis_label='Interface Ratio Difference (%)',
                    yAxis_label='Chain-Pair PAE',
                    name='chain_pair_pae_vs_r_is_diff',
                    yAxis_score=pae_values[valid_indices]
                )

        # Plot RNA ipTM vs r_is_diff
        if not merged_df['af3_rna_ipTM'].isna().all() and not merged_df['r_is_diff'].isna().all():
            def process_tm_values(tm_str, rna_length_str):
                if pd.isna(tm_str) or pd.isna(rna_length_str):
                    return None
                try:
                    # Convert TM string to list
                    if isinstance(tm_str, str):
                        tm_values = eval(tm_str)
                    else:
                        tm_values = [tm_str]
                    
                    # Convert RNA length string to list
                    if isinstance(rna_length_str, str):
                        if '[' in rna_length_str:
                            rna_lengths = eval(rna_length_str)
                        else:
                            rna_lengths = [int(rna_length_str)]
                    else:
                        rna_lengths = [int(rna_length_str)]
                    
                    # Find index of first occurrence of longest RNA
                    if len(rna_lengths) > 1:
                        max_length = max(rna_lengths)
                        max_length_idx = rna_lengths.index(max_length)
                        if max_length_idx < len(tm_values):
                            return tm_values[max_length_idx]
                        return None
                    return tm_values[0]
                except (ValueError, TypeError, SyntaxError, IndexError):
                    return None

            # Process both pTM and ipTM values
            merged_df['processed_pTM'] = merged_df.apply(
                lambda row: process_tm_values(row['af3_rna_pTM'], row['RNALength']), axis=1
            )
            merged_df['processed_ipTM'] = merged_df.apply(
                lambda row: process_tm_values(row['af3_rna_ipTM'], row['RNALength']), axis=1
            )

            # Plot ipTM vs r_is_diff
            iptm_values = merged_df['processed_ipTM'].dropna()
            valid_indices = iptm_values.index.intersection(merged_df.dropna(subset=['r_is_diff']).index)

            if len(valid_indices) > 0:
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=merged_df.loc[valid_indices, 'r_is_diff'],
                    xAxis_label='Interface Ratio Difference (%)',
                    yAxis_label='RNA ipTM',
                    name='rna_iptm_vs_r_is_diff',
                    yAxis_score=iptm_values[valid_indices]
                )

        # Get RMSD data from database with necessary columns for filtering
        rmsd_query = f"""
        SELECT exp_db_id, RNA_RMSD, af3_chain_pair_pae_min, 
               ChainIDpairList_proteinRNA, af3_rna_MSA, RBP_PDB_Templates
        FROM {self.config.pred_table}
        WHERE RNA_RMSD IS NOT NULL
        """
        rmsd_df = pd.read_sql_query(rmsd_query, self.database.connection)

        # Apply the same filtering to RMSD data
        rmsd_df = self.filter_data(rmsd_df, single_chain_only)

        if not rmsd_df.empty:
            # Get first RMSD value for each entry
            def extract_first_rmsd(x):
                if pd.isna(x) or x == '':
                    return None
                try:
                    if isinstance(x, str):
                        if '[' in x:
                            values = eval(x)
                            if isinstance(values, list):
                                return float(values[0])
                        return float(x)
                    return float(x)
                except (ValueError, TypeError, SyntaxError, IndexError):
                    return None

            # Get PAE value for each entry
            # def extract_pae(x):
            #     if pd.isna(x) or x == '':
            #         return None
            #     try:
            #         if isinstance(x, str):
            #             values = eval(x)
            #             return float(values)
            #         return float(x)
            #     except (ValueError, TypeError, SyntaxError):
            #         return None
            def extract_pae(x):
                if pd.isna(x) or x == '':
                    return None

                try:
                    # If it's already numeric, return as float
                    if isinstance(x, (int, float)):
                        return float(x)

                    if isinstance(x, str):
                        # Normalize JSON-like syntax to Python-safe format
                        x_clean = (
                            x.replace("null", "None")
                            .replace("true", "True")
                            .replace("false", "False")
                        )

                        # Try parsing safely without eval
                        try:
                            values = ast.literal_eval(x_clean)
                        except (ValueError, SyntaxError):
                            try:
                                values = json.loads(x)
                            except Exception:
                                return None

                        # If parsed successfully, try to extract a numeric value
                        if isinstance(values, (int, float)):
                            return float(values)
                        elif isinstance(values, (list, tuple)) and len(values) > 0:
                            # Example: return the mean of numeric entries
                            nums = [v for v in values if isinstance(v, (int, float))]
                            return float(sum(nums) / len(nums)) if nums else None
                        elif isinstance(values, dict):
                            # Example: return min or max value depending on your metric
                            nums = []
                            for v in values.values():
                                if isinstance(v, (list, tuple)):
                                    nums.extend([n for n in v if isinstance(n, (int, float))])
                                elif isinstance(v, (int, float)):
                                    nums.append(v)
                            return float(sum(nums) / len(nums)) if nums else None
                        else:
                            return None

                    return None

                except Exception as e:
                    print(f"extract_pae() warning for input: {x} -> {e}")
                    return None

            rmsd_df['RNA_RMSD_first'] = rmsd_df['RNA_RMSD'].apply(extract_first_rmsd)
            rmsd_df['maxMotif_chain_pair_pae'] = rmsd_df['af3_chain_pair_pae_min'].apply(extract_pae)

            # Merge with filtered main dataframe
            df = df.merge(rmsd_df[['exp_db_id', 'RNA_RMSD_first', 'maxMotif_chain_pair_pae']],
                          left_on='exp_db_id',
                          right_on='exp_db_id',
                          how='left')

            # Create scatter plots for AF3 metrics vs RMSD
            metrics_to_plot = {
                'af3_rna_pTM': 'RNA pTM',
                'af3_rna_ipTM': 'RNA ipTM',
                'af3_rna_pLDDT_avg': 'RNA pLDDT',
                'af3_global_pae_avg': 'Global PAE',
                'maxMotif_chain_pair_pae': 'Chain-Pair PAE'
            }

            for column, label in metrics_to_plot.items():
                # Get values, handling lists for pTM and ipTM
                if column in ['af3_rna_pTM', 'af3_rna_ipTM']:
                    values = df.apply(
                        lambda row: process_tm_values(row[column], row['RNALength']), 
                        axis=1
                    ).dropna()
                else:
                    values = df[column].dropna()

                # Create scatter plot only if we have both metrics and RMSD values
                valid_indices = values.index.intersection(df.dropna(subset=['RNA_RMSD_first']).index)
                if len(valid_indices) > 0:
                    # Convert values to float before comparison
                    rmsd_values = pd.to_numeric(df.loc[valid_indices, 'RNA_RMSD_first'], errors='coerce')
                    metric_values = pd.to_numeric(values[valid_indices], errors='coerce')
                    
                    # Add check for zero/negative values before log10
                    valid_mask = (rmsd_values > 0) & (metric_values > 0)

                    if valid_mask.any():
                        plot_creator.get_scatterplot(
                            table_source='af3_metrics',
                            xAxis_score=rmsd_values[valid_mask],
                            xAxis_label='RNA RMSD [Å]',
                            yAxis_label=label,
                            name=f'{column}_vs_RMSD',
                            yAxis_score=metric_values[valid_mask]
                        )

        # Helper to process metrics by length
        def process_metric_by_length(metric_str, length_str):
            if pd.isna(metric_str) or pd.isna(length_str):
                return None
            try:
                # Convert metric string to list
                if isinstance(metric_str, str):
                    metric_values = eval(metric_str)
                else:
                    metric_values = [metric_str]
                # Convert length string to list
                if isinstance(length_str, str):
                    if '[' in length_str:
                        lengths = eval(length_str)
                    else:
                        lengths = [int(length_str)]
                else:
                    lengths = [int(length_str)]
                # Find index of first occurrence of max length
                if len(lengths) > 1:
                    max_length = max(lengths)
                    max_length_idx = lengths.index(max_length)
                    if max_length_idx < len(metric_values):
                        return metric_values[max_length_idx]
                    return None
                return metric_values[0]
            except Exception:
                return None

        def _to_float_list(values):
            """Convert scalar/list-like metric value into a flat float list."""
            if pd.isna(values):
                return []
            try:
                parsed = safe_eval(values) if isinstance(values, str) else values
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    out = []
                    for v in parsed:
                        try:
                            out.append(float(v))
                        except Exception:
                            continue
                    return out
                return [float(parsed)]
            except Exception:
                return []

        def _tm_allchains_to_diag_list(values):
            """
            Convert RNA_TM_AllChains to comparable per-chain values.
            - [[a,b],[c,d]] -> [a,d] (diagonal, same chain index pairing)
            - [[x]] -> [x]
            - [x,y] -> [x,y]
            - x -> [x]
            """
            if pd.isna(values):
                return []
            try:
                parsed = safe_eval(values) if isinstance(values, str) else values
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    if len(parsed) == 0:
                        return []
                    if isinstance(parsed[0], (list, tuple, np.ndarray)):
                        diag_vals = []
                        for i, row in enumerate(parsed):
                            if not isinstance(row, (list, tuple, np.ndarray)):
                                continue
                            if i < len(row):
                                try:
                                    diag_vals.append(float(row[i]))
                                except Exception:
                                    continue
                        return diag_vals
                    return _to_float_list(parsed)
                return [float(parsed)]
            except Exception:
                return []

        def average_metric(metric_str):
            if pd.isna(metric_str):
                return np.nan
            try:
                if isinstance(metric_str, str):
                    if '[' in metric_str:
                        values = eval(metric_str)
                        return float(np.mean(values))
                    return float(metric_str)
                return float(metric_str)
            except Exception:
                return np.nan

        def max_metric(metric_str):
            if pd.isna(metric_str):
                return np.nan
            try:
                if isinstance(metric_str, str):
                    if '[' in metric_str:
                        values = eval(metric_str)
                        return float(np.max(values))
                    return float(metric_str)
                return float(metric_str)
            except Exception:
                return np.nan

        # Use OrderedDict to guarantee order
        metric_keys = [
            'Protein pTM', 'RNA pTM', 'Protein ipTM', 'RNA ipTM',
            'Protein pLDDT', 'RNA pLDDT', 'Fraction Disordered'
        ]
        data_dict = OrderedDict()
        data_dict['Protein pTM'] = df['af3_protein_pTM'].apply(average_metric) if 'af3_protein_pTM' in df.columns else pd.Series(dtype=float)
        data_dict['RNA pTM'] = df['af3_rna_pTM'].apply(average_metric) if 'af3_rna_pTM' in df.columns else pd.Series(dtype=float)
        data_dict['Protein ipTM'] = df['af3_protein_ipTM'].apply(max_metric) if 'af3_protein_ipTM' in df.columns else pd.Series(dtype=float)
        data_dict['RNA ipTM'] = df['af3_rna_ipTM'].apply(max_metric) if 'af3_rna_ipTM' in df.columns else pd.Series(dtype=float)
        data_dict['Protein pLDDT'] = pd.to_numeric(df['af3_protein_pLDDT_avg'], errors='coerce') if 'af3_protein_pLDDT_avg' in df.columns else pd.Series(dtype=float)
        data_dict['RNA pLDDT'] = pd.to_numeric(df['af3_rna_pLDDT_avg'], errors='coerce') if 'af3_rna_pLDDT_avg' in df.columns else pd.Series(dtype=float)
        data_dict['Fraction Disordered'] = pd.to_numeric(df['af3_fraction_disordered'], errors='coerce') if 'af3_fraction_disordered' in df.columns else pd.Series(dtype=float)

        # Ensure all metrics are present, even if empty
        for key in metric_keys:
            if key not in data_dict:
                data_dict[key] = pd.Series(dtype=float)

        data_values = [pd.to_numeric(vals, errors='coerce').dropna().reset_index(drop=True) for vals in data_dict.values()]
        labels = list(data_dict.keys())
        plot_creator.get_violin_plot(
            table_source='af3_metrics',
            data_values=data_values,
            labels=labels,
            name='af3_metrics',
            rotation=90
        )

        # Add scatter plot of RNA pTM vs RNA_TM_AllChains (index-matched pairs)
        rna_ptm_pairs = []
        rna_tm_pairs = []
        if 'af3_rna_pTM' in df.columns and 'RNA_TM_AllChains' in df.columns:
            for _, row in df.iterrows():
                ptm_values = _to_float_list(row['af3_rna_pTM'])
                tm_values = _tm_allchains_to_diag_list(row['RNA_TM_AllChains'])
                if not ptm_values or not tm_values:
                    continue
                n_pairs = min(len(ptm_values), len(tm_values))
                for i in range(n_pairs):
                    rna_ptm_pairs.append(ptm_values[i])
                    rna_tm_pairs.append(tm_values[i])
        if rna_ptm_pairs and rna_tm_pairs:
            plot_creator.get_scatterplot(
                table_source='af3_metrics',
                xAxis_score=rna_ptm_pairs,
                xAxis_label='RNA pTM',
                yAxis_label='RNA TM',
                name='rna_pTM_vs_rna_TM',
                yAxis_score=rna_tm_pairs,
                xAxis_limit=(0, 1),
                yAxis_limit=(0, 1)
            )

        # Add scatter plot of Protein pTM vs Protein_TM if available
        if 'af3_protein_pTM' in df.columns and 'Protein_TM' in df.columns and 'ProteinLength' in df.columns:
            protein_ptm = df.apply(lambda row: process_metric_by_length(row['af3_protein_pTM'], row['ProteinLength']) if (not pd.isna(row['af3_protein_pTM']) and not pd.isna(row['ProteinLength'])) else None, axis=1)
            protein_tm = pd.to_numeric(df['Protein_TM'], errors='coerce')
            # print("dsf", protein_ptm)
            # print("protein_tm", protein_tm)
            valid = (~protein_ptm.isna()) & (~protein_tm.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=protein_ptm[valid],
                    xAxis_label='Protein pTM',
                    yAxis_label='Protein TM',
                    name='protein_pTM_vs_protein_TM',
                    yAxis_score=protein_tm[valid],
                    xAxis_limit=(0, 1),
                    yAxis_limit=(0, 1)
                )

        # Add scatter plot of max af3_rna_ipTM vs RNAmotif_score if available
        if 'af3_rna_ipTM' in df.columns and 'RNAmotif_score' in df.columns:
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
            valid = (~rna_iptm_max.isna()) & (~motif_score.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    yAxis_score=rna_iptm_max[valid],
                    yAxis_label='max RNA ipTM',
                    xAxis_label='RNA motif score',
                    name='rna_ipTMmax_vs_rnamotif_score',
                    xAxis_score=motif_score[valid]
                )

        # Add scatter plot of rna_iptm_max vs Complex_TM if available
        if 'af3_rna_ipTM' in df.columns and 'Complex_TM' in df.columns:
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            complex_tm = pd.to_numeric(df['Complex_TM'], errors='coerce')
            valid = (~rna_iptm_max.isna()) & (~complex_tm.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_iptm_max[valid],
                    xAxis_label='max RNA ipTM',
                    yAxis_label='Complex TM',
                    name='rna_ipTMmax_vs_ComplexTM',
                    yAxis_score=complex_tm[valid],
                    xAxis_limit=(0, 1),
                    yAxis_limit=(0, 1)
                )

        # Add scatter plot of chain_pair_pae vs Complex_TM if available
        if 'af3_chain_pair_pae_min' in df.columns and 'Complex_TM' in df.columns:
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            complex_tm = pd.to_numeric(df['Complex_TM'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~complex_tm.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=chain_pair_pae[valid],
                    xAxis_label='Chain Pair PAE',
                    yAxis_label='Complex TM',
                    name='chain_pair_pae_vs_ComplexTM',
                    yAxis_score=complex_tm[valid],
                    yAxis_limit=(0, 1)
                )

        # Add scatter plot of rna_iptm_max vs iLDDT if available
        if 'af3_rna_ipTM' in df.columns and 'iLDDT' in df.columns:
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            def first_value(val):
                if pd.isna(val):
                    return np.nan
                try:
                    if isinstance(val, str) and '[' in val:
                        values = eval(val)
                        return float(values[0])
                    return float(val)
                except Exception:
                    return np.nan
            complex_rmsd = df['iLDDT'].apply(first_value)
            valid = (~rna_iptm_max.isna()) & (~complex_rmsd.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_iptm_max[valid],
                    xAxis_label='max RNA ipTM',
                    yAxis_label='iLDDT',
                    name='rna_ipTMmax_vs_iLDDT',
                    yAxis_score=complex_rmsd[valid]
                )

        # Add scatter plot of max RNA ipTM vs binding affinity if available
        if 'af3_rna_ipTM' in df.columns and 'Binding_affinity_kd' in df.columns:
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            def process_binding_affinity(val):
                if pd.isna(val):
                    return np.nan
                try:
                    if isinstance(val, str):
                        return float(val)
                    return float(val)
                except Exception:
                    return np.nan
            binding_affinity = df['Binding_affinity_kd'].apply(process_binding_affinity)
            valid = (~rna_iptm_max.isna()) & (~binding_affinity.isna())
            if valid.any():
                # Convert to log scale if values are very small
                if binding_affinity[valid].min() < 1e-10:
                    # Handle zero values before log conversion
                    non_zero_mask = binding_affinity > 0
                    binding_affinity_log = pd.Series(index=binding_affinity.index)
                    binding_affinity_log[non_zero_mask] = np.log10(binding_affinity[non_zero_mask])
                    binding_affinity_log[~non_zero_mask] = np.nan
                    valid = valid & non_zero_mask  # Update valid mask to exclude zeros
                    ylabel = 'Log₁₀[Binding Affinity (M)]'
                else:
                    binding_affinity_log = binding_affinity
                    ylabel = 'Binding Affinity (M)'

                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_iptm_max[valid],
                    xAxis_label='max RNA ipTM',
                    yAxis_label=ylabel,
                    name='rna_ipTMmax_vs_binding_affinity',
                    yAxis_score=binding_affinity_log[valid]
                )

        # Add scatter plot of Chain Pair PAE vs binding affinity if available
        if 'af3_chain_pair_pae_min' in df.columns and 'Binding_affinity_kd' in df.columns:
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            binding_affinity = df['Binding_affinity_kd'].apply(process_binding_affinity)
            valid = (~chain_pair_pae.isna()) & (~binding_affinity.isna())
            if valid.any():
                # Convert to log scale if values are very small
                if binding_affinity[valid].min() < 1e-10:
                    # Handle zero values before log conversion
                    non_zero_mask = binding_affinity > 0
                    binding_affinity_log = pd.Series(index=binding_affinity.index)
                    binding_affinity_log[non_zero_mask] = np.log10(binding_affinity[non_zero_mask])
                    binding_affinity_log[~non_zero_mask] = np.nan
                    valid = valid & non_zero_mask  # Update valid mask to exclude zeros
                    ylabel = 'Log₁₀[Binding Affinity (M)]'
                else:
                    binding_affinity_log = binding_affinity
                    ylabel = 'Binding Affinity (M)'

                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=chain_pair_pae[valid],
                    xAxis_label='Chain Pair PAE',
                    yAxis_label=ylabel,
                    name='chain_pair_pae_vs_binding_affinity',
                    yAxis_score=binding_affinity_log[valid]
                )

        # Add scatter plot of RNA pLDDT vs RNA LDDT if available
        if 'af3_rna_pLDDT_avg' in df.columns and 'RNA_LDDT_allAtom' in df.columns:
            rna_plddt = pd.to_numeric(df['af3_rna_pLDDT_avg'], errors='coerce')
            rna_lddt = pd.to_numeric(df['RNA_LDDT_allAtom'], errors='coerce')
            valid = (~rna_plddt.isna()) & (~rna_lddt.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_plddt[valid],
                    xAxis_label='RNA pLDDT',
                    yAxis_label='RNA LDDT',
                    name='rna_plddt_vs_rna_lddt',
                    yAxis_score=rna_lddt[valid],
                    xAxis_limit=(0, 1),
                    yAxis_limit=(0, 1)
                )

        # Add MSA color-coded scatter plot of RNA pLDDT vs RNA LDDT if available
        # Create separate plot_creator for MSA plots (no filters) so filename doesn't include filter suffixes
        plot_creator_msa = PlotCreator('af3_metrics', None, single_chain_only, None)
        if all(col in df.columns for col in ['af3_rna_pLDDT_avg', 'RNA_LDDT_allAtom', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    # Ensure it's a list
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    # Ensure it's a list
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    # If only one chain, just use its MSA value
                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        # Find index of longest RNA
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            msa_status = df.apply(classify_msa, axis=1)
            rna_plddt = pd.to_numeric(df['af3_rna_pLDDT_avg'], errors='coerce')
            rna_lddt = pd.to_numeric(df['RNA_LDDT_allAtom'], errors='coerce')
            valid = (~rna_plddt.isna()) & (~rna_lddt.isna()) & (~msa_status.isna())
            
            if valid.any():
                color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                plot_creator_msa.get_msa_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_plddt[valid],
                    xAxis_label='RNA pLDDT',
                    name='rna_plddt_vs_rna_lddt_all_chains_msa',
                    yAxis_label='RNA LDDT',
                    yAxis_score=rna_lddt[valid],
                    color_by=msa_status[valid],
                    color_map=color_map
                )
            
            # Add RBP template color-coded scatter plot of RNA pLDDT vs RNA LDDT if available
            # Create separate plot_creator for RBP plots (no filters) so filename doesn't include filter suffixes
            plot_creator_rbp = PlotCreator('af3_metrics', None, single_chain_only, None)
            if 'RBP_PDB_Templates' in df.columns:
                def classify_rbp_template(row):
                    try:
                        rbp_templates = row['RBP_PDB_Templates']
                        if pd.isna(rbp_templates) or rbp_templates == '' or rbp_templates == '[]' or rbp_templates == 'null':
                            return '-RBP'
                        # Check if it's a string that needs parsing
                        if isinstance(rbp_templates, str):
                            try:
                                parsed = safe_eval(rbp_templates)
                                if isinstance(parsed, list):
                                    return '+RBP' if len(parsed) > 0 else '-RBP'
                            except:
                                return '-RBP'
                        # If it's already a list
                        if isinstance(rbp_templates, list):
                            return '+RBP' if len(rbp_templates) > 0 else '-RBP'
                        return '+RBP'
                    except Exception:
                        return np.nan
                
                rbp_status = df.apply(classify_rbp_template, axis=1)
                rna_plddt_rbp = pd.to_numeric(df['af3_rna_pLDDT_avg'], errors='coerce')
                rna_lddt_rbp = pd.to_numeric(df['RNA_LDDT_allAtom'], errors='coerce')
                valid_rbp = (~rna_plddt_rbp.isna()) & (~rna_lddt_rbp.isna()) & (~rbp_status.isna())
                
                if valid_rbp.any():
                    color_map_rbp = {'+RBP': 'green', '-RBP': 'red'}
                    plot_creator_rbp.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=rna_plddt_rbp[valid_rbp],
                        xAxis_label='RNA pLDDT',
                        name='rna_plddt_vs_rna_lddt_all_chains_rbp',
                        yAxis_label='RNA LDDT',
                        yAxis_score=rna_lddt_rbp[valid_rbp],
                        color_by=rbp_status[valid_rbp],
                        color_map=color_map_rbp
                )


        # Add MSA color-coded scatter plot for Global PAE vs RMSD
        # Create separate plot_creator for MSA plots (no filters) so filename doesn't include filter suffixes
        plot_creator_msa = PlotCreator('af3_metrics', None, single_chain_only, None)
        if all(col in df.columns for col in
               ['af3_global_pae_avg', 'Complex_RMSD', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    # Ensure it's a list
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    # Ensure it's a list
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    # If only one chain, just use its MSA value
                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        # Find index of longest RNA
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(
                        f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            def extract_first_rmsd(x):
                if pd.isna(x) or x == '':
                    return None
                try:
                    if isinstance(x, str):
                        if '[' in x:
                            values = eval(x)
                            if isinstance(values, list):
                                return float(values[0])
                        return float(x)
                    return float(x)
                except (ValueError, TypeError, SyntaxError, IndexError):
                    return None

            msa_status = df.apply(classify_msa, axis=1)
            global_pae = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
            rmsd_values = df['Complex_RMSD'].apply(extract_first_rmsd)
            valid = (~global_pae.isna()) & (~rmsd_values.isna()) & (~msa_status.isna())

        if valid.any():
            color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
            plot_creator_msa.get_msa_scatterplot(
                table_source='af3_metrics',
                xAxis_score=rmsd_values[valid],
                xAxis_label='Complex RMSD [Å]',
                name='af3_global_pae_avg_vs_cRMSD_all_chains_msa',
                yAxis_label='Global PAE',
                yAxis_score=global_pae[valid],
                color_by=msa_status[valid],
                color_map=color_map
            )
            
            # Add MSA plot for RNA RMSD
            if 'RNA_RMSD' in df.columns:
                rna_rmsd_values_msa = df['RNA_RMSD'].apply(extract_first_rmsd)
                valid_rna_msa = (~global_pae.isna()) & (~rna_rmsd_values_msa.isna()) & (~msa_status.isna())
                
                if valid_rna_msa.any():
                    plot_creator_msa.get_msa_scatterplot(
                            table_source='af3_metrics',
                            xAxis_score=rna_rmsd_values_msa[valid_rna_msa],
                            xAxis_label='RNA RMSD [Å]',
                            name='af3_global_pae_avg_vs_rRMSD_all_chains_msa',
                            yAxis_label='Global PAE',
                            yAxis_score=global_pae[valid_rna_msa],
                            color_by=msa_status[valid_rna_msa],
                            color_map=color_map
                        )
            
            # Add RBP template color-coded scatter plot for Global PAE vs RMSD
            # Create separate plot_creator for RBP plots (no filters) so filename doesn't include filter suffixes
            plot_creator_rbp = PlotCreator('af3_metrics', None, single_chain_only, None)
            if 'RBP_PDB_Templates' in df.columns:
                def classify_rbp_template(row):
                    try:
                        rbp_templates = row['RBP_PDB_Templates']
                        if pd.isna(rbp_templates) or rbp_templates == '' or rbp_templates == '[]' or rbp_templates == 'null':
                            return '-RBP'
                        if isinstance(rbp_templates, str):
                            try:
                                parsed = safe_eval(rbp_templates)
                                if isinstance(parsed, list):
                                    return '+RBP' if len(parsed) > 0 else '-RBP'
                            except:
                                return '-RBP'
                        if isinstance(rbp_templates, list):
                            return '+RBP' if len(rbp_templates) > 0 else '-RBP'
                        return '+RBP'
                    except Exception:
                        return np.nan
                
                rbp_status = df.apply(classify_rbp_template, axis=1)
                global_pae_rbp = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
                rmsd_values_rbp = df['Complex_RMSD'].apply(extract_first_rmsd)
                valid_rbp = (~global_pae_rbp.isna()) & (~rmsd_values_rbp.isna()) & (~rbp_status.isna())
                
                if valid_rbp.any():
                    color_map_rbp = {'+RBP': 'green', '-RBP': 'red'}
                    plot_creator_rbp.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=rmsd_values_rbp[valid_rbp],
                        xAxis_label='Complex RMSD [Å]',
                        name='af3_global_pae_avg_vs_cRMSD_all_chains_rbp',
                        yAxis_label='Global PAE',
                        yAxis_score=global_pae_rbp[valid_rbp],
                        color_by=rbp_status[valid_rbp],
                        color_map=color_map_rbp
                    )
                    
                    # Add Global PAE vs RNA RMSD plot (not Complex RMSD)
                    if 'RNA_RMSD' in df.columns:
                        rna_rmsd_values = df['RNA_RMSD'].apply(extract_first_rmsd)
                        valid_rna_rmsd = (~global_pae_rbp.isna()) & (~rna_rmsd_values.isna()) & (~rbp_status.isna())
                        
                        if valid_rna_rmsd.any():
                            color_map_rbp_rna = {'+RBP': 'green', '-RBP': 'red'}
                            plot_creator_rbp.get_msa_scatterplot(
                                table_source='af3_metrics',
                                xAxis_score=rna_rmsd_values[valid_rna_rmsd],
                                xAxis_label='RNA RMSD [Å]',
                                name='af3_global_pae_avg_vs_rRMSD_all_chains_rbp',
                                yAxis_label='Global PAE',
                                yAxis_score=global_pae_rbp[valid_rna_rmsd],
                                color_by=rbp_status[valid_rna_rmsd],
                                color_map=color_map_rbp_rna
            )

            # Add MSA color-coded scatter plot for Global PAE vs RNAmotif_score
            if all(col in df.columns for col in
                   ['af3_global_pae_avg', 'RNAmotif_score', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
                def classify_msa(row):
                    try:
                        msa_dict = row['RNA_msa_size']
                        if isinstance(msa_dict, str):
                            msa_dict = safe_eval(msa_dict)
                        if not isinstance(msa_dict, dict):
                            return np.nan

                        chain_ids = row['RNAChainIDs_pred']
                        if isinstance(chain_ids, str):
                            chain_ids = safe_eval(chain_ids)
                        # Ensure it's a list
                        if not isinstance(chain_ids, (list, tuple)):
                            chain_ids = [chain_ids]

                        rna_lengths = row['RNALength']
                        if isinstance(rna_lengths, str):
                            rna_lengths = safe_eval(rna_lengths)
                        # Ensure it's a list
                        if not isinstance(rna_lengths, (list, tuple)):
                            rna_lengths = [rna_lengths]

                        # If only one chain, just use its MSA value
                        if len(msa_dict) == 1:
                            msa_val = list(msa_dict.values())[0]
                        else:
                            # Find index of longest RNA
                            max_len = max(rna_lengths)
                            max_idx = rna_lengths.index(max_len)
                            chain_id = chain_ids[max_idx]
                            msa_val = msa_dict.get(str(chain_id), 1)

                        return '+rMSA' if msa_val > 1 else '-rMSA'
                    except Exception as e:
                        print(
                            f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                        return np.nan

                msa_status = df.apply(classify_msa, axis=1)
                global_pae = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
                motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
                valid = (~global_pae.isna()) & (~motif_score.isna()) & (~msa_status.isna())

                if valid.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=global_pae[valid],
                        xAxis_label='Global PAE',
                        name='af3_global_pae_avg_vs_RNAmotif_score_all_chains',
                        yAxis_label='RNA Motif Score',
                        yAxis_score=motif_score[valid],
                        color_by=msa_status[valid],
                        color_map=color_map
                    )

            if all(col in df.columns for col in
                   ['RNA_TM', 'RNA_GC_Content', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
                def classify_msa(row):
                    try:
                        msa_dict = row['RNA_msa_size']
                        if isinstance(msa_dict, str):
                            msa_dict = safe_eval(msa_dict)
                        if not isinstance(msa_dict, dict):
                            return np.nan

                        chain_ids = row['RNAChainIDs_pred']
                        if isinstance(chain_ids, str):
                            chain_ids = safe_eval(chain_ids)
                        # Ensure it's a list
                        if not isinstance(chain_ids, (list, tuple)):
                            chain_ids = [chain_ids]

                        rna_lengths = row['RNALength']
                        if isinstance(rna_lengths, str):
                            rna_lengths = safe_eval(rna_lengths)
                        # Ensure it's a list
                        if not isinstance(rna_lengths, (list, tuple)):
                            rna_lengths = [rna_lengths]

                        # If only one chain, just use its MSA value
                        if len(msa_dict) == 1:
                            msa_val = list(msa_dict.values())[0]
                        else:
                            # Find index of longest RNA
                            max_len = max(rna_lengths)
                            max_idx = rna_lengths.index(max_len)
                            chain_id = chain_ids[max_idx]
                            msa_val = msa_dict.get(str(chain_id), 1)

                        return '+rMSA' if msa_val > 1 else '-rMSA'
                    except Exception as e:
                        print(
                            f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                        return np.nan

                msa_status = df.apply(classify_msa, axis=1)
                gc_content = pd.to_numeric(df['RNA_GC_Content'], errors='coerce')
                tm_score = pd.to_numeric(df['RNA_TM'], errors='coerce')
                valid = (~gc_content.isna()) & (~tm_score.isna()) & (~msa_status.isna())

                if valid.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=gc_content[valid],
                        xAxis_label='RNA GC-content',
                        name='GC_tm',
                        yAxis_label='RNA TM',
                        yAxis_score=tm_score[valid],
                        color_by=msa_status[valid],
                        color_map=color_map
                    )

                lddt_score = pd.to_numeric(df['RNA_LDDT_allAtom'], errors='coerce')
                valid2 = (~gc_content.isna()) & (~lddt_score.isna()) & (~msa_status.isna())

                if valid2.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=gc_content[valid2],
                        xAxis_label='RNA GC-content',
                        name='GC_lddt',
                        yAxis_label='RNA LDDT',
                        yAxis_score=lddt_score[valid2],
                        color_by=msa_status[valid2],
                        color_map=color_map
                    )

                seq_complex = pd.to_numeric(df['RNA_SequenceComplexity'], errors='coerce')
                valid3 = (~seq_complex.isna()) & (~tm_score.isna()) & (~msa_status.isna())
                valid4 = (~seq_complex.isna()) & (~lddt_score.isna()) & (~msa_status.isna())

                if valid3.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=seq_complex[valid3],
                        xAxis_label='RNA Sequence Complexity',
                        name='seq_complex_tm',
                        yAxis_label='RNA TM',
                        yAxis_score=tm_score[valid3],
                        color_by=msa_status[valid3],
                        color_map=color_map
                    )

                if valid4.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=seq_complex[valid4],
                        xAxis_label='RNA Sequence Complexity',
                        name='seq_complex_lddt',
                        yAxis_label='RNA LDDT',
                        yAxis_score=lddt_score[valid4],
                        color_by=msa_status[valid4],
                        color_map=color_map
                    )

                diversity = pd.to_numeric(df['RNA_Shannon_Entropy_K2'], errors='coerce')
                valid3 = (~diversity.isna()) & (~tm_score.isna()) & (~msa_status.isna())
                valid4 = (~diversity.isna()) & (~lddt_score.isna()) & (~msa_status.isna())

                if valid3.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=diversity[valid3],
                        xAxis_label='RNA Shannon Entropy K2',
                        name='rna_shannon_tm',
                        yAxis_label='RNA TM',
                        yAxis_score=tm_score[valid3],
                        color_by=msa_status[valid3],
                        color_map=color_map
                    )

                if valid4.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=diversity[valid4],
                        xAxis_label='RNA Shannon Entropy K2',
                        name='rna_shannon_lddt',
                        yAxis_label='RNA LDDT',
                        yAxis_score=lddt_score[valid4],
                        color_by=msa_status[valid4],
                        color_map=color_map
                    )

                mfe = pd.to_numeric(df['RNA_MFE_Value'], errors='coerce')
                valid3 = (~mfe.isna()) & (~tm_score.isna()) & (~msa_status.isna())
                valid4 = (~mfe.isna()) & (~lddt_score.isna()) & (~msa_status.isna())

                if valid3.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=mfe[valid3],
                        xAxis_label='RNA MFE',
                        name='rna_mfe_tm',
                        yAxis_label='RNA TM',
                        yAxis_score=tm_score[valid3],
                        color_by=msa_status[valid3],
                        color_map=color_map
                    )

                if valid4.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=mfe[valid4],
                        xAxis_label='RNA MFE',
                        name='rna_mfe_lddt',
                        yAxis_label='RNA LDDT',
                        yAxis_score=lddt_score[valid4],
                        color_by=msa_status[valid4],
                        color_map=color_map
                    )

            # Add MSA color-coded scatter plot for Global PAE vs AAmatch_score
            if all(col in df.columns for col in
                   ['af3_global_pae_avg', 'AAmatch_score', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
                def classify_msa(row):
                    try:
                        msa_dict = row['RNA_msa_size']
                        if isinstance(msa_dict, str):
                            msa_dict = safe_eval(msa_dict)
                        if not isinstance(msa_dict, dict):
                            return np.nan

                        chain_ids = row['RNAChainIDs_pred']
                        if isinstance(chain_ids, str):
                            chain_ids = safe_eval(chain_ids)
                        # Ensure it's a list
                        if not isinstance(chain_ids, (list, tuple)):
                            chain_ids = [chain_ids]

                        rna_lengths = row['RNALength']
                        if isinstance(rna_lengths, str):
                            rna_lengths = safe_eval(rna_lengths)
                        # Ensure it's a list
                        if not isinstance(rna_lengths, (list, tuple)):
                            rna_lengths = [rna_lengths]

                        # If only one chain, just use its MSA value
                        if len(msa_dict) == 1:
                            msa_val = list(msa_dict.values())[0]
                        else:
                            # Find index of longest RNA
                            max_len = max(rna_lengths)
                            max_idx = rna_lengths.index(max_len)
                            chain_id = chain_ids[max_idx]
                            msa_val = msa_dict.get(str(chain_id), 1)

                        return '+rMSA' if msa_val > 1 else '-rMSA'
                    except Exception as e:
                        print(
                            f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                        return np.nan

                msa_status = df.apply(classify_msa, axis=1)
                global_pae = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
                aa_match_score = pd.to_numeric(df['AAmatch_score'], errors='coerce')
                valid = (~global_pae.isna()) & (~aa_match_score.isna()) & (~msa_status.isna())

                if valid.any():
                    color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                    plot_creator.get_msa_scatterplot(
                        table_source='af3_metrics',
                        xAxis_score=global_pae[valid],
                        xAxis_label='Global PAE',
                        name='af3_global_pae_avg_vs_AAmatch_score_all_chains',
                        yAxis_label='AA Match Score',
                        yAxis_score=aa_match_score[valid],
                        color_by=msa_status[valid],
                        color_map=color_map
                    )

        # Add scatter plot of chain_pair_pae vs Complex_RMSD if available
        if 'af3_chain_pair_pae_min' in df.columns and 'Complex_RMSD' in df.columns:
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            def first_value(val):
                if pd.isna(val):
                    return np.nan
                try:
                    if isinstance(val, str) and '[' in val:
                        values = eval(val)
                        return float(values[0])
                    return float(val)
                except Exception:
                    return np.nan
            complex_rmsd = df['Complex_RMSD'].apply(first_value)
            valid = (~chain_pair_pae.isna()) & (~complex_rmsd.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=chain_pair_pae[valid],
                    xAxis_label='Chain Pair PAE',
                    yAxis_label='Complex RMSD',
                    name='chain_pair_pae_vs_ComplexRMSD',
                    yAxis_score=complex_rmsd[valid]
                )

        # Add scatter plot of Fraction Disordered vs RNAmotif_score if available
        if 'af3_fraction_disordered' in df.columns and 'RNAmotif_score' in df.columns:
            fraction_disordered = pd.to_numeric(df['af3_fraction_disordered'], errors='coerce')
            motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
            valid = (~fraction_disordered.isna()) & (~motif_score.isna())
            if valid.any():
                plot_creator.get_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=fraction_disordered[valid],
                    xAxis_label='Fraction Disordered',
                    yAxis_label='RNA motif score',
                    name='fraction_disordered_vs_rnamotif_score',
                    yAxis_score=motif_score[valid]
                )

        # Add 3D scatter plot: Chain Pair PAE, max RNA ipTM, RNAmotif_score
        if 'af3_chain_pair_pae_min' in df.columns and 'af3_rna_ipTM' in df.columns and 'RNAmotif_score' in df.columns:
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~rna_iptm_max.isna()) & (~motif_score.isna())
            if valid.any():
                plot_creator.get_3d_scatterplotA(
                    table_source='af3_metrics',
                    xAxis_score=chain_pair_pae[valid],
                    yAxis_score=rna_iptm_max[valid],
                    zAxis_score=motif_score[valid],
                    xAxis_label='Chain Pair PAE (smaller=better)',
                    yAxis_label='max RNA ipTM (bigger=better)',
                    zAxis_label='RNA motif score',
                    name='chain_pair_pae_rna_ipTMmax_vs_rnamotif_score_3d',
                )

        # Add MSA color-coded 3D scatter plot: Chain Pair PAE, max RNA ipTM, RNAmotif_score
        if all(col in df.columns for col in ['af3_chain_pair_pae_min', 'af3_rna_ipTM', 'RNAmotif_score', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    # Ensure it's a list
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    # Ensure it's a list
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    # If only one chain, just use its MSA value
                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        # Find index of longest RNA
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            msa_status = df.apply(classify_msa, axis=1)
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~rna_iptm_max.isna()) & (~motif_score.isna()) & (~msa_status.isna())
            
            if valid.any():
                # Create 3D scatter plot with MSA color coding
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Separate data by MSA status
                msa_plus_mask = msa_status[valid] == '+rMSA'
                msa_minus_mask = msa_status[valid] == '-rMSA'
                
                # Plot +MSA points in blue
                if msa_plus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_plus_mask], 
                              rna_iptm_max[valid][msa_plus_mask], 
                              motif_score[valid][msa_plus_mask], 
                              c='blue', label='+rMSA', alpha=0.7, s=50)
                
                # Plot -MSA points in orange
                if msa_minus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_minus_mask], 
                              rna_iptm_max[valid][msa_minus_mask], 
                              motif_score[valid][msa_minus_mask], 
                              c='orange', label='-rMSA', alpha=0.7, s=50)
                
                ax.set_xlabel('Chain Pair PAE (smaller=better)')
                ax.set_ylabel('max RNA ipTM (bigger=better)')
                ax.set_zlabel('RNA motif score')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_creator.results_dir, 'chain_pair_pae_rna_ipTMmax_vs_rnamotif_score_3d_msa.png'))
                plt.close()

        # Add 3D scatter plot: Chain Pair PAE, max RNA ipTM, Complex_TM
        if 'af3_chain_pair_pae_min' in df.columns and 'af3_rna_ipTM' in df.columns and 'Complex_TM' in df.columns:
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            complex_tm = pd.to_numeric(df['Complex_TM'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~rna_iptm_max.isna()) & (~complex_tm.isna())
            if valid.any():
                plot_creator.get_3d_scatterplotA(
                    table_source='af3_metrics',
                    xAxis_score=chain_pair_pae[valid],
                    yAxis_score=rna_iptm_max[valid],
                    zAxis_score=complex_tm[valid],
                    xAxis_label='Chain Pair PAE (smaller=better)',
                    yAxis_label='max RNA ipTM (bigger=better)',
                    zAxis_label='Complex TM',
                    name='chain_pair_pae_rna_ipTMmax_vs_ComplexTM_3d',
                )

        # Add MSA color-coded 3D scatter plot: Chain Pair PAE, max RNA ipTM, Complex_TM
        if all(col in df.columns for col in ['af3_chain_pair_pae_min', 'af3_rna_ipTM', 'Complex_TM', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    # Ensure it's a list
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    # Ensure it's a list
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    # If only one chain, just use its MSA value
                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        # Find index of longest RNA
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            msa_status = df.apply(classify_msa, axis=1)
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            rna_iptm_max = df['af3_rna_ipTM'].apply(max_metric)
            complex_tm = pd.to_numeric(df['Complex_TM'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~rna_iptm_max.isna()) & (~complex_tm.isna()) & (~msa_status.isna())
            
            if valid.any():
                # Create 3D scatter plot with MSA color coding
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Separate data by MSA status
                msa_plus_mask = msa_status[valid] == '+rMSA'
                msa_minus_mask = msa_status[valid] == '-rMSA'
                
                # Plot +MSA points in blue
                if msa_plus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_plus_mask], 
                              rna_iptm_max[valid][msa_plus_mask], 
                              complex_tm[valid][msa_plus_mask], 
                              c='blue', label='+rMSA', alpha=0.7, s=50)
                
                # Plot -MSA points in orange
                if msa_minus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_minus_mask], 
                              rna_iptm_max[valid][msa_minus_mask], 
                              complex_tm[valid][msa_minus_mask], 
                              c='orange', label='-rMSA', alpha=0.7, s=50)
                
                ax.set_xlabel('Chain Pair PAE (smaller=better)')
                ax.set_ylabel('max RNA ipTM (bigger=better)')
                ax.set_zlabel('Complex TM')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_creator.results_dir, 'chain_pair_pae_rna_ipTMmax_vs_ComplexTM_3d_msa.png'))
                plt.close()

        # Plot Motif ipTM Match Score for multi-chain complexes if possible
        # if all(col in df.columns for col in ['NumberRNAs','RNAMotifLength','ChainIDpairList_proteinRNA_exp','RNAChainIDs','RNASequence_exp','RNASequence_pred','af3_rna_ipTM']):
        multi_chain_df = self.merged_df[self.merged_df['NumberRNAs'] > 1]
        # print(multi_chain_df)
        if not multi_chain_df.empty:
            motif_scores = multi_chain_df.apply(motif_ipTM_match_score, axis=1).dropna()
            n_perfect = (motif_scores == 1.0).sum()
            n_imperfect = (motif_scores != 1.0).sum()
            data = {
                'Perfect Match (1.0)': n_perfect,
                'Imperfect (<1.0)': n_imperfect
            }
            plot_creator.get_barplot(
                table_source='af3_metrics',
                data=data,
                title='Motif ipTM Match Score Distribution',
                xlabel='Motif ipTM Match',
                ylabel='Count',
                bar_color='#FF8F00',
                rotation=0
            )


        multi_chain_df = self.merged_df[self.merged_df['NumberRNAs'] > 1]
        if not multi_chain_df.empty:
            motif_scores = multi_chain_df.apply(motif_ipTM_match_score, axis=1)

            # Helper for max ipTM
            def get_max_iptm(val):
                if isinstance(val, str):
                    try:
                        vals = ast.literal_eval(val)
                    except Exception:
                        return np.nan
                else:
                    vals = val
                if isinstance(vals, (list, tuple, np.ndarray)):
                    return np.nanmax([float(x) for x in vals if x is not None])
                elif vals is not None:
                    return float(vals)
                return np.nan

            # Helper for chain_pair_pae_min (single value per row)
            def get_chain_pair_pae(val):
                if isinstance(val, str):
                    try:
                        return float(val)
                    except Exception:
                        return np.nan
                elif val is not None:
                    return float(val)
                return np.nan

            # Grouping
            perfect_mask = motif_scores == 1.0
            imperfect_mask = motif_scores != 1.0

            # --- Plot 1: max af3_rna_ipTM ---
            max_iptm = multi_chain_df['af3_rna_ipTM'].apply(get_max_iptm)
            perfect_iptm = max_iptm[perfect_mask].dropna()
            imperfect_iptm = max_iptm[imperfect_mask].dropna()
            
            # Filter out empty arrays and build data/labels lists
            data_iptm = []
            labels_iptm = []
            percentages_iptm = []
            
            if not perfect_iptm.empty:
                data_iptm.append(perfect_iptm)
                labels_iptm.append('Perfect Match (1.0)')
            n_total = len(motif_scores.dropna())
            n_perfect = perfect_mask.sum()
            percent_perfect = 100 * n_perfect / n_total if n_total else 0
            percentages_iptm.append(percent_perfect)
            
            if not imperfect_iptm.empty:
                data_iptm.append(imperfect_iptm)
                labels_iptm.append('Imperfect (<1.0)')
                n_total = len(motif_scores.dropna())
                n_imperfect = imperfect_mask.sum()
                percent_imperfect = 100 * n_imperfect / n_total if n_total else 0
                percentages_iptm.append(percent_imperfect)

            # Only plot if we have data
            if len(data_iptm) > 0:
                # Violin plot with annotation below x-ticks
                fig, ax = plt.subplots(figsize=(8, 6))
                parts = ax.violinplot(data_iptm, showmeans=False, showmedians=True)
                ax.set_xticks(range(1, len(labels_iptm) + 1))
                ax.set_xticklabels(labels_iptm)
                ax.set_ylabel('max af3_rna_ipTM')
                ax.set_title('max af3_rna_ipTM by Motif Match')
                # Annotate percentages below x-ticks
                for i, pct in enumerate(percentages_iptm):
                    ax.text(i + 1, -0.08, f'{pct:.1f}%', ha='center', va='top', fontsize=12, color='black',
                            transform=ax.get_xaxis_transform())
                plt.tight_layout()
                plt.savefig(os.path.join(plot_creator.results_dir, 'max_af3_rna_ipTM_by_motif_match.png'))
                plt.close()

            # --- Plot 2: af3_chain_pair_pae_min ---
            chain_pair_pae = multi_chain_df['af3_chain_pair_pae_min'].apply(get_chain_pair_pae)
            perfect_pae = chain_pair_pae[perfect_mask].dropna()
            imperfect_pae = chain_pair_pae[imperfect_mask].dropna()
            
            # Filter out empty arrays and build data/labels lists
            data_pae = []
            labels_pae = []
            percentages_pae = []
            
            if not perfect_pae.empty:
                data_pae.append(perfect_pae)
                labels_pae.append('Perfect Match (1.0)')
                n_total = len(motif_scores.dropna())
                n_perfect = perfect_mask.sum()
                percent_perfect = 100 * n_perfect / n_total if n_total else 0
                percentages_pae.append(percent_perfect)
            
            if not imperfect_pae.empty:
                data_pae.append(imperfect_pae)
                labels_pae.append('Imperfect (<1.0)')
                n_total = len(motif_scores.dropna())
                n_imperfect = imperfect_mask.sum()
                percent_imperfect = 100 * n_imperfect / n_total if n_total else 0
                percentages_pae.append(percent_imperfect)
            
            # Only plot if we have data
            if len(data_pae) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                parts = ax.violinplot(data_pae, showmeans=False, showmedians=True)
                ax.set_xticks(range(1, len(labels_pae) + 1))
                ax.set_xticklabels(labels_pae)
                ax.set_ylabel('af3_chain_pair_pae_min')
                ax.set_title('max_af3_rna_ipTM_pae by Motif Match')
                # Annotate percentages below x-ticks
                for i, pct in enumerate(percentages_pae):
                    ax.text(i + 1, -0.08, f'{pct:.1f}%', ha='center', va='top', fontsize=12, color='black',
                            transform=ax.get_xaxis_transform())
                plt.tight_layout()
                plt.savefig(os.path.join(plot_creator.results_dir, 'af3_chain_pair_pae_min_by_motif_match.png'))
                plt.close()

        from matplotlib.patches import Patch

        multi_chain_df = self.merged_df[self.merged_df['NumberRNAs'] == 1]
        if not multi_chain_df.empty:
            def get_chain_pair_pae(val):
                if isinstance(val, str):
                    try:
                        return float(val)
                    except Exception:
                        return np.nan
                elif val is not None:
                    return float(val)
                return np.nan

            chain_pair_pae = multi_chain_df['af3_chain_pair_pae_min'].apply(get_chain_pair_pae)
            rna_motif_score = pd.to_numeric(multi_chain_df['RNAmotif_score'], errors='coerce')
            complex_tm_score = pd.to_numeric(multi_chain_df['Complex_TM'], errors='coerce')

            # Classification
            pae_le_10_mask = chain_pair_pae <= 10
            pae_gt_10_mask = chain_pair_pae > 10

            motif_score_le_10 = rna_motif_score[pae_le_10_mask].dropna()
            motif_score_gt_10 = rna_motif_score[pae_gt_10_mask].dropna()
            complex_tm_le_10 = complex_tm_score[pae_le_10_mask].dropna()
            complex_tm_gt_10 = complex_tm_score[pae_gt_10_mask].dropna()

            # Percentages
            n_total = len(chain_pair_pae.dropna())
            n_le_10 = pae_le_10_mask.sum()
            n_gt_10 = pae_gt_10_mask.sum()
            percent_le_10 = 100 * n_le_10 / n_total if n_total else 0
            percent_gt_10 = 100 * n_gt_10 / n_total if n_total else 0
            percentages = [percent_le_10, percent_gt_10]

            # Prepare data and positions: for each group, orange (motif), blue (tm)
            data = [
                motif_score_le_10 if not motif_score_le_10.empty else pd.Series(dtype=float),
                complex_tm_le_10 if not complex_tm_le_10.empty else pd.Series(dtype=float),
                motif_score_gt_10 if not motif_score_gt_10.empty else pd.Series(dtype=float),
                complex_tm_gt_10 if not complex_tm_gt_10.empty else pd.Series(dtype=float)
            ]
            positions = [1, 2, 4, 5]
            colors = ['#FF8F00', '#1976D2', '#FF8F00', '#1976D2']

            fig, ax = plt.subplots(figsize=(10, 6))
            parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.7)

            # Set correct colors: alternate orange/blue
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

            # X-tick labels and annotation
            ax.set_xticks([1.5, 4.5])
            ax.set_xticklabels(['PAE ≤ 10', 'PAE > 10'])
            ax.set_ylabel('Score')
            ax.set_title('RNAmotif_score and Complex_TM by Chain Pair PAE')
            # Annotate percentages below x-ticks
            for i, pct in enumerate(percentages):
                ax.text(1.5 + 3 * i, -0.08, f'{pct:.1f}%', ha='center', va='top', fontsize=12, color='black',
                        transform=ax.get_xaxis_transform())
            # Add legend
            legend_patches = [Patch(facecolor='#FF8F00', label='RNAmotif_score'),
                              Patch(facecolor='#1976D2', label='Complex_TM')]
            ax.legend(handles=legend_patches, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_creator.results_dir, 'RNAmotif_and_ComplexTM_by_chain_pair_pae.png'))
            plt.close()

        def classify_msa(row):
            try:
                msa_dict = row['RNA_msa_size']
                if isinstance(msa_dict, str):
                    msa_dict = safe_eval(msa_dict)
                if not isinstance(msa_dict, dict):
                    return np.nan

                chain_ids = row['RNAChainIDs_pred']
                if isinstance(chain_ids, str):
                    chain_ids = safe_eval(chain_ids)
                # Ensure it's a list
                if not isinstance(chain_ids, (list, tuple)):
                    chain_ids = [chain_ids]

                rna_lengths = row['RNALength']
                if isinstance(rna_lengths, str):
                    rna_lengths = safe_eval(rna_lengths)
                # Ensure it's a list
                if not isinstance(rna_lengths, (list, tuple)):
                    rna_lengths = [rna_lengths]

                # If only one chain, just use its MSA value
                if len(msa_dict) == 1:
                    msa_val = list(msa_dict.values())[0]
                else:
                    # Find index of longest RNA
                    max_len = max(rna_lengths)
                    max_idx = rna_lengths.index(max_len)
                    chain_id = chain_ids[max_idx]
                    msa_val = msa_dict.get(str(chain_id), 1)

                return '+rMSA' if msa_val > 1 else '-rMSA'
            except Exception as e:
                print(
                    f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                return np.nan

        # Prepare data
        df = self.pred_df  # or self.merged_df if you want to include experimental columns
        if all(col in df.columns for col in ['af3_rna_pTM', 'RNA_TM_AllChains', 'RNA_msa_size', 'RNAChainIDs_pred', 'RNALength']):
            msa_status = df.apply(classify_msa, axis=1)
            rna_ptm = []
            rna_tm = []
            msa_for_pairs = []
            rbp_for_pairs = []
            rbp_status = None
            if 'RBP_PDB_Templates' in df.columns:
                def classify_rbp_template(row):
                    try:
                        rbp_templates = row['RBP_PDB_Templates']
                        if pd.isna(rbp_templates) or rbp_templates == '' or rbp_templates == '[]' or rbp_templates == 'null':
                            return '-RBP'
                        if isinstance(rbp_templates, str):
                            try:
                                parsed = safe_eval(rbp_templates)
                                if isinstance(parsed, list):
                                    return '+RBP' if len(parsed) > 0 else '-RBP'
                            except:
                                return '-RBP'
                        if isinstance(rbp_templates, list):
                            return '+RBP' if len(rbp_templates) > 0 else '-RBP'
                        return '+RBP'
                    except Exception:
                        return np.nan
                rbp_status = df.apply(classify_rbp_template, axis=1)

            for idx, row in df.iterrows():
                msa_cat = msa_status.loc[idx]
                if pd.isna(msa_cat):
                    continue
                ptm_values = _to_float_list(row['af3_rna_pTM'])
                tm_values = _tm_allchains_to_diag_list(row['RNA_TM_AllChains'])
                if not ptm_values or not tm_values:
                    continue
                n_pairs = min(len(ptm_values), len(tm_values))
                for i in range(n_pairs):
                    rna_ptm.append(ptm_values[i])
                    rna_tm.append(tm_values[i])
                    msa_for_pairs.append(msa_cat)
                    if rbp_status is not None:
                        rbp_for_pairs.append(rbp_status.loc[idx])

            if rna_ptm and rna_tm:
                color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}

                # Call the color scatterplot
                plot_creator.get_msa_scatterplot(
                    table_source='af3_metrics',
                    xAxis_score=rna_ptm,
                    xAxis_label='RNA pTM',
                    name='msa_rna_pTM_vs_rna_TM_all_chains_msa',
                    yAxis_label='RNA TM',
                    yAxis_score=rna_tm,
                    color_by=msa_for_pairs,
                    color_map=color_map
                )
            
                # Add RBP template color-coded scatter plot for RNA pTM vs RNA TM
                if rbp_status is not None:
                    valid_rbp = [i for i, s in enumerate(rbp_for_pairs) if not pd.isna(s)]
                    if valid_rbp:
                        color_map_rbp = {'+RBP': 'green', '-RBP': 'red'}
                        plot_creator_rbp.get_msa_scatterplot(
                            table_source='af3_metrics',
                            xAxis_score=[rna_ptm[i] for i in valid_rbp],
                            xAxis_label='RNA pTM',
                            name='msa_rna_pTM_vs_rna_TM_all_chains_rbp',
                            yAxis_label='RNA TM',
                            yAxis_score=[rna_tm[i] for i in valid_rbp],
                            color_by=[rbp_for_pairs[i] for i in valid_rbp],
                            color_map=color_map_rbp
                        )

        # Add MSA color-coded 3D scatter plot: Chain Pair PAE, Global PAE, RNAmotif_score
        if all(col in df.columns for col in
               ['af3_chain_pair_pae_min', 'af3_global_pae_avg', 'RNAmotif_score', 'RNA_msa_size',
                'RNAChainIDs_pred', 'RNALength']):
            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(
                        f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            msa_status = df.apply(classify_msa, axis=1)
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            global_pae = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
            motif_score = pd.to_numeric(df['RNAmotif_score'], errors='coerce')
            valid = (~chain_pair_pae.isna()) & (~global_pae.isna()) & (~motif_score.isna()) & (~msa_status.isna())

            if valid.any():
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                msa_plus_mask = msa_status[valid] == '+rMSA'
                msa_minus_mask = msa_status[valid] == '-rMSA'
                if msa_plus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_plus_mask],
                               global_pae[valid][msa_plus_mask],
                               motif_score[valid][msa_plus_mask],
                               c='blue', label='+rMSA', alpha=0.7, s=50)
                if msa_minus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_minus_mask],
                               global_pae[valid][msa_minus_mask],
                               motif_score[valid][msa_minus_mask],
                               c='orange', label='-rMSA', alpha=0.7, s=50)
                ax.set_xlabel('Chain Pair PAE (smaller=better)')
                ax.set_ylabel('Global PAE (smaller=better)')
                ax.set_zlabel('RNA motif score')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_creator.results_dir,
                                         'chain_pair_pae_global_pae_vs_rnamotif_score_3d_msa.png'))
                plt.close()

        # Add MSA color-coded 3D scatter plot: Chain Pair PAE, Global PAE, RNA_RMSD
        if all(col in df.columns for col in
               ['af3_chain_pair_pae_min', 'af3_global_pae_avg', 'RNA_RMSD', 'RNA_msa_size', 'RNAChainIDs_pred',
                'RNALength']):

            def parse_rna_rmsd(val):
                if pd.isna(val):
                    return np.nan
                try:
                    if isinstance(val, str):
                        if '[' in val:
                            values = ast.literal_eval(val)
                            if isinstance(values, list) and len(values) > 0:
                                return float(values[0])
                        return float(val)
                    elif isinstance(val, (list, tuple)):
                        return float(val[0]) if len(val) > 0 else np.nan
                    return float(val)
                except Exception:
                    return np.nan

            def classify_msa(row):
                try:
                    msa_dict = row['RNA_msa_size']
                    if isinstance(msa_dict, str):
                        msa_dict = safe_eval(msa_dict)
                    if not isinstance(msa_dict, dict):
                        return np.nan

                    chain_ids = row['RNAChainIDs_pred']
                    if isinstance(chain_ids, str):
                        chain_ids = safe_eval(chain_ids)
                    if not isinstance(chain_ids, (list, tuple)):
                        chain_ids = [chain_ids]

                    rna_lengths = row['RNALength']
                    if isinstance(rna_lengths, str):
                        rna_lengths = safe_eval(rna_lengths)
                    if not isinstance(rna_lengths, (list, tuple)):
                        rna_lengths = [rna_lengths]

                    if len(msa_dict) == 1:
                        msa_val = list(msa_dict.values())[0]
                    else:
                        max_len = max(rna_lengths)
                        max_idx = rna_lengths.index(max_len)
                        chain_id = chain_ids[max_idx]
                        msa_val = msa_dict.get(str(chain_id), 1)

                    return '+rMSA' if msa_val > 1 else '-rMSA'
                except Exception as e:
                    print(
                        f"Error in classify_msa: {e}. Row data: RNALength={row.get('RNALength')}, RNAChainIDs_pred={row.get('RNAChainIDs_pred')}")
                    return np.nan

            msa_status = df.apply(classify_msa, axis=1)
            chain_pair_pae = pd.to_numeric(df['af3_chain_pair_pae_min'], errors='coerce')
            global_pae = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce')
            rna_rmsd = df['RNA_RMSD'].apply(parse_rna_rmsd)
            valid = (~chain_pair_pae.isna()) & (~global_pae.isna()) & (~rna_rmsd.isna()) & (~msa_status.isna())

            if valid.any():
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                msa_plus_mask = msa_status[valid] == '+rMSA'
                msa_minus_mask = msa_status[valid] == '-rMSA'
                if msa_plus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_plus_mask],
                               global_pae[valid][msa_plus_mask],
                               rna_rmsd[valid][msa_plus_mask],
                               c='blue', label='+rMSA', alpha=0.7, s=50)
                if msa_minus_mask.any():
                    ax.scatter(chain_pair_pae[valid][msa_minus_mask],
                               global_pae[valid][msa_minus_mask],
                               rna_rmsd[valid][msa_minus_mask],
                               c='orange', label='-rMSA', alpha=0.7, s=50)
                ax.set_xlabel('Chain Pair PAE (smaller=better)')
                ax.set_ylabel('Global PAE (smaller=better)')
                ax.set_zlabel('RNA RMSD')
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_creator.results_dir, 'chain_pair_pae_global_pae_vs_rna_rmsd_3d_msa.png'))
                plt.close()

    def plot_global_pae_metrics(self, single_chain_only=False):
        """Plot global PAE metrics separately"""
        df = self.filter_data(self.pred_df, single_chain_only)

        # Define metrics to plot (excluding PAE)
        metrics = {
            'Global PAE': 'af3_global_pae_avg'
        }

        # Process data for each metric
        data_dict = {}
        for label, column in metrics.items():
            values = df[column].dropna()
            if len(values) > 0:
                data_dict[label] = values

        # Create violin plot using PlotCreator with af3_metrics type
        plot_creator = PlotCreator('af3_metrics', self.msa_option, single_chain_only, self.rbp_template_option)
        plot_creator.set_plot_style()

        # Convert data for violin plot
        data_values = [data_dict[key] for key in data_dict.keys()]
        labels = list(data_dict.keys())

        # Clean data_values to ensure all are numeric
        cleaned_data_values = []
        for vals in data_values:
            numeric_vals = pd.to_numeric(vals, errors='coerce').dropna()
            cleaned_data_values.append(numeric_vals)

        plot_creator.get_violin_plot(
            table_source='pae_metrics',
            data_values=cleaned_data_values,
            labels=labels,
            name='global_pae',
            yAxis_label='PAE Score'
        )

    def plot_bond_metrics(self, single_chain_only=False):
        """Plot RNA bond metrics (glycosidic, sugar pucker, gamma)"""
        plot_creator = PlotCreator('bond_metrics', self.msa_option, single_chain_only, self.rbp_template_option)
        plot_creator.set_plot_style()

        # Change filter_chains to filter_data
        df = self.filter_data(self.merged_df, single_chain_only)
        # print("df", self.merged_df)

        # Plot glycosidic bond
        self._plot_glycosidic_bond(plot_creator, df, single_chain_only)

        # Plot sugar pucker
        self._plot_sugar_pucker(plot_creator, df, single_chain_only)

        # Plot gamma angle
        self._plot_gamma_angle(plot_creator, df, single_chain_only)
        
        # Plot bond metrics by MSA (per-sample differences)
        self._plot_bond_metrics_by_msa(plot_creator, df, single_chain_only)
        
        # Plot torsion angle comparisons from DSSR files
        self._plot_dssr_torsion_angles(plot_creator, single_chain_only)

    def _plot_glycosidic_bond(self, plot_creator, df, single_chain_only):
        """Plot glycosidic bond distributions"""
        categories = ['anti', 'syn']

        # Process glycosidic bond data
        chi_counts_pred = {cat: 0 for cat in categories}
        chi_counts_exp = {cat: 0 for cat in categories}

        for chi_dict_str in df['RNA_GlycosidicBond_pred'].dropna():
            counts = self.parse_dict_string(chi_dict_str)
            for cat in categories:
                chi_counts_pred[cat] += counts.get(cat, 0)

        for chi_dict_str in df['RNA_GlycosidicBond_exp'].dropna():
            counts = self.parse_dict_string(chi_dict_str)
            for cat in categories:
                chi_counts_exp[cat] += counts.get(cat, 0)

        # Calculate percentages
        total_pred = sum(chi_counts_pred.values())
        total_exp = sum(chi_counts_exp.values())

        chi_percent_pred = {cat: (count / total_pred * 100 if total_pred > 0 else 0)
                            for cat, count in chi_counts_pred.items()}
        chi_percent_exp = {cat: (count / total_exp * 100 if total_exp > 0 else 0)
                           for cat, count in chi_counts_exp.items()}

        # Create plot
        plt.figure(figsize=(8, 6))
        x = range(len(categories))
        plt.bar(x, [chi_percent_exp[cat] for cat in categories], color='orange', label='Experimental', width=0.35)
        plt.bar([i + 0.35 for i in x], [chi_percent_pred[cat] for cat in categories],
                color='green', alpha=0.7, label='Predicted', width=0.35)
        plt.xticks([i + 0.175 for i in x], categories)
        plt.xlabel("Chi Orientation")
        plt.ylabel("Percentage (%)")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()

        suffix = '_single_chain' if single_chain_only else ''
        plot_creator.save_plot(f'glycosidic_bond{suffix}')

    def _plot_sugar_pucker(self, plot_creator, df, single_chain_only):
        """Plot sugar pucker distributions"""
        categories = ["C3'-endo", "C2'-endo", "C3'-exo", "C4'-exo", "C2'-exo", "O4'-endo", "O4'-exo"]

        # Process data
        pucker_counts_pred = {cat: 0 for cat in categories}
        pucker_counts_exp = {cat: 0 for cat in categories}

        for pucker_dict_str in df['RNA_SugarPucker_pred'].dropna():
            counts = self.parse_dict_string(pucker_dict_str)
            for cat in categories:
                pucker_counts_pred[cat] += counts.get(cat, 0)

        for pucker_dict_str in df['RNA_SugarPucker_exp'].dropna():
            counts = self.parse_dict_string(pucker_dict_str)
            for cat in categories:
                pucker_counts_exp[cat] += counts.get(cat, 0)

        # Calculate percentages and create plot
        total_pred = sum(pucker_counts_pred.values())
        total_exp = sum(pucker_counts_exp.values())

        pucker_percent_pred = {cat: (count / total_pred * 100 if total_pred > 0 else 0)
                               for cat, count in pucker_counts_pred.items()}
        pucker_percent_exp = {cat: (count / total_exp * 100 if total_exp > 0 else 0)
                              for cat, count in pucker_counts_exp.items()}

        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        plt.bar(x, [pucker_percent_exp[cat] for cat in categories], color='orange', label='Experimental', width=0.35)
        plt.bar([i + 0.35 for i in x], [pucker_percent_pred[cat] for cat in categories],
                color='green', alpha=0.7, label='Predicted', width=0.35)
        plt.xticks([i + 0.175 for i in x], categories, rotation=45, ha='right')
        plt.xlabel("Sugar Pucker Type")
        plt.ylabel("Percentage (%)")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()

        suffix = '_single_chain' if single_chain_only else ''
        plot_creator.save_plot(f'sugar_pucker{suffix}')

    def _plot_gamma_angle(self, plot_creator, df, single_chain_only):
        """Plot gamma angle distributions"""
        categories = ['g+', 'g-', 't']

        # Process data
        gamma_counts_pred = {cat: 0 for cat in categories}
        gamma_counts_exp = {cat: 0 for cat in categories}

        for gamma_dict_str in df['RNA_GammaAngle_pred'].dropna():
            counts = self.parse_dict_string(gamma_dict_str)
            for cat in categories:
                gamma_counts_pred[cat] += counts.get(cat, 0)

        for gamma_dict_str in df['RNA_GammaAngle_exp'].dropna():
            counts = self.parse_dict_string(gamma_dict_str)
            for cat in categories:
                gamma_counts_exp[cat] += counts.get(cat, 0)

        # Calculate percentages and create plot
        total_pred = sum(gamma_counts_pred.values())
        total_exp = sum(gamma_counts_exp.values())

        gamma_percent_pred = {cat: (count / total_pred * 100 if total_pred > 0 else 0)
                              for cat, count in gamma_counts_pred.items()}
        gamma_percent_exp = {cat: (count / total_exp * 100 if total_exp > 0 else 0)
                             for cat, count in gamma_counts_exp.items()}

        plt.figure(figsize=(8, 6))
        x = range(len(categories))
        plt.bar(x, [gamma_percent_exp[cat] for cat in categories], color='orange', label='Experimental', width=0.35)
        plt.bar([i + 0.35 for i in x], [gamma_percent_pred[cat] for cat in categories],
                color='green', alpha=0.7, label='Predicted', width=0.35)
        plt.xticks([i + 0.175 for i in x], categories)
        plt.xlabel("Gamma Angle Conformation")
        plt.ylabel("Percentage (%)")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()

        suffix = '_single_chain' if single_chain_only else ''
        plot_creator.save_plot(f'gamma_angle{suffix}')
    
    def _plot_bond_metrics_by_msa(self, plot_creator, df, single_chain_only):
        """Plot error rates per category for bond metrics (gamma_angle, glycosidic_bond, sugar_pucker) by MSA status"""
        from ast import literal_eval as safe_eval
        
        # Classify MSA status for each row
        def classify_msa(row):
            try:
                msa_dict = row.get('RNA_msa_size')
                if pd.isna(msa_dict):
                    return np.nan
                    
                if isinstance(msa_dict, str):
                    msa_dict = safe_eval(msa_dict)
                if not isinstance(msa_dict, dict):
                    return np.nan

                chain_ids = row.get('RNAChainIDs_pred')
                if isinstance(chain_ids, str):
                    chain_ids = safe_eval(chain_ids)
                if not isinstance(chain_ids, (list, tuple)):
                    chain_ids = [chain_ids]

                rna_lengths = row.get('RNALength')
                if isinstance(rna_lengths, str):
                    rna_lengths = safe_eval(rna_lengths)
                if not isinstance(rna_lengths, (list, tuple)):
                    rna_lengths = [rna_lengths]

                # If only one chain, just use its MSA value
                if len(msa_dict) == 1:
                    msa_val = list(msa_dict.values())[0]
                else:
                    # Find index of longest RNA
                    max_len = max(rna_lengths)
                    max_idx = rna_lengths.index(max_len)
                    chain_id = chain_ids[max_idx]
                    msa_val = msa_dict.get(str(chain_id), 1)

                return '+rMSA' if msa_val > 1 else '-rMSA'
            except Exception:
                return np.nan
        
        # Add MSA classification
        df_with_msa = df.copy()
        
        # Check if required columns exist
        required_cols = ['RNA_GammaAngle_pred', 'RNA_GammaAngle_exp', 
                        'RNA_GlycosidicBond_pred', 'RNA_GlycosidicBond_exp',
                        'RNA_SugarPucker_pred', 'RNA_SugarPucker_exp']
        missing_cols = [col for col in required_cols if col not in df_with_msa.columns]
        if missing_cols:
            return
        
        # Check if RNA_msa_size exists
        if 'RNA_msa_size' not in df_with_msa.columns:
            return
        
        df_with_msa['msa_status'] = df_with_msa.apply(classify_msa, axis=1)
        df_with_msa = df_with_msa[df_with_msa['msa_status'].notna()]
        
        if df_with_msa.empty:
            return
        
        # Calculate accuracy rates for all bond metrics and combine into one plot
        # Gamma angle categories
        gamma_categories = ['g+', 'g-', 't']
        gamma_data = self._calculate_accuracy_rates(
            df_with_msa, 'RNA_GammaAngle_pred', 'RNA_GammaAngle_exp',
            gamma_categories
        )
        
        # Glycosidic bond categories
        glycosidic_categories = ['anti', 'syn']
        glycosidic_data = self._calculate_accuracy_rates(
            df_with_msa, 'RNA_GlycosidicBond_pred', 'RNA_GlycosidicBond_exp',
            glycosidic_categories, tolerance=0.15
        )
        
        # Sugar pucker categories (only c3'endo and c2'endo)
        pucker_categories = ["C3'-endo", "C2'-endo"]
        pucker_data = self._calculate_accuracy_rates(
            df_with_msa, 'RNA_SugarPucker_pred', 'RNA_SugarPucker_exp',
            pucker_categories, tolerance=0.15
        )
        
        # Combine all categories with labels and error statistics
        all_categories = []
        all_accuracy_msa = []
        all_accuracy_no_msa = []
        all_errors_msa = []
        all_errors_no_msa = []
        
        # Skip gamma angles - removed per user request
        # Add glycosidic bond with prefix - ensure order matches categories list
        for cat in glycosidic_categories:
            if cat in glycosidic_data:
                all_categories.append(f'Glycosidic-{cat}')
                all_accuracy_msa.append(glycosidic_data[cat]['+rMSA']['accuracy'])
                all_accuracy_no_msa.append(glycosidic_data[cat]['-rMSA']['accuracy'])
                all_errors_msa.append(glycosidic_data[cat]['+rMSA']['error'])
                all_errors_no_msa.append(glycosidic_data[cat]['-rMSA']['error'])
        
        # Add sugar pucker with prefix - ensure order matches categories list
        for cat in pucker_categories:
            if cat in pucker_data:
                all_categories.append(f'Pucker-{cat}')
                all_accuracy_msa.append(pucker_data[cat]['+rMSA']['accuracy'])
                all_accuracy_no_msa.append(pucker_data[cat]['-rMSA']['accuracy'])
                all_errors_msa.append(pucker_data[cat]['+rMSA']['error'])
                all_errors_no_msa.append(pucker_data[cat]['-rMSA']['error'])
        
        # Calculate combined "All MSA" accuracy for each category
        tolerance = 0.15  # 15% relative error tolerance (same as used in _calculate_accuracy_rates)
        all_accuracy_all = []
        all_errors_all = []
        
        # Recalculate combined accuracy (all samples together)
        for cat_idx, cat in enumerate(all_categories):
            # Determine which metric this category belongs to
            if cat.startswith('Glycosidic-'):
                cat_name = cat.replace('Glycosidic-', '')
                pred_col = 'RNA_GlycosidicBond_pred'
                exp_col = 'RNA_GlycosidicBond_exp'
            elif cat.startswith('Pucker-'):
                cat_name = cat.replace('Pucker-', '')
                pred_col = 'RNA_SugarPucker_pred'
                exp_col = 'RNA_SugarPucker_exp'
            else:
                continue
            
            # Calculate combined accuracy (all MSA together)
            total_all = 0
            correct_all = 0
            
            for idx, row in df_with_msa.iterrows():
                pred_dict_str = row.get(pred_col)
                exp_dict_str = row.get(exp_col)
                
                if pd.isna(pred_dict_str) or pd.isna(exp_dict_str):
                    continue
                
                try:
                    pred_dict = self.parse_dict_string(pred_dict_str)
                    exp_dict = self.parse_dict_string(exp_dict_str)
                    
                    exp_count = exp_dict.get(cat_name, 0)
                    pred_count = pred_dict.get(cat_name, 0)
                    
                    if exp_count > 0:
                        total_all += 1
                        # Use tolerance-based matching: count as correct if within 15% relative error
                        # This handles cases like:
                        # - pred=7, exp=8: |7-8|/8 = 0.125 (12.5%) <= 15% → correct
                        # - pred=12, exp=12: |12-12|/12 = 0 (0%) <= 15% → correct
                        # - pred=4, exp=12: |4-12|/12 = 0.667 (66.7%) > 15% → incorrect
                        rel_error = abs(pred_count - exp_count) / float(exp_count)
                        if rel_error <= tolerance:
                            correct_all += 1
                except Exception:
                    continue
            
            accuracy_all = (correct_all / float(total_all) * 100) if total_all > 0 else 0
            p_all = accuracy_all / 100.0
            se_all = (np.sqrt(p_all * (1 - p_all) / total_all) * 100) if total_all > 0 else 0
            
            all_accuracy_all.append(accuracy_all)
            all_errors_all.append(se_all)
        
        # Store accuracy data for statistics
        self.bond_metrics_accuracy_data = {
            'categories': all_categories,
            'accuracy_all': all_accuracy_all,
            'accuracy_msa': all_accuracy_msa,
            'accuracy_no_msa': all_accuracy_no_msa,
            'errors_all': all_errors_all,
            'errors_msa': all_errors_msa,
            'errors_no_msa': all_errors_no_msa
        }
        
        # Create combined grouped bar plot with error bars
        self._plot_accuracy_rates_combined(
            plot_creator, all_categories, all_accuracy_msa, all_accuracy_no_msa,
            all_errors_msa, all_errors_no_msa,
            'Bond Metrics Classification Accuracy Rate (τ=15%)',
            'bond_metrics_accuracy_by_MSA', single_chain_only
        )
    
    def _calculate_accuracy_rates(self, df, pred_col, exp_col, categories, tolerance=0.15):
        """
        Calculate accuracy rates per category per MSA group using percentage-based normalized error.
        
        Args:
            tolerance: Relative error threshold (tau). Default 0.15 (15%).
                      Count as correct only if |pred - exp| / exp <= tolerance
        
        Returns:
            Dictionary: {category: {'+rMSA': {'accuracy': rate, 'error': std_error}, '-rMSA': {...}}}
        """
        # Initialize accuracy tracking: {category: {'+rMSA': {'total': 0, 'correct': 0}, '-rMSA': {...}}}
        accuracy_stats = {cat: {'+rMSA': {'total': 0, 'correct': 0}, '-rMSA': {'total': 0, 'correct': 0}} 
                         for cat in categories}
        
        # Process each sample
        for idx, row in df.iterrows():
            pred_dict_str = row.get(pred_col)
            exp_dict_str = row.get(exp_col)
            msa_status = row.get('msa_status')
            
            if pd.isna(pred_dict_str) or pd.isna(exp_dict_str) or pd.isna(msa_status):
                continue
            
            try:
                pred_dict = self.parse_dict_string(pred_dict_str)
                exp_dict = self.parse_dict_string(exp_dict_str)
                
                # For each category, check if GT has it and calculate relative error
                for cat in categories:
                    exp_count = exp_dict.get(cat, 0)
                    pred_count = pred_dict.get(cat, 0)
                    
                    # If GT has this category (exp_count > 0), count it
                    if exp_count > 0:
                        accuracy_stats[cat][msa_status]['total'] += 1
                        
                        # Calculate relative error: |pred - exp| / exp
                        rel_error = abs(pred_count - exp_count) / float(exp_count)
                        
                        # Count as correct if relative error is within threshold
                        if rel_error <= tolerance:
                            accuracy_stats[cat][msa_status]['correct'] += 1
            except Exception:
                continue
        
        # Calculate accuracy rates and standard errors
        accuracy_results = {}
        for cat in categories:
            # +rMSA accuracy rate
            total_msa = accuracy_stats[cat]['+rMSA']['total']
            correct_msa = accuracy_stats[cat]['+rMSA']['correct']
            # Accuracy rate = (number of correct samples) / (total samples) * 100
            # Higher accuracy_rate means BETTER performance
            accuracy_msa = (correct_msa / float(total_msa) * 100) if total_msa > 0 else 0
            
            # Calculate standard error for proportion (accuracy as proportion)
            # SE = sqrt(p * (1-p) / n) where p is proportion and n is sample size
            p_msa = correct_msa / float(total_msa) if total_msa > 0 else 0
            se_msa = np.sqrt(p_msa * (1 - p_msa) / float(total_msa)) * 100 if total_msa > 0 else 0
            
            # -rMSA accuracy rate
            total_no_msa = accuracy_stats[cat]['-rMSA']['total']
            correct_no_msa = accuracy_stats[cat]['-rMSA']['correct']
            # Accuracy rate = (number of correct samples) / (total samples) * 100
            accuracy_no_msa = (correct_no_msa / float(total_no_msa) * 100) if total_no_msa > 0 else 0
            
            # Calculate standard error for -rMSA
            p_no_msa = correct_no_msa / float(total_no_msa) if total_no_msa > 0 else 0
            se_no_msa = np.sqrt(p_no_msa * (1 - p_no_msa) / float(total_no_msa)) * 100 if total_no_msa > 0 else 0
            
            accuracy_results[cat] = {
                '+rMSA': {'accuracy': accuracy_msa, 'error': se_msa},
                '-rMSA': {'accuracy': accuracy_no_msa, 'error': se_no_msa}
            }
        
        return accuracy_results
    
    def _plot_accuracy_rates_combined(self, plot_creator, categories, accuracy_rates_msa, accuracy_rates_no_msa,
                                      errors_msa, errors_no_msa, title, plot_name, single_chain_only):
        """Create a combined grouped bar plot showing accuracy rates for all bond metrics, grouped by MSA status"""
        import matplotlib.pyplot as plt
        
        # Use plotCreator's style and pattern for grouped barplot with error bars
        plot_creator.set_plot_style()
        
        # Ensure data arrays match categories order
        # Convert to numpy arrays to ensure proper indexing
        accuracy_rates_msa = np.array(accuracy_rates_msa)
        accuracy_rates_no_msa = np.array(accuracy_rates_no_msa)
        errors_msa = np.array(errors_msa)
        errors_no_msa = np.array(errors_no_msa)
        
        groups = [accuracy_rates_msa, accuracy_rates_no_msa]
        legend_labels = ['+rMSA', '-rMSA']
        colors = ['#2E86AB', '#A23B72']
        
        indices = np.arange(len(categories))
        num_series = len(groups)
        group_width = 0.35
        offsets = [i * group_width for i in range(num_series)]
        
        plt.figure(figsize=(14, 6))
        
        # Plot bars with error bars (following plotCreator's grouped_barplot pattern)
        for i, (group, color) in enumerate(zip(groups, colors)):
            x_positions = indices + offsets[i] - group_width * (num_series - 1) / 2
            errors = errors_msa if i == 0 else errors_no_msa
            plt.bar(x_positions, group, group_width, label=legend_labels[i], 
                   color=color, yerr=errors, capsize=3, 
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        plt.xlabel('')
        plt.ylabel('Count Accuracy (%)', fontsize=12)
        plt.xticks(indices, categories, rotation=45, ha='right', fontsize=10)
        plt.legend(fontsize=11)
        
        # Set y-axis limits (0-100% for accuracy)
        plt.ylim([0, 105])
        plt.yticks(np.arange(0, 101, 10))
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add vertical lines to separate metric groups
        # After glycosidic bond (2 categories)
        if len(categories) > 2:
            plt.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        # Save plot using plot_creator's save method
        plot_creator.save_plot(plot_name)
        plt.close()
    
    def _plot_dssr_torsion_angles(self, plot_creator, single_chain_only):
        """Plot average torsion angles (α, β, γ, δ, ε, ζ, χ) from DSSR files comparing pred vs exp"""
        import json
        from ast import literal_eval as safe_eval
        
        # Get DSSR folder path
        parent_folder = self.config.get_parent_folder()
        if not parent_folder:
            print("Warning: Could not determine parent folder for DSSR files")
            return
        
        dssr_folder = os.path.join(parent_folder, 'dssr')
        if not os.path.exists(dssr_folder):
            print(f"Warning: DSSR folder not found at {dssr_folder}")
            return
        
        # Get filtered data to determine which samples to process
        df = self.filter_data(self.pred_df, single_chain_only)
        
        # Classify MSA status for each row
        def classify_msa(row):
            try:
                msa_dict = row.get('RNA_msa_size')
                if pd.isna(msa_dict):
                    return np.nan
                    
                if isinstance(msa_dict, str):
                    msa_dict = safe_eval(msa_dict)
                if not isinstance(msa_dict, dict):
                    return np.nan

                chain_ids = row.get('RNAChainIDs_pred')
                if isinstance(chain_ids, str):
                    chain_ids = safe_eval(chain_ids)
                if not isinstance(chain_ids, (list, tuple)):
                    chain_ids = [chain_ids]

                rna_lengths = row.get('RNALength')
                if isinstance(rna_lengths, str):
                    rna_lengths = safe_eval(rna_lengths)
                if not isinstance(rna_lengths, (list, tuple)):
                    rna_lengths = [rna_lengths]

                # If only one chain, just use its MSA value
                if len(msa_dict) == 1:
                    msa_val = list(msa_dict.values())[0]
                else:
                    # Find index of longest RNA
                    max_len = max(rna_lengths)
                    max_idx = rna_lengths.index(max_len)
                    chain_id = chain_ids[max_idx]
                    msa_val = msa_dict.get(str(chain_id), 1)

                return '+rMSA' if msa_val > 1 else '-rMSA'
            except Exception:
                return np.nan
        
        # Torsion angles to plot
        torsion_angles = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
        torsion_labels = {
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'delta': 'δ',
            'epsilon': 'ε',
            'zeta': 'ζ',
            'chi': 'χ'
        }
        
        # Function to calculate average torsion angle from DSSR file
        def get_avg_torsion_angle(dssr_file, angle_name):
            """Extract average torsion angle from DSSR JSON file"""
            if not os.path.exists(dssr_file):
                return None
            
            try:
                with open(dssr_file, 'r') as f:
                    data = json.load(f)
                
                nts = data.get('nts', [])
                if not nts:
                    return None
                
                # Collect all valid angle values
                angle_values = []
                for nt in nts:
                    angle_value = nt.get(angle_name, None)
                    if angle_value is not None and angle_value != 'unknown':
                        try:
                            angle_values.append(float(angle_value))
                        except (ValueError, TypeError):
                            continue
                
                if angle_values:
                    return np.mean(angle_values)
                return None
            except Exception as e:
                print(f"Error reading DSSR file {dssr_file}: {e}")
                return None
        
        # Collect data for each torsion angle (for scatter plots and combined box plot)
        all_angle_data = {angle: {'pred': [], 'exp': [], 'msa': []} for angle in torsion_angles}
        
        for angle_name in torsion_angles:
            pred_values = []
            exp_values = []
            msa_statuses = []
            
            for idx, row in df.iterrows():
                pdb_id = row.get('exp_db_id')
                if pd.isna(pdb_id):
                    continue
                
                # Get MSA status
                msa_status = classify_msa(row)
                if pd.isna(msa_status):
                    continue
                
                # Construct file paths
                pdb_id_lower = str(pdb_id).lower()
                pred_file = os.path.join(dssr_folder, f"{pdb_id_lower}_af_dssr.json")
                exp_file = os.path.join(dssr_folder, f"{pdb_id_lower}_pdb_dssr.json")
                
                # Get average torsion angles
                pred_angle = get_avg_torsion_angle(pred_file, angle_name)
                exp_angle = get_avg_torsion_angle(exp_file, angle_name)
                
                # Only include if both values are available
                if pred_angle is not None and exp_angle is not None:
                    pred_values.append(pred_angle)
                    exp_values.append(exp_angle)
                    msa_statuses.append(msa_status)
                    
                    # Store for combined box plot
                    all_angle_data[angle_name]['pred'].append(pred_angle)
                    all_angle_data[angle_name]['exp'].append(exp_angle)
                    all_angle_data[angle_name]['msa'].append(msa_status)
            
            # Create scatter plot if we have data
            if len(pred_values) > 0:
                pred_series = pd.Series(pred_values)
                exp_series = pd.Series(exp_values)
                msa_series = pd.Series(msa_statuses)
                
                color_map = {'+rMSA': 'blue', '-rMSA': 'orange'}
                plot_creator.get_msa_scatterplot(
                    table_source='bond_metrics',
                    xAxis_score=exp_series,
                    xAxis_label=f'Experimental {torsion_labels[angle_name]}',
                    name=f'dssr_{angle_name}_pred_vs_exp',
                    yAxis_label=f'Predicted {torsion_labels[angle_name]}',
                    yAxis_score=pred_series,
                    color_by=msa_series,
                    color_map=color_map
                )
        
        # Calculate average torsion angles for statistics
        self.torsion_angle_averages = {}
        for angle_name in torsion_angles:
            pred_data = all_angle_data[angle_name]['pred']
            exp_data = all_angle_data[angle_name]['exp']
            msa_data = all_angle_data[angle_name]['msa']
            
            if len(pred_data) > 0 and len(exp_data) > 0:
                # All MSA (combined)
                avg_pred_all = np.mean(pred_data)
                avg_exp_all = np.mean(exp_data)
                
                # Split by MSA status
                pred_msa_plus = [pred_data[i] for i in range(len(pred_data)) if msa_data[i] == '+rMSA']
                pred_msa_minus = [pred_data[i] for i in range(len(pred_data)) if msa_data[i] == '-rMSA']
                exp_msa_plus = [exp_data[i] for i in range(len(exp_data)) if msa_data[i] == '+rMSA']
                exp_msa_minus = [exp_data[i] for i in range(len(exp_data)) if msa_data[i] == '-rMSA']
                
                avg_pred_msa_plus = np.mean(pred_msa_plus) if len(pred_msa_plus) > 0 else None
                avg_pred_msa_minus = np.mean(pred_msa_minus) if len(pred_msa_minus) > 0 else None
                avg_exp_msa_plus = np.mean(exp_msa_plus) if len(exp_msa_plus) > 0 else None
                avg_exp_msa_minus = np.mean(exp_msa_minus) if len(exp_msa_minus) > 0 else None
                
                self.torsion_angle_averages[angle_name] = {
                    'pred_all': avg_pred_all,
                    'exp_all': avg_exp_all,
                    'pred_msa_plus': avg_pred_msa_plus,
                    'pred_msa_minus': avg_pred_msa_minus,
                    'exp_msa_plus': avg_exp_msa_plus,
                    'exp_msa_minus': avg_exp_msa_minus
                }
        
        # Create combined box plot for all torsion angles
        self._plot_combined_torsion_boxplot(plot_creator, all_angle_data, torsion_angles, torsion_labels)
        
        # Create combined box plot split by MSA status
        self._plot_combined_torsion_boxplot_by_msa(plot_creator, all_angle_data, torsion_angles, torsion_labels)
        
        # Create plot showing which torsion angle has the best average accuracy
        self._plot_torsion_angle_accuracy(plot_creator, all_angle_data, torsion_angles, torsion_labels)
    
    def _plot_combined_torsion_boxplot(self, plot_creator, all_angle_data, torsion_angles, torsion_labels):
        """Create a combined box plot showing all torsion angles for both predicted and experimental"""
        import matplotlib.pyplot as plt
        
        # Prepare data for box plot
        plot_data = []
        plot_labels = []
        
        for angle_name in torsion_angles:
            pred_data = all_angle_data[angle_name]['pred']
            exp_data = all_angle_data[angle_name]['exp']
            
            # Only add if we have data
            if len(pred_data) > 0:
                plot_data.append(pred_data)
                plot_labels.append(torsion_labels[angle_name])
            
            if len(exp_data) > 0:
                plot_data.append(exp_data)
                plot_labels.append(torsion_labels[angle_name])
        
        # Create box plot if we have data
        if len(plot_data) > 0:
            plot_creator.set_plot_style()
            plt.figure(figsize=(16, 8))
            
            # Create box plot
            bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            
            # Color boxes: alternate between pred and exp
            colors = []
            for i in range(len(plot_labels)):
                # Even indices are pred, odd indices are exp (since we add pred then exp for each angle)
                if i % 2 == 0:
                    colors.append('#ff7f0e')  # Orange for predicted
                else:
                    colors.append('#2ecc71')  # Green for experimental
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel('Torsion Angle (degrees)', fontsize=12)
            plt.xlabel('Torsion Angle Type', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff7f0e', alpha=0.7, label='Predicted'),
                Patch(facecolor='#2ecc71', alpha=0.7, label='Experimental')
            ]
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            plot_creator.save_plot('dssr_torsion_angles_combined_boxplot')
            plt.close()
    
    def _plot_combined_torsion_boxplot_by_msa(self, plot_creator, all_angle_data, torsion_angles, torsion_labels):
        """Create a combined box plot showing all torsion angles split by MSA status and pred/exp"""
        import matplotlib.pyplot as plt
        
        # Prepare data for box plot: 4 groups per angle (+MSA Pred, +MSA Exp, -MSA Pred, -MSA Exp)
        plot_data = []
        plot_labels = []
        
        for angle_name in torsion_angles:
            pred_data = all_angle_data[angle_name]['pred']
            exp_data = all_angle_data[angle_name]['exp']
            msa_data = all_angle_data[angle_name]['msa']
            
            # Split by MSA status
            pred_msa_plus = [pred_data[i] for i in range(len(pred_data)) if msa_data[i] == '+rMSA']
            pred_msa_minus = [pred_data[i] for i in range(len(pred_data)) if msa_data[i] == '-rMSA']
            exp_msa_plus = [exp_data[i] for i in range(len(exp_data)) if msa_data[i] == '+rMSA']
            exp_msa_minus = [exp_data[i] for i in range(len(exp_data)) if msa_data[i] == '-rMSA']
            
            angle_label = torsion_labels[angle_name]
            
            # Add data in order: +rMSA Pred, +rMSA Exp, -rMSA Pred, -rMSA Exp
            if len(pred_msa_plus) > 0:
                plot_data.append(pred_msa_plus)
                plot_labels.append(f"{angle_label}\n+rMSA\nPred")
            
            if len(exp_msa_plus) > 0:
                plot_data.append(exp_msa_plus)
                plot_labels.append(f"{angle_label}\n+rMSA\nExp")
            
            if len(pred_msa_minus) > 0:
                plot_data.append(pred_msa_minus)
                plot_labels.append(f"{angle_label}\n-rMSA\nPred")
            
            if len(exp_msa_minus) > 0:
                plot_data.append(exp_msa_minus)
                plot_labels.append(f"{angle_label}\n-rMSA\nExp")
        
        # Create box plot if we have data
        if len(plot_data) > 0:
            plot_creator.set_plot_style()
            plt.figure(figsize=(20, 8))
            
            # Create box plot
            bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            
            # Color boxes based on MSA and pred/exp
            colors = []
            for label in plot_labels:
                if '+rMSA' in label and 'Pred' in label:
                    colors.append('#3498db')  # Blue for +rMSA Pred
                elif '+rMSA' in label and 'Exp' in label:
                    colors.append('#2980b9')  # Darker blue for +rMSA Exp
                elif '-rMSA' in label and 'Pred' in label:
                    colors.append('#e74c3c')  # Red for -rMSA Pred
                elif '-rMSA' in label and 'Exp' in label:
                    colors.append('#c0392b')  # Darker red for -rMSA Exp
                else:
                    colors.append('#95a5a6')  # Gray fallback
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel('Torsion Angle (degrees)', fontsize=12)
            plt.xlabel('Torsion Angle Type', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', alpha=0.7, label='+rMSA Predicted'),
                Patch(facecolor='#2980b9', alpha=0.7, label='+rMSA Experimental'),
                Patch(facecolor='#e74c3c', alpha=0.7, label='-rMSA Predicted'),
                Patch(facecolor='#c0392b', alpha=0.7, label='-rMSA Experimental')
            ]
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            plot_creator.save_plot('dssr_torsion_angles_combined_boxplot_by_msa')
            plt.close()
    
    def _plot_torsion_angle_accuracy(self, plot_creator, all_angle_data, torsion_angles, torsion_labels):
        """Create a bar plot showing average accuracy for each torsion angle, split by MSA status"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate accuracy and standard error for each angle, split by MSA
        angle_accuracies_msa = {}
        angle_errors_msa = {}
        angle_accuracies_no_msa = {}
        angle_errors_no_msa = {}
        
        for angle_name in torsion_angles:
            pred_data = all_angle_data[angle_name]['pred']
            exp_data = all_angle_data[angle_name]['exp']
            msa_data = all_angle_data[angle_name]['msa']
            
            if len(pred_data) == 0 or len(exp_data) == 0:
                continue
            
            # Calculate accuracy for each sample, split by MSA
            sample_accuracies_msa = []
            sample_accuracies_no_msa = []
            
            for p, e, msa in zip(pred_data, exp_data, msa_data):
                if e != 0:
                    # Calculate accuracy using individual experimental value
                    # Use a more robust metric: accuracy = 100 * (1 - min(1, |pred - exp| / |exp|))
                    # This ensures accuracy is always between 0 and 100%
                    abs_error = abs(p - e)
                    abs_exp = abs(e)
                    if abs_exp != 0:
                        relative_error = min(1.0, abs_error / abs_exp)  # Cap at 1.0
                        accuracy = 100 * (1 - relative_error)  # Always between 0 and 100
                        
                        if msa == '+rMSA':
                            sample_accuracies_msa.append(accuracy)
                        elif msa == '-rMSA':
                            sample_accuracies_no_msa.append(accuracy)
            
            # Calculate statistics for +rMSA
            if len(sample_accuracies_msa) > 0:
                mean_accuracy_msa = np.mean(sample_accuracies_msa)
                std_error_msa = np.std(sample_accuracies_msa, ddof=1) / np.sqrt(len(sample_accuracies_msa)) if len(sample_accuracies_msa) > 1 else 0
                angle_accuracies_msa[angle_name] = mean_accuracy_msa
                angle_errors_msa[angle_name] = std_error_msa
            
            # Calculate statistics for -rMSA
            if len(sample_accuracies_no_msa) > 0:
                mean_accuracy_no_msa = np.mean(sample_accuracies_no_msa)
                std_error_no_msa = np.std(sample_accuracies_no_msa, ddof=1) / np.sqrt(len(sample_accuracies_no_msa)) if len(sample_accuracies_no_msa) > 1 else 0
                angle_accuracies_no_msa[angle_name] = mean_accuracy_no_msa
                angle_errors_no_msa[angle_name] = std_error_no_msa
        
        # Store for statistics (combined and by MSA)
        self.torsion_angle_accuracies = {}
        self.torsion_angle_errors = {}
        self.torsion_angle_accuracies_msa = angle_accuracies_msa
        self.torsion_angle_errors_msa = angle_errors_msa
        self.torsion_angle_accuracies_no_msa = angle_accuracies_no_msa
        self.torsion_angle_errors_no_msa = angle_errors_no_msa
        
        # Calculate combined accuracies for overall statistics
        for angle_name in torsion_angles:
            if angle_name in angle_accuracies_msa or angle_name in angle_accuracies_no_msa:
                all_samples = []
                if angle_name in angle_accuracies_msa:
                    all_samples.extend([angle_accuracies_msa[angle_name]] * len([x for x in all_angle_data[angle_name]['msa'] if x == '+rMSA']))
                if angle_name in angle_accuracies_no_msa:
                    all_samples.extend([angle_accuracies_no_msa[angle_name]] * len([x for x in all_angle_data[angle_name]['msa'] if x == '-rMSA']))
                if all_samples:
                    self.torsion_angle_accuracies[angle_name] = np.mean(all_samples)
                    self.torsion_angle_errors[angle_name] = np.std(all_samples, ddof=1) / np.sqrt(len(all_samples)) if len(all_samples) > 1 else 0
        
        # Create grouped bar plot if we have data
        if len(angle_accuracies_msa) > 0 or len(angle_accuracies_no_msa) > 0:
            plot_creator.set_plot_style()
            plt.figure(figsize=(14, 6))
            
            # Order by angle order (alpha, beta, gamma, delta, epsilon, zeta, chi)
            ordered_angles = []
            accuracies_msa = []
            errors_msa = []
            accuracies_no_msa = []
            errors_no_msa = []
            
            for angle_name in torsion_angles:
                if angle_name in angle_accuracies_msa or angle_name in angle_accuracies_no_msa:
                    ordered_angles.append(torsion_labels[angle_name])
                    accuracies_msa.append(angle_accuracies_msa.get(angle_name, 0))
                    errors_msa.append(angle_errors_msa.get(angle_name, 0))
                    accuracies_no_msa.append(angle_accuracies_no_msa.get(angle_name, 0))
                    errors_no_msa.append(angle_errors_no_msa.get(angle_name, 0))
            
            # Create grouped bar plot
            x = np.arange(len(ordered_angles))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, accuracies_msa, width, label='+rMSA', color='#2E86AB',
                          yerr=errors_msa, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            bars2 = plt.bar(x + width/2, accuracies_no_msa, width, label='-rMSA', color='#A23B72',
                          yerr=errors_no_msa, capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            
            plt.ylabel('Average Accuracy (%)', fontsize=12)
            plt.xlabel('Torsion Angle', fontsize=12)
            plt.xticks(x, ordered_angles, rotation=45, ha='right', fontsize=10)
            # Only add legend if there are labels
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(fontsize=11)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            max_acc = max(max(accuracies_msa) if accuracies_msa else [0], max(accuracies_no_msa) if accuracies_no_msa else [0])
            plt.ylim([0, max_acc * 1.2 if max_acc > 0 else 100])
            plt.tight_layout(pad=2.0)
            
            plot_creator.save_plot('dssr_torsion_angles_accuracy_comparison')
            plt.close()

    def plot_rna_sugar_pucker(self, single_chain_only=False):
        """Plot sugar pucker distributions as percentages"""
        df = self.filter_data(self.merged_df, single_chain_only)
        categories = ["C3'-endo", "C2'-endo", "C3'-exo", "C4'-exo", "C2'-exo", "O4'-endo", "O4'-exo"]

        # Parse dictionary strings and aggregate counts
        pucker_counts_pred = {cat: 0 for cat in categories}
        pucker_counts_exp = {cat: 0 for cat in categories}

        for pucker_dict_str in df['RNA_SugarPucker_pred'].dropna():
            counts = self.parse_dict_string(pucker_dict_str)
            for cat in categories:
                pucker_counts_pred[cat] += counts.get(cat, 0)

        for pucker_dict_str in df['RNA_SugarPucker_exp'].dropna():
            counts = self.parse_dict_string(pucker_dict_str)
            for cat in categories:
                pucker_counts_exp[cat] += counts.get(cat, 0)

        # Calculate percentages
        total_pred = sum(pucker_counts_pred.values())
        total_exp = sum(pucker_counts_exp.values())

        pucker_percent_pred = {cat: (count / total_pred * 100 if total_pred > 0 else 0)
                               for cat, count in pucker_counts_pred.items()}
        pucker_percent_exp = {cat: (count / total_exp * 100 if total_exp > 0 else 0)
                              for cat, count in pucker_counts_exp.items()}

        # Create plot
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        plt.bar(x, [pucker_percent_exp[cat] for cat in categories], color='yellow', label='Experimental', width=0.35)
        plt.bar([i + 0.35 for i in x], [pucker_percent_pred[cat] for cat in categories],
                color='green', alpha=0.7, label='Predicted', width=0.35)
        plt.xticks([i + 0.175 for i in x], categories, rotation=45, ha='right')
        plt.xlabel("Sugar Pucker Type")
        plt.ylabel("Percentage (%)")
        plt.legend()
        plt.tight_layout(pad=2.0)

        suffix = '_single_chain' if single_chain_only else ''
        plt.savefig(os.path.join(self.results_dir, f'sugar_pucker{suffix}.png'))
        plt.close()

    def plot_rna_gamma_angle(self, single_chain_only=False):
        """Plot gamma angle distributions as percentages"""
        df = self.filter_data(self.merged_df, single_chain_only)
        categories = ['g+', 'g-', 't']

        # Parse dictionary strings and aggregate counts
        gamma_counts_pred = {cat: 0 for cat in categories}
        gamma_counts_exp = {cat: 0 for cat in categories}

        for gamma_dict_str in df['RNA_GammaAngle_pred'].dropna():
            counts = self.parse_dict_string(gamma_dict_str)
            for cat in categories:
                gamma_counts_pred[cat] += counts.get(cat, 0)

        for gamma_dict_str in df['RNA_GammaAngle_exp'].dropna():
            counts = self.parse_dict_string(gamma_dict_str)
            for cat in categories:
                gamma_counts_exp[cat] += counts.get(cat, 0)

        # Calculate percentages
        total_pred = sum(gamma_counts_pred.values())
        total_exp = sum(gamma_counts_exp.values())

        gamma_percent_pred = {cat: (count / total_pred * 100 if total_pred > 0 else 0)
                              for cat, count in gamma_counts_pred.items()}
        gamma_percent_exp = {cat: (count / total_exp * 100 if total_exp > 0 else 0)
                             for cat, count in gamma_counts_exp.items()}

        # Create plot
        plt.figure(figsize=(8, 6))
        x = range(len(categories))
        plt.bar(x, [gamma_percent_exp[cat] for cat in categories], color='yellow', label='Experimental', width=0.35)
        plt.bar([i + 0.35 for i in x], [gamma_percent_pred[cat] for cat in categories],
                color='green', alpha=0.7, label='Predicted', width=0.35)
        plt.xticks([i + 0.175 for i in x], categories)
        plt.xlabel("Gamma Angle Conformation")
        plt.ylabel("Percentage (%)")
        plt.legend()
        plt.tight_layout(pad=2.0)

        suffix = '_single_chain' if single_chain_only else ''
        plt.savefig(os.path.join(self.results_dir, f'gamma_angle{suffix}.png'))
        plt.close()

    def plot_interface_metrics(self, single_chain_only=False):
        """Plot interface metrics (free energy and binding affinity)"""
        plot_creator = PlotCreator('binding_affinity', self.msa_option, single_chain_only, self.rbp_template_option)

        # Plot free energy
        df = self.filter_chains(self.merged_df, True)  # Always use single_chain=True

        plt.figure()  # Create new figure for free energy plot
        # Plot free energy if data exists
        if df['Free_energy_exp'].notna().sum() > 0 and df['Free_energy_pred'].notna().sum() > 0:
            plot_creator.get_scatterplot(
                table_source='pred_vs_exp',
                xAxis_score=df['Free_energy_exp'].values,
                xAxis_label='Experimental Free Energy (kcal/mol)',
                yAxis_label='Predicted Free Energy (kcal/mol)',
                name='free_energy',
                yAxis_score=df['Free_energy_pred'].values
            )
        else:
            print("No valid free energy data to plot")
            plt.close()  # Close empty figure if no data

        # Plot binding affinity separately
        plt.figure()  # Create new figure for binding affinity plot
        kd_exp = df['Binding_affinity_kd_exp'].apply(self.parse_binding_affinity)
        kd_pred = df['Binding_affinity_kd_pred'].apply(self.parse_binding_affinity)

        # Remove None values and sort
        mask = kd_exp.notna() & kd_pred.notna()
        kd_exp = kd_exp[mask]
        kd_pred = kd_pred[mask]

        if len(kd_exp) > 0:
            # Convert to log scale if values are very small
            if kd_exp.min() < 1e-10 or kd_pred.min() < 1e-10:
                kd_exp = np.log10(kd_exp)
                kd_pred = np.log10(kd_pred)
                xlabel = 'Log₁₀[Experimental Kd (M)]'
                ylabel = 'Log₁₀[Predicted Kd (M)]'
            else:
                xlabel = 'Experimental Kd (M)'
                ylabel = 'Predicted Kd (M)'

            plot_creator.get_scatterplot(
                table_source='pred_vs_exp',
                xAxis_score=kd_exp.values,
                xAxis_label=xlabel,
                yAxis_label=ylabel,
                name='binding_affinity',
                yAxis_score=kd_pred.values
            )
        else:
            print("No valid binding affinity data to plot")
            plt.close()  # Close empty figure if no data

    def parse_dict_string(self, dict_string):
        """Parse string representation of dictionary into actual dictionary"""
        try:
            # Remove curly braces and split by comma
            dict_str = dict_string.strip('{}')
            pairs = dict_str.split(',')
            result = {}

            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':')
                    # Clean up the key and value
                    key = key.strip().strip("'").strip('"')
                    value = int(value.strip())
                    result[key] = value
            return result
        except:
            return {}

    def parse_pae_list(self, pae_string):
        """Parse PAE list string and return maximum value"""
        try:
            pae_list = literal_eval(pae_string)
            return max(pae_list) if pae_list else 0
        except:
            return 0

    def parse_binding_affinity(self, value):
        """Parse binding affinity values, handling various formats"""
        try:
            if pd.isna(value):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            # Handle scientific notation strings
            if 'e' in value.lower():
                return float(value)
            # Handle other string formats
            return float(value)
        except (ValueError, TypeError):
            return None

    def filter_by_msa(self, df, msa_option):
        """Filter dataframe based on MSA option"""
        if msa_option not in ['+MSA', '-MSA']:
            return df

        def get_msa_value(msa_str):
            try:
                if pd.isna(msa_str):
                    return None
                if isinstance(msa_str, str):
                    values = eval(msa_str)
                    return 1 if any(v == 1 for v in values) else 0
                return msa_str
            except:
                return None

        df['msa_value'] = df['af3_rna_MSA'].apply(get_msa_value)
        msa_filter = df['msa_value'] == (1 if msa_option == '+MSA' else 0)
        return df[msa_filter]

    def plot_domain_rmsd(self, single_chain_only=False):
        """Create bar chart of average RMSD by protein domain"""
        try:
            # Check if domain_names column exists
            ref_table = self.config.ref_table
            pred_table = self.config.pred_table
            check_query = f"""
            SELECT COUNT(*) 
            FROM pragma_table_info('{ref_table}') 
            WHERE name='domain_names'
            """
            cursor = self.database.connection.cursor()
            cursor.execute(check_query)
            has_domain = bool(cursor.fetchone()[0])

            if not has_domain:
                print(f"domain_names column not found in {ref_table} table")
                return

            # Get domain and RMSD data from SQLite
            # Add RBP template filter to query
            rbp_filter = ""
            if self.rbp_template_option and 'RBP_PDB_Templates' in self.pred_df.columns:
                rbp_upper = self.rbp_template_option.upper()
                if rbp_upper == '+RBPTEMP' or rbp_upper == '+RBP':
                    rbp_filter = "AND p.RBP_PDB_Templates IS NOT NULL AND p.RBP_PDB_Templates != '' AND p.RBP_PDB_Templates != '[]' AND p.RBP_PDB_Templates != 'null'"
                elif rbp_upper == '-RBPTEMP' or rbp_upper == '-RBP':
                    rbp_filter = "AND (p.RBP_PDB_Templates IS NULL OR p.RBP_PDB_Templates = '' OR p.RBP_PDB_Templates = '[]' OR p.RBP_PDB_Templates = 'null')"
            
            # Add MSA filter to query
            msa_filter = ""
            if self.msa_option == '+MSA':
                msa_filter = "AND (p.af3_rna_MSA = '[1]' OR p.af3_rna_MSA LIKE '[1,1%')"
            elif self.msa_option == '-MSA':
                msa_filter = "AND (p.af3_rna_MSA = '[0]' OR p.af3_rna_MSA LIKE '%0%')"
            
            query = f"""
            SELECT e.domain_names, p.RNA_RMSD
            FROM {ref_table} e
            JOIN {pred_table} p ON e.PDBId = p.exp_db_id
            WHERE e.domain_names IS NOT NULL 
            AND p.RNA_RMSD IS NOT NULL
            {rbp_filter}
            {msa_filter}
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                print("No domain and RMSD data found")
                return

            # Convert to DataFrame to apply MSA and single_chain filters (RBP already filtered in SQL)
            df = pd.DataFrame(rows, columns=['domain_names', 'RNA_RMSD'])
            
            # We need to join with pred_table to get MSA and ChainIDpairList_proteinRNA for filtering
            # But since we already have the data, let's filter by MSA if needed
            # For now, apply single_chain filter if needed (requires ChainIDpairList_proteinRNA)
            # Since we don't have that column, we'll skip single_chain filtering here
            # RBP filtering is already done in SQL
            
            # Apply MSA filter if needed
            if self.msa_option:
                # We need to get MSA data - let's query it separately or add it to the original query
                # For simplicity, let's add MSA to the query
                pass  # MSA filtering will be handled differently since we don't have MSA in this query

            # Process the data
            domain_rmsd_data = []
            for domain_names, rmsd in rows:
                if domain_names and rmsd:
                    # Split domain names
                    domains = domain_names.split(';')
                    # Get RMSD value using the extract_rmsd function
                    rmsd_value = self.extract_rmsd(rmsd)
                    if rmsd_value is not None:
                        # Add each domain with its RMSD
                        for domain in domains:
                            if domain.strip():  # Skip empty domains
                                domain_rmsd_data.append((domain.strip(), rmsd_value))

            if not domain_rmsd_data:
                print("No valid domain-RMSD pairs found")
                return

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(domain_rmsd_data, columns=['domain', 'rmsd'])

            # Calculate statistics for each domain
            domain_stats = df.groupby('domain').agg({
                'rmsd': ['mean', 'std', 'count']
            }).reset_index()
            domain_stats.columns = ['domain', 'mean', 'std', 'count']

            # Sort by mean RMSD
            domain_stats = domain_stats.sort_values('mean')

            # Create plot using PlotCreator
            plot_creator = PlotCreator('domain_metrics', self.msa_option, single_chain_only, self.rbp_template_option)

            # Create larger bar plot with adjusted dimensions
            plt.figure(figsize=(20, 10))  # Increased figure size
            bars = plt.bar(
                range(len(domain_stats)),
                domain_stats['mean'],
                yerr=domain_stats['std'],
                capsize=5
            )

            # Customize plot with larger text and more space
            plt.xticks(
                range(len(domain_stats)),
                domain_stats['domain'],
                rotation=90,
                ha='right',
                fontsize=12  # Increased font size
            )
            plt.xlabel('Protein Domains', fontsize=14, labelpad=10)  # Added padding
            plt.ylabel('Average RNA RMSD [Å]', fontsize=14, labelpad=10)

            # Remove count labels from bars
            # Adjust margins to prevent label cutoff
            plt.margins(x=0.01)  # Reduce horizontal margins
            plt.tight_layout()

            # Save plot
            plot_creator.save_plot('domain_rmsd_distribution')
            plt.close()

        except Exception as e:
            print(f"Error creating domain RMSD plot: {e}")

    def extract_rmsd(self, rmsd_str):
        """Extract first RMSD value from string or float"""
        if pd.isna(rmsd_str):
            return None
        try:
            if isinstance(rmsd_str, (int, float)):
                return float(rmsd_str)
            # Handle list format
            if isinstance(rmsd_str, str) and '[' in rmsd_str:
                values = [float(x.strip()) for x in rmsd_str.strip('[]').split(',')]
                return values[0]  # Take first value for RMSD
            return float(rmsd_str)
        except:
            print(f"Could not parse RMSD: {rmsd_str}")
            return None

    def plot_domain_coverage(self, single_chain_only=False):
        """Plot how well predicted interaction residues align with domain positions"""
        try:
            # Get domain mappings first
            ref_table = self.config.ref_table
            pred_table = self.config.pred_table
            mapping_query = f"""
            SELECT DISTINCT domain_ids, domain_names 
            FROM {ref_table} 
            WHERE domain_ids IS NOT NULL 
            AND domain_names IS NOT NULL
            """
            cursor = self.database.connection.cursor()
            cursor.execute(mapping_query)
            mapping_rows = cursor.fetchall()

            # Create domain ID to name mapping
            domain_map = {}
            for ids, names in mapping_rows:
                id_list = ids.split(';')
                name_list = names.split(';')
                if len(id_list) == len(name_list):
                    domain_map.update(dict(zip(id_list, name_list)))

            # Get domain and AAMotif data with MSA and RBP template columns
            # Add RBP template filter to query
            rbp_filter = ""
            if self.rbp_template_option and 'RBP_PDB_Templates' in self.pred_df.columns:
                rbp_upper = self.rbp_template_option.upper()
                if rbp_upper == '+RBPTEMP' or rbp_upper == '+RBP':
                    rbp_filter = "AND p.RBP_PDB_Templates IS NOT NULL AND p.RBP_PDB_Templates != '' AND p.RBP_PDB_Templates != '[]' AND p.RBP_PDB_Templates != 'null'"
                elif rbp_upper == '-RBPTEMP' or rbp_upper == '-RBP':
                    rbp_filter = "AND (p.RBP_PDB_Templates IS NULL OR p.RBP_PDB_Templates = '' OR p.RBP_PDB_Templates = '[]' OR p.RBP_PDB_Templates = 'null')"
            
            query = f"""
            SELECT e.PDBId, e.domain_ids, e.domain_pos, p.AAMotif, p.ChainIDpairList_proteinRNA, p.af3_rna_MSA
            FROM {ref_table} e
            JOIN {pred_table} p ON e.PDBId = p.exp_db_id
            WHERE e.domain_pos IS NOT NULL 
            AND p.AAMotif IS NOT NULL
            {rbp_filter}
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            # Convert to DataFrame for filtering
            df = pd.DataFrame(rows, columns=['PDBId', 'domain_ids', 'domain_pos', 'AAMotif',
                                             'ChainIDpairList_proteinRNA', 'af3_rna_MSA'])

            # Apply filters (MSA and single_chain, RBP already filtered in SQL)
            df = self.filter_data(df, single_chain_only)

            if df.empty:
                print("No data after filtering")
                return

            # Process filtered data
            rows = df.values.tolist()

            # Process the data
            domain_coverage_data = []
            domain_coverage_types = {}
            pdb_coverage_data = {}
            pdb_categories = {
                'all_in_all': 0,  # All residues in all domains (multiple domains only)
                'partial_in_all': 0,  # Some residues in all domains (multiple domains only)
                'all_in_one': 0,  # All residues in one domain (any number of domains)
                'partial_in_one': 0,  # Some residues in one domain (any number of domains)
                'outside_all': 0  # No residues in any domain
            }

            for pdb_id, domain_ids, domain_pos, aa_motif, chain_pairs, msa in rows:
                try:
                    # Parse domain positions
                    domain_ranges = {}
                    domain_id_list = domain_ids.split(';')
                    domain_pos_list = domain_pos.split(';')

                    # Skip entries with invalid domain positions
                    if all(pos.strip() == '-' for pos in domain_pos_list):
                        continue

                    for d_id, d_pos in zip(domain_id_list, domain_pos_list):
                        d_id = d_id.strip()
                        if d_pos.strip() == '-':
                            continue

                        try:
                            # Extract all ranges for this domain
                            ranges = []
                            if ':' in d_pos:
                                range_str = d_pos.split(':')[1]
                            else:
                                range_str = d_pos

                            # Handle multiple ranges separated by comma
                            for r in range_str.split(','):
                                start, end = map(int, r.strip().split('-'))
                                ranges.append((start, end))

                            domain_ranges[d_id] = ranges
                            # print(f"Processing {pdb_id}:")
                            # print(f"  Domain {d_id}: ranges {ranges}")

                        except ValueError:
                            continue

                    if not domain_ranges:  # Skip if no valid domain ranges found
                        continue

                    # Parse AAMotif indices
                    # try:
                    # print(f"\nProcessing {pdb_id}:")
                    # print(f"Raw AAMotif data: {aa_motif}")

                    if isinstance(aa_motif, str):
                        # Handle string representation of list
                        if aa_motif.startswith('[') and aa_motif.endswith(']'):
                            aa_list = literal_eval(aa_motif)
                        else:
                            aa_list = [aa_motif]
                    else:
                        aa_list = [str(aa_motif)]

                    if not isinstance(aa_list, list):
                        aa_list = [aa_list]

                    # Process each motif list separately
                    for motif_list in aa_list:
                        aa_indices = []
                        if isinstance(motif_list, str):
                            # Extract numbers between parentheses
                            numbers = re.findall(r'\((\d+)\)', motif_list)
                            if numbers:
                                aa_indices = [int(num) for num in numbers]
                                # print(f"  AAMotif residue indices: {aa_indices}")
                            else:
                                # print(f"  No residue indices found in: {motif_list}")
                                continue

                        if aa_indices:
                            # Track coverage for this PDB
                            domains_with_all = 0  # Domains containing all residues
                            domains_with_partial = 0  # Domains containing some residues
                            total_domains = len(domain_ranges)
                            max_coverage = 0  # Initialize max coverage for domain plots

                            for domain_id, ranges in domain_ranges.items():
                                # Initialize coverage type counters if not exists
                                if domain_id not in domain_coverage_types:
                                    domain_coverage_types[domain_id] = {
                                        'within': 0,
                                        'partial': 0,
                                        'outside': 0
                                    }

                                # Count residues within this domain
                                indices_within = sum(1 for idx in aa_indices
                                                     if any(start <= idx <= end
                                                            for start, end in ranges))

                                # Update domain coverage types (for existing plots)
                                if indices_within == len(aa_indices):
                                    domain_coverage_types[domain_id]['within'] += 1
                                    domains_with_all += 1
                                elif indices_within > 0:
                                    domain_coverage_types[domain_id]['partial'] += 1
                                    domains_with_partial += 1
                                else:
                                    domain_coverage_types[domain_id]['outside'] += 1

                                # Calculate coverage for domain plots
                                coverage = indices_within / len(aa_indices)
                                domain_coverage_data.append((domain_id, coverage))
                                max_coverage = max(max_coverage, coverage)

                            # Store maximum coverage for PDB scatter plot
                            pdb_coverage_data[pdb_id] = max_coverage

                            # Categorize this PDB sample
                            if total_domains > 1:  # Multiple domains
                                if domains_with_all == total_domains:
                                    pdb_categories['all_in_all'] += 1
                                elif domains_with_partial == total_domains:
                                    pdb_categories['partial_in_all'] += 1
                                elif domains_with_all > 0:
                                    pdb_categories['all_in_one'] += 1
                                elif domains_with_partial > 0:
                                    pdb_categories['partial_in_one'] += 1
                                else:
                                    pdb_categories['outside_all'] += 1
                            else:  # Single domain
                                if domains_with_all == 1:
                                    pdb_categories['all_in_one'] += 1
                                elif domains_with_partial == 1:
                                    pdb_categories['partial_in_one'] += 1
                                else:
                                    pdb_categories['outside_all'] += 1

                except Exception as e:
                    print(f"Error processing {pdb_id}: {e}")
                    continue

            if not domain_coverage_data:
                print("No valid domain coverage data found")
                return

            # Create plots
            plot_creator = PlotCreator('domain_metrics', self.msa_option, single_chain_only, self.rbp_template_option)

            # Create coverage distribution plot
            plt.figure(figsize=(20, 10))
            domains = list(domain_coverage_types.keys())
            # Map domain IDs to names
            domain_names = [domain_map.get(d, d) for d in domains]

            within_counts = [domain_coverage_types[d]['within'] for d in domains]
            partial_counts = [domain_coverage_types[d]['partial'] for d in domains]
            outside_counts = [domain_coverage_types[d]['outside'] for d in domains]

            # Calculate percentages
            totals = np.array(within_counts) + np.array(partial_counts) + np.array(outside_counts)
            within_pct = 100 * np.array(within_counts) / totals
            partial_pct = 100 * np.array(partial_counts) / totals
            outside_pct = 100 * np.array(outside_counts) / totals

            x = range(len(domains))
            plt.bar(x, within_pct, label='Within Domain', color='green')
            plt.bar(x, partial_pct, bottom=within_pct, label='Partially Within', color='yellow')
            plt.bar(x, outside_pct, bottom=within_pct + partial_pct, label='Outside Domain', color='red')

            # Use domain names instead of IDs
            plt.xticks(x, domain_names, rotation=90, ha='right', fontsize=12)
            plt.xlabel('Protein Domains', fontsize=14, labelpad=10)
            plt.ylabel('Percentage of Samples (%)', fontsize=14, labelpad=10)
            plt.legend()
            plt.margins(x=0.01)
            plt.tight_layout()
            plot_creator.save_plot('domain_coverage_distribution')
            plt.close()

            # Create domain interaction coverage plot
            plt.figure(figsize=(20, 10))
            df = pd.DataFrame(domain_coverage_data, columns=['domain_id', 'coverage'])

            # Map domain IDs to names in DataFrame - fixed inplace warning
            df = df.assign(
                domain_name=df['domain_id'].map(domain_map).fillna(df['domain_id'])
            )

            domain_stats = df.groupby('domain_name').agg({
                'coverage': ['mean', 'std', 'count']
            }).reset_index()
            domain_stats.columns = ['domain_id', 'mean', 'std', 'count']
            domain_stats = domain_stats.sort_values('mean', ascending=False)

            plt.bar(
                range(len(domain_stats)),
                domain_stats['mean'],
                yerr=domain_stats['std'],
                capsize=5
            )

            plt.xticks(
                range(len(domain_stats)),
                domain_stats['domain_id'],
                rotation=90,
                ha='right',
                fontsize=12
            )
            plt.xlabel('Protein Domains', fontsize=14, labelpad=10)
            plt.ylabel('Average Coverage of Predicted Interactions', fontsize=14, labelpad=10)
            plt.ylim(0, 1)
            plt.margins(x=0.01)
            plt.tight_layout()
            plot_creator.save_plot('domain_interaction_coverage')
            plt.close()

            # Create PDB coverage scatter plot
            plt.figure(figsize=(15, 8))
            pdb_ids = list(pdb_coverage_data.keys())
            coverages = list(pdb_coverage_data.values())

            plt.scatter(range(len(pdb_ids)), coverages, alpha=0.6)
            plt.xticks(
                range(len(pdb_ids)),
                pdb_ids,
                rotation=90,
                ha='right',
                fontsize=10
            )
            plt.xlabel('PDB ID', fontsize=12)
            plt.ylabel('Maximum Domain Coverage', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_creator.save_plot('pdb_domain_coverage')
            plt.close()

            # Add new plot for PDB categories
            plt.figure(figsize=(12, 8))
            categories = [
                'All residues\nin all domains\n(multiple domains)',
                'Partial residues\nin all domains\n(multiple domains)',
                'All residues in\none domain',
                'Partial residues in\none domain',
                'Outside all\ndomains'
            ]
            counts = [
                pdb_categories['all_in_all'],
                pdb_categories['partial_in_all'],
                pdb_categories['all_in_one'],
                pdb_categories['partial_in_one'],
                pdb_categories['outside_all']
            ]

            # Calculate percentages
            total_samples = sum(counts)
            percentages = [count / total_samples * 100 for count in counts]

            # Create bar plot
            bars = plt.bar(range(len(categories)), percentages)

            # Add count labels on top of bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'n={count}',
                         ha='center', va='bottom')

            plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
            # plt.xlabel('Coverage Category', fontsize=12)
            plt.ylabel('Percentage of Samples (%)', fontsize=12)
            # plt.title('Distribution of Domain Coverage Patterns', fontsize=14)
            plt.tight_layout()
            plot_creator.save_plot('pdb_coverage_categories')
            plt.close()

        except Exception as e:
            print(f"Error creating domain coverage plot: {e}")

    def filter_data(self, df, single_chain_only):
        """Filter data based on chain pairs, MSA option, and RBP template option"""
        filtered_df = df.copy()

        # Apply single chain filter if requested
        if single_chain_only:
            filtered_df = filtered_df[filtered_df['ChainIDpairList_proteinRNA'].apply(self.is_single_chain)]

        # Apply MSA filter if specified
        if self.msa_option == '+MSA':
            filtered_df = filtered_df[
                (filtered_df['af3_rna_MSA'] == '[1]') |
                (filtered_df['af3_rna_MSA'].str.startswith('[1,1', na=False))
                ]
        elif self.msa_option == '-MSA':
            filtered_df = filtered_df[
                (filtered_df['af3_rna_MSA'] == '[0]') |
                (filtered_df['af3_rna_MSA'].str.contains('0', na=False))
                ]

        # Apply RBP template filter if specified
        if self.rbp_template_option and 'RBP_PDB_Templates' in filtered_df.columns:
            if self.rbp_template_option.upper() == '+RBPTEMP' or self.rbp_template_option.upper() == '+RBP':
                # Filter for samples WITH RBP templates (non-empty)
                filtered_df = filtered_df[
                    (filtered_df['RBP_PDB_Templates'].notna()) &
                    (filtered_df['RBP_PDB_Templates'] != '') &
                    (filtered_df['RBP_PDB_Templates'] != '[]') &
                    (filtered_df['RBP_PDB_Templates'] != 'null')
                ]
            elif self.rbp_template_option.upper() == '-RBPTEMP' or self.rbp_template_option.upper() == '-RBP':
                # Filter for samples WITHOUT RBP templates (empty)
                filtered_df = filtered_df[
                    (filtered_df['RBP_PDB_Templates'].isna()) |
                    (filtered_df['RBP_PDB_Templates'] == '') |
                    (filtered_df['RBP_PDB_Templates'] == '[]') |
                    (filtered_df['RBP_PDB_Templates'] == 'null')
                ]

        return filtered_df

    def plot_rna_metrics(self, single_chain_only=False):
        """Create scatter plots for RNA metrics vs length and RMSD"""
        # Get filtered data first
        df = self.filter_data(self.pred_df, single_chain_only)

        # Create plot creator
        plot_creator = PlotCreator('rna_metrics', self.msa_option, single_chain_only, self.rbp_template_option)

        # Process RNA length values
        def extract_length(length_str):
            if pd.isna(length_str):
                return None
            try:
                if isinstance(length_str, (int, float)):
                    return int(length_str)
                if isinstance(length_str, str) and '[' in length_str:
                    values = [int(x.strip()) for x in length_str.strip('[]').split(',')]
                    return max(values)
                return int(length_str)
            except:
                return None

        # Process RMSD values
        def extract_rmsd(rmsd_str):
            if pd.isna(rmsd_str):
                return None
            try:
                if isinstance(rmsd_str, (int, float)):
                    return float(rmsd_str)
                if isinstance(rmsd_str, str) and '[' in rmsd_str:
                    values = [float(x.strip()) for x in rmsd_str.strip('[]').split(',')]
                    return values[0]
                return float(rmsd_str)
            except:
                return None

        # Apply transformations
        df['RNALength'] = df['RNALength'].apply(extract_length)
        df['RNA_RMSD'] = df['RNA_RMSD'].apply(extract_rmsd)

        # Drop rows with None values and convert to numeric
        df = df.dropna(subset=['RNALength', 'RNA_RMSD'])
        df['RNALength'] = pd.to_numeric(df['RNALength'])
        df['RNA_RMSD'] = pd.to_numeric(df['RNA_RMSD'])

        # Define metrics to plot
        metrics = {
            'F1-Score': 'RNA_f1_score',
            'Precision': 'RNA_precision',
            'Recall': 'RNA_recall',
            'MCC': 'RNA_mcc',
            'WL': 'RNA_wl'
        }

        # # Plot metrics vs length
        # for metric_name, metric_col in metrics.items():
        #     if not df[metric_col].isna().all():
        #         plot_creator.get_scatterplot(
        #             table_source='rna_metrics',
        #             xAxis_score=df['RNALength'],
        #             xAxis_label='RNA Length',
        #             yAxis_label=metric_name,
        #             name=f'{metric_col}_vs_length',
        #             yAxis_score=df[metric_col]
        #         )

        fig_length, axes_length = plt.subplots(2, 3, figsize=(15, 10))
        axes_length = axes_length.flatten()

        for idx, (metric_name, metric_col) in enumerate(metrics.items()):
            ax = axes_length[idx]
            plot_creator.scatter_plot(
                df['RNALength'],
                df[metric_col],
                ax=ax,
                xlabel='RNA Length',
                ylabel=metric_name
            )

        # Remove extra subplot
        if len(axes_length) > len(metrics):
            fig_length.delaxes(axes_length[-1])
        fig_length.tight_layout()

        # # Plot metrics vs RMSD
        # df_rmsd = df.dropna(subset=['RNA_RMSD'])
        # for metric_name, metric_col in metrics.items():
        #     if not df_rmsd[metric_col].isna().all():
        #         plot_creator.get_scatterplot(
        #             table_source='rna_metrics',
        #             xAxis_score=df_rmsd['RNA_RMSD'],
        #             xAxis_label='RNA RMSD',
        #             yAxis_label=metric_name,
        #             name=f'{metric_col}_vs_rmsd',
        #             yAxis_score=df_rmsd[metric_col]
        #         )

        fig_rmsd, axes_rmsd = plt.subplots(2, 3, figsize=(15, 10))
        axes_rmsd = axes_rmsd.flatten()

        # Filter out None RMSD values
        df_rmsd = df.dropna(subset=['RNA_RMSD'])

        for idx, (metric_name, metric_col) in enumerate(metrics.items()):
            ax = axes_rmsd[idx]
            plot_creator.scatter_plot(
                df_rmsd['RNA_RMSD'],
                df_rmsd[metric_col],
                ax=ax,
                xlabel='RNA RMSD',
                ylabel=metric_name
            )

        # Remove extra subplot
        if len(axes_rmsd) > len(metrics):
            fig_rmsd.delaxes(axes_rmsd[-1])

        fig_rmsd.tight_layout()

        plt.figure(fig_length.number)
        plt.savefig(os.path.join(plot_creator.results_dir, 'rna_metrics_vs_length.png'))

        plt.figure(fig_rmsd.number)
        plt.savefig(os.path.join(plot_creator.results_dir, 'rna_metrics_vs_rmsd.png'))

        # ========== Plot metrics vs additional numeric columns ==========
        extra_numeric_cols = [
            'RNA_LDDT_allAtom',
            'Complex_LDDT_allAtom',
            'iLDDT',
            'QS_global',
            'ICS'
        ]

        for col in extra_numeric_cols:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, (metric_name, metric_col) in enumerate(metrics.items()):
                ax = axes[idx]
                plot_creator.scatter_plot(
                    df[col],
                    df[metric_col],
                    ax=ax,
                    xlabel=col.replace('_', ' '),
                    ylabel=metric_name
                )

            if len(axes) > len(metrics):
                for i in range(len(metrics), len(axes)):
                    fig.delaxes(axes[i])

            fig.tight_layout()
            plt.figure(fig.number)
            plt.savefig(os.path.join(plot_creator.results_dir, f'{col}_vs_rna_metrics.png'))
            plt.close(fig)

        plt.close('all')

    def analyze_domain_relationships(self, single_chain_only=False):
        """Analyze relationships between domain counts and various metrics"""
        plot_creator = PlotCreator('domain_count', self.msa_option, single_chain_only, self.rbp_template_option)

        # Query to get required data
        ref_table = self.config.ref_table
        pred_table = self.config.pred_table
        
        # Add RBP template filter to query
        rbp_filter = ""
        if self.rbp_template_option and 'RBP_PDB_Templates' in self.pred_df.columns:
            rbp_upper = self.rbp_template_option.upper()
            if rbp_upper == '+RBPTEMP' or rbp_upper == '+RBP':
                rbp_filter = "AND p.RBP_PDB_Templates IS NOT NULL AND p.RBP_PDB_Templates != '' AND p.RBP_PDB_Templates != '[]' AND p.RBP_PDB_Templates != 'null'"
            elif rbp_upper == '-RBPTEMP' or rbp_upper == '-RBP':
                rbp_filter = "AND (p.RBP_PDB_Templates IS NULL OR p.RBP_PDB_Templates = '' OR p.RBP_PDB_Templates = '[]' OR p.RBP_PDB_Templates = 'null')"
        
        query = f"""
        SELECT p.Binding_affinity_kd, p.af3_rna_MSA, p.RNAmotif_score,
               p.ProteinRNAInterfaceArea, p.RNAMotifLength, e.domain_counts
        FROM {pred_table} p
        JOIN {ref_table} e ON p.exp_db_id = e.PDBId
        WHERE e.domain_counts IS NOT NULL
        {rbp_filter}
        """
        results = self.database.execute_query(query)

        if not results:
            print("No domain relationship data found after filtering")
            return

        # Convert to DataFrame to apply MSA and single_chain filters (RBP already filtered in SQL)
        df = pd.DataFrame(results, columns=['Binding_affinity_kd', 'af3_rna_MSA', 'RNAmotif_score',
                                              'ProteinRNAInterfaceArea', 'RNAMotifLength', 'domain_counts'])
        
        # Apply MSA and single_chain filters (RBP already filtered in SQL)
        # Note: We need to add a dummy column for filter_data to work properly
        if 'ChainIDpairList_proteinRNA' not in df.columns:
            # Add a dummy column if it doesn't exist (for single_chain filtering)
            df['ChainIDpairList_proteinRNA'] = ''
        
        df = self.filter_data(df, single_chain_only)
        
        if df.empty:
            print("No data after filtering")
            return

        # Convert back to list of tuples for processing
        results = [tuple(row) for row in df[['Binding_affinity_kd', 'af3_rna_MSA', 'RNAmotif_score',
                                              'ProteinRNAInterfaceArea', 'RNAMotifLength', 'domain_counts']].values]

        # Process functions remain the same...
        def process_domain_counts(x):
            if pd.notna(x):
                try:
                    counts = [int(count) for count in x.split(';') if count.isdigit()]
                    return sum(counts) if counts else None
                except:
                    return None
            return None

        def process_motif_length(x):
            try:
                if '[' in x:
                    lengths = ast.literal_eval(x)
                    return sum(lengths)
                else:
                    return int(x)
            except:
                return None

        def process_binding_affinity(x):
            try:
                if x and x.lower() != 'none':
                    value = float(x)
                    return np.log10(value) if value > 0 else None
                return None
            except:
                return None

        def process_msa_status(x):
            try:
                if isinstance(x, str):
                    msa_list = ast.literal_eval(x)
                    if len(msa_list) == 1:
                        return 2 if msa_list[0] == 1 else 0
                    elif all(v == 1 for v in msa_list):
                        return 2
                    elif all(v == 0 for v in msa_list):
                        return 0
                    elif 1 in msa_list:
                        return 1
                    return 0
                return 0
            except:
                return 0

        # Create separate paired data for each plot
        motif_score_data = {'domain_counts': [], 'motif_score': []}
        interface_area_data = {'domain_counts': [], 'interface_area': []}
        binding_affinity_data = {'domain_counts': [], 'binding_affinity': []}
        
        # Collect data for MSA analysis
        msa_domain_data = {'domain_counts': [], 'msa_status': []}

        # Process results and create paired data
        for row in results:
            domain_count = process_domain_counts(row[5])
            
            if domain_count is not None:
                # For motif score plot
                motif_score = row[2]
                if motif_score is not None:
                    motif_score_data['domain_counts'].append(domain_count)
                    motif_score_data['motif_score'].append(motif_score)
                
                # For interface area plot
                interface_area = row[3]
                if interface_area is not None:
                    interface_area_data['domain_counts'].append(domain_count)
                    interface_area_data['interface_area'].append(interface_area)
                
                # For binding affinity plot
                binding_aff = process_binding_affinity(row[0])
                if binding_aff is not None:
                    binding_affinity_data['domain_counts'].append(domain_count)
                    binding_affinity_data['binding_affinity'].append(binding_aff)
                
                # For MSA analysis
                msa_status = process_msa_status(row[1])
                if msa_status is not None:
                    msa_domain_data['domain_counts'].append(domain_count)
                    msa_domain_data['msa_status'].append(msa_status)

        # Create scatter plots with paired data
        fig_scatter, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # RNAmotif score vs domain count
        if motif_score_data['domain_counts']:
            plot_creator.scatter_plot(
                x=motif_score_data['domain_counts'],
                y=motif_score_data['motif_score'],
                ax=ax1,
                xlabel='Number of Domains',
                ylabel='RNA Motif Score'
            )

        # Interface area vs domain count
        if interface_area_data['domain_counts']:
            plot_creator.scatter_plot(
                x=interface_area_data['domain_counts'],
                y=interface_area_data['interface_area'],
                ax=ax2,
                xlabel='Number of Domains',
                ylabel='Protein-RNA Interface Area'
            )

        # Binding affinity vs domain count
        if binding_affinity_data['domain_counts']:
            min_aff = min(binding_affinity_data['binding_affinity'])
            ylabel = 'Log₁₀[Binding Affinity (M)]' if min_aff < 1e-10 else 'Binding Affinity (M)'
            plot_creator.scatter_plot(
                x=binding_affinity_data['domain_counts'],
                y=binding_affinity_data['binding_affinity'],
                ax=ax3,
                xlabel='Number of Domains',
                ylabel=ylabel
            )

        fig_scatter.tight_layout()
        plt.savefig(os.path.join(plot_creator.results_dir, 'domain_analysis_scatter_plots.png'))
        plt.close(fig_scatter)

        # Create boxplot for binding affinity vs domain count
        if binding_affinity_data['domain_counts']:
            unique_counts = sorted(set(binding_affinity_data['domain_counts']))
            affinity_by_count = {count: [] for count in unique_counts}
            for count, aff in zip(binding_affinity_data['domain_counts'], binding_affinity_data['binding_affinity']):
                affinity_by_count[count].append(aff)

            min_aff = min(min(vals) for vals in affinity_by_count.values() if vals)
            yAxis_label = 'Log₁₀[Binding Affinity (M)]' if min_aff < 1e-10 else 'Binding Affinity (M)'

            plot_creator.get_boxplot(
                table_source='domain_count',
                data_values=[affinity_by_count[count] for count in unique_counts],
                labels=[f'Domains: {count}' for count in unique_counts],
                name='binding_affinity_vs_domains',
                yAxis_label=yAxis_label,
                rotation=45
            )

        # Create segmented bar plot for MSA status vs domain count distribution
        if msa_domain_data['domain_counts']:
            msa_domain_counts = {
                'No RNA MSA': [],
                'Mixed RNA MSA': [],
                'Full RNA MSA': []
            }

            msa_labels = {0: 'No RNA MSA', 1: 'Mixed RNA MSA', 2: 'Full RNA MSA'}

            for msa, count in zip(msa_domain_data['msa_status'], msa_domain_data['domain_counts']):
                msa_domain_counts[msa_labels[msa]].append(count)

            segment_labels = sorted(set(msa_domain_data['domain_counts']))

            plot_creator.get_segmented_barplot(
                table_source='domain_count',
                data_dict=msa_domain_counts,
                category_labels=['No RNA MSA', 'Mixed RNA MSA', 'Full RNA MSA'],
                segment_labels=segment_labels,
                xlabel='MSA Status',
                ylabel='Number of Structures',
                name='domain_count_distribution_by_msa',
                rotation=0
            )


def _standardize_msa_option(msa_option):
    """Standardize MSA option to uppercase and ensure proper format"""
    if not msa_option:
        return None
    msa_option = msa_option.upper()
    if msa_option in ['+MSA', '-MSA', '+msa', '-msa']:
        # Check for minus sign explicitly
        return '-MSA' if msa_option.startswith('-') else '+MSA'
    return None


def _get_filter_info(plotter):
    """Helper function to get filter information string."""
    filter_parts = []
    if plotter.single_chain_only:
        filter_parts.append("Single-Chain")
    else:
        filter_parts.append("All-Chains")
    
    if plotter.msa_option:
        filter_parts.append(plotter.msa_option)
    
    if plotter.rbp_template_option:
        # Format RBP template option nicely
        if plotter.rbp_template_option.upper() == '+RBPTEMP':
            filter_parts.append("+RBP-template")
        elif plotter.rbp_template_option.upper() == '-RBPTEMP':
            filter_parts.append("-RBP-template")
        else:
            filter_parts.append(plotter.rbp_template_option)
    
    return ", ".join(filter_parts) if filter_parts else "None"

def write_af3_statistics(plotter, single_chain_only):
    """Write statistics for AF3 metrics and global PAE metrics."""
    # Get the correct results directory for af3_metrics plots
    plot_creator = PlotCreator('af3_metrics', plotter.msa_option, single_chain_only, plotter.rbp_template_option)
    output_path = os.path.join(plot_creator.results_dir, 'af3_metrics_statistics.txt')
    
    df = plotter.filter_data(plotter.pred_df, single_chain_only)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AF3 Metrics and Global PAE Metrics Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        filter_info = _get_filter_info(plotter)
        f.write(f"Selected Samples by: {filter_info}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total structures analyzed: {len(df)}\n\n")
        
        # AF3 metrics statistics
        af3_metrics = {
            'af3_ranking_score': 'AF3 Ranking Score',
            'af3_rna_pLDDT_avg': 'RNA pLDDT Average',
            'af3_chain_pair_pae_min': 'Chain-Pair PAE Minimum',
            'af3_rna_ipTM': 'RNA ipTM',
            'af3_rna_pTM': 'RNA pTM'
        }
        
        f.write("AF3 Metrics:\n")
        f.write("-" * 80 + "\n")
        for col, label in af3_metrics.items():
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    f.write(f"{label}:\n")
                    f.write(f"  Count: {len(values)}\n")
                    f.write(f"  Mean: {values.mean():.4f}\n")
                    f.write(f"  Std: {values.std():.4f}\n")
                    f.write(f"  Median: {values.median():.4f}\n")
                    f.write(f"  Min: {values.min():.4f}\n")
                    f.write(f"  Max: {values.max():.4f}\n\n")
        
        # Global PAE statistics
        if 'af3_global_pae_avg' in df.columns:
            pae_values = pd.to_numeric(df['af3_global_pae_avg'], errors='coerce').dropna()
            if len(pae_values) > 0:
                f.write("Global PAE Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Count: {len(pae_values)}\n")
                f.write(f"  Mean: {pae_values.mean():.4f}\n")
                f.write(f"  Std: {pae_values.std():.4f}\n")
                f.write(f"  Median: {pae_values.median():.4f}\n")
                f.write(f"  Min: {pae_values.min():.4f}\n")
                f.write(f"  Max: {pae_values.max():.4f}\n\n")

        f.write("AF3 metrics vs structural outcomes (Spearman correlation)\n")
        f.write("-" * 80 + "\n")
        f.write(
            "For each structural column present in the filtered table, AF3 predictors are ranked by "
            "|Spearman rho| (higher = stronger monotonic association). ipTM/pTM list columns use "
            "max / mean over values like the metric plots.\n\n"
        )

        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None

        if spearmanr is None:
            f.write("(scipy not installed; skipping correlation table.)\n")
        else:
            def _af3_list_mean(metric_str):
                if pd.isna(metric_str):
                    return np.nan
                try:
                    if isinstance(metric_str, str):
                        if '[' in metric_str:
                            values = ast.literal_eval(metric_str)
                            return float(np.mean(values))
                        return float(metric_str)
                    if isinstance(metric_str, (list, tuple, np.ndarray)):
                        return float(np.mean(metric_str))
                    return float(metric_str)
                except Exception:
                    return np.nan

            def _af3_list_max(metric_str):
                if pd.isna(metric_str):
                    return np.nan
                try:
                    if isinstance(metric_str, str):
                        if '[' in metric_str:
                            values = ast.literal_eval(metric_str)
                            return float(np.max(values))
                        return float(metric_str)
                    if isinstance(metric_str, (list, tuple, np.ndarray)):
                        return float(np.max(metric_str))
                    return float(metric_str)
                except Exception:
                    return np.nan

            def _structural_y(col_name):
                if col_name not in df.columns:
                    return None
                s = df[col_name]
                if col_name == 'RNA_RMSD':
                    def _first_rna_rmsd(val):
                        if pd.isna(val):
                            return np.nan
                        try:
                            if isinstance(val, str):
                                if '[' in val:
                                    values = ast.literal_eval(val)
                                    if isinstance(values, list) and len(values) > 0:
                                        return float(values[0])
                                return float(val)
                            if isinstance(val, (list, tuple)):
                                return float(val[0]) if len(val) > 0 else np.nan
                            return float(val)
                        except Exception:
                            return np.nan
                    return s.apply(_first_rna_rmsd)
                return pd.to_numeric(s, errors='coerce')

            skip_af3 = {'af3_rna_MSA'}
            af3_cols = sorted(c for c in df.columns if c.startswith('af3_') and c not in skip_af3)
            pred_series = {}
            for c in af3_cols:
                if c in ('af3_rna_pTM', 'af3_protein_pTM'):
                    pred_series[c] = df[c].apply(_af3_list_mean)
                elif c in ('af3_rna_ipTM', 'af3_protein_ipTM'):
                    pred_series[c] = df[c].apply(_af3_list_max)
                else:
                    pred_series[c] = pd.to_numeric(df[c], errors='coerce')

            structural_targets = [
                'RNA_TM', 'Complex_TM', 'Protein_TM',
                'RNA_RMSD', 'Complex_RMSD', 'Protein_RMSD',
                'RNA_LDDT_allAtom', 'Complex_LDDT_allAtom', 'Protein_LDDT_allAtom',
                'iLDDT', 'RNAmotif_score', 'AAmatch_score',
                'Binding_affinity_kd',
                'RNA_GC_Content', 'RNA_SequenceComplexity',
                'RNA_Shannon_Entropy_K2', 'RNA_MFE_Value',
            ]

            for target in structural_targets:
                y = _structural_y(target)
                if y is None:
                    continue
                rows = []
                for pc, xser in pred_series.items():
                    mask = ~(xser.isna() | y.isna())
                    n = int(mask.sum())
                    if n < 3:
                        continue
                    rho, p = spearmanr(xser[mask].astype(float), y[mask].astype(float))
                    if np.isnan(rho):
                        continue
                    rows.append((pc, float(rho), float(p), n))
                if not rows:
                    continue
                rows.sort(key=lambda t: abs(t[1]), reverse=True)
                best = rows[0]
                best_pos = max((t for t in rows if t[1] > 0), key=lambda t: t[1], default=None)
                best_neg = min((t for t in rows if t[1] < 0), key=lambda t: t[1], default=None)
                f.write(f"{target} (n pairs for best row: {best[3]}):\n")
                f.write(
                    f"  Strongest by |rho|: {best[0]}  rho={best[1]:+.4f}  p={best[2]:.4e}\n"
                )
                if best_pos:
                    f.write(
                        f"  Strongest positive: {best_pos[0]}  rho={best_pos[1]:+.4f}  p={best_pos[2]:.4e}  n={best_pos[3]}\n"
                    )
                if best_neg:
                    f.write(
                        f"  Strongest negative: {best_neg[0]}  rho={best_neg[1]:+.4f}  p={best_neg[2]:.4e}  n={best_neg[3]}\n"
                    )
                f.write("  Full ranking:\n")
                for pc, rho, p, n in rows:
                    f.write(f"    {pc:34s}  rho={rho:+.4f}  p={p:.4e}  n={n}\n")
                f.write("\n")

    print(f"Statistics written to: {output_path}")

def write_bond_statistics(plotter, single_chain_only):
    """Write statistics for bond metrics."""
    # Get the correct results directory for bond_metrics plots
    plot_creator = PlotCreator('bond_metrics', plotter.msa_option, single_chain_only, plotter.rbp_template_option)
    output_path = os.path.join(plot_creator.results_dir, 'bond_metrics_statistics.txt')
    
    df = plotter.filter_data(plotter.merged_df, single_chain_only)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Bond Metrics Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        filter_info = _get_filter_info(plotter)
        f.write(f"Selected Samples by: {filter_info}\n\n")
        
        # Glycosidic bond statistics - calculate actual counts
        if 'RNA_GlycosidicBond_pred' in df.columns and 'RNA_GlycosidicBond_exp' in df.columns:
            f.write("Glycosidic Bond Statistics:\n")
            f.write("-" * 80 + "\n")
            categories = ['anti', 'syn']
            chi_counts_pred = {cat: 0 for cat in categories}
            chi_counts_exp = {cat: 0 for cat in categories}
            
            for chi_dict_str in df['RNA_GlycosidicBond_pred'].dropna():
                counts = plotter.parse_dict_string(chi_dict_str)
                for cat in categories:
                    chi_counts_pred[cat] += counts.get(cat, 0)
            
            for chi_dict_str in df['RNA_GlycosidicBond_exp'].dropna():
                counts = plotter.parse_dict_string(chi_dict_str)
                for cat in categories:
                    chi_counts_exp[cat] += counts.get(cat, 0)
            
            f.write("Predicted:\n")
            for cat in categories:
                f.write(f"  {cat}: {chi_counts_pred[cat]}\n")
            f.write(f"  Total: {sum(chi_counts_pred.values())}\n\n")
            
            f.write("Experimental:\n")
            for cat in categories:
                f.write(f"  {cat}: {chi_counts_exp[cat]}\n")
            f.write(f"  Total: {sum(chi_counts_exp.values())}\n\n")
            
            # Chi-square test for glycosidic bond
            from scipy.stats import chi2_contingency
            pred_array = [chi_counts_pred[cat] for cat in categories]
            exp_array = [chi_counts_exp[cat] for cat in categories]
            contingency_table = np.array([pred_array, exp_array])
            if np.sum(contingency_table) > 0:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    # Determine significance stars
                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    f.write("Chi-square test (Predicted vs Experimental):\n")
                    f.write(f"  Chi-square statistic: {chi2:.4f}{stars}\n")
                    f.write(f"  p-value: {p_value:.4e}\n")
                    f.write(f"  Degrees of freedom: {dof}\n")
                    if stars:
                        f.write(f"  Significance: {stars} (p < 0.05: *, p < 0.01: **, p < 0.001: ***)\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Chi-square test could not be performed: {e}\n\n")
        
        # Sugar pucker statistics - calculate actual counts
        if 'RNA_SugarPucker_pred' in df.columns and 'RNA_SugarPucker_exp' in df.columns:
            f.write("Sugar Pucker Statistics:\n")
            f.write("-" * 80 + "\n")
            categories = ["C3'-endo", "C2'-endo", "C3'-exo", "C4'-exo", "C2'-exo", "O4'-endo", "O4'-exo"]
            pucker_counts_pred = {cat: 0 for cat in categories}
            pucker_counts_exp = {cat: 0 for cat in categories}
            
            for pucker_dict_str in df['RNA_SugarPucker_pred'].dropna():
                counts = plotter.parse_dict_string(pucker_dict_str)
                for cat in categories:
                    pucker_counts_pred[cat] += counts.get(cat, 0)
            
            for pucker_dict_str in df['RNA_SugarPucker_exp'].dropna():
                counts = plotter.parse_dict_string(pucker_dict_str)
                for cat in categories:
                    pucker_counts_exp[cat] += counts.get(cat, 0)
            
            f.write("Predicted:\n")
            for cat in categories:
                f.write(f"  {cat}: {pucker_counts_pred[cat]}\n")
            f.write(f"  Total: {sum(pucker_counts_pred.values())}\n\n")
            
            f.write("Experimental:\n")
            for cat in categories:
                f.write(f"  {cat}: {pucker_counts_exp[cat]}\n")
            f.write(f"  Total: {sum(pucker_counts_exp.values())}\n\n")
            
            # Chi-square test for sugar pucker
            from scipy.stats import chi2_contingency
            pred_array = [pucker_counts_pred[cat] for cat in categories]
            exp_array = [pucker_counts_exp[cat] for cat in categories]
            contingency_table = np.array([pred_array, exp_array])
            if np.sum(contingency_table) > 0:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    # Determine significance stars
                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    f.write("Chi-square test (Predicted vs Experimental):\n")
                    f.write(f"  Chi-square statistic: {chi2:.4f}{stars}\n")
                    f.write(f"  p-value: {p_value:.4e}\n")
                    f.write(f"  Degrees of freedom: {dof}\n")
                    if stars:
                        f.write(f"  Significance: {stars} (p < 0.05: *, p < 0.01: **, p < 0.001: ***)\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Chi-square test could not be performed: {e}\n\n")
        
        # Gamma angle statistics - calculate actual counts
        if 'RNA_GammaAngle_pred' in df.columns and 'RNA_GammaAngle_exp' in df.columns:
            f.write("Gamma Angle Statistics:\n")
            f.write("-" * 80 + "\n")
            categories = ['g+', 'g-', 't']
            gamma_counts_pred = {cat: 0 for cat in categories}
            gamma_counts_exp = {cat: 0 for cat in categories}
            
            for gamma_dict_str in df['RNA_GammaAngle_pred'].dropna():
                counts = plotter.parse_dict_string(gamma_dict_str)
                for cat in categories:
                    gamma_counts_pred[cat] += counts.get(cat, 0)
            
            for gamma_dict_str in df['RNA_GammaAngle_exp'].dropna():
                counts = plotter.parse_dict_string(gamma_dict_str)
                for cat in categories:
                    gamma_counts_exp[cat] += counts.get(cat, 0)
            
            f.write("Predicted:\n")
            for cat in categories:
                f.write(f"  {cat}: {gamma_counts_pred[cat]}\n")
            f.write(f"  Total: {sum(gamma_counts_pred.values())}\n\n")
            
            f.write("Experimental:\n")
            for cat in categories:
                f.write(f"  {cat}: {gamma_counts_exp[cat]}\n")
            f.write(f"  Total: {sum(gamma_counts_exp.values())}\n\n")
            
            # Chi-square test for gamma angle
            from scipy.stats import chi2_contingency
            pred_array = [gamma_counts_pred[cat] for cat in categories]
            exp_array = [gamma_counts_exp[cat] for cat in categories]
            contingency_table = np.array([pred_array, exp_array])
            if np.sum(contingency_table) > 0:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    # Determine significance stars
                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                    f.write("Chi-square test (Predicted vs Experimental):\n")
                    f.write(f"  Chi-square statistic: {chi2:.4f}{stars}\n")
                    f.write(f"  p-value: {p_value:.4e}\n")
                    f.write(f"  Degrees of freedom: {dof}\n")
                    if stars:
                        f.write(f"  Significance: {stars} (p < 0.05: *, p < 0.01: **, p < 0.001: ***)\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Chi-square test could not be performed: {e}\n\n")
        
        # Torsion angle average statistics
        if hasattr(plotter, 'torsion_angle_averages') and plotter.torsion_angle_averages:
            f.write("Torsion Angle Average Statistics:\n")
            f.write("-" * 80 + "\n")
            torsion_labels = {
                'alpha': 'α (Alpha)',
                'beta': 'β (Beta)',
                'gamma': 'γ (Gamma)',
                'delta': 'δ (Delta)',
                'epsilon': 'ε (Epsilon)',
                'zeta': 'ζ (Zeta)',
                'chi': 'χ (Chi)'
            }
            torsion_order = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
            
            for angle_name in torsion_order:
                if angle_name in plotter.torsion_angle_averages:
                    data = plotter.torsion_angle_averages[angle_name]
                    f.write(f"{torsion_labels[angle_name]}:\n")
                    f.write(f"  All MSA:\n")
                    f.write(f"    Predicted: {data['pred_all']:.2f}°\n")
                    f.write(f"    Experimental: {data['exp_all']:.2f}°\n")
                    f.write(f"  +rMSA:\n")
                    if data['pred_msa_plus'] is not None:
                        f.write(f"    Predicted: {data['pred_msa_plus']:.2f}°\n")
                    if data['exp_msa_plus'] is not None:
                        f.write(f"    Experimental: {data['exp_msa_plus']:.2f}°\n")
                    f.write(f"  -rMSA:\n")
                    if data['pred_msa_minus'] is not None:
                        f.write(f"    Predicted: {data['pred_msa_minus']:.2f}°\n")
                    if data['exp_msa_minus'] is not None:
                        f.write(f"    Experimental: {data['exp_msa_minus']:.2f}°\n")
                    f.write("\n")
        
        # Torsion angle accuracy statistics (divided by +/-rMSA)
        if (hasattr(plotter, 'torsion_angle_accuracies_all') and plotter.torsion_angle_accuracies_all) or \
           (hasattr(plotter, 'torsion_angle_accuracies_msa') and plotter.torsion_angle_accuracies_msa) or \
           (hasattr(plotter, 'torsion_angle_accuracies_no_msa') and plotter.torsion_angle_accuracies_no_msa):
            f.write("Torsion Angle Accuracy Statistics:\n")
            f.write("-" * 80 + "\n")
            torsion_labels = {
                'alpha': 'α (Alpha)',
                'beta': 'β (Beta)',
                'gamma': 'γ (Gamma)',
                'delta': 'δ (Delta)',
                'epsilon': 'ε (Epsilon)',
                'zeta': 'ζ (Zeta)',
                'chi': 'χ (Chi)'
            }
            torsion_order = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
            
            for angle_name in torsion_order:
                has_all = hasattr(plotter, 'torsion_angle_accuracies_all') and angle_name in plotter.torsion_angle_accuracies_all
                has_msa = hasattr(plotter, 'torsion_angle_accuracies_msa') and angle_name in plotter.torsion_angle_accuracies_msa
                has_no_msa = hasattr(plotter, 'torsion_angle_accuracies_no_msa') and angle_name in plotter.torsion_angle_accuracies_no_msa
                
                if has_all or has_msa or has_no_msa:
                    f.write(f"{torsion_labels[angle_name]}:\n")
                    if has_all:
                        acc_all = plotter.torsion_angle_accuracies_all[angle_name]
                        err_all = plotter.torsion_angle_errors_all.get(angle_name, 0)
                        f.write(f"  All MSA: {acc_all:.2f}% ± {err_all:.2f}%\n")
                    if has_msa:
                        acc_msa = plotter.torsion_angle_accuracies_msa[angle_name]
                        err_msa = plotter.torsion_angle_errors_msa.get(angle_name, 0)
                        f.write(f"  +rMSA: {acc_msa:.2f}% ± {err_msa:.2f}%\n")
                    if has_no_msa:
                        acc_no_msa = plotter.torsion_angle_accuracies_no_msa[angle_name]
                        err_no_msa = plotter.torsion_angle_errors_no_msa.get(angle_name, 0)
                        f.write(f"  -rMSA: {acc_no_msa:.2f}% ± {err_no_msa:.2f}%\n")
                    f.write("\n")
        
        # Bond metrics accuracy by MSA statistics
        if hasattr(plotter, 'bond_metrics_accuracy_data') and plotter.bond_metrics_accuracy_data:
            f.write("Bond Metrics Classification Accuracy by rMSA Status:\n")
            f.write("-" * 80 + "\n")
            data = plotter.bond_metrics_accuracy_data
            categories = data['categories']
            accuracy_all = data.get('accuracy_all', [])
            accuracy_msa = data['accuracy_msa']
            accuracy_no_msa = data['accuracy_no_msa']
            errors_all = data.get('errors_all', [])
            errors_msa = data['errors_msa']
            errors_no_msa = data['errors_no_msa']
            
            for i, cat in enumerate(categories):
                f.write(f"{cat}:\n")
                if i < len(accuracy_all) and accuracy_all[i] is not None:
                    f.write(f"  All MSA: {accuracy_all[i]:.2f}% ± {errors_all[i]:.2f}%\n")
                f.write(f"  +rMSA: {accuracy_msa[i]:.2f}% ± {errors_msa[i]:.2f}%\n")
                f.write(f"  -rMSA: {accuracy_no_msa[i]:.2f}% ± {errors_no_msa[i]:.2f}%\n\n")
    
    print(f"Statistics written to: {output_path}")

def write_rna_statistics(plotter, single_chain_only):
    """Write statistics for RNA metrics."""
    # Get the correct results directory for rna_metrics plots
    plot_creator = PlotCreator('rna_metrics', plotter.msa_option, single_chain_only, plotter.rbp_template_option)
    output_path = os.path.join(plot_creator.results_dir, 'rna_metrics_statistics.txt')
    
    df = plotter.filter_data(plotter.pred_df, single_chain_only)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RNA Metrics Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        filter_info = _get_filter_info(plotter)
        f.write(f"Selected Samples by: {filter_info}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total structures analyzed: {len(df)}\n\n")
        
        # RNA metrics for correlation analysis
        rna_metrics = {
            'RNA_f1_score': 'F1-Score',
            'RNA_precision': 'Precision',
            'RNA_recall': 'Recall',
            'RNA_mcc': 'MCC',
            'RNA_wl': 'WL'
        }
        
        # Metrics to correlate with (check availability in dataframe)
        correlation_metrics = {}
        if 'RNA_LDDT_allAtom' in df.columns:
            correlation_metrics['RNA_LDDT_allAtom'] = 'RNA LDDT All-Atom'
        if 'RNA_RMSD' in df.columns:
            correlation_metrics['RNA_RMSD'] = 'RMSD'
        if 'iLDDT' in df.columns:
            correlation_metrics['iLDDT'] = 'iLDDT'
        if 'Complex_LDDT_allAtom' in df.columns:
            correlation_metrics['Complex_LDDT_allAtom'] = 'Complex LDDT All-Atom'
        if 'RNALength' in df.columns:
            correlation_metrics['RNALength'] = 'RNA Length'
        
        f.write("Correlation Analysis:\n")
        f.write("-" * 80 + "\n")
        f.write("Correlations between RNA metrics (F1-Score, Precision, Recall, MCC, WL) and quality metrics:\n")
        f.write("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001\n\n")
        
        from scipy.stats import pearsonr, spearmanr
        
        # Process RNA length
        def extract_length(length_str):
            if pd.isna(length_str):
                return None
            try:
                if isinstance(length_str, (int, float)):
                    return int(length_str)
                if isinstance(length_str, str) and '[' in length_str:
                    values = [int(x.strip()) for x in length_str.strip('[]').split(',')]
                    return max(values)
                return int(length_str)
            except:
                return None
        
        # Process RMSD
        def extract_rmsd(rmsd_str):
            if pd.isna(rmsd_str):
                return None
            try:
                if isinstance(rmsd_str, (int, float)):
                    return float(rmsd_str)
                if isinstance(rmsd_str, str) and '[' in rmsd_str:
                    values = [float(x.strip()) for x in rmsd_str.strip('[]').split(',')]
                    return values[0]
                return float(rmsd_str)
            except:
                return None
        
        # Prepare data
        df_corr = df.copy()
        if 'RNALength' in df_corr.columns:
            df_corr['RNALength'] = df_corr['RNALength'].apply(extract_length)
        if 'RNA_RMSD' in df_corr.columns:
            df_corr['RNA_RMSD'] = df_corr['RNA_RMSD'].apply(extract_rmsd)
        
        # Calculate correlations
        for rna_metric_col, rna_metric_label in rna_metrics.items():
            if rna_metric_col not in df_corr.columns:
                continue
            
            rna_values = pd.to_numeric(df_corr[rna_metric_col], errors='coerce')
            
            f.write(f"{rna_metric_label}:\n")
            for corr_metric_col, corr_metric_label in correlation_metrics.items():
                if corr_metric_col not in df_corr.columns:
                    continue
                
                corr_values = pd.to_numeric(df_corr[corr_metric_col], errors='coerce')
                
                # Get valid pairs
                valid_mask = rna_values.notna() & corr_values.notna()
                if valid_mask.sum() >= 2:
                    rna_valid = rna_values[valid_mask]
                    corr_valid = corr_values[valid_mask]
                    
                    try:
                        pearson_corr, pearson_p = pearsonr(rna_valid, corr_valid)
                        spearman_corr, spearman_p = spearmanr(rna_valid, corr_valid)
                        
                        # Determine significance stars for Pearson
                        if pearson_p < 0.001:
                            pearson_stars = "***"
                        elif pearson_p < 0.01:
                            pearson_stars = "**"
                        elif pearson_p < 0.05:
                            pearson_stars = "*"
                        else:
                            pearson_stars = ""
                        
                        # Determine significance stars for Spearman
                        if spearman_p < 0.001:
                            spearman_stars = "***"
                        elif spearman_p < 0.01:
                            spearman_stars = "**"
                        elif spearman_p < 0.05:
                            spearman_stars = "*"
                        else:
                            spearman_stars = ""
                        
                        f.write(f"  vs {corr_metric_label}:\n")
                        f.write(f"    Pearson: r={pearson_corr:.4f}{pearson_stars}, p={pearson_p:.4e} (n={len(rna_valid)})\n")
                        f.write(f"    Spearman: ρ={spearman_corr:.4f}{spearman_stars}, p={spearman_p:.4e}\n")
                    except Exception as e:
                        f.write(f"  vs {corr_metric_label}: Could not calculate correlation\n")
            f.write("\n")
    
    print(f"Statistics written to: {output_path}")

def write_interface_statistics(plotter, single_chain_only):
    """Write statistics for interface metrics."""
    # Get the correct results directory for binding_affinity plots
    plot_creator = PlotCreator('binding_affinity', plotter.msa_option, single_chain_only, plotter.rbp_template_option)
    output_path = os.path.join(plot_creator.results_dir, 'interface_metrics_statistics.txt')
    
    df = plotter.filter_chains(plotter.merged_df, True)  # Always use single_chain=True
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Interface Metrics Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        filter_info = _get_filter_info(plotter)
        f.write(f"Selected Samples by: {filter_info}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total structures analyzed: {len(df)}\n\n")
        
        # Free energy statistics
        if 'Free_energy_pred' in df.columns and 'Free_energy_exp' in df.columns:
            f.write("Free Energy Statistics:\n")
            f.write("-" * 80 + "\n")
            pred_values = pd.to_numeric(df['Free_energy_pred'], errors='coerce').dropna()
            exp_values = pd.to_numeric(df['Free_energy_exp'], errors='coerce').dropna()
            if len(pred_values) > 0:
                f.write("Predicted:\n")
                f.write(f"  Count: {len(pred_values)}\n")
                f.write(f"  Mean: {pred_values.mean():.4f}\n")
                f.write(f"  Std: {pred_values.std():.4f}\n")
                f.write(f"  Median: {pred_values.median():.4f}\n\n")
            if len(exp_values) > 0:
                f.write("Experimental:\n")
                f.write(f"  Count: {len(exp_values)}\n")
                f.write(f"  Mean: {exp_values.mean():.4f}\n")
                f.write(f"  Std: {exp_values.std():.4f}\n")
                f.write(f"  Median: {exp_values.median():.4f}\n\n")
        
        # Binding affinity statistics
        if 'Binding_affinity_kd_pred' in df.columns and 'Binding_affinity_kd_exp' in df.columns:
            f.write("Binding Affinity Statistics:\n")
            f.write("-" * 80 + "\n")
            pred_count = df['Binding_affinity_kd_pred'].notna().sum()
            exp_count = df['Binding_affinity_kd_exp'].notna().sum()
            f.write(f"Structures with predicted data: {pred_count}\n")
            f.write(f"Structures with experimental data: {exp_count}\n\n")
    
    print(f"Statistics written to: {output_path}")

def write_domain_statistics(plotter, single_chain_only):
    """Write combined statistics for all domain-related plots."""
    # Get the correct results directory for domain_metrics plots
    plot_creator = PlotCreator('domain_metrics', plotter.msa_option, single_chain_only, plotter.rbp_template_option)
    output_path = os.path.join(plot_creator.results_dir, 'domain_metrics_statistics.txt')
    
    ref_table = plotter.config.ref_table
    pred_table = plotter.config.pred_table
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Domain Metrics Statistics (Combined)\n")
        f.write("=" * 80 + "\n\n")
        
        filter_info = _get_filter_info(plotter)
        f.write(f"Selected Samples by: {filter_info}\n\n")
        
        # Domain coverage distribution statistics
        # Recalculate domain coverage distribution (similar to plot_domain_coverage)
        query_coverage = f"""
        SELECT e.PDBId, e.domain_ids, e.domain_pos, p.AAMotif, p.ChainIDpairList_proteinRNA
        FROM {ref_table} e
        JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        WHERE e.domain_pos IS NOT NULL 
        AND p.AAMotif IS NOT NULL
        """
        try:
            cursor = plotter.database.connection.cursor()
            cursor.execute(query_coverage)
            rows = cursor.fetchall()
            
            # Convert to DataFrame for filtering
            df_coverage = pd.DataFrame(rows, columns=['PDBId', 'domain_ids', 'domain_pos', 'AAMotif', 'ChainIDpairList_proteinRNA'])
            df_coverage = plotter.filter_data(df_coverage, single_chain_only)
            
            if not df_coverage.empty:
                domain_coverage_types = {}
                total_samples = 0
                
                for _, row in df_coverage.iterrows():
                    pdb_id = row['PDBId']
                    domain_ids = row['domain_ids']
                    domain_pos = row['domain_pos']
                    aa_motif = row['AAMotif']
                    
                    try:
                        if pd.isna(domain_ids) or pd.isna(domain_pos) or pd.isna(aa_motif):
                            continue
                        
                        # Parse domain positions
                        domain_ranges = {}
                        domain_id_list = domain_ids.split(';')
                        domain_pos_list = domain_pos.split(';')
                        
                        if all(pos.strip() == '-' for pos in domain_pos_list):
                            continue
                        
                        for d_id, d_pos in zip(domain_id_list, domain_pos_list):
                            d_id = d_id.strip()
                            if d_pos.strip() == '-':
                                continue
                            
                            try:
                                ranges = []
                                if ':' in d_pos:
                                    range_str = d_pos.split(':')[1]
                                else:
                                    range_str = d_pos
                                
                                for r in range_str.split(','):
                                    start, end = map(int, r.strip().split('-'))
                                    ranges.append((start, end))
                                
                                domain_ranges[d_id] = ranges
                            except ValueError:
                                continue
                        
                        if not domain_ranges:
                            continue
                        
                        # Parse AAMotif indices
                        if isinstance(aa_motif, str):
                            if aa_motif.startswith('[') and aa_motif.endswith(']'):
                                aa_list = literal_eval(aa_motif)
                            else:
                                aa_list = [aa_motif]
                        else:
                            aa_list = [str(aa_motif)]
                        
                        if not isinstance(aa_list, list):
                            aa_list = [aa_list]
                        
                        # Process each motif list
                        for motif_list in aa_list:
                            aa_indices = []
                            if isinstance(motif_list, str):
                                numbers = re.findall(r'\((\d+)\)', motif_list)
                                if numbers:
                                    aa_indices = [int(num) for num in numbers]
                                else:
                                    continue
                            
                            if aa_indices:
                                total_samples += 1
                                for domain_id, ranges in domain_ranges.items():
                                    if domain_id not in domain_coverage_types:
                                        domain_coverage_types[domain_id] = {
                                            'within': 0,
                                            'partial': 0,
                                            'outside': 0
                                        }
                                    
                                    indices_within = sum(1 for idx in aa_indices
                                                         if any(start <= idx <= end
                                                                for start, end in ranges))
                                    
                                    if indices_within == len(aa_indices):
                                        domain_coverage_types[domain_id]['within'] += 1
                                    elif indices_within > 0:
                                        domain_coverage_types[domain_id]['partial'] += 1
                                    else:
                                        domain_coverage_types[domain_id]['outside'] += 1
                    except Exception:
                        continue
                
                if domain_coverage_types and total_samples > 0:
                    f.write("Domain Coverage Distribution:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total samples analyzed: {total_samples}\n\n")
                    
                    # Aggregate across all domains
                    total_within = sum(d['within'] for d in domain_coverage_types.values())
                    total_partial = sum(d['partial'] for d in domain_coverage_types.values())
                    total_outside = sum(d['outside'] for d in domain_coverage_types.values())
                    total_domain_samples = total_within + total_partial + total_outside
                    
                    if total_domain_samples > 0:
                        f.write("Overall Distribution (across all domains):\n")
                        f.write(f"  Within domain: {total_within} ({100*total_within/total_domain_samples:.2f}%)\n")
                        f.write(f"  Partially within domain: {total_partial} ({100*total_partial/total_domain_samples:.2f}%)\n")
                        f.write(f"  Outside domain: {total_outside} ({100*total_outside/total_domain_samples:.2f}%)\n\n")
                    
                    # Per-domain distribution
                    f.write("Per-Domain Distribution:\n")
                    for domain_id in sorted(domain_coverage_types.keys()):
                        d = domain_coverage_types[domain_id]
                        domain_total = d['within'] + d['partial'] + d['outside']
                        if domain_total > 0:
                            f.write(f"  {domain_id}:\n")
                            f.write(f"    Within: {d['within']} ({100*d['within']/domain_total:.2f}%)\n")
                            f.write(f"    Partial: {d['partial']} ({100*d['partial']/domain_total:.2f}%)\n")
                            f.write(f"    Outside: {d['outside']} ({100*d['outside']/domain_total:.2f}%)\n\n")
        except Exception as e:
            f.write(f"Error calculating domain coverage distribution: {e}\n\n")
        
        # Domain RMSD statistics
        query_rmsd = f"""
        SELECT e.domain_names, p.RNA_RMSD
        FROM {ref_table} e
        JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        WHERE e.domain_names IS NOT NULL 
        AND p.RNA_RMSD IS NOT NULL
        """
        try:
            cursor = plotter.database.connection.cursor()
            cursor.execute(query_rmsd)
            rows = cursor.fetchall()
            
            if rows:
                f.write("Domain RMSD Statistics:\n")
                f.write("-" * 80 + "\n")
                domain_rmsd_dict = {}
                for domain_names, rmsd in rows:
                    if domain_names and rmsd:
                        domains = domain_names.split(';')
                        rmsd_value = plotter.extract_rmsd(rmsd)
                        if rmsd_value is not None:
                            for domain in domains:
                                domain = domain.strip()
                                if domain:
                                    if domain not in domain_rmsd_dict:
                                        domain_rmsd_dict[domain] = []
                                    domain_rmsd_dict[domain].append(rmsd_value)
                
                for domain, rmsd_values in sorted(domain_rmsd_dict.items()):
                    if rmsd_values:
                        f.write(f"{domain}:\n")
                        f.write(f"  Count: {len(rmsd_values)}\n")
                        f.write(f"  Mean RMSD: {np.mean(rmsd_values):.4f}\n")
                        f.write(f"  Std RMSD: {np.std(rmsd_values):.4f}\n")
                        f.write(f"  Median RMSD: {np.median(rmsd_values):.4f}\n\n")
        except Exception as e:
            f.write(f"Error retrieving domain RMSD statistics: {e}\n\n")
        
        # Domain coverage distribution statistics
        # Recalculate domain coverage distribution (similar to plot_domain_coverage)
        query_coverage = f"""
        SELECT e.PDBId, e.domain_ids, e.domain_pos, p.AAMotif, p.ChainIDpairList_proteinRNA
        FROM {ref_table} e
        JOIN {pred_table} p ON e.PDBId = p.exp_db_id
        WHERE e.domain_pos IS NOT NULL 
        AND p.AAMotif IS NOT NULL
        """
        try:
            cursor.execute(query_coverage)
            rows = cursor.fetchall()
            
            # Convert to DataFrame for filtering
            df_coverage = pd.DataFrame(rows, columns=['PDBId', 'domain_ids', 'domain_pos', 'AAMotif', 'ChainIDpairList_proteinRNA'])
            df_coverage = plotter.filter_data(df_coverage, single_chain_only)
            
            if not df_coverage.empty:
                domain_coverage_types = {}
                total_samples = 0
                
                for _, row in df_coverage.iterrows():
                    pdb_id = row['PDBId']
                    domain_ids = row['domain_ids']
                    domain_pos = row['domain_pos']
                    aa_motif = row['AAMotif']
                    
                    try:
                        if pd.isna(domain_ids) or pd.isna(domain_pos) or pd.isna(aa_motif):
                            continue
                        
                        # Parse domain positions
                        domain_ranges = {}
                        domain_id_list = domain_ids.split(';')
                        domain_pos_list = domain_pos.split(';')
                        
                        if all(pos.strip() == '-' for pos in domain_pos_list):
                            continue
                        
                        for d_id, d_pos in zip(domain_id_list, domain_pos_list):
                            d_id = d_id.strip()
                            if d_pos.strip() == '-':
                                continue
                            
                            try:
                                ranges = []
                                if ':' in d_pos:
                                    range_str = d_pos.split(':')[1]
                                else:
                                    range_str = d_pos
                                
                                for r in range_str.split(','):
                                    start, end = map(int, r.strip().split('-'))
                                    ranges.append((start, end))
                                
                                domain_ranges[d_id] = ranges
                            except ValueError:
                                continue
                        
                        if not domain_ranges:
                            continue
                        
                        # Parse AAMotif indices
                        if isinstance(aa_motif, str):
                            if aa_motif.startswith('[') and aa_motif.endswith(']'):
                                aa_list = literal_eval(aa_motif)
                            else:
                                aa_list = [aa_motif]
                        else:
                            aa_list = [str(aa_motif)]
                        
                        if not isinstance(aa_list, list):
                            aa_list = [aa_list]
                        
                        # Process each motif list
                        for motif_list in aa_list:
                            aa_indices = []
                            if isinstance(motif_list, str):
                                numbers = re.findall(r'\((\d+)\)', motif_list)
                                if numbers:
                                    aa_indices = [int(num) for num in numbers]
                                else:
                                    continue
                            
                            if aa_indices:
                                total_samples += 1
                                for domain_id, ranges in domain_ranges.items():
                                    if domain_id not in domain_coverage_types:
                                        domain_coverage_types[domain_id] = {
                                            'within': 0,
                                            'partial': 0,
                                            'outside': 0
                                        }
                                    
                                    indices_within = sum(1 for idx in aa_indices
                                                         if any(start <= idx <= end
                                                                for start, end in ranges))
                                    
                                    if indices_within == len(aa_indices):
                                        domain_coverage_types[domain_id]['within'] += 1
                                    elif indices_within > 0:
                                        domain_coverage_types[domain_id]['partial'] += 1
                                    else:
                                        domain_coverage_types[domain_id]['outside'] += 1
                    except Exception:
                        continue
                
                if domain_coverage_types and total_samples > 0:
                    f.write("Domain Coverage Distribution:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total samples analyzed: {total_samples}\n\n")
                    
                    # Aggregate across all domains
                    total_within = sum(d['within'] for d in domain_coverage_types.values())
                    total_partial = sum(d['partial'] for d in domain_coverage_types.values())
                    total_outside = sum(d['outside'] for d in domain_coverage_types.values())
                    total_domain_samples = total_within + total_partial + total_outside
                    
                    if total_domain_samples > 0:
                        f.write("Overall Distribution (across all domains):\n")
                        f.write(f"  Within domain: {total_within} ({100*total_within/total_domain_samples:.2f}%)\n")
                        f.write(f"  Partially within domain: {total_partial} ({100*total_partial/total_domain_samples:.2f}%)\n")
                        f.write(f"  Outside domain: {total_outside} ({100*total_outside/total_domain_samples:.2f}%)\n\n")
                    
                    # Per-domain distribution
                    f.write("Per-Domain Distribution:\n")
                    for domain_id in sorted(domain_coverage_types.keys()):
                        d = domain_coverage_types[domain_id]
                        domain_total = d['within'] + d['partial'] + d['outside']
                        if domain_total > 0:
                            f.write(f"  {domain_id}:\n")
                            f.write(f"    Within: {d['within']} ({100*d['within']/domain_total:.2f}%)\n")
                            f.write(f"    Partial: {d['partial']} ({100*d['partial']/domain_total:.2f}%)\n")
                            f.write(f"    Outside: {d['outside']} ({100*d['outside']/domain_total:.2f}%)\n\n")
        except Exception as e:
            f.write(f"Error calculating domain coverage distribution: {e}\n\n")
        
        # Domain relationships statistics
        query_relationships = f"""
        SELECT e.domain_counts, p.Binding_affinity_kd, p.af3_rna_MSA, 
               p.RNAmotif_score, p.ProteinRNAInterfaceArea, p.RNAMotifLength
        FROM {pred_table} p
        JOIN {ref_table} e ON p.exp_db_id = e.PDBId
        WHERE e.domain_counts IS NOT NULL
        """
        try:
            results = plotter.database.execute_query(query_relationships)
            if results:
                f.write("Domain Relationships Statistics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total structures with domain data: {len(results)}\n\n")
        except Exception as e:
            f.write(f"Error retrieving domain relationships statistics: {e}\n\n")
    
    print(f"Statistics written to: {output_path}")


def main():
    # Parse command-line arguments
    single_chain_only = False
    msa_option = None
    rbp_template_option = None
    
    # Parse arguments (case-insensitive, any order)
    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in ['single_chain', 'all']:
            single_chain_only = (arg_lower == 'single_chain')
        elif arg_lower in ['+msa', '-msa', 'msa']:
            msa_option = _standardize_msa_option(arg)
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case
    
    # Get table names from startConfig
    database = DatabaseMethods()
    config = StartConfig()
    plotter = ResultPlotter(database, config, msa_option, single_chain_only, rbp_template_option)

    def column_exists(table_name, column_name):
        """Check if a column exists in a table."""
        try:
            cursor = database.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            return column_name in columns
        except Exception as e:
            print(f"Error checking column {column_name} in {table_name}: {e}")
            return False

    # Check for AF3-related data in pred_table
    pred_table = config.pred_table
    has_af3_data = column_exists(pred_table, 'af3_global_pae_avg')
    
    # Check for domain-related data in exp_table
    ref_table = config.ref_table
    has_domain_data = column_exists(ref_table, 'domain_names')

    # Create plots and write statistics
    # 1. AF3 metrics and global PAE metrics
    if not single_chain_only:
        if has_af3_data:
            plotter.plot_af3_metrics(single_chain_only)
            plotter.plot_global_pae_metrics(single_chain_only)
            write_af3_statistics(plotter, single_chain_only)
        else:
            print("Skipping AF3 metrics plots: af3_global_pae_avg column not found in pred_table")
        
        # 2. Bond metrics
        plotter.plot_bond_metrics(single_chain_only)
        write_bond_statistics(plotter, single_chain_only)
        
        # 3. RNA metrics
        plotter.plot_rna_metrics(single_chain_only)
        write_rna_statistics(plotter, single_chain_only)

    # 4. Interface metrics (only for single chain)
    if single_chain_only:
        plotter.plot_interface_metrics(single_chain_only)
        write_interface_statistics(plotter, single_chain_only)

    # 5. Domain metrics (combined statistics)
    if has_domain_data:
        plotter.plot_domain_rmsd(single_chain_only)
        plotter.plot_domain_coverage(single_chain_only)
        plotter.analyze_domain_relationships(single_chain_only)
        write_domain_statistics(plotter, single_chain_only)
    else:
        print("Skipping domain plots: domain_names column not found in exp_table")

    print(f"\nPlots generated for {'single chain pairs' if single_chain_only else 'all chains'}")
    if msa_option:
        print(f"MSA filter: {msa_option}")
    if rbp_template_option:
        print(f"RBP template filter: {rbp_template_option}")
    print(f"Results saved in: {os.path.dirname(plotter.results_dir)}")

    # Close database connection
    database.close_connection()


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
# plotResults saves across several plot_types; composition reuses those PNGs
# (run the script first). Panel names are the saved base names.
_COMPOSITE_PLOT_TYPES = (
    "af3_metrics", "bond_metrics", "domain_metrics", "domain_count",
    "binding_affinity", "rna_metrics",
)


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None,
                     rbp_template_option=None):
    """Return {base_name: (draw_fn, title)} from this script's saved PNGs."""
    from plots.publication_figure import saved_png_panels
    return saved_png_panels(
        _COMPOSITE_PLOT_TYPES, names, single_chain_only=single_chain_only,
        msa_option=msa_option, rbp_template_option=rbp_template_option)


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="plot_results_composite"))
    main()