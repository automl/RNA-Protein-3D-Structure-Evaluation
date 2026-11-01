"""
Common utilities for plotting scripts.
Centralizes argument parsing, filter formatting, and common data processing functions.
"""
import sys
import ast
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any


def parse_plotting_args() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Parse common command-line arguments for plotting scripts.
    
    Returns:
        Tuple of (single_chain_only, msa_option, rbp_template_option)
    """
    single_chain_only = False
    msa_option = None
    rbp_template_option = None
    
    # Parse arguments (case-insensitive, any order)
    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in ['single_chain', 'all']:
            single_chain_only = (arg_lower == 'single_chain')
        elif arg_lower in ['+msa', '-msa', 'msa']:
            msa_option = '+MSA' if arg_lower in ['+msa', 'msa'] else '-MSA'
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case
    
    return single_chain_only, msa_option, rbp_template_option


def format_filter_info(single_chain_only: bool = False, 
                       msa_option: Optional[str] = None,
                       rbp_template_option: Optional[str] = None) -> list:
    """
    Format filter information for display in statistics files.
    
    Args:
        single_chain_only: Whether single chain filter is applied
        msa_option: MSA filter option ('+MSA' or '-MSA')
        rbp_template_option: RBP template filter option
    
    Returns:
        List of formatted filter strings
    """
    filter_parts = []
    
    if single_chain_only:
        filter_parts.append("Single-Chain")
    else:
        filter_parts.append("All-Chains")
    
    if msa_option:
        filter_parts.append(msa_option)
    
    if rbp_template_option:
        # Format RBP template option nicely
        rbp_upper = rbp_template_option.upper()
        if rbp_upper in ['+RBPTEMP', '+RBP']:
            filter_parts.append("+RBP-template")
        elif rbp_upper in ['-RBPTEMP', '-RBP']:
            filter_parts.append("-RBP-template")
        else:
            filter_parts.append(rbp_template_option)
    
    return filter_parts


def format_filter_info_from_plot_creator(plot_creator) -> list:
    """
    Format filter information from a PlotCreator instance.
    
    Args:
        plot_creator: PlotCreator instance
    
    Returns:
        List of formatted filter strings
    """
    return format_filter_info(
        single_chain_only=plot_creator.single_chain_only if hasattr(plot_creator, 'single_chain_only') else False,
        msa_option=plot_creator.msa_option if hasattr(plot_creator, 'msa_option') else None,
        rbp_template_option=plot_creator.rbp_template_option if hasattr(plot_creator, 'rbp_template_option') else None
    )


def safe_literal_eval(value: Any) -> Any:
    """
    Safely evaluate a string representation of a Python literal.
    
    Args:
        value: Value to evaluate (can be string, list, dict, etc.)
    
    Returns:
        Evaluated value or original value if evaluation fails
    """
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith('[') and value.endswith(']')) or \
           (value.startswith('{') and value.endswith('}')):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value


def parse_dict_string(dict_string: str) -> Dict[str, Any]:
    """
    Parse string representation of dictionary into actual dictionary.
    Handles formats like: "{'key1': 1, 'key2': 2}" or "{key1: 1, key2: 2}"
    
    Args:
        dict_string: String representation of dictionary
    
    Returns:
        Dictionary or empty dict if parsing fails
    """
    if pd.isna(dict_string) or not dict_string:
        return {}
    
    try:
        # Try literal_eval first (handles proper Python syntax)
        if isinstance(dict_string, str) and (dict_string.strip().startswith('{') or dict_string.strip().startswith('[')):
            return ast.literal_eval(dict_string)
    except (ValueError, SyntaxError):
        pass
    
    try:
        # Fallback: manual parsing for simple formats
        dict_str = str(dict_string).strip('{}')
        if not dict_str:
            return {}
        
        pairs = dict_str.split(',')
        result = {}
        
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                # Clean up the key and value
                key = key.strip().strip("'").strip('"')
                try:
                    value = int(value.strip())
                except ValueError:
                    try:
                        value = float(value.strip())
                    except ValueError:
                        value = value.strip().strip("'").strip('"')
                result[key] = value
        return result
    except Exception:
        return {}


def calculate_mean_std(values: list) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate mean, std, and median for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Tuple of (mean, std, median) or (None, None, None) if no valid values
    """
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


def extract_first_value(value: Any) -> Optional[float]:
    """
    Extract first value from a list string or return float value.
    Handles formats like: "[1.5, 2.3, 3.1]" or "1.5"
    
    Args:
        value: Value that might be a list string or single value
    
    Returns:
        First value as float, or None if extraction fails
    """
    try:
        if isinstance(value, str):
            # Handle list format - strip brackets and split
            value = value.strip('[]').replace(' ', '')
            if ',' in value:
                values = [float(x) for x in value.split(',')]
                return max(values) if values else None
            return float(value)
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list):
            return float(value[0]) if value else None
        return None
    except (ValueError, TypeError, IndexError):
        return None


def normalize_msa_option(msa_option: Optional[str]) -> Optional[str]:
    """
    Normalize MSA option to standard format.
    
    Args:
        msa_option: MSA option string (can be '+msa', '-msa', '+MSA', '-MSA', 'msa', etc.)
    
    Returns:
        Normalized option ('+MSA' or '-MSA') or None
    """
    if msa_option is None:
        return None
    
    msa_lower = str(msa_option).lower()
    if msa_lower in ['+msa', 'msa']:
        return '+MSA'
    elif msa_lower == '-msa':
        return '-MSA'
    
    return msa_option  # Return as-is if already in correct format


def extract_length(length_str: Any) -> Optional[int]:
    """
    Extract length value from string representation or return integer.
    Handles formats like: "[100, 200]" (returns max), "100", or 100.
    
    Args:
        length_str: String, int, or float representing length(s)
    
    Returns:
        Integer length value (max if list), or None if parsing fails
    """
    if pd.isna(length_str):
        return None
    try:
        if isinstance(length_str, (int, float)):
            return int(length_str)
        if isinstance(length_str, str) and '[' in length_str:
            values = [int(x.strip()) for x in length_str.strip('[]').split(',')]
            return max(values) if values else None
        return int(length_str)
    except (ValueError, TypeError):
        return None


def extract_rmsd(rmsd_str: Any) -> Optional[float]:
    """
    Extract RMSD value from string representation or return float.
    Handles formats like: "[1.5, 2.3]" (returns first), "1.5", or 1.5.
    
    Args:
        rmsd_str: String, int, or float representing RMSD value(s)
    
    Returns:
        Float RMSD value (first if list), or None if parsing fails
    """
    if pd.isna(rmsd_str):
        return None
    try:
        if isinstance(rmsd_str, (int, float)):
            return float(rmsd_str)
        if isinstance(rmsd_str, str) and '[' in rmsd_str:
            values = [float(x.strip()) for x in rmsd_str.strip('[]').split(',')]
            return values[0] if values else None
        return float(rmsd_str)
    except (ValueError, TypeError):
        return None


def get_rna_rmsd(val: Any) -> Optional[float]:
    """
    Extract RNA RMSD value from various formats.
    Handles formats like: "[1.5, 2.3]" (returns first), "1.5", list, or float.
    This is a specialized version for RNA RMSD that handles more edge cases.
    
    Args:
        val: Value that might be a list string, list, or single value
    
    Returns:
        Float RMSD value (first if list), or None if parsing fails
    """
    if pd.isna(val):
        return None
    try:
        if isinstance(val, str):
            if '[' in val:
                values = ast.literal_eval(val)
                if isinstance(values, list) and len(values) > 0:
                    return float(values[0])
            return float(val)
        elif isinstance(val, (list, tuple)):
            return float(val[0]) if len(val) > 0 else None
        return float(val)
    except (ValueError, TypeError, SyntaxError):
        return None


def classify_msa(row: pd.Series, msa_col: str = 'RNA_msa_size', 
                 chain_col: str = 'RNAChainIDs_pred', length_col: str = 'RNALength') -> Optional[str]:
    """
    Classify MSA status for a row based on RNA_msa_size, RNAChainIDs, and RNALength.
    
    Args:
        row: pandas Series representing a database row
        msa_col: Column name for MSA size dictionary
        chain_col: Column name for RNA chain IDs
        length_col: Column name for RNA lengths
    
    Returns:
        '+rMSA' if MSA size > 1, '-rMSA' if MSA size <= 1, or None if error
    """
    try:
        msa_dict = row.get(msa_col)
        if pd.isna(msa_dict):
            return None
            
        if isinstance(msa_dict, str):
            msa_dict = safe_literal_eval(msa_dict)
        if not isinstance(msa_dict, dict):
            return None

        chain_ids = row.get(chain_col)
        if pd.isna(chain_ids):
            return None
            
        if isinstance(chain_ids, str):
            chain_ids = safe_literal_eval(chain_ids)
        if not isinstance(chain_ids, (list, tuple)):
            chain_ids = [chain_ids]

        rna_lengths = row.get(length_col)
        if pd.isna(rna_lengths):
            return None
            
        if isinstance(rna_lengths, str):
            rna_lengths = safe_literal_eval(rna_lengths)
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
        # Silently return None on error (can be logged if needed)
        return None


def process_rna_inf(inf_value: Any) -> pd.Series:
    """
    Process RNA_INF column value to extract RNA_inf, RNA_WC, RNA_nonWC, RNA_STACK.
    
    Args:
        inf_value: RNA_INF value (can be string list, list, or other format)
    
    Returns:
        pandas Series with keys: 'RNA_inf', 'RNA_WC', 'RNA_nonWC', 'RNA_STACK'
    """
    if isinstance(inf_value, str):
        try:
            values = ast.literal_eval(inf_value)
            if isinstance(values, list) and len(values) >= 4:
                return pd.Series({
                    'RNA_inf': values[0], 
                    'RNA_WC': values[1], 
                    'RNA_nonWC': values[2], 
                    'RNA_STACK': values[3]
                })
        except (ValueError, SyntaxError):
            pass
        # If parsing fails, return the value for all fields
        return pd.Series({
            'RNA_inf': inf_value, 
            'RNA_WC': inf_value, 
            'RNA_nonWC': inf_value, 
            'RNA_STACK': inf_value
        })
    elif isinstance(inf_value, (list, tuple)) and len(inf_value) >= 4:
        return pd.Series({
            'RNA_inf': inf_value[0], 
            'RNA_WC': inf_value[1], 
            'RNA_nonWC': inf_value[2], 
            'RNA_STACK': inf_value[3]
        })
    # Default: return the value for all fields
    return pd.Series({
        'RNA_inf': inf_value, 
        'RNA_WC': inf_value, 
        'RNA_nonWC': inf_value, 
        'RNA_STACK': inf_value
    })


def process_rmsd(rmsd_str: Any) -> Optional[float]:
    """
    Process RMSD string to get first value.
    Handles formats like: "[1.5, 2.3]" (returns first), "1.5", or 1.5.
    
    Args:
        rmsd_str: RMSD value (string, list, or float)
    
    Returns:
        Float RMSD value (first if list), or None if parsing fails
    """
    if pd.isna(rmsd_str):
        return None
    try:
        if isinstance(rmsd_str, (int, float)):
            return float(rmsd_str)
        if isinstance(rmsd_str, str):
            if '[' in rmsd_str:
                values = ast.literal_eval(rmsd_str)
                if isinstance(values, list) and len(values) > 0:
                    return float(values[0])
            return float(rmsd_str)
        if isinstance(rmsd_str, (list, tuple)) and len(rmsd_str) > 0:
            return float(rmsd_str[0])
        return None
    except (ValueError, TypeError, SyntaxError):
        return None


def process_tm(tm_str: Any) -> Optional[float]:
    """
    Process TM score string to get first value.
    Handles formats like: "[0.8, 0.9]" (returns first), "0.8", or 0.8.
    
    Args:
        tm_str: TM score value (string, list, or float)
    
    Returns:
        Float TM score (first if list), or None if parsing fails
    """
    return process_rmsd(tm_str)  # Same logic as RMSD


def process_lddt(lddt_str: Any) -> Optional[float]:
    """
    Process LDDT string to get first value.
    Handles formats like: "[0.9, 0.85]" (returns first), "0.9", or 0.9.
    
    Args:
        lddt_str: LDDT value (string, list, or float)
    
    Returns:
        Float LDDT value (first if list), or None if parsing fails
    """
    return process_rmsd(lddt_str)  # Same logic as RMSD


def process_pae(pae_str: Any) -> Optional[float]:
    """
    Process PAE string to get first value.
    Handles formats like: "[5.2, 4.8]" (returns first), "5.2", or 5.2.
    
    Args:
        pae_str: PAE value (string, list, or float)
    
    Returns:
        Float PAE value (first if list), or None if parsing fails
    """
    return process_rmsd(pae_str)  # Same logic as RMSD


def process_plddt(plddt_str: Any) -> Optional[float]:
    """
    Process pLDDT string to get first value.
    Handles formats like: "[0.88, 0.82]" (returns first), "0.88", or 0.88.
    
    Args:
        plddt_str: pLDDT value (string, list, or float)
    
    Returns:
        Float pLDDT value (first if list), or None if parsing fails
    """
    return process_rmsd(plddt_str)  # Same logic as RMSD


def process_ptm(ptm_str: Any, rna_length_str: Any) -> Optional[float]:
    """
    Process pTM/ipTM values using the longest RNA index.
    
    Args:
        ptm_str: pTM/ipTM value (string list or list)
        rna_length_str: RNA length value (string list or list)
    
    Returns:
        Float pTM/ipTM value for longest RNA, or None if parsing fails
    """
    if pd.isna(ptm_str) or pd.isna(rna_length_str):
        return None
    try:
        rna_lengths = safe_literal_eval(rna_length_str) if isinstance(rna_length_str, str) else rna_length_str
        if not isinstance(rna_lengths, list):
            rna_lengths = [rna_lengths]
        longest_idx = rna_lengths.index(max(rna_lengths))
        
        ptm_values = safe_literal_eval(ptm_str) if isinstance(ptm_str, str) else ptm_str
        if not isinstance(ptm_values, list):
            ptm_values = [ptm_values]
        
        return float(ptm_values[longest_idx]) if longest_idx < len(ptm_values) else float(ptm_values[0])
    except (ValueError, TypeError, SyntaxError, IndexError):
        return None


def process_rna_length(length_str: Any) -> Optional[int]:
    """
    Process RNA length string to get the maximum length.
    Handles formats like: "[100, 200]" (returns max), "100", or 100.
    
    Args:
        length_str: RNA length value (string, list, or int)
    
    Returns:
        Integer maximum length, or None if parsing fails
    """
    if pd.isna(length_str):
        return None
    try:
        if isinstance(length_str, (int, float)):
            return int(length_str)
        if isinstance(length_str, str):
            if '[' in length_str:
                values = ast.literal_eval(length_str)
                if isinstance(values, list):
                    return max(values)
            return int(length_str)
        if isinstance(length_str, (list, tuple)):
            return max(length_str)
        return None
    except (ValueError, TypeError, SyntaxError):
        return None


def process_msa_sizes(row: pd.Series, msa_col: str = 'RNA_msa_size', 
                      chain_col: str = 'RNAChainIDs', length_col: str = 'RNALength') -> int:
    """
    Process MSA sizes for both single and multiple chains.
    Returns the MSA size for the longest RNA chain.
    
    Args:
        row: pandas Series representing a database row
        msa_col: Column name for MSA size dictionary
        chain_col: Column name for RNA chain IDs
        length_col: Column name for RNA lengths
    
    Returns:
        Integer MSA size (defaults to 1 if error)
    """
    try:
        msa_dict = row.get(msa_col)
        if pd.isna(msa_dict):
            return 1
            
        if isinstance(msa_dict, str):
            msa_dict = safe_literal_eval(msa_dict)
        if not isinstance(msa_dict, dict):
            return 1
        
        # Handle single chain case
        if len(msa_dict) == 1:
            return list(msa_dict.values())[0]
        
        # Multi-chain case: use longest RNA
        rna_lengths = row.get(length_col)
        rna_chain_ids = row.get(chain_col)
        
        if pd.isna(rna_lengths) or pd.isna(rna_chain_ids):
            return list(msa_dict.values())[0]
        
        if isinstance(rna_lengths, str):
            rna_lengths = safe_literal_eval(rna_lengths)
        if not isinstance(rna_lengths, list):
            rna_lengths = [rna_lengths]
            
        if isinstance(rna_chain_ids, str):
            rna_chain_ids = safe_literal_eval(rna_chain_ids)
        if not isinstance(rna_chain_ids, list):
            rna_chain_ids = [rna_chain_ids]
        
        longest_idx = rna_lengths.index(max(rna_lengths))
        if longest_idx < len(rna_chain_ids):
            chain_id = rna_chain_ids[longest_idx]
            return msa_dict.get(str(chain_id), 1)
        
        return list(msa_dict.values())[0]
    except Exception:
        return 1


def create_msa_class_labels(bins: list, labels: list) -> Dict[str, str]:
    """
    Create mapping from MSA class values to display labels.
    
    Args:
        bins: List of bin edges
        labels: List of label names
    
    Returns:
        Dictionary mapping label names to display labels
    """
    # Default mapping - can be customized
    default_mapping = {
        'no_msa': '0 rMSA',
        '0 rMSA': '0 rMSA',
        'small_msa': '0-20 rMSA',
        '0-20 rMSA': '0-20 rMSA',
        'medium_msa': '20-1000 rMSA',
        '20-1000 rMSA': '20-1000 rMSA',
        'large_msa': '+1000 rMSA',
        '+1000 rMSA': '+1000 rMSA'
    }
    
    # Create mapping from provided labels
    label_map = {}
    for label in labels:
        label_map[label] = default_mapping.get(label, label)
    
    return label_map


def create_length_class_labels(bins: list, labels: list) -> Dict[str, str]:
    """
    Create mapping from length class values to display labels.
    
    Args:
        bins: List of bin edges
        labels: List of label names
    
    Returns:
        Dictionary mapping label names to display labels
    """
    # Default mapping - can be customized
    default_mapping = {
        '0-20': '0-20',
        '20-100': '20-100',
        '100-200': '100-200',
        '200+': '200+',
        '0-50': '0-50',
        '50+': '50+'
    }
    
    # Create mapping from provided labels
    label_map = {}
    for label in labels:
        label_map[label] = default_mapping.get(label, label)
    
    return label_map


def process_dataframe_metrics(df: pd.DataFrame, metric_processors: Dict[str, Any]) -> pd.DataFrame:
    """
    Process multiple dataframe columns using specified processing functions.
    
    Args:
        df: DataFrame to process
        metric_processors: Dictionary mapping column names to processing functions or tuples.
                          Format: {
                              'column_name': processing_function,
                              'column_name_with_dependency': (processing_function, 'dependency_column')
                          }
                          For functions that need additional columns (like process_ptm), use tuple format.
    
    Returns:
        Processed DataFrame
    """
    df = df.copy()
    
    for column, processor in metric_processors.items():
        if column not in df.columns:
            continue
        
        try:
            if isinstance(processor, tuple):
                # Processor with dependency (e.g., process_ptm needs RNALength)
                proc_func, dep_column = processor
                if dep_column in df.columns:
                    df[column] = df.apply(lambda row: proc_func(row[column], row[dep_column]), axis=1)
            elif callable(processor):
                # Simple processor function
                df[column] = df[column].apply(processor)
        except Exception as e:
            # Silently skip if processing fails
            continue
    
    return df

