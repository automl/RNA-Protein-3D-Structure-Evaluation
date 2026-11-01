"""
Script to plot metrics based on template presence (RBP_PDB_Templates, Protein_PDB_Templates)
and template count, with MSA size classification.

For each metric, creates plots showing:
1. Whether samples have RBP templates, Protein templates, both, or neither
2. The count of templates in each list (0, 1, 2, 3, 4, etc.)
3. MSA size classification (no_msa, small_msa, medium_msa, large_msa)
4. Interaction plots comparing MSA and template effects
"""

import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import os as _os, sys as _sys  # allow direct execution: python evaluation/<script>.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator
from structures.rna import RNA
from structures.protein import Protein

# Metric label mapping
LABEL_MAP = {
    'Complex_LDDT': r"cLDDT$_{BB}$",
    'Complex_LDDT_allAtom': 'cLDDT',
    'RNA_LDDT': r"rLDDT$_{C3'}$",
    'RNA_LDDT_allAtom': 'rLDDT',
    'RNA_TM': 'rTM',
    'Complex_TM': 'cTM',
    'RNAComplex_TM': 'crTM',
    'iLDDT': 'iLDDT'
}


def parse_template_column(value: Optional[str]) -> Optional[List[str]]:
    """
    Parse template column value from database.
    
    Args:
        value: Database value (can be None, empty string, "[]", or JSON list)
    
    Returns:
        List of PDB IDs or None if empty/invalid
    """
    if value is None or value == "" or value.strip() == "":
        return None
    
    if value.strip() == "[]":
        return []
    
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def get_template_presence(rbp_templates: Optional[List[str]], 
                         protein_templates: Optional[List[str]],
                         exp_db_id: Optional[str] = None,
                         check_exact: bool = False) -> str:
    """
    Determine template presence category.
    
    Args:
        rbp_templates: List of RBP template PDB IDs or None
        protein_templates: List of Protein template PDB IDs or None
        exp_db_id: PDB ID to check for exact match (optional)
        check_exact: If True, check if any template matches exp_db_id
    
    Returns:
        Category: "Neither", "RBP", "Protein", "Both", or "Identical" (if check_exact and match found)
    """
    # Check for exact template match first (if enabled)
    if check_exact and exp_db_id:
        # Normalize exp_db_id: uppercase and strip whitespace
        exp_db_id_normalized = str(exp_db_id).strip().upper()
        all_templates = []
        if rbp_templates:
            all_templates.extend([str(t).strip().upper() for t in rbp_templates])
        if protein_templates:
            all_templates.extend([str(t).strip().upper() for t in protein_templates])
        
        if exp_db_id_normalized in all_templates:
            return "Identical"
    
    has_rbp = rbp_templates is not None and len(rbp_templates) > 0
    has_protein = protein_templates is not None and len(protein_templates) > 0
    
    if has_rbp and has_protein:
        return "Both"
    elif has_rbp:
        return "RBP"
    elif has_protein:
        return "Protein"
    else:
        return "Neither"


def get_template_count(templates: Optional[List[str]]) -> int:
    """
    Get count of templates in list.
    
    Args:
        templates: List of template PDB IDs or None
    
    Returns:
        Count (0 if None or empty)
    """
    if templates is None:
        return 0
    return len(templates)


def process_rna_msa_sizes(row) -> float:
    """
    Process MSA sizes from RNA_msa_size column.
    For multi-chain RNAs, uses the longest chain's MSA size.
    
    Args:
        row: DataFrame row with RNA_msa_size and exp_db_id
    
    Returns:
        MSA size value (1 if no MSA or error)
    """
    try:
        msa_size = row.get('RNA_msa_size')
        if msa_size is None or pd.isna(msa_size):
            return 1.0
        
        if isinstance(msa_size, str):
            msa_dict = json.loads(msa_size) if msa_size.startswith('{') else eval(msa_size)
        else:
            msa_dict = msa_size
        
        if not isinstance(msa_dict, dict):
            return 1.0
        
        # For single chain, use the value
        if len(msa_dict) == 1:
            return float(list(msa_dict.values())[0])
        
        # For multi-chain, use RNA.get_longest_chain_id()
        exp_db_id = row.get('exp_db_id')
        if exp_db_id is None:
            return float(list(msa_dict.values())[0])
        
        try:
            rna = RNA.get_rna_from_db(exp_db_id.upper())
            if rna and rna.is_rna:
                longest_chain_id = rna.get_longest_chain_id()
                if longest_chain_id and longest_chain_id in msa_dict:
                    return float(msa_dict[longest_chain_id])
        except Exception:
            pass
        
        return float(list(msa_dict.values())[0])
    except Exception:
        return 1.0


def process_protein_msa_sizes(row) -> float:
    """
    Process MSA sizes from Protein_msa_size column.
    For multi-chain proteins, uses the longest chain's MSA size.
    
    Args:
        row: DataFrame row with Protein_msa_size and exp_db_id
    
    Returns:
        MSA size value (1 if no MSA or error)
    """
    try:
        msa_size = row.get('Protein_msa_size')
        if msa_size is None or pd.isna(msa_size):
            return 1.0
        
        if isinstance(msa_size, str):
            msa_dict = json.loads(msa_size) if msa_size.startswith('{') else eval(msa_size)
        else:
            msa_dict = msa_size
        
        if not isinstance(msa_dict, dict):
            return 1.0
        
        # For single chain, use the value
        if len(msa_dict) == 1:
            return float(list(msa_dict.values())[0])
        
        # For multi-chain, use Protein.get_longest_chain_id()
        exp_db_id = row.get('exp_db_id')
        if exp_db_id is None:
            return float(list(msa_dict.values())[0])
        
        try:
            protein = Protein.get_protein_from_db(id=exp_db_id.upper())
            if protein and protein.is_protein:
                longest_chain_id = protein.get_longest_chain_id()
                if longest_chain_id and longest_chain_id in msa_dict:
                    return float(msa_dict[longest_chain_id])
        except Exception:
            pass
        
        return float(list(msa_dict.values())[0])
    except Exception:
        return 1.0


def load_data(db_path: str, pred_table: str, check_exact: bool = False) -> pd.DataFrame:
    """
    Load data from database including metrics and template columns.
    
    Args:
        db_path: Path to database
        pred_table: Name of prediction table
    
    Returns:
        DataFrame with metrics and parsed template columns
    """
    conn = sqlite3.connect(db_path)
    
    # Get available columns
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({pred_table})")
    columns_info = cursor.fetchall()
    available_columns = [col[1] for col in columns_info]
    
    # Build query with available metrics and template columns
    metric_columns = [col for col in LABEL_MAP.keys() if col in available_columns]
    template_columns = []
    if 'RBP_PDB_Templates' in available_columns:
        template_columns.append('RBP_PDB_Templates')
    if 'Protein_PDB_Templates' in available_columns:
        template_columns.append('Protein_PDB_Templates')
    
    if not metric_columns:
        print("Warning: No metric columns found in database")
        conn.close()
        return pd.DataFrame()
    
    if not template_columns:
        print("Warning: No template columns (RBP_PDB_Templates or Protein_PDB_Templates) found in database")
        conn.close()
        return pd.DataFrame()
    
    # Check for MSA columns
    msa_columns = []
    if 'RNA_msa_size' in available_columns:
        msa_columns.append('RNA_msa_size')
    if 'Protein_msa_size' in available_columns:
        msa_columns.append('Protein_msa_size')
    
    select_cols = ['exp_db_id'] + metric_columns + template_columns + msa_columns
    query = f"SELECT {', '.join(select_cols)} FROM {pred_table}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse template columns
    if 'RBP_PDB_Templates' in df.columns:
        df['RBP_templates_parsed'] = df['RBP_PDB_Templates'].apply(parse_template_column)
    else:
        df['RBP_templates_parsed'] = None
    
    if 'Protein_PDB_Templates' in df.columns:
        df['Protein_templates_parsed'] = df['Protein_PDB_Templates'].apply(parse_template_column)
    else:
        df['Protein_templates_parsed'] = None
    
    # Create presence category
    df['Template_Presence'] = df.apply(
        lambda row: get_template_presence(
            row.get('RBP_templates_parsed'),
            row.get('Protein_templates_parsed'),
            exp_db_id=row.get('exp_db_id'),
            check_exact=check_exact
        ),
        axis=1
    )
    
    # Create count columns
    df['RBP_Template_Count'] = df['RBP_templates_parsed'].apply(get_template_count)
    df['Protein_Template_Count'] = df['Protein_templates_parsed'].apply(get_template_count)
    
    # Create binary template presence columns
    df['RBP_Template_Presence'] = df['RBP_templates_parsed'].apply(
        lambda x: '+RBP' if x is not None and len(x) > 0 else '-RBP'
    )
    df['Protein_Template_Presence'] = df['Protein_templates_parsed'].apply(
        lambda x: '+Protein' if x is not None and len(x) > 0 else '-Protein'
    )
    
    # Create combined exclusive template category
    def get_exclusive_template_category(row):
        """Get exclusive template category: RBP_only, Protein_only, Both, or Neither"""
        rbp_templates = row.get('RBP_templates_parsed')
        protein_templates = row.get('Protein_templates_parsed')
        
        has_rbp = rbp_templates is not None and len(rbp_templates) > 0
        has_protein = protein_templates is not None and len(protein_templates) > 0
        
        if has_rbp and has_protein:
            return 'Both'
        elif has_rbp:
            return 'RBP_only'
        elif has_protein:
            return 'Protein_only'
        else:
            return 'Neither'
    
    df['Exclusive_Template_Category'] = df.apply(get_exclusive_template_category, axis=1)
    
    # Process RNA MSA sizes (rMSA) - binary classification: has MSA or not
    if 'RNA_msa_size' in df.columns:
        df['rMSAsize'] = df.apply(process_rna_msa_sizes, axis=1)
        # Binary classification: has MSA (>1) or no MSA (≤1)
        df['has_rmsa'] = df['rMSAsize'].apply(lambda x: '+rMSA' if x > 1 else '-rMSA')
        # MSA class bins: [-float('inf'), 1, 20, 1000, float('inf')]
        # Labels: ['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
        df['rmsa_class'] = pd.cut(
            df['rMSAsize'],
            bins=[-float('inf'), 1, 20, 1000, float('inf')],
            labels=['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
        )
    else:
        df['rMSAsize'] = None
        df['has_rmsa'] = None
        df['rmsa_class'] = None
    
    # Process Protein MSA sizes (pMSA) - binary classification: has MSA or not
    if 'Protein_msa_size' in df.columns:
        df['pMSAsize'] = df.apply(process_protein_msa_sizes, axis=1)
        # Binary classification: has MSA (>1) or no MSA (≤1)
        df['has_pmsa'] = df['pMSAsize'].apply(lambda x: '+pMSA' if x > 1 else '-pMSA')
        # MSA class bins: [-float('inf'), 1, 20, 1000, float('inf')]
        # Labels: ['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
        df['pmsa_class'] = pd.cut(
            df['pMSAsize'],
            bins=[-float('inf'), 1, 20, 1000, float('inf')],
            labels=['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
        )
    else:
        df['pMSAsize'] = None
        df['has_pmsa'] = None
        df['pmsa_class'] = None
    
    # For backward compatibility, keep old column names pointing to rMSA
    df['MSAsize'] = df['rMSAsize']
    df['has_msa'] = df['has_rmsa']
    df['msa_class'] = df['rmsa_class']
    
    return df


def prepare_presence_data(df: pd.DataFrame, metric_col: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for presence-based plots.
    
    Args:
        df: DataFrame with metrics and template presence
        metric_col: Name of metric column
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Filter out NaN values
    df_clean = df[[metric_col, 'Template_Presence']].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Order categories: Neither, Protein, RBP, Both, Identical (if present)
    category_order = ["Neither", "Protein", "RBP", "Both", "Identical"]
    
    # Get available categories in data
    available_categories = df_clean['Template_Presence'].unique()
    
    # Filter category_order to only include categories present in data
    filtered_order = [cat for cat in category_order if cat in available_categories]
    
    data_values = []
    labels = []
    
    for category in filtered_order:
        values = df_clean[df_clean['Template_Presence'] == category][metric_col].values
        if len(values) > 0:
            # Convert to numeric and filter out NaN
            numeric_values = pd.to_numeric(values, errors='coerce')
            numeric_values = numeric_values[~pd.isna(numeric_values)]
            if len(numeric_values) > 0:
                data_values.append(numeric_values)
                labels.append(category)
    
    return data_values, labels


def prepare_count_data(df: pd.DataFrame, metric_col: str, template_type: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for count-based plots.
    
    Args:
        df: DataFrame with metrics and template counts
        metric_col: Name of metric column
        template_type: "RBP" or "Protein"
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    count_col = f'{template_type}_Template_Count'
    
    # Filter out NaN values
    df_clean = df[[metric_col, count_col]].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Get unique counts and sort
    unique_counts = sorted(df_clean[count_col].unique())
    
    data_values = []
    labels = []
    
    for count in unique_counts:
        values = df_clean[df_clean[count_col] == count][metric_col].values
        if len(values) > 0:
            # Convert to numeric and filter out NaN
            numeric_values = pd.to_numeric(values, errors='coerce')
            numeric_values = numeric_values[~pd.isna(numeric_values)]
            if len(numeric_values) > 0:
                data_values.append(numeric_values)
                labels.append(str(int(count)))
    
    return data_values, labels


def prepare_template_count_by_msa(df: pd.DataFrame, count_col: str, msa_type: str = 'rMSA') -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for template count by MSA class plots.
    
    Args:
        df: DataFrame with template counts and MSA class
        count_col: Name of template count column ('RBP_Template_Count' or 'Protein_Template_Count')
        msa_type: 'rMSA' or 'pMSA' to specify which MSA to use
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Select MSA class column based on type
    msa_class_col = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    if msa_class_col not in df.columns:
        return [], []
    
    # Filter out NaN values
    df_clean = df[[count_col, msa_class_col]].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Order MSA classes based on type
    if msa_type == 'rMSA':
        msa_order = ['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
    else:
        msa_order = ['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
    
    # Get available MSA classes in data
    available_classes = df_clean[msa_class_col].unique()
    
    # Filter order to only include classes present in data
    filtered_order = [cls for cls in msa_order if cls in available_classes]
    
    data_values = []
    labels = []
    
    for msa_class in filtered_order:
        values = df_clean[df_clean[msa_class_col] == msa_class][count_col].values
        if len(values) > 0:
            # Convert to numeric and filter out NaN
            numeric_values = pd.to_numeric(values, errors='coerce')
            numeric_values = numeric_values[~pd.isna(numeric_values)]
            if len(numeric_values) > 0:
                data_values.append(numeric_values)
                labels.append(msa_class)
    
    return data_values, labels


def prepare_msa_data(df: pd.DataFrame, metric_col: str, msa_type: str = 'rMSA') -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for MSA-based plots (binary: has MSA or not).
    
    Args:
        df: DataFrame with metrics and MSA classification
        metric_col: Name of metric column
        msa_type: 'rMSA' or 'pMSA' to specify which MSA to use
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Select MSA columns based on type
    has_msa_col = 'has_rmsa' if msa_type == 'rMSA' else 'has_pmsa'
    
    if has_msa_col not in df.columns:
        return [], []
    
    # Filter out NaN values
    df_clean = df[[metric_col, has_msa_col]].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Order categories based on type
    if msa_type == 'rMSA':
        category_order = ['-rMSA', '+rMSA']
    else:
        category_order = ['-pMSA', '+pMSA']
    
    data_values = []
    labels = []
    
    for category in category_order:
        values = df_clean[df_clean[has_msa_col] == category][metric_col].values
        if len(values) > 0:
            # Convert to numeric and filter out NaN
            numeric_values = pd.to_numeric(values, errors='coerce')
            numeric_values = numeric_values[~pd.isna(numeric_values)]
            if len(numeric_values) > 0:
                data_values.append(numeric_values)
                labels.append(category)
    
    return data_values, labels


def prepare_interaction_data(df: pd.DataFrame, metric_col: str, 
                            group_by: str, stratify_by: str, msa_type: str = 'rMSA') -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for interaction plots (e.g., MSA class × Template presence).
    
    Args:
        df: DataFrame with metrics and classifications
        metric_col: Name of metric column
        group_by: Primary grouping column (e.g., 'has_msa', 'has_rmsa', 'has_pmsa', 'msa_class', 'rmsa_class', 'pmsa_class')
        stratify_by: Secondary grouping column (e.g., 'Template_Presence')
        msa_type: 'rMSA' or 'pMSA' to specify which MSA to use (only used if group_by/stratify_by contain generic 'msa')
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Map generic 'msa' references to specific MSA type columns
    if group_by == 'has_msa':
        group_by = 'has_rmsa' if msa_type == 'rMSA' else 'has_pmsa'
    elif group_by == 'msa_class':
        group_by = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    if stratify_by == 'has_msa':
        stratify_by = 'has_rmsa' if msa_type == 'rMSA' else 'has_pmsa'
    elif stratify_by == 'msa_class':
        stratify_by = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    # Filter out NaN values
    df_clean = df[[metric_col, group_by, stratify_by]].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Get unique values for both grouping columns with custom ordering
    # Define custom order for MSA class based on type
    if msa_type == 'rMSA':
        msa_order = ['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
    else:
        msa_order = ['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
    
    if group_by in ['rmsa_class', 'pmsa_class']:
        available_group = df_clean[group_by].dropna().unique()
        group_values = [s for s in msa_order if s in available_group]
        # Add any remaining values not in the order
        for val in available_group:
            if val not in group_values:
                group_values.append(val)
    else:
        group_values = sorted(df_clean[group_by].dropna().unique())
    
    # Define custom order for Template_Presence: Neither, Protein, RBP, Both, Identical
    if stratify_by == 'Template_Presence':
        stratify_order = ['Neither', 'Protein', 'RBP', 'Both', 'Identical']
        available_stratify = df_clean[stratify_by].dropna().unique()
        stratify_values = [s for s in stratify_order if s in available_stratify]
        # Add any remaining values not in the order
        for val in available_stratify:
            if val not in stratify_values:
                stratify_values.append(val)
    elif stratify_by in ['rmsa_class', 'pmsa_class']:
        available_stratify = df_clean[stratify_by].dropna().unique()
        stratify_values = [s for s in msa_order if s in available_stratify]
        # Add any remaining values not in the order
        for val in available_stratify:
            if val not in stratify_values:
                stratify_values.append(val)
    elif group_by == 'Exclusive_Template_Category' or stratify_by == 'Exclusive_Template_Category':
        # Custom order for exclusive template categories: Neither first, Both last
        exclusive_order = ['Neither', 'RBP_only', 'Protein_only', 'Both']
        if group_by == 'Exclusive_Template_Category':
            available_group = df_clean[group_by].dropna().unique()
            group_values = [s for s in exclusive_order if s in available_group]
            for val in available_group:
                if val not in group_values:
                    group_values.append(val)
            stratify_values = sorted(df_clean[stratify_by].dropna().unique())
        else:
            available_stratify = df_clean[stratify_by].dropna().unique()
            stratify_values = [s for s in exclusive_order if s in available_stratify]
            for val in available_stratify:
                if val not in stratify_values:
                    stratify_values.append(val)
    else:
        stratify_values = sorted(df_clean[stratify_by].dropna().unique())
    
    data_values = []
    labels = []
    
    # Create labels as "group_value × stratify_value" (single line)
    # For Exclusive_Template_Category × msa_class, ensure Neither is first, Both is last
    if group_by == 'Exclusive_Template_Category' and stratify_by in ['rmsa_class', 'pmsa_class']:
        # Ensure group_values order: Neither, RBP_only, Protein_only, Both
        exclusive_order = ['Neither', 'RBP_only', 'Protein_only', 'Both']
        group_values_ordered = [g for g in exclusive_order if g in group_values]
        for g in group_values:
            if g not in group_values_ordered:
                group_values_ordered.append(g)
        group_values = group_values_ordered
    
    for group_val in group_values:
        for stratify_val in stratify_values:
            mask = (df_clean[group_by] == group_val) & (df_clean[stratify_by] == stratify_val)
            values = df_clean[mask][metric_col].values
            if len(values) > 0:
                # Convert to numeric and filter out NaN
                numeric_values = pd.to_numeric(values, errors='coerce')
                numeric_values = numeric_values[~pd.isna(numeric_values)]
                if len(numeric_values) > 0:
                    data_values.append(numeric_values)
                    labels.append(f"{group_val} × {stratify_val}")
    
    return data_values, labels


def prepare_template_count_msa_interaction(df: pd.DataFrame, metric_col: str, count_col: str, template_type: str = None, msa_type: str = 'rMSA') -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for interaction plots: MSA class × Template count (0-4).
    
    Args:
        df: DataFrame with metrics, MSA class, and template counts
        metric_col: Name of metric column
        count_col: Name of template count column ('RBP_Template_Count' or 'Protein_Template_Count')
        template_type: Type of template ('RBP' or 'Protein') to include in labels
        msa_type: 'rMSA' or 'pMSA' to specify which MSA to use
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Select MSA class column based on type
    msa_class_col = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    if msa_class_col not in df.columns:
        return [], []
    
    # Filter out NaN values
    df_clean = df[[metric_col, msa_class_col, count_col]].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Order MSA classes based on type
    if msa_type == 'rMSA':
        msa_order = ['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
    else:
        msa_order = ['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
    
    available_msa = df_clean[msa_class_col].dropna().unique()
    msa_values = [m for m in msa_order if m in available_msa]
    for val in available_msa:
        if val not in msa_values:
            msa_values.append(val)
    
    # Get unique template counts (0-4) and sort
    template_counts = sorted(df_clean[count_col].dropna().unique())
    # Limit to 0-4 if there are higher counts
    template_counts = [tc for tc in template_counts if 0 <= tc <= 4]
    
    data_values = []
    labels = []
    
    # Create labels as "MSA_class × Template_Count Template_Type"
    for msa_class in msa_values:
        for count in template_counts:
            mask = (df_clean[msa_class_col] == msa_class) & (df_clean[count_col] == count)
            values = df_clean[mask][metric_col].values
            if len(values) > 0:
                # Convert to numeric and filter out NaN
                numeric_values = pd.to_numeric(values, errors='coerce')
                numeric_values = numeric_values[~pd.isna(numeric_values)]
                if len(numeric_values) > 0:
                    data_values.append(numeric_values)
                    # Add template type to label if provided
                    if template_type:
                        labels.append(f"{msa_class} × {int(count)} {template_type}")
                    else:
                        labels.append(f"{msa_class} × {int(count)}")
    
    return data_values, labels


def create_colored_box_plot_by_msa(plot_creator, table_source, data_values, labels, plot_name, y_label):
    """
    Create a box plot with colors based on MSA bins.
    All boxes with the same MSA class get the same color.
    
    Args:
        plot_creator: PlotCreator instance
        table_source: Table source name
        data_values: List of arrays containing data for each group
        labels: List of labels (format: "Category × MSA_class" or "MSA_class × Category")
        plot_name: Name for the plot file
        y_label: Label for y-axis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import re
    
    # Filter out empty arrays and convert to numeric
    filtered = []
    for vals, lab in zip(data_values, labels):
        if len(vals) > 0:
            # Convert to numeric, dropping any non-numeric values
            numeric_vals = pd.to_numeric(vals, errors='coerce')
            numeric_vals = numeric_vals[~pd.isna(numeric_vals)]
            if len(numeric_vals) > 0:
                filtered.append((np.array(numeric_vals), lab))
    
    if not filtered:
        print("No data to plot!")
        return
    
    data_values_filtered, labels_filtered = zip(*filtered)
    data_values_filtered = list(data_values_filtered)
    
    # Extract MSA class from labels (format: "Category × 0 rMSA" or "0 pMSA × Category")
    msa_classes = []
    import re
    
    # Order from most specific to least specific to avoid substring matching issues
    # Only support rMSA and pMSA formats
    msa_order = ['1000+ rMSA', '20-1000 rMSA', '0-20 rMSA', '0 rMSA', 
                  '1000+ pMSA', '20-1000 pMSA', '0-20 pMSA', '0 pMSA']
    
    for label in labels_filtered:
        msa_found = None
        
        # First, try direct matching with the full MSA class strings
        for msa_class in msa_order:
            if msa_class in label:
                msa_found = msa_class
                break
        
        # If not found, try to extract MSA class pattern using regex
        if not msa_found:
            # Pattern to match: number(s) or range, optional space, then rMSA or pMSA
            # This should match patterns like "0 rMSA", "0-20 pMSA", "1000+ rMSA"
            msa_pattern = r'(\d+(?:-\d+)?\+?)\s*(rMSA|pMSA)'
            match = re.search(msa_pattern, label)
            if match:
                size_part = match.group(1)
                msa_type_part = match.group(2)
                
                # Reconstruct the MSA class based on size part
                if size_part == '0':
                    msa_found = f'0 {msa_type_part}'
                elif size_part == '0-20':
                    msa_found = f'0-20 {msa_type_part}'
                elif size_part == '20-1000':
                    msa_found = f'20-1000 {msa_type_part}'
                elif '1000' in size_part:
                    msa_found = f'1000+ {msa_type_part}'
        
        if msa_found:
            msa_classes.append(msa_found)
        else:
            # Debug: uncomment to see which labels are failing
            # print(f"Warning: Could not extract MSA class from label: {label}")
            msa_classes.append(None)  # Default if not found
    
    # Map colors based on MSA size category (0, 0-20, 20-1000, 1000+) regardless of rMSA/pMSA
    # This ensures consistent coloring across both rMSA and pMSA plots
    size_category_map = {
        '0': 0,           # 0 rMSA or 0 pMSA -> index 0
        '0-20': 1,        # 0-20 rMSA or 0-20 pMSA -> index 1
        '20-1000': 2,     # 20-1000 rMSA or 20-1000 pMSA -> index 2
        '1000+': 3        # 1000+ rMSA or 1000+ pMSA -> index 3
    }
    
    # Always use 4 colors for the 4 MSA size categories
    palette_colors = sns.color_palette(n_colors=4)
    
    # Create color map based on size category
    # First, get unique MSA classes to build the map
    unique_msa_classes = list(set([msa for msa in msa_classes if msa is not None]))
    color_map = {}
    for msa_class in unique_msa_classes:
        # Extract size category from label (e.g., "0 rMSA" -> "0", "20-1000 pMSA" -> "20-1000")
        # Check in order from most specific to least specific to avoid wrong matches
        # Order: 1000+, 20-1000, 0-20, 0 (most specific first)
        matched = False
        if '1000+' in msa_class:
            color_map[msa_class] = palette_colors[3]
            matched = True
        elif '20-1000' in msa_class:
            color_map[msa_class] = palette_colors[2]
            matched = True
        elif '0-20' in msa_class:
            color_map[msa_class] = palette_colors[1]
            matched = True
        elif msa_class.startswith('0 ') or msa_class.startswith('0\t'):
            # Check for "0 rMSA" or "0 pMSA" (must be at start to avoid matching "0-20")
            color_map[msa_class] = palette_colors[0]
            matched = True
        
        # If no match found, use default color
        if not matched:
            color_map[msa_class] = palette_colors[0]
    
    # Create list of colors for each box based on MSA class
    # Ensure we have a color for every box, even if MSA class extraction failed
    color_list = []
    for idx, msa_class in enumerate(msa_classes):
        if msa_class and msa_class in color_map:
            color_list.append(color_map[msa_class])
        elif msa_class:
            # MSA class was extracted but not in color_map - extract size directly
            # Check in order from most specific to least specific
            if '1000+' in msa_class:
                color_list.append(palette_colors[3])
            elif '20-1000' in msa_class:
                color_list.append(palette_colors[2])
            elif '0-20' in msa_class:
                color_list.append(palette_colors[1])
            elif msa_class.startswith('0 ') or msa_class.startswith('0\t'):
                color_list.append(palette_colors[0])
            else:
                # No size category found, use first color
                color_list.append(palette_colors[0])
        else:
            # If MSA class extraction failed, try to extract directly from the label
            # This is a fallback to ensure we always have colors
            if idx < len(labels_filtered):
                label = labels_filtered[idx]
                # Try to extract MSA class from label one more time
                msa_pattern = r'(\d+(?:-\d+)?\+?)\s*(rMSA|pMSA)'
                match = re.search(msa_pattern, label)
                if match:
                    size_part = match.group(1)
                    # Map to color based on size category
                    if size_part == '0':
                        color_list.append(palette_colors[0])
                    elif size_part == '0-20':
                        color_list.append(palette_colors[1])
                    elif size_part == '20-1000':
                        color_list.append(palette_colors[2])
                    elif '1000' in size_part:
                        color_list.append(palette_colors[3])
                    else:
                        color_list.append(palette_colors[0])
                else:
                    color_list.append(palette_colors[0])  # Default to first color
            else:
                color_list.append(palette_colors[0])  # Default to first color
    
    # Ensure color_list has the same length as data_values_filtered
    if len(color_list) != len(data_values_filtered):
        # Pad or truncate to match
        if len(color_list) < len(data_values_filtered):
            color_list.extend([sns.color_palette()[0]] * (len(data_values_filtered) - len(color_list)))
        else:
            color_list = color_list[:len(data_values_filtered)]
    
    # Set plot style first (this sets font sizes and other defaults)
    plot_creator.set_plot_style()
    
    # Determine figure size for this specific plot
    num_labels = len(labels_filtered)
    fig_width = max(16, min(24, num_labels * 0.6))
    fig_height = 10
    
    # Create figure with specific size (don't modify global rcParams)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
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
    
    # Font sizes are set by set_plot_style() via rcParams
    plt.ylabel(y_label)
    plt.xlabel('')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)
    
    # Save plot
    title = f"{plot_name}_{table_source}"
    plot_creator.save_plot(title)
    plt.close(fig)


def create_colored_box_plot_by_template_presence(plot_creator, table_source, data_values, labels, plot_name, y_label):
    """
    Create a box plot with colors based on template presence categories.
    All boxes with the same template presence category get the same color.
    
    Args:
        plot_creator: PlotCreator instance
        table_source: Table source name
        data_values: List of arrays containing data for each group
        labels: List of labels (format: "MSA × Template_Presence" or "Template_Presence × MSA")
        plot_name: Name for the plot file
        y_label: Label for y-axis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # Filter out empty arrays and convert to numeric
    filtered = []
    for vals, lab in zip(data_values, labels):
        if len(vals) > 0:
            # Convert to numeric, dropping any non-numeric values
            numeric_vals = pd.to_numeric(vals, errors='coerce')
            numeric_vals = numeric_vals[~pd.isna(numeric_vals)]
            if len(numeric_vals) > 0:
                filtered.append((np.array(numeric_vals), lab))
    
    if not filtered:
        print("No data to plot!")
        return
    
    data_values_filtered, labels_filtered = zip(*filtered)
    data_values_filtered = list(data_values_filtered)
    
    # Extract template presence category from labels (e.g., "Neither" from "-MSA × Neither" or "Neither × -MSA")
    template_categories = []
    template_order = ['Neither', 'Protein', 'RBP', 'Both', 'Identical']
    
    for label in labels_filtered:
        template_found = None
        # Check if label contains any template category (check in order)
        for template_cat in template_order:
            if template_cat in label:
                template_found = template_cat
                break
        if template_found:
            template_categories.append(template_found)
        else:
            template_categories.append(None)  # Default if not found
    
    # Get unique template categories and create color map
    unique_template_cats = [cat for cat in template_order if cat in template_categories]
    # Add any remaining categories not in the order
    for cat in template_categories:
        if cat and cat not in unique_template_cats:
            unique_template_cats.append(cat)
    
    # Use consistent colors for template categories
    palette_colors = sns.color_palette(n_colors=len(unique_template_cats))
    color_map = dict(zip(unique_template_cats, palette_colors))
    
    # Create list of colors for each box based on template category
    color_list = [color_map.get(template_cat, sns.color_palette()[0]) if template_cat else sns.color_palette()[0] 
                  for template_cat in template_categories]
    
    # Set plot style first (this sets font sizes and other defaults)
    plot_creator.set_plot_style()
    
    # Determine figure size for this specific plot
    num_labels = len(labels_filtered)
    fig_width = max(16, min(24, num_labels * 0.6))
    fig_height = 10
    
    # Create figure with specific size (don't modify global rcParams)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
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
    
    # Font sizes are set by set_plot_style() via rcParams
    plt.ylabel(y_label)
    plt.xlabel('')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)
    
    # Save plot
    title = f"{plot_name}_{table_source}"
    plot_creator.save_plot(title)
    plt.close(fig)


def create_colored_box_plot(plot_creator, table_source, data_values, labels, plot_name, y_label):
    """
    Create a box plot with colors based on RBP template count.
    All boxes with the same RBP count get the same color (regardless of Protein count).
    
    Args:
        plot_creator: PlotCreator instance
        table_source: Table source name
        data_values: List of arrays containing data for each group
        labels: List of labels (format: "MSA_class × X Prot Y RBP")
        plot_name: Name for the plot file
        y_label: Label for y-axis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # Filter out empty arrays and convert to numeric
    filtered = []
    for vals, lab in zip(data_values, labels):
        if len(vals) > 0:
            # Convert to numeric, dropping any non-numeric values
            numeric_vals = pd.to_numeric(vals, errors='coerce')
            numeric_vals = numeric_vals[~pd.isna(numeric_vals)]
            if len(numeric_vals) > 0:
                filtered.append((np.array(numeric_vals), lab))
    
    if not filtered:
        print("No data to plot!")
        return
    
    data_values_filtered, labels_filtered = zip(*filtered)
    data_values_filtered = list(data_values_filtered)
    
    # Extract RBP count from labels (e.g., "1" from "0 MSA × 0 Prot 1 RBP")
    rbp_counts = []
    for label in labels_filtered:
        if ' RBP' in label:
            # Extract RBP count: look for pattern "X RBP" where X is a number
            import re
            match = re.search(r'(\d+)\s+RBP', label)
            if match:
                rbp_counts.append(int(match.group(1)))
            else:
                rbp_counts.append(0)  # Default if not found
        else:
            rbp_counts.append(0)  # Default if no RBP in label
    
    # Get unique RBP counts and create color map
    unique_rbp_counts = sorted(set(rbp_counts))
    palette_colors = sns.color_palette(n_colors=len(unique_rbp_counts))
    color_map = dict(zip(unique_rbp_counts, palette_colors))
    
    # Create list of colors for each box based on RBP count
    color_list = [color_map.get(rbp_count, sns.color_palette()[0]) for rbp_count in rbp_counts]
    
    # Set plot style first (this sets font sizes and other defaults)
    plot_creator.set_plot_style()
    
    # Determine figure size for this specific plot
    num_labels = len(labels_filtered)
    fig_width = max(16, min(24, num_labels * 0.6))
    fig_height = 10
    
    # Create figure with specific size (don't modify global rcParams)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create box plot
    ax = plt.gca()
    # Font sizes are set by set_plot_style() via rcParams
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
    
    # Font sizes are set by set_plot_style() via rcParams
    plt.ylabel(y_label)
    plt.xlabel('')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)
    
    # Save plot
    title = f"{plot_name}_{table_source}"
    plot_creator.save_plot(title)
    plt.close(fig)


def prepare_combined_template_count_msa_interaction(df: pd.DataFrame, metric_col: str, msa_type: str = 'rMSA') -> Tuple[List[np.ndarray], List[str]]:
    """
    Prepare data for interaction plots: MSA class × Combined template counts (Protein + RBP, 0-4 each).
    
    Args:
        df: DataFrame with metrics, MSA class, and template counts
        metric_col: Name of metric column
        msa_type: 'rMSA' or 'pMSA' to specify which MSA to use
    
    Returns:
        Tuple of (data_values list, labels list)
    """
    # Select MSA class column based on type
    msa_class_col = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    if msa_class_col not in df.columns:
        return [], []
    
    # Filter out NaN values
    df_clean = df[[metric_col, msa_class_col, 'RBP_Template_Count', 'Protein_Template_Count']].dropna()
    
    if df_clean.empty:
        return [], []
    
    # Order MSA classes based on type
    if msa_type == 'rMSA':
        msa_order = ['0 rMSA', '0-20 rMSA', '20-1000 rMSA', '1000+ rMSA']
    else:
        msa_order = ['0 pMSA', '0-20 pMSA', '20-1000 pMSA', '1000+ pMSA']
    
    available_msa = df_clean[msa_class_col].dropna().unique()
    msa_values = [m for m in msa_order if m in available_msa]
    for val in available_msa:
        if val not in msa_values:
            msa_values.append(val)
    
    # Get unique template counts (0-4) for both types
    rbp_counts = sorted([int(tc) for tc in df_clean['RBP_Template_Count'].dropna().unique() if 0 <= tc <= 4])
    protein_counts = sorted([int(tc) for tc in df_clean['Protein_Template_Count'].dropna().unique() if 0 <= tc <= 4])
    
    data_values = []
    labels = []
    
    # Create labels as "MSA_class × Prot_count Prot RBP_count RBP"
    for msa_class in msa_values:
        for prot_count in protein_counts:
            for rbp_count in rbp_counts:
                mask = (
                    (df_clean[msa_class_col] == msa_class) & 
                    (df_clean['Protein_Template_Count'] == prot_count) & 
                    (df_clean['RBP_Template_Count'] == rbp_count)
                )
                values = df_clean[mask][metric_col].values
                if len(values) > 0:
                    # Convert to numeric and filter out NaN
                    numeric_values = pd.to_numeric(values, errors='coerce')
                    numeric_values = numeric_values[~pd.isna(numeric_values)]
                    if len(numeric_values) > 0:
                        data_values.append(numeric_values)
                        # Create label: "0 MSA × 0 Prot 1 RBP" or "0 MSA × 2 Prot 0 RBP"
                        label = f"{msa_class} × {prot_count} Prot {rbp_count} RBP"
                        labels.append(label)
    
    return data_values, labels


def prepare_stratified_data(df: pd.DataFrame, metric_col: str,
                           primary_group: str, stratify_group: str) -> Dict[str, Tuple[List[np.ndarray], List[str]]]:
    """
    Prepare data for stratified plots (e.g., template presence stratified by MSA class).
    
    Args:
        df: DataFrame with metrics and classifications
        metric_col: Name of metric column
        primary_group: Primary grouping column (e.g., 'Template_Presence')
        stratify_group: Stratification column (e.g., 'msa_class')
    
    Returns:
        Dictionary mapping stratify values to (data_values, labels) tuples
    """
    # Filter out NaN values
    df_clean = df[[metric_col, primary_group, stratify_group]].dropna()
    
    if df_clean.empty:
        return {}
    
    stratify_values = sorted(df_clean[stratify_group].dropna().unique())
    primary_values = sorted(df_clean[primary_group].dropna().unique())
    
    result = {}
    
    for stratify_val in stratify_values:
        df_stratified = df_clean[df_clean[stratify_group] == stratify_val]
        
        data_values = []
        labels = []
        
        for primary_val in primary_values:
            values = df_stratified[df_stratified[primary_group] == primary_val][metric_col].values
            if len(values) > 0:
                data_values.append(values)
                labels.append(str(primary_val))
        
        if data_values:
            result[str(stratify_val)] = (data_values, labels)
    
    return result


def calculate_statistics(df: pd.DataFrame, metric_col: str) -> Dict:
    """
    Calculate average metrics and p-values for template presence categories.
    
    Args:
        df: DataFrame with metrics and template information
        metric_col: Name of metric column
    
    Returns:
        Dictionary with statistics
    """
    stats_dict = {
        'metric': metric_col,
        'metric_label': LABEL_MAP.get(metric_col, metric_col),
        'averages': {},
        'counts': {},
        'p_values': {}
    }
    
    # Filter out NaN values
    df_clean = df[[metric_col, 'Template_Presence']].dropna()
    
    if df_clean.empty:
        return stats_dict
    
    # Get unique categories
    categories = sorted(df_clean['Template_Presence'].unique())
    
    # Calculate averages and counts for each category
    for category in categories:
        values = df_clean[df_clean['Template_Presence'] == category][metric_col].values
        if len(values) > 0:
            # Convert to numeric, coercing errors to NaN
            numeric_values = pd.to_numeric(values, errors='coerce')
            # Remove NaN values
            numeric_values = numeric_values[~pd.isna(numeric_values)]
            
            if len(numeric_values) > 0:
                stats_dict['averages'][category] = np.mean(numeric_values)
                stats_dict['counts'][category] = len(numeric_values)
            else:
                # No valid numeric values for this category
                stats_dict['averages'][category] = None
                stats_dict['counts'][category] = 0
    
    # Calculate p-values between groups (pairwise comparisons)
    # Use Mann-Whitney U test for non-parametric comparison
    p_values = {}
    categories_list = list(categories)
    
    for i, cat1 in enumerate(categories_list):
        for cat2 in categories_list[i+1:]:
            values1_raw = df_clean[df_clean['Template_Presence'] == cat1][metric_col].values
            values2_raw = df_clean[df_clean['Template_Presence'] == cat2][metric_col].values
            
            # Convert to numeric and filter out NaN
            values1 = pd.to_numeric(values1_raw, errors='coerce')
            values1 = values1[~pd.isna(values1)]
            values2 = pd.to_numeric(values2_raw, errors='coerce')
            values2 = values2[~pd.isna(values2)]
            
            if len(values1) > 0 and len(values2) > 0:
                try:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    comparison_key = f"{cat1} vs {cat2}"
                    p_values[comparison_key] = p_value
                except Exception as e:
                    print(f"Warning: Could not calculate p-value for {cat1} vs {cat2}: {e}")
                    p_values[f"{cat1} vs {cat2}"] = None
    
    stats_dict['p_values'] = p_values
    
    return stats_dict


def write_statistics_to_file(df: pd.DataFrame, output_dir: str, check_exact: bool = False):
    """
    Write average metrics and p-values to a text file.
    Args:
        df: DataFrame with metrics and template information
        output_dir: Directory where plots are saved (where to save the statistics file)
        check_exact: Whether exact template matching was used
    """
    # Get available metrics
    available_metrics = [col for col in LABEL_MAP.keys() if col in df.columns]
    
    if not available_metrics:
        print("No metrics found for statistics")
        return
    
    # Create output file path in the same directory as plots
    output_path = os.path.join(output_dir, "template_metrics_statistics.txt")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Template Metrics Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        if check_exact:
            f.write("Note: Exact template matching enabled (Identical category included)\n\n")
        
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Metrics analysed: {len(available_metrics)}\n\n")
        
        # Template presence distribution
        if 'Template_Presence' in df.columns:
            f.write("Template Presence Distribution:\n")
            f.write("-" * 80 + "\n")
            presence_counts = df['Template_Presence'].value_counts()
            for category, count in presence_counts.items():
                f.write(f"  {category}: {count} samples\n")
            f.write("\n")
        
        # MSA class distribution
        if 'msa_class' in df.columns:
            f.write("MSA Class Distribution:\n")
            f.write("-" * 80 + "\n")
            msa_counts = df['msa_class'].value_counts()
            msa_order = ['0 MSA', '0-20 MSA', '20-1000 MSA', '1000+ MSA']
            for msa_class in msa_order:
                if msa_class in msa_counts.index:
                    count = msa_counts[msa_class]
                    f.write(f"  {msa_class}: {count} samples\n")
            f.write("\n")
        
        # Template counts by MSA class
        if 'msa_class' in df.columns:
            if 'RBP_Template_Count' in df.columns:
                f.write("RBP Template Count by MSA Class:\n")
                f.write("-" * 80 + "\n")
                msa_order = ['0 MSA', '0-20 MSA', '20-1000 MSA', '1000+ MSA']
                for msa_class in msa_order:
                    msa_data = df[df['msa_class'] == msa_class]['RBP_Template_Count'].dropna()
                    if len(msa_data) > 0:
                        avg_count = np.mean(msa_data)
                        median_count = np.median(msa_data)
                        f.write(f"  {msa_class}: Mean = {avg_count:.2f}, Median = {median_count:.1f}, n = {len(msa_data)}\n")
                f.write("\n")
            
            if 'Protein_Template_Count' in df.columns:
                f.write("Protein Template Count by MSA Class:\n")
                f.write("-" * 80 + "\n")
                msa_order = ['0 MSA', '0-20 MSA', '20-1000 MSA', '1000+ MSA']
                for msa_class in msa_order:
                    msa_data = df[df['msa_class'] == msa_class]['Protein_Template_Count'].dropna()
                    if len(msa_data) > 0:
                        avg_count = np.mean(msa_data)
                        median_count = np.median(msa_data)
                        f.write(f"  {msa_class}: Mean = {avg_count:.2f}, Median = {median_count:.1f}, n = {len(msa_data)}\n")
                f.write("\n")
        
        # Statistics for each metric
        for metric_col in available_metrics:
            stats_dict = calculate_statistics(df, metric_col)
            metric_label = stats_dict['metric_label']
            
            f.write("=" * 80 + "\n")
            f.write(f"Metric: {metric_col} ({metric_label})\n")
            f.write("=" * 80 + "\n\n")
            
            # Averages
            f.write("Average Values by Template Presence:\n")
            f.write("-" * 80 + "\n")
            if stats_dict['averages']:
                # Order: Neither, Protein, RBP, Both, Identical
                category_order = ["Neither", "Protein", "RBP", "Both", "Identical"]
                for category in category_order:
                    if category in stats_dict['averages']:
                        avg = stats_dict['averages'][category]
                        count = stats_dict['counts'][category]
                        f.write(f"  {category:15s}: {avg:.4f} (n={count})\n")
                # Add any remaining categories not in the order
                for category in stats_dict['averages']:
                    if category not in category_order:
                        avg = stats_dict['averages'][category]
                        count = stats_dict['counts'][category]
                        f.write(f"  {category:15s}: {avg:.4f} (n={count})\n")
            f.write("\n")
            
            # P-values
            f.write("P-values (Mann-Whitney U test):\n")
            f.write("-" * 80 + "\n")
            if stats_dict['p_values']:
                for comparison, p_val in sorted(stats_dict['p_values'].items()):
                    if p_val is not None:
                        significance = ""
                        if p_val < 0.001:
                            significance = "***"
                        elif p_val < 0.01:
                            significance = "**"
                        elif p_val < 0.05:
                            significance = "*"
                        f.write(f"  {comparison:30s}: p = {p_val:.6f} {significance}\n")
                    else:
                        f.write(f"  {comparison:30s}: p = N/A\n")
            else:
                f.write("  No comparisons available (insufficient groups)\n")
            f.write("\n")
            f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
            f.write("\n")
            
            # Statistics by Template Presence and MSA Class
            if 'msa_class' in df.columns and 'has_msa' in df.columns:
                f.write("Metric Values by Template Presence and MSA Class:\n")
                f.write("-" * 80 + "\n")
                
                # Get unique template presence categories
                template_categories = sorted(df['Template_Presence'].dropna().unique())
                msa_order = ['0 MSA', '0-20 MSA', '20-1000 MSA', '1000+ MSA']
                
                # Create a table showing averages for each combination
                for template_cat in template_categories:
                    f.write(f"\n  {template_cat}:\n")
                    for msa_class in msa_order:
                        mask = (df['Template_Presence'] == template_cat) & (df['msa_class'] == msa_class)
                        values = df[mask][metric_col].dropna()
                        if len(values) > 0:
                            numeric_values = pd.to_numeric(values, errors='coerce')
                            numeric_values = numeric_values[~pd.isna(numeric_values)]
                            if len(numeric_values) > 0:
                                avg = np.mean(numeric_values)
                                f.write(f"    {msa_class:20s}: {avg:.4f} (n={len(numeric_values)})\n")
                
                f.write("\n")
                
                # Calculate p-values for template presence within each MSA class
                f.write("P-values: Template Presence comparisons within each MSA Class:\n")
                f.write("-" * 80 + "\n")
                for msa_class in msa_order:
                    msa_df = df[df['msa_class'] == msa_class]
                    if len(msa_df) > 0:
                        f.write(f"\n  {msa_class}:\n")
                        template_cats_in_msa = sorted(msa_df['Template_Presence'].dropna().unique())
                        if len(template_cats_in_msa) > 1:
                            for i, cat1 in enumerate(template_cats_in_msa):
                                for cat2 in template_cats_in_msa[i+1:]:
                                    values1_raw = msa_df[msa_df['Template_Presence'] == cat1][metric_col].values
                                    values2_raw = msa_df[msa_df['Template_Presence'] == cat2][metric_col].values
                                    
                                    values1 = pd.to_numeric(values1_raw, errors='coerce')
                                    values1 = values1[~pd.isna(values1)]
                                    values2 = pd.to_numeric(values2_raw, errors='coerce')
                                    values2 = values2[~pd.isna(values2)]
                                    
                                    if len(values1) > 0 and len(values2) > 0:
                                        try:
                                            statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                                            significance = ""
                                            if p_value < 0.001:
                                                significance = "***"
                                            elif p_value < 0.01:
                                                significance = "**"
                                            elif p_value < 0.05:
                                                significance = "*"
                                            f.write(f"    {cat1} vs {cat2}: p = {p_value:.6f} {significance}\n")
                                        except Exception:
                                            f.write(f"    {cat1} vs {cat2}: p = N/A\n")
                        else:
                            f.write(f"    (Only one template category present)\n")
                f.write("\n")
                
                # Calculate p-values for MSA class within each template presence category
                f.write("P-values: MSA Class comparisons within each Template Presence category:\n")
                f.write("-" * 80 + "\n")
                for template_cat in template_categories:
                    template_df = df[df['Template_Presence'] == template_cat]
                    if len(template_df) > 0:
                        f.write(f"\n  {template_cat}:\n")
                        msa_classes_in_template = [m for m in msa_order if m in template_df['msa_class'].dropna().unique()]
                        if len(msa_classes_in_template) > 1:
                            for i, msa1 in enumerate(msa_classes_in_template):
                                for msa2 in msa_classes_in_template[i+1:]:
                                    values1_raw = template_df[template_df['msa_class'] == msa1][metric_col].values
                                    values2_raw = template_df[template_df['msa_class'] == msa2][metric_col].values
                                    
                                    values1 = pd.to_numeric(values1_raw, errors='coerce')
                                    values1 = values1[~pd.isna(values1)]
                                    values2 = pd.to_numeric(values2_raw, errors='coerce')
                                    values2 = values2[~pd.isna(values2)]
                                    
                                    if len(values1) > 0 and len(values2) > 0:
                                        try:
                                            statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                                            significance = ""
                                            if p_value < 0.001:
                                                significance = "***"
                                            elif p_value < 0.01:
                                                significance = "**"
                                            elif p_value < 0.05:
                                                significance = "*"
                                            f.write(f"    {msa1} vs {msa2}: p = {p_value:.6f} {significance}\n")
                                        except Exception:
                                            f.write(f"    {msa1} vs {msa2}: p = N/A\n")
                        else:
                            f.write(f"    (Only one MSA class present)\n")
                f.write("\n")
    
    print(f"\nStatistics written to: {output_path}")


def create_plots_for_msa_type(df: pd.DataFrame, metric_col: str, metric_label: str, 
                              plot_creator: PlotCreator, pred_table: str, msa_type: str):
    """
    Create all MSA-related plots for a specific metric and MSA type.
    
    Args:
        df: DataFrame with metrics and template information
        metric_col: Name of metric column
        metric_label: Display label for metric
        plot_creator: PlotCreator instance
        pred_table: Name of prediction table
        msa_type: 'rMSA' or 'pMSA'
    """
    # Check if MSA data is available for this type
    has_msa_col = 'has_rmsa' if msa_type == 'rMSA' else 'has_pmsa'
    msa_class_col = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    has_msa = has_msa_col in df.columns and df[has_msa_col].notna().any()
    has_msa_class = msa_class_col in df.columns and df[msa_class_col].notna().any()
    
    # Determine MSA column names for interaction plots
    has_msa_name = 'has_rmsa' if msa_type == 'rMSA' else 'has_pmsa'
    msa_class_name = 'rmsa_class' if msa_type == 'rMSA' else 'pmsa_class'
    
    # Suffix for plot names
    msa_suffix = '_rMSA' if msa_type == 'rMSA' else '_pMSA'
    
    # 4. Interaction plot: MSA × Template presence (if MSA data available)
    if has_msa:
        interaction_data, interaction_labels = prepare_interaction_data(
            df, metric_col, has_msa_name, 'Template_Presence', msa_type=msa_type
        )
        if interaction_data:
            plot_name = f"{metric_col}_MSA_x_Template_presence{msa_suffix}"
            y_label = metric_label
            
            # Use colored box plot with template presence coloring
            create_colored_box_plot_by_template_presence(
                plot_creator, pred_table, interaction_data, 
                interaction_labels, plot_name, y_label
            )
    
    # 5. Plot RBP template count by MSA class
    if has_msa_class and 'RBP_Template_Count' in df.columns:
        rbp_msa_data, rbp_msa_labels = prepare_template_count_by_msa(df, 'RBP_Template_Count', msa_type=msa_type)
        if rbp_msa_data:
            plot_name = f"RBP_template_count_by_MSA_class{msa_suffix}"
            y_label = "Number of RBP Templates"
            
            plot_creator.get_boxplot(
                table_source=pred_table,
                data_values=rbp_msa_data,
                labels=rbp_msa_labels,
                name=plot_name,
                yAxis_label=y_label,
                rotation=0
            )
    
    # 6. Plot Protein template count by MSA class
    if has_msa_class and 'Protein_Template_Count' in df.columns:
        protein_msa_data, protein_msa_labels = prepare_template_count_by_msa(df, 'Protein_Template_Count', msa_type=msa_type)
        if protein_msa_data:
            plot_name = f"Protein_template_count_by_MSA_class{msa_suffix}"
            y_label = "Number of Protein Templates"
            
            plot_creator.get_boxplot(
                table_source=pred_table,
                data_values=protein_msa_data,
                labels=protein_msa_labels,
                name=plot_name,
                yAxis_label=y_label,
                rotation=0
            )
    
    # 7. Interaction plot: RBP Template Presence × MSA Class
    # Use backward compatibility 'msa_class' for rMSA, 'pmsa_class' for pMSA
    msa_class_col = 'msa_class' if msa_type == 'rMSA' else 'pmsa_class'
    if msa_class_col in df.columns and 'RBP_Template_Presence' in df.columns:
        rbp_msa_interaction_data, rbp_msa_interaction_labels = prepare_interaction_data(
            df, metric_col, 'RBP_Template_Presence', msa_class_col, msa_type=msa_type
        )
        if rbp_msa_interaction_data:
            plot_name = f"{metric_col}_RBP_template_x_MSA_class{msa_suffix}"
            y_label = metric_label
            
            # Use colored box plot with MSA bin coloring
            create_colored_box_plot_by_msa(
                plot_creator, pred_table, rbp_msa_interaction_data, 
                rbp_msa_interaction_labels, plot_name, y_label
            )
    
    # 8. Interaction plot: Protein Template Presence × MSA Class
    # Use backward compatibility 'msa_class' for rMSA, 'pmsa_class' for pMSA
    if msa_class_col in df.columns and 'Protein_Template_Presence' in df.columns:
        protein_msa_interaction_data, protein_msa_interaction_labels = prepare_interaction_data(
            df, metric_col, 'Protein_Template_Presence', msa_class_col, msa_type=msa_type
        )
        if protein_msa_interaction_data:
            plot_name = f"{metric_col}_Protein_template_x_MSA_class{msa_suffix}"
            y_label = metric_label
            
            # Use colored box plot with MSA bin coloring
            create_colored_box_plot_by_msa(
                plot_creator, pred_table, protein_msa_interaction_data, 
                protein_msa_interaction_labels, plot_name, y_label
            )
    
    # 9. Combined interaction plot: Exclusive Template Category × MSA Class
    if has_msa_class and 'Exclusive_Template_Category' in df.columns:
        exclusive_msa_interaction_data, exclusive_msa_interaction_labels = prepare_interaction_data(
            df, metric_col, 'Exclusive_Template_Category', msa_class_name, msa_type=msa_type
        )
        if exclusive_msa_interaction_data:
            plot_name = f"{metric_col}_Exclusive_template_x_MSA_class{msa_suffix}"
            y_label = metric_label
            
            # Always use colored box plot with MSA bin coloring
            create_colored_box_plot_by_msa(
                plot_creator, pred_table, exclusive_msa_interaction_data, 
                exclusive_msa_interaction_labels, plot_name, y_label
            )
    
    # 10. Combined interaction plot: MSA bins × (Protein + RBP template counts)
    if has_msa_class and 'RBP_Template_Count' in df.columns and 'Protein_Template_Count' in df.columns:
        combined_count_msa_data, combined_count_msa_labels = prepare_combined_template_count_msa_interaction(
            df, metric_col, msa_type=msa_type
        )
        if combined_count_msa_data:
            plot_name = f"{metric_col}_MSA_class_x_combined_template_count{msa_suffix}"
            y_label = metric_label
            
            # Create box plot with color coding by template count combination
            create_colored_box_plot(
                plot_creator, pred_table, combined_count_msa_data, combined_count_msa_labels,
                plot_name, y_label
            )


def create_plots(df: pd.DataFrame, check_exact: bool = False):
    """
    Create all plots for each metric.
    
    Args:
        df: DataFrame with metrics and template information
        check_exact: Whether exact template matching was used
    
    Returns:
        results_dir: Directory where plots are saved
    """
    config = StartConfig()
    pred_table = config.pred_table
    
    # Create plot creator
    plot_creator = PlotCreator(
        plot_type='template_metrics',
        msa_option=None,
        single_chain_only=False
    )
    
    # Get results directory where plots are saved
    results_dir = plot_creator.results_dir
    
    # Get available metrics
    available_metrics = [col for col in LABEL_MAP.keys() if col in df.columns]
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    has_msa = 'has_msa' in df.columns and df['has_msa'].notna().any()
    
    for metric_col in available_metrics:
        metric_label = LABEL_MAP.get(metric_col, metric_col)
        print(f"\nProcessing {metric_col} ({metric_label})...")
        
        # 1. Plot by template presence
        presence_data, presence_labels = prepare_presence_data(df, metric_col)
        if presence_data:
            plot_name = f"{metric_col}_by_template_presence"
            y_label = metric_label
            
            plot_creator.get_violin_plot(
                table_source=pred_table,
                data_values=presence_data,
                labels=presence_labels,
                name=plot_name,
                yAxis_label=y_label,
                xAxis_label="Template Presence",
                rotation=0
            )
        else:
            print(f"No data for presence plot: {metric_col}")
        
        # 2. Plot by RBP template count
        if 'RBP_Template_Count' in df.columns:
            rbp_count_data, rbp_count_labels = prepare_count_data(df, metric_col, 'RBP')
            if rbp_count_data:
                plot_name = f"{metric_col}_by_RBP_template_count"
                y_label = metric_label
                
                plot_creator.get_violin_plot(
                    table_source=pred_table,
                    data_values=rbp_count_data,
                    labels=rbp_count_labels,
                    name=plot_name,
                    yAxis_label=y_label,
                    xAxis_label="RBP Template",
                    rotation=0
                )
            else:
                print(f"No data for RBP count plot: {metric_col}")
        
        # 3. Plot by Protein template count
        if 'Protein_Template_Count' in df.columns:
            protein_count_data, protein_count_labels = prepare_count_data(df, metric_col, 'Protein')
            if protein_count_data:
                plot_name = f"{metric_col}_by_Protein_template_count"
                y_label = metric_label
                
                plot_creator.get_violin_plot(
                    table_source=pred_table,
                    data_values=protein_count_data,
                    labels=protein_count_labels,
                    name=plot_name,
                    yAxis_label=y_label,
                    xAxis_label="Protein Template",
                    rotation=0
                )
            else:
                print(f"No data for Protein count plot: {metric_col}")
        
        # Generate plots for both rMSA and pMSA
        for msa_type in ['rMSA', 'pMSA']:
            create_plots_for_msa_type(df, metric_col, metric_label, plot_creator, pred_table, msa_type)
    
    return results_dir


def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot metrics based on template presence and count'
    )
    parser.add_argument(
        '--exact',
        action='store_true',
        help='Check for exact template matches (template PDB ID matches exp_db_id)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = StartConfig()
    db_path = config.get_database_path()
    pred_table = config.pred_table
    
    print(f"Database: {db_path}")
    print(f"Table: {pred_table}")
    print(f"Check exact templates: {args.exact}")
    print()
    
    # Load data
    print("Loading data from database...")
    df = load_data(db_path, pred_table, check_exact=args.exact)
    
    if df.empty:
        print("Error: No data loaded from database")
        return
    
    print(f"Loaded {len(df)} samples")
    print(f"Available metrics: {[col for col in LABEL_MAP.keys() if col in df.columns]}")
    if args.exact:
        exact_count = len(df[df['Template_Presence'] == 'Identical'])
        print(f"Identical template matches: {exact_count}")
    print()
    
    # Create plots and get results directory
    results_dir = create_plots(df, check_exact=args.exact)
    
    # Write statistics to file in the same directory as plots
    if results_dir:
        print(f"\nCalculating statistics and writing to: {results_dir}")
        write_statistics_to_file(df, results_dir, check_exact=args.exact)
    else:
        print("\nWarning: Could not determine results directory. Statistics not written.")


# --------------------------------------------------------------------------- #
# Composite-figure panels (for plots.composite_figures)
# --------------------------------------------------------------------------- #
# (suffix, x-axis label, preparer) for the composable violin panels.
_COMPOSITE_KINDS = (
    ("by_template_presence", "Template presence",
     lambda df, m: prepare_presence_data(df, m)),
    ("by_RBP_template_count", "RBP template",
     lambda df, m: prepare_count_data(df, m, "RBP")),
)
COMPOSITE_PANEL_NAMES = tuple(
    f"{metric}_{suffix}" for metric in LABEL_MAP for suffix, _, _ in _COMPOSITE_KINDS)


def _draw_template_panel(df, metric, prep, xlabel):
    from plots.publication_figure import sns_violin
    try:
        data, labels = prep(df, metric)
    except Exception:
        data, labels = [], []

    def draw(ax):
        sns_violin(ax, data, labels, ylabel=LABEL_MAP.get(metric, metric),
                   xlabel=xlabel)
    return draw


def composite_panels(names=None, single_chain_only=False, msa_option=None, db_path=None):
    """Return {base_name: (draw_fn, title)} for plots.composite_figures.

    Panels are ``<metric>_by_template_presence`` and
    ``<metric>_by_RBP_template_count`` for each metric in LABEL_MAP.
    """
    from plots.publication_figure import normalize_panel_name
    wanted = ([normalize_panel_name(n) for n in names]
              if names else list(COMPOSITE_PANEL_NAMES))

    config = StartConfig()
    db_path = db_path or config.get_database_path()
    df = load_data(db_path, config.pred_table)

    registry = {}
    for metric in LABEL_MAP:
        if metric not in df.columns:
            continue
        for suffix, xlabel, prep in _COMPOSITE_KINDS:
            registry[f"{metric}_{suffix}"] = (
                _draw_template_panel(df, metric, prep, xlabel),
                f"{LABEL_MAP.get(metric, metric)} by {xlabel.lower()}")
    return {n: registry[n] for n in wanted if n in registry}


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:2] == ["composite"]:
        from plots.composite_figures import run_composite_cli
        raise SystemExit(run_composite_cli(composite_panels, _sys.argv[2:],
                                           default_out="template_metrics_composite"))
    main()

