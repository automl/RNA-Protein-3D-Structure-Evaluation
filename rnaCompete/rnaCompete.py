import os
import re
import sys
import math
import pandas as pd
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig
from plots.plotCreator import PlotCreator
from Bio.Align import PairwiseAligner
import numpy as np

def find_motif_position(sequence, motif):
    """Find position of motif in sequence"""
    if not motif or not sequence:
        return -1
    print(motif)
    # Clean up the predicted motif format: "C(13), U(14)..." -> "CUCG..."
    if '(' in motif:
        try:
            # Extract just the nucleotides and join them
            clean_motif = ''.join(base.strip() for base in motif.split(',') if base.strip())
            clean_motif = ''.join(base[0] for base in clean_motif.split('('))
            print(clean_motif)
            return sequence.find(clean_motif)
        except:
            return -1
    
    return sequence.find(motif)

def calculate_similarity_score(interface_area, protein_length):
    """Weight-neutral interface score: interface area normalised by sqrt(protein length).

    Replaces the old maxprobe/minprobe weighted formula (1.2 vs 0.8), which baked the
    probe class directly into the score and was therefore circular for maxprobe/minprobe
    separation analyses.

    Normalisation choice (empirical, on the RNAcompete max/minprobe set):
      - Interface area grows sublinearly with protein size (it tracks surface area, not
        residue count), so a sqrt(ProteinLength) divisor is used. It gives the best
        across-protein maxprobe/minprobe separation (global AUC 0.524), beating both raw
        area (0.523) and a linear ProteinLength divisor (0.519, which over-corrects).
        Within a protein it is equivalent to raw area, since ProteinLength is constant
        per protein.
      - No RNALength term: RNALength varies within a protein but normalising by it
        reduces separation (linear: within-protein AUC 0.50; sqrt: 0.525) vs no RNA
        term (0.531), i.e. interface area does not scale with RNA length here.
    """
    if not interface_area or not protein_length or protein_length <= 0:
        return 0.0
    return interface_area / math.sqrt(protein_length)

def get_clean_motif(motif):
    """Extract clean motif sequence from either format"""
    if '(' in motif:  # pred format: "U(11), U(12), G(13)..."
        try:
            # Split into individual base-position pairs
            bases = [part.strip() for part in motif.split(',')]
            
            # Extract bases and positions
            parsed = []
            for base in bases:
                if '(' in base and ')' in base:
                    nuc = base.split('(')[0].strip()
                    pos = int(base.split('(')[1].split(')')[0])
                    parsed.append((pos, nuc))
            
            # Sort by position and join bases
            parsed.sort(key=lambda x: x[0])  # Sort by position
            
            # Check if positions are continuous
            positions = [x[0] for x in parsed]
            if positions == list(range(min(positions), max(positions) + 1)):
                clean_motif = ''.join(x[1] for x in parsed)
                print(f"Cleaned predicted motif: {clean_motif} (from positions {min(positions)}-{max(positions)})")
                return clean_motif
            else:
                print(f"Warning: Non-continuous positions in motif: {positions}")
                return None
                
        except Exception as e:
            print(f"Error cleaning predicted motif: {e}")
            return None
            
    return motif  # exp format: already clean "GUUCGUU"

def get_motif_positions(motif, is_predicted=False):
    """Get positions of motif bases in sequence
    Returns: list of tuples (position, base)
    """
    if is_predicted and '(' in motif:  # pred format: "U(19), A(20)..."
        try:
            positions = []
            bases = [part.strip() for part in motif.split(',')]
            for base in bases:
                if '(' in base and ')' in base:
                    nuc = base.split('(')[0].strip()
                    pos = int(base.split('(')[1].split(')')[0])
                    # Add 1 to match experimental positions
                    positions.append((pos, nuc))
            return sorted(positions)  # Sort by position
        except Exception as e:
            print(f"Error parsing predicted motif: {e}")
            return None
    else:  # exp format: need to find in sequence
        return None  # We'll handle experimental motif differently

def calculate_motif_score(exp_motif, pred_motif, rna_sequence):
    """Calculate similarity score between experimental and predicted motifs
    Returns a score between 0 and 1, where:
    1.0 = perfect match in sequence and position
    0.0 = no match or invalid input
    """
    print("\nCalculating motif score:")
    print(f"Experimental motif: {exp_motif}")
    print(f"Predicted motif: {pred_motif}")
    print(f"RNA sequence: {rna_sequence}")
    
    if not exp_motif or not pred_motif or not rna_sequence:
        print("Missing input data")
        return 0.0

    # Find experimental motif in sequence (1-based)
    exp_start = rna_sequence.find(exp_motif)
    if exp_start == -1:
        print("Experimental motif not found in sequence")
        return 0.0
        
    # Get experimental motif positions (1-based)
    exp_positions = [(exp_start + i + 1, base) for i, base in enumerate(exp_motif)]
    print(f"Experimental motif positions: {exp_positions}")
    
    # Get predicted motif positions (converted to 1-based in get_motif_positions)
    pred_positions = get_motif_positions(pred_motif, is_predicted=True)
    if not pred_positions:
        print("Could not parse predicted motif positions")
        return 0.0
    print(f"Predicted motif positions: {pred_positions}")
    
    # Find matching positions
    matches = 0
    total_positions = max(len(exp_positions), len(pred_positions))
    
    # Create sets of positions for each base type
    exp_pos_dict = {base: set() for base in 'ACGU'}
    pred_pos_dict = {base: set() for base in 'ACGU'}
    
    for pos, base in exp_positions:
        exp_pos_dict[base].add(pos)
    for pos, base in pred_positions:
        pred_pos_dict[base].add(pos)
    
    # Count matches for each base type
    for base in 'ACGU':
        exp_set = exp_pos_dict[base]
        pred_set = pred_pos_dict[base]
        
        # Find closest matches within 3 positions
        while exp_set and pred_set:
            exp_pos = exp_set.pop()
            closest_pred = min(pred_set, key=lambda x: abs(x - exp_pos))
            distance = abs(closest_pred - exp_pos)
            
            if distance == 0:
                matches += 1
            elif distance <= 3:
                matches += 0.5
            else:
                matches += 0.2
            
            pred_set.remove(closest_pred)
    
    # Calculate final score
    score = matches / total_positions if total_positions > 0 else 0
    score = max(0.0, min(1.0, score))  # Ensure between 0 and 1
    
    print(f"Matches: {matches}")
    print(f"Total positions: {total_positions}")
    print(f"Final score: {score}")
    print("-" * 50)
    
    return score

def calculate_motif_in_exp_score(exp_positions, pred_positions):
    """Calculate ratio of predicted motif bases found in experimental motif"""
    print("\nCalculating motif in exp score:")
    print(f"Experimental positions: {exp_positions}")
    print(f"Predicted positions: {pred_positions}")
    
    # Create sets of (position, base) tuples for comparison
    exp_set = set((pos, base) for pos, base in exp_positions)
    pred_set = set((pos, base) for pos, base in pred_positions)
    
    # Find matches
    matches = exp_set.intersection(pred_set)
    
    # Calculate score
    score = len(matches) / len(exp_positions) if exp_positions else 0
    score = round(score, 2)  # Round to 2 decimal places
    
    print(f"Matches found: {matches}")
    print(f"Score: {score} ({len(matches)}/{len(exp_positions)})")
    print("-" * 50)
    
    return score

def update_similarity_scores(database, exp_data):
    """Update database with calculated similarity scores"""
    updates = []
    output_data = []
    
    for row in exp_data:
        pdb_id = row[0]
        print("pdb_id", pdb_id)
        rna_sequence = row[1]
        exp_motif = row[2]
        motif_type = row[3]
        interface_ratio = row[4]
        interface_area = row[5]
        protein_length = row[6]
        
        # Get predicted motif
        query = """
        SELECT RNAMotif 
        FROM pred_protein_rna 
        WHERE exp_db_id = UPPER(?)
        """
        pred_data = database.execute_query(query, (pdb_id,))
        if not pred_data:
            continue
            
        pred_motif = pred_data[0][0]
        
        # Calculate both scores
        motif_score = calculate_motif_score(exp_motif, pred_motif, rna_sequence)
        
        # Get positions for in_exp score
        exp_start = rna_sequence.find(exp_motif)
        if exp_start != -1:
            exp_positions = [(exp_start + i + 1, base) for i, base in enumerate(exp_motif)]
            pred_positions = get_motif_positions(pred_motif, is_predicted=True)
            if pred_positions:
                motif_in_exp_score = calculate_motif_in_exp_score(exp_positions, pred_positions)
            else:
                motif_in_exp_score = 0.0
        else:
            motif_in_exp_score = 0.0

        # Weight-neutral interface score: interface area per protein residue.
        try:
            area_f = float(interface_area) if interface_area else 0.0
        except (ValueError, TypeError):
            area_f = 0.0
        try:
            protlen_f = float(protein_length) if protein_length else 0.0
        except (ValueError, TypeError):
            protlen_f = 0.0
        motif_similarity = calculate_similarity_score(area_f, protlen_f)

        # Round scores
        motif_score = round(motif_score, 2)
        motif_in_exp_score = round(motif_in_exp_score, 2)
        
        # Add to updates for database
        updates.append((motif_score, motif_in_exp_score, motif_similarity, pdb_id))
        
        # Add to output data for plotting
        output_data.append({
            'pdb_id': pdb_id,
            'type': motif_type,
            'proteinRNA_InterfaceRatio': float(interface_ratio) if interface_ratio else 0.0,
            'proteinRNA_InterfaceArea': float(interface_area) if interface_area else 0.0,
            'similarity_score': motif_score,
            'motif_in_exp_score': motif_in_exp_score,
            'InterfaceArea_norm': motif_similarity
        })
    
    # Update database
    if updates:
        update_query = """
        UPDATE pred_protein_rna
        SET RNAmotif_score = ?,
            RNAMotif_inExp = ?,
            InterfaceArea_norm = ?
        WHERE exp_db_id = UPPER(?)
        """
        with database.connection:
            database.connection.executemany(update_query, updates)
    
    print(f"Updated {len(updates)} records with similarity scores")
    return output_data

def _parse_metric_value(raw, kind):
    """Parse a DB cell into a float for stats. kind: float | bracket | log10kd."""
    if raw is None:
        return None
    s = raw.strip() if isinstance(raw, str) else raw
    if isinstance(s, str) and s == '':
        return None
    try:
        if kind == 'bracket':
            txt = str(s).strip().strip('[]')
            if not txt:
                return None
            parts = [float(x) for x in txt.split(',') if x.strip() != '']
            return sum(parts) / len(parts) if parts else None
        if kind == 'log10kd':
            kd = float(s)
            return -1.0 * np.log10(kd) if kd > 0 else None
        return float(s)
    except (ValueError, TypeError):
        return None


def _describe_values(values):
    arr = np.asarray(values, dtype=float)
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan'),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
    }


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float('nan')
    sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float('nan')
    return float((np.mean(a) - np.mean(b)) / np.sqrt(sp2))


def write_minmax_stats(database, out_dir):
    """Write a txt summary of maxprobe vs minprobe distributions for each metric.

    For every metric: per-class n / mean / median / std / IQR, the max-min difference
    in means and medians, Cohen's d, the rank separation AUC P(max > min), and a
    Mann-Whitney U two-sided p-value (if scipy is available).
    """
    # (column, table_alias, label, parse_kind)
    metrics = [
        ('ProteinRNAInterfaceArea', 'p', 'Protein-RNA interface area [A^2]', 'float'),
        ('ProteinRNAInterfaceRatio', 'p', 'Protein-RNA interface ratio [%]', 'float'),
        ('InterfaceArea_norm', 'p', 'Normalized interface area (A/sqrt(Lp))', 'float'),
        ('RNAmotif_score', 'p', 'RNA motif score', 'float'),
        ('RNAMotif_inExp', 'p', 'RNA motif in exp', 'float'),
        ('Binding_affinity_kd', 'p', 'Predicted affinity -log10(Kd)', 'log10kd'),
        ('af3_protein_pTM', 'p', 'AF3 protein pTM', 'bracket'),
        ('af3_rna_pTM', 'p', 'AF3 RNA pTM', 'bracket'),
        ('af3_protein_ipTM', 'p', 'AF3 protein ipTM', 'bracket'),
        ('af3_rna_ipTM', 'p', 'AF3 RNA ipTM', 'bracket'),
        ('af3_protein_pLDDT_avg', 'p', 'AF3 protein pLDDT avg', 'float'),
        ('af3_rna_pLDDT_avg', 'p', 'AF3 RNA pLDDT avg', 'float'),
        ('af3_global_pae_avg', 'p', 'AF3 global PAE avg', 'float'),
        ('af3_chain_pair_pae_min', 'p', 'AF3 chain-pair PAE min', 'bracket'),
        ('af3_fraction_disordered', 'p', 'AF3 fraction disordered', 'float'),
        ('af3_ranking_score', 'p', 'AF3 ranking score', 'float'),
        ('RNALength', 'p', 'RNA length', 'float'),
        ('ProteinLength', 'p', 'Protein length', 'float'),
        ('BindingIntensity', 'e', 'Binding intensity (exp)', 'float'),
        ('RNAMotif_z_score', 'e', 'RNAMotif z-score (exp)', 'float'),
    ]

    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        mannwhitneyu = None

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'minmax_probe_stats.txt')
    lines = []
    lines.append('=' * 78)
    lines.append('Maxprobe vs minprobe: distribution statistics per metric')
    lines.append('=' * 78)
    lines.append('')
    lines.append('AUC = P(random maxprobe value > random minprobe value); 0.5 = no separation.')
    lines.append('Cohen d and AUC are signed as (maxprobe - minprobe). Kd is reported as')
    lines.append('-log10(Kd) so that higher = stronger predicted binding.')
    lines.append('')

    for col, alias, label, kind in metrics:
        query = f"""
        SELECT {alias}.{col}, e.Class
        FROM pred_protein_rna p
        JOIN exp_protein_rna e ON UPPER(e.PDBId) = p.exp_db_id
        WHERE e.Class IN ('maxprobe', 'minprobe')
        """
        try:
            rows = database.execute_query(query)
        except Exception as exc:
            lines.append(f"{label} [{col}]")
            lines.append(f"  (skipped: {exc})")
            lines.append('')
            continue

        mx, mn = [], []
        for value, klass in rows:
            v = _parse_metric_value(value, kind)
            if v is None:
                continue
            if klass == 'maxprobe':
                mx.append(v)
            elif klass == 'minprobe':
                mn.append(v)

        lines.append(f"{label} [{col}]")
        lines.append('-' * 78)
        if not mx or not mn:
            lines.append(f"  insufficient data (max n={len(mx)}, min n={len(mn)})")
            lines.append('')
            continue

        smax = _describe_values(mx)
        smin = _describe_values(mn)
        for tag, s in (('maxprobe', smax), ('minprobe', smin)):
            lines.append(
                f"  {tag:9s} n={s['n']:5d}  mean={s['mean']:12.4f}  median={s['median']:12.4f}"
                f"  std={s['std']:12.4f}  IQR=[{s['q1']:.4f}, {s['q3']:.4f}]"
            )

        d_mean = smax['mean'] - smin['mean']
        d_med = smax['median'] - smin['median']
        d = _cohens_d(mx, mn)

        wins = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in mx for b in mn)
        auc = wins / (len(mx) * len(mn))

        p_str = 'n/a (scipy unavailable)'
        if mannwhitneyu is not None:
            try:
                _, p = mannwhitneyu(mx, mn, alternative='two-sided')
                p_str = f"{p:.3e}"
            except Exception as exc:
                p_str = f"error ({exc})"

        lines.append(
            f"  delta mean (max-min) = {d_mean:+.4f}   delta median = {d_med:+.4f}"
        )
        lines.append(
            f"  Cohen d = {d:+.3f}   AUC(max>min) = {auc:.4f}   Mann-Whitney p = {p_str}"
        )
        lines.append('')

    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"Wrote max/min probe stats to {out_path}")
    return out_path


def create_plots(output_data, msa_option, single_chain_only):
    """Create plots from the processed data"""
    database = DatabaseMethods()
    output_df = pd.DataFrame(output_data)
    if output_df.empty:
        print("No data found for plotting")
        return
        
    plot_creator = PlotCreator('motif_metrics', msa_option, single_chain_only)
    
    # First create the original box plots
    plot_creator.get_boxplot_group(
        table_source='rnaCompete_metrics',
        output_df=output_df,
        column="proteinRNA_InterfaceRatio",
        group_by="type",
        name="Interface_Ratio",
        yAxis_label="Interface Ratio [%]"
    )

    plot_creator.get_boxplot_group(
        table_source='rnaCompete_metrics',
        output_df=output_df,
        column="proteinRNA_InterfaceArea",
        group_by="type",
        name="Interface_Area",
        yAxis_label="Interface Area [A]"
    )

    plot_creator.get_boxplot_group(
        table_source="rnaCompete_metrics",
        output_df=output_df,
        column="similarity_score",
        group_by="type",
        name="Similarity_Score",
        yAxis_label="RNA Motif Similarity Score"
    )

    # Add box plot for RNAMotif_inExp vs Class
    query = """
    SELECT p.RNAMotif_inExp, e.Class
    FROM pred_protein_rna p
    JOIN exp_protein_rna e ON UPPER(e.PDBId) = p.exp_db_id
    WHERE p.RNAMotif_inExp IS NOT NULL 
    AND e.Class IN ('maxprobe', 'minprobe')
    """
    data = database.execute_query(query)
    if data:
        try:
            plot_data = []
            for row in data:
                try:
                    if row[0] is not None and row[1]:  # Changed condition to include zeros
                        plot_data.append({
                            'value': float(row[0]),  # Will now include 0.0
                            'type': row[1]
                        })
                except (ValueError, TypeError):
                    continue
            
            if plot_data:
                df = pd.DataFrame(plot_data)
                plot_creator.get_boxplot_group(
                    table_source='rnaCompete_metrics',
                    output_df=df,
                    column='value',
                    group_by='type',
                    name="RNAMotif_inExp_by_Class",
                    yAxis_label="RNA Motif in Exp Score"
                )
        except Exception as e:
            print(f"Error creating RNAMotif_inExp boxplot: {e}")
    
    # Then create metric box plots by Class
    metrics_for_boxplots = [
        ('Binding_affinity_kd', '-log10(Binding Affinity Kd)'),
        ('af3_rna_pTM', 'RNA pTM'),
        ('af3_rna_ipTM', 'RNA ipTM'),
        ('af3_rna_pLDDT_avg', 'RNA pLDDT Average'),
        ('af3_global_pae_avg', 'Global PAE Average'),
        ('af3_chain_pair_pae_min', 'Chain Pair PAE Min'),
        ('af3_fraction_disordered', 'Fraction Disordered'),
        ('af3_has_clash', 'Has Clash'),
        ('af3_ranking_score', 'Ranking Score')
    ]

    # Get data for boxplots
    for metric, label in metrics_for_boxplots:
        query = f"""
        SELECT p.{metric}, e.Class
        FROM pred_protein_rna p
        JOIN exp_protein_rna e ON UPPER(e.PDBId) = p.exp_db_id
        WHERE p.{metric} IS NOT NULL 
        AND e.Class IN ('maxprobe', 'minprobe')
        AND p.{metric} != ''
        """
        data = database.execute_query(query)
        if data:
            try:
                plot_data = []
                for row in data:
                    try:
                        if row[0] and row[1]:
                            value = row[0]
                            class_type = row[1]
                            
                            # Handle different metric formats
                            if metric in ['af3_rna_pTM', 'af3_rna_ipTM']:
                                # Extract value from [0.09] format
                                value = float(value.strip('[]'))
                            elif metric == 'Binding_affinity_kd':
                                # Convert to -log10 scale
                                kd = float(value)
                                if kd > 0:
                                    value = -1 * np.log10(kd)
                                else:
                                    continue
                            else:
                                value = float(value)
                            
                            plot_data.append({
                                'value': value,
                                'type': class_type
                            })
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                if plot_data:
                    df = pd.DataFrame(plot_data)
                    plot_creator.get_boxplot_group(
                        table_source='rnaCompete_metrics',
                        output_df=df,
                        column='value',
                        group_by='type',
                        name=f"{metric}_by_Class",
                        yAxis_label=label
                    )
            except Exception as e:
                print(f"Error creating boxplot for {metric}: {e}")

    # New scatter plots
    # 1. Binding affinity vs Binding intensity
    query = """
    SELECT p.Binding_affinity_kd, e.BindingIntensity
    FROM pred_protein_rna p
    JOIN exp_protein_rna e ON UPPER(e.PDBId) = p.exp_db_id
    WHERE p.Binding_affinity_kd IS NOT NULL 
    AND e.BindingIntensity IS NOT NULL
    AND p.Binding_affinity_kd != ''
    AND e.BindingIntensity != ''
    """
    data = database.execute_query(query)
    if data:
        try:
            x_vals = []
            y_vals = []
            for row in data:
                try:
                    if row[0] and row[1]:  # Check for non-empty values
                        # Convert scientific notation to float and take -log10
                        kd = float(row[0])
                        if kd > 0:  # Avoid log of negative/zero
                            x = -1 * np.log10(kd)  # Convert to -log10(Kd)
                        else:
                            continue
                            
                        # Convert binding intensity to float
                        y = float(row[1])
                        
                        x_vals.append(x)
                        y_vals.append(y)
                except (ValueError, TypeError):
                    continue
                    
            if x_vals and y_vals:
                plot_creator.get_scatterplot(
                    table_source='rnaCompete_metrics',
                    xAxis_score=x_vals,
                    yAxis_score=y_vals,
                    xAxis_label="-log10(Binding Affinity Kd)",
                    yAxis_label="Binding Intensity",
                    name="Binding_Affinity_vs_Intensity"
                )
        except Exception as e:
            print(f"Error creating binding affinity plot: {e}")

    # 2. Metrics vs Both Similarity Scores
    metrics = [
        ('Binding_affinity_kd', '-log10(Binding Affinity Kd)'),
        ('af3_rna_pTM', 'RNA pTM'),
        ('af3_rna_ipTM', 'RNA ipTM'),
        ('af3_rna_pLDDT_avg', 'RNA pLDDT Average'),
        ('af3_global_pae_avg', 'Global PAE Average'),
        ('af3_chain_pair_pae_min', 'Chain Pair PAE Min'),
        ('af3_fraction_disordered', 'Fraction Disordered'),
        ('af3_has_clash', 'Has Clash'),
        ('af3_ranking_score', 'Ranking Score')
    ]

    for metric, label in metrics:
        # Plot vs RNAmotif_score
        query = f"""
        SELECT {metric}, RNAmotif_score
        FROM pred_protein_rna
        WHERE {metric} IS NOT NULL 
        AND RNAmotif_score IS NOT NULL
        AND {metric} != ''
        """
        data = database.execute_query(query)
        if data:
            try:
                x_vals = []
                y_vals = []
                for row in data:
                    try:
                        if row[0] and row[1]:
                            value = row[0]
                            
                            # Handle different metric formats
                            if metric in ['af3_rna_pTM', 'af3_rna_ipTM']:
                                x = float(value.strip('[]'))
                            elif metric == 'Binding_affinity_kd':
                                kd = float(value)
                                if kd > 0:
                                    x = -1 * np.log10(kd)
                                else:
                                    continue
                            else:
                                x = float(value)
                            
                            y = float(row[1])
                            x_vals.append(x)
                            y_vals.append(y)
                    except (ValueError, TypeError, SyntaxError):
                        continue
                        
                if x_vals and y_vals:
                    plot_creator.get_scatterplot(
                        table_source='rnaCompete_metrics',
                        xAxis_score=x_vals,
                        yAxis_score=y_vals,
                        xAxis_label=label,
                        yAxis_label="RNA Motif Similarity Score",
                        name=f"{metric}_vs_Similarity"
                    )
            except Exception as e:
                print(f"Error processing {metric}: {e}")

        # Plot vs RNAMotif_inExp
        query = f"""
        SELECT {metric}, RNAMotif_inExp
        FROM pred_protein_rna
        WHERE {metric} IS NOT NULL 
        AND RNAMotif_inExp IS NOT NULL
        AND {metric} != ''
        """
        data = database.execute_query(query)
        if data:
            try:
                x_vals = []
                y_vals = []
                for row in data:
                    try:
                        if row[0] is not None and row[1] is not None:
                            value = row[0]
                            
                            # Handle different metric formats
                            if metric in ['af3_rna_pTM', 'af3_rna_ipTM']:
                                x = float(value.strip('[]'))
                            elif metric == 'Binding_affinity_kd':
                                kd = float(value)
                                if kd > 0:
                                    x = -1 * np.log10(kd)
                                else:
                                    continue
                            else:
                                x = float(value)
                            
                            y = float(row[1])
                            x_vals.append(x)
                            y_vals.append(y)
                    except (ValueError, TypeError, SyntaxError):
                        continue
                        
                if x_vals and y_vals:
                    plot_creator.get_scatterplot(
                        table_source='rnaCompete_metrics',
                        xAxis_score=x_vals,
                        yAxis_score=y_vals,
                        xAxis_label=label,
                        yAxis_label="RNA Motif in Exp Score",
                        name=f"{metric}_vs_MotifInExp"
                    )
            except Exception as e:
                print(f"Error processing {metric}: {e}")

    # Write maxprobe vs minprobe distribution stats into the motif_metrics folder.
    try:
        write_minmax_stats(database, plot_creator.results_dir)
    except Exception as e:
        print(f"Error writing max/min stats: {e}")

    database.close_connection()

def get_plot_data(database, msa_filter=""):
    """Get data from database for plotting"""
    query = f"""
    SELECT e.PDBId, e.Class as type,
           p.ProteinRNAInterfaceRatio, p.ProteinRNAInterfaceArea,
           p.RNAmotif_score
    FROM exp_protein_rna e
    JOIN pred_protein_rna p ON UPPER(e.PDBId) = p.exp_db_id
    WHERE p.RNAmotif_score IS NOT NULL {msa_filter}
    """
    
    rows = database.execute_query(query)
    
    output_data = [{
        'pdb_id': row[0],
        'type': row[1],
        'proteinRNA_InterfaceRatio': float(row[2]) if row[2] else 0.0,
        'proteinRNA_InterfaceArea': float(row[3]) if row[3] else 0.0,
        'similarity_score': float(row[4]) if row[4] else 0.0
    } for row in rows]
    
    return output_data

def process_rnacompete_data(single_chain_only=False, msa_option=None, plot_only=False):
    """Process RNAcompete data and update similarity scores"""
    database = DatabaseMethods()
    
    # Get MSA filter
    msa_filter = ""
    if msa_option:
        if msa_option == '+MSA':
            msa_filter = "AND p.af3_rna_MSA = '[1]' OR p.af3_rna_MSA LIKE '[1,1%'"
        elif msa_option == '-MSA':
            msa_filter = "AND (p.af3_rna_MSA = '[0]' OR p.af3_rna_MSA LIKE '%0%')"
    
    if plot_only:
        # Just get data and create plots
        output_data = get_plot_data(database, msa_filter)
        create_plots(output_data, msa_option, single_chain_only)
    else:
        # Get data and update RNAmotif_score
        exp_query = f"""
        SELECT e.PDBId, e.RNASequence, e.RNAMotif, e.Class,
               p.ProteinRNAInterfaceRatio, p.ProteinRNAInterfaceArea, p.ProteinLength
        FROM exp_protein_rna e
        JOIN pred_protein_rna p ON UPPER(e.PDBId) = p.exp_db_id
        WHERE 1=1 {msa_filter}
        """
        
        exp_data = database.execute_query(exp_query)
        output_data = update_similarity_scores(database, exp_data)
        create_plots(output_data, msa_option, single_chain_only)
    
    database.close_connection()

if __name__ == "__main__":
    if len(sys.argv) not in [1, 2, 3, 4]:
        print(f"Usage: python {os.path.basename(sys.argv[0])} [single_chain|all] [+MSA|-MSA] [--plot-only]")
        print("Options:")
        print("  single_chain|all : Process single chain pairs or all chains")
        print("  +MSA|-MSA       : Filter by MSA presence")
        print("  --plot-only     : Only create plots without updating database")
        sys.exit(1)
        
    args = sys.argv[1:]
    single_chain_only = 'single_chain' in args
    plot_only = '--plot-only' in args
    
    # Remove flags from args to check for MSA option
    args = [arg for arg in args if arg not in ['single_chain', 'all', '--plot-only']]
    msa_option = args[0] if args and args[0] in ['+MSA', '-MSA'] else None
    
    if msa_option and msa_option.upper() not in ['+MSA', '-MSA']:
        print("MSA option must be either +MSA or -MSA")
        sys.exit(1)
        
    process_rnacompete_data(single_chain_only, msa_option, plot_only)