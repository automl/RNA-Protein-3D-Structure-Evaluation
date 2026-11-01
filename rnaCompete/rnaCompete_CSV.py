import pandas as pd
import re
import sys
import os
from pathlib import Path
from plots.plotCreator import PlotCreator

# def convert_rna_motif_to_range(motif_str):
#     positions = [int(pos.group(1)) for pos in re.finditer(r'\((\d+)\)', motif_str)]
#     return f"{min(positions)}-{max(positions)}" if positions else None

# Find the longest consecutive sequence and convert it to range
def convert_rna_motif_to_range(motif_str):
    positions = [int(pos.group(1)) for pos in re.finditer(r'\((\d+)\)', motif_str)]
    if not positions:
        return None
    positions.sort()
    longest_seq = []
    current_seq = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1] + 1:
            # If consecutive, extend the current sequence
            current_seq.append(positions[i])
        else:
            # Otherwise, check if the current sequence is the longest
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq
            current_seq = [positions[i]]

    # Check the last sequence in case it's the longest
    if len(current_seq) > len(longest_seq):
        longest_seq = current_seq

    return f"{longest_seq[0]}-{longest_seq[-1]}" if longest_seq else None

# def calculate_similarity(pred_range, exp_range):
#     pred_start, pred_end = map(int, pred_range.split('-'))
#     exp_start, exp_end = map(int, exp_range.split('-'))
#     overlap = set(range(pred_start, pred_end + 1)).intersection(range(exp_start, exp_end + 1))
#     return len(overlap) / (pred_end - pred_start + 1) if pred_end - pred_start + 1 else 0

def calculate_similarity(pred_range, exp_range):
    pred_start, pred_end = map(int, pred_range.split('-'))
    exp_start, exp_end = map(int, exp_range.split('-'))
    pred_length = pred_end - pred_start + 1
    exp_length = exp_end - exp_start + 1

    # Calculate the overlap between the two ranges
    overlap = set(range(pred_start, pred_end + 1)).intersection(range(exp_start, exp_end + 1))
    overlap_length = len(overlap)

    # Calculate the length-dependent similarity
    combined_length = pred_length + exp_length
    return (2 * overlap_length) / combined_length if combined_length else 0

def main(csv_path, tsv_path, plot_creator):
    csv_data = pd.read_csv(csv_path)
    tsv_data = pd.read_csv(tsv_path, sep='\t')
    output_data = []
    output_path = Path(csv_path).parent / 'similarity_table.csv'

    for _, csv_row in csv_data.iterrows():
        pred_motif = csv_row['RNAMotif']
        file_name = csv_row['FileName']
        pred_range = convert_rna_motif_to_range(pred_motif)

        for _, tsv_row in tsv_data[tsv_data['file_id'] == file_name.split('_model_')[0]].iterrows():
            exp_motif = tsv_row['mer_pos']
            similarity_score = calculate_similarity(pred_range, exp_motif)
            motif_type = re.search(r'_(min7mer|max7mer|minprobe|maxprobe)_', csv_row['FileName'])
            motif_type = motif_type.group(1) if motif_type else "unknown"

            output_data.append({
                'file_name': file_name,
                'type': motif_type,
                'pred_motif': pred_range,
                'exp_motif': exp_motif,
                'similarity_score': round(similarity_score,2),
                'proteinRNA_InterfaceArea' : csv_row['ProteinRNAInterfaceArea'],
                'proteinRNA_InterfaceRatio' : csv_row['ProteinRNAInterfaceRatio']
            })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    print(f"Similarity table saved to {output_path}.")

    plot_creator.get_boxplot_group(
        table_source='rnaCompete_metrics',
        output_df=output_df,
        column="proteinRNA_InterfaceRatio",
        group_by="type",
        name="Interface_Ratio",
        yAxis_label="Interface Ratio [%]"
    )

    plot_creator.get_boxplot_group(
        table_source="rnaCompete_metrics",
        output_df=output_df,
        column="similarity_score",
        group_by="type",
        name="Similarity Score",
        yAxis_label="Similarity Score"
    )

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <path_to_csv> <path_to_tsv> [single_chain|all] [+MSA|-MSA]")
    else:
        csv_path = sys.argv[1]
        tsv_path = sys.argv[2]

        single_chain_only = sys.argv[3] == 'single_chain' if len(sys.argv) > 2 else False
        msa_option = sys.argv[4] if len(sys.argv) > 3 else None

        plot_creator = PlotCreator('motif_metrics', msa_option, single_chain_only)

        main(csv_path, tsv_path, plot_creator)

