import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import ast
import difflib
from plots.plotCreator import PlotCreator
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig
from scipy import stats

class ProteinRNAInterface(DatabaseMethods):
    def __init__(self, plot_creator=None, id: str = "", protein_rna_interface_area: str = "", protein_rna_r_is: str = "", protein_rna_aap: str = "",
                 aac: str = ""):
        super().__init__()
        self.plot_creator = plot_creator
        self.id = id
        self.protein_rna_interface_area = protein_rna_interface_area
        self.protein_rna_r_is = protein_rna_r_is
        self.protein_rna_aap = protein_rna_aap
        self.aac = aac

    def get_proteinRNA_metrics(self, pred_table_name, exp_table_name):
        common_pdb = self.get_intersect_values(table1=f"{pred_table_name}",
                                               column1="exp_db_id",
                                               table2=f"{exp_table_name}",
                                               column2="PDBId"
                                               )

        filtered_pdb_ids = self.plot_creator.get_filtered_pdb_ids(pred_table_name)

        common_pdb_ids = list(set(filtered_pdb_ids) & set(common_pdb))

        results = {}
        columns = [
            "ProteinSequence", "ProteinChainIDs", "RNASequence", "RNAChainIDs",
            "AAC", "AAPproteinRNA", "ChainIDpairList_proteinRNA", "Hbond_proteinRNA",
            "vdWbond_proteinRNA", "ProteinRNAInterfaceArea", "ProteinRNAInterfaceRatio",
            "RNALength", "RNAMotif", "RNAMotifLength"
        ]

        # for pdb_id in common_pdb_ids:
                # pred_data = self.get_table_columns(pred_table_name, columns, condition=f"exp_db_id = '{pdb_id}'")
                # exp_data = self.get_table_columns(exp_table_name, columns, condition=f"PDBId = '{pdb_id}'")
                #
                # results[pdb_id] = {
                #     'predicted': pred_data,
                #     'experimental': exp_data
                # }
        for pdb_id in common_pdb_ids:
            # First check if both tables have valid data for this PDB ID
            pred_exists = self.get_table_columns(
                pred_table_name,
                ["exp_db_id"],
                condition=f"exp_db_id = '{pdb_id}' AND ProteinRNAInterfaceArea IS NOT NULL AND ContactList != '[]'"
            )
            exp_exists = self.get_table_columns(
                exp_table_name,
                ["PDBId"],
                condition=f"PDBId = '{pdb_id}' AND ProteinRNAInterfaceArea IS NOT NULL AND ContactList != '[]'"
            )

            # Only proceed if both checks pass
            if pred_exists and exp_exists:
                pred_data = self.get_table_columns(
                    pred_table_name,
                    columns,
                    condition=f"exp_db_id = '{pdb_id}' AND ProteinRNAInterfaceArea IS NOT NULL AND ContactList != '[]'"
                )
                exp_data = self.get_table_columns(
                    exp_table_name,
                    columns,
                    condition=f"PDBId = '{pdb_id}' AND ProteinRNAInterfaceArea IS NOT NULL AND ContactList != '[]'"
                )

                # Additional validation that we got all the data we need
                if all(pred_data) and all(exp_data):
                    results[pdb_id] = {
                        'predicted': pred_data,
                        'experimental': exp_data
                    }
                else:
                    print(f"Warning: Incomplete data for PDB ID {pdb_id}, skipping...")

        return results

class ProteinDNAInterface(DatabaseMethods):
    def __init__(self, plot_creator=None, id: str = "", protein_dna_interface_area: str = "", protein_dna_r_is: str = "", protein_dna_aap: str = "",
                 aac: str = ""):
        super().__init__()
        self.plot_creator = plot_creator
        self.id = id
        self.protein_dna_interface_area = protein_dna_interface_area
        self.protein_dna_r_is = protein_dna_r_is
        self.protein_dna_aap = protein_dna_aap
        self.aac = aac

    def get_proteinDNA_metrics(self, pred_table_name, exp_table_name):
        common_pdb = self.get_intersect_values(table1=f"{pred_table_name}",
                                               column1="exp_db_id",
                                               table2=f"{exp_table_name}",
                                               column2="PDBId"
                                               )

        filtered_pdb_ids = self.plot_creator.get_filtered_pdb_ids(pred_table_name)

        common_pdb_ids = list(set(filtered_pdb_ids) & set(common_pdb))

        results = {}
        columns = [
            "ProteinSequence", "ProteinChainIDs", "DNASequence", "DNAChainIDs",
            "AAC", "AAPproteinDNA", "ChainIDpairList_proteinDNA", "Hbond_proteinDNA",
            "vdWbond_proteinDNA", "ProteinDNAInterfaceArea", "ProteinDNAInterfaceRatio",
            "DNALength", "DNAMotif", "NumberDNAs"
        ]

        for pdb_id in common_pdb_ids:
            pred_data = self.get_table_columns(pred_table_name, columns, condition=f"exp_db_id = '{pdb_id}'")
            exp_data = self.get_table_columns(exp_table_name, columns, condition=f"PDBId = '{pdb_id}'")

            results[pdb_id] = {
                'predicted': pred_data,
                'experimental': exp_data
            }

        return results

class RnaDNAInterface(DatabaseMethods):
    def __init__(self, plot_creator=None, id: str = "", rna_dna_interface_area: str = "", rna_dna_r_is: str = ""):
        super().__init__()
        self.plot_creator = plot_creator
        self.id = id
        self.rna_dna_interface_area = rna_dna_interface_area
        self.rna_dna_r_is = rna_dna_r_is


class RnaRNAInterface(DatabaseMethods):
    def __init__(self, plot_creator=None, id: str = "", rna_rna_interface_area: str = "", rna_rna_r_is: str = ""):
        super().__init__()
        self.plot_creator = plot_creator
        self.id = id
        self.rna_rna_interface_area = rna_rna_interface_area
        self.rna_rna_r_is = rna_rna_r_is

    def get_rnaRNA_metrics(self, pred_table_name, exp_table_name):
        common_pdb = self.get_intersect_values(table1=f"{pred_table_name}",
                                               column1="exp_db_id",
                                               table2=f"{exp_table_name}",
                                               column2="PDBId"
                                               )

        filtered_pdb_ids = self.plot_creator.get_filtered_pdb_ids(pred_table_name)

        common_pdb_ids = list(set(filtered_pdb_ids) & set(common_pdb))

        results = {}
        columns = [
            "RNArnaInterfaceArea", "rnaRNAInterfaceRatio", "RNALength"
        ]

        for pdb_id in common_pdb_ids:
            pred_data = self.get_table_columns(pred_table_name, columns, condition=f"exp_db_id = '{pdb_id}'")
            exp_data = self.get_table_columns(exp_table_name, columns, condition=f"PDBId = '{pdb_id}'")

            results[pdb_id] = {
                'predicted': pred_data,
                'experimental': exp_data
            }

        return results

# def set_plot_style():
#     plt.rcParams.update({
#         'figure.figsize': (8, 6),  # Adjust the base figure size here (proportional to 10:6)
#         'font.size': 20,            # Set a standard font size for all plots
#         'axes.titlesize': 20,       # Title size
#         'axes.labelsize': 18,       # X and Y label size
#         'xtick.labelsize': 16,      # X-axis tick size
#         'ytick.labelsize': 16,      # Y-axis tick size
#         'legend.fontsize': 16,      # Legend font size
#         'figure.autolayout': True   # Automatically adjust layout for better fitting
#     })

# def get_IA_ris_barplots(ref_interface_area, ref_interface_ratio, pred_interface_area, pred_interface_ratio):
#     set_plot_style()
#     labels = ['Interface Area', 'IA/SA Ratio']
#     reference_values = [ref_interface_area, ref_interface_ratio]
#     predicted_values = [pred_interface_area, pred_interface_ratio]
#
#     x = range(len(labels))
#     plt.bar(x, reference_values, width=0.4, label='Reference', align='center')
#     plt.bar(x, predicted_values, width=0.4, label='Predicted', align='edge')
#     plt.xticks(x, labels)
#     plt.ylabel('Values')
#     plt.title('Comparison of Interface Metrics')
#     plt.legend()
#     plt.show()

def parse_string_to_list(list_str):
    if list_str is None:
        return
    list_str = list_str.strip('[]')
    elements = [element.strip() for element in list_str.split(',')]

    list_dict = {}
    for element in elements:
        try:
            # Split each element into key and value (key: value format expected)
            aa, value = element.split(':')
            aa = aa.strip()  # Remove extra whitespace around key
            value = value.strip()  # Remove extra whitespace around value
            list_dict[aa] = float(value)
        except ValueError as e:
            return None

    return list_dict


def convert_to_list_of_dicts(string_data):
    """
    Converts a string representation of a list of '[RES: value, RES: value]' entries
    into a list of dictionaries, preserving list length (empty entries -> {}).
    """
    try:
        if not string_data:
            print("Warning: Empty string data")
            return []

        list_of_strings = ast.literal_eval(string_data)
        list_of_dicts = []

        for item in list_of_strings:
            try:
                # Clean the string
                item_cleaned = item.strip("[]").strip()
                if not item_cleaned:
                    # Preserve empty placeholder
                    list_of_dicts.append({})
                    continue

                # Split and convert to dictionary
                dict_data = {}
                pairs = item_cleaned.split(",")
                for pair in pairs:
                    if ":" not in pair:
                        print(f"Warning: Skipping malformed pair: {pair}")
                        continue
                    try:
                        k, v = pair.split(":")
                        dict_data[k.strip()] = float(v.strip())
                    except ValueError as e:
                        print(f"Warning: Could not parse key-value pair {pair}: {e}")
                        continue

                # Always append, even if dict_data is empty
                list_of_dicts.append(dict_data)

            except Exception as e:
                print(f"Warning: Error processing item {item}: {e}")
                list_of_dicts.append({})  # preserve placeholder
                continue

        return list_of_dicts

    except Exception as e:
        print(f"Error converting string to list of dicts: {e}")
        print(f"Problematic string data: {string_data}")
        return []


def average_aac(aac):
    combined_aac = {}
    for interface in aac:
        interface_dict = parse_string_to_list(interface)
        if interface_dict is None:
            continue

        for aa, value in interface_dict.items():
            if aa in combined_aac:
                combined_aac[aa].append(value)
            else:
                combined_aac[aa] = [value]

    avg_aac = {aa: sum(values) / len(values) for aa, values in combined_aac.items()}
    return avg_aac

# How many amino acids are shared between the predicted and reference data sets relative to their total number
# The size of the intersection divided by the size of the union of the two sets
def jaccard_index(pred_aac, ref_aac):
    pred_set = set(pred_aac.keys())
    ref_set = set(ref_aac.keys())
    intersection = pred_set.intersection(ref_set)
    union = pred_set.union(ref_set)

    jaccard_index = len(intersection) / len(union) if union else 0
    return jaccard_index

# How similar the amino acid frequency distributions are, taking into account the magnitude of frequencies
# Cosine of the angle between two vectors in a multidimensional space
def cosine_similarity_score(pred_aac, ref_aac):
    all_aas = list(set(pred_aac.keys()).union(set(ref_aac.keys())))
    # Create vectors for both predicted and reference data
    pred_vector = [pred_aac.get(aa, 0) for aa in all_aas]
    ref_vector = [ref_aac.get(aa, 0) for aa in all_aas]
    pred_vector = np.array(pred_vector).reshape(1, -1)
    ref_vector = np.array(ref_vector).reshape(1, -1)
    similarity = cosine_similarity(pred_vector, ref_vector)[0][0]
    return similarity

def get_heatmap(aac_diffs, aap_diffs):
    set_plot_style()
    # Convert diffs to DataFrame for heatmap
    aac_diff_df = pd.DataFrame(list(aac_diffs.items()), columns=['Amino Acid', 'AAC Difference'])
    aap_diff_df = pd.DataFrame(list(aap_diffs.items()), columns=['Amino Acid', 'AAP Difference'])

    # Plotting AAC Differences
    # plt.figure(figsize=(10, 6))
    sns.barplot(x='Amino Acid', y='AAC Difference', data=aac_diff_df)
    plt.title('AAC Differences Between Reference and Predicted')
    plt.show()

    # Plotting AAP Differences
    # plt.figure(figsize=(10, 6))
    sns.barplot(x='Amino Acid', y='AAP Difference', data=aap_diff_df)
    plt.title('AAP Differences Between Reference and Predicted')
    plt.show()

def subtract_nested_lists(pred_list, ref_list):
    diff_list = []
    for pred_sublist, ref_sublist in zip(pred_list, ref_list):
        diff_sublist = [p - r for p, r in zip(pred_sublist, ref_sublist)]
        diff_list.append(diff_sublist)
    return diff_list

def find_matching_index(sequence_list, sequence, similarity_threshold=0.976):
    for i, seq in enumerate(sequence_list):
        if seq == sequence:
            return i
    # If no exact match is found, check for a similar match
    for i, seq in enumerate(sequence_list):
        similarity = difflib.SequenceMatcher(None, seq, sequence).ratio()
        if similarity >= similarity_threshold:
            # print(f"Found similar sequence with similarity {similarity:.2f}")
            return i
    return None

def convert_to_list(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return value  # Return if it's already a list

def create_union_chain(protein_dna_pairs, protein_rna_pairs, dna_chain_ids, rna_chain_ids, rna_h_bonds, rna_vdW_bonds, dna_h_bonds, dna_vdW_bonds):
    def safe_literal_eval(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)  # Try converting if it's a string
            except (ValueError, SyntaxError):
                return value
        return value

    dna_chain_ids = safe_literal_eval(dna_chain_ids)
    rna_chain_ids = safe_literal_eval(rna_chain_ids)
    protein_dna_pairs = safe_literal_eval(protein_dna_pairs)
    protein_rna_pairs = safe_literal_eval(protein_rna_pairs)
    rna_h_bonds = safe_literal_eval(rna_h_bonds)
    rna_vdW_bonds = safe_literal_eval(rna_vdW_bonds)
    dna_h_bonds = safe_literal_eval(dna_h_bonds)
    dna_vdW_bonds = safe_literal_eval(dna_vdW_bonds)
    # print("protein_dna_pairs", protein_dna_pairs)
    # print("protein_rna_pairs", protein_rna_pairs)

    # Determine the alphabetically smallest chain between DNA and RNA
    min_dna_chain = min(dna_chain_ids) if dna_chain_ids else None
    min_rna_chain = min(rna_chain_ids) if rna_chain_ids else None
    union_chain = []
    h_bonds = []
    vdW_bonds = []
    # If the smallest chain is from DNA, append DNA-protein pairs first, otherwise RNA-protein pairs
    if min_dna_chain < min_rna_chain:
        union_chain.extend(protein_dna_pairs)
        start_rna_index = len(union_chain)  # Index where RNA pairs start
        union_chain.extend(protein_rna_pairs)
        rna_indices = list(range(start_rna_index, len(union_chain)))  # Indices of RNA pairs
        h_bonds.extend(dna_h_bonds)
        h_bonds.extend(rna_h_bonds)
        vdW_bonds.extend(dna_vdW_bonds)
        vdW_bonds.extend(rna_vdW_bonds)
    else:
        union_chain.extend(protein_rna_pairs)
        rna_indices = list(range(len(union_chain)))
        union_chain.extend(protein_dna_pairs)
        h_bonds.extend(rna_h_bonds)
        h_bonds.extend(dna_h_bonds)
        vdW_bonds.extend(rna_vdW_bonds)
        vdW_bonds.extend(dna_vdW_bonds)
    # print("vdW_bonds", vdW_bonds)
    # print("h_bonds", h_bonds)
    # print("len(protein_rna_pairs)", len(protein_rna_pairs))
    if len(protein_rna_pairs) == 1:
        # print("%£$£& len(protein_rna_pairs) == 1")
        return union_chain, rna_indices[0]
    else:
        max_sum = -1
        max_sum_index = None

        for idx in rna_indices:
            # Sum of vdW_bonds and h_bonds for each RNA-pair
            total_bonds = vdW_bonds[idx] + h_bonds[idx]

            if total_bonds > max_sum:
                max_sum = total_bonds
                max_sum_index = idx

        return union_chain, max_sum_index

def find_max_index_and_chain_pairs(hbonds, vbonds, chainPairs, protChainIDs, protSequence, rnaChainIDs, rnaSequence, target_protSequence, target_rnaSequence, target_ProtChainIDs, target_RNAChainIDs, target_Chain_Pairs):
    chainPairs = convert_to_list(chainPairs)
    if len(chainPairs) == 1: # eg [["A", "B"]] in both cases
        return None, None
    else:
        hbonds = convert_to_list(hbonds)
        vbonds = convert_to_list(vbonds)
        total_bonds = [h + v for h, v in zip(hbonds, vbonds)]
        max_index = total_bonds.index(max(total_bonds))
        ref_pair_index = max_index
        prot, rna = chainPairs[max_index]

        if len(protChainIDs) == 1 and len(rnaChainIDs) != 1:
            # print("HELLO1")
            target_Chain_Pairs = convert_to_list(target_Chain_Pairs)
            if len(target_Chain_Pairs) == 1:
                target_pair_index = None
                return target_pair_index, ref_pair_index
            target_prot_chain = target_ProtChainIDs
            rnaChainIDs = convert_to_list(rnaChainIDs)
            rna_index = rnaChainIDs.index(rna)
            rnaSequence = convert_to_list(rnaSequence)
            rna_seq = rnaSequence[rna_index]
            target_rnaSequence = convert_to_list(target_rnaSequence)
            target_rna_index = find_matching_index(target_rnaSequence, rna_seq)
            target_RNAChainIDs = convert_to_list(target_RNAChainIDs)
            # Validate index before accessing
            if target_rna_index is None or target_rna_index >= len(target_RNAChainIDs):
                # Fallback: use rna_index if valid, otherwise use 0
                if rna_index < len(target_RNAChainIDs):
                    target_rna_index = rna_index
                else:
                    target_rna_index = 0  # Use first chain as fallback
            target_rna_chain = target_RNAChainIDs[target_rna_index]
            for i, pair in enumerate(target_Chain_Pairs):
                if pair == [target_prot_chain, target_rna_chain]:
                    target_pair_index = i
                    break
                elif pair[1] == target_rna_chain and target_prot_chain not in pair:
                    target_pair_index = i
                elif pair[0] == target_prot_chain and target_rna_chain not in pair:
                    target_pair_index = i

        elif len(rnaChainIDs) == 1 and len(protChainIDs) != 1:
            # print("HELLO2")
            target_Chain_Pairs = convert_to_list(target_Chain_Pairs)
            if len(target_Chain_Pairs) == 1:
                target_pair_index = None
                return target_pair_index, ref_pair_index
            target_rna_chain = target_RNAChainIDs
            protChainIDs = convert_to_list(protChainIDs)
            prot_index = protChainIDs.index(prot)
            protSequence = convert_to_list(protSequence)
            prot_seq = protSequence[prot_index]
            # print("target_protSequence", target_protSequence)
            # print("!!!!prot_seq", prot_seq)
            # print("isinstance(target_protSequence, str)", isinstance(target_protSequence, str))
            target_protSequence = convert_to_list(target_protSequence)
            target_prot_index = find_matching_index(target_protSequence, prot_seq)
            target_ProtChainIDs = convert_to_list(target_ProtChainIDs)
            # Validate index before accessing
            if target_prot_index is None or target_prot_index >= len(target_ProtChainIDs):
                # Fallback: use prot_index if valid, otherwise use 0
                if prot_index < len(target_ProtChainIDs):
                    target_prot_index = prot_index
                else:
                    target_prot_index = 0  # Use first chain as fallback
            # print("!!!!target_prot_index", target_prot_index)
            # print("target_ProtChainIDs", target_ProtChainIDs)
            target_prot_chain = target_ProtChainIDs[target_prot_index]
            # print("!!!target_prot_chain", target_prot_chain, target_rna_chain)
            # print("target_Chain_Pairs", target_Chain_Pairs)
            for i, pair in enumerate(target_Chain_Pairs):
                if pair == [target_prot_chain, target_rna_chain]:
                    target_pair_index = i
                    break
                elif pair[1] == target_rna_chain and target_prot_chain not in pair:
                    target_pair_index = i
                elif pair[0] == target_prot_chain and target_rna_chain not in pair:
                    target_pair_index = i

        else:
            # print("HELLO3")
            target_Chain_Pairs = convert_to_list(target_Chain_Pairs)
            if len(target_Chain_Pairs) == 1:
                target_pair_index = None
                return target_pair_index, ref_pair_index
            protChainIDs = convert_to_list(protChainIDs)
            prot_index = protChainIDs.index(prot)
            protSequence = convert_to_list(protSequence)
            prot_seq = protSequence[prot_index]
            target_protSequence = convert_to_list(target_protSequence)
            target_prot_index = find_matching_index(target_protSequence, prot_seq)
            target_ProtChainIDs = convert_to_list(target_ProtChainIDs)
            # Validate index before accessing
            if target_prot_index is None or target_prot_index >= len(target_ProtChainIDs):
                # Fallback: use prot_index if valid, otherwise use 0
                if prot_index < len(target_ProtChainIDs):
                    target_prot_index = prot_index
                else:
                    target_prot_index = 0  # Use first chain as fallback
            target_prot_chain = target_ProtChainIDs[target_prot_index]
            rnaChainIDs = convert_to_list(rnaChainIDs)
            rna_index = rnaChainIDs.index(rna)
            # print("rnaChainIDs, rna_index", rnaChainIDs, rna_index)
            rnaSequence = convert_to_list(rnaSequence)
            rna_seq = rnaSequence[rna_index]
            # print("rnaSequence, rna_seq", rnaSequence, rna_seq)
            target_rnaSequence = convert_to_list(target_rnaSequence)
            target_rna_index = find_matching_index(target_rnaSequence, rna_seq)
            target_RNAChainIDs = convert_to_list(target_RNAChainIDs)
            # Validate index before accessing
            if target_rna_index is None or target_rna_index >= len(target_RNAChainIDs):
                # Fallback: use rna_index if valid, otherwise use 0
                if rna_index < len(target_RNAChainIDs):
                    target_rna_index = rna_index
                else:
                    target_rna_index = 0  # Use first chain as fallback
            # print("@££ target_rna_index", target_rna_index)
            target_rna_chain = target_RNAChainIDs[target_rna_index]
            # print("!!!target_prot_chain", target_prot_chain, target_rna_chain)
            # print("!!@target_Chain_Pairs", target_Chain_Pairs)
            for i, pair in enumerate(target_Chain_Pairs):
                if pair == [target_prot_chain, target_rna_chain]:
                    target_pair_index = i
                    break
                elif pair[1] == target_rna_chain and target_prot_chain not in pair:
                    target_pair_index = i
                elif pair[0] == target_prot_chain and target_rna_chain not in pair:
                    target_pair_index = i

        return target_pair_index, ref_pair_index

def get_rna_dna_id_chain_pairs(result):
    for pdb_id, data in result.items():
        pred_data = data['predicted']
        ref_data = data['experimental']
        pred_rna_chains = pred_data[3]
        pred_rna_chain_id_pairs = pred_data[6]
        ref_rna_chains = ref_data[3]
        ref_rna_chain_id_pairs = ref_data[6]
    return pred_rna_chains, pred_rna_chain_id_pairs, ref_rna_chains, ref_rna_chain_id_pairs

def get_protein_na_plots(result, dna_result, plot_creator):
    # AAC, AAPproteinRNA, ProteinRNAInterfaceArea, ProteinRNAInterfaceRatio
    ia_diff = []
    r_is_diff = []
    aac_diff = []
    total_pred_aac = defaultdict(float)
    total_ref_aac = defaultdict(float)
    total_agreement_jaccard = []
    total_agreement_cosine = []
    rna_sequence_lengths = []
    aac_missing_in_pred_counts = []
    aac_missing_in_ref_counts = []
    pdb_ids = []
    aac_missing_in_pred_counter = Counter()
    aac_missing_in_ref_counter = Counter()
    pred_total_aac_count = 0
    ref_total_aac_count = 0
    pred_h_bonds_list = []
    ref_h_bonds_list = []
    pred_vdW_bonds_list = []
    ref_vdW_bonds_list = []
    rna_sequence_lengths_all = []
    pred_rna_motif_list = []
    ref_rna_motif_list = []
    ref_rna_motif_length_list_all = []
    pred_rna_motif_length_list_all = []
    pred_interface_area_list = []
    exp_interface_area_list = []
    pred_interface_ratio_list = []
    exp_interface_ratio_list = []

    for pdb_id, data in result.items():
        # print(f"PDB ID: {pdb_id}")
        # print("Predicted Data:", data['predicted'])
        # print("Experimental Data:", data['experimental'])
        pred_protein_seq, pred_protein_chains, pred_rna_seq, pred_rna_chains, pred_aac, pred_aap, pred_chain_id_pairs, \
            pred_h_bonds, pred_vdW_bonds, pred_interface_area, pred_interface_ratio, rna_length, pred_rna_motif, pred_rna_motif_length_list = data['predicted']
        ref_protein_seq, ref_protein_chains, ref_rna_seq, ref_rna_chains, ref_aac, ref_aap, ref_chain_id_pairs, \
            ref_h_bonds, ref_vdW_bonds, ref_interface_area, ref_interface_ratio, ref_rna_length, ref_rna_motif, ref_rna_motif_length_list = data['experimental']
        # print("pred_chain_id_pairs", pred_chain_id_pairs)
        # print("ref_chain_id_pairs", ref_chain_id_pairs)
        if dna_result:
            if pdb_id in dna_result:
                dna_data = dna_result[pdb_id]
                pred_protein_seq, pred_protein_chains, pred_dna_seq, pred_dna_chains, pred_aac, pred_aap, pred_dna_chain_id_pairs, \
                    pred_dna_h_bonds, pred_dna_vdW_bonds, pred_interface_area, pred_interface_ratio, dna_length, pred_dna_motif, NumberDNAs = \
                dna_data['predicted']
                ref_protein_seq, ref_protein_chains, ref_dna_seq, ref_dna_chains, ref_aac, ref_aap, ref_dna_chain_id_pairs, \
                    ref_dna_h_bonds, ref_dna_vdW_bonds, ref_interface_area, ref_interface_ratio, ref_dna_length, ref_dna_motif, NumberDNAs = \
                dna_data['experimental']

            pred_chain_id_pairs, target_pair_index = create_union_chain(pred_dna_chain_id_pairs, pred_chain_id_pairs,
                                                          pred_dna_chains, pred_rna_chains, pred_h_bonds, pred_vdW_bonds, pred_dna_h_bonds, pred_dna_vdW_bonds)
            ref_chain_id_pairs, ref_pair_index = create_union_chain(ref_dna_chain_id_pairs, ref_chain_id_pairs, ref_dna_chains,
                                                         ref_rna_chains, ref_h_bonds, ref_vdW_bonds, ref_dna_h_bonds, ref_dna_vdW_bonds)
            # print("@£$^%*$£&%$%&$%&@%@&@%$, ref_chain_id_pairs", ref_chain_id_pairs)
            # print("pred_chain_id_pairs", pred_chain_id_pairs)
        else:
            target_pair_index, ref_pair_index = find_max_index_and_chain_pairs(ref_h_bonds, ref_vdW_bonds, ref_chain_id_pairs, ref_protein_chains,
                                       ref_protein_seq, ref_rna_chains, ref_rna_seq, pred_protein_seq, pred_rna_seq,
                                       pred_protein_chains, pred_rna_chains, pred_chain_id_pairs)
        # Percentage differences
        if ref_interface_area == None or pred_interface_area == None or ref_interface_area == '' or pred_interface_area == '':
            continue
        else:
            # ref_interface_area = float(ref_interface_area)
            # pred_interface_area = float(pred_interface_area)
            interface_area_diff = round((((pred_interface_area - ref_interface_area) / ref_interface_area) * 100), 2)
            ia_diff.append(interface_area_diff)
            ia_to_sa_diff = round((((pred_interface_ratio - ref_interface_ratio) / ref_interface_ratio) * 100), 2)
            r_is_diff.append(ia_to_sa_diff)
        if dna_result:
            pred_aac = convert_to_list_of_dicts(pred_aac)
            pred_aac = pred_aac[target_pair_index]
            ref_aac = convert_to_list_of_dicts(ref_aac)
            ref_aac = ref_aac[ref_pair_index]
        else:
            # print("target_pair_index", target_pair_index)
            # print("ref_pair_index", ref_pair_index)
            # print("ref_chain_id_pairs", ref_chain_id_pairs)
            # print("target_prot_chain, target_rna_chain,", target_prot_chain, target_rna_chain,)
            # print("pred_aac", pred_aac)
            # print("ref_aac", ref_aac)
            # print(isinstance(pred_rna_motif_length_list, str))
            # pred_rna_motif_length_list = ast.literal_eval(pred_rna_motif_length_list)
            # print(isinstance(pred_rna_motif_length_list, str), len(pred_rna_motif_length_list))
            if target_pair_index is None or len(pred_rna_motif_length_list) == 1:
                # print("A pred_aac", pred_aac)
                if pred_aac.count('[') > 2: # check why this would happen other than for prot-dna-rna
                    pred_aac = average_aac(pred_aac)
                    # print("!£@$@$ Average applied!", pred_aac)
                else:
                    pred_aac = parse_string_to_list(pred_aac)
                # print("A", pred_rna_motif_length_list)
            if ref_pair_index is None or len(ref_rna_motif_length_list) == 1:
                ref_aac = parse_string_to_list(ref_aac)
                # print("B", ref_rna_motif_length_list)
            if target_pair_index is not None and target_pair_index >= 0 and len(pred_rna_motif_length_list) > 1:
                # print("B pred_aac", pred_aac)
                pred_aac = convert_to_list_of_dicts(pred_aac)
                pred_aac = pred_aac[target_pair_index]
                # print("C")
            if ref_pair_index is not None and ref_pair_index >= 0 and len(ref_rna_motif_length_list) > 1:
                ref_aac = convert_to_list_of_dicts(ref_aac)
                # print("convert_to_list_of_dicts", ref_aac)
                ref_aac = ref_aac[ref_pair_index]
                # print("d")
        # print("pred_aac post", pred_aac) # 2PO1
        # print("ref_aac post", ref_aac)
        if pred_aac and ref_aac:  # ACC differences %
            aac_diffs = {aa: round(((ref_aac[aa] - pred_aac.get(aa, 0) / ref_aac[aa]) * 100), 2) for aa in ref_aac}
            # print("aac_diffs", aac_diffs)
        # if isinstance(ref_aap, str):
        #     ref_aap = json.loads(ref_aap)
        # if isinstance(pred_aap, str):
        #     pred_aap = json.loads(pred_aap)
        # if pred_aap and ref_aap:  # Aap differences
        #     aap_diffs = {aa: round(((ref_aac[aa] - pred_aap.get(aa, 0) / ref_aap[aa]) * 100), 2) for aa in ref_aap}
        #     aap_diffs = {}
        #     for aa in ref_aap:
        #         if ref_aap[aa] == 0:
        #             continue
        #         aap_diffs[aa] = round(((ref_aac.get(aa, 0) - pred_aap.get(aa, 0)) / ref_aap[aa]) * 100, 2)
        for aa, freq in pred_aac.items():  # Aggregated AAC frequencies
            total_pred_aac[aa] += freq
        for aa, freq in ref_aac.items():  # Aggregated AAC frequencies
            total_ref_aac[aa] += freq
        agreement_jaccard = round(jaccard_index(pred_aac, ref_aac), 2)
        # print(f"Jaccard Index: {agreement_jaccard}")
        total_agreement_jaccard.append(agreement_jaccard)
        agreement_cosine = round(cosine_similarity_score(pred_aac, ref_aac), 2)
        # print(f"Cosine Similarity: {agreement_cosine}")
        total_agreement_cosine.append(agreement_cosine)
        if isinstance(rna_length, int):
            rna_sequence_lengths.append(rna_length)
        else:
            # print("RRRRNNNNARNARNARNA", type(rna_length), rna_length)
            rna_length = ast.literal_eval(rna_length)
            rna_sequence_lengths.append(max(rna_length))
        missing_in_pred = set(ref_aac.keys()) - set(pred_aac.keys())
        # print("missing_in_pred", missing_in_pred)
        missing_in_ref = set(pred_aac.keys()) - set(ref_aac.keys())
        # print("missing_in_ref", missing_in_ref)
        aac_missing_in_pred_counts.append(len(missing_in_pred))
        aac_missing_in_ref_counts.append(len(missing_in_ref))
        pdb_ids.append(pdb_id)
        ref_total_aac_count += len(ref_aac)
        pred_total_aac_count += len(ref_aac)
        aac_missing_in_pred_counter.update(missing_in_pred)
        aac_missing_in_ref_counter.update(missing_in_ref)
        # total_missing_in_pred = sum(aac_missing_in_pred_counter.values())
        # total_missing_in_ref = sum(aac_missing_in_ref_counter.values())
        # percent_missing_in_pred = (total_missing_in_pred / pred_total_aac_count) * 100
        # percent_missing_in_ref = (total_missing_in_ref / ref_total_aac_count) * 100
        pred_h_bonds = convert_to_list(pred_h_bonds)
        pred_h_bonds_list.append(max(pred_h_bonds))
        ref_h_bonds = convert_to_list(ref_h_bonds)
        ref_h_bonds_list.append(max(ref_h_bonds))
        pred_vdW_bonds = convert_to_list(pred_vdW_bonds)
        pred_vdW_bonds_list.append(max(pred_vdW_bonds))
        ref_vdW_bonds = convert_to_list(ref_vdW_bonds)
        ref_vdW_bonds_list.append(max(ref_vdW_bonds))
        # pred_rna_motif = convert_to_list(pred_rna_motif)
        # pred_rna_motif_list.append(pred_rna_motif)
        # ref_rna_motif = convert_to_list(ref_rna_motif)
        # ref_rna_motif_list.append(ref_rna_motif)
        ref_rna_motif_length_list = convert_to_list(ref_rna_motif_length_list)
        if isinstance(ref_rna_motif_length_list, int):
            ref_rna_motif_length_list = [ref_rna_motif_length_list]
        ref_rna_motif_length_list_all.append(max(ref_rna_motif_length_list))
        pred_rna_motif_length_list = convert_to_list(pred_rna_motif_length_list)
        if isinstance(pred_rna_motif_length_list, int):
            pred_rna_motif_length_list = [pred_rna_motif_length_list]
        pred_rna_motif_length_list_all.append(max(pred_rna_motif_length_list))
        pred_interface_area_list.append(pred_interface_area)
        exp_interface_area_list.append(ref_interface_area)
        pred_interface_ratio_list.append(pred_interface_ratio)
        exp_interface_ratio_list.append(ref_interface_ratio)

    plot_creator.get_barplot(
        table_source='interface_metrics',
        data=total_ref_aac,
        title="Aggregated AAC of all exp_protein_rna",
        xlabel="Amino Acids",
        ylabel="Aggregated Frequency",
        bar_color="#FF8F00",  # Optional: Change color
        rotation=45  # Optional: Adjust rotation
    )
    plot_creator.get_barplot(
        table_source='interface_metrics',
        data=total_pred_aac,
        title="Aggregated AAC of all pred_protein_rna",
        xlabel="Amino Acids",
        ylabel="Aggregated Frequency",
        bar_color="#48A860",  # Optional: Change color
        rotation=45  # Optional: Adjust rotation
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Jaccard Index",
        name="Jaccard_RNAlength",
        yAxis_score=total_agreement_jaccard
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Cosine Similarity",
        name="Cosine_RNAlength",
        yAxis_score=total_agreement_cosine
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Difference in Interface Area [%]",
        name="IA_diff_RNAlength",
        yAxis_score=ia_diff
    )
    # (+ if ref > pred, - if pred > ref)", "protein_rna")
    # get_scatterplot(rna_sequence_lengths, r_is_diff, "Difference in Interface Ratio [%]", "protein_rna")
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Difference in Interface Ratio [%]",
        name="IR_diff_RNAlength",
        yAxis_score=r_is_diff
    )
    # (+ if ref > pred, - if pred > ref)", "protein_rna")
    plot_creator.get_grouped_barplot(
        table_source='interface_metrics',
        groups=[aac_missing_in_pred_counts, aac_missing_in_ref_counts],
        group_labels=pdb_ids,
        colors=['orange', 'green'],
        xlabel='PDB ID',
        ylabel='Missing/Additional Protein-RNA Interface AA',
        title='AAC Missing and Additional Counts',
        rotation=90,
        legend_labels=['Missing in Predicted', 'Additional in Predicted']
    )
    # get_barplot_aa(aac_missing_in_pred_counter, aac_missing_in_ref_counter)
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=exp_interface_area_list,
        xAxis_label="Experimental Interface Area [Å]",
        yAxis_label="Predicted Interface Area [Å]",
        name="InterfaceArea_Pred_Exp",
        yAxis_score=pred_interface_area_list
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=exp_interface_ratio_list,
        xAxis_label="Experimental Interface Ratio",
        yAxis_label="Predicted Interface Ratio",
        name="InterfaceRatio_Pred_Exp",
        yAxis_score=pred_interface_ratio_list
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=ref_rna_motif_length_list_all,
        xAxis_label="Experimental RNA Motif Length",
        yAxis_label="Predicted RNA Motif Length",
        name="RNAmotifLength_Pred_Exp",
        yAxis_score=pred_rna_motif_length_list_all
    )

    shorter_than_25_positive = 0
    shorter_than_25_negative = 0
    longer_than_25_positive = 0
    longer_than_25_negative = 0

    for rna_sequence_length, ia_diff_value in zip(rna_sequence_lengths, r_is_diff):
        if rna_sequence_length < 25:
            if ia_diff_value > 0:
                shorter_than_25_positive += 1
            elif ia_diff_value < 0:
                shorter_than_25_negative += 1
        elif rna_sequence_length > 25:
            if ia_diff_value > 0:
                longer_than_25_positive += 1
            elif ia_diff_value < 0:
                longer_than_25_negative += 1

    rna_length_stats = {
        'shorter_than_25_positive': shorter_than_25_positive,
        'shorter_than_25_negative': shorter_than_25_negative,
        'longer_than_25_positive': longer_than_25_positive,
        'longer_than_25_negative': longer_than_25_negative
    }

    # diff_h_bonds = subtract_nested_lists(pred_h_bonds_list, ref_h_bonds_list)
    # diff_vdw_bonds = subtract_nested_lists(pred_vdW_bonds_list, ref_vdW_bonds_list)
    # Only max bonds considered:
    diff_h_bonds = [pred - ref for pred, ref in zip(pred_h_bonds_list, ref_h_bonds_list)]
    diff_vdw_bonds = [pred - ref for pred, ref in zip(pred_vdW_bonds_list, ref_vdW_bonds_list)]
    # print("diff_h_bonds", diff_h_bonds)
    # print("diff_vdw_bonds", diff_vdw_bonds)
    # plot_bond_difference_vs_length(rna_sequence_lengths, diff_h_bonds, "H_Bond_Difference (pred-ref)", "protein_rna")
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="H-Bond Difference",
        name="HBond_Diff_RNAlength",
        yAxis_score=diff_h_bonds
    )
    # plot_bond_difference_vs_length(rna_sequence_lengths, diff_vdw_bonds, "vdW_Bond_Difference (pred-ref)", "protein_rna")
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="vdW-Contact Difference",
        name="vdW_Diff_RNAlength",
        yAxis_score=diff_h_bonds
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=ref_h_bonds_list,
        xAxis_label="Experimental H-Bonds",
        yAxis_label="Predicted H-Bonds",
        name="HBond_Pred_Exp",
        yAxis_score=pred_h_bonds_list
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=ref_vdW_bonds_list,
        xAxis_label="Experimental vdW Interactions",
        yAxis_label="Predicted vdW Interactions",
        name="vdW_Pred_Exp",
        yAxis_score=pred_vdW_bonds_list
    )
    # # Total Bond Difference in both bond types
    # total_bond_diff = [h + v for h, v in zip(diff_h_bonds, diff_vdw_bonds)]
    # # Bond Density as bonds per unit of sequence length
    # h_bond_density = [h / length for h, length in zip(pred_h_bonds, sequence_length)]
    # vdw_bond_density = [v / length for v, length in zip(pred_vdw_bonds, sequence_length)]
    # # Correlation Between Bond Differences and Interface Area
    # interface_area_diff = [pred_area - ref_area for pred_area, ref_area in zip(pred_interface_area, ref_interface_area)]
    # # Percentage difference between predicted and reference bonds for a normalized view
    # percent_diff_h_bonds = [(pred - ref) / ref * 100 for pred, ref in zip(pred_h_bonds, ref_h_bonds)]
    # percent_diff_vdw_bonds = [(pred - ref) / ref * 100 for pred, ref in zip(pred_vdw_bonds, ref_vdw_bonds)]

    def calculate_mean_std(values):
        if not values:
            raise ValueError("The list of values is empty.")

        mean = np.mean(values)
        std_dev = np.std(values)

        median = np.median(values)

        return mean, std_dev, median

    pred_h_stats = calculate_mean_std(pred_h_bonds_list)
    ref_h_stats = calculate_mean_std(ref_h_bonds_list)
    pred_vdW_stats = calculate_mean_std(pred_vdW_bonds_list)
    ref_vdW_stats = calculate_mean_std(ref_vdW_bonds_list)
    ia_diff_stats = calculate_mean_std(ia_diff)
    r_is_diff_stats = calculate_mean_std(r_is_diff)
    jaccard_stats = calculate_mean_std(total_agreement_jaccard)
    cosine_stats = calculate_mean_std(total_agreement_cosine)
    diff_h_stats = calculate_mean_std(diff_h_bonds)
    diff_vdw_stats = calculate_mean_std(diff_vdw_bonds)
    
    stats_dict = {
        'Amount of hydrogen bonds in predicted complex': pred_h_stats,
        'Amount of hydrogen bonds in reference complex': ref_h_stats,
        'Difference in the amount of hydrogen bonds [%]': diff_h_stats,
        'Amount of van der Waals bonds in predicted complex': pred_vdW_stats,
        'Amount of van der Waals bonds in reference complex': ref_vdW_stats,
        'Difference in the amount of van der Waals bonds [%]': diff_vdw_stats,
        'Difference in Interface Area [%]': ia_diff_stats,
        'Difference in Interface Ratio [%]': r_is_diff_stats,
        'Amino acid composition (AAC) similarity scores based on the Jaccard Index': jaccard_stats,
        'Amino acid composition (AAC) cosine similarity scores': cosine_stats
    }
    
    # Write statistics to file
    write_statistics_to_file(stats_dict, plot_creator, rna_length_stats)

    # print(f"Missing amino acids in Predicted set: {dict(aac_missing_in_pred_counter)}")
    # print(f"Missing amino acids in Reference set: {dict(aac_missing_in_ref_counter)}")
    # print(f"Overall percentage of missing amino acids in Predicted set: {percent_missing_in_pred:.2f}%")
    # print(f"Overall percentage of missing amino acids in Reference set: {percent_missing_in_ref:.2f}%")

    plot_creator.get_barplot(
        table_source='interface_metrics',
        data=aac_missing_in_pred_counter,
        title='Missing_AA_in_Predicted',
        xlabel='Amino Acid',
        ylabel='Count',
        rotation=45  # Optional: Adjust rotation
    )

def write_statistics_to_file(stats_dict, plot_creator, rna_length_stats=None):
    """Write statistics to a text file in the same directory as plots."""
    output_path = os.path.join(plot_creator.results_dir, 'interface_metrics_statistics.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Interface Metrics Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        # Add filter information from plot_creator
        filter_parts = []
        if plot_creator.single_chain_only:
            filter_parts.append("Single-Chain")
        else:
            filter_parts.append("All-Chains")
        
        if plot_creator.msa_option:
            filter_parts.append(plot_creator.msa_option)
        
        if plot_creator.rbp_template_option:
            # Format RBP template option nicely
            if plot_creator.rbp_template_option.upper() == '+RBPTEMP':
                filter_parts.append("+RBP-template")
            elif plot_creator.rbp_template_option.upper() == '-RBPTEMP':
                filter_parts.append("-RBP-template")
            else:
                filter_parts.append(plot_creator.rbp_template_option)
        
        if filter_parts:
            f.write("Selected Samples by: " + ", ".join(filter_parts) + "\n\n")
        
        for metric_name, stats_tuple in stats_dict.items():
            if isinstance(stats_tuple, tuple) and len(stats_tuple) == 3:
                mean, std, median = stats_tuple
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {mean:.4f}\n")
                f.write(f"  Std: {std:.4f}\n")
                f.write(f"  Median: {median:.4f}\n\n")
        
        # Add RNA sequence length analysis
        if rna_length_stats is not None and isinstance(rna_length_stats, dict) and len(rna_length_stats) > 0:
            f.write("=" * 60 + "\n")
            f.write("RNA Sequence Length Analysis (Interface Ratio Difference)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Shorter than 25 nt:\n")
            f.write(f"  Positive differences: {rna_length_stats.get('shorter_than_25_positive', 0)}\n")
            f.write(f"  Negative differences: {rna_length_stats.get('shorter_than_25_negative', 0)}\n\n")
            f.write(f"Longer than 25 nt:\n")
            f.write(f"  Positive differences: {rna_length_stats.get('longer_than_25_positive', 0)}\n")
            f.write(f"  Negative differences: {rna_length_stats.get('longer_than_25_negative', 0)}\n\n")
    
    print(f"\nStatistics written to: {output_path}")


def get_na_na_plots(result, plot_creator):
    # AAC, AAPproteinRNA, ProteinRNAInterfaceArea, ProteinRNAInterfaceRatio
    ia_diff = []
    r_is_diff = []
    rna_sequence_lengths = []
    pdb_ids = []

    for pdb_id, data in result.items():
        # print(f"PDB ID: {pdb_id}")
        # print("Predicted Data:", data['predicted'])
        # print("Experimental Data:", data['experimental'])
        pred_interface_area, pred_interface_ratio, rna_length = data['predicted']
        ref_interface_area, ref_interface_ratio, ref_rna_length = data['experimental']
        # print("rna_length", rna_length)
        # print("ref_rna_length", ref_rna_length)
        # Percentage differences
        interface_area_diff = round((((pred_interface_area - ref_interface_area) / ref_interface_area) * 100), 2)
        ia_diff.append(interface_area_diff)
        ia_to_sa_diff = round((((pred_interface_ratio - ref_interface_ratio) / ref_interface_ratio) * 100), 2)
        r_is_diff.append(ia_to_sa_diff)
        pdb_ids.append(pdb_id)

    shorter_than_25_positive = 0
    shorter_than_25_negative = 0
    longer_than_25_positive = 0
    longer_than_25_negative = 0

    for rna_sequence_length, ia_diff_value in zip(rna_sequence_lengths, r_is_diff):
        if rna_sequence_length < 25:
            if ia_diff_value > 0:
                shorter_than_25_positive += 1
            elif ia_diff_value < 0:
                shorter_than_25_negative += 1
        elif rna_sequence_length > 25:
            if ia_diff_value > 0:
                longer_than_25_positive += 1
            elif ia_diff_value < 0:
                longer_than_25_negative += 1

    rna_length_stats = {
        'shorter_than_25_positive': shorter_than_25_positive,
        'shorter_than_25_negative': shorter_than_25_negative,
        'longer_than_25_positive': longer_than_25_positive,
        'longer_than_25_negative': longer_than_25_negative
    }
    
    # Write statistics to file for get_na_na_plots
    def calculate_mean_std(values):
        if not values:
            raise ValueError("The list of values is empty.")
        mean = np.mean(values)
        std_dev = np.std(values)
        median = np.median(values)
        return mean, std_dev, median
    
    stats_dict_na = {
        'Difference in Interface Area [%]': calculate_mean_std(ia_diff),
        'Difference in Interface Ratio [%]': calculate_mean_std(r_is_diff)
    }
    write_statistics_to_file(stats_dict_na, plot_creator, rna_length_stats)
    
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Difference in Interface Area [%]",
        name="IR_diff_RNAlength",
        yAxis_score=ia_diff
    )
    plot_creator.get_scatterplot(
        table_source='interface_metrics',
        xAxis_score=rna_sequence_lengths,
        xAxis_label="RNA Sequence Length",
        yAxis_label="Difference in Interface Ratio [%]",
        name="IR_diff_RNAlength",
        yAxis_score=r_is_diff
    )

if __name__ == "__main__":
    if len(sys.argv) not in [1, 2, 3, 4, 5]:
        print(f"Usage: python {os.path.basename(sys.argv[0])} [single_chain|all] [+msa|-msa] [+rbptemp|-rbptemp]")
        sys.exit(1)

    # Get table names from startConfig
    config = StartConfig()
    pred_table = config.pred_table
    exp_table = config.ref_table
    
    # Parse arguments - handle flexible ordering
    single_chain_only = False
    msa_option = None
    rbp_template_option = None
    
    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg in ['single_chain', 'all']:
            single_chain_only = (arg == 'single_chain')
        elif arg_lower in ['+msa', '-msa', '+msa', '-msa']:
            msa_option = arg  # Keep original case
        elif arg_lower in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
            rbp_template_option = arg  # Keep original case

    plot_creator = PlotCreator('interface_metrics', msa_option, single_chain_only, rbp_template_option)

    if pred_table == "pred_protein_rna":
        protein_rna_metrics = ProteinRNAInterface(plot_creator=plot_creator)
        result = protein_rna_metrics.get_proteinRNA_metrics(pred_table, exp_table)
        get_protein_na_plots(result, None, plot_creator)
        protein_rna_metrics.close_connection()
    elif pred_table == "pred_protein_dna":
        protein_dna_metrics = ProteinDNAInterface(plot_creator=plot_creator)
        result = protein_dna_metrics.get_proteinDNA_metrics(pred_table, exp_table)
        get_protein_na_plots(result, None, plot_creator)
        protein_dna_metrics.close_connection()
    elif pred_table == "pred_rna_rna":
        rna_rna_metrics = RnaRNAInterface(plot_creator=plot_creator)
        result = rna_rna_metrics.get_rnaRNA_metrics(pred_table, exp_table)
        get_na_na_plots(result, plot_creator)
        rna_rna_metrics.close_connection()
    elif pred_table == "pred_protein_rna_dna":
        protein_rna_metrics = ProteinRNAInterface(plot_creator=plot_creator)
        rna_result = protein_rna_metrics.get_proteinRNA_metrics(pred_table, exp_table)
        pred_rna_chains, pred_rna_chain_id_pairs, ref_rna_chains, ref_rna_chain_id_pairs = get_rna_dna_id_chain_pairs(rna_result)
        protein_dna_metrics = ProteinDNAInterface(plot_creator=plot_creator)
        dna_result = protein_dna_metrics.get_proteinDNA_metrics(pred_table, exp_table)
        # Protein-RNA interface
        get_protein_na_plots(rna_result, dna_result, plot_creator)
        protein_dna_metrics.close_connection()
    else:
        print(f"Warning: Unknown table type {pred_table}. Using default behavior.")
        protein_rna_metrics = ProteinRNAInterface(plot_creator=plot_creator)
        result = protein_rna_metrics.get_proteinRNA_metrics(pred_table, exp_table)
        get_protein_na_plots(result, None, plot_creator)
        protein_rna_metrics.close_connection()



